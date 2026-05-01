# -*- coding: utf-8 -*-
# 完整RAG系统：文档分块 + 向量检索 + 重排序 + 火山方舟DeepSeek回答

# 1. 导入所有依赖
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# 2. 加载配置文件（.env）
load_dotenv()

# 3. 全局模型初始化（只加载一次，速度最快）
# 中文文本嵌入模型
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
# 文本重排序模型
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
# 向量数据库（内存存储）
chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

# ---------------------- 核心功能函数 ----------------------
def split_into_chunks(doc_file: str) -> List[str]:
    """读取Markdown文件并分块（修复UTF-8编码报错）"""
    with open(doc_file, 'r', encoding='utf-8') as file:
        content = file.read()
    # 按空行分割，过滤空白内容
    return [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

def embed_chunk(chunk: str) -> List[float]:
    """将文本转换为向量"""
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()

def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    """将文本和向量存入数据库"""
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )

def retrieve(query: str, top_k: int = 5) -> List[str]:
    """根据问题检索相关文本"""
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

def rerank(query: str, retrieved_chunks: List[str], top_k: int = 3) -> List[str]:
    """对检索结果重排序，提升精度"""
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks][:top_k]

def generate(query: str, chunks: List[str]) -> str:
    """
    调用火山方舟DeepSeek生成回答（通用OpenAI接口，无SDK报错）
    """
    # 拼接参考文本
    chunks_content = "\n\n".join(chunks)

    # 提示词模板
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确回答。
用户问题：{query}
相关片段：
{chunks_content}
请基于上述内容作答，不要编造信息。"""

    print(f"===== 系统提示词 =====\n{prompt}\n=====================\n")

    # 火山方舟 OpenAI 兼容配置
    client = OpenAI(
        api_key=os.getenv("VOLC_ACCESS_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )

    # 调用模型
    completion = client.chat.completions.create(
        model="ep-20260427153808-rkvgh",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048
    )

    return completion.choices[0].message.content

# ---------------------- 主程序运行 ----------------------
if __name__ == "__main__":
    try:
        # 1. 读取文档（确保 doc.md 在代码同一文件夹）
        print("正在读取文档并分块...")
        chunks = split_into_chunks("doc.md")

        # 2. 生成向量并入库
        print("正在生成向量并存储...")
        embeddings = [embed_chunk(chunk) for chunk in chunks]
        save_embeddings(chunks, embeddings)

        # 3. 设置你的问题
        query = "哆啦A梦使用的3个秘密道具分别是什么？"
        print(f"\n用户问题：{query}\n")

        # 4. 检索 + 重排序
        print("正在检索相关内容...")
        retrieved_chunks = retrieve(query, top_k=5)
        reranked_chunks = rerank(query, retrieved_chunks, top_k=3)

        # 5. 生成回答
        print("正在调用 DeepSeek 生成回答...\n")
        answer = generate(query, reranked_chunks)

        # 6. 输出结果
        print("=" * 60)
        print("📝 DeepSeek 最终回答：")
        print(answer)
        print("=" * 60)

    except Exception as e:
        print(f"运行出错：{e}")