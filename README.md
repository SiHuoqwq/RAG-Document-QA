# RAG-Document-QA
基于 RAG + 火山方舟 DeepSeek 的本地文档智能问答系统
# 基于 RAG 架构的本地文档智能问答系统
## 项目介绍
独立开发的轻量化本地文档问答系统，基于检索增强生成（RAG）技术，实现对本地 Markdown 文档的智能检索与精准回答。

## 核心功能
- 本地 Markdown 文档自动解析与分块
- 中文文本向量化 + Chroma 向量数据库存储
- 向量检索 + 语义重排序优化
- 集成火山方舟 DeepSeek 大模型生成回答

## 技术栈
Python | Sentence-Transformers | ChromaDB | 火山方舟 DeepSeek | OpenAI 兼容接口

## 运行结果
项目成功实现基于本地知识库的精准问答，可稳定调用大模型生成回答。
