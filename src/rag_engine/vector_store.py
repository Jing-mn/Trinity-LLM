# src/rag_engine/vector_store.py

import os
import logging
from typing import List, Optional

# 【国内网络修复】huggingface.co 在国内经常连不上或报 SSL 证书错误，
# 这里默认切换到国内镜像 hf-mirror.com。
# 注意：必须在导入 langchain_huggingface（会间接导入 huggingface_hub）之前设置，
# 因为 huggingface_hub 在导入时只读取一次 HF_ENDPOINT。
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def get_embeddings_model():
    """
    获取 Embedding 模型（将文字变成数字向量的模型）
    我们选用 BAAI/bge-small-zh-v1.5，专为中文优化，体积小（约 33MB），速度快。
    """
    # 【为什么要这样写？】指定 device="cpu" 确保没有显卡也能跑
    # 如果你有 NVIDIA 显卡，可以改成 device="cuda" 加速
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}  # 归一化，提升检索效果
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def build_vector_store(
    documents: List[Document],
    persist_directory: str = "./data/chroma_db",
    batch_size: int = 100
) -> Chroma:
    """
    将切分好的文档块向量化，并存入 Chroma 持久化数据库。

    参数:
        documents: 切分后的文档块列表 (List[Document])
        persist_directory: 数据库保存路径
        batch_size: 一次性处理多少条（防止内存爆炸）

    返回:
        Chroma: 已构建好的向量库对象
    """
    if not documents:
        raise ValueError("文档列表为空，无法构建向量库！")
    
    logger.info(f"准备构建向量库，共 {len(documents)} 个文档块...")
    logger.info(f"向量库将持久化到: {persist_directory}")
    
    # 获取向量模型
    embeddings = get_embeddings_model()
    
    # 【核心动作】从文档创建 Chroma 向量库
    # Chroma 会自动调用 embeddings 模型，把 doc.page_content 变成向量存起来
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="trinity_rag_collection"  # 给集合起个名字，方便管理
    )
    # 说明：langchain_chroma 的 Chroma 没有 persist() 方法（旧版 langchain_community 才有）。
    # 传入 persist_directory 后，chromadb 使用 PersistentClient，会在 add_documents 时自动落盘，无需手动持久化。

    logger.info(f"✅ 向量库构建完成！已存入 {persist_directory}")
    return vectorstore

def load_vector_store(persist_directory: str = "./data/chroma_db") -> Chroma:
    """
    从硬盘加载已有的向量库（用于以后直接使用，不用重新构建）
    """
    embeddings = get_embeddings_model()
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="trinity_rag_collection"
    )
    logger.info(f"✅ 向量库加载成功！共 {vectorstore._collection.count()} 条向量")
    return vectorstore