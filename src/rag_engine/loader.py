# src/rag_engine/loader.py

import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

# 配置日志（让程序在运行时报出信息，方便你看到进度）
logger = logging.getLogger(__name__)

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    从指定文件夹加载所有 PDF、DOCX、TXT 文件，并转换为 LangChain 的 Document 对象列表。
    
    参数:
        folder_path (str): 存放文档的文件夹路径 (比如 "data/raw_docs")
    
    返回:
        List[Document]: 包含所有文档内容和元数据的列表
    """
    # 【为什么要这样写？】如果路径不存在，提前报错退出，避免后面出现莫名其妙的报错。
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"错误：找不到文件夹！请检查路径: {folder_path}")
    
    all_documents = []
    
    # 遍历文件夹里的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # 忽略文件夹，只处理文件
        if os.path.isdir(file_path):
            continue
        
        logger.info(f"正在读取文件: {file_name}")
        
        try:
            # 【核心逻辑】根据文件后缀名，选择不同的“解析器”提取文字
            if file_name.endswith(".pdf"):
                # PDF 解析器（可以处理扫描件里的文字层）
                loader = PyPDFLoader(file_path)
                
            elif file_name.endswith(".docx"):
                # Word 文档解析器
                loader = Docx2txtLoader(file_path)
                
            elif file_name.endswith(".txt"):
                # 纯文本文件解析器（指定 utf-8 编码，防止中文乱码）
                loader = TextLoader(file_path, encoding="utf-8")
                
            else:
                # 不支持的文件格式就跳过，不报错（因为可能有图片或Excel暂时不处理）
                logger.warning(f"跳过不支持的文件格式: {file_name}")
                continue
            
            # 执行加载，获取此文件里的所有“文档块”（通常一个PDF有几页就有几个块）
            docs = loader.load()
            
            # 【关键步骤】给每个文档块打上“来源标签”（为了以后知道答案出自哪个文件）
            for doc in docs:
                # 如果元数据里没有来源，就手动添加文件路径
                if "source" not in doc.metadata:
                    doc.metadata["source"] = file_path
                # 还可以额外加个文件名，方便展示
                doc.metadata["file_name"] = file_name
                
            all_documents.extend(docs)
            logger.info(f"成功读取 {file_name}，共 {len(docs)} 页/块")
            
        except Exception as e:
            # 【为什么要这样写？】防止因为某一个文件损坏，导致整个程序崩溃
            logger.error(f"读取文件 {file_name} 时出错: {e}")
            continue
    
    logger.info(f"全部加载完成！共加载了 {len(all_documents)} 个文档块。")
    return all_documents