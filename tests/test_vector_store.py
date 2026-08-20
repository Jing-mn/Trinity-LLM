# tests/test_vector_store.py
# 测试 src/rag_engine/vector_store.py 的向量库构建与加载。
#
# 注意：本文件的测试需要加载真实 embedding 模型（BAAI/bge-small-zh-v1.5），
# 每次实例化约需 10-20 秒（模型已缓存在本机时仍要加载进内存），运行较慢属正常。

import pytest
from langchain_core.documents import Document

from src.rag_engine.vector_store import (
    build_vector_store,
    get_embeddings_model,
    load_vector_store,
)


def test_get_embeddings_model():
    """测试 get_embeddings_model 返回正确的 embedding 模型实例。"""
    embeddings = get_embeddings_model()

    assert embeddings is not None
    assert embeddings.model_name == "BAAI/bge-small-zh-v1.5"


def test_build_vector_store_empty_raises(tmp_path):
    """测试空文档构建向量库：应抛出 ValueError。"""
    # build_vector_store 在加载模型之前就会检查空文档，因此该测试是快的
    with pytest.raises(ValueError):
        build_vector_store([], persist_directory=str(tmp_path))


def test_load_vector_store(tmp_path):
    """测试先构建再加载向量库：加载后数量应与构建时一致。"""
    docs = [
        Document(
            page_content="这是用于测试向量库构建与加载的文档内容。",
            metadata={"source": "test.txt"},
        )
    ]
    # 1. 先构建向量库到临时目录
    build_vector_store(docs, persist_directory=str(tmp_path))

    # 2. 再从同一临时目录加载
    store = load_vector_store(str(tmp_path))

    assert store._collection.count() == 1
