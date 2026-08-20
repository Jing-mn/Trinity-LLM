# src/api/models.py

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """客户端发送的提问请求"""
    question: str = Field(..., description="用户的问题", example="这份文档讲了什么？")
    top_k: Optional[int] = Field(4, description="检索时返回的文档块数量", ge=1, le=10)


class SourceInfo(BaseModel):
    """单个引用来源的信息"""
    file_name: str = Field(..., description="文件名")


class QueryResponse(BaseModel):
    """服务器返回的回答响应"""
    answer: str = Field(..., description="AI 生成的回答内容")
    sources: List[str] = Field(..., description="引用来源的文件名列表")