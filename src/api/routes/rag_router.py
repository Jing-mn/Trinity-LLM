# src/api/routes/rag_router.py

import logging
from fastapi import APIRouter, HTTPException, Request
from src.api.models import QueryRequest, QueryResponse
from src.rag_engine.rag_chain import query_with_sources

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG 问答"])

@router.post("/query", response_model=QueryResponse)
async def rag_query(request: Request, req: QueryRequest):
    try:
        chain = request.app.state.rag_chain
        retriever = request.app.state.rag_retriever
        if chain is None or retriever is None:
            raise HTTPException(status_code=503, detail="RAG 系统尚未初始化完成")
        
        logger.info(f"收到提问: {req.question}")
        result = query_with_sources(
            question=req.question,
            chain=chain,
            retriever=retriever,
            top_k=req.top_k
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理提问时出错: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")