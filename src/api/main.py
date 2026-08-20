# src/api/main.py

import sys
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

from src.rag_engine.rag_chain import create_rag_chain
from src.api.routes.rag_router import router as rag_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在启动 RAG 系统，加载向量库和模型...")
    try:
        chain, retriever = create_rag_chain(top_k=4, temperature=0.2)
        app.state.rag_chain = chain
        app.state.rag_retriever = retriever
        logger.info("RAG 系统加载完成，服务已就绪。")
    except Exception as e:
        logger.error(f"RAG 系统加载失败: {e}")
        app.state.rag_chain = None
        app.state.rag_retriever = None
    yield
    logger.info("正在关闭 RAG 系统...")

app = FastAPI(
    title="Trinity-LLM RAG 问答 API",
    description="基于 RAG 检索增强生成的知识库问答系统，支持引用溯源。",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)

@app.get("/")
async def root(request: Request):
    return {
        "service": "Trinity-LLM RAG API",
        "status": "running",
        "rag_ready": request.app.state.rag_chain is not None
    }

@app.get("/health")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "rag_ready": request.app.state.rag_chain is not None
    }