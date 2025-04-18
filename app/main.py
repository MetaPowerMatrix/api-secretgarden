import asyncio
import logging
import uvicorn
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from concurrent import futures

from app.config import settings
from app.api.routes import router as api_router
from app.grpc.server import serve_grpc
from app.services import init_services

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('app.log')  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Python Web Server",
    description="支持REST API和gRPC的Python Web服务器",
    version="0.1.0",
)

# 注册REST API路由
app.include_router(api_router, prefix=settings.API_PREFIX)

# 注册静态文件服务
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

async def start_grpc_server():
    """启动gRPC服务器（在单独的线程中运行）"""
    await asyncio.get_event_loop().run_in_executor(None, serve_grpc)

@app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    logger.info("Starting application...")
    
    # 初始化服务目录
    init_services()
    
    # 启动gRPC服务器
    asyncio.create_task(start_grpc_server())
    logger.info(f"REST API available at http://localhost:{settings.APP_PORT}{settings.API_PREFIX}")
    logger.info(f"gRPC server running on port {settings.GRPC_PORT}")
    logger.info(f"Static files available at http://localhost:{settings.APP_PORT}/static")
    logger.info(f"WebSocket服务正在独立端口运行: ws://localhost:{settings.WEBSOCKET_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行的操作"""
    logger.info("Shutting down application...")

def main():
    """应用主入口"""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=settings.APP_ENV == "development",
    )

if __name__ == "__main__":
    main() 