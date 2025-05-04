import logging
import uvicorn
from fastapi import FastAPI
from app.config import settings
from app.websocket.routes import router as ws_router
from app.services import init_services

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('websocket.log')  # 输出到单独的日志文件
    ]
)
logger = logging.getLogger("websocket")

# 创建WebSocket应用
ws_app = FastAPI(
    title="WebSocket Server",
    description="WebSocket服务器，用于音频数据代理",
    version="0.1.0",
)

# 注册WebSocket路由
ws_app.include_router(ws_router, prefix=settings.WEBSOCKET_PATH)

@ws_app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    logger.info("Starting WebSocket server...")
    # 初始化服务目录
    init_services()
    
    logger.info(f"WebSocket server running on port {settings.WEBSOCKET_PORT}")

@ws_app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行的操作"""
    logger.info("Shutting down WebSocket server...")

def main():
    """WebSocket服务器主入口"""
    uvicorn.run(
        "app.server:ws_app",
        host="0.0.0.0",
        port=settings.WEBSOCKET_PORT,
        reload=settings.APP_ENV == "development",
        timeout_keep_alive=120,        # 将保持连接活跃的超时时间设为120秒
        ws_ping_interval=30,           # 将WebSocket ping间隔设为30秒
        ws_ping_timeout=30,            # 将WebSocket ping超时设为30秒
    )

if __name__ == "__main__":
    main()