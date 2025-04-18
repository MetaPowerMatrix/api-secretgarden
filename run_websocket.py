#!/usr/bin/env python3
import os
import sys
import uvicorn

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.server import ws_app

if __name__ == "__main__":
    uvicorn.run(ws_app, host="0.0.0.0", port=8001)