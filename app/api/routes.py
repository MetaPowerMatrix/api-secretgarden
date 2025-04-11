from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any
import os
from fastapi.responses import JSONResponse
from app.config import settings
import shutil
from datetime import datetime

router = APIRouter()

class Message(BaseModel):
    content: str

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """健康检查接口"""
    return {"status": "ok"}

@router.post("/message")
async def create_message(message: Message) -> Dict[str, Any]:
    """示例POST接口：创建消息"""
    return {
        "id": "msg_123456",
        "content": message.content,
        "created_at": "2023-10-28T12:00:00Z"
    }

@router.get("/message/{message_id}")
async def get_message(message_id: str) -> Dict[str, Any]:
    """示例GET接口：获取消息"""
    # 在实际应用中，这里会从数据库查询消息
    if message_id != "msg_123456":
        raise HTTPException(status_code=404, detail="消息不存在")
    
    return {
        "id": message_id,
        "content": "示例消息内容",
        "created_at": "2023-10-28T12:00:00Z"
    }

@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片文件
    """
    try:
        # 检查文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只允许上传图片文件")

        # 使用原始文件名
        original_filename = file.filename
        
        # 完整的文件路径
        file_path = os.path.join(settings.IMAGE_STORAGE_DIR, original_filename)
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse(
            status_code=200,
            content={
                "code": 0,
                "data": {
                    "filename": original_filename,
                    "fileUrl": "https://static.kalaisai.com/extra-images/" + original_filename
                },
                "message": "图片上传成功"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")
    finally:
        file.file.close() 