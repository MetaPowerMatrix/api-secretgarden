from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

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