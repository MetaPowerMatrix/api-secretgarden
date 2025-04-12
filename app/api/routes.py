from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from fastapi.responses import JSONResponse
from app.config import settings
import shutil
from datetime import datetime
import json
import logging

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()

class Message(BaseModel):
    content: str

class JDAuthToken(BaseModel):
    token: str
    expires_in: int
    time: int
    uid: Optional[str] = ""
    user_nick: str
    venderId: str

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
                    "fileName": original_filename,
                    "fileUrl": "https://static.kalaisai.com/extra-images/" + original_filename
                },
                "message": "图片上传成功"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")
    finally:
        file.file.close()

@router.post("/jd/auth/callback")
async def jd_auth_callback(token: str = Form(...)):
    """
    接收京东授权码的回调接口
    """
    try:
        print(f"收到京东授权回调请求，原始token数据: {token}")
        
        # 解析token字符串为JSON对象
        token_data = json.loads(token)
        print(f"解析后的token数据: {json.dumps(token_data, ensure_ascii=False, indent=2)}")
        
        # 验证数据格式
        auth_token = JDAuthToken(**token_data)
        print(f"数据验证通过，token信息: 用户={auth_token.user_nick}, 商家ID={auth_token.venderId}, 有效期={auth_token.expires_in}秒")
        
        # 存储授权信息
        storage_path = os.path.join(settings.DATA_DIR, "jd_auth.json")
        print(f"准备将授权信息保存到文件: {storage_path}")
        
        with open(storage_path, "w", encoding="utf-8") as f:
            json.dump(token_data, f, ensure_ascii=False, indent=2)
        print("授权信息保存成功")
        
        response = {
            "code": "0",
            "msg": "success",
            "data": ""
        }
        print(f"返回响应: {json.dumps(response, ensure_ascii=False)}")
        
        return response
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
        raise HTTPException(status_code=400, detail="无效的token格式")
    except Exception as e:
        print(f"处理授权码时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理授权码时出错: {str(e)}")