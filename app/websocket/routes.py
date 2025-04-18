import json
import logging
from typing import List, Dict, Optional, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

# 存储活跃的WebSocket连接
active_connections: List[WebSocket] = []

# 音频数据缓冲字典，用于存储每个客户端的音频片段
audio_buffers = {}
# 录音会话标识
recording_sessions = {}

# 对话历史
conversation_history = []

# ===== WebSocket代理相关变量 =====
# 前端客户端连接管理
frontend_clients: Dict[str, WebSocket] = {}
# AI后端连接管理
ai_backend: Optional[WebSocket] = None
# 前端会话ID到客户端映射
session_to_client: Dict[str, str] = {}
# 客户端ID到会话ID映射
client_to_session: Dict[str, str] = {}
# 会话音频数据缓冲
session_audio_buffers: Dict[str, bytearray] = {}
# 等待AI处理的会话队列
pending_sessions: Set[str] = set()


class ConnectionManager:
    """管理WebSocket连接"""
    
    async def connect(self, websocket: WebSocket):
        """建立WebSocket连接"""
        await websocket.accept()
        active_connections.append(websocket)
        logger.info(f"WebSocket连接建立，当前连接数: {len(active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """关闭WebSocket连接"""
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket连接关闭，当前连接数: {len(active_connections)}")
    
    async def broadcast(self, message: Dict):
        """广播消息到所有连接"""
        message_json = json.dumps(message, ensure_ascii=False)
        for connection in active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"发送消息失败: {str(e)}")
    
    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """发送个人消息"""
        message_json = json.dumps(message, ensure_ascii=False)
        await websocket.send_text(message_json)

# 创建连接管理器实例
connection_manager = ConnectionManager()

@router.websocket("/proxy")
async def proxy_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket代理端点
    连接双方：
    1. 前端客户端 - 发送音频数据，接收AI处理后的结果
    2. AI后端 - 接收前端发送的音频数据，处理后返回结果
    """
    await websocket.accept()
    
    # 首先确定连接类型（前端或AI后端）
    try:
        # 等待连接标识消息
        init_message = await websocket.receive_text()
        logger.info(f"初始化消息: {init_message}")
        init_data = json.loads(init_message)
        
        if "client_type" not in init_data:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "缺少客户端类型标识"
            }))
            await websocket.close()
            return
            
        client_type = init_data["client_type"]
        
        if client_type == "ai_backend":
            # 处理AI后端连接
            global ai_backend
            
            # 如果已有AI后端连接，拒绝新连接
            if ai_backend is not None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "已存在AI后端连接"
                }))
                await websocket.close()
                return
                
            # 设置全局AI后端连接
            ai_backend = websocket
            logger.info("AI后端已连接")
            
            # 向AI后端发送确认消息
            await websocket.send_text(json.dumps({
                "type": "status",
                "content": "连接成功"
            }))
            
            try:
                # 监听来自AI后端的消息
                while True:
                    try:
                        # 在接收消息前记录日志
                        logger.debug(f"准备接收来自客户端的消息")
                        message = await websocket.receive()
                        
                        # 检查消息类型
                        if "text" in message:
                            # 解析JSON消息
                            try:
                                data = json.loads(message["text"])
                                
                                if "session_id" in data and "type" in data:
                                    session_id = data["session_id"]
                                    
                                    # 查找对应的前端客户端
                                    if session_id in session_to_client:
                                        client_id = session_to_client[session_id]
                                        
                                        if client_id in frontend_clients:
                                            frontend_ws = frontend_clients[client_id]
                                            
                                            # 转发消息给前端
                                            if data["type"] == "text":
                                                # 文本消息
                                                await frontend_ws.send_text(json.dumps({
                                                    "type": "text",
                                                    "content": data["content"]
                                                }))
                                            elif data["type"] == "processing_complete":
                                                # 处理完成消息
                                                if session_id in pending_sessions:
                                                    pending_sessions.remove(session_id)
                                                
                                                # 通知前端处理完成
                                                await frontend_ws.send_text(json.dumps({
                                                    "type": "status",
                                                    "content": "处理完成"
                                                }))
                                            
                                            logger.info(f"已将AI处理的音频数据块转发至前端客户端 {client_id}, 大小: {len(audio_data)} 字节")
                                        else:
                                            logger.warning(f"找不到客户端ID: {client_id}")
                                    else:
                                        logger.warning(f"找不到会话ID: {session_id}")
                                else:
                                    logger.warning("AI后端消息缺少session_id或type")
                                    
                            except json.JSONDecodeError:
                                logger.error("无法解析AI后端发送的JSON消息")
                        
                        elif "bytes" in message:
                            # 处理二进制数据（音频）
                            binary_data = message["bytes"]
                            
                            # 从二进制数据中提取会话ID（前8字节）
                            if len(binary_data) > 16:
                                # 提取会话ID（假设会话ID是UUID格式，存储在前16字节）
                                session_id_bytes = binary_data[:16]
                                audio_data = binary_data[16:]
                                
                                try:
                                    # 将字节转换为UUID字符串
                                    session_id = str(uuid.UUID(bytes=session_id_bytes))
                                    logger.info(f"接收到AI后端音频数据: {len(audio_data)} 字节, 会话ID: {session_id}")
                                        
                                    # 查找对应的前端客户端
                                    if session_id in session_to_client:
                                        client_id = session_to_client[session_id]
                                        
                                        if client_id in frontend_clients:
                                            frontend_ws = frontend_clients[client_id]
                                            
                                            # 直接转发音频数据到前端
                                            await frontend_ws.send_bytes(audio_data)
                                            logger.info(f"已将AI处理的音频数据转发至前端客户端 {client_id}")
                                        else:
                                            logger.warning(f"找不到客户端ID: {client_id}")
                                    else:
                                        logger.warning(f"转发音频数据失败,找不到会话ID: {session_id}")
                                except ValueError:
                                    logger.error("无法解析会话ID")
                            else:
                                logger.error("音频数据格式不正确")
                    except WebSocketDisconnect:
                        logger.info("AI后端断开连接")
                        break
            except Exception as e:
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                line_number = exc_tb.tb_lineno
                logger.error(f"AI后端连接错误: {str(e)}, 出错行号: {line_number}")
            finally:
                # 清理AI后端连接
                ai_backend = None
                logger.info("AI后端连接已关闭")
                
        elif client_type == "frontend":
            # 处理前端客户端连接
            
            # 生成客户端ID和会话ID
            client_id = f"client_{id(websocket)}"
            session_id = str(uuid.uuid4())
            
            # 记录映射关系
            frontend_clients[client_id] = websocket
            session_to_client[session_id] = client_id
            client_to_session[client_id] = session_id
            
            # 初始化音频缓冲区
            session_audio_buffers[session_id] = bytearray()
            
            logger.info(f"前端客户端已连接: ID={client_id}, 会话ID={session_id}")
            
            # 向前端发送会话信息
            await websocket.send_text(json.dumps({
                "type": "session_info",
                "content": {
                    "session_id": session_id,
                    "client_id": client_id
                }
            }))
            
            try:
                # 监听来自前端的消息
                while True:
                    try:
                        # 在接收消息前记录日志
                        logger.debug(f"准备接收来自客户端{client_id}的消息")
                        message = await websocket.receive()
                        
                        # 检查消息类型
                        if "bytes" in message:
                            # 接收音频数据块
                            audio_data = message["bytes"]
                            # logger.info(f"接收到前端音频数据: {len(audio_data)} 字节")
                            
                            # 将数据添加到缓冲区
                            session_audio_buffers[session_id].extend(audio_data)
                            
                        elif "text" in message:
                            # 解析JSON消息
                            try:
                                data = json.loads(message["text"])
                                
                                if "command" in data:
                                    command = data["command"]
                                    
                                    if command == "audio_complete":
                                        # 前端发送完所有音频数据
                                        if len(session_audio_buffers[session_id]) > 0:
                                            logger.info(f"前端音频传输完成，准备转发到AI后端处理，总大小: {len(session_audio_buffers[session_id])} 字节")
                                            
                                            # 检查AI后端是否连接
                                            if ai_backend is None:
                                                await websocket.send_text(json.dumps({
                                                    "type": "error",
                                                    "content": "AI后端未连接，无法处理请求"
                                                }))
                                                continue
                                            
                                            # 转发音频数据到AI后端
                                            # 创建包含会话ID的二进制数据包
                                            # 会话ID转为二进制
                                            complete_audio_data = bytes(session_audio_buffers[session_id])
                                            session_id_bytes = uuid.UUID(session_id).bytes
                                            data_with_session = session_id_bytes + complete_audio_data

                                            # 加入等待处理队列
                                            pending_sessions.add(session_id)
                                            
                                            # 通知AI后端新的音频处理请求
                                            # await ai_backend.send_text(json.dumps({
                                            #     "type": "new_audio",
                                            #     "session_id": session_id,
                                            #     "audio_size": len(complete_audio_data)
                                            # }))
                                            
                                            # 发送音频数据
                                            await ai_backend.send_bytes(data_with_session)
                                            
                                            # 清空缓冲区，准备下一次录音
                                            session_audio_buffers[session_id] = bytearray()
                                            
                                            # 通知前端
                                            # await websocket.send_text(json.dumps({
                                            #     "type": "status",
                                            #     "content": "音频已转发至AI后端处理"
                                            # }))
                                        else:
                                            await websocket.send_text(json.dumps({
                                                "type": "error",
                                                "content": "没有接收到音频数据"
                                            }))
                                        
                                    elif command == "cancel_processing":
                                        # 取消正在处理的请求
                                        if session_id in pending_sessions:
                                            # 通知AI后端取消处理
                                            if ai_backend is not None:
                                                await ai_backend.send_text(json.dumps({
                                                    "type": "cancel_processing",
                                                    "session_id": session_id
                                                }))
                                                
                                            pending_sessions.remove(session_id)
                                            
                                            await websocket.send_text(json.dumps({
                                                "type": "status",
                                                "content": "处理请求已取消"
                                            }))
                                
                            except json.JSONDecodeError:
                                logger.error("无法解析前端发送的JSON消息")
                            
                    except WebSocketDisconnect:
                        logger.info(f"前端客户端断开连接: {client_id}")
                        break
            except Exception as e:
                import sys
                exc_type, exc_obj, exc_tb = sys.exc_info()
                line_number = exc_tb.tb_lineno
                logger.error(f"前端客户端连接错误: {str(e)}, 出错行号: {line_number}")
            finally:
                # 清理前端客户端资源
                if client_id in frontend_clients:
                    del frontend_clients[client_id]
                
                if client_id in client_to_session:
                    session_id = client_to_session[client_id]
                    
                    # 如果会话还在处理队列中，通知AI后端取消处理
                    if session_id in pending_sessions and ai_backend is not None:
                        try:
                            await ai_backend.send_text(json.dumps({
                                "type": "cancel_processing",
                                "session_id": session_id
                            }))
                        except:
                            pass
                            
                        pending_sessions.remove(session_id)
                    
                    # 清理会话相关资源
                    if session_id in session_to_client:
                        del session_to_client[session_id]
                    
                    if session_id in session_audio_buffers:
                        del session_audio_buffers[session_id]
                        
                    del client_to_session[client_id]
                
                logger.info(f"前端客户端资源已清理: {client_id}")
                logger.debug(f"删除会话映射: 会话ID={session_id}")
        else:
            # 未知客户端类型
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"未知的客户端类型: {client_type}"
            }))
            await websocket.close()
            
    except WebSocketDisconnect:
        logger.info("WebSocket连接断开")
    except json.JSONDecodeError:
        logger.error("无法解析客户端初始化消息")
        await websocket.close()
    except Exception as e:
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        logger.error(f"WebSocket代理错误: {str(e)}, 出错行号: {line_number}")
        await websocket.close()
    