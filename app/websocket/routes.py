import json
import logging
import asyncio
import os
import datetime
import wave
import numpy as np
import requests
from typing import List, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from openai import OpenAI
import tempfile
import io
import edge_tts
import aiofiles
from pydub import AudioSegment

from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# 存储活跃的WebSocket连接
active_connections: List[WebSocket] = []

# 音频数据缓冲字典，用于存储每个客户端的音频片段
audio_buffers = {}
# 录音会话标识
recording_sessions = {}

# 初始化OpenAI客户端
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# 对话历史
conversation_history = []

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

# 音频处理相关函数
async def save_raw_to_wav(raw_data, wav_file_path):
    """将原始PCM数据保存为WAV文件"""
    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(settings.ESP32_CHANNELS)
        wav_file.setsampwidth(settings.ESP32_SAMPLE_WIDTH)
        wav_file.setframerate(settings.ESP32_SAMPLE_RATE)
        wav_file.writeframes(raw_data)
    return wav_file_path

async def transcribe_audio(audio_file_path):
    """使用OpenAI Whisper API将语音转换为文本"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="zh"  # 指定中文
            )
        return transcript.text
    except Exception as e:
        logger.error(f"语音识别错误: {e}")
        return None

async def get_deepseek_response(prompt):
    """调用DeepSeek API获取回复"""
    try:
        # 保存对话历史以维持上下文
        global conversation_history
        
        # 构建完整消息历史
        messages = conversation_history + [{"role": "user", "content": prompt}]
        
        headers = {
            "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800,
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]
            
            # 更新对话历史
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # 保持对话历史在合理长度
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
            return assistant_response
        else:
            logger.error(f"DeepSeek API错误: {response.status_code}")
            logger.error(response.text)
            return "抱歉，我无法获取回复。请稍后再试。"
    except Exception as e:
        logger.error(f"调用DeepSeek API错误: {e}")
        return "抱歉，处理您的请求时出现了错误。"

async def text_to_speech(text):
    """将文本转换为语音并确保返回原始PCM格式"""
    try:
        # 使用edge-tts进行TTS转换
        communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
        
        # 首先将TTS数据收集到临时文件
        temp_wav_path = os.path.join(settings.PROCESSED_DIR, f"temp_tts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
        
        # 使用edge-tts的write_to_file方法直接写入文件
        await communicate.save(temp_wav_path)
        logger.info(f"TTS生成的原始WAV文件已保存: {temp_wav_path}")
        
        # 使用pydub加载WAV文件
        audio = AudioSegment.from_wav(temp_wav_path)
        
        # 转换为ESP32兼容的格式
        audio = audio.set_frame_rate(settings.ESP32_SAMPLE_RATE)
        audio = audio.set_channels(settings.ESP32_CHANNELS)
        audio = audio.set_sample_width(settings.ESP32_SAMPLE_WIDTH)
        
        # 获取原始PCM数据
        pcm_data = audio.raw_data
        
        # 可选：保存转换后的PCM为WAV文件(用于调试)
        converted_wav_path = os.path.join(settings.PROCESSED_DIR, f"converted_tts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
        with wave.open(converted_wav_path, 'wb') as wav_file:
            wav_file.setnchannels(settings.ESP32_CHANNELS)
            wav_file.setsampwidth(settings.ESP32_SAMPLE_WIDTH)
            wav_file.setframerate(settings.ESP32_SAMPLE_RATE)
            wav_file.writeframes(pcm_data)
        
        logger.info(f"转换后的PCM数据大小: {len(pcm_data)}字节")
        return pcm_data
        
    except Exception as e:
        logger.error(f"TTS错误: {e}")
        return None

async def process_audio(raw_audio_data):
    """处理音频数据的完整流程"""
    try:
        # 生成唯一文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_file_path = os.path.join(settings.AUDIO_DIR, f"audio_{timestamp}.wav")
        
        # 将原始数据保存为WAV文件
        await save_raw_to_wav(raw_audio_data, wav_file_path)
        logger.info(f"已保存WAV文件: {wav_file_path}")
        
        # 转录音频
        transcript = await transcribe_audio(wav_file_path)
        if not transcript:
            return None, "抱歉，无法识别您的语音。"
        
        logger.info(f"语音识别结果: {transcript}")
        
        # 如果识别到的文本太短或无意义，则跳过处理
        if len(transcript.strip()) < 2:
            return None, "抱歉，没有识别到有效的语音内容。"
        
        # 获取DeepSeek回复
        ai_response = await get_deepseek_response(transcript)
        logger.info(f"AI回复: {ai_response}")
        
        # 生成语音回复
        audio_response = await text_to_speech(ai_response)
        
        # 如果成功生成语音
        if audio_response:
            return audio_response, ai_response
        
        return None, ai_response
    except Exception as e:
        logger.error(f"处理音频流程出错: {e}")
        return None, "处理请求时发生错误。"

@router.websocket(settings.WEBSOCKET_PATH)
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    # 生成客户端ID
    client_id = f"client_{id(websocket)}"
    
    # 建立连接
    await connection_manager.connect(websocket)
    
    # 初始化客户端的音频缓冲区
    audio_buffers[client_id] = bytearray()
    # 初始化客户端的录音会话标识
    recording_sessions[client_id] = None
    
    try:
        while True:
            # 接收消息，支持文本和二进制数据
            message = await websocket.receive()
            
            # 检查消息类型 (二进制或文本)
            if "bytes" in message:
                # 接收音频数据块
                audio_data = message["bytes"]
                logger.info(f"接收到音频数据块: {len(audio_data)} 字节")
                
                # 将数据添加到缓冲区
                audio_buffers[client_id].extend(audio_data)
                logger.info(f"当前音频缓冲区大小: {len(audio_buffers[client_id])} 字节")
                
            elif "text" in message:
                text_data = message["text"]
                # 尝试解析为JSON (IMU数据或其他命令)
                try:
                    # 解析JSON数据
                    data = json.loads(text_data)
                    logger.info(f"收到JSON消息: {data}")
                    
                    if "imu" in data:
                        # 这是IMU数据
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        imu_file = os.path.join(settings.IMU_DIR, f"imu_{timestamp}.json")
                        
                        with open(imu_file, "w") as f:
                            f.write(text_data)
                        
                        logger.info(f"接收到IMU数据并保存")
                    elif "command" in data:
                        # 处理命令消息
                        command = data["command"]
                        if command == "reset_conversation":
                            conversation_history.clear()
                            await connection_manager.send_personal_message(
                                {"type": "status", "content": "对话已重置"}, 
                                websocket
                            )
                        elif command == "get_audio_params":
                            # 发送音频参数信息
                            await connection_manager.send_personal_message({
                                "type": "audio_params", 
                                "content": {
                                    "sample_rate": settings.ESP32_SAMPLE_RATE,
                                    "channels": settings.ESP32_CHANNELS,
                                    "sample_width": settings.ESP32_SAMPLE_WIDTH
                                }
                            }, websocket)
                        elif command == "recording_stopped":
                            # 处理完整的录音数据
                            if len(audio_buffers[client_id]) > 0:
                                # 保存完整音频数据
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                complete_audio_data = bytes(audio_buffers[client_id])
                                
                                # 保存原始音频数据
                                raw_file = os.path.join(settings.AUDIO_DIR, f"complete_audio_{timestamp}.raw")
                                with open(raw_file, "wb") as f:
                                    f.write(complete_audio_data)
                                
                                logger.info(f"接收到完整录音: {len(complete_audio_data)} 字节, 已保存到 {raw_file}")
                                
                                # 处理音频并获取回复
                                audio_response, text_response = await process_audio(complete_audio_data)
                                
                                # 发送文本回复
                                await connection_manager.send_personal_message(
                                    {"type": "text", "content": text_response}, 
                                    websocket
                                )
                                
                                # 如果有音频回复，发送PCM音频数据
                                if audio_response:
                                    # 发送原始PCM数据
                                    await websocket.send_bytes(audio_response)
                                    logger.info(f"已发送PCM音频回复: {len(audio_response)} 字节")
                                
                                # 清空缓冲区，准备下一次录音
                                audio_buffers[client_id] = bytearray()
                                logger.info("音频缓冲区已清空，准备下一次录音")
                            else:
                                logger.warning("收到录音停止命令，但缓冲区为空")
                                await connection_manager.send_personal_message(
                                    {"type": "error", "content": "没有接收到音频数据"}, 
                                    websocket
                                )
                    elif "heartbeat" in data:
                        logger.debug(f"接收到心跳: {data['heartbeat']}")
                        # 可选：发送心跳响应
                        await connection_manager.send_personal_message(
                            {"type": "heartbeat", "content": "pong"}, 
                            websocket
                        )
                    elif "content" in data:
                        # 标准消息处理
                        logger.info(f"收到标准消息: {data}")
                        
                        response = {
                            "type": "response",
                            "content": data.get("content", ""),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        # 根据消息类型决定是广播还是私人消息
                        if data.get("broadcast", False):
                            await connection_manager.broadcast(response)
                        else:
                            await connection_manager.send_personal_message(response, websocket)
                    else:
                        logger.warning(f"接收到未知的JSON数据: {data}")
                except json.JSONDecodeError:
                    # 如果不是有效的JSON，发送错误消息
                    logger.warning(f"接收到非JSON文本数据: {text_data}")
                    await connection_manager.send_personal_message(
                        {"type": "error", "content": "无效的JSON格式"}, 
                        websocket
                    )
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket错误: {str(e)}")
    finally:
        # 处理可能的未完成录音数据
        if client_id in audio_buffers and len(audio_buffers[client_id]) > 0:
            try:
                logger.info(f"连接关闭前处理剩余音频数据: {len(audio_buffers[client_id])} 字节")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_file = os.path.join(settings.AUDIO_DIR, f"incomplete_audio_{timestamp}.raw")
                with open(raw_file, "wb") as f:
                    f.write(bytes(audio_buffers[client_id]))
            except Exception as e:
                logger.error(f"处理未完成音频数据出错: {e}")
        
        # 清理资源
        if client_id in audio_buffers:
            del audio_buffers[client_id]
        if client_id in recording_sessions:
            del recording_sessions[client_id]
            
        # 断开连接
        await connection_manager.disconnect(websocket) 