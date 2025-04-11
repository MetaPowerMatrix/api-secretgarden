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

def vosk_speech_to_text(audio_path, model_path="vosk-model-cn-0.22"):
    """
    使用Vosk离线识别（需提前下载语言模型）
    :param model_path: 模型目录路径（中文小模型）
    """
    # 加载模型（需从https://alphacephei.com/vosk/models下载）
    model = Model(model_path)
    
    with wave.open(audio_path, 'rb') as wf:
        recognizer = KaldiRecognizer(model, wf.getframerate())
        
        text = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text.append(result.get("text", ""))
        
        final_result = json.loads(recognizer.FinalResult())
        text.append(final_result.get("text", ""))
        
    return " ".join(text)

async def save_raw_to_wav(raw_data, wav_file_path):
    """将原始PCM数据保存为WAV文件"""
    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(ESP32_CHANNELS)
        wav_file.setsampwidth(ESP32_SAMPLE_WIDTH)
        wav_file.setframerate(ESP32_SAMPLE_RATE)
        wav_file.writeframes(raw_data)
    return wav_file_path

async def get_deepseek_response(prompt):
    """调用DeepSeek API获取回复"""
    try:
        # 保存对话历史以维持上下文
        global conversation_history
        
        # 构建完整消息历史
        messages = conversation_history + [{"role": "user", "content": prompt}]
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
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
            print(f"DeepSeek API错误: {response.status_code}")
            print(response.text)
            return "抱歉，我无法获取回复。请稍后再试。"
    except Exception as e:
        print(f"调用DeepSeek API错误: {e}")
        return "抱歉，处理您的请求时出现了错误。"

async def text_to_speech(text):
    try:
        # 生成临时文件路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = os.path.join(PROCESSED_DIR, f"temp_tts_{timestamp}.mp3")
        print(f"mp3生成临时文件路径: {temp_path}")
        
        # 1. 生成MP3
        communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
        await communicate.save(temp_path)
        
        # 2. 用pydub处理
        audio = AudioSegment.from_file(temp_path, format="mp3")
        audio = audio.set_frame_rate(ESP32_SAMPLE_RATE)
        audio = audio.set_channels(ESP32_CHANNELS)
        audio = audio.set_sample_width(ESP32_SAMPLE_WIDTH)
        
        # 3. 导出为PCM
        pcm_data = audio.raw_data
        print(f"pcm_data: {len(pcm_data)}")

        # 调试：验证数据有效性
        assert len(pcm_data) % (ESP32_SAMPLE_WIDTH * ESP32_CHANNELS) == 0, "无效的PCM数据长度"
        
        return pcm_data
        
    except Exception as e:
        print(f"TTS错误: {e}")
        return None
    # finally:
        # if 'temp_path' in locals() and os.path.exists(temp_path):
            # os.remove(temp_path)

async def process_audio(raw_audio_data, websocket):
    """处理音频数据的完整流程"""
    try:
        # 生成唯一文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_file_path = os.path.join(AUDIO_DIR, f"audio_{timestamp}.wav")
        
        # 将原始数据保存为WAV文件
        await save_raw_to_wav(raw_audio_data, wav_file_path)
        print(f"已保存WAV文件: {wav_file_path}")
        
        # 转录音频
        transcript = vosk_speech_to_text(wav_file_path)
        if not transcript:
            return None, "抱歉，无法识别您的语音。"
        
        print(f"语音识别结果: {transcript}")
        await websocket.send(json.dumps({"type": "text", "content": transcript}))
        
        # 获取DeepSeek回复
        ai_response = await get_deepseek_response(transcript)
        print(f"ai_response: {ai_response}")
        await websocket.send(json.dumps({"type": "text", "content": ai_response}))

        # 生成语音回复
        audio_response = await text_to_speech(ai_response)
        print(f"audio_response: {len(audio_response)}")

        # 如果成功生成语音
        if audio_response:
            return audio_response, ai_response
        
        return None, ai_response
    except Exception as e:
        print(f"处理音频流程出错: {e}")
        return None, "处理请求时发生错误。"

@router.websocket(settings.WEBSOCKET_PATH)
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接端点"""
    # 生成客户端ID
    client_id = f"client_{id(websocket)}"
    
    # 注册新客户端
    print(f"客户端已连接: {websocket.remote_address} (ID: {client_id})")
    # 建立连接
    await connection_manager.connect(websocket)
    
    # 初始化客户端的音频缓冲区
    audio_buffers[client_id] = bytearray()
    # 初始化客户端的录音会话标识
    recording_sessions[client_id] = None
    
    try:
        async for message in websocket:
            # 检查消息类型 (二进制或文本)
            if isinstance(message, bytes):
                # 接收音频数据块
                print(f"接收到音频数据块: {len(message)} 字节")
                
                # 将数据添加到缓冲区
                audio_buffers[client_id].extend(message)
                print(f"当前音频缓冲区大小: {len(audio_buffers[client_id])} 字节")
                
                # 检查是否需要处理完整音频 (通过ESP32发送的停止录音命令来触发)
                # 这里不做处理，等待录音停止命令
                
            else:
                # 尝试解析为JSON (IMU数据或其他命令)
                try:
                    data = json.loads(message)
                    if "imu" in data:
                        # 这是IMU数据
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        imu_file = os.path.join(IMU_DIR, f"imu_{timestamp}.json")
                        
                        with open(imu_file, "w") as f:
                            f.write(message)
                        
                        print(f"接收到IMU数据: {message}")
                    elif "command" in data:
                        # 处理命令消息
                        command = data["command"]
                        if command == "reset_conversation":
                            conversation_history.clear()
                            await websocket.send(json.dumps({"type": "status", "content": "对话已重置"}))
                        elif command == "get_audio_params":
                            # 发送音频参数信息
                            await websocket.send(json.dumps({
                                "type": "audio_params", 
                                "content": {
                                    "sample_rate": ESP32_SAMPLE_RATE,
                                    "channels": ESP32_CHANNELS,
                                    "sample_width": ESP32_SAMPLE_WIDTH
                                }
                            }))
                        elif command == "recording_stopped":
                            # 处理完整的录音数据
                            if len(audio_buffers[client_id]) > 0:
                                # 保存完整音频数据
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                complete_audio_data = bytes(audio_buffers[client_id])
                                
                                # 保存原始音频数据
                                raw_file = os.path.join(AUDIO_DIR, f"complete_audio_{timestamp}.raw")
                                with open(raw_file, "wb") as f:
                                    f.write(complete_audio_data)
                                
                                print(f"接收到完整录音: {len(complete_audio_data)} 字节, 已保存到 {raw_file}")
                                
                                # 处理音频并获取回复
                                audio_response, text_response = await process_audio(complete_audio_data, websocket)
                                
                                # 发送文本回复（调试用）
                                # await websocket.send(json.dumps({"type": "text", "content": text_response}))
                                
                                # 如果有音频回复，发送PCM音频数据
                                
                                if audio_response:
                                    total_audio_length = len(audio_response)
                                    while total_audio_length > 0:
                                        send_audio = audio_response[0:5120]
                                        print(f"开始发送PCM音频回复: {len(send_audio)} 字节")
                                        await websocket.send(send_audio)
                                        total_audio_length -= len(send_audio)
                                        audio_response = audio_response[len(send_audio):]
                                        await asyncio.sleep(0.05)
                                
                                # 清空缓冲区，准备下一次录音
                                audio_buffers[client_id] = bytearray()
                                print("音频缓冲区已清空，准备下一次录音")
                            else:
                                print("收到录音停止命令，但缓冲区为空")
                                await websocket.send(json.dumps({"type": "error", "content": "没有接收到音频数据"}))
                    elif "heartbeat" in data:
                        print(f"接收到心跳: {data['heartbeat']}")
                    elif "type" in data and data["type"] == "audio_metadata":
                        print(f"接收到音频元数据: {data}")
                    else:
                        print(f"接收到未知的JSON数据: {message}")
                except json.JSONDecodeError:
                    print(f"接收到未知的文本数据: {message}")

    except websockets.exceptions.ConnectionClosed:
        print("连接已关闭")
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