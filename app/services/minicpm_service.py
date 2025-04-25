"""
MiniCPM模型服务
负责MiniCPM模型的加载、管理和使用
"""
import os
import logging
import traceback
import torch
from fastapi import HTTPException
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any

# 配置日志
logger = logging.getLogger(__name__)

# 全局变量存储MiniCPM模型和相关组件
model = None 
tokenizer = None
loading = False
device = "auto" if torch.cuda.is_available() else "cpu"

def load_model():
    """
    加载MiniCPM-o-2_6模型，只在第一次调用时初始化
    """
    global model, tokenizer, loading, device
    
    # 避免并发初始化
    if loading:
        return False
        
    if model is not None and tokenizer is not None:
        return True
        
    try:
        loading = True
        logger.info("开始加载MiniCPM-o模型...")
        
        # 加载MiniCPM-o-2_6模型
        model_name = "openbmb/MiniCPM-o-2_6"

        # 注意这里使用flash_attention_2替代sdpa来避免滑动窗口注意力警告
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',  # 使用flash_attention_2替代sdpa
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=False,
            init_tts=False
        )

        model = model.eval().to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        logger.info(f"MiniCPM-o模型已加载到{device.upper()}")
        loading = False
        return True
    except Exception as e:
        logger.error(f"加载MiniCPM-o模型失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        loading = False
        return False

async def voice_chat(audio_input, chat_history=None, max_new_tokens=512, temperature=0.3, 
                   voice_config=None, generate_audio=True, stream=False):
    """
    使用MiniCPM模型进行语音对话
    """
    if not load_model():
        raise HTTPException(status_code=500, detail="无法加载MiniCPM-o模型")
    
    try:
        # 准备对话历史(如果有)
        msgs = []
        if chat_history:
            for msg in chat_history:
                if "role" in msg and "content" in msg:
                    msgs.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加用户的新音频消息
        msgs.append({"role": "user", "content": [audio_input]})
        
        # 添加语音配置的系统提示词
        audio_system_prompt = ""
        if voice_config:
            if voice_config.get("voice_id") == "male":
                audio_system_prompt = "You are a male voice assistant."
            elif voice_config.get("voice_id") == "female":
                audio_system_prompt = "You are a female voice assistant."
            else:
                audio_system_prompt = "You are a voice assistant."
                
            # 如果有情感或速度设置，添加到提示词
            if voice_config.get("emotion") not in [None, "neutral"]:
                audio_system_prompt += f" Your voice is {voice_config.get('emotion')}."
            if voice_config.get("speed") != 1.0 and voice_config.get("speed") is not None:
                audio_system_prompt += f" You speak at {voice_config.get('speed')} speed."
        
        # 流式生成或一次性生成
        if stream:
            return model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=max_new_tokens,
                use_tts_template=True,
                generate_audio=False,  # 流式模式下先不生成音频
                temperature=temperature,
                audio_system_prompt=audio_system_prompt,
                stream=True
            )
        else:
            # 端到端生成语音回复
            response = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=max_new_tokens,
                use_tts_template=True, 
                generate_audio=generate_audio,
                temperature=temperature,
                audio_system_prompt=audio_system_prompt,
                stream=False
            )
            return response
        
    except Exception as e:
        logger.error(f"生成语音回复失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"生成语音回复失败: {str(e)}")

def get_model_status():
    """
    获取MiniCPM模型的加载状态
    """
    global model, tokenizer, loading
    
    if loading:
        status = "loading"
    elif model is not None and tokenizer is not None:
        status = "ready"
    else:
        status = "not_loaded"
    
    return {
        "status": status,
        "device": device,
        "gpu_available": torch.cuda.is_available(),
        "model": "openbmb/MiniCPM-o-2_6",
        "capabilities": [
            "端到端语音对话",
            "语音输入到语音输出",
            "可配置声音特性",
            "流式文本响应"
        ]
    }