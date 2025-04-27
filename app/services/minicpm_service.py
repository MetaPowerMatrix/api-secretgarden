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
import librosa

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
            attn_implementation='sdpa',  # 使用flash_attention_2替代sdpa
            torch_dtype=torch.bfloat16,
        )

        model = model.eval().cuda()

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        model.init_tts()
        model.tts.float()

        logger.info(f"MiniCPM-o模型已加载到{device.upper()}")
        loading = False
        return True
    except Exception as e:
        logger.error(f"加载MiniCPM-o模型失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        loading = False
        return False

async def voice_chat(audio_input, ref_audio, output_audio_path, max_new_tokens=128, temperature=0.3, history=None):
    """
    使用MiniCPM模型进行语音对话
    """
    if not load_model():
        raise HTTPException(status_code=500, detail="无法加载MiniCPM-o模型")
    
    try:
        ref_audio, _ = librosa.load(ref_audio, sr=16000, mono=True) # load the reference audio
        sys_prompt = model.get_sys_prompt(ref_audio=ref_audio, mode='audio_roleplay', language='zh')

        # round one
        user_question = {'role': 'user', 'content': [librosa.load(audio_input, sr=16000, mono=True)[0]]}
        msgs = [sys_prompt, user_question]
        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=max_new_tokens,
            use_tts_template=True,
            generate_audio=True,
            temperature=temperature,
            output_audio_path=output_audio_path,
        )
        # history = msgs.append({'role': 'assistant', 'content': res})
        
        return res
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