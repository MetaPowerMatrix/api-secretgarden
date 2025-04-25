"""
DeepSeek大模型服务
负责DeepSeek模型的加载、管理和使用
"""
import os
import logging
import traceback
import torch
from fastapi import HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Any

# 配置日志
logger = logging.getLogger(__name__)

# 全局变量存储加载的DeepSeek模型和tokenizer
model = None
tokenizer = None
loading = False
device = "auto" if torch.cuda.is_available() else "cpu"

def load_model():
    """
    加载DeepSeek-R1模型和tokenizer，只在第一次调用时初始化
    """
    global model, tokenizer, loading, device

    # 避免并发初始化
    if loading:
        return False
        
    if model is not None and tokenizer is not None:
        return True
        
    try:
        loading = True
        logger.info("开始加载DeepSeek-R1模型...")
        
        # 使用绝对路径加载本地模型
        model_path = os.path.abspath("./models")
        
        logger.info(f"加载模型路径: {model_path}")
        
        # 获取 GPU 信息
        n_gpus = torch.cuda.device_count()
        logger.info(f"可用 GPU 数量: {n_gpus}")
        
        # 使用量化配置减少内存占用
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        # 首先加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 加载模型，使用多 GPU 和量化减少内存占用
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,  # 自动分配到可用 GPU
            quantization_config=quantization_config,
            local_files_only=True
        )
        
        logger.info(f"DeepSeek-R1模型已加载到多个 GPU")
        loading = False
        return True
    except Exception as e:
        logger.error(f"加载DeepSeek-R1模型失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        loading = False
        return False

async def chat_with_model(prompt: str, history: List[Dict[str, str]] = None, max_length: int = 2048, 
                         temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
    """
    与DeepSeek模型进行对话
    """
    if not load_model():
        raise HTTPException(status_code=500, detail="无法加载DeepSeek-R1模型")
    
    try:
        # 处理历史记录格式
        messages = []
        
        # 如果有历史消息，添加到对话中
        if history:
            for msg in history:
                if "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加当前用户的消息
        messages.append({"role": "user", "content": prompt})
        
        # 生成回复
        with torch.no_grad():
            # 准备输入
            input_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 编码输入
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # 生成回复
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解码输出
            full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 提取模型的回复（去除输入部分）
            assistant_response = full_response[len(input_text):].strip()
            
            # 更新历史
            new_history = history.copy() if history else []
            new_history.append({"role": "user", "content": prompt})
            new_history.append({"role": "assistant", "content": assistant_response})
            
            return {
                "response": assistant_response,
                "history": new_history
            }
            
    except Exception as e:
        logger.error(f"对话生成失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")

def get_model_status():
    """
    获取DeepSeek模型的加载状态
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
        "model": "deepseek-ai/deepseek-R1"
    }