import logging
import traceback
import torch
from fastapi import HTTPException
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig

# 配置日志
logger = logging.getLogger(__name__)

# 全局变量
model = None
tokenizer = None
loading = False
device = "auto" if torch.cuda.is_available() else "cpu"

# 检查Gryphe/MythoMax-L2-13b模型加载状态
def get_uncensored_status():
    """
    检查Gryphe/MythoMax-L2-13b模型加载状态
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
        "model": "Gryphe/MythoMax-L2-13b"
    }

def load_uncensored_model():
    """
    加载Gryphe/MythoMax-L2-13b模型
    """
    global model, tokenizer, loading, device

    model_name = "Austism/chronos-hermes-13b"

    # 避免并发初始化
    if loading:
        return False

    if model is not None and tokenizer is not None:
        return True

    try:
        loading = True
        logger.info("开始加载Gryphe/MythoMax-L2-13b模型...")

        # 使用量化配置减少内存占用
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device,
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device
        )

        logger.info(f"Gryphe/MythoMax-L2-13b模型已加载到{device.upper()}")
        loading = False
        return True
    except Exception as e:
        logger.error(f"加载Gryphe/MythoMax-L2-13b模型失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        loading = False
        return False

async def chat_with_uncensored(prompt: str):
    """
    与Gryphe/MythoMax-L2-13b模型进行对话（异步版本）
    """
    global model, tokenizer, device

    if not load_uncensored_model():
        raise HTTPException(status_code=500, detail="无法加载Gryphe/MythoMax-L2-13b模型")

    # 将输入移动到CUDA设备
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    # 生成回复
    generate_ids = model.generate(inputs.input_ids, max_length=2048)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return {
        "response": result,
        "history": []
    }

