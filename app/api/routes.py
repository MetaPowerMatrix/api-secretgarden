from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from app.config import settings
import shutil
from datetime import datetime
import json
import logging
import uuid
import requests
import hashlib
import tempfile
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import soundfile as sf
from base64 import b64encode

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 配置日志
logger = logging.getLogger(__name__)

# 全局变量存储加载的Whisper模型和处理器
whisper_model = None
whisper_processor = None
model_loading = False

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
        storage_path = os.path.join(settings.DATA_DIR, "jd/jd_auth.json")
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

class ProductUploadRequest(BaseModel):
    product_id: str

@router.post("/jd/product/upload")
async def upload_product_to_jd(request: ProductUploadRequest):
    """
    上传商品到京东到家平台
    """
    try:
        logger.info(f"开始上传商品: {request.product_id}")
        
        # 1. 加载商品主信息
        main_data_file = os.path.join(settings.DATA_DIR, "jd/database_export.json")
        if not os.path.exists(main_data_file):
            raise HTTPException(status_code=400, detail="商品主数据文件不存在")
        
        with open(main_data_file, "r", encoding="utf-8") as f:
            products = json.load(f)
            
        # 查找指定ID的商品
        product = None
        for p in products:
            if p.get("_id") == request.product_id:
                product = p
                break
                
        if not product:
            raise HTTPException(status_code=404, detail=f"未找到ID为 {request.product_id} 的商品")
        
        # 2. 加载分类信息
        category_file = os.path.join(settings.DATA_DIR, "jd/prod_category.json")
        if not os.path.exists(category_file):
            raise HTTPException(status_code=400, detail="商品分类文件不存在")
            
        with open(category_file, "r", encoding="utf-8") as f:
            categories = json.load(f)
            
        # 查找商品分类
        category_info = None
        for c in categories:
            if c.get("_id") == request.product_id:
                category_info = c
                break
                
        if not category_info or not category_info.get("category_id"):
            raise HTTPException(status_code=400, detail=f"未找到商品 {request.product_id} 的分类信息")
            
        # 3. 加载店内分类信息
        shop_category_file = os.path.join(settings.DATA_DIR, "jd/shop_category.json")
        if not os.path.exists(shop_category_file):
            raise HTTPException(status_code=400, detail="店内分类文件不存在")
            
        with open(shop_category_file, "r", encoding="utf-8") as f:
            shop_categories = json.load(f)
            
        # 查找商品店内分类
        shop_category_info = None
        for sc in shop_categories:
            if sc.get("_id") == request.product_id:
                shop_category_info = sc
                break
        
        # 4. 加载品牌信息
        brand_file = os.path.join(settings.DATA_DIR, "jd/prod_brand.json")
        if not os.path.exists(brand_file):
            raise HTTPException(status_code=400, detail="品牌信息文件不存在")
            
        with open(brand_file, "r", encoding="utf-8") as f:
            brands = json.load(f)
            
        # 查找商品品牌
        brand_info = None
        for b in brands:
            if b.get("_id") == request.product_id:
                brand_info = b
                break
                
        if not brand_info or not brand_info.get("brand_info", {}):
            raise HTTPException(status_code=400, detail=f"未找到商品 {request.product_id} 的品牌信息")
            
        # 5. 构建请求参数
        # 生成唯一的traceId
        trace_id = str(uuid.uuid4()).replace("-", "")
        
        # 获取商品相关数据
        brand_id = brand_info.get("brand_info", {})
        category_id = category_info.get("category_id")
        shop_category_ids = []
        if shop_category_info and shop_category_info.get("category_id"):
            shop_category_ids.append(shop_category_info.get("category_id"))
        
        # 商品主要信息
        sku_name = product.get("title", "")
        # 使用商品ID作为商家商品编码
        out_sku_id = request.product_id
        # 价格转换为分
        price_str = product.get("price", "0")
        try:
            price = int(float(price_str) * 100)  # 转换为分
        except:
            price = 0
            
        # 图片链接处理
        images = []
        if "cover" in product and product["cover"]:
            images.append(product["cover"])
        if "images" in product and isinstance(product["images"], list):
            for img in product["images"]:
                if img not in images:  # 避免重复
                    images.append(img)
        
        # 描述信息
        product_desc = product.get("description", "暂无描述")
        if not product_desc or len(product_desc) < 10:
            product_desc = sku_name + " - " + "秘密花园，隐私发货，品质优选。"  # 确保描述至少10个字符
            
        # 重量信息(默认0.5kg)
        weight = 0.5
        if "weight" in product:
            try:
                weight = float(product["weight"])
            except:
                pass
                
        # 6. 调用京东接口上传商品
        token_file = os.path.join(settings.DATA_DIR, "jd/jd_auth.json")
        if not os.path.exists(token_file):
            raise HTTPException(status_code=400, detail="京东授权Token不存在")
            
        with open(token_file, "r", encoding="utf-8") as f:
            token_data = json.load(f)
            token = token_data.get("token")
            
        if not token:
            raise HTTPException(status_code=400, detail="无效的京东授权Token")
        
        # 京东API配置
        jd_api_url = "https://openapi.jddj.com/djapi/pms/addGoodsV6"
        app_key = os.getenv("JDDJ_APP_KEY")
        app_secret = os.getenv("JDDJ_APP_SECRET")
        
        if not app_key or not app_secret:
            raise HTTPException(status_code=500, detail="缺少API配置")
        
        # 应用级参数
        app_params = {
            "traceId": trace_id,
            "outSkuId": out_sku_id,
            "categoryId": category_id,
            "brandId": brand_id,
            "skuName": sku_name,
            "skuPrice": price,
            "weight": weight,
            "images": images,
            "productDesc": product_desc,
            "ifViewDesc": 0,  # 商品详情在app端展示
            "fixedStatus": 1,  # 上架状态
            "isSale": True,    # 可售
            "transportAttribute": "0",  # 常温
            "liquidStatue": "1",        # 非液体
            "prescripition": "2",       # 非处方药
            "highSingularValue": "1",   # 非高单值
            "isBreakable": "1"          # 非易碎
        }
        
        # 如果有店内分类，添加到参数中
        if shop_category_ids:
            app_params["shopCategories"] = shop_category_ids
        
        # 商品条码
        if "upc" in product and product["upc"]:
            app_params["upc"] = product["upc"]
        
        # 构建系统级参数
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_params = {
            "token": token,
            "app_key": app_key,
            "timestamp": timestamp,
            "format": "json",
            "v": "1.0",
            "jd_param_json": json.dumps(app_params)
        }
        
        # 生成签名
        def generate_sign(params):
            sorted_params = sorted(params.items())
            param_str = "".join([f"{k}{v}" for k, v in sorted_params])
            sign_str = f"{app_secret}{param_str}{app_secret}"
            return hashlib.md5(sign_str.encode('utf-8')).hexdigest().upper()
        
        sign = generate_sign(system_params)
        
        # 完整请求参数
        params = {
            **system_params,
            "sign": sign,
        }
        
        logger.info(f"请求参数: {json.dumps(params, ensure_ascii=False)}")
        
        # 发送请求
        response = requests.get(jd_api_url, params=params)
        result = response.json()
        
        logger.info(f"京东接口返回: {json.dumps(result, ensure_ascii=False)}")
        
        if result.get("code") != "0":
            return {
                "code": result.get("code", "-1"),
                "message": result.get("msg", "上传失败"),
                "data": None
            }
        
        # 保存上传结果
        upload_result = {
            "product_id": request.product_id,
            "sku_id": result.get("data", {}).get("result", {}).get("skuId"),
            "out_sku_id": out_sku_id,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success"
        }
        
        # 将结果保存到上传记录文件
        upload_log_file = os.path.join(settings.DATA_DIR, "jd/upload_log.json")
        upload_logs = []
        
        if os.path.exists(upload_log_file):
            try:
                with open(upload_log_file, "r", encoding="utf-8") as f:
                    upload_logs = json.load(f)
            except:
                upload_logs = []
        
        upload_logs.append(upload_result)
        
        with open(upload_log_file, "w", encoding="utf-8") as f:
            json.dump(upload_logs, f, ensure_ascii=False, indent=2)
            
        return {
            "code": "0",
            "message": "商品上传成功",
            "data": {
                "sku_id": result.get("data", {}).get("result", {}).get("skuId"),
                "out_sku_id": out_sku_id
            }
        }
        
    except HTTPException as e:
        # 直接抛出HTTP异常
        raise e
    except Exception as e:
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        logger.error(f"上传商品时出错: {str(e)}, 出错行号: {line_number}")
        raise HTTPException(status_code=500, detail=f"上传商品时出错: {str(e)}, 出错行号: {line_number}")

class ProductUpdateRequest(BaseModel):
    product_id: str
    update_fields: Dict[str, Any]

@router.post("/jd/product/update")
async def update_product_on_jd(request: ProductUpdateRequest):
    """
    更新京东到家平台商品信息
    """
    try:
        logger.info(f"开始更新商品: {request.product_id}, 更新字段: {request.update_fields}")
        
        # 检查上传记录，获取skuId
        upload_log_file = os.path.join(settings.DATA_DIR, "jd/upload_log.json")
        if not os.path.exists(upload_log_file):
            raise HTTPException(status_code=400, detail="未找到商品上传记录，请先上传商品")
            
        with open(upload_log_file, "r", encoding="utf-8") as f:
            upload_logs = json.load(f)
        
        # 查找商品记录
        product_log = None
        for log in upload_logs:
            if log.get("product_id") == request.product_id:
                product_log = log
                break
                
        if not product_log:
            raise HTTPException(status_code=404, detail=f"未找到商品 {request.product_id} 的上传记录，请先上传商品")
        
        # 获取outSkuId
        out_sku_id = product_log.get("out_sku_id", request.product_id)
        
        # 加载token
        token_file = os.path.join(settings.DATA_DIR, "jd/jd_auth.json")
        if not os.path.exists(token_file):
            raise HTTPException(status_code=400, detail="京东授权Token不存在")
            
        with open(token_file, "r", encoding="utf-8") as f:
            token_data = json.load(f)
            token = token_data.get("token")
            
        if not token:
            raise HTTPException(status_code=400, detail="无效的京东授权Token")
        
        # 京东API配置
        jd_api_url = "https://openapi.jddj.com/djapi/pms/updateSku"
        app_key = os.getenv("JDDJ_APP_KEY")
        app_secret = os.getenv("JDDJ_APP_SECRET")
        
        if not app_key or not app_secret:
            raise HTTPException(status_code=500, detail="缺少API配置")
        
        # 生成唯一的traceId
        trace_id = str(uuid.uuid4()).replace("-", "")
        
        # 构建应用级参数
        app_params = {
            "traceId": trace_id,
            "outSkuId": out_sku_id,
        }
        
        # 添加更新字段
        for key, value in request.update_fields.items():
            # 特殊处理价格字段，需要转换为分
            if key == "skuPrice" and isinstance(value, (int, float, str)):
                try:
                    app_params[key] = int(float(value) * 100)
                except:
                    app_params[key] = value
            else:
                app_params[key] = value
        
        # 构建系统级参数
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_params = {
            "token": token,
            "app_key": app_key,
            "timestamp": timestamp,
            "format": "json",
            "v": "1.0",
            "jd_param_json": json.dumps(app_params)
        }
        
        # 生成签名
        def generate_sign(params):
            sorted_params = sorted(params.items())
            param_str = "".join([f"{k}{v}" for k, v in sorted_params])
            sign_str = f"{app_secret}{param_str}{app_secret}"
            return hashlib.md5(sign_str.encode('utf-8')).hexdigest().upper()
        
        sign = generate_sign(system_params)
        
        # 完整请求参数
        params = {
            **system_params,
            "sign": sign,
        }
        
        logger.info(f"请求参数: {json.dumps(params, ensure_ascii=False)}")
        
        # 发送请求
        response = requests.get(jd_api_url, params=params)
        result = response.json()
        
        logger.info(f"京东接口返回: {json.dumps(result, ensure_ascii=False)}")
        
        if result.get("code") != "0":
            return {
                "code": result.get("code", "-1"),
                "message": result.get("msg", "更新失败"),
                "data": None
            }
        
        # 更新上传记录
        product_log["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        product_log["update_fields"] = request.update_fields
        
        # 保存更新后的记录
        with open(upload_log_file, "w", encoding="utf-8") as f:
            json.dump(upload_logs, f, ensure_ascii=False, indent=2)
            
        return {
            "code": "0",
            "message": "商品更新成功",
            "data": {
                "sku_id": result.get("data", {}).get("result", {}).get("skuId"),
                "out_sku_id": out_sku_id,
                "updated_fields": list(request.update_fields.keys())
            }
        }
        
    except HTTPException as e:
        # 直接抛出HTTP异常
        raise e
    except Exception as e:
        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_number = exc_tb.tb_lineno
        logger.error(f"更新商品时出错: {str(e)}, 出错行号: {line_number}")
        raise HTTPException(status_code=500, detail=f"更新商品时出错: {str(e)}, 出错行号: {line_number}")

from packaging import version

def is_torch_greater_or_equal_than_1_13():
    return version.parse(torch.__version__) >= version.parse("1.13.0")

def load_whisper_model():
    """
    加载Whisper模型和处理器，只在第一次调用时初始化
    """
    global whisper_model, whisper_processor, model_loading
    
    whisper_device = "cuda:1" if torch.cuda.is_available() else "cpu"

    logger.info("准备加载Whisper模型...")

    # 避免并发初始化
    if model_loading:
        return False
        
    if whisper_model is not None and whisper_processor is not None:
        return True
        
    try:
        model_loading = True
        logger.info("开始加载Whisper模型...")
        
        # 加载小型中文模型，也可以选择其他大小的模型
        model_id = "openai/whisper-large-v3"
        
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, 
            use_safetensors=True,
            device_map=whisper_device
        )

        whisper_processor = AutoProcessor.from_pretrained(model_id)

        # 将模型移至GPU（如果可用）
        if torch.cuda.is_available():
            whisper_model = whisper_model.to("cuda")
            logger.info("Whisper模型已加载到GPU")
        else:
            logger.info("Whisper模型已加载到CPU")
            
        model_loading = False
        return True
    except Exception as e:
        logger.error(f"加载Whisper模型失败: {str(e)}")
        model_loading = False
        return False

async def transcribe_audio(audio_file_path, language="zh"):
    """
    使用Whisper模型进行语音转文字
    """
    global whisper_model, whisper_processor
    
    # 确保模型已加载
    if not load_whisper_model():
        raise HTTPException(status_code=500, detail="无法加载Whisper模型")
    
    try:
        # 读取音频文件
        audio_array, sampling_rate = librosa.load(audio_file_path, sr=16000)
        
        # 处理音频
        input_features = whisper_processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features
        
        # 将输入移至GPU（如果可用）
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
        
        # 使用模型生成转录
        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features, language=language)
        
        # 解码预测的token为文本
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
    except Exception as e:
        logger.error(f"语音转文字失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"语音转文字失败: {str(e)}")

class TranscriptionRequest(BaseModel):
    language: Optional[str] = "zh"
    
@router.post("/speech-to-text")
async def speech_to_text(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form("zh")
):
    """
    语音转文字API接口
    接受音频文件，返回转录文本
    支持的格式: WAV, MP3, OGG等常见音频格式
    """
    try:
        # 验证文件类型
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="只接受音频文件")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # 保存上传的音频
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 进行语音转文字
        transcription = await transcribe_audio(temp_file_path, language)
        
        # 在后台任务中删除临时文件
        background_tasks.add_task(os.unlink, temp_file_path)
        
        return {
            "code": 0,
            "message": "语音转文字成功",
            "data": {
                "transcription": transcription,
                "language": language
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"处理语音文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理语音文件失败: {str(e)}")

@router.get("/speech-to-text/status")
async def speech_to_text_status(background_tasks: BackgroundTasks):
    """
    检查Whisper模型加载状态
    """
    global whisper_model, whisper_processor, model_loading
    
    if model_loading:
        status = "loading"
    elif whisper_model is not None and whisper_processor is not None:
        status = "ready"
    else:
        status = "not_loaded"
    
    print(status)

    # 尝试触发模型加载（如果尚未加载）
    if status == "not_loaded":
        # 在后台触发模型加载
        background_tasks.add_task(load_whisper_model)
        status = "loading"
    
    return {
        "status": status,
        "gpu_available": torch.cuda.is_available(),
        "model": "openai/whisper-small"
    }

# 全局变量存储加载的DeepSeek-v3模型和tokenizer
deepseek_model = None
deepseek_tokenizer = None
deepseek_loading = False
deepseek_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_deepseek_model():
    """
    加载DeepSeek-R1模型和tokenizer，只在第一次调用时初始化
    """
    global deepseek_model, deepseek_tokenizer, deepseek_loading, deepseek_device

    # 避免并发初始化
    if deepseek_loading:
        return False
        
    if deepseek_model is not None and deepseek_tokenizer is not None:
        return True
        
    try:
        deepseek_loading = True
        logger.info("开始加载DeepSeek-R1模型...")
        
        # 使用绝对路径加载本地模型
        model_path = os.path.abspath("./models")
        
        logger.info(f"加载模型路径: {model_path}")
        
        # 获取 GPU 信息
        n_gpus = torch.cuda.device_count()
        logger.info(f"可用 GPU 数量: {n_gpus}")
        
        # 使用量化配置减少内存占用
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        # 首先加载tokenizer
        deepseek_tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 加载模型，使用多 GPU 和量化减少内存占用
        deepseek_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",  # 自动分配到可用 GPU
            quantization_config=quantization_config,
            local_files_only=True
        )
        
        logger.info(f"DeepSeek-R1模型已加载到多个 GPU")
        deepseek_loading = False
        return True
    except Exception as e:
        logger.error(f"加载DeepSeek-R1模型失败: {str(e)}")
        deepseek_loading = False
        return False

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[Dict[str, str]]] = []
    max_length: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    
@router.post("/chat")
async def chat_with_model(request: ChatRequest):
    """
    与DeepSeek-v3模型进行对话
    """
    global deepseek_model, deepseek_tokenizer
    
    # 确保模型已加载
    if not load_deepseek_model():
        raise HTTPException(status_code=500, detail="无法加载DeepSeek-v3模型")
    
    try:
        # 处理历史记录格式
        messages = []
        
        # 如果有历史消息，添加到对话中
        if request.history:
            for msg in request.history:
                if "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加当前用户的消息
        messages.append({"role": "user", "content": request.prompt})
        
        # 生成回复
        with torch.no_grad():
            # 准备输入
            input_text = deepseek_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 编码输入
            inputs = deepseek_tokenizer(input_text, return_tensors="pt").to(deepseek_device)
            
            # 生成回复
            outputs = deepseek_model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=deepseek_tokenizer.eos_token_id
            )
            
            # 解码输出
            full_response = deepseek_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 提取模型的回复（去除输入部分）
            assistant_response = full_response[len(input_text):].strip()
            
            # 更新历史
            new_history = request.history.copy() if request.history else []
            new_history.append({"role": "user", "content": request.prompt})
            new_history.append({"role": "assistant", "content": assistant_response})
            
            return {
                "code": 0,
                "message": "对话成功",
                "data": {
                    "response": assistant_response,
                    "history": new_history
                }
            }
            
    except Exception as e:
        logger.error(f"对话生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")

@router.get("/chat/status")
async def chat_model_status(background_tasks: BackgroundTasks):
    """
    检查DeepSeek模型加载状态
    """
    global deepseek_model, deepseek_tokenizer, deepseek_loading
    
    if deepseek_loading:
        status = "loading"
    elif deepseek_model is not None and deepseek_tokenizer is not None:
        status = "ready"
    else:
        status = "not_loaded"
        
    # 尝试触发模型加载（如果尚未加载）
    if status == "not_loaded":
        background_tasks.add_task(load_deepseek_model)
        status = "loading"
    
    return {
        "status": status,
        "device": deepseek_device,
        "gpu_available": torch.cuda.is_available(),
        "model": "deepseek-ai/deepseek-moe-16b-chat"
    }

# 全局变量存储MiniCPM模型和相关组件
minicpm_model = None 
minicpm_tokenizer = None
minicpm_loading = False

def load_minicpm_model():
    """
    加载MiniCPM-o-2_6模型，只在第一次调用时初始化
    """
    global minicpm_model, minicpm_tokenizer, minicpm_loading, deepseek_device
    
    # 避免并发初始化
    if minicpm_loading:
        return False
        
    if minicpm_model is not None and minicpm_tokenizer is not None:
        return True
        
    try:
        minicpm_loading = True
        logger.info("开始加载MiniCPM-o模型...")
        
        # 加载MiniCPM-o-2_6模型
        model_name = "openbmb/MiniCPM-o-2_6"
        
        # 加载tokenizer
        minicpm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 加载模型，使用半精度
        minicpm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=deepseek_device  # 复用已有的设备变量
        )
        
        logger.info(f"MiniCPM-o模型已加载到{deepseek_device.upper()}")
        minicpm_loading = False
        return True
    except Exception as e:
        logger.error(f"加载MiniCPM-o模型失败: {str(e)}")
        minicpm_loading = False
        return False

class VoiceChatRequest(BaseModel):
    history: Optional[List[Dict[str, str]]] = []
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3
    voice_config: Optional[Dict[str, Any]] = {
        "voice_id": "default",
        "speed": 1.0,
        "emotion": "neutral"
    }
    
@router.post("/voice-chat")
async def voice_chat(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    history: Optional[str] = Form("[]"),
    max_new_tokens: Optional[int] = Form(512),
    temperature: Optional[float] = Form(0.3),
    voice_config: Optional[str] = Form('{"voice_id": "default", "speed": 1.0, "emotion": "neutral"}')
):
    """
    端到端语音对话API接口
    直接接收语音输入并返回语音输出，无需中间文本转换
    """
    try:
        # 解析历史记录和语音配置
        chat_history = json.loads(history)
        voice_config_dict = json.loads(voice_config)
        
        # 确保MiniCPM模型已加载
        if not load_minicpm_model():
            raise HTTPException(status_code=500, detail="无法加载MiniCPM-o模型")
        
        # 保存上传的音频到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 加载音频数据
        audio_input, sr = librosa.load(temp_file_path, sr=16000, mono=True)
        
        # 在后台任务中删除临时音频文件
        background_tasks.add_task(os.unlink, temp_file_path)
        
        # 准备对话历史(如果有)
        msgs = []
        for msg in chat_history:
            if "role" in msg and "content" in msg:
                msgs.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加用户的新音频消息
        msgs.append({"role": "user", "content": [audio_input]})
        
        # 添加语音配置的系统提示词
        audio_system_prompt = ""
        if voice_config_dict.get("voice_id") == "male":
            audio_system_prompt = "You are a male voice assistant."
        elif voice_config_dict.get("voice_id") == "female":
            audio_system_prompt = "You are a female voice assistant."
        else:
            audio_system_prompt = "You are a voice assistant."
            
        # 如果有情感或速度设置，添加到提示词
        if voice_config_dict.get("emotion") not in [None, "neutral"]:
            audio_system_prompt += f" Your voice is {voice_config_dict.get('emotion')}."
        if voice_config_dict.get("speed") != 1.0 and voice_config_dict.get("speed") is not None:
            audio_system_prompt += f" You speak at {voice_config_dict.get('speed')} speed."
            
        # 端到端生成语音回复
        try:
            response = minicpm_model.chat(
                msgs=msgs,
                tokenizer=minicpm_tokenizer,
                sampling=True,
                max_new_tokens=max_new_tokens,
                use_tts_template=True,  # 启用TTS模板
                generate_audio=True,    # 生成音频
                temperature=temperature,
                audio_system_prompt=audio_system_prompt,  # 配置语音特性
                stream=False
            )
            
            # 保存回复的音频文件
            output_audio_path = tempfile.mktemp(suffix=".wav")
            sf.write(output_audio_path, response["response_audio"], 16000)
            
            # 读取并编码音频文件
            with open(output_audio_path, "rb") as audio_file:
                encoded_audio = b64encode(audio_file.read()).decode("utf-8")
            
            # 清理临时文件
            background_tasks.add_task(os.unlink, output_audio_path)
            
            # 更新对话历史
            new_history = chat_history.copy()
            new_history.append({"role": "user", "content": "语音输入"})  # 占位符
            new_history.append({"role": "assistant", "content": response["text"]})
            
            return {
                "code": 0,
                "message": "语音对话成功",
                "data": {
                    "text": response["text"],  # 模型生成的文本
                    "audio": encoded_audio,     # base64编码的音频
                    "history": new_history
                }
            }
        
        except Exception as e:
            logger.error(f"生成语音回复失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"生成语音回复失败: {str(e)}")
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="历史记录或语音配置格式无效")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"语音对话处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"语音对话处理失败: {str(e)}")

@router.post("/voice-chat-streaming")
async def voice_chat_streaming(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    history: Optional[str] = Form("[]"),
    max_new_tokens: Optional[int] = Form(512),
    temperature: Optional[float] = Form(0.3),
    voice_config: Optional[str] = Form('{"voice_id": "default", "speed": 1.0, "emotion": "neutral"}')
):
    """
    端到端流式语音对话API接口
    支持流式返回文本结果
    """
    
    async def stream_generator():
        try:
            # 解析历史记录和语音配置
            chat_history = json.loads(history)
            voice_config_dict = json.loads(voice_config)
            
            # 确保MiniCPM模型已加载
            if not load_minicpm_model():
                yield json.dumps({"error": "无法加载MiniCPM-o模型"})
                return
            
            # 保存上传的音频到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # 加载音频数据
            audio_input, sr = librosa.load(temp_file_path, sr=16000, mono=True)
            
            # 在后台任务中删除临时音频文件
            background_tasks.add_task(os.unlink, temp_file_path)
            
            # 准备对话历史
            msgs = []
            for msg in chat_history:
                if "role" in msg and "content" in msg:
                    msgs.append({"role": msg["role"], "content": msg["content"]})
            
            # 添加用户的新音频消息
            msgs.append({"role": "user", "content": [audio_input]})
            
            # 添加语音配置
            audio_system_prompt = f"You are a {voice_config_dict.get('voice_id', 'default')} voice assistant."
            
            # 流式生成文本结果
            generator = minicpm_model.chat(
                msgs=msgs,
                tokenizer=minicpm_tokenizer,
                sampling=True,
                max_new_tokens=max_new_tokens,
                use_tts_template=True,
                generate_audio=False,  # 流式模式下先不生成音频
                temperature=temperature,
                audio_system_prompt=audio_system_prompt,
                stream=True  # 启用流式输出
            )
            
            # 流式输出文本结果
            full_text = ""
            for text_chunk in generator:
                full_text += text_chunk
                yield json.dumps({"text": text_chunk}) + "\n"
            
            # 完成后生成完整的音频回复
            msgs = []
            for msg in chat_history:
                if "role" in msg and "content" in msg:
                    msgs.append({"role": msg["role"], "content": msg["content"]})
            
            msgs.append({"role": "user", "content": [audio_input]})
            
            # 生成最终的音频结果
            response = minicpm_model.chat(
                msgs=msgs,
                tokenizer=minicpm_tokenizer,
                sampling=True,
                max_new_tokens=max_new_tokens,
                use_tts_template=True,
                generate_audio=True,
                temperature=temperature,
                audio_system_prompt=audio_system_prompt,
                stream=False
            )
            
            # 保存回复的音频文件
            output_audio_path = tempfile.mktemp(suffix=".wav")
            sf.write(output_audio_path, response["response_audio"], 16000)
            
            # 读取并编码音频文件
            with open(output_audio_path, "rb") as audio_file:
                encoded_audio = b64encode(audio_file.read()).decode("utf-8")
            
            # 清理临时文件
            background_tasks.add_task(os.unlink, output_audio_path)
            
            # 发送完整的音频结果
            yield json.dumps({"audio": encoded_audio, "complete": True, "full_text": full_text})
            
        except Exception as e:
            logger.error(f"流式语音对话处理失败: {str(e)}")
            yield json.dumps({"error": f"流式语音对话处理失败: {str(e)}"})
    
    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@router.get("/minicpm/status")
async def minicpm_status(background_tasks: BackgroundTasks):
    """
    检查MiniCPM-o模型加载状态
    """
    global minicpm_model, minicpm_tokenizer, minicpm_loading
    
    if minicpm_loading:
        status = "loading"
    elif minicpm_model is not None and minicpm_tokenizer is not None:
        status = "ready"
    else:
        status = "not_loaded"
        
    # 尝试触发模型加载（如果尚未加载）
    if status == "not_loaded":
        background_tasks.add_task(load_minicpm_model)
        status = "loading"
    
    return {
        "status": status,
        "device": deepseek_device,
        "gpu_available": torch.cuda.is_available(),
        "model": "openbmb/MiniCPM-o-2_6",
        "capabilities": [
            "端到端语音对话",
            "语音输入到语音输出",
            "可配置声音特性",
            "流式文本响应"
        ]
    }

@router.post("/megatts/upload-voice")
async def upload_voice_reference(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    voice_id: str = Form(...)
):
    """
    上传声音参考文件
    用于零样本声音克隆
    """
    try:
        # 创建voices目录（如果不存在）
        os.makedirs("voices", exist_ok=True)
        
        # 保存上传的音频到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # 从音频提取潜在表示
        try:
            # 加载音频
            audio_array, sr = librosa.load(temp_file_path, sr=22050, mono=True)
            
            # 使用音频分词器提取潜在表示
            audio_tokenizer = AutoTokenizer.from_pretrained("/root/smart-yolo/Megatts/checkpoints/g2p/")
            with torch.no_grad():
                # 将音频转换为张量
                audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0).to(megatts_device)
                
                # 提取潜在表示
                latent = audio_tokenizer.encode(audio_tensor)
                
                # 保存潜在表示
                latent_np = latent.cpu().numpy()
                np.save(f"voices/{voice_id}.npy", latent_np)
                
                # 同时保存原始音频作为参考
                import shutil
                shutil.copy(temp_file_path, f"voices/{voice_id}.wav")
        
        except Exception as e:
            logger.error(f"声音特征提取失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"声音特征提取失败: {str(e)}")
        finally:
            # 在后台任务中删除临时文件
            background_tasks.add_task(os.unlink, temp_file_path)
        
        return {
            "code": 0,
            "message": "声音参考上传并处理成功",
            "data": {
                "voice_id": voice_id
            }
        }
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"声音参考处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"声音参考处理失败: {str(e)}")
