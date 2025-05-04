from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Body
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
import librosa
import soundfile as sf
from base64 import b64encode

# 导入模型服务
from app.services.whisper_service import transcribe_audio, get_model_status as get_whisper_status, load_model as load_whisper_model
# from app.services.deepseek_service import chat_with_v30324 as deepseek_chat, get_model_status as get_deepseek_status
from app.services.minicpm_service import voice_chat as minicpm_voice_chat, get_model_status as get_minicpm_status
from app.services.qwen_service import get_model_status as get_qwen_status, chat_with_model as qwen_chat

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
        if file.content_type is not None:
            if not file.content_type.startswith('audio/'):
                # 检查文件扩展名
                filename = file.filename.lower()
                audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
                is_audio = any(filename.endswith(ext) for ext in audio_extensions)
                
                if not is_audio:
                    raise HTTPException(status_code=400, detail="只接受音频文件")
        else:
            # 当 content_type 为 None 时，检查文件扩展名
            filename = file.filename.lower() if file.filename else ""
            audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
            is_audio = any(filename.endswith(ext) for ext in audio_extensions)
            
            if not is_audio:
                raise HTTPException(status_code=400, detail="只接受音频文件且未提供内容类型")
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # 保存上传的音频
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="上传的文件为空")
                
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
    status_info = get_whisper_status()
    
    # 尝试触发模型加载（如果尚未加载）
    if status_info["status"] == "not_loaded":
        # 在后台触发模型加载
        background_tasks.add_task(load_whisper_model)
        status_info["status"] = "loading"
    
    return status_info

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
    try:
        # 调用deepseek_service中的chat_with_model函数
        result = await deepseek_chat(
            prompt=request.prompt,
            history=request.history,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "code": 0,
            "message": "对话成功",
            "data": result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"对话生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"对话生成失败: {str(e)}")

@router.get("/chat/status")
async def chat_model_status(background_tasks: BackgroundTasks):
    """
    检查DeepSeek模型加载状态
    """
    status_info = get_deepseek_status()
    
    # 尝试触发模型加载（如果尚未加载）
    if status_info["status"] == "not_loaded":
        from app.services.deepseek_service import load_model
        background_tasks.add_task(load_model)
        status_info["status"] = "loading"
    
    return status_info

class VoiceChatRequest(BaseModel):
    audio_input: str
    ref_audio: str
    output_audio_path: str
    session_id: str
    
@router.post("/voice-chat")
async def voice_chat(
    request: VoiceChatRequest = Body(...),
):
    """
    端到端语音对话API接口
    直接接收语音输入并返回语音输出，无需中间文本转换
    """
    try:
        # 使用minicpm_service进行语音对话
        response = await minicpm_voice_chat(
            audio_input=request.audio_input,
            ref_audio=request.ref_audio,
            output_audio_path=request.output_audio_path,
        )
        
        return {
            "code": 0,
            "message": "语音对话成功",
            "data": {
                "text": response,  # 模型生成的文本
                "output_audio_path": request.output_audio_path,
            }
        }
    
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
            
            # 保存上传的音频到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # 加载音频数据
            audio_input, sr = librosa.load(temp_file_path, sr=16000, mono=True)
            
            # 在后台任务中删除临时音频文件
            background_tasks.add_task(os.unlink, temp_file_path)
            
            # 流式生成文本结果
            generator = await minicpm_voice_chat(
                audio_input=audio_input,
                chat_history=chat_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                voice_config=voice_config_dict,
                generate_audio=False,
                stream=True
            )
            
            # 流式输出文本结果
            full_text = ""
            for text_chunk in generator:
                full_text += text_chunk
                yield json.dumps({"text": text_chunk}) + "\n"
            
            # 完成后生成完整的音频回复
            response = await minicpm_voice_chat(
                audio_input=audio_input,
                chat_history=chat_history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                voice_config=voice_config_dict,
                generate_audio=True,
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
    status_info = get_minicpm_status()
    
    # 尝试触发模型加载（如果尚未加载）
    if status_info["status"] == "not_loaded":
        from app.services.minicpm_service import load_model
        background_tasks.add_task(load_model)
        status_info["status"] = "loading"
    
    return status_info

@router.post("/chat/qwen")
async def chat_with_qwen(request: ChatRequest):
    """
    与Qwen-QwQ-32B模型进行对话
    """
    try:
        # 调用qwen_service中的chat_with_model函数
        result = await qwen_chat(
            prompt=request.prompt,
            history=request.history,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "code": 0,
            "message": "对话成功",
            "data": result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Qwen对话生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Qwen对话生成失败: {str(e)}")

@router.get("/qwen/status")
async def qwen_status(background_tasks: BackgroundTasks):
    """
    检查Qwen模型加载状态
    """
    status_info = get_qwen_status()
    
    # 尝试触发模型加载（如果尚未加载）
    if status_info["status"] == "not_loaded":
        from app.services.qwen_service import load_model
        background_tasks.add_task(load_model)
        status_info["status"] = "loading"
    
    return status_info
