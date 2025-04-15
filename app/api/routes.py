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
import uuid
import requests
import hashlib

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
                
        if not brand_info or not brand_info.get("brand_info", {}).get("brandId"):
            raise HTTPException(status_code=400, detail=f"未找到商品 {request.product_id} 的品牌信息")
            
        # 5. 构建请求参数
        # 生成唯一的traceId
        trace_id = str(uuid.uuid4()).replace("-", "")
        
        # 获取商品相关数据
        brand_id = brand_info.get("brand_info", {}).get("brandId")
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
            product_desc = sku_name + "。" + "这是一个优质商品。" * 3  # 确保描述至少10个字符
            
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
        logger.error(f"上传商品时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传商品时出错: {str(e)}")
