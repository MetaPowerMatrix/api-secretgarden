import os
import json
import requests
import logging
import hashlib
import time
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JDDJCategoryTool:
    def __init__(self):
        self.base_url = "https://openapi.jddj.com/djapi/api/queryChildCategoriesForOP"
        self.data_dir = "/data/app/jd"  # 修改为指定目录
        os.makedirs(self.data_dir, exist_ok=True)
        self.app_key = os.getenv("JDDJ_APP_KEY")
        self.app_secret = os.getenv("JDDJ_APP_SECRET")
        
        if not self.app_key or not self.app_secret:
            raise ValueError("请在.env文件中设置JDDJ_APP_KEY和JDDJ_APP_SECRET")
        
        # 请求频率控制
        self.max_requests_per_minute = 45
        self.request_timestamps = []
        self.request_interval = 60.0 / self.max_requests_per_minute  # 每次请求的最小间隔（秒）
        
    def wait_for_rate_limit(self):
        """等待直到可以发送下一个请求"""
        current_time = time.time()
        
        # 清理超过1分钟的时间戳
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts < 60.0]
        
        # 如果已经达到限制，等待
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60.0 - (current_time - self.request_timestamps[0])
            if wait_time > 0:
                logger.info(f"达到请求频率限制，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
                current_time = time.time()
        
        # 如果距离上次请求时间太短，等待
        if self.request_timestamps:
            time_since_last = current_time - self.request_timestamps[-1]
            if time_since_last < self.request_interval:
                wait_time = self.request_interval - time_since_last
                logger.info(f"请求间隔太短，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
        
        # 记录当前请求时间
        self.request_timestamps.append(time.time())
        
    def load_token(self) -> Optional[str]:
        """从文件中加载token"""
        token_file = os.path.join(self.data_dir, "jd_auth.json")
        if not os.path.exists(token_file):
            logger.error("未找到token文件，请先运行授权流程")
            return None
            
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                token_data = json.load(f)
                return token_data.get("token")
        except Exception as e:
            logger.error(f"读取token文件失败: {str(e)}")
            return None

    def generate_sign(self, params: Dict) -> str:
        """生成签名"""
        # 按参数名升序排列
        sorted_params = sorted(params.items())
        # 拼接参数
        param_str = "".join([f"{k}{v}" for k, v in sorted_params])
        # 添加app_secret
        sign_str = f"{self.app_secret}{param_str}{self.app_secret}"
        # MD5加密
        return hashlib.md5(sign_str.encode('utf-8')).hexdigest().upper()

    def get_categories(self, parent_id: str = "0", level: int = 1) -> List[Dict]:
        """获取指定父级ID下的子类目"""
        token = self.load_token()
        if not token:
            return []

        try:
            # 等待直到可以发送请求
            self.wait_for_rate_limit()
            
            # 系统级参数
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            system_params = {
                "token": token,
                "app_key": self.app_key,
                "timestamp": timestamp,
                "format": "json",
                "v": "1.0",
                # 应用级参数
                "jd_param_json": json.dumps({"id": parent_id, "fields": [
                    "ID", "CATEGORY_NAME", "CATEGORY_LEVEL",
                    "CHECK_UPC_STATUS", "WEIGHT_MARK", "PACKAGE_FEE_MARK", "LEAF"
                ]})
            }
            
            # 生成签名（只使用系统级参数）
            sign = self.generate_sign(system_params)
            
            # 完整的请求参数
            params = {
                **system_params,
                "sign": sign,
            }
            
            logger.info(f"请求参数: {json.dumps(params, ensure_ascii=False)}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"响应结果: {json.dumps(result, ensure_ascii=False)}")
            
            if result.get("code") != "0":
                logger.error(f"获取类目失败: {result.get('msg')}")
                return []
                
            # 提取类目数据
            categories = result.get("data", {}).get("result", [])
            logger.info(f"获取到{level}级类目数量: {len(categories)}")
            
            # 递归获取子类目
            for category in categories:
                if level < 5 and category.get("leaf") != 1:  # 最多获取5级类目，且不是末级类目
                    category["children"] = self.get_categories(
                        str(category.get("id", "")),
                        level + 1
                    )
                    
            return categories
            
        except Exception as e:
            logger.error(f"获取类目时出错: {str(e)}")
            return []

    def save_categories(self, categories: List[Dict]):
        """保存类目信息到文件"""
        try:
            output_file = os.path.join(self.data_dir, "category.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(categories, f, ensure_ascii=False, indent=2)
            logger.info(f"类目信息已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存类目信息失败: {str(e)}")

def main():
    """主函数"""
    tool = JDDJCategoryTool()
    logger.info("开始获取京东到家商品类目...")
    
    # 从根类目开始获取
    categories = tool.get_categories()
    
    if categories:
        tool.save_categories(categories)
        logger.info("类目获取完成")
    else:
        logger.error("获取类目失败")

if __name__ == "__main__":
    main()