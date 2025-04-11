import os
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 应用环境：development, production, testing
APP_ENV = os.getenv("APP_ENV", "development")

# FastAPI服务端口
APP_PORT = int(os.getenv("APP_PORT", 8000))

# gRPC服务端口
GRPC_PORT = int(os.getenv("GRPC_PORT", 50051))

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 其他配置项
API_PREFIX = "/api/v1"
WEBSOCKET_PATH = "/ws"

# API密钥配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 音频处理配置
# 存储音频和IMU数据的目录
DATA_DIR = os.getenv("DATA_DIR", "esp32_data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
IMU_DIR = os.path.join(DATA_DIR, "imu")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# 确保目录存在
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMU_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ESP32音频参数设置
ESP32_SAMPLE_RATE = int(os.getenv("ESP32_SAMPLE_RATE", 44100))  # ESP32使用的采样率，需匹配ESP32的I2S配置
ESP32_CHANNELS = int(os.getenv("ESP32_CHANNELS", 1))            # 单声道
ESP32_SAMPLE_WIDTH = int(os.getenv("ESP32_SAMPLE_WIDTH", 2))    # 16位