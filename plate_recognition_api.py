#!/usr/bin/env python3
"""
车牌识别API服务
基于已训练好的模型提供RESTful API接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pathlib import Path
import json
import base64
import io
import cv2
from typing import List, Dict, Any
from pydantic import BaseModel
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 车牌字符映射
CHARACTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '京', '津', '冀', '晋', '蒙', '辽', '吉', '黑',
    '沪', '苏', '浙', '皖', '闽', '赣', '鲁', '豫',
    '鄂', '湘', '粤', '桂', '琼', '渝', '川', '贵',
    '云', '藏', '陕', '甘', '青', '宁', '新', '使',
    '领', '警', '学', '港', '澳'
]

# 车牌类型
PLATE_TYPES = ['蓝牌', '黄牌', '绿牌', '白牌', '黑牌', '警车', '军车', '使馆', '教练车']

class PlateRecognitionModel(nn.Module):
    """车牌识别模型"""
    def __init__(self, num_chars=74, max_length=8, num_plate_types=9):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_plate_types = num_plate_types

        # 使用ResNet34作为骨干网络
        import torchvision.models as models
        resnet = models.resnet34(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Sigmoid()
        )

        # 特征增强
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 字符分类器 (简化版)
        self.char_classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_chars)
        )

        # 类型分类器 (根据实际模型参数)
        self.type_classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_plate_types)
        )

        # 位置编码 (根据实际形状)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, 128))

    def forward(self, x):
        batch_size = x.size(0)

        # 骨干网络特征提取
        features = self.backbone(x)  # [B, 512, H, W]

        # 简化为64通道特征
        reduced_features = F.adaptive_avg_pool2d(features, (8, 8))
        reduced_features = reduced_features.mean(dim=1, keepdim=True)  # 减少到64通道

        # 特征增强
        enhanced_features = self.feature_enhancement(reduced_features)

        # 全局平均池化用于类型分类
        global_features = enhanced_features.mean(dim=[2, 3])  # [B, 64]

        # 序列特征用于字符分类
        seq_features = F.adaptive_avg_pool2d(enhanced_features, (self.max_length, 1))
        seq_features = seq_features.squeeze(-1).transpose(1, 2)  # [B, L, C]

        # 添加位置编码
        seq_features = seq_features + self.positional_encoding

        # 分类
        char_logits = self.char_classifier(seq_features)
        type_logits = self.type_classifier(global_features)

        return char_logits, type_logits

class RecognitionResult(BaseModel):
    """识别结果模型"""
    plate_number: str
    plate_type: str
    confidence: float
    processing_time: float

# 创建FastAPI应用
app = FastAPI(title="车牌识别系统", description="基于深度学习的中国车牌识别API服务")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局变量
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = None

def load_model():
    """加载预训练模型"""
    global model, transform

    try:
        model_path = "best_fast_high_accuracy_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        model = PlateRecognitionModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # 定义图像预处理
        transform = transforms.Compose([
            transforms.Resize((96, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"模型加载成功，设备: {device}")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False

def decode_plate_number(char_logits):
    """解码车牌号码"""
    char_probs = torch.softmax(char_logits, dim=-1)
    char_indices = torch.argmax(char_probs, dim=-1)

    plate_chars = []
    for idx in char_indices[0]:
        if idx < len(CHARACTERS):
            plate_chars.append(CHARACTERS[idx])

    return ''.join(plate_chars[:8])  # 最多8个字符

def get_plate_type(type_logits):
    """获取车牌类型"""
    type_probs = torch.softmax(type_logits, dim=-1)
    type_idx = torch.argmax(type_probs, dim=-1)

    if type_idx < len(PLATE_TYPES):
        return PLATE_TYPES[type_idx]
    return "未知"

def recognize_plate(image: Image.Image) -> Dict[str, Any]:
    """识别车牌"""
    import time
    start_time = time.time()

    try:
        # 图像预处理
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_tensor = transform(image).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            char_logits, type_logits = model(img_tensor)

        # 解码结果
        plate_number = decode_plate_number(char_logits)
        plate_type = get_plate_type(type_logits)

        # 计算置信度
        char_probs = torch.softmax(char_logits, dim=-1)
        confidence = torch.max(char_probs).item()

        processing_time = time.time() - start_time

        return {
            "plate_number": plate_number,
            "plate_type": plate_type,
            "confidence": confidence,
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"识别失败: {e}")
        return {
            "plate_number": "识别失败",
            "plate_type": "未知",
            "confidence": 0.0,
            "processing_time": 0.0,
            "error": str(e)
        }

# API端点
@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    load_model()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>车牌识别系统</title>
        <meta charset="utf-8">
        <style>
            body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); padding: 40px; text-align: center; }
            h1 { color: #333; margin-bottom: 30px; font-size: 2.5em; }
            .btn { display: inline-block; background: linear-gradient(45deg, #667eea, #764ba2); color: white; text-decoration: none; padding: 15px 30px; border-radius: 25px; margin: 10px; font-size: 1.1em; transition: all 0.3s ease; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
            .info { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 车牌识别系统</h1>
            <div class="info">
                <p>基于深度学习的智能车牌识别解决方案</p>
                <p>支持单张图片识别、批量处理和实时视频识别</p>
            </div>
            <div>
                <a href="/web" class="btn">🎯 进入Web界面</a>
                <a href="/docs" class="btn">📚 API文档</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Web界面"""
    static_file = Path("static/index.html")
    if static_file.exists():
        return FileResponse(static_file)
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Web界面未找到</title>
        <meta charset="utf-8">
        <style>
            body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center; max-width: 500px; }
            h1 { color: #d32f2f; margin-bottom: 20px; }
            .error { background: #ffebee; padding: 20px; border-radius: 10px; border-left: 4px solid #d32f2f; margin: 20px 0; }
            .back { display: inline-block; background: #667eea; color: white; text-decoration: none; padding: 10px 20px; border-radius: 5px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>❌ Web界面未找到</h1>
            <div class="error">
                <p>Web界面文件不存在，请检查static/index.html文件是否存在</p>
            </div>
            <a href="/" class="back">返回首页</a>
        </div>
    </body>
    </html>
    """

@app.post("/recognize", response_model=RecognitionResult)
async def recognize_plate_api(file: UploadFile = File(...)):
    """单张图片识别接口"""
    if not model:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 识别车牌
        result = recognize_plate(image)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return RecognitionResult(
            plate_number=result["plate_number"],
            plate_type=result["plate_type"],
            confidence=result["confidence"],
            processing_time=result["processing_time"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

@app.post("/recognize_batch")
async def recognize_batch_api(files: List[UploadFile] = File(...)):
    """批量识别接口"""
    if not model:
        raise HTTPException(status_code=500, detail="模型未加载")

    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            result = recognize_plate(image)
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"results": results}

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    return {
        "device": str(device),
        "model_loaded": model is not None,
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "max_file_size": "10MB"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)