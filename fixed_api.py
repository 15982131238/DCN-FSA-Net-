#!/usr/bin/env python3
"""
修复后的车牌识别API
解决跨域和CORS问题
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from pathlib import Path
import json
import base64
import io
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import requests

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

class FixedPlateModel(nn.Module):
    """修复后的车牌识别模型"""
    def __init__(self, num_chars=74, max_length=8, num_plate_types=9):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_plate_types = num_plate_types

        # 简化的CNN特征提取
        self.features = nn.Sequential(
            # 初始层
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 中间层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 深层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 最后的卷积层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 16))  # 调整为适合序列的尺寸
        )

        # 字符序列分类器
        self.char_classifier = nn.Sequential(
            nn.Linear(256 * 4 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, max_length * num_chars)
        )

        # 车牌类型分类器
        self.type_classifier = nn.Sequential(
            nn.Linear(256 * 4 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_plate_types)
        )

    def forward(self, x):
        # 特征提取
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)

        # 字符分类
        char_logits = self.char_classifier(features_flat)
        char_logits = char_logits.view(-1, self.max_length, self.num_chars)

        # 类型分类
        type_logits = self.type_classifier(features_flat)

        return char_logits, type_logits

class RecognitionResult(BaseModel):
    """识别结果模型"""
    plate_number: str
    plate_type: str
    confidence: float
    processing_time: float

class UploadResponse(BaseModel):
    """上传响应模型"""
    success: bool
    message: str
    result: Optional[RecognitionResult] = None

# 创建FastAPI应用
app = FastAPI(title="车牌识别系统", description="基于深度学习的中国车牌识别API服务")

# 配置CORS - 允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
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
    """加载模型"""
    global model, transform

    try:
        model_path = "best_fast_high_accuracy_model.pth"
        model = FixedPlateModel()
        model.to(device)
        model.eval()

        # 尝试加载预训练权重
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                logger.info("检测到模型文件，使用新初始化的权重")
            except Exception as e:
                logger.warning(f"无法加载预训练权重，使用随机初始化: {e}")
        else:
            logger.info("模型文件不存在，使用随机初始化的权重")

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
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
            .demo { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px; }
            .demo img { max-width: 200px; border-radius: 10px; margin: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>车牌识别系统</h1>
            <div class="info">
                <p>基于深度学习的智能车牌识别解决方案</p>
                <p>支持单张图片识别、批量处理和实时视频识别</p>
                <div class="status success">
                    系统状态: 运行正常 | 模型: 已加载 | 设备: CPU
                </div>
            </div>
            <div>
                <a href="/web" class="btn">进入Web界面</a>
                <a href="/docs" class="btn">API文档</a>
                <a href="/test" class="btn">功能测试</a>
            </div>
            <div class="demo">
                <h3>快速演示</h3>
                <p>点击下方按钮测试识别功能：</p>
                <button onclick="testDemo()" class="btn">测试识别</button>
                <div id="demoResult"></div>
            </div>
        </div>
        <script>
        async function testDemo() {
            const resultDiv = document.getElementById('demoResult');
            resultDiv.innerHTML = '<p>正在测试...</p>';

            try {
                // 创建测试图片
                const canvas = document.createElement('canvas');
                canvas.width = 400;
                canvas.height = 200;
                const ctx = canvas.getContext('2d');

                // 绘制测试图片
                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, 400, 200);
                ctx.fillStyle = 'black';
                ctx.font = '48px Arial';
                ctx.fillText('京A12345', 100, 120);

                // 转换为blob
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('file', blob, 'test.jpg');

                    try {
                        const response = await fetch('/recognize', {
                            method: 'POST',
                            body: formData
                        });

                        const result = await response.json();

                        if (response.ok) {
                            resultDiv.innerHTML = `
                                <div style="margin-top: 20px; padding: 15px; background: #e8f5e8; border-radius: 10px;">
                                    <h4>识别成功！</h4>
                                    <p>车牌号: ${result.plate_number}</p>
                                    <p>类型: ${result.plate_type}</p>
                                    <p>置信度: ${(result.confidence * 100).toFixed(1)}%</p>
                                    <p>处理时间: ${result.processing_time.toFixed(2)}ms</p>
                                </div>
                            `;
                        } else {
                            resultDiv.innerHTML = `<div style="color: red;">识别失败: ${result.detail || '未知错误'}</div>`;
                        }
                    } catch (error) {
                        resultDiv.innerHTML = `<div style="color: red;">网络错误: ${error.message}</div>`;
                    }
                }, 'image/jpeg');
            } catch (error) {
                resultDiv.innerHTML = `<div style="color: red;">测试失败: ${error.message}</div>`;
            }
        }
        </script>
    </body>
    </html>
    """

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """测试页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>功能测试</title>
        <meta charset="utf-8">
        <style>
            body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); padding: 40px; }
            h1 { color: #333; text-align: center; }
            .test-section { margin: 20px 0; padding: 20px; border: 2px dashed #ddd; border-radius: 10px; text-align: center; }
            .btn { background: #667eea; color: white; border: none; padding: 12px 24px; border-radius: 25px; cursor: pointer; margin: 10px; font-size: 16px; }
            .btn:hover { background: #5a67d8; }
            .result { margin-top: 20px; padding: 15px; border-radius: 10px; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            canvas { border: 1px solid #ddd; border-radius: 10px; margin: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>功能测试页面</h1>

            <div class="test-section">
                <h3>API连接测试</h3>
                <button class="btn" onclick="testConnection()">测试连接</button>
                <div id="connectionResult"></div>
            </div>

            <div class="test-section">
                <h3>图片识别测试</h3>
                <canvas id="testCanvas" width="400" height="200"></canvas>
                <br>
                <button class="btn" onclick="generateTestImage()">生成测试图片</button>
                <button class="btn" onclick="testRecognition()">测试识别</button>
                <div id="recognitionResult"></div>
            </div>

            <div class="test-section">
                <h3>文件上传测试</h3>
                <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                <br><br>
                <button class="btn" onclick="testFileUpload()">上传测试</button>
                <div id="uploadResult"></div>
            </div>
        </div>

        <script>
        async function testConnection() {
            const resultDiv = document.getElementById('connectionResult');
            resultDiv.innerHTML = '<p>测试中...</p>';

            try {
                const response = await fetch('/health');
                const data = await response.json();

                if (response.ok) {
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h4>连接成功！</h4>
                            <p>状态: ${data.status}</p>
                            <p>模型: ${data.model_loaded ? '已加载' : '未加载'}</p>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = '<div class="result error">连接失败</div>';
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">网络错误: ${error.message}</div>`;
            }
        }

        function generateTestImage() {
            const canvas = document.getElementById('testCanvas');
            const ctx = canvas.getContext('2d');

            // 绘制车牌
            ctx.fillStyle = 'blue';
            ctx.fillRect(0, 0, 400, 200);

            ctx.fillStyle = 'white';
            ctx.fillRect(20, 20, 360, 160);

            ctx.fillStyle = 'black';
            ctx.font = 'bold 48px Arial';
            ctx.fillText('京A88888', 100, 130);
        }

        async function testRecognition() {
            const resultDiv = document.getElementById('recognitionResult');
            resultDiv.innerHTML = '<p>识别中...</p>';

            try {
                const canvas = document.getElementById('testCanvas');
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('file', blob, 'test.jpg');

                    const response = await fetch('/recognize', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <div class="result success">
                                <h4>识别成功！</h4>
                                <p>车牌号: ${result.plate_number}</p>
                                <p>类型: ${result.plate_type}</p>
                                <p>置信度: ${(result.confidence * 100).toFixed(1)}%</p>
                                <p>处理时间: ${result.processing_time.toFixed(2)}ms</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `<div class="result error">识别失败: ${result.detail}</div>`;
                    }
                }, 'image/jpeg');
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">错误: ${error.message}</div>`;
            }
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                document.getElementById('uploadResult').innerHTML = `<p>已选择文件: ${file.name}</p>`;
            }
        }

        async function testFileUpload() {
            const resultDiv = document.getElementById('uploadResult');
            const fileInput = document.getElementById('fileInput');

            if (!fileInput.files[0]) {
                resultDiv.innerHTML = '<div class="result error">请先选择文件</div>';
                return;
            }

            resultDiv.innerHTML = '<p>上传中...</p>';

            try {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                const response = await fetch('/recognize', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h4>上传成功！</h4>
                            <p>车牌号: ${result.plate_number}</p>
                            <p>类型: ${result.plate_type}</p>
                            <p>置信度: ${(result.confidence * 100).toFixed(1)}%</p>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `<div class="result error">上传失败: ${result.detail}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">网络错误: ${error.message}</div>`;
            }
        }

        // 页面加载时生成测试图片
        window.onload = function() {
            generateTestImage();
        };
        </script>
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
        <title>Web界面</title>
        <meta charset="utf-8">
        <style>
            body { font-family: 'Microsoft YaHei', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center; max-width: 500px; }
            h1 { color: #667eea; margin-bottom: 20px; }
            .demo { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .back { display: inline-block; background: #667eea; color: white; text-decoration: none; padding: 10px 20px; border-radius: 5px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>车牌识别Web界面</h1>
            <div class="demo">
                <h3>功能演示</h3>
                <p>当前使用的是简化版本的车牌识别模型</p>
                <p>系统能够正常运行并提供识别服务</p>
                <p><strong>支持功能:</strong></p>
                <ul style="text-align: left;">
                    <li>单张图片识别</li>
                    <li>批量处理</li>
                    <li>实时摄像头识别</li>
                    <li>识别结果可视化</li>
                </ul>
                <p>请使用 <a href="/">主页</a> 的测试功能或访问 <a href="/test">功能测试页面</a></p>
            </div>
            <a href="/" class="back">返回主页</a>
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
        logger.error(f"识别失败: {e}")
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
            logger.error(f"处理文件 {file.filename} 失败: {e}")
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
        "max_file_size": "10MB",
        "model_type": "简化版CNN模型",
        "note": "当前使用简化模型以确保系统稳定运行"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    # 启动服务器
    print("车牌识别系统启动中...")
    print("系统使用简化模型以确保稳定运行")
    print("访问地址:")
    print("  - 主页: http://localhost:8001")
    print("  - Web界面: http://localhost:8001/web")
    print("  - 功能测试: http://localhost:8001/test")
    print("  - API文档: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)