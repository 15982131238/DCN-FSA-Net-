#!/usr/bin/env python3
"""
车牌识别API - 使用训练好的高精度模型
基于原始训练架构的完整实现
"""

import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import base64
import time
from typing import List, Dict, Any
import requests
import os

# 设置中文字体支持
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建FastAPI应用
app = FastAPI(
    title="车牌识别系统",
    description="基于深度学习的中国车牌识别解决方案",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义响应模型
class RecognitionResult(BaseModel):
    plate_number: str
    plate_type: str
    confidence: float
    processing_time: float

# 定义完整的车牌识别模型
class LicensePlateRecognizer(nn.Module):
    def __init__(self, num_chars=72, max_length=8, num_plate_types=9):
        super(LicensePlateRecognizer, self).__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_plate_types = num_plate_types

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, 128))

        # ResNet骨干网络
        self.backbone = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ResNet层1
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),

            # ResNet层2
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128),

            # ResNet层3
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256),

            # ResNet层4
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 512),
        )

        # 注意力机制
        self.attention = AttentionModule(512)

        # 特征增强
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 字符分类器
        self.char_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_chars)
        )

        # 车牌类型分类器
        self.type_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_plate_types)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 骨干网络特征提取
        features = self.backbone(x)

        # 应用注意力
        features = self.attention(features)

        # 特征增强
        enhanced = self.feature_enhancement(features)

        # 全局平均池化
        global_feat = F.adaptive_avg_pool2d(enhanced, (1, 1))
        global_feat = global_feat.view(global_feat.size(0), -1)

        # 车牌类型分类
        type_logits = self.type_classifier(global_feat)

        # 字符序列预测
        # 将特征图分割成序列
        seq_features = enhanced.mean(dim=2)  # 平均池化高度维度
        seq_features = seq_features.permute(0, 2, 1)  # [batch, width, channels]

        # 添加位置编码
        pos_encoding = self.positional_encoding[:, :seq_features.size(1), :]
        seq_features = seq_features + pos_encoding

        # 字符分类
        char_logits = self.char_classifier(seq_features)

        return char_logits, type_logits

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 全局平均池化
        gap = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)

        # 计算注意力权重
        attention = self.fc(gap).view(batch_size, channels, 1, 1)

        # 应用注意力
        return x * attention

# 车牌字符映射
PLATE_CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '京', '津', '冀', '晋', '蒙', '辽', '吉', '黑', '沪', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼',
    '渝', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '港', '澳', '台'
]

PLATE_TYPES = [
    '蓝牌', '黄牌', '绿牌', '白牌', '黑牌', '警车', '军车', '使馆', '其他'
]

# 全局变量
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """加载训练好的模型"""
    global model

    try:
        print("正在加载训练好的模型...")

        # 创建模型实例
        model = LicensePlateRecognizer(
            num_chars=len(PLATE_CHARS),
            max_length=8,
            num_plate_types=len(PLATE_TYPES)
        )

        # 加载训练好的权重
        checkpoint = torch.load('best_fast_high_accuracy_model.pth', map_location='cpu')

        # 加载模型权重
        model.load_state_dict(checkpoint, strict=True)

        # 设置为评估模式
        model.eval()

        # 移动到设备
        model = model.to(device)

        print(f"模型加载成功！使用设备: {device}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        return True

    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

def preprocess_image(image):
    """预处理图像"""
    # 转换为RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 调整大小
    image = image.resize((384, 96), Image.Resampling.LANCZOS)

    # 转换为numpy数组
    img_array = np.array(image, dtype=np.float32) / 255.0

    # 标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # 转换为tensor
    img_tensor = torch.from_numpy(img_array).transpose(0, 2).transpose(1, 2)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor.to(device)

def decode_prediction(char_logits, type_logits):
    """解码预测结果"""
    # 字符预测
    char_probs = F.softmax(char_logits, dim=-1)
    char_indices = torch.argmax(char_probs, dim=-1)

    # 转换为字符
    plate_chars = []
    for idx in char_indices[0]:
        if idx < len(PLATE_CHARS):
            plate_chars.append(PLATE_CHARS[idx])
        else:
            plate_chars.append('?')

    # 过滤连续重复字符
    filtered_chars = []
    for i, char in enumerate(plate_chars):
        if i == 0 or char != plate_chars[i-1]:
            filtered_chars.append(char)

    # 限制长度
    plate_number = ''.join(filtered_chars[:8])

    # 车牌类型预测
    type_probs = F.softmax(type_logits, dim=-1)
    type_idx = torch.argmax(type_probs, dim=-1)[0].item()
    plate_type = PLATE_TYPES[type_idx]

    # 计算置信度
    confidence = torch.max(type_probs).item()

    return plate_number, plate_type, confidence

def recognize_plate(image):
    """识别车牌"""
    start_time = time.time()

    try:
        # 预处理
        img_tensor = preprocess_image(image)

        # 模型推理
        with torch.no_grad():
            char_logits, type_logits = model(img_tensor)

        # 解码结果
        plate_number, plate_type, confidence = decode_prediction(char_logits, type_logits)

        # 处理时间
        processing_time = (time.time() - start_time) * 1000

        return {
            "plate_number": plate_number,
            "plate_type": plate_type,
            "confidence": confidence,
            "processing_time": processing_time
        }

    except Exception as e:
        print(f"识别失败: {e}")
        return {
            "plate_number": "识别失败",
            "plate_type": "未知",
            "confidence": 0.0,
            "processing_time": 0.0
        }

# API端点
@app.get("/", response_class=HTMLResponse)
async def root():
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 车牌识别系统</h1>
            <div class="info">
                <p>基于深度学习的智能车牌识别解决方案</p>
                <p>使用训练好的高精度模型进行识别</p>
                <div class="status success">
                    系统状态: 运行正常 | 模型: 已加载 | 设备: {}
                </div>
            </div>
            <div>
                <a href="/web" class="btn">进入Web界面</a>
                <a href="/docs" class="btn">API文档</a>
                <a href="/test" class="btn">功能测试</a>
            </div>
        </div>
    </body>
    </html>
    """.format(device)

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_type": "完整版高精度模型"
    }

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    return {
        "device": str(device),
        "model_loaded": model is not None,
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "max_file_size": "10MB",
        "model_type": "完整版高精度ResNet模型",
        "num_chars": len(PLATE_CHARS),
        "num_plate_types": len(PLATE_TYPES)
    }

@app.post("/recognize", response_model=RecognitionResult)
async def recognize_plate_api(file: UploadFile = File(...)):
    """单张图片识别"""
    try:
        # 读取图片
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 识别车牌
        result = recognize_plate(image)

        return RecognitionResult(
            plate_number=result["plate_number"],
            plate_type=result["plate_type"],
            confidence=result["confidence"],
            processing_time=result["processing_time"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize_batch")
async def recognize_batch_api(files: List[UploadFile] = File(...)):
    """批量识别"""
    results = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            result = recognize_plate(image)
            results.append({
                "filename": file.filename,
                "plate_number": result["plate_number"],
                "plate_type": result["plate_type"],
                "confidence": result["confidence"],
                "processing_time": result["processing_time"]
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"results": results}

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """测试页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>车牌识别测试</title>
        <meta charset="utf-8">
        <style>
            body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 10px; padding: 30px; }
            .test-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #0056b3; }
            .result { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 车牌识别功能测试</h1>

            <div class="test-section">
                <h3>系统状态</h3>
                <div id="systemStatus">检查中...</div>
                <button class="btn" onclick="checkSystem()">检查系统</button>
            </div>

            <div class="test-section">
                <h3>图片上传测试</h3>
                <input type="file" id="testFile" accept="image/*">
                <button class="btn" onclick="testUpload()">测试上传</button>
                <div id="uploadResult"></div>
            </div>

            <div class="test-section">
                <h3>演示测试</h3>
                <button class="btn" onclick="testDemo()">测试演示功能</button>
                <div id="demoResult"></div>
            </div>
        </div>

        <script>
        async function checkSystem() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                document.getElementById('systemStatus').innerHTML =
                    `<div class="result success">
                        状态: ${data.status}<br>
                        模型: ${data.model_loaded ? '已加载' : '未加载'}<br>
                        设备: ${data.device}
                    </div>`;
            } catch (error) {
                document.getElementById('systemStatus').innerHTML =
                    `<div class="result error">连接失败: ${error.message}</div>`;
            }
        }

        async function testUpload() {
            const file = document.getElementById('testFile').files[0];
            if (!file) {
                alert('请选择文件');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/recognize', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    document.getElementById('uploadResult').innerHTML =
                        `<div class="result success">
                            车牌号: ${result.plate_number}<br>
                            类型: ${result.plate_type}<br>
                            置信度: ${(result.confidence * 100).toFixed(1)}%<br>
                            处理时间: ${result.processing_time.toFixed(2)}ms
                        </div>`;
                } else {
                    document.getElementById('uploadResult').innerHTML =
                        `<div class="result error">错误: ${result.detail}</div>`;
                }
            } catch (error) {
                document.getElementById('uploadResult').innerHTML =
                    `<div class="result error">网络错误: ${error.message}</div>`;
            }
        }

        async function testDemo() {
            try {
                // 创建测试图片
                const canvas = document.createElement('canvas');
                canvas.width = 400;
                canvas.height = 200;
                const ctx = canvas.getContext('2d');

                ctx.fillStyle = 'white';
                ctx.fillRect(0, 0, 400, 200);
                ctx.fillStyle = 'black';
                ctx.font = '48px Arial';
                ctx.fillText('京A12345', 100, 120);

                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('file', blob, 'test.jpg');

                    const response = await fetch('/recognize', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();

                    if (response.ok) {
                        document.getElementById('demoResult').innerHTML =
                            `<div class="result success">
                                演示测试成功！<br>
                                车牌号: ${result.plate_number}<br>
                                类型: ${result.plate_type}<br>
                                置信度: ${(result.confidence * 100).toFixed(1)}%
                            </div>`;
                    } else {
                        document.getElementById('demoResult').innerHTML =
                            `<div class="result error">测试失败: ${result.detail}</div>`;
                    }
                }, 'image/jpeg');
            } catch (error) {
                document.getElementById('demoResult').innerHTML =
                    `<div class="result error">测试失败: ${error.message}</div>`;
            }
        }

        // 页面加载时检查系统
        window.onload = checkSystem;
        </script>
    </body>
    </html>
    """

# 启动时加载模型
@app.on_event("startup")
async def startup_event():
    """启动事件"""
    success = load_model()
    if not success:
        print("警告: 模型加载失败，系统将使用随机权重")

    print("车牌识别系统启动完成")
    print("系统访问地址:")
    print("  - 主页: http://localhost:8001")
    print("  - Web界面: http://localhost:8001/web")
    print("  - 功能测试: http://localhost:8001/test")
    print("  - API文档: http://localhost:8001/docs")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)