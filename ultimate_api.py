#!/usr/bin/env python3
"""
车牌识别API - 完全匹配原始训练模型
基于UltimatePlateModel架构的精确实现
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
import torchvision.transforms as transforms
import torchvision.models as models
import time
from typing import List, Dict, Any
import os

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

# 完全匹配原始训练的UltimatePlateModel
class UltimatePlateModel(nn.Module):
    """终极车牌模型 - 完全匹配训练架构"""
    def __init__(self, num_chars=74, max_length=8, num_plate_types=9):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_plate_types = num_plate_types

        # 使用ResNet34作为骨干网络
        resnet = torchvision.models.resnet34(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 多级特征提取
        self.feature_pyramid = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 64, 1)
        ])

        # 高级注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Sigmoid()
        )

        # 双向GRU序列建模
        self.char_gru = nn.GRU(64, 128, bidirectional=True, batch_first=True, dropout=0.2)

        # 字符分类器
        self.char_classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_chars)
        )

        # 类型分类器
        self.type_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_plate_types)
        )

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, 256))

    def forward(self, x):
        batch_size = x.size(0)

        # 骨干网络特征提取
        features = self.backbone(x)  # [B, 512, H, W]

        # 特征金字塔
        pyramid_features = []
        for i, conv in enumerate(self.feature_pyramid):
            features = conv(features)
            pyramid_features.append(features)

        # 使用最细粒度的特征
        fine_features = pyramid_features[-1]

        # 注意力机制
        attention_weights = self.attention(fine_features)
        attended_features = fine_features * attention_weights

        # 全局平均池化用于类型分类
        global_features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

        # 序列特征用于字符分类
        seq_features = F.adaptive_avg_pool2d(attended_features, (self.max_length, 1))
        seq_features = seq_features.squeeze(-1).transpose(1, 2)  # [B, L, C]

        # GRU序列建模
        gru_out, _ = self.char_gru(seq_features)

        # 添加位置编码
        gru_out = gru_out + self.positional_encoding

        # 分类
        char_logits = self.char_classifier(gru_out)
        type_logits = self.type_classifier(global_features)

        return char_logits, type_logits


# 车牌字符映射 - 完全匹配训练时的字符集
PLATE_CHARS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领警学挂港澳'

PLATE_TYPES = [
    '普通蓝牌', '新能源小型车', '新能源大型车', '单层黄牌',
    '黑色车牌', '白色车牌', '双层黄牌', '拖拉机绿牌', '其他类型'
]

# 全局变量
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet标准输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model():
    """加载训练好的模型"""
    global model

    try:
        print("正在加载训练好的模型...")

        # 创建模型实例
        model = UltimatePlateModel(
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

    # 应用预处理变换
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度

    return img_tensor.to(device)

def decode_prediction(char_logits, type_logits):
    """解码预测结果"""
    # 字符预测
    char_probs = F.softmax(char_logits, dim=-1)
    char_indices = torch.argmax(char_logits, dim=-1)

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
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>车牌识别系统</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); padding: 40px; text-align: center; }}
            h1 {{ color: #333; margin-bottom: 30px; font-size: 2.5em; }}
            .btn {{ display: inline-block; background: linear-gradient(45deg, #667eea, #764ba2); color: white; text-decoration: none; padding: 15px 30px; border-radius: 25px; margin: 10px; font-size: 1.1em; transition: all 0.3s ease; }}
            .btn:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }}
            .info {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .status.success {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 车牌识别系统</h1>
            <div class="info">
                <p>基于深度学习的智能车牌识别解决方案</p>
                <p>使用原始训练好的高精度模型 (UltimatePlateModel)</p>
                <div class="status success">
                    系统状态: 运行正常 | 模型: 已加载 | 设备: {device}
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
    """

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_type": "UltimatePlateModel - ResNet34+GRU"
    }

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    return {
        "device": str(device),
        "model_loaded": model is not None,
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "max_file_size": "10MB",
        "model_type": "UltimatePlateModel",
        "num_chars": len(PLATE_CHARS),
        "num_plate_types": len(PLATE_TYPES),
        "input_size": "224x224"
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

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Web界面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>车牌识别系统</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Microsoft YaHei', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); overflow: hidden; }
            .header { background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 30px; text-align: center; }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .main-content { padding: 40px; }
            .upload-section { border: 3px dashed #ddd; border-radius: 10px; padding: 40px; text-align: center; margin-bottom: 30px; }
            .upload-btn { background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; padding: 15px 30px; border-radius: 25px; font-size: 1.1em; cursor: pointer; margin: 10px; }
            .file-input { display: none; }
            .preview-section { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 30px; }
            .image-preview, .result-preview { background: #f8f9fa; border-radius: 10px; padding: 20px; text-align: center; }
            .plate-number { font-size: 2em; font-weight: bold; color: #667eea; text-align: center; padding: 15px; background: linear-gradient(45deg, #f0f4ff, #e8f0ff); border-radius: 10px; margin: 15px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚗 车牌识别系统</h1>
                <p>使用原始训练模型的高精度识别</p>
            </div>
            <div class="main-content">
                <div class="upload-section">
                    <h2>📤 上传图片进行识别</h2>
                    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">选择图片</button>
                    <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFile(this.files[0])">
                </div>
                <div class="preview-section" id="previewSection" style="display: none;">
                    <div class="image-preview">
                        <h3>📷 原始图片</h3>
                        <img id="previewImage" alt="预览图片" style="max-width: 100%; max-height: 300px;">
                    </div>
                    <div class="result-preview">
                        <h3>🎯 识别结果</h3>
                        <div class="plate-number" id="plateNumber">等待识别...</div>
                        <div><strong>车牌类型:</strong> <span id="plateType">-</span></div>
                        <div><strong>置信度:</strong> <span id="confidence">-</span></div>
                        <div><strong>处理时间:</strong> <span id="processingTime">-</span></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
        async function handleFile(file) {
            if (!file) return;

            const previewImage = document.getElementById('previewImage');
            const previewSection = document.getElementById('previewSection');

            // 显示预览
            previewImage.src = URL.createObjectURL(file);
            previewSection.style.display = 'grid';

            // 上传识别
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/recognize', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    document.getElementById('plateNumber').textContent = result.plate_number;
                    document.getElementById('plateType').textContent = result.plate_type;
                    document.getElementById('confidence').textContent = Math.round(result.confidence * 100) + '%';
                    document.getElementById('processingTime').textContent = result.processing_time.toFixed(2) + 'ms';
                } else {
                    alert('识别失败: ' + result.detail);
                }
            } catch (error) {
                alert('网络错误: ' + error.message);
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
                        设备: ${data.device}<br>
                        类型: ${data.model_type}
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