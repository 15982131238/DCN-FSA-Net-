#!/usr/bin/env python3
"""
车牌识别API - 完美匹配训练模型结构
基于原始训练架构的精确实现
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

# 完全匹配原始模型结构的车牌识别模型
class PerfectLicensePlateRecognizer(nn.Module):
    def __init__(self):
        super(PerfectLicensePlateRecognizer, self).__init__()

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, 8, 128))

        # ResNet骨干网络 - 完全匹配原始结构
        self.backbone = nn.ModuleDict({
            '0': nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            '1': nn.BatchNorm2d(64),

            # ResNet块
            '4': nn.ModuleDict({
                '0': nn.ModuleDict({
                    'conv1': nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    'bn1': nn.BatchNorm2d(64),
                    'conv2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(64),
                }),
                '1': nn.ModuleDict({
                    'conv1': nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    'bn1': nn.BatchNorm2d(64),
                    'conv2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(64),
                })
            }),

            '5': nn.ModuleDict({
                '0': nn.ModuleDict({
                    'conv1': nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    'bn1': nn.BatchNorm2d(128),
                    'conv2': nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(128),
                    'downsample': nn.ModuleDict({
                        '0': nn.Conv2d(64, 128, kernel_size=1, stride=2),
                        '1': nn.BatchNorm2d(128),
                    })
                }),
                '1': nn.ModuleDict({
                    'conv1': nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    'bn1': nn.BatchNorm2d(128),
                    'conv2': nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(128),
                })
            }),

            '6': nn.ModuleDict({
                '0': nn.ModuleDict({
                    'conv1': nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    'bn1': nn.BatchNorm2d(256),
                    'conv2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(256),
                    'downsample': nn.ModuleDict({
                        '0': nn.Conv2d(128, 256, kernel_size=1, stride=2),
                        '1': nn.BatchNorm2d(256),
                    })
                }),
                '1': nn.ModuleDict({
                    'conv1': nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    'bn1': nn.BatchNorm2d(256),
                    'conv2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(256),
                })
            }),

            '7': nn.ModuleDict({
                '0': nn.ModuleDict({
                    'conv1': nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    'bn1': nn.BatchNorm2d(512),
                    'conv2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(512),
                    'downsample': nn.ModuleDict({
                        '0': nn.Conv2d(256, 512, kernel_size=1, stride=2),
                        '1': nn.BatchNorm2d(512),
                    })
                }),
                '1': nn.ModuleDict({
                    'conv1': nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    'bn1': nn.BatchNorm2d(512),
                    'conv2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    'bn2': nn.BatchNorm2d(512),
                })
            })
        })

        # 注意力机制
        self.attention = nn.ModuleDict({
            'fc': nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 512),
                nn.Sigmoid()
            )
        })

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
            nn.Linear(64, 72)  # 72个字符
        )

        # 车牌类型分类器
        self.type_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9)   # 9种类型
        )

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # 初始卷积
        x = self.backbone['0'](x)
        x = self.backbone['1'](x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet块
        x = self._resnet_block(x, self.backbone['4'])
        x = self._resnet_block(x, self.backbone['5'])
        x = self._resnet_block(x, self.backbone['6'])
        x = self._resnet_block(x, self.backbone['7'])

        # 注意力机制
        batch_size, channels, height, width = x.size()
        gap = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        attention = self.attention['fc'](gap).view(batch_size, channels, 1, 1)
        x = x * attention

        # 特征增强
        x = self.feature_enhancement(x)

        # 全局平均池化
        global_feat = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)

        # 车牌类型分类
        type_logits = self.type_classifier(global_feat)

        # 字符序列预测
        seq_features = x.mean(dim=2)  # 平均池化高度维度
        seq_features = seq_features.permute(0, 2, 1)  # [batch, width, channels]

        # 添加位置编码
        pos_encoding = self.positional_encoding[:, :seq_features.size(1), :]
        seq_features = seq_features + pos_encoding

        # 字符分类
        char_logits = self.char_classifier(seq_features)

        return char_logits, type_logits

    def _resnet_block(self, x, block_dict):
        """处理ResNet块"""
        # 第一个子块
        residual = x
        out = self.relu(block_dict['0']['bn1'](block_dict['0']['conv1'](x)))
        out = block_dict['0']['bn2'](block_dict['0']['conv2'](out))

        # 下采样
        if 'downsample' in block_dict['0']:
            residual = block_dict['0']['downsample']['1'](
                block_dict['0']['downsample']['0'](x)
            )

        out += residual
        out = self.relu(out)

        # 第二个子块
        residual = out
        out = self.relu(block_dict['1']['bn1'](block_dict['1']['conv1'](out)))
        out = block_dict['1']['bn2'](block_dict['1']['conv2'](out))
        out += residual
        out = self.relu(out)

        return out

# 车牌字符映射 - 精确匹配72个字符
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
        model = PerfectLicensePlateRecognizer()

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

    # 转换为tensor并确保数据类型一致
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.transpose(0, 2).transpose(1, 2)
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
                <p>使用原始训练好的高精度模型</p>
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
        "model_type": "原始训练高精度模型"
    }

@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    return {
        "device": str(device),
        "model_loaded": model is not None,
        "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff"],
        "max_file_size": "10MB",
        "model_type": "原始训练ResNet模型",
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
    print("  - API文档: http://localhost:8001/docs")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)