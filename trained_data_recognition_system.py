#!/usr/bin/env python3
"""
训练数据驱动的车牌识别系统
使用现有的训练模型和数据确保识别准确性
"""

import os
import sys
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import io
import sqlite3
import base64

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from starlette.middleware.cors import CORSMiddleware

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="训练数据驱动车牌识别系统", version="12.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 车牌字符集 (与训练数据保持一致)
PLATE_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
PLATE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
PLATE_NUMBERS = "0123456789"
PLATE_CHARS = PLATE_PROVINCES + PLATE_LETTERS + PLATE_NUMBERS

# 车牌类型映射
PLATE_TYPES = {
    0: "蓝牌", 1: "黄牌", 2: "白牌", 3: "黑牌", 4: "新能源车牌",
    5: "使馆车牌", 6: "港澳车牌", 7: "军牌", 8: "警牌"
}

class ExactModel(nn.Module):
    """与训练模型完全一致的架构"""

    def __init__(self, num_chars=72, max_length=8, num_plate_types=9):
        super().__init__()
        self.num_chars = num_chars
        self.max_length = max_length
        self.num_plate_types = num_plate_types

        # 使用与训练模型完全相同的架构
        # === BACKBONE STRUCTURE ===
        self.backbone_0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.backbone_1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.backbone_4_0_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backbone_4_0_bn1 = nn.BatchNorm2d(64)
        self.backbone_4_0_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backbone_4_0_bn2 = nn.BatchNorm2d(64)

        self.backbone_4_1_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backbone_4_1_bn1 = nn.BatchNorm2d(64)
        self.backbone_4_1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backbone_4_1_bn2 = nn.BatchNorm2d(64)

        self.backbone_5_0_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.backbone_5_0_bn1 = nn.BatchNorm2d(128)
        self.backbone_5_0_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.backbone_5_0_bn2 = nn.BatchNorm2d(128)
        self.backbone_5_0_downsample_0 = nn.Conv2d(64, 128, kernel_size=1)
        self.backbone_5_0_downsample_1 = nn.BatchNorm2d(128)

        self.backbone_5_1_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.backbone_5_1_bn1 = nn.BatchNorm2d(128)
        self.backbone_5_1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.backbone_5_1_bn2 = nn.BatchNorm2d(128)

        self.backbone_6_0_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.backbone_6_0_bn1 = nn.BatchNorm2d(256)
        self.backbone_6_0_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.backbone_6_0_bn2 = nn.BatchNorm2d(256)
        self.backbone_6_0_downsample_0 = nn.Conv2d(128, 256, kernel_size=1)
        self.backbone_6_0_downsample_1 = nn.BatchNorm2d(256)

        self.backbone_6_1_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.backbone_6_1_bn1 = nn.BatchNorm2d(256)
        self.backbone_6_1_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.backbone_6_1_bn2 = nn.BatchNorm2d(256)

        self.backbone_7_0_conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.backbone_7_0_bn1 = nn.BatchNorm2d(512)
        self.backbone_7_0_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.backbone_7_0_bn2 = nn.BatchNorm2d(512)
        self.backbone_7_0_downsample_0 = nn.Conv2d(256, 512, kernel_size=1)
        self.backbone_7_0_downsample_1 = nn.BatchNorm2d(512)

        self.backbone_7_1_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.backbone_7_1_bn1 = nn.BatchNorm2d(512)
        self.backbone_7_1_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.backbone_7_1_bn2 = nn.BatchNorm2d(512)

        # Feature enhancement
        self.feature_enhancement_0 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.feature_enhancement_1 = nn.BatchNorm2d(256)
        self.feature_enhancement_4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.feature_enhancement_5 = nn.BatchNorm2d(128)

        # Attention mechanism
        self.attention_fc_0 = nn.Linear(512, 64)
        self.attention_fc_2 = nn.Linear(64, 512)

        # Classifiers
        self.char_classifier_0 = nn.Linear(128, 64)
        self.char_classifier_3 = nn.Linear(64, num_chars)
        self.type_classifier_0 = nn.Linear(128, 64)
        self.type_classifier_3 = nn.Linear(64, num_plate_types)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, 128))

        # Activation functions
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone forward pass
        x = self.backbone_0(x)
        x = self.relu(x)
        x = self.backbone_1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        # Residual blocks
        # Layer 4
        identity = x
        out = self.relu(self.backbone_4_0_bn1(self.backbone_4_0_conv1(x)))
        out = self.backbone_4_0_bn2(self.backbone_4_0_conv2(out))
        out += identity
        x = self.relu(out)

        identity = x
        out = self.relu(self.backbone_4_1_bn1(self.backbone_4_1_conv1(x)))
        out = self.backbone_4_1_bn2(self.backbone_4_1_conv2(out))
        out += identity
        x = self.relu(out)

        # Layer 5
        identity = x
        out = self.relu(self.backbone_5_0_bn1(self.backbone_5_0_conv1(x)))
        out = self.backbone_5_0_bn2(self.backbone_5_0_conv2(out))
        identity = self.backbone_5_0_downsample_1(self.backbone_5_0_downsample_0(identity))
        out += identity
        x = self.relu(out)

        identity = x
        out = self.relu(self.backbone_5_1_bn1(self.backbone_5_1_conv1(x)))
        out = self.backbone_5_1_bn2(self.backbone_5_1_conv2(out))
        out += identity
        x = self.relu(out)

        # Layer 6
        identity = x
        out = self.relu(self.backbone_6_0_bn1(self.backbone_6_0_conv1(x)))
        out = self.backbone_6_0_bn2(self.backbone_6_0_conv2(out))
        identity = self.backbone_6_0_downsample_1(self.backbone_6_0_downsample_0(identity))
        out += identity
        x = self.relu(out)

        identity = x
        out = self.relu(self.backbone_6_1_bn1(self.backbone_6_1_conv1(x)))
        out = self.backbone_6_1_bn2(self.backbone_6_1_conv2(out))
        out += identity
        x = self.relu(out)

        # Layer 7
        identity = x
        out = self.relu(self.backbone_7_0_bn1(self.backbone_7_0_conv1(x)))
        out = self.backbone_7_0_bn2(self.backbone_7_0_conv2(out))
        identity = self.backbone_7_0_downsample_1(self.backbone_7_0_downsample_0(identity))
        out += identity
        x = self.relu(out)

        identity = x
        out = self.relu(self.backbone_7_1_bn1(self.backbone_7_1_conv1(x)))
        out = self.backbone_7_1_bn2(self.backbone_7_1_conv2(out))
        out += identity
        x = self.relu(out)

        features_512 = x

        # Feature enhancement
        x = self.relu(self.feature_enhancement_1(self.feature_enhancement_0(features_512)))
        x = self.relu(self.feature_enhancement_5(self.feature_enhancement_4(x)))
        features_128 = x

        # Attention mechanism
        global_features = F.adaptive_avg_pool2d(features_512, (1, 1)).squeeze(-1).squeeze(-1)
        attention_weights = self.attention_fc_0(global_features)
        attention_weights = torch.sigmoid(attention_weights)
        attention_weights = self.attention_fc_2(attention_weights)
        attention_weights = torch.sigmoid(attention_weights)

        B, C, H, W = features_512.shape
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        attended_features = features_512 * attention_weights
        attended_features = self.relu(self.feature_enhancement_1(self.feature_enhancement_0(attended_features)))
        attended_features = self.relu(self.feature_enhancement_5(self.feature_enhancement_4(attended_features)))

        global_features = F.adaptive_avg_pool2d(attended_features, (1, 1)).squeeze(-1).squeeze(-1)
        seq_features = F.adaptive_avg_pool2d(features_128, (self.max_length, 1))
        seq_features = seq_features.squeeze(-1).transpose(1, 2)
        seq_features = seq_features + self.positional_encoding

        char_logits = self.char_classifier_3(self.relu(self.char_classifier_0(seq_features)))
        type_logits = self.type_classifier_3(self.relu(self.type_classifier_0(global_features)))

        return char_logits, type_logits

class TrainingDataRecognizer:
    """基于训练数据的车牌识别器"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        self.db_path = "data/trained_recognition_history.db"

        # 初始化数据库
        self.init_database()

        # 尝试加载模型
        self.load_model()

        logger.info(f"初始化训练数据驱动识别器，设备: {self.device}")
        logger.info(f"模型加载状态: {self.model_loaded}")

    def init_database(self):
        """初始化数据库"""
        try:
            os.makedirs("data", exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    plate_number TEXT,
                    plate_type TEXT,
                    confidence REAL,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_data BLOB
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("数据库初始化成功")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")

    def load_model(self):
        """加载训练好的模型"""
        model_path = "best_fast_high_accuracy_model.pth"

        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return False

        try:
            # 创建模型
            self.model = ExactModel(num_chars=72, max_length=8, num_plate_types=9)
            self.model.to(self.device)
            self.model.eval()

            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)

            # 尝试直接加载
            try:
                self.model.load_state_dict(checkpoint, strict=True)
                logger.info("模型权重加载成功（严格模式）")
            except Exception as e:
                logger.warning(f"严格模式加载失败: {e}")
                # 尝试部分加载
                model_dict = self.model.state_dict()
                pretrained_dict = {}
                for k, v in checkpoint.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        pretrained_dict[k] = v
                    else:
                        logger.warning(f"跳过参数: {k}, 形状: {v.shape}")

                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                logger.info(f"部分加载: {len(pretrained_dict)}/{len(checkpoint)} 参数")

            self.model_loaded = True
            logger.info("模型加载成功")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model_loaded = False
            return False

    def safe_open_image(self, image_data: bytes, filename: str) -> Optional[np.ndarray]:
        """安全打开图片文件"""
        try:
            # 尝试使用PIL打开
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 转换为numpy数组
            image_array = np.array(image)

            # 验证图片尺寸
            if image_array.shape[0] < 20 or image_array.shape[1] < 20:
                logger.warning(f"图片尺寸过小: {image_array.shape}")
                return None

            return image_array

        except Exception as e:
            logger.error(f"PIL打开失败，尝试OpenCV: {e}")

            # 尝试OpenCV
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if cv_image is not None:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    return cv_image
            except Exception as e2:
                logger.error(f"OpenCV打开也失败: {e2}")

            return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        try:
            # 调整大小到模型输入尺寸
            height, width = image.shape[:2]
            if width != 224 or height != 224:
                image = cv2.resize(image, (224, 224))

            # 归一化
            image = image.astype(np.float32) / 255.0

            # 标准化 (ImageNet均值和标准差)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std

            # 调整维度顺序
            image = np.transpose(image, (2, 0, 1))

            return image

        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return image

    def decode_plate_number(self, char_logits: torch.Tensor) -> str:
        """解码车牌号码"""
        try:
            # 使用贪心解码
            _, predicted = torch.max(char_logits, dim=-1)
            predicted = predicted.squeeze(0).cpu().numpy()

            # 转换为字符
            plate_chars = []
            for idx in predicted:
                if idx < len(PLATE_CHARS):
                    plate_chars.append(PLATE_CHARS[idx])
                else:
                    break

            plate_number = ''.join(plate_chars)

            # 验证车牌格式
            if self.validate_plate_format(plate_number):
                return plate_number
            else:
                # 如果格式不正确，尝试修复
                return self.fix_plate_format(plate_number)

        except Exception as e:
            logger.error(f"车牌解码失败: {e}")
            return ""

    def validate_plate_format(self, text: str) -> bool:
        """验证车牌格式"""
        if len(text) < 7 or len(text) > 8:
            return False

        # 检查省份简称
        if text[0] not in PLATE_PROVINCES:
            return False

        # 检查第二个字符是否是字母
        if text[1] not in PLATE_LETTERS:
            return False

        # 检查剩余字符
        remaining_chars = text[2:]
        valid_chars = all(c in PLATE_LETTERS + PLATE_NUMBERS for c in remaining_chars)

        return valid_chars

    def fix_plate_format(self, text: str) -> str:
        """修复车牌格式"""
        if not text:
            return text

        # 确保至少有省份和字母
        if len(text) < 2:
            return text

        # 确保第一个字符是省份
        if text[0] not in PLATE_PROVINCES:
            text = '京' + text[1:]  # 默认使用北京

        # 确保第二个字符是字母
        if text[1] not in PLATE_LETTERS:
            text = text[0] + 'A' + text[2:]

        # 限制长度
        if len(text) > 8:
            text = text[:8]

        # 确保其他字符有效
        fixed_chars = []
        for i, c in enumerate(text[2:], 2):
            if c in PLATE_LETTERS + PLATE_NUMBERS:
                fixed_chars.append(c)
            else:
                if i < 7:  # 前7位用数字填充
                    fixed_chars.append('0')

        return text[:2] + ''.join(fixed_chars[:6])

    def recognize_with_model(self, image: np.ndarray) -> Dict[str, Any]:
        """使用模型进行识别"""
        if not self.model_loaded:
            return {
                "plate_number": "模型未加载",
                "plate_type": "未知",
                "confidence": 0.0,
                "success": False,
                "method": "model_not_loaded"
            }

        try:
            # 预处理图像
            processed_image = self.preprocess_image(image)

            # 转换为tensor
            input_tensor = torch.FloatTensor(processed_image).unsqueeze(0).to(self.device)

            # 模型推理
            with torch.no_grad():
                char_logits, type_logits = self.model(input_tensor)

                # 获取类型预测
                type_probs = torch.softmax(type_logits, dim=-1)
                type_pred = torch.argmax(type_probs, dim=-1).item()
                type_confidence = type_probs[0][type_pred].item()

                # 解码车牌号码
                plate_number = self.decode_plate_number(char_logits)

                # 计算字符置信度
                char_probs = torch.softmax(char_logits, dim=-1)
                char_confidence = torch.max(char_probs, dim=-1)[0].mean().item()

                # 综合置信度
                confidence = (char_confidence + type_confidence) / 2

                if plate_number:
                    plate_type = PLATE_TYPES.get(type_pred, "未知")

                    return {
                        "plate_number": plate_number,
                        "plate_type": plate_type,
                        "confidence": confidence,
                        "success": True,
                        "method": "trained_model",
                        "model_type": "ExactModel",
                        "type_confidence": type_confidence,
                        "char_confidence": char_confidence
                    }
                else:
                    return {
                        "plate_number": "识别失败",
                        "plate_type": "未知",
                        "confidence": 0.0,
                        "success": False,
                        "method": "model_failed"
                    }

        except Exception as e:
            logger.error(f"模型识别失败: {e}")
            return {
                "plate_number": "模型错误",
                "plate_type": "未知",
                "confidence": 0.0,
                "success": False,
                "method": "model_exception"
            }

    def save_to_database(self, filename: str, result: Dict[str, Any], image_data: bytes):
        """保存识别结果到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO recognition_history
                (filename, plate_number, plate_type, confidence, processing_time, image_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                result.get('plate_number', ''),
                result.get('plate_type', '未知'),
                result.get('confidence', 0.0),
                result.get('processing_time', 0.0),
                image_data
            ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"数据库保存失败: {e}")

    def recognize_plate(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """主识别函数"""
        start_time = time.time()

        try:
            # 安全打开图片
            image = self.safe_open_image(image_data, filename)
            if image is None:
                return {
                    "plate_number": "图片格式错误",
                    "plate_type": "未知",
                    "confidence": 0.0,
                    "processing_time": (time.time() - start_time) * 1000,
                    "success": False,
                    "method": "invalid_image"
                }

            # 使用训练模型进行识别
            result = self.recognize_with_model(image)

            # 添加处理时间
            result['processing_time'] = (time.time() - start_time) * 1000

            # 保存到数据库
            if result['success']:
                self.save_to_database(filename, result, image_data)

            return result

        except Exception as e:
            logger.error(f"识别失败: {e}")
            return {
                "plate_number": "处理异常",
                "plate_type": "未知",
                "confidence": 0.0,
                "processing_time": (time.time() - start_time) * 1000,
                "success": False,
                "method": "exception"
            }

# 创建识别器实例
recognizer = TrainingDataRecognizer()

# 数据模型
class RecognitionResult(BaseModel):
    plate_number: str
    plate_type: str
    confidence: float
    processing_time: float
    success: bool

# API端点
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "训练数据驱动的车牌识别系统",
        "version": "12.0.0",
        "model_loaded": recognizer.model_loaded,
        "device": str(recognizer.device)
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_type": "ExactModel",
        "device": str(recognizer.device),
        "model_loaded": recognizer.model_loaded,
        "training_data_available": os.path.exists("best_fast_high_accuracy_model.pth")
    }

@app.post("/recognize")
async def recognize_plate(file: UploadFile = File(...)):
    """车牌识别端点"""
    try:
        start_time = time.time()

        # 读取文件
        contents = await file.read()

        # 进行识别
        result = recognizer.recognize_plate(contents, file.filename)

        return result

    except Exception as e:
        logger.error(f"识别失败: {e}")
        return {
            "plate_number": "处理异常",
            "plate_type": "未知",
            "confidence": 0.0,
            "processing_time": (time.time() - start_time) * 1000,
            "success": False,
            "method": "exception"
        }

@app.get("/history")
async def get_history(limit: int = 10):
    """获取识别历史"""
    try:
        conn = sqlite3.connect(recognizer.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT plate_number, plate_type, confidence, processing_time, timestamp
            FROM recognition_history
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))

        history = []
        for row in cursor.fetchall():
            history.append({
                "plate_number": row[0],
                "plate_type": row[1],
                "confidence": row[2],
                "processing_time": row[3],
                "timestamp": row[4]
            })

        conn.close()
        return {"history": history}

    except Exception as e:
        logger.error(f"获取历史失败: {e}")
        return {"history": []}

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    try:
        conn = sqlite3.connect(recognizer.db_path)
        cursor = conn.cursor()

        # 总识别次数
        cursor.execute("SELECT COUNT(*) FROM recognition_history")
        total_count = cursor.fetchone()[0]

        # 成功次数
        cursor.execute("SELECT COUNT(*) FROM recognition_history WHERE confidence > 0.5")
        success_count = cursor.fetchone()[0]

        # 平均置信度
        cursor.execute("SELECT AVG(confidence) FROM recognition_history")
        avg_confidence = cursor.fetchone()[0] or 0.0

        # 平均处理时间
        cursor.execute("SELECT AVG(processing_time) FROM recognition_history")
        avg_time = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_recognitions": total_count,
            "successful_recognitions": success_count,
            "success_rate": (success_count / total_count * 100) if total_count > 0 else 0,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_time
        }

    except Exception as e:
        logger.error(f"获取统计失败: {e}")
        return {
            "total_recognitions": 0,
            "successful_recognitions": 0,
            "success_rate": 0,
            "average_confidence": 0,
            "average_processing_time": 0
        }

if __name__ == "__main__":
    print("启动训练数据驱动的车牌识别系统...")
    uvicorn.run(app, host="0.0.0.0", port=8027, reload=False)