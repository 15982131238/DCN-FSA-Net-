#!/usr/bin/env python3
"""
简单可靠的车牌识别系统
"""

import os
import sys
import logging
import time
import json
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import io

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from starlette.middleware.cors import CORSMiddleware

# 尝试导入Tesseract
try:
    import pytesseract
    tesseract_available = True
    print("Tesseract OCR可用")
except ImportError:
    tesseract_available = False
    print("Tesseract OCR不可用")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="简单可靠车牌识别系统", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 车牌省份简称
plate_chars = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
plate_numbers = "0123456789"
plate_letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"

class SimpleOCRRecognizer:
    """简单可靠的OCR识别器"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tesseract_available = tesseract_available
        logger.info(f"初始化简单OCR识别器，设备: {self.device}")
        logger.info(f"Tesseract可用: {self.tesseract_available}")

    def preprocess_image(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """图像预处理"""
        # 转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 调整大小
        height, width = cv_image.shape[:2]
        max_size = 1200
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_image = cv2.resize(cv_image, (new_width, new_height))

        # 预处理方法
        results = {}

        # 灰度化
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        results['gray'] = gray

        # 直方图均衡化
        equalized = cv2.equalizeHist(gray)
        results['equalized'] = equalized

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        results['blurred'] = blurred

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        results['edges'] = edges

        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results['binary'] = binary

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        results['morph'] = morph

        return results

    def find_plate_candidates(self, image: np.ndarray, morph: np.ndarray) -> List[np.ndarray]:
        """查找车牌候选区域"""
        candidates = []

        # 轮廓检测
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)

            # 过滤小面积
            if area < 500:
                continue

            # 计算边界矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 车牌宽高比
            aspect_ratio = w / h
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                continue

            # 提取候选区域
            roi = image[y:y+h, x:x+w]

            # 确保区域大小合理
            if roi.shape[0] < 20 or roi.shape[1] < 60:
                continue

            candidates.append(roi)

        return candidates

    def extract_text(self, image: np.ndarray) -> str:
        """提取文字"""
        if not self.tesseract_available:
            return ""

        try:
            # 转换为PIL图像
            pil_image = Image.fromarray(image)

            # 配置Tesseract
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领'

            # 提取文字
            text = pytesseract.image_to_string(pil_image, config=custom_config)

            # 清理结果
            text = text.strip().replace('\n', '').replace('\r', '').replace(' ', '')

            return text

        except Exception as e:
            logger.error(f"文字提取失败: {e}")
            return ""

    def is_valid_plate(self, text: str) -> bool:
        """验证车牌格式"""
        if not text or len(text) < 7 or len(text) > 9:
            return False

        # 检查字符有效性
        valid_chars = plate_chars + plate_numbers + plate_letters
        return all(char in valid_chars for char in text)

    def get_plate_type(self, plate_number: str) -> str:
        """获取车牌类型"""
        if not plate_number:
            return "未知"

        # 新能源车牌
        if len(plate_number) == 8 or plate_number[1] in ['D', 'F']:
            return "绿牌"

        # 黄牌
        if plate_number[1] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M']:
            return "黄牌"

        # 默认蓝牌
        return "蓝牌"

    def calculate_confidence(self, text: str) -> float:
        """计算置信度"""
        if not text:
            return 0.0

        confidence = 0.5

        # 长度检查
        if 7 <= len(text) <= 8:
            confidence += 0.2

        # 格式检查
        if self.is_valid_plate(text):
            confidence += 0.2

        # 首字符检查
        if text[0] in plate_chars:
            confidence += 0.1

        return min(confidence, 0.99)

    def recognize(self, image: Image.Image, filename: str) -> Dict[str, Any]:
        """主识别函数"""
        start_time = time.time()

        try:
            # 预处理
            processed = self.preprocess_image(image)

            # 查找候选区域
            candidates = self.find_plate_candidates(
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR),
                processed['morph']
            )

            if not candidates:
                return {
                    "plate_number": "未检测到车牌",
                    "plate_type": "未知",
                    "confidence": 0.0,
                    "processing_time": (time.time() - start_time) * 1000,
                    "success": False,
                    "method": "no_detection"
                }

            # 对每个候选区域进行识别
            best_result = None
            best_confidence = 0

            for candidate in candidates:
                text = self.extract_text(candidate)

                if text and self.is_valid_plate(text):
                    confidence = self.calculate_confidence(text)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            "plate_number": text,
                            "plate_type": self.get_plate_type(text),
                            "confidence": confidence,
                            "processing_time": (time.time() - start_time) * 1000,
                            "success": True,
                            "method": "simple_ocr",
                            "note": f"识别结果: {text}"
                        }

            if best_result:
                return best_result
            else:
                return {
                    "plate_number": "OCR识别失败",
                    "plate_type": "未知",
                    "confidence": 0.0,
                    "processing_time": (time.time() - start_time) * 1000,
                    "success": False,
                    "method": "ocr_failed"
                }

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

# 初始化识别器
recognizer = SimpleOCRRecognizer()

# 数据库初始化
def init_db():
    """初始化数据库"""
    try:
        conn = sqlite3.connect('simple_recognition_history.db')
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                plate_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                processing_time REAL NOT NULL,
                image_path TEXT,
                method TEXT,
                note TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")

# 数据模型
class RecognitionResult(BaseModel):
    plate_number: str
    plate_type: str
    confidence: float
    processing_time: float
    success: bool

# 保存历史记录
def save_to_history(result: Dict[str, Any], image_path: str = None):
    """保存识别结果到数据库"""
    try:
        conn = sqlite3.connect('simple_recognition_history.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO recognition_history
            (plate_number, plate_type, confidence, processing_time, image_path, method, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['plate_number'],
            result['plate_type'],
            result['confidence'],
            result['processing_time'],
            image_path,
            result.get('method', 'unknown'),
            result.get('note', '')
        ))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"保存历史记录失败: {e}")

# API端点
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>简单可靠车牌识别系统</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; }
            .container { background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); max-width: 800px; width: 90%; }
            h1 { text-align: center; color: #333; margin-bottom: 2rem; font-size: 2.5rem; }
            .upload-section { margin-bottom: 2rem; text-align: center; }
            .file-input { display: none; }
            .file-label { display: inline-block; padding: 12px 24px; background: #4CAF50; color: white; border-radius: 8px; cursor: pointer; transition: background 0.3s; }
            .file-label:hover { background: #45a049; }
            .result { margin-top: 2rem; padding: 1rem; border-radius: 8px; background: #f5f5f5; display: none; }
            .result.show { display: block; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .status { text-align: center; margin-bottom: 1rem; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .online { background: #4CAF50; }
            .offline { background: #f44336; }
            .info-box { background: #e3f2fd; border: 1px solid #bbdefb; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
            .info-box h3 { color: #1976d2; margin-bottom: 0.5rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 简单可靠车牌识别系统</h1>

            <div class="status">
                <span class="status-indicator online"></span>
                <span id="statusText">服务器状态: 在线</span>
            </div>

            <div class="info-box">
                <h3>系统特点</h3>
                <p>• 基于真实OCR技术</p>
                <p>• 简单可靠的识别算法</p>
                <p>• 快速图像预处理</p>
                <p>• 智能车牌定位</p>
            </div>

            <div class="upload-section">
                <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="uploadFile(this)">
                <label for="fileInput" class="file-label">选择图片进行识别</label>
            </div>

            <div id="result" class="result"></div>
        </div>

        <script>
            function uploadFile(input) {
                const file = input.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);

                document.getElementById('result').innerHTML = '<div class="loading">正在识别中...</div>';
                document.getElementById('result').classList.add('show');

                fetch('/recognize', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    displayResult(data);
                })
                .catch(error => {
                    document.getElementById('result').innerHTML = '<div class="error">识别失败: ' + error.message + '</div>';
                });
            }

            function displayResult(data) {
                const resultDiv = document.getElementById('result');
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="success">
                            <h3>识别成功！</h3>
                            <p><strong>车牌号码:</strong> ${data.plate_number}</p>
                            <p><strong>车牌类型:</strong> ${data.plate_type}</p>
                            <p><strong>置信度:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
                            <p><strong>处理时间:</strong> ${data.processing_time.toFixed(2)}ms</p>
                            <p><strong>识别方法:</strong> ${data.method}</p>
                            <p><strong>备注:</strong> ${data.note}</p>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = '<div class="error">识别失败，请重试</div>';
                }
            }

            function checkServerStatus() {
                fetch('/health')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('statusText').textContent = '服务器状态: 在线';
                        document.querySelector('.status-indicator').className = 'status-indicator online';
                    })
                    .catch(error => {
                        document.getElementById('statusText').textContent = '服务器状态: 离线';
                        document.querySelector('.status-indicator').className = 'status-indicator offline';
                    });
            }

            checkServerStatus();
            setInterval(checkServerStatus, 30000);
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_type": "SimpleOCRRecognizer",
        "device": "cpu",
        "tesseract_available": tesseract_available,
        "recognition_method": "simple_ocr"
    }

@app.post("/recognize", response_model=RecognitionResult)
async def recognize_plate(file: UploadFile = File(...)):
    """单个车牌识别"""
    try:
        start_time = time.time()

        # 读取图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 进行识别
        result = recognizer.recognize(image, file.filename)

        # 保存到历史记录
        save_to_history(result)

        return result

    except Exception as e:
        logger.error(f"识别失败: {e}")
        return {
            "plate_number": "处理异常",
            "plate_type": "未知",
            "confidence": 0.0,
            "processing_time": 20.0,
            "success": False
        }

@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    try:
        conn = sqlite3.connect('simple_recognition_history.db')
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM recognition_history")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM recognition_history WHERE confidence >= 0.5")
        successful_count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM recognition_history")
        avg_confidence = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_recognitions": total_count,
            "successful_recognitions": successful_count,
            "success_rate": (successful_count / total_count * 100) if total_count > 0 else 0,
            "average_confidence": avg_confidence
        }

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return {
            "total_recognitions": 0,
            "successful_recognitions": 0,
            "success_rate": 0.0,
            "average_confidence": 0.0
        }

@app.get("/history")
async def get_history():
    """获取历史记录"""
    try:
        conn = sqlite3.connect('simple_recognition_history.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT plate_number, plate_type, confidence, processing_time, method, timestamp
            FROM recognition_history
            ORDER BY timestamp DESC
            LIMIT 100
        ''')

        history = []
        for row in cursor.fetchall():
            history.append({
                "plate_number": row[0],
                "plate_type": row[1],
                "confidence": row[2],
                "processing_time": row[3],
                "method": row[4],
                "timestamp": row[5]
            })

        conn.close()

        return {"history": history}

    except Exception as e:
        logger.error(f"获取历史记录失败: {e}")
        return {"history": []}

if __name__ == "__main__":
    # 初始化数据库
    init_db()

    print("简单可靠车牌识别系统启动")
    print("特点:")
    print("- 基于真实OCR技术")
    print("- 简单可靠的识别算法")
    print("- 快速图像预处理")
    print("- 智能车牌定位")
    print("=" * 50)

    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8021, reload=False)