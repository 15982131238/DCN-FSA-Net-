#!/usr/bin/env python3
"""
生产级车牌识别系统 - 优化检测算法，确保准确识别
"""

import os
import sys
import logging
import time
import json
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
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
app = FastAPI(title="生产级车牌识别系统", version="2.0.0")

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

class ProductionOCRRecognizer:
    """生产级OCR识别器"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tesseract_available = tesseract_available
        logger.info(f"初始化生产级OCR识别器，设备: {self.device}")
        logger.info(f"Tesseract可用: {self.tesseract_available}")

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """图像增强"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 降噪
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        return denoised

    def detect_by_color(self, image: np.ndarray) -> List[np.ndarray]:
        """基于颜色检测车牌"""
        plates = []

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 蓝色车牌范围
        lower_blue = np.array([100, 80, 46])
        upper_blue = np.array([124, 255, 255])

        # 绿色车牌范围（新能源）
        lower_green = np.array([35, 80, 46])
        upper_green = np.array([77, 255, 255])

        # 黄色车牌范围
        lower_yellow = np.array([20, 80, 46])
        upper_yellow = np.array([35, 255, 255])

        # 检测蓝色车牌
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_result = cv2.bitwise_and(image, image, mask=blue_mask)

        # 检测绿色车牌
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_result = cv2.bitwise_and(image, image, mask=green_mask)

        # 检测黄色车牌
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_result = cv2.bitwise_and(image, image, mask=yellow_mask)

        # 合并结果
        combined_mask = blue_mask | green_mask | yellow_mask
        combined_result = cv2.bitwise_and(image, image, mask=combined_mask)

        # 转换为灰度图
        gray = cv2.cvtColor(combined_result, cv2.COLOR_BGR2GRAY)

        # 二值化
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            # 获取最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 计算宽高比
            width = max(rect[1][0], rect[1][1])
            height = min(rect[1][0], rect[1][1])
            aspect_ratio = width / height if height > 0 else 0

            if 2.0 <= aspect_ratio <= 5.5:
                # 获取旋转后的车牌区域
                x, y, w, h = cv2.boundingRect(contour)
                plate_roi = image[y:y+h, x:x+w]

                if plate_roi.size > 0:
                    plates.append(plate_roi)

        return plates

    def detect_by_contours(self, image: np.ndarray, enhanced: np.ndarray) -> List[np.ndarray]:
        """基于轮廓检测车牌"""
        plates = []

        # 边缘检测
        edges = cv2.Canny(enhanced, 50, 150)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            # 计算边界矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 宽高比检查
            aspect_ratio = w / h
            if 1.5 <= aspect_ratio <= 6.0:
                plate_roi = image[y:y+h, x:x+w]
                if plate_roi.size > 0:
                    plates.append(plate_roi)

        return plates

    def detect_by_gradient(self, image: np.ndarray) -> List[np.ndarray]:
        """基于梯度检测车牌"""
        plates = []

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sobel算子
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)

        # 二值化
        _, binary = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if 1.5 <= aspect_ratio <= 6.0:
                plate_roi = image[y:y+h, x:x+w]
                if plate_roi.size > 0:
                    plates.append(plate_roi)

        return plates

    def extract_text_advanced(self, image: np.ndarray) -> str:
        """高级文字提取"""
        if not self.tesseract_available:
            return ""

        try:
            # 图像预处理
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 调整大小
            height, width = gray.shape
            if width < 200:
                scale = 200 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height))

            # 自适应阈值
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # 降噪
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 转换为PIL图像
            pil_image = Image.fromarray(binary)

            # 配置Tesseract
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领'

            # 提取文字
            text = pytesseract.image_to_string(pil_image, config=custom_config)

            # 清理结果
            text = text.strip().replace('\n', '').replace('\r', '').replace(' ', '')

            return text

        except Exception as e:
            logger.error(f"高级文字提取失败: {e}")
            return ""

    def validate_plate_number(self, text: str) -> bool:
        """验证车牌号码"""
        if not text or len(text) < 7 or len(text) > 9:
            return False

        # 检查字符有效性
        valid_chars = plate_chars + plate_numbers + plate_letters
        return all(char in valid_chars for char in text)

    def determine_plate_type(self, plate_number: str) -> str:
        """确定车牌类型"""
        if not plate_number:
            return "未知"

        # 新能源车牌
        if len(plate_number) == 8:
            return "绿牌"

        if len(plate_number) == 7 and plate_number[1] in ['D', 'F']:
            return "绿牌"

        # 黄牌
        if plate_number[1] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M']:
            if plate_number.startswith('使'):
                return "使领馆"
            return "黄牌"

        # 默认蓝牌
        return "蓝牌"

    def calculate_confidence_score(self, text: str, image_quality: float) -> float:
        """计算置信度分数"""
        if not text:
            return 0.0

        confidence = 0.3

        # 长度检查
        if len(text) in [7, 8]:
            confidence += 0.2

        # 格式检查
        if self.validate_plate_number(text):
            confidence += 0.2

        # 首字符检查
        if text[0] in plate_chars:
            confidence += 0.1

        # 第二字符检查
        if text[1] in plate_letters:
            confidence += 0.1

        # 图像质量
        confidence += image_quality * 0.1

        return min(confidence, 0.99)

    def recognize_license_plate(self, image: Image.Image, filename: str) -> Dict[str, Any]:
        """主识别函数"""
        start_time = time.time()

        try:
            # 转换为OpenCV格式
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # 图像增强
            enhanced = self.enhance_image(cv_image)

            # 多种检测方法
            all_plates = []

            # 方法1：颜色检测
            color_plates = self.detect_by_color(cv_image)
            all_plates.extend([(plate, 'color') for plate in color_plates])

            # 方法2：轮廓检测
            contour_plates = self.detect_by_contours(cv_image, enhanced)
            all_plates.extend([(plate, 'contour') for plate in contour_plates])

            # 方法3：梯度检测
            gradient_plates = self.detect_by_gradient(cv_image)
            all_plates.extend([(plate, 'gradient') for plate in gradient_plates])

            if not all_plates:
                return {
                    "plate_number": "未检测到车牌",
                    "plate_type": "未知",
                    "confidence": 0.0,
                    "processing_time": (time.time() - start_time) * 1000,
                    "success": False,
                    "method": "detection_failed",
                    "note": "未找到车牌区域"
                }

            # 对每个候选车牌进行识别
            best_result = None
            best_confidence = 0

            for plate_image, method in all_plates:
                # 提取文字
                text = self.extract_text_advanced(plate_image)

                if text and self.validate_plate_number(text):
                    # 计算图像质量
                    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image
                    image_quality = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000

                    # 计算置信度
                    confidence = self.calculate_confidence_score(text, image_quality)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            "plate_number": text,
                            "plate_type": self.determine_plate_type(text),
                            "confidence": confidence,
                            "processing_time": (time.time() - start_time) * 1000,
                            "success": True,
                            "method": f"production_{method}",
                            "note": f"真实OCR识别: {text}",
                            "detection_count": len(all_plates)
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
                    "method": "ocr_failed",
                    "note": f"检测到{len(all_plates)}个候选区域，但OCR失败"
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
recognizer = ProductionOCRRecognizer()

# 数据库初始化
def init_db():
    """初始化数据库"""
    try:
        conn = sqlite3.connect('production_recognition_history.db')
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
                detection_count INTEGER,
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
        conn = sqlite3.connect('production_recognition_history.db')
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO recognition_history
            (plate_number, plate_type, confidence, processing_time, image_path, method, note, detection_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['plate_number'],
            result['plate_type'],
            result['confidence'],
            result['processing_time'],
            image_path,
            result.get('method', 'unknown'),
            result.get('note', ''),
            result.get('detection_count', 0)
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
        <title>生产级车牌识别系统</title>
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
            <h1>🚗 生产级车牌识别系统</h1>

            <div class="status">
                <span class="status-indicator online"></span>
                <span id="statusText">服务器状态: 在线</span>
            </div>

            <div class="info-box">
                <h3>系统特点</h3>
                <p>• 多种检测算法（颜色、轮廓、梯度）</p>
                <p>• 高级图像增强技术</p>
                <p>• 真实OCR文字提取</p>
                <p>• 智能车牌类型识别</p>
                <p>• 工程级可靠性</p>
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
                            <p><strong>检测数量:</strong> ${data.detection_count || 0}</p>
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
        "model_type": "ProductionOCRRecognizer",
        "device": "cpu",
        "tesseract_available": tesseract_available,
        "recognition_method": "production_ocr",
        "detection_methods": ["color", "contour", "gradient"]
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
        result = recognizer.recognize_license_plate(image, file.filename)

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
        conn = sqlite3.connect('production_recognition_history.db')
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
        conn = sqlite3.connect('production_recognition_history.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT plate_number, plate_type, confidence, processing_time, method, note, detection_count, timestamp
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
                "note": row[5],
                "detection_count": row[6],
                "timestamp": row[7]
            })

        conn.close()

        return {"history": history}

    except Exception as e:
        logger.error(f"获取历史记录失败: {e}")
        return {"history": []}

if __name__ == "__main__":
    # 初始化数据库
    init_db()

    print("生产级车牌识别系统启动")
    print("特点:")
    print("- 多种检测算法（颜色、轮廓、梯度）")
    print("- 高级图像增强技术")
    print("- 真实OCR文字提取")
    print("- 智能车牌类型识别")
    print("- 工程级可靠性")
    print("=" * 50)

    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8022, reload=False)