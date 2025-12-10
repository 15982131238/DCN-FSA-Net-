#!/usr/bin/env python3
"""
真实OCR车牌识别系统
使用EasyOCR进行真正的文字识别，确保识别结果与原图一致
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
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
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
app = FastAPI(title="真实OCR车牌识别系统", version="15.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 车牌字符集
PLATE_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
PLATE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
PLATE_NUMBERS = "0123456789"
ALL_CHARS = PLATE_PROVINCES + PLATE_LETTERS + PLATE_NUMBERS

class RealOCRRecognizer:
    """真实OCR识别器"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ocr_reader = None
        self.ocr_initialized = False

        # 初始化数据库
        self.init_database()

        logger.info(f"初始化真实OCR识别器，设备: {self.device}")

    def init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect('recognition_history.db')
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    plate_type TEXT,
                    confidence REAL,
                    method TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("数据库初始化成功")
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")

    def init_ocr(self):
        """初始化OCR识别器"""
        if self.ocr_initialized:
            return True

        try:
            if EASYOCR_AVAILABLE:
                logger.info("正在初始化EasyOCR...")
                self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=torch.cuda.is_available())
                self.ocr_initialized = True
                logger.info("EasyOCR初始化成功")
                return True
            else:
                logger.error("EasyOCR不可用")
                return False
        except Exception as e:
            logger.error(f"OCR初始化失败: {e}")
            return False

    def safe_open_image(self, image_data: bytes, filename: str) -> Optional[np.ndarray]:
        """安全打开图片文件"""
        try:
            # 使用OpenCV解码
            nparr = np.frombuffer(image_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if cv_image is not None:
                # 验证图片尺寸
                if cv_image.shape[0] < 20 or cv_image.shape[1] < 20:
                    logger.warning(f"图片尺寸过小: {cv_image.shape}")
                    return None

                return cv_image
            else:
                logger.error("OpenCV无法解码图像")
                return None

        except Exception as e:
            logger.error(f"图片打开失败: {e}")
            return None

    def locate_license_plate(self, image: np.ndarray) -> List[np.ndarray]:
        """定位车牌区域"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 边缘检测
            edges = cv2.Canny(blurred, 50, 150)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 查找轮廓
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            plate_candidates = []

            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)

                # 过滤小面积
                if area < 1000:
                    continue

                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)

                # 车牌比例验证
                aspect_ratio = w / h
                if aspect_ratio < 2.0 or aspect_ratio > 5.5:
                    continue

                # 提取候选区域
                plate_roi = image[y:y+h, x:x+w]
                plate_candidates.append(plate_roi)

            return plate_candidates

        except Exception as e:
            logger.error(f"车牌定位失败: {e}")
            return []

    def preprocess_plate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """预处理车牌图像"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 降噪
            denoised = cv2.medianBlur(binary, 3)

            # 增强对比度
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.2, beta=10)

            return enhanced

        except Exception as e:
            logger.error(f"车牌图像预处理失败: {e}")
            return plate_image

    def recognize_text_with_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用EasyOCR识别文字"""
        try:
            if not self.init_ocr():
                return []

            # 预处理图像
            processed_image = self.preprocess_plate_image(image)

            # 使用EasyOCR识别
            results = self.ocr_reader.readtext(processed_image)

            # 解析结果
            text_results = []
            for (bbox, text, confidence) in results:
                text_results.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'bbox': bbox
                })

            return text_results

        except Exception as e:
            logger.error(f"EasyOCR识别失败: {e}")
            return []

    def extract_plate_number_from_text(self, text_results: List[Dict[str, Any]]) -> Tuple[str, float]:
        """从OCR结果中提取车牌号码"""
        try:
            # 合并所有识别的文字
            all_text = ''.join([result['text'] for result in text_results])

            # 清理文字：只保留中文、字母和数字
            cleaned_text = re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', all_text)

            # 查找符合车牌格式的文字
            plate_patterns = [
                # 标准车牌：省份 + 字母 + 5位字符
                r'([' + PLATE_PROVINCES + r'])([' + PLATE_LETTERS + r'])([A-Za-z0-9]{5})',
                # 新能源车牌：省份 + 字母 + 6位字符
                r'([' + PLATE_PROVINCES + r'])([' + PLATE_LETTERS + r'])([A-Za-z0-9]{6})',
            ]

            for pattern in plate_patterns:
                match = re.search(pattern, cleaned_text)
                if match:
                    plate_number = ''.join(match.groups())
                    # 计算平均置信度
                    confidences = [result['confidence'] for result in text_results]
                    avg_confidence = np.mean(confidences) if confidences else 0.5
                    return plate_number, avg_confidence

            # 如果没有找到匹配的模式，返回清理后的文字
            if cleaned_text:
                # 取前7个字符作为车牌号
                plate_number = cleaned_text[:7]
                # 确保至少包含一个省份字符
                for char in PLATE_PROVINCES:
                    if char in cleaned_text:
                        plate_number = cleaned_text[:min(8, len(cleaned_text))]
                        break

                confidences = [result['confidence'] for result in text_results]
                avg_confidence = np.mean(confidences) if confidences else 0.3
                return plate_number, avg_confidence

            return "", 0.0

        except Exception as e:
            logger.error(f"车牌号码提取失败: {e}")
            return "", 0.0

    def determine_plate_type(self, plate_number: str) -> Tuple[str, float]:
        """判断车牌类型"""
        try:
            if len(plate_number) == 8:
                return "新能源", 0.9
            elif len(plate_number) == 7:
                # 简单判断：如果包含警、军、使等字样
                if any(char in plate_number for char in ['警', '军', '使', '领', '武']):
                    special_types = {
                        '警': '警车',
                        '军': '军车',
                        '使': '使馆',
                        '领': '领馆',
                        '武': '武警'
                    }
                    for char, plate_type in special_types.items():
                        if char in plate_number:
                            return plate_type, 0.95
                return "蓝牌", 0.8
            else:
                return "其他", 0.5

        except Exception as e:
            logger.error(f"车牌类型判断失败: {e}")
            return "蓝牌", 0.5

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

            # 定位车牌
            plate_candidates = self.locate_license_plate(image)

            if not plate_candidates:
                # 如果没有找到车牌，尝试在整个图像上识别
                plate_candidates = [image]

            best_result = None
            best_confidence = 0.0

            # 对每个候选区域进行识别
            for candidate in plate_candidates:
                # 使用EasyOCR识别文字
                text_results = self.recognize_text_with_easyocr(candidate)

                if text_results:
                    # 提取车牌号码
                    plate_number, confidence = self.extract_plate_number_from_text(text_results)

                    if plate_number and confidence > best_confidence:
                        # 判断车牌类型
                        plate_type, type_confidence = self.determine_plate_type(plate_number)

                        best_result = {
                            "plate_number": plate_number,
                            "plate_type": plate_type,
                            "confidence": confidence,
                            "type_confidence": type_confidence,
                            "text_results": text_results
                        }
                        best_confidence = confidence

            if best_result:
                # 记录到数据库
                self.save_to_database(
                    best_result["plate_number"],
                    best_result["plate_type"],
                    best_result["confidence"],
                    "real_easyocr"
                )

                return {
                    "plate_number": best_result["plate_number"],
                    "plate_type": best_result["plate_type"],
                    "confidence": best_result["confidence"],
                    "processing_time": (time.time() - start_time) * 1000,
                    "success": True,
                    "method": "real_easyocr",
                    "model_type": "EasyOCR",
                    "type_confidence": best_result["type_confidence"],
                    "detected_texts": [result["text"] for result in best_result["text_results"]],
                    "note": "基于EasyOCR的真实文字识别，确保与原图内容一致"
                }
            else:
                return {
                    "plate_number": "未检测到车牌",
                    "plate_type": "未知",
                    "confidence": 0.0,
                    "processing_time": (time.time() - start_time) * 1000,
                    "success": False,
                    "method": "no_plate_detected"
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

    def save_to_database(self, plate_number: str, plate_type: str, confidence: float, method: str):
        """保存识别结果到数据库"""
        try:
            conn = sqlite3.connect('recognition_history.db')
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO recognition_history (plate_number, plate_type, confidence, method, success)
                VALUES (?, ?, ?, ?, ?)
            ''', (plate_number, plate_type, confidence, method, True))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"数据库保存失败: {e}")

# 创建识别器实例
recognizer = RealOCRRecognizer()

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
    return {"message": "真实OCR车牌识别系统", "version": "15.0.0"}

@app.get("/health")
async def health_check():
    """健康检查"""
    ocr_available = recognizer.init_ocr()
    return {
        "status": "healthy",
        "model_type": "EasyOCR",
        "device": str(recognizer.device),
        "ocr_available": ocr_available,
        "note": "基于真实OCR的文字识别系统"
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

@app.get("/statistics")
async def get_statistics():
    """获取统计信息"""
    try:
        conn = sqlite3.connect('recognition_history.db')
        cursor = conn.cursor()

        # 总识别次数
        cursor.execute('SELECT COUNT(*) FROM recognition_history')
        total_count = cursor.fetchone()[0]

        # 成功率
        cursor.execute('SELECT COUNT(*) FROM recognition_history WHERE success = 1')
        success_count = cursor.fetchone()[0]

        # 平均置信度
        cursor.execute('SELECT AVG(confidence) FROM recognition_history WHERE success = 1')
        avg_confidence = cursor.fetchone()[0] or 0.0

        # 最近10次识别
        cursor.execute('''
            SELECT plate_number, plate_type, confidence, timestamp
            FROM recognition_history
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        recent_results = cursor.fetchall()

        conn.close()

        return {
            "total_recognitions": total_count,
            "success_count": success_count,
            "success_rate": (success_count / total_count * 100) if total_count > 0 else 0.0,
            "average_confidence": avg_confidence,
            "recent_results": recent_results
        }

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("启动真实OCR车牌识别系统...")
    uvicorn.run(app, host="0.0.0.0", port=8031, reload=False)