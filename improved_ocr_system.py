#!/usr/bin/env python3
"""
改进版真实OCR车牌识别系统
修复图片处理和OCR识别问题，确保准确识别
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import cv2
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from starlette.middleware.cors import CORSMiddleware

# 尝试导入OCR库
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("Tesseract OCR已加载")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract OCR不可用，将使用OpenCV进行文字检测")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="改进版真实OCR车牌识别系统", version="8.0.0")

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

class ImprovedOCRRecognizer:
    """改进版OCR识别器 - 解决图片处理问题"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tesseract_available = TESSERACT_AVAILABLE
        logger.info(f"初始化改进版OCR识别器，设备: {self.device}")
        logger.info(f"Tesseract可用: {self.tesseract_available}")

    def safe_open_image(self, image_data: bytes, filename: str) -> Optional[Image.Image]:
        """安全打开图片文件，处理各种格式"""
        try:
            # 尝试直接打开
            image = Image.open(io.BytesIO(image_data))

            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 验证图片尺寸
            if image.size[0] < 20 or image.size[1] < 20:
                logger.warning(f"图片尺寸过小: {image.size}")
                return None

            return image

        except Exception as e:
            logger.error(f"图片打开失败: {e}")

            # 尝试使用OpenCV作为备选方案
            try:
                nparr = np.frombuffer(image_data, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if cv_image is not None:
                    # 转换为PIL格式
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(cv_image)
                    return image
            except Exception as e2:
                logger.error(f"OpenCV备选方案也失败: {e2}")

            return None

    def preprocess_image(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """改进的图像预处理"""
        # 转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 调整大小，保持宽高比
        height, width = cv_image.shape[:2]
        max_size = 1200
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_image = cv2.resize(cv_image, (new_width, new_height))

        original = cv_image.copy()

        # 多种预处理方法
        results = {}

        # 1. 标准灰度化
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        results['gray'] = gray

        # 2. 直方图均衡化
        equalized = cv2.equalizeHist(gray)
        results['equalized'] = equalized

        # 3. 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        results['blurred'] = blurred

        # 4. 自适应阈值
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results['adaptive'] = adaptive

        # 5. 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        results['edges'] = edges

        return results

    def locate_license_plates(self, image: np.ndarray) -> List[np.ndarray]:
        """改进的车牌定位算法"""
        plates = []

        try:
            # 方法1：基于轮廓检测
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)

                # 过滤小面积
                if area < 1000:
                    continue

                # 计算边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # 车牌宽高比通常在2-6之间
                if 1.5 <= aspect_ratio <= 7.0:
                    # 提取候选区域
                    plate_roi = image[y:y+h, x:x+w]
                    if plate_roi.size > 0:
                        plates.append(plate_roi)

            # 方法2：基于形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

            morph_contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in morph_contours:
                area = cv2.contourArea(contour)
                if area < 800:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                if 1.5 <= aspect_ratio <= 7.0:
                    plate_roi = image[y:y+h, x:x+w]
                    if plate_roi.size > 0:
                        plates.append(plate_roi)

        except Exception as e:
            logger.error(f"车牌定位失败: {e}")

        return plates

    def extract_text_robust(self, image: np.ndarray) -> str:
        """鲁棒的文字提取"""
        if not self.tesseract_available:
            return ""

        try:
            # 转换为PIL格式
            if len(image.shape) == 2:
                # 灰度图
                pil_image = Image.fromarray(image)
            else:
                # 彩色图
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # 增强对比度
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)

            # 调整大小
            width, height = pil_image.size
            if width < 200:
                new_width = 400
                new_height = int(height * new_width / width)
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

            # Tesseract配置
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领'

            # 尝试OCR识别
            text = pytesseract.image_to_string(pil_image, config=config)

            # 清理结果
            text = re.sub(r'[^A-Z0-9京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]', '', text.upper())

            return text.strip()

        except Exception as e:
            logger.error(f"文字提取失败: {e}")
            return ""

    def validate_plate_format(self, text: str) -> bool:
        """验证车牌格式"""
        if len(text) < 7 or len(text) > 10:
            return False

        # 检查是否包含省份简称
        has_province = any(text.startswith(province) for province in PLATE_PROVINCES)

        # 检查是否包含字母和数字
        has_letters = any(c in PLATE_LETTERS for c in text)
        has_numbers = any(c in PLATE_NUMBERS for c in text)

        return has_province and has_letters and has_numbers

    def determine_plate_type(self, text: str) -> str:
        """确定车牌类型"""
        if len(text) == 8:
            return "新能源车牌"
        elif text[0] in PLATE_PROVINCES:
            return "蓝牌"
        else:
            return "未知"

    def calculate_confidence(self, text: str) -> float:
        """计算置信度"""
        if not text:
            return 0.0

        # 基础置信度
        confidence = 0.5

        # 长度加分
        if 7 <= len(text) <= 8:
            confidence += 0.2

        # 省份简称加分
        if text[0] in PLATE_PROVINCES:
            confidence += 0.2

        # 字符构成加分
        valid_chars = sum(1 for c in text if c in PLATE_PROVINCES + PLATE_LETTERS + PLATE_NUMBERS)
        confidence += (valid_chars / len(text)) * 0.1

        return min(confidence, 1.0)

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

            # 图像预处理
            processed_images = self.preprocess_image(image)

            # 尝试不同预处理方法
            all_plates = []

            for method_name, processed_image in processed_images.items():
                if method_name in ['adaptive', 'equalized', 'blurred']:
                    plates = self.locate_license_plates(processed_image)
                    all_plates.extend([(plate, method_name) for plate in plates])

            # 如果没有检测到车牌，尝试使用原图
            if not all_plates:
                gray_plates = self.locate_license_plates(processed_images['gray'])
                all_plates.extend([(plate, 'gray') for plate in gray_plates])

            if not all_plates:
                return {
                    "plate_number": "未检测到车牌",
                    "plate_type": "未知",
                    "confidence": 0.0,
                    "processing_time": (time.time() - start_time) * 1000,
                    "success": False,
                    "method": "detection_failed"
                }

            # 对每个候选车牌进行OCR识别
            best_result = None
            best_confidence = 0

            for plate_image, method_name in all_plates:
                extracted_text = self.extract_text_robust(plate_image)

                if extracted_text and self.validate_plate_format(extracted_text):
                    plate_type = self.determine_plate_type(extracted_text)
                    confidence = self.calculate_confidence(extracted_text)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            "plate_number": extracted_text,
                            "plate_type": plate_type,
                            "confidence": confidence,
                            "processing_time": (time.time() - start_time) * 1000,
                            "success": True,
                            "method": f"improved_ocr_{method_name}",
                            "note": f"真实OCR识别结果"
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

# 创建识别器实例
recognizer = ImprovedOCRRecognizer()

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
    return {"message": "改进版真实OCR车牌识别系统", "version": "8.0.0"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_type": "ImprovedOCRRecognizer",
        "device": str(recognizer.device),
        "tesseract_available": recognizer.tesseract_available
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

if __name__ == "__main__":
    print("启动改进版真实OCR车牌识别系统...")
    uvicorn.run(app, host="0.0.0.0", port=8023, reload=False)