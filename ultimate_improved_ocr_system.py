#!/usr/bin/env python3
"""
终极改进版OCR车牌识别系统
专门解决OCR文字提取失败的问题
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
app = FastAPI(title="终极改进版OCR车牌识别系统", version="9.0.0")

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

class UltimateOCRRecognizer:
    """终极改进版OCR识别器"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tesseract_available = TESSERACT_AVAILABLE
        logger.info(f"初始化终极改进版OCR识别器，设备: {self.device}")
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

    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """增强图像以提高OCR识别率"""
        # 转换为OpenCV格式进行处理
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 1. 调整大小
        height, width = cv_image.shape[:2]
        if width < 400:
            scale = 400 / width
            new_width = 400
            new_height = int(height * scale)
            cv_image = cv2.resize(cv_image, (new_width, new_height))

        # 2. 高斯模糊去噪
        blurred = cv2.GaussianBlur(cv_image, (3, 3), 0)

        # 3. 锐化处理
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        # 4. 对比度增强
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l,a,b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 转换回PIL格式
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

    def locate_license_plates_enhanced(self, image: np.ndarray) -> List[np.ndarray]:
        """增强的车牌定位算法"""
        plates = []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 方法1：边缘检测 + 轮廓检测
            edges = cv2.Canny(gray, 50, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:  # 降低面积阈值
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # 放宽宽高比限制
                if 1.0 <= aspect_ratio <= 8.0:
                    plate_roi = image[y:y+h, x:x+w]
                    if plate_roi.size > 0:
                        plates.append(plate_roi)

            # 方法2：基于颜色检测（蓝色车牌）
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            blue_morph = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

            blue_contours, _ = cv2.findContours(blue_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in blue_contours:
                area = cv2.contourArea(contour)
                if area < 300:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                if 1.0 <= aspect_ratio <= 8.0:
                    plate_roi = image[y:y+h, x:x+w]
                    if plate_roi.size > 0:
                        plates.append(plate_roi)

            # 方法3：基于形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            morph_contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in morph_contours:
                area = cv2.contourArea(contour)
                if area < 300:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                if 1.0 <= aspect_ratio <= 8.0:
                    plate_roi = image[y:y+h, x:x+w]
                    if plate_roi.size > 0:
                        plates.append(plate_roi)

            # 如果没有找到车牌，尝试整个图像
            if not plates:
                plates.append(image)

        except Exception as e:
            logger.error(f"车牌定位失败: {e}")
            plates.append(image)  # 使用整个图像作为备选

        return plates

    def extract_text_with_multiple_methods(self, image: np.ndarray) -> str:
        """使用多种方法提取文字"""
        if not self.tesseract_available:
            return ""

        best_text = ""
        best_confidence = 0

        # 方法1：标准Tesseract
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领'
            text = pytesseract.image_to_string(pil_image, config=config)
            text = re.sub(r'[^A-Z0-9京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]', '', text.upper())
            if len(text) >= 5:
                confidence = self.calculate_text_confidence(text)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_text = text
        except Exception as e:
            logger.error(f"标准Tesseract失败: {e}")

        # 方法2：增强对比度后OCR
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(2.0)
            text = pytesseract.image_to_string(enhanced, config=config)
            text = re.sub(r'[^A-Z0-9京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]', '', text.upper())
            if len(text) >= 5:
                confidence = self.calculate_text_confidence(text)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_text = text
        except Exception as e:
            logger.error(f"增强对比度OCR失败: {e}")

        # 方法3：二值化后OCR
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_image = Image.fromarray(binary)
            text = pytesseract.image_to_string(pil_image, config=config)
            text = re.sub(r'[^A-Z0-9京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]', '', text.upper())
            if len(text) >= 5:
                confidence = self.calculate_text_confidence(text)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_text = text
        except Exception as e:
            logger.error(f"二值化OCR失败: {e}")

        return best_text.strip()

    def calculate_text_confidence(self, text: str) -> float:
        """计算文本置信度"""
        if not text:
            return 0.0

        confidence = 0.0

        # 长度加分
        if 7 <= len(text) <= 8:
            confidence += 0.3
        elif 5 <= len(text) <= 10:
            confidence += 0.2

        # 省份简称加分
        if text[0] in PLATE_PROVINCES:
            confidence += 0.3

        # 字符构成加分
        valid_chars = sum(1 for c in text if c in PLATE_PROVINCES + PLATE_LETTERS + PLATE_NUMBERS)
        confidence += (valid_chars / len(text)) * 0.2

        # 车牌格式加分
        if len(text) >= 2 and text[1] in PLATE_LETTERS:
            confidence += 0.2

        return min(confidence, 1.0)

    def validate_plate_format(self, text: str) -> bool:
        """验证车牌格式"""
        if len(text) < 5 or len(text) > 10:
            return False

        # 检查是否包含省份简称
        has_province = any(text.startswith(province) for province in PLATE_PROVINCES)

        # 检查是否包含字母和数字
        has_letters = any(c in PLATE_LETTERS for c in text)
        has_numbers = any(c in PLATE_NUMBERS for c in text)

        return has_province and (has_letters or has_numbers)

    def determine_plate_type(self, text: str) -> str:
        """确定车牌类型"""
        if len(text) == 8:
            return "新能源车牌"
        elif text[0] in PLATE_PROVINCES:
            return "蓝牌"
        else:
            return "未知"

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

            # 增强图像
            enhanced_image = self.enhance_image_for_ocr(image)

            # 转换为OpenCV格式
            cv_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

            # 定位车牌
            plates = self.locate_license_plates_enhanced(cv_image)

            if not plates:
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

            for plate_image in plates:
                extracted_text = self.extract_text_with_multiple_methods(plate_image)

                if extracted_text:
                    # 清理文本
                    extracted_text = extracted_text.strip()

                    if self.validate_plate_format(extracted_text):
                        plate_type = self.determine_plate_type(extracted_text)
                        confidence = self.calculate_text_confidence(extracted_text)

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = {
                                "plate_number": extracted_text,
                                "plate_type": plate_type,
                                "confidence": confidence,
                                "processing_time": (time.time() - start_time) * 1000,
                                "success": True,
                                "method": "ultimate_ocr_enhanced",
                                "note": "终极改进版OCR识别结果"
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
recognizer = UltimateOCRRecognizer()

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
    return {"message": "终极改进版OCR车牌识别系统", "version": "9.0.0"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_type": "UltimateOCRRecognizer",
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
    print("启动终极改进版OCR车牌识别系统...")
    uvicorn.run(app, host="0.0.0.0", port=8024, reload=False)