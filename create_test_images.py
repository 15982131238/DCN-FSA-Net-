#!/usr/bin/env python3
"""
创建真实的车牌测试图片
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_plate_image(plate_text, filename):
    """创建包含指定车牌文字的测试图片"""

    # 创建蓝色背景图片（标准车牌尺寸：440x140）
    img = Image.new('RGB', (440, 140), color=(0, 51, 102))  # 深蓝色背景
    draw = ImageDraw.Draw(img)

    try:
        # 尝试使用较大的字体
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
        except:
            # 使用默认字体
            font = ImageFont.load_default()

    # 绘制白色文字
    draw.text((20, 40), plate_text, font=font, fill='white')

    # 添加车牌边框
    draw.rectangle([5, 5, 435, 135], outline='white', width=3)

    # 保存图片
    img.save(filename)
    print(f"已创建测试图片: {filename}, 车牌号: {plate_text}")

# 创建几个测试图片
test_plates = [
    ("京A12345", "test_beijing_plate.jpg"),
    ("沪B67890", "test_shanghai_plate.jpg"),
    ("粤C24680", "test_guangdong_plate.jpg"),
    ("浙E86420", "test_zhejiang_plate.jpg")
]

for plate, filename in test_plates:
    create_plate_image(plate, filename)

print("所有测试图片创建完成!")