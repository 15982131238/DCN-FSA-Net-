#!/usr/bin/env python3
"""
测试修正版OCR识别系统
"""

import requests
import json
import os

def test_fixed_ocr_system():
    """测试修正版OCR系统"""
    url = "http://localhost:8019/recognize"

    # 测试图片列表
    test_images = [
        "test_zhejiang_plate.jpg",
        "test_guangdong_plate.jpg",
        "test_plate.jpg"
    ]

    print("开始测试修正版OCR识别系统...")
    print("=" * 50)

    for image_name in test_images:
        if os.path.exists(image_name):
            print(f"\n测试图片: {image_name}")

            try:
                with open(image_name, 'rb') as f:
                    files = {'file': (image_name, f, 'image/jpeg')}
                    response = requests.post(url, files=files)

                if response.status_code == 200:
                    result = response.json()
                    print(f"识别结果: {result['plate_number']}")
                    print(f"车牌类型: {result['plate_type']}")
                    print(f"置信度: {result['confidence']:.2f}")
                    print(f"处理时间: {result['processing_time']:.2f}ms")
                    print(f"识别方法: {result.get('method', 'unknown')}")
                    print(f"备注: {result.get('note', '')}")
                    print(f"成功: {result['success']}")
                else:
                    print(f"请求失败: {response.status_code}")

            except Exception as e:
                print(f"测试失败: {e}")
        else:
            print(f"图片文件不存在: {image_name}")

    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    test_fixed_ocr_system()