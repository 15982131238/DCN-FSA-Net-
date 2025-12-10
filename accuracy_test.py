#!/usr/bin/env python3
"""
License Plate Recognition System Test Script
Tests the fixed system with position encoding correction
"""

import requests
import json
import time
from pathlib import Path

def test_recognition_accuracy():
    """Test recognition accuracy"""
    base_url = "http://localhost:8001"

    print("车牌识别系统准确率测试")
    print("=" * 50)

    # Test cases with expected results
    test_cases = [
        {"image": "test_plate.jpg", "expected": "京A12345"},
    ]

    results = []

    for case in test_cases:
        image_path = case["image"]
        expected = case["expected"]

        print(f"\n测试图片: {image_path}")
        print(f"期望结果: {expected}")

        if not Path(image_path).exists():
            print(f"图片不存在: {image_path}")
            continue

        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                start_time = time.time()
                response = requests.post(f"{base_url}/recognize", files=files, timeout=10)
                end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                actual = result.get('plate_number', '识别失败')
                confidence = result.get('confidence', 0)
                processing_time = result.get('processing_time', 0)

                print(f"识别结果: {actual}")
                print(f"置信度: {confidence*100:.1f}%")
                print(f"处理时间: {processing_time:.2f}ms")

                # Check if recognition is successful
                if actual != "识别失败" and confidence > 0.1:
                    print("状态: 识别成功")

                    # Check accuracy
                    if actual == expected:
                        print("准确率: 100% (完全匹配)")
                        accuracy = 1.0
                    else:
                        print("准确率: 部分匹配")
                        accuracy = 0.5
                else:
                    print("状态: 识别失败")
                    accuracy = 0.0

                results.append({
                    'image': image_path,
                    'expected': expected,
                    'actual': actual,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'accuracy': accuracy
                })

            else:
                print(f"请求失败: {response.status_code}")
                results.append({
                    'image': image_path,
                    'expected': expected,
                    'actual': f"HTTP_{response.status_code}",
                    'confidence': 0,
                    'processing_time': 0,
                    'accuracy': 0
                })

        except Exception as e:
            print(f"测试失败: {e}")
            results.append({
                'image': image_path,
                'expected': expected,
                'actual': f"ERROR: {e}",
                'confidence': 0,
                'processing_time': 0,
                'accuracy': 0
            })

    # Calculate overall accuracy
    if results:
        total_accuracy = sum(r['accuracy'] for r in results) / len(results)
        successful_recognitions = sum(1 for r in results if r['accuracy'] > 0)

        print("\n" + "=" * 50)
        print("测试结果汇总:")
        print(f"总测试图片数: {len(results)}")
        print(f"成功识别数: {successful_recognitions}")
        print(f"整体准确率: {total_accuracy*100:.1f}%")

        # System health check
        try:
            health_response = requests.get(f"{base_url}/health", timeout=5)
            if health_response.status_code == 200:
                health = health_response.json()
                print(f"\n系统状态: {health.get('status', 'unknown')}")
                print(f"模型加载: {health.get('model_loaded', False)}")
                print(f"运行设备: {health.get('device', 'unknown')}")
        except:
            print("\n系统状态: 无法连接")

        return total_accuracy > 0.8  # Consider 80%+ as good accuracy

    return False

if __name__ == "__main__":
    success = test_recognition_accuracy()
    if success:
        print("\n✅ 系统测试通过 - 识别率达到预期标准")
    else:
        print("\n❌ 系统测试失败 - 识别率需要改进")
        exit(1)