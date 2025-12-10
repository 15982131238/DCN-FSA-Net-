#!/usr/bin/env python3
"""
测试所有OCR识别系统
"""

import requests
import json
import os
import sys

def print_safe(text):
    """安全打印函数，处理Windows编码问题"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 移除emoji表情符号，只保留文本
        import re
        clean_text = re.sub(r'[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]', '', text)
        print(clean_text)

def test_system(system_name, port, test_images):
    """测试单个系统"""
    url = f"http://localhost:{port}/recognize"

    print_safe(f"\n{'='*60}")
    print_safe(f"测试 {system_name} (端口 {port})")
    print_safe(f"{'='*60}")

    success_count = 0
    total_tests = 0

    for image_name in test_images:
        if os.path.exists(image_name):
            print_safe(f"\n测试图片: {image_name}")
            total_tests += 1

            try:
                with open(image_name, 'rb') as f:
                    files = {'file': (image_name, f, 'image/jpeg')}
                    response = requests.post(url, files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    print_safe(f"  识别结果: {result['plate_number']}")
                    print_safe(f"  车牌类型: {result['plate_type']}")
                    print_safe(f"  置信度: {result['confidence']:.2f}")
                    print_safe(f"  处理时间: {result['processing_time']:.2f}ms")
                    print_safe(f"  识别方法: {result.get('method', 'unknown')}")
                    print_safe(f"  成功: {result['success']}")

                    if result['success']:
                        success_count += 1
                else:
                    print_safe(f"  请求失败: {response.status_code}")

            except requests.exceptions.ConnectionError:
                print_safe(f"  无法连接到服务器 (端口 {port})")
                break
            except Exception as e:
                print_safe(f"  测试失败: {e}")
        else:
            print_safe(f"  图片文件不存在: {image_name}")

    print_safe(f"\n{system_name} 测试结果: {success_count}/{total_tests} 成功")
    return success_count, total_tests

def check_system_health(system_name, port):
    """检查系统健康状态"""
    url = f"http://localhost:{port}/health"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            health = response.json()
            print_safe(f"  {system_name}: 在线")
            print_safe(f"     模型: {health.get('model_type', 'unknown')}")
            print_safe(f"     设备: {health.get('device', 'unknown')}")
            print_safe(f"     OCR可用: {health.get('tesseract_available', False)}")
            return True
        else:
            print_safe(f"  {system_name}: 响应异常 ({response.status_code})")
            return False
    except Exception as e:
        print_safe(f"  {system_name}: 离线 ({str(e)})")
        return False

def main():
    """主测试函数"""
    print_safe("开始测试所有OCR识别系统")
    print_safe("=" * 80)

    # 测试图片列表
    test_images = [
        "test_zhejiang_plate.jpg",
        "test_guangdong_plate.jpg",
        "test_beijing_plate.jpg",
        "test_shanghai_plate.jpg"
    ]

    # 系统列表
    systems = [
        ("生产级OCR系统", 8022),
        ("简单可靠OCR系统", 8021),
        ("修正版OCR系统", 8019),
        ("最终版OCR系统", 8018),
        ("终极OCR系统", 8017),
        ("简化版OpenCV系统", 8026),
        ("OpenCV OCR系统", 8025),
        ("终极改进版OCR系统", 8024),
        ("改进版OCR系统", 8023)
    ]

    # 检查所有系统状态
    print_safe("\n检查系统状态:")
    print_safe("-" * 40)

    healthy_systems = []
    for system_name, port in systems:
        if check_system_health(system_name, port):
            healthy_systems.append((system_name, port))

    # 测试健康系统
    print_safe(f"\n开始测试 {len(healthy_systems)} 个健康系统")

    total_success = 0
    total_tests = 0

    for system_name, port in healthy_systems:
        success, tests = test_system(system_name, port, test_images)
        total_success += success
        total_tests += tests

    # 总结
    print_safe(f"\n{'='*80}")
    print_safe("测试总结")
    print_safe(f"{'='*80}")
    print_safe(f"总测试数: {total_tests}")
    print_safe(f"成功识别: {total_success}")
    print_safe(f"成功率: {(total_success/total_tests*100):.1f}%" if total_tests > 0 else "成功率: 0%")
    print_safe(f"健康系统: {len(healthy_systems)}/{len(systems)}")

    if len(healthy_systems) == 0:
        print_safe("\n没有健康的服务系统，请检查服务器是否启动")
    else:
        print_safe(f"\n最佳系统推荐: {healthy_systems[0][0]} (端口 {healthy_systems[0][1]})")

if __name__ == "__main__":
    main()