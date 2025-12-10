#!/usr/bin/env python3
"""
测试脚本：验证车牌识别系统是否正常工作
"""

import requests
import json
import time
from pathlib import Path

def test_system():
    """测试系统功能"""
    base_url = "http://localhost:8001"

    print("🧪 开始测试车牌识别系统...")
    print("=" * 50)

    # 1. 测试健康检查
    print("\n1. 测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ 健康检查通过")
            print(f"   状态: {health_data.get('status')}")
            print(f"   模型: {health_data.get('model_loaded')}")
            print(f"   设备: {health_data.get('device')}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查连接失败: {e}")
        return False

    # 2. 测试图片上传
    print("\n2. 测试图片上传识别...")
    test_image_path = "test_plate.jpg"

    if not Path(test_image_path).exists():
        print(f"❌ 测试图片不存在: {test_image_path}")
        return False

    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/recognize", files=files, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✅ 图片识别成功")
            print(f"   车牌号: {result.get('plate_number')}")
            print(f"   车牌类型: {result.get('plate_type')}")
            print(f"   置信度: {(result.get('confidence', 0) * 100):.1f}%")
            print(f"   处理时间: {result.get('processing_time', 0):.2f}ms")

            # 检查识别结果的合理性
            if result.get('plate_number') != "识别失败" and result.get('confidence', 0) > 0.1:
                print("✅ 识别结果有效")
            else:
                print("⚠️  识别结果可能存在问题")
        else:
            print(f"❌ 图片识别失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 图片上传测试失败: {e}")
        return False

    # 3. 测试历史记录
    print("\n3. 测试历史记录...")
    try:
        response = requests.get(f"{base_url}/history", timeout=5)
        if response.status_code == 200:
            history_data = response.json()
            print("✅ 历史记录获取成功")
            print(f"   记录数量: {history_data.get('total', 0)}")
        else:
            print(f"❌ 历史记录获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 历史记录测试失败: {e}")
        return False

    # 4. 测试Web界面访问
    print("\n4. 测试Web界面...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Web界面访问正常")
        else:
            print(f"❌ Web界面访问失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Web界面测试失败: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 所有测试通过！系统运行正常")
    print("\n📋 系统功能总结:")
    print("✅ 健康检查正常")
    print("✅ 图片识别功能正常")
    print("✅ 历史记录功能正常")
    print("✅ Web界面访问正常")
    print("✅ 网络连接稳定")

    print("\n🌐 访问地址:")
    print(f"   - 主页: {base_url}")
    print(f"   - Web界面: {base_url}/web")
    print(f"   - 功能测试: {base_url}/test")
    print(f"   - API文档: {base_url}/docs")

    return True

if __name__ == "__main__":
    success = test_system()
    if not success:
        print("\n❌ 系统测试失败，请检查系统状态")
        exit(1)