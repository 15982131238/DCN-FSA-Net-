#!/usr/bin/env python3
"""
快速测试车牌识别系统
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def test_api():
    """测试API"""
    print("正在测试API服务...")

    try:
        # 测试健康检查
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"健康检查: {health}")

            if health.get("model_loaded"):
                print("✓ 模型已加载")
            else:
                print("✗ 模型未加载")
        else:
            print(f"✗ 健康检查失败: {response.status_code}")
            return False

        # 测试统计信息
        response = requests.get("http://localhost:8000/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"系统统计: {stats}")
        else:
            print(f"✗ 统计信息获取失败: {response.status_code}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"✗ API测试失败: {e}")
        return False

def test_image_recognition():
    """测试图片识别"""
    print("正在测试图片识别...")

    # 创建测试图片
    from PIL import Image
    import io

    # 创建一个简单的测试图片
    img = Image.new('RGB', (400, 200), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    try:
        # 发送识别请求
        response = requests.post(
            "http://localhost:8000/recognize",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print(f"识别结果: {result}")
            return True
        else:
            print(f"✗ 识别失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"✗ 图片识别测试失败: {e}")
        return False

def main():
    """主函数"""
    print("车牌识别系统快速测试")
    print("=" * 40)

    # 检查必要文件
    required_files = ["working_api.py", "static/index.html"]
    for file in required_files:
        if not Path(file).exists():
            print(f"✗ 缺少文件: {file}")
            return False

    print("文件检查通过")

    # 启动服务器
    print("正在启动服务器...")
    try:
        server_process = subprocess.Popen([
            sys.executable, "working_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 等待服务器启动
        print("等待服务器启动...")
        time.sleep(5)

        # 测试API
        if test_api():
            print("✓ API测试通过")

            # 测试图片识别
            if test_image_recognition():
                print("✓ 图片识别测试通过")
                print("\n🎉 系统测试成功！")
                print("访问地址:")
                print("  - 主页: http://localhost:8000")
                print("  - Web界面: http://localhost:8000/web")
                print("  - API文档: http://localhost:8000/docs")
            else:
                print("✗ 图片识别测试失败")

        else:
            print("✗ API测试失败")

    except KeyboardInterrupt:
        print("\n测试中断")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        # 停止服务器
        if 'server_process' in locals():
            print("正在停止服务器...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

if __name__ == "__main__":
    main()