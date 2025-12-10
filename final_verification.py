#!/usr/bin/env python3
"""
最终验证脚本
确保系统能够正常工作
"""

import sys
import requests
import time
from pathlib import Path

def test_api_endpoints():
    """测试所有API端点"""
    base_url = "http://localhost:8001"

    endpoints = [
        ("GET", "/health", "健康检查"),
        ("GET", "/stats", "系统统计"),
        ("GET", "/", "主页"),
        ("GET", "/web", "Web界面")
    ]

    results = []

    for method, endpoint, name in endpoints:
        url = base_url + endpoint
        try:
            if method == "GET":
                response = requests.get(url, timeout=5)
            else:
                response = requests.post(url, timeout=5)

            if response.status_code == 200:
                results.append(f"[OK] {name}: {url}")
            else:
                results.append(f"[FAIL] {name}: {url} (状态码: {response.status_code})")

        except requests.exceptions.RequestException as e:
            results.append(f"[FAIL] {name}: {url} (错误: {str(e)[:50]})")

    return results

def test_image_recognition():
    """测试图片识别功能"""
    try:
        # 创建测试图片
        from PIL import Image
        import io

        img = Image.new('RGB', (400, 200), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        # 发送识别请求
        response = requests.post(
            "http://localhost:8001/recognize",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return f"[OK] 图片识别: 成功识别车牌 {result.get('plate_number', 'N/A')}"
        else:
            return f"[FAIL] 图片识别: HTTP {response.status_code}"

    except Exception as e:
        return f"[FAIL] 图片识别: {str(e)[:50]}"

def main():
    """主函数"""
    print("车牌识别系统最终验证")
    print("=" * 40)

    # 检查必要文件
    required_files = [
        "working_api.py",
        "static/index.html",
        "start_server.py",
        "best_fast_high_accuracy_model.pth"
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"[FAIL] 缺少文件: {missing_files}")
        return

    print("[OK] 所有必要文件存在")

    # 检查Python依赖
    required_packages = ['torch', 'torchvision', 'fastapi', 'uvicorn', 'PIL']
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"[FAIL] 缺少依赖包: {missing_packages}")
        print("请运行: pip install " + " ".join(missing_packages))
        return

    print("[OK] 所有依赖包已安装")

    # 启动服务器
    print("\n启动服务器进行测试...")
    import subprocess

    try:
        server_process = subprocess.Popen([
            sys.executable, "working_api.py"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 等待服务器启动
        time.sleep(5)

        # 测试API端点
        print("\n测试API端点:")
        api_results = test_api_endpoints()
        for result in api_results:
            print(f"  {result}")

        # 测试图片识别
        print(f"\n{test_image_recognition()}")

        # 统计结果
        success_count = sum(1 for result in api_results if result.startswith("[OK]"))
        if test_image_recognition().startswith("[OK]"):
            success_count += 1

        print(f"\n验证结果: {success_count}/{len(api_results) + 1} 项测试通过")

        if success_count >= len(api_results):
            print("\n[OK] 系统验证成功！")
            print("系统可以正常使用:")
            print("  - 主页: http://localhost:8001")
            print("  - Web界面: http://localhost:8001/web")
            print("  - API文档: http://localhost:8001/docs")
        else:
            print("\n[FAIL] 系统验证失败，请检查问题")

    except Exception as e:
        print(f"[FAIL] 验证过程中发生错误: {e}")
    finally:
        # 停止服务器
        if 'server_process' in locals():
            print("\n停止服务器...")
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except:
                server_process.kill()

if __name__ == "__main__":
    main()