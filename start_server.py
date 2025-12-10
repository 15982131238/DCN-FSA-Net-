#!/usr/bin/env python3
"""
车牌识别系统启动脚本
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'torch', 'torchvision', 'fastapi', 'uvicorn',
        'pillow', 'numpy', 'opencv-python'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def check_model():
    """检查模型文件"""
    model_path = "best_fast_high_accuracy_model.pth"
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保模型文件在当前目录中")
        return False
    return True

def check_static_files():
    """检查静态文件"""
    static_dir = Path("static")
    index_file = static_dir / "index.html"

    if not static_dir.exists():
        print("创建static目录...")
        static_dir.mkdir(exist_ok=True)

    if not index_file.exists():
        print("❌ Web界面文件不存在: static/index.html")
        return False

    return True

def start_server():
    """启动服务器"""
    print("🚀 启动车牌识别系统...")

    # 启动FastAPI服务器
    cmd = [sys.executable, "working_api.py"]
    process = subprocess.Popen(cmd)

    # 等待服务器启动
    time.sleep(3)

    # 打开浏览器
    url = "http://localhost:8001"
    print(f"打开浏览器: {url}")
    webbrowser.open(url)

    print("服务器启动成功!")
    print("API文档: http://localhost:8001/docs")
    print("Web界面: http://localhost:8001/web")
    print("按 Ctrl+C 停止服务器")

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n🛑 正在停止服务器...")
        process.terminate()
        process.wait()
        print("✅ 服务器已停止")

def main():
    """主函数"""
    print("🚗 车牌识别系统启动检查...")

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 检查模型
    if not check_model():
        sys.exit(1)

    # 检查静态文件
    if not check_static_files():
        print("Web界面文件缺失，但API服务仍可正常运行")

    # 启动服务器
    start_server()

if __name__ == "__main__":
    main()