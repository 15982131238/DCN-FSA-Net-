#!/usr/bin/env python3
"""
车牌识别系统演示脚本
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

def show_banner():
    """显示系统横幅"""
    banner = """
    ================================================================
     __     ______  _    _   __      _ _______ _______
     \ \   / / __ \| |  | |  \ \    / |__   __|__   __|
      \ \_/ / |  | | |  | |   \ \  / /   | |     | |
       \   /| |  | | |  | |    \ \/ /    | |     | |
        | | | |__| | |__| |     \  /     | |     | |
        |_|  \____/ \____/       \/      |_|     |_|
    ================================================================
                基于深度学习的中国车牌识别系统
    ================================================================
    """
    print(banner)

def check_system():
    """检查系统状态"""
    print("正在检查系统状态...")

    # 检查模型文件
    model_path = "best_fast_high_accuracy_model.pth"
    if not Path(model_path).exists():
        print(f"错误: 模型文件 {model_path} 不存在")
        return False

    # 检查关键文件
    required_files = [
        "plate_recognition_api.py",
        "static/index.html",
        "video_processor.py"
    ]

    for file in required_files:
        if not Path(file).exists():
            print(f"错误: 文件 {file} 不存在")
            return False

    print("系统文件检查完成")
    return True

def run_test():
    """运行系统测试"""
    print("正在运行系统测试...")

    try:
        result = subprocess.run([sys.executable, "test_system.py"],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("系统测试通过")
            return True
        else:
            print("系统测试失败")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"测试运行失败: {e}")
        return False

def start_system():
    """启动系统"""
    print("正在启动车牌识别系统...")

    # 启动服务器
    try:
        process = subprocess.Popen([sys.executable, "start_server.py"])

        # 等待服务器启动
        print("等待服务器启动...")
        time.sleep(5)

        # 打开浏览器
        url = "http://localhost:8000"
        print(f"正在打开浏览器: {url}")
        webbrowser.open(url)

        print("系统启动成功!")
        print("访问地址:")
        print("  - 主页: http://localhost:8000")
        print("  - Web界面: http://localhost:8000/web")
        print("  - API文档: http://localhost:8000/docs")
        print("\n按 Ctrl+C 停止服务器")

        # 等待用户中断
        process.wait()

    except KeyboardInterrupt:
        print("\n正在停止系统...")
        process.terminate()
        process.wait()
        print("系统已停止")
    except Exception as e:
        print(f"启动失败: {e}")

def show_menu():
    """显示菜单"""
    print("\n请选择操作:")
    print("1. 系统测试")
    print("2. 启动Web服务")
    print("3. 视频处理演示")
    print("4. 安装依赖包")
    print("5. 查看使用说明")
    print("0. 退出")

    choice = input("请输入选项 (0-5): ").strip()
    return choice

def main():
    """主函数"""
    show_banner()

    if not check_system():
        print("系统检查失败，请确保所有文件都存在")
        return

    while True:
        choice = show_menu()

        if choice == '0':
            print("感谢使用车牌识别系统!")
            break

        elif choice == '1':
            run_test()

        elif choice == '2':
            start_system()

        elif choice == '3':
            if Path("video_demo.py").exists():
                subprocess.run([sys.executable, "video_demo.py"])
            else:
                print("视频演示文件不存在")

        elif choice == '4':
            subprocess.run([sys.executable, "install_deps.py"])

        elif choice == '5':
            if Path("使用说明.md").exists():
                print("使用说明:")
                with open("使用说明.md", "r", encoding="utf-8") as f:
                    print(f.read()[:1000] + "...")
            else:
                print("使用说明文件不存在")

        else:
            print("无效选项，请重新选择")

if __name__ == "__main__":
    main()