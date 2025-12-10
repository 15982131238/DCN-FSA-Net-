#!/usr/bin/env python3
"""
简单系统测试
"""

import subprocess
import time
import sys
from pathlib import Path

def main():
    """主函数"""
    print("车牌识别系统测试")
    print("=" * 30)

    # 检查文件
    required_files = ["working_api.py", "static/index.html"]
    missing_files = []

    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"缺少文件: {missing_files}")
        return

    print("文件检查通过")

    # 启动服务器
    print("正在启动服务器...")
    try:
        server_process = subprocess.Popen([
            sys.executable, "working_api.py"
        ])

        # 等待服务器启动
        print("等待服务器启动...")
        time.sleep(5)

        # 检查服务是否运行
        try:
            import requests
            response = requests.get("http://localhost:8001/health", timeout=3)
            if response.status_code == 200:
                health = response.json()
                print(f"健康检查: {health}")
                print("API服务运行正常")
                print("访问地址:")
                print("  - 主页: http://localhost:8001")
                print("  - Web界面: http://localhost:8001/web")
                print("  - API文档: http://localhost:8001/docs")
            else:
                print("API服务异常")
        except:
            print("无法连接到API服务")

    except Exception as e:
        print(f"启动失败: {e}")
    finally:
        # 停止服务器
        if 'server_process' in locals():
            print("正在停止服务器...")
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except:
                server_process.kill()

if __name__ == "__main__":
    main()