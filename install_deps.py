#!/usr/bin/env python3
"""
依赖包安装脚本
"""

import subprocess
import sys
import os

def install_package(package):
    """安装单个包"""
    try:
        print(f"正在安装 {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print(f"[OK] {package} 安装成功")
            return True
        else:
            print(f"[FAIL] {package} 安装失败")
            print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"[FAIL] 安装 {package} 时发生错误: {e}")
        return False

def main():
    """主函数"""
    print("车牌识别系统依赖包安装")
    print("=" * 50)

    # 需要安装的包
    packages = [
        "fastapi",
        "pillow",
        "opencv-python",
        "pydantic<2.0"  # 修复 pydantic 兼容性问题
    ]

    print("开始安装依赖包...")

    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1

    print("\n" + "=" * 50)
    print(f"安装结果: {success_count}/{len(packages)} 成功")

    if success_count == len(packages):
        print("[OK] 所有依赖包安装成功")
        print("\n现在可以运行以下命令测试系统:")
        print("python test_system.py")
    else:
        print("[FAIL] 部分依赖包安装失败")
        print("请手动安装失败的包，或检查网络连接")

    return success_count == len(packages)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)