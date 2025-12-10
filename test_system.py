#!/usr/bin/env python3
"""
车牌识别系统测试脚本
"""

import sys
import os
from pathlib import Path

def test_dependencies():
    """测试依赖包"""
    print("正在检查依赖包...")

    packages = ['torch', 'torchvision', 'fastapi', 'uvicorn', 'pillow', 'numpy', 'opencv-python']
    missing = []

    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package}")
            missing.append(package)

    if missing:
        print(f"\n缺少依赖包: {missing}")
        print("请运行: pip install " + " ".join(missing))
        return False

    print("所有依赖包已安装")
    return True

def test_model():
    """测试模型文件"""
    print("\n正在检查模型文件...")

    model_path = "best_fast_high_accuracy_model.pth"
    if Path(model_path).exists():
        print(f"[OK] 模型文件存在: {model_path}")
        return True
    else:
        print(f"[FAIL] 模型文件不存在: {model_path}")
        return False

def test_api():
    """测试API"""
    print("\n正在测试API...")

    try:
        from plate_recognition_api import load_model, PlateRecognitionModel
        print("[OK] API模块导入成功")

        # 测试模型加载
        model = PlateRecognitionModel()
        print("[OK] 模型创建成功")

        return True
    except Exception as e:
        print(f"[FAIL] API测试失败: {e}")
        return False

def test_static_files():
    """测试静态文件"""
    print("\n正在检查静态文件...")

    static_dir = Path("static")
    if not static_dir.exists():
        print("[FAIL] static目录不存在")
        return False

    index_file = static_dir / "index.html"
    if index_file.exists():
        print("[OK] Web界面文件存在")
        return True
    else:
        print("[FAIL] Web界面文件不存在")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("车牌识别系统测试")
    print("=" * 50)

    tests = [
        ("依赖包检查", test_dependencies),
        ("模型文件检查", test_model),
        ("API模块测试", test_api),
        ("静态文件检查", test_static_files)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[{name}]")
        if test_func():
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")

    if failed == 0:
        print("[OK] 所有测试通过，系统可以正常运行")
        print("\n启动方式:")
        print("1. 运行: python start_server.py")
        print("2. 访问: http://localhost:8000")
    else:
        print("[FAIL] 部分测试失败，请检查问题后再试")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)