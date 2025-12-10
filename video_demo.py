#!/usr/bin/env python3
"""
视频处理演示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_processor import VideoStreamer
from plate_recognition_api import load_model, model

def main():
    # 加载模型
    if not load_model():
        print("❌ 模型加载失败")
        return

    # 创建视频流处理器
    streamer = VideoStreamer(model, model.device)

    print("🚗 车牌识别视频处理演示")
    print("1. 摄像头实时识别")
    print("2. 视频文件处理")
    print("3. 退出")

    while True:
        choice = input("请选择功能 (1-3): ").strip()

        if choice == '1':
            camera_id = input("请输入摄像头ID (默认0): ").strip()
            camera_id = int(camera_id) if camera_id.isdigit() else 0
            streamer.start_camera(camera_id)

        elif choice == '2':
            video_path = input("请输入视频文件路径: ").strip()
            if os.path.exists(video_path):
                output_path = input("请输入输出文件路径 (可选): ").strip()
                output_path = output_path if output_path else None
                streamer.process_video_file(video_path, output_path)
            else:
                print("❌ 文件不存在")

        elif choice == '3':
            print("👋 再见!")
            break

        else:
            print("❌ 无效选择")

if __name__ == "__main__":
    main()
