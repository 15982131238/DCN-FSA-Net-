#!/usr/bin/env python3
"""
视频处理模块
支持实时视频流和文件视频处理
"""

import cv2
import numpy as np
import torch
import time
from PIL import Image
from typing import List, Dict, Any, Optional, Callable
import threading
import queue
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器"""

    def __init__(self, model, device, recognition_callback: Optional[Callable] = None):
        self.model = model
        self.device = device
        self.recognition_callback = recognition_callback
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.capture_thread = None

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """处理单帧图像"""
        try:
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 转换为PIL图像
            image = Image.fromarray(frame_rgb)

            # 调用识别函数
            from plate_recognition_api import recognize_plate
            result = recognize_plate(image)

            return result

        except Exception as e:
            logger.error(f"帧处理失败: {e}")
            return {
                "plate_number": "处理失败",
                "plate_type": "未知",
                "confidence": 0.0,
                "processing_time": 0.0,
                "error": str(e)
            }

    def draw_result(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """在帧上绘制识别结果"""
        try:
            # 复制帧
            output_frame = frame.copy()

            # 获取帧尺寸
            height, width = frame.shape[:2]

            # 绘制半透明背景
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (0, height-100), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)

            # 绘制识别结果
            plate_text = f"车牌: {result.get('plate_number', 'N/A')}"
            type_text = f"类型: {result.get('plate_type', 'N/A')}"
            confidence_text = f"置信度: {result.get('confidence', 0)*100:.1f}%"

            # 设置字体
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2

            # 计算文本位置
            y_offset = height - 70

            # 绘制文本
            cv2.putText(output_frame, plate_text, (10, y_offset),
                       font, font_scale, (255, 255, 255), font_thickness)
            cv2.putText(output_frame, type_text, (10, y_offset + 25),
                       font, font_scale, (255, 255, 255), font_thickness)
            cv2.putText(output_frame, confidence_text, (10, y_offset + 50),
                       font, font_scale, (255, 255, 255), font_thickness)

            # 绘制置信度条
            if 'confidence' in result:
                confidence = result['confidence']
                bar_width = int((width - 20) * confidence)
                cv2.rectangle(output_frame, (10, y_offset + 65),
                             (10 + bar_width, y_offset + 75), (0, 255, 0), -1)
                cv2.rectangle(output_frame, (10, y_offset + 65),
                             (width - 10, y_offset + 75), (255, 255, 255), 2)

            return output_frame

        except Exception as e:
            logger.error(f"绘制结果失败: {e}")
            return frame

    def capture_frames(self, source, fps_limit=10):
        """捕获帧线程"""
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error("无法打开视频源")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"视频源信息: {width}x{height}, {fps:.2f}fps")

        # 计算帧间隔
        frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0
        last_time = time.time()

        while self.is_running:
            ret, frame = cap.read()

            if not ret:
                logger.info("视频结束或读取失败")
                break

            # 控制帧率
            current_time = time.time()
            if current_time - last_time < frame_interval:
                time.sleep(0.001)
                continue

            last_time = current_time

            # 将帧放入队列
            try:
                self.frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                # 队列满，丢弃最旧的帧
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Empty:
                    pass

        cap.release()
        logger.info("帧捕获线程结束")

    def process_frames(self):
        """处理帧线程"""
        while self.is_running:
            try:
                # 从队列获取帧
                frame = self.frame_queue.get(timeout=1.0)

                # 处理帧
                result = self.process_frame(frame)

                # 绘制结果
                output_frame = self.draw_result(frame, result)

                # 将结果放入队列
                self.result_queue.put((output_frame, result), timeout=0.1)

                # 调用回调函数
                if self.recognition_callback:
                    self.recognition_callback(result)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"帧处理错误: {e}")

        logger.info("帧处理线程结束")

    def start(self, source, fps_limit=10):
        """启动视频处理"""
        if self.is_running:
            logger.warning("视频处理器已在运行")
            return False

        self.is_running = True

        # 启动捕获线程
        self.capture_thread = threading.Thread(
            target=self.capture_frames,
            args=(source, fps_limit)
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        logger.info("视频处理器启动成功")
        return True

    def stop(self):
        """停止视频处理"""
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)

        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("视频处理器已停止")

    def get_result(self):
        """获取处理结果"""
        try:
            return self.result_queue.get(timeout=1.0)
        except queue.Empty:
            return None, None

class VideoStreamer:
    """视频流处理器"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.processor = None
        self.window_name = "车牌识别实时视频"

    def start_camera(self, camera_id=0, fps_limit=10):
        """启动摄像头"""
        self.processor = VideoProcessor(
            self.model,
            self.device,
            self.on_recognition_result
        )

        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        # 启动处理器
        if self.processor.start(camera_id, fps_limit):
            print("📹 摄像头启动成功")
            print("按 ESC 键退出")

            # 显示循环
            while True:
                frame, result = self.processor.get_result()

                if frame is not None:
                    cv2.imshow(self.window_name, frame)

                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break

            # 停止处理器
            self.processor.stop()
            cv2.destroyAllWindows()
            print("🛑 摄像头已停止")
        else:
            print("❌ 摄像头启动失败")

    def process_video_file(self, video_path, output_path=None, fps_limit=10):
        """处理视频文件"""
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            return False

        self.processor = VideoProcessor(
            self.model,
            self.device,
            self.on_recognition_result
        )

        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)

        # 设置视频写入器
        video_writer = None
        if output_path:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 启动处理器
        if self.processor.start(video_path, fps_limit):
            print(f"📹 开始处理视频: {video_path}")
            print("按 ESC 键退出")

            # 显示循环
            while True:
                frame, result = self.processor.get_result()

                if frame is not None:
                    cv2.imshow(self.window_name, frame)

                    # 写入输出文件
                    if video_writer:
                        video_writer.write(frame)

                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键
                    break

                # 检查是否处理完成
                if not self.processor.is_running and self.processor.frame_queue.empty():
                    break

            # 停止处理器
            self.processor.stop()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            print("🛑 视频处理完成")

            if output_path:
                print(f"📁 输出文件: {output_path}")

            return True
        else:
            print("❌ 视频处理失败")
            return False

    def on_recognition_result(self, result):
        """识别结果回调"""
        if result.get('plate_number') != '处理失败':
            print(f"🚗 识别结果: {result.get('plate_number')} "
                  f"({result.get('plate_type')}) "
                  f"置信度: {result.get('confidence', 0)*100:.1f}%")

def create_video_demo():
    """创建视频演示脚本"""
    script_content = '''#!/usr/bin/env python3
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
'''

    with open("video_demo.py", "w", encoding="utf-8") as f:
        f.write(script_content)

    print("视频演示脚本已创建: video_demo.py")

if __name__ == "__main__":
    # 创建视频演示脚本
    create_video_demo()