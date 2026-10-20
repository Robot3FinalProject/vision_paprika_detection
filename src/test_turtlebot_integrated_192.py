#!/usr/bin/env python3
"""
File Name: test_turtlebot_integrated_192.py
Description: 터틀봇 192x192 NCNN FP16 통합 벤치마크 (Ctrl+C: 강제 종료 / q키: 정상 종료 및 MP4·CSV 저장)
...
"""

import os
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import select
import termios
import tty
import time
import csv
import cv2
import numpy as np
import threading
import queue
from datetime import datetime
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)

# ==========================================
# 1. 설정 및 경로 지정
# ==========================================
VERSION_TAG = "v3.0_Integrated_192_NCNN"
MODEL_PATH = "/home/ubuntu/models/v3.x/yolo11_seg_2class_synth_and_simul_and_mokup_v3.0/weights/best_192_ncnn_model"
SAVE_DIR = "/home/ubuntu/result/v3.x/phase1_integrated_test_speedup"
CONF_THRESHOLD = 0.25
IMGSZ = 192

os.makedirs(SAVE_DIR, exist_ok=True)
CSV_FILE = os.path.join(SAVE_DIR, f"realtime_metrics_{VERSION_TAG}.csv")
VIDEO_SAVE_PATH = os.path.join(SAVE_DIR, f"realtime_test_{VERSION_TAG}.mp4")

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Frame_ID", "FPS", "Latency_ms", "Det_Count", "Avg_Conf"])

print(f"[{VERSION_TAG}] Loading NCNN YOLO Model (imgsz={IMGSZ})...")
model = YOLO(MODEL_PATH)

# C270 웹캠 /dev/video1에 맞추어 인덱스 1 오픈
print("[INFO] Opening camera /dev/video1 (Index 1)...")
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MP4 인코더 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(VIDEO_SAVE_PATH, fourcc, 15.0, (640, 480))

# 웹 송출 전용 큐 버퍼
web_frame_queue = queue.Queue(maxsize=2)
running = True
frame_id = 0
cleanup_lock = threading.Lock()

def cleanup_and_exit(reason="User Quit ('q' pressed)"):
    global running, video_writer, cap
    with cleanup_lock:
        if not running:
            return
        running = False
        print(f"\n==========================================")
        print(f"[{VERSION_TAG}] Normal Shutdown ({reason})...")
        print("Saving MP4 Video and CSV Metrics Safely...")
        time.sleep(0.5)
        
        if video_writer is not None:
            video_writer.release()
            print(f"[SUCCESS] MP4 Saved: {VIDEO_SAVE_PATH}")
        if cap is not None and cap.isOpened():
            cap.release()
            print("[SUCCESS] Camera Released.")
        print(f"[SUCCESS] CSV Saved: {CSV_FILE}")
        print("==========================================\n")

# ==========================================
# 2. 통합 추론 및 OSD 시각화 메인 루프
# ==========================================
def inference_worker():
    global frame_id, running
    
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
            
        frame_id += 1
        t_start = time.perf_counter()
        
        # 192x192 NCNN 추론
        results = model.predict(
                    source=frame,
                    imgsz=IMGSZ,
                    conf=CONF_THRESHOLD,
                    device="cpu",
                    verbose=False
                )[0]
        
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0.0
        
        boxes = results.boxes
        det_count = len(boxes) if boxes is not None else 0
        avg_conf = float(boxes.conf.mean().cpu().numpy()) if det_count > 0 else 0.0
        
        # 세그멘테이션 마스크 및 바운딩 박스 시각화
        res_frame = results.plot()

        # OSD 표기
        osd_text_1 = f"NCNN 192x192 FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms"
        osd_text_2 = f"Conf: {CONF_THRESHOLD:.2f} | Dets: {det_count} | Avg Conf: {avg_conf:.2%}"
        
        cv2.rectangle(res_frame, (5, 5), (635, 65), (0, 0, 0), -1)
        cv2.putText(res_frame, osd_text_1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(res_frame, osd_text_2, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # CSV 실시간 기록
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, frame_id, f"{fps:.2f}", f"{latency_ms:.2f}", det_count, f"{avg_conf:.4f}"])

        # MP4 프레임 저장
        if video_writer is not None:
            video_writer.write(res_frame)

        # 웹 송출용 큐 업데이트
        if not web_frame_queue.full():
            web_frame_queue.put(res_frame)

def keyboard_listener_thread():
    global running
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 'q':
                    cleanup_and_exit(reason="Key 'q' pressed")
                    os._exit(0)
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

# ==========================================
# 3. 비동기 웹 스트리밍 루프 (Flask)
# ==========================================
def generate_frames():
    while running:
        try:
            display_frame = web_frame_queue.get(timeout=0.1)
            ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except queue.Empty:
            continue

@app.route('/')
@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=inference_worker, daemon=True).start()
    threading.Thread(target=keyboard_listener_thread, daemon=True).start()
    
    print(f"[{VERSION_TAG}] Integrated Real-time Streaming & Inference Running...")
    print(f"-> Access Web Browser: http://<터틀봇_IP>:5000/video")
    print(f"-> Press 'q' key to QUIT NORMALLY and SAVE files.")
    print(f"-> Press 'Ctrl+C' to FORCE QUIT immediately (without saving).")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        # Ctrl+C 누르면 저장 없이 즉시 강제 종료
        print("\n[FORCE QUIT] Ctrl+C pressed. Exiting immediately without saving.")
        os._exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")