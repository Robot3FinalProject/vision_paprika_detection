import sys
import os

sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

import cv2
import time
import numpy as np
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LatestHarvestVisionNode(Node):
    def __init__(self):
        super().__init__('latest_harvest_vision_node')
        
        # [ROS 2 Publisher] 가장 최근 투입된 객체 상태 전송
        self.publisher_ = self.create_publisher(String, '/vision/latest_status', 10)

        # 1. 설정 및 모델 경로 스펙
        self.MODEL_PATH = "/home/heh/pj_detection_paprika/models/v3.x/yolo11_seg_2class_synth_and_simul_and_mokup_v3.0/weights/best.pt"
        self.CAM_INDEX = 4  
        self.CONF_THRESHOLD = 0.65  

        self.CLASS_NAMES = {0: 'Red', 1: 'Yellow'}
        self.CLASS_COLORS = {
            'Red': (0, 0, 255),      
            'Yellow': (0, 215, 255)  
        }
        
        self.OUTPUT_DIR = "/home/heh/pj_detection_paprika/result/latest_object_inspection"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.LOG_FILE_PATH = os.path.join(self.OUTPUT_DIR, "latest_inspection_log.txt")

        self.get_logger().info(f"📦 Loading YOLO11-seg model from: {self.MODEL_PATH}")
        self.model = YOLO(self.MODEL_PATH)

        self.get_logger().info(f"🎥 Connecting to Logitech Webcam (Index: {self.CAM_INDEX})...")
        self.cap = cv2.VideoCapture(self.CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Error: Could not open webcam at index {self.CAM_INDEX}.")
            exit()

        # 최신 객체 상태 관리 변수
        self.latest_detected_color = "None"
        self.latest_conf = 0.0
        self.latest_box = None
        self.latest_mask = None

        print("\n🚀 Starting [Latest Object] Harvest Box Inspection...")
        print("---------------------------------------------------------")
        print("[Controls]")
        print("  - Press 's' : Save current snapshot and log")
        print("  - Press 'r' : Reset latest status")
        print("  - Press 'q' : Quit program")
        print("---------------------------------------------------------\n")

        # 메인 비전 루프 타이머 (10fps -> 0.1초)
        self.timer = self.create_timer(0.1, self.vision_loop_callback)

    def vision_loop_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("❌ Error: Failed to grab frame from webcam.")
            return

        results = self.model.predict(frame, conf=self.CONF_THRESHOLD, verbose=False)[0]
        
        current_detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            masks_xy = results.masks.xy if results.masks is not None else [None] * len(boxes)

            for box, cls_id, conf, mask in zip(boxes, classes, confs, masks_xy):
                class_name = self.CLASS_NAMES.get(int(cls_id), None)
                if class_name not in ['Red', 'Yellow']:
                    continue

                x1, y1, x2, y2 = map(int, box)
                current_detections.append({
                    "color": class_name,
                    "conf": float(conf),
                    "box": (x1, y1, x2, y2),
                    "mask": mask,
                    # 화면 상단(y1이 작은 값)에 가까울수록 최근에 들어온 객체이거나, 면적이 큰 객체 우선순위 부여 가능
                    "y1": y1 
                })

        # 가장 최근에 투입되었거나 확실하게 감지된 주요 객체 선정 (여기서는 탐지된 것 중 가장 높은 신뢰도 또는 가장 최근 객체 선택)
        if current_detections:
            # 신뢰도가 가장 높거나 화면 상단에 진입한 객체를 최신 객체로 지정
            best_det = max(current_detections, key=lambda x: x["conf"])
            
            self.latest_detected_color = best_det["color"]
            self.latest_conf = best_det["conf"]
            self.latest_box = best_det["box"]
            self.latest_mask = best_det["mask"]
        else:
            # 감지된 객체가 없으면 초기화 (필요에 따라 유지할 수도 있음)
            pass

        # [ROS 2 퍼블리시] 강화학습 파트에 최신 객체 색상 정보 전송
        status_msg = String()
        status_msg.data = f"Latest_Color: {self.latest_detected_color} | Conf: {self.latest_conf:.2f}"
        self.publisher_.publish(status_msg)

        # 시각화 오버레이 (세그멘테이션 마스크)
        overlay = frame.copy()
        if self.latest_mask is not None and self.latest_detected_color in self.CLASS_COLORS:
            bgr_color = self.CLASS_COLORS[self.latest_detected_color]
            pts = self.latest_mask.astype(np.int32)
            cv2.fillPoly(overlay, [pts], bgr_color)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 바운딩 박스 및 텍스트 표기
        if self.latest_box is not None:
            x1, y1, x2, y2 = self.latest_box
            bgr_color = self.CLASS_COLORS.get(self.latest_detected_color, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 3)
            
            label = f"LATEST: {self.latest_detected_color} ({self.latest_conf:.2f})"
            cv2.putText(frame, label, (x1, max(35, y1 - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_color, 2)

        # 상단 통계 정보 패널
        stats_bg = frame.copy()
        cv2.rectangle(stats_bg, (15, 15), (650, 95), (30, 30, 30), -1)
        cv2.addWeighted(stats_bg, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "[RL Target Inspection: Latest Object Only]", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Latest Inserted Color : {self.latest_detected_color}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Harvest Box Latest Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_filename = f"latest_snapshot_{timestamp}.jpg"
            img_save_path = os.path.join(self.OUTPUT_DIR, img_filename)
            cv2.imwrite(img_save_path, frame)
            print(f"\n📸 Snapshot saved: {img_save_path}\n")

        elif key == ord('r'):
            self.latest_detected_color = "None"
            self.latest_conf = 0.0
            self.latest_box = None
            self.latest_mask = None
            print("\n🔄 Reset latest status.\n")

        elif key == ord('q'):
            print("🚪 Exiting program...")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LatestHarvestVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("✅ Program terminated safely.")


if __name__ == '__main__':
    main()