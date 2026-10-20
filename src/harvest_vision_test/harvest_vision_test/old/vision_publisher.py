import sys
import os

sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

import cv2
import time
import numpy as np
from ultralytics import YOLO

# [ROS 2 관련 임포트 추가]
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HarvestVisionNode(Node):
    def __init__(self):
        super().__init__('harvest_vision_node')
        
        # [ROS 2 Publisher 생성]
        # '/vision/box_status' 토픽으로 상자 상태 문자열을 발행합니다.
        self.publisher_ = self.create_publisher(String, '/vision/box_status', 10)

        # =========================================================
        # 1. 설정 및 모델 경로 스펙
        # =========================================================
        self.MODEL_PATH = "/home/heh/pj_detection_paprika/models/v3.x/yolo11_seg_2class_synth_and_simul_and_mokup_v3.0/weights/best.pt"
        self.CAM_INDEX = 4  

        # 임계값 설정
        # 1) 초근접 화각에서 쪼개짐 방지를 위한 신뢰도 유지
        self.CONF_THRESHOLD = 0.65                  # 카메라 : 상자 직착형
        # self.CONF_THRESHOLD = 0.55  

        # 2) 파프리카 지름(300~350px)을 완벽히 수용하기 위해 매칭 거리 확대 (260 -> 350)
        self.ACCUMULATED_MATCH_THRESHOLD = 350      # 카메라 : 상자 직착형
        # self.ACCUMULATED_MATCH_THRESHOLD = 130
        # self.ACCUMULATED_MATCH_THRESHOLD = 260
        
        # 3) 초근접 픽셀 흔들림 오차 보정을 위한 정지 허용 범위 확대 (45 -> 60)
        self.POSITION_TOLERANCE = 60                # 카메라 : 상자 직착형
        # self.POSITION_TOLERANCE = 25
        # self.POSITION_TOLERANCE = 45  
        
        # 4) 정착 완료 시간 
        self.SETTLE_TIME = 0.6                      # 카메라 : 상자 직착형
        # self.SETTLE_TIME = 0.5



        self.CLASS_NAMES = {0: 'Red', 1: 'Yellow'}
        self.CLASS_COLORS = {
            'Red': (0, 0, 255),      
            'Yellow': (0, 215, 255)  
        }
        
        self.OUTPUT_DIR = "/home/heh/pj_detection_paprika/result/box_inspection"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.LOG_FILE_PATH = os.path.join(self.OUTPUT_DIR, "harvest_box_inspection_log.txt")

        self.get_logger().info(f"📦 Loading YOLO11-seg model from: {self.MODEL_PATH}")
        self.model = YOLO(self.MODEL_PATH)

        self.get_logger().info(f"🎥 Connecting to Logitech Webcam (Index: {self.CAM_INDEX})...")
        self.cap = cv2.VideoCapture(self.CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Error: Could not open webcam at index {self.CAM_INDEX}.")
            exit()

        # 영구 누적 리스트 및 후보군 관리 딕셔너리
        self.accumulated_paprikas = []  # [{seq, color, center, box, mask, conf, visible}]
        self.candidates = {}            # 새로 진입해 굴러다니는 임시 후보군

        print("\n🚀 Starting Robust Harvest Box Inspection...")
        print("---------------------------------------------------------")
        print("[Controls]")
        print("  - Press 's' : Save current snapshot and log to file")
        print("  - Press 'r' : Reset accumulated box (Clear all)")
        print("  - Press 'q' : Quit program")
        print("---------------------------------------------------------\n")

        # [ROS 2 타이머 설정]
        # 주기적으로 메인 비전 루프가 돌 수 있도록 타이머 설정 (약 10fps 기준 0.1초)
        self.timer = self.create_timer(0.1, self.vision_loop_callback)

    def vision_loop_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("❌ Error: Failed to grab frame from webcam.")
            return

        results = self.model.predict(frame, conf=self.CONF_THRESHOLD, verbose=False)[0]
        
        current_frame_detections = []

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
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                box_width = x2 - x1
                box_height = y2 - y1
                print(f"[{class_name}] Paprika Size - Width: {box_width}px, Height: {box_height}px")

                current_frame_detections.append({
                    "color": class_name,
                    "conf": float(conf),
                    "box": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "mask": mask
                })

        current_time = time.time()

        # 1) 모든 누적 객체의 가시성 초기화
        for acc in self.accumulated_paprikas:
            acc["visible"] = False

        # 2) [최우선 처리] 현재 감지된 객체들을 순회하며 '이미 누적된 기존 객체'와 매칭
        unmatched_detections = []
        
        for det in current_frame_detections:
            det_cx, det_cy = det["center"]
            det_color = det["color"]

            matched_acc = None
            min_dist = float('inf')

            for acc in self.accumulated_paprikas:
                if acc["color"] == det_color:
                    acc_cx, acc_cy = acc["center"]
                    dist = np.sqrt((det_cx - acc_cx)**2 + (det_cy - acc_cy)**2)
                    if dist < self.ACCUMULATED_MATCH_THRESHOLD and dist < min_dist:
                        min_dist = dist
                        matched_acc = acc

            if matched_acc is not None:
                matched_acc["center"] = (det_cx, det_cy)
                matched_acc["box"] = det["box"]
                matched_acc["mask"] = det["mask"]
                matched_acc["conf"] = det["conf"]
                matched_acc["visible"] = True
            else:
                unmatched_detections.append(det)

        # 3) [후보군 처리] 누적 객체에 매칭되지 않은 남은 탐지 건들만 후보군(candidates)으로 진입
        for det in unmatched_detections:
            det_cx, det_cy = det["center"]
            det_color = det["color"]

            matched_cand_key = None
            min_cand_dist = float('inf')
            
            for cand_key, cand in self.candidates.items():
                if cand["color"] == det_color:
                    dist = np.sqrt((det_cx - cand["center"][0])**2 + (det_cy - cand["center"][1])**2)
                    if dist < 80 and dist < min_cand_dist:
                        min_cand_dist = dist
                        matched_cand_key = cand_key

            if matched_cand_key is not None:
                cand = self.candidates[matched_cand_key]
                dist_moved = np.sqrt((det_cx - cand["center"][0])**2 + (det_cy - cand["center"][1])**2)
                
                cand["center"] = (det_cx, det_cy)
                cand["box"] = det["box"]
                cand["mask"] = det["mask"]
                cand["conf"] = det["conf"]

                if dist_moved < self.POSITION_TOLERANCE:
                    if (current_time - cand["first_seen"]) >= self.SETTLE_TIME:
                        new_seq = len(self.accumulated_paprikas) + 1
                        self.accumulated_paprikas.append({
                            "seq": new_seq,
                            "color": cand["color"],
                            "center": cand["center"],
                            "box": cand["box"],
                            "mask": cand["mask"],
                            "conf": cand["conf"],
                            "visible": True
                        })
                        print(f"✨ Settled & Committed: #{new_seq} {cand['color']}")
                        del self.candidates[matched_cand_key]
                        continue
                else:
                    cand["first_seen"] = current_time
            else:
                cand_key = f"{det_color}_{int(det_cx)}_{int(det_cy)}_{int(current_time*1000)}"
                self.candidates[cand_key] = {
                    "color": det_color,
                    "center": (det_cx, det_cy),
                    "box": det["box"],
                    "mask": det["mask"],
                    "conf": det["conf"],
                    "first_seen": current_time
                }

        # 오랫동안 처리되지 않은 죽은 후보군 청소
        stale_keys = [k for k, v in self.candidates.items() if (current_time - v["first_seen"]) > 3.0]
        for k in stale_keys:
            del self.candidates[k]

        # 4) 정렬 및 통계 집계
        self.accumulated_paprikas = sorted(self.accumulated_paprikas, key=lambda x: x["seq"])
        total_count = len(self.accumulated_paprikas)
        red_count = sum(1 for p in self.accumulated_paprikas if p["color"] == 'Red')
        yellow_count = sum(1 for p in self.accumulated_paprikas if p["color"] == 'Yellow')

        # [ROS 2 퍼블리시 추가]
        # 비전 처리 결과(수량 및 순서)를 토픽 메시지로 만들어 로봇팔 파트에 전송합니다.
        status_msg = String()
        order_str = " -> ".join([f"#{p['seq']}:{p['color']}" for p in self.accumulated_paprikas]) if self.accumulated_paprikas else "Empty"
        status_msg.data = f"Total: {total_count} | Red: {red_count} | Yellow: {yellow_count} | Order: [{order_str}]"
        self.publisher_.publish(status_msg)

        # 5) 시각화 오버레이
        overlay = frame.copy()
        for p in self.accumulated_paprikas:
            if p["visible"] and p["mask"] is not None:
                bgr_color = self.CLASS_COLORS.get(p["color"], (255, 255, 255))
                pts = p["mask"].astype(np.int32)
                cv2.fillPoly(overlay, [pts], bgr_color)

        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # 바운딩 박스 및 번호 표기
        for p in self.accumulated_paprikas:
            if p["visible"]:
                x1, y1, x2, y2 = p["box"]
                color_name = p["color"]
                seq = p["seq"]
                conf = p["conf"]
                bgr_color = self.CLASS_COLORS.get(color_name, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                label = f"#{seq} {color_name} ({conf:.2f})"
                cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)

        # 6) 상단 통계 정보 패널
        stats_bg = frame.copy()
        cv2.rectangle(stats_bg, (15, 15), (750, 125), (30, 30, 30), -1)
        cv2.addWeighted(stats_bg, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "[Robust Harvest Box Status]", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Accumulated: {total_count}  |  Red: {red_count}  |  Yellow: {yellow_count}", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Order: {order_str}", (25, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Harvest Box Robust Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_filename = f"box_robust_snapshot_{timestamp}.jpg"
            img_save_path = os.path.join(self.OUTPUT_DIR, img_filename)
            
            cv2.imwrite(img_save_path, frame)

            box_order_list = [f"#{p['seq']} {p['color']}" for p in self.accumulated_paprikas]
            log_entry = (
                f"[{timestamp}] ROBUST SNAPSHOT SAVED\n"
                f"  - Total Count : {total_count} (Red: {red_count}, Yellow: {yellow_count})\n"
                f"  - Box Order   : {box_order_list}\n"
                f"  - Image Path  : {img_save_path}\n"
                f"{'-'*60}\n"
            )
            
            print("\n" + log_entry)
            with open(self.LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(log_entry)

        elif key == ord('r'):
            self.accumulated_paprikas.clear()
            self.candidates.clear()
            print("\n🔄 Accumulated box and candidates have been RESET.\n")

        elif key == ord('q'):
            print("🚪 Exiting inspection program...")
            rclpy.shutdown()


def main(args=None):
    # [ROS 2 초기화 및 노드 실행]
    rclpy.init(args=args)
    vision_node = HarvestVisionNode()
    
    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_node.cap.release()
        cv2.destroyAllWindows()
        vision_node.destroy_node()
        rclpy.shutdown()
        print("✅ Inspection program terminated safely.")


if __name__ == '__main__':
    main()