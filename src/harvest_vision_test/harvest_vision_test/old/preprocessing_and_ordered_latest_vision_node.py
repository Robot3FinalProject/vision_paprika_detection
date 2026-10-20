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


class PreprocessedOrderedLatestVisionNode(Node):
    def __init__(self):
        super().__init__('preprocessed_ordered_latest_vision_node')
        
        # [ROS 2 Publisher] 가장 최근 투입된 객체의 순서와 색상 전송
        self.publisher_ = self.create_publisher(String, '/vision/latest_status', 10)

        # 1. 설정 및 모델 경로 스펙
        self.MODEL_PATH = "/home/heh/pj_detection_paprika/models/v3.x/yolo11_seg_2class_synth_and_simul_and_mokup_v3.0/weights/best.pt"
        self.CAM_INDEX = 4  
        self.CONF_THRESHOLD = 0.55  # 테스트 및 왜곡 보정을 위해 조절 가능한 임계값

        # =========================================================
        # [OpenCV 전처리 설정] 상자 안쪽 유효 영역 ROI 좌표 지정
        # (카메라 화면 1280x720 기준, 상자 내부만 남기기 위한 Crop 박스)
        # ※ 현장 환경에 맞춰 x1, y1, x2, y2 값을 세밀하게 조절하세요!
        # =========================================================
        self.ROI_X1, self.ROI_Y1 = 200, 50
        self.ROI_X2, self.ROI_Y2 = 1080, 670

        # 매칭 및 정착 파라미터
        self.MATCH_THRESHOLD = 350       # 기존 객체와 동일성 판단 거리
        self.SETTLE_TIME = 0.5           # 후보군이 정착으로 인정되는 시간 (초)
        self.POSITION_TOLERANCE = 60     # 후보군 유지 중 위치 오차 허용 범위

        self.CLASS_NAMES = {0: 'Red', 1: 'Yellow'}
        self.CLASS_COLORS = {
            'Red': (0, 0, 255),      
            'Yellow': (0, 215, 255)  
        }
        
        self.OUTPUT_DIR = "/home/heh/pj_detection_paprika/result/preprocessing_and_latest_object_inspection"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.get_logger().info(f"📦 Loading YOLO11-seg model from: {self.MODEL_PATH}")
        self.model = YOLO(self.MODEL_PATH)

        self.get_logger().info(f"🎥 Connecting to Logitech Webcam (Index: {self.CAM_INDEX})...")
        self.cap = cv2.VideoCapture(self.CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Error: Could not open webcam at index {self.CAM_INDEX}.")
            exit()

        # 상태 관리 구조체
        self.accumulated_paprikas = []  # [{seq, color, center, box, mask, conf}]
        self.candidates = {}            # {candidate_id: {color, center, first_seen, last_seen, box, mask, conf}}
        self.next_candidate_id = 0
        self.global_seq = 0             # 투입 순서 번호

        print("\n🚀 Starting [Preprocessed & Sequence-based] Harvest Box Inspection...")
        print("---------------------------------------------------------")
        print("[Controls]")
        print("  - Press 's' : Save current snapshot")
        print("  - Press 'r' : Reset all accumulated state")
        print("  - Press 'q' : Quit program")
        print("---------------------------------------------------------\n")

        self.timer = self.create_timer(0.1, self.vision_loop_callback)

    def vision_loop_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("❌ Error: Failed to grab frame from webcam.")
            return

        h, w, _ = frame.shape
        x1, y1, x2, y2 = self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2
        
        # 1. OpenCV 전처리: 원본 화면에 ROI 영역 시각화 (디버깅용 초록색 박스)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 2. 상자 내부 영역만 잘라내기 (Crop 전처리)
        cropped_roi = frame[y1:y2, x1:x2]

        current_detections = []
        if cropped_roi.size > 0:
            # 3. 크롭된 ROI 이미지로 YOLO 추론 수행
            results = self.model.predict(cropped_roi, conf=self.CONF_THRESHOLD, verbose=False)[0]
            
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                masks_xy = results.masks.xy if results.masks is not None else [None] * len(boxes)

                for box, cls_id, conf, mask in zip(boxes, classes, confs, masks_xy):
                    class_name = self.CLASS_NAMES.get(int(cls_id), None)
                    if class_name not in ['Red', 'Yellow']:
                        continue

                    bx1, by1, bx2, by2 = map(int, box)
                    
                    # 💡 핵심: 크롭된 로컬 좌표계를 다시 전체 원본 프레임(Frame) 좌표계로 역변환
                    orig_bx1 = bx1 + x1
                    orig_by1 = by1 + y1
                    orig_bx2 = bx2 + x1
                    orig_by2 = by2 + y1

                    center_x = int((orig_bx1 + orig_bx2) / 2)
                    center_y = int((orig_by1 + orig_by2) / 2)

                    current_detections.append({
                        "color": class_name,
                        "conf": float(conf),
                        "box": (orig_bx1, orig_by1, orig_bx2, orig_by2),
                        "center": (center_x, center_y),
                        "mask": mask
                    })

        current_time = time.time()

        # 4) 이미 확정된 누적 객체들과 현재 감지 매칭 (밀려나거나 위치가 바뀐 기존 객체 갱신)
        matched_acc_indices = set()
        unmatched_detections = []

        for det in current_detections:
            det_cx, det_cy = det["center"]
            det_color = det["color"]

            matched_acc = None
            min_dist = float('inf')

            for idx, acc in enumerate(self.accumulated_paprikas):
                if acc["color"] == det_color and idx not in matched_acc_indices:
                    acc_cx, acc_cy = acc["center"]
                    dist = np.sqrt((det_cx - acc_cx)**2 + (det_cy - acc_cy)**2)
                    if dist < self.MATCH_THRESHOLD and dist < min_dist:
                        min_dist = dist
                        matched_acc = acc
                        matched_acc_idx = idx

            if matched_acc is not None:
                matched_acc["center"] = (det_cx, det_cy)
                matched_acc["box"] = det["box"]
                matched_acc["mask"] = det["mask"]
                matched_acc["conf"] = det["conf"]
                matched_acc_indices.add(matched_acc_idx)
            else:
                unmatched_detections.append(det)

        # 5) 누적 객체에 매칭되지 않은 신규 탐지 건들을 후보군(Candidates)으로 관리
        current_candidate_keys = set()

        for det in unmatched_detections:
            det_cx, det_cy = det["center"]
            det_color = det["color"]

            matched_cand_key = None
            min_cand_dist = float('inf')

            for cand_key, cand in self.candidates.items():
                if cand["color"] == det_color:
                    dist = np.sqrt((det_cx - cand["center"][0])**2 + (det_cy - cand["center"][1])**2)
                    if dist < self.POSITION_TOLERANCE and dist < min_cand_dist:
                        min_cand_dist = dist
                        matched_cand_key = cand_key

            if matched_cand_key is not None:
                self.candidates[matched_cand_key]["center"] = (det_cx, det_cy)
                self.candidates[matched_cand_key]["box"] = det["box"]
                self.candidates[matched_cand_key]["mask"] = det["mask"]
                self.candidates[matched_cand_key]["conf"] = det["conf"]
                self.candidates[matched_cand_key]["last_seen"] = current_time
                current_candidate_keys.add(matched_cand_key)
            else:
                new_key = self.next_candidate_id
                self.next_candidate_id += 1
                self.candidates[new_key] = {
                    "color": det_color,
                    "center": (det_cx, det_cy),
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "box": det["box"],
                    "mask": det["mask"],
                    "conf": det["conf"]
                }
                current_candidate_keys.add(new_key)

        # 6) 오랫동안 정착된 후보군을 '정식 누적 객체(Seq 부여)'로 승격
        expired_candidates = []
        for cand_key, cand in self.candidates.items():
            duration = cand["last_seen"] - cand["first_seen"]
            if duration >= self.SETTLE_TIME:
                too_close = False
                for acc in self.accumulated_paprikas:
                    if np.sqrt((cand["center"][0] - acc["center"][0])**2 + (cand["center"][1] - acc["center"][1])**2) < 80:
                        too_close = True
                        break
                
                if not too_close:
                    self.global_seq += 1
                    new_item = {
                        "seq": self.global_seq,
                        "color": cand["color"],
                        "center": cand["center"],
                        "box": cand["box"],
                        "mask": cand["mask"],
                        "conf": cand["conf"]
                    }
                    self.accumulated_paprikas.append(new_item)
                    self.get_logger().info(f"✨ Settled & Committed: #{self.global_seq} ({cand['color']})")
                
                expired_candidates.append(cand_key)

        for cand_key in expired_candidates:
            del self.candidates[cand_key]

        # 사라진 후보군 정리
        stale_keys = [k for k, v in self.candidates.items() if current_time - v["last_seen"] > 0.4]
        for k in stale_keys:
            del self.candidates[k]

        # 7) [가장 최근 객체 판정] 누적된 객체 중 seq가 가장 높은 객체를 최신 타겟으로 지정
        latest_obj = None
        if self.accumulated_paprikas:
            latest_obj = max(self.accumulated_paprikas, key=lambda x: x["seq"])

        # [ROS 2 퍼블리시] 최신 투입 객체 상태 전송
        status_msg = String()
        if latest_obj:
            status_msg.data = f"Latest_Seq: #{latest_obj['seq']} | Color: {latest_obj['color']}"
        else:
            status_msg.data = "Latest_Seq: None | Color: None"
        self.publisher_.publish(status_msg)

        # =========================================================
        # 시각화 오버레이
        # =========================================================
        for acc in self.accumulated_paprikas:
            bx1, by1, bx2, by2 = acc["box"]
            color_name = acc["color"]
            bgr_color = self.CLASS_COLORS.get(color_name, (255, 255, 255))
            
            is_latest = (latest_obj and latest_obj["seq"] == acc["seq"])
            thickness = 4 if is_latest else 2

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), bgr_color, thickness)
            
            label = f"#{acc['seq']} {color_name} ({acc['conf']:.2f})"
            if is_latest:
                label = f"LATEST -> " + label

            cv2.putText(frame, label, (bx1, max(25, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)

        # 상단 통계 패널
        stats_bg = frame.copy()
        cv2.rectangle(stats_bg, (15, 15), (720, 105), (30, 30, 30), -1)
        cv2.addWeighted(stats_bg, 0.7, frame, 0.3, 0, frame)

        total_count = len(self.accumulated_paprikas)
        red_count = sum(1 for x in self.accumulated_paprikas if x["color"] == 'Red')
        yellow_count = sum(1 for x in self.accumulated_paprikas if x["color"] == 'Yellow')
        latest_str = f"#{latest_obj['seq']}: {latest_obj['color']}" if latest_obj else "None"

        cv2.putText(frame, f"[Preprocess ROI + Sequence Tracking]", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {total_count} | Red: {red_count} | Yellow: {yellow_count}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Latest Inserted Target -> {latest_str}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

        cv2.imshow("Harvest Box Sequence Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_filename = f"preprocess_sequence_{timestamp}.jpg"
            img_save_path = os.path.join(self.OUTPUT_DIR, img_filename)
            cv2.imwrite(img_save_path, frame)
            print(f"\n📸 Snapshot saved: {img_save_path}\n")

        elif key == ord('r'):
            self.accumulated_paprikas.clear()
            self.candidates.clear()
            self.global_seq = 0
            print("\n🔄 Reset all accumulated state.\n")

        elif key == ord('q'):
            print("🚪 Exiting program...")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessedOrderedLatestVisionNode()
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