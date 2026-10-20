import sys
import os

sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

import cv2
import time
import numpy as np
from ultralytics import YOLO

# ROS 2 관련 임포트
try:
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# =========================================================
# 1. 설정 및 모델 경로 스펙
# =========================================================
MODEL_PATH = "/home/heh/pj_detection_paprika/models/v3.x/yolo11_seg_2class_synth_and_simul_and_mokup_v3.0/weights/best.pt"
# CAM_INDEX = 0
CAM_INDEX = 4
CONF_THRESHOLD = 0.40

ACCUMULATED_MATCH_THRESHOLD = 170
SETTLE_TIME = 0.3
POSITION_TOLERANCE = 25
IOU_MATCH_THRESHOLD = 0.20

CLASS_NAMES = {0: 'Red', 1: 'Yellow'}
CLASS_COLORS = {
    'Red': (0, 0, 255),      
    'Yellow': (0, 215, 255)  
}
OUTPUT_DIR = "/home/heh/pj_detection_paprika/result/box_inspection"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "harvest_box_inspection_log.txt")

# =========================================================
# 2. 세그멘테이션 마스크 IoU 계산 함수
# =========================================================
def calculate_mask_iou(mask1, mask2, img_shape):
    if mask1 is None or mask2 is None:
        return 0.0
    
    h, w = img_shape[:2]
    
    def polygon_to_mask(mask_pts, height, width):
        m = np.zeros((height, width), dtype=np.uint8)
        if mask_pts is not None and len(mask_pts) > 0:
            pts = np.array(mask_pts, dtype=np.int32)
            cv2.fillPoly(m, [pts], 1)
        return m

    m1 = polygon_to_mask(mask1, h, w)
    m2 = polygon_to_mask(mask2, h, w)
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 0.0
    return intersection / union


# =========================================================
# 3. ROS 2 비전 서버 노드 클래스 (화면 표시 상태 변수 추가)
# =========================================================
class HarvestVisionServerNode(Node):
    def __init__(self):
        super().__init__('harvest_box_inspector_node')
        self.latest_inserted_object = None
        self.has_new_insertion = False  
        
        # 화면에 실시간으로 띄울 상태 메시지 및 색상 관리
        self.display_status = "WAITING (No Request)"
        self.status_color = (200, 200, 200) # 회색
        
        # ROS 2 Service Server 생성
        self.srv = self.create_service(
            Trigger, 
            'get_latest_inserted_object', 
            self.handle_get_latest_object
        )
        self.get_logger().info("🤖 ROS 2 Service Server 'get_latest_inserted_object' is ready.")

    def handle_get_latest_object(self, request, response):
        if self.has_new_insertion and self.latest_inserted_object:
            response.success = True
            response.message = self.latest_inserted_object
            
            # 성공 시 화면 표시 상태 업데이트 (초록/노랑 계열)
            self.display_status = f"SUCCESS: {self.latest_inserted_object} Inserted!"
            self.status_color = CLASS_COLORS.get(self.latest_inserted_object, (0, 255, 0))
            
            # 요청 처리 후 플래그 초기화
            self.has_new_insertion = False 
        else:
            response.success = False
            response.message = "Fail"
            
            # 실패 시 화면 표시 상태 업데이트 (빨간색)
            self.display_status = "FAIL: No New Object Detected!"
            self.status_color = (0, 0, 255)
            
        return response


def main():
    print(f"📦 Loading YOLO11-seg model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"🎥 Connecting to Logitech Webcam (Index: {CAM_INDEX})...")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam at index {CAM_INDEX}.")
        return

    accumulated_paprikas = []  
    candidates = {}            

    if ROS2_AVAILABLE and not rclpy.ok():
        rclpy.init()
    
    vision_node = HarvestVisionServerNode()

    print("\n🚀 Starting Harvest Box Inspection with ROS 2 Service Interface...")
    print("---------------------------------------------------------")
    print("[Controls]")
    print("  - Press 's' : Save current snapshot and log to file")
    print("  - Press 'r' : Reset accumulated box (Clear all)")
    print("  - Press 'q' : Quit program")
    print("---------------------------------------------------------\n")

    try:
        while cap.isOpened():
            if ROS2_AVAILABLE:
                rclpy.spin_once(vision_node, timeout_sec=0.01)

            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to grab frame from webcam.")
                break

            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            
            current_frame_detections = []

            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                masks_xy = results.masks.xy if results.masks is not None else [None] * len(boxes)

                for box, cls_id, conf, mask in zip(boxes, classes, confs, masks_xy):
                    class_name = CLASS_NAMES.get(int(cls_id), None)
                    if class_name not in ['Red', 'Yellow']:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    current_frame_detections.append({
                        "color": class_name,
                        "conf": float(conf),
                        "box": (x1, y1, x2, y2),
                        "center": (center_x, center_y),
                        "mask": mask
                    })

            current_time = time.time()

            for acc in accumulated_paprikas:
                acc["visible"] = False

            unmatched_detections = []
            
            for det in current_frame_detections:
                det_cx, det_cy = det["center"]
                det_color = det["color"]

                matched_acc = None
                best_score = -float('inf')

                for acc in accumulated_paprikas:
                    if acc["color"] == det_color:
                        acc_cx, acc_cy = acc["center"]
                        dist = np.sqrt((det_cx - acc_cx)**2 + (det_cy - acc_cy)**2)
                        iou = calculate_mask_iou(acc["mask"], det["mask"], frame.shape)

                        if dist < ACCUMULATED_MATCH_THRESHOLD or iou > IOU_MATCH_THRESHOLD:
                            score = (iou * 100.0) - (dist * 0.1)
                            if score > best_score:
                                best_score = score
                                matched_acc = acc

                if matched_acc is not None:
                    matched_acc["center"] = (det_cx, det_cy)
                    matched_acc["box"] = det["box"]
                    matched_acc["mask"] = det["mask"]
                    matched_acc["conf"] = det["conf"]
                    matched_acc["visible"] = True
                else:
                    unmatched_detections.append(det)

            for det in unmatched_detections:
                det_cx, det_cy = det["center"]
                det_color = det["color"]

                matched_cand_key = None
                min_cand_dist = float('inf')
                
                for cand_key, cand in candidates.items():
                    if cand["color"] == det_color:
                        dist = np.sqrt((det_cx - cand["center"][0])**2 + (det_cy - cand["center"][1])**2)
                        if dist < 80 and dist < min_cand_dist:
                            min_cand_dist = dist
                            matched_cand_key = cand_key

                if matched_cand_key is not None:
                    cand = candidates[matched_cand_key]
                    dist_moved = np.sqrt((det_cx - cand["center"][0])**2 + (det_cy - cand["center"][1])**2)
                    
                    cand["center"] = (det_cx, det_cy)
                    cand["box"] = det["box"]
                    cand["mask"] = det["mask"]
                    cand["conf"] = det["conf"]

                    if dist_moved < POSITION_TOLERANCE:
                        if (current_time - cand["first_seen"]) >= SETTLE_TIME:
                            new_seq = len(accumulated_paprikas) + 1
                            new_obj = {
                                "seq": new_seq,
                                "color": cand["color"],
                                "center": cand["center"],
                                "box": cand["box"],
                                "mask": cand["mask"],
                                "conf": cand["conf"],
                                "visible": True
                            }
                            accumulated_paprikas.append(new_obj)
                            
                            vision_node.latest_inserted_object = cand["color"]
                            vision_node.has_new_insertion = True
                            print(f"✨ [ROS 2 Service Target] New Object Settled! -> Latest Inserted: {vision_node.latest_inserted_object} (#{new_seq})")
                            
                            del candidates[matched_cand_key]
                            continue
                    else:
                        cand["first_seen"] = current_time
                else:
                    cand_key = f"{det_color}_{int(det_cx)}_{int(det_cy)}_{int(current_time*1000)}"
                    candidates[cand_key] = {
                        "color": det_color,
                        "center": (det_cx, det_cy),
                        "box": det["box"],
                        "mask": det["mask"],
                        "conf": det["conf"],
                        "first_seen": current_time
                    }

            stale_keys = [k for k, v in candidates.items() if (current_time - v["first_seen"]) > 3.0]
            for k in stale_keys:
                del candidates[k]

            accumulated_paprikas = sorted(accumulated_paprikas, key=lambda x: x["seq"])
            total_count = len(accumulated_paprikas)
            red_count = sum(1 for p in accumulated_paprikas if p["color"] == 'Red')
            yellow_count = sum(1 for p in accumulated_paprikas if p["color"] == 'Yellow')

            # 시각화 오버레이
            overlay = frame.copy()
            for p in accumulated_paprikas:
                if p["visible"] and p["mask"] is not None:
                    bgr_color = CLASS_COLORS.get(p["color"], (255, 255, 255))
                    pts = np.array(p["mask"], dtype=np.int32)
                    if len(pts) > 0:
                        cv2.fillPoly(overlay, [pts], bgr_color)

            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            for p in accumulated_paprikas:
                if p["visible"]:
                    x1, y1, x2, y2 = p["box"]
                    color_name = p["color"]
                    seq = p["seq"]
                    conf = p["conf"]
                    bgr_color = CLASS_COLORS.get(color_name, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                    label = f"#{seq} {color_name} ({conf:.2f})"
                    cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)

            # 상단 통계 정보 패널 (실패/성공 상태 UI 텍스트 추가)
            stats_bg = frame.copy()
            cv2.rectangle(stats_bg, (15, 15), (820, 150), (30, 30, 30), -1)
            cv2.addWeighted(stats_bg, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, "[Harvest Box ROS 2 Service Status]", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Total: {total_count}  |  Red: {red_count}  |  Yellow: {yellow_count}", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 실시간 서비스 요청 결과(성공/실패) 색상 반영 출력
            cv2.putText(frame, f"Request Status -> {vision_node.display_status}", (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vision_node.status_color, 2)

            order_str = " -> ".join([f"#{p['seq']}:{p['color']}" for p in accumulated_paprikas]) if accumulated_paprikas else "Box is Empty"
            cv2.putText(frame, f"Order: {order_str}", (25, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Harvest Box Robust Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_filename = f"box_robust_snapshot_{timestamp}.jpg"
                img_save_path = os.path.join(OUTPUT_DIR, img_filename)
                
                cv2.imwrite(img_save_path, frame)
                print(f"\n📸 Snapshot saved: {img_save_path}\n")

            elif key == ord('r'):
                accumulated_paprikas.clear()
                candidates.clear()
                vision_node.latest_inserted_object = None
                vision_node.has_new_insertion = False
                vision_node.display_status = "RESET (Box Cleared)"
                vision_node.status_color = (200, 200, 200)
                print("\n🔄 Accumulated box, candidates, and ROS 2 target have been RESET.\n")

            elif key == ord('q'):
                print("🚪 Exiting inspection program...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ROS2_AVAILABLE:
            vision_node.destroy_node()
            rclpy.shutdown()
        print("✅ Inspection program terminated safely.")

if __name__ == '__main__':
    main()









# import sys
# import os

# sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

# import cv2
# import time
# import numpy as np
# from ultralytics import YOLO

# # ROS 2 관련 임포트
# try:
#     import rclpy
#     from rclpy.node import Node
#     from std_srvs.srv import Trigger
#     ROS2_AVAILABLE = True
# except ImportError:
#     ROS2_AVAILABLE = False

# # =========================================================
# # 1. 설정 및 모델 경로 스펙 (현재 성능 유지)
# # =========================================================
# MODEL_PATH = "/home/heh/pj_detection_paprika/models/v3.x/yolo11_seg_2class_synth_and_simul_and_mokup_v3.0/weights/best.pt"
# CAM_INDEX = 0
# CONF_THRESHOLD = 0.40

# # 검증된 파라미터 유지
# ACCUMULATED_MATCH_THRESHOLD = 170
# SETTLE_TIME = 0.3
# POSITION_TOLERANCE = 25
# IOU_MATCH_THRESHOLD = 0.20

# CLASS_NAMES = {0: 'Red', 1: 'Yellow'}
# CLASS_COLORS = {
#     'Red': (0, 0, 255),      
#     'Yellow': (0, 215, 255)  
# }
# OUTPUT_DIR = "/home/heh/pj_detection_paprika/result/box_inspection"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "harvest_box_inspection_log.txt")

# # =========================================================
# # 2. 세그멘테이션 마스크 IoU 계산 함수 (유지)
# # =========================================================
# def calculate_mask_iou(mask1, mask2, img_shape):
#     if mask1 is None or mask2 is None:
#         return 0.0
    
#     h, w = img_shape[:2]
    
#     def polygon_to_mask(mask_pts, height, width):
#         m = np.zeros((height, width), dtype=np.uint8)
#         if mask_pts is not None and len(mask_pts) > 0:
#             pts = np.array(mask_pts, dtype=np.int32)
#             cv2.fillPoly(m, [pts], 1)
#         return m

#     m1 = polygon_to_mask(mask1, h, w)
#     m2 = polygon_to_mask(mask2, h, w)
    
#     intersection = np.logical_and(m1, m2).sum()
#     union = np.logical_or(m1, m2).sum()
    
#     if union == 0:
#         return 0.0
#     return intersection / union


# # =========================================================
# # 3. ROS 2 비전 서버 노드 클래스 정의 (성공/실패 피드백 추가)
# # =========================================================
# class HarvestVisionServerNode(Node):
#     def __init__(self):
#         super().__init__('harvest_box_inspector_node')
#         self.latest_inserted_object = None
#         self.has_new_insertion = False  # 새로운 객체 유입 여부 플래그
        
#         # ROS 2 Service Server 생성
#         self.srv = self.create_service(
#             Trigger, 
#             'get_latest_inserted_object', 
#             self.handle_get_latest_object
#         )
#         self.get_logger().info("🤖 ROS 2 Service Server 'get_latest_inserted_object' is ready.")

#     def handle_get_latest_object(self, request, response):
#         if self.has_new_insertion and self.latest_inserted_object:
#             response.success = True
#             response.message = self.latest_inserted_object
#             # 한 번 요청을 받아가면 플래그를 초기화하여 다음 요청 시 중복 성공 방지 (선택적)
#             self.has_new_insertion = False 
#         else:
#             response.success = False
#             response.message = "Fail"
            
#         return response


# def main():
#     print(f"📦 Loading YOLO11-seg model from: {MODEL_PATH}")
#     model = YOLO(MODEL_PATH)

#     print(f"🎥 Connecting to Logitech Webcam (Index: {CAM_INDEX})...")
#     cap = cv2.VideoCapture(CAM_INDEX)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     if not cap.isOpened():
#         print(f"❌ Error: Could not open webcam at index {CAM_INDEX}.")
#         return

#     accumulated_paprikas = []  
#     candidates = {}            

#     # ROS 2 초기화
#     if ROS2_AVAILABLE and not rclpy.ok():
#         rclpy.init()
    
#     vision_node = HarvestVisionServerNode()

#     print("\n🚀 Starting Harvest Box Inspection with ROS 2 Service Interface...")
#     print("---------------------------------------------------------")
#     print("[Controls]")
#     print("  - Press 's' : Save current snapshot and log to file")
#     print("  - Press 'r' : Reset accumulated box (Clear all)")
#     print("  - Press 'q' : Quit program")
#     print("---------------------------------------------------------\n")

#     try:
#         while cap.isOpened():
#             if ROS2_AVAILABLE:
#                 rclpy.spin_once(vision_node, timeout_sec=0.01)

#             ret, frame = cap.read()
#             if not ret:
#                 print("❌ Error: Failed to grab frame from webcam.")
#                 break

#             results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            
#             current_frame_detections = []

#             if results.boxes is not None:
#                 boxes = results.boxes.xyxy.cpu().numpy()
#                 classes = results.boxes.cls.cpu().numpy()
#                 confs = results.boxes.conf.cpu().numpy()
#                 masks_xy = results.masks.xy if results.masks is not None else [None] * len(boxes)

#                 for box, cls_id, conf, mask in zip(boxes, classes, confs, masks_xy):
#                     class_name = CLASS_NAMES.get(int(cls_id), None)
#                     if class_name not in ['Red', 'Yellow']:
#                         continue

#                     x1, y1, x2, y2 = map(int, box)
#                     center_x = int((x1 + x2) / 2)
#                     center_y = int((y1 + y2) / 2)

#                     current_frame_detections.append({
#                         "color": class_name,
#                         "conf": float(conf),
#                         "box": (x1, y1, x2, y2),
#                         "center": (center_x, center_y),
#                         "mask": mask
#                     })

#             current_time = time.time()

#             for acc in accumulated_paprikas:
#                 acc["visible"] = False

#             unmatched_detections = []
            
#             for det in current_frame_detections:
#                 det_cx, det_cy = det["center"]
#                 det_color = det["color"]

#                 matched_acc = None
#                 best_score = -float('inf')

#                 for acc in accumulated_paprikas:
#                     if acc["color"] == det_color:
#                         acc_cx, acc_cy = acc["center"]
#                         dist = np.sqrt((det_cx - acc_cx)**2 + (det_cy - acc_cy)**2)
#                         iou = calculate_mask_iou(acc["mask"], det["mask"], frame.shape)

#                         if dist < ACCUMULATED_MATCH_THRESHOLD or iou > IOU_MATCH_THRESHOLD:
#                             score = (iou * 100.0) - (dist * 0.1)
#                             if score > best_score:
#                                 best_score = score
#                                 matched_acc = acc

#                 if matched_acc is not None:
#                     matched_acc["center"] = (det_cx, det_cy)
#                     matched_acc["box"] = det["box"]
#                     matched_acc["mask"] = det["mask"]
#                     matched_acc["conf"] = det["conf"]
#                     matched_acc["visible"] = True
#                 else:
#                     unmatched_detections.append(det)

#             for det in unmatched_detections:
#                 det_cx, det_cy = det["center"]
#                 det_color = det["color"]

#                 matched_cand_key = None
#                 min_cand_dist = float('inf')
                
#                 for cand_key, cand in candidates.items():
#                     if cand["color"] == det_color:
#                         dist = np.sqrt((det_cx - cand["center"][0])**2 + (det_cy - cand["center"][1])**2)
#                         if dist < 80 and dist < min_cand_dist:
#                             min_cand_dist = dist
#                             matched_cand_key = cand_key

#                 if matched_cand_key is not None:
#                     cand = candidates[matched_cand_key]
#                     dist_moved = np.sqrt((det_cx - cand["center"][0])**2 + (det_cy - cand["center"][1])**2)
                    
#                     cand["center"] = (det_cx, det_cy)
#                     cand["box"] = det["box"]
#                     cand["mask"] = det["mask"]
#                     cand["conf"] = det["conf"]

#                     if dist_moved < POSITION_TOLERANCE:
#                         if (current_time - cand["first_seen"]) >= SETTLE_TIME:
#                             new_seq = len(accumulated_paprikas) + 1
#                             new_obj = {
#                                 "seq": new_seq,
#                                 "color": cand["color"],
#                                 "center": cand["center"],
#                                 "box": cand["box"],
#                                 "mask": cand["mask"],
#                                 "conf": cand["conf"],
#                                 "visible": True
#                             }
#                             accumulated_paprikas.append(new_obj)
                            
#                             # 최신 유입 객체 정보 및 성공 플래그 업데이트
#                             vision_node.latest_inserted_object = cand["color"]
#                             vision_node.has_new_insertion = True
#                             print(f"✨ [ROS 2 Service Target] New Object Settled! -> Latest Inserted: {vision_node.latest_inserted_object} (#{new_seq})")
                            
#                             del candidates[matched_cand_key]
#                             continue
#                     else:
#                         cand["first_seen"] = current_time
#                 else:
#                     cand_key = f"{det_color}_{int(det_cx)}_{int(det_cy)}_{int(current_time*1000)}"
#                     candidates[cand_key] = {
#                         "color": det_color,
#                         "center": (det_cx, det_cy),
#                         "box": det["box"],
#                         "mask": det["mask"],
#                         "conf": det["conf"],
#                         "first_seen": current_time
#                     }

#             stale_keys = [k for k, v in candidates.items() if (current_time - v["first_seen"]) > 3.0]
#             for k in stale_keys:
#                 del candidates[k]

#             accumulated_paprikas = sorted(accumulated_paprikas, key=lambda x: x["seq"])
#             total_count = len(accumulated_paprikas)
#             red_count = sum(1 for p in accumulated_paprikas if p["color"] == 'Red')
#             yellow_count = sum(1 for p in accumulated_paprikas if p["color"] == 'Yellow')

#             # 시각화 오버레이
#             overlay = frame.copy()
#             for p in accumulated_paprikas:
#                 if p["visible"] and p["mask"] is not None:
#                     bgr_color = CLASS_COLORS.get(p["color"], (255, 255, 255))
#                     pts = np.array(p["mask"], dtype=np.int32)
#                     if len(pts) > 0:
#                         cv2.fillPoly(overlay, [pts], bgr_color)

#             cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

#             for p in accumulated_paprikas:
#                 if p["visible"]:
#                     x1, y1, x2, y2 = p["box"]
#                     color_name = p["color"]
#                     seq = p["seq"]
#                     conf = p["conf"]
#                     bgr_color = CLASS_COLORS.get(color_name, (255, 255, 255))

#                     cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
#                     label = f"#{seq} {color_name} ({conf:.2f})"
#                     cv2.putText(frame, label, (x1, max(25, y1 - 10)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)

#             # 상단 통계 정보 패널
#             stats_bg = frame.copy()
#             cv2.rectangle(stats_bg, (15, 15), (780, 150), (30, 30, 30), -1)
#             cv2.addWeighted(stats_bg, 0.7, frame, 0.3, 0, frame)

#             cv2.putText(frame, "[Harvest Box ROS 2 Service Status]", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             cv2.putText(frame, f"Total: {total_count}  |  Red: {red_count}  |  Yellow: {yellow_count}", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             latest_str = vision_node.latest_inserted_object if vision_node.latest_inserted_object else "None"
#             cv2.putText(frame, f"Latest Service Response: {latest_str}", (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

#             order_str = " -> ".join([f"#{p['seq']}:{p['color']}" for p in accumulated_paprikas]) if accumulated_paprikas else "Box is Empty"
#             cv2.putText(frame, f"Order: {order_str}", (25, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

#             cv2.imshow("Harvest Box Robust Tracking", frame)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('s'):
#                 timestamp = time.strftime("%Y%m%d_%H%M%S")
#                 img_filename = f"box_robust_snapshot_{timestamp}.jpg"
#                 img_save_path = os.path.join(OUTPUT_DIR, img_filename)
                
#                 cv2.imwrite(img_save_path, frame)

#                 box_order_list = [f"#{p['seq']} {p['color']}" for p in accumulated_paprikas]
#                 log_entry = (
#                     f"[{timestamp}] ROBUST SNAPSHOT SAVED\n"
#                     f"  - Total Count : {total_count} (Red: {red_count}, Yellow: {yellow_count})\n"
#                     f"  - Latest Serv : {vision_node.latest_inserted_object}\n"
#                     f"  - Box Order   : {box_order_list}\n"
#                     f"  - Image Path  : {img_save_path}\n"
#                     f"{'-'*60}\n"
#                 )
                
#                 print("\n" + log_entry)
#                 with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
#                     f.write(log_entry)

#             elif key == ord('r'):
#                 accumulated_paprikas.clear()
#                 candidates.clear()
#                 vision_node.latest_inserted_object = None
#                 vision_node.has_new_insertion = False
#                 print("\n🔄 Accumulated box, candidates, and ROS 2 target have been RESET.\n")

#             elif key == ord('q'):
#                 print("🚪 Exiting inspection program...")
#                 break

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         if ROS2_AVAILABLE:
#             vision_node.destroy_node()
#             rclpy.shutdown()
#         print("✅ Inspection program terminated safely.")

# if __name__ == '__main__':
#     main()