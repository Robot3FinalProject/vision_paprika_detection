
import os
import sys
import time

# 시스템 Python 패키지와 가상환경 패키지 충돌 방지 (기존 설정 유지)
sys.path = [
    p for p in sys.path
    if "/usr/lib/python3/dist-packages" not in p
]

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    OUTPUT_DIR = (
        "/home/heh/pj_detection_paprika/result/depth_inspection"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    started = False

    # RGB와 Depth 스트림 설정
    config.enable_stream(
        rs.stream.color, 640, 480, rs.format.bgr8, 30
    )
    config.enable_stream(
        rs.stream.depth, 640, 480, rs.format.z16, 30
    )

    try:
        print("\nInitializing RealSense...")

        # 중요: 자동 hardware_reset() 호출 제거
        profile = pipeline.start(config)
        started = True

        print("Pipeline started!")

        device = profile.get_device()
        print("Camera:", device.get_info(rs.camera_info.name))

        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale:.6f} m/unit")

        print("\nControls: [s] Save snapshot | [q] Quit")

        consecutive_timeouts = 0

        while True:
            try:
                frames = pipeline.wait_for_frames(
                    timeout_ms=10000
                )
            except RuntimeError as e:
                consecutive_timeouts += 1
                print(
                    f"[WARNING] Frame timeout "
                    f"({consecutive_timeouts}/3): {e}"
                )

                if consecutive_timeouts >= 3:
                    print(
                        "\n[ERROR] No frames received repeatedly."
                        "\nCheck USB connection, camera streaming, "
                        "and other applications using the camera."
                    )
                    break

                continue

            consecutive_timeouts = 0

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("[WARNING] Missing RGB or Depth frame")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Depth 시각화
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )

            h, w = depth_image.shape
            center_y, center_x = h // 2, w // 2

            center_depth_raw = int(
                depth_image[center_y, center_x]
            )
            center_distance_m = center_depth_raw * depth_scale

            if center_depth_raw == 0:
                distance_text = "Center: No depth"
            else:
                distance_text = (
                    f"Center: {center_distance_m:.3f} m"
                )

            cv2.putText(
                color_image,
                f"[RGB] {distance_text}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                depth_colormap,
                f"[Depth] {distance_text} "
                f"(Raw: {center_depth_raw})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            combined_view = np.hstack(
                (color_image, depth_colormap)
            )

            cv2.imshow(
                "RealSense RGB-D Inspection",
                combined_view,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                rgb_path = os.path.join(
                    OUTPUT_DIR,
                    f"realsense_rgb_{timestamp}.jpg",
                )
                depth_path = os.path.join(
                    OUTPUT_DIR,
                    f"realsense_depth_{timestamp}.jpg",
                )

                rgb_ok = cv2.imwrite(rgb_path, color_image)
                depth_ok = cv2.imwrite(
                    depth_path, depth_colormap
                )

                if rgb_ok and depth_ok:
                    print("Snapshots saved:")
                    print(" RGB:", rgb_path)
                    print("Depth:", depth_path)
                else:
                    print("[ERROR] Failed to save snapshots")

            elif key == ord("q"):
                print("Exiting viewer...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")

    finally:
        if started:
            try:
                pipeline.stop()
                print("Pipeline stopped.")
            except Exception as e:
                print(f"Pipeline cleanup warning: {e}")

        cv2.destroyAllWindows()
        print("Program terminated.")


if __name__ == "__main__":
    main()
