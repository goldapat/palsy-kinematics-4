import cv2
import mediapipe as mp
import glob
import os
import math
import csv
import re

''' 
VERSION 2
เป้าหมาย = ดู MINI-RGBD file 01 
Tracking with LEFT and RIGHT wrist (joint 20, 21) with mediapipe. Preview with cv2 window แล้วเทียบกับค่าจริง (ground truth)

Check line 127 for more info

'''

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
seq_path = os.path.join(BASE_DIR, "MINI-RGBD_web", "01", "rgb")
gt_path = os.path.join(BASE_DIR, "MINI-RGBD_web", "01", "joints_2Ddep")
csv_output_path = os.path.join(BASE_DIR, "pose_tracking_results.csv")

frame_files = sorted(glob.glob(os.path.join(seq_path, "*.png")))

def get_gt_coords_from_file(gt_file_path, joint_id):
    if not os.path.exists(gt_file_path):
        return None, None
        
    try:
        with open(gt_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) > joint_id:
                parts = lines[joint_id].split()
                if len(parts) >= 2:
                    return float(parts[0]), float(parts[1])
    except Exception:
        pass
        
    return None, None

rw_error_list = []
lw_error_list = []

with open(csv_output_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "filename", 
        "mp_rw_x", "mp_rw_y", "gt_rw_x", "gt_rw_y", "rw_error_px",
        "mp_lw_x", "mp_lw_y", "gt_lw_x", "gt_lw_y", "lw_error_px",
        "detected"
    ])

    for f in frame_files:
        frame = cv2.imread(f)
        if frame is None:
            continue
            
        h, w, _ = frame.shape
        filename = os.path.basename(f)
        
        numbers_only = re.sub(r'\D', '', filename)
        gt_filename = f"syn_joints_2Ddep_{numbers_only}.txt"
        gt_file_path = os.path.join(gt_path, gt_filename)
        
        # MINI-RGBD indices: Joint 21 = Right Wrist, Joint 20 = Left Wrist
        gt_rw_x, gt_rw_y = get_gt_coords_from_file(gt_file_path, joint_id=21)
        gt_lw_x, gt_lw_y = get_gt_coords_from_file(gt_file_path, joint_id=20)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # --- Draw Ground Truth Points (RED) ---
        if gt_rw_x is not None and gt_rw_y is not None:
            cv2.circle(frame, (int(gt_rw_x), int(gt_rw_y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, "GT_R", (int(gt_rw_x) + 5, int(gt_rw_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if gt_lw_x is not None and gt_lw_y is not None:
            cv2.circle(frame, (int(gt_lw_x), int(gt_lw_y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, "GT_L", (int(gt_lw_x) + 5, int(gt_lw_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # --- Process MediaPipe Landmarks ---
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract Right Wrist
            rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            mp_rw_x, mp_rw_y = int(rw.x * w), int(rw.y * h)
            cv2.circle(frame, (mp_rw_x, mp_rw_y), 5, (0, 255, 0), -1)

            # Extract Left Wrist
            lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            mp_lw_x, mp_lw_y = int(lw.x * w), int(lw.y * h)
            cv2.circle(frame, (mp_lw_x, mp_lw_y), 5, (255, 0, 0), -1) # Blue dot for Left Wrist

            # Right Wrist Error calculation
            rw_error = None
            if gt_rw_x is not None and gt_rw_y is not None:
                rw_error = math.sqrt((mp_rw_x - gt_rw_x)**2 + (mp_rw_y - gt_rw_y)**2)
                rw_error_list.append(rw_error)
                cv2.line(frame, (int(gt_rw_x), int(gt_rw_y)), (mp_rw_x, mp_rw_y), (255, 255, 0), 1)

            # Left Wrist Error calculation
            lw_error = None
            if gt_lw_x is not None and gt_lw_y is not None:
                lw_error = math.sqrt((mp_lw_x - gt_lw_x)**2 + (mp_lw_y - gt_lw_y)**2)
                lw_error_list.append(lw_error)
                cv2.line(frame, (int(gt_lw_x), int(gt_lw_y)), (mp_lw_x, mp_lw_y), (255, 255, 0), 1)

            # Write data to CSV
            writer.writerow([
                filename, 
                mp_rw_x, mp_rw_y, round(gt_rw_x, 2) if gt_rw_x else None, round(gt_rw_y, 2) if gt_rw_y else None, round(rw_error, 2) if rw_error else None,
                mp_lw_x, mp_lw_y, round(gt_lw_x, 2) if gt_lw_x else None, round(gt_lw_y, 2) if gt_lw_y else None, round(lw_error, 2) if lw_error else None,
                True
            ])
        else:
            writer.writerow([
                filename, 
                None, None, round(gt_rw_x, 2) if gt_rw_x else None, round(gt_rw_y, 2) if gt_rw_y else None, None,
                None, None, round(gt_lw_x, 2) if gt_lw_x else None, round(gt_lw_y, 2) if gt_lw_y else None, None,
                False
            ])

        cv2.imshow("MINI-RGBD Validation | Red=GT, Green=MP Right Wrist, Blue=MP Left Wrist", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # cv2.waitKey(30) = Default cv2.waitKey(1) = Fast // Press Q to stop
            break

cv2.destroyAllWindows()

# Print statistics
avg_rw_err = sum(rw_error_list) / len(rw_error_list) if rw_error_list else 0
avg_lw_err = sum(lw_error_list) / len(lw_error_list) if lw_error_list else 0

print(f"\nAverage Right Wrist Error: {avg_rw_err:.2f} px")
print(f"Average Left Wrist Error:  {avg_lw_err:.2f} px")