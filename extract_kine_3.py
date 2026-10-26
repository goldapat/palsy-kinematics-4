import cv2
import mediapipe as mp
import glob
import os
import math
import csv
import re

''' 
VERSION 3
เป้าหมาย = ดู MINI-RGBD file 01 
Tracking with LEFT/RIGHT wrists (joint 20, 21) and LEFT/RIGHT ankles (joint 7, 8) with mediapipe.
Preview with cv2 window แล้วเทียบกับค่าจริง (ground truth)

September 7 2026
check line 147

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
ra_error_list = []
la_error_list = []

with open(csv_output_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "filename", 
        "mp_rw_x", "mp_rw_y", "gt_rw_x", "gt_rw_y", "rw_error_px",
        "mp_lw_x", "mp_lw_y", "gt_lw_x", "gt_lw_y", "lw_error_px",
        "mp_ra_x", "mp_ra_y", "gt_ra_x", "gt_ra_y", "ra_error_px",
        "mp_la_x", "mp_la_y", "gt_la_x", "gt_la_y", "la_error_px",
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
        
        # MINI-RGBD indices: Joint 21 = R Wrist, 20 = L Wrist, 8 = R Ankle, 7 = L Ankle
        gt_rw_x, gt_rw_y = get_gt_coords_from_file(gt_file_path, joint_id=21)
        gt_lw_x, gt_lw_y = get_gt_coords_from_file(gt_file_path, joint_id=20)
        gt_ra_x, gt_ra_y = get_gt_coords_from_file(gt_file_path, joint_id=8)
        gt_la_x, gt_la_y = get_gt_coords_from_file(gt_file_path, joint_id=7)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # --- Draw Ground Truth Points (RED) ---
        for gt_x, gt_y, label in [(gt_rw_x, gt_rw_y, "GT_RW"), 
                                  (gt_lw_x, gt_lw_y, "GT_LW"),
                                  (gt_ra_x, gt_ra_y, "GT_RA"),
                                  (gt_la_x, gt_la_y, "GT_LA")]:
            if gt_x is not None and gt_y is not None:
                cv2.circle(frame, (int(gt_x), int(gt_y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, label, (int(gt_x) + 5, int(gt_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # --- Process MediaPipe Landmarks ---
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract Wrists
            rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            mp_rw_x, mp_rw_y = int(rw.x * w), int(rw.y * h)
            cv2.circle(frame, (mp_rw_x, mp_rw_y), 5, (0, 255, 0), -1)

            lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            mp_lw_x, mp_lw_y = int(lw.x * w), int(lw.y * h)
            cv2.circle(frame, (mp_lw_x, mp_lw_y), 5, (255, 0, 0), -1)

            # Extract Ankles
            ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            mp_ra_x, mp_ra_y = int(ra.x * w), int(ra.y * h)
            cv2.circle(frame, (mp_ra_x, mp_ra_y), 5, (0, 255, 255), -1)

            la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            mp_la_x, mp_la_y = int(la.x * w), int(la.y * h)
            cv2.circle(frame, (mp_la_x, mp_la_y), 5, (255, 255, 0), -1)

            # Error calculations
            def calc_error(mp_x, mp_y, gt_x, gt_y, error_list):
                if gt_x is not None and gt_y is not None:
                    err = math.sqrt((mp_x - gt_x)**2 + (mp_y - gt_y)**2)
                    error_list.append(err)
                    cv2.line(frame, (int(gt_x), int(gt_y)), (mp_x, mp_y), (255, 255, 0), 1)
                    return err
                return None

            rw_error = calc_error(mp_rw_x, mp_rw_y, gt_rw_x, gt_rw_y, rw_error_list)
            lw_error = calc_error(mp_lw_x, mp_lw_y, gt_lw_x, gt_lw_y, lw_error_list)
            ra_error = calc_error(mp_ra_x, mp_ra_y, gt_ra_x, gt_ra_y, ra_error_list)
            la_error = calc_error(mp_la_x, mp_la_y, gt_la_x, gt_la_y, la_error_list)

            # Write data to CSV
            writer.writerow([
                filename, 
                mp_rw_x, mp_rw_y, round(gt_rw_x, 2) if gt_rw_x else None, round(gt_rw_y, 2) if gt_rw_y else None, round(rw_error, 2) if rw_error else None,
                mp_lw_x, mp_lw_y, round(gt_lw_x, 2) if gt_lw_x else None, round(gt_lw_y, 2) if gt_lw_y else None, round(lw_error, 2) if lw_error else None,
                mp_ra_x, mp_ra_y, round(gt_ra_x, 2) if gt_ra_x else None, round(gt_ra_y, 2) if gt_ra_y else None, round(ra_error, 2) if ra_error else None,
                mp_la_x, mp_la_y, round(gt_la_x, 2) if gt_la_x else None, round(gt_la_y, 2) if gt_la_y else None, round(la_error, 2) if la_error else None,
                True
            ])
        else:
            writer.writerow([
                filename, 
                None, None, round(gt_rw_x, 2) if gt_rw_x else None, round(gt_rw_y, 2) if gt_rw_y else None, None,
                None, None, round(gt_lw_x, 2) if gt_lw_x else None, round(gt_lw_y, 2) if gt_lw_y else None, None,
                None, None, round(gt_ra_x, 2) if gt_ra_x else None, round(gt_ra_y, 2) if gt_ra_y else None, None,
                None, None, round(gt_la_x, 2) if gt_la_x else None, round(gt_la_y, 2) if gt_la_y else None, None,
                False
            ])

        cv2.imshow("MINI-RGBD Validation | Wrists & Ankles", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # cv2.waitKey(30) = Default cv2.waitKey(1) = Fast // Press Q to stop
            break

cv2.destroyAllWindows()

# Print statistics
avg_rw_err = sum(rw_error_list) / len(rw_error_list) if rw_error_list else 0
avg_lw_err = sum(lw_error_list) / len(lw_error_list) if lw_error_list else 0
avg_ra_err = sum(ra_error_list) / len(ra_error_list) if ra_error_list else 0
avg_la_err = sum(la_error_list) / len(la_error_list) if la_error_list else 0

print(f"\nAverage Right Wrist Error: {avg_rw_err:.2f} px")
print(f"Average Left Wrist Error:  {avg_lw_err:.2f} px")
print(f"Average Right Ankle Error: {avg_ra_err:.2f} px")
print(f"Average Left Ankle Error:  {avg_la_err:.2f} px")