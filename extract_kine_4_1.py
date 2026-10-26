import cv2
import mediapipe as mp
import os

''' 
VERSION 4.1 (Video Mode - Popup & Console Error Logging Only)
Target = Process an MP4 video file
Tracking with LEFT/RIGHT wrists and LEFT/RIGHT ankles with MediaPipe.
Preview with cv2 window and print missing detection errors to console.

10 26 2026
'''

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=2,           # Heavy model
    smooth_landmarks=True,        # Enable smoothing for video tracking
    min_detection_confidence=0.7, # Raise confidence to reduce false readings
    min_tracking_confidence=0.7   # Raise tracking confidence
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: Update this path to your target MP4 file when ready
video_path = r"YOUR VIDEO PATH YOUR VIDEO PATH" <--- YOUR PATH YOUR PATH YOUR PATH YOUR PATH VIDEO IS IN README

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing video: {video_path} ({fps} FPS, {total_frames} frames)")

frame_idx = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        frame_idx += 1
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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
        else:
            print(f"Warning: Frame {frame_idx} - Pose not detected!")

        cv2.imshow("Video Pose Tracking | Wrists & Ankles", frame)
        
        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete! Video window closed.")
