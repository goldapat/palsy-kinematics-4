import os
import cv2

# Configuration
input_dir = r"E:\GOLD\school\Project Cerebral palsy\cp_screening_project\MINI-RGBD_web\01\rgb"
output_video_path = r"E:\GOLD\school\Project Cerebral palsy\cp_screening_project\rgb_output.mp4"
fps = 30

# Read the first frame to get exact dimensions
first_frame_path = os.path.join(input_dir, "syn_00000.png")
first_frame = cv2.imread(first_frame_path)

if first_frame is None:
    raise FileNotFoundError(f"Could not read the first frame at: {first_frame_path}")

height, width, _ = first_frame.shape

# Initialize VideoWriter (mp4v codec works universally with .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing frames from syn_00000.png to syn_00999.png...")

# Loop through syn_00000 to syn_00999
for i in range(1000):
    filename = f"syn_{i:05d}.png"
    file_path = os.path.join(input_dir, filename)
    
    frame = cv2.imread(file_path)
    if frame is not None:
        out.write(frame)
    else:
        print(f"Warning: Missing or unreadable frame {filename}, skipping.")

out.release()
print(f"Successfully saved video to: {output_video_path}")