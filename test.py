from ultralytics import YOLO
import cv2
import time
 
# Load the exported OpenVINO model
ov_model = YOLO("best_openvino_model/")
 
# Open the video file
cap = cv2.VideoCapture("clip.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
 
frame_count = 0
start_time = time.time() 
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Run inference (you can specify device="intel:gpu" if needed)
    _ = ov_model(frame, device="intel:gpu")
    frame_count += 1
 
end_time = time.time()
elapsed = end_time - start_time
avg_fps = frame_count / elapsed if elapsed > 0 else 0
 
print(f"Processed {frame_count} frames in {elapsed:.2f} seconds.")
print(f"Average inference FPS: {avg_fps:.2f}")
 
cap.release()