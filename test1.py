import cv2
import threading
import time
import os

cameras = [
    {'name': 'Camera_1', 'url': 'rtsp://admin:private123@192.168.3.26:554/cam/realmonitor?channel=1&subtype=0'},
    {'name': 'Camera_2', 'url': 'rtsp://admin:private123@192.168.3.16:554/cam/realmonitor?channel=1&subtype=0'},
    {'name': 'Camera_3', 'url': 'rtsp://admin:private123@192.168.3.14:554/cam/realmonitor?channel=1&subtype=0'},
    {'name': 'Camera_11', 'url': 'rtsp://admin:private123@192.168.3.18:554/cam/realmonitor?channel=1&subtype=0'},
    {'name': 'Camera_22', 'url': 'rtsp://admin:private123@192.168.3.15:554/cam/realmonitor?channel=1&subtype=0'},
]

output_dir = 'camera_snapshots'
os.makedirs(output_dir, exist_ok=True)

def capture_frames(camera):
    cam_name = camera['name']
    stream_url = camera['url']
    cam_dir = os.path.join(output_dir, cam_name)
    os.makedirs(cam_dir, exist_ok=True)

    frame_counter = 1

    print(f"[INFO] Starting capture for {cam_name}")

    while True:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"[ERROR] Cannot connect to {cam_name} at {stream_url}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        print(f"[INFO] Connected to {cam_name}. Starting frame capture.")

        last_saved_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Lost connection to {cam_name}. Reconnecting in 5 seconds...")
                cap.release()
                time.sleep(5)
                break

            current_time = time.time()

            # Only save one frame per second
            if current_time - last_saved_time >= 1.0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(cam_dir, f"{timestamp}_frame{frame_counter}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] {cam_name}: Saved frame {frame_counter} as {filename}")

                frame_counter += 1
                last_saved_time = current_time

            # No sleep here – keep reading frames fast and discard them if not time yet

        cap.release()


# Start threads
threads = []
for cam in cameras:
    t = threading.Thread(target=capture_frames, args=(cam,))
    t.daemon = True
    t.start()
    threads.append(t)

try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\n[INFO] Stopping all captures.")
