import time
from src.frame_reader import cv2_frame_reader

def producer(rtsp_url, frame_queue, stats, cam_name, lock):
    while True:
        try:
            for frame in cv2_frame_reader(rtsp_url):
                if not frame_queue.full():
                    frame_queue.put((frame, cam_name))
                    with lock:
                        stats[cam_name]["frames_received"] += 1
            print(f"[{cam_name}] Stream ended, restarting in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"[{cam_name}] Producer error: {e}, restarting in 5s...")
            time.sleep(5)
