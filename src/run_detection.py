import threading, queue
from src.producer import producer
from src.consumer import consumer

def run_detection(cameras):
    stats = {}
    lock = threading.Lock()
    for cam in cameras:
        stats[cam.name] = {"frames_received": 0}
        rtsp_url = f"rtsp://admin:private123@{cam.ip}:554/cam/realmonitor?channel=1&subtype=0"
        frame_queue = queue.Queue(maxsize=10)
        threading.Thread(target=producer, args=(rtsp_url, frame_queue, stats, cam.name, lock), daemon=True).start()
        threading.Thread(target=consumer, args=(frame_queue, cam.name, stats, lock), daemon=True).start()
