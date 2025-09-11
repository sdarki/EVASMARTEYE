from src.frame_reader import ffmpeg_frame_reader
import queue

def producer(rtsp_url, frame_queue, stats, cam_name, lock):
    print(f"[{cam_name}] Producer started")
    for frame in ffmpeg_frame_reader(rtsp_url):
        with lock:
            stats[cam_name]["frames_received"] += 1
        try:
            # Always keep only the freshest frame
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # discard oldest frame
                except queue.Empty:
                    pass
            frame_queue.put_nowait(frame)
        except:
            print(f"[{cam_name}] Frame queue error, dropping frame...")
    frame_queue.put(None)  # signal consumer to stop
    print(f"[{cam_name}] Producer stopped")
