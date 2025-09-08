import ffmpeg
import numpy as np
import queue
import threading
import time  # <-- Add this import
import json  # <-- Add this import
import os
import cv2
from ultralytics import YOLO
from .model_utils import ModelHandler
from collections import Counter

model_handler = None  # Global model handler

def ffmpeg_frame_reader(rtsp_url, width=1280, height=736, fps=1, cam_name="unknown"):
    log_dir = f"logs/{cam_name}"
    os.makedirs(log_dir, exist_ok=True)
    ffmpeg_log_path = os.path.join(log_dir, "ffmpeg_stderr.log")

    process = (
        ffmpeg
        .input(rtsp_url, rtsp_transport='tcp')
        .output(
            'pipe:',
            format='rawvideo',
            pix_fmt='rgb24',
            vf=f'fps={fps},scale={width}:{height}'
        )
        # .global_args('-hwaccel', 'cuda')
        .global_args('-fflags', '+discardcorrupt+nobuffer')
        .global_args('-flags', '+low_delay')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = width * height * 3

    # Thread to log FFmpeg stderr in real time
    def log_ffmpeg_stderr(stderr, log_path):
        with open(log_path, "ab") as log_file:
            while True:
                line = stderr.readline()
                if not line:
                    break
                log_file.write(line)
                log_file.flush()

    stderr_thread = threading.Thread(target=log_ffmpeg_stderr, args=(process.stderr, ffmpeg_log_path), daemon=True)
    stderr_thread.start()

    try:
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes or len(in_bytes) < frame_size:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            yield frame
    finally:
        process.stdout.close()
        process.stderr.close()
        process.wait()


def producer(rtsp_url, frame_queue, stats, cam_name, lock):
    while True:
        try:
            for frame in ffmpeg_frame_reader(rtsp_url, cam_name=cam_name):
                if not frame_queue.full():
                    frame_queue.put(frame)
                    with lock:
                        stats[cam_name]["frames_received"] += 1
            print(f"[{cam_name}] FFmpeg stopped producing frames, restarting in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"[{cam_name}] Producer error: {e}, restarting in 5 seconds...")
            time.sleep(5)

def consumer(frame_queue, cam_name, stats, lock, log_dir="reports", img_dir="output_frames"):
    """
    Consumer function that processes frames, saves per-second representative results,
    and stores sampled frames for validation (no mode logic).
    """
    global model_handler
    frame_count = 0
    duplicate_count = 0
    total_frames = 0
    total_processing_time = 0.0
    start_time = time.time()

    last_frame_hash = None
    last_timestamp = 0
    min_time_diff = 0.8  # seconds

    # For saving logs/images
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(img_dir, cam_name), exist_ok=True)

    json_file = os.path.join(log_dir, f"{cam_name}_reports.json")
    img_output_dir = os.path.join(img_dir, cam_name)

    print(f"[{cam_name}] Consumer started")

    save_every_n_frames = 10  # save 1 out of every 10 frames

    while True:
        try:
            frame = frame_queue.get(timeout=30)
            if frame is None:
                break

            current_timestamp = time.time()
            total_frames += 1

            # --- Frame skipping logic ---
            if current_timestamp - last_timestamp < min_time_diff:
                duplicate_count += 1
                frame_queue.task_done()
                continue

            frame_sample = frame[::20, ::20].tobytes()
            frame_hash = hash(frame_sample)
            if frame_hash == last_frame_hash:
                duplicate_count += 1
                frame_queue.task_done()
                continue

            last_frame_hash = frame_hash
            last_timestamp = current_timestamp

            # --- Inference ---
            process_start = time.time()
            results = model_handler.predict(frame, imgsz=(1280, 736))
            process_end = time.time()
            total_processing_time += (process_end - process_start)

            helmet_count = 0
            no_helmet_count = 0
            annotated_frame = frame.copy()

            for r in results:
                if hasattr(r, "boxes"):
                    for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                        x1, y1, x2, y2 = map(int, box)
                        if int(cls) == 0:
                            helmet_count += 1
                            color = (0, 255, 0)
                            label = "Helmet"
                        else:
                            no_helmet_count += 1
                            color = (0, 0, 255)
                            label = "No Helmet"

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            img_name = f"{cam_name}_{int(current_timestamp)}.jpg"
            cv2.imwrite(os.path.join(img_output_dir, img_name),
                       cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            frame_queue.task_done()

            # ---- Every second → save report ----
            if int(current_timestamp) != int(start_time):
                report = {
                    "camera": cam_name,
                    "timestamp": int(current_timestamp),
                    "helmet:no_helmet": f"{helmet_count}:{no_helmet_count}",
                    "frames_processed_this_second": frame_count,
                    "duplicates_skipped": duplicate_count,
                    "avg_processing_time_sec": round(total_processing_time / max(frame_count, 1), 4),
                    "total_frames": total_frames
                }

                print(f"[{cam_name}] Per-second Report: {report}")

                # Append JSON
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data = json.load(f)
                else:
                    data = []

                data.append(report)
                with open(json_file, "w") as f:
                    json.dump(data, f, indent=4)

                # Reset counters for next second
                frame_count = 0
                duplicate_count = 0
                total_processing_time = 0.0
                start_time = current_timestamp

        except queue.Empty:
            print(f"[{cam_name}] No frames received for 30 seconds...")
            continue
        except Exception as e:
            print(f"[{cam_name}] Consumer error: {e}")
            break

    print(f"[{cam_name}] Consumer stopped")


def run_detection(cameras):
    global model_handler
    model_handler = ModelHandler("./instance/best_openvino_model/")  # Load once globally

    stats = {}
    lock = threading.Lock()
    for cam in cameras:
        stats[cam.name] = {"frames_received": 0}
        rtsp_url = f"rtsp://admin:private123@{cam.ip}:554/cam/realmonitor?channel=1&subtype=0"
        frame_queue = queue.Queue(maxsize=10)
        threading.Thread(target=producer, args=(rtsp_url, frame_queue, stats, cam.name, lock), daemon=True).start()
        threading.Thread(target=consumer, args=(frame_queue, cam.name, stats, lock), daemon=True).start()