import ffmpeg
import numpy as np
import queue
import threading
import time
import json
import os
import cv2
import openvino as ov
from src.post_process import yolo_postprocess

# --- Load OpenVINO model globally ---
core = ov.Core()
print("Available devices:", core.available_devices)  # e.g., ['CPU', 'GPU']
model_path = "./instance/best_openvino_model/best.xml"
ov_model = core.read_model(model_path)
compiled_model = core.compile_model(ov_model, "GPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

def ffmpeg_frame_reader(rtsp_url, width=1280, height=736, fps=1, cam_name="unknown"):
    log_dir = f"logs/{cam_name}"
    os.makedirs(log_dir, exist_ok=True)
    ffmpeg_log_path = os.path.join(log_dir, "ffmpeg_stderr.log")

    process = (
        ffmpeg.input(rtsp_url, rtsp_transport='tcp')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', vf=f'fps={fps},scale={width}:{height}')
        .global_args('-fflags', '+discardcorrupt+nobuffer')
        .global_args('-flags', '+low_delay')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = width * height * 3

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
        # --- Create a thread-local compiled model ---
    local_compiled_model = core.compile_model(ov_model, "GPU")
    local_input_layer = local_compiled_model.input(0)
    local_output_layer = local_compiled_model.output(0)
    frame_count = 0
    duplicate_count = 0
    total_frames = 0
    total_processing_time = 0.0
    start_time = time.time()
    last_frame_hash = None
    last_timestamp = 0
    min_time_diff = 0.8

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(img_dir, cam_name), exist_ok=True)
    json_file = os.path.join(log_dir, f"{cam_name}_reports.json")
    img_output_dir = os.path.join(img_dir, cam_name)

    print(f"[{cam_name}] Consumer started")

    while True:
        try:
            frame = frame_queue.get(timeout=30)
            if frame is None:
                break

            current_timestamp = time.time()
            total_frames += 1

            # --- Frame skipping ---
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

            # --- OpenVINO inference ---
            model_h, model_w = local_input_layer.shape[2], local_input_layer.shape[3]
            frame_resized = cv2.resize(frame, (model_w, model_h))
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            input_tensor = frame_resized.transpose(2, 0, 1)[None].astype("float16")  # NCHW FP16
            process_start = time.time()
            results = local_compiled_model([input_tensor])[local_output_layer]
            print("Results type:", type(results))
            print("Results shape:", np.array(results).shape)
            print("Sample column:", np.array(results).squeeze(0)[:, 0])
            process_end = time.time()
            total_processing_time += (process_end - process_start)

            # --- Post-processing and counting ---
            boxes, class_ids, scores = yolo_postprocess(np.array(results), frame.shape, conf_threshold=0.25)

            helmet_count = 0
            no_helmet_count = 0
            annotated_frame = frame.copy()

            # Get original and model sizes
            orig_h, orig_w = frame.shape[:2]
            model_h, model_w = frame_resized.shape[:2]
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h

            for box, cls, score in zip(boxes, class_ids, scores):
                x1, y1, x2, y2 = map(int, box)
                # Scale coordinates
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
                if int(cls) == 0:  # assuming class 0 = Helmet
                    helmet_count += 1
                    color = (0, 255, 0)
                    label = "Helmet"
                else:
                    no_helmet_count += 1
                    color = (0, 0, 255)
                    label = "No Helmet"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label,  (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # --- Save annotated frame ---


            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_filename = os.path.join(img_output_dir, f"{int(current_timestamp)}.jpg")
            cv2.imwrite(frame_filename, annotated_frame)
            frame_queue.task_done()
            frame_count += 1

            # --- Per-second report ---
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

                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data = json.load(f)
                else:
                    data = []

                data.append(report)
                with open(json_file, "w") as f:
                    json.dump(data, f, indent=4)

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
    stats = {}
    lock = threading.Lock()
    for cam in cameras:
        stats[cam.name] = {"frames_received": 0}
        rtsp_url = f"rtsp://admin:private123@{cam.ip}:554/cam/realmonitor?channel=1&subtype=0"
        frame_queue = queue.Queue(maxsize=10)
        threading.Thread(target=producer, args=(rtsp_url, frame_queue, stats, cam.name, lock), daemon=True).start()
        threading.Thread(target=consumer, args=(frame_queue, cam.name, stats, lock), daemon=True).start()
