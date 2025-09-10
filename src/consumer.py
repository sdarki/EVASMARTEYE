import os, time, json, queue, cv2
from src.model_handler import ModelHandler

def consumer(frame_queue, cam_name, stats, lock, log_dir="reports", img_dir="output_frames"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(img_dir, cam_name), exist_ok=True)

    json_file = os.path.join(log_dir, f"{cam_name}_reports.json")
    img_output_dir = os.path.join(img_dir, cam_name)
    model_handler = ModelHandler()

    frame_count, duplicate_count, total_frames = 0, 0, 0
    total_processing_time, last_timestamp = 0.0, 0
    start_time = time.time()
    last_frame_hash = None
    min_time_diff = 0.8

    print(f"[{cam_name}] Consumer started")

    while True:
        try:
            frame = frame_queue.get(timeout=30)
            if frame is None:
                break

            current_timestamp = time.time()
            total_frames += 1

            # Skip near-duplicates
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

            last_frame_hash, last_timestamp = frame_hash, current_timestamp

            # Run inference
            process_start = time.time()
            results = model_handler.infer(frame)
            process_end = time.time()
            total_processing_time += (process_end - process_start)

            # Annotate & save
            annotated_frame = results[0].plot()
            cv2.imwrite(os.path.join(img_output_dir, f"{int(current_timestamp)}.jpg"), annotated_frame)

            # Count helmets
            boxes = results[0].boxes
            helmet_count = sum(boxes.cls == 0)
            no_helmet_count = sum(boxes.cls != 0)

            frame_count += 1
            frame_queue.task_done()

            # Per-second report
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
                print(f"[{cam_name}] Report: {report}")

                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        data = json.load(f)
                else:
                    data = []
                data.append(report)
                with open(json_file, "w") as f:
                    json.dump(data, f, indent=4)

                frame_count, duplicate_count, total_processing_time = 0, 0, 0.0
                start_time = current_timestamp

        except queue.Empty:
            print(f"[{cam_name}] No frames for 30s...")
            continue
        except Exception as e:
            print(f"[{cam_name}] Consumer error: {e}")
            break

    print(f"[{cam_name}] Consumer stopped")
