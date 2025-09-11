import os, time, json, queue, cv2
from collections import Counter
from src.model_handler import ModelHandler

# Load model globally once
global_model_handler = ModelHandler()

def consumer(frame_queue, cam_name, stats, lock, length, log_dir="reports", img_dir="output_frames"):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(img_dir, cam_name), exist_ok=True)

    json_file = os.path.join(log_dir, f"{cam_name}_reports.json")
    img_output_dir = os.path.join(img_dir, cam_name)

    frame_count, duplicate_count, total_frames = 0, 0, 0
    total_processing_time, last_timestamp = 0.0, 0
    start_time = time.time()
    last_frame_hash = None
    min_time_diff = 0.8

    # --- Mode Buffers ---
    frame_buffer = []        # stores last 10 frames
    hash_buffer = []         # stores their hashes
    mode_buffer = []         # stores selected 6 mode frames

    print(f"[{cam_name}] Consumer started (Total cameras: {length})")

    # --- For 10-second reports ---
    ten_sec_reports = []
    ten_sec_start = time.time()
    log_dir_10s = os.path.join(log_dir, "10s_reports")
    os.makedirs(log_dir_10s, exist_ok=True)
    json_file_10s = os.path.join(log_dir_10s, f"{cam_name}_10s_reports.json")

    # --- For 1-minute reports ---
    one_min_modes = []
    log_dir_1min = os.path.join(log_dir, "1min_reports")
    os.makedirs(log_dir_1min, exist_ok=True)
    json_file_1min = os.path.join(log_dir_1min, f"{cam_name}_1min_reports.json")

    # --- For 30-minute reports---
    thirty_min_modes = []
    log_dir_30min = os.path.join(log_dir, "30min_reports")
    os.makedirs(log_dir_30min, exist_ok=True)
    json_file_30min = os.path.join(log_dir_30min, f"{cam_name}_30min_reports.json")

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
            results = global_model_handler.infer(frame)
            process_end = time.time()
            total_processing_time += (process_end - process_start)

            # Count helmets
            boxes = results[0].boxes
            helmet_count = int(sum(boxes.cls == 0))
            no_helmet_count = int(sum(boxes.cls != 0))

            frame_count += 1
            frame_queue.task_done()

            # --- Collect frames for mode selection ---
            frame_buffer.append(frame)
            hash_buffer.append(frame_hash)

            # When we reach 10 frames, pick mode
            if len(frame_buffer) == 10:
                # Find most frequent frame hash
                freq = Counter(hash_buffer)
                mode_hash, _ = freq.most_common(1)[0]
                mode_index = hash_buffer.index(mode_hash)
                mode_frame = frame_buffer[mode_index]

                # Save for second-level mode
                mode_buffer.append(mode_frame)

                # Reset buffers
                frame_buffer.clear()
                hash_buffer.clear()

                # When we reach 6 mode frames, pick final mode
                if len(mode_buffer) == 6:
                    # Use same hash approach
                    second_hashes = [hash(f[::20, ::20].tobytes()) for f in mode_buffer]
                    freq2 = Counter(second_hashes)
                    final_hash, _ = freq2.most_common(1)[0]
                    final_index = second_hashes.index(final_hash)
                    final_frame = mode_buffer[final_index]

                    # Annotate & Save final selected frame
                    annotated_frame = results[0].plot()
                    save_path = os.path.join(img_output_dir, f"final_{int(current_timestamp)}.jpg")
                    cv2.imwrite(save_path, annotated_frame)
                    print(f"[{cam_name}] Final Mode Frame Saved: {save_path}")

                    # Reset second-level buffer
                    mode_buffer.clear()

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
                
                # --- Append to 10s buffer ---
                ten_sec_reports.append(report["helmet:no_helmet"])
                
                # Check if 10s completed
                if time.time() - ten_sec_start >= 10:
                    # Find mode of helmet:no_helmet
                    from collections import Counter
                    mode_val, _ = Counter(ten_sec_reports).most_common(1)[0]
                    report_10s = {
                    "camera": cam_name,
                    "timestamp": int(ten_sec_start),
                    "helmet:no_helmet": ten_sec_reports.copy(),
                    "mode_frame": mode_val
                    }

                    # Save 10s report JSON
                    if os.path.exists(json_file_10s):
                        with open(json_file_10s, "r") as f:
                            data_10s = json.load(f)
                    else:
                        data_10s = []
                    data_10s.append(report_10s)
                    with open(json_file_10s, "w") as f:
                        json.dump(data_10s, f, indent=4)
                    print(f"[{cam_name}] 10s Report: {report_10s}")

                    # --- Append mode frame for 1-minute report ---
                    one_min_modes.append(mode_val)
                    if len(one_min_modes) == 6:
                        from collections import Counter
                        final_mode, _ = Counter(one_min_modes).most_common(1)[0]
                        report_1min = {
                            "camera": cam_name,
                            "timestamp": int(current_timestamp),
                            "mode_frames_1min": one_min_modes.copy(),
                            "final_mode_frame": final_mode
                        }

                        # Save 1-min report JSON
                        if os.path.exists(json_file_1min):
                            with open(json_file_1min, "r") as f:
                                data_1min = json.load(f)
                        else:
                            data_1min = []
                        data_1min.append(report_1min)
                        with open(json_file_1min, "w") as f:
                            json.dump(data_1min, f, indent=4)
                        print(f"[{cam_name}] 1-Min Report: {report_1min}")

                        # --- Append mode frame for 30-minute report ---
                        thirty_min_modes.append(final_mode)
                        if len(thirty_min_modes) == 30:
                            from collections import Counter
                            final_30min_mode, _ = Counter(thirty_min_modes).most_common(1)[0]
                            report_30min = {
                                "camera": cam_name,
                                "timestamp": int(current_timestamp),
                                "mode_frames_30min": thirty_min_modes.copy(),
                                "final_mode_frame_30min": final_30min_mode
                            }

                            # Save 30-min report JSON
                            if os.path.exists(json_file_30min):
                                with open(json_file_30min, "r") as f:
                                    data_30min = json.load(f)
                            else:
                                data_30min = []
                            data_30min.append(report_30min)
                            with open(json_file_30min, "w") as f:
                                json.dump(data_30min, f, indent=4)
                            print(f"[{cam_name}] 30-Min Report: {report_30min}")

                            # Reset 30-min buffer
                            thirty_min_modes.clear()
                            
                        # Reset 1-min buffer
                        one_min_modes.clear()
                    
                    # Reset 10s buffer
                    ten_sec_reports.clear()
                    ten_sec_start = time.time()

                # Reset per-second counters
                frame_count, duplicate_count, total_processing_time = 0, 0, 0.0
                start_time = current_timestamp

        except queue.Empty:
            print(f"[{cam_name}] No frames for 30s...")
            continue
        except Exception as e:
            print(f"[{cam_name}] Consumer error: {e}")
            break

    print(f"[{cam_name}] Consumer stopped")
