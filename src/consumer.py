import os, time, json, queue, cv2
from src.model_handler import ModelHandler

# Global model handler instance
global_model_handler = None

def consumer(frame_queue, cam_name, stats, lock, log_dir="reports", img_dir="output_frames"):
    global global_model_handler
    if global_model_handler is None:
        global_model_handler = ModelHandler()
    model_handler = global_model_handler

    total_frames = 0
    total_infer_time = 0.0

    print(f"[{cam_name}] Consumer started (inference only, no saving)")

    while True:
        try:
            frame, cam_name_actual = frame_queue.get(timeout=30)  # Unpack tuple
            if frame is None:
                break

            t0 = time.time()
            _ = model_handler.infer(frame)
            t1 = time.time()

            infer_time = (t1 - t0) * 1000  # ms
            total_infer_time += infer_time
            total_frames += 1

            print(f"[{cam_name_actual}] Inference time: {infer_time:.1f} ms (frame {total_frames})")

            frame_queue.task_done()

        except queue.Empty:
            print(f"[{cam_name}] No frames for 30s...")
            continue
        except Exception as e:
            print(f"[{cam_name}] Consumer error: {e}")
            break

    print(f"[{cam_name}] Consumer stopped. Avg inference: {total_infer_time/total_frames if total_frames else 0:.1f} ms")
