import ffmpeg
import numpy as np
import threading
import time
import os
import select

cameras = [
    {'name': 'Camera_1', 'url': 'rtsp://admin:private123@192.168.3.26:554/cam/realmonitor?channel=1&subtype=0'},
    {'name': 'Camera_2', 'url': 'rtsp://admin:private123@192.168.3.16:554/cam/realmonitor?channel=1&subtype=0'},
    {'name': 'Camera_3', 'url': 'rtsp://admin:private123@192.168.3.14:554/cam/realmonitor?channel=1&subtype=0'},
    {'name': 'Camera_11', 'url': 'rtsp://admin:private123@192.168.3.18:554/cam/realmonitor?channel=1&subtype=0'},
    {'name': 'Camera_22', 'url': 'rtsp://admin:private123@192.168.3.15:554/cam/realmonitor?channel=1&subtype=0'},
]

output_dir = 'camera_snapshots'
os.makedirs(output_dir, exist_ok=True)

def ffmpeg_frame_reader(rtsp_url, width=1280, height=736, fps=1):
    frame_size = width * height * 3

    while True:
        process = (
            ffmpeg.input(rtsp_url, rtsp_transport="tcp")
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", vf=f"fps={fps},scale={width}:{height}")
            .global_args("-fflags", "+discardcorrupt+nobuffer")
            .global_args("-flags", "+low_delay")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        try:
            while True:
                # Use select with 10s timeout
                rlist, _, _ = select.select([process.stdout], [], [], 10)

                if rlist:
                    in_bytes = process.stdout.read(frame_size)

                    if not in_bytes or len(in_bytes) < frame_size:
                        print("[WARNING] Incomplete frame received, reconnecting in 5s...")
                        break

                    frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                    yield frame
                else:
                    print("[WARNING] No data received for 10s, reconnecting...")
                    break  # No data for 10s – assume hang or disconnect

        except Exception as e:
            print(f"[ERROR] ffmpeg exception: {e}")

        finally:
            process.stdout.close()
            process.stderr.close()
            process.wait()

        time.sleep(5)  # Wait before reconnecting

def capture_frames(camera):
    cam_name = camera['name']
    stream_url = camera['url']
    cam_dir = os.path.join(output_dir, cam_name)
    os.makedirs(cam_dir, exist_ok=True)

    frame_counter = 1
    print(f"[INFO] Starting capture for {cam_name}")

    for frame in ffmpeg_frame_reader(stream_url):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(cam_dir, f"{timestamp}_frame{frame_counter}.jpg")
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(filename)

        print(f"[INFO] {cam_name}: Saved frame {frame_counter} as {filename}")

        frame_counter += 1
        time.sleep(1)  # Save roughly 1 frame per second


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
