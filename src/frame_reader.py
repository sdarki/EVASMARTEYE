import ffmpeg
import numpy as np
import time
import select

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
                # Add select with 10-second timeout
                rlist, _, _ = select.select([process.stdout], [], [], 10)

                if rlist:
                    in_bytes = process.stdout.read(frame_size)

                    if not in_bytes or len(in_bytes) < frame_size:
                        print("[WARNING] Incomplete frame received, reconnecting in 5s...")
                        break  # Reconnect

                    frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                    yield frame
                else:
                    print("[WARNING] No frame data for 10s, reconnecting...")
                    break  # Reconnect

        except Exception as e:
            print(f"[ERROR] ffmpeg exception: {e}")

        finally:
            process.stdout.close()
            process.stderr.close()
            process.wait()

        time.sleep(5)  # Wait before reconnecting
