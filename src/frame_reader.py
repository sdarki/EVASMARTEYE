import ffmpeg
import numpy as np

def ffmpeg_frame_reader(rtsp_url, width=1280, height=736, fps=1):
    process = (
        ffmpeg.input(rtsp_url, rtsp_transport="tcp")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", vf=f"fps={fps},scale={width}:{height}")
        .global_args("-fflags", "+discardcorrupt+nobuffer")
        .global_args("-flags", "+low_delay")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = width * height * 3
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
