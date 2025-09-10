import cv2

def cv2_frame_reader(rtsp_url, width=1280, height=736, fps=1):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Could not open stream {rtsp_url}")
        return

    frame_interval = int(cap.get(cv2.CAP_PROP_FPS) // fps) if fps > 0 else 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        frame = cv2.resize(frame, (width, height))
        yield frame

    cap.release()