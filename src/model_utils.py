# import numpy as np
# import cv2
# from ultralytics import YOLO
# class ModelHandler:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)

#     def predict(self, frame, imgsz=(1280, 736), device="cpu"):
#         return self.model(frame, imgsz=imgsz, device=device)

# Initialize OpenVINO Core

import openvino as ov
import time

core = ov.Core()
print("Available devices:", core.available_devices)  # should show ['CPU', 'GPU'] if iGPU is usable
 
# Load and compile the OpenVINO model for GPU
model_path = "yolo11s_openvino_model/yolo11s.xml"
ov_model = core.read_model(model_path)
compiled_model = core.compile_model(ov_model, "GPU")

start_time = time.time()
for idx, frame in enumerate(frames):
    # Preprocess to NCHW FP32
    input_tensor = frame.transpose(2, 0, 1)[None].astype("float16")
    results = compiled_model([input_tensor])
    # process results here if needed
end_time = time.time()
 
elapsed = end_time - start_time
actual_fps = len(frames) / elapsed if elapsed > 0 else 0
print(f"Processed {len(frames)} frames in {elapsed:.2f} seconds ({actual_fps:.2f} FPS)")