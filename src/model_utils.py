import numpy as np
import cv2
from ultralytics import YOLO
class ModelHandler:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, frame, imgsz=(1280, 736), device="cpu"):
        return self.model(frame, imgsz=imgsz, device=device)