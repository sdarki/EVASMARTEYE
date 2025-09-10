import cv2
from ultralytics import YOLO

class ModelHandler:
    def __init__(self, model_path="best_openvino_model/"):
        self.model = YOLO(model_path)
        print("Model loaded:", self.model)

    def infer(self, frame):
        # No color conversion needed if model expects BGR
        return self.model(frame, device="intel:gpu")
