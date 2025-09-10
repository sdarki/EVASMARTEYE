import cv2
from ultralytics import YOLO

class ModelHandler:
    def __init__(self, model_path="best_openvino_model/"):
        self.model = YOLO(model_path)
        print("Model loaded:", self.model)

    def infer(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.model(frame_rgb, device="intel:gpu")
