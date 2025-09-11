import cv2
from ultralytics import YOLO

class ModelHandler:
    _model = None  # shared global model instance

    def __init__(self, model_path="best_openvino_model/"):
        if ModelHandler._model is None:
            print(f"[ModelHandler] Loading model from {model_path} ...")
            ModelHandler._model = YOLO(model_path)
            print("[ModelHandler] Model loaded successfully")
        self.model = ModelHandler._model

    def infer(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.model(frame_rgb, device="intel:gpu")
