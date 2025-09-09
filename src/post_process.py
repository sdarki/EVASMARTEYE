import numpy as np
import cv2

# # --- Helper for YOLO-style post-processing ---
# def yolo_postprocess(output, img_shape, conf_threshold=0.25):
#     """
#     Decode YOLO outputs from OpenVINO model into bounding boxes and class ids.
#     output: numpy array, shape [1, 6, N]
#     img_shape: (height, width, channels)
#     """
#     h, w, _ = img_shape

#     if len(output.shape) == 3:   # (1, 6, N)
#         output = output[0]       # (6, N)
#         output = output.T        # (N, 6)

#     boxes = []
#     class_ids = []
#     scores = []

#     for det in output:
#         x_center, y_center, width, height, conf, cls = det

#         if conf < conf_threshold:
#             continue

#         # Convert xywh -> x1y1x2y2 (normalized to image size)
#         x1 = int((x_center - width / 2) * w)
#         y1 = int((y_center - height / 2) * h)
#         x2 = int((x_center + width / 2) * w)
#         y2 = int((y_center + height / 2) * h)

#         boxes.append([x1, y1, x2, y2])
#         class_ids.append(int(cls))
#         scores.append(float(conf))

#     return boxes, class_ids, scores

def yolo_postprocess(output, img_shape, conf_threshold=0.25):
    h, w, _ = img_shape

    if len(output.shape) == 3:   # (1, 6, N) or (1, 85, N)
        output = output[0].T        # (N, 6) or (N, 85)

    boxes, class_ids, scores = [], [], []

    for det in output:
        x_center, y_center, width, height, obj_conf = det[:5]
        class_probs = det[5:]          # remaining entries
        cls_id = np.argmax(class_probs)
        conf = obj_conf * class_probs[cls_id]

        if conf < conf_threshold:
            continue

        # Convert xywh to x1y1x2y2
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        boxes.append([x1, y1, x2, y2])
        class_ids.append(cls_id)
        scores.append(float(conf))

    return boxes, class_ids, scores
