import time
import threading
from ultralytics import YOLO
import config

class Detector:
    def __init__(self, model_path=config.YOLO_MODEL_PATH):
        self.model = YOLO(model_path, task="detect")
        self.results = []
        self.centered = None
        self.lock = threading.Lock()
        self.last_infer_time = 0.0

    def infer(self, frame):
        t0 = time.time()
        preds = self.model(frame, conf=config.CONF_THRESHOLD,
                            iou=config.IOU_THRESHOLD, verbose=False)[0]
        self.last_infer_time = time.time() - t0

        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        detections, best, best_dist = [], None, float("inf")

        for box in preds.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            name = self.model.names[int(box.cls[0])]
            det = {"box": (int(x1), int(y1), int(x2), int(y2)),
                   "class": name, "conf": conf}
            detections.append(det)

            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best = dist, det

        with self.lock:
            self.results, self.centered = detections, best
        return detections, best

    def get_latest(self):
        with self.lock:
            return list(self.results), self.centered
