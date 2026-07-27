import threading, time
import cv2
from flask import Flask, Response, jsonify, render_template

import config
from camera import CameraStream
from detector import Detector
import recognizer

app = Flask(__name__)
camera = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH,
                       config.FRAME_HEIGHT, config.FRAME_RATE).start()
detector = Detector()
state = {"fps": 0.0}

def detection_loop():
    n, t0 = 0, time.time()
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01); continue
        n += 1
        if n % config.DETECT_EVERY_N_FRAMES == 0:
            detector.infer(frame)
        if time.time() - t0 >= 1.0:
            state["fps"] = n / (time.time() - t0)
            n, t0 = 0, time.time()
        time.sleep(0.001)

threading.Thread(target=detection_loop, daemon=True).start()

def draw_overlay(frame):
    detections, centered = detector.get_latest()
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        is_center = centered is not None and det is centered
        color = (0, 165, 255) if is_center else (60, 200, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_center else 2)
        cv2.putText(frame, f'{det["class"]} {det["conf"]*100:.0f}%',
                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    cv2.putText(frame, f'FPS: {state["fps"]:.1f}', (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def mjpeg_generator():
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01); continue
        frame = draw_overlay(frame)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/describe", methods=["POST"])
def describe():
    frame = camera.read()
    if frame is None:
        return jsonify({"error": "Camera not ready"}), 400

    _, centered = detector.get_latest()
    h, w = frame.shape[:2]

    if centered is not None:
        x1, y1, x2, y2 = centered["box"]
        pad = 15
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        yolo_class, yolo_conf = centered["class"], round(centered["conf"] * 100, 1)
    else:
        # YOLO didn't recognize anything here — it only knows 80 COCO classes.
        # Fall back to a generous crop around the center so the VLM can still
        # identify things like a PCB, oscilloscope, motor driver, etc.
        cw, ch = int(w * 0.5), int(h * 0.5)
        cx, cy = w // 2, h // 2
        x1, y1 = max(0, cx - cw // 2), max(0, cy - ch // 2)
        x2, y2 = min(w, cx + cw // 2), min(h, cy + ch // 2)
        yolo_class, yolo_conf = None, None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return jsonify({"error": "Empty crop"}), 400

    result = recognizer.describe_object(crop)
    result["yolo_class"], result["yolo_conf"] = yolo_class, yolo_conf
    return jsonify(result)

if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, threaded=True)
