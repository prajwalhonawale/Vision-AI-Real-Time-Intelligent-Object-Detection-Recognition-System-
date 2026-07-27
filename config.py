import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30

# Detection
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo", "yolo11n_ncnn_model")
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
DETECT_EVERY_N_FRAMES = 2  # skip frames to save CPU

# VLM
VLM_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
VLM_TIMEOUT = 90  # Pi 4B is slow, give it room

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
