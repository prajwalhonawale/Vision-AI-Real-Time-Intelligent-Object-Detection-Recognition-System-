import base64
import time
import requests
import cv2
import config

PROMPT = ("You are looking at a cropped photo of a single object. "
          "Identify what the object is in 2-4 words, then give a one-sentence "
          "description. Format exactly as:\nNAME: <short name>\nDESCRIPTION: <sentence>")

def _encode(bgr_image):
    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode crop")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

def describe_object(bgr_crop):
    payload = {
        "model": "moondream2",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": _encode(bgr_crop)}}
            ]
        }],
        "max_tokens": 120,
        "temperature": 0.2
    }
    t0 = time.time()
    try:
        r = requests.post(config.VLM_SERVER_URL, json=payload, timeout=config.VLM_TIMEOUT)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return {"name": "Unknown", "description": f"VLM error: {e}",
                "latency": round(time.time() - t0, 1)}

    name, description = "Unknown", content.strip()
    for line in content.splitlines():
        if line.upper().startswith("NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.upper().startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()

    return {"name": name, "description": description, "latency": round(time.time() - t0, 1)}
