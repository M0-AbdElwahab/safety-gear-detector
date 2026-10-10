"""FastAPI app: detect helmets / vests / people with the exported ONNX model.

Runs inside Docker (needs onnxruntime, not PyTorch).
Endpoints:
  GET  /health         -> is the server up and which model is loaded
  POST /predict        -> upload an image, get detections as JSON
  POST /predict/image  -> upload an image, get the same image back with boxes drawn
"""
import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageOps

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).parent / "models" / "best.onnx"))
CLASS_NAMES = ["boots", "helmet", "nohelmet", "novest", "person", "vest"]  # same order as data.yaml
VIOLATION_CLASSES = {"nohelmet", "novest"}
INPUT_SIZE = 640          # the model was exported with a fixed 640x640 input
CONF_THRESHOLD = 0.25     # ignore detections the model is less sure about
IOU_THRESHOLD = 0.7       # boxes overlapping more than this are treated as duplicates
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Load the model once when the server starts, not on every request
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

app = FastAPI(title="Safety Gear Detector")


def preprocess(image):
    """Resize to 640x640 without stretching (add grey padding), scale to 0-1, shape (1,3,640,640)."""
    w, h = image.size
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    pad_x = (INPUT_SIZE - new_w) // 2
    pad_y = (INPUT_SIZE - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
    return arr, scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold):
    """Non-max suppression: keep the best box, drop boxes that overlap it too much, repeat."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def postprocess(output, scale, pad_x, pad_y, orig_w, orig_h):
    """Turn the raw model output (1, 10, 8400) into a clean list of detections."""
    preds = output[0].T                      # (8400, 10): 4 box numbers + 6 class scores
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    mask = confidences >= CONF_THRESHOLD
    preds, class_ids, confidences = preds[mask], class_ids[mask], confidences[mask]
    if len(preds) == 0:
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # Shifting boxes by class makes NMS compare boxes of the same class only
    keep = nms(boxes + class_ids[:, None] * 4096.0, confidences, IOU_THRESHOLD)

    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        # Undo the padding and resizing so the box fits the ORIGINAL image
        x1 = float(np.clip((x1 - pad_x) / scale, 0, orig_w))
        x2 = float(np.clip((x2 - pad_x) / scale, 0, orig_w))
        y1 = float(np.clip((y1 - pad_y) / scale, 0, orig_h))
        y2 = float(np.clip((y2 - pad_y) / scale, 0, orig_h))
        detections.append({
            "class": CLASS_NAMES[int(class_ids[i])],
            "confidence": round(float(confidences[i]), 3),
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    return detections


def detect(file):
    """Read the upload, validate it, run the model. Returns (image, detections)."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")   # fixes rotated phone photos
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    arr, scale, pad_x, pad_y = preprocess(image)
    output = session.run(None, {input_name: arr})[0]
    detections = postprocess(output, scale, pad_x, pad_y, image.width, image.height)
    return image, detections


@app.get("/health")
def health():
    return {"status": "ok", "model": Path(MODEL_PATH).name, "classes": CLASS_NAMES}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image, detections = detect(file)
    return {
        "image_width": image.width,
        "image_height": image.height,
        "detections": detections,
        "violation_found": any(d["class"] in VIOLATION_CLASSES for d in detections),
    }


@app.post("/predict/image")
def predict_image(file: UploadFile = File(...)):
    image, detections = detect(file)
    draw = ImageDraw.Draw(image)
    for d in detections:
        color = "red" if d["class"] in VIOLATION_CLASSES else "lime"
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 3, y1 + 3), f'{d["class"]} {d["confidence"]:.2f}', fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")