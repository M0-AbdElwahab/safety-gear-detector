"""Convert the trained PyTorch weights (best.pt) to ONNX.

Runs on Colab (needs ultralytics), not on a laptop. The resulting
best.onnx is what the FastAPI app loads with onnxruntime, so the API
does not need PyTorch at all.
"""
from ultralytics import YOLO

WEIGHTS = "/content/best.pt"


def main():
    model = YOLO(WEIGHTS)
    onnx_path = model.export(
        format="onnx",
        imgsz=640,       # fixed 640x640 input, same size used in training
        simplify=True,   # remove redundant operations from the graph
    )
    print("exported:", onnx_path)


if __name__ == "__main__":
    main()