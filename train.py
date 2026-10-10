"""Train YOLOv8s on the PPE_master dataset (6 classes).

Runs on a Colab T4 GPU, not on a laptop. These are the exact settings
used for the run reported in the README.
"""
from ultralytics import YOLO

DATA_YAML = "/content/ppe_data/data_fixed.yaml"   # written in Cell 6
PROJECT_DIR = "/content/drive/MyDrive/ppe_project/runs"


def main():
    # Start from COCO-pretrained weights (transfer learning), then fine-tune
    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA_YAML,
        epochs=60,       # at most 60 passes over the training set
        patience=15,     # stop early if validation stalls for 15 epochs
        imgsz=640,       # images are resized to 640x640
        batch=16,        # 16 images per step (fits a T4)
        workers=2,       # Colab only has 2 CPUs
        seed=0,          # makes the run more repeatable
        project=PROJECT_DIR,
        name="yolov8s_60ep",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()