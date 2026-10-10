"""Evaluate the trained model on a dataset split and print a results table.

Runs on Colab (needs ultralytics and the downloaded dataset), not on a laptop.
Usage:  python evaluate.py test     (default)
        python evaluate.py val
"""
import sys
from ultralytics import YOLO

WEIGHTS = "/content/best.pt"
DATA_YAML = "/content/ppe_data/data_fixed.yaml"
OUT_DIR = "/content/drive/MyDrive/ppe_project/final"


def main(split="test"):
    model = YOLO(WEIGHTS)
    metrics = model.val(
        data=DATA_YAML,
        split=split,     # "test" = images never used for training or model selection
        imgsz=640,
        batch=16,
        workers=2,
        plots=True,      # also saves the confusion matrix and curves
        project=OUT_DIR,
        name=f"{split}_eval",
        exist_ok=True,
    )

    box = metrics.box
    print(f"\n| Class | Precision | Recall | mAP50 | mAP50-95 |")
    print("|---|---|---|---|---|")
    print(f"| all | {box.mp:.3f} | {box.mr:.3f} | {box.map50:.3f} | {box.map:.3f} |")
    # box.p, box.r, box.ap50, box.ap hold one value per class that has labels
    for row, class_id in enumerate(box.ap_class_index):
        name = metrics.names[class_id]
        print(f"| {name} | {box.p[row]:.3f} | {box.r[row]:.3f} "
              f"| {box.ap50[row]:.3f} | {box.ap[row]:.3f} |")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    main(split)