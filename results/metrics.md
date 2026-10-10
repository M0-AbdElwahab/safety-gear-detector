# Results (yolov8s, 60 epochs)

Model: YOLOv8s fine-tuned from COCO weights, imgsz 640, batch 16, seed 0, 60 epochs (0.624 h on a Colab T4).
Software: Ultralytics 8.4.175, Python 3.13.15, torch 2.11.0+cu130.
Dataset: PPE_master v1 by harshaJSN (CC BY 4.0), 2,615 images: train 2,092 / valid 392 / test 131.
One training image was skipped as corrupt (labels mix segment and detection rows), so 2,091 were used.

## Data caveat: overlap between splits
Unique source names (text before ".rf." in file names): train 1,421 / valid 343 / test 119.
59 of the 119 test names and 147 of the 343 valid names also appear in train, so some images are
augmented copies of the same source photo. Scores below are likely optimistic. Matching is by
file name only, so near-duplicates with different names may remain.

## 1) Validation split (392 images, 2,887 boxes) - most optimistic; best.pt was selected on it
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.906 | 0.887 | 0.914 | 0.628 |
| boots | 0.955 | 0.957 | 0.975 | 0.754 |
| helmet | 0.921 | 0.963 | 0.969 | 0.681 |
| nohelmet | 0.855 | 0.699 | 0.755 | 0.317 |
| novest | 0.834 | 0.824 | 0.854 | 0.456 |
| person | 0.926 | 0.932 | 0.961 | 0.789 |
| vest | 0.942 | 0.948 | 0.969 | 0.769 |

## 2) Full test split (131 images, 844 boxes)
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.913 | 0.881 | 0.926 | 0.656 |
| boots | 0.943 | 0.930 | 0.959 | 0.713 |
| helmet | 0.971 | 0.970 | 0.991 | 0.730 |
| nohelmet | 0.805 | 0.778 | 0.831 | 0.422 |
| novest | 0.883 | 0.695 | 0.820 | 0.439 |
| person | 0.928 | 0.947 | 0.982 | 0.828 |
| vest | 0.949 | 0.967 | 0.975 | 0.804 |

## 3) Name-filtered test (62 of 131 images, 334 boxes) - most conservative, headline
Only test images whose source name does not appear in train.
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.868 | 0.875 | 0.920 | 0.614 |
| boots | 0.767 | 0.933 | 0.957 | 0.721 |
| helmet | 0.948 | 0.969 | 0.993 | 0.718 |
| nohelmet | 0.805 | 0.765 | 0.832 | 0.419 |
| novest | 0.887 | 0.695 | 0.820 | 0.444 |
| person | 0.880 | 0.907 | 0.966 | 0.711 |
| vest | 0.917 | 0.980 | 0.949 | 0.674 |

Notes: boots has only 15 boxes in 6 images in table 3, and nohelmet has 27 boxes, so those rows are noisy.
Inference time reported by Ultralytics on a T4: 7.0 ms (val), 11.4 ms (test), 11.7 ms (filtered test).
This is model time per image, not API latency.