# Results (yolov8s, 60 epochs)

Model: YOLOv8s fine-tuned from COCO weights, imgsz 640, batch 16, seed 0, 60 epochs (0.624 h on a Colab T4).
Software: Ultralytics 8.4.175, Python 3.13.15, torch 2.11.0+cu130 (training, tables 1-3). Script re-runs and table 4 used Ultralytics 8.4.178 in Colab.
Dataset: PPE_master v1 by harshaJSN (CC BY 4.0), 2,615 images: train 2,092 / valid 392 / test 131.
One training image was skipped as corrupt (labels mix segment and detection rows), so 2,091 were used.

## Data caveat: overlap between splits
Unique source names (text before ".rf." in file names): train 1,421 / valid 343 / test 119.
59 of the 119 test names and 147 of the 343 valid names also appear in train, which is consistent with
some images being augmented copies of the same source photo. Scores below may be optimistic. Matching is by
file name only, so near-duplicates with different names may remain, and different photos sharing a name could be dropped from table 3.

## 1) Validation split (392 images, 2,887 boxes) - best.pt was selected on it
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

## 3) Name-filtered test (62 of 131 images, 334 boxes) - headline; lowest precision, recall and mAP50-95 of the three sets, but its mAP50 is similar
Only test images whose source name does not appear in train. The filter removed 69 images, all without a violation label.
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.868 | 0.875 | 0.920 | 0.614 |
| boots | 0.767 | 0.933 | 0.957 | 0.721 |
| helmet | 0.948 | 0.969 | 0.993 | 0.718 |
| nohelmet | 0.805 | 0.765 | 0.832 | 0.419 |
| novest | 0.887 | 0.695 | 0.820 | 0.444 |
| person | 0.880 | 0.907 | 0.966 | 0.711 |
| vest | 0.917 | 0.980 | 0.949 | 0.674 |

## 4) Exported best.onnx on the full test split (Ultralytics ONNX validation, batch 1, GPU)
The committed best.onnx and a fresh export from export.py gave identical results.
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.926 | 0.850 | 0.930 | 0.657 |
| boots | 0.957 | 0.927 | 0.964 | 0.708 |
| helmet | 0.966 | 0.964 | 0.992 | 0.736 |
| nohelmet | 0.857 | 0.667 | 0.837 | 0.427 |
| novest | 0.890 | 0.644 | 0.841 | 0.440 |
| person | 0.940 | 0.940 | 0.969 | 0.822 |
| vest | 0.944 | 0.958 | 0.974 | 0.808 |

## 5) app.py's own pipeline, image level (confidence 0.25)
| | Full test (131) | Name-filtered test (62) |
|---|---|---|
| Images with a violation label | 33 | 33 |
| ...app returned a violation box | 30 | 30 |
| Images with no violation label | 98 | 29 |
| ...app returned a violation box | 2 | 2 |
| Images labelled nohelmet: app returned nohelmet | 12 of 12 | 12 of 12 |
| Images labelled novest: app returned novest | 30 of 33 | 30 of 33 |

An image counts as flagged if any violation box appears, even on the wrong person. This is not box-level accuracy.

Notes: boots has only 15 boxes in 6 images in table 3, and nohelmet has 27 boxes, so those rows are noisy.
Inference time reported by Ultralytics for best.pt on a T4 GPU: 7.0 ms (val), 11.4 ms (test), 11.7 ms (filtered test); 10.6 ms in the evaluate.py re-run.
Table 4 inference: 14.4 ms (committed) and 13.5 ms (fresh export) per image through ONNX Runtime's CUDA provider.
These exclude pre- and post-processing and are not API latency (the Docker app runs on CPU and was not timed).
The committed best.onnx and the fresh export have the same size (44,754,255 bytes) but different SHA-256 hashes (first 16 hex characters: f7abd19f20a694c6 and 32b888253aa15a56).
Scores in tables 1-3 are for best.pt; table 4 is the exported model.
