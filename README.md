# Safety Gear Detector

YOLOv8 object detection for personal protective equipment (PPE), trained on a free Colab T4 and served through FastAPI in Docker with an ONNX model (no PyTorch needed at serving time).

**Classes (6):** `boots`, `helmet`, `nohelmet`, `novest`, `person`, `vest`.
`nohelmet` and `novest` are the violation classes. By design, the JSON endpoint sets `violation_found: true` when it detects either one. I saw it turn true in one test on a dataset image, using FastAPI's test client (see below). I have not seen the Docker container return true.

## Results

Model: YOLOv8s fine-tuned from COCO weights (imgsz 640, batch 16, 60 epochs, seed 0, 0.624 h on a Colab T4).

| Evaluation set | Images | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|---|
| Validation (used to pick best.pt) | 392 | 0.914 | 0.628 | 0.906 | 0.887 |
| Full test | 131 | 0.926 | 0.656 | 0.913 | 0.881 |
| **Name-filtered test (headline)** | 62 | **0.920** | **0.614** | 0.868 | 0.875 |

Per-class results on the name-filtered test set:

| Class | Precision | Recall | mAP@50 | mAP@50-95 | Boxes |
|---|---|---|---|---|---|
| boots | 0.767 | 0.933 | 0.957 | 0.721 | 15 |
| helmet | 0.948 | 0.969 | 0.993 | 0.718 | 76 |
| nohelmet | 0.805 | 0.765 | 0.832 | 0.419 | 27 |
| novest | 0.887 | 0.695 | 0.820 | 0.444 | 59 |
| person | 0.880 | 0.907 | 0.966 | 0.711 | 107 |
| vest | 0.917 | 0.980 | 0.949 | 0.674 | 50 |

Full tables for all sets, software versions and notes: [results/metrics.md](results/metrics.md).
Confusion matrices and curves: [results/](results/) (validation) and [results/test/](results/test/) (full test).

### Check of the API's own code (image level)
I ran `app.py`'s own preprocessing, ONNX inference and NMS (confidence threshold 0.25) on the test images and counted per image. An image counts as flagged if any violation box appears, even on the wrong person, so this is a sanity check, not an accuracy score, and it says nothing about box position.

| | Full test (131 images) | Name-filtered test (62 images) |
|---|---|---|
| Images with a violation label | 33 | 33 |
| ...app returned a violation box | 30 | 30 |
| Images with no violation label | 98 | 29 |
| ...app returned a violation box | 2 | 2 |
| Images labelled `nohelmet`: app returned `nohelmet` | 12 of 12 | 12 of 12 |
| Images labelled `novest`: app returned `novest` | 30 of 33 | 30 of 33 |

### Data caveat
The dataset likely has overlap between splits. The training split has 2,092 images but only 1,421 unique source names (text before `.rf.` in the file name), which fits with many images being augmented copies of the same source photo. 59 of the 119 unique test source names also appear in train. So the validation and full-test scores may be optimistic. The name-filtered test keeps only test images whose source name is not in train, so I use it as the headline: it is the set least likely to overlap with training. It has lower precision, recall and mAP@50-95 than the full test, but its mAP@50 (0.920) is close to the full test (0.926) and validation (0.914). It is also a smaller, different mix of images: the filter removed 69 images, all without a violation label, so 33 of the 62 filtered images have a violation label, against 33 of 131 in the full test. That is also why its `nohelmet` and `novest` rows are almost identical to the full-test rows, and why the differences are not purely due to leakage. Matching is by file name only: near-duplicates with different names may remain, and different photos that share a name could also be dropped. The name-filtered numbers come from a one-off Colab cell that drops test images whose source name appears in train and then runs the same validation on the rest; that cell is not included as a script in this repo.

### Limitations
- The violation classes are the weakest by mAP@50-95 and recall: `nohelmet` (27 filtered-test boxes) and `novest` (59 boxes) have the lowest mAP@50-95, and `novest` recall is 0.695. Missed violations are possible (see the demo).
- The test sets are small (62 and 131 images), so per-class numbers are noisy. `boots` has only 15 boxes (in 6 images) in the filtered set.
- The scores above are for `best.pt`. The exported `best.onnx` scored mAP@50 0.930, mAP@50-95 0.657, precision 0.926 and recall 0.850 on the full test split in Ultralytics' own ONNX validation (GPU, batch 1). Recall was lower than `best.pt` (0.881), including `nohelmet` 0.667 (vs 0.778) and `novest` 0.644 (vs 0.695); I did not investigate why. The API's own preprocessing and NMS were checked only at image level (table above), not with box-level mAP.
- On one dataset test image (labelled 2 person, 2 nohelmet, 2 novest), the app returned 4 person, 6 nohelmet and 5 novest boxes. I did not check whether the extra boxes are false positives or missing labels.
- Inference time reported by Ultralytics for `best.pt` on a T4 GPU was 7.0 ms (validation), 11.4 ms (full test) and 11.7 ms (filtered test) per image. This excludes pre- and post-processing and is not API latency: the Docker app runs on CPU and I have not measured its speed.
- One training image was skipped as corrupt (labels mix segment and detection rows), so 2,091 were used.
- `violation_found` is true only when the model itself predicts `nohelmet` or `novest`. A person with no helmet box at all is not flagged, because the app does not link helmets and vests to individual people.
- API testing was manual. Through the Swagger page at `/docs` I used the three endpoints. With `curl.exe` I sent one sample photo and an 11 MB dummy file. `violation_found: true` and the red-box drawing were checked outside the Docker container, using FastAPI's test client and a direct function call.

## Demo

![API output on a test photo](assets/demo_web_photo_1_small.jpg)

Output of `POST /predict/image` on a stock photo I downloaded for this demo. The model found both people (0.92, 0.91), both vests (0.89, 0.81) and the helmet on the right (0.84). **It missed the violation:** the man on the left wears no helmet, but the model produced no `nohelmet` box, so `violation_found` was `false`. This is consistent with the weak `nohelmet` recall in the results above, but it is a single image, not a measure of accuracy.

Photo by [Murat IŞIK](https://www.pexels.com/@bymuratisikofficial/) on [Pexels](https://www.pexels.com/photo/construction-workers-in-safety-gear-on-site-34670929/) (Pexels free license).

## Run the API (Docker)

```
docker build -t safety-gear-detector .
docker run --rm -p 8000:8000 safety-gear-detector
```

Then open http://localhost:8000/docs.

| Endpoint | What it does |
|---|---|
| `GET /health` | Server status, model name, class list |
| `POST /predict` | Upload an image, get detections as JSON (`class`, `confidence`, `box`, plus `violation_found`) |
| `POST /predict/image` | Upload an image, get it back with boxes drawn (red = violation, green = other) |

Example (Windows PowerShell; I ran this form with a sample photo and got valid JSON back):

```
curl.exe -X POST -F "file=@my_photo.jpg" http://localhost:8000/predict
```

Uploads over 10 MB are rejected with 413 (tested with an 11 MB dummy file), and a file that is not an image returned 400 in my test. Send the original photo, not a copy with boxes already drawn on it: in my tests an annotated copy returned no detections.

## Project files

| File | Purpose |
|---|---|
| `train.py` | Fine-tunes YOLOv8s (runs on Colab; same settings as the notebook cell I ran, but the script itself was not re-run) |
| `evaluate.py` | Scores the model on a split and prints a results table (runs on Colab; run for the test split, it reproduced the full-test numbers exactly) |
| `export.py` | Converts `best.pt` to ONNX (runs on Colab; its output has the same size as the committed model but a different hash, and both scored identically) |
| `app.py` | FastAPI server: preprocessing, ONNX inference, NMS, JSON/image responses |
| `Dockerfile` | Builds the serving image (Python 3.14-slim, onnxruntime) |
| `requirements.txt` | Serving dependencies only |
| `models/best.onnx` | Exported model used by the API |

The PyTorch weights (`best.pt`) are not in the repo.

## Reproduce training (Colab)

1. Colab with a T4 GPU, `pip install ultralytics roboflow`, Roboflow API key in Colab Secrets, Google Drive mounted.
2. Download PPE_master v1 in YOLOv8 format to `/content/ppe_data` and write `/content/ppe_data/data_fixed.yaml` with absolute paths (the downloaded `data.yaml` uses relative paths).
3. Run `train.py`. It saves to `/content/drive/MyDrive/ppe_project/runs/yolov8s_60ep`. Copy `weights/best.pt` from there to `/content/best.pt`, which is the path `evaluate.py` and `export.py` read, then run those two.

The reported numbers come from Ultralytics 8.4.175 (training, tables 1-3). I re-ran `evaluate.py` (test split) and `export.py` from a fresh clone on Ultralytics 8.4.178 using the same `best.pt`; I did not re-run `train.py`.

## Dataset and licenses

- Dataset: [PPE_master](https://universe.roboflow.com/harshajsn/ppe_master) v1 by harshaJSN on Roboflow Universe, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). I used the images and labels exactly as provided in version 1 and did not modify them. The name-filtered test set is a subset I selected for evaluation only. Some images look like stock photos with watermarks, so check the source before redistributing any dataset images.
- YOLOv8 comes from Ultralytics. Check its current license terms before commercial use or redistribution (I believe it is AGPL-3.0, with a separate enterprise license).
