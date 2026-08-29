# Swift Robotics — Door Open / Closed Detection Pipeline
> **Perception Subsystem for Autonomous Mobile Robots (AMRs)**
> **Candidate:** Saitejesh Tanuku | **Role:** Junior AI Engineer Technical Evaluation

[![Python 3.12](https://img.shields.io/badge/Python-3.12.4-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![ONNX Opset 12](https://img.shields.io/badge/ONNX-Opset%2012-005CED.svg?logo=onnx&logoColor=white)](https://onnx.ai/)
[![License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 1. Executive Summary

End-to-end pipeline to detect whether a doorway is **`door_open`** (traversable) or **`door_closed`** (obstacle).

```
[Multi-Source Raw Data] ──► [aHash Dedup (-14.7%)] ──► [6 Controlled Experiments] ──► [Held-Out Test] ──► [ONNX Export]
      (2,512 images)              (2,143 clean)           (LR, Scale, Aug, Size)       (95.7% F1)          (ONNXRuntime CPU)
```

**Key achievements:**
- **Deduplication:** Pruned 369 redundant CCTV burst frames (14.7%) via 256-bit aHash before splitting to prevent train/test leakage.
- **Controlled ablations:** 6 training experiments each isolating exactly one factor group (LR schedule, model capacity, resolution, augmentation).
- **Best model (`lr_schedule`) on held-out test (N=281):** Precision 97.64%, Recall 93.87%, **F1 95.72%**, mAP@0.5 98.07%, mAP@0.5:0.95 84.52%.
- **Safety asymmetry:** Closed→Open false positive rate < 1% (1/103) on the test set.
- **ONNX export** with 3-tier validation (structure, runtime, PyTorch/ONNX output parity).

---

## 2. Dataset Preparation & Deduplication

Three public Roboflow sources were merged, annotation formats normalized, and near-duplicates removed before any train/val/test split:

| Source | Raw | Retained | Pruned | Format | Domain |
|---|---:|---:|---:|---|---|
| `vikashs_1527` | 1,527 | 1,522 | 5 | 10-pt Polygon → BBox | Residential and office doorways |
| `fiw_706` | 691 | 327 | 364 | Bounding Box | Commercial storage corridor CCTV |
| `utfyu_116` | 294 | 294 | 0 | Bounding Box | Apartment hallways, handheld photos |
| **Total** | **2,512** | **2,143** | **369 (14.7%)** | 2 classes | **1,541 Train / 321 Val / 281 Test** |

> **Note on `fiw_706`:** These are fixed-camera CCTV frames from a commercial corridor — visually distinct from residential hallways. The 364 near-duplicate burst frames were removed before splitting; retaining them would have caused severe train/test leakage and inflated benchmark scores.

> **Why dedup before splitting:** A static camera at 30 FPS produces nearly identical consecutive frames. Deduplicating the full pool first (rather than per-split) is the correct approach — it prevents duplicate pairs from straddling the train/test boundary.

---

## 3. Hyperparameter Experiments

Six experiments on the **validation split (N=321)**, each changing exactly one factor with everything else frozen (same seed, same base model unless noted):

| # | Experiment | Model | ImgSz | Key Variable | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Latency |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `baseline` | YOLOv8n | 640 | COCO defaults (reference) | 0.9704 | 0.9690 | 0.9697 | 0.9757 | 0.8355 | 22.05 ms |
| 2 | `augmentation` | YOLOv8n | 640 | +HSV jitter, shear, mixup | 0.9696 | 0.9645 | 0.9670 | 0.9846 | 0.8197 | 21.23 ms |
| 3 | `high_resolution` | YOLOv8n | 960 | Resolution 640→960, batch 8 | 0.9791 | 0.9468 | 0.9627 | 0.9865 | 0.8327 | 26.56 ms |
| 4 | `final` | YOLOv8n | 800 | Intermediate resolution | 0.9696 | 0.9673 | 0.9684 | 0.9844 | 0.8126 | 24.94 ms |
| **5** | **`lr_schedule` ✅** | **YOLOv8n** | **640** | **LR 0.01→0.001, AdamW** | **0.9680** | **0.9738** | **0.9709** | **0.9806** | **0.8462** | **17.73 ms** |
| 6 | `model_size` | YOLOv8s | 640 | Larger backbone (11.1M params) | 0.9800 | 0.9651 | 0.9725 | 0.9900 | 0.8455 | 18.80 ms |

> **Confidence threshold sweep (post-hoc, Exp 7):** Swept `conf` from 0.10 to 0.60 on the baseline model. F1 peaks at `conf=0.25` (F1=0.9718). This threshold is applied for all final evaluations.

---

## 4. Model Selection

**Selected model: `lr_schedule` (Exp 5)**

Selection criteria in priority order:

1. **Hard latency cap ≤ 30 ms** — all experiments pass.
2. **Highest mAP@0.5:0.95** (strict localization across IoU thresholds 0.50–0.95) — `lr_schedule` wins at **0.8462**, just ahead of `model_size` (0.8455) and well above `baseline` (0.8355).
3. **Highest F1 at this architecture tier** — `lr_schedule` at 0.9709 vs baseline's 0.9697.
4. **Lowest latency** — `lr_schedule` at **17.73 ms (56.4 FPS)** is 4.3 ms faster than baseline.

`model_size` (Exp 6, YOLOv8s) achieves higher precision and mAP@0.5 but is a different architecture (3.7× more parameters), making it a capacity trade-off rather than a hyperparameter comparison within the same model family.

**Why `lr_schedule` beats baseline:** Pre-trained YOLO weights carry useful low-level edge filters tuned for 80 COCO classes. The default `lr0=0.01` applies large early gradient updates that partially overwrite those filters before settling. Dropping to `lr0=0.001` with AdamW lets the regression head converge smoothly around rectangular doorframe contours, yielding better strict IoU localization (mAP@0.5:0.95) and recall.

**Where the model still fails:**
- **Low contrast / backlighting** — jamb edges vanish against a bright exterior.
- **Partial occlusion** — a cart or person in the foreground breaks the door contour.
- **Glass / transparent doors** — model relies on frame edges; specular reflections from a closed glass door can resemble an open corridor.
- **Ajar (5°–15° open)** — ambiguous narrow gap; model tends toward `door_closed` which is the safe failure.

---

## 5. Held-Out Test Results

**`lr_schedule` evaluated once on the permanently locked test set (N=281 images, 281 instances):**

| Metric | All | `door_open` | `door_closed` |
|---|---:|---:|---:|
| Precision | **97.64%** | 100.00% | 95.28% |
| Recall | **93.87%** | 93.60% | 94.17% |
| F1 | **95.72%** | 96.69% | 94.72% |
| mAP@0.5 | **98.07%** | 97.32% | 98.82% |
| mAP@0.5:0.95 | **84.52%** | 85.40% | 83.70% |

**Safety asymmetry:**
- Closed predicted as Open (collision hazard): **1 out of 103 (0.97%)**.
- Open predicted as Closed (fail-safe pause): 5 out of 178 (2.8%).

A 3-frame temporal consensus filter (require 3 consecutive agreeing frames before commanding movement) reduces the single-frame hazard rate further.

---

## 6. Example Predictions

Six held-out test images across residential, office, and corridor environments:

![Prediction showcase](results/predictions/example_predictions_showcase.jpg)

- **Green badge** = `door_open` (traversable)
- **Red badge** = `door_closed` (obstacle)

---

## 7. ONNX Export & Benchmarking

**Export:**
```bash
python src/export_onnx.py --weights runs/detect/lr_schedule/weights/best.pt --imgsz 640 --opset 12
```

`export_onnx.py` runs 3-tier validation automatically:
1. `onnx.checker.check_model` — graph topology and tensor shape validation.
2. Zero-crash runtime test via `onnxruntime.InferenceSession`.
3. Max absolute difference between PyTorch and ONNX outputs < 1e-4.

**Latency benchmark:**

| Engine | Device | Mean latency | Throughput |
|---|---|---:|---:|
| PyTorch FP16 | NVIDIA RTX 3050 (CUDA 12.1) | 17.73 ms | ~56 FPS |
| ONNXRuntime (CPU EP) | Intel Core i7 | 48.14 ms | ~21 FPS |

> Full CPU profiling results in `results/benchmark_best_onnx_cpu.json` (100 iterations, 10 warmup). CUDA EP requires `onnxruntime-gpu`; the CPU number is the portable cross-platform baseline.

---

## 8. Failure Modes

| Failure | Cause | Mitigation |
|---|---|---|
| Low-light / backlit | Jamb edges vanish | Histogram equalization pre-processing |
| Partial occlusion | Broken door contour | Cutout augmentation with foreground objects |
| Glass / transparent door | Frame-only cue unreliable | Fuse with depth sensor or LiDAR |
| Ajar door (5°–15°) | Ambiguous open gap | Depth-based aperture width estimate |
| Small/distant door (>10 m) | <2% of frame | Adaptive ROI crop |

---

## 9. Reproduction

```bash
git clone https://github.com/tanukusaitejesh-prog/YOLO-TASK.git
cd YOLO-TASK
pip install -r requirements.txt

# Train
python src/train.py --experiment lr_schedule

# Evaluate on test set
python src/evaluate.py --weights runs/detect/lr_schedule/weights/best.pt --split test --imgsz 640

# Benchmark latency
python src/benchmark.py --weights runs/detect/lr_schedule/weights/best.pt --model-type pytorch --imgsz 640

# Export to ONNX
python src/export_onnx.py --weights runs/detect/lr_schedule/weights/best.pt --imgsz 640 --opset 12
```

> **Dataset path:** Update `path:` in `data/data.yaml` to point to your local dataset directory before training.
