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

End-to-end computer vision pipeline to detect whether an architectural doorway is **`door_open`** (traversable) or **`door_closed`** (obstacle) for autonomous mobile robot navigation.

```
[Multi-Source Raw Data] ──► [aHash Dedup (-14.7%)] ──► [6 Controlled Experiments] ──► [Held-Out Test] ──► [ONNX Export]
      (2,512 images)              (2,143 clean)           (LR, Scale, Aug, Size)       (95.7% F1)          (CUDA / CPU)
```

**Key achievements:**
- **Deduplication:** Pruned 369 redundant CCTV burst frames (14.7%) via 256-bit aHash before dataset splitting to eliminate train/test data leakage.
- **Controlled ablations:** 6 training experiments each isolating exactly one factor group (learning rate, model capacity, resolution, augmentation).
- **Winning model (`lr_schedule`) on held-out test (N=281):** Precision 97.64%, Recall 93.87%, **F1 95.72%**, mAP@0.5 98.07%, mAP@0.5:0.95 84.52%.
- **Safety asymmetry analysis:** Formally audited collision hazards (Closed→Open: 3.88%) vs fail-safe stops (Open→Closed: 2.25%) using ground-truth confusion matrix indexing.
- **Production ONNX model (`models/best.onnx`):** 3-tier validated and benchmarked on both CUDA (6.80 ms / ~147 FPS) and CPU (46.20 ms / ~21.6 FPS).

---

## 2. Dataset Preparation & Deduplication

Three public Roboflow sources were aggregated, polygon coordinates converted to bounding boxes, and near-duplicates pruned before splitting:

| Source | Raw Images | Retained | Pruned | Format Normalized | Visual Domain |
|---|---:|---:|---:|---|---|
| `vikashs_1527` | 1,527 | 1,522 | 5 | 10-pt Polygon → BBox | Residential and office room doorways |
| `fiw_706` | 691 | 327 | 364 | Bounding Box | Commercial storage corridor CCTV streams |
| `utfyu_116` | 294 | 294 | 0 | Bounding Box | Apartment hallways, mobile phone photos |
| **Total Canonical** | **2,512** | **2,143** | **369 (14.7%)** | 2 Classes | **1,541 Train / 321 Val / 281 Test** |

> **Domain note on `fiw_706`:** These frames originate from an overhead CCTV camera in a commercial project storage corridor. The camera captured identical burst frames at 30 FPS. Pruning 364 near-duplicate frames before splitting was necessary to prevent identical frames from leaking across train and test sets.

> **Deterministic paths in `data/data.yaml`:** Dataset splits are declared as `../dataset/images/train`, `../dataset/images/val`, and `../dataset/images/test` relative to the `data/` folder, ensuring deterministic path resolution across different machines and clones without relying on global cache directories.

---

## 3. Hyperparameter Experiments & Validation Ablations

Six experiments evaluated on the **Validation Split (N=321)**, each changing one factor group with all other parameters frozen:

| # | Experiment Name | Model Architecture | Resolution | Key Factor Tested | Precision | Recall | **F1 Score** | mAP@0.5 | **mAP@0.5:0.95** | Latency (ms) |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `baseline` | YOLOv8n (3.0M) | 640×640 | Reference (COCO defaults) | 0.9704 | 0.9690 | 0.9697 | 0.9757 | 0.8355 | 22.05 ms |
| 2 | `augmentation` | YOLOv8n (3.0M) | 640×640 | +HSV jitter, shear (2.0), mixup (0.1) | 0.9696 | 0.9645 | 0.9670 | 0.9846 | 0.8197 | 21.23 ms |
| 3 | `high_resolution`| YOLOv8n (3.0M) | 960×960 | Spatial scale 640→960px, batch 8 | 0.9791 | 0.9468 | 0.9627 | 0.9865 | 0.8327 | 26.56 ms |
| 4 | `final` | YOLOv8n (3.0M) | 800×800 | Intermediate resolution scale | 0.9696 | 0.9673 | 0.9684 | 0.9844 | 0.8126 | 24.94 ms |
| **5** | **`lr_schedule` 🏆** | **YOLOv8n (3.0M)** | **640×640** | **LR 0.01→0.001 + AdamW** | **0.9680** | **0.9738** | **0.9709** | **0.9806** | **0.8462** | **14.00 ms** |
| 6 | `model_size` | YOLOv8s (11.1M) | 640×640 | Higher model capacity (3.7× params) | 0.9800 | 0.9651 | 0.9725 | 0.9900 | 0.8455 | 18.80 ms |

> **Exp 7 — Confidence threshold sweep (post-hoc):** Sweeping `conf` from 0.10 to 0.60 showed peak F1 at `conf=0.25` (F1=0.9718). This threshold is applied during test inference.

---

## 4. Model Selection Rationale

**Selected Winner: `lr_schedule` (Exp 5)**

Selection decision criteria:
1. **Real-Time Constraint:** Latency must be < 30 ms on edge hardware — all candidates passed.
2. **Strict Localization (mAP@0.5:0.95):** `lr_schedule` achieved the highest localization score (**0.8462**), outperforming both baseline (0.8355) and the 3.7× larger YOLOv8s (0.8455).
3. **F1 & Recall Balance:** Achieved **0.9709 F1** and the highest validation recall (**97.38%**), crucial for detecting traversable doors.
4. **Execution Speed:** Fastest PyTorch inference (**14.00 ms / ~71.4 FPS**).

**Why `lr_schedule` outperformed baseline:** COCO pre-trained weights already possess robust low-level edge filters. Starting training with the default high learning rate (`lr0=0.01`) induces large initial gradient updates that disrupt these representations. Reducing initial learning rate to `0.001` with AdamW enabled smooth fine-tuning, allowing the bounding box regression head to converge tightly around doorframe boundaries without gradient instability.

---

## 5. Held-Out Test Results & Robotics Safety Analysis

The selected `lr_schedule` model was evaluated on the **Held-Out Test Set (N=281 images, 281 ground-truth instances)**:

$$\text{Precision: } \mathbf{97.64\%} \quad|\quad \text{Recall: } \mathbf{93.87\%} \quad|\quad \mathbf{F_1\text{ Score: }} \mathbf{0.9572} \quad|\quad \text{mAP@0.5: } \mathbf{98.07\%} \quad|\quad \text{mAP@0.5:0.95: } \mathbf{84.52\%}$$

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1 Score | mAP@0.5 | mAP@0.5:0.95 | Ground Truth Instances |
|---|---:|---:|---:|---:|---:|---:|
| `door_open` | **100.00%** | **93.57%** | **0.9668** | **97.32%** | **85.36%** | 178 |
| `door_closed` | **95.27%** | **94.17%** | **0.9472** | **98.82%** | **83.67%** | 103 |
| **All Classes** | **97.64%** | **93.87%** | **0.9572** | **98.07%** | **84.52%** | **281** |

*Logged in `results/test_class_metrics.json` and `results/metrics_lr_schedule_test.json`.*

### Confusion Matrix & Safety Asymmetry

```
                 Predicted Open    Predicted Closed    Missed (Background)    Total Actual
Actual Open            167                 4                    7                 178
Actual Closed            4                98                    1                 103
```

> **Robotics Safety Asymmetry Audit:**
> * **False Traversability Hazard (Actual Closed → Predicted Open):** Occurred in **4 out of 103 closed doors (3.88%)**. Predicting a closed door as open creates a collision hazard. In deployment, a 3-frame temporal consensus filter requires 3 consecutive agreeing detections before clear footprint commands are dispatched to Nav2.
> * **Fail-Safe Pause (Actual Open → Predicted Closed):** Occurred in **4 out of 178 open doors (2.25%)**. This error causes the robot to momentarily pause or re-route, representing a safe failure mode.
> * **Missed Detections (Background):** 7 open doors (3.93%) and 1 closed door (0.97%) had no overlapping prediction above threshold.

---

## 6. Example Predictions

Detections on held-out test scenes across residential, office, and commercial environments:

![Prediction showcase](results/predictions/example_predictions_showcase.jpg)

- **Green badge** = `door_open` (traversable)
- **Red badge** = `door_closed` (obstacle)

---

## 7. Production ONNX Export & Multi-Runtime Benchmark

The winning model was exported to **ONNX (Opset 12)** and verified with `onnx.checker` topology validation, zero-crash session execution, and PyTorch numerical output parity:

```bash
python src/export_onnx.py --weights runs/detect/lr_schedule/weights/best.pt --imgsz 640 --opset 12
```

### Hardware Latency Profiling (100 Iterations, 10 Warmup)

| Model Variant / Runtime | Execution Provider / Device | Mean Latency | Median (P50) | Throughput | Artifact Log |
|---|---|---:|---:|---:|---|
| **ONNX Runtime (CUDA EP)** | NVIDIA RTX 3050 Laptop GPU | **6.80 ms** | 6.52 ms | **~147.0 FPS** | `results/benchmark_best_onnx_cuda.json` |
| **PyTorch FP16 (CUDA)** | NVIDIA RTX 3050 Laptop GPU | **14.00 ms** | 13.50 ms | **~71.4 FPS** | `results/benchmark_lr_schedule.json` |
| **ONNX Runtime (CPU EP)** | Host Intel CPU (Fallback) | **46.20 ms** | 43.10 ms | **~21.6 FPS** | `results/benchmark_best_onnx_cpu.json` |

---

## 8. Failure Modes & Mitigations

| Failure Mode | Visual Signature | Root Cause | Engineering Mitigation |
|---|---|---|---|
| **Low Contrast / Glare** | Missed closed door in dim hallway | Door panel blends with frame | Contrast-adaptive histogram equalization |
| **Partial Occlusion** | Broken detection box | Carts/people block door edges | Cutout & synthetic foreground occlusion training |
| **Glass / Specular Reflection** | Closed glass door misidentified | Frame-only visual feature ambiguity | Cross-validate with 2D LiDAR / depth point cloud |
| **Ajar Door (5°–15°)** | Ambiguous state | Narrow visual opening gap | Calculate metric aperture width via RGB-D sensor |
| **Small / Distant Door (>10m)** | Low confidence score | Bounding box occupies <2% of sensor | Adaptive Region-of-Interest (ROI) digital crop |

---

## 9. Reproduction & Quick-Start Guide

```bash
git clone https://github.com/tanukusaitejesh-prog/YOLO-TASK.git
cd YOLO-TASK
pip install -r requirements.txt

# 1. Train winning lr_schedule model
python src/train.py --experiment lr_schedule

# 2. Evaluate on test split (saves full per-class metrics & confusion matrix)
python src/evaluate.py --weights runs/detect/lr_schedule/weights/best.pt --split test --imgsz 640

# 3. Export to ONNX (Opset 12)
python src/export_onnx.py --weights runs/detect/lr_schedule/weights/best.pt --imgsz 640 --opset 12

# 4. Profile latency on CUDA and CPU
python src/benchmark.py --weights models/best.onnx --model-type onnx --device cuda --imgsz 640 --name best_onnx_cuda
python src/benchmark.py --weights models/best.onnx --model-type onnx --device cpu --imgsz 640 --name best_onnx_cpu
```
