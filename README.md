# Swift Robotics — Door Open / Closed Detection Pipeline
> **Perception Subsystem for Autonomous Mobile Robots (AMRs)**  
> **Candidate:** Saitejesh Tanuku | **Role:** Junior AI Engineer Technical Evaluation

[![Python 3.12](https://img.shields.io/badge/Python-3.12.4-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![ONNX Opset 12](https://img.shields.io/badge/ONNX-Opset%2012-005CED.svg?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Code License: MIT](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 1. Executive Summary

End-to-end computer vision pipeline to detect whether an architectural doorway is **`door_open`** (traversable) or **`door_closed`** (obstacle) for autonomous mobile robot navigation.

```
[Multi-Source Raw Data] --> [aHash Dedup (-14.7%)] --> [6 Controlled Experiments] --> [Held-Out Test] --> [ONNX Export]
      (2,512 images)              (2,143 clean)           (LR, Scale, Aug, Size)       (95.7% F1)          (CUDA / CPU)
```

**Key achievements:**
- **Deduplication:** Pruned 369 redundant CCTV burst frames (14.7%) via 256-bit aHash before dataset splitting to eliminate train/test data leakage.
- **Controlled ablations:** 6 experiments — 5 training runs isolating Learning Rate schedules, Model Capacity, Spatial Resolution, and Domain Augmentation, plus a post-hoc confidence threshold sweep.
- **Winning model (`lr_schedule`) on held-out test (N=281):** Precision 97.64%, Recall 93.87%, **F1 95.72%**, mAP@0.5 98.07%, mAP@0.5:0.95 84.52%.
- **Safety asymmetry audit:** Evaluated collision hazards (Closed -> Open: 3.88%) vs fail-safe stops (Open -> Closed: 2.25%) using ground-truth confusion matrix indexing.
- **Production ONNX model (`models/best.onnx`):** 4-tier validated (including numerical output tensor parity) and profiled on both CUDA (6.90 ms / ~145 FPS) and CPU (51.64 ms / ~19 FPS).

---

## 2. Dataset Preparation & Deduplication

Three public Roboflow sources were merged, polygon coordinates normalized to bounding boxes, and near-duplicates removed before splitting:

| Source | Raw Images | Retained | Pruned | Format Normalized | Visual Domain |
|---|---:|---:|---:|---|---|
| `vikashs_1527` | 1,527 | 1,522 | 5 | 10-pt Polygon -> BBox | Residential and office room doorways |
| `fiw_706` | 691 | 327 | 364 | Bounding Box | Commercial warehouse, loading docks & storage corridor CCTV |
| `utfyu_116` | 294 | 294 | 0 | Bounding Box | Apartment hallways, mobile phone photos |
| **Total Canonical** | **2,512** | **2,143** | **369 (14.7%)** | 2 Classes | **1,541 Train / 321 Val / 281 Test** |

> **Domain & source notes:** `vikashs_1527`, `fiw_706`, and `utfyu_116` represent upstream Roboflow Universe dataset project identifiers; the Raw Images column indicates the actual annotated image count validated and ingested during pipeline preprocessing. Frames in `fiw_706` originate from overhead CCTV cameras in commercial facility storage rooms and loading dock hallways: because fixed-angle surveillance captures continuous video bursts at 30 FPS, removing 364 near-duplicate frames before dataset splitting was essential to prevent identical scenes from leaking across train and test sets.

> **Deterministic paths in `data/data.yaml`:** Dataset splits are declared as `../dataset/images/train`, `../dataset/images/val`, and `../dataset/images/test` relative to the `data/` folder, ensuring deterministic path resolution across different machines and clones without relying on global cache directories.

---

## 3. Hyperparameter Experiments & Validation Ablations

Five experiments evaluated on the **Validation Split (N=321)**, each changing exactly one factor group with all other parameters frozen:

| # | Experiment Name | Model Architecture | Resolution | Key Factor Tested | Precision | Recall | **F1 Score** | mAP@0.5 | **mAP@0.5:0.95** | Latency (ms) |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `baseline` | YOLOv8n (3.0M) | 640×640 | Reference (COCO defaults) | 0.9704 | 0.9690 | 0.9697 | 0.9757 | 0.8355 | 22.05 ms |
| 2 | `augmentation` | YOLOv8n (3.0M) | 640×640 | +HSV jitter, shear (2.0), mixup (0.1) | 0.9696 | 0.9645 | 0.9670 | 0.9846 | 0.8197 | 21.23 ms |
| 3 | `high_resolution`| YOLOv8n (3.0M) | 960×960 | Spatial scale 640->960px, batch 8 | 0.9791 | 0.9468 | 0.9627 | 0.9865 | 0.8327 | 26.56 ms |
| **4** | **`lr_schedule` (Best)** | **YOLOv8n (3.0M)** | **640×640** | **Cosine annealing floor `lrf` 0.01->0.001** | **0.9680** | **0.9738** | **0.9709** | **0.9806** | **0.8462** | **17.73 ms** |
| 5 | `model_size` | YOLOv8s (11.2M) | 640×640 | Higher model capacity (3.7× params) | 0.9800 | 0.9651 | 0.9725 | 0.9900 | 0.8455 | 18.80 ms |

> **Exp 6 — Confidence threshold sweep (post-hoc):** Sweeping `conf` from 0.10 to 0.60 showed peak F1 at `conf=0.25` (F1=0.9718). This threshold is applied during test inference.

---

## 4. Model Selection Rationale & Hyperparameter Impact Analysis

**Selected Best Model: `lr_schedule` (Experiment 4)**

### Why It Performed Better
1. **Strict Localization Superiority (mAP@0.5:0.95):** `lr_schedule` achieved the highest localization score (**0.8462**), outperforming baseline (0.8355), high resolution (0.8327), and the 3.7× larger YOLOv8s (0.8455). Accurate boundary localization is vital for mobile robots to compute door opening aperture widths accurately.
2. **High Traversability Recall (97.38%):** Maximizes detection of open passageways, preventing unnecessary robot hesitations or costly re-routing.
3. **Edge Latency & Efficiency:** Executes in **12.52 ms** (PyTorch FP16) and **6.90 ms** (ONNX CUDA), leaving substantial compute headroom for simultaneous SLAM and path planning.

### Which Hyperparameters Had the Biggest Impact
* **Cosine Annealing Floor (`lrf`):** Setting `lrf=0.001` (vs baseline `0.01`) was the primary performance driver. While Ultralytics `optimizer="auto"` resolves to AdamW ($\text{lr} \approx 0.00167$) for both baseline and `lr_schedule` ($\text{iterations} = 2,500 < 10,000$), the tighter decay floor allowed late-epoch learning rates to decay down to $1.8 \times 10^{-5}$ (vs $3.3 \times 10^{-5}$). This prevented regression head oscillation around thin door jambs, boosting mAP@0.5:0.95 by **+1.07%**.
* **Spatial Resolution (`imgsz`):** Increasing resolution to 960×960 improved precision (+0.87%) but dropped recall (-2.22%) and increased latency by +20.4%, proving suboptimal for real-time edge robotics.
* **Model Scale (`yolov8s`):** Scaling to 11.2M parameters improved mAP@0.5 to 0.9900 but did not surpass `lr_schedule` on mAP@0.5:0.95 (0.8455 vs 0.8462) despite consuming $3.5\times$ more compute.

*Note: `src/train.py` explicitly supports `optimizer = cfg.get("optimizer", "auto")`, allowing full manual control over optimizer family and learning rate dynamics.*

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
> * **False Traversability Hazard (Actual Closed -> Predicted Open):** Occurred in **4 out of 103 closed doors (3.88%)**. Predicting a closed door as open creates a collision hazard. In deployment, a 3-frame temporal consensus filter requires 3 consecutive agreeing detections before clear footprint commands are dispatched to Nav2.
> * **Fail-Safe Pause (Actual Open -> Predicted Closed):** Occurred in **4 out of 178 open doors (2.25%)**. This error causes the robot to momentarily pause or re-route, representing a safe failure mode.
> * **Missed Detections (Background):** 7 open doors (3.93%) and 1 closed door (0.97%) had no overlapping prediction above threshold.

---

## 6. Example Predictions

Detections generated directly from the winning `lr_schedule` model on held-out test scenes across residential, office, and commercial environments:

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

| Model Variant / Runtime | Execution Provider / Device | Mean Latency | Median (P50) | 95th %ile (P95) | Throughput | Artifact Log |
|---|---|---:|---:|---:|---:|---|
| **ONNX Runtime (CUDA EP)** | NVIDIA RTX 3050 Laptop GPU | **6.90 ms** | 6.77 ms | 7.96 ms | **~144.9 FPS** | `results/benchmark_best_onnx_cuda.json` |
| **PyTorch FP16 (CUDA)** | NVIDIA RTX 3050 Laptop GPU | **12.52 ms** | 11.22 ms | 18.58 ms | **~79.9 FPS** | `results/benchmark_lr_schedule.json` |
| **ONNX Runtime (CPU EP)** | Host Intel CPU (Fallback) | **51.64 ms** | 48.17 ms | 78.68 ms | **~19.4 FPS** | `results/benchmark_best_onnx_cpu.json` |

---

## 8. Failure Modes & Mitigations

| Failure Mode | Visual Signature | Root Cause | Engineering Mitigation |
|---|---|---|---|
| **Low Contrast / Glare** | Missed closed door in dim hallway | Door panel blends with frame | Contrast-adaptive histogram equalization |
| **Partial Occlusion** | Broken detection box | Carts/people block door edges | Cutout & synthetic foreground occlusion training |
| **Glass / Specular Reflection** | Closed glass door misidentified | Frame-only visual feature ambiguity | Cross-validate with 2D LiDAR / depth point cloud |
| **Ajar Door (5 deg - 15 deg)** | Ambiguous state | Narrow visual opening gap | Calculate metric aperture width via RGB-D sensor |
| **Small / Distant Door (>10m)** | Low confidence score | Bounding box occupies <2% of sensor | Adaptive Region-of-Interest (ROI) digital crop |

---

## 9. Deliverables Mapping & Submission Index

| Deliverable Requested in Task | Location in Repository | Summary of Deliverable |
|---|---|---|
| **1. Training code** | [`src/train.py`](src/train.py) | CLI training orchestrator with deterministic seeding & config pass-through |
| **2. Dataset configuration** | [`data/data.yaml`](data/data.yaml) | Portable relative paths (`../dataset/images/*`), 2 classes (`door_open`, `door_closed`) |
| **3. Hyperparameter experiment results** | [`results/experiment_results.csv`](results/experiment_results.csv) | Centralized table of 5 controlled ablations + held-out test + confidence sweep |
| **4. Best model metrics** | [`results/test_class_metrics.json`](results/test_class_metrics.json) | Per-class P/R/F1/AP breakdown, $3\times3$ confusion matrix & safety audit metrics |
| **5. 3–5 example predictions** | [`results/predictions/`](results/predictions/) | 6 individual test scene predictions + 1 master showcase montage with green/red badges |
| **6. ONNX model** | [`models/best.onnx`](models/best.onnx) | 12.3 MB Opset 12 static model with verified numerical tensor parity |
| **7. Short README.md** | [`README.md`](README.md) | Structured technical evaluation report covering methodology, trade-offs & results |

---

## 10. Reproduction & Quick-Start Guide

```bash
git clone https://github.com/tanukusaitejesh-prog/YOLO-TASK.git
cd YOLO-TASK
python -m venv venv
# Linux/macOS: source venv/bin/activate | Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Train winning lr_schedule model
python src/train.py --experiment lr_schedule

# 2. Evaluate on test split (saves full per-class metrics & confusion matrix)
python src/evaluate.py --weights runs/detect/lr_schedule/weights/best.pt --split test --imgsz 640

# 3. Export to ONNX (Opset 12) with 4-tier validation
python src/export_onnx.py --weights runs/detect/lr_schedule/weights/best.pt --imgsz 640 --opset 12

# 4. Profile latency on CUDA and CPU
python src/benchmark.py --weights models/best.onnx --model-type onnx --device cuda --imgsz 640 --name best_onnx_cuda
python src/benchmark.py --weights models/best.onnx --model-type onnx --device cpu --imgsz 640 --name best_onnx_cpu
```
