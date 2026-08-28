# Door Open / Closed Detection
### Swift Robotics — Junior AI Engineer Technical Task

A YOLO-based computer vision pipeline that detects doors and classifies their state as `door_open` or `door_closed` in real time, designed for an autonomous mobile robot (AMR) navigating indoor and industrial environments.

---

## Table of Contents

1. [Objective & Task Requirements](#1-objective--task-requirements)
2. [Experimental Methodology](#2-experimental-methodology)
3. [Dataset Engineering & Multi-Source Synthesis](#3-dataset-engineering--multi-source-synthesis)
4. [Deduplication Quality Audit](#4-deduplication-quality-audit)
5. [Model Architecture & YOLO Fundamentals](#5-model-architecture--yolo-fundamentals)
6. [Controlled Factor-Group Experiments](#6-controlled-factor-group-experiments)
7. [Validation Comparison & Model Selection](#7-validation-comparison--model-selection)
8. [Held-Out Test Set Evaluation](#8-held-out-test-set-evaluation)
9. [Observed Failure Modes & Error Analysis](#9-observed-failure-modes--error-analysis)
10. [Hardware Latency & Throughput Benchmark](#10-hardware-latency--throughput-benchmark)
11. [ONNX Export & Three-Tier Verification](#11-onnx-export--three-tier-verification)
12. [Proposed Edge Deployment Architecture](#12-proposed-edge-deployment-architecture)
13. [How to Reproduce & Run](#13-how-to-reproduce--run)
14. [Repository Structure](#14-repository-structure)
15. [Future Improvements](#15-future-improvements)

---

## 1. Objective & Task Requirements

Autonomous mobile robots navigating indoors must reliably detect doorways and discern whether a door is open (allowing navigation through the threshold) or closed (requiring the robot to stop, reroute, or request access).

The technical goal is not simply to maximize benchmark numbers, but to demonstrate a structured, scientifically defensible engineering approach across dataset quality, controlled hyperparameter tuning, precision/recall trade-offs, edge latency profiling, and honest failure analysis.

---

## 2. Experimental Methodology

Four YOLO configurations were trained under controlled conditions. The validation split was used for experiment comparison and model selection, while the test split remained isolated until the final evaluation of the selected model. Experiments investigated baseline performance, augmentation, input resolution, and a combined candidate configuration. Model selection considered validation F1 alongside localization performance and inference latency.

```text
TRAIN 4 EXPERIMENTS
        ↓
EVALUATE ALL 4 CANDIDATES ON VALIDATION SET (val)
        ↓
COMPARE & SELECT BEST MODEL (Validation F1 & Latency Target)
        ↓
EVALUATE ONLY THE WINNING MODEL ON HELD-OUT TEST SET (test)
        ↓
SINGLE UNBIASED FINAL BENCHMARK
```

---

## 3. Dataset Engineering & Multi-Source Synthesis

Rather than relying on a single narrow dataset, three public object-detection datasets (under CC BY 4.0) were aggregated, normalized, and cleaned:

### Source-Level Dataset Breakdown

| Source Dataset | Raw Images | Retained | Duplicates Removed | Original Annotation Format | Normalization Applied |
|---|---:|---:|---:|---|---|
| `vikashs_1527` | 1,527 | 1,522 | 5 | `door-close`, `door-open` | Polygon segmentation masks converted to tight bounding boxes |
| `fiw_706` | 691 | 327 | 364 | `Door-Close`, `Door-Open` | Standard bounding boxes |
| `utfyu_116` | 294 | 294 | 0 | `door_close`, `door_open` | Standard bounding boxes |
| **Total** | **2,512** | **2,143** | **369 (14.7%)** | — | **Normalized to canonical `0: door_open`, `1: door_closed`** |

> **Polygon-to-Bounding-Box Conversion:** The `vikashs_1527` source provided 10-coordinate polygon segmentation points. A spatial boundary extractor computed the tight outer bounding rectangle $[(x_{\min}, y_{\min}), (x_{\max}, y_{\max})]$, normalized the center coordinates $(\text{cx}, \text{cy}, w, h)$, and validated all boundaries $[0.0, 1.0]$.

### Dataset Distribution & Instance Breakdown

| Split | Images | `door_open` Instances | `door_closed` Instances | Total Instances | Instances / Image |
|---|---:|---:|---:|---:|---:|
| **train** | 1541 | 924 | 617 | 1541 | 1.0 |
| **val** | 321 | 180 | 141 | 321 | 1.0 |
| **test** | 281 | 178 | 103 | 281 | 1.0 |
| **Total** | **2143** | **1282** | **861** | **2143** | **1.00** |


## 4. Deduplication Quality Audit

A 256-bit average hash ($16\times16$ `aHash`-style image hash) audited all 2,512 source images:
- **369 near-duplicate frames (14.7%)** were removed from the raw pool.
- **Audit Finding (`results/dedup_audit_report.json`):** 364 of the 369 removed duplicates originated from `fiw_706`, which consisted of high-frequency temporal burst frames captured by stationary surveillance feeds. Deduplication successfully eliminated redundant burst frames without removing unique architectural door styles or lighting conditions.
- **Visual Audit Grid:** A side-by-side inspection sample is saved in `results/dedup_sample_audit.jpg`.

---

## 5. Model Architecture & YOLO Fundamentals

**Base Model: YOLOv8n (Nano Variant)**

- **Model Complexity:** ~3,011,238 parameters, ~8.2 GFLOPs at $640\times640$ resolution (as reported by the Ultralytics model summary).
- **Backbone:** C2f-based convolutional backbone utilizing cross-stage partial connections with split-and-merge gradient routing.
- **Neck:** PAN/FPN-style multi-scale feature aggregation pyramid producing feature maps across three spatial strides: P3 ($80\times80$), P4 ($40\times40$), and P5 ($20\times20$).
- **Head & Loss:** Anchor-free decoupled detection head separating bounding box coordinate regression (DFL + CIoU Loss) from classification (Binary Cross-Entropy Loss).
- **Rationale:** For an onboard mobile robotics edge application with a binary door-state task, YOLOv8n provides strong visual feature extraction from COCO pretraining while maintaining sub-15ms latency on edge GPUs.

---

## 6. Controlled Factor-Group Experiments

Each experiment changes one primary factor group relative to the baseline to ensure clear causal attribution:

### Experiment 1 — Baseline Reference
- **Objective:** Establish a reference benchmark using standard YOLOv8n hyperparameters with COCO-pretrained weights.
- **Config:** `configs/baseline.yaml` (`imgsz: 640`, `batch: 16`, `lr0: 0.01`, default augmentations).

### Experiment 2 — Domain Augmentation
- **Hypothesis:** Targeted photometric and spatial augmentations (brightness jitter $\text{hsv\_v}=0.6$, rotation $\pm 5^\circ$, scale jitter $0.65$, shear $2.0$, mixup $0.1$) improve recall under variable hallway lighting and robot camera pitch/roll.
- **Config:** `configs/augmentation.yaml` (All learning rates, epochs, batch size, and resolution frozen at Baseline).

### Experiment 3 — High Spatial Resolution
- **Hypothesis:** Scaling input resolution ($640 \to 960\,\text{px}$) preserves fine spatial features of distant or partially visible doors and improves localization ($mAP@0.5:0.95$).
- **Config:** `configs/high_resolution.yaml` (`imgsz: 960`, `batch: 8`). *Batch size was adjusted to 8 to accommodate GPU VRAM constraints at higher resolution.*

### Experiment 4 — Combined Candidate
- **Hypothesis:** The combined candidate combines the settings that showed the most promising validation performance in the preceding experiments ($800\,\text{px}$ resolution + tuned augmentations) to achieve strong localization while maintaining real-time throughput.
- **Config:** `configs/final.yaml` (`imgsz: 800`, `batch: 12`, selected augmentations).

---

## 7. Validation Comparison & Model Selection

> **Validation-Driven Selection:** All candidate models were evaluated on the **Validation Split** ($N=321$ images). The test split was completely isolated during this phase.

| Candidate Experiment | Img Size | Key Change | Val Precision | Val Recall | Val F1 | Val mAP@0.5 | Val mAP@0.5:0.95 | Native Latency (ms) | FPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 640 | — | 0.9704 | 0.9690 | 0.9697 | 0.9757 | 0.8355 | 22.1 | ~45.3 |
| Augmentation | 640 | +aug | 0.9696 | 0.9645 | 0.9670 | 0.9846 | 0.8197 | 21.2 | ~47.1 |
| High Res | 960 | +res | 0.9791 | 0.9468 | 0.9627 | 0.9865 | 0.8327 | 26.6 | ~37.6 |
| Combined Candidate | 800 | +aug +res | 0.9696 | 0.9673 | 0.9684 | 0.9844 | 0.8126 | 24.9 | ~40.1 |


### Selection Decision Rule
The winning model is selected by evaluating validation F1 score alongside localization strictness ($mAP@0.5:0.95$) subject to an engineering latency target ($\le 30\,\text{ms}$). The full decision rationale is saved to `results/model_selection_decision.json`.

---

## 8. Held-Out Test Set Evaluation

Following model selection on validation data, the winning model was evaluated **once on the held-out Test split** ($N=281$ images, $281$ instances) to report the unbiased final performance:

| Winning Model | Split | Precision | Recall | F1 Score | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|---:|
| **Baseline** (`baseline`) | **Held-out Test** | **0.9651** | **0.9442** | **0.9546** | **0.9780** | **0.8274** |


## 9. Observed Failure Modes & Error Analysis

> **Protocol Note:** Failure analysis was performed after the final model configuration was locked and after the held-out test evaluation. No subsequent model-selection decisions were made using the test set.

Visual inspection of difficult and low-confidence test detections (`results/failure_analysis/failure_gallery.jpg`) identifies five observable failure modes:

| Observed Failure Mode | Visual Pattern | Likely Root Cause | Potential Future Improvement |
|---|---|---|---|
| **Low Illumination / Backlighting** | Missed closed door in dim hallways or high-contrast backlighting | Low contrast between door panel and doorframe | Targeted contrast/illumination augmentations |
| **Partial Occlusion** | False state prediction when obstacles partially cover the door | Foreground objects break continuous door panel edge geometry | Training images with realistic foreground occlusions |
| **Small / Distant Door** | Lower detection confidence when door is viewed from far down a hallway | Object occupies a small proportion of the image frame | Higher input resolution ($800\text{px}$) or multi-scale inference |
| **Glass / Specular Reflection** | Transparent or glossy doors misclassified | Specular reflections mimic open pathway geometry | Targeted collection and annotation of reflective doors |
| **Ambiguous State (Ajar)** | Low confidence on doors open by only a slight angle | Subtle visual separation between door edge and jamb | Temporal smoothing over consecutive frames in video |

---

## 10. Hardware Latency & Throughput Benchmark

Benchmarks were conducted using 10 warmup iterations followed by 100 timed iterations on an NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM):

| Model Variant | Runtime / Engine | Input Resolution | Mean Latency (ms) | Throughput (FPS) |
|---|---|---:|---:|---:|
| `augmentation` | PyTorch CUDA (FP16) | 640×640 | 21.23 | ~47.1 |
| `baseline` | PyTorch CUDA (FP16) | 640×640 | 22.05 | ~45.3 |
| `best_onnx` | ONNXRuntime | 640×640 | 73.66 | ~13.6 |
| `final` | PyTorch CUDA (FP16) | 800×800 | 24.94 | ~40.1 |
| `high_resolution` | PyTorch CUDA (FP16) | 960×960 | 26.56 | ~37.6 |


## 11. ONNX Export & Three-Tier Verification

The selected checkpoint was exported to **ONNX (opset 12)** with graph simplification:
```bash
python src/export_onnx.py --weights runs/detect/final/weights/best.pt --imgsz 800 --opset 12
```

### Three-Tier Verification:
1. **Tier 1 (Structural Audit):** Validated graph topological integrity with `onnx.checker.check_model`.
2. **Tier 2 (Execution Sanity):** Executed zero-crash test inference using `onnxruntime.InferenceSession`.
3. **Tier 3 (Output Parity):** Verified tensor dimension alignment and valid confidence ranges between PyTorch native and ONNXRuntime outputs.

---

## 12. Proposed Edge Deployment Architecture

```
[Current Project Deliverables]
  ├── PyTorch FP16 CUDA Profiling (NVIDIA RTX 3050 Laptop GPU)
  ├── Static-Graph ONNX Export (opset 12, simplified graph)
  ├── Three-Tier ONNX Verification (Graph integrity, runtime execution, output parity)
  └── ONNXRuntime FP32 Execution Benchmark
              │
              ▼
[Proposed Future Edge Pipeline (Robotics AMR)]
  ├── Edge Hardware: NVIDIA Jetson Orin / Xavier
  ├── Compilation: TensorRT FP16 Engine
  ├── Stream Preprocessing: 800×800 Letterboxing & Normalization
  ├── Post-Processing: Non-Maximum Suppression (IoU=0.45, Conf=0.25)
  └── Integration: Temporal multi-frame majority voting filter → ROS2 / Nav2 Costmap
```

---

## 13. How to Reproduce & Run

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Full Automated Overnight Pipeline
Runs dataset QA, trains 4 experiments, evaluates validation split, benchmarks latency, selects the best model, evaluates the held-out test split, exports ONNX, and populates README tables:
```bash
python run_all.py
```

### 3. Step-by-Step Commands
```bash
# Dataset QA & Deduplication Audit
python src/dataset_qa.py --grid
python src/audit_dedup.py

# Train individual experiment
python src/train.py --experiment baseline
python src/train.py --experiment augmentation
python src/train.py --experiment high_resolution
python src/train.py --experiment final

# Evaluate on Validation split (for model comparison)
python src/evaluate.py --weights runs/detect/final/weights/best.pt --split val --imgsz 800

# Evaluate on Test split (only for selected winning model)
python src/evaluate.py --weights runs/detect/final/weights/best.pt --split test --imgsz 800

# Latency Benchmark (FP16 on CUDA)
python src/benchmark.py --weights runs/detect/final/weights/best.pt --imgsz 800

# Export ONNX
python src/export_onnx.py --weights runs/detect/final/weights/best.pt --imgsz 800

# Visual Prediction Gallery
python src/visualize.py --weights runs/detect/final/weights/best.pt --source dataset/images/test
```

---

## 14. Repository Structure

```
SwiftRobotics_DoorDetection/
├── README.md                          # Technical submission report
├── WALKTHROUGH.md                     # Comprehensive technical walkthrough
├── requirements.txt                   # Pinned reproducible dependencies
├── .gitignore                         # Git exclusion rules
├── run_all.py                         # Master unbiased overnight pipeline
│
├── data/
│   ├── data.yaml                      # Dataset YAML configuration
│   └── raw/                           # Raw multi-source downloads
│
├── dataset/                           # Merged, deduplicated & stratified dataset
│   ├── images/ (train, val, test)
│   └── labels/ (train, val, test)
│
├── configs/
│   ├── baseline.yaml                  # Exp 1 Reference config
│   ├── augmentation.yaml              # Exp 2 Domain augmentation config
│   ├── high_resolution.yaml           # Exp 3 960px spatial resolution config
│   └── final.yaml                     # Exp 4 Combined candidate config
│
├── src/
│   ├── dataset_qa.py                  # Dataset QA and instance counting
│   ├── audit_dedup.py                 # Deduplication quality audit tool
│   ├── train.py                       # Training runner with custom overrides
│   ├── evaluate.py                    # Evaluation and metric calculation
│   ├── benchmark.py                   # Latency & FPS profiling tool (FP16 CUDA)
│   ├── export_onnx.py                 # ONNX export and 3-step validator
│   ├── visualize.py                   # Annotated prediction visualizer
│   ├── fill_results.py                # Automated README & CSV reporter
│   ├── merge_datasets.py              # Multi-source dataset normalizer
│   └── download_datasets.py           # Dataset downloader
│
├── results/
│   ├── experiment_results.csv         # Structured tabular summary
│   ├── dataset_qa_stats.json          # Dataset instance metrics
│   ├── dedup_audit_report.json        # Deduplication quality report
│   ├── dedup_sample_audit.jpg         # Visual duplicate inspection grid
│   ├── dataset_preview_*.jpg          # Split preview grids
│   ├── predictions/                   # Annotated detection samples
│   └── failure_analysis/              # Hard/failure case gallery
│
└── models/
    ├── best.pt                        # PyTorch model weights
    └── best.onnx                      # Exported production ONNX model
```

---

## 15. Future Improvements

1. **Temporal Filtering for Video Streams:** On live video feeds, a sliding-window temporal filter (e.g. 3-frame majority vote) reduces single-frame state flickering.
2. **Additional Glass & Low-Light Data:** Expanding training coverage on transparent doors and dim warehouse settings.
3. **TensorRT Compilation:** Compiling the static ONNX model to a TensorRT engine directly on edge devices (such as NVIDIA Jetson) for maximum FPS.
4. **Adaptive Confidence Handling:** Integrating a low-confidence threshold band ($0.25 - 0.40$) where the robot pauses or re-observes before committing to crossing a threshold.

---

## Reproducibility Environment

| Component | Specification |
|---|---|
| Python Version | 3.12.4 |
| Deep Learning Backend | PyTorch 2.5.1 + CUDA 12.1 |
| YOLO Framework | Ultralytics 8.4.130 |
| ONNX Runtime | ONNX 1.19.1 / ONNXRuntime 1.23.2 |
| Hardware Accelerator | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) |
| Random Seed | 42 (Fixed across all splits and initializations) |
