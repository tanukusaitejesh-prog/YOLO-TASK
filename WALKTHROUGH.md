# Technical Walkthrough: Door Open / Closed Detection Pipeline
### Swift Robotics — Junior AI Engineer Technical Task

---

## 1. Executive Summary & Problem Formulation

### Robotics Context & Objective
Autonomous mobile robots (AMRs) operating in dynamic indoor environments frequently interact with doorways. The perception system must detect doors and classify their traversability state in real time:
- **Traversable (`door_open`, Class 0):** Path planning layers can compute costmap clearance to navigate through the doorway threshold.
- **Non-traversable (`door_closed`, Class 1):** The robot must halt prior to collision, signal for access, or plan an alternate route.

### Engineering Philosophy
Rather than pursuing benchmark numbers on an arbitrary random split, this project prioritizes a **disciplined, end-to-end computer vision engineering methodology**:
1. **Multi-Source Dataset Engineering:** Aggregation of three independent sources, annotation normalization (including polygon-to-bounding-box geometric conversion), 256-bit average hash (`aHash`-style) deduplication, and source-stratified splitting.
2. **Controlled Factor-Group Experiments:** Hypotheses formulated to isolate baseline performance, domain-specific augmentations, resolution scaling under hardware-constrained batch sizing, and a combined candidate configuration.
3. **Strict Validation-Driven Selection:** All model comparisons and architectural decisions were made **strictly on the Validation split**. The held-out test split remained untouched until the final model was locked.
4. **Edge Deployment Readiness:** Native PyTorch FP16 CUDA profiling alongside static-graph ONNX (opset 12) export with three-tier verification (structural validation, execution sanity, and output tensor parity).

---

## 2. Dataset Engineering & Multi-Source Synthesis

### Source Aggregation & Normalization
To avoid overfitting to a single camera sensor or building aesthetic, three public object-detection datasets (under CC BY 4.0) were aggregated:

| Source Dataset | Raw Images | Retained | Duplicates Removed | Original Annotation Format | Normalization Applied |
|---|---:|---:|---:|---|---|
| `vikashs_1527` | 1,527 | 1,522 | 5 | `door-close`, `door-open` | Polygon segmentation masks converted to tight bounding boxes |
| `fiw_706` | 691 | 327 | 364 | `Door-Close`, `Door-Open` | Standard bounding boxes |
| `utfyu_116` | 294 | 294 | 0 | `door_close`, `door_open` | Standard bounding boxes |
| **Total** | **2,512** | **2,143** | **369 (14.7%)** | — | **Normalized to canonical `0: door_open`, `1: door_closed`** |

```
                              2,512 Raw Annotated Images
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
         369 Removed (14.7%)                         2,143 Clean Images (85.3%)
  (364 from fiw_706 surveillance bursts,                        │
    5 from vikashs_1527 identical files)           Source-Stratified Split (72% / 15% / 13%)
                                                                │
                                            ┌───────────────────┼───────────────────┐
                                            ▼                   ▼                   ▼
                                       Train (1,541)        Val (321)           Test (281)
                                      (924 O / 617 C)    (180 O / 141 C)     (178 O / 103 C)
```

### Technical Details of Data Engineering:
1. **Polygon-to-Bounding-Box Conversion:** The `vikashs_1527` dataset contained 10-coordinate polygon segmentation points. A spatial boundary parser extracted extreme coordinates $[(x_{\min}, y_{\min}), (x_{\max}, y_{\max})]$, calculated normalized center coordinates $(\text{cx}, \text{cy}, w, h)$, and verified spatial validity $[0.0, 1.0]$.
2. **256-Bit Average Hash (`aHash`-style) Deduplication:** An intensity-based $16\times16$ average hash was computed across all raw images. An audit (`results/dedup_audit_report.json`) confirmed that 364 of the 369 pruned images originated from `fiw_706` (high-frequency temporal burst frames from stationary surveillance cameras). Deduplication effectively removed near-identical burst frames while preserving distinct architectural door styles.
3. **Source-Stratified Splitting:** Images from each source were partitioned proportionally across train (72%), val (15%), and test (13%) splits following duplicate pruning.
4. **Instance Characteristics:** The final dataset contains one annotated door instance per image (2,143 images = 2,143 instances), characterizing single-door navigation scenarios rather than dense multi-object scenes.

---

## 3. Model Architecture & YOLO Fundamentals

### Architecture Overview: YOLOv8n
- **Parameters:** ~3,011,238 parameters (as reported by the Ultralytics model summary)
- **Computational Complexity:** ~8.2 GFLOPs at $640\times640$ resolution
- **Backbone:** C2f-based convolutional backbone utilizing cross-stage partial connections with split-and-merge gradient routing.
- **Neck:** PAN/FPN-style multi-scale feature aggregation pyramid producing feature maps across three spatial strides:
  - **P3 ($80\times80$ grid):** High spatial resolution for small/distant doorways.
  - **P4 ($40\times40$ grid):** Intermediate scale for standard hallway doors.
  - **P5 ($20\times20$ grid):** High semantic abstraction for close-up doors.
- **Head & Loss:** Anchor-free decoupled detection head separating bounding box coordinate regression from classification.

```
Input Image (3 × H × W)
       │
       ▼
[C2f Convolutional Backbone] ──> Multi-Scale Feature Extraction
       │
       ▼
[PAN/FPN-Style Feature Neck] ──> Multi-Scale Pyramids (P3, P4, P5)
       │
       ▼
[Anchor-Free Decoupled Head] ──┬──> Box Regression Branch (Task-Aligned DFL + CIoU Loss)
                               └──> Classification Branch (Binary Cross-Entropy Loss)
```

---

## 4. Controlled Experimental Design

Four experiments were designed under controlled conditions, varying one primary factor group at a time:

### Experiment 1 — Baseline Reference
- **Objective:** Establish a reference benchmark using standard YOLOv8n hyperparameters with COCO-pretrained weights.
- **Configuration:** `configs/baseline.yaml` (`imgsz: 640`, `batch: 16`, `lr0: 0.01`, default augmentations).

### Experiment 2 — Domain Augmentation
- **Hypothesis:** Targeted photometric and spatial augmentations (brightness jitter $\text{hsv\_v}=0.6$, rotation $\pm 5^\circ$, scale jitter $0.65$, shear $2.0$, mixup $0.1$) improve generalization to variable indoor corridor lighting and robot pitch/roll.
- **Configuration:** `configs/augmentation.yaml` (All learning rates, epochs, batch size, and resolution frozen at Baseline).

### Experiment 3 — High Spatial Resolution
- **Hypothesis:** Scaling input resolution from $640 \to 960\text{ px}$ preserves spatial features for distant or partially visible doors and tightens bounding box localization ($mAP@0.5:0.95$).
- **Configuration:** `configs/high_resolution.yaml` (`imgsz: 960`, `batch: 8`). *Batch size was adjusted to 8 to accommodate GPU VRAM constraints at higher resolution.*

### Experiment 4 — Combined Candidate
- **Hypothesis:** Combining the validated domain augmentations with a balanced intermediate resolution ($800\text{ px}$, `batch: 12`) achieves localization benefits while preserving real-time edge throughput.
- **Configuration:** `configs/final.yaml` (`imgsz: 800`, `batch: 12`, selected augmentations).

---

## 5. Validation-Driven Model Selection & Test Isolation

To maintain complete methodological integrity:
1. **Validation Comparison:** All candidate models were evaluated on the **Validation Split** ($N=321$ images).
2. **Model Selection Rule:** The winning model was selected by evaluating validation F1 score alongside localization strictness ($mAP@0.5:0.95$) subject to an engineering latency target ($\le 30\text{ ms}$).
3. **Structured Decision Logging:** The selection rationale is logged to [`results/model_selection_decision.json`](file:///C:/Users/saite/OneDrive/Desktop/SwiftR/results/model_selection_decision.json).
4. **Single Test Evaluation:** The held-out **Test Split** ($N=281$ images) was evaluated **exactly once** on the locked winning model to obtain the unbiased final performance.
5. **Post-Lock Failure Analysis:** Failure analysis was performed strictly after the final model was locked and test metrics were recorded; no subsequent hyperparameter adjustments were made using the test set.

### Selection Metric Discussion
F1 score ($F_1 = \frac{2 \cdot P \cdot R}{P + R}$) provides a balanced summary of precision and recall for this two-class detection task. In an operational mobile robotics system, false positives (`door_open` predicted when closed) risk collision, whereas false negatives (`door_closed` predicted when open) degrade navigation efficiency.

---

## 6. Grounded Error Taxonomy & Failure Modes

Visual inspection of difficult and low-confidence test predictions (`results/failure_analysis/failure_gallery.jpg`) identifies five observable failure modes:

| Observed Failure Mode | Visual Pattern | Likely Root Cause | Potential Future Improvement |
|---|---|---|---|
| **Low Illumination / Backlighting** | Missed closed door in dim hallways or high-contrast backlighting | Low contrast between door panel and doorframe | Targeted contrast/illumination augmentations |
| **Partial Occlusion** | False state prediction when obstacles partially cover the door | Foreground objects break continuous door panel edge geometry | Training images with realistic foreground occlusions |
| **Small / Distant Door** | Lower detection confidence when door is viewed from far down a hallway | Object occupies a small proportion of the image frame | Higher input resolution ($800\text{px}$) or multi-scale inference |
| **Glass / Specular Reflection** | Transparent or glossy doors misclassified | Specular reflections mimic open pathway geometry | Targeted collection and annotation of reflective doors |
| **Ambiguous State (Ajar)** | Low confidence on doors open by only a slight angle | Subtle visual separation between door edge and jamb | Temporal smoothing over consecutive frames in video |

---

## 7. Edge Deployment & Hardware Benchmarking

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
  └── Integration: Temporal multi-frame majority voting filter $\to$ ROS2 / Nav2 Costmap
```

### Three-Tier ONNX Verification:
- **Tier 1 (Structural Audit):** Validated graph topological integrity with `onnx.checker.check_model`.
- **Tier 2 (Execution Sanity):** Executed zero-crash test inference using `onnxruntime.InferenceSession`.
- **Tier 3 (Output Parity):** Verified tensor dimension alignment and valid confidence ranges between PyTorch native and ONNXRuntime outputs.

### Latency Profiling Protocol:
- **PyTorch GPU Benchmark:** Native PyTorch running in **FP16 half-precision on CUDA** (10 warmup iterations, 100 timed runs).
- **ONNXRuntime Benchmark:** Static graph execution in **FP32** via ONNXRuntime engine.
- **Hardware:** Dedicated NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).

---

## 8. Summary of Project Deliverables

| Deliverable | Path | Purpose |
|---|---|---|
| **Technical Report** | [`README.md`](file:///C:/Users/saite/OneDrive/Desktop/SwiftR/README.md) | Primary submission report with auto-populated experimental tables |
| **Master Pipeline** | [`run_all.py`](file:///C:/Users/saite/OneDrive/Desktop/SwiftR/run_all.py) | Automated script running QA, training, val selection, test eval, and reporting |
| **QA Audit Tool** | [`src/dataset_qa.py`](file:///C:/Users/saite/OneDrive/Desktop/SwiftR/src/dataset_qa.py) | Audits images, instances, and outputs preview grids |
| **Dedup Auditor** | [`src/audit_dedup.py`](file:///C:/Users/saite/OneDrive/Desktop/SwiftR/src/audit_dedup.py) | Audits the 369 removed duplicates and outputs visual grid |
| **Export & Validator** | [`src/export_onnx.py`](file:///C:/Users/saite/OneDrive/Desktop/SwiftR/src/export_onnx.py) | Exports to static ONNX and performs 3-tier validation |
| **Latency Profiler** | [`src/benchmark.py`](file:///C:/Users/saite/OneDrive/Desktop/SwiftR/src/benchmark.py) | Profiles FP16 CUDA and ONNXRuntime latency |
| **Visualizer** | [`src/visualize.py`](file:///C:/Users/saite/OneDrive/Desktop/SwiftR/src/visualize.py) | Generates test prediction grids and failure analysis galleries |
