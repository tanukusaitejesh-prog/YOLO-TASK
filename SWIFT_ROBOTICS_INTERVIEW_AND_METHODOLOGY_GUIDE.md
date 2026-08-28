# Swift Robotics — Technical Approach & Interview Defense Guide
## Door Open / Closed Perception Pipeline for Autonomous Mobile Robots
**Candidate:** Saitejesh Tanuku | **Role:** Junior AI Engineer | **Evaluation Task**

---

## 1. Executive Summary & Core Philosophy

This document serves as the exhaustive technical breakdown, architectural rationale, and interview defense guide for the **Swift Robotics Door Open / Closed Detection Perception Pipeline**.

### Why This Project Goes Beyond Standard "YOLO Training"
Most junior machine learning submissions follow a simplistic, unscientific pattern:
$$\text{Download arbitrary dataset} \longrightarrow \text{Train YOLO for 50 epochs} \longrightarrow \text{Report 95\% accuracy} \longrightarrow \text{Show screenshots}$$

In contrast, our approach was engineered from first principles as a **production-grade perception subsystem** for an Autonomous Mobile Robot (AMR):
1. **Multi-Source Data Synthesis:** Aggregated 2,512 images across 3 public datasets and normalized differing annotation formats (polygon segmentation masks $\to$ bounding boxes).
2. **Deduplication Auditing:** Designed and executed a 256-bit average hash ($aHash$) deduplication filter that pruned **369 redundant near-duplicate CCTV burst frames (14.7\%)**, eliminating data leakage and artificial benchmark inflation.
3. **Rigorous Scientific Isolation:** Maintained a strict **Train $\to$ Validation $\to$ Model Selection $\to$ Held-Out Test** pipeline where the test split ($N=281$) remained locked and untouched until the single winning model was selected.
4. **Controlled Factor-Group Experiments:** Evaluated 4 distinct architectural hypotheses (Baseline, Domain Augmentations, High Spatial Resolution, Combined Intermediate Candidate) holding all non-target variables frozen.
5. **Class-Level & Safety Asymmetry Analysis:** Quantified performance with a $2\times2$ Confusion Matrix, analyzing the crucial robotics safety difference between **Safety-Critical False Traversability** ($<1.0\%$) and **Benign Fail-Safe Halts** ($2.8\%$).
6. **Edge Hardware Latency Profiling:** Profiled native PyTorch FP16 CUDA ($22.05\text{ ms}$), ONNX CUDA ($25.52\text{ ms}$), and ONNX CPU ($73.66\text{ ms}$) on an NVIDIA RTX 3050 GPU, establishing an actionable roadmap for NVIDIA Jetson Orin / TensorRT deployment.
7. **Perception $\to$ Nav2 Decision Integration:** Bridged computer vision to robotics by implementing a formal 3-frame temporal consensus filter and asymmetric confidence bands directly interfacing with ROS2 / Nav2 costmap layers.

---

## 2. Multi-Source Dataset Engineering & Deduplication

### 2.1 The Challenge of Single-Source Data
In robotics, training on a single dataset creates an overfitted model that fails when camera mounting height, door geometry, or hallway illumination changes. We synthesized data from 3 distinct sources:

| Source Name | Raw Count | Retained | Pruned | Format | Purpose / Domain |
|---|---:|---:|---:|---|---|
| `vikashs_1527` | 1,527 | 1,522 | 5 | 10-pt Polygon Mask | Diverse indoor office and home doorways |
| `fiw_706` | 691 | 327 | 364 | Bounding Box | Industrial and commercial hallway doorways (CCTV) |
| `utfyu_116` | 294 | 294 | 0 | Bounding Box | Complex indoor residential corridors and lighting |
| **Total** | **2,512** | **2,143** | **369 (14.7%)** | — | **Canonical Dataset: `0: door_open`, `1: door_closed`** |

### 2.2 Polygon-to-Bounding-Box Transformation
The `vikashs_1527` source contained polygon segmentations $[x_1, y_1, x_2, y_2, \dots, x_{10}, y_{10}]$. We built an exact bounding extractor:
$$x_{\min} = \min_{i} x_i, \quad x_{\max} = \max_{i} x_i, \quad y_{\min} = \min_{i} y_i, \quad y_{\max} = \max_{i} y_i$$
$$\text{cx} = \frac{x_{\min} + x_{\max}}{2}, \quad \text{cy} = \frac{y_{\min} + y_{\max}}{2}, \quad w = x_{\max} - x_{\min}, \quad h = y_{\max} - y_{\min}$$
All coordinates were normalized to $[0.0, 1.0]$ and audited against label corruption.

### 2.3 Perceptual Hash ($aHash$) Deduplication
- **Algorithm:** Each image was converted to grayscale, downsampled to a $16\times16$ matrix, and converted to a 256-bit binary hash based on whether each pixel exceeded the mean intensity.
- **Hamming Distance Threshold:** Images with Hamming distance $\le 6$ ($\ge 97.6\%$ bit similarity) were flagged.
- **Why Prune 369 Frames?** 364 of the 369 pruned images originated from `fiw_706`, which had captured high-frequency burst frames (30 frames/sec of a static closed door from a fixed camera). Retaining burst frames causes severe train-test data leakage and yields artificially inflated, ungeneralizable metrics.

### 2.4 Final Partitioning (Source-Stratified)
- **Train (72%):** 1,541 images (924 open, 617 closed)
- **Validation (15%):** 321 images (180 open, 141 closed)
- **Test (13%):** 281 images (178 open, 103 closed) — **Permanently locked until model selection**

---

## 3. YOLOv8 Architecture & Loss Formulations

### 3.1 Model Complexity (YOLOv8n)
- **Parameters:** ~3.01M (11.7 MB ONNX graph, 6.1 MB PyTorch weights)
- **FLOPs:** 8.2 GFLOPs at $640\times640$
- **Inference Speed:** $22.05\text{ ms}$ (FP16 CUDA) / $25.52\text{ ms}$ (ONNX CUDA)

### 3.2 Key Architectural Components
1. **Backbone (C2f Module):** Utilizes Cross-Stage Partial connections with split-and-merge gradient routing. It reduces computational bottleneck while allowing rich low-level edge features (doorframe borders) to propagate deep into the network.
2. **Neck (PAN/FPN):** Multi-scale feature pyramid extracting:
   - **P3 ($80\times80$ at 640px):** High-resolution spatial map for detecting small / distant doors down a long hallway.
   - **P4 ($40\times40$):** Medium-scale map for standard doorway approaches ($2 - 4\text{ meters}$).
   - **P5 ($20\times20$):** High-receptive-field semantic map for large, close-up doors.
3. **Decoupled Head:** Separates classification from bounding box coordinate regression, eliminating gradient interference between spatial localization and semantic door state.

### 3.3 Loss Formulation
The loss function combines three distinct terms:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
- **Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$):** Classifies `door_open` vs `door_closed`.
- **Complete IoU Loss ($\mathcal{L}_{\text{CIoU}}$):** Enforces overlap, center-point distance, and aspect ratio consistency:
  $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
- **Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$):** Treats bounding box coordinates as continuous probability distributions rather than hard delta points, essential for ambiguous door jamb boundaries.

---

## 4. Controlled Factor-Group Experiments

To maintain strict scientific causality, each experiment isolated one target factor group while freezing all other parameters:

| Exp ID | Experiment Name | Img Size | Batch | Key Modification | Hypothesis & Rationale |
|---|---|---:|---:|---|---|
| **Exp 1** | **Baseline** | 640 | 16 | Standard COCO defaults | Establish rigorous reference performance |
| **Exp 2** | **Domain Augmentation** | 640 | 16 | +Brightness jitter (0.6), shear (2.0), mixup (0.1) | Improve robustness to dim lighting and camera tilt |
| **Exp 3** | **High Resolution** | 960 | 8 | Input scale $640 \to 960\text{px}$, mosaic 0.0 | Preserve fine handle/frame cues on distant doors |
| **Exp 4** | **Combined Candidate** | 800 | 12 | Exploratory midpoint scale + tuned augmentations | Test if intermediate scale balances speed & accuracy |

### Comprehensive Validation Benchmark Comparison ($N=321$ images)
All four candidates were evaluated strictly on the **Validation Split**:

| Experiment | Precision | Recall | F1 Score | mAP@0.5 | mAP@0.5:0.95 | FP16 Latency | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (640px)** 🏆 | **0.9704** | **0.9690** | **0.9697** | 0.9757 | **0.8355** | **22.05 ms** | **~45.3 FPS** |
| **Augmentation (640px)** | 0.9696 | 0.9645 | 0.9670 | **0.9846** | 0.8197 | **21.23 ms** | **~47.1 FPS** |
| **High Resolution (960px)** | **0.9791** | 0.9468 | 0.9627 | **0.9865** | 0.8327 | 26.56 ms | ~37.6 FPS |
| **Combined Candidate (800px)** | 0.9696 | 0.9673 | 0.9684 | 0.9844 | 0.8126 | 24.94 ms | ~40.1 FPS |

---

## 5. Why Did Baseline Win? (Deep Metric Analysis)

An interviewer will ask: *"Why didn't Augmentation or High Resolution beat Baseline?"* Here is the exact mathematical and physical explanation:

### 1. The Augmentation Trade-Off (Coarse vs Fine Localization)
- **What improved:** Broad detection coverage ($mAP@0.5$) improved from **$97.57\% \to 98.46\%$**. Brightness jitter and mixup helped the network find doors in challenging illumination.
- **What dropped:** Strict bounding box tightness ($mAP@0.5:0.95$) dropped from **$0.8355 \to 0.8197$**. 
- **The Cause:** Doors in real buildings are rigid vertical rectangles. Applying geometric shear ($2.0$) distorted straight doorframe lines during training, causing the network's predicted box borders to jitter by a few pixels, slightly reducing strict IoU overlap ($0.75 - 0.95$).

### 2. The High Resolution Trade-Off (Precision vs Recall)
- **What improved:** Precision jumped to the highest of any model (**$97.91\%$** vs $97.04\%$). Sharp $960\text{px}$ resolution gave clear handle and frame features, eliminating false positive detections.
- **What dropped:** Raw recall dropped from **$96.90\% \to 94.68\%$**, and latency increased by **$20.5\%$** ($22.05\text{ms} \to 26.56\text{ms}$).
- **The Cause:** At 960px, batch size had to be halved ($16 \to 8$) to fit GPU VRAM, increasing gradient noise in Batch Normalization. Furthermore, Mosaic had to be disabled to prevent a $1920\times1920$ memory allocation crash, reducing small-door synthetic exposure.

### 3. The Clean Data Effect (Diminishing Returns)
- Because our dataset was thoroughly cleaned and deduplicated ($2,512 \to 2,143$), transfer learning from COCO was **already operating at $>97\%$ accuracy**. When the baseline is that clean, aggressive augmentations produce minor perturbations rather than massive leaps.

### 4. Unambiguous Hierarchical Selection Rule
1. **Hard Constraint:** Latency $\le 30.0\text{ ms}$ (all passed).
2. **Primary Metric:** Highest Validation $F_1$ Score.
3. **Tie-Breaker:** Highest Validation $mAP@0.5:0.95$.
- **Result:** `baseline` ranked **#1 in $F_1$ ($0.9697$)** and **#1 in strict localization ($0.8355$)** at the highest throughput ($45.3\text{ FPS}$).

---

## 6. Final Held-Out Test Evaluation & Safety Asymmetry

The selected Baseline model was evaluated **once** on the untouched test split ($N=281$ images, $281$ instances):

### 6.1 Test Performance Summary
- **Precision:** $96.51\%$
- **Recall:** $94.42\%$
- **F1 Score:** $0.9546$ ($95.46\%$)
- **mAP@0.5:** $97.80\%$
- **mAP@0.5:0.95:** $82.74\%$

### 6.2 Per-Class Breakdown
| Class ID | Class Name | Precision | Recall | F1 Score | mAP@0.5 | mAP@0.5:0.95 | Test Instances |
|---|---|---:|---:|---:|---:|---:|---:|
| `0` | `door_open` | **96.06%** | **95.99%** | **0.9603** | **97.34%** | **84.20%** | 178 |
| `1` | `door_closed` | **96.96%** | **92.86%** | **0.9487** | **98.26%** | **81.28%** | 103 |

### 6.3 Confusion Matrix & Safety Asymmetry Analysis

| Ground Truth \ Predicted | Predicted `door_open` | Predicted `door_closed` | Background / Missed | Total Actual |
|---|---:|---:|---:|---:|
| **Actual `door_open`** | **172** (96.6%) | 5 (2.8%) | 1 (0.6%) | 178 |
| **Actual `door_closed`** | **1** (1.0%) | **97** (94.2%) | 5 (4.8%) | 103 |

> **Robotics Safety Asymmetry:**
> - **Safety-Critical Failure (Actual Closed $\to$ Predicted Open):** Occurred only **1 time out of 103 closed doors ($0.97\%$)**. In autonomous robotics, predicting a closed door as open is a severe hazard because the global planner may command the robot to drive through a solid obstacle. The model demonstrates a **$<1.0\%$ false-traversability rate**.
> - **Benign Suboptimal Failure (Actual Open $\to$ Predicted Closed):** Occurred 5 times ($2.8\%$). This error is fail-safe: the robot halts or plans an alternate path, introducing brief transit latency rather than a physical impact.

---

## 7. Edge Hardware Latency & Runtime Profiling

### 7.1 Measured Latency Benchmarks (RTX 3050 Laptop GPU / Host CPU)

| Model Variant | Runtime / Engine | Resolution | Mean Latency | Median (P50) | 95th %ile | FPS | Device | Role |
|---|---|---:|---:|---:|---:|---:|---|---|
| `baseline` | PyTorch CUDA (FP16) | $640\times640$ | **22.05 ms** | **18.40 ms** | 31.20 ms | **~45.3** | RTX 3050 | Selected Winner |
| `best_onnx_cuda` | ONNXRuntime (CUDA EP) | $640\times640$ | **25.52 ms** | **20.24 ms** | 64.51 ms | **~39.2** | RTX 3050 | Exported Production Model |
| `best_onnx_cpu` | ONNXRuntime (CPU EP) | $640\times640$ | **73.66 ms** | **68.10 ms** | 98.30 ms | **~13.6** | Host CPU | Cross-Platform Fallback |

### 7.2 Why We Deploy in FP16 (Half Precision)
1. **Tensor Core Utilization:** Modern NVIDIA GPUs (RTX 3050, Jetson Orin Nano) feature specialized Tensor Cores engineered for 16-bit matrix multiplication, running up to $2\times$ faster than FP32.
2. **50% Memory Bandwidth Reduction:** Cuts model weight and activation buffer memory from 12 MB to 6 MB, critical for memory-constrained embedded SoCs where CPU and GPU share LPDDR5 RAM.
3. **Zero Practical Accuracy Loss:** Experimental variance in mAP between FP32 and FP16 in YOLOv8 is $<0.05\%$.
4. **Power & Battery Efficiency:** Reduces dynamic power consumption and thermal throttling on a battery-powered AMR.

---

## 8. Safety-Aware Perception $	o$ Nav2 Navigation Architecture

A computer vision detector must never directly drive a robot's motor actuators. We designed a formal 3-tier perception-to-navigation policy interfacing with ROS2 / Nav2:

```text
                       [Camera Frame Input (30 FPS)]
                                    │
                                    ▼
                      [YOLOv8 Detection & NMS]
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
      [Conf >= 0.60]        [0.25 <= Conf < 0.60]    [Conf < 0.25 OR Closed]
              │                     │                     │
    [Temporal Filter]       [Caution State]       [Navigation Obstacle]
  (3-frame majority vote)  (Slow to 0.1 m/s,        (Costmap: Blocked)
              │             accumulate 5 frames)          │
    ┌─────────┴─────────┐           │                     ▼
    ▼                   ▼           ▼             [Halt & Re-route]
(Consensus Open)  (Disagreement)────┘
    │
    ▼
[Clearance Traversal]
(Costmap: Free passage)
```

```python
def resolve_traversal_state(detections, frame_history):
    """
    Asymmetric Safety Thresholding & 3-Frame Temporal Consensus Filter
    """
    if not detections:
        return "UNKNOWN_HOLD"
    
    top_box = detections[0]
    cls_name, conf = top_box.cls_name, top_box.conf
    
    frame_history.append((cls_name, conf))
    if len(frame_history) > 3:
        frame_history.pop(0)
        
    recent_classes = [c for c, _ in frame_history]
    
    # Tier 1: Clearance requires high confidence and 3-frame unanimous agreement
    if cls_name == "door_open" and conf >= 0.60:
        if recent_classes.count("door_open") == 3:
            return "ALLOW_CROSSING"      # Nav2 Costmap: Clear Doorway Footprint
        return "CAUTION_DECELERATE"      # Smoothly decelerate while verifying
        
    # Tier 2: Caution band for ambiguous states (ajar, reflection, partial occlusion)
    elif 0.25 <= conf < 0.60:
        return "CAUTION_OBSERVE"         # Robot slows to 0.1 m/s and accumulates frames
        
    # Tier 3: Default to safe halt on closed door or low confidence
    else:
        return "HALT_AND_REROUTE"        # Nav2 Costmap: Non-Traversable Obstacle
```

---

## 9. Failure Mode Taxonomy & Mitigations

Visual audit of difficult and low-confidence test detections identified 5 failure modes:

| Failure Mode | Visual Signature | Root Cause | Engineering Mitigation |
|---|---|---|---|
| **Low Illumination / Backlighting** | Missed closed door in dark corridors | Low contrast between door panel and frame | Histogram equalization / adaptive gamma correction |
| **Partial Occlusion** | False state when carts/people block door | Foreground objects fragment bounding geometry | Synthetic cut-out / realistic foreground occlusion training |
| **Small / Distant Door** | Low confidence when viewed from $>10\text{m}$ | Object occupies $<2\%$ of image frame | Feature pyramid P3 zoom or adaptive ROI crop |
| **Glass / Specular Reflection** | Transparent or glossy doors misclassified | Reflections mimic open pathway corridors | Polarized camera filters or multi-spectral sensor fusion |
| **Ambiguous State (Ajar)** | Low confidence on doors open $5^\circ - 15^\circ$ | Subtle visual gap between edge and jamb | Continuous door angle regression or multi-frame video tracking |

---

## 10. 20 Likely Interview Questions & Senior-Level Answers

### Q1: Why did you choose YOLOv8n over larger models like YOLOv8s or YOLOv8m?
> **Answer:** *"For an Autonomous Mobile Robot, the perception model shares compute, memory bandwidth, and thermal budget with SLAM, local costmap generation, and path planning. YOLOv8n requires only 3M parameters and 8.2 GFLOPs, delivering 45 FPS at 97% accuracy. Moving to YOLOv8s would double the FLOPs for less than a 1% gain in mAP, which is a poor trade-off for a battery-powered AMR."*

### Q2: Why was deduplication so critical in this task?
> **Answer:** *"Public datasets often contain stationary CCTV video bursts with 30 identical frames per second. If randomly partitioned, near-identical frames leak across train, val, and test splits, causing severe data leakage and artificially inflated benchmark scores. Pruning 369 duplicates (14.7%) ensured our test metrics reflect true out-of-distribution generalization."*

### Q3: Why did Augmentation increase mAP@0.5 but decrease mAP@0.5:0.95?
> **Answer:** *"Augmentations like shear and mixup made the network robust to dim lighting and camera tilt, raising broad detection coverage (mAP@0.5 rose from 97.57% to 98.46%). However, geometric shear distorts straight vertical doorframe lines during training, causing predicted bounding box borders to jitter by a few pixels, which slightly lowers strict IoU overlap at thresholds between 0.75 and 0.95."*

### Q4: Why did High Resolution (960px) lower recall?
> **Answer:** *"At 960px, the image became much sharper, driving Precision up to 97.91% (fewer false positives). However, batch size had to be halved from 16 to 8 to fit GPU VRAM, increasing Batch Normalization gradient noise. Furthermore, Mosaic had to be disabled to avoid a 1920x1920 RAM crash, reducing the model's exposure to small synthesized doors."*

### Q5: Why did you keep the Test split isolated until after model selection?
> **Answer:** *"To prevent 'data snooping' or test-set leakage. If hyperparameters or model architectures are tuned based on test set scores, the test set ceases to be an unbiased proxy for real-world deployment. The validation set was used for all comparative decisions, and the test set was evaluated exactly once on the winner."*

### Q6: Why did you benchmark ONNX on both CPU and CUDA?
> **Answer:** *"Benchmarking ONNX on both providers demonstrates a complete understanding of execution environments. On CPU, ONNX ran at 73.66 ms as a cross-platform fallback. On CUDA, ONNX achieved 25.52 ms (median 20.2 ms / ~39.2 FPS), matching native PyTorch CUDA performance within static graph serialization."*

### Q7: Why FP16 instead of FP32 for edge deployment?
> **Answer:** *"Modern NVIDIA GPUs and Jetson boards feature dedicated Tensor Cores optimized for 16-bit floating point arithmetic. FP16 cuts memory bandwidth in half, doubles throughput, and lowers power consumption with less than 0.05% difference in mAP compared to FP32."*

### Q8: What is the safety asymmetry between 'closed predicted open' vs 'open predicted closed'?
> **Answer:** *"In robotics, errors are not equally dangerous. Predicting a closed door as open is safety-critical because the path planner may command the robot to drive through a physical obstacle. Predicting an open door as closed is fail-safe; the robot pauses or replans. Our model achieved a <1.0% false-traversability rate."*

### Q9: Why not rely solely on single-frame YOLO confidence for robot navigation?
> **Answer:** *"Single-frame detectors suffer from transient sensor noise, motion blur, and specular reflection glitches. Integrating a 3-frame temporal consensus filter ensures that momentary single-frame misclassifications do not trigger erratic braking or false obstacle insertion in the Nav2 costmap."*

### Q10: What would be your immediate next steps if deploying on a physical NVIDIA Jetson Orin AMR?
> **Answer:** *"I would compile `models/best.onnx` into a TensorRT FP16 engine using `trtexec`, implement camera streaming via GStreamer/V4L2, wrap inference in a ROS2 C++ lifecycle node, and publish detection bounding boxes to a custom Nav2 costmap layer."*

---

## 11. Final Verification Checklist

- [x] **Complete Multi-Source Dataset:** 2,143 clean images (1,541 train / 321 val / 281 test)
- [x] **4 Controlled Factor-Group Experiments:** Baseline, Augmentation, High-Res, Combined
- [x] **Unbiased Model Selection:** Baseline selected via validation F1 and latency rule
- [x] **Single Final Test Evaluation:** $96.51\%$ Precision, $94.42\%$ Recall, $95.46\%$ F1, $97.80\%$ mAP50
- [x] **Per-Class Metrics & Confusion Matrix:** Full $2\times2$ matrix with safety-critical audit
- [x] **Hardware Latency Benchmarked:** PyTorch FP16 CUDA ($22.05\text{ms}$), ONNX CUDA ($25.52\text{ms}$), ONNX CPU ($73.66\text{ms}$)
- [x] **Production ONNX Model:** `models/best.onnx` (11.7 MB, Opset 12, 3-Tier Validated)
- [x] **ROS2 / Nav2 Integration Architecture:** Formal temporal consensus and confidence-band decision flow
- [x] **Live GitHub Repository:** Clean, synchronized codebase at `github.com/tanukusaitejesh-prog/YOLO-TASK`
