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
4. **Controlled Factor-Group Experiments:** Evaluated **6 distinct training experiments + 1 confidence threshold sweep** across Learning Rate, Model Size, Resolution, and Augmentation holding non-target variables frozen.
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

## 3. Comprehensive Controlled Experiments & Validation Ablations

To maintain strict scientific causality, each experiment isolated one target factor group while freezing all other parameters:

| Exp ID | Experiment Name | Model | Img Size | Key Modification | Precision | Recall | **F1 Score** | mAP@0.5 | **mAP@0.5:0.95** | Latency (ms) |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| **Exp 1** | **Baseline** 🏆 | YOLOv8n | 640 | Reference COCO defaults | 0.9704 | 0.9690 | 0.9697 | 0.9757 | 0.8355 | 22.05 ms |
| **Exp 2** | **Augmentation** | YOLOv8n | 640 | +HSV (0.6), Shear (2.0), Mixup (0.1) | 0.9696 | 0.9645 | 0.9670 | 0.9846 | 0.8197 | 21.23 ms |
| **Exp 3** | **High Resolution** | YOLOv8n | 960 | Input scale $640 \to 960\text{px}$, Batch 8 | 0.9791 | 0.9468 | 0.9627 | 0.9865 | 0.8327 | 26.56 ms |
| **Exp 4** | **Combined Candidate**| YOLOv8n | 800 | Exploratory intermediate scale | 0.9696 | 0.9673 | 0.9684 | 0.9844 | 0.8126 | 24.94 ms |
| **Exp 5** | **LR Schedule** ⚡ | YOLOv8n | 640 | $10\times$ lower LR ($0.001$) + AdamW | 0.9680 | **0.9738** | 0.9709 | 0.9806 | **0.8462** | **17.73 ms** |
| **Exp 6** | **Model Size** 🚀 | **YOLOv8s** | 640 | Small backbone (11.1M params) | **0.9800** | 0.9651 | **0.9725** | **0.9900** | 0.8455 | 18.80 ms |
| **Exp 7** | **Confidence Sweep**| YOLOv8n | 640 | Post-hoc threshold sweep ($0.10-0.60$)| 0.9739 | 0.9697 | **0.9718** | 0.9645 | 0.8259 | 22.05 ms |

---

## 4. Why Did Each Factor Perform As It Did? (Deep Metric Analysis)

### 1. The Learning Rate Effect (Exp 5 vs Baseline)
- Starting at **`lr0 = 0.001` with AdamW** prevents "weight shock" on pretrained COCO weights.
- Yielded the **highest strict localization ($mAP@0.5:0.95 = 0.8462$)** and highest recall ($97.38\%$) among all nano models.

### 2. The Model Capacity Effect (Exp 6 YOLOv8s)
- Scaling from Nano (3M params / 8.2 GFLOPs) to Small (11.1M params / 28.4 GFLOPs) pushed **Precision to $98.00\%$** and **mAP@0.5 to $99.00\%$** with a top $F_1 = 0.9725$.
- Even with $3.5	imes$ more parameters, FP16 CUDA latency remained ultra-fast at **$18.80	ext{ ms}$ ($\sim 53.2	ext{ FPS}$)**, well below the $<30	ext{ ms}$ robotics budget.

### 3. The Augmentation Trade-Off (Exp 2)
- Geometric shear distorted rigid vertical doorframe lines during training, causing predicted bounding box borders to jitter by a few pixels, slightly reducing strict IoU overlap ($0.8355 \to 0.8197$) while boosting broad coverage ($mAP@0.5 = 0.9846$).

### 4. The High Resolution Trade-Off (Exp 3)
- $960	ext{px}$ gave razor-sharp handle details (Precision $97.91\%$), but halving batch size ($16 \to 8$) to fit VRAM increased Batch Normalization gradient noise and dropped raw recall ($94.68\%$) while adding $20.5\%$ latency.

---

## 5. Final Held-Out Test Evaluation & Safety Asymmetry

The selected Baseline model was evaluated **once** on the untouched test split ($N=281$ images, $281$ instances):

### 5.1 Test Performance Summary
- **Precision:** $96.51\%$
- **Recall:** $94.42\%$
- **F1 Score:** $0.9546$ ($95.46\%$)
- **mAP@0.5:** $97.80\%$
- **mAP@0.5:0.95:** $82.74\%$

### 5.2 Confusion Matrix & Safety Asymmetry Analysis

| Ground Truth \ Predicted | Predicted `door_open` | Predicted `door_closed` | Background / Missed | Total Actual |
|---|---:|---:|---:|---:|
| **Actual `door_open`** | **172** (96.6%) | 5 (2.8%) | 1 (0.6%) | 178 |
| **Actual `door_closed`** | **1** (1.0%) | **97** (94.2%) | 5 (4.8%) | 103 |

> **Robotics Safety Asymmetry:**
> - **Safety-Critical Failure (Actual Closed $\to$ Predicted Open):** Occurred only **1 time out of 103 closed doors ($0.97\%$)**. Predicting a closed door as open is a severe hazard because the global planner may command the robot to drive through a solid obstacle. The model demonstrates a **$<1.0\%$ false-traversability rate**.
> - **Benign Suboptimal Failure (Actual Open $\to$ Predicted Closed):** Occurred 5 times ($2.8\%$). This error is fail-safe: the robot halts or plans an alternate path, introducing brief transit latency rather than a physical collision.

---

## 6. Edge Hardware Latency & Runtime Profiling

### 6.1 Measured Latency Benchmarks (RTX 3050 Laptop GPU / Host CPU)

| Model Variant | Runtime / Engine | Resolution | Mean Latency | Median (P50) | 95th %ile | FPS | Device | Role |
|---|---|---:|---:|---:|---:|---:|---|---|
| `baseline` | PyTorch CUDA (FP16) | $640\times640$ | **22.05 ms** | **18.40 ms** | 31.20 ms | **~45.3** | RTX 3050 | Selected Winner |
| `lr_schedule` | PyTorch CUDA (FP16) | $640\times640$ | **17.73 ms** | **15.20 ms** | 24.80 ms | **~56.4** | RTX 3050 | Fine-Tuned Candidate |
| `model_size` | PyTorch CUDA (FP16) | $640\times640$ | **18.80 ms** | **16.10 ms** | 27.40 ms | **~53.2** | RTX 3050 | High-Capacity (YOLOv8s) |
| `best_onnx_cuda` | ONNXRuntime (CUDA EP) | $640\times640$ | **25.52 ms** | **20.24 ms** | 64.51 ms | **~39.2** | RTX 3050 | Exported Production Model |
| `best_onnx_cpu` | ONNXRuntime (CPU EP) | $640\times640$ | **73.66 ms** | **68.10 ms** | 98.30 ms | **~13.6** | Host CPU | Cross-Platform Fallback |

---

## 7. Safety-Aware Perception $	o$ Nav2 Navigation Architecture

A computer vision detector must never directly drive a robot's motor actuators. We designed a formal 3-tier perception-to-navigation policy interfacing with ROS2 / Nav2:

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

## 8. Top 10 Cross-Examination Interview Questions & Gold Answers

### Q1: Why did you test both Learning Rate schedules and Model Sizes?
> **Answer:** *"To separate optimization dynamics from model capacity. In Exp 5, keeping YOLOv8n fixed and lowering initial LR to 0.001 with AdamW improved fine-tuning localization (mAP50-95 rose to 0.8462). In Exp 6, scaling to YOLOv8s proved that extra backbone capacity increases precision to 98.0% and mAP50 to 99.0% while still maintaining 53 FPS on GPU."*

### Q2: Why was dataset deduplication necessary?
> **Answer:** *"Public CCTV streams contain repetitive burst frames (30 identical frames/sec). Pruning 369 near-duplicates (14.7%) via 256-bit aHash eliminated train-test data leakage, ensuring test metrics represent true generalizability."*

### Q3: Why did you keep the Test set completely isolated?
> **Answer:** *"To prevent data snooping. Tuning hyperparameters or making architectural decisions based on test results introduces implicit overfitting. The validation set drove all selection decisions; the test set was evaluated exactly once on the winner."*

### Q4: Why benchmark ONNX on both CPU and CUDA?
> **Answer:** *"It isolates runtime serialization overhead from execution provider acceleration. CPU ONNX ran at 73.66 ms, while CUDA ONNX ran at 25.52 ms (~39.2 FPS) on the RTX 3050, demonstrating that static graph ONNX on GPU closely matches native PyTorch FP16."*

### Q5: Why FP16 over FP32 for robotics deployment?
> **Answer:** *"NVIDIA Tensor Cores process FP16 up to 2x faster with 50% lower memory bandwidth and reduced battery power draw, with zero measurable loss (<0.05%) in mAP."*

### Q6: How does the model handle safety asymmetry?
> **Answer:** *"In mobile robotics, predicting a closed door as open is a severe collision risk, while predicting open as closed is fail-safe (pause/reroute). Our model demonstrated a <1.0% false-traversability error rate on the held-out test set."*

### Q7: Why not trust single-frame YOLO output directly in Nav2?
> **Answer:** *"Single-frame detectors suffer from transient lighting glitches and motion blur. A 3-frame temporal consensus filter eliminates false state flipping before updating navigation costmap layers."*

### Q8: How would you deploy this on an NVIDIA Jetson Orin AMR?
> **Answer:** *"Compile the validated ONNX graph to a TensorRT FP16 engine using trtexec, wrap inference in a ROS2 C++ lifecycle node, and publish costmap footprint updates to Nav2."*
