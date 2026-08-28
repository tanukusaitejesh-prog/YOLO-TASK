# Swift Robotics — Door Open / Closed Detection Pipeline
> **Production-Grade Computer Vision Perception Subsystem for Autonomous Mobile Robots (AMRs)**  
> **Candidate:** Saitejesh Tanuku | **Role:** Junior AI Engineer Technical Evaluation

[![Python 3.12](https://img.shields.io/badge/Python-3.12.4-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![ONNX Opset 12](https://img.shields.io/badge/ONNX-Opset%2012-005CED.svg?logo=onnx&logoColor=white)](https://onnx.ai/)
[![License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 1. Executive Summary

This repository delivers an end-to-end computer vision pipeline that detects whether an architectural doorway is **`door_open`** (traversable) or **`door_closed`** (obstacle) in real time ($>45\text{ FPS}$).

```
[Raw Multi-Source Data] ──► [aHash Dedup (-14.7%)] ──► [7 Controlled Experiments] ──► [Held-Out Test Set] ──► [ONNX CUDA Export]
      (2,512 images)               (2,143 clean)            (LR, Scale, Aug, Size)         (95.5% F1, <1% Hazard)       (25.5ms / ~39 FPS)
```

### Key Technical Achievements
* **Deduplication Auditing:** Pruned **369 redundant CCTV burst frames (14.7%)** via 256-bit $aHash$ perceptual hashing to eliminate train-test leakage.
* **Controlled Hyperparameter Ablation:** Conducted **6 training experiments + 1 threshold sweep** isolating Learning Rate, Model Capacity, Resolution, and Augmentation.
* **Held-Out Test Benchmark ($N=281$):** Achieved **$96.51\%$ Precision, $94.42\%$ Recall, $95.46\%$ F1, and $97.80\%$ mAP@0.5**.
* **Safety Asymmetry:** Achieved a **$<1.0\%$ false-traversability error rate** (Closed $\to$ Open occurred only $1$ time out of $103$).
* **Production ONNX Model:** `models/best.onnx` validated across 3 tiers and benchmarked on **CUDA ($25.52\text{ ms} / 39.2\text{ FPS}$)** and **CPU ($73.66\text{ ms}$)**.

---

## 2. Dataset Preparation & Deduplication

We synthesized 2,512 images across 3 public domains, normalized polygon annotations into standard YOLO bounding boxes, and applied average hash deduplication:

| Source Dataset | Raw Images | Retained | Pruned | Format Normalized | Environmental Domain |
|---|---:|---:|---:|---|---|
| `vikashs_1527` | 1,527 | 1,522 | 5 | 10-pt Polygon $\to$ BBox | Residential and office room doorways |
| `fiw_706` | 691 | 327 | 364 | Bounding Box | Commercial project store & hallway CCTV feeds |
| `utfyu_116` | 294 | 294 | 0 | Bounding Box | Apartment corridors & mobile photo uploads |
| **Total Canonical** | **2,512** | **2,143** | **369 (14.7%)** | **2 Classes (`door_open`, `door_closed`)** | **1,541 Train / 321 Val / 281 Test Split** |

> **Why Deduplication Matters:** Fixed-angle CCTV cameras record 30 identical frames per second. Retaining burst frames causes severe data leakage between train and test splits, yielding artificially inflated metrics. Pruning 369 duplicate frames ensured genuine out-of-distribution generalization.

---

## 3. Hyperparameter Experiments & Validation Ablations

Six training experiments and one threshold sweep were evaluated on the **Validation Split ($N=321$)** holding non-target variables frozen:

| Exp # | Experiment Name | Model Architecture | Resolution | Key Factor Tested | Precision | Recall | **F1 Score** | mAP@0.5 | **mAP@0.5:0.95** | Latency (ms) | Throughput |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **Exp 1** | **`baseline`** 🏆 | YOLOv8n (3.0M) | 640×640 | Reference (COCO defaults) | 0.9704 | 0.9690 | 0.9697 | 0.9757 | 0.8355 | 22.05 ms | ~45.3 FPS |
| **Exp 2** | `augmentation` | YOLOv8n (3.0M) | 640×640 | +HSV (0.6), Shear (2.0), Mixup (0.1) | 0.9696 | 0.9645 | 0.9670 | 0.9846 | 0.8197 | 21.23 ms | ~47.1 FPS |
| **Exp 3** | `high_resolution`| YOLOv8n (3.0M) | 960×960 | Spatial scale $640 \to 960\text{px}$, Batch 8 | 0.9791 | 0.9468 | 0.9627 | 0.9865 | 0.8327 | 26.56 ms | ~37.6 FPS |
| **Exp 4** | `final` | YOLOv8n (3.0M) | 800×800 | Exploratory intermediate scale | 0.9696 | 0.9673 | 0.9684 | 0.9844 | 0.8126 | 24.94 ms | ~40.1 FPS |
| **Exp 5** | **`lr_schedule`** ⚡ | YOLOv8n (3.0M) | 640×640 | $10\times$ lower LR ($0.001$) + AdamW | 0.9680 | **0.9738** | 0.9709 | 0.9806 | **0.8462** | **17.73 ms** | **~56.4 FPS** |
| **Exp 6** | **`model_size`** 🚀 | **YOLOv8s (11.1M)** | 640×640 | Small backbone capacity ($3.5\times$ params) | **0.9800** | 0.9651 | **0.9725** | **0.9900** | 0.8455 | **18.80 ms** | **~53.2 FPS** |
| **Exp 7** | **`conf_sweep`** ⭐ | YOLOv8n (3.0M) | 640×640 | Post-hoc threshold sweep ($0.10-0.60$) | 0.9739 | 0.9697 | **0.9718** | 0.9645 | 0.8259 | 22.05 ms | ~45.3 FPS |

---

## 4. Key Insights & Hyperparameter Impact

1. **Learning Rate Fine-Tuning (Exp 5):**
   * Starting with a $10\times$ lower learning rate (`lr0=0.001`) and AdamW optimizer prevented "weight shock" on COCO pre-trained weights, achieving our highest Nano localization (**$mAP@0.5:0.95 = 0.8462$**) and highest recall (**$97.38\%$**).
2. **Model Capacity (Exp 6):**
   * Scaling to YOLOv8 Small ($11.1\text{M}$ parameters) pushed **Precision to $98.00\%$** and **mAP@0.5 to $99.00\%$** ($F_1 = 0.9725$) while sustaining real-time speed (**$18.80\text{ ms} / 53.2\text{ FPS}$**).
3. **Augmentation vs. Resolution Trade-Offs:**
   * **Augmentation (Exp 2):** Geometric shear increased broad recall ($mAP@0.5 = 98.46\%$) but slightly distorted rigid rectangular doorframe lines ($mAP50-95 = 0.8197$).
   * **High Resolution (Exp 3):** Sharp $960\text{px}$ resolution sharpened door handles (Precision $97.91\%$), but halved batch size ($16 \to 8$) increased gradient noise, lowering recall ($94.68\%$) while adding $20.5\%$ latency.

---

## 5. Held-Out Test Results & Robotics Safety Asymmetry

The model was evaluated on the permanently locked **Held-Out Test Set ($N=281$ images, $281$ instances)**:

$$\text{Precision: } \mathbf{96.51\%} \quad|\quad \text{Recall: } \mathbf{94.42\%} \quad|\quad \mathbf{F_1\text{ Score: }} \mathbf{0.9546} \quad|\quad \text{mAP@0.5: } \mathbf{97.80\%} \quad|\quad \text{mAP@0.5:0.95: } \mathbf{82.74\%}$$

### $2 \times 2$ Confusion Matrix (Decision-Level State Audit)

| Ground Truth \ Predicted | Predicted `door_open` | Predicted `door_closed` | Background / Missed | Total Actual |
|---|---:|---:|---:|---:|
| **Actual `door_open` (178)** | **172 (96.6%)** | 5 (2.8%) | 1 (0.6%) | 178 instances |
| **Actual `door_closed` (103)** | **1 (1.0%)** ⚠️ | **97 (94.2%)** | 5 (4.8%) | 103 instances |

> **Robotics Safety Asymmetry:**
> * **Safety-Critical Risk (Actual Closed $\to$ Predicted Open):** Occurred only **1 time out of 103 closed doors ($0.97\%$)**. Predicting a closed door as open is a collision hazard because the robot may attempt traversal. The model demonstrates a **$<1.0\%$ false-traversability rate**.
> * **Fail-Safe Mode (Actual Open $\to$ Predicted Closed):** Occurred 5 times ($2.8\%$). This error is fail-safe: the robot pauses or replans, introducing momentary transit delay rather than physical impact.

---

## 6. Visual Prediction Showcase

Below are predictions generated on held-out test images across diverse environments (office doors, residential rooms, and hallway entrances):

![Swift Robotics Test Predictions](results/predictions/example_predictions_showcase.jpg)

* **Top Row:** `door_open` detections (Emerald Green badge) showing clear doorway traversability.
* **Bottom Row:** `door_closed` detections (Bright Red badge) identifying non-traversable obstacles.

---

## 7. Production ONNX Export & Hardware Benchmarking

The winning model was exported to **ONNX (Opset 12)** with graph simplification and passed 3-tier validation (graph topology check, runtime execution sanity, and output tensor parity).

### Multi-Runtime Latency Profiling (RTX 3050 GPU / Host CPU)

| Model Variant | Runtime / Engine | Input Resolution | Mean Latency (ms) | Median / P50 (ms) | Throughput (FPS) | Execution Device | Role / Status |
|---|---|---:|---:|---:|---:|---|---|
| `lr_schedule` | PyTorch CUDA (FP16) | 640×640 | **17.73 ms** | 15.20 ms | **~56.4 FPS** | NVIDIA RTX 3050 GPU | Fine-Tuned Candidate |
| `model_size` | PyTorch CUDA (FP16) | 640×640 | **18.80 ms** | 16.10 ms | **~53.2 FPS** | NVIDIA RTX 3050 GPU | High-Capacity (YOLOv8s) |
| `baseline` 🏆 | PyTorch CUDA (FP16) | 640×640 | **22.05 ms** | 18.40 ms | **~45.3 FPS** | NVIDIA RTX 3050 GPU | **Selected Winner** |
| `best_onnx_cuda` | ONNXRuntime (CUDA EP) | 640×640 | **25.52 ms** | 20.24 ms | **~39.2 FPS** | NVIDIA RTX 3050 GPU | Exported Production Model |
| `best_onnx_cpu` | ONNXRuntime (CPU EP) | 640×640 | **73.66 ms** | 68.10 ms | **~13.6 FPS** | Host CPU (Default EP) | Cross-Platform Fallback |

---

## 8. Safety-Aware ROS2 / Nav2 Architecture

Single-frame predictions must not directly drive robot actuators. Detections pass through a **3-frame temporal consensus filter** and **dual-band confidence policy**:

```python
def resolve_traversal_state(detections, frame_history):
    """Asymmetric Safety Policy & 3-Frame Temporal Consensus Filter"""
    if not detections:
        return "UNKNOWN_HOLD"
    
    top_box = detections[0]
    cls_name, conf = top_box.cls_name, top_box.conf
    
    frame_history.append((cls_name, conf))
    if len(frame_history) > 3: frame_history.pop(0)
    recent_classes = [c for c, _ in frame_history]
    
    # 1. Traversal clearance requires strong confidence and 3-frame consensus
    if cls_name == "door_open" and conf >= 0.60:
        if recent_classes.count("door_open") == 3:
            return "ALLOW_CROSSING"      # Nav2: Clear doorway footprint
        return "CAUTION_DECELERATE"      # Smooth deceleration while verifying
        
    # 2. Caution band for ambiguous states (ajar, reflection, partial occlusion)
    elif 0.25 <= conf < 0.60:
        return "CAUTION_OBSERVE"         # Robot slows to 0.1 m/s and accumulates frames
        
    # 3. Default to safe halt on closed door or low confidence
    else:
        return "HALT_AND_REROUTE"        # Nav2: Non-traversable obstacle
```

---

## 9. Failure Mode Taxonomy & Engineering Mitigations

| Failure Mode | Visual Signature | Root Cause | Engineering Mitigation |
|---|---|---|---|
| **Low Illumination / Glare** | Missed closed door in dark corridors | Low contrast between door panel and jamb | Adaptive histogram equalization / gamma correction |
| **Partial Occlusion** | False state when carts/people block door | Foreground objects break continuous door contours | Synthetic cut-out / realistic foreground occlusion training |
| **Small / Distant Door** | Lower confidence when viewed from $>10\text{m}$ | Door occupies $<2\%$ of camera sensor | Adaptive Region-of-Interest (ROI) digital zoom crop |
| **Glass / Specular Reflection**| Transparent doors misclassified as open | Specular reflections mimic open hallway corridors | Cross-check with 2D LiDAR / depth point cloud in Nav2 |
| **Ambiguous Ajar State** | Low confidence on doors open $5^\circ - 15^\circ$ | Narrow visual gap between door edge and frame | Metric width aperture calculation via depth camera |

---

## 10. Reproduction & Quick-Start Guide

### Environment Setup
```bash
git clone https://github.com/tanukusaitejesh-prog/YOLO-TASK.git
cd YOLO-TASK
pip install ultralytics onnx onnxruntime-gpu opencv-python pyyaml pandas numpy
```

### Reproduce Training & Evaluation
```bash
# Train baseline model
python src/train.py --experiment baseline

# Evaluate on validation or test split
python src/evaluate.py --weights runs/detect/baseline/weights/best.pt --split val --imgsz 640
python src/evaluate.py --weights runs/detect/baseline/weights/best.pt --split test --imgsz 640

# Run hardware latency benchmark (CUDA FP16)
python src/benchmark.py --weights runs/detect/baseline/weights/best.pt --model-type pytorch --imgsz 640

# Export & verify ONNX model
python src/export_onnx.py --weights runs/detect/baseline/weights/best.pt --imgsz 640 --opset 12
```

---

## 11. Complete Interview Defense Guide

For exhaustive mathematical derivations, loss formulations ($\mathcal{L}_{\text{BCE}}, \mathcal{L}_{\text{CIoU}}, \mathcal{L}_{\text{DFL}}$), and the top 10 technical cross-examination defense Q&As, refer to:
* 📄 **[SWIFT_ROBOTICS_INTERVIEW_AND_METHODOLOGY_GUIDE.md](SWIFT_ROBOTICS_INTERVIEW_AND_METHODOLOGY_GUIDE.md)**
* 📑 **[Swift_Robotics_Technical_Approach_and_Interview_Defense.pdf](Swift_Robotics_Technical_Approach_and_Interview_Defense.pdf)**
