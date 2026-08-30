"""
Robotics Deployment Robustness Evaluation (Spatial IoU >= 0.50)
===============================================================
Evaluates the winning lr_schedule model against realistic visual
corruptions encountered in AMR robotics deployment using strict
spatial bounding box IoU >= 0.50 matching.

Corruptions evaluated:
1. Normal (Clean Held-Out Test Set baseline)
2. Low Light (Gamma 2.2 non-linear dimming / underexposure)
3. Motion Blur (15px linear camera vibration / velocity blur)
4. Partial Occlusion (25% center bounding box cutout mask)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def apply_low_light(img: np.ndarray) -> np.ndarray:
    """Non-linear gamma curve simulating dim hallways (gamma=2.2)."""
    table = np.array([((i / 255.0) ** 2.2) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_motion_blur(img: np.ndarray) -> np.ndarray:
    """Linear horizontal motion blur simulating robot velocity / camera shake."""
    kernel = np.zeros((15, 15))
    kernel[7, :] = np.ones(15) / 15.0
    return cv2.filter2D(img, -1, kernel)


def apply_occlusion(img: np.ndarray) -> np.ndarray:
    """25% central rectangular gray mask simulating passing obstacles / people."""
    out = img.copy()
    h, w = out.shape[:2]
    ch, cw = int(h * 0.25), int(w * 0.25)
    cy, cx = h // 2, w // 2
    out[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2] = 128
    return out


def box_iou(box1: list, box2: list) -> float:
    """Compute standard Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_split(
    model: YOLO,
    test_imgs: list,
    test_lbls: list,
    transform_fn,
    name: str,
    conf: float = 0.25,
    iou_thresh: float = 0.50,
    imgsz: int = 640,
) -> dict:
    tp = 0
    fp = 0
    fn = 0

    for img_path, lbl_path in zip(test_imgs, test_lbls):
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        if transform_fn is not None:
            img = transform_fn(img)

        # Parse ground-truth bounding boxes
        gt_boxes = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                    gt_boxes.append({"cls": cls_id, "box": [x1, y1, x2, y2], "matched": False})

        # Run inference with explicit imgsz
        res = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        pred_boxes = []
        for b in res.boxes:
            pred_boxes.append({
                "cls": int(b.cls[0]),
                "conf": float(b.conf[0]),
                "box": [float(v) for v in b.xyxy[0]],
            })

        # Sort predictions by confidence descending
        pred_boxes.sort(key=lambda x: x["conf"], reverse=True)

        # Spatial IoU greedy matching
        for p in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for idx, g in enumerate(gt_boxes):
                if not g["matched"] and g["cls"] == p["cls"]:
                    iou = box_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                tp += 1
                gt_boxes[best_gt_idx]["matched"] = True
            else:
                fp += 1

        fn += sum(1 for g in gt_boxes if not g["matched"])

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "condition": name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "eval_protocol": f"Spatial IoU >= {iou_thresh:.2f}, conf = {conf:.2f}, imgsz = {imgsz}",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness under simulated deployment conditions.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "detect" / "lr_schedule" / "weights" / "best.pt"),
        help="Path to model weights.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution.")
    parser.add_argument("--iou", type=float, default=0.50, help="Spatial IoU threshold.")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        fallback = PROJECT_ROOT / "models" / "best.pt"
        if fallback.exists():
            weights_path = fallback
        else:
            raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    test_dir = PROJECT_ROOT / "dataset" / "images" / "test"
    test_imgs = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    test_lbls = [PROJECT_ROOT / "dataset" / "labels" / "test" / (p.stem + ".txt") for p in test_imgs]

    print("=" * 72)
    print(f"  Robotics Deployment Robustness Benchmark (N=281 Test Images)")
    print(f"  Model: {weights_path.name}  |  conf = {args.conf}  |  imgsz = {args.imgsz}  |  IoU >= {args.iou}")
    print("=" * 72)
    print(f"{'Condition / Perturbation':<35} | {'Precision':>9} | {'Recall':>8} | {'F1 Score':>8}")
    print("-" * 72)

    conditions = [
        ("Normal (Clean Held-Out Test)", None),
        ("Low Light (Gamma 2.2 Dimming)", apply_low_light),
        ("Motion Blur (15px Linear Shake)", apply_motion_blur),
        ("Partial Occlusion (25% Center Mask)", apply_occlusion),
    ]

    records = []
    for name, fn in conditions:
        rec = evaluate_split(
            model,
            test_imgs,
            test_lbls,
            fn,
            name,
            conf=args.conf,
            iou_thresh=args.iou,
            imgsz=args.imgsz,
        )
        records.append(rec)
        print(f"  {name:<33} | {rec['precision']*100:>8.1f}% | {rec['recall']*100:>7.1f}% | {rec['f1']*100:>7.1f}%")

    print("=" * 72)

    out_file = RESULTS_DIR / "robustness_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nRobustness report saved to: {out_file}\n")


if __name__ == "__main__":
    main()
