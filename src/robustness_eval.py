"""
Robustness Evaluation under Realistic Deployment Corruptions
=============================================================
Evaluates the winning lr_schedule model against realistic visual
corruptions encountered in AMR robotics deployment:
1. Normal (Held-Out Test Set baseline)
2. Low Light (Gamma 2.2 dimming / underexposure)
3. Motion Blur (15px horizontal camera vibration / velocity blur)
4. Partial Occlusion (25% synthetic bounding box cutout)
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


def evaluate_split(model: YOLO, test_imgs: list, test_lbls: list, transform_fn, name: str, conf: float = 0.25) -> dict:
    tp = 0
    fp = 0
    fn = 0

    for img_path, lbl_path in zip(test_imgs, test_lbls):
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if transform_fn is not None:
            img = transform_fn(img)

        # Ground truth classes
        gt_classes = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gt_classes.append(int(line.split()[0]))

        # Predict
        res = model.predict(img, conf=conf, verbose=False)[0]
        pred_classes = [int(b.cls[0]) for b in res.boxes]

        # Greedy class matching
        matched_preds = list(pred_classes)
        for gt in gt_classes:
            if gt in matched_preds:
                tp += 1
                matched_preds.remove(gt)
            else:
                fn += 1
        fp += len(matched_preds)

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

    print("=" * 68)
    print("  Robotics Deployment Robustness Benchmark (N=281 Test Images)")
    print(f"  Model: {weights_path.name}  |  Confidence Threshold: {args.conf}")
    print("=" * 68)
    print(f"{'Condition / Perturbation':<35} | {'Precision':>9} | {'Recall':>8} | {'F1 Score':>8}")
    print("-" * 68)

    conditions = [
        ("Normal (Clean Held-Out Test)", None),
        ("Low Light (Gamma 2.2 Dimming)", apply_low_light),
        ("Motion Blur (15px Linear Shake)", apply_motion_blur),
        ("Partial Occlusion (25% Center Mask)", apply_occlusion),
    ]

    records = []
    for name, fn in conditions:
        rec = evaluate_split(model, test_imgs, test_lbls, fn, name, conf=args.conf)
        records.append(rec)
        print(f"  {name:<33} | {rec['precision']*100:>8.1f}% | {rec['recall']*100:>7.1f}% | {rec['f1']*100:>7.1f}%")

    print("=" * 68)

    out_file = RESULTS_DIR / "robustness_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nRobustness report saved to: {out_file}\n")


if __name__ == "__main__":
    main()
