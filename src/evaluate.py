import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
evaluate.py  —  Evaluate a trained model and report full metrics.

Runs the model on either val or test split and prints / saves:
    Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, per-class AP

Why evaluate separately from training?
    Ultralytics reports val metrics during training, but the test set must
    only be evaluated once — after the best model has been selected.
    This script makes that single evaluation explicit and saves results
    so nothing has to be re-run.

Usage
-----
    python src/evaluate.py --weights runs/detect/baseline/weights/best.pt
    python src/evaluate.py --weights runs/detect/final/weights/best.pt --split test --imgsz 800
"""

import argparse
import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(weights: str, split: str, imgsz: int) -> dict:
    from ultralytics import YOLO

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Infer experiment name from path  (runs/detect/baseline/weights/best.pt → baseline)
    try:
        experiment = weights_path.parts[-3]
    except IndexError:
        experiment = weights_path.stem

    print(f"\n{'='*55}")
    print(f"  Evaluating : {experiment}  ({split} split)")
    print(f"  Weights    : {weights_path}")
    print(f"  Image size : {imgsz}")
    print(f"{'='*55}\n")

    model   = YOLO(str(weights_path))
    metrics = model.val(
        data    = str(DATA_YAML),
        split   = split,
        imgsz   = imgsz,
        name    = f"eval_{experiment}_{split}",
        verbose = True,
    )

    # ── Core metrics ──────────────────────────────────────────────────────────
    precision = float(metrics.box.mp)    # mean precision across classes
    recall    = float(metrics.box.mr)    # mean recall across classes
    map50     = float(metrics.box.map50) # mAP @ IoU=0.50
    map50_95  = float(metrics.box.map)   # mAP @ IoU=0.50:0.95

    # F1: Ultralytics doesn't surface a single F1 value directly, so we compute
    # the harmonic mean ourselves.  This is the standard F1 definition.
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    result = {
        "experiment" : experiment,
        "split"      : split,
        "imgsz"      : imgsz,
        "precision"  : round(precision,  4),
        "recall"     : round(recall,     4),
        "f1"         : round(f1,         4),
        "map50"      : round(map50,      4),
        "map50_95"   : round(map50_95,   4),
    }

    # ── Per-class breakdown ────────────────────────────────────────────────────
    # Per-class AP helps identify whether one door state is harder than the other.
    names = model.names
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        per_class = {}
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = names[int(cls_idx)]
            if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
                per_class[cls_name] = {"ap50": round(float(metrics.box.ap50[i]), 4)}
        if per_class:
            result["per_class"] = per_class

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Results — {experiment}  ({split})")
    print(f"{'─'*55}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1             : {f1:.4f}")
    print(f"  mAP@0.5        : {map50:.4f}")
    print(f"  mAP@0.5:0.95   : {map50_95:.4f}")
    if "per_class" in result:
        print(f"{'─'*55}")
        for cls_name, vals in result["per_class"].items():
            print(f"  {cls_name:<16} AP@0.5 = {vals['ap50']:.4f}")
    print(f"{'='*55}\n")

    # ── Save to JSON ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"metrics_{experiment}_{split}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO model on val or test split."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to .pt checkpoint  (e.g. runs/detect/baseline/weights/best.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size — should match the training imgsz for this experiment.",
    )
    args = parser.parse_args()
    evaluate(args.weights, args.split, args.imgsz)


if __name__ == "__main__":
    main()
