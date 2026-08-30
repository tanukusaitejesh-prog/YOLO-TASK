"""
Confidence Threshold Sweep
==========================
Evaluates the winning lr_schedule model (runs/detect/lr_schedule/weights/best.pt)
across multiple confidence thresholds on the validation split to map the
Precision-Recall trade-off curve.

Thresholds tested: [0.10, 0.25, 0.35, 0.40, 0.50, 0.60]
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"


def sweep_thresholds(weights_path: Path, split: str = "val", imgsz: int = 640) -> list:
    from ultralytics import YOLO

    if not weights_path.exists():
        fallback = PROJECT_ROOT / "models" / "best.pt"
        if fallback.exists():
            weights_path = fallback
        else:
            raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    thresholds = [0.10, 0.25, 0.35, 0.40, 0.50, 0.60]

    print("=" * 65)
    print("  Confidence Threshold Sweep")
    print(f"  Model: {weights_path.name} ({weights_path.parts[-3] if len(weights_path.parts) >= 3 else 'lr_schedule'})")
    print(f"  Split: {split}  |  imgsz: {imgsz}")
    print("=" * 65)
    print(f"{'Conf':>6} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'mAP50':>8} | {'mAP50-95':>10}")
    print("-" * 65)

    records = []
    for conf in thresholds:
        results = model.val(
            data=str(DATA_YAML),
            split=split,
            imgsz=imgsz,
            conf=conf,
            iou=0.45,
            verbose=False,
        )

        p   = float(results.results_dict.get("metrics/precision(B)", 0))
        r   = float(results.results_dict.get("metrics/recall(B)", 0))
        f1  = (2 * p * r / (p + r + 1e-9)) if (p + r) > 0 else 0.0
        m50 = float(results.results_dict.get("metrics/mAP50(B)", 0))
        m95 = float(results.results_dict.get("metrics/mAP50-95(B)", 0))

        print(f"  {conf:>4.2f} | {p:>10.4f} | {r:>8.4f} | {f1:>8.4f} | {m50:>8.4f} | {m95:>10.4f}")

        records.append({
            "experiment": "conf_threshold_sweep",
            "model": "lr_schedule",
            "conf_threshold": conf,
            "split": split,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "map50": round(m50, 4),
            "map50_95": round(m95, 4),
        })

    print("=" * 65)

    # Save JSON
    out_json = RESULTS_DIR / "conf_threshold_sweep.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Summary
    best = max(records, key=lambda x: x["f1"])
    print(f"\nBest F1 point: conf={best['conf_threshold']}  "
          f"P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}\n")

    return records


def main():
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds on validation set.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "detect" / "lr_schedule" / "weights" / "best.pt"),
        help="Path to trained checkpoint (defaults to lr_schedule).",
    )
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or test).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution.")
    args = parser.parse_args()

    sweep_thresholds(Path(args.weights), args.split, args.imgsz)


if __name__ == "__main__":
    main()
