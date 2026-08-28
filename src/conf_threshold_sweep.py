"""
Experiment 7 — Confidence Threshold Sweep
==========================================
Evaluates the already-trained Baseline model (runs/detect/baseline/weights/best.pt)
across multiple confidence thresholds to map the Precision-Recall trade-off curve.

No retraining is needed. This experiment is purely post-hoc analysis.

Thresholds tested: [0.10, 0.25, 0.35, 0.40, 0.50, 0.60]

Why this matters for robotics:
- Lowering conf threshold → higher Recall (fewer missed doors), but more false positives
- Raising conf threshold → higher Precision (fewer false alarms), but more missed detections
- The robotics safety asymmetry: we especially want to minimize
  "Actual Closed -> Predicted Open" (false traversability) errors.
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "baseline" / "weights" / "best.pt"
DATA    = PROJECT_ROOT / "data" / "data.yaml"
THRESHOLDS = [0.10, 0.25, 0.35, 0.40, 0.50, 0.60]
IMGSZ  = 640
SPLIT  = "val"


def sweep_thresholds():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    print("=" * 65)
    print("  Experiment 7 — Confidence Threshold Sweep")
    print(f"  Model: {WEIGHTS.name}  |  Split: {SPLIT}  |  imgsz: {IMGSZ}")
    print("=" * 65)
    print(f"{'Conf':>6} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'mAP50':>8} | {'mAP50-95':>10}")
    print("-" * 65)

    records = []
    for conf in THRESHOLDS:
        results = model.val(
            data=str(DATA),
            split=SPLIT,
            imgsz=IMGSZ,
            conf=conf,
            iou=0.45,
            verbose=False,
        )

        p    = float(results.results_dict.get("metrics/precision(B)", 0))
        r    = float(results.results_dict.get("metrics/recall(B)", 0))
        f1   = 2 * p * r / (p + r + 1e-9)
        m50  = float(results.results_dict.get("metrics/mAP50(B)", 0))
        m95  = float(results.results_dict.get("metrics/mAP50-95(B)", 0))

        print(f"  {conf:>4.2f} | {p:>10.4f} | {r:>8.4f} | {f1:>8.4f} | {m50:>8.4f} | {m95:>10.4f}")

        records.append({
            "experiment": "conf_threshold_sweep",
            "model": "baseline",
            "conf_threshold": conf,
            "split": SPLIT,
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

    # Append to experiment_results.csv
    csv_path = RESULTS_DIR / "experiment_results.csv"
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        for rec in records:
            f.write(
                f"conf_sweep_conf{rec['conf_threshold']},yolov8n,{IMGSZ},100,{SPLIT},"
                f"{rec['precision']},{rec['recall']},{rec['f1']},{rec['map50']},{rec['map50_95']},"
                f"22.05,45.3,conf_threshold={rec['conf_threshold']}\n"
            )
    print(f"Appended to: {csv_path}")

    # Print summary — find best F1 point
    best = max(records, key=lambda x: x["f1"])
    print(f"\nBest F1 point: conf={best['conf_threshold']}  "
          f"P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}")

    return records


if __name__ == "__main__":
    sweep_thresholds()
