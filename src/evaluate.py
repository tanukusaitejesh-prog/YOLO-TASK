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
    Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, per-class metrics, and confusion matrix
"""

import argparse
import json
from pathlib import Path
import numpy as np

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

    # F1: harmonic mean of overall precision and recall
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
    names = model.names
    per_class = {}
    
    # Extract arrays from metrics.box
    p_arr = getattr(metrics.box, 'p', None)
    r_arr = getattr(metrics.box, 'r', None)
    f1_arr = getattr(metrics.box, 'f1', None)
    ap50_arr = getattr(metrics.box, 'ap50', None)
    ap_arr = getattr(metrics.box, 'ap', None)
    cls_indices = getattr(metrics.box, 'ap_class_index', range(len(names)))

    for i, cls_idx in enumerate(cls_indices):
        cls_name = names[int(cls_idx)]
        c_p = float(p_arr[i]) if p_arr is not None and len(p_arr) > i else 0.0
        c_r = float(r_arr[i]) if r_arr is not None and len(r_arr) > i else 0.0
        c_f1 = float(f1_arr[i]) if f1_arr is not None and len(f1_arr) > i else ((2*c_p*c_r/(c_p+c_r)) if (c_p+c_r)>0 else 0.0)
        c_ap50 = float(ap50_arr[i]) if ap50_arr is not None and len(ap50_arr) > i else 0.0
        c_ap = float(ap_arr[i]) if ap_arr is not None and len(ap_arr) > i else 0.0

        per_class[cls_name] = {
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1": round(c_f1, 4),
            "ap50": round(c_ap50, 4),
            "ap50_95": round(c_ap, 4)
        }
    result["per_class"] = per_class

    # ── Confusion Matrix Extraction ───────────────────────────────────────────
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        cm_matrix = metrics.confusion_matrix.matrix.tolist()
        result["confusion_matrix"] = {
            "matrix": cm_matrix,
            "labels": ["door_open", "door_closed", "background"]
        }
        # Explicit counts for robotics safety audit
        if len(cm_matrix) >= 2 and len(cm_matrix[0]) >= 3:
            # Row 0: True door_open -> [pred_open, pred_closed, missed_bg]
            # Row 1: True door_closed -> [pred_open, pred_closed, missed_bg]
            true_open_pred_open = int(cm_matrix[0][0])
            true_open_pred_closed = int(cm_matrix[0][1])
            true_open_missed = int(cm_matrix[0][2])
            total_open = true_open_pred_open + true_open_pred_closed + true_open_missed

            true_closed_pred_open = int(cm_matrix[1][0]) # SAFETY HAZARD
            true_closed_pred_closed = int(cm_matrix[1][1])
            true_closed_missed = int(cm_matrix[1][2])
            total_closed = true_closed_pred_open + true_closed_pred_closed + true_closed_missed

            result["safety_audit"] = {
                "total_open_instances": total_open,
                "open_pred_open": true_open_pred_open,
                "open_pred_closed_failsafe": true_open_pred_closed,
                "open_missed_bg": true_open_missed,
                "total_closed_instances": total_closed,
                "closed_pred_closed": true_closed_pred_closed,
                "closed_pred_open_hazard": true_closed_pred_open,
                "closed_missed_bg": true_closed_missed,
                "hazard_rate_percent": round((true_closed_pred_open / total_closed * 100) if total_closed > 0 else 0.0, 2),
                "failsafe_rate_percent": round((true_open_pred_closed / total_open * 100) if total_open > 0 else 0.0, 2)
            }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Results — {experiment}  ({split})")
    print(f"{'─'*55}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1             : {f1:.4f}")
    print(f"  mAP@0.5        : {map50:.4f}")
    print(f"  mAP@0.5:0.95   : {map50_95:.4f}")
    print(f"{'─'*55}")
    for cls_name, vals in result["per_class"].items():
        print(f"  {cls_name:<14} P={vals['precision']:.4f} R={vals['recall']:.4f} F1={vals['f1']:.4f} AP50={vals['ap50']:.4f} AP50-95={vals['ap50_95']:.4f}")
    
    if "safety_audit" in result:
        sa = result["safety_audit"]
        print(f"{'─'*55}")
        print(f"  Safety Audit:")
        print(f"    Closed -> Open (Hazard)   : {sa['closed_pred_open_hazard']} / {sa['total_closed_instances']} ({sa['hazard_rate_percent']}%)")
        print(f"    Open -> Closed (Fail-safe): {sa['open_pred_closed_failsafe']} / {sa['total_open_instances']} ({sa['failsafe_rate_percent']}%)")
    print(f"{'='*55}\n")

    # ── Save to JSON ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"metrics_{experiment}_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    if split == "test":
        test_class_path = RESULTS_DIR / "test_class_metrics.json"
        with open(test_class_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {test_class_path}\n")

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
