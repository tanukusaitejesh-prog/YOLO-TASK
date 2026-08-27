import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
robustness_eval.py  —  Evaluate the best model on DeepDoors2 (external test).

What this does:
    The model was trained on merged Roboflow datasets.
    DeepDoors2 was NOT used during training at all.

    Running inference on DeepDoors2 images tells us whether the model
    generalises beyond its training distribution — a much stronger claim
    than good performance on a random split of the same data.

    This is what "real-world generalisation" looks like before deployment.

DeepDoors2:
    ~3,000 RGB images with realistic conditions:
    obstacles, blur, glass doors, double doors, outdoor doors, people
    Original labels: open / closed / semi-open
    Semi-open images are EXCLUDED from this evaluation (ambiguous state).

    Dataset paper:
    "DeepDoors2: A Real-Time 2D and 3D Door Detection and
     State Classification on Low-Power Devices"

    Download: https://zenodo.org/record/7884745  (or GitHub companion repo)

Usage
-----
    # With pre-organised DeepDoors2 images:
    python src/robustness_eval.py --weights models/best.pt --source data/deepdoors2/

    # Preview what it would check (no model needed):
    python src/robustness_eval.py --source data/deepdoors2/ --dry-run
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
RESULTS_DIR   = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp"}

# DeepDoors2 uses these folder/label names — adapt if your download differs
DD2_OPEN_DIRS   = {"open",   "Open",   "OPEN",   "door_open"}
DD2_CLOSED_DIRS = {"closed", "Closed", "CLOSED", "door_closed"}
DD2_SKIP_DIRS   = {"semi-open", "semi_open", "ajar", "semiopen"}


def collect_deepdoors2_images(source: Path) -> list:
    """
    Walk source/ and collect (image_path, true_class) pairs.
    Skips 'semi-open' images (genuinely ambiguous — not a fair test).

    Expects either:
        source/open/*.jpg
        source/closed/*.jpg
    or flat:
        source/*.jpg  (no ground truth — inference only)
    """
    records = []
    subdirs = [d for d in source.iterdir() if d.is_dir()]

    if subdirs:
        for d in subdirs:
            name_lower = d.name.lower().replace("-", "_").replace(" ", "_")
            if name_lower in {s.lower() for s in DD2_SKIP_DIRS}:
                print(f"  Skipping semi-open directory: {d.name}")
                continue
            if name_lower in {s.lower() for s in DD2_OPEN_DIRS}:
                gt = 0   # door_open
            elif name_lower in {s.lower() for s in DD2_CLOSED_DIRS}:
                gt = 1   # door_closed
            else:
                print(f"  Unknown subfolder: {d.name} — skipping")
                continue
            for img in d.glob("*"):
                if img.suffix.lower() in IMG_EXTS:
                    records.append({"path": img, "gt": gt})
    else:
        # Flat directory — no ground truth, inference only
        for img in source.glob("*"):
            if img.suffix.lower() in IMG_EXTS:
                records.append({"path": img, "gt": None})

    return records


def evaluate_robustness(weights: str, source: Path, conf: float, max_images: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)

    records = collect_deepdoors2_images(source)
    if not records:
        print(f"No images found in {source}"); return {}

    random.seed(42)
    if max_images and len(records) > max_images:
        records = random.sample(records, max_images)

    print(f"\n  Evaluating {len(records)} DeepDoors2 images (conf ≥ {conf})")
    has_gt = any(r["gt"] is not None for r in records)

    tp, fp, fn, tn = 0, 0, 0, 0
    no_detection   = 0
    results_log    = []

    for rec in records:
        img_bgr  = cv2.imread(str(rec["path"]))
        if img_bgr is None: continue

        preds = model.predict(img_bgr, conf=conf, verbose=False)[0]
        gt    = rec["gt"]

        if len(preds.boxes) == 0:
            no_detection += 1
            pred_cls = None
        else:
            # Take highest-confidence detection
            best_box = max(preds.boxes, key=lambda b: float(b.conf[0]))
            pred_cls = int(best_box.cls[0])

        if has_gt:
            if pred_cls is None:
                fn += 1
            elif pred_cls == gt:
                if gt == 0: tp += 1
                else:       tn += 1
            else:
                if gt == 0: fn += 1   # predicted closed, actually open
                else:       fp += 1   # predicted open, actually closed

        results_log.append({
            "image"    : rec["path"].name,
            "gt"       : gt,
            "predicted": pred_cls,
            "n_dets"   : len(preds.boxes),
        })

    # ── Metrics ───────────────────────────────────────────────────────────────
    out = {
        "dataset"       : "DeepDoors2",
        "n_images"      : len(results_log),
        "no_detection"  : no_detection,
        "conf_threshold": conf,
    }

    if has_gt:
        total_positive = tp + fn
        total_negative = tn + fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc       = (tp + tn) / len(results_log) if results_log else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0

        out.update({
            "accuracy"       : round(acc, 4),
            "precision"      : round(precision, 4),
            "recall"         : round(recall, 4),
            "f1"             : round(f1, 4),
            "tp"             : tp, "fp": fp,
            "fn"             : fn, "tn": tn,
        })

        print(f"\n{'='*55}")
        print(f"  Robustness Eval — DeepDoors2 (unseen distribution)")
        print(f"{'─'*55}")
        print(f"  Images evaluated : {len(results_log)}")
        print(f"  No detection     : {no_detection}")
        print(f"  Accuracy         : {acc:.4f}")
        print(f"  Precision        : {precision:.4f}")
        print(f"  Recall           : {recall:.4f}")
        print(f"  F1               : {f1:.4f}")
        print(f"{'─'*55}")
        print(f"  open  images:  {total_positive}  | closed images: {total_negative}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"{'='*55}\n")
    else:
        print(f"  No ground truth — inference only. Detections logged.")

    # Save log
    out["log"] = results_log
    out_path   = RESULTS_DIR / "robustness_deepdoors2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results: {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness evaluation on DeepDoors2 (external distribution)."
    )
    parser.add_argument("--weights",    type=str, required=False, default=None,
                        help="Path to best .pt model.")
    parser.add_argument("--source",     type=str,
                        default=str(PROJECT_ROOT / "data" / "deepdoors2"),
                        help="Path to DeepDoors2 directory.")
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--max-images", type=int,   default=500)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Just count images, do not run model.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"\nDeepDoors2 not found at: {source}")
        print("\nTo set up DeepDoors2:")
        print("  1. Download from https://zenodo.org/record/7884745")
        print("  2. Extract into data/deepdoors2/")
        print("  3. Structure should be:")
        print("       data/deepdoors2/open/    *.jpg")
        print("       data/deepdoors2/closed/  *.jpg")
        print("  (semi-open images are automatically skipped)")
        return

    records = collect_deepdoors2_images(source)
    print(f"\nFound {len(records)} usable DeepDoors2 images")

    if args.dry_run or not args.weights:
        by_class = {0: 0, 1: 0, None: 0}
        for r in records:
            by_class[r["gt"]] = by_class.get(r["gt"], 0) + 1
        print(f"  door_open   : {by_class.get(0, 0)}")
        print(f"  door_closed : {by_class.get(1, 0)}")
        print(f"  unlabelled  : {by_class.get(None, 0)}")
        return

    evaluate_robustness(args.weights, source, args.conf, args.max_images)


if __name__ == "__main__":
    main()
