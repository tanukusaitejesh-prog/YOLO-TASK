import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
merge_datasets.py  —  Normalise, deduplicate, and split multi-source data.
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
DATASET_DIR  = PROJECT_ROOT / "dataset"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OPEN_ALIASES = {
    "door_open", "door-open", "door open", "open", "dooropen",
    "door_open_", "door-opened", "opened", "door opened",
    "door_openned",
    "door_open".upper(), "door open".upper(), "open".upper(),
    "door-open".upper(),
}
CLOSED_ALIASES = {
    "door_closed", "door-closed", "door closed", "closed", "doorclosed",
    "door_closed_", "door_close", "door-close", "door close", "close",
    "door_close".upper(), "door closed".upper(), "closed".upper(),
    "door_closed".upper(), "door-close".upper(),
}

CLASS_MAP: dict[str, int] = {}
for alias in OPEN_ALIASES:
    CLASS_MAP[alias.lower().strip()] = 0
for alias in CLOSED_ALIASES:
    CLASS_MAP[alias.lower().strip()] = 1

CANONICAL_NAMES = {0: "door_open", 1: "door_closed"}
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def phash(img: np.ndarray, size: int = 16) -> str:
    """16x16 average hash for robust deduplication."""
    gray = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    return "".join("1" if p > mean else "0" for p in gray.flatten())


def parse_source(source_dir: Path) -> tuple[list, set]:
    records      = []
    unknown_cls  = set()
    source_name  = source_dir.name

    yaml_path = source_dir / "data.yaml"
    src_class_names: dict[int, str] = {}
    if yaml_path.exists():
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", {})
        if isinstance(names, list):
            src_class_names = {i: n for i, n in enumerate(names)}
        elif isinstance(names, dict):
            src_class_names = {int(k): v for k, v in names.items()}

    for split_dir_name in ("train", "valid", "val", "test"):
        images_dir = source_dir / split_dir_name / "images"
        labels_dir = source_dir / split_dir_name / "labels"

        if not images_dir.exists():
            images_dir = source_dir / split_dir_name
            labels_dir = source_dir / split_dir_name

        if not images_dir.exists():
            continue

        for img_path in images_dir.glob("*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            annotations = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    src_cls_id = int(parts[0])
                    src_cls_name = src_class_names.get(src_cls_id, str(src_cls_id))
                    canonical_id = CLASS_MAP.get(src_cls_name.lower().strip())
                    if canonical_id is None:
                        unknown_cls.add(f"{source_name}::{src_cls_name}")
                        continue

                    try:
                        floats = [float(c) for c in parts[1:]]
                        if len(floats) == 4:
                            cx, cy, w, h = floats
                        else:
                            # Polygon points conversion to bbox
                            xs = floats[0::2]
                            ys = floats[1::2]
                            min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
                            min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
                            cx = (min_x + max_x) / 2.0
                            cy = (min_y + max_y) / 2.0
                            w  = max_x - min_x
                            h  = max_y - min_y

                        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                            annotations.append((canonical_id, cx, cy, w, h))
                    except ValueError:
                        pass

            if annotations:
                records.append({
                    "img_path"    : img_path,
                    "annotations" : annotations,
                    "source"      : source_name,
                })

    return records, unknown_cls


def deduplicate(records: list) -> tuple[list, list]:
    seen: dict[str, dict] = {}
    kept: list = []
    removed: list = []

    for rec in records:
        img = cv2.imread(str(rec["img_path"]))
        if img is None:
            removed.append({**rec, "reason": "corrupted"})
            continue
        h = phash(img)
        if h in seen:
            removed.append({**rec, "reason": "near_duplicate"})
        else:
            seen[h] = rec
            kept.append(rec)

    return kept, removed


def group_aware_split(
    records: list,
    train_ratio: float,
    val_ratio:   float,
    test_ratio:  float,
    seed: int = 42,
) -> dict[str, list]:
    rng = random.Random(seed)
    by_source: dict[str, list] = defaultdict(list)
    for rec in records:
        by_source[rec["source"]].append(rec)

    splits: dict[str, list] = {"train": [], "val": [], "test": []}

    for source, recs in by_source.items():
        rng.shuffle(recs)
        n       = len(recs)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        splits["train"].extend(recs[:n_train])
        splits["val"].extend(recs[n_train : n_train + n_val])
        splits["test"].extend(recs[n_train + n_val :])

    return splits


def write_dataset(splits: dict[str, list], dry_run: bool = False) -> dict:
    if not dry_run:
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        DATASET_DIR.mkdir(parents=True)

    stats = {}
    for split_name, records in splits.items():
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name

        if not dry_run:
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

        class_counts = defaultdict(int)
        for i, rec in enumerate(records):
            src_short = rec["source"][:6]
            stem      = f"{src_short}_{rec['img_path'].stem}_{i}"
            ext       = rec["img_path"].suffix.lower()
            img_dest  = img_dir / (stem + ext)
            lbl_dest  = lbl_dir / (stem + ".txt")

            if not dry_run:
                shutil.copy2(rec["img_path"], img_dest)
                with open(lbl_dest, "w", encoding="utf-8") as f:
                    for ann in rec["annotations"]:
                        cls_id, cx, cy, w, h = ann
                        f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        class_counts[cls_id] += 1

        stats[split_name] = {
            "images"       : len(records),
            "door_open"    : class_counts.get(0, 0),
            "door_closed"  : class_counts.get(1, 0),
        }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalise, deduplicate and split multi-source door dataset."
    )
    parser.add_argument("--raw-dir",  default=str(RAW_DIR))
    parser.add_argument("--train",    type=float, default=0.72)
    parser.add_argument("--val",      type=float, default=0.15)
    parser.add_argument("--test",     type=float, default=0.13)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    sources = [d for d in raw_dir.iterdir() if d.is_dir()]

    if not sources:
        print(f"No source datasets found in {raw_dir}")
        return

    print("\n" + "="*60)
    print("  Merge + Normalise + Deduplicate + Split")
    print("="*60 + "\n")

    all_records  = []
    all_unknown  = set()
    source_stats = {}

    for src_dir in sorted(sources):
        print(f"  Parsing source: {src_dir.name}")
        records, unknown = parse_source(src_dir)
        all_records.extend(records)
        all_unknown.update(unknown)
        source_stats[src_dir.name] = len(records)
        print(f"    -> {len(records)} valid images")

    print(f"\n  Total before dedup: {len(all_records)} images")

    if all_unknown:
        print("\n  [!] UNMAPPED CLASS NAMES (need review):")
        for u in sorted(all_unknown):
            print(f"     {u}")
        if not args.dry_run:
            return

    print("\n  Deduplicating...")
    kept, removed = deduplicate(all_records)
    n_corrupt = sum(1 for r in removed if r.get("reason") == "corrupted")
    n_dupes   = sum(1 for r in removed if r.get("reason") == "near_duplicate")
    print(f"    Kept      : {len(kept)}")
    print(f"    Corrupted : {n_corrupt}")
    print(f"    Duplicates: {n_dupes}")

    print(f"\n  Splitting (train={args.train}, val={args.val}, test={args.test})...")
    splits = group_aware_split(kept, args.train, args.val, args.test, args.seed)
    for split_name, recs in splits.items():
        sources_in_split = set(r["source"] for r in recs)
        print(f"    {split_name:<6} {len(recs):>5} images  |  sources: {sources_in_split}")

    print(f"\n  {'[DRY RUN] ' if args.dry_run else ''}Writing dataset/...")
    stats = write_dataset(splits, dry_run=args.dry_run)

    report = {
        "sources"         : source_stats,
        "total_raw"       : len(all_records),
        "removed"         : {"corrupted": n_corrupt, "duplicates": n_dupes},
        "kept"            : len(kept),
        "unmapped_classes": list(all_unknown),
        "splits"          : stats,
        "seed"            : args.seed,
    }
    report_path = RESULTS_DIR / "dataset_merge_report.json"
    if not args.dry_run:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n  QA report: {report_path}")

    print("\n" + "="*60)
    print("  Final dataset summary")
    print("-" * 60)
    if stats:
        for split_name, s in stats.items():
            print(f"  {split_name:<6}  images={s['images']:>4}  "
                  f"door_open={s['door_open']:>4}  door_closed={s['door_closed']:>4}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
