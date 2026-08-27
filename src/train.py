import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
train.py — Training entry-point for Swift Robotics Door Detection.
"""

import argparse
import time
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR  = PROJECT_ROOT / "configs"
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"
VALID_EXPERIMENTS = ["baseline", "augmentation", "high_resolution", "final"]


def load_config(experiment: str) -> dict:
    config_path = CONFIGS_DIR / f"{experiment}.yaml"
    if not config_path.exists():
        available = [p.stem for p in CONFIGS_DIR.glob("*.yaml")]
        raise FileNotFoundError(f"Config not found: {config_path}\nAvailable: {available}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train(experiment: str, resume: bool = False) -> None:
    from ultralytics import YOLO

    cfg = load_config(experiment)
    last_weights = PROJECT_ROOT / "runs" / "detect" / experiment / "weights" / "last.pt"

    print(f"\n{'='*60}")
    print(f"  Swift Robotics — Door Detection Training")
    print(f"{'='*60}")
    print(f"  Experiment : {experiment}")
    print(f"  Model      : {cfg['model']}")
    print(f"  Image size : {cfg['imgsz']}")
    print(f"  Epochs     : {cfg['epochs']}")
    print(f"  Batch      : {cfg['batch']}")
    print(f"  Resume     : {resume} (last.pt exists: {last_weights.exists()})")
    print(f"  Seed       : {cfg.get('seed', 42)}")
    print(f"  Data       : {DATA_YAML}")
    print(f"{'='*60}\n")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}.")

    t_start = time.time()

    if resume and last_weights.exists():
        print(f"  Resuming training from checkpoint: {last_weights}")
        model = YOLO(str(last_weights))
        model.train(resume=True)
    else:
        model = YOLO(cfg["model"])
        model.train(
            data        = str(DATA_YAML),
            epochs      = cfg["epochs"],
            imgsz       = cfg["imgsz"],
            batch       = cfg["batch"],
            lr0         = cfg.get("lr0",           0.01),
            lrf         = cfg.get("lrf",           0.01),
            momentum    = cfg.get("momentum",      0.937),
            weight_decay= cfg.get("weight_decay",  0.0005),
            warmup_epochs=cfg.get("warmup_epochs", 3),
            seed        = cfg.get("seed",          42),
            # ── Augmentation ──────────────────────────────────────────────────────
            hsv_h       = cfg.get("hsv_h",        0.015),
            hsv_s       = cfg.get("hsv_s",        0.7),
            hsv_v       = cfg.get("hsv_v",        0.4),
            degrees     = cfg.get("degrees",      0.0),
            translate   = cfg.get("translate",    0.1),
            scale       = cfg.get("scale",        0.5),
            shear       = cfg.get("shear",        0.0),
            perspective = cfg.get("perspective",  0.0),
            flipud      = cfg.get("flipud",       0.0),
            fliplr      = cfg.get("fliplr",       0.5),
            mosaic      = cfg.get("mosaic",       1.0),
            mixup       = cfg.get("mixup",        0.0),
            copy_paste  = cfg.get("copy_paste",   0.0),
            # ── Output ────────────────────────────────────────────────────────────
            project     = str(PROJECT_ROOT / "runs" / "detect"),
            name        = experiment,
            exist_ok    = True,
            device      = 0,
            workers     = 2,
            verbose     = True,
        )

    elapsed = time.time() - t_start
    best_weights = PROJECT_ROOT / "runs" / "detect" / experiment / "weights" / "best.pt"

    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best weights : {best_weights}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO door detector.")
    parser.add_argument("--experiment", type=str, required=True, choices=VALID_EXPERIMENTS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.experiment, args.resume)


if __name__ == "__main__":
    main()
