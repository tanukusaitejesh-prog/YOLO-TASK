import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
visualize.py  —  Run inference and save annotated prediction images.

Creates two types of output:
    1. Individual annotated images in results/predictions/
    2. A prediction grid (N images laid out in a grid) for the README

Colour convention (consistent with robotics intent):
    door_open   →  Green  (robot can proceed)
    door_closed →  Red    (robot must stop)

Usage
-----
    # Random sample from the test set
    python src/visualize.py --weights runs/detect/lr_schedule/weights/best.pt \
        --source dataset/images/test

    # Specific image
    python src/visualize.py --weights runs/detect/lr_schedule/weights/best.pt \
        --source path/to/image.jpg

    # Low-confidence detections (useful for failure analysis)
    python src/visualize.py --weights runs/detect/lr_schedule/weights/best.pt \
        --source dataset/images/test --conf 0.20
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR      = PROJECT_ROOT / "results" / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
# OpenCV uses BGR
CLASS_COLORS = {
    "door_open"   : (50,  205, 50),   # green
    "door_closed" : (60,  20, 220),   # red
}
FALLBACK_COLOR = (180, 180, 180)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_result(image: np.ndarray, result) -> np.ndarray:
    """Draw bounding boxes + confidence labels on a BGR image."""
    img   = image.copy()
    names = result.names

    for box in result.boxes:
        cls_id          = int(box.cls[0])
        conf            = float(box.conf[0])
        cls_name        = names[cls_id]
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        color           = CLASS_COLORS.get(cls_name, FALLBACK_COLOR)
        label           = f"{cls_name}  {conf:.2f}"

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return img


def make_grid(images: list, cols: int = 3, cell_w: int = 400, cell_h: int = 300) -> np.ndarray:
    """Lay out a list of BGR images into a uniform grid."""
    resized = [cv2.resize(img, (cell_w, cell_h)) for img in images]
    rows = []
    for i in range(0, len(resized), cols):
        row = resized[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def visualize(
    weights: str,
    source: str,
    conf: float,
    max_images: int,
    cols: int,
    seed: int,
) -> None:
    from ultralytics import YOLO

    model      = YOLO(weights)
    source_path = Path(source)

    # Collect images
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = [p for p in source_path.rglob("*") if p.suffix.lower() in IMG_EXTS]

    if not image_paths:
        print(f"No images found in: {source}")
        return

    random.seed(seed)
    selected = random.sample(image_paths, min(max_images, len(image_paths)))
    print(f"Running inference on {len(selected)} image(s)…")

    annotated = []
    for img_path in selected:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Skipping (unreadable): {img_path.name}")
            continue

        results = model.predict(img_bgr, conf=conf, verbose=False)
        drawn   = draw_result(img_bgr, results[0])
        annotated.append(drawn)

        out = OUT_DIR / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(out), drawn)
        n = len(results[0].boxes)
        print(f"  {img_path.name}  →  {n} detection(s)  →  {out.name}")

    # Save grid
    if annotated:
        grid      = make_grid(annotated, cols=cols)
        grid_path = OUT_DIR / "prediction_grid.jpg"
        cv2.imwrite(str(grid_path), grid)
        print(f"\nGrid saved: {grid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise model predictions and save annotated images."
    )
    parser.add_argument("--weights",    required=True,       help="Path to .pt weights.")
    parser.add_argument("--source",     required=True,       help="Image file or directory.")
    parser.add_argument("--conf",       type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-images", type=int,   default=9,    help="Max images to process.")
    parser.add_argument("--cols",       type=int,   default=3,    help="Grid columns.")
    parser.add_argument("--seed",       type=int,   default=42,   help="Sampling seed.")
    args = parser.parse_args()

    visualize(args.weights, args.source, args.conf, args.max_images, args.cols, args.seed)


if __name__ == "__main__":
    main()
