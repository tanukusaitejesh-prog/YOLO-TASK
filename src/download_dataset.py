import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
download_dataset.py — Download a door detection dataset automatically.

Uses Roboflow Universe to grab a pre-annotated door dataset in YOLOv8 format,
then organises it into the expected directory structure.

If you already have a dataset, skip this and populate dataset/ manually.

Usage
-----
    python src/download_dataset.py                    # interactive prompt
    python src/download_dataset.py --api-key YOUR_KEY # non-interactive
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"


def download_via_roboflow(api_key: str) -> None:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)

    # This is a public Roboflow door detection dataset (door_open / door_closed)
    # Universe link: https://universe.roboflow.com/roboflow-100/door-detection-3bmkt
    print("\nConnecting to Roboflow Universe...")
    project  = rf.workspace("roboflow-100").project("door-detection-3bmkt")
    version  = project.version(2)
    dataset  = version.download("yolov8", location=str(DATASET_ROOT / "_rf_download"))
    print(f"Downloaded to: {dataset.location}")
    _reorganise(Path(dataset.location))


def _reorganise(source: Path) -> None:
    """
    Roboflow downloads into train/valid/test subfolders with images/ and labels/.
    Reorganise into the project's expected layout:
        dataset/images/{train,val,test}/
        dataset/labels/{train,val,test}/
    """
    DATASET_ROOT.mkdir(exist_ok=True)

    split_map = {"train": "train", "valid": "val", "test": "test"}

    for rf_split, our_split in split_map.items():
        for kind in ("images", "labels"):
            src = source / rf_split / kind
            dst = DATASET_ROOT / kind / our_split
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.glob("*"):
                    shutil.copy2(f, dst / f.name)

    # Clean up the raw download folder
    shutil.rmtree(source, ignore_errors=True)
    print("\nDataset organised:")
    for split in ("train", "val", "test"):
        n_imgs = len(list((DATASET_ROOT / "images" / split).glob("*")))
        n_lbs  = len(list((DATASET_ROOT / "labels" / split).glob("*.txt")))
        print(f"  {split:<5}  images: {n_imgs}  labels: {n_lbs}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a door detection dataset from Roboflow Universe."
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Your Roboflow API key. Get one free at https://app.roboflow.com",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        print("\nRoboflow API key required.")
        print("Get one free (no card needed) at: https://app.roboflow.com → Settings → API")
        api_key = input("\nPaste your API key: ").strip()
        if not api_key:
            print("No key provided. Exiting.")
            return

    download_via_roboflow(api_key)
    print("\nDone. Run QA next:")
    print("  python src/dataset_qa.py --grid\n")


if __name__ == "__main__":
    main()
