import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
download_datasets.py  —  Download all three Roboflow detection datasets.

Sources (all CC BY 4.0, object-detection format):
    1. DOOR_OPEN_CLOSE       ~3,459 images  (primary)
    2. Open/Close Detection  ~748 images    (supplement)
    3. FIW Door Open-Close   ~706 images    (supplement)

Each is downloaded into data/raw/{source_name}/ in YOLOv8 format.
The merge step (merge_datasets.py) handles normalisation and splitting.

Usage
-----
    python src/download_datasets.py --api-key YOUR_KEY
    python src/download_datasets.py           # will prompt if key missing

Get a FREE Roboflow key at: https://app.roboflow.com → Settings → API
(email signup, no card required, takes 90 seconds)
"""

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

# ── Dataset registry ───────────────────────────────────────────────────────────
# Each entry: (workspace, project, version, short_name)
# To find these for a new dataset: open the Roboflow page → Export → get snippet
DATASETS = [
    # Primary — 3,459 images, DOOR_OPEN / DOOR_CLOSE
    # https://universe.roboflow.com/search?q=DOOR_OPEN_CLOSE&t=object-detection
    {
        "workspace" : "door-open-close-dataset",   # update if slug differs
        "project"   : "door_open_close",
        "version"   : 1,
        "name"      : "door_open_close_3459",
    },
    # Supplement A — 748 images
    # https://universe.roboflow.com/openclose-door-dataset/open-close-door-detection
    {
        "workspace" : "openclose-door-dataset",
        "project"   : "open-close-door-detection",
        "version"   : 1,
        "name"      : "openclose_748",
    },
    # Supplement B — 706 images
    # https://universe.roboflow.com/fiw-benbo/door-open-close-prgfz
    {
        "workspace" : "fiw-benbo",
        "project"   : "door-open-close-prgfz",
        "version"   : 1,
        "name"      : "fiw_706",
    },
]


def download_one(rf, spec: dict) -> Path:
    """Download one Roboflow dataset and return its local root path."""
    dest = RAW_DIR / spec["name"]
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading: {spec['workspace']}/{spec['project']} v{spec['version']}")
    project  = rf.workspace(spec["workspace"]).project(spec["project"])
    version  = project.version(spec["version"])
    dataset  = version.download("yolov8", location=str(dest), overwrite=True)
    print(f"  Saved to:    {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all door detection datasets from Roboflow Universe."
    )
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("\nRoboflow API key required.")
        print("Get one free at: https://app.roboflow.com  (Settings → API)")
        api_key = input("Paste API key: ").strip()
        if not api_key:
            print("No key — exiting."); return

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {len(DATASETS)} datasets to {RAW_DIR}\n")

    downloaded = []
    for spec in DATASETS:
        try:
            path = download_one(rf, spec)
            downloaded.append((spec["name"], path))
        except Exception as e:
            print(f"  WARNING: Failed to download {spec['name']}: {e}")
            print(f"  → Download manually from Roboflow and place in {RAW_DIR / spec['name']}")

    print(f"\n{'='*55}")
    print(f"  Downloaded {len(downloaded)}/{len(DATASETS)} datasets.")
    print(f"  Next step:")
    print(f"    python src/merge_datasets.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
