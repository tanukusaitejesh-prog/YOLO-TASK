import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
export_onnx.py  —  Export the best model to ONNX and validate it.

Why ONNX?
    ONNX is a hardware-agnostic interchange format.  Once exported,
    the model can run on ONNXRuntime (cross-platform CPU/GPU), TensorRT
    (NVIDIA Jetson, data-centre GPU), or OpenVINO (Intel edge devices) —
    without rewriting any training code.

This script exports, then validates with four checks:
    1. ONNX model structure passes onnx.checker
    2. ONNXRuntime can load and execute the model
    3. A sample input produces output with the expected shape
    4. Numerical output parity between PyTorch and ONNXRuntime (np.allclose)

Usage
-----
    python src/export_onnx.py --weights runs/detect/lr_schedule/weights/best.pt
    python src/export_onnx.py --weights runs/detect/baseline/weights/best.pt --imgsz 640
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def export_and_validate(weights: str, imgsz: int, opset: int) -> None:
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort
    import torch

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"\n{'='*55}")
    print(f"  ONNX Export")
    print(f"  Source  : {weights_path}")
    print(f"  Imgsz   : {imgsz}  |  Opset: {opset}")
    print(f"{'='*55}\n")

    # ── Step 1: Export ────────────────────────────────────────────────────────
    model = YOLO(str(weights_path))
    exported_path = model.export(
        format   = "onnx",
        imgsz    = imgsz,
        opset    = opset,
        simplify = True,   # onnx-simplifier removes redundant nodes → smaller file
        dynamic  = False,  # static input shape is more compatible with edge runtimes
    )

    onnx_src  = Path(exported_path)
    onnx_dest = MODELS_DIR / "best.onnx"
    pt_dest   = MODELS_DIR / "best.pt"

    shutil.copy2(onnx_src,   onnx_dest)
    shutil.copy2(weights_path, pt_dest)

    print(f"  ONNX model: {onnx_dest}  ({onnx_dest.stat().st_size/1e6:.1f} MB)")
    print(f"  PT model  : {pt_dest}\n")

    # ── Step 2: Structural validation ─────────────────────────────────────────
    print("  Checking ONNX model structure…")
    onnx_model = onnx.load(str(onnx_dest))
    onnx.checker.check_model(onnx_model)
    print("  ✓  Structure valid\n")

    # ── Step 3: ONNXRuntime inference check ───────────────────────────────────
    print("  Running test inference with ONNXRuntime…")
    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    session    = ort.InferenceSession(str(onnx_dest), providers=providers)
    in_name    = session.get_inputs()[0].name
    in_shape   = session.get_inputs()[0].shape
    out_shape  = session.get_outputs()[0].shape

    print(f"  Input  name  : {in_name}")
    print(f"  Input  shape : {in_shape}")
    print(f"  Output shape : {out_shape}")

    dummy   = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    onnx_outputs = session.run(None, {in_name: dummy})
    print(f"  ✓  Inference succeeded  (output[0].shape = {onnx_outputs[0].shape})\n")

    # ── Step 4: Numerical Parity Check (PyTorch vs ONNX) ─────────────────────
    print("  Evaluating numerical parity (PyTorch vs ONNX raw tensor outputs)…")
    dummy_tensor = torch.from_numpy(dummy)
    model.model.eval()
    with torch.no_grad():
        pt_raw = model.model(dummy_tensor)
        if isinstance(pt_raw, (list, tuple)):
            pt_out = pt_raw[0].detach().cpu().numpy()
        else:
            pt_out = pt_raw.detach().cpu().numpy()

    onnx_out = onnx_outputs[0]
    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    mean_diff = float(np.mean(np.abs(pt_out - onnx_out)))
    is_close = np.allclose(pt_out, onnx_out, atol=1e-3, rtol=1e-3)

    print(f"  PyTorch raw tensor : {pt_out.shape}")
    print(f"  ONNX raw tensor    : {onnx_out.shape}")
    print(f"  Max absolute error : {max_diff:.2e}")
    print(f"  Mean absolute error: {mean_diff:.2e}")
    if is_close:
        print(f"  ✓  Numerical parity confirmed (np.allclose atol=1e-3, rtol=1e-3)\n")
    else:
        print(f"  ⚠  WARNING: Numerical discrepancy detected! Max delta ({max_diff:.2e}) exceeds tolerance (1e-3)\n")

    print(f"{'='*55}")
    print(f"  Export done.  Deliver: models/best.pt  +  models/best.onnx")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the best YOLO checkpoint to ONNX and validate."
    )
    parser.add_argument(
        "--weights", required=True,
        help="Path to best .pt checkpoint.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (must match training imgsz).",
    )
    parser.add_argument(
        "--opset", type=int, default=12,
        help="ONNX opset version.  12 is widely compatible with TensorRT and ORT.",
    )
    args = parser.parse_args()
    export_and_validate(args.weights, args.imgsz, args.opset)


if __name__ == "__main__":
    main()
