import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
benchmark.py — Profile inference latency and FPS for PyTorch (FP16/CUDA) & ONNX.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def hardware_info() -> dict:
    info = {
        "os"     : platform.platform(),
        "python" : platform.python_version(),
        "cpu"    : platform.processor(),
    }
    try:
        import torch
        info["torch"]          = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"]  = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        else:
            info["gpu"] = "none (CPU only)"
    except ImportError:
        info["torch"] = "not installed"
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return info


def benchmark_pytorch(weights: str, imgsz: int, warmup: int, n_runs: int) -> dict:
    from ultralytics import YOLO
    import torch

    model  = YOLO(weights)
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    precision = "FP16 (CUDA)" if use_cuda else "FP32 (CPU)"

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    print(f"  Warming up ({warmup} runs on {device} with {precision})…", flush=True)
    for _ in range(warmup):
        model.predict(dummy, verbose=False, imgsz=imgsz, half=use_cuda, device=device)

    print(f"  Benchmarking ({n_runs} runs)…", flush=True)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, imgsz=imgsz, half=use_cuda, device=device)
        latencies.append((time.perf_counter() - t0) * 1000)

    return _report(latencies, runtime=f"PyTorch {precision}", device=device)


def benchmark_onnx(weights: str, imgsz: int, warmup: int, n_runs: int) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available_eps = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_eps
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(weights, opts, providers=providers)

    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
    print(f"  Warming up ({warmup} runs on {device})…", flush=True)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    print(f"  Benchmarking ({n_runs} runs)…", flush=True)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    return _report(latencies, runtime="ONNXRuntime (FP32 Static Graph)", device=device)


def _report(latencies: list, runtime: str, device: str) -> dict:
    arr = np.array(latencies)
    result = {
        "runtime"  : runtime,
        "device"   : device,
        "mean_ms"  : round(float(arr.mean()), 2),
        "std_ms"   : round(float(arr.std()),  2),
        "min_ms"   : round(float(arr.min()),  2),
        "max_ms"   : round(float(arr.max()),  2),
        "fps"      : round(1000.0 / float(arr.mean()), 1),
        "n_runs"   : len(latencies),
    }
    print(f"\n  {'─'*45}")
    print(f"  Runtime  : {runtime}  |  Device: {device}")
    print(f"  Mean     : {result['mean_ms']:.2f} ms  ±  {result['std_ms']:.2f} ms")
    print(f"  FPS      : ~{result['fps']:.1f}")
    print(f"  {'─'*45}\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference latency.")
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--model-type", default="pytorch", choices=["pytorch", "onnx"])
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--warmup",     type=int, default=10)
    parser.add_argument("--runs",       type=int, default=100)
    parser.add_argument("--name",       type=str, default=None)
    args = parser.parse_args()

    hw = hardware_info()
    if args.model_type == "onnx":
        result = benchmark_onnx(args.weights, args.imgsz, args.warmup, args.runs)
    else:
        result = benchmark_pytorch(args.weights, args.imgsz, args.warmup, args.runs)

    result["hardware"] = hw
    result["imgsz"]    = args.imgsz
    result["warmup"]   = args.warmup
    result["weights"]  = args.weights

    label    = args.name or Path(args.weights).stem
    out_path = RESULTS_DIR / f"benchmark_{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
