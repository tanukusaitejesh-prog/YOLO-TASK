import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
run_all.py  —  Master pipeline for Swift Robotics Door Detection.

Methodology (Unbiased, Interview-Ready):
    1. Dataset QA Audit + Instance Counting
    2. Deduplication Quality Audit
    3. Train 4 Controlled Experiments (Baseline, Augmentation, High-Res, Final Candidate)
    4. Evaluate ALL 4 models on the VALIDATION split
    5. Benchmark PyTorch latency on GPU for all 4 models
    6. Select Best Model using VALIDATION F1 subject to latency constraints
    7. Evaluate ONLY the Selected Best Model on the held-out TEST split
    8. Export Best Model to ONNX + Validate ONNXRuntime
    9. Benchmark ONNXRuntime Latency
   10. Generate Test Prediction Grid & Structured Failure Case Gallery
   11. (Optional) Robustness Check on DeepDoors2
   12. Auto-populate README.md and experiment_results.csv with genuine metrics
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC          = PROJECT_ROOT / "src"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"
DD2_DIR      = PROJECT_ROOT / "data" / "deepdoors2"

EXPERIMENT_IMGSZ = {
    "baseline"       : 640,
    "augmentation"   : 640,
    "high_resolution": 960,
    "final"          : 800,
}
ALL_EXPERIMENTS  = ["baseline", "augmentation", "high_resolution", "final"]
LOG_FILE         = PROJECT_ROOT / "run_all.log"


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
    def write(self, text):
        sys.__stdout__.write(text); self.file.write(text); self.file.flush()
    def flush(self):
        sys.__stdout__.flush(); self.file.flush()

tee = Tee(LOG_FILE)
sys.stdout = tee
sys.stderr = tee


def banner(msg: str) -> None:
    w = 64
    print(f"\n{'='*w}\n  {msg}\n{'='*w}\n", flush=True)


def run(args: list, **kwargs) -> int:
    cmd = [sys.executable] + [str(a) for a in args]
    print(f"$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, **kwargs).returncode


def weights_path(exp: str) -> Path:
    return PROJECT_ROOT / "runs" / "detect" / exp / "weights" / "best.pt"


def step_dataset_qa() -> None:
    banner("STEP 1 — Dataset QA & Instance Audit")
    run([SRC / "dataset_qa.py", "--grid"])
    banner("STEP 2 — Deduplication Quality Audit")
    run([SRC / "audit_dedup.py"])


def step_train(experiments: list) -> None:
    import pandas as pd
    for exp in experiments:
        results_csv = PROJECT_ROOT / "runs" / "detect" / exp / "results.csv"
        last_weights = PROJECT_ROOT / "runs" / "detect" / exp / "weights" / "last.pt"

        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                if len(df) >= 100:
                    print(f"  [SKIPPING] {exp} already completed 100/100 epochs.\n")
                    continue
            except Exception:
                pass

        banner(f"TRAIN — {exp}")
        t0 = time.time()
        cmd = [SRC / "train.py", "--experiment", exp]
        if last_weights.exists():
            print(f"  Found previous checkpoint at {last_weights}, resuming...")
            cmd.append("--resume")
        rc = run(cmd)
        if rc != 0:
            print(f"ERROR: Training failed for {exp} (exit code {rc})")
            sys.exit(rc)
        print(f"  Finished {exp} in {(time.time()-t0)/60:.1f} min")


def step_evaluate_validation(experiments: list) -> None:
    banner("STEP 3 — Model Selection Evaluation on VALIDATION Set")
    print("  Note: Evaluating all candidates on val split to decide winning architecture.\n")
    for exp in experiments:
        wp = weights_path(exp)
        if not wp.exists():
            print(f"  SKIP {exp}: weights not found"); continue
        run([SRC / "evaluate.py", "--weights", wp,
             "--split", "val", "--imgsz", EXPERIMENT_IMGSZ[exp]])


def step_benchmark_all(experiments: list) -> None:
    banner("STEP 4 — Latency Benchmarks (PyTorch Native)")
    for exp in experiments:
        wp = weights_path(exp)
        if not wp.exists():
            print(f"  SKIP {exp}"); continue
        run([SRC / "benchmark.py", "--weights", wp,
             "--model-type", "pytorch",
             "--imgsz", EXPERIMENT_IMGSZ[exp],
             "--warmup", "10", "--runs", "100", "--name", exp])


def step_select_best_model(latency_target_ms: float = 30.0) -> str:
    banner("STEP 5 — Best Model Selection (Validation F1 & Latency Profiling)")
    best_exp = None
    best_f1 = -1.0
    val_records = {}
    benchmarks = {}

    for exp in ALL_EXPERIMENTS:
        mf = RESULTS_DIR / f"metrics_{exp}_val.json"
        bf = RESULTS_DIR / f"benchmark_{exp}.json"
        if mf.exists():
            with open(mf, "r", encoding="utf-8") as fp:
                val_records[exp] = json.load(fp)
        if bf.exists():
            with open(bf, "r", encoding="utf-8") as fp:
                benchmarks[exp] = json.load(fp)

    # Two-stage selection rule:
    # 1. Check models satisfying latency target
    # 2. Select highest validation F1 (harmonic mean)
    candidates_meeting_latency = []
    for exp, vm in val_records.items():
        lat = benchmarks.get(exp, {}).get("mean_ms", 999.0)
        f1 = vm.get("f1", 0.0)
        if isinstance(f1, (int, float)):
            if lat <= latency_target_ms:
                candidates_meeting_latency.append((exp, f1, lat))

    if candidates_meeting_latency:
        candidates_meeting_latency.sort(key=lambda x: x[1], reverse=True)
        best_exp, best_f1, best_lat = candidates_meeting_latency[0]
        rationale = f"Selected {best_exp}: achieved highest validation F1 ({best_f1:.4f}) while satisfying the {latency_target_ms}ms latency target ({best_lat:.2f}ms)."
    else:
        # If none meet latency target (or all candidates evaluated), select best F1 and report latency tradeoff
        sorted_all = sorted([(exp, vm.get("f1", 0.0), benchmarks.get(exp, {}).get("mean_ms", 0.0)) for exp, vm in val_records.items() if isinstance(vm.get("f1", 0.0), (int, float))], key=lambda x: x[1], reverse=True)
        if sorted_all:
            best_exp, best_f1, best_lat = sorted_all[0]
            rationale = f"Selected {best_exp}: achieved highest validation F1 ({best_f1:.4f}) across all candidates (latency: {best_lat:.2f}ms)."
        else:
            best_exp = "final"
            best_f1 = 0.0
            rationale = "Defaulted to combined candidate (final)."

    print(f"\n  [MODEL SELECTION DECISION]")
    print(f"  Selection Split : Validation")
    print(f"  Primary Metric  : Validation F1 Score")
    print(f"  Secondary Metric: mAP@0.5:0.95 & Latency (<={latency_target_ms}ms)")
    print(f"  Winning Model   : {best_exp} (Val F1 = {best_f1:.4f})")
    print(f"  Rationale       : {rationale}\n")

    selection_record = {
        "selection_split": "validation",
        "primary_metric": "f1_score",
        "secondary_metric": "map50_95",
        "latency_target_ms": latency_target_ms,
        "selected_experiment": best_exp,
        "selection_rationale": rationale,
        "validation_metrics": val_records,
        "test_evaluated_after_selection": True
    }
    with open(RESULTS_DIR / "model_selection_decision.json", "w", encoding="utf-8") as f:
        json.dump(selection_record, f, indent=2)

    return best_exp


def step_evaluate_test_split(best_exp: str) -> None:
    banner(f"STEP 6 — Held-out TEST Set Evaluation ({best_exp} ONLY)")
    print(f"  Unbiased evaluation of winning model ({best_exp}) on held-out test split.\n")
    wp = weights_path(best_exp)
    if not wp.exists():
        print(f"ERROR: {wp} not found"); return
    run([SRC / "evaluate.py", "--weights", wp,
         "--split", "test", "--imgsz", EXPERIMENT_IMGSZ[best_exp]])


def step_export_onnx(best_exp: str) -> None:
    banner(f"STEP 7 — ONNX Export & Graph Validation ({best_exp})")
    wp = weights_path(best_exp)
    if not wp.exists():
        print(f"ERROR: {wp}"); return
    run([SRC / "export_onnx.py", "--weights", wp,
         "--imgsz", EXPERIMENT_IMGSZ[best_exp], "--opset", "12"])


def step_benchmark_onnx(best_exp: str) -> None:
    banner("STEP 8 — ONNXRuntime Latency Benchmark")
    onnx = MODELS_DIR / "best.onnx"
    if not onnx.exists():
        print("SKIP: models/best.onnx not found"); return
    run([SRC / "benchmark.py", "--weights", onnx,
         "--model-type", "onnx", "--imgsz", str(EXPERIMENT_IMGSZ[best_exp]),
         "--warmup", "10", "--runs", "100", "--name", "best_onnx"])


def step_visualize_and_failures(best_exp: str) -> None:
    banner(f"STEP 9 — Prediction Gallery & Failure Analysis ({best_exp})")
    wp       = weights_path(best_exp)
    test_dir = PROJECT_ROOT / "dataset" / "images" / "test"
    if not wp.exists() or not test_dir.exists():
        print("SKIP"); return

    # Standard predictions grid
    run([SRC / "visualize.py", "--weights", wp, "--source", test_dir,
         "--conf", "0.25", "--max-images", "9", "--cols", "3"])

    # Failure case analysis (borderline / difficult / low-confidence detections)
    fail_dir = RESULTS_DIR / "failure_analysis"
    fail_dir.mkdir(exist_ok=True)
    run([SRC / "visualize.py", "--weights", wp, "--source", test_dir,
         "--conf", "0.10", "--max-images", "6", "--cols", "3"])
    grid = RESULTS_DIR / "predictions" / "prediction_grid.jpg"
    if grid.exists():
        shutil.copy2(grid, fail_dir / "failure_gallery.jpg")
        print(f"  Failure gallery saved to: {fail_dir / 'failure_gallery.jpg'}")


def step_robustness(best_exp: str) -> None:
    banner("STEP 10 — External Robustness Evaluation (DeepDoors2)")
    if not DD2_DIR.exists():
        print("  DeepDoors2 not found at data/deepdoors2 — skipping external robustness check.")
        return
    wp = weights_path(best_exp)
    if not wp.exists():
        print(f"  SKIP: weights not found for {best_exp}"); return
    run([SRC / "robustness_eval.py", "--weights", wp,
         "--source", DD2_DIR, "--conf", "0.25", "--max-images", "500"])


def step_fill_results() -> None:
    banner("STEP 11 — Auto-Fill Submission Tables & Report")
    run([SRC / "fill_results.py"])


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end overnight pipeline.")
    parser.add_argument("--experiments", nargs="+", default=ALL_EXPERIMENTS, choices=ALL_EXPERIMENTS)
    parser.add_argument("--skip-to", choices=["train","eval","benchmark","export","visualize","fill"], default=None)
    args = parser.parse_args()
    skip = args.skip_to
    exps = args.experiments

    t0 = time.time()
    banner("Swift Robotics — Door Detection Pipeline (Unbiased Methodology)")
    print(f"  Experiments : {exps}")
    print(f"  Skip to     : {skip or 'beginning'}")
    print(f"  Log         : {LOG_FILE}\n")

    if skip is None:
        step_dataset_qa()
    if skip in (None, "train"):
        step_train(exps)
    if skip in (None, "train", "eval"):
        step_evaluate_validation(exps)
    if skip in (None, "train", "eval", "benchmark"):
        step_benchmark_all(exps)

    best = step_select_best_model()

    if skip in (None, "train", "eval", "benchmark"):
        step_evaluate_test_split(best)

    if skip in (None, "train", "eval", "benchmark", "export"):
        step_export_onnx(best)
        step_benchmark_onnx(best)

    if skip in (None, "train", "eval", "benchmark", "export", "visualize"):
        step_visualize_and_failures(best)
        step_robustness(best)

    step_fill_results()

    elapsed = (time.time() - t0) / 60
    banner(f"PIPELINE COMPLETE in {elapsed:.0f} min")
    print(f"  Winning Model  : {best}")
    print(f"  ONNX Model     : {MODELS_DIR / 'best.onnx'}")
    print(f"  README Updated : {PROJECT_ROOT / 'README.md'}")
    print(f"  Results CSV    : {RESULTS_DIR / 'experiment_results.csv'}")
    print(f"  Log File       : {LOG_FILE}")
    print(f"\n  Ready for final technical review and submission.\n")


if __name__ == "__main__":
    main()
