import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""
fill_results.py — Auto-populate README.md and experiment_results.csv.

Maintains strict separation between:
    - Validation Set: used for hyperparameter comparison & model selection
    - Test Set: single unbiased evaluation of the winning model
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
README_PATH  = PROJECT_ROOT / "README.md"
CSV_PATH     = RESULTS_DIR / "experiment_results.csv"

EXPERIMENT_ORDER = ["baseline", "augmentation", "high_resolution", "final"]
EXPERIMENT_LABELS = {
    "baseline":        ("Baseline",           "640",  "—"),
    "augmentation":    ("Augmentation",       "640",  "+aug"),
    "high_resolution": ("High Res",           "960",  "+res"),
    "final":           ("Combined Candidate", "800",  "+aug +res"),
}


def load_val_metrics() -> dict:
    metrics = {}
    for f in RESULTS_DIR.glob("metrics_*_val.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        exp = data.get("experiment", f.stem.replace("metrics_", "").replace("_val", ""))
        metrics[exp] = data
    return metrics


def load_test_metrics() -> dict:
    metrics = {}
    for f in RESULTS_DIR.glob("metrics_*_test.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        exp = data.get("experiment", f.stem.replace("metrics_", "").replace("_test", ""))
        metrics[exp] = data
    return metrics


def load_benchmarks() -> dict:
    benchmarks = {}
    for f in RESULTS_DIR.glob("benchmark_*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        name = f.stem.replace("benchmark_", "")
        benchmarks[name] = data
    return benchmarks


def load_dataset_qa() -> dict:
    qa_file = RESULTS_DIR / "dataset_qa_stats.json"
    if qa_file.exists():
        with open(qa_file, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def fmt(val, decimals: int = 4) -> str:
    if isinstance(val, (float, int)):
        return f"{val:.{decimals}f}" if isinstance(val, float) else str(val)
    return str(val) if val is not None else "_fill_"


def build_validation_table(val_metrics: dict, benchmarks: dict) -> str:
    header = (
        "| Candidate Experiment | Img Size | Key Change | Val Precision | Val Recall | Val F1 | "
        "Val mAP@0.5 | Val mAP@0.5:0.95 | Native Latency (ms) | FPS |\n"
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = ""
    for exp in EXPERIMENT_ORDER:
        label, imgsz, change = EXPERIMENT_LABELS.get(exp, (exp, "?", "?"))
        m = val_metrics.get(exp, {})
        p     = fmt(m.get("precision"))
        r     = fmt(m.get("recall"))
        f1    = fmt(m.get("f1"))
        m50   = fmt(m.get("map50"))
        m5095 = fmt(m.get("map50_95"))

        lat = "_fill_"
        fps = "_fill_"
        if exp in benchmarks:
            lat = f"{benchmarks[exp].get('mean_ms', 0):.1f}"
            fps = f"~{benchmarks[exp].get('fps', 0):.1f}"

        row = (
            f"| {label} | {imgsz} | {change} | {p} | {r} | {f1} | {m50} | {m5095} | {lat} | {fps} |\n"
        )
        rows += row
    return header + rows


def build_final_test_table(best_exp: str, test_metrics: dict) -> str:
    m = test_metrics.get(best_exp, {})
    label = EXPERIMENT_LABELS.get(best_exp, (best_exp,))[0]
    header = (
        "| Winning Model | Split | Precision | Recall | F1 Score | mAP@0.5 | mAP@0.5:0.95 |\n"
        "|---|---|---:|---:|---:|---:|---:|\n"
    )
    p     = fmt(m.get("precision"))
    r     = fmt(m.get("recall"))
    f1    = fmt(m.get("f1"))
    m50   = fmt(m.get("map50"))
    m5095 = fmt(m.get("map50_95"))
    row = f"| **{label}** (`{best_exp}`) | **Held-out Test** | **{p}** | **{r}** | **{f1}** | **{m50}** | **{m5095}** |\n"
    return header + row


def build_dataset_table(qa_stats: dict) -> str:
    if not qa_stats:
        return None
    header = (
        "| Split | Images | `door_open` Instances | `door_closed` Instances | Total Instances | Instances / Image |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    rows = ""
    tot_img, tot_open, tot_closed, tot_inst = 0, 0, 0, 0
    for split in ["train", "val", "test"]:
        s = qa_stats.get(split, {})
        img = s.get("images", 0)
        opn = s.get("open_instances", 0)
        cls = s.get("closed_instances", 0)
        ins = s.get("total_instances", 0)
        ipi = s.get("instances_per_image", 1.0)
        tot_img += img; tot_open += opn; tot_closed += cls; tot_inst += ins
        rows += f"| **{split}** | {img} | {opn} | {cls} | {ins} | {ipi} |\n"
    rows += f"| **Total** | **{tot_img}** | **{tot_open}** | **{tot_closed}** | **{tot_inst}** | **{tot_inst/max(tot_img,1):.2f}** |\n"
    return header + rows


def update_readme(dry_run: bool = False) -> None:
    if not README_PATH.exists():
        print(f"README not found: {README_PATH}")
        return

    val_metrics  = load_val_metrics()
    test_metrics = load_test_metrics()
    benchmarks   = load_benchmarks()
    qa_stats     = load_dataset_qa()

    best_exp = "final"
    decision_file = RESULTS_DIR / "model_selection_decision.json"
    if decision_file.exists():
        with open(decision_file, "r", encoding="utf-8") as fp:
            best_exp = json.load(fp).get("selected_model", "final")
    elif test_metrics:
        best_exp = list(test_metrics.keys())[0]

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Dataset table
    dt_table = build_dataset_table(qa_stats)
    if dt_table:
        content = re.sub(
            r"\| Split \| Images \|.*?\n(\|[-| :]+\n)([\s\S]*?)(?=\n>|\n##|\Z)",
            dt_table + "\n",
            content,
            count=1,
            flags=re.MULTILINE
        )

    # 2. Validation Candidate Comparison Table
    if val_metrics:
        val_table = build_validation_table(val_metrics, benchmarks)
        content = re.sub(
            r"\| Candidate Experiment \| Img Size \|.*?\n(\|[-| :]+\n)([\s\S]*?)(?=\n>|\n##|\Z)",
            val_table + "\n",
            content,
            flags=re.MULTILINE
        )

    # 3. Final Test Table
    if test_metrics:
        test_table = build_final_test_table(best_exp, test_metrics)
        content = re.sub(
            r"\| Winning Model \| Split \|.*?\n(\|[-| :]+\n)([\s\S]*?)(?=\n>|\n##|\Z)",
            test_table + "\n",
            content,
            flags=re.MULTILINE
        )

    # 4. Latency Table
    if benchmarks:
        lat_header = "| Model Variant | Runtime / Engine | Input Resolution | Mean Latency (ms) | Throughput (FPS) |\n|---|---|---:|---:|---:|\n"
        lat_rows = ""
        for name, b in sorted(benchmarks.items()):
            runtime = "ONNXRuntime" if "onnx" in name else "PyTorch CUDA (FP16)"
            imgsz   = b.get("imgsz", 640)
            mean_ms = f"{b.get('mean_ms', 0):.2f}"
            fps     = f"~{b.get('fps', 0):.1f}"
            lat_rows += f"| `{name}` | {runtime} | {imgsz}×{imgsz} | {mean_ms} | {fps} |\n"
        content = re.sub(
            r"\| Model Variant \| Runtime / Engine \|.*?\n(\|[-| :]+\n)([\s\S]*?)(?=\n>|\n##|\Z)",
            lat_header + lat_rows + "\n",
            content,
            flags=re.MULTILINE
        )

    if not dry_run:
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"README updated: {README_PATH}")
        update_csv(val_metrics, test_metrics, benchmarks)


def update_csv(val_metrics: dict, test_metrics: dict, benchmarks: dict) -> None:
    lines = ["experiment,model,imgsz,epochs,split,precision,recall,f1,map50,map50_95,latency_ms,fps,notes"]
    for exp in EXPERIMENT_ORDER:
        vm = val_metrics.get(exp, {})
        lat = benchmarks.get(exp, {}).get("mean_ms", "")
        fps = benchmarks.get(exp, {}).get("fps", "")
        lines.append(f"{exp},yolov8n,{EXPERIMENT_LABELS.get(exp, ('','640'))[1]},100,val,{fmt(vm.get('precision'))},{fmt(vm.get('recall'))},{fmt(vm.get('f1'))},{fmt(vm.get('map50'))},{fmt(vm.get('map50_95'))},{lat},{fps},validation_candidate")

    for exp, tm in test_metrics.items():
        lat = benchmarks.get(exp, {}).get("mean_ms", "")
        fps = benchmarks.get(exp, {}).get("fps", "")
        lines.append(f"{exp},yolov8n,{EXPERIMENT_LABELS.get(exp, ('','640'))[1]},100,test,{fmt(tm.get('precision'))},{fmt(tm.get('recall'))},{fmt(tm.get('f1'))},{fmt(tm.get('map50'))},{fmt(tm.get('map50_95'))},{lat},{fps},winning_model_heldout_test")

    with open(CSV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"CSV updated:    {CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-fill README tables.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    update_readme(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
