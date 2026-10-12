"""Fine-tuning method ablation: TF-IDF baseline vs full fine-tuning vs LoRA / rsLoRA (r=16, 64) vs QLoRA.

Every transformer configuration is trained with three seeds; the report gives mean +/- std on the
held-out test set and the out-of-distribution challenge set, plus parameter and memory cost.

    uv run python scripts/run_classification_ablation.py
"""

import copy
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluate import Evaluator
from src.model_utils import load_config, require_deep_learning
from src.train import ClassificationTrainer
from src.train_transformer import load_classification_splits, run

SEEDS = [42, 7, 1234]
VARIANTS = [
    # (name, method, lora overrides)
    ("full_ft", "full", None),
    ("lora_r16", "lora", {"r": 16, "lora_alpha": 32}),
    ("rslora_r16", "rslora", {"r": 16, "lora_alpha": 32}),
    ("lora_r64", "lora", {"r": 64, "lora_alpha": 128}),
    ("rslora_r64", "rslora", {"r": 64, "lora_alpha": 128}),
    ("qlora_r16", "qlora", {"r": 16, "lora_alpha": 32}),
]


def baseline_scores(cfg):
    splits, _ = load_classification_splits(cfg)
    clf = ClassificationTrainer().train(*splits["train"])
    out = {}
    for name in ("test", "challenge"):
        x, y = splits[name]
        pred = clf.predict(x).tolist()
        out[name] = Evaluator().compute_metrics(y, pred)
        out[name]["accuracy_wilson_95ci"] = Evaluator.wilson_interval(sum(int(p == t) for p, t in zip(pred, y)), len(y))
    return out


def fmt(values, pct=False):
    arr = np.array(values, dtype=float) * (100 if pct else 1)
    return f"{arr.mean():.1f} ± {arr.std():.1f}" if pct else f"{arr.mean():.3f} ± {arr.std():.3f}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", action="store_true", help="reuse per-run metrics already in results/metrics")
    args = parser.parse_args()
    require_deep_learning()
    cfg = load_config(os.path.join(BASE_DIR, "configs", "classification.yaml"))
    base_model = cfg["model"]["base_model"]
    results = {}

    for name, method, overrides in VARIANTS:
        results[name] = []
        for seed in SEEDS:
            run_name = f"ablation_{name}_s{seed}"
            cached = os.path.join(BASE_DIR, "results", "metrics", f"classification_{run_name}.json")
            if args.resume and os.path.exists(cached):
                with open(cached, encoding="utf-8") as f:
                    results[name].append(json.load(f))
                print(f"[{run_name}] reused cached metrics")
                continue
            run_cfg = copy.deepcopy(cfg)
            run_cfg["seed"] = seed
            results[name].append(run(run_cfg, method, base_model, run_name,
                                     save_model=False, lora_overrides=overrides))

    base = baseline_scores(cfg)
    out_dir = os.path.join(BASE_DIR, "results")
    with open(os.path.join(out_dir, "metrics", "classification_ablation.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline": base, "runs": results}, f, indent=2)

    lines = [
        "# Fine-tuning method ablation",
        "",
        f"Base model `{base_model}`, seeds {SEEDS}, train/dev/test = 102/24/42, challenge = 30 plain-English questions.",
        "Model selection (early stopping) uses the dev set only; test and challenge sets are never seen during training.",
        "Values are mean ± std over seeds. Accuracy is in %.",
        "",
        "| Variant | Trainable params | Test acc | Test log-loss | Challenge acc | Challenge macro-F1 | Challenge log-loss | Epochs | Train time (s) | Peak GPU (MiB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| TF-IDF + LogReg baseline | - | {base['test']['accuracy'] * 100:.1f} | - | {base['challenge']['accuracy'] * 100:.1f} | "
        f"{base['challenge']['macro_f1']:.3f} | - | - | - | - |",
    ]
    for name, runs in results.items():
        r0 = runs[0]
        lines.append(
            f"| {name} | {r0['trainable_params']:,} ({100 * r0['trainable_params'] / r0['total_params']:.2f}%) | "
            f"{fmt([r['test']['accuracy'] for r in runs], pct=True)} | {fmt([r['test']['log_loss'] for r in runs])} | "
            f"{fmt([r['challenge']['accuracy'] for r in runs], pct=True)} | {fmt([r['challenge']['macro_f1'] for r in runs])} | "
            f"{fmt([r['challenge']['log_loss'] for r in runs])} | {np.mean([r['epochs_run'] for r in runs]):.1f} | "
            f"{np.mean([r['train_seconds'] for r in runs]):.1f} | {np.mean([r['peak_gpu_mib'] or 0 for r in runs]):.0f} |"
        )

    lines += ["", "## Per-seed 95% confidence intervals (seed 42)", "",
              "| Variant | Test acc (Wilson CI) | Challenge acc (Wilson CI) | Challenge macro-F1 (bootstrap CI) |", "|---|---|---|---|"]
    bt, bc = base["test"]["accuracy_wilson_95ci"], base["challenge"]["accuracy_wilson_95ci"]
    lines.append(f"| TF-IDF baseline | {base['test']['accuracy'] * 100:.1f} [{bt[0] * 100:.1f}, {bt[1] * 100:.1f}] | "
                 f"{base['challenge']['accuracy'] * 100:.1f} [{bc[0] * 100:.1f}, {bc[1] * 100:.1f}] | - |")
    for name, runs in results.items():
        t, c = runs[0]["test"], runs[0]["challenge"]
        lines.append(
            f"| {name} | {t['accuracy'] * 100:.1f} [{t['accuracy_wilson_95ci'][0] * 100:.1f}, {t['accuracy_wilson_95ci'][1] * 100:.1f}] | "
            f"{c['accuracy'] * 100:.1f} [{c['accuracy_wilson_95ci'][0] * 100:.1f}, {c['accuracy_wilson_95ci'][1] * 100:.1f}] | "
            f"{c['macro_f1']:.3f} [{c['macro_f1_95ci'][0]:.3f}, {c['macro_f1_95ci'][1]:.3f}] |")

    report = os.path.join(out_dir, "classification_ablation.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved ablation report to {report}")


if __name__ == "__main__":
    main()
