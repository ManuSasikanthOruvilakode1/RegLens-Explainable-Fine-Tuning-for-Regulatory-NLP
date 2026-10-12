"""Run the full transformer suite in order. Each stage is a separate process so GPU memory is
released between stages; a failing stage is reported and the remaining stages still run.

    uv run python scripts/run_transformer_suite.py                 # every stage
    uv run python scripts/run_transformer_suite.py --skip ablation # skip the 18-run ablation
"""

import argparse
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = [
    ("data", ["scripts/build_datasets.py"]),
    ("baseline", ["scripts/run_all.py"]),
    ("ablation", ["scripts/run_classification_ablation.py"]),
    ("lora", ["src/train_transformer.py", "--method", "lora"]),
    ("dapt", ["src/domain_adapt.py"]),
    ("dapt_lora", ["src/train_transformer.py", "--method", "lora",
                   "--base-model", "models/domain_adapted/bert-base-uncased-au", "--run-name", "dapt_bert-base-uncased_lora"]),
    ("distill", ["src/distill.py"]),
    ("ner", ["src/train_ner.py"]),
    ("multitask", ["src/train_multitask.py", "--compare"]),
    ("summarise", ["src/train_summarisation.py"]),
    ("explain", ["src/explain.py"]),
    ("export", ["src/export.py"]),
    ("notebooks", ["scripts/build_notebooks.py"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip", nargs="*", default=[], choices=[s for s, _ in STAGES])
    parser.add_argument("--only", nargs="*", default=None, choices=[s for s, _ in STAGES])
    args = parser.parse_args()

    status = []
    for name, cmd in STAGES:
        if name in args.skip or (args.only and name not in args.only):
            continue
        print(f"\n{'=' * 70}\n stage: {name}\n{'=' * 70}", flush=True)
        start = time.perf_counter()
        code = subprocess.call([sys.executable, *cmd], cwd=BASE_DIR)
        status.append((name, code, time.perf_counter() - start))

    print("\nSummary")
    for name, code, seconds in status:
        print(f"   {name:10s} {'ok' if code == 0 else f'FAILED ({code})':12s} {seconds:7.1f} s")
    sys.exit(1 if any(code for _, code, _ in status) else 0)


if __name__ == "__main__":
    main()
