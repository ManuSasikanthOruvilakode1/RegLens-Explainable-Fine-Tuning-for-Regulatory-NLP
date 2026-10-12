"""Knowledge distillation loss and student-teacher supervision.

    L_total = alpha * T^2 * KL(p_teacher^T || p_student^T) + (1 - alpha) * CE(y, p_student)

`DistillationLoss` is a dependency-free NumPy reference implementation; `torch_distillation_loss`
is the differentiable version used for training. Running this module trains a roberta-base teacher,
then a distilbert student with and without the teacher's soft targets:

    uv run python src/distill.py
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class DistillationLoss:
    """Blends soft KL divergence teacher loss and hard ground-truth cross-entropy loss."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha

    def compute_soft_targets(self, teacher_logits: np.ndarray) -> np.ndarray:
        """Applies temperature scaling to soften probability distribution."""
        scaled = teacher_logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    def blend_loss(
        self,
        student_probs: np.ndarray,
        teacher_soft_probs: np.ndarray,
        true_labels: np.ndarray
    ) -> float:
        """Computes blended scalar loss value for verification."""
        # Cross entropy loss
        n = len(true_labels)
        log_preds = np.log(np.clip(student_probs[np.arange(n), true_labels], 1e-12, 1.0))
        ce_loss = -np.mean(log_preds)

        # KL divergence
        kl_div = np.sum(teacher_soft_probs * (np.log(np.clip(teacher_soft_probs, 1e-12, 1.0)) - np.log(np.clip(student_probs, 1e-12, 1.0))), axis=1)
        kl_loss = float(np.mean(kl_div))

        total_loss = self.alpha * (self.temperature ** 2) * kl_loss + (1 - self.alpha) * ce_loss
        return float(total_loss)


def torch_distillation_loss(student_logits, teacher_logits, labels, temperature: float = 4.0, alpha: float = 0.7):
    """Differentiable blended loss. The T^2 factor keeps soft-target gradients on the same scale as CE."""
    import torch.nn.functional as F

    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean")
    ce = F.cross_entropy(student_logits, labels)
    return alpha * temperature ** 2 * kl + (1 - alpha) * ce


def _summary(name, model_id, clf, metrics, challenge, seconds):
    return {"model": name, "base_model": model_id, "method": clf.method,
            "total_params": clf.param_counts["total"], "trainable_params": clf.param_counts["trainable"],
            "train_seconds": round(seconds, 1), "test": metrics, "challenge": challenge}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", default="roberta-base")
    parser.add_argument("--student", default="distilbert-base-uncased")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    from src.model_utils import load_config, require_deep_learning, set_seed
    from src.train_transformer import TransformerClassifier, evaluate_on_test, load_classification_splits
    require_deep_learning()
    import torch

    cfg = load_config(os.path.join(BASE_DIR, "configs", "classification.yaml"))
    splits, id2label = load_classification_splits(cfg)
    (train_x, train_y), (dev_x, dev_y) = splits["train"], splits["dev"]
    max_len = cfg["model"]["max_seq_length"]
    train_cfg = copy.deepcopy(cfg["training"])
    results = []

    def evaluate(name, model_id, clf, seconds):
        metrics, _, _ = evaluate_on_test(clf, *splits["test"], id2label)
        challenge, _, _ = evaluate_on_test(clf, *splits["challenge"], id2label)
        results.append(_summary(name, model_id, clf, metrics, challenge, seconds))
        print(f"   {name}: test acc {metrics['accuracy'] * 100:.1f}%  challenge acc {challenge['accuracy'] * 100:.1f}%  "
              f"latency {metrics['latency_ms_per_sample']} ms")

    # 1. Teacher: full fine-tuning of the larger model
    print(f"\n[teacher] full fine-tuning {args.teacher}")
    teacher = TransformerClassifier(args.teacher, id2label, "full", train_cfg=train_cfg, max_seq_length=max_len)
    start = time.perf_counter()
    teacher.fit(train_x, train_y, dev_x, dev_y)
    evaluate("teacher", args.teacher, teacher, time.perf_counter() - start)
    teacher_logits = torch.tensor(teacher.predict_logits(train_x))
    del teacher
    torch.cuda.empty_cache()

    # 2. Student baseline: same recipe, hard labels only
    print(f"\n[student-ce] full fine-tuning {args.student} on hard labels")
    set_seed(42)
    student_ce = TransformerClassifier(args.student, id2label, "full", train_cfg=train_cfg, max_seq_length=max_len)
    start = time.perf_counter()
    student_ce.fit(train_x, train_y, dev_x, dev_y)
    evaluate("student_hard_labels", args.student, student_ce, time.perf_counter() - start)
    del student_ce
    torch.cuda.empty_cache()

    # 3. Distilled student: blended soft (teacher) + hard (label) loss
    print(f"\n[student-kd] distilling into {args.student} (T={args.temperature}, alpha={args.alpha})")
    set_seed(42)
    student_kd = TransformerClassifier(args.student, id2label, "full", train_cfg=train_cfg, max_seq_length=max_len)

    def kd_loss(logits, labels, idx):
        t = teacher_logits[idx].to(logits.device)
        return torch_distillation_loss(logits, t, labels, args.temperature, args.alpha)

    start = time.perf_counter()
    student_kd.fit(train_x, train_y, dev_x, dev_y, loss_fn=kd_loss)
    evaluate("student_distilled", args.student, student_kd, time.perf_counter() - start)
    student_kd.save(os.path.join(BASE_DIR, "models", "distilled", "distilbert_student"),
                    {"teacher": args.teacher, "temperature": args.temperature, "alpha": args.alpha})

    os.makedirs(os.path.join(BASE_DIR, "results", "metrics"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "results", "metrics", "distillation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    def ci(m):
        w = m["accuracy_wilson_95ci"]
        return f"{m['accuracy'] * 100:.1f} [{w[0] * 100:.1f}, {w[1] * 100:.1f}]"

    lines = [
        "# Knowledge distillation: teacher vs student", "",
        f"Teacher `{args.teacher}` (full fine-tuning) -> student `{args.student}`. "
        f"Loss = {args.alpha} * T^2 * KL + {1 - args.alpha:.1f} * CE with T = {args.temperature}. "
        "Teacher soft targets are computed on the training split only; dev is used for early stopping.", "",
        "| Model | Params | Test acc % (Wilson 95% CI) | Test macro-F1 (bootstrap CI) | Challenge acc % (Wilson 95% CI) | Challenge macro-F1 | Challenge log-loss | Latency ms/sample |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        t, c = r["test"], r["challenge"]
        lines.append(f"| {r['model']} | {r['total_params'] / 1e6:.1f}M | {ci(t)} | {t['macro_f1']:.3f} "
                     f"[{t['macro_f1_95ci'][0]:.3f}, {t['macro_f1_95ci'][1]:.3f}] | {ci(c)} | {c['macro_f1']:.3f} | "
                     f"{c['log_loss']} | {t['latency_ms_per_sample']} |")
    with open(os.path.join(BASE_DIR, "results", "distillation_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    from src.tracking import ExperimentTracker
    for r in results:
        with ExperimentTracker("distillation", r["model"]) as tracker:
            tracker.log_params({"base_model": r["base_model"], "teacher": args.teacher,
                                "temperature": args.temperature, "alpha": args.alpha})
            tracker.log_metrics({"test": r["test"], "challenge": r["challenge"], "total_params": r["total_params"],
                                 "train_seconds": r["train_seconds"]})
            tracker.log_artifact(os.path.join(BASE_DIR, "results", "distillation_comparison.md"))
    print("\nSaved results/distillation_comparison.md")


if __name__ == "__main__":
    main()
