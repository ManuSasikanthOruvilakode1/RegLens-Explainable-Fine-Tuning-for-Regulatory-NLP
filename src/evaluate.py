"""Systematic evaluation with 95% bootstrap confidence intervals."""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


class Evaluator:
    """Evaluates classification models with rigorous bootstrap confidence intervals."""

    def __init__(self, bootstrap_rounds: int = 500, ci_percentile: float = 95.0, seed: int = 42):
        self.bootstrap_rounds = bootstrap_rounds
        self.ci_percentile = ci_percentile
        self.seed = seed

    def compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculates standard classification performance metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def compute_bootstrap_ci(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Tuple[float, float]]:
        """Computes empirical 95% bootstrap confidence intervals."""
        np.random.seed(self.seed)
        n = len(y_true)
        if n == 0:
            return {}

        boot_f1s = []
        boot_accs = []
        alpha = (100.0 - self.ci_percentile) / 2.0

        for _ in range(self.bootstrap_rounds):
            indices = np.random.choice(n, size=n, replace=True)
            sample_true = [y_true[i] for i in indices]
            sample_pred = [y_pred[i] for i in indices]
            boot_f1s.append(f1_score(sample_true, sample_pred, average="macro", zero_division=0))
            boot_accs.append(accuracy_score(sample_true, sample_pred))

        ci_f1 = (
            float(np.percentile(boot_f1s, alpha)),
            float(np.percentile(boot_f1s, 100.0 - alpha))
        )
        ci_acc = (
            float(np.percentile(boot_accs, alpha)),
            float(np.percentile(boot_accs, 100.0 - alpha))
        )

        return {
            "macro_f1_95ci": (round(ci_f1[0], 4), round(ci_f1[1], 4)),
            "accuracy_95ci": (round(ci_acc[0], 4), round(ci_acc[1], 4))
        }

    @staticmethod
    def wilson_interval(successes: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
        """Wilson score interval for a proportion.

        Unlike the percentile bootstrap it stays informative at the boundaries: a perfect
        42/42 gives roughly [0.916, 1.0] instead of a degenerate [1.0, 1.0].
        """
        if n == 0:
            return (0.0, 1.0)
        p = successes / n
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
        return (round(float(max(0.0, centre - half)), 4), round(float(min(1.0, centre + half)), 4))
