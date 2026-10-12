"""Unit tests for Evaluator."""

import pytest
from src.evaluate import Evaluator


def test_evaluator_metrics():
    evaluator = Evaluator(bootstrap_rounds=50)
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 2]  # 5 out of 6 correct

    metrics = evaluator.compute_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    ci = evaluator.compute_bootstrap_ci(y_true, y_pred)
    assert "macro_f1_95ci" in ci
    low, high = ci["macro_f1_95ci"]
    assert low <= high
