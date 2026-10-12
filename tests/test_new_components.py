"""Unit tests for tracking, summarisation and multi-task helpers (no model downloads)."""

import pytest

from src.tracking import ExperimentTracker, flatten


def test_flatten_nested_params():
    assert flatten({"lora": {"r": 16, "alpha": 32}, "lr": 2e-4}) == {"lora.r": 16, "lora.alpha": 32, "lr": 2e-4}


def test_tracker_is_noop_when_disabled():
    with ExperimentTracker("unit-test", "noop", enabled=False) as tracker:
        tracker.log_params({"a": 1})
        tracker.log_metrics({"x": 0.5, "ci": (0.1, 0.9)})
    assert tracker._mlflow is None


def test_lead_sentence_and_rouge():
    pytest.importorskip("rouge_score")
    from src.train_summarisation import lead_sentence, rouge_scores, summarise_scores
    doc = "The ATO administers tax law. It also issues public rulings."
    assert lead_sentence(doc) == "The ATO administers tax law."
    per_doc = rouge_scores(["the ato administers tax law"], ["The ATO administers tax law."])
    assert per_doc[0]["rouge1"] == pytest.approx(1.0)
    summary = summarise_scores(per_doc * 3, rounds=20)
    assert summary["rougeL_95ci"][0] <= summary["rougeL"] <= summary["rougeL_95ci"][1]


def test_squad_metrics_and_round_robin():
    pytest.importorskip("torch")
    from src.train_multitask import qa_exact, qa_f1, round_robin_schedule
    assert qa_exact("The 21 days.", "21 days") == 1.0
    assert qa_f1("within 21 days", "21 days") == pytest.approx(0.8)
    schedule = round_robin_schedule({"classification": 3, "ner": 1, "qa": 2})
    assert schedule[:3] == ["classification", "ner", "qa"]
    assert len(schedule) == 9 and schedule.count("ner") == 3
