"""Unit tests for dataset construction helpers and interval estimates (no deep-learning deps)."""

from src.evaluate import Evaluator
from src.model_utils import EarlyStopping, lora_scaling
from src.preprocess import TextPreprocessor


def test_parse_entity_markup_produces_bio_tags():
    tokens, tags = TextPreprocessor.parse_entity_markup(
        "The [Fair Work Commission](GOV_BODY) applies the [Fair Work Act 2009](REGULATORY_ACT) .")
    assert tokens == ["The", "Fair", "Work", "Commission", "applies", "the", "Fair", "Work", "Act", "2009", "."]
    assert tags == ["O", "B-GOV_BODY", "I-GOV_BODY", "I-GOV_BODY", "O", "O",
                    "B-REGULATORY_ACT", "I-REGULATORY_ACT", "I-REGULATORY_ACT", "I-REGULATORY_ACT", "O"]


def test_stratified_split_keeps_label_balance_and_is_disjoint():
    records = [{"text": f"{label} {i}", "label": label} for label in ("a", "b", "c") for i in range(20)]
    keep, held = TextPreprocessor.stratified_split(records, "label", test_fraction=0.25, seed=1)
    assert len(held) == 15 and len(keep) == 45
    assert {r["text"] for r in keep}.isdisjoint({r["text"] for r in held})
    assert sorted(r["label"] for r in held).count("a") == 5


def test_wilson_interval_is_informative_at_perfect_accuracy():
    low, high = Evaluator.wilson_interval(42, 42)
    assert high == 1.0
    assert 0.90 < low < 0.93


def test_wilson_interval_contains_point_estimate():
    low, high = Evaluator.wilson_interval(22, 30)
    assert low < 22 / 30 < high


def test_rslora_scaling_differs_from_lora():
    assert lora_scaling(32, 16, rank_stabilized=False) == 2.0
    assert lora_scaling(32, 16, rank_stabilized=True) == 8.0
    # rsLoRA keeps the effective scale from collapsing as rank grows
    assert lora_scaling(128, 64, True) / lora_scaling(32, 16, True) == 2.0


def test_early_stopping_patience():
    stopper = EarlyStopping(patience=2)
    assert stopper.step(0.5) and not stopper.should_stop
    stopper.step(0.4)
    stopper.step(0.5)
    assert stopper.should_stop
