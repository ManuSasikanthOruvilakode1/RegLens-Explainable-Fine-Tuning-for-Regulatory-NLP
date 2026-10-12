"""Unit tests for the PyTorch components. Skipped automatically without the deeplearning extra."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.distill import DistillationLoss, torch_distillation_loss
from src.domain_adapt import IGNORE_INDEX, dynamic_mask
from src.train_ner import GazetteerTagger, align_labels_to_subwords, build_label_list, repair_bio


def test_torch_distillation_loss_matches_numpy_reference():
    rng = np.random.default_rng(0)
    student, teacher = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    labels = np.array([0, 1, 2, 0, 1, 2])
    # with T = 1 the NumPy reference (shared probabilities for KL and CE) is directly comparable
    ref = DistillationLoss(temperature=1.0, alpha=0.7)
    s_probs = ref.compute_soft_targets(student)
    expected = ref.blend_loss(s_probs, ref.compute_soft_targets(teacher), labels)
    got = torch_distillation_loss(torch.tensor(student), torch.tensor(teacher), torch.tensor(labels), 1.0, 0.7)
    assert abs(float(got) - expected) < 1e-6


def test_distillation_loss_is_zero_kl_when_student_matches_teacher():
    logits = torch.tensor([[2.0, 0.1, -1.0]])
    pure_kd = torch_distillation_loss(logits, logits.clone(), torch.tensor([0]), temperature=4.0, alpha=1.0)
    assert float(pure_kd) == pytest.approx(0.0, abs=1e-6)


def test_dynamic_mask_respects_ratios_and_special_tokens():
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(5, 1000, (64, 128))
    special = torch.zeros_like(ids)
    special[:, 0] = 1
    special[:, -1] = 1
    inputs, labels = dynamic_mask(ids, special, mask_token_id=3, vocab_size=1000, mlm_probability=0.15, generator=gen)

    selected = labels != IGNORE_INDEX
    assert not selected[:, 0].any() and not selected[:, -1].any()
    assert 0.13 < selected.float().mean().item() < 0.17
    assert torch.equal(labels[selected], ids[selected])
    masked_share = (inputs[selected] == 3).float().mean().item()
    assert 0.76 < masked_share < 0.84
    assert torch.equal(inputs[~selected], ids[~selected])


def test_label_alignment_uses_first_subword_only():
    word_ids = [None, 0, 0, 1, 2, 2, 2, None]
    assert align_labels_to_subwords(word_ids, [1, 0, 3]) == [-100, 1, -100, 0, 3, -100, -100, -100]


def test_label_list_has_nine_bio_labels():
    labels = build_label_list(["ORG", "REGULATORY_ACT", "GOV_BODY", "FINANCIAL_TERM"])
    assert len(labels) == 9 and labels[0] == "O"


def test_repair_bio_fixes_orphan_inside_tags():
    assert repair_bio(["O", "I-ORG", "I-ORG", "B-GOV_BODY", "I-ORG"]) == ["O", "B-ORG", "I-ORG", "B-GOV_BODY", "B-ORG"]


def test_gazetteer_longest_match():
    train = [{"tokens": ["The", "Fair", "Work", "Act", "applies"],
              "labels": ["O", "B-REGULATORY_ACT", "I-REGULATORY_ACT", "I-REGULATORY_ACT", "O"]},
             {"tokens": ["Fair", "Work"], "labels": ["B-ORG", "I-ORG"]}]
    tagger = GazetteerTagger().fit(train)
    assert tagger.predict(["under", "the", "fair", "work", "act"]) == [
        "O", "O", "B-REGULATORY_ACT", "I-REGULATORY_ACT", "I-REGULATORY_ACT"]
