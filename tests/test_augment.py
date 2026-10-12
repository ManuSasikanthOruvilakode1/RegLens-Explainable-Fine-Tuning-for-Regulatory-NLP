"""Unit tests for TextAugmenter."""

import pytest
from src.augment import TextAugmenter


def test_synonym_replacement():
    aug = TextAugmenter(seed=42)
    text = "The director reviewed the statutory obligations."
    augmented = aug.synonym_replacement(text, n=1)
    assert isinstance(augmented, str)
    assert len(augmented.split()) == len(text.split())


def test_augment_dataset():
    aug = TextAugmenter(seed=42)
    records = [{"text": "The director must comply.", "label": "corporate_governance"}]
    augmented = aug.augment_dataset(records, multiplier=2)
    assert len(augmented) == 3  # Original + 2 synthetic
