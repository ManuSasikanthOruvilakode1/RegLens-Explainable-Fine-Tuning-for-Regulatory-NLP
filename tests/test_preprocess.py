"""Unit tests for TextPreprocessor."""

import pytest
from src.preprocess import TextPreprocessor


def test_clean_text():
    pre = TextPreprocessor(lower=True)
    raw = "  Section  180   Corporations Act  "
    cleaned = pre.clean_text(raw)
    assert cleaned == "section 180 corporations act"


def test_encode_labels():
    labels = ["taxation", "workplace_relations", "taxation"]
    encoded, l2i, i2l = TextPreprocessor.encode_labels(labels)
    assert len(encoded) == 3
    assert encoded[0] == encoded[2]
    assert i2l[encoded[0]] == "taxation"
