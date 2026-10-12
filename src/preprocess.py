"""Preprocessing, tokenization and dataset loading."""

import json
import os
import random
import re
from typing import List, Dict, Any, Tuple


class TextPreprocessor:
    """Cleans and tokenizes text for downstream classification and transformer ingestion."""

    def __init__(self, lower: bool = True):
        self.lower = lower

    def clean_text(self, text: str) -> str:
        """Normalises whitespace and basic punctuation."""
        text = re.sub(r"\s+", " ", text).strip()
        if self.lower:
            text = text.lower()
        return text

    @staticmethod
    def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
        """Loads JSONL dataset file."""
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def encode_labels(labels: List[str]) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
        """Encodes string labels into integer indices with bidirectional mapping."""
        unique_labels = sorted(list(set(labels)))
        label2id = {l: i for i, l in enumerate(unique_labels)}
        id2label = {i: l for i, l in enumerate(unique_labels)}
        encoded = [label2id[l] for l in labels]
        return encoded, label2id, id2label

    @staticmethod
    def save_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
        """Writes records to a JSONL file, creating parent directories."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def parse_entity_markup(line: str) -> Tuple[List[str], List[str]]:
        """Converts '[span words](TYPE) other words' markup into tokens and BIO tags."""
        tokens, tags = [], []
        cursor = 0
        for match in ENTITY_MARKUP.finditer(line):
            for tok in line[cursor:match.start()].split():
                tokens.append(tok)
                tags.append("O")
            span_tokens = match.group(1).split()
            for i, tok in enumerate(span_tokens):
                tokens.append(tok)
                tags.append(("B-" if i == 0 else "I-") + match.group(2))
            cursor = match.end()
        for tok in line[cursor:].split():
            tokens.append(tok)
            tags.append("O")
        return tokens, tags

    @staticmethod
    def stratified_split(
        records: List[Dict[str, Any]],
        label_key: str,
        test_fraction: float,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Splits records so every label keeps the same proportion in both partitions."""
        rng = random.Random(seed)
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            by_label.setdefault(rec[label_key], []).append(rec)

        keep, held_out = [], []
        for label in sorted(by_label):
            group = list(by_label[label])
            rng.shuffle(group)
            n_held = max(1, round(len(group) * test_fraction))
            held_out.extend(group[:n_held])
            keep.extend(group[n_held:])
        rng.shuffle(keep)
        rng.shuffle(held_out)
        return keep, held_out


ENTITY_MARKUP = re.compile(r"\[([^\]]+)\]\(([A-Z_]+)\)")
