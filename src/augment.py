"""Easy Data Augmentation (EDA) techniques for text classification."""

import random
from typing import List, Dict, Any


class TextAugmenter:
    """Performs rule-based data augmentation (synonym swap, random insertion, random swap)."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.synonyms = {
            "director": ["executive", "board member", "officer"],
            "statutory": ["legislated", "legal", "prescribed"],
            "employee": ["worker", "staff member", "personnel"],
            "employer": ["company", "enterprise", "business"],
            "rates": ["tariffs", "schedules", "percentages"],
            "obligations": ["duties", "responsibilities", "requirements"],
        }

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replaces up to n words in the text with known domain synonyms."""
        words = text.split()
        replaceable = [i for i, w in enumerate(words) if w.lower() in self.synonyms]
        if not replaceable:
            return text

        targets = random.sample(replaceable, min(n, len(replaceable)))
        for idx in targets:
            w = words[idx].lower()
            syn = random.choice(self.synonyms[w])
            words[idx] = syn

        return " ".join(words)

    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swaps two words in the text n times."""
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def augment_dataset(self, records: List[Dict[str, Any]], multiplier: int = 1) -> List[Dict[str, Any]]:
        """Augments dataset records by producing synthetic variants."""
        augmented = list(records)
        for rec in records:
            for _ in range(multiplier):
                new_text = self.synonym_replacement(rec["text"], n=1)
                augmented.append({
                    "text": new_text,
                    "label": rec["label"],
                    "augmented": True
                })
        return augmented
