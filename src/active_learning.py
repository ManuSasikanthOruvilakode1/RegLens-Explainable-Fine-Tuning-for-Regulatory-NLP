"""Active learning uncertainty sampling algorithms."""

from typing import List, Dict, Any
import numpy as np


class ActiveLearner:
    """Selects high-value unlabelled instances using prediction entropy and margin sampling."""

    @staticmethod
    def compute_entropy(probs: np.ndarray) -> np.ndarray:
        """Computes Shannon entropy across probability distributions."""
        # Clip to prevent log(0)
        clipped = np.clip(probs, 1e-12, 1.0)
        return -np.sum(clipped * np.log2(clipped), axis=1)

    @classmethod
    def select_uncertain_samples(
        cls,
        unlabelled_texts: List[str],
        probs: np.ndarray,
        budget: int = 5
    ) -> List[Dict[str, Any]]:
        """Selects the top-budget samples with highest prediction uncertainty."""
        entropies = cls.compute_entropy(probs)
        ranked_indices = np.argsort(-entropies)[:budget]

        selected = []
        for idx in ranked_indices:
            selected.append({
                "text": unlabelled_texts[idx],
                "entropy": round(float(entropies[idx]), 4),
                "top_prob": round(float(np.max(probs[idx])), 4)
            })

        return selected
