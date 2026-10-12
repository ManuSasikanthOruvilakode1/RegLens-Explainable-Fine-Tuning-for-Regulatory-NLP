"""Curriculum learning difficulty estimation and sample scheduling."""

from typing import List, Dict, Any


class CurriculumScheduler:
    """Scores training sample difficulty and orders batches from easy to hard."""

    @staticmethod
    def score_difficulty(text: str) -> float:
        """Estimates sample difficulty based on length and lexical diversity."""
        tokens = text.split()
        if not tokens:
            return 0.0
        unique_ratio = len(set(tokens)) / len(tokens)
        # Difficulty is a composite of length and vocabulary complexity
        difficulty = (len(tokens) / 50.0) * 0.5 + unique_ratio * 0.5
        return round(float(difficulty), 4)

    def order_curriculum(self, records: List[Dict[str, Any]], schedule: str = "easy_to_hard") -> List[Dict[str, Any]]:
        """Orders samples according to curriculum strategy."""
        scored = []
        for r in records:
            diff = self.score_difficulty(r["text"])
            scored.append({**r, "difficulty_score": diff})

        if schedule == "easy_to_hard":
            scored.sort(key=lambda x: x["difficulty_score"])
        elif schedule == "anti_curriculum":
            scored.sort(key=lambda x: x["difficulty_score"], reverse=True)
        return scored
