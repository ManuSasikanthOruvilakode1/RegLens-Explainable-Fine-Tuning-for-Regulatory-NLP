"""Deep dive error analysis and misclassification inspection."""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class ErrorAnalyzer:
    """Categorises model prediction errors and identifies hardest failure cases."""

    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label

    def analyze(
        self,
        texts: List[str],
        y_true: List[int],
        y_pred: List[int],
        probs: np.ndarray
    ) -> Dict[str, Any]:
        """Produces confusion matrix and ranked misclassified samples."""
        labels = sorted(list(self.id2label.keys()))
        target_names = [self.id2label[i] for i in labels]

        cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)

        misclassified = []
        for i, (text, true_lbl, pred_lbl) in enumerate(zip(texts, y_true, y_pred)):
            if true_lbl != pred_lbl:
                pred_prob = float(probs[i][pred_lbl])
                true_prob = float(probs[i][true_lbl])
                # Margin between predicted probability and true label probability
                margin = pred_prob - true_prob
                misclassified.append({
                    "text": text,
                    "true_label": self.id2label[true_lbl],
                    "predicted_label": self.id2label[pred_lbl],
                    "predicted_prob": round(pred_prob, 4),
                    "true_prob": round(true_prob, 4),
                    "confidence_error_margin": round(margin, 4)
                })

        # Rank by hardest error margin
        misclassified.sort(key=lambda x: x["confidence_error_margin"], reverse=True)

        return {
            "confusion_matrix": cm,
            "classification_report": report,
            "total_errors": len(misclassified),
            "error_rate": round(len(misclassified) / max(len(texts), 1), 4),
            "hardest_samples": misclassified
        }
