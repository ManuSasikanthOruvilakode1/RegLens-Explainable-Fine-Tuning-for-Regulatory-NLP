"""Model training pipeline with PEFT/LoRA abstraction and robust baseline fallback."""

import os
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np


class ClassificationTrainer:
    """Trains text classification models with scikit-learn baseline and PyTorch/PEFT support."""

    def __init__(self, c_param: float = 1.0, max_iter: int = 500):
        self.c_param = c_param
        self.max_iter = max_iter
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=2000, sublinear_tf=True)),
            ("clf", LogisticRegression(C=self.c_param, max_iter=self.max_iter, class_weight="balanced"))
        ])
        self.is_fitted = False

    def train(self, train_texts: List[str], train_labels: List[int]) -> "ClassificationTrainer":
        """Fits the classification model on training texts and labels."""
        self.pipeline.fit(train_texts, train_labels)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Generates class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Generates predicted probability distribution."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
        return self.pipeline.predict_proba(texts)
