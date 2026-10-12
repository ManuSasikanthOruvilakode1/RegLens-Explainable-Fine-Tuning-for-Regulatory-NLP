"""Inference wrappers around the saved models, shared by the CLI demo and the web app."""

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE_DIR, "models")
ENTITY_TYPES = ["ORG", "REGULATORY_ACT", "GOV_BODY", "FINANCIAL_TERM"]


def default_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DomainClassifier:
    """LoRA-tuned BERT regulatory-domain classifier with Integrated Gradients explanations."""

    def __init__(self, model_dir: str = os.path.join(MODELS, "finetuned", "bert-base-uncased_lora"), device=None):
        from src.explain import IntegratedGradientsExplainer, load_classifier
        self.device = device or default_device()
        self.model, self.tokenizer = load_classifier(model_dir, self.device)
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.explainer = IntegratedGradientsExplainer(self.model, self.tokenizer, self.device)

    def predict(self, text: str) -> Dict[str, float]:
        import torch
        with torch.no_grad():
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            probs = torch.softmax(self.model(**enc).logits, -1)[0].cpu().numpy()
        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}

    def explain(self, text: str) -> List[Tuple[str, float]]:
        """Word-level attributions towards the predicted class, scaled to [-1, 1]."""
        e = self.explainer.explain(text)
        peak = max((abs(v) for v in e["attributions"]), default=1.0) or 1.0
        return [(w, v / peak) for w, v in zip(e["words"], e["attributions"])]


class EntityTagger:
    """LoRA-tuned RoBERTa token classifier for Australian regulatory entities."""

    def __init__(self, model_dir: str = os.path.join(MODELS, "finetuned", "ner_roberta-base_lora"), device=None):
        from peft import PeftModel
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        from src.train_ner import build_label_list
        self.device = device or default_device()
        self.labels = build_label_list(ENTITY_TYPES)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
        base = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=len(self.labels))
        self.model = PeftModel.from_pretrained(base, model_dir).merge_and_unload().to(self.device).eval()

    def tag(self, sentence: str) -> List[Tuple[str, str]]:
        """Returns (word, BIO tag) pairs; punctuation is split from words first."""
        import re
        import torch
        from src.train_ner import repair_bio
        words = re.findall(r"[\w'&()-]+|[^\w\s]", sentence)
        if not words:
            return []
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True,
                             max_length=128).to(self.device)
        with torch.no_grad():
            pred = self.model(**enc).logits.argmax(-1)[0].tolist()
        tags, seen = ["O"] * len(words), set()
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None and wid not in seen:
                tags[wid] = self.labels[pred[pos]]
                seen.add(wid)
        return list(zip(words, repair_bio(tags)))

    def spans(self, sentence: str) -> List[Tuple[str, str]]:
        """Groups BIO tags into (text, entity_type or None) spans, ready for highlighting."""
        out: List[Tuple[str, str]] = []
        for word, tag in self.tag(sentence):
            etype = tag[2:] if tag != "O" else None
            if tag.startswith("I-") and out and out[-1][1] == etype:
                out[-1] = (out[-1][0] + " " + word, etype)
            else:
                out.append((word, etype))
        return out


class Summariser:
    """flan-t5-base with the summarisation LoRA adapter; the adapter can be bypassed for zero-shot output."""

    PROMPT = "Summarize in one sentence: "

    def __init__(self, model_dir: str = os.path.join(MODELS, "finetuned", "summariser_lora"), device=None):
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = device or default_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.model = PeftModel.from_pretrained(base, model_dir).to(self.device).eval()

    def _generate(self, passage: str) -> str:
        import torch
        enc = self.tokenizer(self.PROMPT + passage, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**enc, max_new_tokens=64, num_beams=4)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    def summarise(self, passage: str) -> Dict[str, str]:
        with self.model.disable_adapter():
            zero_shot = self._generate(passage)
        return {"zero_shot": zero_shot, "fine_tuned": self._generate(passage)}


class OnnxDomainClassifier:
    """The same classifier exported to ONNX and INT8-quantised, running on CPU."""

    def __init__(self, model_name: str = "bert-base-uncased_lora", threads: int = 4):
        from transformers import AutoTokenizer
        from src.export import OnnxClassifier
        model_dir = os.path.join(MODELS, "finetuned", model_name)
        with open(os.path.join(model_dir, "training_metadata.json"), encoding="utf-8") as f:
            self.id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
        self.clf = OnnxClassifier(os.path.join(MODELS, "onnx", f"{model_name}.int8.onnx"),
                                  AutoTokenizer.from_pretrained(model_dir), threads)
        self.clf.logits(["warm-up"])

    def predict(self, text: str) -> Tuple[Dict[str, float], float]:
        start = time.perf_counter()
        logits = self.clf.logits([text])[0]
        ms = (time.perf_counter() - start) * 1000
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}, ms
