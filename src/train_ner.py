"""Token classification (NER) for Australian regulatory entities in CoNLL BIO format.

Entity types: GOV_BODY, REGULATORY_ACT, FINANCIAL_TERM, ORG (9 BIO labels including O).
Evaluated with seqeval in strict IOB2 mode (exact span and type must match), with 95% bootstrap
confidence intervals resampled over sentences. A training-set gazetteer gives a non-neural baseline.

    uv run python src/train_ner.py
    uv run python src/train_ner.py --method full
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.model_utils import (
    EarlyStopping, build_optimizer_and_scheduler, count_parameters, get_device,
    load_config, load_peft_model, require_deep_learning, set_seed,
)
from src.preprocess import TextPreprocessor

IGNORE_INDEX = -100


def build_label_list(entities: Sequence[str]) -> List[str]:
    return ["O"] + [f"{prefix}-{ent}" for ent in entities for prefix in ("B", "I")]


def align_labels_to_subwords(word_ids: Sequence[Optional[int]], word_labels: Sequence[int]) -> List[int]:
    """Gives each word's label to its first sub-token; continuation sub-tokens and special tokens get -100."""
    aligned, previous = [], None
    for wid in word_ids:
        if wid is None or wid == previous:
            aligned.append(IGNORE_INDEX)
        else:
            aligned.append(word_labels[wid])
        previous = wid
    return aligned


def span_f1(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Any]:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
    from seqeval.scheme import IOB2
    kw = {"mode": "strict", "scheme": IOB2}
    return {
        "precision": float(precision_score(y_true, y_pred, **kw)),
        "recall": float(recall_score(y_true, y_pred, **kw)),
        "f1": float(f1_score(y_true, y_pred, **kw)),
        "per_type": classification_report(y_true, y_pred, output_dict=True, zero_division=0, **kw),
    }


def bootstrap_span_f1(y_true, y_pred, rounds: int = 500, seed: int = 42):
    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        scores.append(f1_score([y_true[i] for i in idx], [y_pred[i] for i in idx],
                               mode="strict", scheme=IOB2, zero_division=0))
    return (round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4))


class GazetteerTagger:
    """Longest-match dictionary tagger built from the entity spans seen in training."""

    def __init__(self):
        self.spans: Dict[tuple, str] = {}
        self.max_len = 1

    def fit(self, sentences: List[Dict[str, Any]]) -> "GazetteerTagger":
        for s in sentences:
            current, etype = [], None
            for tok, lab in list(zip(s["tokens"], s["labels"])) + [("", "O")]:
                if lab.startswith("I-") and current:
                    current.append(tok.lower())
                    continue
                if current:
                    self.spans[tuple(current)] = etype
                    self.max_len = max(self.max_len, len(current))
                current, etype = ([tok.lower()], lab[2:]) if lab.startswith("B-") else ([], None)
        return self

    def predict(self, tokens: List[str]) -> List[str]:
        lowered = [t.lower() for t in tokens]
        tags, i = ["O"] * len(tokens), 0
        while i < len(tokens):
            for length in range(min(self.max_len, len(tokens) - i), 0, -1):
                etype = self.spans.get(tuple(lowered[i:i + length]))
                if etype:
                    tags[i] = f"B-{etype}"
                    for j in range(i + 1, i + length):
                        tags[j] = f"I-{etype}"
                    i += length
                    break
            else:
                i += 1
        return tags


class TransformerTokenClassifier:
    """Fine-tunes an encoder for BIO token classification with optional LoRA adapters."""

    def __init__(self, base_model: str, labels: List[str], method: str = "lora",
                 lora_cfg: Optional[Dict[str, Any]] = None, train_cfg: Optional[Dict[str, Any]] = None,
                 max_seq_length: int = 128, device: str = "auto", seed: int = 42):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        set_seed(seed)
        self.seed = seed
        self.labels = labels
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for i, l in enumerate(labels)}
        self.train_cfg = train_cfg or {}
        self.max_seq_length = max_seq_length
        self.device = get_device(device)
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, add_prefix_space=True)
        self.model, self.method = load_peft_model(
            AutoModelForTokenClassification, base_model, method, lora_cfg or {}, task_type="TOKEN_CLS",
            device=self.device, num_labels=len(labels), id2label=self.id2label, label2id=self.label2id,
        )
        self.param_counts = count_parameters(self.model)

    def _encode(self, sentences: List[Dict[str, Any]], with_labels: bool = True):
        import torch
        enc = self.tokenizer([s["tokens"] for s in sentences], is_split_into_words=True, truncation=True,
                             max_length=self.max_seq_length, padding=True, return_tensors="pt")
        word_ids = [enc.word_ids(i) for i in range(len(sentences))]
        batch = {k: v.to(self.device) for k, v in enc.items()}
        if with_labels:
            batch["labels"] = torch.tensor([
                align_labels_to_subwords(word_ids[i], [self.label2id[l] for l in s["labels"]])
                for i, s in enumerate(sentences)], device=self.device)
        return batch, word_ids

    def fit(self, train: List[Dict[str, Any]], dev: Optional[List[Dict[str, Any]]] = None):
        import torch
        cfg = self.train_cfg
        bs, epochs = int(cfg.get("batch_size", 8)), int(cfg.get("epochs", 20))
        lr = float(cfg.get("learning_rate_full" if self.method == "full" else "learning_rate_peft", 5e-4))
        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model, lr, float(cfg.get("weight_decay", 0.01)), -(-len(train) // bs) * epochs,
            float(cfg.get("warmup_ratio", 0.1)), cfg.get("scheduler", "cosine"))
        stopper = EarlyStopping(patience=int(cfg.get("early_stopping_patience", 4)))
        rng = np.random.default_rng(self.seed)
        best_state, history = None, []

        for epoch in range(epochs):
            self.model.train()
            order = rng.permutation(len(train))
            losses = []
            for start in range(0, len(order), bs):
                batch, _ = self._encode([train[i] for i in order[start:start + bs]])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    loss = self.model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                               float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
            record = {"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 4)}
            if dev:
                record["dev_span_f1"] = round(span_f1([s["labels"] for s in dev], self.predict(dev))["f1"], 4)
                if stopper.step(record["dev_span_f1"]):
                    best_state = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}
            history.append(record)
            print(f"   epoch {record['epoch']}: " + ", ".join(f"{k}={v}" for k, v in record.items() if k != "epoch"))
            if dev and stopper.should_stop:
                print(f"   early stopping (no dev improvement for {stopper.patience} epochs)")
                break

        if best_state is not None:
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for n, v in best_state.items():
                    params[n].copy_(v.to(params[n].device))
        return history

    def predict(self, sentences: List[Dict[str, Any]], batch_size: int = 32) -> List[List[str]]:
        import torch
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(sentences), batch_size):
                chunk = sentences[start:start + batch_size]
                batch, word_ids = self._encode(chunk, with_labels=False)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    pred_ids = self.model(**batch).logits.argmax(-1).cpu().numpy()
                for i, s in enumerate(chunk):
                    tags, seen = ["O"] * len(s["tokens"]), set()
                    for pos, wid in enumerate(word_ids[i]):
                        if wid is not None and wid not in seen:
                            tags[wid] = self.id2label[int(pred_ids[i][pos])]
                            seen.add(wid)
                    predictions.append(repair_bio(tags))
        return predictions

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return output_dir


def repair_bio(tags: List[str]) -> List[str]:
    """Converts an orphan I-X (not preceded by B-X / I-X) into B-X so spans are well formed."""
    fixed = list(tags)
    for i, tag in enumerate(fixed):
        if tag.startswith("I-") and (i == 0 or fixed[i - 1][2:] != tag[2:]):
            fixed[i] = "B-" + tag[2:]
    return fixed


def format_scores(name: str, scores: Dict[str, Any], ci) -> str:
    return f"| {name} | {scores['precision']:.3f} | {scores['recall']:.3f} | {scores['f1']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "configs", "ner.yaml"))
    parser.add_argument("--method", choices=["full", "lora", "rslora"], default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    train, dev, test = (TextPreprocessor.load_jsonl(os.path.join(BASE_DIR, ds[f"{k}_path"])) for k in ("train", "dev", "test"))
    labels = build_label_list(ds["entities"])
    y_test = [s["labels"] for s in test]
    rounds = cfg["evaluation"].get("bootstrap_iterations", 500)
    n_entities = sum(t.startswith("B-") for s in test for t in s["labels"])
    print(f"NER: train={len(train)} dev={len(dev)} test={len(test)} sentences ({n_entities} test entities), {len(labels)} labels")

    gaz_pred = [GazetteerTagger().fit(train).predict(s["tokens"]) for s in test]
    gaz = span_f1(y_test, gaz_pred)
    gaz_ci = bootstrap_span_f1(y_test, gaz_pred, rounds)
    print(f"   gazetteer baseline span-F1 {gaz['f1']:.3f} {gaz_ci}")

    require_deep_learning()
    method = args.method or cfg["training"]["method"]
    base_model = args.base_model or cfg["model"]["base_model"]
    print(f"\n[{base_model} / {method}]")
    tagger = TransformerTokenClassifier(base_model, labels, method, cfg.get("lora"), cfg["training"],
                                        cfg["model"]["max_seq_length"], args.device, cfg.get("seed", 42))
    pc = tagger.param_counts
    start = time.perf_counter()
    history = tagger.fit(train, dev)
    seconds = time.perf_counter() - start
    pred = tagger.predict(test)
    scores = span_f1(y_test, pred)
    ci = bootstrap_span_f1(y_test, pred, rounds)
    print(f"   test strict span-F1 {scores['f1']:.3f} {ci}")

    run_name = f"{os.path.basename(base_model.rstrip('/'))}_{tagger.method}"
    tagger.save(os.path.join(BASE_DIR, "models", "finetuned", f"ner_{run_name}"))

    lines = [
        "# Named entity recognition (strict span-level, seqeval IOB2)", "",
        f"- **Data**: {len(train)} / {len(dev)} / {len(test)} train / dev / test sentences; {n_entities} gold entities in test",
        f"- **Model**: `{base_model}`, method `{tagger.method}`, trainable {pc['trainable']:,} of {pc['total']:,} parameters",
        f"- **Training**: {len(history)} epochs (early stopping on dev span-F1), {seconds:.1f} s",
        "- **CI**: 95% bootstrap over test sentences (500 resamples)", "",
        "| System | Precision | Recall | F1 | F1 95% CI |", "|---|---|---|---|---|",
        format_scores("Gazetteer (training spans, longest match)", gaz, gaz_ci),
        format_scores(f"{base_model} + {tagger.method}", scores, ci),
        "", "## Per entity type (transformer)", "", "| Type | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    for ent in ds["entities"]:
        m = scores["per_type"].get(ent)
        if m:
            lines.append(f"| {ent} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1-score']:.2f} | {int(m['support'])} |")
    lines += ["", "## Test predictions with errors", ""]
    for s, p in zip(test, pred):
        if p != s["labels"]:
            diff = [f"{tok} ({g} -> {q})" for tok, g, q in zip(s["tokens"], s["labels"], p) if g != q]
            lines.append(f"- {' '.join(s['tokens'])}  \n  differences: {'; '.join(diff)}")
    os.makedirs(os.path.join(BASE_DIR, "results", "metrics"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "results", "ner_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE_DIR, "results", "metrics", "ner.json"), "w", encoding="utf-8") as f:
        json.dump({"gazetteer": {**{k: v for k, v in gaz.items() if k != "per_type"}, "f1_95ci": gaz_ci},
                   "transformer": {"base_model": base_model, "method": tagger.method,
                                   **{k: v for k, v in scores.items() if k != "per_type"}, "f1_95ci": ci,
                                   "epochs": len(history), "train_seconds": round(seconds, 1)}}, f, indent=2)
    from src.tracking import ExperimentTracker
    with ExperimentTracker("ner", run_name) as tracker:
        tracker.log_params({"base_model": base_model, "method": tagger.method, "training": cfg["training"],
                            "lora": cfg.get("lora", {})})
        for h in history:
            tracker.log_metrics({k: v for k, v in h.items() if k != "epoch"}, step=h["epoch"])
        tracker.log_metrics({"test_span_f1": scores["f1"], "test_precision": scores["precision"],
                             "test_recall": scores["recall"], "test_span_f1_95ci": ci, "gazetteer_span_f1": gaz["f1"]})
        tracker.log_artifact(os.path.join(BASE_DIR, "results", "ner_report.md"))
    print("Saved results/ner_report.md")


if __name__ == "__main__":
    main()
