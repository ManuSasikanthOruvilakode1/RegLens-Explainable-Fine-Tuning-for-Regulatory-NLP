"""Transformer sequence classification with full fine-tuning, LoRA, rank-stabilised LoRA and 4-bit QLoRA.

Usage (from the project root):
    uv run python src/train_transformer.py --method lora
    uv run python src/train_transformer.py --method qlora --base-model roberta-base
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import log_loss

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluate import Evaluator
from src.error_analysis import ErrorAnalyzer
from src.model_utils import (
    EarlyStopping, build_optimizer_and_scheduler, count_parameters, get_device,
    load_config, load_peft_model, require_deep_learning, set_seed,
)
from src.preprocess import TextPreprocessor
from src.tracking import ExperimentTracker


class TransformerClassifier:
    """Fine-tunes a pretrained encoder for sequence classification with an optional PEFT method."""

    def __init__(self, base_model: str, id2label: Dict[int, str], method: str = "lora",
                 lora_cfg: Optional[Dict[str, Any]] = None, train_cfg: Optional[Dict[str, Any]] = None,
                 quant_cfg: Optional[Dict[str, Any]] = None, max_seq_length: int = 128,
                 device: str = "auto", seed: int = 42, reinit_pooler: bool = False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.seed = seed
        set_seed(seed)
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.train_cfg = train_cfg or {}
        self.max_seq_length = max_seq_length
        self.device = get_device(device)
        self.base_model = base_model

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model, self.method = load_peft_model(
            AutoModelForSequenceClassification, base_model, method, lora_cfg or {},
            task_type="SEQ_CLS", device=self.device, quant_cfg=quant_cfg,
            num_labels=len(self.id2label), id2label=self.id2label, label2id=self.label2id,
        )
        if self.method == "qlora":
            self.device = torch.device("cuda")
        if reinit_pooler:
            self._reinit_pooler()
        self.param_counts = count_parameters(self.model)

    def _reinit_pooler(self) -> None:
        """Replaces the pretrained pooler with a freshly initialised one (control experiment)."""
        import torch
        std = getattr(self.model.config, "initializer_range", 0.02)
        for name, module in self.model.named_modules():
            if name.endswith("pooler.dense"):
                with torch.no_grad():
                    module.weight.normal_(0.0, std)
                    module.bias.zero_()
                return
        raise ValueError("model has no pooler to re-initialise")

    def _batches(self, texts: List[str], labels: Optional[List[int]], batch_size: int, shuffle: bool):
        import torch
        order = np.arange(len(texts))
        if shuffle:
            np.random.default_rng(self.seed + self._epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_seq_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if labels is not None:
                enc["labels"] = torch.tensor([labels[i] for i in idx], device=self.device)
            yield idx, enc

    def _autocast(self):
        import contextlib
        import torch
        if self.device.type == "cuda" and self.method != "qlora":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def fit(self, train_texts: List[str], train_labels: List[int],
            dev_texts: Optional[List[str]] = None, dev_labels: Optional[List[int]] = None,
            shuffle: bool = True, loss_fn: Optional[Callable] = None) -> List[Dict[str, float]]:
        """Trains with gradient accumulation, clipping, cosine warmup and dev-set early stopping.

        Set shuffle=False to preserve a curriculum ordering of the training texts.
        loss_fn(logits, labels, sample_indices) replaces the default cross-entropy, e.g. for
        knowledge distillation against pre-computed teacher logits.
        """
        import torch

        cfg = self.train_cfg
        batch_size = int(cfg.get("batch_size", 16))
        accum = int(cfg.get("gradient_accumulation_steps", 1))
        epochs = int(cfg.get("epochs", 5))
        lr = float(cfg.get("learning_rate_full" if self.method == "full" else "learning_rate_peft", 2e-4))
        steps_per_epoch = -(-len(train_texts) // batch_size)
        total_steps = -(-steps_per_epoch * epochs // accum)

        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model, lr, float(cfg.get("weight_decay", 0.01)), total_steps,
            float(cfg.get("warmup_ratio", 0.1)), cfg.get("scheduler", "cosine"),
        )
        stopper = EarlyStopping(patience=int(cfg.get("early_stopping_patience", 2)))
        max_norm = float(cfg.get("max_grad_norm", 1.0))
        best_state = None
        history = []

        for epoch in range(epochs):
            self._epoch = epoch
            self.model.train()
            running, n_batches = 0.0, 0
            for step, (idx, batch) in enumerate(self._batches(train_texts, train_labels, batch_size, shuffle)):
                with self._autocast():
                    if loss_fn is None:
                        loss = self.model(**batch).loss / accum
                    else:
                        labels = batch.pop("labels")
                        loss = loss_fn(self.model(**batch).logits.float(), labels, idx) / accum
                loss.backward()
                running += loss.item() * accum
                n_batches += 1
                if (step + 1) % accum == 0 or (step + 1) == steps_per_epoch:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            record = {"epoch": epoch + 1, "train_loss": round(running / max(n_batches, 1), 4)}
            if dev_texts:
                dev_pred = self.predict(dev_texts)
                evaluator = Evaluator()
                record["dev_macro_f1"] = round(evaluator.compute_metrics(dev_labels, dev_pred.tolist())["macro_f1"], 4)
                if stopper.step(record["dev_macro_f1"]):
                    best_state = {n: p.detach().cpu().clone()
                                  for n, p in self.model.named_parameters() if p.requires_grad}
            history.append(record)
            print(f"   epoch {record['epoch']}: " + ", ".join(f"{k}={v}" for k, v in record.items() if k != "epoch"))
            if dev_texts and stopper.should_stop:
                print(f"   early stopping (no dev improvement for {stopper.patience} epochs)")
                break

        if best_state is not None:
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for n, value in best_state.items():
                    params[n].copy_(value.to(params[n].device))
        return history

    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        import torch
        self.model.eval()
        self._epoch = 0
        out = []
        with torch.no_grad():
            for _, batch in self._batches(texts, None, batch_size, shuffle=False):
                with self._autocast():
                    logits = self.model(**batch).logits.float()
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict_logits(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        import torch
        self.model.eval()
        self._epoch = 0
        out = []
        with torch.no_grad():
            for _, batch in self._batches(texts, None, batch_size, shuffle=False):
                with self._autocast():
                    out.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, texts: List[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)

    def save(self, output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"base_model": self.base_model, "method": self.method,
                       "id2label": self.id2label, **(metadata or {})}, f, indent=2)
        return output_dir


def load_classification_splits(cfg: Dict[str, Any]):
    ds = cfg["dataset"]
    splits = {name: TextPreprocessor.load_jsonl(os.path.join(BASE_DIR, ds[f"{name}_path"]))
              for name in ("train", "dev", "test", "challenge")
              if os.path.exists(os.path.join(BASE_DIR, ds.get(f"{name}_path", "__missing__")))}
    labels = sorted({r[ds["label_column"]] for recs in splits.values() for r in recs})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    def unpack(records):
        return [r[ds["text_column"]] for r in records], [label2id[r[ds["label_column"]]] for r in records]

    return {name: unpack(recs) for name, recs in splits.items()}, id2label


def evaluate_on_test(clf: TransformerClassifier, test_texts, test_labels, id2label, bootstrap_rounds=500):
    start = time.perf_counter()
    probs = clf.predict_proba(test_texts)
    latency_ms = (time.perf_counter() - start) * 1000 / max(len(test_texts), 1)
    preds = np.argmax(probs, axis=1).tolist()
    evaluator = Evaluator(bootstrap_rounds=bootstrap_rounds, seed=42)
    metrics = evaluator.compute_metrics(test_labels, preds)
    metrics.update(evaluator.compute_bootstrap_ci(test_labels, preds))
    n_correct = int(sum(int(p == t) for p, t in zip(preds, test_labels)))
    metrics["accuracy_wilson_95ci"] = Evaluator.wilson_interval(n_correct, len(test_labels))
    metrics["log_loss"] = round(float(log_loss(test_labels, probs, labels=sorted(id2label))), 4)
    metrics["latency_ms_per_sample"] = round(latency_ms, 3)
    errors = ErrorAnalyzer(id2label).analyze(test_texts, test_labels, preds, probs)
    return metrics, errors, probs


def metric_lines(metrics: Dict[str, Any]) -> List[str]:
    acc_ci, f1_ci, wil = metrics["accuracy_95ci"], metrics["macro_f1_95ci"], metrics["accuracy_wilson_95ci"]
    return [
        f"- **Accuracy**: {metrics['accuracy'] * 100:.1f}% (bootstrap 95% CI: [{acc_ci[0] * 100:.1f}%, {acc_ci[1] * 100:.1f}%]; "
        f"Wilson 95% CI: [{wil[0] * 100:.1f}%, {wil[1] * 100:.1f}%])",
        f"- **Macro-F1**: {metrics['macro_f1']:.4f} (bootstrap 95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}])",
        f"- **Macro precision / recall**: {metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f}",
        f"- **Log-loss**: {metrics['log_loss']}",
        f"- **Inference latency**: {metrics['latency_ms_per_sample']} ms per sample",
    ]


def write_report(path: str, title: str, run_info: Dict[str, Any], metrics: Dict[str, Any],
                 errors: Dict[str, Any], history: List[Dict[str, float]],
                 challenge: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"# {title}", ""]
    lines += [f"- **{k}**: {v}" for k, v in run_info.items()]
    lines += ["", "## Held-out test metrics", ""] + metric_lines(metrics)
    if challenge:
        lines += ["", "## Plain-English challenge set (out-of-distribution)", ""] + metric_lines(challenge["metrics"])
    lines += [
        "", "## Per-class performance (test)", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    for name, m in errors["classification_report"].items():
        if isinstance(m, dict):
            lines.append(f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1-score']:.2f} | {int(m['support'])} |")
    lines += ["", "## Confusion matrix (rows = true, columns = predicted)", ""]
    lines += ["    " + str(row) for row in errors["confusion_matrix"]]
    lines += ["", "## Training history", "", "| Epoch | Train loss | Dev macro-F1 |", "|---|---|---|"]
    lines += [f"| {h['epoch']} | {h['train_loss']} | {h.get('dev_macro_f1', '-')} |" for h in history]
    for heading, errs in [("test", errors), ("challenge", challenge["errors"] if challenge else None)]:
        if errs and errs["hardest_samples"]:
            lines += ["", f"## Hardest misclassifications ({heading})", "",
                      "| Text | True | Predicted | p(pred) |", "|---|---|---|---|"]
            for s in errs["hardest_samples"][:5]:
                lines.append(f"| {s['text']} | {s['true_label']} | {s['predicted_label']} | {s['predicted_prob']:.2f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run(cfg: Dict[str, Any], method: str, base_model: str, run_name: str, device: str = "auto",
        save_model: bool = True, lora_overrides: Optional[Dict[str, Any]] = None,
        reinit_pooler: bool = False) -> Dict[str, Any]:
    require_deep_learning()
    splits, id2label = load_classification_splits(cfg)
    (train_x, train_y), (dev_x, dev_y), (test_x, test_y) = splits["train"], splits["dev"], splits["test"]
    lora_cfg = {**cfg.get("lora", {}), **(lora_overrides or {})}

    print(f"\n[{run_name}] base={base_model} method={method} lora={lora_cfg if method != 'full' else '-'}")
    clf = TransformerClassifier(base_model, id2label, method, lora_cfg, cfg["training"],
                                cfg.get("quantization"), cfg["model"]["max_seq_length"], device,
                                cfg.get("seed", 42), reinit_pooler)
    pc = clf.param_counts
    print(f"   trainable params: {pc['trainable']:,} / {pc['total']:,} ({100 * pc['trainable'] / pc['total']:.2f}%)")

    start = time.perf_counter()
    history = clf.fit(train_x, train_y, dev_x, dev_y)
    train_seconds = time.perf_counter() - start

    metrics, errors, _ = evaluate_on_test(clf, test_x, test_y, id2label,
                                          cfg.get("evaluation", {}).get("bootstrap_iterations", 500))
    challenge = None
    if "challenge" in splits:
        ch_metrics, ch_errors, _ = evaluate_on_test(clf, *splits["challenge"], id2label,
                                                    cfg.get("evaluation", {}).get("bootstrap_iterations", 500))
        challenge = {"metrics": ch_metrics, "errors": ch_errors}
    for name, m in [("test", metrics), ("challenge", challenge["metrics"] if challenge else None)]:
        if m:
            wil = m["accuracy_wilson_95ci"]
            print(f"   {name:9s} accuracy {m['accuracy'] * 100:.1f}% (Wilson [{wil[0] * 100:.1f}, {wil[1] * 100:.1f}])  "
                  f"macro-F1 {m['macro_f1']:.4f}  log-loss {m['log_loss']}")

    peak_mem = None
    import torch
    if torch.cuda.is_available():
        peak_mem = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        torch.cuda.reset_peak_memory_stats()

    run_info = {
        "Base model": base_model, "Method": clf.method,
        "Trainable parameters": f"{pc['trainable']:,} of {pc['total']:,} ({100 * pc['trainable'] / pc['total']:.2f}%)",
        "Train / dev / test sizes": f"{len(train_x)} / {len(dev_x)} / {len(test_x)}",
        "Training time": f"{train_seconds:.1f} s", "Peak GPU memory": f"{peak_mem} MiB" if peak_mem else "n/a (CPU)",
    }
    if clf.method != "full":
        run_info["LoRA"] = f"r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}, dropout={lora_cfg['lora_dropout']}, targets={lora_cfg['target_modules']}"
    report_path = os.path.join(BASE_DIR, "results", "classification", f"{run_name}.md")
    write_report(report_path, f"Transformer classification - {run_name}", run_info, metrics, errors, history, challenge)

    if save_model:
        clf.save(os.path.join(BASE_DIR, "models", "finetuned", run_name), {"test_metrics": metrics})

    result = {"run_name": run_name, "base_model": base_model, "method": clf.method,
              "trainable_params": pc["trainable"], "total_params": pc["total"],
              "train_seconds": round(train_seconds, 1), "peak_gpu_mib": peak_mem,
              "epochs_run": len(history), "test": metrics,
              "challenge": challenge["metrics"] if challenge else None}
    with ExperimentTracker("classification", run_name) as tracker:
        tracker.log_params({"base_model": base_model, "method": clf.method, "seed": cfg.get("seed", 42),
                            "training": cfg["training"], "lora": lora_cfg if clf.method != "full" else {}})
        for h in history:
            tracker.log_metrics({k: v for k, v in h.items() if k != "epoch"}, step=h["epoch"])
        tracker.log_metrics({"test": metrics, "challenge": challenge["metrics"] if challenge else {},
                             "trainable_params": pc["trainable"], "train_seconds": train_seconds,
                             "peak_gpu_mib": peak_mem or 0})
        tracker.log_artifact(report_path)
    metrics_dir = os.path.join(BASE_DIR, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, f"classification_{run_name}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    del clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "configs", "classification.yaml"))
    parser.add_argument("--method", choices=["full", "lora", "rslora", "qlora"], default=None)
    parser.add_argument("--base-model", default=None, help="HF model id or local path (e.g. a DAPT checkpoint)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--rank", type=int, default=None, help="Override LoRA rank r (alpha set to 2r)")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--reinit-pooler", action="store_true",
                        help="replace the pretrained pooler with a random one (control experiment)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    method = args.method or cfg["training"]["method"]
    base_model = args.base_model or cfg["model"]["base_model"]
    overrides = {"r": args.rank, "lora_alpha": 2 * args.rank} if args.rank else None
    run_name = args.run_name or f"{os.path.basename(base_model.rstrip('/'))}_{method}"
    run(cfg, method, base_model, run_name, args.device, not args.no_save, overrides, args.reinit_pooler)


if __name__ == "__main__":
    main()
