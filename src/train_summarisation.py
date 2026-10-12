"""Abstractive summarisation of Australian regulatory passages with a LoRA-tuned seq2seq model.

Three systems are scored on the same held-out passages with ROUGE-1 / ROUGE-2 / ROUGE-L (F1, Porter
stemming) and 95% bootstrap confidence intervals resampled over documents:

    lead-1       first sentence of the passage (strong extractive baseline for news-style text)
    zero-shot    the instruction-tuned base model with a summarisation prompt, no training
    fine-tuned   the same model after LoRA fine-tuning on the training pairs

    uv run python src/train_summarisation.py
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

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
from src.tracking import ExperimentTracker

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")


def lead_sentence(document: str) -> str:
    """Returns the first sentence (split on '.', '!' or '?' followed by whitespace and a capital)."""
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", document.strip(), maxsplit=1)[0]


def rouge_scores(predictions: List[str], references: List[str]) -> List[Dict[str, float]]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
    return [{k: v.fmeasure for k, v in scorer.score(ref, pred).items()} for pred, ref in zip(predictions, references)]


def summarise_scores(per_doc: List[Dict[str, float]], rounds: int = 500, seed: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    out = {}
    for key in ROUGE_TYPES:
        values = np.array([d[key] for d in per_doc])
        boot = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(rounds)]
        out[key] = round(float(values.mean()), 4)
        out[f"{key}_95ci"] = (round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4))
    return out


class Seq2SeqSummariser:
    def __init__(self, cfg: Dict[str, Any], method: str = "lora", device: str = "auto"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        self.seed = cfg.get("seed", 42)
        set_seed(self.seed)
        m = cfg["model"]
        self.prompt = m["prompt"]
        self.max_src, self.max_tgt, self.beams = m["max_source_length"], m["max_target_length"], m["num_beams"]
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(m["base_model"])
        self.model, self.method = load_peft_model(AutoModelForSeq2SeqLM, m["base_model"], method,
                                                  cfg.get("lora", {}), task_type="SEQ_2_SEQ_LM", device=self.device)
        self.param_counts = count_parameters(self.model)

    def generate(self, documents: List[str], batch_size: int = 8) -> List[str]:
        import torch
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(documents), batch_size):
                enc = self.tokenizer([self.prompt + d for d in documents[start:start + batch_size]], padding=True,
                                     truncation=True, max_length=self.max_src, return_tensors="pt").to(self.device)
                ids = self.model.generate(**enc, max_new_tokens=self.max_tgt, num_beams=self.beams, early_stopping=True)
                outputs += self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        return [o.strip() for o in outputs]

    def fit(self, train: List[Dict[str, str]], dev: List[Dict[str, str]]) -> List[Dict[str, float]]:
        import torch
        tc = self.cfg["training"]
        bs, epochs = int(tc["batch_size"]), int(tc["epochs"])
        lr = float(tc["learning_rate_full" if self.method == "full" else "learning_rate_peft"])
        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model, lr, float(tc["weight_decay"]), -(-len(train) // bs) * epochs,
            float(tc["warmup_ratio"]), tc.get("scheduler", "linear"))
        stopper = EarlyStopping(patience=int(tc["early_stopping_patience"]))
        rng = np.random.default_rng(self.seed)
        best_state, history = None, []

        for epoch in range(epochs):
            self.model.train()
            losses = []
            order = rng.permutation(len(train))
            for start in range(0, len(train), bs):
                batch = [train[i] for i in order[start:start + bs]]
                enc = self.tokenizer([self.prompt + r["document"] for r in batch], padding=True, truncation=True,
                                     max_length=self.max_src, return_tensors="pt")
                labels = self.tokenizer(text_target=[r["summary"] for r in batch], padding=True, truncation=True,
                                        max_length=self.max_tgt, return_tensors="pt")["input_ids"]
                labels[labels == self.tokenizer.pad_token_id] = -100
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    loss = self.model(**enc, labels=labels.to(self.device)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                               float(tc["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())

            dev_rouge = summarise_scores(rouge_scores(self.generate([d["document"] for d in dev]),
                                                      [d["summary"] for d in dev]), rounds=1)["rougeL"]
            record = {"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 4), "dev_rougeL": dev_rouge}
            history.append(record)
            print(f"   epoch {epoch + 1}: train_loss={record['train_loss']}, dev_rougeL={dev_rouge}")
            if stopper.step(dev_rouge):
                best_state = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}
            if stopper.should_stop:
                print(f"   early stopping (no dev improvement for {stopper.patience} epochs)")
                break

        if best_state is not None:
            params = dict(self.model.named_parameters())
            with torch.no_grad():
                for n, v in best_state.items():
                    params[n].copy_(v.to(params[n].device))
        return history

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "configs", "summarisation.yaml"))
    parser.add_argument("--method", choices=["full", "lora", "rslora"], default=None)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    train_all = TextPreprocessor.load_jsonl(os.path.join(BASE_DIR, ds["train_path"]))
    test = TextPreprocessor.load_jsonl(os.path.join(BASE_DIR, ds["test_path"]))
    dev, train = train_all[:ds["dev_size"]], train_all[ds["dev_size"]:]
    refs = [r["summary"] for r in test]
    rounds = cfg["evaluation"]["bootstrap_iterations"]
    print(f"Summarisation: train={len(train)} dev={len(dev)} test={len(test)} passages")

    systems: Dict[str, Dict[str, Any]] = {}
    lead = [lead_sentence(r["document"]) for r in test]
    systems["lead-1 (first sentence)"] = {"predictions": lead, **summarise_scores(rouge_scores(lead, refs), rounds)}

    require_deep_learning()
    method = args.method or cfg["training"]["method"]
    summariser = Seq2SeqSummariser(cfg, method, args.device)
    base = cfg["model"]["base_model"]

    zero = summariser.generate([r["document"] for r in test])
    systems[f"{base} zero-shot"] = {"predictions": zero, **summarise_scores(rouge_scores(zero, refs), rounds)}

    print(f"\n[{base} / {summariser.method}] trainable {summariser.param_counts['trainable']:,} of "
          f"{summariser.param_counts['total']:,}")
    start = time.perf_counter()
    history = summariser.fit(train, dev)
    seconds = time.perf_counter() - start
    tuned = summariser.generate([r["document"] for r in test])
    tuned_name = f"{base} + {summariser.method}"
    systems[tuned_name] = {"predictions": tuned, **summarise_scores(rouge_scores(tuned, refs), rounds)}
    summariser.save(os.path.join(BASE_DIR, "models", "finetuned", f"summariser_{summariser.method}"))

    for name, s in systems.items():
        print(f"   {name:35s} R1 {s['rouge1']:.3f}  R2 {s['rouge2']:.3f}  RL {s['rougeL']:.3f} {s['rougeL_95ci']}")

    lines = [
        "# Abstractive summarisation", "",
        f"- **Data**: {len(train)} / {len(dev)} / {len(test)} train / dev / test passage-summary pairs (split by passage)",
        f"- **Model**: `{base}` with `{summariser.method}` (trainable {summariser.param_counts['trainable']:,} of "
        f"{summariser.param_counts['total']:,} parameters), {len(history)} epochs, early stopping on dev ROUGE-L, {seconds:.1f} s",
        f"- **Decoding**: beam search ({cfg['model']['num_beams']} beams), max {cfg['model']['max_target_length']} new tokens",
        "- **Metric**: ROUGE F1 with Porter stemming; 95% bootstrap CI over test documents", "",
        "| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-L 95% CI | Mean words |", "|---|---|---|---|---|---|",
    ]
    for name, s in systems.items():
        mean_words = np.mean([len(p.split()) for p in s["predictions"]])
        lines.append(f"| {name} | {s['rouge1']:.3f} | {s['rouge2']:.3f} | {s['rougeL']:.3f} | "
                     f"[{s['rougeL_95ci'][0]:.3f}, {s['rougeL_95ci'][1]:.3f}] | {mean_words:.1f} |")
    lines.append(f"| reference summaries | - | - | - | - | {np.mean([len(r.split()) for r in refs]):.1f} |")
    lines += ["", "## Test examples", ""]
    for i, r in enumerate(test):
        lines += [f"**{r['id']}**", "", f"- Reference: {r['summary']}", f"- Zero-shot: {zero[i]}", f"- Fine-tuned: {tuned[i]}", ""]
    report = os.path.join(BASE_DIR, "results", "summarisation_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE_DIR, "results", "metrics", "summarisation.json"), "w", encoding="utf-8") as f:
        json.dump({"systems": systems, "history": history, "train_seconds": round(seconds, 1)}, f, indent=2)

    with ExperimentTracker("summarisation", tuned_name.replace("/", "_")) as tracker:
        tracker.log_params({"base_model": base, "method": summariser.method, "training": cfg["training"],
                            "lora": cfg.get("lora", {})})
        for h in history:
            tracker.log_metrics({k: v for k, v in h.items() if k != "epoch"}, step=h["epoch"])
        for tag, s in zip(("lead1", "zero_shot", "fine_tuned"), systems.values()):
            tracker.log_metrics({f"{tag}_{k}": v for k, v in s.items() if k != "predictions"})
        tracker.log_artifact(report)
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
