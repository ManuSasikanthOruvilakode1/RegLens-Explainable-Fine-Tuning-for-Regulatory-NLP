"""Multi-task learning with a shared transformer encoder and three task heads.

    Head A  regulatory classification   Linear(hidden, 3)   on the first (<s>/[CLS]) token
    Head B  entity recognition (BIO)    Linear(hidden, 9)   on every token
    Head C  extractive QA               Linear(hidden, 2)   start / end span logits

Batches are drawn round-robin (one batch per task in turn) so the smaller tasks are not swamped by
the larger ones. Each task loss is multiplied by its configured weight. Passing --tasks with a single
task trains the identical architecture on that task alone, which gives the single-task reference
used to measure positive or negative transfer.

    uv run python src/train_multitask.py                   # all three tasks
    uv run python src/train_multitask.py --compare         # multi-task vs each task alone
"""

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from typing import Any, Dict, List

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluate import Evaluator
from src.model_utils import EarlyStopping, get_device, load_config, require_deep_learning, set_seed
from src.preprocess import TextPreprocessor
from src.train_ner import align_labels_to_subwords, build_label_list, repair_bio, span_f1

ALL_TASKS = ("classification", "ner", "qa")


# ----------------------------------------------------------------------------- data

def load_squad(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]
    return [{"id": qa["id"], "question": qa["question"], "context": p["context"],
             "answer": qa["answers"][0]["text"], "answer_start": qa["answers"][0]["answer_start"]}
            for article in data for p in article["paragraphs"] for qa in p["qas"]]


def round_robin_schedule(batches_per_task: Dict[str, int]) -> List[str]:
    """One batch per task in turn, cycling until the largest task has been seen once.

    Smaller tasks are revisited (re-shuffled) rather than dropped, so every step trains every task
    equally often and no single task dominates the shared encoder updates.
    """
    n_rounds = max(batches_per_task.values())
    tasks = [t for t in batches_per_task if batches_per_task[t] > 0]
    return [t for _ in range(n_rounds) for t in tasks]


# ----------------------------------------------------------------------------- squad metrics

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(prediction: str, truth: str) -> float:
    pred, gold = normalize_answer(prediction).split(), normalize_answer(truth).split()
    common = collections.Counter(pred) & collections.Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def qa_exact(prediction: str, truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(truth))


# ----------------------------------------------------------------------------- model

class MultiTaskModel:
    """Shared encoder plus task-specific linear heads."""

    def __init__(self, encoder_name: str, n_classes: int, n_tags: int, device):
        import torch.nn as nn
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name).to(device)
        hidden = self.encoder.config.hidden_size
        dropout = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.heads = nn.ModuleDict({
            "classification": nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes)),
            "ner": nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_tags)),
            "qa": nn.Linear(hidden, 2),
        }).to(device)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.heads.parameters())

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.heads.train(mode)

    def forward(self, task: str, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if task == "classification":
            return self.heads["classification"](hidden[:, 0])
        if task == "ner":
            return self.heads["ner"](hidden)
        start, end = self.heads["qa"](hidden).split(1, dim=-1)
        return start.squeeze(-1), end.squeeze(-1)


class MultiTaskTrainer:
    def __init__(self, cfg: Dict[str, Any], tasks: List[str], device: str = "auto"):
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.seed = cfg.get("seed", 42)
        set_seed(self.seed)
        self.tasks = tasks
        self.weights = {t["name"]: float(t.get("weight", 1.0)) for t in cfg["tasks"]}
        self.device = get_device(device)
        encoder = cfg["model"]["shared_encoder"]
        self.tokenizer = AutoTokenizer.from_pretrained(encoder, add_prefix_space=True)
        self.max_len = cfg["model"]["max_seq_length"]
        self.qa_max_len = cfg["model"]["qa_max_seq_length"]
        self._load_data()
        self.model = MultiTaskModel(encoder, len(self.id2label), len(self.tag_list), self.device)

    def _load_data(self):
        cls_dir = os.path.join(BASE_DIR, "data", "processed", "classification")
        self.cls = {s: TextPreprocessor.load_jsonl(os.path.join(cls_dir, f"{s}.jsonl"))
                    for s in ("train", "dev", "test", "challenge")}
        labels = sorted({r["label"] for r in self.cls["train"]})
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        ner_dir = os.path.join(BASE_DIR, "data", "processed", "ner")
        self.ner = {s: TextPreprocessor.load_jsonl(os.path.join(ner_dir, f"{s}.jsonl")) for s in ("train", "dev", "test")}
        self.tag_list = build_label_list(["ORG", "REGULATORY_ACT", "GOV_BODY", "FINANCIAL_TERM"])
        self.tag2id = {t: i for i, t in enumerate(self.tag_list)}

        qa_dir = os.path.join(BASE_DIR, "data", "processed", "qa")
        self.qa = {s: load_squad(os.path.join(qa_dir, f"{s}.json")) for s in ("train", "test")}

    # ---- batch builders
    def _to_device(self, enc):
        return {k: v.to(self.device) for k, v in enc.items()}

    def _cls_batch(self, records):
        import torch
        enc = self.tokenizer([r["text"] for r in records], padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        return self._to_device(enc), torch.tensor([self.label2id[r["label"]] for r in records], device=self.device)

    def _ner_batch(self, sents, with_labels=True):
        import torch
        enc = self.tokenizer([s["tokens"] for s in sents], is_split_into_words=True, padding=True,
                             truncation=True, max_length=self.max_len, return_tensors="pt")
        word_ids = [enc.word_ids(i) for i in range(len(sents))]
        labels = None
        if with_labels:
            labels = torch.tensor([align_labels_to_subwords(word_ids[i], [self.tag2id[t] for t in s["labels"]])
                                   for i, s in enumerate(sents)], device=self.device)
        return self._to_device(enc), labels, word_ids

    def _qa_batch(self, items):
        import torch
        enc = self.tokenizer([q["question"] for q in items], [q["context"] for q in items], padding=True,
                             truncation="only_second", max_length=self.qa_max_len,
                             return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        starts, ends = [], []
        for i, q in enumerate(items):
            seq_ids = enc.sequence_ids(i)
            a_start, a_end = q["answer_start"], q["answer_start"] + len(q["answer"])
            s_tok = e_tok = 0
            for t, (o_s, o_e) in enumerate(offsets[i].tolist()):
                if seq_ids[t] != 1:
                    continue
                if o_s <= a_start < o_e:
                    s_tok = t
                if o_s < a_end <= o_e:
                    e_tok = t
            starts.append(s_tok)
            ends.append(e_tok)
        return self._to_device(enc), torch.tensor(starts, device=self.device), torch.tensor(ends, device=self.device), offsets, enc

    # ---- training
    def _task_loss(self, task, items):
        import torch.nn.functional as F
        if task == "classification":
            enc, labels = self._cls_batch(items)
            return F.cross_entropy(self.model.forward(task, enc["input_ids"], enc["attention_mask"]), labels)
        if task == "ner":
            enc, labels, _ = self._ner_batch(items)
            logits = self.model.forward(task, enc["input_ids"], enc["attention_mask"])
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        enc, s, e, _, _ = self._qa_batch(items)
        s_logits, e_logits = self.model.forward(task, enc["input_ids"], enc["attention_mask"])
        return (F.cross_entropy(s_logits, s) + F.cross_entropy(e_logits, e)) / 2

    def fit(self):
        import torch
        from transformers import get_scheduler

        tc = self.cfg["training"]
        bs, epochs = int(tc["batch_size"]), int(tc["epochs"])
        pools = {"classification": self.cls["train"], "ner": self.ner["train"], "qa": self.qa["train"]}
        pools = {t: pools[t] for t in self.tasks}
        per_task = {t: -(-len(p) // bs) for t, p in pools.items()}
        schedule = round_robin_schedule(per_task)
        total_steps = len(schedule) * epochs

        optimizer = torch.optim.AdamW([
            {"params": self.model.encoder.parameters(), "lr": float(tc["learning_rate"])},
            {"params": self.model.heads.parameters(), "lr": float(tc["head_learning_rate"])},
        ], betas=(0.9, 0.999), eps=1e-8, weight_decay=float(tc["weight_decay"]))
        scheduler = get_scheduler(tc.get("scheduler", "linear"), optimizer,
                                  num_warmup_steps=int(total_steps * float(tc["warmup_ratio"])),
                                  num_training_steps=total_steps)
        stopper = EarlyStopping(patience=int(tc["early_stopping_patience"]))
        rng = np.random.default_rng(self.seed)
        cursors = {t: (rng.permutation(len(p)), 0) for t, p in pools.items()}
        best_state, history = None, []

        def next_batch(task):
            order, pos = cursors[task]
            if pos >= len(order):
                order, pos = rng.permutation(len(pools[task])), 0
            cursors[task] = (order, pos + bs)
            return [pools[task][i] for i in order[pos:pos + bs]]

        for epoch in range(epochs):
            self.model.train()
            losses = collections.defaultdict(list)
            for task in schedule:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    loss = self._task_loss(task, next_batch(task))
                (self.weights.get(task, 1.0) * loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(tc["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses[task].append(loss.item())

            dev = self.evaluate(split="dev")
            # QA has no dev split (contexts are scarce), so selection uses the tasks that do
            dev_score = float(np.mean([dev[t]["selection_score"] for t in dev])) if dev else float("nan")
            record = {"epoch": epoch + 1, **{f"{t}_loss": round(float(np.mean(v)), 4) for t, v in losses.items()},
                      "dev_score": round(dev_score, 4)}
            history.append(record)
            print(f"   epoch {epoch + 1}: " + ", ".join(f"{k}={v}" for k, v in record.items() if k != "epoch"))
            if not dev:
                continue
            if stopper.step(dev_score):
                best_state = {k: v.detach().cpu().clone() for k, v in
                              list(self.model.encoder.state_dict().items()) +
                              [("heads." + k, v) for k, v in self.model.heads.state_dict().items()]}
            if stopper.should_stop:
                print(f"   early stopping (no dev improvement for {stopper.patience} epochs)")
                break

        if best_state is not None:
            self.model.encoder.load_state_dict({k: v for k, v in best_state.items() if not k.startswith("heads.")})
            self.model.heads.load_state_dict({k[6:]: v for k, v in best_state.items() if k.startswith("heads.")})
        return history

    # ---- evaluation
    def evaluate(self, split: str = "test") -> Dict[str, Dict[str, Any]]:
        import torch
        self.model.train(False)
        out = {}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            if "classification" in self.tasks:
                for name in ([split, "challenge"] if split == "test" else [split]):
                    recs = self.cls[name]
                    enc, labels = self._cls_batch(recs)
                    pred = self.model.forward("classification", enc["input_ids"], enc["attention_mask"]).argmax(-1).tolist()
                    y = labels.tolist()
                    ev = Evaluator(bootstrap_rounds=500)
                    m = ev.compute_metrics(y, pred)
                    m.update(ev.compute_bootstrap_ci(y, pred))
                    m["accuracy_wilson_95ci"] = Evaluator.wilson_interval(sum(int(a == b) for a, b in zip(pred, y)), len(y))
                    m["selection_score"] = m["macro_f1"]
                    out["classification" if name == split else "classification_challenge"] = m
            if "ner" in self.tasks:
                sents = self.ner[split]
                enc, _, word_ids = self._ner_batch(sents, with_labels=False)
                pred_ids = self.model.forward("ner", enc["input_ids"], enc["attention_mask"]).argmax(-1).cpu().numpy()
                preds = []
                for i, s in enumerate(sents):
                    tags, seen = ["O"] * len(s["tokens"]), set()
                    for pos, wid in enumerate(word_ids[i]):
                        if wid is not None and wid not in seen:
                            tags[wid] = self.tag_list[int(pred_ids[i][pos])]
                            seen.add(wid)
                    preds.append(repair_bio(tags))
                scores = span_f1([s["labels"] for s in sents], preds)
                out["ner"] = {k: v for k, v in scores.items() if k != "per_type"}
                out["ner"]["selection_score"] = scores["f1"]
            if "qa" in self.tasks and split in self.qa:
                out["qa"] = self._evaluate_qa(self.qa[split])
        return out

    def _evaluate_qa(self, items, max_answer_tokens: int = 30):
        enc, _, _, offsets, raw = self._qa_batch(items)
        s_logits, e_logits = self.model.forward("qa", enc["input_ids"], enc["attention_mask"])
        s_logits, e_logits = s_logits.float().cpu().numpy(), e_logits.float().cpu().numpy()
        ems, f1s, examples = [], [], []
        for i, q in enumerate(items):
            ctx_tokens = [t for t, sid in enumerate(raw.sequence_ids(i)) if sid == 1]
            best, best_score = (ctx_tokens[0], ctx_tokens[0]), -np.inf
            for s in ctx_tokens:
                for e in ctx_tokens:
                    if s <= e < s + max_answer_tokens and s_logits[i][s] + e_logits[i][e] > best_score:
                        best_score, best = s_logits[i][s] + e_logits[i][e], (s, e)
            span = offsets[i].tolist()
            prediction = q["context"][span[best[0]][0]:span[best[1]][1]].strip()
            ems.append(qa_exact(prediction, q["answer"]))
            f1s.append(qa_f1(prediction, q["answer"]))
            examples.append({"question": q["question"], "gold": q["answer"], "predicted": prediction})
        rng = np.random.default_rng(42)
        boot = [np.mean(np.array(f1s)[rng.integers(0, len(f1s), len(f1s))]) for _ in range(500)]
        return {"exact_match": float(np.mean(ems)), "f1": float(np.mean(f1s)),
                "f1_95ci": (round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)),
                "n_questions": len(items), "examples": examples, "selection_score": float(np.mean(f1s))}


def train_once(cfg, tasks, device="auto"):
    print(f"\n[tasks: {', '.join(tasks)}]")
    trainer = MultiTaskTrainer(cfg, tasks, device)
    start = time.perf_counter()
    history = trainer.fit()
    seconds = time.perf_counter() - start
    result = trainer.evaluate("test")
    result["_meta"] = {"tasks": tasks, "epochs": len(history), "train_seconds": round(seconds, 1)}
    for task, m in result.items():
        if task == "_meta":
            continue
        key = "macro_f1" if task.startswith("classification") else "f1"
        print(f"   {task}: {key}={m[key]:.4f}" + (f"  EM={m['exact_match']:.4f}" if task == "qa" else ""))
    import torch
    del trainer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(BASE_DIR, "configs", "multitask.yaml"))
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=list(ALL_TASKS))
    parser.add_argument("--compare", action="store_true", help="also train each task alone for a transfer comparison")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 1234])
    args = parser.parse_args()
    require_deep_learning()
    cfg = load_config(args.config)

    configs = {"multitask": args.tasks}
    if args.compare:
        configs.update({f"single_{task}": [task] for task in args.tasks})
    runs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in configs}
    for seed in args.seeds:
        for name, tasks in configs.items():
            run_cfg = {**cfg, "seed": seed}
            result = train_once(run_cfg, tasks, args.device)
            result["_meta"]["seed"] = seed
            runs[name].append(result)

    os.makedirs(os.path.join(BASE_DIR, "results", "metrics"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "results", "metrics", "multitask.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    columns = [("classification", "macro_f1", "Classification macro-F1 (test)"),
               ("classification_challenge", "macro_f1", "Classification macro-F1 (challenge)"),
               ("ner", "f1", "NER strict span-F1"), ("qa", "exact_match", "QA exact match"), ("qa", "f1", "QA token F1")]

    def mean_std(seed_runs, task, key):
        values = [r[task][key] for r in seed_runs if task in r]
        return f"{np.mean(values):.3f} ± {np.std(values):.3f}" if values else "-"

    lines = [
        "# Multi-task shared encoder", "",
        f"Shared encoder `{cfg['model']['shared_encoder']}` (full fine-tuning, encoder lr {cfg['training']['learning_rate']}, "
        f"head lr {cfg['training']['head_learning_rate']}), round-robin task sampling, task weights "
        + ", ".join(f"{t['name']}={t['weight']}" for t in cfg["tasks"]) + ".",
        "Early stopping uses the mean dev score of classification (macro-F1) and NER (span-F1); QA has no dev split.",
        f"Values are mean ± std over seeds {args.seeds}.", "",
        "| Run | " + " | ".join(c[2] for c in columns) + " | Epochs |",
        "|---|" + "---|" * (len(columns) + 1),
    ]
    for name, seed_runs in runs.items():
        lines.append(f"| {name} | " + " | ".join(mean_std(seed_runs, t, k) for t, k, _ in columns) +
                     f" | {np.mean([r['_meta']['epochs'] for r in seed_runs]):.1f} |")
    lines += ["", "## Per-seed results", "", "| Run | Seed | " + " | ".join(c[2] for c in columns) + " |",
              "|---|---|" + "---|" * len(columns)]
    for name, seed_runs in runs.items():
        for r in seed_runs:
            lines.append(f"| {name} | {r['_meta']['seed']} | " +
                         " | ".join(f"{r[t][k]:.3f}" if t in r else "-" for t, k, _ in columns) + " |")
    first = runs["multitask"][0]
    if "qa" in first:
        lines += ["", f"## QA predictions (multi-task model, seed {first['_meta']['seed']}, test contexts never seen in training)", "",
                  "| Question | Gold | Predicted |", "|---|---|---|"]
        for ex in first["qa"]["examples"]:
            lines.append(f"| {ex['question']} | {ex['gold']} | {ex['predicted']} |")
    with open(os.path.join(BASE_DIR, "results", "multitask_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    from src.tracking import ExperimentTracker
    for name, seed_runs in runs.items():
        for run in seed_runs:
            with ExperimentTracker("multitask", f"{name}_s{run['_meta']['seed']}") as tracker:
                tracker.log_params({"tasks": ",".join(run["_meta"]["tasks"]), "seed": run["_meta"]["seed"],
                                    "encoder": cfg["model"]["shared_encoder"], "training": cfg["training"]})
                tracker.log_metrics({task: {k: v for k, v in m.items() if k != "examples"}
                                     for task, m in run.items() if task != "_meta"})
                tracker.log_metrics({"epochs": run["_meta"]["epochs"], "train_seconds": run["_meta"]["train_seconds"]})
                tracker.log_artifact(os.path.join(BASE_DIR, "results", "multitask_report.md"))
    print("Saved results/multitask_report.md")


if __name__ == "__main__":
    main()
