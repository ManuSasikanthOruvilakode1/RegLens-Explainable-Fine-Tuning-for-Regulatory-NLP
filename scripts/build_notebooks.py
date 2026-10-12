"""Generate and execute the analysis notebooks from the saved results.

The notebooks read only `data/processed/` and `results/metrics/*.json`, so they run on CPU in seconds
without any model weights. Re-run after the pipelines to refresh every chart and table:

    uv sync --extra notebooks
    uv run python scripts/build_notebooks.py
"""

import os
import sys

import nbformat
from nbclient import NotebookClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(BASE_DIR, "notebooks")

SETUP = """\
import json, os, sys
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT)
from src.plotting import apply_style, hbar_groups, SERIES, MUTED, TEXT_SECONDARY
from src.preprocess import TextPreprocessor

apply_style()
pd.set_option("display.precision", 3)

def metrics(name):
    with open(os.path.join(ROOT, "results", "metrics", f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)

def split(task, name):
    return TextPreprocessor.load_jsonl(os.path.join(ROOT, "data", "processed", task, f"{name}.jsonl"))"""

NOTEBOOKS = {
    "01_data_analysis.ipynb": [
        ("md", "# 01 · Data analysis\n\nSize, balance and vocabulary of every dataset. The last section measures "
               "*why* the plain-English challenge set is harder than the in-distribution test set."),
        ("code", SETUP),
        ("md", "## Dataset sizes"),
        ("code", """\
cls = {s: split("classification", s) for s in ("train", "dev", "test", "challenge")}
ner = {s: split("ner", s) for s in ("train", "dev", "test")}
summ = {s: split("summarisation", s) for s in ("train", "test")}
qa = {}
for s in ("train", "test"):
    with open(os.path.join(ROOT, "data", "processed", "qa", f"{s}.json"), encoding="utf-8") as f:
        qa[s] = sum(len(p["qas"]) for d in json.load(f)["data"] for p in d["paragraphs"])
pd.DataFrame([
    {"task": "classification", **{k: len(v) for k, v in cls.items()}},
    {"task": "ner (sentences)", **{k: len(v) for k, v in ner.items()}},
    {"task": "qa (questions)", "train": qa["train"], "test": qa["test"]},
    {"task": "summarisation (pairs)", **{k: len(v) for k, v in summ.items()}},
]).set_index("task").fillna("-")"""),
        ("md", "## Classification: label balance per split"),
        ("code", """\
balance = pd.DataFrame({s: Counter(r["label"] for r in recs) for s, recs in cls.items()}).T
balance"""),
        ("md", "## Sentence length by class (training split)\n\nOne panel per class, shared axes, so lengths compare directly."),
        ("code", """\
labels = sorted(balance.columns)
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True, sharey=True)
bins = np.arange(5, 35, 2)
for ax, label in zip(axes, labels):
    lengths = [len(r["text"].split()) for r in cls["train"] if r["label"] == label]
    ax.hist(lengths, bins=bins, color=SERIES[0], edgecolor="white", linewidth=1)
    ax.set_title(f"{label} (median {np.median(lengths):.0f})", fontsize=10)
    ax.set_xlabel("words")
axes[0].set_ylabel("sentences")
fig.suptitle("Sentence length distribution, training split", x=0.01, ha="left", fontweight="semibold")
fig.tight_layout()"""),
        ("md", "## Why the challenge set is harder: vocabulary overlap with training\n\n"
               "Share of each sentence's content words that never occur in the training split."),
        ("code", """\
STOP = set("the a an of to and or in on for is are be by with as at that this it from must may can do i my we our".split())
def content(text):
    return [w.strip(".,?!'\\"").lower() for w in text.split() if w.strip(".,?!'\\"").lower() not in STOP]
train_vocab = {w for r in cls["train"] for w in content(r["text"])}
oov = {s: [np.mean([w not in train_vocab for w in content(r["text"])]) for r in cls[s]] for s in ("test", "challenge")}
table = pd.DataFrame({s: {"mean OOV share": np.mean(v), "median": np.median(v)} for s, v in oov.items()}).T
fig, ax = plt.subplots(figsize=(7, 2.2))
hbar_groups(ax, list(oov), {"mean share of unseen content words": [100 * np.mean(v) for v in oov.values()]},
            fmt="{:.0f}%", xlabel="% of content words not seen in training", xlim=(0, 100))
ax.set_title("Challenge sentences contain more vocabulary unseen in training")
fig.tight_layout()
table"""),
        ("md", "## NER entity inventory"),
        ("code", """\
ents = Counter(t[2:] for s in ner["train"] + ner["dev"] + ner["test"] for t in s["labels"] if t.startswith("B-"))
pd.DataFrame.from_dict(ents, orient="index", columns=["entities"]).sort_values("entities", ascending=False)"""),
    ],
    "02_classification_ablation.ipynb": [
        ("md", "# 02 · Fine-tuning method ablation\n\nFull fine-tuning vs LoRA / rsLoRA (r = 16, 64) vs 4-bit QLoRA on "
               "`bert-base-uncased`, three seeds each. Test = 42 in-distribution sentences; challenge = 30 plain-English questions."),
        ("code", SETUP),
        ("code", """\
abl = metrics("classification_ablation")
rows = []
for name, runs in abl["runs"].items():
    for r in runs:
        rows.append({"variant": name, "seed": r["run_name"].rsplit("_s", 1)[1],
                     "test_acc": 100 * r["test"]["accuracy"], "challenge_acc": 100 * r["challenge"]["accuracy"],
                     "challenge_f1": r["challenge"]["macro_f1"], "trainable_params": r["trainable_params"],
                     "peak_gpu_mib": r["peak_gpu_mib"], "epochs": r["epochs_run"]})
runs = pd.DataFrame(rows)
summary = runs.groupby("variant", sort=False).agg(
    test_mean=("test_acc", "mean"), test_std=("test_acc", "std"),
    challenge_mean=("challenge_acc", "mean"), challenge_std=("challenge_acc", "std"),
    trainable_params=("trainable_params", "first"), peak_gpu_mib=("peak_gpu_mib", "mean"))
summary"""),
        ("md", "## Accuracy by method (mean ± std over 3 seeds)\n\nThe grey reference line is the TF-IDF + logistic-regression baseline on the challenge set."),
        ("code", """\
fig, ax = plt.subplots(figsize=(8, 4.6))
base = 100 * abl["baseline"]["challenge"]["accuracy"]
ax.axvline(base, color=MUTED, linestyle="--", linewidth=1.2, label=f"TF-IDF baseline, challenge ({base:.1f}%)")
hbar_groups(ax, list(summary.index),
            {"test (in-distribution)": summary["test_mean"].tolist(), "challenge (plain English)": summary["challenge_mean"].tolist()},
            errors={"test (in-distribution)": summary["test_std"].tolist(), "challenge (plain English)": summary["challenge_std"].tolist()},
            xlabel="accuracy (%)", xlim=(0, 112))
ax.set_title("Every method saturates the test set; the challenge set separates them")
fig.tight_layout()"""),
        ("md", "## Seed-to-seed spread\n\nEach dot is one seed. The spread is wider than most gaps between methods, "
               "which is why no single-seed comparison is reported."),
        ("code", """\
order = list(summary.index)
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, v in enumerate(order):
    vals = runs.loc[runs.variant == v, "challenge_acc"].to_numpy()
    ax.scatter(vals, [i] * len(vals), s=60, color=SERIES[0], edgecolor="white", linewidth=1.5, zorder=3)
    ax.plot([vals.min(), vals.max()], [i, i], color=SERIES[0], alpha=0.35, linewidth=2)
ax.set_yticks(range(len(order)), order)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlim(40, 100)
ax.set_xlabel("challenge accuracy (%) per seed")
ax.set_title("Challenge accuracy for each seed")
fig.tight_layout()"""),
        ("md", "## Cost: trainable parameters and peak GPU memory"),
        ("code", """\
cost = summary[["trainable_params", "peak_gpu_mib"]].copy()
cost["trainable_%"] = 100 * cost["trainable_params"] / runs.loc[runs.variant == "full_ft", "trainable_params"].iloc[0]
fig, ax = plt.subplots(figsize=(8, 3.2))
hbar_groups(ax, list(cost.index), {"peak GPU memory (MiB)": cost["peak_gpu_mib"].tolist()}, fmt="{:.0f}",
            xlabel="MiB", xlim=(0, 2500))
ax.set_title("QLoRA trains in 313 MiB, about 7x less than full fine-tuning")
fig.tight_layout()
cost"""),
    ],
    "03_dapt_distillation_multitask.ipynb": [
        ("md", "# 03 · Domain adaptation, distillation and multi-task learning"),
        ("code", SETUP),
        ("md", "## Domain-adaptive pre-training (DAPT) and the pooler control\n\n"
               "Masked-LM continued pre-training lowers held-out perplexity, but the classifier result depends on a "
               "second factor: BERT's pooler layer. It is pretrained for next-sentence prediction and stays frozen under "
               "LoRA. A 2 x 2 design (encoder x pooler), three seeds per cell, separates the two effects."),
        ("code", """\
dapt = metrics("dapt")
abl = metrics("classification_ablation")
cells = {
    ("bert-base-uncased", "pretrained pooler"): abl["runs"]["lora_r16"],
    ("bert-base-uncased", "random pooler"): [metrics(f"classification_control_random_pooler_lora_s{s}") for s in (42, 7, 1234)],
    ("DAPT encoder", "pretrained pooler"): [metrics(n) for n in ("classification_dapt_bert-base-uncased_lora",
                                                                "classification_dapt_lora_s7", "classification_dapt_lora_s1234")],
    ("DAPT encoder", "random pooler"): [metrics(f"classification_dapt_random_pooler_s{s}") for s in (42, 7, 1234)],
}
grid = pd.DataFrame([{"encoder": e, "pooler": p,
                      "test acc %": np.mean([100 * r["test"]["accuracy"] for r in runs]),
                      "challenge acc %": np.mean([100 * r["challenge"]["accuracy"] for r in runs]),
                      "challenge std": np.std([100 * r["challenge"]["accuracy"] for r in runs])}
                     for (e, p), runs in cells.items()])
print(f"Held-out masked-token perplexity: {dapt['perplexity_before']:.2f} -> {dapt['perplexity_after']:.2f}")
grid"""),
        ("code", """\
encoders = ["bert-base-uncased", "DAPT encoder"]
pick = lambda p, col: [grid[(grid.encoder == e) & (grid.pooler == p)][col].item() for e in encoders]
fig, ax = plt.subplots(figsize=(8, 2.9))
hbar_groups(ax, encoders, {p: pick(p, "challenge acc %") for p in ("pretrained pooler", "random pooler")},
            errors={p: pick(p, "challenge std") for p in ("pretrained pooler", "random pooler")},
            xlabel="challenge accuracy (%)", xlim=(0, 105))
ax.set_title("The pooler, not domain adaptation, drives the challenge-set gain")
fig.tight_layout()"""),
        ("md", "## Knowledge distillation: roberta-base teacher -> distilbert student"),
        ("code", """\
dist = metrics("distillation")
dt = pd.DataFrame([{"model": r["model"], "params (M)": r["total_params"] / 1e6, "test acc %": 100 * r["test"]["accuracy"],
                    "challenge acc %": 100 * r["challenge"]["accuracy"],
                    "latency ms": r["test"]["latency_ms_per_sample"]} for r in dist]).set_index("model")
fig, axes = plt.subplots(1, 2, figsize=(10, 2.6), gridspec_kw={"width_ratios": [3, 2]})
hbar_groups(axes[0], list(dt.index), {"test": dt["test acc %"].tolist(), "challenge": dt["challenge acc %"].tolist()},
            xlabel="accuracy (%)", xlim=(0, 115))
axes[0].set_title("Accuracy")
hbar_groups(axes[1], list(dt.index), {"GPU latency": dt["latency ms"].tolist()}, fmt="{:.1f} ms", xlabel="ms per sample", xlim=(0, 8.5))
axes[1].set_yticklabels([])
axes[1].set_title("Inference latency")
fig.tight_layout()
dt"""),
        ("md", "The distilled student is no better than the student trained on hard labels; the gap is well inside the "
               "confidence intervals. With 102 training sentences the teacher's soft targets add little information."),
        ("md", "## Multi-task shared encoder vs single-task models"),
        ("code", """\
mt = metrics("multitask")
cols = {"classification macro-F1 (challenge)": ("classification_challenge", "macro_f1"),
        "NER strict span-F1": ("ner", "f1"), "QA token F1": ("qa", "f1")}
single = {"classification_challenge": "single_classification", "ner": "single_ner", "qa": "single_qa"}
def seed_values(run, task, key):
    return [r[task][key] for r in mt[run]]
mtab = pd.DataFrame({
    "multi-task": [np.mean(seed_values("multitask", t, k)) for t, k in cols.values()],
    "multi-task std": [np.std(seed_values("multitask", t, k)) for t, k in cols.values()],
    "single-task": [np.mean(seed_values(single[t], t, k)) for t, k in cols.values()],
    "single-task std": [np.std(seed_values(single[t], t, k)) for t, k in cols.values()],
}, index=list(cols))
fig, ax = plt.subplots(figsize=(8, 3.0))
hbar_groups(ax, list(mtab.index), {c: mtab[c].tolist() for c in ("multi-task", "single-task")},
            errors={c: mtab[f"{c} std"].tolist() for c in ("multi-task", "single-task")},
            fmt="{:.2f}", xlabel="score (mean ± std over 3 seeds)", xlim=(0, 1.1))
ax.set_title("Multi-task vs single-task, three seeds each")
fig.tight_layout()
mtab"""),
    ],
    "04_ner_qa_summarisation.ipynb": [
        ("md", "# 04 · NER, extractive QA and summarisation"),
        ("code", SETUP),
        ("md", "## NER: transformer vs gazetteer (strict span-F1, 95% bootstrap CI)"),
        ("code", """\
ner = metrics("ner")
nt = pd.DataFrame({k: {"precision": v["precision"], "recall": v["recall"], "f1": v["f1"],
                       "ci_low": v["f1_95ci"][0], "ci_high": v["f1_95ci"][1]} for k, v in ner.items()}).T
fig, ax = plt.subplots(figsize=(8, 2.2))
f1 = nt["f1"].tolist()
hbar_groups(ax, ["gazetteer", "roberta-base + LoRA"], {"span-F1": f1},
            errors={"span-F1": [[f - lo for f, lo in zip(f1, nt["ci_low"])], [hi - f for f, hi in zip(f1, nt["ci_high"])]]},
            fmt="{:.2f}", xlabel="strict span-F1", xlim=(0, 1.05))
ax.set_title("The transformer nearly doubles span-F1; the gazetteer is precise but misses most entities")
fig.tight_layout()
nt"""),
        ("md", "## Extractive QA examples (multi-task model, unseen contexts)"),
        ("code", """\
qa = metrics("multitask")["multitask"][0]["qa"]   # seed 42
ex = pd.DataFrame(qa["examples"])
ex["exact"] = [a.lower().strip(". ") == b.lower().strip(". ") for a, b in zip(ex.gold, ex.predicted)]
print(f"EM {qa['exact_match']:.3f}   token F1 {qa['f1']:.3f}  95% CI {qa['f1_95ci']}")
ex"""),
        ("md", "## Summarisation: ROUGE with 95% bootstrap CI"),
        ("code", """\
sm = metrics("summarisation")["systems"]
st = pd.DataFrame({k: {m: v[m] for m in ("rouge1", "rouge2", "rougeL")} | {"rougeL_low": v["rougeL_95ci"][0], "rougeL_high": v["rougeL_95ci"][1]}
                   for k, v in sm.items()}).T
fig, ax = plt.subplots(figsize=(8, 2.8))
hbar_groups(ax, list(st.index), {"ROUGE-1": st["rouge1"].tolist(), "ROUGE-2": st["rouge2"].tolist(), "ROUGE-L": st["rougeL"].tolist()},
            fmt="{:.2f}", xlabel="ROUGE F1", xlim=(0, 0.6))
ax.set_title("LoRA fine-tuning beats both the lead sentence and zero-shot generation")
fig.tight_layout()
st"""),
        ("code", """\
names = list(sm)
pd.DataFrame({"zero-shot": sm[names[1]]["predictions"], "fine-tuned": sm[names[2]]["predictions"]}).head(6)"""),
    ],
    "05_explainability_and_export.ipynb": [
        ("md", "# 05 · Explainability and deployment\n\nIntegrated Gradients token attributions for the LoRA classifier, "
               "then the ONNX / INT8 export benchmark."),
        ("code", SETUP),
        ("code", """\
expl = metrics("explanations")
print(f"{len(expl)} sentences explained; mean |convergence delta| {np.mean([e['convergence_delta'] for e in expl]):.3f}")
per_class = {}
for e in expl:
    for w, v in zip(e["words"], e["attributions"]):
        if w.isalpha() and len(w) > 2:
            per_class.setdefault(e["predicted"], {}).setdefault(w.lower(), []).append(v)"""),
        ("md", "## Words that push the model towards each class\n\nMean attribution across all test and challenge "
               "sentences predicted as that class (words seen at least twice)."),
        ("code", """\
classes = sorted(per_class)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, cls_name in zip(axes, classes):
    top = sorted(((w, np.mean(v)) for w, v in per_class[cls_name].items() if len(v) >= 2), key=lambda x: -x[1])[:8]
    hbar_groups(ax, [w for w, _ in top], {"attribution": [s for _, s in top]}, fmt="{:+.2f}",
                xlim=(0, max(s for _, s in top) * 1.35))
    ax.set_title(cls_name, fontsize=10)
fig.suptitle("Top attributed words per predicted class", x=0.01, ha="left", fontweight="semibold")
fig.tight_layout()"""),
        ("md", "## Misclassified sentences and their most influential words"),
        ("code", """\
wrong = [e for e in expl if e["gold"] != e["predicted"]]
pd.DataFrame([{"split": e["split"], "text": e["text"], "gold": e["gold"], "predicted": e["predicted"],
               "top words": ", ".join(w for w, _ in sorted(zip(e["words"], e["attributions"]), key=lambda x: -abs(x[1]))[:4])}
              for e in wrong])"""),
        ("md", "## ONNX Runtime export and dynamic INT8 quantisation (CPU, batch size 1)\n\n"
               "Size and latency use different units, so they are separate charts rather than one dual-axis plot."),
        ("code", """\
ex = metrics("export")
et = pd.DataFrame({k: {"size_mb": v["size_mb"], "accuracy %": 100 * v["accuracy"], "agreement %": 100 * v["agreement"],
                       "median ms": v["latency_ms_median"]} for k, v in ex.items()}).T
fig, axes = plt.subplots(1, 2, figsize=(10, 2.4))
hbar_groups(axes[0], list(et.index), {"median latency": et["median ms"].tolist()}, fmt="{:.1f} ms", xlabel="ms", xlim=(0, 28))
axes[0].set_title("Median latency")
hbar_groups(axes[1], list(et.index), {"size": et["size_mb"].tolist()}, fmt="{:.0f} MB", xlabel="MB", xlim=(0, 520))
axes[1].set_yticklabels([])
axes[1].set_title("Model size")
fig.tight_layout()
et"""),
    ],
}


def build(name: str, cells) -> str:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src) for kind, src in cells]
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": NB_DIR}}).execute()
    path = os.path.join(NB_DIR, name)
    nbformat.write(nb, path)
    return path


def main():
    os.makedirs(NB_DIR, exist_ok=True)
    only = sys.argv[1:]
    for name, cells in NOTEBOOKS.items():
        if only and not any(o in name for o in only):
            continue
        print(f"executed {build(name, cells)}")


if __name__ == "__main__":
    main()
