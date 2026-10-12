"""Token attribution for the fine-tuned regulatory classifier with Captum Integrated Gradients.

Attributions are computed on the word-embedding layer against a [PAD] baseline (special tokens kept),
using 50 Riemann steps. Sub-word scores are summed back to whole words. The convergence delta
(sum of attributions minus the change in model output) is reported as a completeness check.

    uv run python src/explain.py --model-dir models/finetuned/bert-base-uncased_lora
"""

import argparse
import html
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.model_utils import get_device, require_deep_learning
from src.preprocess import TextPreprocessor


def load_classifier(model_dir: str, device):
    """Loads either a PEFT adapter directory or a fully fine-tuned checkpoint."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        from peft import PeftConfig, PeftModel
        with open(os.path.join(model_dir, "training_metadata.json"), encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
        base = AutoModelForSequenceClassification.from_pretrained(
            PeftConfig.from_pretrained(model_dir).base_model_name_or_path, num_labels=len(id2label),
            id2label=id2label, label2id={v: k for k, v in id2label.items()})
        model = PeftModel.from_pretrained(base, model_dir).merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model.to(device).eval(), tokenizer


def merge_subwords(tokens: List[str], scores: np.ndarray, word_ids: List[Any]):
    """Sums sub-word attributions into whole words, dropping special tokens."""
    words, values, last = [], [], None
    for tok, score, wid in zip(tokens, scores, word_ids):
        if wid is None:
            continue
        piece = tok[2:] if tok.startswith("##") else tok.lstrip("Ġ▁")
        if wid == last:
            words[-1] += piece
            values[-1] += float(score)
        else:
            words.append(piece)
            values.append(float(score))
        last = wid
    return words, values


class IntegratedGradientsExplainer:
    def __init__(self, model, tokenizer, device, n_steps: int = 50):
        from captum.attr import LayerIntegratedGradients
        self.model, self.tokenizer, self.device, self.n_steps = model, tokenizer, device, n_steps
        self.lig = LayerIntegratedGradients(self._forward, model.get_input_embeddings())

    def _forward(self, input_ids, attention_mask):
        import torch
        return torch.softmax(self.model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1)

    def explain(self, text: str, target: int = None) -> Dict[str, Any]:
        import torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        input_ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        special = torch.tensor(self.tokenizer.get_special_tokens_mask(
            enc["input_ids"][0].tolist(), already_has_special_tokens=True), device=self.device).bool()
        baseline = torch.where(special.unsqueeze(0), input_ids, torch.full_like(input_ids, self.tokenizer.pad_token_id))

        with torch.no_grad():
            probs = self._forward(input_ids, mask)[0].cpu().numpy()
        target = int(np.argmax(probs)) if target is None else target
        attributions, delta = self.lig.attribute(
            inputs=input_ids, baselines=baseline, additional_forward_args=(mask,), target=target,
            n_steps=self.n_steps, return_convergence_delta=True, internal_batch_size=25)
        scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        words, values = merge_subwords(tokens, scores, enc.word_ids(0))
        return {"text": text, "target": target, "probs": probs.tolist(), "words": words,
                "attributions": values, "convergence_delta": float(delta.abs().item())}


def html_heatmap(words: List[str], values: List[float]) -> str:
    peak = max((abs(v) for v in values), default=1.0) or 1.0
    spans = []
    for w, v in zip(words, values):
        alpha = min(abs(v) / peak, 1.0)
        rgb = "22, 163, 74" if v >= 0 else "220, 38, 38"
        spans.append(f'<span title="{v:+.3f}" style="background: rgba({rgb}, {alpha:.2f}); padding: 1px 3px; '
                     f'border-radius: 3px">{html.escape(w)}</span>')
    return " ".join(spans)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=os.path.join(BASE_DIR, "models", "finetuned", "bert-base-uncased_lora"))
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    require_deep_learning()

    if not os.path.isdir(args.model_dir):
        raise SystemExit(f"No model at {args.model_dir}. Train one first: uv run python src/train_transformer.py --method lora")

    device = get_device()
    model, tokenizer = load_classifier(args.model_dir, device)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    explainer = IntegratedGradientsExplainer(model, tokenizer, device, args.n_steps)

    cls_dir = os.path.join(BASE_DIR, "data", "processed", "classification")
    explained = []
    for split in ("test", "challenge"):
        for rec in TextPreprocessor.load_jsonl(os.path.join(cls_dir, f"{split}.jsonl")):
            e = explainer.explain(rec["text"])
            e.update({"split": split, "gold": rec["label"], "predicted": id2label[e["target"]]})
            explained.append(e)
    print(f"Explained {len(explained)} sentences; mean |convergence delta| = "
          f"{np.mean([e['convergence_delta'] for e in explained]):.4f}")

    # Global view: mean attribution of each word towards the predicted class, per class
    per_class = defaultdict(lambda: defaultdict(list))
    for e in explained:
        for w, v in zip(e["words"], e["attributions"]):
            if w.isalpha() and len(w) > 2:
                per_class[e["predicted"]][w.lower()].append(v)

    lines = ["# Integrated Gradients token attributions", "",
             f"- **Model**: `{os.path.relpath(args.model_dir, BASE_DIR)}`",
             f"- **Method**: Captum LayerIntegratedGradients on word embeddings, {args.n_steps} steps, [PAD] baseline",
             f"- **Completeness check**: mean |convergence delta| = {np.mean([e['convergence_delta'] for e in explained]):.4f} "
             f"(max {np.max([e['convergence_delta'] for e in explained]):.4f})",
             "- Positive scores push the model towards its predicted class; negative scores push away.", "",
             "## Most influential words per predicted class (mean attribution, words seen at least twice)", ""]
    for cls in sorted(per_class):
        ranked = sorted(((w, float(np.mean(v)), len(v)) for w, v in per_class[cls].items() if len(v) >= 2),
                        key=lambda x: -x[1])[:10]
        lines.append(f"- **{cls}**: " + ", ".join(f"{w} ({s:+.2f})" for w, s, _ in ranked))

    lines += ["", "## Misclassified and challenge examples", ""]
    shown = [e for e in explained if e["gold"] != e["predicted"]] + \
            [e for e in explained if e["split"] == "challenge" and e["gold"] == e["predicted"]][:4]
    for e in shown:
        top = sorted(zip(e["words"], e["attributions"]), key=lambda x: -abs(x[1]))[:args.top_k]
        status = "correct" if e["gold"] == e["predicted"] else "WRONG"
        lines.append(f"- [{e['split']}, {status}] *{e['text']}*  \n  gold `{e['gold']}`, predicted `{e['predicted']}` "
                     f"(p={max(e['probs']):.2f}); top words: " + ", ".join(f"{w} ({v:+.2f})" for w, v in top))

    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "results", "explanations.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    rows = "\n".join(
        f"<tr><td>{e['split']}</td><td>{e['gold']}</td><td>{e['predicted']}</td><td>{max(e['probs']):.2f}</td>"
        f"<td>{html_heatmap(e['words'], e['attributions'])}</td></tr>" for e in explained)
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>Token attributions</title><style>"
            "body{font-family:system-ui,sans-serif;margin:24px;line-height:1.9}table{border-collapse:collapse}"
            "td,th{border-bottom:1px solid #ddd;padding:6px 10px;vertical-align:top;text-align:left}</style></head><body>"
            "<h1>Integrated Gradients token attributions</h1><p>Green pushes towards the predicted class, red away. "
            "Hover a word for its score.</p><table><tr><th>Split</th><th>Gold</th><th>Predicted</th><th>p</th>"
            f"<th>Sentence</th></tr>{rows}</table></body></html>")
    with open(os.path.join(BASE_DIR, "results", "explanations.html"), "w", encoding="utf-8") as f:
        f.write(page)
    with open(os.path.join(BASE_DIR, "results", "metrics", "explanations.json"), "w", encoding="utf-8") as f:
        json.dump(explained, f, indent=2)
    print("Saved results/explanations.md and results/explanations.html")


if __name__ == "__main__":
    main()
