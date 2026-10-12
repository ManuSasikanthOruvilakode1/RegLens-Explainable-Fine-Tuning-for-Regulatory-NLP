"""Domain-adaptive pre-training (DAPT): continued masked-language-model training on unlabelled
Australian regulatory text before task fine-tuning.

Dynamic masking re-samples the mask every time a batch is built: 15% of non-special tokens are
selected, of which 80% become [MASK], 10% a random token and 10% stay unchanged.

    uv run python src/domain_adapt.py
    uv run python src/train_transformer.py --base-model models/domain_adapted/bert-base-uncased-au --run-name dapt_bert_lora
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.tracking import ExperimentTracker
from src.model_utils import build_optimizer_and_scheduler, get_device, require_deep_learning, set_seed

IGNORE_INDEX = -100


def dynamic_mask(input_ids, special_tokens_mask, mask_token_id: int, vocab_size: int,
                 mlm_probability: float = 0.15, generator=None) -> Tuple["object", "object"]:
    """Applies BERT-style 80/10/10 masking. Returns (masked_input_ids, labels).

    labels hold the original id at selected positions and -100 elsewhere, so the loss is only
    computed on the tokens the model has to reconstruct.
    """
    import torch

    labels = input_ids.clone()
    inputs = input_ids.clone()
    prob = torch.full(labels.shape, mlm_probability)
    prob.masked_fill_(special_tokens_mask.bool(), 0.0)
    selected = torch.bernoulli(prob, generator=generator).bool()
    labels[~selected] = IGNORE_INDEX

    replace_with_mask = torch.bernoulli(torch.full(labels.shape, 0.8), generator=generator).bool() & selected
    inputs[replace_with_mask] = mask_token_id

    # half of the remaining 20% -> random token, the other half left unchanged
    replace_random = torch.bernoulli(torch.full(labels.shape, 0.5), generator=generator).bool() & selected & ~replace_with_mask
    random_tokens = torch.randint(vocab_size, labels.shape, dtype=input_ids.dtype, generator=generator)
    inputs[replace_random] = random_tokens[replace_random]
    return inputs, labels


class DomainAdaptivePretrainer:
    """Continues MLM pre-training of an encoder on an in-domain corpus."""

    def __init__(self, base_model: str = "bert-base-uncased", max_seq_length: int = 128,
                 mlm_probability: float = 0.15, device: str = "auto", seed: int = 42):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        set_seed(seed)
        self.seed = seed
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.device = get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForMaskedLM.from_pretrained(base_model).to(self.device)

    def _encode(self, passages: List[str]) -> Dict[str, "object"]:
        return self.tokenizer(passages, padding=True, truncation=True, max_length=self.max_seq_length,
                              return_special_tokens_mask=True, return_tensors="pt")

    def _masked_batch(self, passages: List[str], generator) -> Dict[str, "object"]:
        enc = self._encode(passages)
        special = enc.pop("special_tokens_mask") | (enc["attention_mask"] == 0).long()
        inputs, labels = dynamic_mask(enc["input_ids"], special, self.tokenizer.mask_token_id,
                                      len(self.tokenizer), self.mlm_probability, generator)
        enc["input_ids"], enc["labels"] = inputs, labels
        return {k: v.to(self.device) for k, v in enc.items()}

    def pseudo_perplexity(self, passages: List[str], repeats: int = 5, batch_size: int = 16) -> float:
        """Masked-token perplexity averaged over several fixed-seed masking draws (lower is better)."""
        import torch

        self.model.eval()
        generator = torch.Generator().manual_seed(1234)
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for _ in range(repeats):
                for start in range(0, len(passages), batch_size):
                    batch = self._masked_batch(passages[start:start + batch_size], generator)
                    n_tokens = int((batch["labels"] != IGNORE_INDEX).sum())
                    if n_tokens == 0:
                        continue
                    total_loss += self.model(**batch).loss.item() * n_tokens
                    total_tokens += n_tokens
        return math.exp(total_loss / max(total_tokens, 1))

    def train(self, passages: List[str], epochs: int = 10, batch_size: int = 16,
              learning_rate: float = 5e-5, weight_decay: float = 0.01, warmup_ratio: float = 0.1,
              max_grad_norm: float = 1.0) -> List[Dict[str, float]]:
        import torch

        steps_per_epoch = -(-len(passages) // batch_size)
        optimizer, scheduler = build_optimizer_and_scheduler(
            self.model, learning_rate, weight_decay, steps_per_epoch * epochs, warmup_ratio, "linear")
        generator = torch.Generator().manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        history = []
        for epoch in range(epochs):
            self.model.train()
            order = rng.permutation(len(passages))
            losses = []
            for start in range(0, len(order), batch_size):
                batch = self._masked_batch([passages[i] for i in order[start:start + batch_size]], generator)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    loss = self.model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
            history.append({"epoch": epoch + 1, "mlm_loss": round(float(np.mean(losses)), 4)})
            print(f"   epoch {epoch + 1}: mlm_loss={history[-1]['mlm_loss']}")
        return history

    def save(self, output_dir: str) -> str:
        """Saves the adapted encoder, carrying over the base model's pretrained pooler.

        Masked-LM checkpoints have no pooler layer, so a sequence classifier loaded from them would get
        a freshly random pooler on every load and not be reproducible from disk.
        """
        from safetensors.torch import load_file, save_file
        from transformers import AutoModel

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        pooler = getattr(AutoModel.from_pretrained(self.base_model), "pooler", None)
        weights = os.path.join(output_dir, "model.safetensors")
        if pooler is not None and os.path.exists(weights):
            state = load_file(weights)
            prefix = self.model.base_model_prefix
            for name, tensor in pooler.state_dict().items():
                state[f"{prefix}.pooler.{name}"] = tensor.detach().cpu().contiguous()
            save_file(state, weights, metadata={"format": "pt"})
        return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--corpus", default=os.path.join(BASE_DIR, "data", "processed", "dapt", "corpus.txt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    args = parser.parse_args()
    require_deep_learning()

    with open(args.corpus, encoding="utf-8") as f:
        passages = [line.strip() for line in f if line.strip()]
    rng = np.random.default_rng(42)
    order = rng.permutation(len(passages))
    n_held = max(1, len(passages) // 10)
    held_out = [passages[i] for i in order[:n_held]]
    train = [passages[i] for i in order[n_held:]]
    print(f"DAPT corpus: {len(train)} training passages, {len(held_out)} held-out passages")

    adapter = DomainAdaptivePretrainer(args.base_model, mlm_probability=args.mlm_probability, device=args.device)
    ppl_before = adapter.pseudo_perplexity(held_out)
    print(f"   held-out masked-token perplexity before: {ppl_before:.2f}")
    start = time.perf_counter()
    history = adapter.train(train, epochs=args.epochs, learning_rate=args.learning_rate)
    seconds = time.perf_counter() - start
    ppl_after = adapter.pseudo_perplexity(held_out)
    print(f"   held-out masked-token perplexity after:  {ppl_after:.2f}")

    name = os.path.basename(args.base_model.rstrip("/")) + "-au"
    output_dir = args.output_dir or os.path.join(BASE_DIR, "models", "domain_adapted", name)
    adapter.save(output_dir)

    summary = {"base_model": args.base_model, "train_passages": len(train), "held_out_passages": len(held_out),
               "train_words": sum(len(p.split()) for p in train), "epochs": args.epochs,
               "learning_rate": args.learning_rate, "mlm_probability": args.mlm_probability,
               "perplexity_before": round(ppl_before, 3), "perplexity_after": round(ppl_after, 3),
               "train_seconds": round(seconds, 1), "history": history, "output_dir": os.path.relpath(output_dir, BASE_DIR)}
    os.makedirs(os.path.join(BASE_DIR, "results", "metrics"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "results", "metrics", "dapt.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with ExperimentTracker("domain_adaptation", name) as tracker:
        tracker.log_params({k: v for k, v in summary.items() if k not in ("history", "perplexity_before", "perplexity_after")})
        for h in history:
            tracker.log_metrics({"mlm_loss": h["mlm_loss"]}, step=h["epoch"])
        tracker.log_metrics({"perplexity_before": ppl_before, "perplexity_after": ppl_after})
    print(f"Saved domain-adapted model to {output_dir}")


if __name__ == "__main__":
    main()
