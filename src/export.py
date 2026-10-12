"""Exporting and persisting models for downstream serving.

`ModelExporter` pickles scikit-learn pipelines. Running this module exports a fine-tuned transformer
classifier (LoRA adapters merged into the base weights) to ONNX, applies dynamic INT8 weight
quantisation, and benchmarks PyTorch vs ONNX Runtime FP32 vs ONNX Runtime INT8 on CPU:

    uv run python src/export.py --model-dir models/finetuned/bert-base-uncased_lora
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class ModelExporter:
    """Serialises trained models and metadata."""

    @staticmethod
    def export(model: Any, export_path: str, metadata: Dict[str, Any]) -> str:
        """Saves model and metadata to specified path."""
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        payload = {
            "model": model,
            "metadata": metadata
        }
        with open(export_path, "wb") as f:
            pickle.dump(payload, f)
        return export_path

    @staticmethod
    def load(export_path: str) -> Dict[str, Any]:
        """Loads serialized model and metadata."""
        with open(export_path, "rb") as f:
            return pickle.load(f)


def dir_size_mb(paths: List[str]) -> float:
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p)) / 2 ** 20


def export_onnx(model, tokenizer, onnx_path: str, opset: int = 17) -> str:
    """Exports a sequence classifier with dynamic batch and sequence axes."""
    import torch

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model = model.cpu().eval()
    sample = tokenizer(["Directors must act with care and diligence."], return_tensors="pt")
    args = (sample["input_ids"], sample["attention_mask"])

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    with torch.no_grad():
        torch.onnx.export(
            LogitsOnly(model), args, onnx_path, input_names=["input_ids", "attention_mask"],
            output_names=["logits"], opset_version=opset, dynamo=False,
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
        )
    return onnx_path


def quantize_int8(onnx_path: str, int8_path: str) -> str:
    """Dynamic INT8 quantisation of MatMul / Gemm weights; activations are quantised at run time."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
    return int8_path


class OnnxClassifier:
    def __init__(self, onnx_path: str, tokenizer, threads: int = 4):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.tokenizer = tokenizer

    def logits(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="np")
        return self.session.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                             "attention_mask": enc["attention_mask"].astype(np.int64)})[0]


def benchmark(predict_logits, texts: List[str], labels: List[int], repeats: int = 3) -> Dict[str, Any]:
    """Accuracy plus single-sentence CPU latency (median over texts, best of `repeats` passes)."""
    logits = np.concatenate([predict_logits(texts[i:i + 16]) for i in range(0, len(texts), 16)])
    predict_logits(texts[:1])  # warm-up
    timings = []
    for text in texts:
        best = min(_timed(predict_logits, text) for _ in range(repeats))
        timings.append(best)
    preds = logits.argmax(-1)
    return {"logits": logits, "accuracy": float(np.mean(preds == np.array(labels))),
            "latency_ms_median": round(float(np.median(timings)) * 1000, 2),
            "latency_ms_p90": round(float(np.percentile(timings, 90)) * 1000, 2)}


def _timed(fn, text):
    start = time.perf_counter()
    fn([text])
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=os.path.join(BASE_DIR, "models", "finetuned", "bert-base-uncased_lora"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "models", "onnx"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    from src.model_utils import require_deep_learning
    require_deep_learning()
    import torch
    from src.explain import load_classifier
    from src.preprocess import TextPreprocessor
    from src.evaluate import Evaluator
    from src.tracking import ExperimentTracker

    torch.set_num_threads(args.threads)
    model, tokenizer = load_classifier(args.model_dir, torch.device("cpu"))
    label2id = {v: int(k) for k, v in model.config.id2label.items()}
    name = os.path.basename(args.model_dir.rstrip("/\\"))

    cls_dir = os.path.join(BASE_DIR, "data", "processed", "classification")
    records = [r for s in ("test", "challenge") for r in TextPreprocessor.load_jsonl(os.path.join(cls_dir, f"{s}.jsonl"))]
    texts, labels = [r["text"] for r in records], [label2id[r["label"]] for r in records]

    def torch_logits(batch):
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            return model(**enc).logits.numpy()

    # Benchmark PyTorch before exporting: tracing leaves state on the module that changes later outputs
    torch_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    results = {"PyTorch FP32": {**benchmark(torch_logits, texts, labels), "size_mb": round(torch_size, 1)}}

    fp32_path = export_onnx(model, tokenizer, os.path.join(args.output_dir, f"{name}.onnx"))
    int8_path = quantize_int8(fp32_path, os.path.join(args.output_dir, f"{name}.int8.onnx"))
    print(f"Exported {fp32_path} and {int8_path}")

    fp32_files = [fp32_path] + [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir)
                                if f.startswith(f"{name}.onnx") and f != f"{name}.onnx"]
    runs = {
        "ONNX Runtime FP32": (OnnxClassifier(fp32_path, tokenizer, args.threads).logits, dir_size_mb(fp32_files)),
        "ONNX Runtime INT8 (dynamic)": (OnnxClassifier(int8_path, tokenizer, args.threads).logits, dir_size_mb([int8_path])),
    }
    for label, (fn, size) in runs.items():
        results[label] = {**benchmark(fn, texts, labels), "size_mb": round(size, 1)}
        print(f"   {label:28s} acc {results[label]['accuracy'] * 100:.1f}%  median {results[label]['latency_ms_median']} ms  "
              f"size {results[label]['size_mb']} MB")

    reference = results["PyTorch FP32"]["logits"]
    n = len(texts)
    lines = [
        "# ONNX export and INT8 quantisation", "",
        f"- **Model**: `{os.path.relpath(args.model_dir, BASE_DIR)}` (LoRA adapters merged into the base weights)",
        f"- **Evaluation texts**: {n} (42 test + 30 challenge sentences)",
        f"- **Hardware**: CPU only, {args.threads} threads; latency is per single sentence (batch size 1), "
        "median and 90th percentile, best of 3 passes",
        "- Accuracy CI: Wilson 95%. Agreement: share of predictions identical to PyTorch FP32.", "",
        "| Runtime | Size (MB) | Accuracy % (Wilson CI) | Agreement with PyTorch | Max abs logit diff | Median latency (ms) | p90 latency (ms) | Speed-up |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base_latency = results["PyTorch FP32"]["latency_ms_median"]
    summary = {}
    for label, r in results.items():
        agree = float(np.mean(r["logits"].argmax(-1) == reference.argmax(-1)))
        diff = float(np.max(np.abs(r["logits"] - reference)))
        wil = Evaluator.wilson_interval(int(round(r["accuracy"] * n)), n)
        lines.append(f"| {label} | {r['size_mb']} | {r['accuracy'] * 100:.1f} [{wil[0] * 100:.1f}, {wil[1] * 100:.1f}] | "
                     f"{agree * 100:.1f}% | {diff:.4f} | {r['latency_ms_median']} | {r['latency_ms_p90']} | "
                     f"{base_latency / r['latency_ms_median']:.2f}x |")
        summary[label] = {k: v for k, v in r.items() if k != "logits"} | {"agreement": agree, "max_abs_logit_diff": diff}

    report = os.path.join(BASE_DIR, "results", "export_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(BASE_DIR, "results", "metrics", "export.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with ExperimentTracker("export", name) as tracker:
        tracker.log_params({"model_dir": os.path.relpath(args.model_dir, BASE_DIR), "threads": args.threads})
        for tag, (label, s) in zip(("torch_fp32", "onnx_fp32", "onnx_int8"), summary.items()):
            tracker.log_metrics({f"{tag}_{k}": v for k, v in s.items()})
        tracker.log_artifact(report)
    print(f"Saved {report}")


if __name__ == "__main__":
    main()
