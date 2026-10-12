"""Command-line demo of the trained models on new text (the web version is src/ui.py).

    uv run python scripts/demo.py
    uv run python scripts/demo.py "Can my boss make me work on Anzac Day?"
"""

import os
import sys

os.environ.setdefault("NLP_TRACKING", "0")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.inference import DomainClassifier, EntityTagger, OnnxDomainClassifier, Summariser
from src.ui import CLASSIFY_EXAMPLES, NER_EXAMPLES, SUMMARY_EXAMPLE


def heading(text):
    print(f"\n{'=' * 78}\n {text}\n{'=' * 78}")


def main():
    questions = sys.argv[1:] or CLASSIFY_EXAMPLES[:3]

    heading("1. Regulatory domain classification  (BERT + LoRA, Integrated Gradients)")
    classifier = DomainClassifier()
    for text in questions:
        probs = classifier.predict(text)
        label = max(probs, key=probs.get)
        top = sorted(classifier.explain(text), key=lambda x: -x[1])[:4]
        print(f"\n  \"{text}\"\n    -> {label} (p={probs[label]:.2f}); key words: " +
              ", ".join(f"{w} {v:+.2f}" for w, v in top))

    heading("2. Named entity recognition  (RoBERTa + LoRA)")
    tagger = EntityTagger()
    for sentence in NER_EXAMPLES:
        print(f"\n  \"{sentence}\"")
        for chunk, etype in tagger.spans(sentence):
            if etype:
                print(f"    [{etype:15s}] {chunk}")

    heading("3. Abstractive summarisation  (flan-t5-base + LoRA)")
    out = Summariser().summarise(SUMMARY_EXAMPLE)
    print(f"\n  zero-shot : {out['zero_shot']}\n  fine-tuned: {out['fine_tuned']}")

    heading("4. CPU deployment  (ONNX Runtime INT8)")
    fast = OnnxDomainClassifier()
    for text in questions:
        probs, ms = fast.predict(text)
        label = max(probs, key=probs.get)
        print(f"  {label:22s} p={probs[label]:.2f}  {ms:5.1f} ms  \"{text[:55]}\"")


if __name__ == "__main__":
    main()
