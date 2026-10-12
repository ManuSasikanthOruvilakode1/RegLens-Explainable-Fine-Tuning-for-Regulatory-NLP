"""Build processed train/dev/test splits for every task from the raw source files.

Splits are deterministic (seed 42) and leakage-free:
  - classification: stratified 60/15/25 split by label
  - ner: random 65/15/20 split by sentence
  - qa: split by context paragraph so no test passage is seen during training
  - dapt: unlabelled corpus built only from raw prose plus *training* texts
"""

import csv
import json
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import TextPreprocessor

SEED = 42
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def build_classification():
    with open(os.path.join(RAW_DIR, "classification_corpus.tsv"), encoding="utf-8") as f:
        records = [{"text": row["text"], "label": row["label"]} for row in csv.DictReader(f, delimiter="\t")]

    texts = [r["text"] for r in records]
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate texts found in classification corpus")

    train_dev, test = TextPreprocessor.stratified_split(records, "label", test_fraction=0.25, seed=SEED)
    train, dev = TextPreprocessor.stratified_split(train_dev, "label", test_fraction=0.2, seed=SEED)

    # Out-of-distribution challenge set: plain-English questions with no statutory vocabulary
    with open(os.path.join(RAW_DIR, "classification_challenge.tsv"), encoding="utf-8") as f:
        challenge = [{"text": row["text"], "label": row["label"]} for row in csv.DictReader(f, delimiter="\t")]

    out_dir = os.path.join(PROCESSED_DIR, "classification")
    for name, split in [("train", train), ("dev", dev), ("test", test), ("challenge", challenge)]:
        TextPreprocessor.save_jsonl(split, os.path.join(out_dir, f"{name}.jsonl"))
    print(f"classification: train={len(train)} dev={len(dev)} test={len(test)} challenge={len(challenge)}")
    return train


def build_ner():
    sentences = []
    with open(os.path.join(RAW_DIR, "ner_annotated.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens, labels = TextPreprocessor.parse_entity_markup(line)
            sentences.append({"tokens": tokens, "labels": labels})

    rng = random.Random(SEED)
    rng.shuffle(sentences)
    n_test = round(len(sentences) * 0.20)
    n_dev = round(len(sentences) * 0.15)
    test, dev, train = sentences[:n_test], sentences[n_test:n_test + n_dev], sentences[n_test + n_dev:]

    out_dir = os.path.join(PROCESSED_DIR, "ner")
    for name, split in [("train", train), ("dev", dev), ("test", test)]:
        TextPreprocessor.save_jsonl(split, os.path.join(out_dir, f"{name}.jsonl"))
    print(f"ner: train={len(train)} dev={len(dev)} test={len(test)}")


def build_qa():
    with open(os.path.join(RAW_DIR, "qa_source.json"), encoding="utf-8") as f:
        articles = json.load(f)

    rng = random.Random(SEED)
    rng.shuffle(articles)
    n_test = round(len(articles) * 0.3)
    splits = {"test": articles[:n_test], "train": articles[n_test:]}

    out_dir = os.path.join(PROCESSED_DIR, "qa")
    os.makedirs(out_dir, exist_ok=True)
    for name, split_articles in splits.items():
        data = []
        for a_idx, article in enumerate(split_articles):
            context = article["context"]
            qas = []
            for q_idx, qa in enumerate(article["qas"]):
                start = context.find(qa["answer"])
                if start < 0:
                    raise ValueError(f"Answer not found in context: {qa['answer']!r} ({article['title']})")
                qas.append({
                    "id": f"{name}-{a_idx}-{q_idx}",
                    "question": qa["question"],
                    "answers": [{"text": qa["answer"], "answer_start": start}],
                })
            data.append({"title": article["title"], "paragraphs": [{"context": context, "qas": qas}]})
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "data": data}, f, indent=2, ensure_ascii=False)
        n_q = sum(len(p["qas"]) for d in data for p in d["paragraphs"])
        print(f"qa {name}: contexts={len(data)} questions={n_q}")
    return splits["train"]


def build_dapt_corpus(classification_train, qa_train_articles):
    with open(os.path.join(RAW_DIR, "regulatory_corpus.txt"), encoding="utf-8") as f:
        paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]
    paragraphs += [a["context"] for a in qa_train_articles]
    paragraphs += [r["text"] for r in classification_train]

    out_dir = os.path.join(PROCESSED_DIR, "dapt")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(paragraphs) + "\n")
    print(f"dapt: {len(paragraphs)} passages, {sum(len(p.split()) for p in paragraphs)} words")


def build_summarisation():
    with open(os.path.join(RAW_DIR, "regulatory_corpus.txt"), encoding="utf-8") as f:
        passages = {f"corpus-{i}": p.strip() for i, p in enumerate(p for p in f.read().split("\n\n") if p.strip())}
    with open(os.path.join(RAW_DIR, "qa_source.json"), encoding="utf-8") as f:
        passages.update({f"qa:{a['title']}": a["context"] for a in json.load(f)})
    with open(os.path.join(RAW_DIR, "summarisation_references.tsv"), encoding="utf-8") as f:
        pairs = [{"id": row["source"], "document": passages[row["source"]], "summary": row["summary"]}
                 for row in csv.DictReader(f, delimiter="\t")]

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n_test = round(len(pairs) * 0.3)
    out_dir = os.path.join(PROCESSED_DIR, "summarisation")
    TextPreprocessor.save_jsonl(pairs[n_test:], os.path.join(out_dir, "train.jsonl"))
    TextPreprocessor.save_jsonl(pairs[:n_test], os.path.join(out_dir, "test.jsonl"))
    print(f"summarisation: train={len(pairs) - n_test} test={n_test} document-summary pairs")


def main():
    classification_train = build_classification()
    build_ner()
    qa_train = build_qa()
    build_dapt_corpus(classification_train, qa_train)
    build_summarisation()


if __name__ == "__main__":
    main()
