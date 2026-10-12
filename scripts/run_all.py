"""Run end-to-end NLP baseline training, evaluation, curriculum analysis, and error reports."""

import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocess import TextPreprocessor
from src.augment import TextAugmenter
from src.curriculum import CurriculumScheduler
from src.train import ClassificationTrainer
from src.evaluate import Evaluator
from src.error_analysis import ErrorAnalyzer
from src.active_learning import ActiveLearner
from src.export import ModelExporter


def main():
    print("=" * 70)
    print(" Australian NLP Fine-Tuning Suite - Baseline Execution")
    print("=" * 70)

    train_path = os.path.join(BASE_DIR, "data", "processed", "classification", "train.jsonl")
    test_path = os.path.join(BASE_DIR, "data", "processed", "classification", "test.jsonl")

    # 1. Load dataset
    print(f"\n1. Loading datasets from {train_path} and {test_path}...")
    train_records = TextPreprocessor.load_jsonl(train_path)
    test_records = TextPreprocessor.load_jsonl(test_path)
    print(f"   Loaded {len(train_records)} train records, {len(test_records)} test records.")

    all_labels = [r["label"] for r in train_records] + [r["label"] for r in test_records]
    _, l2i, i2l = TextPreprocessor.encode_labels(all_labels)
    print(f"   Detected {len(l2i)} target classes: {list(l2i.keys())}")

    # 2. Data Augmentation
    print("\n2. Applying domain data augmentation (EDA)...")
    augmenter = TextAugmenter(seed=42)
    augmented_train = augmenter.augment_dataset(train_records, multiplier=1)
    print(f"   Expanded training set from {len(train_records)} to {len(augmented_train)} records.")

    # 3. Curriculum Scheduling
    print("\n3. Computing sample difficulty scores and curriculum ordering...")
    scheduler = CurriculumScheduler()
    curriculum_train = scheduler.order_curriculum(augmented_train, schedule="easy_to_hard")
    print(f"   Easiest sample: '{curriculum_train[0]['text'][:45]}...' (Score: {curriculum_train[0]['difficulty_score']})")
    print(f"   Hardest sample: '{curriculum_train[-1]['text'][:45]}...' (Score: {curriculum_train[-1]['difficulty_score']})")

    # 4. Model Training
    print("\n4. Fitting baseline model...")
    X_train = [r["text"] for r in curriculum_train]
    y_train = [l2i[r["label"]] for r in curriculum_train]
    X_test = [r["text"] for r in test_records]
    y_test = [l2i[r["label"]] for r in test_records]

    trainer = ClassificationTrainer()
    trainer.train(X_train, y_train)
    print("   Model training complete.")

    # 5. Evaluation with Bootstrap Confidence Intervals
    print("\n5. Running systematic evaluation with 95% bootstrap confidence intervals...")
    y_pred = trainer.predict(X_test).tolist()
    probs = trainer.predict_proba(X_test)

    evaluator = Evaluator(bootstrap_rounds=500, seed=42)
    metrics = evaluator.compute_metrics(y_test, y_pred)
    cis = evaluator.compute_bootstrap_ci(y_test, y_pred)

    print(f"   Accuracy: {metrics['accuracy'] * 100:.1f}% (95% CI: [{cis['accuracy_95ci'][0]*100:.1f}%, {cis['accuracy_95ci'][1]*100:.1f}%])")
    print(f"   Macro-F1: {metrics['macro_f1']:.4f} (95% CI: [{cis['macro_f1_95ci'][0]:.4f}, {cis['macro_f1_95ci'][1]:.4f}])")

    # 6. Error Analysis
    print("\n6. Performing error analysis and confusion matrix calculation...")
    analyzer = ErrorAnalyzer(id2label=i2l)
    error_report = analyzer.analyze(X_test, y_test, y_pred, probs)
    print(f"   Total Errors: {error_report['total_errors']} / {len(X_test)} (Error rate: {error_report['error_rate']*100:.1f}%)")

    # 7. Active Learning Sample Selection
    print("\n7. Simulating active learning uncertainty sampling for next annotation batch...")
    candidate_unlabelled = [
        "The Australian Securities and Investments Commission imposed civil penalties for breach of director duties.",
        "Employees under the modern award are eligible for higher duties allowance when acting up.",
        "Fringe benefits tax returns must be lodged annually by employers providing taxable fringe benefits."
    ]
    cand_probs = trainer.predict_proba(candidate_unlabelled)
    selected_queries = ActiveLearner.select_uncertain_samples(candidate_unlabelled, cand_probs, budget=2)
    for i, s in enumerate(selected_queries, 1):
        print(f"   Selected #{i} (Entropy: {s['entropy']}): '{s['text'][:55]}...'")

    # 8. Save Model and Reports
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_file = os.path.join(results_dir, "classification_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Australian NLP Classification Baseline Report\n\n")
        f.write(f"- **Accuracy**: {metrics['accuracy'] * 100:.1f}%\n")
        f.write(f"- **Macro-F1**: {metrics['macro_f1']:.4f} (95% CI: [{cis['macro_f1_95ci'][0]:.4f}, {cis['macro_f1_95ci'][1]:.4f}])\n\n")
        f.write("## Per-Class Performance\n\n")
        for cls_name, cls_metrics in error_report["classification_report"].items():
            if isinstance(cls_metrics, dict):
                f.write(f"- **{cls_name}**: Precision={cls_metrics.get('precision', 0):.2f}, Recall={cls_metrics.get('recall', 0):.2f}, F1={cls_metrics.get('f1-score', 0):.2f}\n")

    model_dir = os.path.join(BASE_DIR, "models", "finetuned")
    os.makedirs(model_dir, exist_ok=True)
    export_path = os.path.join(model_dir, "baseline_classifier.pkl")
    ModelExporter.export(trainer.pipeline, export_path, {"id2label": i2l, "metrics": metrics})
    print(f"\nSaved model and artifacts to: {export_path}")
    print(f"Saved evaluation report to: {report_file}")
    print("\nNLP baseline execution complete!")


if __name__ == "__main__":
    main()
