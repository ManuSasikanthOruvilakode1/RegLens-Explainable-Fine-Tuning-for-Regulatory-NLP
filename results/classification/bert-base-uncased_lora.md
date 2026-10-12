# Transformer classification - bert-base-uncased_lora

- **Base model**: bert-base-uncased
- **Method**: lora
- **Trainable parameters**: 592,131 of 110,076,678 (0.54%)
- **Train / dev / test sizes**: 102 / 24 / 42
- **Training time**: 8.8 s
- **Peak GPU memory**: 686.2 MiB
- **LoRA**: r=16, alpha=32, dropout=0.05, targets=['query', 'value']

## Held-out test metrics

- **Accuracy**: 100.0% (bootstrap 95% CI: [100.0%, 100.0%]; Wilson 95% CI: [91.6%, 100.0%])
- **Macro-F1**: 1.0000 (bootstrap 95% CI: [1.0000, 1.0000])
- **Macro precision / recall**: 1.0000 / 1.0000
- **Log-loss**: 0.0251
- **Inference latency**: 1.624 ms per sample

## Plain-English challenge set (out-of-distribution)

- **Accuracy**: 73.3% (bootstrap 95% CI: [56.7%, 90.0%]; Wilson 95% CI: [55.5%, 85.8%])
- **Macro-F1**: 0.7381 (bootstrap 95% CI: [0.5426, 0.8880])
- **Macro precision / recall**: 0.8519 / 0.7333
- **Log-loss**: 0.6485
- **Inference latency**: 0.837 ms per sample

## Per-class performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corporate_governance | 1.00 | 1.00 | 1.00 | 14 |
| taxation | 1.00 | 1.00 | 1.00 | 14 |
| workplace_relations | 1.00 | 1.00 | 1.00 | 14 |
| macro avg | 1.00 | 1.00 | 1.00 | 42 |
| weighted avg | 1.00 | 1.00 | 1.00 | 42 |

## Confusion matrix (rows = true, columns = predicted)

    [14, 0, 0]
    [0, 14, 0]
    [0, 0, 14]

## Training history

| Epoch | Train loss | Dev macro-F1 |
|---|---|---|
| 1 | 1.1247 | 0.1985 |
| 2 | 1.098 | 0.3659 |
| 3 | 1.0734 | 0.6852 |
| 4 | 0.9762 | 0.6674 |
| 5 | 0.7284 | 0.7775 |
| 6 | 0.4151 | 0.8721 |
| 7 | 0.1624 | 0.9582 |
| 8 | 0.0744 | 0.8737 |
| 9 | 0.0222 | 1.0 |
| 10 | 0.0117 | 0.9582 |
| 11 | 0.0059 | 0.9582 |
| 12 | 0.0051 | 0.9582 |
| 13 | 0.0042 | 0.9582 |

## Hardest misclassifications (challenge)

| Text | True | Predicted | p(pred) |
|---|---|---|---|
| An employee reported fraud by the chief executive, how are they protected? | corporate_governance | workplace_relations | 0.97 |
| What happens if I pay my quarterly instalment a month late? | taxation | workplace_relations | 0.95 |
| Can I claim the electricity I use while working from my kitchen table? | taxation | workplace_relations | 0.90 |
| My company car is mostly used on weekends, will my boss have to pay something extra? | taxation | workplace_relations | 0.87 |
| The business is running out of cash and can't pay suppliers, should the directors keep trading? | corporate_governance | workplace_relations | 0.77 |
