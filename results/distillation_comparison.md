# Knowledge distillation: teacher vs student

Teacher `roberta-base` (full fine-tuning) -> student `distilbert-base-uncased`. Loss = 0.7 * T^2 * KL + 0.3 * CE with T = 4.0. Teacher soft targets are computed on the training split only; dev is used for early stopping.

| Model | Params | Test acc % (Wilson 95% CI) | Test macro-F1 (bootstrap CI) | Challenge acc % (Wilson 95% CI) | Challenge macro-F1 | Challenge log-loss | Latency ms/sample |
|---|---|---|---|---|---|---|---|
| teacher | 124.6M | 95.2 [84.2, 98.7] | 0.953 [0.879, 1.000] | 86.7 [70.3, 94.7] | 0.868 | 0.8014 | 0.8 |
| student_hard_labels | 67.0M | 100.0 [91.6, 100.0] | 1.000 [1.000, 1.000] | 83.3 [66.4, 92.7] | 0.837 | 0.4136 | 0.451 |
| student_distilled | 67.0M | 100.0 [91.6, 100.0] | 1.000 [1.000, 1.000] | 80.0 [62.7, 90.5] | 0.803 | 0.6561 | 0.4 |
