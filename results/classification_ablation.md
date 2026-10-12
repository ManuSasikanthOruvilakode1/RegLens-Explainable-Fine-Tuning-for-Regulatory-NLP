# Fine-tuning method ablation

Base model `bert-base-uncased`, seeds [42, 7, 1234], train/dev/test = 102/24/42, challenge = 30 plain-English questions.
Model selection (early stopping) uses the dev set only; test and challenge sets are never seen during training.
Values are mean ± std over seeds. Accuracy is in %.

| Variant | Trainable params | Test acc | Test log-loss | Challenge acc | Challenge macro-F1 | Challenge log-loss | Epochs | Train time (s) | Peak GPU (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg baseline | - | 81.0 | - | 66.7 | 0.661 | - | - | - | - |
| full_ft | 109,484,547 (100.00%) | 98.4 ± 2.2 | 0.167 ± 0.073 | 72.2 ± 18.1 | 0.693 ± 0.218 | 0.771 ± 0.127 | 9.3 | 74.0 | 2115 |
| lora_r16 | 592,131 (0.54%) | 97.6 ± 1.9 | 0.090 ± 0.059 | 63.3 ± 9.8 | 0.580 ± 0.144 | 1.078 ± 0.389 | 15.7 | 30.0 | 687 |
| rslora_r16 | 592,131 (0.54%) | 96.8 ± 3.0 | 0.117 ± 0.072 | 68.9 ± 18.5 | 0.649 ± 0.236 | 0.848 ± 0.405 | 10.7 | 7.8 | 687 |
| lora_r64 | 2,361,603 (2.11%) | 96.8 ± 3.0 | 0.124 ± 0.134 | 61.1 ± 13.4 | 0.542 ± 0.187 | 0.927 ± 0.219 | 10.7 | 8.3 | 711 |
| rslora_r64 | 2,361,603 (2.11%) | 97.6 ± 0.0 | 0.112 ± 0.045 | 74.4 ± 8.3 | 0.719 ± 0.122 | 0.759 ± 0.346 | 8.7 | 7.5 | 711 |
| qlora_r16 | 592,131 (0.88%) | 98.4 ± 1.1 | 0.080 ± 0.035 | 72.2 ± 4.2 | 0.664 ± 0.066 | 0.840 ± 0.099 | 13.3 | 13.4 | 313 |

## Per-seed 95% confidence intervals (seed 42)

| Variant | Test acc (Wilson CI) | Challenge acc (Wilson CI) | Challenge macro-F1 (bootstrap CI) |
|---|---|---|---|
| TF-IDF baseline | 81.0 [66.7, 90.0] | 66.7 [48.8, 80.8] | - |
| full_ft | 100.0 [91.6, 100.0] | 86.7 [70.3, 94.7] | 0.862 [0.701, 0.968] |
| lora_r16 | 100.0 [91.6, 100.0] | 73.3 [55.5, 85.8] | 0.738 [0.543, 0.888] |
| rslora_r16 | 100.0 [91.6, 100.0] | 86.7 [70.3, 94.7] | 0.870 [0.717, 0.973] |
| lora_r64 | 100.0 [91.6, 100.0] | 80.0 [62.7, 90.5] | 0.805 [0.616, 0.934] |
| rslora_r64 | 97.6 [87.7, 99.6] | 83.3 [66.4, 92.7] | 0.838 [0.692, 0.953] |
| qlora_r16 | 97.6 [87.7, 99.6] | 66.7 [48.8, 80.8] | 0.580 [0.428, 0.741] |
