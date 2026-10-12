# ONNX export and INT8 quantisation

- **Model**: `models\finetuned\bert-base-uncased_lora` (LoRA adapters merged into the base weights)
- **Evaluation texts**: 72 (42 test + 30 challenge sentences)
- **Hardware**: CPU only, 4 threads; latency is per single sentence (batch size 1), median and 90th percentile, best of 3 passes
- Accuracy CI: Wilson 95%. Agreement: share of predictions identical to PyTorch FP32.

| Runtime | Size (MB) | Accuracy % (Wilson CI) | Agreement with PyTorch | Max abs logit diff | Median latency (ms) | p90 latency (ms) | Speed-up |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 417.7 | 88.9 [79.6, 94.3] | 100.0% | 0.0000 | 21.42 | 23.37 | 1.00x |
| ONNX Runtime FP32 | 417.9 | 88.9 [79.6, 94.3] | 100.0% | 0.0000 | 16.83 | 17.89 | 1.27x |
| ONNX Runtime INT8 (dynamic) | 105.2 | 91.7 [83.0, 96.1] | 94.4% | 1.8445 | 7.47 | 8.49 | 2.87x |
