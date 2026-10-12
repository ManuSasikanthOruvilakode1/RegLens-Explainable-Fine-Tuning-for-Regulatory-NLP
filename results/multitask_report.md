# Multi-task shared encoder

Shared encoder `roberta-base` (full fine-tuning, encoder lr 3e-05, head lr 0.001), round-robin task sampling, task weights classification=1.0, ner=1.5, qa=1.0.
Early stopping uses the mean dev score of classification (macro-F1) and NER (span-F1); QA has no dev split.
Values are mean ± std over seeds [42, 7, 1234].

| Run | Classification macro-F1 (test) | Classification macro-F1 (challenge) | NER strict span-F1 | QA exact match | QA token F1 | Epochs |
|---|---|---|---|---|---|---|
| multitask | 0.984 ± 0.023 | 0.905 ± 0.064 | 0.784 ± 0.090 | 0.513 ± 0.131 | 0.675 ± 0.063 | 8.3 |
| single_classification | 0.953 ± 0.019 | 0.850 ± 0.062 | - | - | - | 9.3 |
| single_ner | - | - | 0.736 ± 0.035 | - | - | 16.0 |
| single_qa | - | - | - | 0.538 ± 0.063 | 0.729 ± 0.031 | 20.0 |

## Per-seed results

| Run | Seed | Classification macro-F1 (test) | Classification macro-F1 (challenge) | NER strict span-F1 | QA exact match | QA token F1 |
|---|---|---|---|---|---|---|
| multitask | 42 | 1.000 | 0.931 | 0.800 | 0.462 | 0.606 |
| multitask | 7 | 1.000 | 0.967 | 0.885 | 0.692 | 0.759 |
| multitask | 1234 | 0.952 | 0.817 | 0.667 | 0.385 | 0.660 |
| single_classification | 42 | 0.976 | 0.828 | - | - | - |
| single_classification | 7 | 0.952 | 0.789 | - | - | - |
| single_classification | 1234 | 0.930 | 0.935 | - | - | - |
| single_ner | 42 | - | - | 0.717 | - | - |
| single_ner | 7 | - | - | 0.784 | - | - |
| single_ner | 1234 | - | - | 0.706 | - | - |
| single_qa | 42 | - | - | - | 0.538 | 0.715 |
| single_qa | 7 | - | - | - | 0.615 | 0.772 |
| single_qa | 1234 | - | - | - | 0.462 | 0.700 |

## QA predictions (multi-task model, seed 42, test contexts never seen in training)

| Question | Gold | Predicted |
|---|---|---|
| Who pays fringe benefits tax? | employers | Fringe benefits tax is paid by employers |
| When does the FBT year begin? | 1 April | 31 March |
| What is the FBT rate? | 47 percent | 47 percent |
| What level of government levies payroll tax? | state and territory | Payroll tax |
| Which provisions combine the wages of related businesses? | grouping provisions | grouping provisions |
| Where do employers register for payroll tax? | the revenue office in each state where they pay wages | with the revenue office |
| Which listing rule sets out the continuous disclosure obligation? | Listing Rule 3.1 | 3.1 |
| Which section of the Corporations Act gives continuous disclosure statutory force? | section 674 | section 674 |
| What may ASIC issue for suspected breaches? | infringement notices | infringement notices |
| How long must an individual hold an asset to receive the CGT discount? | at least twelve months | one-third |
| What discount do complying superannuation funds receive? | one-third | one-third |
| Which entities are not entitled to the CGT discount? | Companies | Companies |
| Against what can capital losses be offset? | capital gains | against capital gains |
