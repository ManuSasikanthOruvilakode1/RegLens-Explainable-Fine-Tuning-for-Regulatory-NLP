# Named entity recognition (strict span-level, seqeval IOB2)

- **Data**: 47 / 11 / 15 train / dev / test sentences; 25 gold entities in test
- **Model**: `roberta-base`, method `lora`, trainable 596,745 of 124,658,706 parameters
- **Training**: 17 epochs (early stopping on dev span-F1), 5.7 s
- **CI**: 95% bootstrap over test sentences (500 resamples)

| System | Precision | Recall | F1 | F1 95% CI |
|---|---|---|---|---|
| Gazetteer (training spans, longest match) | 0.857 | 0.240 | 0.375 | [0.180, 0.500] |
| roberta-base + lora | 0.739 | 0.680 | 0.708 | [0.569, 0.842] |

## Per entity type (transformer)

| Type | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ORG | 0.50 | 1.00 | 0.67 | 1 |
| REGULATORY_ACT | 1.00 | 1.00 | 1.00 | 4 |
| GOV_BODY | 0.86 | 0.75 | 0.80 | 8 |
| FINANCIAL_TERM | 0.60 | 0.50 | 0.55 | 12 |

## Test predictions with errors

- A business activity statement must be lodged with the ATO each quarter .  
  differences: statement (I-FINANCIAL_TERM -> O)
- The Fair Work Ombudsman recovered unpaid penalty rates for hospitality workers .  
  differences: penalty (B-FINANCIAL_TERM -> O); rates (I-FINANCIAL_TERM -> O)
- Dividend imputation prevents company profits from being taxed twice .  
  differences: Dividend (B-FINANCIAL_TERM -> O); imputation (I-FINANCIAL_TERM -> O)
- Capital gains tax applies when a CGT event happens to an asset .  
  differences: CGT (O -> B-FINANCIAL_TERM)
- Division 293 tax applies to high-income superannuation contributions .  
  differences: Division (B-FINANCIAL_TERM -> O); 293 (I-FINANCIAL_TERM -> O); tax (I-FINANCIAL_TERM -> O); superannuation (O -> B-FINANCIAL_TERM)
- AUSTRAC monitors reporting entities under the Anti-Money Laundering and Counter-Terrorism Financing Act 2006 .  
  differences: AUSTRAC (B-GOV_BODY -> O)
- PAYG withholding amounts are reported through Single Touch Payroll .  
  differences: Single (B-FINANCIAL_TERM -> O); Touch (I-FINANCIAL_TERM -> B-FINANCIAL_TERM); Payroll (I-FINANCIAL_TERM -> B-GOV_BODY)
- Research and development tax incentive claims are jointly administered by AusIndustry and the ATO .  
  differences: AusIndustry (B-GOV_BODY -> B-ORG)
- BHP paid fully franked dividends to its shareholders .  
  differences: dividends (B-FINANCIAL_TERM -> O)
