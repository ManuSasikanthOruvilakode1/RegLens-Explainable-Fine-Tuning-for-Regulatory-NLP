# Abstractive summarisation

- **Data**: 22 / 5 / 12 train / dev / test passage-summary pairs (split by passage)
- **Model**: `google/flan-t5-base` with `lora` (trainable 1,769,472 of 249,347,328 parameters), 15 epochs, early stopping on dev ROUGE-L, 42.2 s
- **Decoding**: beam search (4 beams), max 64 new tokens
- **Metric**: ROUGE F1 with Porter stemming; 95% bootstrap CI over test documents

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-L 95% CI | Mean words |
|---|---|---|---|---|---|
| lead-1 (first sentence) | 0.364 | 0.181 | 0.301 | [0.232, 0.374] | 23.8 |
| google/flan-t5-base zero-shot | 0.266 | 0.126 | 0.227 | [0.181, 0.278] | 10.1 |
| google/flan-t5-base + lora | 0.458 | 0.217 | 0.376 | [0.315, 0.428] | 29.0 |
| reference summaries | - | - | - | - | 20.0 |

## Test examples

**corpus-23**

- Reference: APRA supervises banks, insurers and most super funds through prudential standards on capital, liquidity, governance and risk.
- Zero-shot: The Australian Prudential Regulation Authority (APRA) regulates deposit-taking institutions, general insurers, life insurers, private health insurers and most superannuation funds.
- Fine-tuned: The Australian Prudential Regulation Authority supervises authorised deposit-taking institutions, general insurers, life insurers, private health insurers and superannuation funds.

**qa:Fringe benefits tax**

- Reference: Employers pay FBT at 47 percent on non-cash benefits given to employees, in a year running from April to March.
- Zero-shot: Employers in England and Wales are required to pay a fringe benefits tax (FBT) rate of 47 percent.
- Fine-tuned: Employers pay fringe benefits tax on employee benefits, which run from April to March, and the FBT rate is 47 percent, equal to the top marginal tax rate plus the Medicare levy.

**corpus-3**

- Reference: GST applies to taxable supplies made in an enterprise, but input-taxed supplies such as residential rent carry no GST and no input tax credits.
- Zero-shot: Identify the goods and services tax (GST).
- Fine-tuned: Input-taxed supplies, such as residential rent and financial supplies, do not attract GST and the supplier cannot claim input tax credits for acquisitions relating to them.

**corpus-11**

- Reference: Employees who have served the minimum period and are under the income threshold or covered by an award are protected from harsh, unjust or unreasonable dismissal.
- Zero-shot: Find out if you are protected from unfair dismissal.
- Fine-tuned: Employees are protected from unfair dismissal if they earn less than the high income threshold or are covered by an award or enterprise agreement.

**corpus-10**

- Reference: Enterprise bargaining lets workplaces negotiate tailored agreements, with bargaining representatives bound by good faith requirements.
- Zero-shot: Understand enterprise bargaining.
- Fine-tuned: Enterprise bargaining allows employers, employees and bargaining representatives to negotiate a specific workplace agreement, and bargaining representatives must meet good faith bargaining requirements.

**qa:Flexible working arrangements**

- Reference: Eligible employees can request flexible work in writing, and employers must respond within 21 days and refuse only on reasonable business grounds.
- Zero-shot: Request a flexible working arrangement if you are a parent or carer.
- Fine-tuned: Employees with at least twelve months of continuous service can request flexible working arrangements if they are a parent, carer, 55 or older, pregnant or experiencing family and domestic violence.

**qa:Goods and services tax**

- Reference: GST is a 10 percent tax that businesses must register for once turnover reaches $75,000, with credits available for business purchases.
- Zero-shot: Register for GST.
- Fine-tuned: Goods and services tax is a broad-based tax of 10 percent on goods, services and other items sold or consumed in Australia. Registered businesses report GST on their business activity statement and can claim credits for GST included in the price of business purchases.

**qa:Annual leave**

- Reference: Employees accrue four weeks of paid annual leave a year, while casuals receive a loading instead.
- Zero-shot: Understand the benefits of paid annual leave.
- Fine-tuned: Full-time and part-time employees accrue four weeks paid annual leave for each year, while shift workers receive an additional week. Casual employees do not receive paid annual leave but are compensated through a casual loading of 25 percent.

**qa:Capital gains tax discount**

- Reference: Individuals and trusts holding assets for twelve months get a 50 percent CGT discount, super funds get one-third and companies get none.
- Zero-shot: Understand how a CGT event occurs. Understand how a CGT event occurs.
- Fine-tuned: Individuals and trusts can reduce capital gains by 50 percent, while superannuation funds receive a discount of one-third. Capital losses cannot be offset against capital gains and can be carried forward indefinitely.

**corpus-22**

- Reference: Market integrity rules add requirements for market participants, and insider trading means trading on inside information.
- Zero-shot: In the UK, insider trading is regulated by the Financial Conduct Authority (FCA).
- Fine-tuned: Market integrity rules supplement the Corporations Act to set market participants on licensed financial markets, and insider trading occurs when a person in possession of inside information trades in relevant financial products.

**qa:Unfair dismissal**

- Reference: Unfair dismissal claims go to the Fair Work Commission within 21 days and may lead to reinstatement or capped compensation.
- Zero-shot: Apply to the Fair Work Commission if you were unfairly dismissed.
- Fine-tuned: Employees who claim unfair dismissal can apply to the Fair Work Commission within 21 days of the dismissal.

**corpus-21**

- Reference: Financial services businesses need an Australian financial services licence and must meet obligations on resources, competence, risk management and dispute resolution.
- Zero-shot: Obtain an Australian financial services licence.
- Fine-tuned: A financial services licence is required to run a financial services business in Australia and licensees must comply with general obligations, including a dispute resolution system and external complaints authority.

