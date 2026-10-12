# Integrated Gradients token attributions

- **Model**: `models\finetuned\bert-base-uncased_lora`
- **Method**: Captum LayerIntegratedGradients on word embeddings, 50 steps, [PAD] baseline
- **Completeness check**: mean |convergence delta| = 0.0498 (max 0.3147)
- Positive scores push the model towards its predicted class; negative scores push away.

## Most influential words per predicted class (mean attribution, words seen at least twice)

- **corporate_governance**: companies (+0.29), members (+0.28), business (+0.27), from (+0.23), directors (+0.22), corporations (+0.22), board (+0.20), company (+0.18), act (+0.16), with (+0.16)
- **taxation**: tax (+0.09), interest (+0.08), superannuation (+0.08), can (+0.07), contributions (+0.06), bought (+0.05), property (+0.03), income (+0.03), benefits (+0.02), paid (+0.01)
- **workplace_relations**: employment (+0.33), work (+0.26), day (+0.21), hours (+0.20), employee (+0.20), service (+0.20), working (+0.16), leave (+0.15), employers (+0.15), days (+0.14)

## Misclassified and challenge examples

- [challenge, WRONG] *My company car is mostly used on weekends, will my boss have to pay something extra?*  
  gold `taxation`, predicted `workplace_relations` (p=0.87); top words: weekends (+0.09), boss (+0.08), on (+0.07), extra (+0.07), will (+0.06)
- [challenge, WRONG] *How much do I need to earn before I have to start paying anything to the government?*  
  gold `taxation`, predicted `workplace_relations` (p=0.61); top words: i (+0.13), the (+0.12), have (+0.09), before (+0.06), how (+0.06)
- [challenge, WRONG] *Can I claim the electricity I use while working from my kitchen table?*  
  gold `taxation`, predicted `workplace_relations` (p=0.90); top words: kitchen (+0.16), working (+0.14), table (+0.13), i (+0.12), from (+0.11)
- [challenge, WRONG] *What happens if I pay my quarterly instalment a month late?*  
  gold `taxation`, predicted `workplace_relations` (p=0.95); top words: late (+0.45), month (+0.17), instalment (+0.15), if (-0.13), my (+0.13)
- [challenge, WRONG] *The business is running out of cash and can't pay suppliers, should the directors keep trading?*  
  gold `corporate_governance`, predicted `workplace_relations` (p=0.77); top words: directors (-0.36), trading (-0.32), , (+0.31), ? (+0.29), the (+0.19)
- [challenge, WRONG] *I own six percent of a listed miner, do I have to tell anyone?*  
  gold `corporate_governance`, predicted `workplace_relations` (p=0.44); top words: own (-0.41), miner (+0.14), , (+0.14), six (+0.13), have (+0.12)
- [challenge, WRONG] *An employee reported fraud by the chief executive, how are they protected?*  
  gold `corporate_governance`, predicted `workplace_relations` (p=0.97); top words: employee (+0.25), how (+0.19), an (+0.17), ? (+0.16), chief (-0.16)
- [challenge, WRONG] *Shareholders voted against the pay report twice, what happens now?*  
  gold `corporate_governance`, predicted `workplace_relations` (p=0.50); top words: shareholders (-0.46), , (+0.33), happens (+0.27), the (-0.21), pay (+0.19)
- [challenge, correct] *I sold some shares I bought three years ago, do I get a discount on what I owe?*  
  gold `taxation`, predicted `taxation` (p=0.89); top words: owe (+0.46), i (-0.25), i (-0.21), i (+0.18), shares (-0.17)
- [challenge, correct] *Can I write off the laptop I bought for my side business this year?*  
  gold `taxation`, predicted `taxation` (p=0.51); top words: can (+0.16), i (-0.15), ? (+0.12), this (-0.12), i (-0.09)
- [challenge, correct] *We just crossed seventy-five thousand in sales, do we now need to add ten percent to our prices?*  
  gold `taxation`, predicted `taxation` (p=0.70); top words: prices (+0.47), our (-0.28), just (-0.16), , (+0.14), to (-0.12)
- [challenge, correct] *Do I have to declare the rent I get from the granny flat?*  
  gold `taxation`, predicted `taxation` (p=0.50); top words: i (-0.26), i (-0.22), do (+0.17), from (-0.14), rent (+0.14)
