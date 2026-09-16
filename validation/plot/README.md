# Figures

| Directory | Contents |
| --- | --- |
| [recall](recall/) | Recall@K by domain |
| [accuracy](accuracy/) | Accuracy and significance by domain |
| [effect_size](effect_size/) | Paired effect sizes and confidence intervals |

Generate figures from saved results:

```bash
python validation/plot/recall/scripts/plot_individual_panels.py
python validation/plot/accuracy/scripts/plot_accuracy_panels.py
python validation/plot/effect_size/plot.py
```

Figures are saved in each subdirectory's `figures/` folder.
