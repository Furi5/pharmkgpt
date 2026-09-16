# Corrected Supplementary Table 1

## Publication-ready caption

**Supplementary Table 1 | Accuracy and paired effect-size comparisons of PharmKGPT versus eight baselines across the overall benchmark and five pharmacogenomics domains.** Accuracy is reported as mean ± SD (%) over five independent runs. Missing or invalid responses were counted as incorrect. The primary effect size is the paired absolute accuracy difference (ΔAcc, percentage points) between PharmKGPT and each comparator. Ninety-five percent confidence intervals were estimated using 20,000 nonparametric bootstrap resamples clustered by question, retaining all five runs and all methods within each resampled question. Relative error reduction (RER) is reported as a secondary effect size. Two-sided paired t tests were performed on per-question five-run mean correctness values, with Holm adjustment across the eight PharmKGPT comparisons within each domain. The overall benchmark contains 1,045 questions; domain sample sizes are shown in the table.

## A. Corrected accuracy results

| Domain | n | PharmKGPT | HippoRAG 2 | SemanticRAG | LightRAG | Gemma3-27B | LLaMA3-8B | LLaMA2-13B | DeepSeek-32B | Qwen-32B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 1,045 | **95.62 ± 0.23** | 88.38 ± 0.87 | 84.96 ± 0.27 | 84.13 ± 1.23 | 81.91 ± 0.18 | 80.88 ± 0.28 | 75.60 ± 0.61 | 73.38 ± 0.27 | 69.93 ± 0.27 |
| Genetic | 247 | **89.31 ± 0.61** | 84.62 ± 0.91 | 84.94 ± 0.53 | 76.60 ± 2.55 | 65.18 ± 0.50 | 61.05 ± 0.92 | 55.14 ± 2.17 | 59.43 ± 1.01 | 60.00 ± 1.38 |
| Clinical | 233 | **98.03 ± 0.49** | 90.82 ± 1.16 | 84.21 ± 0.71 | 87.90 ± 1.53 | 83.69 ± 0.74 | 83.69 ± 1.77 | 82.58 ± 2.07 | 73.48 ± 2.32 | 64.72 ± 0.98 |
| Cell | 213 | **98.12 ± 0.33** | 88.92 ± 1.39 | 85.73 ± 0.54 | 90.70 ± 2.03 | 93.80 ± 0.21 | 95.96 ± 0.42 | 88.92 ± 1.35 | 81.97 ± 1.39 | 78.69 ± 1.35 |
| Pathway | 230 | **97.22 ± 0.39** | 87.65 ± 1.09 | 85.39 ± 0.50 | 82.96 ± 2.35 | 86.09 ± 0.31 | 87.39 ± 0.87 | 79.48 ± 1.08 | 78.09 ± 1.36 | 76.17 ± 1.17 |
| Metabolism | 122 | **96.39 ± 0.93** | 91.80 ± 1.74 | 84.26 ± 1.35 | 82.95 ± 1.35 | 83.77 ± 0.90 | 77.05 ± 1.74 | 73.11 ± 2.27 | 77.54 ± 1.80 | 72.95 ± 1.53 |

Accuracy values are percentages. Bold indicates the corrected PharmKGPT values.

## B. PharmKGPT count-level audit

| Domain | Correct counts in runs 1–5 | Correct / total across five runs | Mean accuracy (%) |
| --- | --- | ---: | ---: |
| Overall | 1,000; 998; 998; 1,003; 997 | 4,996 / 5,225 | **95.62** |
| Genetic | 222; 221; 221; 221; 218 | 1,103 / 1,235 | **89.31** |
| Clinical | 229; 227; 228; 230; 228 | 1,142 / 1,165 | **98.03** |
| Cell | 208; 209; 209; 210; 209 | 1,045 / 1,065 | **98.12** |
| Pathway | 223; 223; 224; 223; 225 | 1,118 / 1,150 | **97.22** |
| Metabolism | 118; 118; 116; 119; 117 | 588 / 610 | **96.39** |

## C. Corrected paired effect sizes

| Domain | Comparator | Comparator accuracy (%) | ΔAcc (pp) [95% CI] | RER (%) [95% CI] | Holm-adjusted p |
| --- | --- | ---: | ---: | ---: | ---: |
| Overall | HippoRAG 2 | 88.38 | 7.23 [5.51, 9.00] | 62.27 [51.85, 71.50] | <0.0001 |
| Overall | SemanticRAG | 84.96 | 10.66 [8.63, 12.75] | 70.87 [62.73, 78.16] | <0.0001 |
| Overall | LightRAG | 84.13 | 11.48 [9.95, 13.00] | 72.38 [65.12, 79.16] | <0.0001 |
| Overall | Gemma3-27B | 81.91 | 13.70 [11.35, 16.06] | 75.77 [68.84, 81.89] | <0.0001 |
| Overall | LLaMA3-8B | 80.88 | 14.74 [12.36, 17.09] | 77.08 [70.35, 82.97] | <0.0001 |
| Overall | LLaMA2-13B | 75.60 | 20.02 [17.59, 22.53] | 82.04 [77.04, 86.66] | <0.0001 |
| Overall | DeepSeek-32B | 73.38 | 22.24 [20.02, 24.52] | 83.54 [79.14, 87.62] | <0.0001 |
| Overall | Qwen-32B | 69.93 | 25.68 [23.27, 28.11] | 85.42 [81.40, 89.14] | <0.0001 |
| Genetic | HippoRAG 2 | 84.62 | 4.70 [0.49, 8.91] | 30.53 [3.88, 51.93] | 0.0571 |
| Genetic | SemanticRAG | 84.94 | 4.37 [0.08, 8.74] | 29.03 [0.56, 51.35] | 0.0571 |
| Genetic | LightRAG | 76.60 | 12.71 [8.83, 16.44] | 54.33 [38.96, 68.40] | <0.0001 |
| Genetic | Gemma3-27B | 65.18 | 24.13 [17.73, 30.53] | 69.30 [57.18, 79.76] | <0.0001 |
| Genetic | LLaMA3-8B | 61.05 | 28.26 [21.94, 34.66] | 72.56 [61.85, 81.91] | <0.0001 |
| Genetic | LLaMA2-13B | 55.14 | 34.17 [27.77, 40.49] | 76.17 [67.11, 84.25] | <0.0001 |
| Genetic | DeepSeek-32B | 59.43 | 29.88 [24.21, 35.47] | 73.65 [64.06, 82.26] | <0.0001 |
| Genetic | Qwen-32B | 60.00 | 29.31 [23.40, 35.30] | 73.28 [63.33, 82.17] | <0.0001 |
| Clinical | HippoRAG 2 | 90.82 | 7.21 [4.21, 10.64] | 78.50 [59.55, 92.55] | <0.0001 |
| Clinical | SemanticRAG | 84.21 | 13.82 [9.36, 18.54] | 87.50 [76.14, 95.85] | <0.0001 |
| Clinical | LightRAG | 87.90 | 10.13 [7.21, 13.13] | 83.69 [69.05, 94.74] | <0.0001 |
| Clinical | Gemma3-27B | 83.69 | 14.33 [9.70, 19.23] | 87.89 [76.32, 96.02] | <0.0001 |
| Clinical | LLaMA3-8B | 83.69 | 14.33 [9.96, 18.88] | 87.89 [76.92, 95.95] | <0.0001 |
| Clinical | LLaMA2-13B | 82.58 | 15.45 [11.24, 19.83] | 88.67 [78.19, 96.36] | <0.0001 |
| Clinical | DeepSeek-32B | 73.48 | 24.55 [20.17, 29.18] | 92.56 [86.36, 97.55] | <0.0001 |
| Clinical | Qwen-32B | 64.72 | 33.30 [28.24, 38.54] | 94.40 [89.64, 98.19] | <0.0001 |
| Cell | HippoRAG 2 | 88.92 | 9.20 [5.92, 12.86] | 83.05 [67.83, 95.10] | <0.0001 |
| Cell | SemanticRAG | 85.73 | 12.39 [8.17, 16.90] | 86.84 [74.71, 96.41] | <0.0001 |
| Cell | LightRAG | 90.70 | 7.42 [4.60, 10.33] | 79.80 [57.89, 95.00] | <0.0001 |
| Cell | Gemma3-27B | 93.80 | 4.32 [0.75, 8.08] | 69.70 [21.05, 92.86] | 0.0435 |
| Cell | LLaMA3-8B | 95.96 | 2.16 [-0.75, 5.16] | 53.49 [-32.52, 88.71] | 0.1534 |
| Cell | LLaMA2-13B | 88.92 | 9.20 [5.16, 13.43] | 83.05 [62.07, 95.83] | <0.0001 |
| Cell | DeepSeek-32B | 81.97 | 16.15 [12.02, 20.47] | 89.58 [79.17, 97.35] | <0.0001 |
| Cell | Qwen-32B | 78.69 | 19.44 [14.93, 24.23] | 91.19 [82.61, 97.71] | <0.0001 |
| Pathway | HippoRAG 2 | 87.65 | 9.57 [5.30, 14.09] | 77.46 [55.56, 91.93] | <0.0001 |
| Pathway | SemanticRAG | 85.39 | 11.83 [7.57, 16.26] | 80.95 [65.65, 92.96] | <0.0001 |
| Pathway | LightRAG | 82.96 | 14.26 [11.04, 17.48] | 83.67 [71.66, 93.69] | <0.0001 |
| Pathway | Gemma3-27B | 86.09 | 11.13 [6.96, 15.65] | 80.00 [63.92, 92.22] | <0.0001 |
| Pathway | LLaMA3-8B | 87.39 | 9.83 [5.91, 14.00] | 77.93 [59.32, 91.56] | <0.0001 |
| Pathway | LLaMA2-13B | 79.48 | 17.74 [12.87, 22.87] | 86.44 [75.86, 94.87] | <0.0001 |
| Pathway | DeepSeek-32B | 78.09 | 19.13 [14.87, 23.57] | 87.30 [77.56, 95.22] | <0.0001 |
| Pathway | Qwen-32B | 76.17 | 21.04 [16.35, 25.91] | 88.32 [79.10, 95.53] | <0.0001 |
| Metabolism | HippoRAG 2 | 91.80 | 4.59 [1.64, 8.03] | 56.00 [26.67, 84.91] | 0.0066 |
| Metabolism | SemanticRAG | 84.26 | 12.13 [6.89, 17.87] | 77.08 [57.69, 93.15] | <0.0001 |
| Metabolism | LightRAG | 82.95 | 13.44 [9.51, 17.38] | 78.85 [61.61, 93.81] | <0.0001 |
| Metabolism | Gemma3-27B | 83.77 | 12.62 [7.05, 18.85] | 77.78 [58.46, 93.51] | 0.0001 |
| Metabolism | LLaMA3-8B | 77.05 | 19.34 [12.30, 26.72] | 84.29 [68.55, 95.80] | <0.0001 |
| Metabolism | LLaMA2-13B | 73.11 | 23.28 [16.23, 30.82] | 86.59 [74.00, 96.37] | <0.0001 |
| Metabolism | DeepSeek-32B | 77.54 | 18.85 [13.11, 24.92] | 83.94 [69.81, 95.32] | <0.0001 |
| Metabolism | Qwen-32B | 72.95 | 23.44 [17.38, 29.84] | 86.67 [74.42, 96.27] | <0.0001 |

ΔAcc is PharmKGPT accuracy minus comparator accuracy in percentage points. RER is relative error reduction. Domain analyses are secondary; the overall 1,045-question comparison is the primary analysis. A Holm-adjusted p value of at least 0.05 is not statistically significant.

## Provenance

- Authoritative answer matrix: `data/all_answer.csv`
- Accuracy summary: `results/method_accuracy_summary.csv`
- Run-level audit: `results/run_accuracy_summary.csv`
- Overall effects: `results/canonical_effect_sizes_overall.csv`
- Domain effects: `results/canonical_effect_sizes_by_domain.csv`
- Bootstrap replicates: 20,000
- Bootstrap seed: 20260803
- Missing/invalid response policy: counted as incorrect

