# Canonical effect-size method

## Estimand

For method \(m\), question \(q\), and run \(r\), let \(Y_{mqr}=1\) when the returned option equals the gold option and 0 otherwise. Missing and invalid responses are assigned 0.

The reported accuracy is:

\[
\hat{p}_m = \frac{1}{Q}\sum_{q=1}^{Q}\left(\frac{1}{5}\sum_{r=1}^{5}Y_{mqr}\right),
\]

where \(Q=1{,}045\). This reproduces the manuscript's PharmkGPT result of 95.62%.

## Effect sizes

The primary paired effect size against comparator \(c\) is the absolute accuracy difference:

\[
\Delta_c = \hat{p}_{PharmkGPT} - \hat{p}_c.
\]

The secondary effect size is relative error reduction:

\[
RER_c = \frac{\hat{p}_{PharmkGPT}-\hat{p}_c}{1-\hat{p}_c}.
\]

## Uncertainty and multiplicity

Confidence intervals use 20,000 nonparametric percentile-bootstrap resamples of questions. Resampling a question retains all methods and all five runs, preserving the paired and repeated-run structure. Two-sided paired t tests operate on the per-question five-run mean-correctness values. Holm adjustment is applied across the eight PharmkGPT comparisons within each analysis category.

Domain analyses are secondary. The overall 1,045-question analysis is the reviewer-facing main comparison.
