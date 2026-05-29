# Probe-run comparison: filtered vs with_truncated

**Run filtered** — `data/concepts/sycophancy_qa_v2/stage1_probes_filtered`
**Run with_truncated** — `data/concepts/sycophancy_qa_v2/stage1_probes_with_truncated`

## Sample sizes (identical across all (layer, position) pairs in a run)

| split | filtered pos | filtered neg | with_truncated pos | with_truncated neg | Δpos | Δneg |
|---|---|---|---|---|---|---|
| train | 114 | 360 | 132 | 359 | -18 | +1 |
| val | 34 | 87 | 31 | 93 | +3 | -6 |
| ood | 7 | 376 | 7 | 378 | +0 | -2 |

Negative Δ in run filtered = run filtered dropped that many rollouts.

## Top 8 probes by run filtered's OOD pooled DiffMean

| probe | filtered val DM | with_truncated val DM | Δ | filtered OOD DM | with_truncated OOD DM | Δ | filtered OOD LR | with_truncated OOD LR | Δ |
|---|---|---|---|---|---|---|---|---|---|
| layer40_last_token | 0.891 | 0.907 | -0.016 | 0.807 | 0.797 | +0.010 | 0.856 | 0.864 | -0.008 |
| layer8_answer_mean_pool | 0.875 | 0.939 | -0.064 | 0.804 | 0.793 | +0.011 | 0.618 | 0.594 | +0.024 |
| layer56_end_of_reasoning | 0.904 | 0.908 | -0.004 | 0.799 | 0.803 | -0.004 | 0.823 | 0.862 | -0.039 |
| layer60_end_of_reasoning | 0.901 | 0.920 | -0.019 | 0.787 | 0.819 | -0.032 | 0.845 | 0.930 | -0.085 |
| layer44_last_token | 0.888 | 0.899 | -0.012 | 0.784 | 0.768 | +0.017 | 0.873 | 0.887 | -0.014 |
| layer52_end_of_reasoning | 0.904 | 0.926 | -0.021 | 0.768 | 0.772 | -0.004 | 0.852 | 0.926 | -0.074 |
| layer12_answer_mean_pool | 0.883 | 0.921 | -0.037 | 0.760 | 0.846 | -0.086 | 0.543 | 0.642 | -0.098 |
| layer44_answer_mean_pool | 0.902 | 0.948 | -0.047 | 0.752 | 0.852 | -0.101 | 0.820 | 0.890 | -0.070 |

**Δ = filtered − with_truncated**. Positive Δ on OOD = run filtered generalizes better.

## Calibration delta (DiffMean, OOD)

| probe | filtered TPR@5% | with_truncated TPR@5% | filtered TPR@10% | with_truncated TPR@10% | filtered catch-all FPR | with_truncated catch-all FPR | Δ catch-all |
|---|---|---|---|---|---|---|---|
| layer40_last_token | None% | — | None% | — | 56.4% | — | — |
| layer8_answer_mean_pool | None% | — | None% | — | 42.0% | — | — |
| layer56_end_of_reasoning | None% | — | None% | — | 54.0% | — | — |

Δ catch-all < 0 means run filtered needs lower FPR to catch all OOD positives — i.e., better calibration.
