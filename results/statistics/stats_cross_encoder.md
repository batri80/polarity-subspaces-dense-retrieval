# Cross-encoder statistics: subspace removal (content readout)

NevIR: exact McNemar pooled over seeds (per-seed differences in brackets). SciFact: per-query differences averaged over seeds, bootstrap 95% CI, Wilcoxon. The failed GTE run (ft_gte, NaN loss) is excluded; GTE plain FT = ft_gte_fp32.

## A. NevIR pairwise accuracy

| model | seeds | full | polarity removed | random removed | removed vs full: diff, W/L, p | removed vs random: diff, W/L, p |
|---|---|---|---|---|---|---|
| E5 zero-shot | 1 | 0.187 | 0.105 | 0.190 | -0.082 [-0.082], 22/136, p=2.9e-21 | -0.085, 19/137, p=3.1e-23 |
| GTE zero-shot | 1 | 0.146 | 0.080 | 0.149 | -0.067 [-0.067], 18/110, p=2.7e-17 | -0.069, 18/114, p=3e-18 |
| MPNet zero-shot | 1 | 0.081 | 0.062 | 0.082 | -0.019 [-0.019], 14/40, p=0.00054 | -0.020, 14/41, p=0.00036 |
| Contriever zero-shot | 1 | 0.097 | 0.078 | 0.096 | -0.019 [-0.019], 16/42, p=0.00086 | -0.018, 17/42, p=0.0015 |
| E5 plain FT | 3 | 0.464 | 0.245 | 0.463 | -0.220 [-0.216, -0.240, -0.202], 198/1109, p=6.8e-154 | -0.218, 197/1102, p=8.4e-153 |
| E5 anchored frame | 3 | 0.430 | 0.216 | 0.425 | -0.213 [-0.234, -0.236, -0.169], 236/1121, p=4.5e-138 | -0.209, 215/1081, p=4.3e-139 |
| MPNet plain FT | 2 | 0.525 | 0.292 | 0.525 | -0.232 [-0.246, -0.219], 82/725, p=1.8e-129 | -0.232, 83/726, p=4.9e-129 |
| GTE plain FT | 2 | 0.333 | 0.162 | 0.332 | -0.171 [-0.171, -0.171], 96/569, p=1.1e-82 | -0.170, 99/570, p=2.7e-81 |

## B. SciFact: content readout vs full readout

| model | all: diff [95% CI], p | negated (n=35): diff [CI], p | affirmative (n=265): diff [CI], p | content vs random-U, all: diff [CI], p |
|---|---|---|---|---|
| E5 zero-shot | +0.006 [-0.001, +0.014], p=0.21 | -0.005 [-0.030, +0.012], p=nan | +0.007 [-0.000, +0.016], p=0.21 | +0.003 [-0.004, +0.011], p=0.64 |
| GTE zero-shot | -0.002 [-0.010, +0.006], p=0.43 | -0.007 [-0.027, +0.005], p=nan | -0.001 [-0.009, +0.007], p=0.46 | +0.000 [-0.007, +0.008], p=0.81 |
| MPNet zero-shot | +0.002 [-0.003, +0.007], p=0.37 | +0.011 [+0.000, +0.026], p=nan | +0.000 [-0.005, +0.006], p=0.91 | +0.003 [-0.002, +0.008], p=0.073 |
| Contriever zero-shot | +0.001 [-0.004, +0.007], p=0.47 | +0.002 [-0.003, +0.007], p=nan | +0.001 [-0.005, +0.007], p=0.57 | -0.001 [-0.006, +0.005], p=0.7 |
| E5 plain FT | +0.058 [+0.042, +0.075], p=9.2e-12 | +0.141 [+0.080, +0.208], p=0.00034 | +0.047 [+0.032, +0.064], p=1.2e-08 | +0.059 [+0.043, +0.076], p=1.9e-11 |
| E5 anchored frame | +0.084 [+0.062, +0.107], p=3.3e-12 | +0.265 [+0.173, +0.360], p=3.1e-05 | +0.060 [+0.040, +0.080], p=6e-08 | +0.062 [+0.043, +0.082], p=1.8e-09 |
| MPNet plain FT | +0.020 [+0.008, +0.033], p=0.001 | +0.067 [+0.011, +0.135], p=0.041 | +0.014 [+0.004, +0.025], p=0.0074 | +0.019 [+0.008, +0.031], p=0.00087 |
| GTE plain FT | +0.003 [-0.004, +0.011], p=0.31 | +0.008 [-0.018, +0.040], p=nan | +0.003 [-0.005, +0.011], p=0.29 | +0.004 [-0.004, +0.012], p=0.3 |

## C. Training damage on SciFact relative to the same encoder zero-shot

| model | drop on negated | drop on affirmative | difference (neg - aff) [95% CI] | p (bootstrap) |
|---|---|---|---|---|
| E5 plain FT | -0.193 | -0.093 | -0.100 [-0.197, -0.009] | 0.028 |
| E5 anchored frame | -0.320 | -0.130 | -0.189 [-0.305, -0.082] | 0.0002 |
| MPNet plain FT | -0.079 | -0.012 | -0.067 [-0.134, -0.000] | 0.05 |
| GTE plain FT | -0.084 | -0.056 | -0.028 [-0.113, +0.049] | 0.51 |

## D. ExcluIR pairwise RR: content readout vs full

| model | full | content | diff, W/L, p |
|---|---|---|---|
| E5 zero-shot | 0.603 | 0.600 | -0.0029, 11/21, p=0.11 |
| GTE zero-shot | 0.629 | 0.631 | +0.0017, 17/11, p=0.34 |
| MPNet zero-shot | 0.713 | 0.712 | -0.0003, 11/12, p=1 |
| Contriever zero-shot | 0.617 | 0.618 | +0.0009, 8/5, p=0.58 |
| E5 plain FT | 0.613 | 0.619 | +0.0060, 279/217, p=0.0061 |
| E5 anchored frame | 0.623 | 0.626 | +0.0027, 252/224, p=0.22 |
| MPNet plain FT | 0.678 | 0.677 | -0.0013, 105/114, p=0.59 |
| GTE plain FT | 0.630 | 0.634 | +0.0046, 83/51, p=0.0072 |
