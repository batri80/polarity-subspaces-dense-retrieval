# Semantic Odometry: statistics for the three-seed run (tests3)

E5-base-v2; v2 frame recipe; seeds 1-3. Binary outcomes: exact McNemar with
discordant pairs pooled over seeds (the per-seed column is the more
conservative evidence, since pooling treats seed-instance pairs as
independent). SciFact: per-query nDCG@10, paired differences averaged over
seeds, query-level bootstrap 95% CI and Wilcoxon signed-rank test.
Difference-in-differences rows compare negated (n=35) with affirmative
(n=265) claims, with a bootstrap over queries within each group.

| comparison | mean diff | per seed / by group | CI or win/loss | p |
|---|---|---|---|---|
| ## A. ExcluIR pairwise RR (McNemar, pooled over 3 seeds) |  |  |  |  |
| frame vs zeroshot | +0.0201 | [+0.020, +0.020, +0.020] | 1104/896 | 3.6e-06 |
| frame vs frame_randomU | +0.0119 | [+0.013, +0.011, +0.011] | 411/288 | 3.7e-06 |
| frame vs plain_ft | +0.0100 | [+0.028, +0.016, -0.013] | 427/323 | 0.00017 |
| plain_ft vs zeroshot | +0.0100 | [-0.008, +0.005, +0.034] | 1042/938 | 0.021 |
| frame_randomU vs zeroshot | +0.0082 | [+0.007, +0.009, +0.009] | 1008/923 | 0.056 |
| ## B. Reflection test, ExcluIR RR: reflect vs own (P1) and vs random-U reflect (P2) |  |  |  |  |
| zeroshot: reflect vs own | -0.0009 | [-0.001, -0.001, -0.001] | 6/15 | 0.078 |
| plain_ft: reflect vs own | +0.0045 | [+0.003, +0.008, +0.002] | 134/87 | 0.0019 |
| frame: reflect vs own | +0.0016 | [-0.002, -0.003, +0.010] | 174/157 | 0.38 |
| frame_randomU: reflect vs own | +0.0015 | [+0.003, -0.002, +0.004] | 53/37 | 0.11 |
| plain_ft: reflect vs reflect_randU | +0.0035 | [+0.001, +0.007, +0.002] | 130/94 | 0.019 |
| frame: reflect vs reflect_randU | +0.0012 | [+0.000, -0.002, +0.005] | 102/90 | 0.43 |
| frame_randomU: reflect vs reflect_randU | -0.0001 | [+0.000, -0.001, +0.000] | 26/27 | 1 |
| ## B3. Reflection, NevIR: reflect vs own (P3) |  |  |  |  |
| plain_ft: reflect vs own (NevIR) | -0.0441 | [-0.038, -0.043, -0.052] | 46/229 | 2.2e-30 |
| frame: reflect vs own (NevIR) | -0.0431 | [-0.047, -0.036, -0.046] | 92/271 | 1.3e-21 |
| frame_randomU: reflect vs own (NevIR) | +0.0005 | [-0.001, -0.001, +0.004] | 17/15 | 0.86 |
| ## C. SciFact damage from training (vs zeroshot), negated vs affirmative claims |  |  |  |  |
| plain_ft: (drop on negated) - (drop on affirmative) | -0.0997 | neg -0.193 / aff -0.093 | 95% CI [-0.196, -0.009] | 0.034 (bootstrap) |
| frame: (drop on negated) - (drop on affirmative) | -0.1893 | neg -0.320 / aff -0.130 | 95% CI [-0.301, -0.082] | 0.0004 (bootstrap) |
| frame_randomU: (drop on negated) - (drop on affirmative) | -0.0795 | neg -0.151 / aff -0.071 | 95% CI [-0.179, +0.012] | 0.094 (bootstrap) |
| plain_ft vs zeroshot, all claims | -0.1047 | [-0.146, -0.096, -0.072] | 95% CI [-0.130, -0.078] | 1.2e-14 |
| plain_ft vs zeroshot, negated claims (n=35) | -0.1927 | [-0.309, -0.156, -0.114] | 95% CI [-0.287, -0.108] | 0.00029 |
| plain_ft vs zeroshot, affirmative claims (n=265) | -0.0930 | [-0.125, -0.088, -0.066] | 95% CI [-0.121, -0.067] | 1.1e-11 |
| frame vs zeroshot, all claims | -0.1526 | [-0.212, -0.155, -0.091] | 95% CI [-0.183, -0.124] | 1.2e-19 |
| frame vs zeroshot, negated claims (n=35) | -0.3198 | [-0.432, -0.315, -0.213] | 95% CI [-0.432, -0.212] | 2.3e-05 |
| frame vs zeroshot, affirmative claims (n=265) | -0.1305 | [-0.183, -0.134, -0.075] | 95% CI [-0.160, -0.101] | 2.1e-15 |
| frame_randomU vs zeroshot, all claims | -0.0805 | [-0.082, -0.095, -0.064] | 95% CI [-0.106, -0.055] | 8.3e-10 |
| frame_randomU vs zeroshot, negated claims (n=35) | -0.1507 | [-0.157, -0.157, -0.139] | 95% CI [-0.247, -0.066] | 0.0024 |
| frame_randomU vs zeroshot, affirmative claims (n=265) | -0.0712 | [-0.072, -0.087, -0.054] | 95% CI [-0.097, -0.045] | 7.3e-08 |
| ## D. Content readout on SciFact: content vs full (C1), content vs random-U (C2) |  |  |  |  |
| zeroshot: content vs full, all | +0.0059 | [+0.006, +0.006, +0.006] | 95% CI [-0.001, +0.013] | 0.21 |
| zeroshot: content vs full, negated | -0.0052 | [-0.005, -0.005, -0.005] | 95% CI [-0.030, +0.012] | nan |
| zeroshot: content vs full, affirmative | +0.0073 | [+0.007, +0.007, +0.007] | 95% CI [+0.000, +0.016] | 0.21 |
| zeroshot: content vs content_randU, all | +0.0032 | [+0.003, +0.003, +0.003] | 95% CI [-0.004, +0.011] | 0.64 |
| plain_ft: content vs full, all | +0.0582 | [+0.083, +0.060, +0.031] | 95% CI [+0.042, +0.074] | 9.2e-12 |
| plain_ft: content vs full, negated | +0.1406 | [+0.224, +0.147, +0.051] | 95% CI [+0.080, +0.208] | 0.00034 |
| plain_ft: content vs full, affirmative | +0.0473 | [+0.064, +0.049, +0.029] | 95% CI [+0.032, +0.064] | 1.2e-08 |
| plain_ft: content vs content_randU, all | +0.0591 | [+0.083, +0.065, +0.030] | 95% CI [+0.043, +0.076] | 1.9e-11 |
| frame: content vs full, all | +0.0841 | [+0.139, +0.082, +0.031] | 95% CI [+0.062, +0.106] | 3.3e-12 |
| frame: content vs full, negated | +0.2649 | [+0.401, +0.245, +0.149] | 95% CI [+0.175, +0.360] | 3.1e-05 |
| frame: content vs full, affirmative | +0.0603 | [+0.105, +0.060, +0.016] | 95% CI [+0.040, +0.080] | 6e-08 |
| frame: content vs content_randU, all | +0.0620 | [+0.103, +0.065, +0.018] | 95% CI [+0.043, +0.082] | 1.8e-09 |
| frame_randomU: content vs full, all | +0.0049 | [+0.004, +0.005, +0.005] | 95% CI [+0.001, +0.009] | 0.0077 |
| frame_randomU: content vs full, negated | +0.0127 | [+0.018, -0.005, +0.025] | 95% CI [-0.004, +0.032] | 0.21 |
| frame_randomU: content vs full, affirmative | +0.0039 | [+0.002, +0.006, +0.003] | 95% CI [+0.001, +0.008] | 0.019 |
| frame_randomU: content vs content_randU, all | +0.0052 | [+0.006, +0.008, +0.002] | 95% CI [+0.002, +0.009] | 0.0045 |
| frame: recovery on negated - recovery on affirmative | +0.2046 | neg +0.265 / aff +0.060 | 95% CI [+0.111, +0.304] | 0.0002 (bootstrap) |
| plain_ft: recovery on negated - recovery on affirmative | +0.0932 | neg +0.141 / aff +0.047 | 95% CI [+0.033, +0.162] | 0.0016 (bootstrap) |
| frame(content) vs plain_ft(content), all (C4) | -0.0220 | [-0.009, -0.038, -0.019] | 95% CI [-0.033, -0.011] | 0.00051 |
| ## E. Content readout, NevIR: content vs full (C3) and vs random-U |  |  |  |  |
| zeroshot: content vs full (NevIR) | -0.0824 | [-0.082, -0.082, -0.082] | 66/408 | 3.1e-61 |
| zeroshot: content vs content_randU (NevIR) | -0.0853 | [-0.085, -0.085, -0.085] | 57/411 | 3.4e-67 |
| plain_ft: content vs full (NevIR) | -0.2196 | [-0.216, -0.240, -0.202] | 198/1109 | 6.8e-154 |
| plain_ft: content vs content_randU (NevIR) | -0.2181 | [-0.215, -0.241, -0.199] | 197/1102 | 8.4e-153 |
| frame: content vs full (NevIR) | -0.2133 | [-0.234, -0.236, -0.169] | 236/1121 | 4.5e-138 |
| frame: content vs content_randU (NevIR) | -0.2087 | [-0.234, -0.230, -0.162] | 215/1081 | 4.3e-139 |
| frame_randomU: content vs full (NevIR) | -0.0007 | [+0.004, -0.004, -0.003] | 25/28 | 0.78 |
| frame_randomU: content vs content_randU (NevIR) | -0.0019 | [+0.001, -0.001, -0.006] | 19/27 | 0.3 |
| ## F. Content readout, ExcluIR: does the exclusion gain survive removing U? |  |  |  |  |
| plain_ft: content vs full (ExcluIR) | +0.0060 | [+0.009, +0.006, +0.003] | 279/217 | 0.0061 |
| frame: content vs full (ExcluIR) | +0.0027 | [-0.001, -0.001, +0.010] | 252/224 | 0.22 |
| frame_randomU: content vs full (ExcluIR) | +0.0001 | [+0.001, -0.002, +0.001] | 60/59 | 1 |
| frame(content) vs plain_ft(full) (ExcluIR) | +0.0127 | [+0.027, +0.014, -0.003] | 485/353 | 5.8e-06 |
