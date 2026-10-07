## Labs
| lab | n | rate (Hz) | EEG channels | recording reference | system | mains (Hz) |
|---|---|---|---|---|---|---|
| Dartmouth | 28 | 500 | 32 | M2 | Brain Products | 60 |
| Florida | 50 | 500 | 129 | E129 | EGI | 60 |
| Mainz | 25 | 1000 | 60 | FCz | Brain Products | 50 |
| Münster | 30 | 1024 | 69 | Reference free | Biosemi | 50 |
| NorthDakota | 28 | 250 | 66 | Reference free | Biosemi | 60 |
| Ohio | 24 | 1000 | 32 | Cz | Brain Products | 60 |
| Reykjavik | 30 | 1000 | 32 | Fz | Brain Products | 50 |
| Sheffield | 25 | 512 | 66 | Reference free | Biosemi | 50 |
| Zürich (Prof. Langer) | 40 | 500 | 125 | Cpz | ANT Neuro | 50 |
| Zürich (Prof. Sauseng) | 24 | 1000 | 64 | FCz | Brain Products | 50 |

## BWE
| condition | deficit dB | loud / mid / quiet dB | coh | coh > 40 Hz | kappa | SNR dB | vs ceiling dB | LB coh | CRPS | PIT low / high | gap HB (indep.) dB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ceiling: true low band, empty high band | -62.00 | -62.1 / -61.8 / -61.8 | -- | -- | -- | 12.32 | 0.00 | 1.000 | 8.237 |  |  |
| naive (polyphase sinc) | -51.12 | -50.7 / -51.0 / -51.5 | -- | -- | -- | 12.21 | -0.11 | 1.000 | 6.593 |  |  |
| det_paper: L1 only (LISA's released loss) | -23.32 | -22.5 / -23.7 / -24.0 | 0.020 | 0.002 | 0.328 | 12.21 | -0.12 | 1.000 | 2.901 |  |  |
| det: L1 + 0.1 MSSTFT | -7.82 | -8.3 / -7.4 / -7.1 | 0.023 | 0.001 | 0.104 | 11.42 | -0.91 | 1.000 | 1.170 |  |  |
| es_marg at tau = 0 | -10.89 | -10.7 / -10.8 / -11.2 | 0.021 | 0.001 | 0.123 | 11.78 | -0.54 | 1.000 | 1.509 |  |  |
| es_marg: energy score, one draw (M = 8) | -2.08 | -2.5 / -1.6 / -1.2 | 0.018 | 0.001 | 0.013 | 8.62 | -3.71 | 1.000 | 0.673 | 0.098 / 0.316 | 2.66 (2.91) |
| shaped noise: unconditional, mean HB spectrum (M = 8) | 4.25 | 2.9 / 5.1 / 6.4 | 0.012 | 0.001 | 0.011 | 6.05 | -6.28 | 1.000 | 0.801 | 0.368 / 0.081 | 4.39 (4.38) |
| det_paper, raw | -23.20 | -22.4 / -23.5 / -23.9 | 0.023 | 0.002 | 0.315 | 12.24 | -0.08 | 1.000 | 2.900 |  |  |
| det, raw | -7.75 | -8.3 / -7.3 / -7.0 | 0.028 | 0.001 | 0.107 | 11.35 | -0.98 | 0.999 | 1.172 |  |  |
| es_marg, raw | -2.29 | -2.8 / -1.8 / -1.5 | 0.022 | 0.001 | 0.015 | 7.43 | -4.90 | 0.976 | 0.686 | 0.094 / 0.327 | 2.59 (2.82) |

Ideals at M = 8: PIT low = high = 0.111; gap equals its independent-draw value when draws are as diverse as their level allows; a calibrated sampler has gap 2.50 dB.

High-band envelope from the low band alone (ridge on 5 log band energies, training subjects -> held-out): R^2 pooled 0.529; within a recording r = 0.574 (R^2 0.289), 83160 frames.

| condition | envelope slope | r | sd ratio |
|---|---|---|---|
| det: L1 + 0.1 MSSTFT | 0.528 | 0.569 | 0.929 |
| es_marg at tau = 0 | 0.646 | 0.597 | 1.081 |
| es_marg: energy score, one draw (M = 8) | 0.410 | 0.295 | 1.388 |
| shaped noise: unconditional, mean HB spectrum (M = 8) | 0.013 | 0.037 | 0.351 |

| lab | det deficit | es_marg deficit | det coh | es_marg coh | HB share of energy |
|---|---|---|---|---|---|
| Dartmouth | -7.02 | -1.24 | 0.027 | 0.019 | 6.1 % |
| Florida | -6.32 | 1.09 | 0.014 | 0.028 | 9.2 % |
| Mainz | -6.53 | -1.00 | 0.021 | 0.014 | 5.4 % |
| Münster | -6.15 | -0.75 | 0.028 | 0.017 | 7.5 % |
| NorthDakota | -7.59 | -2.11 | 0.021 | 0.015 | 10.6 % |
| Ohio | -4.36 | 0.91 | 0.028 | 0.018 | 5.2 % |
| Reykjavik | -8.95 | -3.34 | 0.022 | 0.016 | 10.7 % |
| Sheffield | -6.93 | -1.02 | 0.023 | 0.015 | 7.5 % |
| Zürich (Prof. Langer) | -9.51 | -4.09 | 0.023 | 0.016 | 14.3 % |
| Zürich (Prof. Sauseng) | -14.83 | -9.24 | 0.021 | 0.022 | 41.6 % |

## Coherence of the CDA
n = 248 subjects.

| quantity | mean | median | t vs 0 |
|---|---|---|---|
| c, 0.3-0.9 s, broadband | 0.0200 | 0.0102 | 11.3 |
| p (ERP energy / trial energy) | 0.0251 | 0.0151 | 14.3 |
| kappa = c / p | 0.5075 | 0.6605 | 15.2 |
| c, 0-2 Hz | 0.0431 | 0.0203 | 11.8 |
| c, 2-4 Hz | 0.0062 | 0.0039 | 11.4 |
| c, 4-8 Hz | 0.0023 | 0.0016 | 9.7 |
| c, 8-13 Hz | 0.0026 | 0.0011 | 7.9 |
| c, 13-30 Hz | 0.0007 | 0.0004 | 5.6 |
| alpha waveform c (band-passed) | 0.0024 | 0.0011 | 8.4 |
| alpha envelope c (log power) | 0.0216 | 0.0091 | 10.6 |
| alpha lateralization, log contra/ipsi | -0.0854 | -0.0712 | -13.7 |
| beta waveform c | 0.0006 | 0.0002 | 4.7 |
| beta envelope c | 0.0051 | 0.0012 | 6.7 |
| theta waveform c | 0.0023 | 0.0016 | 10.4 |
| theta envelope c | 0.0009 | -0.0007 | 2.6 |

## Spatial
n = 30 held-out subjects, every electrode with a 10-05 label left out in turn; medians over subjects.

| band (Hz) | 1-4 | 4-8 | 8-13 | 13-30 | 30-60 | 60-125 |
|---|---|---|---|---|---|---|
| spline: c | 0.88 | 0.91 | 0.91 | 0.69 | 0.43 | 0.39 |
| spline: p | 0.91 | 0.93 | 0.95 | 0.86 | 0.75 | 0.68 |
| spline: SNR dB | 8.95 | 9.56 | 9.22 | 2.89 | 0.74 | 0.47 |
| Wiener: c | 0.96 | 0.95 | 0.95 | 0.79 | 0.57 | 0.52 |
| Wiener: p | 0.98 | 0.96 | 0.96 | 0.80 | 0.59 | 0.54 |
| Wiener: SNR dB | 12.43 | 12.27 | 12.51 | 6.33 | 3.53 | 2.67 |
| Gaussian draw: c | 0.96 | 0.95 | 0.95 | 0.79 | 0.57 | 0.52 |
| Gaussian draw: p | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.01 |
| Gaussian draw: SNR dB | 10.44 | 9.93 | 10.00 | 3.71 | 0.97 | 0.14 |

## CDA -> K
n = 248 subjects, 10 labs. CRPS in units of K; CRPSS = 1 - CRPS / CRPS(climatology), 95 % paired-bootstrap CI.

| feature | r | r within lab | CRPS clim | CRPSS cda | CRPSS cda, lab-centred | CRPSS lab oracle | point sd / sd(K) | PIT 90 % cover |
|---|---|---|---|---|---|---|---|---|
| CDA set size 4 | -0.283 (p 5.8e-06) | -0.146 | 0.462 | +0.042 [+0.010, +0.072] | +0.001 | +0.158 | 0.278 | 0.88 |
| CDA mean of set sizes 4 and 6 | -0.333 (p 7.5e-08) | -0.195 | 0.462 | +0.058 [+0.022, +0.094] | +0.007 | +0.158 | 0.326 | 0.90 |
| CDA increase 2 -> 4 | -0.226 (p 0.00033) | -0.111 | 0.462 | +0.026 [-0.003, +0.056] | -0.004 | +0.158 | 0.233 | 0.87 |
| CDA slope over set size | -0.312 (p 5.3e-07) | -0.215 | 0.462 | +0.051 [+0.013, +0.088] | +0.011 | +0.158 | 0.303 | 0.90 |

Split-half reliability (odd/even trials, Spearman-Brown): cda2 0.70, cda4 0.83, cda6 0.82, cda46_minus_2 0.35

Set size: CDA -0.497 ± 0.042 / -0.794 ± 0.052 / -0.747 ± 0.048 µV; K 1.71 / 2.41 / 2.26; CDA 4 vs 2: t = -7.90, p = 9.2e-14.

## CDA -> K, strict eye criterion (largest-residual-HEOG quarter dropped)
n = 186 subjects, 10 labs. CRPS in units of K; CRPSS = 1 - CRPS / CRPS(climatology), 95 % paired-bootstrap CI.

| feature | r | r within lab | CRPS clim | CRPSS cda | CRPSS cda, lab-centred | CRPSS lab oracle | point sd / sd(K) | PIT 90 % cover |
|---|---|---|---|---|---|---|---|---|
| CDA set size 4 | -0.282 (p 9.5e-05) | -0.143 | 0.475 | +0.035 [-0.006, +0.074] | -0.004 | +0.164 | 0.287 | 0.91 |
| CDA mean of set sizes 4 and 6 | -0.332 (p 3.7e-06) | -0.193 | 0.475 | +0.052 [+0.005, +0.099] | +0.004 | +0.164 | 0.333 | 0.90 |
| CDA increase 2 -> 4 | -0.210 (p 0.0041) | -0.102 | 0.475 | +0.015 [-0.022, +0.047] | -0.012 | +0.164 | 0.230 | 0.88 |
| CDA slope over set size | -0.287 (p 7.2e-05) | -0.200 | 0.475 | +0.039 [-0.004, +0.077] | +0.007 | +0.164 | 0.279 | 0.90 |

Split-half reliability (odd/even trials, Spearman-Brown): cda2 0.63, cda4 0.84, cda6 0.79, cda46_minus_2 0.35

Set size: CDA -0.456 ± 0.045 / -0.730 ± 0.060 / -0.674 ± 0.055 µV; K 1.72 / 2.37 / 2.18; CDA 4 vs 2: t = -6.18, p = 4e-09.
