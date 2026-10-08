# Ideas — what the literature suggests, and what the data said

Written after three literature passes (EEG / time series / dynamical systems) and tested on the same 248
subjects (176,917 trials). Every test here takes minutes of CPU; nothing was trained. Scripts: `src/ideas/`,
outputs: `results/ideas/`. Per-trial table: `python -I src/trials.py data/prep results`.

Context from the literature: the published replication of this dataset (Strzelczyk et al. 2026, *Cortex*)
puts the CDA–capacity correlation at r ≈ 0.15 against 0.78 in the original. Nobody has scored single-trial EEG
with proper scores. Nobody has tested load hysteresis or critical slowing down before working-memory lapses.

## The headline: the "capacity" signal is mostly an engagement signal

`src/ideas/engagement.py`.

**How capacity was split.** Cowan's K mixes two things: how many items fit, and how often the person is
attending at all. At 2 items almost everyone is under capacity, so errors there are lapses. Attention is
therefore a = K₂/2 (Rouder et al. 2008), and capacity on attended trials is K/a.

| | r with lab-mean CDA (10 labs) | r with CDA within lab (n = 248) |
|---|---|---|
| K (what the field calls capacity) | −0.86 | −0.20 |
| attention (from 2-item trials only) | −0.85 | −0.19 |
| capacity on attended trials | −0.78 | −0.15 |

- **Between labs, attention explains capacity.** Lab-mean attention and lab-mean K correlate at **+0.90**. Florida's
  participants miss 24 % of 2-item trials; Münster's miss 6 %.
- **Within labs, attention carries more of the CDA.** When the CDA is regressed on both components, attention
  takes the larger unique share (β −0.16 ± 0.07 vs −0.11 ± 0.07).

**Reading.** Much of what this literature calls "working-memory capacity", and of what the CDA tracks, is
**how engaged the person was**. That is a different construct. It moves with sleep, motivation and stimulants
far more easily than storage does.

**Caveat.** The attention estimate assumes everyone can hold at least 2 items. Sample composition does not
explain the lab differences: lab-mean K against lab-mean age gives r = +0.05, against % female +0.25.

**Literature anchors.**
- The replication paper finds a within-lab CDA–K correlation of .13–.17, the same as ours. It does not analyse
  the between-lab component.
- Its Discussion offers anatomical gain as an untested explanation for between-person CDA variance.
- The storage-vs-attention question is open (Feldmann-Wüstefeld 2018; Hakim 2019). Graded attention
  fluctuations explain capacity differences (Adam 2015). Lapses co-occur with working-memory failures
  (deBettencourt 2019).

## Single-trial brain-reading has zero forecast value

`src/ideas/trial_forecast.py`.

**The test.** Predict whether *this* trial will be answered correctly, starting from the person's own accuracy
at this load. Then add the single-trial CDA, alpha lateralization, pre-stimulus alpha and eye drift, with
weights fitted on nine labs and scored on the tenth.

**The result.** Brier skill **+0.00 % [−0.02, +0.02]** over 144,831 trials. The CDA coefficient points the
published way (Adam et al. 2018: bigger CDA, more likely correct), but its forecast value is nil.

Adam et al. 2018 found poor-trial CDAs smaller when trials are binned and averaged. That holds here too, but
per trial it buys nothing.

**Reading.** This is the lisa-rtm point, transplanted. A trial-averaged effect can be real and still carry no
usable information per trial: a single CDA trial is ~98 % noise (coherent fraction 2 %). Any closed-loop or
real-time "working-memory state" product built on this signal starts from zero.

## Mental inertia: the brain stays in high-load mode for one trial

`src/ideas/hysteresis.py`.

- **CDA carry-over.** A 2-item trial right after a 6-item trial shows a CDA **0.16 µV larger** (p = 0.003),
  about half the normal 2→4-item step. The effect does not depend on whether the cued side repeats, so it is
  not a spatial memory trace. It is a carried-over control state.
- **Faster responses.** After harder trials people answer faster (−16 ms after 6 items, t = −8, all 10 labs).
  Trials run on a fixed clock, so this is not pacing.
- **A left-over hemisphere.** When the cued side repeats, alpha in the hemisphere about to work is higher
  before the array (t = +7.9). That looks like a recovery rebound from the previous trial, still visible ~4 s
  later.

No published study has looked at trial-history effects on the CDA or alpha lateralization. The nearest work is
on proactive interference (Hartshorne 2008; Lin & Luck 2012) and serial bias (Barbosa 2020).

**Status:** exploratory. The CDA effect survives a ×3 correction, not a family-wide one. The individual
"inertia" this suggests is untested as a trait.

## Lapses: flickering, not critical slowing down

`src/ideas/ews.py`.

**The test.** Do lapses (errors on 2-item trials) show early-warning signals in the 10 trials before them? The
comparison is against correct 2-item trials of the same person, with time-on-task trends removed.

- **RT variance rises before a lapse** (t = +4.1), and RTs get slightly *faster* (−7 ms).
- **Autocorrelation does not rise** (t = +1.6, n.s.).
- **No EEG precursor survives.** Before detrending, pre-stimulus alpha looked *lower* before lapses
  (t = −5.4, against O'Connell et al. 2009). That was a time-on-task confound: lapses cluster early in the
  session, when alpha is lower.

**Reading.** Lapses are announced by erratic, rushed responding (Esterman et al. 2013's "out of the zone"),
not by the critical-slowing-down signature of a tipping point. A behavioural early warning exists; a neural
one does not show up here.

## Do the eyes carry the capacity signal?

`src/ideas/eyes.py`.

**What the eyes do.** Both eye signals drift toward the remembered side with exactly the CDA's load profile:
they rise from 2 to 4 items, then plateau.

**What they predict.**
- Eye-tracker drift slope correlates with capacity at r = +0.21 within lab (n = 110, 4 labs). That is as
  strong as the CDA, and equally useless out of lab (CRPS skill ≈ 0).
- Controlling for the eyes leaves the CDA–capacity link unchanged (−0.19 → −0.18).

This replicates Kang & Woodman 2014: gaze bias scales with load, but it neither produces the CDA nor
predicts K.

**Reading.** A camera is as good a "capacity" readout as a 64-channel cap. That is because neither is much
good.

## Ideas from the literature not yet tested

1. **Aperiodic exponent during the delay as an independent capacity predictor.** Virtue-Griffiths et al. 2025
   report r ≈ 0.28 (n = 84). Testable from the epochs and the 180 s excerpt with specparam.
2. **Criticality (DFA of the alpha envelope, damping of the alpha mode) as a trait.** Read against the
   between- vs within-lab split. With 180 s per subject this is exploratory at best.
3. **GLM-HMM engagement states** (Ashwood et al. 2022). A model-based version of the attention split above,
   scored by Brier decomposition. It would turn the headline into a per-trial state estimate.
4. **Baseline-shift test** (Nikulin; van Dijk et al. 2010). Is the CDA produced by asymmetric alpha amplitude
   modulation? Needs the excerpt's low-frequency content, which our 0.5 Hz high-pass limits.
5. **Is the between-lab CDA difference hardware?** Most of it is behaviour (the labs differ in attention), but
   the ROI composition does differ by montage.

## Overnight results (2026-10-08)

- Engagement states (2-state HMM, `src/ideas/hmm_states.py`): no held-out gain over i.i.d. (+0.0009/trial); CDA unchanged in disengaged stretches (t = 0.04). Weakens the engagement headline: the 2-item attention score is not momentary engagement.
- Spectral traits (`src/ideas/spectral.py`): 1/f exponent, offset, alpha frequency and power, alpha DFA, HF floor: none predicts K within lab (|r| <= 0.11).
- Hysteresis replication (`src/ideas/extra.py`): same sign in 9/10 labs; significant in one random half (p < 0.001), not the other (p = 0.50). Suggestive.
- Capacity as a bifurcation: no variability peak at load = K (p = 0.12).
- Depletion: recent load does not predict errors beyond time on task (p = 0.085).

The data explorer dashboard reads `dashboard_data/*.json` (built by `src/dashboard_export.py`); its page is `dashboard/`.
