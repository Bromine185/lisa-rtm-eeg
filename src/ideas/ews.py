"""Idea: early-warning signals before attention lapses (critical slowing down, Scheffer et al. 2009).

A lapse = an error at set size 2 (below almost everyone's capacity). For every set-size-2 trial whose preceding
W trials lie in the same block, take the preceding window of three per-trial series, each residualized on set
size within subject: pre-array posterior alpha power (log, both hemispheres), RT, and single-trial CDA. Per
window: mean, variance, lag-1 autocorrelation. Per subject: (mean over lapse trials) - (mean over correct
set-size-2 trials); tested across subjects (>= 3 lapses). Critical slowing down predicts higher variance and
autocorrelation before lapses; O'Connell et al. (2009) predict higher alpha.
A within-subject shuffle null (labels permuted among the subject's set-size-2 trials) checks the pipeline.

    python -I src/ideas/ews.py RESULTS_DIR [W] [--detrend]   (--detrend: remove block means and within-block
                                                                linear trends first: time on task)
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

SERIES = ("alpha_pre", "rt", "cda")
DETREND = False


def window_feats(x):
    k = np.isfinite(x)
    if k.sum() < 0.6 * len(x):
        return np.nan, np.nan, np.nan
    v = x[k]; m = v.mean(); var = v.var()
    xx = np.where(k, x - m, np.nan); a, b = xx[1:], xx[:-1]; kk = np.isfinite(a) & np.isfinite(b)
    ac1 = np.sum(a[kk] * b[kk]) / max(np.sum(v ** 2 - 0) / len(v) * kk.sum(), 1e-12) if kk.sum() > 3 else np.nan
    return m, var, ac1


def main():
    global DETREND
    DETREND = "--detrend" in sys.argv; argv = [a for a in sys.argv if a != "--detrend"]
    res = pathlib.Path(argv[1]); W = int(argv[2]) if len(argv) > 2 else 10
    out = res / "ideas"; out.mkdir(exist_ok=True)
    d = pd.read_csv(res / "trials.csv.gz").sort_values(["sub", "trial"])
    for s in ("alpha_pre", "cda"):
        d.loc[~d.clean, s] = np.nan
    for s in SERIES:
        d[s] = d[s] - d.groupby(["sub", "setsize"])[s].transform("mean")
    if DETREND:
        # time on task: remove each block's mean and linear trend, per subject (lapses cluster early in a session)
        tr = d.trial.astype(float)
        for s in SERIES:
            def resid(y, t=tr):
                tt = t.loc[y.index].values; v = y.values; k = np.isfinite(v)
                if k.sum() <= 5:
                    return y
                return pd.Series(v - np.polyval(np.polyfit(tt[k], v[k], 1), tt), index=y.index)
            d[s] = d.groupby(["sub", "block"])[s].transform(resid)
    rng = np.random.default_rng(0)
    rows, null = [], []
    for sub, s in d.groupby("sub"):
        s = s.reset_index(drop=True)
        idx = [t for t in range(W, len(s)) if s.setsize[t] == 2 and s.outcome[t] != "none"
               and s.block[t - W] == s.block[t]]
        if not idx:
            continue
        err = np.array([s.correct[t] == 0 for t in idx])
        if err.sum() < 3 or (~err).sum() < 10:
            continue
        F = np.array([[f for ser in SERIES for f in window_feats(s[ser].values[t - W:t])] for t in idx])
        names = [f"{ser}:{k}" for ser in SERIES for k in ("mean", "var", "ac1")]
        r = {"sub": sub, "lab": s.lab[0], "n_lapse": int(err.sum())}
        for j, n in enumerate(names):
            r[n] = np.nanmean(F[err, j]) - np.nanmean(F[~err, j])
        rows.append(r)
        perm = rng.permutation(err)
        null.append({n: np.nanmean(F[perm, j]) - np.nanmean(F[~perm, j]) for j, n in enumerate(names)})
    R, N = pd.DataFrame(rows), pd.DataFrame(null)
    summ = {"W": W, "n_subjects": len(R), "median_lapses": float(R.n_lapse.median())}
    print(f"window {W} trials; {len(R)} subjects with >= 3 lapses at set size 2 (median {R.n_lapse.median():.0f})")
    for n in [c for c in R.columns if ":" in c]:
        v = R[n].dropna(); t, p = stats.ttest_1samp(v, 0); tn, pn = stats.ttest_1samp(N[n].dropna(), 0)
        dz = v.mean() / v.std()
        summ[n] = {"mean": v.mean(), "t": t, "p": p, "dz": dz, "null_t": tn}
        print(f"   {n:15s} lapse - correct {v.mean():+.4f}  dz {dz:+.3f}  t {t:+5.2f}  p {p:.3f}   | shuffled null t {tn:+5.2f}")
    summ["detrend"] = DETREND
    json.dump(summ, open(out / f"ews_W{W}{'_detrend' if DETREND else ''}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
