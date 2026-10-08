"""Engagement states: a 2-state hidden Markov model of each person's 720-trial accuracy sequence.

Emissions: correct ~ Bernoulli(p[state, set size]); state 0 = engaged (higher accuracy averaged over set sizes).
Fit by Baum-Welch per subject; posterior state per trial. Then:
  - is accuracy state-structured? held-out log-likelihood (last 20 % of trials) vs an i.i.d. model
  - fraction of trials disengaged, mean dwell time
  - per-trial EEG/eye by state at matched set size: CDA, alpha lateralization, pre-array alpha, HEOG
  - K in the engaged state only, and the CDA-K correlations redone with it (within and between labs)
Ashwood et al. 2022 (GLM-HMM) is the model family; this is its smallest member.

    python -I src/ideas/hmm_states.py RESULTS_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats


def fit(y, ss, iters=200, seed=0):
    """y: 0/1 with NaN for missing; ss: index 0..2. Returns params and posterior P(state)."""
    T = len(y); rng = np.random.default_rng(seed)
    p = np.array([[0.95, 0.85, 0.75], [0.6, 0.5, 0.45]]) + rng.normal(0, 0.02, (2, 3))
    A = np.array([[0.95, 0.05], [0.15, 0.85]]); pi = np.array([0.8, 0.2]); ll_old = -np.inf
    obs = np.isfinite(y)
    for _ in range(iters):
        B = np.ones((T, 2))
        for k in range(2):
            pk = np.clip(p[k, ss], 1e-4, 1 - 1e-4)
            B[obs, k] = np.where(y[obs] == 1, pk[obs], 1 - pk[obs])
        a = np.zeros((T, 2)); c = np.zeros(T)
        a[0] = pi * B[0]; c[0] = a[0].sum(); a[0] /= c[0]
        for t in range(1, T):
            a[t] = (a[t - 1] @ A) * B[t]; c[t] = a[t].sum(); a[t] /= c[t]
        b = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            b[t] = (A @ (B[t + 1] * b[t + 1])) / c[t + 1]
        g = a * b; g /= g.sum(1, keepdims=True)
        xi = (a[:-1, :, None] * A[None] * (B[1:] * b[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(0) / xi.sum((0, 2))[:, None]; pi = g[0]
        for k in range(2):
            for j in range(3):
                m = obs & (ss == j)
                p[k, j] = (g[m, k] * y[m]).sum() / max(g[m, k].sum(), 1e-9)
        ll = np.log(c).sum()
        if ll - ll_old < 1e-6:
            break
        ll_old = ll
    if p[0].mean() < p[1].mean():                       # state 0 = engaged
        p, A, pi, g = p[::-1], A[::-1, ::-1], pi[::-1], g[:, ::-1]
    return p, A, pi, g, ll


def loglik(y, ss, p, A, pi):
    T = len(y); obs = np.isfinite(y); B = np.ones((T, 2))
    for k in range(2):
        pk = np.clip(p[k, ss], 1e-4, 1 - 1e-4); B[obs, k] = np.where(y[obs] == 1, pk[obs], 1 - pk[obs])
    a = pi * B[0]; ll = np.log(a.sum()); a /= a.sum()
    for t in range(1, T):
        a = (a @ A) * B[t]; s = a.sum(); ll += np.log(s); a /= s
    return ll


def main():
    res = pathlib.Path(sys.argv[1]); out = res / "ideas"; out.mkdir(exist_ok=True)
    d = pd.read_csv(res / "trials.csv.gz").sort_values(["sub", "trial"])
    S = pd.read_csv(res / "subjects.tsv", sep="\t"); S = S[~S.excluded].set_index("sub")
    rows, trial_states = [], []
    for sub, s in d.groupby("sub"):
        s = s.reset_index(drop=True)
        y = s.correct.values.astype(float); ss = (s.setsize.values // 2 - 1).astype(int)
        n = len(y); cut = int(0.8 * n)
        best = max((fit(y[:cut], ss[:cut], seed=k) for k in range(3)), key=lambda r: r[4])
        p, A, pi, _, _ = best
        ll_hmm = loglik(y[cut:], ss[cut:], p, A, best[3][-1] @ A)
        pi_iid = np.array([[np.nanmean(y[:cut][ss[:cut] == j]) for j in range(3)]])
        yt, st = y[cut:], ss[cut:]; k = np.isfinite(yt)
        ll_iid = np.sum(np.log(np.where(yt[k] == 1, pi_iid[0, st[k]], 1 - pi_iid[0, st[k]])))
        p, A, pi, g, _ = max((fit(y, ss, seed=k) for k in range(3)), key=lambda r: r[4])
        dis = g[:, 1] > 0.5
        r = {"sub": sub, "lab": s.lab[0], "frac_dis": dis.mean(), "dwell_dis": 1 / max(A[1, 0], 1e-3),
             "dwell_eng": 1 / max(A[0, 1], 1e-3), "p_eng": p[0].tolist(), "p_dis": p[1].tolist(),
             "dll_per_trial": (ll_hmm - ll_iid) / max(k.sum(), 1), "sep": float(p[0].mean() - p[1].mean())}
        # K in the engaged state: hits/FA from engaged trials
        for nset in (4, 6):
            e = (~dis) & (s.setsize.values == nset); o = s.outcome.values
            h = (o[e] == "hit").sum() / max(((o[e] == "hit") | (o[e] == "miss")).sum(), 1)
            f = (o[e] == "fa").sum() / max(((o[e] == "fa") | (o[e] == "cr")).sum(), 1)
            r[f"Keng{nset}"] = nset * (h - f)
        cl = s.clean.values
        for m in ("cda", "alpha_lat", "alpha_pre", "heog", "rt"):
            diffs = []
            for nset in (2, 4, 6):
                a_ = s[m].values[(s.setsize.values == nset) & cl & dis]; b_ = s[m].values[(s.setsize.values == nset) & cl & ~dis]
                if len(a_) >= 5 and len(b_) >= 5:
                    diffs.append(np.nanmean(a_) - np.nanmean(b_))
            r[f"{m}_dis_minus_eng"] = np.mean(diffs) if diffs else np.nan
        rows.append(r)
        trial_states.append(pd.DataFrame({"sub": sub, "trial": s.trial, "p_dis": g[:, 1]}))
    R = pd.DataFrame(rows); R.to_csv(out / "hmm_subjects.tsv", sep="\t", index=False)
    pd.concat(trial_states).to_csv(out / "hmm_trial_states.csv.gz", index=False)
    summ = {"n": len(R)}
    t, pv = stats.ttest_1samp(R.dll_per_trial, 0)
    summ["heldout_dll_per_trial"] = {"mean": R.dll_per_trial.mean(), "t": t, "p": pv, "frac_subjects_better": float((R.dll_per_trial > 0).mean())}
    print(f"{len(R)} subjects. Held-out log-lik gain of the 2-state model over i.i.d.: {R.dll_per_trial.mean():+.4f}/trial "
          f"(t {t:+.2f}, better in {100*(R.dll_per_trial>0).mean():.0f} % of people)")
    print(f"disengaged fraction: median {R.frac_dis.median():.2f}; dwell disengaged {R.dwell_dis.median():.1f} trials, "
          f"engaged {R.dwell_eng.median():.1f}; accuracy engaged {np.mean(np.stack(R.p_eng), 0).round(3)} vs "
          f"disengaged {np.mean(np.stack(R.p_dis), 0).round(3)}")
    for m in ("cda", "alpha_lat", "alpha_pre", "heog", "rt"):
        v = R[f"{m}_dis_minus_eng"].dropna(); t, pv = stats.ttest_1samp(v, 0)
        summ[f"{m}_dis_minus_eng"] = {"mean": v.mean(), "se": v.sem(), "t": t, "p": pv, "n": len(v)}
        print(f"   {m:10s} disengaged - engaged (matched set size): {v.mean():+.4f} ± {v.sem():.4f}  t {t:+.2f}  p {pv:.1e}  n {len(v)}")
    R = R.set_index("sub").join(S[["K", "cda4", "cda6", "K2"]])
    R["Keng"] = R[["Keng4", "Keng6"]].mean(axis=1); R["cda46"] = R[["cda4", "cda6"]].mean(axis=1)
    for v in ("frac_dis", "Keng", "K"):
        L = R.groupby("lab")[[v, "cda46"]].mean(); rl = stats.pearsonr(L[v], L.cda46)[0]
        W = R[[v, "cda46"]] - R.groupby("lab")[[v, "cda46"]].transform("mean"); rw, pw = stats.pearsonr(W[v], W.cda46)
        summ[f"cda_vs_{v}"] = {"r_between_labs": rl, "r_within_lab": rw, "p_within": pw}
        print(f"   CDA vs {v:8s}: between labs r {rl:+.3f}; within labs r {rw:+.3f} (p {pw:.1e})")
    L = R.groupby("lab")[["frac_dis", "K"]].mean(); summ["lab_K_vs_frac_dis"] = stats.pearsonr(L.frac_dis, L.K)[0]
    print(f"   lab-mean K vs lab-mean disengaged fraction: r {summ['lab_K_vs_frac_dis']:+.3f}")
    json.dump(summ, open(out / "hmm.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
