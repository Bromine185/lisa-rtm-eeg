"""Idea: a proper-scored single-trial brain readout. Does this trial's EEG/eye signal forecast whether THIS trial
will be answered correctly, beyond who the person is and how hard the trial is?

Reference forecast p0: the subject's leave-one-trial-out accuracy at this set size. Brain forecast:
logit p = logit p0 + sum_k b_k z_k, z_k = single-trial features z-scored within subject x set size
(cda, cda_late, alpha_lat, alpha_pre, heog). b is fitted on nine labs, scored on the tenth (leave one lab
out). Brier skill score BSS = 1 - Brier(brain) / Brier(p0), with a subject-level bootstrap CI; plus the
reliability (calibration) and resolution terms of the Brier decomposition (Murphy 1973) in 10 bins.

    python -I src/ideas/trial_forecast.py RESULTS_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd

FEATS = ["cda", "cda_late", "alpha_lat", "alpha_pre", "heog"]


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))


def fit_offset_logistic(X, y, off, iters=25, ridge=1e-3):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + X @ b)))
        g = X.T @ (y - p) - ridge * b; H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def murphy(p, y, bins=10):
    e = np.clip((p * bins).astype(int), 0, bins - 1); rel = res = 0.0; yb = y.mean()
    for k in range(bins):
        m = e == k
        if m.any():
            rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2; res += m.sum() * (y[m].mean() - yb) ** 2
    return rel / len(y), res / len(y)


def main():
    res = pathlib.Path(sys.argv[1]); out = res / "ideas"; out.mkdir(exist_ok=True)
    d = pd.read_csv(res / "trials.csv.gz")
    d = d[d.clean & d.correct.notna()].copy()
    g = d.groupby(["sub", "setsize"]).correct
    n, s = g.transform("size"), g.transform("sum")
    d["p0"] = ((s - d.correct) / (n - 1)).clip(0.01, 0.99)
    for f in FEATS:
        gg = d.groupby(["sub", "setsize"])[f]
        d[f + "_z"] = ((d[f] - gg.transform("mean")) / gg.transform("std")).fillna(0).clip(-5, 5)
    Z = [f + "_z" for f in FEATS]
    preds, coefs = [], {}
    for lab in sorted(d.lab.unique()):
        tr, te = d[d.lab != lab], d[d.lab == lab]
        b = fit_offset_logistic(tr[Z].values, tr.correct.values, logit(tr.p0.values))
        coefs[lab] = dict(zip(FEATS, b))
        te = te.assign(p1=1 / (1 + np.exp(-(logit(te.p0.values) + te[Z].values @ b))))
        preds.append(te[["sub", "lab", "setsize", "correct", "p0", "p1"]])
    P = pd.concat(preds)
    B0, B1 = (P.p0 - P.correct) ** 2, (P.p1 - P.correct) ** 2
    bss = 1 - B1.mean() / B0.mean()
    subs = P["sub"].unique(); rng = np.random.default_rng(0)
    per = pd.DataFrame({"b0": B0, "b1": B1, "sub": P["sub"]}).groupby("sub").sum()
    boot = [1 - per.b1.values[i].sum() / per.b0.values[i].sum() for i in rng.integers(0, len(per), (2000, len(per)))]
    out_d = {"n_trials": len(P), "n_subjects": len(subs), "bss": bss, "ci95": [np.percentile(boot, 2.5), np.percentile(boot, 97.5)]}
    print(f"{len(P)} trials, {len(subs)} subjects. Brier: reference {B0.mean():.5f}, brain {B1.mean():.5f}, "
          f"BSS {100*bss:+.2f} % [{100*out_d['ci95'][0]:+.2f}, {100*out_d['ci95'][1]:+.2f}]")
    for n_ in (2, 4, 6):
        m = P.setsize == n_; v = 1 - B1[m].mean() / B0[m].mean(); out_d[f"bss_ss{n_}"] = v
        print(f"   set size {n_}: BSS {100*v:+.2f} %")
    for name, col in (("reference", "p0"), ("brain", "p1")):
        rel, rs = murphy(P[col].values, P.correct.values); out_d[f"{name}_reliability"], out_d[f"{name}_resolution"] = rel, rs
        print(f"   {name:9s} reliability (calibration error) {rel:.5f}  resolution {rs:.5f}")
    cm = pd.DataFrame(coefs).T
    print("coefficients (log-odds per within-condition SD), mean over the ten fits:")
    print(cm.mean().round(3).to_string()); out_d["coef_mean"] = cm.mean().to_dict()
    # single features alone
    for f in FEATS:
        pr = []
        for lab in sorted(d.lab.unique()):
            tr, te = d[d.lab != lab], d[d.lab == lab]
            b = fit_offset_logistic(tr[[f + "_z"]].values, tr.correct.values, logit(tr.p0.values))
            pr.append(((1 / (1 + np.exp(-(logit(te.p0.values) + te[f + "_z"].values * b[0])))) - te.correct) ** 2)
        v = 1 - pd.concat(pr).mean() / B0.mean(); out_d[f"bss_only_{f}"] = v
        print(f"   {f:10s} alone: BSS {100*v:+.3f} %")
    json.dump(out_d, open(out / "trial_forecast.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
