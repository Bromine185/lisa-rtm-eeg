"""Experiment C: does the CDA predict working-memory capacity in a lab it has never seen? Scored as lisa-rtm scores
a sampler: a proper score (CRPS) against a reference forecast, PIT calibration, and the spread of the point
forecast relative to the truth (regression to the mean).

Leave-one-lab-out: fit on nine labs, forecast every subject of the tenth. Outcome K = mean of Cowan's K at set
sizes 4 and 6. Forecasts are Gaussian, so CRPS is closed-form (Gneiting & Raftery 2007):
    CRPS(N(mu, s^2), k) = s [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (k - mu) / s.
  clim        N(mean K, sd K) of the nine training labs: the forecast that knows nothing about the subject
  lab_clim    N(mean K of the held-out lab's own subjects except this one, pooled sd) -- an oracle on lab level,
              the ceiling for what knowing "which lab" buys
  cda         N(a + b x, residual sd), x = the subject's CDA feature, OLS on the nine labs
  cda_labc    the same with x centred within each lab (the held-out lab's centre uses its CDA only, never K)
Point-forecast spread: sd(mu) / sd(K). A calibrated forecast distribution has a flat PIT; the point forecast's
sd ratio equals |r|: it compresses the individual differences it is meant to measure, as every lisa-rtm arm
compresses per-utterance level.

    python -I src/predict.py RESULTS_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

FEATURES = {"cda4": "CDA set size 4", "cda46": "CDA mean of set sizes 4 and 6", "cda4m2": "CDA increase 2 -> 4",
            "cda_slope": "CDA slope over set size"}


def crps_gauss(mu, s, k):
    z = (k - mu) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def features(df):
    df = df.copy()
    df["cda46"] = df[["cda4", "cda6"]].mean(axis=1)
    df["cda4m2"] = df["cda4"] - df["cda2"]
    df["cda_slope"] = (df["cda6"] - df["cda2"]) / 4
    for f in FEATURES:
        df[f + "_labc"] = df[f] - df.groupby("lab")[f].transform("mean")
    return df


def lolo(df, feat):
    rows = []
    for lab in sorted(df.lab.unique()):
        tr, te = df[df.lab != lab], df[df.lab == lab]
        k_tr = tr.K.values
        fc = {"clim": (np.full(len(te), k_tr.mean()), np.full(len(te), k_tr.std(ddof=1)))}
        resid_sd = np.sqrt(np.mean([g.K.var(ddof=1) for _, g in tr.groupby("lab")]))
        fc["lab_clim"] = (np.array([te.K.drop(i).mean() for i in te.index]), np.full(len(te), resid_sd))
        for name, col in (("cda", feat), ("cda_labc", feat + "_labc")):
            b, a = np.polyfit(tr[col].values, k_tr, 1)
            s = np.std(k_tr - (a + b * tr[col].values), ddof=2)
            fc[name] = (a + b * te[col].values, np.full(len(te), s))
        for name, (mu, s) in fc.items():
            for i, (m, sd, k, sub) in enumerate(zip(mu, s, te.K.values, te["sub"].values)):
                rows.append({"lab": lab, "sub": sub, "model": name, "mu": m, "sd": sd, "K": k,
                             "crps": crps_gauss(m, sd, k), "pit": stats.norm.cdf((k - m) / sd)})
    return pd.DataFrame(rows)


def boot_ci(x, y, n=2000, seed=0):
    """CI of 1 - mean(x)/mean(y) over subjects (paired bootstrap)."""
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (n, len(x)))
    v = 1 - x[idx].mean(axis=1) / y[idx].mean(axis=1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    res = pathlib.Path(sys.argv[1])
    S = pd.read_csv(res / "subjects.tsv", sep="\t")
    S["heog_resid"] = S[["heog2", "heog4", "heog6"]].abs().max(axis=1)
    if "--strict-eye" in sys.argv:          # robustness: drop the quarter of subjects with the largest residual eye signal
        S = S[S.heog_resid <= S[~S.excluded].heog_resid.quantile(0.75)]
        res = res / "strict_eye"; res.mkdir(exist_ok=True)
    df = features(S[~S.excluded].dropna(subset=["cda2", "cda4", "cda6", "K"]))
    out = {"n": len(df), "labs": int(df.lab.nunique()), "features": {}}
    # reliability of the CDA (split-half, Spearman-Brown) and correlations
    for f, label in FEATURES.items():
        r, p = stats.pearsonr(df[f], df.K); rs = stats.spearmanr(df[f], df.K)[0]
        rl = stats.pearsonr(df[f + "_labc"], df.K - df.groupby("lab").K.transform("mean"))[0]
        P = lolo(df, f)
        agg = {}
        for m, g in P.groupby("model"):
            agg[m] = {"crps": float(g.crps.mean()), "pit_cover50": float(((g.pit > .25) & (g.pit < .75)).mean()),
                      "pit_cover90": float(((g.pit > .05) & (g.pit < .95)).mean()),
                      "pit_lo10": float((g.pit < .1).mean()), "pit_hi10": float((g.pit > .9).mean()),
                      "sd_ratio_point": float(g.mu.std() / g.K.std()),
                      "calib_slope": float(np.polyfit(g.mu, g.K, 1)[0]) if g.mu.std() > 0 else None}
        clim = P[P.model == "clim"].sort_values("sub").crps.values
        for m in ("cda", "cda_labc", "lab_clim"):
            x = P[P.model == m].sort_values("sub").crps.values
            agg[m]["crpss_vs_clim"] = float(1 - x.mean() / clim.mean()); agg[m]["crpss_ci95"] = boot_ci(x, clim)
        per_lab = {lab: float(stats.pearsonr(g[f], g.K)[0]) for lab, g in df.groupby("lab") if len(g) > 4}
        out["features"][f] = {"label": label, "r": r, "p": p, "spearman": rs, "r_within_lab": rl,
                              "r_per_lab": per_lab, "scores": agg}
        P.to_csv(res / f"lolo_{f}.tsv", sep="\t", index=False)
    # split-half reliability of the CDA features
    rel = {}
    for n in (2, 4, 6):
        a, b = df[f"cda{n}_odd"], df[f"cda{n}_even"]
        r = stats.pearsonr(a, b)[0]; rel[f"cda{n}"] = {"split_half_r": r, "spearman_brown": 2 * r / (1 + r)}
    a = df[["cda4_odd", "cda6_odd"]].mean(axis=1) - df["cda2_odd"]; b = df[["cda4_even", "cda6_even"]].mean(axis=1) - df["cda2_even"]
    r = stats.pearsonr(a, b)[0]; rel["cda46_minus_2"] = {"split_half_r": r, "spearman_brown": 2 * r / (1 + r)}
    out["reliability"] = rel
    out["set_size"] = {f"cda{n}": {"mean": float(df[f"cda{n}"].mean()), "se": float(df[f"cda{n}"].sem())} for n in (2, 4, 6)}
    out["set_size"].update({f"K{n}": {"mean": float(df[f"K{n}"].mean()), "se": float(df[f"K{n}"].sem())} for n in (2, 4, 6)})
    t46 = stats.ttest_rel(df.cda4, df.cda2); out["cda4_vs_cda2"] = {"t": float(t46.statistic), "p": float(t46.pvalue)}
    json.dump(out, open(res / "predict.json", "w"), indent=1, default=float)
    for f, v in out["features"].items():
        s = v["scores"]
        print(f"{f:10s} r {v['r']:+.3f} (within-lab {v['r_within_lab']:+.3f})  CRPS clim {s['clim']['crps']:.3f}  "
              f"cda {s['cda']['crps']:.3f} (CRPSS {s['cda']['crpss_vs_clim']:+.3f} {np.round(s['cda']['crpss_ci95'], 3)})  "
              f"labc {s['cda_labc']['crps']:.3f} (CRPSS {s['cda_labc']['crpss_vs_clim']:+.3f})  lab oracle CRPSS "
              f"{s['lab_clim']['crpss_vs_clim']:+.3f}  point sd ratio {s['cda']['sd_ratio_point']:.3f}  "
              f"PIT90 {s['cda']['pit_cover90']:.2f}")
    print(json.dumps(out["reliability"], indent=1))


if __name__ == "__main__":
    main()
