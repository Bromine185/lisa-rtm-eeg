"""Three remaining tests from the dynamical-systems list.

1. Hysteresis replication: the CDA carry-over (2-item trial after a 6-item trial vs after a 2-item trial) in two
   random halves of the people, and in each lab.
2. Capacity as a bifurcation: does single-trial CDA variability (SD across clean trials, minus the person's
   pre-stimulus noise SD) peak where load meets the person's own capacity (load - K near 0)?
3. Depletion vs time: does the recent load (mean set size of the previous 10 trials) predict an error beyond
   time on task, set size and block? Per-person logistic regression, coefficients tested across people.

    python -I src/ideas/extra.py RESULTS_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats


def logit_fit(X, y, iters=30, ridge=1e-2):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b); b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def main():
    res = pathlib.Path(sys.argv[1]); out = res / "ideas"
    d = pd.read_csv(res / "trials.csv.gz").sort_values(["sub", "trial"])
    S = pd.read_csv(res / "subjects.tsv", sep="\t"); S = S[~S.excluded].set_index("sub")
    summ = {}
    # 1 -------------------------------------------------------------------------------------------------
    g = d.groupby("sub"); d["prev_ss"] = g.setsize.shift(1); d["prev_block"] = g.block.shift(1)
    x = d[d.clean & (d.prev_block == d.block) & (d.setsize == 2)]
    eff = x.groupby("sub").apply(lambda s: s[s.prev_ss == 6].cda.mean() - s[s.prev_ss == 2].cda.mean(), include_groups=False).dropna()
    rng = np.random.default_rng(42); subs = eff.index.values.copy(); rng.shuffle(subs)
    halves = [subs[: len(subs) // 2], subs[len(subs) // 2:]]
    summ["hysteresis_halves"] = []
    for i, h in enumerate(halves):
        v = eff.loc[h]; t, p = stats.ttest_1samp(v, 0)
        summ["hysteresis_halves"].append({"half": i + 1, "n": len(v), "mean": v.mean(), "t": t, "p": p})
        print(f"hysteresis, half {i+1}: {v.mean():+.3f} uV (t {t:+.2f}, p {p:.3f}, n {len(v)})")
    lab = S.lab.reindex(eff.index); bl = eff.groupby(lab).mean()
    summ["hysteresis_labs_negative"] = int((bl < 0).sum()); print(f"   labs with a larger CDA after 6 items: {(bl < 0).sum()}/10")
    # 2 -------------------------------------------------------------------------------------------------
    c = d[d.clean]
    sd = c.groupby(["sub", "setsize"]).cda.std().rename("sd").reset_index()
    sd = sd.join(S[["K"]], on="sub").dropna()
    sd["dist"] = sd.setsize - sd.K
    sd["sd_rel"] = sd.sd - sd.groupby("sub").sd.transform("mean")             # within-person
    bins = [-10, -2, -1, 0, 1, 2, 10]
    sd["bin"] = pd.cut(sd.dist, bins)
    tab = sd.groupby("bin", observed=True).sd_rel.agg(["mean", "sem", "size"])
    print("single-trial CDA SD (within person) by load minus own capacity:"); print(tab.round(3).to_string())
    summ["bifurcation"] = [{"bin": str(k), "mean": r["mean"], "sem": r["sem"], "n": int(r["size"])} for k, r in tab.iterrows()]
    q = sd.assign(d2=sd.dist ** 2)
    rows = []
    for sub, s in q.groupby("sub"):
        if len(s) == 3:
            A = np.c_[np.ones(3), s.dist, s.d2]; rows.append(np.linalg.lstsq(A, s.sd.values, rcond=None)[0][2])
    t, p = stats.ttest_1samp(rows, 0); summ["bifurcation_quadratic"] = {"mean": float(np.mean(rows)), "t": t, "p": p}
    print(f"   curvature of SD in (load - K), per person: {np.mean(rows):+.3f} (t {t:+.2f}, p {p:.3f}); negative = peak near capacity")
    # 3 -------------------------------------------------------------------------------------------------
    d["recent_load"] = g.setsize.transform(lambda s: s.shift(1).rolling(10, min_periods=10).mean())
    coefs = []
    for sub, s in d[d.correct.notna() & d.recent_load.notna()].groupby("sub"):
        X = np.c_[np.ones(len(s)), (s.setsize == 4), (s.setsize == 6), (s.t_min - s.t_min.mean()) / s.t_min.std(),
                  (s.recent_load - s.recent_load.mean()) / s.recent_load.std()]
        coefs.append(logit_fit(X.astype(float), (1 - s.correct.values)))
    B = np.array(coefs)
    for j, name in ((3, "time on task"), (4, "recent load (last 10 trials)")):
        t, p = stats.ttest_1samp(B[:, j], 0); summ[f"error_logit_{name}"] = {"mean": B[:, j].mean(), "t": t, "p": p}
        print(f"error log-odds per SD of {name}: {B[:, j].mean():+.4f} (t {t:+.2f}, p {p:.3f})")
    json.dump(summ, open(out / "extra.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
