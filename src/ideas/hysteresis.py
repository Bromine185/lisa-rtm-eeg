"""Idea: load hysteresis. Does the previous trial's memory load shift the current trial, in the brain and in behaviour?

An attractor (or a synaptic trace that outlives the trial) predicts that the same current load produces a
different CDA / alpha lateralization / accuracy depending on what came before. Two-stage summary statistics:
per subject, OLS of each measure on current-load dummies + previous-load dummies + previous-side-repeat +
previous-correct, on clean trials whose predecessor is in the same block; then coefficients are tested across
subjects (t), and within each lab.

    python -I src/ideas/hysteresis.py RESULTS_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

MEASURES = {"cda": "single-trial CDA (uV, 300-900 ms)", "cda_late": "late CDA (600-900 ms)",
            "alpha_lat": "alpha lateralization (log contra/ipsi)", "correct": "accuracy", "rt": "RT (s)",
            "alpha_pre_lat": "pre-array alpha lateralization", "heog": "toward-cue HEOG (uV)"}


def design(d):
    X = pd.DataFrame({"const": 1.0,
                      "cur4": (d.setsize == 4).astype(float), "cur6": (d.setsize == 6).astype(float),
                      "prev4": (d.prev_ss == 4).astype(float), "prev6": (d.prev_ss == 6).astype(float),
                      "same_side": d.same_side.astype(float), "prev_correct": d.prev_correct.astype(float)})
    return X


def main():
    res = pathlib.Path(sys.argv[1]); out = res / "ideas"; out.mkdir(exist_ok=True)
    df = pd.read_csv(res / "trials.csv.gz")
    df = df.sort_values(["sub", "trial"])
    g = df.groupby("sub")
    df["prev_ss"] = g.setsize.shift(1); df["prev_side"] = g.side.shift(1); df["prev_block"] = g.block.shift(1)
    df["prev_correct"] = g.correct.shift(1)
    df["same_side"] = df.side == df.prev_side
    d = df[df.clean & (df.prev_block == df.block) & df.prev_ss.notna() & df.prev_correct.notna() & df.correct.notna()].copy()
    rows = []
    for sub, s in d.groupby("sub"):
        X = design(s)
        r = {"sub": sub, "lab": s.lab.iloc[0], "n": len(s)}
        for m in MEASURES:
            y = s[m].values; k = np.isfinite(y)
            beta = np.linalg.lstsq(X.values[k], y[k], rcond=None)[0]
            for name, b in zip(X.columns, beta):
                r[f"{m}:{name}"] = b
        # the direct hysteresis contrast: current load 4, preceded by 6 vs by 2
        c4 = s[s.setsize == 4]
        for m in ("cda", "alpha_lat", "correct"):
            r[f"{m}:4after6-4after2"] = c4[c4.prev_ss == 6][m].mean() - c4[c4.prev_ss == 2][m].mean()
        rows.append(r)
    B = pd.DataFrame(rows); B.to_csv(out / "hysteresis_coefs.tsv", sep="\t", index=False)
    summ = {}
    print(f"{len(B)} subjects, {int(B.n.sum())} trials\n")
    for m, label in MEASURES.items():
        print(label)
        for term in ("cur4", "cur6", "prev4", "prev6", "same_side", "prev_correct"):
            v = B[f"{m}:{term}"].dropna(); t, p = stats.ttest_1samp(v, 0)
            labs_sig = (B.groupby("lab")[f"{m}:{term}"].mean() * np.sign(v.mean()) > 0).sum()
            summ[f"{m}:{term}"] = {"mean": v.mean(), "se": v.sem(), "t": t, "p": p, "labs_same_sign": int(labs_sig)}
            print(f"   {term:13s} {v.mean():+.4f} ± {v.sem():.4f}  t {t:+6.2f}  p {p:.1e}  labs same sign {labs_sig}/10")
        # previous-load effect relative to the current-load effect (the 2 -> 4 step)
        ratio = B[f"{m}:prev6"].mean() / B[f"{m}:cur4"].mean() if B[f"{m}:cur4"].mean() != 0 else np.nan
        summ[f"{m}:prev6_over_cur4"] = ratio
        print(f"   previous-load-6 effect as a fraction of the current 2->4 effect: {ratio:+.2f}")
    for m in ("cda", "alpha_lat", "correct"):
        v = B[f"{m}:4after6-4after2"].dropna(); t, p = stats.ttest_1samp(v, 0)
        summ[f"{m}:4after6-4after2"] = {"mean": v.mean(), "se": v.sem(), "t": t, "p": p}
        print(f"{m}: load 4 after 6 minus load 4 after 2 = {v.mean():+.4f} ± {v.sem():.4f} (t {t:+.2f}, p {p:.1e})")
    json.dump(summ, open(out / "hysteresis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
