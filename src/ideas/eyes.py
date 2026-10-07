"""Idea: do the eyes carry the capacity signal? Toward-cue eye drift during retention vs the CDA as predictors of K.

Per subject (clean trials): mean toward-cue HEOG (all labs, uV; scale differs by derivation, so every feature is
z-scored within lab) and toward-cue gaze shift (eye-tracker labs, native units), at set sizes 2/4/6, and the
load slope (6 - 2) / 4. Compared with the CDA on: correlation with K within lab, and leave-one-lab-out CRPS skill
of a Gaussian forecast on lab-z-scored features (the reference is a lab-z-scored climatology, so only
within-lab individual differences can score).

    python -I src/ideas/eyes.py RESULTS_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from predict import crps_gauss


def zlab(df, col):
    return df.groupby("lab")[col].transform(lambda v: (v - v.mean()) / v.std(ddof=1))


def lolo_skill(df, cols):
    """Leave-one-lab-out Gaussian forecast of within-lab-z K from within-lab-z features. CRPSS vs N(0, 1)."""
    out = []
    for lab in df.lab.unique():
        tr, te = df[df.lab != lab], df[df.lab == lab]
        A = np.c_[tr[cols].values, np.ones(len(tr))]; b = np.linalg.lstsq(A, tr.Kz.values, rcond=None)[0]
        s = np.std(tr.Kz.values - A @ b, ddof=len(cols) + 1)
        mu = np.c_[te[cols].values, np.ones(len(te))] @ b
        out.append(pd.DataFrame({"m": crps_gauss(mu, s, te.Kz.values), "c": crps_gauss(0.0, 1.0, te.Kz.values)}))
    o = pd.concat(out)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(o), (2000, len(o)))
    boot = 1 - o.m.values[idx].mean(1) / o.c.values[idx].mean(1)
    return float(1 - o.m.mean() / o.c.mean()), (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def main():
    res = pathlib.Path(sys.argv[1]); out = res / "ideas"; out.mkdir(exist_ok=True)
    d = pd.read_csv(res / "trials.csv.gz"); d = d[d.clean]
    S = pd.read_csv(res / "subjects.tsv", sep="\t"); S = S[~S.excluded].set_index("sub")
    f = d.groupby(["sub", "setsize"])[["heog", "gaze", "cda", "alpha_lat"]].mean().unstack()
    F = pd.DataFrame(index=f.index)
    for m in ("heog", "gaze", "cda", "alpha_lat"):
        F[f"{m}46"] = f[(m, 4)].add(f[(m, 6)]) / 2
        F[f"{m}_slope"] = (f[(m, 6)] - f[(m, 2)]) / 4
        F[f"{m}_4m2"] = f[(m, 4)] - f[(m, 2)]
    F = F.join(S[["lab", "K"]])
    for c in [c for c in F.columns if c not in ("lab", "K")]:
        F[c + "_z"] = zlab(F, c)
    F["Kz"] = zlab(F, "K")
    rows = {}
    for feat in ("cda46", "cda_slope", "alpha_lat46", "heog46", "heog_slope", "heog_4m2", "gaze46", "gaze_slope"):
        sub = F.dropna(subset=[feat + "_z", "Kz"])
        r, p = stats.pearsonr(sub[feat + "_z"], sub.Kz)
        sk, ci = lolo_skill(sub, [feat + "_z"])
        rows[feat] = {"n": len(sub), "labs": int(sub.lab.nunique()), "r_within_lab": r, "p": p, "crpss": sk, "ci95": ci}
        print(f"{feat:12s} n {len(sub):3d} ({sub.lab.nunique():2d} labs)  r within lab {r:+.3f} (p {p:.1e})  "
              f"LOLO CRPSS {sk:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    for combo in (["cda46_z", "heog46_z"], ["cda46_z", "heog46_z", "alpha_lat46_z"]):
        sub = F.dropna(subset=combo + ["Kz"]); sk, ci = lolo_skill(sub, combo)
        rows["+".join(combo)] = {"n": len(sub), "crpss": sk, "ci95": ci}
        print(f"{'+'.join(combo):40s} n {len(sub)}  LOLO CRPSS {sk:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    # how much of the CDA-K link runs through the eyes: partial correlation
    sub = F.dropna(subset=["cda46_z", "heog46_z", "Kz"])
    rk = stats.pearsonr(sub.cda46_z, sub.Kz)[0]
    res_c = sub.cda46_z - np.polyval(np.polyfit(sub.heog46_z, sub.cda46_z, 1), sub.heog46_z)
    res_k = sub.Kz - np.polyval(np.polyfit(sub.heog46_z, sub.Kz, 1), sub.heog46_z)
    rows["partial_cda_K_given_heog"] = float(stats.pearsonr(res_c, res_k)[0]); rows["cda_K"] = float(rk)
    rows["r_cda_heog"] = float(stats.pearsonr(sub.cda46_z, sub.heog46_z)[0])
    print(f"CDA-K within lab {rk:+.3f}; partial, controlling eye drift {rows['partial_cda_K_given_heog']:+.3f}; "
          f"r(CDA, eye drift) {rows['r_cda_heog']:+.3f}")
    json.dump(rows, open(out / "eyes.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
