"""Idea: is the CDA a capacity marker or an engagement marker?

Cowan's K mixes two things: how many items fit (capacity) and how often the person is attending at all.
At set size 2 almost everyone is under capacity, so errors there are mostly lapses. With the attention
parameter of Rouder et al. (2008), a = (H2 - F2) (K2 = 2a when capacity >= 2), clipped to [0.05, 1]:
  attention a        fraction of trials on which the array was encoded at all
  capacity  Kc       mean of K4 and K6 divided by a (capacity on attended trials)
Then: which one does the CDA track, between labs and within labs?

    python -I src/ideas/engagement.py RESULTS_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import stats


def main():
    res = pathlib.Path(sys.argv[1]); out = res / "ideas"; out.mkdir(exist_ok=True)
    S = pd.read_csv(res / "subjects.tsv", sep="\t"); S = S[~S.excluded].copy()
    S["cda46"] = S[["cda4", "cda6"]].mean(axis=1)
    S["attn"] = (S.K2 / 2).clip(0.05, 1.0)
    S["Kc"] = S.K / S.attn
    S["lapse"] = 1 - S.attn
    lab = S.groupby("lab")[["cda46", "K", "attn", "Kc", "acc"]].mean()
    res_d = {"lab_level": {}, "within_lab": {}, "n": len(S)}
    print("between labs (10 lab means):")
    for v in ("K", "attn", "Kc"):
        r, p = stats.pearsonr(lab.cda46, lab[v]); res_d["lab_level"][v] = {"r": r, "p": p}
        print(f"   r(CDA, {v:4s}) = {r:+.3f} (p {p:.3f})")
    r, p = stats.pearsonr(lab.attn, lab.K); res_d["lab_level"]["attn_K"] = {"r": r, "p": p}
    print(f"   r(attention, K) = {r:+.3f} (p {p:.3f})")
    print(lab.round(3).sort_values("cda46").to_string())
    W = S.copy()
    for v in ("cda46", "K", "attn", "Kc"):
        W[v] = W[v] - W.groupby("lab")[v].transform("mean")
    print("within labs (lab-centred, n = %d):" % len(W))
    for v in ("K", "attn", "Kc"):
        r, p = stats.pearsonr(W.cda46, W[v]); res_d["within_lab"][v] = {"r": r, "p": p}
        print(f"   r(CDA, {v:4s}) = {r:+.3f} (p {p:.1e})")
    # unique contributions: CDA ~ attn + Kc (standardized), within lab
    Z = (W[["cda46", "attn", "Kc"]] - W[["cda46", "attn", "Kc"]].mean()) / W[["cda46", "attn", "Kc"]].std()
    A = np.c_[Z.attn, Z.Kc, np.ones(len(Z))]; b, *_ = np.linalg.lstsq(A, Z.cda46, rcond=None)
    e = Z.cda46 - A @ b; se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * e.var(ddof=3))
    res_d["within_lab"]["beta_attn"], res_d["within_lab"]["beta_Kc"] = (b[0], se[0]), (b[1], se[1])
    print(f"   standardized CDA ~ attention {b[0]:+.3f} ± {se[0]:.3f},  capacity-on-attended-trials {b[1]:+.3f} ± {se[1]:.3f}")
    r, p = stats.pearsonr(W.attn, W.K); print(f"   r(attention, K) within lab = {r:+.3f}")
    res_d["within_lab"]["attn_K"] = {"r": r, "p": p}
    json.dump(res_d, open(out / "engagement.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
