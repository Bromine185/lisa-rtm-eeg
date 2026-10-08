"""Traits from 3 minutes of task EEG: aperiodic (1/f) exponent and offset, individual alpha frequency and power,
alpha long-range temporal correlations (DFA), and a high-frequency noise floor. Per subject, posterior channels
of the 180 s excerpt (average reference, 250 Hz).

Questions: (1) does any trait predict capacity K within lab (vs the CDA's r = -0.19)? (2) how much of each trait
is LAB (intraclass correlation) -- a multi-site warning; (3) can the lab be identified from these few numbers?
(4) does any trait explain the attention (lapse) component better than capacity?

    python -I src/ideas/spectral.py PREP_DIR RESULTS_DIR
"""
import json, pathlib, sys, warnings
import numpy as np, pandas as pd
from scipy import signal, stats
warnings.filterwarnings("ignore")
FS = 250


def dfa(x, scales):
    y = np.cumsum(x - x.mean()); F = []
    for s in scales:
        n = len(y) // s; seg = y[: n * s].reshape(n, s); t = np.arange(s)
        res = [np.sqrt(np.mean((g - np.polyval(np.polyfit(t, g, 1), t)) ** 2)) for g in seg]
        F.append(np.mean(res))
    return np.polyfit(np.log(scales), np.log(F), 1)[0]


def subject(prep, sub):
    from fooof import FOOOF
    j = json.loads((prep / f"{sub}.json").read_text()); A = np.load(prep / f"{sub}.npz")["sr"].astype(np.float64)
    names = j["sr_names"]
    post = [i for i, n in enumerate(names) if n.upper().startswith(("P", "O", "PO")) and not n.upper().startswith("FP")]
    if len(post) < 3:
        return None
    X = A[post]
    f, P = signal.welch(X, FS, nperseg=2 * FS); Pm = np.median(P, 0)
    fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    fm.fit(f, Pm, [2, 40])
    ap = fm.get_params("aperiodic_params")
    pk = fm.get_params("peak_params"); pk = np.atleast_2d(pk)
    alpha = pk[(pk[:, 0] >= 7) & (pk[:, 0] <= 14)] if pk.size else np.empty((0, 3))
    iaf, apow = (alpha[np.argmax(alpha[:, 1])][:2] if len(alpha) else (np.nan, np.nan))
    env = np.abs(signal.hilbert(signal.sosfiltfilt(signal.butter(4, [8, 13], "bandpass", fs=FS, output="sos"), X, axis=1), axis=1)).mean(0)
    env = signal.resample_poly(env, 1, 10)                                  # 25 Hz
    scales = np.unique(np.logspace(np.log10(25), np.log10(18 * 25), 12).astype(int))
    hf = (f >= 45) & (f <= 95) & (np.abs(f - 50) > 3) & (np.abs(f - 60) > 3)
    return {"sub": sub, "exponent": ap[1], "offset": ap[0], "iaf": iaf, "alpha_peak_pow": apow,
            "r2_fit": fm.r_squared_, "dfa_alpha": dfa(env, scales), "hf_floor": float(np.log10(np.median(Pm[hf]))),
            "n_post": len(post)}


def icc_lab(v, lab):
    g = pd.DataFrame({"v": v, "lab": lab}).dropna(); k = g.groupby("lab").size().mean()
    msb = g.groupby("lab").v.mean().var() * k; msw = g.groupby("lab").v.var().mean()
    return (msb - msw) / (msb + (k - 1) * msw)


def main():
    prep, res = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out = res / "ideas"
    S = pd.read_csv(res / "subjects.tsv", sep="\t"); S = S[~S.excluded].copy()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(2) as ex:
        rows = [r for r in ex.map(subject, [prep] * len(S), S["sub"]) if r]
    T = pd.DataFrame(rows).merge(S[["sub", "lab", "K", "K2", "cda4", "cda6"]], on="sub")
    T["attn"] = (T.K2 / 2).clip(0.05, 1); T["cda46"] = T[["cda4", "cda6"]].mean(axis=1)
    T.to_csv(out / "spectral_traits.tsv", sep="\t", index=False)
    feats = ["exponent", "offset", "iaf", "alpha_peak_pow", "dfa_alpha", "hf_floor"]
    summ = {"n": len(T), "median_fit_r2": float(T.r2_fit.median())}
    print(f"{len(T)} subjects; specparam fit r2 median {T.r2_fit.median():.3f}")
    W = T.copy()
    for c in feats + ["K", "attn", "cda46"]:
        W[c] = W[c] - W.groupby("lab")[c].transform("mean")
    for c in feats:
        k = W[c].notna()
        rK, pK = stats.pearsonr(W[c][k], W.K[k]); rA, pA = stats.pearsonr(W[c][k], W.attn[k]); rC = stats.pearsonr(W[c][k], W.cda46[k])[0]
        L = T.groupby("lab")[[c, "K"]].mean(); rL = stats.pearsonr(L[c], L.K)[0]
        icc = icc_lab(T[c], T.lab)
        summ[c] = {"r_K_within": rK, "p_K": pK, "r_attn_within": rA, "p_attn": pA, "r_cda_within": rC,
                   "r_K_between_labs": rL, "icc_lab": icc, "mean": float(T[c].mean())}
        print(f"  {c:15s} within-lab r with K {rK:+.3f} (p {pK:.3f}), with attention {rA:+.3f} (p {pA:.3f}), with CDA {rC:+.3f}"
              f" | between labs r(K) {rL:+.2f} | lab ICC {icc:.2f}")
    # can the lab be identified from these numbers?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    Z = T.dropna(subset=feats)
    acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Z[feats], Z.lab,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    summ["lab_decoding_accuracy"] = acc; summ["chance"] = float(Z.lab.value_counts(normalize=True).max())
    print(f"  lab identified from {len(feats)} numbers: {100*acc:.0f} % (chance {100*summ['chance']:.0f} %)")
    json.dump(summ, open(out / "spectral.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
