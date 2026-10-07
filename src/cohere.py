"""Experiment B: the lisa-rtm diagnostics on the CDA itself.

The trial-averaged CDA is a conditional mean: the minimiser of squared error for predicting a single trial's
lateralized signal from its condition (set size). lisa-rtm's identity for any point predictor w of a target y,

    ||y - w||^2 = E (1 + p - 2c),    E = ||y||^2,  p = ||w||^2 / E,  c = Re<y, w> / E  (coherent fraction),

then says how much of the single-trial signal the ERP can carry. With w the leave-one-out condition mean (so the
trial is never in its own predictor), c is the phase-locked (evoked) fraction of the trial's energy, per
frequency band: a predictability spectrum. lisa-rtm's second finding was that an incoherent band can still have
a predictable envelope; here that is the induced alpha lateralization, scored the same way on log band power.

    python -I src/cohere.py PREP_DIR RESULTS_DIR      (needs RESULTS_DIR/subjects.tsv from src/cda.py)
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import signal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cda import T, ROI, STEP_THR, win, step_max

FS = 250
W = win(0.3, 0.9)                                      # the CDA window, 150 samples
BANDS = {"0-2": (0, 2.5), "2-4": (2.5, 4.5), "4-8": (4.5, 8.5), "8-13": (8.5, 13.5), "13-30": (13.5, 30.5)}
ENV_BANDS = {"theta 4-8": (4, 8), "alpha 8-13": (8, 13), "beta 13-30": (13, 30)}


def loo_mean(Y, groups):
    """Leave-one-out mean within each group (set size). Y: (trials, ...)."""
    out = np.empty_like(Y)
    for g in np.unique(groups):
        k = groups == g; n = k.sum(); s = Y[k].sum(0)
        out[k] = (s[None] - Y[k]) / max(n - 1, 1)
    return out


def coherence_terms(Y, Wh):
    """Pooled over trials: E, cross, P per frequency bin. Y, Wh: (trials, n_freq) complex."""
    return (np.abs(Y) ** 2).sum(0), (Y * np.conj(Wh)).real.sum(0), (np.abs(Wh) ** 2).sum(0)


def subject(prep, sub, heog_kind):
    j = json.loads((prep / f"{sub}.json").read_text()); d = np.load(prep / f"{sub}.npz")
    X, H = d["X"].astype(np.float64), d["H"].astype(np.float64)
    names = j["lat_names"]; tr = pd.DataFrame(j["trials"])
    X -= X[:, :, win(-0.2, 0)].mean(-1, keepdims=True); H -= H[:, win(-0.2, 0)].mean(-1, keepdims=True)
    side = tr["side"].to_numpy(dtype=str); ss = tr["setsize"].to_numpy(int)
    w = win(-0.2, 1.0)
    ok = (step_max(H[:, w]) < STEP_THR[heog_kind]) & (np.ptp(X[:, :, w], -1).max(1) < 150) & np.isin(side, ["left", "right"])
    pairs = [(a, b) for a, b in ROI if a in names and b in names]
    il = [names.index(a) for a, _ in pairs]; ir = [names.index(b) for _, b in pairs]
    L, R = X[:, il].mean(1), X[:, ir].mean(1)                                 # ROI average per hemisphere
    contra = np.where(side[:, None] == "left", R, L); ipsi = np.where(side[:, None] == "left", L, R)
    lat = (contra - ipsi)[ok]; g = ss[ok]
    # waveform coherent fraction, time domain and per band (rfft of the 0.3-0.9 s window, no taper: DC kept)
    y = lat[:, W]; wh = loo_mean(y, g)
    E, C, P = (y ** 2).sum(), (y * wh).sum(), (wh ** 2).sum()
    out = {"sub": sub, "n": int(ok.sum()), "c_time": C / E, "p_time": P / E, "kappa_time": C / P,
           "identity_err": float(abs(((y - wh) ** 2).sum() / E - (1 + P / E - 2 * C / E)))}
    Yf, Wf = np.fft.rfft(y, axis=1), np.fft.rfft(wh, axis=1); f = np.fft.rfftfreq(y.shape[1], 1 / FS)
    e, c, p = coherence_terms(Yf, Wf)
    for b, (lo, hi) in BANDS.items():
        s = (f >= lo) & (f < hi)
        out[f"c_{b}"] = c[s].sum() / e[s].sum(); out[f"E_{b}"] = e[s].sum() / ok.sum()
    out["_spec"] = (f, c, e)
    # envelope: per-trial log band power, contra - ipsi (the lateralization index), its coherent fraction
    for b, (lo, hi) in ENV_BANDS.items():
        sos = signal.butter(4, [lo, hi], "bandpass", fs=FS, output="sos")
        env = lambda x: np.log(np.mean(np.abs(signal.hilbert(signal.sosfiltfilt(sos, x, axis=-1), axis=-1))[..., W] ** 2, -1))
        ali = (env(contra) - env(ipsi))[ok]                                   # (trials,)
        # waveform coherence of the same band, for contrast
        yb = signal.sosfiltfilt(sos, lat, axis=-1)[:, W]; whb = loo_mean(yb, g)
        out[f"cwave_{b}"] = (yb * whb).sum() / (yb ** 2).sum()
        wa = loo_mean(ali, g)                                                 # the mean is kept: it is the signal
        out[f"cenv_{b}"] = (ali * wa).sum() / (ali ** 2).sum()
        out[f"ali_{b}"] = float(ali.mean())
        out[f"ali_sd_{b}"] = float(ali.std())
    return out


def main():
    prep, res = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    S = pd.read_csv(res / "subjects.tsv", sep="\t")
    S = S[~S.excluded]
    rows, spec = [], []
    for _, r in S.iterrows():
        o = subject(prep, r["sub"], r["heog"]); spec.append(o.pop("_spec")); o["lab"] = r["lab"]; rows.append(o)
    df = pd.DataFrame(rows); df.to_csv(res / "coherence.tsv", sep="\t", index=False)
    f = spec[0][0]; c = np.sum([s[1] for s in spec], 0); e = np.sum([s[2] for s in spec], 0)
    np.savez(res / "coherence_spectrum.npz", f=f, c_pooled=c / e,
             c_per_sub=np.stack([s[1] / s[2] for s in spec]))
    cols = ["c_time", "p_time", "kappa_time"] + [f"c_{b}" for b in BANDS] + \
           [k for b in ENV_BANDS for k in (f"cwave_{b}", f"cenv_{b}", f"ali_{b}")]
    summ = df[cols].agg(["mean", "median", "std"]).T
    summ["t_vs_0"] = df[cols].mean() / (df[cols].std() / np.sqrt(len(df)))
    print(summ.round(4))
    print("max identity error:", df.identity_err.max())
    json.dump({"n_subjects": len(df), "summary": summ.round(6).to_dict(orient="index")},
              open(res / "coherence_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
