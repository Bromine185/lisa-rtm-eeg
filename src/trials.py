"""One row per trial for every included subject: the shared table behind the 'ideas' analyses.

    python -I src/trials.py PREP_DIR RESULTS_DIR      -> RESULTS_DIR/trials.csv.gz

Columns: sub, lab, trial (0..719 in presentation order), block, t_min (minutes since first trial), setsize, side,
truth (change/same), outcome (hit/miss/cr/fa/none), correct, rt, clean (eye step + amplitude criteria of cda.py),
cda (contra - ipsi, ROI mean, 300-900 ms, uV), cda_late (600-900 ms), alpha_lat (log contra/ipsi alpha power,
300-900 ms), alpha_pre (log mean alpha power both hemispheres, -500..0 ms), alpha_pre_lat (pre-stimulus log
contra/ipsi alpha), heog (toward-cue HEOG, 300-900 ms), gaze (toward-cue gaze X shift, 300-900 ms minus -200..0,
native units; NaN without an eye tracker), lat_ac1 (lag-1 autocorrelation of the lateralized signal, 300-900 ms,
after removing the subject's condition mean), lat_var (its variance).
"""
import json, pathlib, sys
import numpy as np, pandas as pd
from scipy import signal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cda import T, ROI, STEP_THR, win, step_max

FS = 250
SOS_A = signal.butter(4, [8, 13], "bandpass", fs=FS, output="sos")


def logpow(x, w):
    return np.log(np.mean(np.abs(signal.hilbert(signal.sosfiltfilt(SOS_A, x, axis=-1), axis=-1))[..., w] ** 2, -1))


def subject(prep, sub, heog_kind, lab):
    j = json.loads((prep / f"{sub}.json").read_text()); d = np.load(prep / f"{sub}.npz")
    X, H = d["X"].astype(np.float64), d["H"].astype(np.float64); G = d["G"].astype(np.float64)
    tr = pd.DataFrame(j["trials"]); names = j["lat_names"]
    base = win(-0.2, 0)
    Xb = X - X[:, :, base].mean(-1, keepdims=True); Hb = H - H[:, base].mean(-1, keepdims=True)
    side = tr["side"].to_numpy(dtype=str); ss = tr["setsize"].to_numpy(int)
    w = win(-0.2, 1.0)
    clean = (step_max(Hb[:, w]) < STEP_THR[heog_kind]) & (np.ptp(Xb[:, :, w], -1).max(1) < 150) & np.isin(side, ["left", "right"])
    pairs = [(a, b) for a, b in ROI if a in names and b in names]
    L = Xb[:, [names.index(a) for a, _ in pairs]].mean(1); R = Xb[:, [names.index(b) for _, b in pairs]].mean(1)
    Lr = X[:, [names.index(a) for a, _ in pairs]].mean(1); Rr = X[:, [names.index(b) for _, b in pairs]].mean(1)
    left = side[:, None] == "left"
    con, ips = np.where(left, R, L), np.where(left, L, R)
    conr, ipsr = np.where(left, Rr, Lr), np.where(left, Lr, Rr)
    lat = con - ips
    W, Wl, Wp = win(0.3, 0.9), win(0.6, 0.9), win(-0.5, 0.0)
    sgn = np.where(side == "left", 1.0, -1.0)                   # L - R eye derivations go negative for a rightward gaze
    out = pd.DataFrame({"sub": sub, "lab": lab, "trial": np.arange(len(tr)), "setsize": ss, "side": side,
                        "truth": tr["truth"].to_numpy(dtype=str), "outcome": tr["outcome"].to_numpy(dtype=str),
                        "rt": tr["rt"].to_numpy(float), "onset": tr["onset"].to_numpy(float), "clean": clean,
                        "cda": lat[:, W].mean(1), "cda_late": lat[:, Wl].mean(1),
                        "alpha_lat": logpow(conr, W) - logpow(ipsr, W),
                        "alpha_pre": 0.5 * (logpow(conr, Wp) + logpow(ipsr, Wp)),
                        "alpha_pre_lat": logpow(conr, Wp) - logpow(ipsr, Wp),
                        "heog": sgn * Hb[:, W].mean(1)})
    if G.size:
        g = G - G[:, base].mean(-1, keepdims=True)
        out["gaze"] = -sgn * g[:, W].mean(1)                     # gaze X grows rightward: toward a right cue = +, left = -
    else:
        out["gaze"] = np.nan
    # residual dynamics in the retention window: remove the subject's condition mean (the CDA), keep the rest
    resid = lat.copy()
    for n in (2, 4, 6):
        k = (ss == n) & clean
        if k.any():
            resid[ss == n] -= lat[k].mean(0)
    r = resid[:, W] - resid[:, W].mean(1, keepdims=True)
    out["lat_var"] = r.var(1)
    out["lat_ac1"] = (r[:, 1:] * r[:, :-1]).mean(1) / np.maximum(r.var(1), 1e-12)
    out["correct"] = out.outcome.isin(["hit", "cr"]).astype(float).where(out.outcome != "none")
    gaps = np.diff(out.onset.values, prepend=out.onset.values[0])
    out["block"] = np.cumsum(gaps > 15)
    out["t_min"] = (out.onset - out.onset.iloc[0]) / 60
    return out.drop(columns="onset")


def main():
    prep, res = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    S = pd.read_csv(res / "subjects.tsv", sep="\t"); S = S[~S.excluded]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        dfs = list(ex.map(subject, [prep] * len(S), S["sub"], S["heog"], S["lab"]))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(res / "trials.csv.gz", index=False)
    print(df.shape, df.groupby("lab").sub.nunique().to_dict())


if __name__ == "__main__":
    main()
