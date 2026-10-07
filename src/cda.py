"""Gate: does this pipeline show the CDA? Per-subject CDA by set size and Cowan's K, from src/prep.py output.

    python -I src/cda.py PREP_DIR META_DIR OUT_DIR

Per trial: baseline -200..0 ms; reject if the horizontal-eye step (max |mean(next 100 ms) - mean(previous
100 ms)| over -200..1000 ms) exceeds the threshold for the subject's eye channel, or any lateral channel's
peak-to-peak exceeds 150 uV over -200..1000 ms. CDA = contralateral - ipsilateral, averaged over the ROI pairs,
300-900 ms (Vogel & Machizawa 2004). K = N (H - FA) per set size (Cowan). The contra-ipsi difference cancels the
recording reference, so the ten labs' different references do not enter.

Subject exclusion: fewer than 50 % of trials surviving, overall accuracy below 0.6, fewer than 30 clean trials in
any set size. The residual lateralized eye signal (mean toward-cue HEOG, 300-900 ms) is kept per subject and
used as a robustness check in src/predict.py, not as an exclusion: its scale differs between a bipolar EOG, a
single mastoid-referenced EOG electrode and the F7 - F8 proxy, so one microvolt cut-off does not mean one thing.
One eye-step threshold (20 uV) for every eye channel. Four labs have no EOG and get F7 - F8 as a proxy, which
sees a smaller eye signal; the threshold was raised from 12 uV after it rejected a median 75 % of their trials
(set before any CDA-K analysis was run).
"""
import json, pathlib, sys
import numpy as np, pandas as pd

FS, T0 = 250, -150
T = (np.arange(500) + T0) / FS
ROI = [("PO7", "PO8"), ("P7", "P8"), ("O1", "O2"), ("PO3", "PO4"), ("P5", "P6")]
STEP_THR = {"bipolar": 20.0, "single": 20.0, "F7-F8 proxy": 20.0, "none": np.inf}
WIN = (0.3, 0.9)


def win(a, b):
    return (T >= a) & (T < b)


def step_max(h, w=25, hop=5):
    """Max over the epoch of |mean of the next w samples - mean of the previous w| (Luck's step function)."""
    c = np.cumsum(np.pad(h, ((0, 0), (1, 0))), 1)
    m = lambda s: (c[:, s + w] - c[:, s]) / w
    starts = np.arange(0, h.shape[1] - 2 * w, hop)
    return np.max(np.abs(np.stack([m(s + w) - m(s) for s in starts], 1)), 1)


def lateralized(X, names, side, pairs):
    """(trials, time) contra - ipsi, averaged over the pairs present."""
    out = []
    for L, R in pairs:
        if L in names and R in names:
            l, r = X[:, names.index(L)], X[:, names.index(R)]
            out.append(np.where(side[:, None] == "left", r - l, l - r))
    return np.mean(out, 0) if out else None


def subject(prep, sub):
    j = json.loads((prep / f"{sub}.json").read_text()); d = np.load(prep / f"{sub}.npz")
    X, H = d["X"].astype(np.float64), d["H"].astype(np.float64)
    names = j["lat_names"]; tr = pd.DataFrame(j["trials"])
    base = win(-0.2, 0)
    X -= X[:, :, base].mean(-1, keepdims=True); H -= H[:, base].mean(-1, keepdims=True)
    w = win(-0.2, 1.0)
    step = step_max(H[:, w])
    ptp = np.ptp(X[:, :, w], axis=-1).max(1)
    side = tr["side"].to_numpy(dtype=str)
    ok = (step < STEP_THR[j["heog"]]) & (ptp < 150) & np.isin(side, ["left", "right"])
    lat = lateralized(X, names, side, ROI)
    po = lateralized(X, names, side, [("PO7", "PO8")])
    heog_lat = np.where(side[:, None] == "left", -H, H)        # + = toward the cued side for an L-R derivation
    row = {"sub": sub, "heog": j["heog"], "n_trials": len(tr), "kept": float(ok.mean()),
           "roi_pairs": ",".join(f"{a}/{b}" for a, b in ROI if a in names and b in names)}
    acc = tr["outcome"].isin(["hit", "cr"]).sum() / max(tr["outcome"].isin(["hit", "cr", "miss", "fa"]).sum(), 1)
    row["acc"] = float(acc)
    cw = win(*WIN)
    for n in (2, 4, 6):
        s = tr["setsize"].values == n
        o = tr["outcome"].to_numpy(dtype=str)
        h = (o[s] == "hit").sum() / max(((o[s] == "hit") | (o[s] == "miss")).sum(), 1)
        fa = (o[s] == "fa").sum() / max(((o[s] == "fa") | (o[s] == "cr")).sum(), 1)
        row[f"K{n}"] = float(n * (h - fa))
        k = s & ok
        row[f"n{n}"] = int(k.sum())
        row[f"cda{n}"] = float(lat[k][:, cw].mean()) if k.any() else np.nan
        row[f"cda{n}_po"] = float(po[k][:, cw].mean()) if (po is not None and k.any()) else np.nan
        row[f"heog{n}"] = float(heog_lat[k][:, cw].mean()) if k.any() else np.nan
        # split-half (odd/even clean trials) for reliability
        idx = np.where(k)[0]
        row[f"cda{n}_odd"] = float(lat[idx[0::2]][:, cw].mean()) if len(idx) > 1 else np.nan
        row[f"cda{n}_even"] = float(lat[idx[1::2]][:, cw].mean()) if len(idx) > 1 else np.nan
    wave = {n: lat[(tr["setsize"].values == n) & ok].mean(0) for n in (2, 4, 6)}
    return row, wave


def main():
    prep, meta, out = map(pathlib.Path, sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True)
    sv = pd.read_csv(meta / "survey.tsv", sep="\t").set_index("sub")
    rows, waves = [], {}
    for p in sorted(prep.glob("*.json")):
        try:
            r, w = subject(prep, p.stem)
        except Exception as e:
            print(p.stem, "skip", repr(e)); continue
        r["lab"] = sv.loc[p.stem, "lab"]; rows.append(r); waves[p.stem] = w
    df = pd.DataFrame(rows)
    df["K"] = df[["K4", "K6"]].mean(axis=1)
    df["excluded"] = (df.kept < 0.5) | (df.acc < 0.6) | (df[["n2", "n4", "n6"]].min(axis=1) < 30)
    df.to_csv(out / "subjects.tsv", sep="\t", index=False)
    np.savez(out / "cda_waves.npz", subs=np.array(list(waves)), t=T,
             **{f"ss{n}": np.stack([waves[s][n] for s in waves]) for n in (2, 4, 6)})
    g = df[~df.excluded]
    print(f"{len(df)} subjects, {int(df.excluded.sum())} excluded, {len(g)} analysed")
    print(g.groupby("lab")[["cda2", "cda4", "cda6", "K"]].mean().round(2))
    print("grand mean CDA (uV):", g[["cda2", "cda4", "cda6"]].mean().round(3).to_dict(),
          "| K:", g[["K2", "K4", "K6"]].mean().round(2).to_dict())


if __name__ == "__main__":
    main()
