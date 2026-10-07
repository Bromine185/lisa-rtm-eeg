"""Experiment A': the same coherent-fraction measurement for spatial super-resolution (a missing electrode).

For each held-out BWE test subject, every electrode with a standard 10-05 label is predicted from all the others
three ways, and scored per frequency band with the lisa-rtm identity (coherent fraction c, level p, kappa = c / p):
  spline   spherical spline (Perrin et al. 1989; MNE's bad-channel repair): fixed by geometry, no fitting
  wiener   the MSE-optimal linear map per band (the conditional mean), fitted on the first 90 s, scored on the last 90 s
  sampler  wiener + complex Gaussian noise at the conditional variance: one draw of the exact Gaussian conditional
Each target is re-referenced to the average of the OTHER electrodes; a common average that includes the target
makes it exactly minus the sum of the rest and every fitted map would score c = 1. Where c is near 1 a deterministic
interpolator is all one needs; where c is near 0 the missing signal is a distribution, and the RTM argument applies.

    python -I src/spatial.py PREP_DIR META_DIR OUT_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd
import mne
from mne.channels.interpolation import _make_interpolation_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from splits import bwe_split

FS = 250
BANDS = {"1-4": (1, 4), "4-8": (4, 8), "8-13": (8, 13), "13-30": (13, 30), "30-60": (30, 60), "60-125": (60, 125)}
LINE = [50.0, 60.0, 100.0, 120.0]


def main():
    prep, meta, out = map(pathlib.Path, sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True)
    _, test, lab = bwe_split(meta)
    std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    lower = {k.lower(): v for k, v in std.items()}
    rows = []
    for sub in test:
        if not (prep / f"{sub}.npz").exists():
            print(sub, "not preprocessed, skipped", flush=True); continue
        j = json.loads((prep / f"{sub}.json").read_text()); A = np.load(prep / f"{sub}.npz")["sr"].astype(np.float64)
        keep = [i for i, n in enumerate(j["sr_names"]) if n.lower() in lower]
        names = [j["sr_names"][i] for i in keep]; A = A[keep]
        pos = np.array([lower[n.lower()] for n in names])
        half = A.shape[1] // 2
        f = np.fft.rfftfreq(half, 1 / FS); noline = np.all([np.abs(f - l) > 2 for l in LINE], 0)
        sel = {b: (f >= lo) & (f < hi) & noline for b, (lo, hi) in BANDS.items()}
        rng = np.random.default_rng(int(sub[-4:]))
        acc = {(m, b): [0.0, 0.0, 0.0] for m in ("spline", "wiener", "sampler") for b in BANDS}
        for t in range(len(names)):
            src = [i for i in range(len(names)) if i != t]
            ref = A[src].mean(0)                                             # reference that excludes the target
            Xs, Xt = A[src] - ref, A[t] - ref
            Ff, Ft = np.fft.rfft(Xs[:, :half], axis=1), np.fft.rfft(Xs[:, half:2 * half], axis=1)
            Yf, Yt = np.fft.rfft(Xt[:half]), np.fft.rfft(Xt[half:2 * half])
            Wsp = _make_interpolation_matrix(pos[src], pos[[t]])[0]          # spline: fixed by geometry
            for b, sb in sel.items():
                # Wiener (the conditional mean among linear maps), fitted on the first half, per band
                Fs = Ff[:, sb]; Sss = Fs @ Fs.conj().T; Sts = Yf[sb] @ Fs.conj().T
                Sss += 1e-6 * np.trace(Sss).real / len(src) * np.eye(len(src))
                w = np.linalg.solve(Sss.T, Sts)                              # w^T = Sts Sss^-1
                var_c = max((np.sum(np.abs(Yf[sb]) ** 2) - (w @ Sts.conj()).real) / sb.sum(), 0.0)
                Y = Yt[sb]; Fb = Ft[:, sb]
                Pw = w @ Fb
                Z = (rng.standard_normal(len(Y)) + 1j * rng.standard_normal(len(Y))) * np.sqrt(var_c / 2)
                for m, P in (("spline", Wsp @ Fb), ("wiener", Pw), ("sampler", Pw + Z)):
                    a = acc[(m, b)]
                    a[0] += np.sum(np.abs(Y) ** 2); a[1] += np.sum((Y * np.conj(P)).real); a[2] += np.sum(np.abs(P) ** 2)
        r = {"sub": sub, "lab": lab[sub], "n_ch": len(names)}
        for (m, b), (E, C, Pw) in acc.items():
            r[f"{m}_c_{b}"] = C / E; r[f"{m}_p_{b}"] = Pw / E; r[f"{m}_snr_{b}"] = float(10 * np.log10(E / (E + Pw - 2 * C)))
        rows.append(r)
        print(sub, lab[sub], len(names), {b: (round(r[f"spline_c_{b}"], 2), round(r[f"wiener_c_{b}"], 2), round(r[f"wiener_p_{b}"], 2)) for b in BANDS}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "spatial.tsv", sep="\t", index=False)
    summ = {m: {b: {k: float(df[f"{m}_{k}_{b}"].median()) for k in ("c", "p", "snr")} for b in BANDS}
            for m in ("spline", "wiener", "sampler")}
    json.dump({"n_subjects": len(df), "median": summ}, open(out / "spatial_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
