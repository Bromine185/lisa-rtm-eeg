"""Figures for the README. Reads results/ only.

    python -I src/figs.py RESULTS_DIR FIG_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference palette (dataviz skill, light steps), fixed order; neutrals for baselines
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#4a3aa7"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2})


def label_end(ax, x, y, text, color=INK2, dx=0.0):
    ax.annotate(text, (x, y), xytext=(4 + dx, 0), textcoords="offset points", va="center", fontsize=9, color=color)


def fig_cda(res, out):
    w = np.load(res / "cda_waves.npz"); S = pd.read_csv(res / "subjects.tsv", sep="\t").set_index("sub")
    keep = np.array([not S.loc[s, "excluded"] for s in w["subs"]])
    t = w["t"] * 1000
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for n, col in zip((2, 4, 6), (C["blue"], C["orange"], C["aqua"])):
        m = w[f"ss{n}"][keep].mean(0); se = w[f"ss{n}"][keep].std(0) / np.sqrt(keep.sum())
        a.fill_between(t, m - se, m + se, color=col, alpha=0.18, lw=0)
        a.plot(t, m, color=col, label=f"set size {n}")
    a.axvspan(300, 900, color=GRID, alpha=0.6, lw=0, zorder=0)
    a.axhline(0, color=MUTED, lw=1); a.axvline(0, color=MUTED, lw=1)
    a.set_xlim(-200, 1250); a.set_xlabel("time from memory array (ms)"); a.set_ylabel("contra - ipsi (µV)")
    a.set_title(f"CDA, {int(keep.sum())} subjects, 10 labs (mean ± SE)"); a.legend(loc="lower left")
    g = S[~S.excluded]
    x = np.arange(3)
    for i, (n, col) in enumerate(zip((2, 4, 6), (C["blue"], C["orange"], C["aqua"]))):
        b.bar(i, g[f"K{n}"].mean(), yerr=g[f"K{n}"].sem(), color=col, width=0.6, capsize=3, ecolor=INK2)
        b.annotate(f"{g[f'K{n}'].mean():.2f}", (i, g[f"K{n}"].mean() + g[f"K{n}"].sem()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    b.set_xticks(x, ["2", "4", "6"]); b.set_xlabel("set size"); b.set_ylabel("Cowan's K"); b.set_title("Capacity K")
    fig.tight_layout(); fig.savefig(out / "fig1_cda.png", dpi=150); plt.close(fig)


ARM_STYLE = {"det_paper | pt": (C["violet"], "det_paper (L1, LISA)"), "det | pt": (C["blue"], "det (L1 + 0.1 MSSTFT)"),
             "es_marg | pt": (C["orange"], "es_marg (energy score), one draw"),
             "shaped_noise": (C["aqua"], "shaped noise (unconditional)")}


def fig_bwe(res, out):
    rows = pd.read_json(res / "score_rows.json"); ex = json.load(open(res / "score_extra.json"))
    fc = ex["band_centres_hz"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (col, lab) in ARM_STYLE.items():
        g = rows[rows.cond == k]
        if not len(g):
            continue
        curve = np.mean(np.stack(g.curve.values), 0); coh = np.mean(np.stack(g.coh_curve.values), 0)
        a.plot(fc, curve, "-o", color=col, ms=5, label=lab); b.plot(fc, coh, "-o", color=col, ms=5, label=lab)
    a.axhline(0, color=MUTED, lw=1); b.axhline(0, color=MUTED, lw=1)
    a.set_xscale("log"); b.set_xscale("log")
    for ax in (a, b):
        ax.set_xticks([31.5, 40, 50, 63, 80, 100, 125], ["31.5", "40", "50", "63", "80", "100", "125"]); ax.set_xlabel("third-octave band centre (Hz)")
    a.set_ylabel("output / true energy (dB)"); a.set_title("Deficit: high-band level, held-out subjects")
    b.set_ylabel("coherent fraction c"); b.set_title("Coherent fraction: how much is predicted")
    a.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "fig2_bwe.png", dpi=150); plt.close(fig)


def fig_predictability(res, out):
    sp = np.load(res / "coherence_spectrum.npz"); co = pd.read_csv(res / "coherence.tsv", sep="\t")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    f, c = sp["f"], sp["c_pooled"]; s = f <= 30
    a.plot(f[s], c[s], "-o", color=C["blue"], ms=5)
    a.set_xlabel("frequency (Hz), 0.3-0.9 s window"); a.set_ylabel("coherent fraction c")
    a.set_title("Predictability spectrum of a single CDA trial")
    for i in np.where(s)[0][:2]:
        a.annotate(f"{c[i]:.3f}", (f[i], c[i]), xytext=(6, 2), textcoords="offset points", fontsize=9, color=INK2)
    bands = ["theta 4-8", "alpha 8-13", "beta 13-30"]
    x = np.arange(len(bands)); wv = [co[f"cwave_{k}"].mean() for k in bands]; en = [co[f"cenv_{k}"].mean() for k in bands]
    ew = [co[f"cwave_{k}"].sem() for k in bands]; ee = [co[f"cenv_{k}"].sem() for k in bands]
    b.bar(x - 0.2, wv, 0.38, yerr=ew, color=C["blue"], label="waveform (evoked)", capsize=3, ecolor=INK2)
    b.bar(x + 0.2, en, 0.38, yerr=ee, color=C["orange"], label="envelope (log power)", capsize=3, ecolor=INK2)
    for xi, v, e in zip(x + 0.2, en, ee):
        b.annotate(f"{v:.3f}", (xi, v + e), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    for xi, v, e in zip(x - 0.2, wv, ew):
        b.annotate(f"{v:.3f}", (xi, v + e), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    b.set_xticks(x, bands); b.set_ylabel("coherent fraction c"); b.set_title("Same band: waveform vs envelope")
    b.legend(loc="upper left", fontsize=8.5); b.set_ylim(0, max(np.add(en, ee)) * 1.5)
    fig.tight_layout(); fig.savefig(out / "fig3_predictability.png", dpi=150); plt.close(fig)


def fig_spatial(res, out):
    d = json.load(open(res / "spatial_summary.json"))["median"]
    bands = list(d["spline"]); x = np.arange(len(bands))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for m, col, lab in (("spline", C["violet"], "spherical spline"), ("wiener", C["blue"], "Wiener (conditional mean)"),
                        ("sampler", C["orange"], "Gaussian conditional, one draw")):
        a.plot(x, [d[m][k]["c"] for k in bands], "-o", color=col, ms=5, label=lab)
        b.plot(x, [d[m][k]["p"] for k in bands], "-o", color=col, ms=5, label=lab)
    for ax in (a, b):
        ax.set_xticks(x, [k + " Hz" for k in bands], fontsize=8.5); ax.set_ylim(0, 1.1)
    a.set_ylabel("coherent fraction c"); a.set_title("A missing electrode: predicted fraction")
    b.set_ylabel("level p = output / true energy"); b.set_title("...and the level each estimator outputs")
    a.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "fig4_spatial.png", dpi=150); plt.close(fig)


def fig_predict(res, out):
    S = pd.read_csv(res / "subjects.tsv", sep="\t"); S = S[~S.excluded].copy()
    S["cda46"] = S[["cda4", "cda6"]].mean(axis=1)
    P = pd.read_csv(res / "lolo_cda46.tsv", sep="\t")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    a.scatter(S.cda46, S.K, s=22, color=C["blue"], alpha=0.75, edgecolor=SURF, linewidth=0.8)
    bb, aa = np.polyfit(S.cda46, S.K, 1); xs = np.linspace(S.cda46.min(), S.cda46.max(), 2)
    a.plot(xs, aa + bb * xs, color=INK2, lw=1.5)
    r = np.corrcoef(S.cda46, S.K)[0, 1]
    a.set_xlabel("CDA, mean of set sizes 4 and 6 (µV)"); a.set_ylabel("K (mean of set sizes 4, 6)")
    a.set_title(f"CDA vs capacity, n = {len(S)}, r = {r:+.2f}")
    bins = np.linspace(0, 1, 11)
    for m, col, lab in (("clim", MUTED, "climatology"), ("cda", C["orange"], "CDA forecast")):
        h, _ = np.histogram(P[P.model == m].pit, bins)
        b.step(bins, np.r_[h, h[-1]] / len(P[P.model == m]) * 10, where="post", color=col, label=lab)
    b.axhline(1, color=MUTED, lw=1, ls="--"); b.set_ylim(0, 2)
    b.set_xlabel("PIT of held-out-lab forecasts"); b.set_ylabel("density (flat = calibrated)")
    b.set_title("Leave-one-lab-out calibration"); b.legend(loc="upper center", fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "fig5_predict.png", dpi=150); plt.close(fig)


def main():
    res, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    for fn, need in ((fig_cda, "cda_waves.npz"), (fig_bwe, "score_rows.json"), (fig_predictability, "coherence.tsv"),
                     (fig_spatial, "spatial_summary.json"), (fig_predict, "lolo_cda46.tsv")):
        if (res / need).exists():
            fn(res, out); print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
