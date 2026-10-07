"""Markdown tables for the README, every number read from results/.

    python -I src/tables.py RESULTS_DIR META_DIR > results/tables.md
"""
import json, pathlib, sys
import numpy as np, pandas as pd


def f(v, d=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def survey(meta):
    sv = pd.read_csv(meta / "survey.tsv", sep="\t")
    g = sv.groupby("lab").agg(n=("sub", "size"), fs=("fs", lambda x: "/".join(map(str, sorted(set(x))))),
                              ch=("n_eeg", lambda x: "/".join(map(str, sorted(set(x))))),
                              ref=("ref", lambda x: "/".join(sorted(set(map(str, x))))),
                              maker=("maker", "first"), line=("line", "first"))
    L = ["| lab | n | rate (Hz) | EEG channels | recording reference | system | mains (Hz) |", "|---|---|---|---|---|---|---|"]
    for lab, r in g.iterrows():
        L.append(f"| {lab} | {r.n} | {r.fs} | {r.ch} | {r.ref} | {r.maker} | {r.line} |")
    return "\n".join(L)


def bwe(res):
    rows = pd.read_json(res / "score_rows.json"); ex = json.load(open(res / "score_extra.json"))
    order = ["ceiling", "naive", "det_paper | pt", "det | pt", "es_marg tau0 | pt", "es_marg | pt", "shaped_noise",
             "det_paper", "det", "es_marg"]
    names = {"ceiling": "ceiling: true low band, empty high band", "naive": "naive (polyphase sinc)",
             "det_paper | pt": "det_paper: L1 only (LISA's released loss)", "det | pt": "det: L1 + 0.1 MSSTFT",
             "es_marg tau0 | pt": "es_marg at tau = 0", "es_marg | pt": "es_marg: energy score, one draw (M = 8)",
             "shaped_noise": "shaped noise: unconditional, mean HB spectrum (M = 8)",
             "det_paper": "det_paper, raw", "det": "det, raw", "es_marg": "es_marg, raw"}
    g = rows.groupby("cond")
    ceil = rows[rows.cond == "ceiling"].set_index("item").snr
    L = ["| condition | deficit dB | loud / mid / quiet dB | coh | coh > 40 Hz | kappa | SNR dB | vs ceiling dB | LB coh | CRPS | PIT low / high | gap HB (indep.) dB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        if k not in g.groups:
            continue
        d = g.get_group(k); vs = (d.set_index("item").snr - ceil).mean()
        empty = k in ("ceiling", "naive")
        pit = ""
        if "_pit" in d and d["_pit"].notna().any():
            h = np.sum([p for p in d["_pit"] if isinstance(p, list)], 0).astype(float); h /= h.sum(); pit = f"{h[0]:.3f} / {h[-1]:.3f}"
        gap = f"{d.gap_hb.mean():.2f} ({d.gap_indep_hb.mean():.2f})" if "gap_hb" in d and d.gap_hb.notna().any() else ""
        L.append(f"| {names[k]} | {f(d.deficit.mean())} | {f(d.loud.mean(), 1)} / {f(d.mid.mean(), 1)} / {f(d.quiet.mean(), 1)} | "
                 f"{'--' if empty else f(d.coh.mean(), 3)} | {'--' if empty else f(np.mean(np.stack(d.coh_curve.values)[:, 1:]), 3)} | "
                 f"{'--' if empty else f(d.kappa.mean(), 3)} | {f(d.snr.mean())} | {f(vs)} | "
                 f"{f(d.lb_coh.mean(), 3)} | {f(d.crps.mean(), 3)} | {pit} | {gap} |")
    M = 8
    L.append("")
    L.append(f"Ideals at M = {M}: PIT low = high = {1/(M+1):.3f}; gap equals its independent-draw value when draws are as "
             f"diverse as their level allows; a calibrated sampler has gap {10*np.log10(2/(1+1/M)):.2f} dB.")
    e = ex["env_pred_from_lb"]
    L.append("")
    L.append(f"High-band envelope from the low band alone (ridge on 5 log band energies, training subjects -> held-out): "
             f"R^2 pooled {e['r2_pooled']:.3f}; within a recording r = {e['r_within_recording']:.3f} "
             f"(R^2 {e['r2_within_recording']:.3f}), {e['n_frames']} frames.")
    er = ex["env_regression_within_item"]
    L.append("")
    L.append("| condition | envelope slope | r | sd ratio |")
    L.append("|---|---|---|---|")
    for k in ("det | pt", "es_marg tau0 | pt", "es_marg | pt", "shaped_noise"):
        if er.get(k):
            L.append(f"| {names[k]} | {er[k]['slope']:.3f} | {er[k]['r']:.3f} | {er[k]['sd_ratio']:.3f} |")
    # per lab deficit and coh
    L.append("")
    L.append("| lab | det deficit | es_marg deficit | det coh | es_marg coh | HB share of energy |")
    L.append("|---|---|---|---|---|---|")
    for lab, d in rows.groupby("lab"):
        a, b = d[d.cond == "det | pt"], d[d.cond == "es_marg | pt"]
        L.append(f"| {lab} | {f(a.deficit.mean())} | {f(b.deficit.mean())} | {f(a.coh.mean(), 3)} | {f(b.coh.mean(), 3)} | "
                 f"{f(100 * d.hb_share.mean(), 1)} % |")
    return "\n".join(L)


def coherence(res):
    s = json.load(open(res / "coherence_summary.json")); S = s["summary"]
    L = [f"n = {s['n_subjects']} subjects.", "", "| quantity | mean | median | t vs 0 |", "|---|---|---|---|"]
    for k, lab in (("c_time", "c, 0.3-0.9 s, broadband"), ("p_time", "p (ERP energy / trial energy)"),
                   ("kappa_time", "kappa = c / p"), ("c_0-2", "c, 0-2 Hz"), ("c_2-4", "c, 2-4 Hz"), ("c_4-8", "c, 4-8 Hz"),
                   ("c_8-13", "c, 8-13 Hz"), ("c_13-30", "c, 13-30 Hz"),
                   ("cwave_alpha 8-13", "alpha waveform c (band-passed)"), ("cenv_alpha 8-13", "alpha envelope c (log power)"),
                   ("ali_alpha 8-13", "alpha lateralization, log contra/ipsi"), ("cwave_beta 13-30", "beta waveform c"),
                   ("cenv_beta 13-30", "beta envelope c"), ("cwave_theta 4-8", "theta waveform c"), ("cenv_theta 4-8", "theta envelope c")):
        v = S[k]; L.append(f"| {lab} | {v['mean']:.4f} | {v['median']:.4f} | {v['t_vs_0']:.1f} |")
    return "\n".join(L)


def spatial(res):
    d = json.load(open(res / "spatial_summary.json"))
    m = d["median"]; bands = list(m["spline"])
    L = [f"n = {d['n_subjects']} held-out subjects, every electrode with a 10-05 label left out in turn; medians over subjects.", "",
         "| band (Hz) | " + " | ".join(f"{b}" for b in bands) + " |", "|---|" + "---|" * len(bands)]
    for est, name in (("spline", "spline"), ("wiener", "Wiener"), ("sampler", "Gaussian draw")):
        for q, ql in (("c", "c"), ("p", "p"), ("snr", "SNR dB")):
            L.append(f"| {name}: {ql} | " + " | ".join(f"{m[est][b][q]:.2f}" for b in bands) + " |")
    return "\n".join(L)


def predict(res):
    p = json.load(open(res / "predict.json"))
    L = [f"n = {p['n']} subjects, {p['labs']} labs. CRPS in units of K; CRPSS = 1 - CRPS / CRPS(climatology), 95 % paired-bootstrap CI.", "",
         "| feature | r | r within lab | CRPS clim | CRPSS cda | CRPSS cda, lab-centred | CRPSS lab oracle | point sd / sd(K) | PIT 90 % cover |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, v in p["features"].items():
        s = v["scores"]
        L.append(f"| {v['label']} | {v['r']:+.3f} (p {v['p']:.2g}) | {v['r_within_lab']:+.3f} | {s['clim']['crps']:.3f} | "
                 f"{s['cda']['crpss_vs_clim']:+.3f} [{s['cda']['crpss_ci95'][0]:+.3f}, {s['cda']['crpss_ci95'][1]:+.3f}] | "
                 f"{s['cda_labc']['crpss_vs_clim']:+.3f} | {s['lab_clim']['crpss_vs_clim']:+.3f} | {s['cda']['sd_ratio_point']:.3f} | "
                 f"{s['cda']['pit_cover90']:.2f} |")
    L.append("")
    L.append("Split-half reliability (odd/even trials, Spearman-Brown): " + ", ".join(
        f"{k} {v['spearman_brown']:.2f}" for k, v in p["reliability"].items()))
    ss = p["set_size"]
    L.append("")
    L.append("Set size: CDA " + " / ".join(f"{ss[f'cda{n}']['mean']:.3f} ± {ss[f'cda{n}']['se']:.3f}" for n in (2, 4, 6)) +
             " µV; K " + " / ".join(f"{ss[f'K{n}']['mean']:.2f}" for n in (2, 4, 6)) +
             f"; CDA 4 vs 2: t = {p['cda4_vs_cda2']['t']:.2f}, p = {p['cda4_vs_cda2']['p']:.2g}.")
    return "\n".join(L)


def main():
    res, meta = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out = ["## Labs", survey(meta)]
    for name, fn, sub, need in (("BWE", bwe, "bwe", "score_rows.json"),
                                ("Coherence of the CDA", coherence, "", "coherence_summary.json"),
                                ("Spatial", spatial, "", "spatial_summary.json"), ("CDA -> K", predict, "", "predict.json")):
        if (res / sub / need).exists():
            out += ["", f"## {name}", fn(res / sub)]
    if (res / "strict_eye" / "predict.json").exists():
        out += ["", "## CDA -> K, strict eye criterion (largest-residual-HEOG quarter dropped)", predict(res / "strict_eye")]
    print("\n".join(out))


if __name__ == "__main__":
    main()
