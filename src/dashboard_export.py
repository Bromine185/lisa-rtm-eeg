"""Export everything the dashboard shows as JSON row files (one per dataset), from results/ and the prep store.

    python -I src/dashboard_export.py PREP_DIR META_DIR RESULTS_DIR OUT_DIR
"""
import json, pathlib, sys
import numpy as np, pandas as pd

R = lambda x, d=3: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def rows(df, d=4):
    out = []
    for r in df.to_dict(orient="records"):
        o = {}
        for k, v in r.items():
            if isinstance(v, (float, np.floating)):
                o[k] = None if not np.isfinite(v) else round(float(v), d)
            elif isinstance(v, (np.integer,)):
                o[k] = int(v)
            elif isinstance(v, (np.bool_,)):
                o[k] = bool(v)
            else:
                o[k] = v
        out.append(o)
    return out


def main():
    prep, meta, res, out = map(pathlib.Path, sys.argv[1:5]); out.mkdir(parents=True, exist_ok=True)
    W = lambda name, data: (out / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
    sv = pd.read_csv(res / "survey.tsv", sep="\t")
    S = pd.read_csv(res / "subjects.tsv", sep="\t")
    P = pd.read_csv(meta / "participants.tsv", sep="\t").rename(columns={"participant_id": "sub"})
    # ---- subjects ---------------------------------------------------------------------------------------
    S["attention"] = (S.K2 / 2).clip(0.05, 1)
    S["cda46"] = S[["cda4", "cda6"]].mean(axis=1)
    S["heog_resid"] = S[["heog2", "heog4", "heog6"]].abs().max(axis=1)
    S["reason"] = np.select([S.kept < 0.5, S.acc < 0.6, S[["n2", "n4", "n6"]].min(axis=1) < 30],
                            ["under 50 % clean trials", "accuracy under 0.6", "under 30 clean trials in a set size"], "")
    S = S.merge(sv[["sub", "fs", "n_eeg", "maker"]], on="sub", how="left")
    co = pd.read_csv(res / "coherence.tsv", sep="\t")
    S = S.merge(co[["sub", "c_time", "c_0-2", "cenv_alpha 8-13", "cwave_alpha 8-13", "ali_alpha 8-13"]]
                .rename(columns={"c_0-2": "c_slow", "cenv_alpha 8-13": "c_alpha_env", "cwave_alpha 8-13": "c_alpha_wave",
                                 "ali_alpha 8-13": "alpha_lat"}), on="sub", how="left")
    hy = res / "ideas" / "hysteresis_coefs.tsv"
    if hy.exists():
        H = pd.read_csv(hy, sep="\t")[["sub", "cda:prev6", "rt:prev6"]].rename(columns={"cda:prev6": "hyst_cda", "rt:prev6": "hyst_rt"})
        S = S.merge(H, on="sub", how="left")
    hm = res / "ideas" / "hmm_subjects.tsv"
    if hm.exists():
        S = S.merge(pd.read_csv(hm, sep="\t")[["sub", "frac_dis", "dwell_dis"]], on="sub", how="left")
    sp = res / "ideas" / "spectral_traits.tsv"
    if sp.exists():
        S = S.merge(pd.read_csv(sp, sep="\t")[["sub", "exponent", "offset", "iaf", "dfa_alpha", "hf_floor"]], on="sub", how="left")
    cols = [c for c in ["sub", "lab", "excluded", "reason", "fs", "n_eeg", "maker", "heog", "n_trials", "kept", "acc", "K2", "K4", "K6",
                        "K", "attention", "cda2", "cda4", "cda6", "cda46", "cda4_po", "heog_resid", "c_time", "c_slow",
                        "c_alpha_wave", "c_alpha_env", "alpha_lat", "hyst_cda", "hyst_rt", "frac_dis", "dwell_dis",
                        "exponent", "offset", "iaf", "dfa_alpha", "hf_floor"] if c in S.columns]
    W("subjects", rows(S[cols]))
    # ---- labs -------------------------------------------------------------------------------------------
    g = S.groupby("lab")
    L = pd.DataFrame({"n": g.size(), "included": g.excluded.apply(lambda x: int((~x).sum()))})
    inc = S[~S.excluded].groupby("lab")
    for c in ["acc", "K2", "K4", "K6", "K", "attention", "cda2", "cda4", "cda6", "kept"]:
        L[c] = inc[c].mean()
    meta_l = sv.groupby("lab").agg(fs=("fs", "first"), n_eeg=("n_eeg", lambda x: "/".join(map(str, sorted(set(x))))),
                                   reference=("ref", "first"), system=("maker", "first"), mains=("line", "first"),
                                   hw_lowpass=("hw_lp", "first"))
    pl = P.groupby("lab").agg(age=("age", "mean"), female=("sex", lambda x: (x == "f").mean()))
    L = L.join(meta_l).join(pl).reset_index()
    eye = S.groupby("lab").heog.first()
    L["eye_channel"] = L.lab.map(eye)
    L["eye_tracker"] = L.lab.isin(["Münster", "NorthDakota", "Ohio", "Zürich (Prof. Langer)"])
    W("labs", rows(L))
    # ---- waveforms ----------------------------------------------------------------------------------------
    w = np.load(res / "cda_waves.npz")
    t = (w["t"] * 1000)
    keep_t = (t >= -200) & (t <= 1300); idx = np.where(keep_t)[0][::2]       # 125 Hz
    subs = list(w["subs"]); lab_of = S.set_index("sub").lab.to_dict(); exc = S.set_index("sub").excluded.to_dict()
    sw = []
    for n in (2, 4, 6):
        A = w[f"ss{n}"]
        for i, s in enumerate(subs):
            sw.append({"sub": s, "setsize": n, "t0_ms": R(t[idx[0]], 0), "dt_ms": R(t[idx[1]] - t[idx[0]], 0),
                       "w": [R(v, 3) for v in A[i, idx]]})
    W("cda_subject_waves", sw)
    lw = []
    for n in (2, 4, 6):
        A = w[f"ss{n}"]
        for lab in ["All labs"] + sorted(set(lab_of.values())):
            k = np.array([(not exc[s]) and (lab == "All labs" or lab_of[s] == lab) for s in subs])
            m, se = np.nanmean(A[k], 0), np.nanstd(A[k], 0) / np.sqrt(k.sum())
            for j in idx:
                lw.append({"lab": lab, "setsize": n, "t_ms": R(t[j], 0), "uv": R(m[j], 4), "se": R(se[j], 4)})
    W("cda_lab_waves", lw)
    # ---- spectra ----------------------------------------------------------------------------------------
    fgrid = np.arange(1, 126, 1.0); ps = []
    for s in S["sub"]:
        d = np.load(prep / f"{s}.npz"); f, p = d["psd_f"], d["psd"]
        ps.append({"sub": s, "f0_hz": 1, "df_hz": 1, "log10_power": [R(v, 3) if fv <= f.max() else None for v, fv in zip(np.interp(fgrid, f, p), fgrid)]})
    W("psd_subjects", ps)
    # ---- trials aggregates --------------------------------------------------------------------------------
    T = pd.read_csv(res / "trials.csv.gz")
    T["minute5"] = (T.t_min // 5 * 5).clip(upper=60)
    c = T[T.clean]
    by_ss = T.groupby(["lab", "setsize"]).agg(acc=("correct", "mean"), rt=("rt", "median"), n=("trial", "size")).reset_index()
    cs = c.groupby(["lab", "setsize"]).agg(cda=("cda", "mean"), alpha_lat=("alpha_lat", "mean"), heog=("heog", "mean"),
                                         gaze=("gaze", "mean"), clean=("trial", "size")).reset_index()
    W("lab_setsize", rows(by_ss.merge(cs, on=["lab", "setsize"])))
    tc = T.groupby(["lab", "minute5"]).agg(acc=("correct", "mean"), rt=("rt", "median"), n=("trial", "size")).reset_index()
    tc2 = c.groupby(["lab", "minute5"]).agg(alpha_pre=("alpha_pre", "mean")).reset_index()
    tca = T.groupby("minute5").agg(acc=("correct", "mean"), rt=("rt", "median"), n=("trial", "size")).reset_index().assign(lab="All labs")
    tca2 = c.groupby("minute5").agg(alpha_pre=("alpha_pre", "mean")).reset_index().assign(lab="All labs")
    W("time_on_task", rows(pd.concat([tc.merge(tc2, on=["lab", "minute5"]), tca.merge(tca2, on=["lab", "minute5"])])))
    bins = np.linspace(-30, 30, 61); hist = []
    for n in (2, 4, 6):
        h, _ = np.histogram(c[c.setsize == n].cda, bins)
        hist += [{"setsize": n, "lo": R(bins[i], 1), "hi": R(bins[i + 1], 1), "count": int(h[i])} for i in range(len(h))]
    W("cda_single_trial_hist", hist)
    ts = T.groupby(["sub", "setsize"]).agg(acc=("correct", "mean"), rt=("rt", "median"), clean=("clean", "mean")).reset_index()
    ts2 = c.groupby(["sub", "setsize"]).agg(cda=("cda", "mean"), cda_sd=("cda", "std"), alpha_lat=("alpha_lat", "mean"),
                                           heog=("heog", "mean")).reset_index()
    W("subject_setsize", rows(ts.merge(ts2, on=["sub", "setsize"], how="left")))
    # ---- analyses ---------------------------------------------------------------------------------------
    sp_ = np.load(res / "coherence_spectrum.npz")
    W("coherence_spectrum", [{"f": R(f, 2), "c": R(v, 5)} for f, v in zip(sp_["f"], sp_["c_pooled"]) if f <= 40])
    spat = json.load(open(res / "spatial_summary.json"))["median"]
    W("spatial", [{"method": m, "band": b, **{k: R(v, 4) for k, v in d.items()}} for m, bb in spat.items() for b, d in bb.items()])
    br = pd.read_json(res / "bwe" / "score_rows.json"); ex = json.load(open(res / "bwe" / "score_extra.json"))
    order = ["naive", "ceiling", "det_paper | pt", "det | pt", "es_marg tau0 | pt", "es_marg | pt", "shaped_noise"]
    bw = []
    for k in order:
        d = br[br.cond == k]
        cur = np.mean(np.stack(d.curve.values), 0); coh = np.mean(np.stack(d.coh_curve.values), 0)
        bw.append({"condition": k, "deficit": R(d.deficit.mean()), "snr": R(d.snr.mean()), "crps": R(d.crps.mean()),
                   "coh_above_40": None if k in ("naive", "ceiling") else R(np.mean(np.stack(d.coh_curve.values)[:, 1:]), 4),
                   "loud": R(d.loud.mean()), "quiet": R(d.quiet.mean()),
                   **{f"def_{int(round(fc))}": R(v) for fc, v in zip(ex["band_centres_hz"], cur)},
                   **{f"coh_{int(round(fc))}": R(v, 4) for fc, v in zip(ex["band_centres_hz"], coh)}})
    W("bwe", bw)
    pr = json.load(open(res / "predict.json"))
    W("predict", [{"feature": v["label"], "key": k, "r": R(v["r"]), "r_within_lab": R(v["r_within_lab"]),
                   **{f"crpss_{m}": R(s.get("crpss_vs_clim")) for m, s in v["scores"].items() if "crpss_vs_clim" in s},
                   "point_sd_ratio": R(v["scores"]["cda"]["sd_ratio_point"]), "pit90": R(v["scores"]["cda"]["pit_cover90"])}
                  for k, v in pr["features"].items()])
    # ---- the ideas, one row per headline number --------------------------------------------------------------
    I = res / "ideas"; ideas = []
    add = lambda idea, test, value, unit="", t=None, p=None, note="": ideas.append(
        {"idea": idea, "test": test, "value": R(value, 4), "unit": unit, "t": R(t, 2), "p": None if p is None else float(f"{p:.2g}"), "note": note})
    e = json.load(open(I / "engagement.json"))
    add("Engagement vs capacity", "Lab-mean attention vs lab-mean K", e["lab_level"]["attn_K"]["r"], "r", p=e["lab_level"]["attn_K"]["p"])
    add("Engagement vs capacity", "CDA vs attention, between labs", e["lab_level"]["attn"]["r"], "r", p=e["lab_level"]["attn"]["p"])
    add("Engagement vs capacity", "CDA vs capacity on attended trials, between labs", e["lab_level"]["Kc"]["r"], "r", p=e["lab_level"]["Kc"]["p"])
    add("Engagement vs capacity", "CDA ~ attention, within lab (std. beta)", e["within_lab"]["beta_attn"][0], "beta", t=e["within_lab"]["beta_attn"][0] / e["within_lab"]["beta_attn"][1])
    add("Engagement vs capacity", "CDA ~ capacity on attended trials, within lab (std. beta)", e["within_lab"]["beta_Kc"][0], "beta", t=e["within_lab"]["beta_Kc"][0] / e["within_lab"]["beta_Kc"][1])
    tf = json.load(open(I / "trial_forecast.json"))
    add("Single-trial forecast", "Brier skill of trial EEG + eyes over the person's own accuracy", 100 * tf["bss"], "%", note=f"95% CI {100*tf['ci95'][0]:+.2f} to {100*tf['ci95'][1]:+.2f} %, {tf['n_trials']:,} trials")
    hy_ = json.load(open(I / "hysteresis.json"))
    for k, lab in (("cda:prev6", "CDA after a 6-item trial (vs after 2)"), ("rt:prev6", "RT after a 6-item trial (vs after 2), s"),
                   ("correct:prev6", "Accuracy after a 6-item trial (vs after 2)"), ("alpha_pre_lat:same_side", "Pre-array alpha lateralization when the cued side repeats")):
        v = hy_[k]; add("Load hysteresis", lab, v["mean"], "", t=v["t"], p=v["p"], note=f"same sign in {v['labs_same_sign']}/10 labs")
    ew = json.load(open(I / "ews_W10_detrend.json"))
    for k, lab in (("rt:var", "RT variance before lapses"), ("rt:ac1", "RT autocorrelation before lapses"), ("rt:mean", "RT mean before lapses (s)"),
                   ("alpha_pre:mean", "Pre-trial alpha before lapses"), ("cda:ac1", "CDA autocorrelation before lapses")):
        v = ew[k]; add("Early warning before lapses", lab, v["mean"], "", t=v["t"], p=v["p"], note="time-on-task detrended, 10-trial window")
    ey = json.load(open(I / "eyes.json"))
    for k, lab in (("cda46", "CDA vs K within lab"), ("heog_slope", "HEOG load slope vs K within lab"), ("gaze_slope", "Eye-tracker load slope vs K within lab")):
        v = ey[k]; add("Eyes vs CDA", lab, v["r_within_lab"], "r", p=v["p"], note=f"n {v['n']}, out-of-lab CRPS skill {100*v['crpss']:+.1f} %")
    for f, lab in (("hmm.json", None), ("spectral.json", None)):
        if (I / f).exists():
            d = json.load(open(I / f))
            if f == "hmm.json":
                v = d["heldout_dll_per_trial"]; add("Engagement states (HMM)", "Held-out log-likelihood gain over i.i.d., per trial", v["mean"], "nats", t=v["t"], p=v["p"], note=f"better in {100*v['frac_subjects_better']:.0f} % of people")
                for m in ("cda", "alpha_lat", "alpha_pre", "heog", "rt"):
                    v = d[f"{m}_dis_minus_eng"]; add("Engagement states (HMM)", f"{m}: disengaged minus engaged trials", v["mean"], "", t=v["t"], p=v["p"], note=f"n {v['n']}, matched set size")
                for v_ in ("frac_dis", "Keng", "K"):
                    v = d[f"cda_vs_{v_}"]; add("Engagement states (HMM)", f"CDA vs {v_}: between labs / within labs", v["r_within_lab"], "r", p=v["p_within"], note=f"between labs r {v['r_between_labs']:+.2f}")
            else:
                for k in ("exponent", "offset", "iaf", "alpha_peak_pow", "dfa_alpha", "hf_floor"):
                    v = d[k]; add("Spectral traits", f"{k}: r with K within lab", v["r_K_within"], "r", p=v["p_K"], note=f"attention r {v['r_attn_within']:+.2f}, lab ICC {v['icc_lab']:.2f}")
                add("Spectral traits", "Lab identified from 6 spectral numbers", 100 * d["lab_decoding_accuracy"], "%", note=f"chance {100*d['chance']:.0f} %")
    W("ideas", ideas)
    print({p.name: p.stat().st_size for p in out.glob("*.json")})


if __name__ == "__main__":
    main()
