"""Experiment A: EEG bandwidth extension with LISA / LISAS, judged by the lisa-rtm scorer.

Task: 62.5 Hz -> 250 Hz (R = 4, the lisa-rtm ratio). The given band is 0-31.25 Hz, the band every ERP / CDA
analysis keeps; the missing band is 31.25-125 Hz (gamma, muscle). Line-noise bins (50/60/100/120 Hz +- 2 Hz)
are notched in preprocessing and excluded from every high-band (HB) metric.

    python -I src/bwe.py train PREP_DIR META_DIR OUT_DIR [--steps 6000]
    python -I src/bwe.py score PREP_DIR META_DIR OUT_DIR

Split: 3 held-out subjects per lab (30 subjects, ten labs), never seen in training. Each training channel is
scaled by the RMS of its own decimated input, a level known at inference.
"""
import json, pathlib, sys, time, copy
import numpy as np, pandas as pd, torch
from torch.optim.lr_scheduler import MultiStepLR

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lisa_eeg import (CFG, LISAS, arm_loss, decimate, naive_upsample, stft, logmag, snr_db, third_octave_edges,
                      crps_ensemble, crps_fair, pit_ranks)

FS, R = CFG.fs_hi, CFG.upsample
CUT = CFG.fs_lo / 2                      # 31.25 Hz
ALIGN = 27.5                             # gain alignment below this, as score.py aligns below 5.5 kHz
SEG = 500                                # 2 s training segments at 250 Hz
ARMS = {"det_paper": ("det_paper", 0.0), "det": ("det", 0.1), "es_marg": ("es_marg", 0.1)}
LINE = [50.0, 60.0, 100.0, 120.0]
M_DRAWS = 8


def split(prep, meta):
    from splits import bwe_split
    return bwe_split(meta)


def channels(prep, sub, k, rng):
    sr = np.load(prep / f"{sub}.npz")["sr"]
    pick = rng.choice(sr.shape[0], size=min(k, sr.shape[0]), replace=False)
    return sr[np.sort(pick)].astype(np.float64)


class Corpus:
    def __init__(self, prep, subs, k=12, seed=1):
        rng = np.random.default_rng(seed)
        Y = np.concatenate([channels(prep, s, k, rng) for s in subs])
        X = decimate(Y)
        sc = X.std(1, keepdims=True) + 1e-9
        self.Y, self.X = (Y / sc).astype(np.float32), (X / sc).astype(np.float32)
        self.n, self.L = self.Y.shape
        print(f"corpus: {len(subs)} subjects, {self.n} channel recordings, {self.n * self.L / FS / 3600:.1f} h", flush=True)

    def batch(self, rng, B):
        i = rng.integers(self.n, size=B)
        s = rng.integers(0, (self.L - SEG) // R, size=B) * R
        y = self.Y[i[:, None], s[:, None] + np.arange(SEG)]
        x = self.X[i[:, None], (s // R)[:, None] + np.arange(SEG // R)]
        return torch.from_numpy(x), torch.from_numpy(y)

    def hb_psd(self, n=7500, k=400, seed=2):
        """Mean HB power per rfft bin over k random training windows: the shaped-noise control's spectrum."""
        rng = np.random.default_rng(seed)
        i = rng.integers(self.n, size=k); s = rng.integers(0, self.L - n, size=k)
        Yf = np.fft.rfft(self.Y[i[:, None], s[:, None] + np.arange(n)].astype(np.float64), axis=1)
        return (np.abs(Yf) ** 2).mean(0)


def train(prep, meta, out, steps, batch=32, lr=1e-3, clip=1e-3, seed=0):
    torch.manual_seed(seed)
    tr, te, _ = split(prep, meta)
    corpus = Corpus(prep, tr)
    np.save(out / "hb_psd.npy", corpus.hb_psd())
    base = LISAS(CFG)
    models = {k: copy.deepcopy(base) for k in ARMS}
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=lr) for k in ARMS}
    sch = {k: MultiStepLR(opts[k], [int(0.5 * steps), int(0.75 * steps)], 0.5) for k in ARMS}
    hist = {k: [] for k in ARMS}
    rng = np.random.default_rng(seed)
    print(f"LISAS params {sum(p.numel() for p in base.parameters()):,}; arms {ARMS}", flush=True)
    t0 = time.time()
    for step in range(steps):
        x, y = corpus.batch(rng, batch)                    # identical batches for every arm
        for k, (kind, lam) in ARMS.items():
            m = models[k]; m.train()
            loss, terms = arm_loss(kind, lam, m, x, y)
            opts[k].zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opts[k].step(); sch[k].step()
            if step % 50 == 0:
                hist[k].append({"step": step, "loss": loss.item(), **terms})
        if step % 250 == 0 or step + 1 == steps:
            print(f"step {step:>6}/{steps} [{time.time()-t0:5.0f}s] " + " | ".join(
                f"{k}: {hist[k][-1]['loss']:.4f} (w {hist[k][-1]['wave']:.4f} s {hist[k][-1]['spec']:.3f}"
                + (f" sp {hist[k][-1]['spread']:.3f}" if 'spread' in hist[k][-1] else "") + ")" for k in ARMS), flush=True)
    for k, m in models.items():
        torch.save({"model": m.state_dict(), "arm": ARMS[k], "steps": steps}, out / f"{k}.pt")
    json.dump({"hist": hist, "train": tr, "test": te, "wall_s": time.time() - t0}, open(out / "train.json", "w"))


# ---- scoring (sota/score.py, adapted) -----------------------------------------------------------------
def fft_split(x, lo=True, cut=CUT):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS)
    X[(f >= cut) if lo else (f < cut)] = 0
    return np.fft.irfft(X, n=len(x))


def align(y, w):
    Y, W = np.fft.rfft(y), np.fft.rfft(w); lo = np.fft.rfftfreq(len(y), 1 / FS) < ALIGN
    g = np.sqrt(np.sum(np.abs(Y[lo]) ** 2) / max(np.sum(np.abs(W[lo]) ** 2), 1e-30))
    return w * g, float(g)


FR = np.fft.rfftfreq(CFG.eval_n_fft, 1 / FS)
NOLINE = np.all([np.abs(FR - f) > 2.0 for f in LINE], 0)
HBK = (FR >= CUT) & NOLINE
EDGES = third_octave_edges(CUT, FS / 2)
HB_SEL = [(FR >= a) & (FR < b) & NOLINE for a, b in zip(EDGES[:-1], EDGES[1:])]
HB_SEL = [s for s in HB_SEL if s.any()]
HB_CENTRES = [float(np.sqrt(a * b)) for (a, b), s in zip(zip(EDGES[:-1], EDGES[1:]),
              [(FR >= a) & (FR < b) & NOLINE for a, b in zip(EDGES[:-1], EDGES[1:])]) if s.any()]
LB_BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 31.25)]


def S_(w):
    return stft(w, CFG.eval_n_fft, CFG.eval_hop)


def hb_frame_logE(S):
    return np.log10(np.sum(np.abs(S[:, HBK]) ** 2, 1) + 1e-20)


def lb_feats(S):
    return np.stack([np.log10(np.sum(np.abs(S[:, (FR >= a) & (FR < b)]) ** 2, 1) + 1e-20) for a, b in LB_BANDS], 1)


def one(y, w, Sy, masks):
    n = len(y); Y, W = np.fft.rfft(y), np.fft.rfft(w); f = np.fft.rfftfreq(n, 1 / FS)
    hb = (f >= CUT) & np.all([np.abs(f - l) > 2.0 for l in LINE], 0); lo = f < ALIGN
    Sw = S_(w); Py, Pw = np.abs(Sy) ** 2, np.abs(Sw) ** 2
    curve = np.array([10 * np.log10((Pw[:, s].sum() + 1e-20) / (Py[:, s].sum() + 1e-20)) for s in HB_SEL])
    yy = np.array([Py[:, s].sum() for s in HB_SEL]); ww = np.array([Pw[:, s].sum() for s in HB_SEL])
    yw = np.array([np.sum((Sy[:, s] * np.conj(Sw[:, s])).real) for s in HB_SEL])
    r = {"snr": snr_db(y, w), "deficit": float(curve.mean()), "curve": curve.tolist(),
         "deficit_bb": float(10 * np.log10((Pw[:, HBK].sum() + 1e-20) / (Py[:, HBK].sum() + 1e-20))),
         "coh": float(np.mean(yw / np.maximum(yy, 1e-20))), "coh_curve": (yw / np.maximum(yy, 1e-20)).tolist(),
         "kappa": float(np.mean(yw / np.maximum(ww, 1e-20))),
         "coh_fft": float(np.sum((Y[hb] * np.conj(W[hb])).real) / np.sum(np.abs(Y[hb]) ** 2)),
         "lb_coh": float(np.sum((Y[lo] * np.conj(W[lo])).real) / np.sum(np.abs(Y[lo]) ** 2))}
    for k, fr in masks.items():
        r[k] = float(np.mean([10 * np.log10((Pw[fr][:, s].sum() + 1e-20) / (Py[fr][:, s].sum() + 1e-20)) for s in HB_SEL]))
    return r, hb_frame_logE(Sw)


def ensemble(y, D, truth_lm):
    M = len(D); out = {}
    T = fft_split(y, False); X = np.stack([fft_split(d, False) for d in D]); XB = X.mean(0)
    ed = np.mean([np.sum((T - x) ** 2) for x in X]); em = np.sum((T - XB) ** 2); s2 = np.sum((X - XB[None]) ** 2) / (M - 1)
    E, P = float(np.sum(T ** 2)), float(np.mean([np.sum(x ** 2) for x in X]))
    out.update(gap_hb=float(10 * np.log10(ed / em)), gap_indep_hb=float(10 * np.log10((E + P) / (E + P / M))),
               _s2=float(s2), _em=float(em), _E=E, _P=P)
    ens = np.stack([logmag(S_(d))[:, HBK] for d in D])
    out["crps"] = crps_ensemble(ens, truth_lm); out["crps_fair"] = crps_fair(ens, truth_lm)
    out["_pit"] = np.bincount(pit_ranks(ens, truth_lm), minlength=M + 1).tolist()
    out["snr_mean"] = snr_db(y, D.mean(0))
    return out


@torch.no_grad()
def run_model(m, x_lo, n, tau, gen):
    xt = torch.from_numpy(x_lo.astype(np.float32))[None]
    eps = None if tau == 0 else m.sample_eps(xt, tau, generator=gen)
    return m(xt, eps=eps)[0].numpy().astype(np.float64)[:n]


def score(prep, meta, out, n_ch=4, chunks=(0, 2, 4), L=7500):
    torch.set_num_threads(4)
    tj = json.load(open(out / "train.json")); te = tj["test"]
    _, _, lab = split(prep, meta)
    models = {}
    for k in ARMS:
        m = LISAS(CFG); m.load_state_dict(torch.load(out / f"{k}.pt")["model"]); m.eval(); models[k] = m
    hb_psd = np.load(out / "hb_psd.npy")
    fL = np.fft.rfftfreq(L, 1 / FS); hbL = fL >= CUT
    rng = np.random.default_rng(5); gen = torch.Generator().manual_seed(7)
    rows, env, lbf = [], {}, []
    t0 = time.time()
    for si, sub in enumerate(te):
        Ych = channels(prep, sub, n_ch, rng)
        for ci, yfull in enumerate(Ych):
            for c in chunks:
                y = yfull[c * L:(c + 1) * L]
                x = decimate(y); sc = x.std() + 1e-9; y, x = y / sc, x / sc
                nv = naive_upsample(x, L); nv_lo = fft_split(nv, True)
                pt = lambda w: nv_lo + fft_split(w, False)
                Sy = S_(y); truth_lm = logmag(Sy)[:, HBK]
                fe = 20 * np.log10(np.abs(Sy).sum(1) + 1e-12); p25, p75 = np.percentile(fe, [25, 75])
                masks = {"loud": fe >= p75, "mid": (fe >= p25) & (fe < p75), "quiet": fe < p25}
                conds = {"naive": np.stack([nv]), "ceiling": np.stack([fft_split(y, True)])}
                noise = []
                for _ in range(M_DRAWS):
                    Z = (rng.standard_normal(len(fL)) + 1j * rng.standard_normal(len(fL))) / np.sqrt(2)
                    noise.append(nv_lo + np.fft.irfft(np.where(hbL, Z * np.sqrt(hb_psd), 0), n=L))
                conds["shaped_noise"] = np.stack(noise)
                for k, m in models.items():
                    if ARMS[k][0].startswith("det"):
                        conds[k] = np.stack([run_model(m, x, L, 0.0, gen)])
                    else:
                        conds[f"{k} tau0"] = np.stack([run_model(m, x, L, 0.0, gen)])
                        conds[k] = np.stack([run_model(m, x, L, 1.0, gen) for _ in range(M_DRAWS)])
                for k in list(conds):
                    if k.split(" ")[0] in ARMS:
                        conds[f"{k} | pt"] = np.stack([pt(d) for d in conds[k]])
                item = f"{sub}/ch{ci}/c{c}"
                env.setdefault("truth", []).append(hb_frame_logE(Sy)); lbf.append(lb_feats(Sy))
                for name, D in conds.items():
                    al = [align(y, d) for d in D]; D = np.stack([a[0] for a in al])
                    per = [one(y, d, Sy, masks) for d in D]
                    row = {"item": item, "sub": sub, "lab": lab[sub], "cond": name, "M": len(D),
                           "gain": float(np.mean([a[1] for a in al])), "hb_share": float(np.sum(fft_split(y, False) ** 2) / np.sum(y ** 2))}
                    for key in per[0][0]:
                        v = [p[0][key] for p in per]
                        row[key] = np.mean(v, 0).tolist() if isinstance(v[0], list) else float(np.mean(v))
                    env.setdefault(name, []).append(per[0][1])
                    if len(D) >= 2:
                        row.update(ensemble(y, D, truth_lm))
                    else:
                        row["crps"] = row["crps_fair"] = crps_ensemble(logmag(S_(D[0]))[None][:, :, HBK], truth_lm)
                    rows.append(row)
        print(f"  [{time.time()-t0:5.0f}s] {si+1}/{len(te)} {sub} ({lab[sub]})", flush=True)
    # envelope predictability from the low band alone (no model): ridge on training-lab frames, R^2 on test frames
    tr = tj["train"]; rng2 = np.random.default_rng(11); Xtr, ytr = [], []
    for sub in tr[::3]:
        for yfull in channels(prep, sub, 2, rng2):
            y = yfull[:L]; y = y / (decimate(y).std() + 1e-9); S = S_(y)
            Xtr.append(lb_feats(S)); ytr.append(hb_frame_logE(S))
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    A = np.c_[Xtr, np.ones(len(Xtr))]; beta = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ ytr)
    Xte, yte = np.concatenate(lbf), np.concatenate(env["truth"])
    pred = np.c_[Xte, np.ones(len(Xte))] @ beta
    item_id = np.concatenate([np.full(len(e), i) for i, e in enumerate(env["truth"])])
    dm = lambda v: v - pd.Series(v).groupby(item_id).transform("mean").values
    env_pred = {"r2_pooled": float(1 - np.var(yte - pred) / np.var(yte)),
                "r2_within_item": float(1 - np.var(dm(yte) - dm(pred)) / np.var(dm(yte))),
                "r_within_item": float(np.corrcoef(dm(yte), dm(pred))[0, 1]), "n_frames": int(len(yte))}
    env_reg = {}
    t = np.concatenate(env["truth"])
    for name in env:
        if name == "truth":
            continue
        e = np.concatenate(env[name])
        a, b = dm(t), dm(e)
        env_reg[name] = {"slope": float(np.polyfit(a, b, 1)[0]), "r": float(np.corrcoef(a, b)[0, 1]),
                         "sd_ratio": float(b.std() / a.std())} if b.std() > 0 else None
    pd.DataFrame(rows).to_json(out / "score_rows.json", orient="records")
    json.dump({"env_pred_from_lb": env_pred, "env_regression_within_item": env_reg, "band_centres_hz": HB_CENTRES,
               "n_items": len(set(r["item"] for r in rows))}, open(out / "score_extra.json", "w"), indent=1)
    print(json.dumps(env_pred, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    steps = 6000
    if "--steps" in a:
        i = a.index("--steps"); steps = int(a[i + 1]); del a[i:i + 2]
    cmd, prep, meta, out = a[0], pathlib.Path(a[1]), pathlib.Path(a[2]), pathlib.Path(a[3])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == "train":
        train(prep, meta, out, steps)
    elif cmd == "score":
        score(prep, meta, out)
