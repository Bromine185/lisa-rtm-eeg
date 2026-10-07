"""Stream-preprocess one EEGManyLabs Vogel & Machizawa (2004) subject straight from GIN.

    python -I src/prep.py OUT_DIR META_DIR sub-Voge0001 [sub-Voge0002 ...] [--workers 6]

Every CDA recording in the dataset is BrainVision, IEEE float32, VECTORIZED (one contiguous block per channel),
and GIN serves no byte ranges. So the binary is read once off the socket, channel by channel, and only two things
are kept; nothing raw ever touches the disk:

  cda   lateral posterior pairs + a horizontal-eye signal (+ eye-tracker gaze X where recorded), whole session,
        resampled to 250 Hz, band-passed 0.1-40 Hz, epoched -0.6..1.4 s around memory-array onset (500 samples).
        The contralateral-minus-ipsilateral difference cancels the recording reference, so none is applied.
  sr    every EEG channel for 180 s from the middle of the task, common average reference, line-noise notches,
        high-pass 0.5 Hz, resampled to 250 Hz: the corpus for the bandwidth-extension port (src/bwe.py) and
        the spatial-interpolation check. Plus the native-rate mean log-PSD, to see each lab's hardware band.

Writes OUT_DIR/<sub>.npz and OUT_DIR/<sub>.json (trial table, channel lists, QC).
"""
import csv, io, json, math, os, pathlib, re, sys, time, urllib.request
import numpy as np
from scipy import signal

BASE = "https://gin.g-node.org/EEGManyLabs/EEGManyLabs_Replication_VogelMachizawa2004_Raw/raw/master"
FS = 250
PAIRS = [("PO7", "PO8"), ("P7", "P8"), ("O1", "O2"), ("P3", "P4"), ("PO3", "PO4"), ("P5", "P6"), ("PO9", "PO10")]
EPOCH = (-150, 350)                     # samples at 250 Hz: -0.6 .. +1.4 s around memory-array onset
SR_SEC, SR_PAD = 180.0, 5.0
ALIAS = {"T5-P7": "P7", "T6-P8": "P8", "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz", "AFZ": "AFz"}
HEOG_PAIRS = [("HEOG_L", "HEOG_R"), ("HEOGL", "HEOGR")]
RESP = {"response_change_correct": ("change", "hit"), "response_nochange_incorrect": ("change", "miss"),
        "response_nochange_correct": ("same", "cr"), "response_change_incorrect": ("same", "fa")}


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** (k + 1))


def canon(name):
    return ALIAS.get(name, ALIAS.get(name.upper(), name))


def parse_vhdr(text):
    def val(key):
        m = re.search(rf"^{key}=(.*)$", text, re.M)
        return m.group(1).strip() if m else None
    assert val("DataOrientation") == "VECTORIZED", val("DataOrientation")
    assert val("BinaryFormat") == "IEEE_FLOAT_32", val("BinaryFormat")
    names, res = [], []
    for m in re.finditer(r"^Ch(\d+)=([^\n]*)$", text.split("[Channel Infos]")[1], re.M):
        f = m.group(2).split(",")
        names.append(f[0].replace(r"\1", ","))
        res.append(float(f[2]) if len(f) > 2 and f[2].strip() else 1.0)
    return dict(names=names, res=np.array(res), n=int(val("DataPoints")), fs=1e6 / float(val("SamplingInterval")))


def trials_from_events(ev):
    """One row per memory array: set size, cued side, change/same, outcome, onset (s)."""
    rows, cue, cur = [], None, None
    for _, e in ev.iterrows():
        tt, t = e["trial_type"], float(e["onset"])
        if tt in ("cue_left", "cue_right"):
            cue = (tt[4:], t)
        elif isinstance(tt, str) and tt.startswith("setsize") and tt.endswith("_onset"):
            cur = {"onset": t, "setsize": int(tt[7]), "side": cue[0] if cue and t - cue[1] < 2.5 else "n/a",
                   "truth": "n/a", "outcome": "none", "rt": float("nan")}
            rows.append(cur)
        elif isinstance(tt, str) and tt.endswith("_test") and cur is not None:
            cur["test"] = t
        elif tt in RESP and cur is not None and cur["outcome"] == "none":
            cur["truth"], cur["outcome"] = RESP[tt]
            cur["rt"] = t - cur.get("test", float("nan"))
    return rows


def stream_channels(url, info, keep_full, sr_slice, sr_set):
    """Read the VECTORIZED float32 body off the socket. Returns {name: full array} and {name: slice array}."""
    n = info["n"]; full, part = {}, {}
    s0, s1 = sr_slice
    with urllib.request.urlopen(url, timeout=300) as r:
        for ci, name in enumerate(info["names"]):
            buf = np.empty(n, np.float32); mv = memoryview(buf).cast("B"); got = 0
            while got < mv.nbytes:
                k = r.readinto(mv[got:min(got + (16 << 20), mv.nbytes)])
                if not k:
                    raise IOError(f"short read on {name}: {got}/{mv.nbytes}")
                got += k
            buf *= info["res"][ci]
            if name in keep_full:
                full[name] = buf
            if name in sr_set:
                part[name] = buf[s0:s1].copy()
    return full, part


def resample(x, fs):
    """x (..., n) at fs -> 250 Hz with the polyphase anti-alias filter. fs need not be an integer."""
    if abs(fs - FS) < 1e-6:
        return x.astype(np.float64)
    frac = __import__("fractions").Fraction(FS / fs).limit_denominator(4096)
    return signal.resample_poly(x.astype(np.float64), frac.numerator, frac.denominator, axis=-1)


def interp_nan(x):
    x = np.asarray(x, np.float64).copy(); bad = ~np.isfinite(x)
    if bad.all():
        return np.zeros_like(x)
    if bad.any():
        idx = np.arange(len(x)); x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def process(sub, out, meta):
    t_start = time.time()
    chan_tsv = (meta / "sidecars" / f"{sub}_channels.tsv").read_text()
    ctype = {r["name"]: r["type"] for r in csv.DictReader(chan_tsv.splitlines(), delimiter="\t")}
    ej = json.loads((meta / "sidecars" / f"{sub}_eeg.json").read_text())
    line = float(ej.get("PowerLineFrequency") or 50)
    vhdr = get(f"{BASE}/{sub}/eeg/{sub}_task-CDA_eeg.vhdr").decode("utf8", "replace")
    info = parse_vhdr(vhdr); fs = info["fs"]
    import pandas as pd
    ev = pd.read_csv(io.BytesIO(get(f"{BASE}/{sub}/eeg/{sub}_task-CDA_events.tsv")), sep="\t")
    try:
        el = pd.read_csv(io.BytesIO(get(f"{BASE}/{sub}/eeg/{sub}_electrodes.tsv")), sep="\t")
        pos = {canon(str(r["name"])): (float(r["x"]), float(r["y"]), float(r["z"])) for _, r in el.iterrows()
               if all(np.isfinite([pd.to_numeric(r[k], errors="coerce") for k in "xyz"]))}
    except Exception:
        pos = {}
    trials = trials_from_events(ev)
    tsv_names = list(ctype)
    if len(tsv_names) == len(info["names"]) and tsv_names != info["names"]:
        info["names"] = tsv_names           # EGI headers say E1..E129; channels.tsv has the 10-10 labels, same order
    names = info["names"]; cn = {canon(n): n for n in names}
    pairs = [(a, b) for a, b in PAIRS if a in cn and b in cn]
    heog = next(((cn[a], cn[b]) for a, b in HEOG_PAIRS if a in cn and b in cn), None)
    heog_kind = "bipolar"
    if heog is None and "HEOG" in cn:
        heog, heog_kind = (cn["HEOG"], None), "single"
    if heog is None and "F7" in cn and "F8" in cn:
        heog, heog_kind = (cn["F7"], cn["F8"]), "F7-F8 proxy"
    gaze = cn.get("L-GAZE-X")
    keep_full = {cn[a] for p in pairs for a in p} | {h for h in (heog or ()) if h} | ({gaze} if gaze else set())
    eeg = [n for n in names if ctype.get(n, "EEG").upper() == "EEG"]
    # SR window: 180 s centred on the task (first block_start .. last block_end)
    bs = ev.loc[ev.trial_type == "block_start", "onset"]; be = ev.loc[ev.trial_type == "block_end", "onset"]
    mid = 0.5 * (float(bs.min()) + float(be.max())) if len(bs) and len(be) else info["n"] / fs / 2
    s0 = max(0, int((mid - SR_SEC / 2 - SR_PAD) * fs)); s1 = min(info["n"], int((mid + SR_SEC / 2 + SR_PAD) * fs))
    full, part = stream_channels(f"{BASE}/{sub}/eeg/{sub}_task-CDA_eeg.eeg", info, keep_full, (s0, s1), set(eeg))
    t_dl = time.time() - t_start

    # ---- CDA epochs -------------------------------------------------------------------------------------
    bp = signal.butter(2, 0.1, "highpass", fs=FS, output="sos"), signal.butter(4, 40.0, "lowpass", fs=FS, output="sos")
    def band(x):
        x = resample(x - np.mean(x[: int(10 * fs)]), fs)
        return signal.sosfiltfilt(bp[1], signal.sosfiltfilt(bp[0], x))
    lat_names = [canon(n) for p in pairs for n in (cn[p[0]], cn[p[1]])]
    lat = np.stack([band(full[cn[n]]) for n in lat_names]) if pairs else np.zeros((0, 1))
    if heog is None:
        h = np.zeros(lat.shape[-1])
    elif heog[1] is None:
        h = band(full[heog[0]])
    else:
        h = band(full[heog[0]].astype(np.float64) - full[heog[1]])
    g = resample(interp_nan(full[gaze]), fs) if gaze else None
    n250 = lat.shape[-1]
    on = np.array([int(round(t["onset"] * FS)) for t in trials])
    ok = (on + EPOCH[0] >= 0) & (on + EPOCH[1] <= n250)
    idx = on[ok][:, None] + np.arange(*EPOCH)[None]
    X = lat[:, idx].transpose(1, 0, 2).astype(np.float32)                     # (trials, chans, 500)
    H = h[idx].astype(np.float32)
    G = g[idx].astype(np.float32) if g is not None else np.zeros((0,), np.float32)
    trials = [t for t, k in zip(trials, ok) if k]

    # ---- SR excerpt -------------------------------------------------------------------------------------
    sr_names = [n for n in eeg if n in part]
    A = np.stack([part[n].astype(np.float64) for n in sr_names])
    A -= A.mean(1, keepdims=True)
    sd = A.std(1)
    good = (sd > 1e-3) & (sd < 50 * np.median(sd))                            # flat (the reference) or dead
    A = A[good]; sr_names = [canon(n) for n, k in zip(sr_names, good) if k]
    A -= A.mean(0, keepdims=True)                                             # common average reference
    f_w, P = signal.welch(A, fs, nperseg=int(2 * fs), axis=-1)
    psd = np.log10(np.median(P, 0) + 1e-30)
    for k in range(1, 20):
        f0 = k * line
        if f0 < min(fs / 2 - 1, 130):
            b, a = signal.iirnotch(f0, 30.0, fs); A = signal.filtfilt(b, a, A, axis=-1)
    A = signal.sosfiltfilt(signal.butter(2, 0.5, "highpass", fs=fs, output="sos"), A, axis=-1)
    A = resample(A, fs)
    p0 = int(round(SR_PAD * FS)); A = A[:, p0:p0 + int(SR_SEC * FS)].astype(np.float32)
    P_xyz = np.array([pos.get(n, (np.nan,) * 3) for n in sr_names], np.float32)

    np.savez(out / f"{sub}.npz", X=X, H=H, G=G, sr=A, sr_pos=P_xyz, psd=psd.astype(np.float32), psd_f=f_w.astype(np.float32))
    js = {"sub": sub, "fs_native": fs, "line": line, "pairs": [list(p) for p in pairs], "lat_names": lat_names,
          "heog": heog_kind if heog else "none", "gaze": bool(gaze), "sr_names": sr_names, "n_trials": len(trials),
          "trials": trials, "t_download": t_dl, "t_total": time.time() - t_start, "sr_ok": True}
    (out / f"{sub}.json").write_text(json.dumps(js))
    return f"{sub}: fs {fs:.0f}, {len(pairs)} pairs, heog {js['heog']}, {len(trials)} trials, sr {A.shape}, " \
           f"{t_dl:.0f}s dl / {js['t_total']:.0f}s"


def safe(args):
    sub, out, meta = args
    if (out / f"{sub}.json").exists():
        return f"{sub}: cached"
    try:
        return process(sub, out, meta)
    except Exception as e:
        (out / f"{sub}.err").write_text(repr(e))
        return f"{sub}: FAILED {e!r}"


def main():
    a = sys.argv[1:]
    workers = 1
    if "--workers" in a:
        i = a.index("--workers"); workers = int(a[i + 1]); del a[i:i + 2]
    out, meta, subs = pathlib.Path(a[0]), pathlib.Path(a[1]), a[2:]
    out.mkdir(parents=True, exist_ok=True)
    if subs == ["all"]:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from splits import round_robin
        subs = round_robin(meta)
    jobs = [(s, out, meta) for s in subs]
    t0 = time.time()
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(safe, jobs)):
                print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(jobs)} {r}", flush=True)
    else:
        for i, j in enumerate(jobs):
            print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(jobs)} {safe(j)}", flush=True)


if __name__ == "__main__":
    main()
