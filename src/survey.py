"""Metadata survey: one row per subject from the BIDS sidecars (no EEG binaries downloaded).

    python -I src/survey.py OUT_DIR
Writes OUT_DIR/survey.tsv and OUT_DIR/sidecars/<sub>_{eeg.json,channels.tsv,files.txt}.
"""
import csv, json, pathlib, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://gin.g-node.org/EEGManyLabs/EEGManyLabs_Replication_VogelMachizawa2004_Raw"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def one(sub, out):
    d = out / "sidecars"; d.mkdir(parents=True, exist_ok=True)
    html = get(f"{BASE}/src/master/{sub}/eeg").decode("utf8", "replace")
    files = sorted(set(re.findall(rf'/src/master/{sub}/eeg/({sub}[^"?]+)"', html)))
    (d / f"{sub}_files.txt").write_text("\n".join(files))
    row = {"sub": sub, "files": ";".join(f.replace(sub + "_", "") for f in files)}
    js = [f for f in files if f.endswith("task-CDA_eeg.json")]
    if js:
        j = json.loads(get(f"{BASE}/raw/master/{sub}/eeg/{js[0]}"))
        (d / f"{sub}_eeg.json").write_text(json.dumps(j, indent=1))
        hw = j.get("HardwareFilters", {}) if isinstance(j.get("HardwareFilters"), dict) else {}
        lp = hw.get("Lowpass", {}) if isinstance(hw.get("Lowpass"), dict) else {}
        row.update(fs=j.get("SamplingFrequency"), ref=j.get("EEGReference"), line=j.get("PowerLineFrequency"),
                   n_eeg=j.get("EEGChannelCount"), n_eog=j.get("EOGChannelCount"), dur=j.get("RecordingDuration"),
                   maker=j.get("Manufacturer"), model=j.get("ManufacturersModelName"), hw_lp=lp.get("CutoffFrequency"),
                   inst=j.get("InstitutionName"))
    ch = [f for f in files if f.endswith("task-CDA_channels.tsv")]
    if ch:
        t = get(f"{BASE}/raw/master/{sub}/eeg/{ch[0]}").decode("utf8", "replace")
        (d / f"{sub}_channels.tsv").write_text(t)
        rows = list(csv.DictReader(t.splitlines(), delimiter="\t"))
        row["channels"] = ",".join(r["name"] for r in rows)
    return row


def main():
    out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    parts = list(csv.DictReader(get(f"{BASE}/raw/master/participants.tsv").decode().splitlines(), delimiter="\t"))
    (out / "participants.tsv").write_bytes(get(f"{BASE}/raw/master/participants.tsv"))
    lab = {p["participant_id"]: p["lab"] for p in parts}
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda s: one(s, out), sorted(lab)))
    for r in rows:
        r["lab"] = lab[r["sub"]]
    keys = ["sub", "lab", "fs", "n_eeg", "n_eog", "ref", "line", "hw_lp", "dur", "maker", "model", "inst", "files", "channels"]
    with open(out / "survey.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} subjects -> {out/'survey.tsv'}")


if __name__ == "__main__":
    main()
