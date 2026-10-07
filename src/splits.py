"""One deterministic ordering of the 304 subjects, shared by every script.

Per lab, a seeded permutation: the first 3 are the held-out BWE test subjects, the next 8 the BWE training
subjects. src/prep.py processes subjects round-robin across labs in this order, so the BWE subsets exist early.
The CDA analyses use every subject that passes the exclusion rules, regardless of this split.
"""
import numpy as np, pandas as pd

N_TEST, N_TRAIN = 3, 8


def order(meta):
    sv = pd.read_csv(meta / "survey.tsv", sep="\t")
    rng = np.random.default_rng(0)
    per = {lab: list(rng.permutation(sorted(g["sub"]))) for lab, g in sorted(sv.groupby("lab"), key=lambda x: x[0])}
    return per, sv.set_index("sub")["lab"].to_dict()


def bwe_split(meta):
    per, lab = order(meta)
    test = sorted(s for v in per.values() for s in v[:N_TEST])
    train = sorted(s for v in per.values() for s in v[N_TEST:N_TEST + N_TRAIN])
    return train, test, lab


def round_robin(meta):
    per, _ = order(meta)
    out, i = [], 0
    while any(i < len(v) for v in per.values()):
        out += [v[i] for v in per.values() if i < len(v)]
        i += 1
    return out
