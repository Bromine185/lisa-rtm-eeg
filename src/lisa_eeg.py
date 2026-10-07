"""LISA / LISAS and the lisa-rtm objectives, ported verbatim from lisa-rtm to a 250 Hz single-channel EEG target.

Source of every definition (lisa-rtm, commit cb304b7):
  LISAS, decode, arm_loss (det / det_paper / es_marg)  overnight2/c1_model.py
  LISADecoder, MultiScaleSTFTLoss                       build_notebook.py
  stft, snr_db, third_octave_edges, crps_ensemble, pit_ranks, spread_skill   build_notebook.py
Only the configuration changes: fs_hi 48 kHz -> 250 Hz, n_fft 2048 -> 256 (the training STFT scales are
(256, 64), (128, 32), (64, 16): 1.0 / 0.5 / 0.26 s windows at 250 Hz, because 2048 samples is 8 s of EEG).
The architecture is the 88k-parameter one exactly: encoder (16, 32, 64, 32) x kernels (7, 3, 3, 1), decoder
5 x 144 ReLU MLP on [coord, z_{i-1}, z_i, z_{i+1}], 8 Gaussian noise channels at the encoder input.
"""
from dataclasses import dataclass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import signal as sps


@dataclass
class Config:
    fs_hi: int = 250
    upsample: int = 4
    enc_channels: tuple = (16, 32, 64, 32)
    enc_kernels: tuple = (7, 3, 3, 1)
    dec_hidden: int = 144
    dec_layers: int = 5
    n_fft: int = 256            # training (transport) basis
    eval_n_fft: int = 128       # evaluation basis, deliberately different (as in lisa-rtm)
    eval_hop: int = 32

    @property
    def fs_lo(self):
        return self.fs_hi / self.upsample

    @property
    def eval_k_cut(self):
        return int(np.ceil((self.fs_lo / 2) * self.eval_n_fft / self.fs_hi))


CFG = Config()
N_NOISE = 8


class LISADecoder(nn.Module):
    def __init__(self, latent, cfg):
        super().__init__()
        layers, d = [], 1 + 3 * latent
        for _ in range(cfg.dec_layers - 1):
            layers += [nn.Linear(d, cfg.dec_hidden), nn.ReLU()]
            d = cfg.dec_hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        return self.net(f).squeeze(-1)


class LISAS(nn.Module):
    """LISA with n_noise Gaussian channels concatenated to the waveform at the encoder input. eps = 0 is LISA."""
    def __init__(self, cfg=CFG, n_noise=N_NOISE):
        super().__init__()
        self.n_noise, self.R = n_noise, cfg.upsample
        layers, c_in = [], 1 + n_noise
        for i, (c, k) in enumerate(zip(cfg.enc_channels, cfg.enc_kernels)):
            layers.append(nn.Conv1d(c_in, c, k, padding=k // 2))
            if i < len(cfg.enc_channels) - 1:
                layers.append(nn.ReLU())
            c_in = c
        self.enc, self.dim = nn.Sequential(*layers), c_in
        self.dec = LISADecoder(self.dim, cfg)
        self.tau = 0.0

    def sample_eps(self, x_lo, tau=1.0, generator=None):
        B, L = x_lo.shape
        return tau * torch.randn(B, self.n_noise, L, generator=generator)

    def encode(self, x_lo, eps=None):
        B, L = x_lo.shape
        if eps is None:
            eps = torch.zeros(B, self.n_noise, L, dtype=x_lo.dtype)
        return self.enc(torch.cat([x_lo.unsqueeze(1), eps], 1)).transpose(1, 2)

    def decode(self, z, perturb=False):
        B, L_lo, C = z.shape
        q = (torch.arange(L_lo * self.R, dtype=torch.float32) / self.R).unsqueeze(0).expand(B, -1)
        anchor = q + torch.randn_like(q) * 0.5 if perturb else q
        idx = torch.floor(anchor).long().clamp(0, L_lo - 1)
        coord = (2.0 * (q - idx.float()) - 1.0).unsqueeze(-1)

        def take(ii):
            return torch.gather(z, 1, ii.clamp(0, L_lo - 1).unsqueeze(-1).expand(-1, -1, C))
        return self.dec(torch.cat([coord, take(idx - 1), take(idx), take(idx + 1)], -1))

    def forward(self, x_lo, perturb=False, eps=None):
        return self.decode(self.encode(x_lo, eps), perturb)


# ---- distances and losses (c1_model.py / build_notebook.py) ------------------------------------------
SCALES = [(CFG.n_fft, CFG.n_fft // 4), (CFG.n_fft // 2, CFG.n_fft // 8), (CFG.n_fft // 4, CFG.n_fft // 16)]
_WIN = {}


def _win(n):
    if n not in _WIN:
        _WIN[n] = torch.hann_window(n)
    return _WIN[n]


def logmag_feats(y):
    return [torch.log(torch.stft(y, n, h, window=_win(n), return_complex=True).abs() + 1e-7) for n, h in SCALES]


def d_wave(a, b):
    return (a - b).abs().mean(dim=tuple(range(1, a.ndim)))


def d_logmag(fa, fb):
    return sum((A - B).abs().mean(dim=(1, 2)) for A, B in zip(fa, fb)) / len(fa)


class MultiScaleSTFTLoss(nn.Module):
    def forward(self, y, y_hat):
        total = 0.0
        for n, h in SCALES:
            Y = torch.stft(y, n, h, window=_win(n), return_complex=True).abs()
            H = torch.stft(y_hat, n, h, window=_win(n), return_complex=True).abs()
            sc = torch.norm(Y - H, p="fro") / (torch.norm(Y, p="fro") + 1e-8)
            lm = F.l1_loss(torch.log(H + 1e-7), torch.log(Y + 1e-7))
            total = total + sc + lm
        return total / len(SCALES)


SPEC = MultiScaleSTFTLoss()


def arm_loss(kind, lam, m, x, y):
    """det_paper: L1 (LISA's released config). det: L1 + lam * MSSTFT. es_marg: two-draw energy score with
    d = L1(wave) + lam * L1(log|STFT|), ES = 1/2 [d(y,y1) + d(y,y2)] - 1/2 d(y1,y2)."""
    if kind in ("det", "det_paper"):
        y_hat = m(x, perturb=True)
        l_w = F.l1_loss(y_hat, y)
        if kind == "det_paper":
            return l_w, {"wave": l_w.item(), "spec": 0.0}
        l_s = SPEC(y, y_hat)
        return l_w + lam * l_s, {"wave": l_w.item(), "spec": l_s.item()}
    B = x.shape[0]
    eps = m.sample_eps(x.repeat(2, 1), 1.0)
    yh = m(x.repeat(2, 1), perturb=True, eps=eps)
    y1, y2 = yh[:B], yh[B:]
    dw = 0.5 * (d_wave(y, y1) + d_wave(y, y2)) - 0.5 * d_wave(y1, y2)
    if kind == "es_wave":
        return dw.mean(), {"wave": dw.mean().item(), "spec": 0.0, "spread": d_wave(y1, y2).mean().item()}
    fy, f1, f2 = logmag_feats(y), logmag_feats(y1), logmag_feats(y2)
    ds = 0.5 * (d_logmag(fy, f1) + d_logmag(fy, f2)) - 0.5 * d_logmag(f1, f2)
    return (dw + lam * ds).mean(), {"wave": dw.mean().item(), "spec": ds.mean().item(),
                                     "spread": d_wave(y1, y2).mean().item()}


# ---- numpy metrics (build_notebook.py, verbatim) ------------------------------------------------------
def _hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft(x, n_fft, hop):
    x = np.asarray(x, dtype=np.float64)
    xp = np.pad(x, (n_fft, 2 * n_fft))
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * _hann(n_fft), axis=-1)


def logmag(S, eps=1e-8):
    return np.log(np.abs(S) + eps)


def snr_db(y, y_hat):
    e = np.asarray(y, float) - np.asarray(y_hat, float)
    return 10.0 * np.log10(np.sum(np.asarray(y, float) ** 2) / max(np.sum(e ** 2), 1e-20))


def third_octave_edges(f_lo, f_hi):
    f = [f_lo]
    while f[-1] < f_hi * 0.999:
        f.append(min(f[-1] * 2 ** (1 / 3), f_hi))
    return np.array(f)


def crps_ensemble(samples, truth):
    t1 = np.abs(samples - truth[None]).mean(0)
    d = np.abs(samples[:, None] - samples[None, :]).mean((0, 1))
    return float(np.mean(t1 - 0.5 * d))


def crps_fair(samples, truth):
    M = samples.shape[0]
    t1 = np.abs(samples - truth[None]).mean(0)
    d = np.abs(samples[:, None] - samples[None, :]).sum((0, 1)) / (M * (M - 1))
    return float(np.mean(t1 - 0.5 * d))


def pit_ranks(samples, truth):
    return (samples < truth[None]).sum(0).ravel()


def decimate(x, R=CFG.upsample):
    return sps.resample_poly(np.asarray(x, np.float64), 1, R, axis=-1)


def naive_upsample(x_lo, n, R=CFG.upsample):
    return sps.resample_poly(np.asarray(x_lo, np.float64), R, 1, axis=-1)[..., :n]
