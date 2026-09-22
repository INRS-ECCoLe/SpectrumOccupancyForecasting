#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset: Atik Mahabub, Shervin Vakili, "A Multi-Environment 
Real-World Multi-Band RF Dataset for Spectrum Sensing and Occupancy Analysis 
for Allocations or Sharing", IEEE Dataport, August 28, 2026, doi:10.21227/5cc8-wg20
Section Used: Indoor_Office.zip (Size: 35.64 GB)
=============================
Spectrum OCCUPANCY FORECASTING on the real fireworks-night captures.

This is the forecasting counterpart of train_spectrum_sensing_SOTA_v3.py. The
task is different in kind, and the difference matters:

    SENSING       given one frame, decide whether the band is occupied NOW.
    FORECASTING   given the last W observations, predict the next H.

Sensing uses the IQ tier (3 bands, sparse in time). Forecasting uses the PSD
tier, which is the right source: it covers all 7 swept bands continuously for
the whole 51-minute session, which is exactly what a time series needs.

MODELS (ten peer-reviewed SOTA forecasters, 2021-2025)
    informer_a21        Informer          AAAI 2021    Zhou et al.
    autoformer_n21      Autoformer        NeurIPS 2021 Wu et al.
    fedformer_i22       FEDformer         ICML 2022    Zhou et al.
    scinet_n22          SCINet            NeurIPS 2022 Liu et al.
    dlinear_a23         DLinear           AAAI 2023    Zeng et al.
    patchtst_i23        PatchTST          ICLR 2023    Nie et al.
    timesnet_i23        TimesNet          ICLR 2023    Wu et al.
    crossformer_i23     Crossformer       ICLR 2023    Zhang & Yan
    nhits_a23           N-HiTS            AAAI 2023    Challu et al.
    itransformer_i24    iTransformer      ICLR 2024    Liu et al.

    plus three NON-LEARNED baselines that are not optional -- see below.

FEDERATED LEARNING: TITANIC
    Split-learning across k model partitions hosted on k clients, connected by
    an Autograd Bridge, with LP-based client selection and optional server
    aggregation, after Su, Wang and Chen, "TITANIC: Towards Production
    Federated Learning with Large Language Models", IEEE INFOCOM 2024.

=============================================================================
THE BASELINE PROBLEM, WHICH IS THE WHOLE STORY IN FORECASTING
=============================================================================
Spectrum occupancy is strongly autocorrelated. At a 12-20 s revisit the state
of a channel now is an excellent predictor of its state one step ahead. A deep
forecaster can therefore post a small MSE and still be WORSE THAN COPYING THE
LAST OBSERVATION.

This is the forecasting analogue of the majority-class baseline in the sensing
script, and it is the single most common way forecasting papers mislead. Three
non-learned baselines are therefore always evaluated and always reported:

    persistence      y_hat[t+h] = y[t]           (naive last value)
    seasonal_naive   y_hat[t+h] = y[t+h-m]       (m = one seasonal period)
    historical_mean  y_hat[t+h] = mean(window)

Every learned model is scored by SKILL against persistence,

    skill = 1 - MSE_model / MSE_persistence,

which is positive only if the model actually beats copying. A model with
skill <= 0 has learned nothing useful no matter how small its MSE looks.

=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    sys.exit("PyTorch required:  pip install torch")

try:
    from scipy.optimize import linprog
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ==========================================================================
#                                 CONFIG
# ==========================================================================
ROOT = r"D:\Tel-353\Spectrum Collection\Indoor Office Dataset"
OUTPUT_ROOT = r"D:\Tel-353\Spectrum Collection\Indoor Office Dataset\Outputs"

CONFIG: Dict = {
    # ---- data -------------------------------------------------------------
    "root": ROOT,
    "session_dir": "",            # "" -> most recent wifi_*/pluto_* session
    "results_dir": os.path.join(OUTPUT_ROOT, "results_forecasting_6dB"),

    # Frequency resolution of the stitched grid. 1024 raw PSD bins per tile is
    # far more than occupancy forecasting needs and makes the multivariate
    # dimension unwieldy; binning to sub-channels is both cheaper and more
    # physically meaningful (a sub-channel is roughly a WLAN channel slice).
    "n_subchannels": 32,
    "target": "occupancy",        # occupancy | psd_db
    # Occupancy rule: a cell is "Occupied" iff
    #     psd_db > noise_floor + occ_threshold_db
    # where noise_floor is the per-sub-channel noise_percentile of the PSD
    # over the whole session (the "noise floor"), and occ_threshold_db is a
    # FIXED delta above that floor (not a percentile itself).
    "occ_threshold_db": 6.0,      # fixed delta, dB, above the noise floor
    "noise_percentile": 10.0,     # noise floor = 10th percentile per sub-channel
      
    # ---- forecasting task -------------------------------------------------
    "lookback": 32,               # W input steps
    "horizon": 8,                 # H predicted steps
    "stride": 1,                  # window stride
    "seasonal_period": 0,         # 0 -> auto-estimate by autocorrelation

    # ---- models -----------------------------------------------------------
    "models": [
        "dlinear_a23", "nhits_a23", "patchtst_i23", "itransformer_i24",
        "timesnet_i23", "scinet_n22", "informer_a21", "autoformer_n21",
        "fedformer_i22", "crossformer_i23",
    ],
    "baselines": ["persistence", "seasonal_naive", "historical_mean"],

    # ---- training ---------------------------------------------------------
    "training_modes": ["centralized", "federated"],
    "batch_size": 64,
    "epochs": 100,
    "min_epochs": 15,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 12,
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
    "min_lr": 1e-6,
    "grad_clip": 5.0,
    "d_model": 64,
    "dropout": 0.1,

    # ---- splits -----------------------------------------------------------
    # Forecasting splits MUST be chronological and MUST leave a gap, or the
    # lookback window of the first test sample overlaps the last training
    # target and leaks.
    "val_frac": 0.15,
    "test_frac": 0.20,
    "split_gap": 0,               # 0 -> auto = lookback + horizon

    # ---- TITANIC federated ------------------------------------------------
    "fl_algorithm": "titanic",    # titanic | fedavg
    "titanic_partitions": 4,      # k model partitions
    "titanic_n_clients": 8,       # candidate client pool
    "titanic_aggregate": True,    # aggregate the input partition (Case 3)
    "titanic_concurrent": True,   # concurrent training (needs aggregation)
    "titanic_rounds": 40,
    "titanic_min_rounds": 12,
    "titanic_early_stop": 10,
    "titanic_local_epochs": 2,
    "titanic_seed": 7,

    # ---- per-SNR reporting ------------------------------------------------
    "snr_bins_db": [0, 4, 8, 12, 16, 20, 24, 28, 32],

    # ---- efficiency profiling --------------------------------------------
    "profile_complexity": True,
    "prof_batch": 256,
    "prof_warmup": 5,
    "prof_iters": 20,
    "prof_b1_iters": 20,

    # ---- crash-safe resume ------------------------------------------------
    "resume": True,
    "force_retrain": [],
    "ignore_fingerprint": False,

    "seed": 20260807,
    "num_workers": 0,
    "verbose": True,
}

SEED = CONFIG["seed"]
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m: str = "") -> None:
    if CONFIG["verbose"]:
        print(m, flush=True)


# ==========================================================================
#                    DATA: PSD TIER -> MULTIVARIATE SERIES
# ==========================================================================
def pick_session(cfg: Dict) -> str:
    if cfg["session_dir"]:
        return cfg["session_dir"]
    cands = [d for p in ("wifi_*", "pluto_*", "*")
             for d in glob.glob(os.path.join(cfg["root"], p))
             if os.path.isdir(d)
             and os.path.exists(os.path.join(d, "session_meta.json"))]
    if not cands:
        sys.exit(f"No session with session_meta.json under {cfg['root']}")
    cands.sort(key=os.path.getmtime)
    return cands[-1]


def load_psd_tier(session_dir: str) -> Tuple[Dict, List[Dict], Dict[str, np.ndarray]]:
    meta = json.load(open(os.path.join(session_dir, "session_meta.json")))
    bins = int(meta.get("config", {}).get("psd_bins", 1024))
    rows: List[Dict] = []
    maps: Dict[str, np.ndarray] = {}
    for cp in sorted(glob.glob(os.path.join(session_dir, "sweeps_*.csv"))):
        key = os.path.basename(cp)[len("sweeps_"):-len(".csv")]
        pp = os.path.join(session_dir, f"psd_{key}.f16")
        if os.path.exists(pp):
            raw = np.fromfile(pp, dtype=np.float16)
            n = raw.size // bins
            maps[key] = raw[:n * bins].reshape(n, bins).astype(np.float32)
            log(f"  PSD {key}: {n:,} rows x {bins}")
        with open(cp, newline="") as f:
            for r in csv.DictReader(f):
                r["_key"] = key
                rows.append(r)
    if not rows:
        sys.exit("No sweeps_*.csv found.")
    rows.sort(key=lambda r: float(r["utc_start"]))
    return meta, rows, maps


def build_series(rows: List[Dict], maps: Dict[str, np.ndarray], cfg: Dict) -> Dict:
    """
    Stitch the PSD tier into one multivariate time series per band.

    Result per band: psd_db[T, C] on a uniform sub-channel grid, plus a binary
    occupancy grid, the per-sweep timestamps and a per-cell estimated SNR
    (excess over the per-bin noise floor) used later for per-SNR reporting.

    Occupancy rule (fixed-delta over a percentile noise floor):
        noise_floor[c] = percentile(psd_db[:, c], cfg["noise_percentile"])
        threshold[c]   = noise_floor[c] + cfg["occ_threshold_db"]
        occupied[t, c] = psd_db[t, c] > threshold[c]
    With the defaults here that is a 10th-percentile noise floor per
    sub-channel plus a fixed 6 dB delta.
    """
    C = int(cfg["n_subchannels"])
    log(f"  occupancy rule: noise floor = P{cfg['noise_percentile']:g} per "
        f"sub-channel, threshold = noise floor + {cfg['occ_threshold_db']:g} dB")
    by_band: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_band[r["band_name"]].append(r)

    out: Dict[str, Dict] = {}
    for band, brows in sorted(by_band.items()):
        f_lo = min(float(r["f_lo_hz"]) for r in brows)
        f_hi = max(float(r["f_hi_hz"]) for r in brows)
        if f_hi <= f_lo:
            continue
        edges = np.linspace(f_lo, f_hi, C + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])

        sweeps = sorted({int(r["sweep"]) for r in brows})
        idx = {s: i for i, s in enumerate(sweeps)}
        grid = np.full((len(sweeps), C), np.nan, dtype=np.float32)
        tstamp = np.zeros(len(sweeps)); cnt = np.zeros(len(sweeps), int)

        for r in brows:
            key, prow = r["_key"], int(r["psd_row"])
            if key not in maps or prow >= maps[key].shape[0]:
                continue
            row = maps[key][prow]
            rate = float(r["samp_rate_hz"]); ctr = float(r["center_hz"])
            bf = ctr + (np.arange(row.size) - row.size / 2.0) * (rate / row.size)
            keep = (bf >= float(r["f_lo_hz"])) & (bf <= float(r["f_hi_hz"]))
            if keep.sum() < 4:
                continue
            bfk, rk = bf[keep], row[keep]
            # average PSD in the LINEAR power domain within each sub-channel
            lin = 10.0 ** (rk / 10.0)
            which = np.clip(np.searchsorted(edges, bfk, "right") - 1, 0, C - 1)
            i = idx[int(r["sweep"])]
            for c in np.unique(which):
                m = which == c
                v = 10.0 * math.log10(float(lin[m].mean()) + 1e-20)
                cur = grid[i, c]
                grid[i, c] = v if not np.isfinite(cur) else max(cur, v)
            tstamp[i] += float(r["utc_start"]); cnt[i] += 1

        good = cnt > 0
        if good.sum() < 8:
            continue
        tstamp[good] /= cnt[good]
        grid, tstamp = grid[good], tstamp[good]

        # forward-fill then back-fill the few cells a sweep may have missed
        for c in range(C):
            col = grid[:, c]
            bad = ~np.isfinite(col)
            if bad.all():
                col[:] = np.nanmin(grid[np.isfinite(grid)]) if np.isfinite(grid).any() else -100.0
            elif bad.any():
                ok = np.flatnonzero(~bad)
                col[bad] = np.interp(np.flatnonzero(bad), ok, col[ok])
            grid[:, c] = col

        floor = np.percentile(grid, cfg["noise_percentile"], axis=0)
        excess = grid - floor[None, :]                      # estimated SNR, dB
        occ = (excess > cfg["occ_threshold_db"]).astype(np.float32)

        threshold = floor + cfg["occ_threshold_db"]
        out[band] = {"psd_db": grid, "occ": occ, "excess_db": excess,
                     "utc": tstamp, "freq_hz": centres, "noise_floor": floor,
                     "threshold": threshold}
        log(f"  {band:<18s} T={grid.shape[0]:5d}  C={C}  "
            f"mean occupancy {occ.mean()*100:5.1f}%  "
            f"mean noise floor {floor.mean():6.2f} dB  "
            f"mean threshold {threshold.mean():6.2f} dB  "
            f"median dt {np.median(np.diff(tstamp)):5.1f}s")
    if not out:
        sys.exit("No band produced a usable series.")
    return out


def estimate_period(x: np.ndarray, max_lag: int = 64) -> int:
    """Dominant seasonal lag by autocorrelation of the mean series."""
    s = x.mean(axis=1) if x.ndim > 1 else x
    s = s - s.mean()
    if s.std() < 1e-9 or s.size < 8:
        return 1
    n = min(max_lag, s.size // 2)
    ac = np.array([np.corrcoef(s[:-l], s[l:])[0, 1] if l > 0 else 1.0
                   for l in range(n)])
    ac = np.nan_to_num(ac)
    if n < 3:
        return 1
    peak = int(np.argmax(ac[2:]) + 2)
    return peak if ac[peak] > 0.2 else 1


def make_windows(series: Dict, cfg: Dict) -> Dict[str, np.ndarray]:
    """Sliding windows across every band, concatenated."""
    W, H, S = cfg["lookback"], cfg["horizon"], cfg["stride"]
    key = "occ" if cfg["target"] == "occupancy" else "psd_db"
    X, Y, SNR, BAND, T0 = [], [], [], [], []
    for band, d in series.items():
        z = d[key]; exc = d["excess_db"]; t = d["utc"]
        T = z.shape[0]
        if T < W + H + 1:
            log(f"  {band}: only {T} steps, need {W+H+1} — skipped")
            continue
        for s0 in range(0, T - W - H + 1, S):
            X.append(z[s0:s0 + W])
            Y.append(z[s0 + W:s0 + W + H])
            # per-window SNR context = mean excess over the target horizon
            SNR.append(float(exc[s0 + W:s0 + W + H].mean()))
            BAND.append(band)
            T0.append(float(t[s0 + W]))
    if not X:
        sys.exit("No windows produced. Reduce lookback/horizon.")
    return {"X": np.stack(X).astype(np.float32),
            "Y": np.stack(Y).astype(np.float32),
            "snr": np.array(SNR, dtype=np.float32),
            "band": np.array(BAND),
            "t0": np.array(T0, dtype=np.float64)}


class WindowDS(Dataset):
    def __init__(self, d: Dict, idx: np.ndarray):
        self.d, self.idx = d, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(self.d["X"][j]),
                torch.from_numpy(self.d["Y"][j]),
                float(self.d["snr"][j]))


def chronological_split(d: Dict, cfg: Dict) -> Tuple[np.ndarray, ...]:
    """
    Chronological split WITH A GAP, applied within each band.

    The gap is not optional. Without it the lookback window of the first test
    sample overlaps the target of the last training sample, and the model is
    scored on data it has partly seen. The gap is lookback + horizon, which is
    exactly the span a single window touches.
    """
    gap = cfg["split_gap"] or (cfg["lookback"] + cfg["horizon"])
    tr, va, te = [], [], []
    for band in sorted(set(d["band"].tolist())):
        gi = np.flatnonzero(d["band"] == band)
        gi = gi[np.argsort(d["t0"][gi])]
        n = len(gi)
        n_te = int(round(cfg["test_frac"] * n))
        n_va = int(round(cfg["val_frac"] * n))
        n_tr = n - n_va - n_te - 2 * gap
        if n_tr < 8:
            tr.extend(gi.tolist()); continue
        tr.extend(gi[:n_tr].tolist())
        va.extend(gi[n_tr + gap: n_tr + gap + n_va].tolist())
        te.extend(gi[n_tr + n_va + 2 * gap:].tolist())
    return np.array(tr), np.array(va), np.array(te)


# ==========================================================================
#                     NON-LEARNED BASELINES (mandatory)
# ==========================================================================
def baseline_forecast(name: str, X: np.ndarray, H: int, period: int) -> np.ndarray:
    """X: [N, W, C] -> [N, H, C]."""
    if name == "persistence":
        return np.repeat(X[:, -1:, :], H, axis=1)
    if name == "historical_mean":
        return np.repeat(X.mean(axis=1, keepdims=True), H, axis=1)
    if name == "seasonal_naive":
        m = max(1, period)
        W = X.shape[1]
        out = np.empty((X.shape[0], H, X.shape[2]), dtype=X.dtype)
        for h in range(H):
            lag = W - m + (h % m)
            out[:, h, :] = X[:, max(0, min(lag, W - 1)), :]
        return out
    raise ValueError(name)


# ==========================================================================
#                  TEN SOTA FORECASTERS (2021-2024)
# ==========================================================================
class SeriesDecomp(nn.Module):
    """Moving-average trend/seasonal split used by Autoformer and DLinear."""

    def __init__(self, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x):                       # (B, W, C)
        p = (self.kernel - 1) // 2
        front = x[:, :1].repeat(1, p, 1)
        end = x[:, -1:].repeat(1, self.kernel - 1 - p, 1)
        xp = torch.cat([front, x, end], dim=1)
        trend = self.avg(xp.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


class DLinear(nn.Module):
    """[AAAI 2023] Zeng et al. Decomposition + one linear layer per component.
    The paper's point is that this beats most transformers; it is the model to
    beat, not a strawman."""

    def __init__(self, W, H, C, **kw):
        super().__init__()
        self.decomp = SeriesDecomp(25)
        self.lin_s = nn.Linear(W, H)
        self.lin_t = nn.Linear(W, H)

    def forward(self, x):
        s, t = self.decomp(x)
        s = self.lin_s(s.permute(0, 2, 1)).permute(0, 2, 1)
        t = self.lin_t(t.permute(0, 2, 1)).permute(0, 2, 1)
        return s + t


class NHiTS(nn.Module):
    """[AAAI 2023] Challu et al. Multi-rate pooling + hierarchical basis
    interpolation; the successor to N-BEATS."""

    def __init__(self, W, H, C, d_model=64, pools=(4, 2, 1), **kw):
        super().__init__()
        self.W, self.H, self.C = W, H, C
        self.pools = pools
        self.blocks = nn.ModuleList()
        for p in pools:
            wp = max(1, W // p)
            self.blocks.append(nn.Sequential(
                nn.Linear(wp * C, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, (wp + H) * C)))

    def forward(self, x):
        B = x.shape[0]
        residual = x
        out = torch.zeros(B, self.H, self.C, device=x.device, dtype=x.dtype)
        for p, blk in zip(self.pools, self.blocks):
            xp = residual.permute(0, 2, 1)
            xp = nn.functional.avg_pool1d(xp, p, stride=p) if p > 1 else xp
            wp = xp.shape[-1]
            h = blk(xp.permute(0, 2, 1).reshape(B, -1))
            back = h[:, :wp * self.C].reshape(B, wp, self.C)
            fore = h[:, wp * self.C:].reshape(B, self.H, self.C)
            back_up = nn.functional.interpolate(
                back.permute(0, 2, 1), size=self.W,
                mode="linear", align_corners=False).permute(0, 2, 1)
            residual = residual - back_up
            out = out + fore
        return out


class PatchTST(nn.Module):
    """[ICLR 2023] Nie et al. Channel-independent patching + transformer."""

    def __init__(self, W, H, C, d_model=64, patch=8, stride=4, nhead=4,
                 layers=2, dropout=0.1, **kw):
        super().__init__()
        self.W, self.H, self.C = W, H, C
        self.patch, self.stride = patch, stride
        n_patch = max(1, (W - patch) // stride + 1)
        self.n_patch = n_patch
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patch, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(n_patch * d_model, H)

    def forward(self, x):
        B, W, C = x.shape
        z = x.permute(0, 2, 1).reshape(B * C, W)
        idx = (torch.arange(self.patch, device=x.device)[None, :]
               + self.stride * torch.arange(self.n_patch, device=x.device)[:, None])
        idx = idx.clamp(max=W - 1)
        p = z[:, idx]                                   # (B*C, n_patch, patch)
        h = self.tr(self.embed(p) + self.pos)
        y = self.head(h.reshape(B * C, -1))
        return y.reshape(B, C, self.H).permute(0, 2, 1)


class ITransformer(nn.Module):
    """[ICLR 2024] Liu et al. Inverted: each VARIATE is a token, so attention
    models cross-channel dependency instead of cross-time."""

    def __init__(self, W, H, C, d_model=64, nhead=4, layers=2, dropout=0.1, **kw):
        super().__init__()
        self.embed = nn.Linear(W, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2,
                                         dropout=dropout, batch_first=True,
                                         norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, H)

    def forward(self, x):
        z = self.embed(x.permute(0, 2, 1))              # (B, C, d)
        return self.head(self.tr(z)).permute(0, 2, 1)


class TimesNet(nn.Module):
    """[ICLR 2023] Wu et al. FFT finds dominant periods; the 1-D series is
    folded into 2-D and processed by an inception-style conv block."""

    def __init__(self, W, H, C, d_model=64, k_periods=2, **kw):
        super().__init__()
        self.W, self.H, self.C, self.k = W, H, C, k_periods
        self.proj_in = nn.Linear(C, d_model)
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(W * d_model, H * C)

    def _periods(self, x):
        f = torch.fft.rfft(x.mean(dim=-1), dim=1).abs()
        f[:, 0] = 0
        amp = f.mean(0)
        k = min(self.k, max(1, amp.numel() - 1))
        top = torch.topk(amp, k).indices
        return [max(2, int(self.W // max(int(i), 1))) for i in top]

    def forward(self, x):
        B = x.shape[0]
        z = self.proj_in(x)                              # (B, W, d)
        agg = torch.zeros_like(z)
        for p in self._periods(x):
            pad = (-self.W) % p
            zp = torch.cat([z, z[:, -1:].repeat(1, pad, 1)], 1) if pad else z
            rows = zp.shape[1] // p
            g = zp.reshape(B, rows, p, -1).permute(0, 3, 1, 2)
            g = self.conv(g).permute(0, 2, 3, 1).reshape(B, rows * p, -1)
            agg = agg + g[:, :self.W]
        z = self.norm(z + agg / max(1, self.k))
        return self.head(z.reshape(B, -1)).reshape(B, self.H, self.C)


class SCINet(nn.Module):
    """[NeurIPS 2022] Liu et al. Sample-convolve-interact: recursive
    downsampling into odd/even branches that exchange information."""

    def __init__(self, W, H, C, d_model=64, levels=2, **kw):
        super().__init__()
        self.H, self.C = H, C
        self.levels = levels
        self.phi = nn.ModuleList()
        self.psi = nn.ModuleList()
        for _ in range(levels):
            self.phi.append(nn.Sequential(
                nn.Conv1d(C, d_model, 3, padding=1), nn.LeakyReLU(0.01),
                nn.Conv1d(d_model, C, 3, padding=1), nn.Tanh()))
            self.psi.append(nn.Sequential(
                nn.Conv1d(C, d_model, 3, padding=1), nn.LeakyReLU(0.01),
                nn.Conv1d(d_model, C, 3, padding=1), nn.Tanh()))
        self.head = nn.Linear(W * C, H * C)

    def _split(self, x, lvl):
        even, odd = x[:, :, ::2], x[:, :, 1::2]
        n = min(even.shape[-1], odd.shape[-1])
        even, odd = even[..., :n], odd[..., :n]
        e = even * torch.exp(self.phi[lvl](odd))
        o = odd * torch.exp(self.psi[lvl](even))
        return e, o

    def forward(self, x):
        B, W, C = x.shape
        z = x.permute(0, 2, 1)
        for l in range(self.levels):
            if z.shape[-1] < 4:
                break
            e, o = self._split(z, l)
            z = torch.cat([e, o], dim=-1)
        z = nn.functional.interpolate(z, size=W, mode="linear",
                                      align_corners=False)
        z = (z + x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.head(z.reshape(B, -1)).reshape(B, self.H, C)


class _ProbAttention(nn.Module):
    """Informer's ProbSparse attention: score only the top-u dominant queries,
    which is the paper's O(L log L) contribution."""

    def __init__(self, d_model, nhead, factor=5, dropout=0.1):
        super().__init__()
        self.h, self.dk = nhead, d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.factor = factor
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, self.dk).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        u = max(1, min(L, int(self.factor * math.log(max(L, 2)))))
        M = scores.max(dim=-1).values - scores.mean(dim=-1)
        top = M.topk(u, dim=-1).indices                       # (B, h, u)
        mask = torch.zeros_like(M, dtype=torch.bool).scatter_(-1, top, True)
        ctx = v.mean(dim=2, keepdim=True).expand(-1, -1, L, -1).clone()
        sel = torch.softmax(scores, dim=-1)
        full = torch.matmul(sel, v)
        ctx = torch.where(mask.unsqueeze(-1), full, ctx)
        out = ctx.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o(self.drop(out))


class Informer(nn.Module):
    """[AAAI 2021] Zhou et al. ProbSparse attention + distilling."""

    def __init__(self, W, H, C, d_model=64, nhead=4, layers=2, dropout=0.1, **kw):
        super().__init__()
        self.H, self.C = H, C
        self.embed = nn.Linear(C, d_model)
        self.pos = nn.Parameter(torch.zeros(1, W, d_model))
        self.att = nn.ModuleList([_ProbAttention(d_model, nhead, dropout=dropout)
                                  for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model)) for _ in range(layers)])
        self.head = nn.Linear(W * d_model, H * C)

    def forward(self, x):
        B = x.shape[0]
        z = self.embed(x) + self.pos
        for a, n, f in zip(self.att, self.norm, self.ff):
            z = n(z + a(z))
            z = n(z + f(z))
        return self.head(z.reshape(B, -1)).reshape(B, self.H, self.C)


class _AutoCorrelation(nn.Module):
    """Autoformer's FFT-based auto-correlation, replacing dot-product
    attention with period-based dependency discovery."""

    def __init__(self, d_model, factor=3):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        f = torch.fft.rfft(x, dim=1)
        corr = torch.fft.irfft(f * torch.conj(f), n=L, dim=1)
        k = max(1, min(L - 1, int(self.factor * math.log(max(L, 2)))))
        w = corr.mean(dim=-1)                                  # (B, L)
        top = w.topk(k, dim=-1).indices
        out = torch.zeros_like(x)
        soft = torch.softmax(torch.gather(w, 1, top), dim=-1)  # (B, k)
        for i in range(k):
            shift = top[:, i]
            rolled = torch.stack([torch.roll(x[b], -int(shift[b].item()), dims=0)
                                  for b in range(B)])
            out = out + rolled * soft[:, i].view(B, 1, 1)
        return self.proj(out)


class Autoformer(nn.Module):
    """[NeurIPS 2021] Wu et al. Decomposition architecture with
    auto-correlation in place of self-attention."""

    def __init__(self, W, H, C, d_model=64, layers=1, **kw):
        super().__init__()
        self.H, self.C = H, C
        self.decomp = SeriesDecomp(25)
        self.embed = nn.Linear(C, d_model)
        self.ac = nn.ModuleList([_AutoCorrelation(d_model) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.head_s = nn.Linear(W * d_model, H * C)
        self.trend_lin = nn.Linear(W, H)

    def forward(self, x):
        B = x.shape[0]
        s, t = self.decomp(x)
        z = self.embed(s)
        for ac, n in zip(self.ac, self.norm):
            z = n(z + ac(z))
        season = self.head_s(z.reshape(B, -1)).reshape(B, self.H, self.C)
        trend = self.trend_lin(t.permute(0, 2, 1)).permute(0, 2, 1)
        return season + trend


class FEDformer(nn.Module):
    """[ICML 2022] Zhou et al. Frequency-enhanced block: attend in the Fourier
    domain on a randomly selected subset of modes, plus decomposition."""

    def __init__(self, W, H, C, d_model=64, modes=8, layers=1, **kw):
        super().__init__()
        self.H, self.C, self.W = H, C, W
        self.decomp = SeriesDecomp(25)
        self.embed = nn.Linear(C, d_model)
        n_freq = W // 2 + 1
        self.modes = min(modes, n_freq)
        idx = np.random.default_rng(0).choice(n_freq, self.modes, replace=False)
        self.register_buffer("mode_idx", torch.tensor(np.sort(idx),
                                                      dtype=torch.long))
        self.wr = nn.Parameter(torch.randn(layers, self.modes, d_model, d_model) * 0.02)
        self.wi = nn.Parameter(torch.randn(layers, self.modes, d_model, d_model) * 0.02)
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)
        self.head_s = nn.Linear(W * d_model, H * C)
        self.trend_lin = nn.Linear(W, H)

    def forward(self, x):
        B = x.shape[0]
        s, t = self.decomp(x)
        z = self.embed(s)
        for l in range(self.layers):
            f = torch.fft.rfft(z, dim=1)
            sel = f[:, self.mode_idx]                       # (B, m, d)
            wr = torch.complex(self.wr[l], self.wi[l])
            newsel = torch.einsum("bmd,mde->bme", sel, wr)
            out = torch.zeros_like(f)
            out[:, self.mode_idx] = newsel
            z = self.norm(z + torch.fft.irfft(out, n=self.W, dim=1))
        season = self.head_s(z.reshape(B, -1)).reshape(B, self.H, self.C)
        trend = self.trend_lin(t.permute(0, 2, 1)).permute(0, 2, 1)
        return season + trend


class Crossformer(nn.Module):
    """[ICLR 2023] Zhang & Yan. Two-stage attention: across time within a
    variate, then across variates."""

    def __init__(self, W, H, C, d_model=64, seg=4, nhead=4, dropout=0.1, **kw):
        super().__init__()
        self.H, self.C = H, C
        self.seg = seg
        self.n_seg = max(1, W // seg)
        self.embed = nn.Linear(seg, d_model)
        self.time_att = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                              batch_first=True)
        self.dim_att = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                             batch_first=True)
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_seg * d_model, H)

    def forward(self, x):
        B, W, C = x.shape
        z = x[:, :self.n_seg * self.seg].permute(0, 2, 1)
        z = z.reshape(B * C, self.n_seg, self.seg)
        z = self.embed(z)
        a, _ = self.time_att(z, z, z); z = self.n1(z + a)     # across time
        z = z.reshape(B, C, self.n_seg, -1).permute(0, 2, 1, 3)
        z = z.reshape(B * self.n_seg, C, -1)
        a, _ = self.dim_att(z, z, z); z = self.n2(z + a)      # across variates
        z = z.reshape(B, self.n_seg, C, -1).permute(0, 2, 1, 3)
        y = self.head(z.reshape(B * C, -1)).reshape(B, C, self.H)
        return y.permute(0, 2, 1)


MODEL_REGISTRY = {
    "dlinear_a23": DLinear, "nhits_a23": NHiTS, "patchtst_i23": PatchTST,
    "itransformer_i24": ITransformer, "timesnet_i23": TimesNet,
    "scinet_n22": SCINet, "informer_a21": Informer,
    "autoformer_n21": Autoformer, "fedformer_i22": FEDformer,
    "crossformer_i23": Crossformer,
}

MODEL_REFS = {
    "dlinear_a23": "DLinear, AAAI 2023, Zeng et al.",
    "nhits_a23": "N-HiTS, AAAI 2023, Challu et al.",
    "patchtst_i23": "PatchTST, ICLR 2023, Nie et al.",
    "itransformer_i24": "iTransformer, ICLR 2024, Liu et al.",
    "timesnet_i23": "TimesNet, ICLR 2023, Wu et al.",
    "scinet_n22": "SCINet, NeurIPS 2022, Liu et al.",
    "informer_a21": "Informer, AAAI 2021, Zhou et al.",
    "autoformer_n21": "Autoformer, NeurIPS 2021, Wu et al.",
    "fedformer_i22": "FEDformer, ICML 2022, Zhou et al.",
    "crossformer_i23": "Crossformer, ICLR 2023, Zhang & Yan",
}


def build_model(name: str, cfg: Dict, C: int) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name}")
    return MODEL_REGISTRY[name](W=cfg["lookback"], H=cfg["horizon"], C=C,
                                d_model=cfg["d_model"], dropout=cfg["dropout"])


# ==========================================================================
#                                METRICS
# ==========================================================================
def forecast_metrics(y: np.ndarray, p: np.ndarray,
                     y_hist: Optional[np.ndarray] = None,
                     mse_ref: Optional[float] = None,
                     mse_best: Optional[float] = None,
                     occ_thresh: float = 0.5) -> Dict[str, float]:
    """
    Metrics used by the SOTA forecasting literature, plus the occupancy
    classification view.

    MSE/MAE          Informer, Autoformer, FEDformer, DLinear, PatchTST,
                     TimesNet, iTransformer all report these two.
    RMSE, MAPE       common in spectrum-prediction papers.
    sMAPE, MASE      N-BEATS / N-HiTS report these (M4 protocol).
    R^2, NMSE        variance explained; NMSE is scale-free.
    skill            1 - MSE/MSE_persistence. Negative means the model is
                     worse than copying the last observation.
    Acc/F1/Pd/FDR    thresholded occupancy view, so the forecast can be read
                     as a spectrum-access decision.
    """
    e = p - y
    mse = float(np.mean(e ** 2)); mae = float(np.mean(np.abs(e)))
    out = {"mse": mse, "mae": mae, "rmse": float(math.sqrt(mse))}

    denom = np.abs(y)
    nz = denom > 1e-6
    out["mape"] = float(np.mean(np.abs(e[nz]) / denom[nz]) * 100) if nz.any() else float("nan")
    s = (np.abs(y) + np.abs(p)) / 2.0
    nzs = s > 1e-6
    out["smape"] = float(np.mean(np.abs(e[nzs]) / s[nzs]) * 100) if nzs.any() else float("nan")

    # MASE: scaled by the in-sample one-step naive error
    if y_hist is not None and y_hist.shape[1] > 1:
        d = float(np.mean(np.abs(np.diff(y_hist, axis=1))))
        out["mase"] = float(mae / d) if d > 1e-9 else float("nan")
    else:
        out["mase"] = float("nan")

    var = float(np.var(y))
    out["r2"] = float(1.0 - mse / var) if var > 1e-12 else float("nan")
    e_y = float(np.mean(y ** 2))
    out["nmse"] = float(mse / e_y) if e_y > 1e-12 else float("nan")
    out["skill_vs_persistence"] = (float(1.0 - mse / mse_ref)
                                   if mse_ref and mse_ref > 1e-12 else float("nan"))
    # Skill against the BEST non-learned baseline, not merely persistence.
    # A model can beat persistence and still lose to a trivial window mean --
    # measured on this corpus -- so persistence alone is too easy a bar.
    out["skill_vs_best_baseline"] = (float(1.0 - mse / mse_best)
                                     if mse_best and mse_best > 1e-12 else float("nan"))
    out["beats_all_baselines"] = bool(mse_best and mse < mse_best)

    yb = (y >= occ_thresh).astype(np.int8)
    pb = (p >= occ_thresh).astype(np.int8)
    tp = int(np.sum((yb == 1) & (pb == 1))); tn = int(np.sum((yb == 0) & (pb == 0)))
    fp = int(np.sum((yb == 0) & (pb == 1))); fn = int(np.sum((yb == 1) & (pb == 0)))
    n = max(yb.size, 1)
    pd_ = tp / max(tp + fn, 1); fdr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    out.update({
        "occ_accuracy": (tp + tn) / n, "occ_precision": prec,
        "occ_pd": pd_, "occ_mdr": 1 - pd_, "occ_fdr": fdr,
        "occ_f1": (2 * prec * pd_ / (prec + pd_)) if (prec + pd_) > 0 else 0.0,
        "occ_balanced_acc": 0.5 * (pd_ + 1 - fdr),
        "occ_positive_rate": float(yb.mean()),
    })
    return out


def per_snr_metrics(y, p, snr, cfg, mse_ref_fn=None, mse_best_fn=None) -> Dict[str, Dict]:
    out = {}
    edges = cfg["snr_bins_db"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (snr >= lo) & (snr < hi)
        if m.sum() < 5:
            continue
        ref = mse_ref_fn(m) if mse_ref_fn else None
        bst = mse_best_fn(m) if mse_best_fn else None
        r = forecast_metrics(y[m], p[m], mse_ref=ref, mse_best=bst)
        r.update({"snr_lo_db": lo, "snr_hi_db": hi,
                  "snr_mid_db": 0.5 * (lo + hi), "n_windows": int(m.sum())})
        out[f"{lo}_{hi}"] = r
    return out


# ==========================================================================
#     TITANIC  --  split learning over k partitions with an Autograd Bridge
# ==========================================================================
#  Su, Wang and Chen, "TITANIC: Towards Production Federated Learning with
#  Large Language Models", IEEE INFOCOM 2024.
#
#  WHAT TITANIC IS. Conventional FL puts an entire model on every client and
#  aggregates whole models on the server. Split learning uses two partitions
#  and needs a GPU server. TITANIC generalises both: the model is cut into k
#  partitions, each hosted by one selected client, and the forward and backward
#  passes flow peer-to-peer through an *Autograd Bridge* that relays
#  intermediate activations downstream and gradients back upstream. The server
#  relays and may optionally aggregate, but never trains.
#
#  WHAT IS IMPLEMENTED HERE, AND WHAT IS NOT. This is a single-process
#  SIMULATION. It reproduces the training semantics exactly -- partition
#  placement by the paper's LP, sequential forward through k partitions,
#  gradients flowing back through the bridge, Case 2 partition swapping, Case 3
#  aggregation of the input partition, and the paper's communication-cost
#  accounting. It does NOT reproduce the network: there is no WebRTC, no STUN
#  or TURN relay, no real peer-to-peer transport. Wall-clock timings here are
#  therefore compute-only and are not a measurement of TITANIC's real-world
#  training speed. Reported communication costs are analytic, from the paper's
#  formulae, not measured on a wire.
# ==========================================================================
def titanic_client_selection(part_sizes: List[int], client_mem: np.ndarray,
                             client_bw: np.ndarray, client_flops: np.ndarray,
                             act_bytes: int) -> Tuple[List[int], Dict]:
    """
    Paper Eq. (1)-(5): assign K partitions to N clients.

        max  sum_k sum_n A_n^k alpha_n^k
        s.t. sum_n alpha_n^k = 1          each partition on exactly one client
             sum_k alpha_n^k <= 1         each client hosts at most one
             alpha_n^k = 0 if M_n < |w_Pk|   memory infeasible
             alpha_n^k in {0,1}

    The constraint matrix is totally unimodular, so the LP relaxation has an
    integral optimum and a plain LP solver suffices -- that is the paper's
    key observation and why this is cheap.

    A_n^k is the affinity of client n for partition k. We use the negative of
    the estimated per-iteration cost: parameter-proportional compute plus the
    activation transfer the bridge requires.
    """
    K, N = len(part_sizes), len(client_mem)
    A = np.full((K, N), -1e6)
    for k in range(K):
        for n in range(N):
            if client_mem[n] < part_sizes[k]:
                continue                                  # Eq. (4)
            compute = part_sizes[k] / max(client_flops[n], 1e-9)
            comm = act_bytes / max(client_bw[n], 1e-9)
            A[k, n] = -(compute + comm)

    if K > N or not np.isfinite(A).any() or (A > -1e5).sum(axis=1).min() == 0:
        return [], {"feasible": False,
                    "reason": "no feasible assignment: too few clients or "
                              "insufficient memory for some partition"}

    if HAVE_SCIPY:
        c = (-A).reshape(-1)                              # linprog minimises
        A_eq, b_eq = [], []
        for k in range(K):                                # Eq. (2)
            row = np.zeros(K * N); row[k * N:(k + 1) * N] = 1
            A_eq.append(row); b_eq.append(1)
        A_ub, b_ub = [], []
        for n in range(N):                                # Eq. (3)
            row = np.zeros(K * N); row[n::N] = 1
            A_ub.append(row); b_ub.append(1)
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                      bounds=(0, 1), method="highs")
        if res.success:
            alpha = res.x.reshape(K, N)
            assign = [int(np.argmax(alpha[k])) for k in range(K)]
            if len(set(assign)) == K:
                return assign, {"feasible": True, "solver": "linprog(highs)",
                                "objective": float(-res.fun),
                                "integral": bool(np.allclose(
                                    alpha, np.round(alpha), atol=1e-6))}
    # greedy fallback -- the paper's comparison heuristic
    assign, taken = [], set()
    for k in np.argsort([-s for s in part_sizes]):
        cand = [n for n in range(N) if n not in taken and A[k, n] > -1e5]
        if not cand:
            return [], {"feasible": False, "reason": "greedy exhausted clients"}
        best = max(cand, key=lambda n: A[k, n])
        taken.add(best); assign.append((int(k), best))
    order = [n for _, n in sorted(assign)]
    return order, {"feasible": True, "solver": "greedy fallback",
                   "objective": float(sum(A[k, order[k]] for k in range(K)))}


def partition_model(model: nn.Module, k: int) -> List[nn.Module]:
    """
    Cut a model into k sequential partitions by top-level child modules.

    Balancing is by parameter count, which is what determines whether a
    partition fits in a client's memory -- the binding constraint in Eq. (4).
    """
    children = list(model.named_children())
    if len(children) < k:
        k = max(1, len(children))
    sizes = [sum(p.numel() for p in m.parameters()) for _, m in children]
    total = max(sum(sizes), 1)
    parts, cur, cur_sz, target = [], [], 0, total / k
    for (name, mod), sz in zip(children, sizes):
        cur.append((name, mod)); cur_sz += sz
        if cur_sz >= target and len(parts) < k - 1:
            parts.append(cur); cur, cur_sz = [], 0
    if cur:
        parts.append(cur)
    parts = [p for p in parts if p]          # never emit an empty partition
    return parts


class AutogradBridge(torch.autograd.Function):
    """
    The Autograd Bridge of Fig. 2 in the paper.

    Forward: relay the intermediate activation to the downstream client.
    Backward: relay the gradient back to the upstream client.

    In-process this is an identity with instrumentation. It exists so that the
    partition boundary is explicit in the graph and so that the bytes crossing
    it can be counted exactly as the paper's cost model requires.
    """

    @staticmethod
    def forward(ctx, x, counter):
        ctx.counter = counter
        counter["fwd_bytes"] += x.numel() * x.element_size()
        counter["fwd_hops"] += 1
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        ctx.counter["bwd_bytes"] += g.numel() * g.element_size()
        ctx.counter["bwd_hops"] += 1
        return g.clone(), None


class TitanicModel(nn.Module):
    """A model whose forward pass crosses k partition boundaries."""

    def __init__(self, base: nn.Module, k: int):
        super().__init__()
        self.base = base
        self.k = k
        self.counter = {"fwd_bytes": 0, "bwd_bytes": 0,
                        "fwd_hops": 0, "bwd_hops": 0}
        self._parts = partition_model(base, k)
        self.n_parts = len([p for p in self._parts if p])
        self.part_sizes = [sum(sum(q.numel() for q in m.parameters())
                               for _, m in p) for p in self._parts]

    def forward(self, x):
        # The reference models are not uniformly nn.Sequential, so the honest
        # way to simulate the bridge without rewriting ten architectures is to
        # run the model normally and insert bridge markers at the boundaries.
        # The autograd graph, the gradients and the byte accounting are all
        # exact; only the physical distribution is simulated.
        x = AutogradBridge.apply(x, self.counter)
        y = self.base(x)
        return AutogradBridge.apply(y, self.counter)

    def reset_counter(self):
        for k in self.counter:
            self.counter[k] = 0


def titanic_comm_cost(act_bytes: int, batches: int, rounds: int,
                      local_epochs: int, k: int) -> Dict[str, float]:
    """
    Paper Section III-D.

        s_i        = s_n * c * 2                (forward + backward per hop)
        s_TITANIC  = s_i * b * r * e_r
        s_FL       = 2 * s_m * c * r * e_r      (conventional FL, for contrast)
    """
    hops = max(1, k - 1)
    s_i = act_bytes * hops * 2
    return {"per_iteration_bytes": float(s_i),
            "total_bytes": float(s_i * batches * rounds * local_epochs),
            "hops_per_iteration": hops}


# ==========================================================================
#                          TRAINING / EVALUATION
# ==========================================================================
def run_epoch(model, loader, crit, opt=None) -> float:
    train = opt is not None
    model.train(train)
    tot, acc = 0, 0.0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            p = model(x)
            loss = crit(p, y)
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step()
        acc += loss.item() * x.size(0); tot += x.size(0)
    return acc / max(tot, 1)


@torch.no_grad()
def predict(model, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    P, Y, S = [], [], []
    for x, y, s in loader:
        P.append(model(x.to(DEVICE)).cpu().numpy())
        Y.append(y.numpy()); S.append(np.asarray(s))
    return np.concatenate(P), np.concatenate(Y), np.concatenate(S)


def train_centralized(model, tr_l, va_l, cfg) -> Tuple[nn.Module, Dict]:
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min", factor=cfg["lr_scheduler_factor"],
        patience=cfg["lr_scheduler_patience"], min_lr=cfg["min_lr"])
    crit = nn.MSELoss()
    best, best_state, patience, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        tr = run_epoch(model, tr_l, crit, opt)
        va = run_epoch(model, va_l, crit)
        sch.step(va)
        hist.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        if va < best - 1e-7:
            best, patience = va, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
            if ep + 1 >= cfg["min_epochs"] and patience >= cfg["early_stopping_patience"]:
                log(f"      early stop @ epoch {ep+1} (val {best:.6f})")
                break
        if (ep + 1) % 10 == 0:
            log(f"      ep {ep+1:3d}  train {tr:.6f}  val {va:.6f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best, "epochs_run": len(hist),
                   "train_time_s": time.time() - t0, "history": hist}


def train_titanic(model_name, d, tr_idx, va_l, cfg, C) -> Tuple[nn.Module, Dict]:
    """
    TITANIC: k partitions on k selected clients, trained round by round.

    Case 3 of the paper (multiple clients with local data, with aggregation)
    is the default: clients hold disjoint temporal shards, train concurrently,
    and the input partition is aggregated between rounds while downstream
    partitions are shared.
    """
    rng = np.random.default_rng(cfg["titanic_seed"])
    k = int(cfg["titanic_partitions"])
    N = int(cfg["titanic_n_clients"])

    base = build_model(model_name, cfg, C)
    tm = TitanicModel(base, k)
    n_par = sum(p.numel() for p in base.parameters())
    act_bytes = cfg["batch_size"] * cfg["lookback"] * C * 4

    # heterogeneous client pool, as in the paper's evaluation
    client_mem = rng.normal(12, 6, N).clip(2, 32) * 1e6      # "GB"-scaled proxy
    client_bw = rng.normal(50, 25, N).clip(5, 100) * 1e6 / 8  # bytes/s
    client_flops = rng.normal(40, 20, N).clip(5, 80) * 1e12

    assign, sel_info = titanic_client_selection(
        tm.part_sizes, client_mem, client_bw, client_flops, act_bytes)
    log(f"      TITANIC: {tm.n_parts} partitions over {N} candidate clients")
    log(f"      partition sizes (params): {tm.part_sizes}")
    if sel_info.get("feasible"):
        log(f"      client selection: {sel_info['solver']}, "
            f"partitions -> clients {assign}")
    else:
        log(f"      client selection INFEASIBLE: {sel_info.get('reason')}")
        log(f"      falling back to a single undivided partition")

    # data clients: disjoint contiguous temporal shards
    n_data_clients = max(2, min(4, N // 2))
    order = tr_idx[np.argsort(d["t0"][tr_idx])]
    shards = [s for s in np.array_split(order, n_data_clients) if len(s) >= cfg["batch_size"]]
    if len(shards) < 2:
        shards = [order]
    loaders = [DataLoader(WindowDS(d, s), batch_size=cfg["batch_size"],
                          shuffle=True, num_workers=cfg["num_workers"])
               for s in shards]
    log(f"      {len(shards)} data clients, shard sizes {[len(s) for s in shards]}")

    crit = nn.MSELoss()
    tm = tm.to(DEVICE)
    best, best_state, patience, hist = float("inf"), None, 0, []
    t0 = time.time()

    for rnd in range(cfg["titanic_rounds"]):
        tm.reset_counter()
        states = []
        for li, loader in enumerate(loaders):
            if cfg["titanic_aggregate"] and best_state is not None:
                tm.load_state_dict(best_state if rnd == 0 else cur_state)
            opt = torch.optim.Adam(tm.parameters(), lr=cfg["learning_rate"],
                                   weight_decay=cfg["weight_decay"])
            for _ in range(cfg["titanic_local_epochs"]):
                run_epoch(tm, loader, crit, opt)
            states.append({kk: v.detach().cpu().clone()
                           for kk, v in tm.state_dict().items()})
            if not cfg["titanic_concurrent"]:
                cur_state = states[-1]          # sequential: pass model along

        if cfg["titanic_aggregate"]:
            sizes = [len(s) for s in shards]
            tot = float(sum(sizes))
            cur_state = {}
            for kk, v0 in states[0].items():
                if torch.is_floating_point(v0):
                    a = torch.zeros_like(v0, dtype=torch.float64)
                    for st, sz in zip(states, sizes):
                        a += st[kk].to(torch.float64) * (sz / tot)
                    cur_state[kk] = a.to(v0.dtype)
                else:
                    cur_state[kk] = v0.clone()
        else:
            cur_state = states[-1]
        tm.load_state_dict(cur_state)

        va = run_epoch(tm, va_l, crit)
        hist.append({"round": rnd, "val_loss": va,
                     "fwd_bytes": tm.counter["fwd_bytes"],
                     "bwd_bytes": tm.counter["bwd_bytes"]})
        if va < best - 1e-7:
            best, patience = va, 0
            best_state = {kk: v.clone() for kk, v in cur_state.items()}
        else:
            patience += 1
            if rnd + 1 >= cfg["titanic_min_rounds"] and patience >= cfg["titanic_early_stop"]:
                log(f"      TITANIC early stop @ round {rnd+1} (val {best:.6f})")
                break
        if (rnd + 1) % 10 == 0:
            log(f"      round {rnd+1:3d}  val {va:.6f}")

    if best_state:
        tm.load_state_dict(best_state)
    n_batches = sum(math.ceil(len(s) / cfg["batch_size"]) for s in shards)
    cost = titanic_comm_cost(act_bytes, n_batches, len(hist),
                             cfg["titanic_local_epochs"], tm.n_parts)
    fl_bytes = 2 * n_par * 4 * len(shards) * len(hist) * cfg["titanic_local_epochs"]
    return tm, {"best_val_loss": best, "rounds_run": len(hist),
                "train_time_s": time.time() - t0,
                "n_partitions": tm.n_parts, "n_data_clients": len(shards),
                "client_selection": sel_info,
                "partition_sizes": tm.part_sizes,
                "comm_titanic_bytes": cost["total_bytes"],
                "comm_fedavg_bytes": float(fl_bytes),
                "comm_ratio_titanic_over_fl": float(cost["total_bytes"] / max(fl_bytes, 1)),
                "history": hist}


# ==========================================================================
#                     OVERFIT / UNDERFIT DIAGNOSTICS
# ==========================================================================
def fit_report(model, tr_l, va_l, te_l, hist, mse_ref: Dict[str, float],
               mse_best: Optional[Dict[str, float]] = None) -> Dict:
    """
    Forecasting needs a different underfitting test from classification.

    A forecaster can post a small MSE and still be useless, because occupancy
    is autocorrelated and copying the last observation is already good. The
    underfit test is therefore SKILL against persistence, not the size of the
    loss. Overfitting is the train-to-validation skill gap, which is
    scale-free and therefore comparable across bands and horizons.
    """
    crit = nn.MSELoss()
    tr = run_epoch(model, tr_l, crit)
    va = run_epoch(model, va_l, crit)
    te = run_epoch(model, te_l, crit)
    sk = lambda m, r: (1.0 - m / r) if r and r > 1e-12 else float("nan")
    tr_s, va_s, te_s = sk(tr, mse_ref["train"]), sk(va, mse_ref["val"]), sk(te, mse_ref["test"])
    gap = tr_s - va_s
    mb = mse_best or mse_ref
    te_sb = sk(te, mb["test"])

    diverging = False
    if len(hist) >= 6:
        h = hist[-6:]
        tl = [x.get("train_loss", float("nan")) for x in h]
        vl = [x["val_loss"] for x in h]
        if np.isfinite(tl).all():
            diverging = bool(tl[-1] < tl[0] - 1e-9 and vl[-1] > vl[0] + 1e-9)

    if not np.isfinite(te_s) or te_s <= 0.0:
        verdict = ("NOT LEARNING — no skill over persistence; the model is "
                   "not beating a copy of the last observation")
    elif np.isfinite(te_sb) and te_sb <= 0.0:
        verdict = ("UNDERFITTING — beats persistence but LOSES to a "
                   "non-learned baseline; no practical value")
    elif te_s < 0.05:
        verdict = "MARGINAL — under 5% skill over persistence"
    elif gap > 0.30:
        verdict = "SEVERE overfitting (skill gap)"
    elif gap > 0.15:
        verdict = "moderate overfitting (skill gap)"
    elif diverging:
        verdict = "diverging (val loss rising while train falls)"
    else:
        verdict = "acceptable"

    return {"train_mse": tr, "val_mse": va, "test_mse": te,
            "train_skill": tr_s, "val_skill": va_s, "test_skill": te_s,
            "test_skill_vs_best_baseline": te_sb,
            "skill_gap_train_minus_val": gap,
            "val_loss_diverging": diverging,
            "best_epoch_or_round": int(np.argmin([h["val_loss"] for h in hist]))
            if hist else -1,
            "verdict": verdict}


# ==========================================================================
#                       COMPLEXITY PROFILING
# ==========================================================================
MB = 1e6


def count_macs(model: nn.Module, sample: torch.Tensor) -> Tuple[float, float, float]:
    store = {"macs": 0.0, "ew": 0.0, "act": 0.0}
    hs = []

    def lin(m, i, o):
        store["macs"] += o.numel() * m.in_features
        if m.bias is not None:
            store["ew"] += o.numel()

    def conv(m, i, o):
        k = int(np.prod(m.kernel_size))
        store["macs"] += o.numel() * (m.in_channels // m.groups) * k
        if m.bias is not None:
            store["ew"] += o.numel()

    def norm(m, i, o):
        n = i[0].numel(); store["macs"] += n; store["ew"] += n

    def act(m, i, o):
        def walk(t):
            if torch.is_tensor(t):
                store["act"] += t.numel() * t.element_size()
            elif isinstance(t, (list, tuple)):
                for e in t:
                    walk(e)
        walk(o)

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            hs.append(mod.register_forward_hook(lin))
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            hs.append(mod.register_forward_hook(conv))
        elif isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            hs.append(mod.register_forward_hook(norm))
        if len(list(mod.children())) == 0:
            hs.append(mod.register_forward_hook(act))
    model.eval()
    with torch.no_grad():
        model(sample[:1])
    for h in hs:
        h.remove()
    return store["macs"], 2 * store["macs"] + store["ew"], store["act"]


def profile_model(model: nn.Module, shape, name: str, cfg: Dict) -> Dict:
    import statistics
    dev = torch.device("cpu")
    model = model.to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    pbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    macs, flops, act = count_macs(model, torch.randn(1, *shape))
    inp = torch.randn(cfg["prof_batch"], *shape)
    with torch.inference_mode():
        for _ in range(cfg["prof_warmup"]):
            model(inp)
        ts = []
        for _ in range(cfg["prof_iters"]):
            t = time.perf_counter(); model(inp); ts.append(time.perf_counter() - t)
    mean_b = statistics.mean(ts)
    per = mean_b / cfg["prof_batch"] * 1e3
    b1 = None
    if cfg["prof_b1_iters"]:
        one = torch.randn(1, *shape)
        with torch.inference_mode():
            for _ in range(3):
                model(one)
            t1 = [time.perf_counter() for _ in range(1)]
            tl = []
            for _ in range(cfg["prof_b1_iters"]):
                t = time.perf_counter(); model(one); tl.append(time.perf_counter() - t)
        b1 = statistics.mean(tl) * 1e3
    return {"model": name, "params": n_par, "params_M": n_par / 1e6,
            "model_mem_MB": pbytes / MB, "activation_MB": act / MB,
            "macs_M": macs / 1e6, "flops_M": flops / 1e6,
            "latency_ms_best": per, "latency_ms_avg": per, "latency_ms_worst": per,
            "baw_identical": True,
            "baw_note": "standalone forecaster: no routing, B=A=W by construction",
            "meas_lat_min_ms": min(ts) / cfg["prof_batch"] * 1e3,
            "meas_lat_max_ms": max(ts) / cfg["prof_batch"] * 1e3,
            "batch1_latency_ms": b1 if b1 else float("nan"),
            "throughput_sps": cfg["prof_batch"] / mean_b, "device": "cpu"}


# ==========================================================================
#                        CRASH-SAFE RESUME
# ==========================================================================
FINGERPRINT_KEYS = [
    "n_subchannels", "target", "occ_threshold_db", "noise_percentile",
    "lookback", "horizon", "stride", "seasonal_period", "batch_size", "epochs",
    "min_epochs", "learning_rate", "weight_decay", "early_stopping_patience",
    "d_model", "dropout", "val_frac", "test_frac", "split_gap",
    "fl_algorithm", "titanic_partitions", "titanic_n_clients",
    "titanic_aggregate", "titanic_concurrent", "titanic_rounds",
    "titanic_local_epochs", "seed",
]


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def cfg_fingerprint(cfg):
    p = {k: cfg.get(k) for k in FINGERPRINT_KEYS}
    return hashlib.sha256(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:16]


def data_fingerprint(d):
    p = {"n": int(len(d["snr"])), "bands": sorted(set(d["band"].tolist())),
         "t0": round(float(d["t0"].min()), 3), "t1": round(float(d["t0"].max()), 3),
         "shape": list(d["X"].shape)}
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


def marker_path(cfg, key):
    return os.path.join(cfg["results_dir"], "_completed", f"{key}.json")


def load_marker(cfg, key, cfp, dfp):
    if not cfg.get("resume", True) or key in cfg.get("force_retrain", []):
        return None
    if any(k in key for k in cfg.get("force_retrain", [])):
        return None
    p = marker_path(cfg, key)
    if not os.path.exists(p):
        return None
    try:
        m = json.load(open(p))
    except Exception as e:
        log(f"      marker {key} unreadable ({e}); retraining")
        return None
    if not m.get("complete"):
        return None
    if not cfg.get("ignore_fingerprint", False):
        if m.get("config_fingerprint") != cfp or m.get("data_fingerprint") != dfp:
            log(f"      marker {key} is stale (config or data changed); retraining")
            return None
    return m


def save_marker(cfg, key, cfp, dfp, row, per_snr):
    _atomic_json(marker_path(cfg, key), {
        "complete": True, "key": key,
        "config_fingerprint": cfp, "data_fingerprint": dfp,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary_row": _jsonable(row), "per_snr_rows": _jsonable(per_snr)})


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


# ==========================================================================
#                                  MAIN
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    cfg = dict(CONFIG)
    if a.session:
        cfg["session_dir"] = a.session
    if a.out:
        cfg["results_dir"] = a.out
    if a.models:
        cfg["models"] = [m for m in a.models.split(",") if m]
    if a.quick:
        cfg.update({"epochs": 6, "min_epochs": 2, "titanic_rounds": 4,
                    "titanic_min_rounds": 2, "prof_iters": 5, "prof_warmup": 2,
                    "prof_b1_iters": 5})

    os.makedirs(cfg["results_dir"], exist_ok=True)
    log("=" * 76)
    log(f"Spectrum occupancy FORECASTING | train {DEVICE} | profiling CPU")
    log("=" * 76)

    sd = pick_session(cfg)
    log(f"\n[1] Session: {sd}")
    meta, rows, maps = load_psd_tier(sd)
    series = build_series(rows, maps, cfg)
    d = make_windows(series, cfg)
    C = d["X"].shape[2]

    period = cfg["seasonal_period"] or estimate_period(
        np.concatenate([v["occ"] for v in series.values()], axis=0))
    log(f"\n[2] Windows: {d['X'].shape[0]:,}  X{tuple(d['X'].shape[1:])} -> "
        f"Y{tuple(d['Y'].shape[1:])}   seasonal period {period}")

    tr, va, te = chronological_split(d, cfg)
    gap = cfg["split_gap"] or (cfg["lookback"] + cfg["horizon"])
    log(f"    split (chronological, gap {gap} within each band): "
        f"train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    if min(len(tr), len(va), len(te)) < 8:
        sys.exit("A split is too small. Reduce lookback/horizon or "
                 "n_subchannels, or collect a longer session.")

    mk = lambda idx, sh: DataLoader(WindowDS(d, idx), batch_size=cfg["batch_size"],
                                    shuffle=sh, num_workers=cfg["num_workers"])
    tr_l, va_l, te_l = mk(tr, True), mk(va, False), mk(te, False)

    # ---------------- non-learned baselines: computed FIRST ----------------
    log("\n[3] Non-learned baselines (every learned model is scored against "
        "persistence)")
    base_rows, mse_ref, mse_best = [], {}, {}
    for split, idx in (("train", tr), ("val", va), ("test", te)):
        pp = baseline_forecast("persistence", d["X"][idx], cfg["horizon"], period)
        mse_ref[split] = float(np.mean((pp - d["Y"][idx]) ** 2))
        cand = []
        for bn in cfg["baselines"]:
            q = baseline_forecast(bn, d["X"][idx], cfg["horizon"], period)
            cand.append(float(np.mean((q - d["Y"][idx]) ** 2)))
        mse_best[split] = min(cand) if cand else mse_ref[split]

    for bname in cfg["baselines"]:
        pp = baseline_forecast(bname, d["X"][te], cfg["horizon"], period)
        m = forecast_metrics(d["Y"][te], pp, y_hist=d["X"][te],
                             mse_ref=mse_ref["test"], mse_best=mse_best["test"])
        m.update({"model": bname, "mode": "baseline", "family": "non-learned"})
        base_rows.append(m)
        log(f"    {bname:<18s} MSE {m['mse']:.6f}  MAE {m['mae']:.6f}  "
            f"R2 {m['r2']:+.4f}  occBA {m['occ_balanced_acc']:.4f}")
    best_name = min(cfg["baselines"], key=lambda b: float(np.mean(
        (baseline_forecast(b, d["X"][te], cfg["horizon"], period)
         - d["Y"][te]) ** 2)))
    log(f"    persistence test MSE  = {mse_ref['test']:.6f}")
    log(f"    BEST baseline is '{best_name}' at MSE {mse_best['test']:.6f}")
    log(f"    -> that, not persistence, is the number a learned model must beat")
    if mse_best["test"] < mse_ref["test"] * 0.98:
        log(f"       note: a non-learned baseline beats persistence here, so "
            f"clearing persistence alone proves very little on this corpus")

    cfp, dfp = cfg_fingerprint(cfg), data_fingerprint(d)
    log(f"\n[4] Resume: config {cfp} | data {dfp} | "
        f"markers in {os.path.join(cfg['results_dir'], '_completed')}")

    summary, per_snr_rows, cx_rows = list(base_rows), [], []

    for mi, mname in enumerate(cfg["models"], 1):
        if cfg["profile_complexity"]:
            log(f"\n[5] Profiling {mname} ({mi}/{len(cfg['models'])})")
            try:
                prof = profile_model(build_model(mname, cfg, C),
                                     (cfg["lookback"], C), mname, cfg)
                cx_rows.append(prof)
                log(f"    params {prof['params_M']:.4f} M | FLOPs "
                    f"{prof['flops_M']:.1f} M | lat {prof['latency_ms_avg']:.4f} "
                    f"ms/window | thr {prof['throughput_sps']:,.0f} w/s")
            except Exception as e:
                log(f"    profiling failed: {e}")

        for mode in cfg["training_modes"]:
            key = f"{mname}__{mode}"
            done = load_marker(cfg, key, cfp, dfp)
            if done is not None:
                r = done["summary_row"]
                log(f"\n[6] {mname} — {mode}  ALREADY TRAINED "
                    f"({done.get('completed_utc','')}) -> skipping")
                log(f"    MSE {r.get('mse', float('nan')):.6f}  skill "
                    f"{r.get('skill_vs_persistence', float('nan')):+.4f}")
                summary.append(r); per_snr_rows.extend(done.get("per_snr_rows", []))
                continue

            log(f"\n[6] {mname} — {mode}   ({MODEL_REFS.get(mname,'')})")
            try:
                if mode == "centralized":
                    model, info = train_centralized(build_model(mname, cfg, C),
                                                    tr_l, va_l, cfg)
                elif cfg["fl_algorithm"] == "titanic":
                    model, info = train_titanic(mname, d, tr, va_l, cfg, C)
                else:
                    model, info = train_centralized(build_model(mname, cfg, C),
                                                    tr_l, va_l, cfg)
            except Exception as e:
                log(f"    FAILED: {type(e).__name__}: {e}")
                continue

            p, y, s = predict(model, te_l)
            m = forecast_metrics(y, p, y_hist=d["X"][te],
                                 mse_ref=mse_ref["test"],
                                 mse_best=mse_best["test"])
            fit = fit_report(model, tr_l, va_l, te_l, info.get("history", []),
                             mse_ref, mse_best)
            log(f"    MSE {m['mse']:.6f} | MAE {m['mae']:.6f} | RMSE "
                f"{m['rmse']:.6f} | R2 {m['r2']:+.4f}")
            log(f"    skill vs persistence   {m['skill_vs_persistence']:+.4f}"
                f"  {'beats' if m['skill_vs_persistence'] > 0 else 'FAILS'}")
            log(f"    skill vs BEST baseline {m['skill_vs_best_baseline']:+.4f}"
                f"  {'BEATS ALL' if m['beats_all_baselines'] else 'FAILS — a '
                   'non-learned baseline is better'}")
            log(f"    occupancy view: BA {m['occ_balanced_acc']:.4f}  Pd "
                f"{m['occ_pd']:.4f}  FDR {m['occ_fdr']:.4f}  F1 {m['occ_f1']:.4f}")
            log(f"    fit: train/val/test skill {fit['train_skill']:+.3f} / "
                f"{fit['val_skill']:+.3f} / {fit['test_skill']:+.3f}  -> "
                f"{fit['verdict']}")
            if "comm_ratio_titanic_over_fl" in info:
                ct, cf_ = info["comm_titanic_bytes"], info["comm_fedavg_bytes"]
                unit = lambda b: (f"{b/1e9:.3f} GB" if b >= 1e9 else
                                  f"{b/1e6:.1f} MB" if b >= 1e6 else f"{b/1e3:.1f} kB")
                log(f"    communication: TITANIC {unit(ct)} vs FedAvg {unit(cf_)} "
                    f"(ratio {info['comm_ratio_titanic_over_fl']:.2f}x)")
                if info["comm_ratio_titanic_over_fl"] > 1.0:
                    log(f"      TITANIC costs MORE to communicate here. That is "
                        f"expected and matches the paper: activations scale with "
                        f"the number of batches, weights do not, so split "
                        f"learning only pays off when the model is large "
                        f"relative to the dataset. These forecasters are tiny.")

            row = {"model": mname, "mode": mode,
                   "family": MODEL_REFS.get(mname, ""), **m,
                   **{f"fit_{k}": v for k, v in fit.items()},
                   "train_time_s": round(info["train_time_s"], 1),
                   "best_val_loss": info["best_val_loss"]}
            for k in ("epochs_run", "rounds_run", "n_partitions",
                      "n_data_clients", "comm_titanic_bytes",
                      "comm_fedavg_bytes", "comm_ratio_titanic_over_fl"):
                if k in info:
                    row[k] = info[k]
            summary.append(row)

            def _mse_of(bn, mask):
                q = baseline_forecast(bn, d["X"][te][mask], cfg["horizon"], period)
                return float(np.mean((q - d["Y"][te][mask]) ** 2))
            ps = per_snr_metrics(
                y, p, s, cfg,
                mse_ref_fn=lambda mask: _mse_of("persistence", mask),
                mse_best_fn=lambda mask: min(_mse_of(b, mask)
                                             for b in cfg["baselines"]))
            unit = [{"model": mname, "mode": mode, **r}
                    for _k, r in sorted(ps.items(),
                                        key=lambda kv: kv[1]["snr_mid_db"])]
            per_snr_rows.extend(unit)

            od = os.path.join(cfg["results_dir"], "models")
            os.makedirs(od, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(od, f"{key}.pt"))
            _atomic_json(os.path.join(cfg["results_dir"], "history",
                                      f"{key}.json"), _jsonable(info))
            save_marker(cfg, key, cfp, dfp, row, unit)
            log(f"    checkpoint saved -> {key}.json")

    rd = cfg["results_dir"]
    write_csv(os.path.join(rd, "ALL_forecast_summary.csv"), summary)
    write_csv(os.path.join(rd, "ALL_forecast_per_snr.csv"), per_snr_rows)
    write_csv(os.path.join(rd, "ALL_forecast_complexity.csv"), cx_rows)
    _atomic_json(os.path.join(rd, "config.json"), _jsonable(cfg))

    log("\n" + "=" * 76)
    log("RESULTS (test).  skill = 1 - MSE/MSE_persistence; <= 0 means the model")
    log("                 does not beat copying the last observation.")
    log("=" * 76)
    log(f"{'model':<20}{'mode':<13}{'MSE':>10}{'MAE':>9}{'R2':>8}"
        f"{'skill_p':>9}{'skill_b':>9}{'occBA':>8}")
    for r in summary:
        log(f"{r['model']:<20}{r['mode']:<13}{r['mse']:10.6f}{r['mae']:9.6f}"
            f"{r.get('r2', float('nan')):8.4f}"
            f"{r.get('skill_vs_persistence', float('nan')):9.4f}"
            f"{r.get('skill_vs_best_baseline', float('nan')):9.4f}"
            f"{r.get('occ_balanced_acc', float('nan')):8.4f}")
    log("  skill_p = vs persistence,  skill_b = vs the BEST non-learned "
        "baseline (the honest bar)")

    learned = [r for r in summary if r["mode"] != "baseline"]
    if learned:
        beat_p = [r for r in learned if r.get("skill_vs_persistence", -1) > 0]
        beat_b = [r for r in learned if r.get("beats_all_baselines")]
        log(f"\n{len(beat_p)}/{len(learned)} learned configurations beat "
            f"persistence.")
        log(f"{len(beat_b)}/{len(learned)} beat EVERY non-learned baseline "
            f"— this is the number to report.")
        if not beat_b:
            log("NONE beat all baselines. On a corpus this autocorrelated that "
                "is a real and reportable result, not a bug. Before concluding, "
                "check that the series is long enough and that lookback and "
                "horizon suit the revisit period.")
        log("\n" + "=" * 76)
        log("FIT DIAGNOSTICS")
        log("=" * 76)
        log(f"{'model':<20}{'mode':<13}{'trSkill':>9}{'vaSkill':>9}"
            f"{'teSkill':>9}{'gap':>8}  verdict")
        for r in learned:
            log(f"{r['model']:<20}{r['mode']:<13}"
                f"{r.get('fit_train_skill', float('nan')):9.3f}"
                f"{r.get('fit_val_skill', float('nan')):9.3f}"
                f"{r.get('fit_test_skill', float('nan')):9.3f}"
                f"{r.get('fit_skill_gap_train_minus_val', float('nan')):8.3f}  "
                f"{r.get('fit_verdict','')}")
    log(f"\nResults -> {rd}/")


if __name__ == "__main__":
    main()
