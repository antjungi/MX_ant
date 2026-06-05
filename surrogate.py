# -*- coding: utf-8 -*-
"""Surrogate-only training: encoder + S-param MLP (no decoder, standalone)."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib
for _b in ("TkAgg", "Qt5Agg", "Qt6Agg", "wxAgg", "MacOSX"):
    try:
        matplotlib.use(_b)
        break
    except Exception:
        continue

_MPL_BACKEND = matplotlib.get_backend()

import os
import re
import glob
import math
import time
import random
import copy
import sys as _sys
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import pandas as pd
except ImportError:
    pd = None


# ══════════════════════════════════════════════════════════════
# DeepCAD token constants
# ══════════════════════════════════════════════════════════════
LINE = 0
ARC = 1
CIRCLE = 2
SOL = 3
EXT = 4
EOS = 5

N_BIT = 10
N_QUANT = 2 ** N_BIT
QMAX = N_QUANT - 1

N_CMD = 6
N_ARGS = 16

PAD_V = -1
PAD_CMD_INDEX = N_CMD
PAD_PARAM_INDEX = N_QUANT

VALID_PAR = {
    LINE: 2,
    ARC: 4,
    CIRCLE: 3,
    SOL: 0,
    EXT: 8,
    EOS: 0,
}

CMD_NAME = {
    LINE: "LINE",
    ARC: "ARC",
    CIRCLE: "CIRCLE",
    SOL: "SOL",
    EXT: "EXT",
    EOS: "EOS",
}

RETURN_PORT_PAIRS = ((1, 1), (2, 2), (3, 3))
RETURN_LABELS = ["S11", "S22", "S33"]


# ── Param slot indices in EXT token ──
P_SCALE = 0
P_PX = 1
P_PY = 2
P_PZ = 3
P_E1 = 4
P_E2 = 5
P_BOOL = 6
P_ETYPE = 7


# ── Paper-style muted palette (그림 2/4/8 geometry view 전용) ──
PAL_PAPER = [
    "#2E4172",   # deep slate blue
    "#8B5A3C",   # sienna brown
    "#3F6E5C",   # deep teal-green
    "#6B5B7A",   # muted mauve
    "#555555",   # charcoal grey
    "#8B7E3D",   # dark gold
]
PAL = PAL_PAPER


# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
@dataclass
class CFG:
    preset: str = "small"
    seed: int = 7
    run_name: str = "medium_var_fixed"

    # data
    npy_dirs: tuple = ("hfss_results/step_test",)
    sparam_globs: tuple = ("hfss_results/sparam/[12].*",)
    type_names: tuple = ("type1",)
    n_samples_per_type: tuple = (800,)
    sample_mode: str = "random"

    # frequency
    raw_n_freq: int = 401
    n_freq: int = 81
    freq_select_mode: str = "linspace"
    freq_start: float = 1.0
    freq_end: float = 5.0

    # optional dB clipping
    clip_db_enable: bool = False
    clip_db_min: float = -50.0
    clip_db_max: float = 5.0

    # training
    epochs: int = 160
    warmup: int = 15
    batch_size: int = 8
    lr_ae: float = 3e-4
    lr_mlp: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    val_ratio: float = 0.15

    # AE
    d_model: int = 256
    d_param: int = 64
    nhead: int = 8
    n_enc: int = 3
    n_dec: int = 3
    d_ff: int = 1024
    latent: int = 256
    mem_tokens: int = 12
    dropout: float = 0.1
    max_len_cap: int = 256

    n_pool: int = 12
    n_freq_bands: int = 6

    # aux numeric
    aux_numeric: bool = True
    aux_hidden_mult: float = 2.0

    # ★ Checkpoint save/load (학습 후 가중치 + cfg + source 저장 → 나중에 인버스만 가능)
    ckpt_save_path: str = "ckpt/surrogate_last.pt" # 빈 문자열 = 저장 안 함
    ckpt_load_path: str = ""                       # 비어 있으면 학습; 경로 주면 로드 후 학습 skip
    save_ckpt_after_train: bool = True

    # loss weights
    w_cmd: float = 1.0
    w_prm: float = 1.0
    w_aux: float = 2.0
    w_sparam: float = 5.0

    # dB loss scale
    db_loss_scale: float = 20.0

    # fixed medium_var VICReg
    use_vicreg: bool = True
    w_var: float = 0.5
    w_cov: float = 0.05
    vicreg_var_target: float = 1.0

    # residual MLP
    mlp_hidden_mult: float = 2.0
    mlp_dropout: float = 0.3
    residual_scale: float = 1.0
    zero_init_residual: bool = True

    # misc
    n_preview: int = 2
    log_verbosity: str = "simple"
    show_figures: bool = True


PRESETS = {
    "tiny": dict(
        n_samples_each=300,
        epochs=80,
        warmup=8,
        batch_size=8,
        d_model=128,
        d_param=32,
        nhead=4,
        n_enc=2,
        n_dec=2,
        d_ff=512,
        latent=128,
        mem_tokens=8,
        dropout=0.1,
        mlp_hidden_mult=2.0,
        mlp_dropout=0.3,
        weight_decay=1e-3,
    ),
    "small": dict(
        n_samples_each=800,
        epochs=160,
        warmup=15,
        batch_size=8,
        d_model=256,
        d_param=64,
        nhead=8,
        n_enc=3,
        n_dec=3,
        d_ff=1024,
        latent=256,
        mem_tokens=12,
        dropout=0.1,
        mlp_hidden_mult=2.0,
        mlp_dropout=0.3,
        weight_decay=1e-3,
    ),
    "full": dict(
        n_samples_each=0,
        epochs=250,
        warmup=20,
        batch_size=4,
        d_model=384,
        d_param=96,
        nhead=8,
        n_enc=4,
        n_dec=4,
        d_ff=1536,
        latent=512,
        mem_tokens=16,
        dropout=0.1,
        mlp_hidden_mult=2.0,
        mlp_dropout=0.3,
        weight_decay=1e-3,
    ),
}


def apply_preset(cfg):
    name = (cfg.preset or "custom").lower()

    if name == "custom":
        return cfg

    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {cfg.preset}")

    patch = dict(PRESETS[name])
    n_each = patch.pop("n_samples_each", None)

    for k, v in patch.items():
        setattr(cfg, k, v)

    # ★ preset이 정의되면 preset의 n_samples_each가 우선.
    if n_each is not None:
        cfg.n_samples_per_type = tuple([n_each] * len(cfg.npy_dirs))

    return cfg


# ══════════════════════════════════════════════════════════════
# Utils
# ══════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(path):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def causal_mask(sz, device):
    return torch.triu(
        torch.full((sz, sz), float("-inf"), device=device),
        diagonal=1,
    )


def trim_after_eos(tokens):
    if tokens.ndim != 2 or tokens.shape[1] != 17:
        return tokens

    rows = []

    for i in range(tokens.shape[0]):
        c = int(tokens[i, 0])

        if c < 0:
            break

        rows.append(tokens[i])

        if c == EOS:
            break

    if not rows:
        return np.zeros((0, 17), dtype=np.int32)

    return np.stack(rows, axis=0).astype(np.int32)


def ensure_eos_when_truncated(t, max_len):
    if t.shape[0] <= max_len:
        return t.astype(np.int32)

    out = t[:max_len].copy().astype(np.int32)

    eos_row = np.full((17,), PAD_V, dtype=np.int32)
    eos_row[0] = EOS
    out[-1] = eos_row

    return out


def select_frequency_indices(raw_n_freq, target_n_freq, freq_start, freq_end, mode="linspace"):
    raw_n_freq = int(raw_n_freq)
    target_n_freq = int(target_n_freq)

    if target_n_freq <= 0 or target_n_freq >= raw_n_freq:
        idx_sel = np.arange(raw_n_freq, dtype=np.int64)
    else:
        mode = (mode or "linspace").lower()

        if mode == "linspace":
            idx_sel = np.round(
                np.linspace(0, raw_n_freq - 1, target_n_freq)
            ).astype(np.int64)

            idx_sel = np.unique(idx_sel)

            if len(idx_sel) != target_n_freq:
                idx_sel = np.linspace(
                    0, raw_n_freq - 1, target_n_freq
                ).astype(np.int64)
                idx_sel = np.unique(idx_sel)

            if len(idx_sel) != target_n_freq:
                raise RuntimeError(
                    f"frequency selection failed: target={target_n_freq}, unique={len(idx_sel)}"
                )
        else:
            raise ValueError(f"Unknown freq_select_mode: {mode}")

    freqs_full = np.linspace(freq_start, freq_end, raw_n_freq).astype(np.float32)
    freqs_sel = freqs_full[idx_sel].astype(np.float32)

    return idx_sel, freqs_full, freqs_sel


def build_interp_matrix_np(freqs_sel, freqs_full):
    freqs_sel = np.asarray(freqs_sel, dtype=np.float32)
    freqs_full = np.asarray(freqs_full, dtype=np.float32)

    n_sel = len(freqs_sel)
    n_full = len(freqs_full)

    if n_sel < 2:
        raise ValueError("n_sel must be >= 2 for interpolation.")

    W = np.zeros((n_full, n_sel), dtype=np.float32)

    for i, f in enumerate(freqs_full):
        if f <= freqs_sel[0]:
            W[i, 0] = 1.0
        elif f >= freqs_sel[-1]:
            W[i, -1] = 1.0
        else:
            hi = int(np.searchsorted(freqs_sel, f, side="right"))
            lo = hi - 1

            f_lo = freqs_sel[lo]
            f_hi = freqs_sel[hi]

            t = float((f - f_lo) / max(f_hi - f_lo, 1e-12))

            W[i, lo] = 1.0 - t
            W[i, hi] = t

    return W


def interpolate_selected_to_full_torch(pred_sel, interp_w):
    return torch.einsum("fs,bsc->bfc", interp_w, pred_sel)


def interpolate_selected_to_full_np(pred_sel, freqs_sel, freqs_full):
    pred_sel = np.asarray(pred_sel, dtype=np.float32)
    freqs_sel = np.asarray(freqs_sel, dtype=np.float32)
    freqs_full = np.asarray(freqs_full, dtype=np.float32)

    single = False

    if pred_sel.ndim == 2:
        pred_sel = pred_sel[None, ...]
        single = True

    if pred_sel.ndim != 3:
        raise ValueError(f"pred_sel must be (N,n_sel,C) or (n_sel,C), got {pred_sel.shape}")

    N, n_sel, C = pred_sel.shape

    if n_sel != len(freqs_sel):
        raise ValueError(f"n_sel mismatch: pred={n_sel}, freqs_sel={len(freqs_sel)}")

    out = np.zeros((N, len(freqs_full), C), dtype=np.float32)

    for i in range(N):
        for c in range(C):
            out[i, :, c] = np.interp(
                freqs_full,
                freqs_sel,
                pred_sel[i, :, c],
            ).astype(np.float32)

    return out[0] if single else out


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def setup_log_file(path):
    f = open(path, "w", encoding="utf-8", buffering=1)
    _sys.stdout = _Tee(_sys.stdout, f)
    _sys.stderr = _Tee(_sys.stderr, f)
    return f


def _hr(ch="═", w=72):
    return ch * w


def section(title, ch="═"):
    print(f"\n{_hr(ch)}\n  {title}\n{_hr(ch)}")


def subsection(title, ch="─"):
    print(f"\n{_hr(ch)}\n  {title}\n{_hr(ch)}")


# ══════════════════════════════════════════════════════════════
# ★ Inverse design 결과 파일 저장 (txt + csv + png)
# ══════════════════════════════════════════════════════════════

def _detect_sparam_cols(header):
    col_idx = []
    col_names = []
    for i, h in enumerate(header):
        hl = str(h).strip().lower()
        if hl.startswith("re(") or hl.startswith("im("):
            col_idx.append(i)
            col_names.append(str(h).strip())
    return col_idx, col_names


def load_sparam_data(sparam_files, raw_n_freq, expected_cols=None, verbose=True):
    if pd is None:
        raise ImportError("pandas 필요: pip install pandas openpyxl")

    if not sparam_files:
        raise FileNotFoundError("S-param 파일이 없습니다.")

    sparam_files = sorted(sparam_files, key=natural_key)

    if verbose:
        print(f"  S-param files: {len(sparam_files)}")

    all_s = []
    ref_cols = expected_cols

    for fi, fp in enumerate(sparam_files):
        if verbose:
            print(f"    [{fi + 1}/{len(sparam_files)}] {os.path.basename(fp)}")

        ext = os.path.splitext(fp)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(fp, header=0)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(fp, header=0, engine="openpyxl")
        else:
            if verbose:
                print(f"      skip unsupported extension: {ext}")
            continue

        headers = [str(c) for c in df.columns]
        data = df.values
        col_idx, col_names = _detect_sparam_cols(headers)
        if not col_idx:
            if verbose:
                print("      re/im columns not found, skipped")
            continue

        if ref_cols is None:
            ref_cols = col_names
        elif col_names != ref_cols:
            raise ValueError(
                f"S-param column mismatch: {os.path.basename(fp)}\n"
                f"expected={ref_cols}\n"
                f"found   ={col_names}"
            )

        sd = data[:, col_idx].astype(np.float64)
        n_block = sd.shape[0] // raw_n_freq
        for si in range(n_block):
            chunk = sd[si * raw_n_freq:(si + 1) * raw_n_freq]
            if chunk.shape == (raw_n_freq, len(ref_cols)):
                all_s.append(chunk)

    if not all_s:
        raise ValueError("No valid S-param data loaded.")

    arr = np.stack(all_s, axis=0)
    if verbose:
        print(f"  loaded S-param raw shape: {arr.shape}")
    return arr, ref_cols


def _parse_sparam_pair(name):
    s = str(name).lower().replace(" ", "")
    m = re.search(r"port(\d+),port(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"port(\d+)port(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"s\((\d+),(\d+)\)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"s(\d)(\d)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _is_re_col(name):
    return str(name).strip().lower().startswith("re(")


def _is_im_col(name):
    return str(name).strip().lower().startswith("im(")


def filter_return3_sparams(sparam_all, sparam_names, return_pairs=RETURN_PORT_PAIRS, verbose=True):
    pair_to_cols = {}
    for i, name in enumerate(sparam_names):
        pair = _parse_sparam_pair(name)
        if pair is None:
            continue
        pair_to_cols.setdefault(pair, {"re": None, "im": None})
        if _is_re_col(name):
            pair_to_cols[pair]["re"] = i
        elif _is_im_col(name):
            pair_to_cols[pair]["im"] = i

    selected_cols = []
    selected_names = []
    for pair in return_pairs:
        if pair not in pair_to_cols:
            raise ValueError(
                f"S{pair[0]}{pair[1]} column not found.\n"
                f"Current columns: {sparam_names}"
            )
        re_idx = pair_to_cols[pair]["re"]
        im_idx = pair_to_cols[pair]["im"]
        if re_idx is None or im_idx is None:
            raise ValueError(
                f"S{pair[0]}{pair[1]} requires both re/im columns. "
                f"re={re_idx}, im={im_idx}"
            )
        selected_cols.extend([re_idx, im_idx])
        selected_names.extend([sparam_names[re_idx], sparam_names[im_idx]])

    out = sparam_all[:, :, selected_cols]

    if verbose:
        print("\n  Return-3 filtering")
        print("    target: S11, S22, S33 only")
        print(f"    before re/im: {sparam_all.shape}")
        print(f"    after  re/im: {out.shape}")
        print("    selected columns:")
        for n in selected_names:
            print(f"      {n}")

    return out, selected_names


def return3_reim_to_db_np(return3_reim):
    outs = []
    for k in (0, 2, 4):
        re_v = return3_reim[..., k]
        im_v = return3_reim[..., k + 1]
        mag = np.sqrt(re_v ** 2 + im_v ** 2)
        db = 20.0 * np.log10(np.clip(mag, 1e-12, None))
        outs.append(db)
    return np.stack(outs, axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════
class JointDataset(Dataset):
    def __init__(
        self,
        npy_files,
        max_len,
        sparam_db,
        freqs,
        sparam_db_full,
        freqs_full,
        freq_idx,
        interp_matrix,
        type_ids=None,
        type_names=None,
    ):
        self.npy_files = list(npy_files)
        self.max_len = max_len

        self.json_files = []
        for f in self.npy_files:
            jf = f.replace("_tokens.npy", "_deepcad.json")
            self.json_files.append(jf if os.path.exists(jf) else None)

        self.raw = []
        self.padded = []

        for f in self.npy_files:
            t = np.load(f).astype(np.int32)
            t = ensure_eos_when_truncated(t, max_len)
            self.raw.append(t)
            self.padded.append(self._pad(t))

        self.sparam_db = sparam_db.astype(np.float32)
        self.freqs = np.asarray(freqs, dtype=np.float32)
        self.sparam_db_full = sparam_db_full.astype(np.float32)
        self.freqs_full = np.asarray(freqs_full, dtype=np.float32)
        self.freq_idx = np.asarray(freq_idx, dtype=np.int64)
        self.interp_matrix = interp_matrix.astype(np.float32)

        N = len(self.padded)
        max_cmd = -1
        max_prm = -1
        for t in self.raw:
            if t.size == 0:
                continue
            cmd_valid = t[:, 0][t[:, 0] >= 0]
            prm_valid = t[:, 1:][t[:, 1:] >= 0]
            if cmd_valid.size:
                max_cmd = max(max_cmd, int(cmd_valid.max()))
            if prm_valid.size:
                max_prm = max(max_prm, int(prm_valid.max()))

        if max_cmd >= N_CMD:
            raise ValueError(f"cmd max={max_cmd} >= N_CMD={N_CMD}")
        if max_prm >= N_QUANT:
            raise ValueError(
                f"param max={max_prm} >= N_QUANT={N_QUANT}. "
                f"N_BIT mismatch likely."
            )

        if type_ids is None:
            self.type_ids = np.zeros(N, dtype=np.int64)
        else:
            self.type_ids = np.asarray(type_ids, dtype=np.int64)
            assert len(self.type_ids) == N

        self.type_names = list(type_names) if type_names else ["type1"]

        dist = ", ".join(
            f"{name}:{int((self.type_ids == i).sum())}"
            for i, name in enumerate(self.type_names)
        )

        print(f"  loaded {N} token sequences + Return-3 dB target")
        print(f"    type distribution       : {dist}")
        print(f"    token range             : cmd=[0..{max_cmd}], param=[0..{max_prm}]")
        print(f"    selected S-param shape  : {self.sparam_db.shape}")
        print(f"    selected frequency shape: {self.freqs.shape}")
        print(f"    selected freq range     : {self.freqs[0]:.4f} ~ {self.freqs[-1]:.4f} GHz")
        print(f"    full S-param shape      : {self.sparam_db_full.shape}")
        print(f"    full frequency shape    : {self.freqs_full.shape}")
        print(f"    full freq range         : {self.freqs_full[0]:.4f} ~ {self.freqs_full[-1]:.4f} GHz")
        print(f"    interp matrix shape     : {self.interp_matrix.shape}")

    def _pad(self, t):
        L = t.shape[0]
        if L >= self.max_len:
            return t[:self.max_len].astype(np.float32)
        pad = np.full((self.max_len - L, 17), PAD_V, dtype=np.float32)
        return np.concatenate([t, pad], axis=0).astype(np.float32)

    def load_json(self, idx):
        jf = self.json_files[idx] if idx < len(self.json_files) else None
        if jf is None:
            return None
        try:
            import json as _json
            with open(jf, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            return None

    def __len__(self):
        return len(self.padded)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.padded[idx], dtype=torch.float32),
            torch.tensor(self.sparam_db[idx], dtype=torch.float32),
            torch.tensor(self.sparam_db_full[idx], dtype=torch.float32),
            torch.tensor(int(self.type_ids[idx]), dtype=torch.long),
        )


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]


# ============================================================
# main_patched_part2.py
# ============================================================
# 이 파일은 main_patched_part1.py 의 *연속* 입니다.
# 사용법: Part 1 의 끝에 Part 2 의 내용을 그대로 이어붙여 하나의
# 파일로 합쳐서 실행하세요.
#
# Part 1 끝   = SubsetDataset 클래스
# Part 2 시작 = AE model (SinPosEnc / DeepCADBaselineAE)
# Part 2 끝   = main 블록 (USE_TYPES = [1, 2], 두 타입 학습)
# 두 파일을 합치면 총 3110줄.
# ============================================================

# ══════════════════════════════════════════════════════════════
# AE model
# ══════════════════════════════════════════════════════════════
class SinPosEnc(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])



class SparamCommonResidualMLP(nn.Module):
    def __init__(
        self, latent_dim, n_freq, common_curve,
        hidden_mult=2.0, dropout=0.3,
        residual_scale=1.0, zero_init_residual=True,
    ):
        super().__init__()

        self.n_freq = int(n_freq)
        self.n_out = 3
        self.residual_scale = float(residual_scale)

        common_curve = torch.as_tensor(common_curve, dtype=torch.float32)
        if common_curve.shape != (self.n_freq, 3):
            raise ValueError(
                f"common_curve shape mismatch. expected ({self.n_freq},3), got {tuple(common_curve.shape)}"
            )
        self.register_buffer("common_curve", common_curve)

        h1 = max(int(latent_dim * hidden_mult), 128)
        h2 = max(int(latent_dim * hidden_mult), 128)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, self.n_freq * 3),
        )

        if zero_init_residual:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z, return_parts=False):
        residual_raw = self.net(z).view(-1, self.n_freq, 3)
        residual_scaled = self.residual_scale * residual_raw
        pred = self.common_curve.unsqueeze(0) + residual_scaled
        if return_parts:
            return pred, residual_scaled
        return pred


# ══════════════════════════════════════════════════════════════
# Losses
# ══════════════════════════════════════════════════════════════
def ae_loss_fn(cmd_logits, prm_logits, aux_numeric, target, cfg):
    B, L, _ = target.shape

    cmd_t = target[:, :, 0].long()
    valid_cmd = cmd_t >= 0

    cmd_ce = F.cross_entropy(
        cmd_logits.reshape(B * L, N_CMD),
        cmd_t.clamp(0, N_CMD - 1).reshape(B * L),
        reduction="none",
    ).reshape(B, L)
    cmd_loss = (cmd_ce * valid_cmd.float()).sum() / (valid_cmd.float().sum() + 1e-8)

    prm_t = target[:, :, 1:].long()
    valid_prm = prm_t >= 0

    prm_ce = F.cross_entropy(
        prm_logits.reshape(B * L * N_ARGS, N_QUANT),
        prm_t.clamp(0, N_QUANT - 1).reshape(B * L * N_ARGS),
        reduction="none",
    ).reshape(B, L, N_ARGS)
    prm_loss = (prm_ce * valid_prm.float()).sum() / (valid_prm.float().sum() + 1e-8)

    aux_loss = target.new_zeros(())
    if aux_numeric is not None and cfg.w_aux > 0:
        aux_pred = aux_numeric[:, :L, :]
        prm_float = prm_t.clamp(0, QMAX).float() / float(QMAX)
        sq = (aux_pred - prm_float) ** 2
        aux_loss = (sq * valid_prm.float()).sum() / (valid_prm.float().sum() + 1e-8)

    total = cfg.w_cmd * cmd_loss + cfg.w_prm * prm_loss + cfg.w_aux * aux_loss

    comp = {
        "cmd": float(cmd_loss.item()),
        "prm": float(prm_loss.item()),
        "aux": float(aux_loss.item()),
        "ae_total": float(total.item()),
    }
    return total, comp


def sparam_full_interp_loss(pred_sel_db, true_sel_db, true_full_db, interp_w, cfg):
    pred_full_db = interpolate_selected_to_full_torch(pred_sel_db, interp_w)
    full_loss = ((pred_full_db - true_full_db) / cfg.db_loss_scale).pow(2).mean()
    full_rmse_db = torch.sqrt(((pred_full_db - true_full_db) ** 2).mean())
    sel_rmse_db = torch.sqrt(((pred_sel_db - true_sel_db) ** 2).mean())
    comp = {
        "sp_loss": float(full_loss.item()),
        "rmse_db_full": float(full_rmse_db.item()),
        "rmse_db_sel": float(sel_rmse_db.item()),
    }
    return full_loss, comp, pred_full_db


def vicreg_z_loss(z, var_target=1.0, eps=1e-4):
    if z.dim() != 2:
        raise ValueError(f"z shape must be (B,D), got {tuple(z.shape)}")
    B, D = z.shape
    if B <= 1:
        zero = z.new_zeros(())
        return zero, zero
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(var_target - std).mean()
    cov = (zc.T @ zc) / max(B - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / D
    return var_loss, cov_loss



def build_common_curve_and_print_baseline(dataset, train_idx, val_idx_per_type, type_names):
    section("Common curve baseline RMSE")

    train_curves = dataset.sparam_db[train_idx]
    common_sel = train_curves.mean(axis=0).astype(np.float32)

    common_full = interpolate_selected_to_full_np(
        common_sel, dataset.freqs, dataset.freqs_full,
    )

    total_sel_sq = 0.0; total_sel_n = 0
    total_full_sq = 0.0; total_full_n = 0

    for ti, tname in enumerate(type_names):
        idxs = val_idx_per_type.get(ti, [])
        if not idxs:
            continue

        true_sel = dataset.sparam_db[idxs]
        true_full = dataset.sparam_db_full[idxs]
        pred_sel = np.broadcast_to(common_sel[None, :, :], true_sel.shape)
        pred_full = np.broadcast_to(common_full[None, :, :], true_full.shape)

        sel_rmse = float(np.sqrt(np.mean((pred_sel - true_sel) ** 2)))
        full_rmse = float(np.sqrt(np.mean((pred_full - true_full) ** 2)))

        print(f"  [{tname}]")
        print(f"    common selected RMSE : {sel_rmse:.3f} dB")
        print(f"    common full RMSE     : {full_rmse:.3f} dB")

        total_sel_sq += float(np.sum((pred_sel - true_sel) ** 2))
        total_sel_n += int(np.prod(true_sel.shape))
        total_full_sq += float(np.sum((pred_full - true_full) ** 2))
        total_full_n += int(np.prod(true_full.shape))

    if total_sel_n > 0:
        print(f"\n  TOTAL common selected RMSE : {math.sqrt(total_sel_sq / total_sel_n):.3f} dB")
    if total_full_n > 0:
        print(f"  TOTAL common full RMSE     : {math.sqrt(total_full_sq / total_full_n):.3f} dB")

    return common_sel


@torch.no_grad()

def plot_training_curves(hist):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="white")
    ep = np.arange(1, len(hist["tr_full"]) + 1)

    axes[0].plot(ep, hist["tr_full"], label="train full RMSE")
    axes[0].plot(ep, hist["va_full"], label="val full RMSE")
    axes[0].set_title("Full-grid RMSE")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("dB")
    axes[0].grid(True, alpha=0.25); axes[0].legend()

    axes[1].plot(ep, hist["tr_total"], label="train total")
    axes[1].plot(ep, hist["va_total"], label="val total")
    axes[1].set_title("Total loss")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.25); axes[1].legend()

    axes[2].plot(ep, hist["z_std"], label="z_std")
    axes[2].plot(ep, hist["var"], label="var loss")
    axes[2].plot(ep, hist["cov"], label="cov loss")
    axes[2].set_title("VICReg diagnostics")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.25); axes[2].legend()

    plt.tight_layout()
    print("  ✓ training curves figure generated")


@torch.no_grad()
def collect_latents(ae, dataset, indices, device, batch_size=32):
    ae.eval()
    z_list = []
    type_list = []

    for i in range(0, len(indices), batch_size):
        bi = indices[i:i + batch_size]
        toks = torch.stack([
            torch.tensor(dataset.padded[k], dtype=torch.float32)
            for k in bi
        ]).to(device)
        z = ae.encode(toks)
        z_list.append(z.cpu().numpy())
        type_list.extend([int(dataset.type_ids[k]) for k in bi])

    z_all = np.concatenate(z_list, axis=0)
    type_ids_arr = np.asarray(type_list, dtype=np.int64)
    return z_all, type_ids_arr


def analyze_latent_space(z_all, type_ids=None, type_names=None):
    N, D = z_all.shape
    type_ids = np.asarray(type_ids) if type_ids is not None else np.zeros(N, dtype=int)
    type_names = list(type_names) if type_names else ["all"]
    n_types = len(type_names)

    colors_t = ["#4C9BE8", "#E05C5C", "#6DBF67", "#F4A340", "#A57CC1", "#4ECBCB"]

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  ⚠ sklearn 없음: latent 분석 skip (pip install scikit-learn)")
        return

    try:
        pca = PCA(n_components=min(2, D))
        z_pca = pca.fit_transform(z_all)
        pca_full = PCA(n_components=min(N - 1, D))
        pca_full.fit(z_all)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        ev2 = pca.explained_variance_ratio_
    except Exception as e:
        print(f"  ⚠ PCA 실패: {type(e).__name__}: {e}")
        return

    z_tsne = None
    try:
        from sklearn.manifold import TSNE
        if N >= 10:
            perp = min(30, max(5, N // 4))
            z_tsne = TSNE(
                n_components=2, perplexity=perp,
                random_state=0, init="pca", learning_rate="auto",
            ).fit_transform(z_all)
    except Exception as e:
        print(f"  ⚠ t-SNE skip: {type(e).__name__}: {e}")

    n_panels = 2 + (1 if z_tsne is not None else 0)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 4.5), facecolor="white",
    )
    if n_panels == 1:
        axes = [axes]
    pi = 0

    ax = axes[pi]; pi += 1
    for ti in range(n_types):
        m = (type_ids == ti)
        if not m.any():
            continue
        ax.scatter(
            z_pca[m, 0], z_pca[m, 1],
            c=colors_t[ti % len(colors_t)],
            s=18, alpha=0.7,
            label=type_names[ti], edgecolor="none",
        )
    ax.set_title(f"PCA  (var={ev2[0]:.2f}+{ev2[1]:.2f})", fontweight="normal")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    if n_types > 1:
        ax.legend(fontsize=9)

    if z_tsne is not None:
        ax = axes[pi]; pi += 1
        for ti in range(n_types):
            m = (type_ids == ti)
            if not m.any():
                continue
            ax.scatter(
                z_tsne[m, 0], z_tsne[m, 1],
                c=colors_t[ti % len(colors_t)],
                s=18, alpha=0.7,
                label=type_names[ti], edgecolor="none",
            )
        ax.set_title("t-SNE", fontweight="normal")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.grid(True, alpha=0.2)
        if n_types > 1:
            ax.legend(fontsize=9)

    ax = axes[pi]; pi += 1
    ax.plot(
        np.arange(1, len(cumvar) + 1), cumvar,
        lw=1.8, color="#4C9BE8", marker="o", markersize=3,
    )
    for thr, c in [(0.5, "#94A3B8"), (0.9, "#E07B5B"), (0.95, "#3F6E5C")]:
        ax.axhline(thr, ls="--", color=c, lw=1, alpha=0.6)
        if (cumvar >= thr).any():
            idx = int(np.argmax(cumvar >= thr)) + 1
            ax.text(idx, thr, f" dim{idx}", color=c, fontsize=8, va="bottom")
    ax.set_xlabel("# PCA dim")
    ax.set_ylabel("cum explained var")
    ax.set_title(f"Latent intrinsic dim  (D={D})", fontweight="normal")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    print("  ✓ latent 분석 figure 생성 (PCA + t-SNE + cumvar)")


@torch.no_grad()
def visualize_sparam_predictions(
    ae, mlp, dataset, val_indices, device,
    n_samples=3, seed=0,
):
    ae.eval(); mlp.eval()

    val_indices = list(val_indices)
    if len(val_indices) == 0:
        print("  ⚠ sparam viz: val 인덱스 없음")
        return

    n_pick = min(int(n_samples), len(val_indices))
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(val_indices), size=n_pick, replace=False)
    sel = sorted(val_indices[i] for i in sel)

    freqs_full = np.asarray(dataset.freqs_full, dtype=np.float32)
    freqs_sel = np.asarray(dataset.freqs, dtype=np.float32)

    common_sel = mlp.common_curve.detach().cpu().numpy()
    common_full = interpolate_selected_to_full_np(common_sel, freqs_sel, freqs_full)

    fig, axes = plt.subplots(
        n_pick, 3, figsize=(15, 3.5 * n_pick),
        facecolor="white", sharex=True,
    )
    if n_pick == 1:
        axes = axes.reshape(1, 3)

    truth_color = "#2E4172"
    pred_color = "#E07B5B"
    baseline_color = "#94A3B8"

    for row, idx in enumerate(sel):
        tok = torch.tensor(
            dataset.padded[idx], dtype=torch.float32,
        ).unsqueeze(0).to(device)

        z = ae.encode(tok)
        pred_sel_db, residual = mlp(z, return_parts=True)
        pred_sel_db = pred_sel_db.cpu().numpy()[0]

        pred_full_db = interpolate_selected_to_full_np(
            pred_sel_db, freqs_sel, freqs_full,
        )

        true_full_db = dataset.sparam_db_full[idx]
        res_abs = float(np.abs(residual.cpu().numpy()).mean())

        try:
            t_id = int(dataset.type_ids[idx])
            t_name = dataset.type_names[t_id]
        except Exception:
            t_name = "?"

        for col, lbl in enumerate(RETURN_LABELS):
            ax = axes[row, col]

            ax.plot(
                freqs_full, common_full[:, col],
                color=baseline_color, lw=1.4, ls=":", alpha=0.9,
                label="common curve" if (row == 0 and col == 0) else None,
            )
            ax.plot(
                freqs_full, true_full_db[:, col],
                color=truth_color, lw=2.0,
                label="truth" if (row == 0 and col == 0) else None,
            )
            ax.plot(
                freqs_full, pred_full_db[:, col],
                color=pred_color, lw=1.6, ls="--", alpha=0.9,
                label="pred" if (row == 0 and col == 0) else None,
            )

            rmse = float(np.sqrt(np.mean(
                (pred_full_db[:, col] - true_full_db[:, col]) ** 2
            )))

            ax.set_title(
                f"[{t_name}] idx={idx}  {lbl}  RMSE={rmse:.2f} dB",
                fontsize=9, fontweight="normal",
            )
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel("|S| [dB]")
            if row == n_pick - 1:
                ax.set_xlabel("frequency [GHz]")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="best", framealpha=0.85)

        axes[row, 0].text(
            -0.18, 0.5,
            f"|residual|\n  mean\n  ={res_abs:.2f} dB",
            transform=axes[row, 0].transAxes,
            fontsize=8, color="#555", ha="right", va="center",
        )

    plt.tight_layout()
    print(f"  ✓ S-param prediction figure 생성 (n={n_pick} samples, freqs={len(freqs_full)})")


# ══════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════

def load_multitype_data(cfg, script_dir):
    n_types = len(cfg.npy_dirs)

    if n_types != len(cfg.sparam_globs):
        raise ValueError("npy_dirs and sparam_globs length mismatch.")

    type_names = (
        tuple(cfg.type_names)
        if len(cfg.type_names) == n_types
        else tuple(f"type{i + 1}" for i in range(n_types))
    )

    n_samples_per_type = (
        tuple(cfg.n_samples_per_type)
        if len(cfg.n_samples_per_type) == n_types
        else tuple([0] * n_types)
    )

    print(f"  Types: {list(type_names)}")
    print(f"  n_samples per type: {list(n_samples_per_type)}")
    print(f"  raw_n_freq={cfg.raw_n_freq}, selected n_freq target={cfg.n_freq}")

    freq_idx, freqs_full, freqs_sel = select_frequency_indices(
        raw_n_freq=cfg.raw_n_freq,
        target_n_freq=cfg.n_freq,
        freq_start=cfg.freq_start,
        freq_end=cfg.freq_end,
        mode=cfg.freq_select_mode,
    )

    cfg.n_freq = int(len(freq_idx))
    interp_matrix = build_interp_matrix_np(freqs_sel, freqs_full)

    print(f"\n  Frequency point selection")
    print(f"    mode             : {cfg.freq_select_mode}")
    print(f"    raw_n_freq       : {cfg.raw_n_freq}")
    print(f"    selected n_freq  : {cfg.n_freq}")
    print(f"    first/last index : {int(freq_idx[0])} / {int(freq_idx[-1])}")
    print(f"    first/last freq  : {freqs_sel[0]:.4f} / {freqs_sel[-1]:.4f} GHz")
    if len(freq_idx) >= 2:
        print(f"    approx df        : {float(freqs_sel[1] - freqs_sel[0]):.4f} GHz")
    print(f"    interp matrix    : {interp_matrix.shape}")

    all_npy = []
    all_sp_reim = []
    all_tid = []
    sparam_names_ref = None

    for ti in range(n_types):
        npy_dir = cfg.npy_dirs[ti]
        sp_glob = cfg.sparam_globs[ti]
        tname = type_names[ti]
        n_pick = int(n_samples_per_type[ti])

        npy_dir_abs = npy_dir if os.path.isabs(npy_dir) else os.path.join(script_dir, npy_dir)
        sp_glob_abs = sp_glob if os.path.isabs(sp_glob) else os.path.join(script_dir, sp_glob)

        print(f"\n  · [{tname}]")
        print(f"    npy_dir     = {npy_dir_abs}")
        print(f"    sparam_glob = {sp_glob_abs}")

        npy_files = sorted(
            glob.glob(os.path.join(npy_dir_abs, "*_tokens.npy")),
            key=natural_key,
        )
        if not npy_files:
            raise FileNotFoundError(f"No npy files: {npy_dir_abs}")

        sp_files = sorted(glob.glob(sp_glob_abs), key=natural_key)
        if not sp_files:
            raise FileNotFoundError(f"No S-param files: {sp_glob_abs}")

        sp_arr_raw, sp_names = load_sparam_data(
            sp_files, cfg.raw_n_freq,
            expected_cols=sparam_names_ref, verbose=False,
        )

        if sparam_names_ref is None:
            sparam_names_ref = sp_names

        n_match = min(len(npy_files), sp_arr_raw.shape[0])
        if len(npy_files) != sp_arr_raw.shape[0]:
            print(f"    count mismatch token/S-param → use first {n_match}")
            print(f"      token files : {len(npy_files)}")
            print(f"      sparam rows : {sp_arr_raw.shape[0]}")

        npy_files = npy_files[:n_match]
        sp_arr_raw = sp_arr_raw[:n_match]

        if 0 < n_pick < n_match:
            mode = (cfg.sample_mode or "random").lower()
            if mode == "sequential":
                pick = list(range(n_pick))
            else:
                rng = random.Random(cfg.seed + ti)
                pick = sorted(rng.sample(range(n_match), n_pick))
            npy_files = [npy_files[i] for i in pick]
            sp_arr_raw = sp_arr_raw[pick]
            print(f"    selected: {len(npy_files)} by {mode}")
        else:
            print(f"    use all: {len(npy_files)}")

        all_npy.extend(npy_files)
        all_sp_reim.append(sp_arr_raw)
        all_tid.extend([ti] * len(npy_files))

    sparam_reim_all_raw = np.concatenate(all_sp_reim, axis=0)
    type_ids = np.asarray(all_tid, dtype=np.int64)

    sparam_return3_reim_raw, _sparam_return3_names = filter_return3_sparams(
        sparam_reim_all_raw, sparam_names_ref,
        return_pairs=RETURN_PORT_PAIRS, verbose=True,
    )

    sparam_db_full = return3_reim_to_db_np(sparam_return3_reim_raw)

    if cfg.clip_db_enable:
        print(f"\n  dB clipping enabled: [{cfg.clip_db_min}, {cfg.clip_db_max}]")
        sparam_db_full = np.clip(
            sparam_db_full, cfg.clip_db_min, cfg.clip_db_max,
        ).astype(np.float32)

    sparam_db = sparam_db_full[:, freq_idx, :]

    print(f"\n  Convert Return-3 re/im → dB target")
    print(f"    raw re/im shape     : {sparam_return3_reim_raw.shape}")
    print(f"    full dB shape       : {sparam_db_full.shape}")
    print(f"    selected dB shape   : {sparam_db.shape}")
    print(f"    full dB range       : min={sparam_db_full.min():.2f}, max={sparam_db_full.max():.2f}")
    print(f"    selected dB range   : min={sparam_db.min():.2f}, max={sparam_db.max():.2f}")

    max_len = min(
        max(np.load(f).shape[0] for f in all_npy) + 4,
        cfg.max_len_cap,
    )

    dataset = JointDataset(
        all_npy, max_len, sparam_db,
        freqs=freqs_sel,
        sparam_db_full=sparam_db_full,
        freqs_full=freqs_full,
        freq_idx=freq_idx,
        interp_matrix=interp_matrix,
        type_ids=type_ids,
        type_names=type_names,
    )

    return dataset, all_npy, type_ids, type_names, sparam_db, max_len


def make_stratified_split(cfg, dataset, type_ids, type_names):
    n_types = len(type_names)
    n_use = len(dataset)
    rng = random.Random(cfg.seed)

    train_idx = []
    val_idx = []
    val_idx_per_type = {}

    for ti in range(n_types):
        idx_t = [i for i in range(n_use) if int(type_ids[i]) == ti]
        rng.shuffle(idx_t)
        n_val_t = max(1, int(len(idx_t) * cfg.val_ratio))
        val_idx_per_type[ti] = idx_t[:n_val_t]
        val_idx.extend(idx_t[:n_val_t])
        train_idx.extend(idx_t[n_val_t:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    per_type_msg = ", ".join(
        f"{type_names[ti]}:{int((type_ids == ti).sum()) - len(val_idx_per_type[ti])}/{len(val_idx_per_type[ti])}"
        for ti in range(n_types)
    )

    print(f"  train={len(train_idx)} val={len(val_idx)}")
    print(f"  per-type train/val: {per_type_msg}")

    return train_idx, val_idx, val_idx_per_type


# ══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
#  Surrogate Model:  tokens → encoder → latent z → MLP → S-param
# ═══════════════════════════════════════════════════════════════
class SurrogateModel(nn.Module):
    """Encoder + S-param surrogate MLP. Decoder 없음.

      tokens (B, L, 17)
        → cmd_emb + param_embs(+fourier) → param_proj → emb_norm
        → pos_enc → Transformer encoder ×n_enc → h
        → PMA attention pooling (n_pool queries) → to_z → z
        → SparamCommonResidualMLP → S-param (B, n_freq, 3)
    """

    def __init__(
        self, max_len, d_model, d_param, nhead, n_enc, d_ff,
        latent, dropout, n_pool, n_freq_bands,
        n_freq, common_curve,
        mlp_hidden_mult=2.0, mlp_dropout=0.3,
        residual_scale=1.0, zero_init_residual=True,
    ):
        super().__init__()
        assert d_model % nhead == 0

        self.max_len = max_len
        self.d_model = d_model
        self.d_param = d_param
        self.latent = latent
        self.n_pool = int(n_pool)
        self.n_freq_bands = int(n_freq_bands)

        self.cmd_emb = nn.Embedding(
            N_CMD + 1, d_model, padding_idx=PAD_CMD_INDEX,
        )
        self.param_embs = nn.ModuleList([
            nn.Embedding(N_QUANT + 1, d_param, padding_idx=PAD_PARAM_INDEX)
            for _ in range(N_ARGS)
        ])

        if self.n_freq_bands > 0:
            freq_bands = (
                2.0 ** torch.arange(self.n_freq_bands, dtype=torch.float32)
            ) * math.pi
            self.register_buffer("fourier_freqs", freq_bands)
            param_proj_in = N_ARGS * d_param + N_ARGS * 2 * self.n_freq_bands
        else:
            self.fourier_freqs = None
            param_proj_in = N_ARGS * d_param

        self.param_proj = nn.Linear(param_proj_in, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.pos_enc = SinPosEnc(d_model, max_len=max_len + 16, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_enc, norm=nn.LayerNorm(d_model),
        )

        if self.n_pool >= 2:
            self.pool_queries = nn.Parameter(
                torch.randn(self.n_pool, d_model) * 0.02
            )
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead,
                dropout=dropout, batch_first=True,
            )
            self.pool_norm = nn.LayerNorm(d_model)
            self.to_z = nn.Linear(self.n_pool * d_model, latent)
        else:
            self.pool_queries = None
            self.pool_attn = None
            self.pool_norm = None
            self.to_z = nn.Linear(d_model, latent)

        self.surrogate = SparamCommonResidualMLP(
            latent_dim=latent, n_freq=n_freq, common_curve=common_curve,
            hidden_mult=mlp_hidden_mult, dropout=mlp_dropout,
            residual_scale=residual_scale,
            zero_init_residual=zero_init_residual,
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _content_embed(self, x):
        cmd = x[:, :, 0].long()
        params = x[:, :, 1:].long()
        pad_mask = cmd < 0

        cmd_idx = cmd.clone()
        cmd_idx[pad_mask] = PAD_CMD_INDEX
        cmd_idx = cmd_idx.clamp(0, PAD_CMD_INDEX)
        cmd_e = self.cmd_emb(cmd_idx)

        param_idx = params.clone()
        param_idx[param_idx < 0] = PAD_PARAM_INDEX
        param_idx = param_idx.clamp(0, PAD_PARAM_INDEX)

        slot_embs = [
            self.param_embs[s](param_idx[:, :, s]) for s in range(N_ARGS)
        ]
        param_concat = torch.cat(slot_embs, dim=-1)

        if self.n_freq_bands > 0 and self.fourier_freqs is not None:
            p_int = param_idx.clone()
            pad_p = params < 0
            p_int[pad_p] = 0
            p_f = p_int.float() / float(QMAX)
            ang = p_f.unsqueeze(-1) * self.fourier_freqs.view(1, 1, 1, -1)
            sin_p = torch.sin(ang); cos_p = torch.cos(ang)
            mask = 1.0 - pad_p.unsqueeze(-1).float()
            sin_p = sin_p * mask; cos_p = cos_p * mask
            B, L, _ = params.shape
            fourier = torch.cat([sin_p, cos_p], dim=-1)
            fourier = fourier.reshape(B, L, N_ARGS * 2 * self.n_freq_bands)
            param_concat = torch.cat([param_concat, fourier], dim=-1)

        param_e = self.param_proj(param_concat)
        emb = self.emb_norm(cmd_e + param_e)
        emb = emb * (~pad_mask).unsqueeze(-1)
        return emb, pad_mask

    def encode(self, x):
        emb, pad_mask = self._content_embed(x)
        emb = self.pos_enc(emb)
        h = self.encoder(emb, src_key_padding_mask=pad_mask)

        if self.n_pool >= 2:
            B = h.size(0)
            q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
            pooled, _ = self.pool_attn(
                q, h, h, key_padding_mask=pad_mask, need_weights=False,
            )
            pooled = self.pool_norm(pooled)
            pooled = pooled.reshape(B, self.n_pool * self.d_model)
            z = self.to_z(pooled)
        else:
            valid = (~pad_mask).float().unsqueeze(-1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            z = self.to_z(pooled)
        return z

    def forward(self, x):
        z = self.encode(x)
        sparam_pred_sel = self.surrogate(z)
        return sparam_pred_sel, z


# ═══════════════════════════════════════════════════════════════
#  Training loop  (loss = S-param + optional VICReg)
# ═══════════════════════════════════════════════════════════════
def run_epoch_surrogate(model, loader, optimizer, scheduler, device, cfg,
                        interp_w, train_mode=True):
    model.train() if train_mode else model.eval()
    sums = dict(total=0.0, sparam=0.0, rmse_db_sel=0.0, rmse_db_full=0.0)
    n_total = 0
    vic_var = vic_cov = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            tok = batch[0].to(device)
            sp_sel = batch[1].to(device)
            sp_full = batch[2].to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            sparam_pred_sel, z = model(tok)

            sp_loss, sp_comp, _pred_full = sparam_full_interp_loss(
                sparam_pred_sel, sp_sel, sp_full, interp_w, cfg,
            )

            vic_total = z.new_zeros(())
            if cfg.use_vicreg:
                var_l, cov_l = vicreg_z_loss(
                    z, var_target=cfg.vicreg_var_target,
                )
                vic_total = cfg.w_var * var_l + cfg.w_cov * cov_l
                vic_var += float(var_l.item())
                vic_cov += float(cov_l.item())

            total = cfg.w_sparam * sp_loss + vic_total

            if train_mode:
                total.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            B = tok.size(0)
            sums["total"] += float(total.item()) * B
            sums["sparam"] += float(sp_comp["sp_loss"]) * B
            sums["rmse_db_sel"] += float(sp_comp["rmse_db_sel"]) * B
            sums["rmse_db_full"] += float(sp_comp["rmse_db_full"]) * B
            n_total += B
            n_batches += 1

    n = max(n_total, 1)
    nb = max(n_batches, 1)
    return {
        "total": sums["total"] / n,
        "sparam": sums["sparam"] / n,
        "rmse_db_sel": sums["rmse_db_sel"] / n,
        "rmse_db_full": sums["rmse_db_full"] / n,
        "var": vic_var / nb,
        "cov": vic_cov / nb,
    }


def train_surrogate(cfg, dataset, train_idx, val_idx, common_curve, device):
    set_seed(cfg.seed)

    train_set = SubsetDataset(dataset, train_idx)
    val_set = SubsetDataset(dataset, val_idx) if len(val_idx) > 0 \
              else SubsetDataset(dataset, train_idx[:1])

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
    )

    interp_w = torch.tensor(
        dataset.interp_matrix, dtype=torch.float32, device=device,
    )

    model = SurrogateModel(
        max_len=dataset.max_len,
        d_model=cfg.d_model, d_param=cfg.d_param, nhead=cfg.nhead,
        n_enc=cfg.n_enc, d_ff=cfg.d_ff, latent=cfg.latent,
        dropout=cfg.dropout, n_pool=cfg.n_pool, n_freq_bands=cfg.n_freq_bands,
        n_freq=cfg.n_freq, common_curve=common_curve,
        mlp_hidden_mult=cfg.mlp_hidden_mult, mlp_dropout=cfg.mlp_dropout,
        residual_scale=cfg.residual_scale,
        zero_init_residual=cfg.zero_init_residual,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    section("BUILD MODEL")
    print(f"  Surrogate-only model params: {n_params:,}  ({n_params/1e6:.2f} M)")
    print(f"  Loss: S-param dB RMSE + VICReg({cfg.use_vicreg})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_ae, weight_decay=cfg.weight_decay,
    )

    n_steps = max(cfg.epochs * max(len(train_loader), 1), 1)
    warmup_steps = max(cfg.warmup * max(len(train_loader), 1), 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    section("SURROGATE TRAINING")
    print(f"  {'ep':>4s} | {'tr_total':>9s} {'tr_sp':>8s} {'tr_full':>8s} {'tr_sel':>8s} "
          f"{'var':>7s} {'cov':>7s} | "
          f"{'va_total':>9s} {'va_full':>8s} {'va_sel':>8s} | {'lr':>8s}")
    print("  " + "-" * 120)

    best_metric = float("inf")
    best_state = None
    log_every = max(1, cfg.epochs // 30)

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch_surrogate(model, train_loader, optimizer, scheduler, device, cfg,
                                  interp_w, train_mode=True)
        va = run_epoch_surrogate(model, val_loader, None, None, device, cfg,
                                  interp_w, train_mode=False)

        val_metric = va["rmse_db_full"]
        if val_metric < best_metric:
            best_metric = val_metric
            best_state = {
                k: v.detach().clone().cpu()
                for k, v in model.state_dict().items()
            }

        if ep == 1 or ep % log_every == 0 or ep == cfg.epochs:
            print(
                f"  {ep:4d} | "
                f"{tr['total']:9.4f} {tr['sparam']:8.4f} {tr['rmse_db_full']:8.3f} {tr['rmse_db_sel']:8.3f} "
                f"{tr['var']:7.4f} {tr['cov']:7.4f} | "
                f"{va['total']:9.4f} {va['rmse_db_full']:8.3f} {va['rmse_db_sel']:8.3f} | "
                f"{optimizer.param_groups[0]['lr']:8.2e}"
            )

    print(f"\n  Best full-grid val RMSE dB: {best_metric:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ── ckpt save (source code + cfg embed) ──
    ckpt_save_path = getattr(cfg, "ckpt_save_path", "ckpt/surrogate_last.pt")
    if ckpt_save_path and getattr(cfg, "save_ckpt_after_train", True):
        try:
            _dir = os.path.dirname(ckpt_save_path)
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            _src = None
            _src_name = None
            try:
                _src_path = os.path.abspath(__file__)
                with open(_src_path, "r", encoding="utf-8") as _f:
                    _src = _f.read()
                _src_name = os.path.basename(_src_path)
            except Exception:
                pass
            try:
                import dataclasses as _dc
                _cfg_dict = _dc.asdict(cfg)
            except Exception:
                _cfg_dict = None
            cc_np = (common_curve.detach().cpu().numpy()
                     if hasattr(common_curve, "detach")
                     else np.asarray(common_curve))
            torch.save({
                "model_state_dict": model.state_dict(),
                "max_len": dataset.max_len,
                "type_names": list(getattr(dataset, "type_names", [])),
                "common_curve": cc_np,
                "freqs_full": np.asarray(dataset.freqs_full, dtype=np.float32),
                "freqs_sel": np.asarray(dataset.freqs, dtype=np.float32),
                "cfg_repr": repr(cfg),
                "cfg_dict": _cfg_dict,
                "best_val_metric": float(best_metric),
                "source_code": _src,
                "source_filename": _src_name,
            }, ckpt_save_path)
            extra = f"  (+source: {_src_name})" if _src else ""
            print(f"\n  ✓ saved surrogate ckpt → {ckpt_save_path}  "
                  f"(val RMSE dB={best_metric:.4f}){extra}\n")
        except Exception as e:
            print(f"  ⚠ failed to save ckpt: {e}")

    return model, best_metric


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def main():
    cfg = CFG()
    cfg.run_name = "surrogate_only"
    apply_preset(cfg)
    cfg.ckpt_save_path = "ckpt/surrogate_last.pt"
    cfg.w_sparam = 1.0
    cfg.w_cmd = 0.0
    cfg.w_prm = 0.0
    cfg.w_aux = 0.0

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    section("SURROGATE-ONLY TRAINING (standalone)")
    print(f"  Device : {device}" + (f" ({torch.cuda.get_device_name(0)})"
                                     if device.type == "cuda" else ""))
    print(f"  Preset : {cfg.preset}")

    section("LOAD DATA")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset, _npy, type_ids, type_names, _sp, _ml = load_multitype_data(
        cfg, script_dir,
    )

    section("SPLIT")
    train_idx, val_idx, val_idx_per_type = make_stratified_split(
        cfg, dataset, type_ids, type_names,
    )

    common_curve = build_common_curve_and_print_baseline(
        dataset=dataset, train_idx=train_idx,
        val_idx_per_type=val_idx_per_type, type_names=type_names,
    )

    model, best = train_surrogate(cfg, dataset, train_idx, val_idx,
                                   common_curve, device)

    section("DONE")


if __name__ == "__main__":
    main()
