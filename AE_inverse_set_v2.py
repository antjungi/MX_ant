#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Body-Set DeepCAD AE + Return-3 Common-Curve Residual Surrogate (v2)
====================================================================

v1(slot-decoder + Hungarian, GPT 생성판) 대비 변경점:
  1) Hungarian matching cost 를 vectorize (Python 이중 루프 제거)
     → 매 step 당 CE 호출이 M*Nt 번 → log_softmax + gather 몇 번
  2) body 내부 decoder 에 causal self-attention mask 추가
     → 토큰 위치 t 는 t 이하만 attend → 닫힌 loop 의 chained 의미 학습 보조
  3) matching cost 에 presence term 추가
     → -log σ(presence_logit) 가중치를 cost 에 더해서 confidently-present 슬롯이
       매칭에 우선 선택되도록 함 (DETR 정석)

핵심 아키텍처
-------------
- Encoder: SOL...EXT body 분리 → body 내부 token encoder → body set encoder(no PE)
           → PMA pooling → z
- Decoder: z → fixed body slots → 각 slot 의 body token sequence를 parallel
           (causal self-attn) 로 decode
- Loss: Hungarian body-slot matching + S-param surrogate + VICReg
- Surrogate: z → common_curve + residual(z) for Return-3 dB
"""

import os
import re
import glob
import math
import time
import random
import sys as _sys
from dataclasses import dataclass

import numpy as np

import matplotlib
for _b in ("TkAgg", "Qt5Agg", "Qt6Agg", "wxAgg", "MacOSX"):
    try:
        matplotlib.use(_b)
        break
    except Exception:
        continue
_MPL_BACKEND = matplotlib.get_backend()

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy.optimize import linear_sum_assignment as _hungarian
except Exception:
    _hungarian = None


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

# EXT parameter slot indices
P_SCALE = 0
P_PX = 1
P_PY = 2
P_PZ = 3
P_E1 = 4
P_E2 = 5
P_BOOL = 6
P_ETYPE = 7


# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
@dataclass
class CFG:
    preset: str = "tiny"       # tiny / small / full / custom
    seed: int = 7
    run_name: str = "bodyset_ae_v2"

    # data
    npy_dirs: tuple = ("hfss_results/step_test",)
    sparam_globs: tuple = ("hfss_results/sparam/[12].*",)
    type_names: tuple = ("type1",)
    n_samples_per_type: tuple = (300,)
    sample_mode: str = "random"   # random / sequential

    # frequency
    raw_n_freq: int = 401
    n_freq: int = 81
    freq_select_mode: str = "linspace"
    freq_start: float = 1.0
    freq_end: float = 5.0

    # token/body caps
    max_bodies_cap: int = 16
    max_body_len_cap: int = 128

    # training
    epochs: int = 80
    warmup: int = 8
    batch_size: int = 8
    lr_ae: float = 3e-4
    lr_mlp: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    val_ratio: float = 0.15

    # model
    d_model: int = 128
    d_param: int = 32
    nhead: int = 4
    n_body_enc: int = 2
    n_set_enc: int = 1
    n_body_dec: int = 2
    d_ff: int = 512
    latent: int = 128
    mem_tokens: int = 8
    dropout: float = 0.1
    n_pool: int = 8
    n_freq_bands: int = 6

    # loss weights
    w_cmd: float = 1.0
    w_prm: float = 1.0
    w_presence: float = 1.0
    w_match_presence: float = 0.5     # v2: matching cost 안의 presence 가중치
    w_sparam: float = 5.0
    db_loss_scale: float = 20.0

    # VICReg latent regularizer
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
    n_preview: int = 3
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
        n_body_enc=2,
        n_set_enc=1,
        n_body_dec=2,
        d_ff=512,
        latent=128,
        mem_tokens=8,
        dropout=0.1,
        n_pool=8,
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
        n_body_enc=3,
        n_set_enc=1,
        n_body_dec=3,
        d_ff=1024,
        latent=256,
        mem_tokens=12,
        dropout=0.1,
        n_pool=12,
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
        n_body_enc=4,
        n_set_enc=2,
        n_body_dec=4,
        d_ff=1536,
        latent=512,
        mem_tokens=16,
        dropout=0.1,
        n_pool=16,
        mlp_hidden_mult=2.0,
        mlp_dropout=0.3,
        weight_decay=1e-3,
    ),
}


def apply_preset(cfg: CFG):
    name = (cfg.preset or "custom").lower()

    if name == "custom":
        return cfg

    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {cfg.preset}")

    patch = dict(PRESETS[name])
    n_each = patch.pop("n_samples_each", None)

    for k, v in patch.items():
        setattr(cfg, k, v)

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


def _hr(ch="═", w=78):
    return ch * w


def section(title, ch="═"):
    print(f"\n{_hr(ch)}\n  {title}\n{_hr(ch)}")


def subsection(title, ch="─"):
    print(f"\n{_hr(ch)}\n  {title}\n{_hr(ch)}")


def trim_after_eos(tokens: np.ndarray) -> np.ndarray:
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
        out = np.full((1, 17), PAD_V, dtype=np.int32)
        out[0, 0] = EOS
        return out

    return np.stack(rows, axis=0).astype(np.int32)


def ensure_eos(tokens: np.ndarray) -> np.ndarray:
    tokens = trim_after_eos(tokens)

    if len(tokens) == 0 or int(tokens[-1, 0]) != EOS:
        eos = np.full((1, 17), PAD_V, dtype=np.int32)
        eos[0, 0] = EOS
        tokens = np.concatenate([tokens, eos], axis=0)

    return tokens.astype(np.int32)


def split_deepcad_bodies_np(tokens: np.ndarray):
    """
    DeepCAD token sequence를 SOL...EXT 단위 body/sketch segment로 분리.
    여러 SOL loop가 하나의 EXT 앞에 있어도 첫 SOL부터 EXT까지 한 body로 본다.
    """
    tokens = trim_after_eos(tokens)

    bodies = []
    start = None
    valid_end = 0

    for i in range(tokens.shape[0]):
        c = int(tokens[i, 0])

        if c < 0 or c == EOS:
            break

        valid_end = i + 1

        if start is None:
            start = i

        if c == EXT:
            bodies.append(tokens[start:i + 1].copy())
            start = None

    if start is not None and start < valid_end:
        bodies.append(tokens[start:valid_end].copy())

    if not bodies:
        sol = np.full((1, 17), PAD_V, dtype=np.int32)
        sol[0, 0] = SOL

        ext = np.full((1, 17), PAD_V, dtype=np.int32)
        ext[0, 0] = EXT
        ext[0, 1:1 + VALID_PAR[EXT]] = 0

        bodies = [np.concatenate([sol, ext], axis=0)]

    return bodies


def _safe_q(v, default=0.5):
    try:
        v = int(v)

        if v < 0:
            return float(default)

        return float(v) / float(QMAX)

    except Exception:
        return float(default)


def body_sort_key_np(body: np.ndarray):
    """
    재구성/출력용 canonical order.
    학습 loss는 Hungarian이라 순서에 의존하지 않음.
    """
    if body is None or len(body) == 0:
        return (999, 999, 999, 999, 999)

    cmds = body[:, 0].astype(np.int64)
    ext_rows = body[cmds == EXT]

    if len(ext_rows):
        p = ext_rows[0, 1:]

        scale = _safe_q(p[P_SCALE], 0.5)
        px = _safe_q(p[P_PX], 0.5)
        py = _safe_q(p[P_PY], 0.5)
        pz = _safe_q(p[P_PZ], 0.5)
        e1 = _safe_q(p[P_E1], 0.5)

    else:
        scale, px, py, pz, e1 = 0.5, 0.5, 0.5, 0.5, 0.5

    return (-scale, pz, px, py, -e1, len(body))


def canonicalize_body_list(bodies):
    return sorted(list(bodies), key=body_sort_key_np)


def bodies_to_sequence_np(bodies, sort_bodies=True):
    bodies = [b for b in bodies if b is not None and len(b) > 0]

    if sort_bodies:
        bodies = canonicalize_body_list(bodies)

    eos = np.full((1, 17), PAD_V, dtype=np.int32)
    eos[0, 0] = EOS

    if not bodies:
        return eos

    return np.concatenate(
        [b.astype(np.int32) for b in bodies] + [eos],
        axis=0,
    ).astype(np.int32)


def pad_body_np(body: np.ndarray, max_body_len: int):
    body = body.astype(np.int32)

    if body.shape[0] > max_body_len:
        body = body[:max_body_len].copy()

        if not (body[:, 0] == EXT).any():
            ext = np.full((17,), PAD_V, dtype=np.int32)
            ext[0] = EXT
            ext[1:1 + VALID_PAR[EXT]] = 0
            body[-1] = ext

    L = body.shape[0]

    if L < max_body_len:
        pad = np.full((max_body_len - L, 17), PAD_V, dtype=np.int32)
        body = np.concatenate([body, pad], axis=0)

    return body.astype(np.float32)


def decode_logits_to_bodies_np(
    cmd_logits,
    prm_logits,
    present_logits,
    threshold=0.5,
    force_topk=None,
):
    """
    model output → body list.

    cmd_logits     : (M,T,N_CMD)
    prm_logits     : (M,T,N_ARGS,N_QUANT)
    present_logits : (M,)
    """
    cmd = np.argmax(cmd_logits, axis=-1).astype(np.int32)
    prm = np.argmax(prm_logits, axis=-1).astype(np.int32)

    prob = 1.0 / (1.0 + np.exp(-present_logits.astype(np.float32)))

    M, T = cmd.shape

    if force_topk is not None:
        keep = np.zeros(M, dtype=bool)
        idx = np.argsort(-prob)[:int(force_topk)]
        keep[idx] = True

    else:
        keep = prob >= threshold

        if not keep.any():
            keep[np.argmax(prob)] = True

    bodies = []

    for m in range(M):
        if not keep[m]:
            continue

        rows = []

        for t in range(T):
            c = int(cmd[m, t])

            row = np.full((17,), PAD_V, dtype=np.int32)
            row[0] = c

            n_valid = VALID_PAR.get(c, 0)

            if n_valid > 0:
                row[1:1 + n_valid] = prm[m, t, :n_valid]

            rows.append(row)

            if c == EXT:
                break

        if not rows:
            continue

        if int(rows[0][0]) != SOL:
            sol = np.full((17,), PAD_V, dtype=np.int32)
            sol[0] = SOL
            rows.insert(0, sol)

        if not any(int(r[0]) == EXT for r in rows):
            ext = np.full((17,), PAD_V, dtype=np.int32)
            ext[0] = EXT
            ext[1:1 + VALID_PAR[EXT]] = 0
            rows.append(ext)

        bodies.append(np.stack(rows, axis=0).astype(np.int32))

    return bodies


def summarize_tokens(tokens: np.ndarray, max_rows=80):
    tokens = trim_after_eos(tokens)
    lines = []

    for i, row in enumerate(tokens[:max_rows]):
        c = int(row[0])
        name = CMD_NAME.get(c, str(c))
        nv = VALID_PAR.get(c, 0)
        args = row[1:1 + nv].tolist() if nv else []

        lines.append(f"{i:03d}: {name:<6s} {args}")

    if len(tokens) > max_rows:
        lines.append(f"... ({len(tokens) - max_rows} more rows)")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Frequency / interpolation
# ══════════════════════════════════════════════════════════════
def select_frequency_indices(
    raw_n_freq,
    target_n_freq,
    freq_start,
    freq_end,
    mode="linspace",
):
    raw_n_freq = int(raw_n_freq)
    target_n_freq = int(target_n_freq)

    if target_n_freq <= 0 or target_n_freq >= raw_n_freq:
        idx_sel = np.arange(raw_n_freq, dtype=np.int64)

    else:
        if (mode or "linspace").lower() != "linspace":
            raise ValueError(f"Unknown freq_select_mode: {mode}")

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

    freqs_full = np.linspace(freq_start, freq_end, raw_n_freq).astype(np.float32)
    freqs_sel = freqs_full[idx_sel].astype(np.float32)

    return idx_sel, freqs_full, freqs_sel


def build_interp_matrix_np(freqs_sel, freqs_full):
    freqs_sel = np.asarray(freqs_sel, dtype=np.float32)
    freqs_full = np.asarray(freqs_full, dtype=np.float32)

    W = np.zeros((len(freqs_full), len(freqs_sel)), dtype=np.float32)

    for i, f in enumerate(freqs_full):
        if f <= freqs_sel[0]:
            W[i, 0] = 1.0

        elif f >= freqs_sel[-1]:
            W[i, -1] = 1.0

        else:
            hi = int(np.searchsorted(freqs_sel, f, side="right"))
            lo = hi - 1

            t = float(
                (f - freqs_sel[lo])
                / max(freqs_sel[hi] - freqs_sel[lo], 1e-12)
            )

            W[i, lo] = 1.0 - t
            W[i, hi] = t

    return W


def interpolate_selected_to_full_torch(pred_sel, interp_w):
    return torch.einsum("fs,bsc->bfc", interp_w, pred_sel)


def interpolate_selected_to_full_np(pred_sel, freqs_sel, freqs_full):
    pred_sel = np.asarray(pred_sel, dtype=np.float32)

    single = False

    if pred_sel.ndim == 2:
        pred_sel = pred_sel[None, ...]
        single = True

    out = np.zeros(
        (pred_sel.shape[0], len(freqs_full), pred_sel.shape[-1]),
        dtype=np.float32,
    )

    for n in range(pred_sel.shape[0]):
        for c in range(pred_sel.shape[-1]):
            out[n, :, c] = np.interp(
                freqs_full,
                freqs_sel,
                pred_sel[n, :, c],
            ).astype(np.float32)

    return out[0] if single else out


# ══════════════════════════════════════════════════════════════
# S-param loading
# ══════════════════════════════════════════════════════════════
def _detect_sparam_cols(header):
    col_idx, col_names = [], []

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

    all_s = []
    ref_cols = expected_cols

    if verbose:
        print(f"  S-param files: {len(sparam_files)}")

    for fi, fp in enumerate(sparam_files):
        ext = os.path.splitext(fp)[1].lower()

        if verbose:
            print(f"    [{fi + 1}/{len(sparam_files)}] {os.path.basename(fp)}")

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
            raise ValueError(f"S-param column mismatch: {os.path.basename(fp)}")

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


def filter_return3_sparams(
    sparam_all,
    sparam_names,
    return_pairs=RETURN_PORT_PAIRS,
    verbose=True,
):
    pair_to_cols = {}

    for i, name in enumerate(sparam_names):
        pair = _parse_sparam_pair(name)

        if pair is None:
            continue

        pair_to_cols.setdefault(pair, {"re": None, "im": None})

        if str(name).strip().lower().startswith("re("):
            pair_to_cols[pair]["re"] = i

        elif str(name).strip().lower().startswith("im("):
            pair_to_cols[pair]["im"] = i

    selected_cols, selected_names = [], []

    for pair in return_pairs:
        if pair not in pair_to_cols:
            raise ValueError(
                f"S{pair[0]}{pair[1]} column not found. Current columns: {sparam_names}"
            )

        re_idx = pair_to_cols[pair]["re"]
        im_idx = pair_to_cols[pair]["im"]

        if re_idx is None or im_idx is None:
            raise ValueError(f"S{pair[0]}{pair[1]} requires both re/im columns.")

        selected_cols.extend([re_idx, im_idx])
        selected_names.extend([sparam_names[re_idx], sparam_names[im_idx]])

    out = sparam_all[:, :, selected_cols]

    if verbose:
        print("\n  Return-3 filtering")
        print("    target: S11, S22, S33 only")
        print(f"    before re/im: {sparam_all.shape}")
        print(f"    after  re/im: {out.shape}")

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
class BodySetJointDataset(Dataset):
    def __init__(
        self,
        npy_files,
        sparam_db,
        freqs,
        sparam_db_full,
        freqs_full,
        freq_idx,
        interp_matrix,
        type_ids=None,
        type_names=None,
        max_bodies_cap=16,
        max_body_len_cap=128,
    ):
        self.npy_files = list(npy_files)
        self.type_names = list(type_names) if type_names else ["type1"]

        self.freqs = np.asarray(freqs, dtype=np.float32)
        self.freqs_full = np.asarray(freqs_full, dtype=np.float32)
        self.freq_idx = np.asarray(freq_idx, dtype=np.int64)
        self.interp_matrix = interp_matrix.astype(np.float32)

        self.sparam_db = sparam_db.astype(np.float32)
        self.sparam_db_full = sparam_db_full.astype(np.float32)

        N = len(self.npy_files)

        self.type_ids = (
            np.zeros(N, dtype=np.int64)
            if type_ids is None
            else np.asarray(type_ids, dtype=np.int64)
        )

        self.raw_tokens = []
        self.raw_bodies = []

        body_counts, body_lens = [], []

        for f in self.npy_files:
            t = np.load(f).astype(np.int32)
            t = ensure_eos(t)

            bodies = split_deepcad_bodies_np(t)
            bodies_for_store = canonicalize_body_list(bodies)

            self.raw_bodies.append(bodies_for_store)
            self.raw_tokens.append(bodies_to_sequence_np(bodies_for_store, sort_bodies=False))

            body_counts.append(len(bodies_for_store))

            for b in bodies_for_store:
                body_lens.append(len(b))

        self.max_bodies = min(max(body_counts), int(max_bodies_cap))
        self.max_body_len = min(max(body_lens), int(max_body_len_cap))

        self.body_tokens = []
        self.body_present = []

        for bodies in self.raw_bodies:
            bodies = canonicalize_body_list(bodies)[:self.max_bodies]

            bt = np.full(
                (self.max_bodies, self.max_body_len, 17),
                PAD_V,
                dtype=np.float32,
            )

            bp = np.zeros((self.max_bodies,), dtype=np.float32)

            for i, b in enumerate(bodies):
                bt[i] = pad_body_np(b, self.max_body_len)
                bp[i] = 1.0

            self.body_tokens.append(bt)
            self.body_present.append(bp)

        self.body_tokens = np.stack(self.body_tokens, axis=0).astype(np.float32)
        self.body_present = np.stack(self.body_present, axis=0).astype(np.float32)

        max_cmd, max_prm = -1, -1

        for t in self.raw_tokens:
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
                f"param max={max_prm} >= N_QUANT={N_QUANT}. N_BIT mismatch likely."
            )

        dist = ", ".join(
            f"{name}:{int((self.type_ids == i).sum())}"
            for i, name in enumerate(self.type_names)
        )

        print(f"  loaded {N} body-set token samples + Return-3 dB target")
        print(f"    type distribution       : {dist}")
        print(f"    max_bodies              : {self.max_bodies}")
        print(f"    max_body_len            : {self.max_body_len}")
        print(f"    body count range        : {min(body_counts)} ~ {max(body_counts)}")
        print(f"    body len range          : {min(body_lens)} ~ {max(body_lens)}")
        print(f"    token range             : cmd=[0..{max_cmd}], param=[0..{max_prm}]")
        print(f"    selected S-param shape  : {self.sparam_db.shape}")
        print(f"    full S-param shape      : {self.sparam_db_full.shape}")
        print(f"    selected freq range     : {self.freqs[0]:.4f} ~ {self.freqs[-1]:.4f} GHz")
        print(f"    full freq range         : {self.freqs_full[0]:.4f} ~ {self.freqs_full[-1]:.4f} GHz")

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.body_tokens[idx], dtype=torch.float32),
            torch.tensor(self.body_present[idx], dtype=torch.float32),
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

    freq_idx, freqs_full, freqs_sel = select_frequency_indices(
        raw_n_freq=cfg.raw_n_freq,
        target_n_freq=cfg.n_freq,
        freq_start=cfg.freq_start,
        freq_end=cfg.freq_end,
        mode=cfg.freq_select_mode,
    )

    cfg.n_freq = int(len(freq_idx))
    interp_matrix = build_interp_matrix_np(freqs_sel, freqs_full)

    print("\n  Frequency point selection")
    print(f"    raw_n_freq       : {cfg.raw_n_freq}")
    print(f"    selected n_freq  : {cfg.n_freq}")
    print(f"    first/last index : {int(freq_idx[0])} / {int(freq_idx[-1])}")
    print(f"    first/last freq  : {freqs_sel[0]:.4f} / {freqs_sel[-1]:.4f} GHz")
    print(f"    interp matrix    : {interp_matrix.shape}")

    all_npy, all_sp_reim, all_tid = [], [], []
    sparam_names_ref = None

    for ti in range(n_types):
        npy_dir = cfg.npy_dirs[ti]
        sp_glob = cfg.sparam_globs[ti]
        tname = type_names[ti]
        n_pick = int(n_samples_per_type[ti])

        npy_dir_abs = (
            npy_dir
            if os.path.isabs(npy_dir)
            else os.path.join(script_dir, npy_dir)
        )

        sp_glob_abs = (
            sp_glob
            if os.path.isabs(sp_glob)
            else os.path.join(script_dir, sp_glob)
        )

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
            sp_files,
            cfg.raw_n_freq,
            expected_cols=sparam_names_ref,
            verbose=False,
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

    sparam_return3_reim_raw, _ = filter_return3_sparams(
        sparam_reim_all_raw,
        sparam_names_ref,
        return_pairs=RETURN_PORT_PAIRS,
        verbose=True,
    )

    sparam_db_full = return3_reim_to_db_np(sparam_return3_reim_raw)
    sparam_db = sparam_db_full[:, freq_idx, :]

    print("\n  Convert Return-3 re/im → dB target")
    print(f"    raw re/im shape     : {sparam_return3_reim_raw.shape}")
    print(f"    full dB shape       : {sparam_db_full.shape}")
    print(f"    selected dB shape   : {sparam_db.shape}")
    print(f"    full dB range       : min={sparam_db_full.min():.2f}, max={sparam_db_full.max():.2f}")

    dataset = BodySetJointDataset(
        all_npy,
        sparam_db=sparam_db,
        freqs=freqs_sel,
        sparam_db_full=sparam_db_full,
        freqs_full=freqs_full,
        freq_idx=freq_idx,
        interp_matrix=interp_matrix,
        type_ids=type_ids,
        type_names=type_names,
        max_bodies_cap=cfg.max_bodies_cap,
        max_body_len_cap=cfg.max_body_len_cap,
    )

    return dataset, all_npy, type_ids, type_names, sparam_db


def make_stratified_split(cfg, dataset, type_ids, type_names):
    n_types = len(type_names)
    rng = random.Random(cfg.seed)

    train_idx, val_idx, val_idx_per_type = [], [], {}

    for ti in range(n_types):
        idx_t = [i for i in range(len(dataset)) if int(type_ids[i]) == ti]

        rng.shuffle(idx_t)

        n_val_t = max(1, int(len(idx_t) * cfg.val_ratio))

        val_idx_per_type[ti] = idx_t[:n_val_t]
        val_idx.extend(idx_t[:n_val_t])
        train_idx.extend(idx_t[n_val_t:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    msg = ", ".join(
        f"{type_names[ti]}:{int((type_ids == ti).sum()) - len(val_idx_per_type[ti])}/{len(val_idx_per_type[ti])}"
        for ti in range(n_types)
    )

    print(f"  train={len(train_idx)} val={len(val_idx)}")
    print(f"  per-type train/val: {msg}")

    return train_idx, val_idx, val_idx_per_type


# ══════════════════════════════════════════════════════════════
# Model
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


def causal_mask(sz, device):
    return torch.triu(
        torch.ones(sz, sz, device=device, dtype=torch.bool),
        diagonal=1,
    )


class BodySetDeepCADAE(nn.Module):
    def __init__(
        self,
        max_bodies,
        max_body_len,
        d_model,
        d_param,
        nhead,
        n_body_enc,
        n_set_enc,
        n_body_dec,
        d_ff,
        latent,
        mem_tokens,
        dropout,
        n_pool,
        n_freq_bands,
    ):
        super().__init__()

        assert d_model % nhead == 0

        self.max_bodies = int(max_bodies)
        self.max_body_len = int(max_body_len)
        self.d_model = int(d_model)
        self.d_param = int(d_param)
        self.latent = int(latent)
        self.mem_tokens = int(mem_tokens)
        self.n_pool = int(n_pool)
        self.n_freq_bands = int(n_freq_bands)

        self.cmd_emb = nn.Embedding(
            N_CMD + 1,
            d_model,
            padding_idx=PAD_CMD_INDEX,
        )

        self.param_embs = nn.ModuleList([
            nn.Embedding(
                N_QUANT + 1,
                d_param,
                padding_idx=PAD_PARAM_INDEX,
            )
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

        # body-internal encoder
        self.body_pos_enc = SinPosEnc(
            d_model,
            max_len=max_body_len + 8,
            dropout=dropout,
        )

        body_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.body_encoder = nn.TransformerEncoder(
            body_layer,
            num_layers=n_body_enc,
            norm=nn.LayerNorm(d_model),
        )

        self.body_out_norm = nn.LayerNorm(d_model)

        # body-set encoder: body 간 positional encoding 없음
        set_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.set_encoder = nn.TransformerEncoder(
            set_layer,
            num_layers=n_set_enc,
            norm=nn.LayerNorm(d_model),
        )

        # PMA pooling
        if self.n_pool >= 2:
            self.pool_queries = nn.Parameter(
                torch.randn(self.n_pool, d_model) * 0.02
            )

            self.pool_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=True,
            )

            self.pool_norm = nn.LayerNorm(d_model)
            self.to_z = nn.Linear(self.n_pool * d_model, latent)

        else:
            self.pool_queries = None
            self.pool_attn = None
            self.pool_norm = None
            self.to_z = nn.Linear(d_model, latent)

        # z memory
        self.from_z = nn.Linear(latent, mem_tokens * d_model)

        # body-set decoder
        self.dec_body_slots = nn.Parameter(
            torch.randn(max_bodies, d_model) * 0.02
        )

        self.slot_cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.slot_norm = nn.LayerNorm(d_model)
        self.present_head = nn.Linear(d_model, 1)

        self.dec_token_queries = nn.Parameter(
            torch.randn(max_body_len, d_model) * 0.02
        )

        body_dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.body_decoder = nn.TransformerDecoder(
            body_dec_layer,
            num_layers=n_body_dec,
            norm=nn.LayerNorm(d_model),
        )

        self.cmd_head = nn.Linear(d_model, N_CMD)
        self.param_head = nn.Linear(d_model, N_ARGS * N_QUANT)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _content_embed(self, x):
        """
        x: (B, L, 17)

        return:
          emb      : (B,L,d_model)
          pad_mask : (B,L)
        """
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
            self.param_embs[s](param_idx[:, :, s])
            for s in range(N_ARGS)
        ]

        param_concat = torch.cat(slot_embs, dim=-1)

        if self.n_freq_bands > 0 and self.fourier_freqs is not None:
            p_int = param_idx.clone()
            pad_p = params < 0

            p_int[pad_p] = 0

            p_f = p_int.float() / float(QMAX)

            ang = p_f.unsqueeze(-1) * self.fourier_freqs.view(1, 1, 1, -1)

            sin_p = torch.sin(ang)
            cos_p = torch.cos(ang)

            mask = 1.0 - pad_p.unsqueeze(-1).float()

            sin_p = sin_p * mask
            cos_p = cos_p * mask

            B, L, _ = params.shape

            fourier = torch.cat([sin_p, cos_p], dim=-1)
            fourier = fourier.reshape(
                B,
                L,
                N_ARGS * 2 * self.n_freq_bands,
            )

            param_concat = torch.cat([param_concat, fourier], dim=-1)

        param_e = self.param_proj(param_concat)

        emb = self.emb_norm(cmd_e + param_e)
        emb = emb * (~pad_mask).unsqueeze(-1).float()

        return emb, pad_mask

    def encode(self, body_tokens, body_present):
        """
        body_tokens : (B,M,T,17)
        body_present: (B,M), 1 if valid body
        """
        B, M, T, A = body_tokens.shape

        flat = body_tokens.reshape(B * M, T, A)

        emb, tok_pad = self._content_embed(flat)
        emb = self.body_pos_enc(emb)

        # all-pad body는 attention NaN 방지용으로 첫 token만 unmask
        tok_pad_attn = tok_pad.clone()
        all_pad = tok_pad_attn.all(dim=1)

        if all_pad.any():
            tok_pad_attn[all_pad, 0] = False

        h_tok = self.body_encoder(
            emb,
            src_key_padding_mask=tok_pad_attn,
        )

        valid_tok = (~tok_pad).float().unsqueeze(-1)

        body_vec = (h_tok * valid_tok).sum(dim=1) / valid_tok.sum(dim=1).clamp(min=1.0)
        body_vec = self.body_out_norm(body_vec).view(B, M, self.d_model)

        body_pad = body_present <= 0.5
        body_vec = body_vec * (~body_pad).unsqueeze(-1).float()

        # body-set transformer: no positional encoding
        h_body = self.set_encoder(
            body_vec,
            src_key_padding_mask=body_pad,
        )

        h_body = h_body * (~body_pad).unsqueeze(-1).float()

        if self.n_pool >= 2:
            q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)

            pooled, _ = self.pool_attn(
                q,
                h_body,
                h_body,
                key_padding_mask=body_pad,
                need_weights=False,
            )

            pooled = self.pool_norm(pooled)
            pooled = pooled.reshape(B, self.n_pool * self.d_model)

            z = self.to_z(pooled)

        else:
            valid_body = (~body_pad).float().unsqueeze(-1)
            pooled = (h_body * valid_body).sum(dim=1) / valid_body.sum(dim=1).clamp(min=1.0)

            z = self.to_z(pooled)

        return z

    def _memory_from_z(self, z):
        return self.from_z(z).view(
            z.size(0),
            self.mem_tokens,
            self.d_model,
        )

    def decode_body_set(self, z):
        """
        z → body slots → body internal token sequence.

        v2 변경: body decoder self-attention 에 causal mask 추가.

        return:
          present_logits: (B,M)
          cmd_logits    : (B,M,T,N_CMD)
          prm_logits    : (B,M,T,N_ARGS,N_QUANT)
        """
        B = z.size(0)

        memory = self._memory_from_z(z)

        slot_q = self.dec_body_slots.unsqueeze(0).expand(B, -1, -1)

        slot_ctx, _ = self.slot_cross_attn(
            slot_q,
            memory,
            memory,
            need_weights=False,
        )

        slot = self.slot_norm(slot_q + slot_ctx)

        present_logits = self.present_head(slot).squeeze(-1)

        tok_q = slot.unsqueeze(2) + self.dec_token_queries.view(
            1,
            1,
            self.max_body_len,
            self.d_model,
        )

        tok_q = tok_q.reshape(
            B * self.max_bodies,
            self.max_body_len,
            self.d_model,
        )

        mem_rep = memory.unsqueeze(1).expand(
            B,
            self.max_bodies,
            self.mem_tokens,
            self.d_model,
        )

        mem_rep = mem_rep.reshape(
            B * self.max_bodies,
            self.mem_tokens,
            self.d_model,
        )

        # v2: body 내부 token 위치 t 는 0..t 만 attend 하도록 causal mask
        tgt_mask = causal_mask(self.max_body_len, z.device)

        h = self.body_decoder(
            tgt=tok_q,
            memory=mem_rep,
            tgt_mask=tgt_mask,
        )

        cmd_logits = self.cmd_head(h).view(
            B,
            self.max_bodies,
            self.max_body_len,
            N_CMD,
        )

        prm_logits = self.param_head(h).view(
            B,
            self.max_bodies,
            self.max_body_len,
            N_ARGS,
            N_QUANT,
        )

        return present_logits, cmd_logits, prm_logits

    def forward(self, body_tokens, body_present):
        z = self.encode(body_tokens, body_present)

        present_logits, cmd_logits, prm_logits = self.decode_body_set(z)

        return present_logits, cmd_logits, prm_logits, z

    @torch.no_grad()
    def generate(self, z, threshold=0.5, force_topk=None):
        self.eval()

        present_logits, cmd_logits, prm_logits = self.decode_body_set(z)

        out = []

        for b in range(z.size(0)):
            bodies = decode_logits_to_bodies_np(
                cmd_logits[b].detach().cpu().numpy(),
                prm_logits[b].detach().cpu().numpy(),
                present_logits[b].detach().cpu().numpy(),
                threshold=threshold,
                force_topk=force_topk,
            )

            seq = bodies_to_sequence_np(bodies, sort_bodies=True)
            out.append(seq)

        return out


class SparamCommonResidualMLP(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_freq,
        common_curve,
        hidden_mult=2.0,
        dropout=0.3,
        residual_scale=1.0,
        zero_init_residual=True,
    ):
        super().__init__()

        self.n_freq = int(n_freq)
        self.n_out = 3
        self.residual_scale = float(residual_scale)

        common_curve = torch.as_tensor(common_curve, dtype=torch.float32)

        if common_curve.shape != (self.n_freq, 3):
            raise ValueError(f"common_curve shape mismatch: {tuple(common_curve.shape)}")

        self.register_buffer("common_curve", common_curve)

        h1 = max(int(latent_dim * hidden_mult), 128)
        h2 = max(int(latent_dim * hidden_mult), 128)

        self.net = nn.Sequential(
            nn.Linear(latent_dim, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(h2, self.n_freq * 3),
        )

        if zero_init_residual:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, z, return_parts=False):
        residual_raw = self.net(z).view(-1, self.n_freq, 3)
        residual = self.residual_scale * residual_raw

        pred = self.common_curve.unsqueeze(0) + residual

        if return_parts:
            return pred, residual

        return pred


# ══════════════════════════════════════════════════════════════
# Losses
# ══════════════════════════════════════════════════════════════
def _solve_assignment(cost):
    if _hungarian is not None:
        return _hungarian(cost)

    # fallback: greedy
    cost = np.asarray(cost)
    rows, cols = [], []
    used_r, used_c = set(), set()
    n_r, n_c = cost.shape

    for _ in range(min(n_r, n_c)):
        best = None
        for r in range(n_r):
            if r in used_r:
                continue
            for c in range(n_c):
                if c in used_c:
                    continue
                v = float(cost[r, c])
                if best is None or v < best[0]:
                    best = (v, r, c)
        if best is None:
            break
        _, r, c = best
        used_r.add(r)
        used_c.add(c)
        rows.append(r)
        cols.append(c)

    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def _vectorized_pair_cost(
    cmd_logits_b,
    prm_logits_b,
    target_bodies_b,
    true_ids,
    present_logits_b,
    cfg,
):
    """
    v2: 벡터화된 (M, Nt) cost matrix.

    cmd_logits_b     : (M,T,N_CMD)
    prm_logits_b     : (M,T,N_ARGS,N_QUANT)
    target_bodies_b  : (M,T,17)
    true_ids         : list[int] length=Nt
    present_logits_b : (M,)

    returns: torch tensor (M, Nt). caller 는 detach 후 numpy 로 변환해 Hungarian 에 넘김.
    """
    M, T, _ = cmd_logits_b.shape
    Nt = len(true_ids)

    if Nt == 0:
        return None

    device = cmd_logits_b.device

    true_t = target_bodies_b[true_ids]                # (Nt,T,17)
    cmd_t = true_t[:, :, 0].long()                    # (Nt,T)
    prm_t = true_t[:, :, 1:].long()                   # (Nt,T,N_ARGS)

    valid_cmd = (cmd_t >= 0).float()                  # (Nt,T)
    valid_prm = (prm_t >= 0).float()                  # (Nt,T,N_ARGS)

    cmd_target = cmd_t.clamp(0, N_CMD - 1)            # (Nt,T)
    prm_target = prm_t.clamp(0, QMAX)                 # (Nt,T,N_ARGS)

    log_p_cmd = F.log_softmax(cmd_logits_b, dim=-1)   # (M,T,N_CMD)
    log_p_prm = F.log_softmax(prm_logits_b, dim=-1)   # (M,T,N_ARGS,N_QUANT)

    # ─ cmd cost: 완전 벡터화 (N_CMD 작아서 expand 비용 작음) ─
    cmd_target_g = cmd_target.unsqueeze(0).expand(M, -1, -1).unsqueeze(-1)   # (M,Nt,T,1)
    log_p_cmd_g = log_p_cmd.unsqueeze(1).expand(-1, Nt, -1, -1)              # (M,Nt,T,N_CMD)
    cmd_nll = -log_p_cmd_g.gather(-1, cmd_target_g).squeeze(-1)              # (M,Nt,T)

    valid_cmd_b = valid_cmd.unsqueeze(0)                                     # (1,Nt,T)
    cmd_cnt = valid_cmd_b.sum(dim=-1).clamp(min=1.0)                         # (1,Nt)
    cmd_cost = (cmd_nll * valid_cmd_b).sum(dim=-1) / cmd_cnt                 # (M,Nt)

    # ─ prm cost: N_QUANT 가 커서 전체 expand 는 메모리 폭발 → j 루프 (Nt 이내) ─
    prm_cost = torch.zeros(M, Nt, device=device, dtype=log_p_prm.dtype)

    for j in range(Nt):
        idx_j = prm_target[j].unsqueeze(0).unsqueeze(-1).expand(M, -1, -1, -1)  # (M,T,N_ARGS,1)
        nll_j = -log_p_prm.gather(-1, idx_j).squeeze(-1)                       # (M,T,N_ARGS)

        v_j = valid_prm[j].unsqueeze(0)                                        # (1,T,N_ARGS)
        cnt_j = v_j.sum().clamp(min=1.0)
        prm_cost[:, j] = (nll_j * v_j).sum(dim=(-2, -1)) / cnt_j

    base = cfg.w_cmd * cmd_cost + cfg.w_prm * prm_cost                         # (M,Nt)

    # ─ presence term: confidently-present slot 이 매칭에 우선 ─
    # NLL of "slot is present" = -log σ(present_logit)
    if getattr(cfg, "w_match_presence", 0.0) > 0:
        pres_nll = -F.logsigmoid(present_logits_b)                             # (M,)
        base = base + cfg.w_match_presence * pres_nll.unsqueeze(-1)            # broadcast to (M,Nt)

    return base


def _pair_token_loss(cmd_logits, prm_logits, target_body, cfg):
    """
    Gradient-flow 용 단일 (slot, true body) pair loss.
    매칭된 pair 만 호출되므로 batch 전체에서 O(n_pairs) 회 호출.
    """
    T = target_body.size(0)

    cmd_t = target_body[:, 0].long()
    valid_cmd = cmd_t >= 0

    if valid_cmd.any():
        cmd_ce = F.cross_entropy(
            cmd_logits.reshape(T, N_CMD),
            cmd_t.clamp(0, N_CMD - 1),
            reduction="none",
        )

        cmd_loss = (
            cmd_ce * valid_cmd.float()
        ).sum() / valid_cmd.float().sum().clamp(min=1.0)

    else:
        cmd_loss = target_body.new_zeros(())

    prm_t = target_body[:, 1:].long()
    valid_prm = prm_t >= 0

    if valid_prm.any():
        prm_ce = F.cross_entropy(
            prm_logits.reshape(T * N_ARGS, N_QUANT),
            prm_t.clamp(0, QMAX).reshape(T * N_ARGS),
            reduction="none",
        ).reshape(T, N_ARGS)

        prm_loss = (
            prm_ce * valid_prm.float()
        ).sum() / valid_prm.float().sum().clamp(min=1.0)

    else:
        prm_loss = target_body.new_zeros(())

    return cfg.w_cmd * cmd_loss + cfg.w_prm * prm_loss, cmd_loss, prm_loss


def bodyset_ae_loss_fn(
    present_logits,
    cmd_logits,
    prm_logits,
    target_bodies,
    target_present,
    cfg,
):
    """
    Hungarian matching 기반 body-set reconstruction loss (v2).

    변경점:
      - cost matrix 계산: Python 이중 루프 → 벡터화
      - cost matrix 에 presence NLL 가중치 합산
    """
    B, M, T, _ = target_bodies.shape

    total_token = target_bodies.new_zeros(())
    total_cmd = target_bodies.new_zeros(())
    total_prm = target_bodies.new_zeros(())

    n_pairs = 0

    presence_labels = torch.zeros_like(present_logits)

    with torch.no_grad():
        assignments = []

        for b in range(B):
            true_mask = target_present[b] > 0.5
            true_ids = torch.where(true_mask)[0].detach().cpu().numpy().tolist()

            if not true_ids:
                assignments.append([])
                continue

            cost = _vectorized_pair_cost(
                cmd_logits[b].detach(),
                prm_logits[b].detach(),
                target_bodies[b].detach(),
                true_ids,
                present_logits[b].detach(),
                cfg,
            )

            if not torch.isfinite(cost).all():
                cost = torch.nan_to_num(
                    cost,
                    nan=1e6,
                    posinf=1e6,
                    neginf=-1e6,
                )

            cost_np = cost.detach().cpu().numpy()

            row_ind, col_ind = _solve_assignment(cost_np)

            pairs = []
            for r, c in zip(row_ind, col_ind):
                if c < len(true_ids):
                    pairs.append((int(r), int(true_ids[int(c)])))

            assignments.append(pairs)

    for b in range(B):
        for m, tid in assignments[b]:
            presence_labels[b, m] = 1.0

            pair_loss, cmd_l, prm_l = _pair_token_loss(
                cmd_logits[b, m],
                prm_logits[b, m],
                target_bodies[b, tid],
                cfg,
            )

            total_token = total_token + pair_loss
            total_cmd = total_cmd + cmd_l
            total_prm = total_prm + prm_l

            n_pairs += 1

    if n_pairs > 0:
        total_token = total_token / n_pairs
        total_cmd = total_cmd / n_pairs
        total_prm = total_prm / n_pairs

    presence_loss = F.binary_cross_entropy_with_logits(
        present_logits,
        presence_labels,
    )

    total = total_token + cfg.w_presence * presence_loss

    comp = {
        "ae_total": float(total.item()),
        "token": float(total_token.item()),
        "cmd": float(total_cmd.item()),
        "prm": float(total_prm.item()),
        "presence": float(presence_loss.item()),
        "pairs": float(n_pairs),
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
    if z.dim() != 2 or z.size(0) <= 1:
        zero = z.new_zeros(())
        return zero, zero

    B, D = z.shape

    zc = z - z.mean(dim=0, keepdim=True)

    std = torch.sqrt(zc.var(dim=0, unbiased=False) + eps)

    var_loss = F.relu(var_target - std).mean()

    cov = (zc.T @ zc) / max(B - 1, 1)

    off = cov - torch.diag(torch.diag(cov))

    cov_loss = (off ** 2).sum() / D

    return var_loss, cov_loss


# ══════════════════════════════════════════════════════════════
# Training / evaluation
# ══════════════════════════════════════════════════════════════
def run_epoch(ae, mlp, loader, optimizer, device, cfg, interp_w, train_mode=True):
    if train_mode:
        ae.train()
        mlp.train()
    else:
        ae.eval()
        mlp.eval()

    acc = {
        "total": 0.0,
        "ae": 0.0,
        "token": 0.0,
        "cmd": 0.0,
        "prm": 0.0,
        "presence": 0.0,
        "sp": 0.0,
        "rmse_db_full": 0.0,
        "rmse_db_sel": 0.0,
        "var": 0.0,
        "cov": 0.0,
        "z_std": 0.0,
        "z_norm": 0.0,
        "res_abs": 0.0,
        "grad_norm": 0.0,
    }

    n = 0

    for batch in loader:
        body_tok, body_pres, sel_db, full_db, _tid = batch

        body_tok = body_tok.to(device)
        body_pres = body_pres.to(device)
        sel_db = sel_db.to(device)
        full_db = full_db.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            present_logits, cmd_logits, prm_logits, z = ae(body_tok, body_pres)

            ae_loss, ae_comp = bodyset_ae_loss_fn(
                present_logits,
                cmd_logits,
                prm_logits,
                target_bodies=body_tok,
                target_present=body_pres,
                cfg=cfg,
            )

            pred_sel_db, residual = mlp(z, return_parts=True)

            sp_loss, sp_comp, _ = sparam_full_interp_loss(
                pred_sel_db,
                sel_db,
                full_db,
                interp_w,
                cfg,
            )

            total = ae_loss + cfg.w_sparam * sp_loss

            if cfg.use_vicreg and z.size(0) > 1:
                var_loss, cov_loss = vicreg_z_loss(
                    z,
                    var_target=cfg.vicreg_var_target,
                )

                total = total + cfg.w_var * var_loss + cfg.w_cov * cov_loss

            else:
                var_loss = z.new_zeros(())
                cov_loss = z.new_zeros(())

            if train_mode:
                total.backward()

                params = list(ae.parameters()) + list(mlp.parameters())

                gn = nn.utils.clip_grad_norm_(params, cfg.grad_clip)

                optimizer.step()

                acc["grad_norm"] += float(gn)

        with torch.no_grad():
            acc["total"] += float(total.item())
            acc["ae"] += ae_comp["ae_total"]
            acc["token"] += ae_comp["token"]
            acc["cmd"] += ae_comp["cmd"]
            acc["prm"] += ae_comp["prm"]
            acc["presence"] += ae_comp["presence"]

            acc["sp"] += float(sp_loss.item())
            acc["rmse_db_full"] += sp_comp["rmse_db_full"]
            acc["rmse_db_sel"] += sp_comp["rmse_db_sel"]

            acc["var"] += float(var_loss.item())
            acc["cov"] += float(cov_loss.item())

            if z.size(0) > 1:
                acc["z_std"] += float(z.std(dim=0).mean().item())

            acc["z_norm"] += float(z.norm(dim=1).mean().item())
            acc["res_abs"] += float(residual.abs().mean().item())

        n += 1

    if n == 0:
        return {**acc, "n_batches": 0}

    out = {k: v / n for k, v in acc.items()}
    out["n_batches"] = n

    return out


def build_common_curve_and_print_baseline(
    dataset,
    train_idx,
    val_idx_per_type,
    type_names,
):
    section("Common curve baseline RMSE")

    train_curves = dataset.sparam_db[train_idx]
    common_sel = train_curves.mean(axis=0).astype(np.float32)

    common_full = interpolate_selected_to_full_np(
        common_sel,
        dataset.freqs,
        dataset.freqs_full,
    )

    total_sel_sq, total_sel_n = 0.0, 0
    total_full_sq, total_full_n = 0.0, 0

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

        print(f"  [{tname}] common selected RMSE : {sel_rmse:.3f} dB")
        print(f"  [{tname}] common full RMSE     : {full_rmse:.3f} dB")

        total_sel_sq += float(np.sum((pred_sel - true_sel) ** 2))
        total_sel_n += int(np.prod(true_sel.shape))

        total_full_sq += float(np.sum((pred_full - true_full) ** 2))
        total_full_n += int(np.prod(true_full.shape))

    if total_sel_n:
        print(f"\n  TOTAL common selected RMSE : {math.sqrt(total_sel_sq / total_sel_n):.3f} dB")

    if total_full_n:
        print(f"  TOTAL common full RMSE     : {math.sqrt(total_full_sq / total_full_n):.3f} dB")

    return common_sel


@torch.no_grad()
def collect_latents(ae, dataset, indices, device, batch_size=32):
    ae.eval()

    z_list, type_list = [], []

    for i in range(0, len(indices), batch_size):
        bi = indices[i:i + batch_size]

        body_tok = torch.stack([
            torch.tensor(dataset.body_tokens[k], dtype=torch.float32)
            for k in bi
        ]).to(device)

        body_pres = torch.stack([
            torch.tensor(dataset.body_present[k], dtype=torch.float32)
            for k in bi
        ]).to(device)

        z = ae.encode(body_tok, body_pres)

        z_list.append(z.cpu().numpy())

        type_list.extend([int(dataset.type_ids[k]) for k in bi])

    return np.concatenate(z_list, axis=0), np.asarray(type_list, dtype=np.int64)


@torch.no_grad()
def evaluate_sparam_predictions(ae, mlp, dataset, val_idx_per_type, type_names, device):
    ae.eval()
    mlp.eval()

    total_sel_sq = 0.0
    total_sel_n = 0

    total_full_sq = 0.0
    total_full_n = 0

    per_type = {}

    for ti, tname in enumerate(type_names):
        idxs = val_idx_per_type.get(ti, [])

        if not idxs:
            continue

        pred_sel_all, true_sel_all, true_full_all = [], [], []

        for idx in idxs:
            body_tok = torch.tensor(
                dataset.body_tokens[idx],
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

            body_pres = torch.tensor(
                dataset.body_present[idx],
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

            z = ae.encode(body_tok, body_pres)

            pred_sel = mlp(z).cpu().numpy()[0]

            pred_sel_all.append(pred_sel)
            true_sel_all.append(dataset.sparam_db[idx])
            true_full_all.append(dataset.sparam_db_full[idx])

        pred_sel_all = np.asarray(pred_sel_all, dtype=np.float32)
        true_sel_all = np.asarray(true_sel_all, dtype=np.float32)
        true_full_all = np.asarray(true_full_all, dtype=np.float32)

        pred_full_all = interpolate_selected_to_full_np(
            pred_sel_all,
            dataset.freqs,
            dataset.freqs_full,
        )

        ch_sel, ch_full = [], []

        for c in range(3):
            ch_sel.append(float(np.sqrt(np.mean(
                (pred_sel_all[:, :, c] - true_sel_all[:, :, c]) ** 2
            ))))

            ch_full.append(float(np.sqrt(np.mean(
                (pred_full_all[:, :, c] - true_full_all[:, :, c]) ** 2
            ))))

        avg_sel = float(np.sqrt(np.mean((pred_sel_all - true_sel_all) ** 2)))
        avg_full = float(np.sqrt(np.mean((pred_full_all - true_full_all) ** 2)))

        per_type[tname] = {
            "n": len(idxs),
            "ch_sel": ch_sel,
            "ch_full": ch_full,
            "avg_sel": avg_sel,
            "avg_full": avg_full,
        }

        total_sel_sq += float(np.sum((pred_sel_all - true_sel_all) ** 2))
        total_sel_n += int(np.prod(true_sel_all.shape))

        total_full_sq += float(np.sum((pred_full_all - true_full_all) ** 2))
        total_full_n += int(np.prod(true_full_all.shape))

    return {
        "per_type": per_type,
        "overall_selected_rmse": math.sqrt(total_sel_sq / max(total_sel_n, 1)),
        "overall_full_rmse": math.sqrt(total_full_sq / max(total_full_n, 1)),
    }


@torch.no_grad()
def diagnose_reconstruction(ae, dataset, indices, device, max_samples=8):
    ae.eval()

    sel = list(indices)[:max_samples]

    cmd_correct = 0
    cmd_total = 0

    prm_abs = 0.0
    prm_count = 0

    body_count_match = 0

    for idx in sel:
        body_tok = torch.tensor(
            dataset.body_tokens[idx],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        body_pres = torch.tensor(
            dataset.body_present[idx],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        z = ae.encode(body_tok, body_pres)

        n_true = int(dataset.body_present[idx].sum())

        gen_seq = ae.generate(z, force_topk=n_true)[0]
        true_seq = dataset.raw_tokens[idx]

        pred_b = split_deepcad_bodies_np(gen_seq)
        true_b = split_deepcad_bodies_np(true_seq)

        if len(pred_b) == len(true_b):
            body_count_match += 1

        M_pred, M_true = len(pred_b), len(true_b)

        if M_pred and M_true:
            cost = np.zeros((M_pred, M_true), dtype=np.float32)

            for i, pb in enumerate(pred_b):
                for j, tb in enumerate(true_b):
                    L = min(len(pb), len(tb))

                    if L <= 0:
                        cost[i, j] = 1e6

                    else:
                        cost[i, j] = float(
                            np.mean(pb[:L, 0] != tb[:L, 0])
                            + abs(len(pb) - len(tb)) * 0.1
                        )

            rows, cols = _solve_assignment(cost)

            for r, c in zip(rows, cols):
                pb, tb = pred_b[int(r)], true_b[int(c)]
                L = min(len(pb), len(tb))

                if L <= 0:
                    continue

                cmd_correct += int((pb[:L, 0] == tb[:L, 0]).sum())
                cmd_total += L

                pv = pb[:L, 1:]
                tv = tb[:L, 1:]

                valid = (pv >= 0) & (tv >= 0)

                if valid.any():
                    prm_abs += float(
                        np.abs(
                            pv[valid].astype(np.int64)
                            - tv[valid].astype(np.int64)
                        ).sum()
                    )
                    prm_count += int(valid.sum())

    cmd_acc = cmd_correct / max(cmd_total, 1)
    prm_mae = prm_abs / max(prm_count, 1)

    print("  Recon diagnostic:")
    print(f"    samples evaluated : {len(sel)}")
    print(f"    body count match  : {body_count_match}/{len(sel)}")
    print(f"    cmd acc matched   : {cmd_acc * 100:.2f}%")
    print(f"    prm MAE quant     : {prm_mae:.3f}")
    print(f"    prm MAE norm      : {prm_mae / QMAX:.5f}")

    return {
        "cmd_acc": cmd_acc,
        "prm_mae": prm_mae,
        "body_count_match": body_count_match,
    }


def print_eval_diagnostics(eval_metrics, recon_diag, latent_diag, type_names):
    section("EVALUATION DIAGNOSTICS")

    for tname in type_names:
        if tname not in eval_metrics["per_type"]:
            continue

        r = eval_metrics["per_type"][tname]

        print(f"\n  [{tname}] Return-3 RMSE")
        print(f"  {'channel':<8s} {'selected':>10s} {'full401':>10s}")
        print("  " + "-" * 32)

        for i, lbl in enumerate(RETURN_LABELS):
            print(f"  {lbl:<8s} {r['ch_sel'][i]:>10.3f} {r['ch_full'][i]:>10.3f}")

        print("  " + "-" * 32)
        print(f"  {'AVG':<8s} {r['avg_sel']:>10.3f} {r['avg_full']:>10.3f}")

    print(f"\n  Overall selected-point RMSE : {eval_metrics['overall_selected_rmse']:.4f} dB")
    print(f"  Overall full-grid RMSE     : {eval_metrics['overall_full_rmse']:.4f} dB")

    print("\n  Reconstruction")
    print(f"    body count match : {recon_diag['body_count_match']}")
    print(f"    cmd_acc          : {recon_diag['cmd_acc'] * 100:.2f}%")
    print(f"    prm_MAE          : {recon_diag['prm_mae']:.3f}")

    print("\n  Latent")
    print(f"    z shape          : {latent_diag['z_shape']}")
    print(f"    z_std mean       : {latent_diag['z_std_mean']:.4f}")
    print(f"    z_norm mean      : {latent_diag['z_norm_mean']:.4f}")
    print(f"    dead dims        : {latent_diag['dead_dims']} / {latent_diag['latent_dim']}")


def diagnose_latent(ae, dataset, indices, device):
    z_all, _ = collect_latents(
        ae,
        dataset,
        indices,
        device,
        batch_size=32,
    )

    std = z_all.std(axis=0)
    norm = np.linalg.norm(z_all, axis=1)

    return {
        "z_shape": z_all.shape,
        "z_std_mean": float(std.mean()),
        "z_norm_mean": float(norm.mean()),
        "dead_dims": int((std < 1e-3).sum()),
        "latent_dim": int(z_all.shape[1]),
    }


def plot_training_curves(hist):
    ep = np.arange(1, len(hist["tr_full"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="white")

    axes[0].plot(ep, hist["tr_full"], label="train full RMSE")
    axes[0].plot(ep, hist["va_full"], label="val full RMSE")
    axes[0].set_title("Full-grid RMSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("dB")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(ep, hist["tr_total"], label="train total")
    axes[1].plot(ep, hist["va_total"], label="val total")
    axes[1].set_title("Total loss")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(ep, hist["presence"], label="presence")
    axes[2].plot(ep, hist["cmd"], label="cmd")
    axes[2].plot(ep, hist["prm"], label="prm")
    axes[2].set_title("Body-set AE losses")
    axes[2].set_xlabel("epoch")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    plt.tight_layout()


@torch.no_grad()
def visualize_sparam_predictions(
    ae,
    mlp,
    dataset,
    val_indices,
    device,
    n_samples=3,
    seed=0,
):
    ae.eval()
    mlp.eval()

    val_indices = list(val_indices)

    if not val_indices:
        return

    rng = np.random.default_rng(seed)

    n_pick = min(n_samples, len(val_indices))

    sel = sorted([
        val_indices[i]
        for i in rng.choice(len(val_indices), size=n_pick, replace=False)
    ])

    freqs_full = dataset.freqs_full
    freqs_sel = dataset.freqs

    common_sel = mlp.common_curve.detach().cpu().numpy()
    common_full = interpolate_selected_to_full_np(
        common_sel,
        freqs_sel,
        freqs_full,
    )

    fig, axes = plt.subplots(
        n_pick,
        3,
        figsize=(15, 3.5 * n_pick),
        facecolor="white",
        sharex=True,
    )

    if n_pick == 1:
        axes = axes.reshape(1, 3)

    for row, idx in enumerate(sel):
        body_tok = torch.tensor(
            dataset.body_tokens[idx],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        body_pres = torch.tensor(
            dataset.body_present[idx],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        z = ae.encode(body_tok, body_pres)

        pred_sel = mlp(z).cpu().numpy()[0]
        pred_full = interpolate_selected_to_full_np(
            pred_sel,
            freqs_sel,
            freqs_full,
        )

        true_full = dataset.sparam_db_full[idx]

        t_name = dataset.type_names[int(dataset.type_ids[idx])]

        for c, lbl in enumerate(RETURN_LABELS):
            ax = axes[row, c]

            ax.plot(
                freqs_full,
                common_full[:, c],
                lw=1.2,
                ls=":",
                label="common" if row == 0 and c == 0 else None,
            )

            ax.plot(
                freqs_full,
                true_full[:, c],
                lw=1.8,
                label="truth" if row == 0 and c == 0 else None,
            )

            ax.plot(
                freqs_full,
                pred_full[:, c],
                lw=1.5,
                ls="--",
                label="pred" if row == 0 and c == 0 else None,
            )

            rmse = float(np.sqrt(np.mean((pred_full[:, c] - true_full[:, c]) ** 2)))

            ax.set_title(
                f"[{t_name}] idx={idx} {lbl} RMSE={rmse:.2f} dB",
                fontsize=9,
            )

            ax.grid(True, alpha=0.25)

            if c == 0:
                ax.set_ylabel("|S| [dB]")

            if row == n_pick - 1:
                ax.set_xlabel("frequency [GHz]")

            if row == 0 and c == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()


@torch.no_grad()
def print_reconstruction_examples(ae, dataset, indices, device, n_preview=3):
    subsection("Body-set reconstruction examples")

    for k, idx in enumerate(list(indices)[:n_preview]):
        body_tok = torch.tensor(
            dataset.body_tokens[idx],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        body_pres = torch.tensor(
            dataset.body_present[idx],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        n_true = int(dataset.body_present[idx].sum())

        z = ae.encode(body_tok, body_pres)

        gen_seq = ae.generate(z, force_topk=n_true)[0]
        true_seq = dataset.raw_tokens[idx]

        t_name = dataset.type_names[int(dataset.type_ids[idx])]

        print("\n" + "-" * 72)
        print(f"[example {k + 1}] idx={idx}, type={t_name}, true bodies={n_true}")

        print("TRUE:")
        print(summarize_tokens(true_seq, max_rows=40))

        print("\nRECON:")
        print(summarize_tokens(gen_seq, max_rows=40))


def train_bodyset_model(
    cfg,
    dataset,
    type_names,
    train_idx,
    val_idx,
    val_idx_per_type,
    common_curve,
    device,
):
    set_seed(cfg.seed)

    interp_w = torch.tensor(
        dataset.interp_matrix,
        dtype=torch.float32,
        device=device,
    )

    section("BUILD MODEL — Body-Set Encoder + Body-Set Decoder (v2)")

    ae = BodySetDeepCADAE(
        max_bodies=dataset.max_bodies,
        max_body_len=dataset.max_body_len,
        d_model=cfg.d_model,
        d_param=cfg.d_param,
        nhead=cfg.nhead,
        n_body_enc=cfg.n_body_enc,
        n_set_enc=cfg.n_set_enc,
        n_body_dec=cfg.n_body_dec,
        d_ff=cfg.d_ff,
        latent=cfg.latent,
        mem_tokens=cfg.mem_tokens,
        dropout=cfg.dropout,
        n_pool=cfg.n_pool,
        n_freq_bands=cfg.n_freq_bands,
    ).to(device)

    mlp = SparamCommonResidualMLP(
        latent_dim=cfg.latent,
        n_freq=cfg.n_freq,
        common_curve=common_curve,
        hidden_mult=cfg.mlp_hidden_mult,
        dropout=cfg.mlp_dropout,
        residual_scale=cfg.residual_scale,
        zero_init_residual=cfg.zero_init_residual,
    ).to(device)

    ae_params = sum(p.numel() for p in ae.parameters())
    mlp_params = sum(p.numel() for p in mlp.parameters())

    print(f"  AE params           : {ae_params:,}")
    print(f"  Residual MLP params : {mlp_params:,}")
    print(f"  Total params        : {ae_params + mlp_params:,}")
    print(f"  max_bodies/body_len : {dataset.max_bodies} / {dataset.max_body_len}")
    print(f"  latent              : {cfg.latent}")
    print(f"  surrogate           : pred = common_curve + residual(z)")
    print(f"  matching            : vectorized Hungarian + presence (w={cfg.w_match_presence})")
    print(f"  body decoder        : causal self-attention")

    train_loader = DataLoader(
        SubsetDataset(dataset, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        SubsetDataset(dataset, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": list(ae.parameters()), "lr": cfg.lr_ae},
            {"params": list(mlp.parameters()), "lr": cfg.lr_mlp},
        ],
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
    )

    def lr_sched(ep):
        if ep < cfg.warmup:
            return (ep + 1) / max(1, cfg.warmup)

        return 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (ep - cfg.warmup)
                / max(1, cfg.epochs - cfg.warmup)
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    best_metric = float("inf")
    best_ae_state = None
    best_mlp_state = None

    hist = {
        "tr_total": [],
        "va_total": [],
        "tr_full": [],
        "va_full": [],
        "presence": [],
        "cmd": [],
        "prm": [],
    }

    log_every = max(1, cfg.epochs // 8)

    section("JOINT TRAINING")

    print(f"  epochs={cfg.epochs}, batch_size={cfg.batch_size}")
    print(f"  loss = bodyset_AE + {cfg.w_sparam} * sparam_full_interp + VICReg")

    print(
        f"\n  {'ep':>4s} | {'tr_tot':>8s} {'cmd':>6s} {'prm':>6s} {'pres':>6s} "
        f"{'tr_full':>8s} {'tr_sel':>8s} {'zstd':>6s} "
        f"{'va_tot':>8s} {'va_full':>8s} {'va_sel':>8s} | {'lr':>8s}"
    )

    print("  " + "-" * 110)

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(
            ae,
            mlp,
            train_loader,
            optimizer,
            device,
            cfg,
            interp_w,
            train_mode=True,
        )

        va = run_epoch(
            ae,
            mlp,
            val_loader,
            optimizer,
            device,
            cfg,
            interp_w,
            train_mode=False,
        )

        scheduler.step()

        hist["tr_total"].append(tr["total"])
        hist["va_total"].append(va["total"])
        hist["tr_full"].append(tr["rmse_db_full"])
        hist["va_full"].append(va["rmse_db_full"])
        hist["presence"].append(va["presence"])
        hist["cmd"].append(va["cmd"])
        hist["prm"].append(va["prm"])

        if va["rmse_db_full"] < best_metric:
            best_metric = va["rmse_db_full"]

            best_ae_state = {
                k: v.detach().clone().cpu()
                for k, v in ae.state_dict().items()
            }

            best_mlp_state = {
                k: v.detach().clone().cpu()
                for k, v in mlp.state_dict().items()
            }

        if ep == 1 or ep % log_every == 0 or ep == cfg.epochs:
            print(
                f"  {ep:4d} | "
                f"{tr['total']:8.4f} {tr['cmd']:6.3f} {tr['prm']:6.3f} {tr['presence']:6.3f} "
                f"{tr['rmse_db_full']:8.3f} {tr['rmse_db_sel']:8.3f} {tr['z_std']:6.3f} "
                f"{va['total']:8.4f} {va['rmse_db_full']:8.3f} {va['rmse_db_sel']:8.3f} | "
                f"{optimizer.param_groups[0]['lr']:8.2e}"
            )

    print(f"\n  Best full-grid val RMSE dB: {best_metric:.4f}")

    if best_ae_state is not None:
        ae.load_state_dict({k: v.to(device) for k, v in best_ae_state.items()})

    if best_mlp_state is not None:
        mlp.load_state_dict({k: v.to(device) for k, v in best_mlp_state.items()})

    section("EVALUATION")

    recon_diag = diagnose_reconstruction(
        ae,
        dataset,
        val_idx if val_idx else train_idx,
        device,
        max_samples=min(8, len(val_idx) or len(train_idx)),
    )

    eval_metrics = evaluate_sparam_predictions(
        ae,
        mlp,
        dataset,
        val_idx_per_type,
        type_names,
        device,
    )

    latent_diag = diagnose_latent(
        ae,
        dataset,
        val_idx if val_idx else train_idx,
        device,
    )

    print_eval_diagnostics(
        eval_metrics,
        recon_diag,
        latent_diag,
        type_names,
    )

    if cfg.show_figures:
        try:
            plot_training_curves(hist)

        except Exception as e:
            print(f"  ⚠ plot_training_curves failed: {type(e).__name__}: {e}")

        try:
            visualize_sparam_predictions(
                ae,
                mlp,
                dataset,
                val_idx,
                device,
                n_samples=min(3, len(val_idx)),
                seed=cfg.seed,
            )

        except Exception as e:
            print(f"  ⚠ visualize_sparam_predictions failed: {type(e).__name__}: {e}")

        try:
            print_reconstruction_examples(
                ae,
                dataset,
                val_idx if val_idx else train_idx,
                device,
                n_preview=cfg.n_preview,
            )

        except Exception as e:
            print(f"  ⚠ print_reconstruction_examples failed: {type(e).__name__}: {e}")

    result = {
        "best_val": float(best_metric),
        "eval_full": float(eval_metrics["overall_full_rmse"]),
        "eval_sel": float(eval_metrics["overall_selected_rmse"]),
        "z_std": float(latent_diag["z_std_mean"]),
        "z_norm": float(latent_diag["z_norm_mean"]),
        "dead": int(latent_diag["dead_dims"]),
        "latent_dim": int(latent_diag["latent_dim"]),
    }

    section("FINAL SUMMARY")

    print("  AE type      : Body-set encoder + body-set decoder (v2)")
    print("  Decoder loss : Vectorized Hungarian + presence-aware matching")
    print(f"  best_val     : {result['best_val']:.4f} dB")
    print(f"  eval_full    : {result['eval_full']:.4f} dB")
    print(f"  eval_sel     : {result['eval_sel']:.4f} dB")
    print(f"  z_std        : {result['z_std']:.4f}")
    print(f"  z_norm       : {result['z_norm']:.4f}")
    print(f"  dead dims    : {result['dead']} / {result['latent_dim']}")

    return ae, mlp, result


# ══════════════════════════════════════════════════════════════
# Inverse design
# ══════════════════════════════════════════════════════════════
def make_target_db_curve(
    freqs_full,
    channel_target_freqs,
    bandwidth_ghz,
    deep_db=-15.0,
    flat_db=0.0,
):
    freqs_full = np.asarray(freqs_full, dtype=np.float32)

    n_freq = len(freqs_full)
    n_ch = len(channel_target_freqs)

    target = np.full((n_freq, n_ch), flat_db, dtype=np.float32)
    masks = np.zeros((n_freq, n_ch), dtype=bool)

    half_bw = bandwidth_ghz / 2.0

    for c, f0 in enumerate(channel_target_freqs):
        m = np.abs(freqs_full - f0) <= half_bw

        target[m, c] = deep_db
        masks[:, c] = m

    return target, masks


@torch.no_grad()
def _latent_prior_stats(ae, dataset, device):
    z_all, _ = collect_latents(
        ae,
        dataset,
        list(range(len(dataset))),
        device,
        batch_size=32,
    )

    z_t = torch.tensor(z_all, dtype=torch.float32, device=device)

    return z_t, z_t.mean(dim=0), z_t.std(dim=0).clamp(min=1e-3)


def inverse_design_optimize(
    ae,
    mlp,
    dataset,
    channel_target_freqs=(2.0, 3.0, 4.0),
    bandwidth_ghz=0.1,
    device="cuda",
    n_starts=32,
    n_iters=1000,
    lr=5e-2,
    in_band_weight=10.0,
    out_band_weight=0.0,
    z_prior_weight=1e-3,
    z_prior_weight_end=1e-5,
    deep_db=-15.0,
    seed=7,
    verbose_every=100,
):
    ae.eval()
    mlp.eval()

    z_prior, z_mean, z_std = _latent_prior_stats(
        ae,
        dataset,
        device,
    )

    n_starts = min(int(n_starts), z_prior.size(0))

    rng = np.random.default_rng(seed)

    init_idx = rng.choice(
        z_prior.size(0),
        size=n_starts,
        replace=False,
    )

    z = z_prior[init_idx].clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([z], lr=lr)

    target_full, masks = make_target_db_curve(
        dataset.freqs_full,
        channel_target_freqs,
        bandwidth_ghz,
        deep_db=deep_db,
        flat_db=0.0,
    )

    target_t = torch.tensor(
        target_full,
        dtype=torch.float32,
        device=device,
    )

    masks_t = torch.tensor(
        masks,
        dtype=torch.bool,
        device=device,
    )

    interp_w = torch.tensor(
        dataset.interp_matrix,
        dtype=torch.float32,
        device=device,
    )

    in_count = masks_t.float().sum(dim=0).clamp(min=1.0).unsqueeze(0)
    out_count = (~masks_t).float().sum(dim=0).clamp(min=1.0).unsqueeze(0)

    best_loss = float("inf")
    best_z = None
    best_pred_full = None

    print("\n  multi-start latent optimization")
    print(f"    targets={channel_target_freqs}, bw={bandwidth_ghz} GHz, deep_db={deep_db}")
    print(f"    n_starts={n_starts}, n_iters={n_iters}, lr={lr}")

    for it in range(n_iters):
        progress = it / max(n_iters - 1, 1)

        cur_prior_w = z_prior_weight + (
            z_prior_weight_end - z_prior_weight
        ) * progress

        cur_lr = lr * (
            0.05
            + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        )

        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)

        pred_sel = mlp(z)
        pred_full = interpolate_selected_to_full_torch(pred_sel, interp_w)

        diff = pred_full - target_t.unsqueeze(0)

        in_band = (
            (F.relu(diff) ** 2)
            * masks_t.unsqueeze(0).float()
        ).sum(dim=1) / in_count

        out_band = (
            (diff ** 2)
            * (~masks_t).unsqueeze(0).float()
        ).sum(dim=1) / out_count

        match_per = (
            in_band_weight * in_band
            + out_band_weight * out_band
        ).mean(dim=-1)

        z_reg = (((z - z_mean) / z_std) ** 2).mean(dim=-1)

        total_per = match_per + cur_prior_w * z_reg

        loss = total_per.sum()

        loss.backward()
        optimizer.step()

        # step 후 다시 평가해서 best_z와 best_pred를 같은 시점으로 맞춤
        with torch.no_grad():
            pred_sel_new = mlp(z)
            pred_full_new = interpolate_selected_to_full_torch(
                pred_sel_new,
                interp_w,
            )

            diff_new = pred_full_new - target_t.unsqueeze(0)

            in_band_new = (
                (F.relu(diff_new) ** 2)
                * masks_t.unsqueeze(0).float()
            ).sum(dim=1) / in_count

            out_band_new = (
                (diff_new ** 2)
                * (~masks_t).unsqueeze(0).float()
            ).sum(dim=1) / out_count

            match_new = (
                in_band_weight * in_band_new
                + out_band_weight * out_band_new
            ).mean(dim=-1)

            bi = int(torch.argmin(match_new).item())
            cur = float(match_new[bi].item())

            if cur < best_loss:
                best_loss = cur
                best_z = z[bi:bi + 1].detach().clone()
                best_pred_full = pred_full_new[bi].detach().cpu().numpy()

        if it == 0 or (it + 1) % verbose_every == 0 or it == n_iters - 1:
            print(
                f"    iter {it + 1:4d}/{n_iters} | "
                f"best={best_loss:.4f} | "
                f"avg={float(match_per.mean().item()):.4f} | "
                f"lr={cur_lr:.2e}"
            )

    return {
        "best_z": best_z,
        "best_loss": best_loss,
        "best_pred_full": best_pred_full,
        "target_full": target_full,
        "masks": masks,
        "freqs_full": dataset.freqs_full,
        "channel_target_freqs": list(channel_target_freqs),
        "bandwidth_ghz": float(bandwidth_ghz),
        "deep_db": float(deep_db),
    }


def visualize_inverse_design_curve(result):
    freqs = result["freqs_full"]
    pred = result["best_pred_full"]

    target_freqs = result["channel_target_freqs"]
    bw = result["bandwidth_ghz"]
    deep_db = result["deep_db"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        facecolor="white",
    )

    fig.suptitle(
        f"Inverse design surrogate prediction, spec ≤ {deep_db:.0f} dB",
        fontsize=11,
    )

    for c, lbl in enumerate(RETURN_LABELS):
        ax = axes[c]

        f0 = target_freqs[c]
        f_lo, f_hi = f0 - bw / 2.0, f0 + bw / 2.0

        ax.plot(
            freqs,
            pred[:, c],
            lw=1.6,
            label="surrogate pred",
        )

        ax.hlines(
            deep_db,
            f_lo,
            f_hi,
            linestyles="-",
            lw=1.2,
            label="spec",
        )

        ax.axvline(f_lo, ls="--", lw=0.8)
        ax.axvline(f_hi, ls="--", lw=0.8)

        ax.axhline(
            -10.0,
            ls=":",
            lw=0.8,
            label="-10 dB ref",
        )

        band = (freqs >= f_lo) & (freqs <= f_hi)

        if band.any():
            worst = float(pred[band, c].max())
            margin = deep_db - worst
            ok = "OK" if margin >= 0 else "NG"

            ax.set_title(
                f"{lbl} f={f0:.2f} GHz worst={worst:.1f} margin={margin:+.1f} {ok}",
                fontsize=9,
            )

        else:
            ax.set_title(f"{lbl} f={f0:.2f} GHz", fontsize=9)

        ax.set_xlabel("frequency [GHz]")
        ax.set_ylabel("|S| [dB]")

        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plt.tight_layout()


def run_inverse_design_pipeline(ae, mlp, dataset, device):
    section("INVERSE DESIGN — latent search + body-set decode")

    result = inverse_design_optimize(
        ae=ae,
        mlp=mlp,
        dataset=dataset,
        channel_target_freqs=(2.0, 3.0, 4.0),
        bandwidth_ghz=0.1,
        device=device,
        n_starts=32,
        n_iters=1000,
        lr=5e-2,
        in_band_weight=10.0,
        out_band_weight=0.0,
        z_prior_weight=1e-3,
        z_prior_weight_end=1e-5,
        deep_db=-15.0,
        seed=7,
    )

    print("\n  Optimization result")
    print(f"    best match loss: {result['best_loss']:.4f}")

    if result["best_z"] is not None:
        with torch.no_grad():
            gen_seq = ae.generate(
                result["best_z"],
                threshold=0.5,
                force_topk=None,
            )[0]

        print("\n  Decoded inverse-design token sequence:")
        print(summarize_tokens(gen_seq, max_rows=160))

        result["decoded_tokens"] = gen_seq

    try:
        visualize_inverse_design_curve(result)

    except Exception as e:
        print(f"  ⚠ inverse curve figure failed: {type(e).__name__}: {e}")

    return result


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    t0 = time.time()

    # ---------------------------------------------------------
    # 실행 설정
    # ---------------------------------------------------------
    PRESET = "tiny"          # tiny / small / full / custom
    USE_TYPES = [1, 2, 3]

    RAW_N_FREQ = 401
    SELECTED_N_FREQ = 81

    # PRESET="custom"일 때만 그대로 반영됨.
    # tiny/small/full은 preset의 n_samples_each가 우선.
    SAMPLES_PER_TYPE = (800, 800, 800)

    SHOW_FIGURES = True
    RUN_INVERSE_DESIGN = True
    N_PREVIEW = 3

    USE_VICREG = True
    VICREG_W_VAR = 0.5
    VICREG_W_COV = 0.05
    VICREG_VAR_TARGET = 1.0

    _TYPE_DATA = {
        1: (
            os.path.join(script_dir, "hfss_results", "step_test"),
            os.path.join(script_dir, "hfss_results", "sparam", "[12].*"),
        ),
        2: (
            os.path.join(script_dir, "hfss_results", "step_test1"),
            os.path.join(script_dir, "hfss_results", "sparam", "3.*"),
        ),
        3: (
            os.path.join(script_dir, "hfss_results", "step_test2"),
            os.path.join(script_dir, "hfss_results", "sparam", "4.*"),
        ),
    }

    if not USE_TYPES:
        raise ValueError("USE_TYPES is empty.")

    for t in USE_TYPES:
        if t not in _TYPE_DATA:
            raise ValueError(f"Undefined type number: {t}")

    npy_dirs = tuple(_TYPE_DATA[t][0] for t in USE_TYPES)
    sparam_globs = tuple(_TYPE_DATA[t][1] for t in USE_TYPES)
    type_names = tuple(f"type{t}" for t in USE_TYPES)

    cfg = CFG(
        preset=PRESET,
        seed=7,
        run_name="bodyset_deepcad_ae_v2",

        npy_dirs=npy_dirs,
        sparam_globs=sparam_globs,
        type_names=type_names,
        n_samples_per_type=SAMPLES_PER_TYPE,
        sample_mode="random",

        raw_n_freq=RAW_N_FREQ,
        n_freq=SELECTED_N_FREQ,
        freq_select_mode="linspace",
        freq_start=1.0,
        freq_end=5.0,

        max_bodies_cap=16,
        max_body_len_cap=128,

        lr_ae=3e-4,
        lr_mlp=1e-3,
        weight_decay=1e-3,
        grad_clip=1.0,
        val_ratio=0.15,

        w_cmd=1.0,
        w_prm=1.0,
        w_presence=1.0,
        w_match_presence=0.5,
        w_sparam=5.0,
        db_loss_scale=20.0,

        use_vicreg=USE_VICREG,
        w_var=VICREG_W_VAR,
        w_cov=VICREG_W_COV,
        vicreg_var_target=VICREG_VAR_TARGET,

        mlp_hidden_mult=2.0,
        mlp_dropout=0.3,
        residual_scale=1.0,
        zero_init_residual=True,

        n_preview=N_PREVIEW,
        show_figures=SHOW_FIGURES,
    )

    try:
        apply_preset(cfg)

        # preset 후 고정값 재적용
        cfg.use_vicreg = USE_VICREG
        cfg.w_var = VICREG_W_VAR
        cfg.w_cov = VICREG_W_COV
        cfg.vicreg_var_target = VICREG_VAR_TARGET
        cfg.n_preview = N_PREVIEW
        cfg.show_figures = SHOW_FIGURES

        set_seed(cfg.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        section("Body-Set DeepCAD AE v2 + Return-3 Common-Curve Residual Surrogate")

        print(f"  Python : {_sys.version.split()[0]}")
        print(f"  Torch  : {torch.__version__}")
        print(
            f"  Device : {device}"
            + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        )
        print(f"  Backend: {_MPL_BACKEND}")
        print(f"  Preset : {cfg.preset}")
        print(f"  Types  : {type_names}")
        print(f"  AE     : body-set encoder + body-set decoder (v2)")
        print(f"  Loss   : Vectorized Hungarian + presence-aware matching + S-param surrogate")
        print(f"  Decoder: causal self-attention within body")

        section("LOAD DATA")

        dataset, npy_files, type_ids, type_names, sparam_db = load_multitype_data(
            cfg,
            script_dir,
        )

        section("SPLIT")

        train_idx, val_idx, val_idx_per_type = make_stratified_split(
            cfg,
            dataset,
            type_ids,
            type_names,
        )

        common_curve = build_common_curve_and_print_baseline(
            dataset,
            train_idx,
            val_idx_per_type,
            type_names,
        )

        ae, mlp, result = train_bodyset_model(
            cfg=cfg,
            dataset=dataset,
            type_names=type_names,
            train_idx=train_idx,
            val_idx=val_idx,
            val_idx_per_type=val_idx_per_type,
            common_curve=common_curve,
            device=device,
        )

        if RUN_INVERSE_DESIGN:
            try:
                inv_result = run_inverse_design_pipeline(
                    ae,
                    mlp,
                    dataset,
                    device,
                )

            except Exception as e:
                import traceback

                print(f"\n[INVERSE DESIGN] failed: {type(e).__name__}: {e}")
                traceback.print_exc()

        section("DONE")

        print(f"  Final eval_full RMSE : {result['eval_full']:.4f} dB")
        print(f"  figures: {len(plt.get_fignums())} (backend={_MPL_BACKEND})")

        elapsed = time.time() - t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = elapsed % 60

        print(f"\n  경과 시간 (figure 표시 전): {h}h {m}m {s:.1f}s")

        import signal as _signal

        try:
            _signal.signal(_signal.SIGINT, _signal.SIG_DFL)

        except Exception:
            pass

        if cfg.show_figures and len(plt.get_fignums()) > 0:
            try:
                plt.show()

            except KeyboardInterrupt:
                print("\n[Ctrl+C] closing figures and exiting")

            except Exception as e:
                print(f"  ⚠ plt.show() failed: {type(e).__name__}: {e}")

            finally:
                try:
                    plt.close("all")

                except Exception:
                    pass

    except Exception as e:
        import traceback

        print(f"\n[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()

        elapsed = time.time() - t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = elapsed % 60

        print(f"\n  경과 시간 (실패 시점): {h}h {m}m {s:.1f}s")
