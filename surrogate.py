#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Surrogate-only training: encoder + S-param MLP (standalone, no decoder)."""

# (originally adapted from AE_main.py)


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

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
ROLE = 6           # ★ NEW: chunk 시작을 알리는 role 헤더 토큰. param[0] = role_id

N_BIT = 10
N_QUANT = 2 ** N_BIT
QMAX = N_QUANT - 1

N_CMD = 7          # ★ LINE/ARC/CIRCLE/SOL/EXT/EOS/ROLE
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
    ROLE: 1,        # ★ NEW: role_id 한 슬롯만 사용
}

CMD_NAME = {
    LINE: "LINE",
    ARC: "ARC",
    CIRCLE: "CIRCLE",
    SOL: "SOL",
    EXT: "EXT",
    EOS: "EOS",
    ROLE: "ROLE",   # ★ NEW
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
    run_name: str = "compact_vicreg"

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

    # ★ ROLE 토큰화 + 같은 role coplanar chunk 자동 병합
    use_json_names: bool = True              # JSON metadata names → ROLE 토큰 삽입
    merge_same_role_chunks: bool = True      # 같은 role + 같은 plane + 같은 extrude → 1 chunk(multi-loop)
    validate_json_alignment: bool = True     # token 파일명과 JSON metadata.source_file 일치 여부 점검

    # ★ Checkpoint save/load (학습 후 가중치 저장, 나중에 인버스만 돌릴 때 로드)
    ckpt_save_path: str = "ckpt/AE_main_last.pt"  # 빈 문자열 = 저장 안 함
    ckpt_load_path: str = ""                 # 비어 있으면 학습; 경로 주면 로드 후 학습 skip
    save_ckpt_after_train: bool = True

    # loss weights
    w_cmd: float = 1.0
    w_prm: float = 1.0
    w_aux: float = 2.0
    w_sparam: float = 5.0

    # dB loss scale
    db_loss_scale: float = 20.0

    # compact VICReg for latent stability
    use_vicreg: bool = True
    w_var: float = 0.2
    w_cov: float = 0.02
    vicreg_var_target: float = 0.7

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


def sample_id_from_path(path):
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    stem = re.sub(r"(_tokens(?:_float)?|_deepcad)$", "", stem)
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None


def causal_mask(sz, device):
    return torch.triu(
        torch.ones((sz, sz), dtype=torch.bool, device=device),
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


_CMD_PATTERN_CHARS = {
    LINE: "L",
    ARC: "A",
    CIRCLE: "C",
    SOL: "S",
    EXT: "E",
}




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


_LOG_VERBOSITY = "simple"


def set_log_verbosity(v):
    global _LOG_VERBOSITY
    _LOG_VERBOSITY = str(v or "simple").lower()


def log_is_compact():
    return _LOG_VERBOSITY in ("compact", "quiet", "minimal")


def log_is_quiet():
    return _LOG_VERBOSITY in ("quiet", "minimal")


def log_detail(*args, **kwargs):
    if not log_is_compact():
        print(*args, **kwargs)


def section(title, ch="═"):
    if log_is_compact():
        print(f"\n[{title}]")
        return
    print(f"\n{_hr(ch)}\n  {title}\n{_hr(ch)}")


def subsection(title, ch="─"):
    if log_is_compact():
        print(f"\n- {title}")
        return
    print(f"\n{_hr(ch)}\n  {title}\n{_hr(ch)}")


# ══════════════════════════════════════════════════════════════
# ★ Inverse design 결과 파일 저장 (txt + csv + png)
# ══════════════════════════════════════════════════════════════

def _v3(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=float)


def extract_dequant_meta(json_data: dict) -> dict:
    seq = json_data["sequence"]
    meta = json_data.get("metadata", {})

    pts = []
    for op in seq:
        if op["type"] == "Sketch":
            pts.append(_v3(op["plane"]["origin"]).tolist())
    for g in meta.get("solids", []):
        pts.append(g.get("aabb_min", [0, 0, 0]))
        pts.append(g.get("aabb_max", [0, 0, 0]))
    arr = np.array(pts) if pts else np.array([[0, 0, 0], [1, 1, 1]])
    g_min3, g_max3 = arr.min(axis=0), arr.max(axis=0)
    g_max_dim = float((g_max3 - g_min3).max())
    if g_max_dim < 1e-8:
        g_max_dim = 1.0

    sketch_metas = []
    all_extents = []

    for op in seq:
        if op["type"] == "Sketch":
            profile = op.get("profile", {})
            xs, ys = [], []
            for loop in profile.get("children", []):
                for e in loop.get("children", []):
                    for key in ("start_point", "end_point", "mid_point", "center_point"):
                        if key in e:
                            xs.append(e[key]["x"])
                            ys.append(e[key]["y"])
            if xs:
                xy_min = np.array([min(xs), min(ys)])
                xy_max = np.array([max(xs), max(ys)])
            else:
                xy_min = np.array([0.0, 0.0])
                xy_max = np.array([1.0, 1.0])
            sk_scale = max(float((xy_max - xy_min).max()), 1e-8)

            plane = op["plane"]
            sketch_metas.append({
                "xy_min": xy_min, "xy_max": xy_max, "sk_scale": sk_scale,
                "origin": _v3(plane["origin"]),
                "x_axis": _v3(plane["x_axis"]),
                "y_axis": _v3(plane["y_axis"]),
                "z_axis": _v3(plane["z_axis"]),
            })
        elif op["type"] == "Extrude":
            all_extents.append(op.get("extent_one", {}).get("distance", 0.0))
            all_extents.append(op.get("extent_two", {}).get("distance", 0.0))

    max_ext = max(all_extents) if all_extents else 1.0
    if max_ext < 1e-8:
        max_ext = 1.0

    return {
        "g_min3": g_min3, "g_max3": g_max3, "g_max_dim": g_max_dim,
        "max_ext": max_ext, "sketches": sketch_metas,
    }


# ══════════════════════════════════════════════════════════════
# ★ JSON merging (same-role + coplanar + same extrude → multi-loop) + retokenizer
# ══════════════════════════════════════════════════════════════
_BOOL_MAP_TOK  = {"NewBody": 0, "Join": 1, "Cut": 2, "Intersect": 3}
_ETYPE_MAP_TOK = {"OneSide": 0, "Symmetric": 1, "TwoSide": 2}


def _q01_tok(v):
    return int(round(float(np.clip(v, 0.0, 1.0)) * QMAX))


def _norm_tok(v, lo, hi):
    if hi - lo < 1e-8:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def json_to_tokens(json_data):
    """JSON dict → (N, 17) int32 token array.  extract_dequant_meta 활용."""
    meta = extract_dequant_meta(json_data)
    g_min3, g_max3, g_max_dim = meta["g_min3"], meta["g_max3"], meta["g_max_dim"]
    max_ext = meta["max_ext"]
    sk_metas = meta["sketches"]

    def row(cmd, *args):
        r = [PAD_V] * (1 + N_ARGS)
        r[0] = cmd
        for i, v in enumerate(args):
            r[1 + i] = int(v)
        return r

    rows = []
    seq = json_data["sequence"]
    sk_idx = 0
    for op in seq:
        if op["type"] == "Sketch":
            sm = sk_metas[sk_idx]
            xy_min, xy_max, sk_scale = sm["xy_min"], sm["xy_max"], sm["sk_scale"]

            def q2d(x, y):
                return (_q01_tok(_norm_tok(x, xy_min[0], xy_max[0])),
                        _q01_tok(_norm_tok(y, xy_min[1], xy_max[1])))

            def qr(r):
                return _q01_tok(_norm_tok(r, 0.0, sk_scale))

            for loop in op.get("profile", {}).get("children", []):
                rows.append(row(SOL))
                for e in loop.get("children", []):
                    et = e["type"]
                    if et == "Line":
                        qx, qy = q2d(e["end_point"]["x"], e["end_point"]["y"])
                        rows.append(row(LINE, qx, qy))
                    elif et == "Arc":
                        qmx, qmy = q2d(e["mid_point"]["x"], e["mid_point"]["y"])
                        qex, qey = q2d(e["end_point"]["x"], e["end_point"]["y"])
                        rows.append(row(ARC, qmx, qmy, qex, qey))
                    elif et == "Circle":
                        qcx, qcy = q2d(e["center_point"]["x"], e["center_point"]["y"])
                        rows.append(row(CIRCLE, qcx, qcy, qr(e["radius"])))
            sk_idx += 1
        elif op["type"] == "Extrude":
            sm = sk_metas[sk_idx - 1] if sk_idx > 0 else sk_metas[0]
            origin = sm["origin"]
            q_sk_scale = _q01_tok(_norm_tok(sm["sk_scale"] / g_max_dim, 0.0, 1.0))
            q_ox = _q01_tok(_norm_tok(origin[0], g_min3[0], g_max3[0]))
            q_oy = _q01_tok(_norm_tok(origin[1], g_min3[1], g_max3[1]))
            q_oz = _q01_tok(_norm_tok(origin[2], g_min3[2], g_max3[2]))
            e1 = op.get("extent_one", {}).get("distance", 0.0)
            e2 = op.get("extent_two", {}).get("distance", 0.0)
            q_e1 = _q01_tok(_norm_tok(e1, 0.0, max_ext))
            q_e2 = _q01_tok(_norm_tok(e2, 0.0, max_ext))
            q_bool = _BOOL_MAP_TOK.get(op.get("operation", "NewBody"), 0)
            q_etype = _ETYPE_MAP_TOK.get(op.get("extent_type", "OneSide"), 0)
            rows.append(row(EXT, q_sk_scale, q_ox, q_oy, q_oz,
                            q_e1, q_e2, q_bool, q_etype))
    rows.append(row(EOS))
    return np.asarray(rows, dtype=np.int32)




def merge_coplanar_same_role_sketches(json_data, names,
                                       plane_tol=1e-3, extent_tol=1e-3,
                                       verbose=False):
    """JSON sequence 에서 연속된 Sketch+Extrude pair 중
    같은 role + coplanar + 같은 extrude 인 것들을 1개의 Sketch (multi-loop) + 1개의 Extrude 로 병합."""
    seq = json_data["sequence"]
    names = list(names) if names is not None else []

    def is_pair(idx):
        return (
            idx + 1 < len(seq)
            and seq[idx].get("type") == "Sketch"
            and seq[idx + 1].get("type") == "Extrude"
        )

    def planes_match(p1, p2):
        for k in ("origin", "x_axis", "y_axis", "z_axis"):
            v1 = _v3(p1[k]); v2 = _v3(p2[k])
            if np.linalg.norm(v1 - v2) > plane_tol:
                return False
        return True

    def extrudes_match(e1, e2):
        d1 = float(e1.get("extent_one", {}).get("distance", 0.0))
        d2 = float(e2.get("extent_one", {}).get("distance", 0.0))
        if abs(d1 - d2) > extent_tol:
            return False
        s1 = float(e1.get("extent_two", {}).get("distance", 0.0))
        s2 = float(e2.get("extent_two", {}).get("distance", 0.0))
        if abs(s1 - s2) > extent_tol:
            return False
        if e1.get("extent_type", "OneSide") != e2.get("extent_type", "OneSide"):
            return False
        if e1.get("operation", "NewBody") != e2.get("operation", "NewBody"):
            return False
        return True

    new_seq = []
    new_names = []
    n_pairs = 0
    n_merged = 0
    pair_idx = 0
    i = 0
    while i < len(seq):
        if not is_pair(i):
            new_seq.append(seq[i])
            i += 1
            continue

        sk, ex = seq[i], seq[i + 1]
        nm = names[pair_idx] if pair_idx < len(names) else None
        pair_idx += 1
        n_pairs += 1

        group = [(sk, ex)]
        i += 2

        while is_pair(i):
            sk2, ex2 = seq[i], seq[i + 1]
            nm2 = names[pair_idx] if pair_idx < len(names) else None
            if (
                nm is not None and nm == nm2
                and planes_match(sk["plane"], sk2["plane"])
                and extrudes_match(ex, ex2)
            ):
                group.append((sk2, ex2))
                pair_idx += 1
                n_pairs += 1
                i += 2
            else:
                break

        if len(group) > 1:
            fixed_loops = []
            for sk_g, _ex_g in group:
                for lp in sk_g.get("profile", {}).get("children", []):
                    fixed_loops.append(copy.deepcopy(lp))

            new_sk = copy.deepcopy(sk)
            new_sk["profile"] = copy.deepcopy(sk.get("profile", {}))
            new_sk["profile"]["children"] = fixed_loops
            new_seq.append(new_sk)
            new_seq.append(copy.deepcopy(ex))
            n_merged += len(group) - 1
            if verbose:
                print(f"    merged role={nm!r}: {len(group)} chunks → 1 chunk "
                      f"({len(fixed_loops)} loops total)")
        else:
            new_seq.append(sk)
            new_seq.append(ex)
        new_names.append(nm)

    if names and len(names) != n_pairs and verbose:
        print(f"    [warn] role-name count mismatch: names={len(names)} pairs={n_pairs}")

    new_json = dict(json_data)
    new_json["sequence"] = new_seq

    return new_json, new_names, n_merged


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
        remainder = sd.shape[0] % raw_n_freq
        if verbose and remainder:
            print(
                f"      warn: {remainder} trailing row(s) ignored "
                f"(rows={sd.shape[0]}, raw_n_freq={raw_n_freq})"
            )
        if verbose and n_block == 0:
            print(f"      warn: no complete {raw_n_freq}-row S-param block")
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


def check_token_json_alignment(token_path, json_data):
    token_id = sample_id_from_path(token_path)
    source = json_data.get("metadata", {}).get("source_file", "")
    source_id = sample_id_from_path(source)
    if token_id is None or source_id is None:
        return None
    if token_id != source_id:
        return (
            f"token id {token_id} != JSON metadata.source_file id {source_id} "
            f"({os.path.basename(token_path)} vs {source})"
        )
    return None


# ══════════════════════════════════════════════════════════════
# Dataset

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
        use_json_names=True,
        merge_same_role_chunks=True,
        validate_json_alignment=True,
    ):
        self.npy_files = list(npy_files)
        self.max_len = max_len
        self.use_json_names = bool(use_json_names)
        self.merge_same_role_chunks = bool(merge_same_role_chunks)
        self.validate_json_alignment = bool(validate_json_alignment)
        self.merged_json_cache = [None] * len(self.npy_files)

        self.json_files = []
        for f in self.npy_files:
            jf = f.replace("_tokens.npy", "_deepcad.json")
            self.json_files.append(jf if os.path.exists(jf) else None)




        # ★ JSON metadata.solids[i].name → global role_id 매핑 구축
        per_sample_names = []
        alignment_warnings = []
        if self.use_json_names:
            for tok_path, jf in zip(self.npy_files, self.json_files):
                if jf is None or not os.path.exists(jf):
                    per_sample_names.append(None)
                    continue
                try:
                    import json as _json
                    with open(jf, "r", encoding="utf-8") as f:
                        jd = _json.load(f)
                    if self.validate_json_alignment:
                        msg = check_token_json_alignment(tok_path, jd)
                        if msg is not None:
                            alignment_warnings.append(msg)
                    solids_meta = jd.get("metadata", {}).get("solids", [])
                    names = [str(s.get("name", "")) for s in solids_meta]
                    per_sample_names.append(names if any(names) else None)
                except Exception:
                    per_sample_names.append(None)
        else:
            per_sample_names = [None] * len(self.npy_files)

        all_names_set = set()
        for names in per_sample_names:
            if names is None:
                continue
            for n in names:
                if n:
                    all_names_set.add(n)
        self.role_name_to_id = {n: i for i, n in enumerate(sorted(all_names_set))}
        self.role_id_to_name = {i: n for n, i in self.role_name_to_id.items()}
        if len(self.role_name_to_id) > N_QUANT:
            raise ValueError(
                f"too many ROLE ids: {len(self.role_name_to_id)} > param vocab {N_QUANT}"
            )
        n_with_names = sum(1 for n in per_sample_names if n is not None)
        if self.role_name_to_id:
            if log_is_compact():
                print(
                    f"  roles: {len(self.role_name_to_id)} "
                    f"from {n_with_names}/{len(self.npy_files)} JSON sample(s)"
                )
            else:
                print(f"  [roletoken] {len(self.role_name_to_id)} unique role(s) "
                      f"from {n_with_names}/{len(self.npy_files)} samples with JSON names")
                for n, i in sorted(self.role_name_to_id.items(), key=lambda kv: kv[1]):
                    print(f"    role_id={i:>2d}  {n!r}")
        else:
            reason = "disabled by cfg.use_json_names=False" if not self.use_json_names else "no JSON names"
            print(f"  [roletoken] {reason} — ROLE tokens not inserted (base behavior)")
        if alignment_warnings:
            print(f"  [data-check] JSON source_file mismatch: {len(alignment_warnings)} sample(s)")
            if not log_is_compact():
                for msg in alignment_warnings[:5]:
                    print(f"    {msg}")
                if len(alignment_warnings) > 5:
                    print(f"    ... {len(alignment_warnings) - 5} more")

        self.raw = []
        self.padded = []
        n_inserted_total = 0
        n_merge_total = 0
        n_samples_merged = 0
        n_truncated = 0
        max_seen_len = 0

        for i, (f, names) in enumerate(zip(self.npy_files, per_sample_names)):
            jf = self.json_files[i]
            t = None
            updated_names = names

            # ★ JSON 레벨에서 same-role + coplanar + same-extrude chunk 자동 병합
            if (self.merge_same_role_chunks and names is not None
                    and jf is not None and os.path.exists(jf)):
                try:
                    import json as _json
                    with open(jf, "r", encoding="utf-8") as fh:
                        jd_orig = _json.load(fh)
                    jd_merged, new_names, n_merged = merge_coplanar_same_role_sketches(
                        jd_orig, names, verbose=(i < 3),
                    )
                    if n_merged > 0:
                        t = json_to_tokens(jd_merged)
                        updated_names = new_names
                        self.merged_json_cache[i] = jd_merged
                        n_merge_total += n_merged
                        n_samples_merged += 1
                        if i < 3 and not log_is_compact():
                            print(f"    sample {i}: {len(names)} → {len(new_names)} chunks "
                                  f"(merged {n_merged})")
                except Exception as e:
                    print(f"    [warn] sample {i}: merge failed: {e}")
                    t = None

            if t is None:
                t = np.load(f).astype(np.int32)

            t, n_inserted = self._insert_role_tokens(t, updated_names)
            n_inserted_total += n_inserted
            max_seen_len = max(max_seen_len, int(t.shape[0]))
            if t.shape[0] > max_len:
                n_truncated += 1
            t = ensure_eos_when_truncated(t, max_len)
            self.raw.append(t)
            self.padded.append(self._pad(t))

        if self.role_name_to_id:
            if log_is_compact():
                print(f"  role tokens: {n_inserted_total} inserted")
            else:
                print(f"  [roletoken] inserted {n_inserted_total} ROLE tokens "
                      f"across {len(self.npy_files)} samples")
        if self.merge_same_role_chunks:
            if log_is_compact():
                print(
                    f"  chunk merge: {n_merge_total} merged "
                    f"({n_samples_merged}/{len(self.npy_files)} sample(s))"
                )
            else:
                print(f"  [roletoken] merged {n_merge_total} chunks across "
                      f"{n_samples_merged}/{len(self.npy_files)} samples "
                      f"(same role + coplanar + same extrude → multi-loop)")
        if n_truncated:
            print(f"  [warn] {n_truncated}/{len(self.npy_files)} token sequence(s) truncated "
                  f"to max_len={max_len} (max observed after ROLE={max_seen_len})")




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

        if log_is_compact():
            print(
                f"  dataset: {N} samples | {dist} | "
                f"tokens cmd=[0..{max_cmd}], prm=[0..{max_prm}] | "
                f"S-param sel={self.sparam_db.shape}, full={self.sparam_db_full.shape}"
            )
        else:
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

    def _insert_role_tokens(self, tokens, names):
        if not self.role_name_to_id or not names:
            return tokens, 0
        out_rows = []
        chunk_idx = -1
        expecting_first_sol = True
        n_inserted = 0
        for i in range(tokens.shape[0]):
            c = int(tokens[i, 0])
            if c == EOS or c < 0:
                out_rows.append(tokens[i])
                continue
            if c == SOL:
                if expecting_first_sol:
                    chunk_idx += 1
                    if 0 <= chunk_idx < len(names):
                        nm = names[chunk_idx]
                        rid = self.role_name_to_id.get(nm)
                        if rid is not None:
                            role_row = np.full(17, PAD_V, dtype=tokens.dtype)
                            role_row[0] = ROLE
                            role_row[1] = int(rid)
                            out_rows.append(role_row)
                            n_inserted += 1
                    expecting_first_sol = False
                out_rows.append(tokens[i])
            elif c == EXT:
                out_rows.append(tokens[i])
                expecting_first_sol = True
            else:
                out_rows.append(tokens[i])
        return np.stack(out_rows, axis=0).astype(tokens.dtype), n_inserted

    def _pad(self, t):
        L = t.shape[0]
        if L >= self.max_len:
            return t[:self.max_len].astype(np.float32)
        pad = np.full((self.max_len - L, 17), PAD_V, dtype=np.float32)
        return np.concatenate([t, pad], axis=0).astype(np.float32)

    def load_json(self, idx):
        if 0 <= idx < len(self.merged_json_cache):
            cached = self.merged_json_cache[idx]
            if cached is not None:
                return cached
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


# ══════════════════════════════════════════════════════════════
# AE model

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



# ══════════════════════════════════════════════════════════════
# Epoch

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

        if not log_is_compact():
            print(f"  [{tname}]")
            print(f"    common selected RMSE : {sel_rmse:.3f} dB")
            print(f"    common full RMSE     : {full_rmse:.3f} dB")

        total_sel_sq += float(np.sum((pred_sel - true_sel) ** 2))
        total_sel_n += int(np.prod(true_sel.shape))
        total_full_sq += float(np.sum((pred_full - true_full) ** 2))
        total_full_n += int(np.prod(true_full.shape))

    if log_is_compact():
        sel_msg = math.sqrt(total_sel_sq / total_sel_n) if total_sel_n > 0 else float("nan")
        full_msg = math.sqrt(total_full_sq / total_full_n) if total_full_n > 0 else float("nan")
        print(f"  common baseline RMSE: selected={sel_msg:.3f} dB, full={full_msg:.3f} dB")
    else:
        if total_sel_n > 0:
            print(f"\n  TOTAL common selected RMSE : {math.sqrt(total_sel_sq / total_sel_n):.3f} dB")
        if total_full_n > 0:
            print(f"  TOTAL common full RMSE     : {math.sqrt(total_full_sq / total_full_n):.3f} dB")

    return common_sel



@torch.no_grad()
def evaluate_sparam_predictions(ae, mlp, dataset, val_idx_per_type, type_names, cfg, device):
    ae.eval(); mlp.eval()

    freqs_sel = dataset.freqs
    freqs_full = dataset.freqs_full

    total_sel_sq = 0.0; total_sel_n = 0
    total_full_sq = 0.0; total_full_n = 0
    total_res_abs = 0.0; total_res_count = 0
    per_type = {}

    for ti, tname in enumerate(type_names):
        idxs = val_idx_per_type.get(ti, [])
        if not idxs:
            continue

        all_pred_sel = []
        all_true_sel = []
        all_true_full = []
        all_residual = []

        for idx in idxs:
            tok, true_sel_db, true_full_db, _tid = dataset[idx]
            tok = tok.unsqueeze(0).to(device)

            z = ae.encode(tok)
            pred_sel_db, residual = mlp(z, return_parts=True)

            all_pred_sel.append(pred_sel_db.cpu().numpy()[0])
            all_residual.append(residual.cpu().numpy()[0])
            all_true_sel.append(true_sel_db.numpy())
            all_true_full.append(true_full_db.numpy())

        all_pred_sel = np.asarray(all_pred_sel, dtype=np.float32)
        all_true_sel = np.asarray(all_true_sel, dtype=np.float32)
        all_true_full = np.asarray(all_true_full, dtype=np.float32)
        all_residual = np.asarray(all_residual, dtype=np.float32)

        all_pred_full = interpolate_selected_to_full_np(
            all_pred_sel, freqs_sel, freqs_full,
        )

        ch_sel = []; ch_full = []
        for c, lbl in enumerate(RETURN_LABELS):
            rmse_sel = float(np.sqrt(np.mean(
                (all_pred_sel[:, :, c] - all_true_sel[:, :, c]) ** 2
            )))
            rmse_full = float(np.sqrt(np.mean(
                (all_pred_full[:, :, c] - all_true_full[:, :, c]) ** 2
            )))
            ch_sel.append(rmse_sel)
            ch_full.append(rmse_full)

        avg_sel = float(np.sqrt(np.mean((all_pred_sel - all_true_sel) ** 2)))
        avg_full = float(np.sqrt(np.mean((all_pred_full - all_true_full) ** 2)))
        res_abs = float(np.mean(np.abs(all_residual)))

        per_type[tname] = {
            "n": len(idxs),
            "ch_sel": ch_sel, "ch_full": ch_full,
            "avg_sel": avg_sel, "avg_full": avg_full,
            "res_abs": res_abs,
        }

        total_sel_sq += float(np.sum((all_pred_sel - all_true_sel) ** 2))
        total_sel_n += int(np.prod(all_true_sel.shape))
        total_full_sq += float(np.sum((all_pred_full - all_true_full) ** 2))
        total_full_n += int(np.prod(all_true_full.shape))
        total_res_abs += float(np.sum(np.abs(all_residual)))
        total_res_count += int(np.prod(all_residual.shape))

    overall_sel = math.sqrt(total_sel_sq / max(total_sel_n, 1))
    overall_full = math.sqrt(total_full_sq / max(total_full_n, 1))
    overall_res = total_res_abs / max(total_res_count, 1)

    return {
        "per_type": per_type,
        "overall_selected_rmse": overall_sel,
        "overall_full_rmse": overall_full,
        "overall_residual_abs": overall_res,
    }



@torch.no_grad()
def diagnose_latent_simple(ae, dataset, indices, device):
    ae.eval()
    z_list = []

    with torch.no_grad():
        for i in range(0, len(indices), 32):
            bi = indices[i:i + 32]
            toks = torch.stack([
                torch.tensor(dataset.padded[k], dtype=torch.float32)
                for k in bi
            ]).to(device)
            z = ae.encode(toks)
            z_list.append(z.cpu().numpy())

    z_all = np.concatenate(z_list, axis=0)
    std = z_all.std(axis=0)
    norm = np.linalg.norm(z_all, axis=1)

    out = {
        "z_shape": z_all.shape,
        "z_std_mean": float(std.mean()),
        "z_std_median": float(np.median(std)),
        "z_std_min": float(std.min()),
        "z_std_max": float(std.max()),
        "z_norm_mean": float(norm.mean()),
        "dead_dims": int((std < 1e-3).sum()),
        "latent_dim": int(z_all.shape[1]),
    }
    return out


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
    log_detail("  ✓ training curves figure generated")



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
    log_detail("  ✓ latent 분석 figure 생성 (PCA + t-SNE + cumvar)")




# ══════════════════════════════════════════════════════════════
# 3D structure rendering (JSON → real-coord 3D view; viz only)
# ══════════════════════════════════════════════════════════════
def _arc_pts(sx, sy, mx, my, ex, ey, n=32):
    D = 2 * (sx * (my - ey) + mx * (ey - sy) + ex * (sy - my))
    if abs(D) < 1e-9:
        t = np.linspace(0, 1, n)
        return list(zip(sx + (ex - sx) * t, sy + (ey - sy) * t))
    ux = ((sx ** 2 + sy ** 2) * (my - ey) + (mx ** 2 + my ** 2) * (ey - sy) + (ex ** 2 + ey ** 2) * (sy - my)) / D
    uy = ((sx ** 2 + sy ** 2) * (ex - mx) + (mx ** 2 + my ** 2) * (sx - ex) + (ex ** 2 + ey ** 2) * (mx - sx)) / D
    r = math.sqrt((sx - ux) ** 2 + (sy - uy) ** 2)
    a1 = math.atan2(sy - uy, sx - ux)
    am = math.atan2(my - uy, mx - ux)
    a2 = math.atan2(ey - uy, ex - ux)

    def fix(a, ref):
        while a - ref > math.pi:
            a -= 2 * math.pi
        while a - ref < -math.pi:
            a += 2 * math.pi
        return a

    am = fix(am, a1)
    a2 = fix(a2, a1)
    if not (min(a1, a2) <= am <= max(a1, a2)):
        a2 = a2 - 2 * math.pi if a2 > a1 else a2 + 2 * math.pi
    return [(ux + r * math.cos(a), uy + r * math.sin(a)) for a in np.linspace(a1, a2, n)]


def _circle_pts(cx, cy, r, n=48):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in a]


def json_to_real_sketches(json_data: dict) -> list:
    seq = json_data["sequence"]
    sketches = []
    i = 0
    while i < len(seq):
        if seq[i]["type"] == "Sketch" and i + 1 < len(seq) and seq[i + 1]["type"] == "Extrude":
            sk = seq[i]
            ex = seq[i + 1]
            plane = sk["plane"]
            origin = _v3(plane["origin"])
            xa = _v3(plane["x_axis"])
            ya = _v3(plane["y_axis"])
            za = _v3(plane["z_axis"])
            extent_one = float(ex.get("extent_one", {}).get("distance", 0.0))
            extent_two = float(ex.get("extent_two", {}).get("distance", 0.0))
            normal = za / (np.linalg.norm(za) + 1e-12)

            loops_3d = []
            for loop in sk.get("profile", {}).get("children", []):
                pts = []
                for ei, e in enumerate(loop.get("children", [])):
                    et = e["type"]
                    if et == "Line":
                        sp = e["start_point"]
                        ep = e["end_point"]
                        if ei == 0:
                            pts.append(origin + sp["x"] * xa + sp["y"] * ya)
                        pts.append(origin + ep["x"] * xa + ep["y"] * ya)
                    elif et == "Arc":
                        sp = e["start_point"]
                        mp = e["mid_point"]
                        ep = e["end_point"]
                        a2d = _arc_pts(sp["x"], sp["y"], mp["x"], mp["y"], ep["x"], ep["y"])
                        si = 0 if ei == 0 else 1
                        for ax2, ay2 in a2d[si:]:
                            pts.append(origin + ax2 * xa + ay2 * ya)
                    elif et == "Circle":
                        cp = e["center_point"]
                        r = e["radius"]
                        c2d = _circle_pts(cp["x"], cp["y"], r)
                        for cx2, cy2 in c2d:
                            pts.append(origin + cx2 * xa + cy2 * ya)
                if len(pts) >= 2:
                    loops_3d.append(pts)
            if loops_3d:
                sketches.append({
                    "loops_3d": loops_3d,
                    "normal": normal,
                    "extent": extent_one,
                    "z_axis_raw": za,
                    "origin_z": float(origin[2]),
                    "extent_one": extent_one,
                    "extent_two": extent_two,
                })
            i += 2
        else:
            i += 1
    return sketches


def _render_3d_real_simple(ax, sketches):
    all_pts = []
    for idx, sk in enumerate(sketches):
        color = PAL[idx % len(PAL)]
        normal = np.array(sk["normal"])
        extent = sk["extent"]
        for loop_pts in sk["loops_3d"]:
            if len(loop_pts) < 2:
                continue
            bot = [np.array(p) for p in loop_pts]
            top = [p + normal * extent for p in bot]
            all_pts.extend([p.tolist() for p in bot])
            all_pts.extend([p.tolist() for p in top])
            xs_b = [p[0] for p in bot]; ys_b = [p[1] for p in bot]; zs_b = [p[2] for p in bot]
            xs_t = [p[0] for p in top]; ys_t = [p[1] for p in top]; zs_t = [p[2] for p in top]
            ax.plot(xs_b + [xs_b[0]], ys_b + [ys_b[0]], zs_b + [zs_b[0]],
                    color=color, lw=1.4, alpha=0.95)
            ax.plot(xs_t + [xs_t[0]], ys_t + [ys_t[0]], zs_t + [zs_t[0]],
                    color=color, lw=1.0, alpha=0.7)
            if len(bot) >= 3:
                pf_b = Poly3DCollection([[p.tolist() for p in bot]], alpha=0.30)
                pf_b.set_facecolor(color); pf_b.set_edgecolor("none")
                ax.add_collection3d(pf_b)
                pf_t = Poly3DCollection([[p.tolist() for p in top]], alpha=0.35)
                pf_t.set_facecolor(color); pf_t.set_edgecolor("none")
                ax.add_collection3d(pf_t)
    return all_pts


def _style_struct_ax(ax, title):
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((1, 1, 1, 0))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_title(title, color="#222", fontsize=9, fontweight="normal", pad=4)


def _set_axes_struct(ax, pts):
    if not pts:
        return
    arr = np.array(pts)
    ext = arr.max(axis=0) - arr.min(axis=0)
    pad = max(float(ext.max()) * 0.12, 0.1)
    ax.set_xlim(arr[:, 0].min() - pad, arr[:, 0].max() + pad)
    ax.set_ylim(arr[:, 1].min() - pad, arr[:, 1].max() + pad)
    ax.set_zlim(arr[:, 2].min() - pad, arr[:, 2].max() + pad)
    extents = np.array([
        ax.get_xlim3d()[1] - ax.get_xlim3d()[0],
        ax.get_ylim3d()[1] - ax.get_ylim3d()[0],
        ax.get_zlim3d()[1] - ax.get_zlim3d()[0],
    ])
    floor = max(float(extents.max()) * 0.02, 1e-3)
    extents = np.maximum(extents, floor)
    try:
        ax.set_box_aspect(tuple(extents))
    except Exception:
        pass


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

    fig = plt.figure(figsize=(18, 3.8 * n_pick), facecolor="white")
    # 4 cols: structure (3D) + S11 + S22 + S33
    gs = fig.add_gridspec(n_pick, 4, width_ratios=[1.0, 1.1, 1.1, 1.1])

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

        # ── col 0: 구조 3D view (JSON 실좌표) ──
        ax_st = fig.add_subplot(gs[row, 0], projection="3d")
        json_data = None
        try:
            json_data = dataset.load_json(idx)
        except Exception:
            json_data = None
        sketches = []
        if json_data is not None:
            try:
                sketches = json_to_real_sketches(json_data)
            except Exception:
                sketches = []
        if sketches:
            pts = _render_3d_real_simple(ax_st, sketches)
            _set_axes_struct(ax_st, pts)
            ne = len(sketches)
            nl = sum(len(s["loops_3d"]) for s in sketches)
            _style_struct_ax(ax_st, f"[{t_name}] idx={idx}\n{ne} extrude · {nl} loop")
            ax_st.view_init(elev=25, azim=-55)
        else:
            ax_st.text2D(
                0.5, 0.5, "(no JSON)",
                color="#888", ha="center", va="center",
                fontsize=10, transform=ax_st.transAxes,
            )
            _style_struct_ax(ax_st, f"[{t_name}] idx={idx}")

        # ── col 1~3: S11, S22, S33 ──
        for col, lbl in enumerate(RETURN_LABELS):
            ax = fig.add_subplot(gs[row, col + 1])

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
                f"{lbl}  RMSE={rmse:.2f} dB",
                fontsize=9, fontweight="normal",
            )
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel("|S| [dB]")
            if row == n_pick - 1:
                ax.set_xlabel("frequency [GHz]")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="best", framealpha=0.85)

        # residual label (figure-level text, 행 가운데 왼쪽)
        fig.text(
            0.005, 1.0 - (row + 0.5) / n_pick,
            f"|res|\n={res_abs:.2f} dB",
            fontsize=8, color="#555", ha="left", va="center",
        )

    plt.tight_layout()
    log_detail(f"  ✓ S-param prediction + structure figure 생성 (n={n_pick} samples)")


# ══════════════════════════════════════════════════════════════
# Data loading

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

    if log_is_compact():
        print(
            f"  data config: types={list(type_names)}, "
            f"samples/type={list(n_samples_per_type)}, freq={cfg.raw_n_freq}->{cfg.n_freq}"
        )
    else:
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

    if log_is_compact():
        df_msg = f", df≈{float(freqs_sel[1] - freqs_sel[0]):.4f}GHz" if len(freq_idx) >= 2 else ""
        print(
            f"  frequency: {cfg.n_freq} points "
            f"({freqs_sel[0]:.2f}-{freqs_sel[-1]:.2f}GHz{df_msg})"
        )
    else:
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

        if not log_is_compact():
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

        npy_ids_all = [sample_id_from_path(p) for p in npy_files]
        sp_ids_all = [sample_id_from_path(p) for p in sp_files]
        if (
            len(sp_files) == len(npy_files)
            and all(i is not None for i in npy_ids_all)
            and all(i is not None for i in sp_ids_all)
            and len(set(npy_ids_all)) == len(npy_ids_all)
            and len(set(sp_ids_all)) == len(sp_ids_all)
        ):
            if set(npy_ids_all) == set(sp_ids_all):
                sp_by_id = {sample_id_from_path(p): p for p in sp_files}
                aligned_sp_files = [sp_by_id[i] for i in npy_ids_all]
                if aligned_sp_files != sp_files:
                    log_detail("    [data-check] reordered S-param files by sample id")
                    sp_files = aligned_sp_files
            else:
                print(f"    [{tname}] token/S-param ids differ; using natural order")

        sp_arr_raw, sp_names = load_sparam_data(
            sp_files, cfg.raw_n_freq,
            expected_cols=sparam_names_ref, verbose=False,
        )

        if sparam_names_ref is None:
            sparam_names_ref = sp_names

        n_match = min(len(npy_files), sp_arr_raw.shape[0])
        if len(npy_files) != sp_arr_raw.shape[0]:
            print(
                f"    [{tname}] count mismatch token/S-param: "
                f"tokens={len(npy_files)}, sparam={sp_arr_raw.shape[0]} → use {n_match}"
            )

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
            if log_is_compact():
                print(f"  [{tname}] selected {len(npy_files)} sample(s) by {mode}")
            else:
                print(f"    selected: {len(npy_files)} by {mode}")
        else:
            if log_is_compact():
                print(f"  [{tname}] using all {len(npy_files)} sample(s)")
            else:
                print(f"    use all: {len(npy_files)}")

        sample_ids = [sample_id_from_path(p) for p in npy_files]
        if any(sid is None for sid in sample_ids):
            print(f"    [{tname}] some token file ids could not be parsed")
        elif len(set(sample_ids)) != len(sample_ids):
            print(f"    [{tname}] duplicate token file ids detected")
        if len(sp_files) != len(npy_files):
            log_detail("    [data-check] token/S-param pairing assumes natural token order == CSV block order")
        else:
            sp_ids = [sample_id_from_path(p) for p in sp_files]
            if all(sid is not None for sid in sp_ids) and set(sp_ids) != set(sample_ids):
                print(f"    [{tname}] S-param file ids differ from selected token ids")

        all_npy.extend(npy_files)
        all_sp_reim.append(sp_arr_raw)
        all_tid.extend([ti] * len(npy_files))

    sparam_reim_all_raw = np.concatenate(all_sp_reim, axis=0)
    type_ids = np.asarray(all_tid, dtype=np.int64)

    sparam_return3_reim_raw, _sparam_return3_names = filter_return3_sparams(
        sparam_reim_all_raw, sparam_names_ref,
        return_pairs=RETURN_PORT_PAIRS, verbose=not log_is_compact(),
    )

    sparam_db_full = return3_reim_to_db_np(sparam_return3_reim_raw)

    if cfg.clip_db_enable:
        print(f"\n  dB clipping enabled: [{cfg.clip_db_min}, {cfg.clip_db_max}]")
        sparam_db_full = np.clip(
            sparam_db_full, cfg.clip_db_min, cfg.clip_db_max,
        ).astype(np.float32)

    sparam_db = sparam_db_full[:, freq_idx, :]

    if log_is_compact():
        print(
            f"  S-param dB: selected={sparam_db.shape}, full={sparam_db_full.shape}, "
            f"range={sparam_db_full.min():.2f}..{sparam_db_full.max():.2f} dB"
        )
    else:
        print(f"\n  Convert Return-3 re/im → dB target")
        print(f"    raw re/im shape     : {sparam_return3_reim_raw.shape}")
        print(f"    full dB shape       : {sparam_db_full.shape}")
        print(f"    selected dB shape   : {sparam_db.shape}")
        print(f"    full dB range       : min={sparam_db_full.min():.2f}, max={sparam_db_full.max():.2f}")
        print(f"    selected dB range   : min={sparam_db.min():.2f}, max={sparam_db.max():.2f}")

    def _post_role_len(p):
        t = np.load(p)
        n_chunks = int((t[:, 0] == EXT).sum())
        return t.shape[0] + n_chunks
    max_len = min(
        max(_post_role_len(f) for f in all_npy) + 4,
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
        use_json_names=getattr(cfg, "use_json_names", True),
        merge_same_role_chunks=getattr(cfg, "merge_same_role_chunks", True),
        validate_json_alignment=getattr(cfg, "validate_json_alignment", True),
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

    if log_is_compact():
        print(f"  split: train={len(train_idx)}, val={len(val_idx)} | {per_type_msg}")
    else:
        print(f"  train={len(train_idx)} val={len(val_idx)}")
        print(f"  per-type train/val: {per_type_msg}")

    return train_idx, val_idx, val_idx_per_type


# ══════════════════════════════════════════════════════════════
# Train



# ══════════════════════════════════════════════════════════════
# Surrogate Model: encoder + S-param MLP only (no decoder)
# ══════════════════════════════════════════════════════════════
class SurrogateModel(nn.Module):
    """tokens → cmd/param embed (+fourier+ROLE) → Transformer encoder → PMA pool → z
    → SparamCommonResidualMLP (common_curve + residual) → S-param.

    AE_main 의 인코더 부분과 동일 구조. 디코더, cmd_head, param_head,
    aux_numeric_head, decode codebook 관련 로직 전부 제거.
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
            enable_nested_tensor=False,
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

        # ★ Surrogate MLP: z → S-param
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

    def forward(self, x, return_parts=False):
        z = self.encode(x)
        if return_parts:
            pred_sel, residual = self.surrogate(z, return_parts=True)
            return pred_sel, z, residual
        pred_sel = self.surrogate(z)
        return pred_sel, z


# ══════════════════════════════════════════════════════════════
# Surrogate training loop (only S-param + VICReg loss)
# ══════════════════════════════════════════════════════════════
def run_epoch_surrogate(model, loader, optimizer, device, cfg, interp_w, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    acc = {
        "total": 0.0, "sp": 0.0,
        "rmse_db_full": 0.0, "rmse_db_sel": 0.0,
        "var": 0.0, "cov": 0.0,
        "z_std": 0.0, "z_norm": 0.0,
        "res_abs": 0.0, "grad_norm": 0.0,
    }
    n = 0

    for batch in loader:
        batch_tok, batch_sel_db, batch_full_db, _tid = batch
        batch_tok = batch_tok.to(device)
        batch_sel_db = batch_sel_db.to(device)
        batch_full_db = batch_full_db.to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            pred_sel_db, z, residual = model(batch_tok, return_parts=True)

            sp_loss, sp_comp, _pred_full_db = sparam_full_interp_loss(
                pred_sel_db=pred_sel_db,
                true_sel_db=batch_sel_db,
                true_full_db=batch_full_db,
                interp_w=interp_w, cfg=cfg,
            )

            total = cfg.w_sparam * sp_loss

            if cfg.use_vicreg and (cfg.w_var > 0 or cfg.w_cov > 0) and z.size(0) > 1:
                var_loss, cov_loss = vicreg_z_loss(
                    z, var_target=cfg.vicreg_var_target,
                )
                total = total + cfg.w_var * var_loss + cfg.w_cov * cov_loss
            else:
                var_loss = z.new_zeros(())
                cov_loss = z.new_zeros(())

            if train_mode:
                total.backward()
                gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                acc["grad_norm"] += float(gn)

        with torch.no_grad():
            acc["total"] += float(total.item())
            acc["sp"] += float(sp_loss.item())
            acc["rmse_db_full"] += sp_comp["rmse_db_full"]
            acc["rmse_db_sel"] += sp_comp["rmse_db_sel"]
            acc["var"] += float(var_loss.item())
            acc["cov"] += float(cov_loss.item())
            acc["z_std"] += float(z.std(dim=0).mean().item()) if z.size(0) > 1 else 0.0
            acc["z_norm"] += float(z.norm(dim=1).mean().item())
            acc["res_abs"] += float(residual.abs().mean().item())

        n += 1

    if n == 0:
        return {**acc, "n_batches": 0}

    out = {k: v / n for k, v in acc.items()}
    out["n_batches"] = n
    return out


def print_surrogate_eval_diagnostics(eval_metrics, latent_diag, type_names):
    section("EVALUATION DIAGNOSTICS")

    if log_is_compact():
        print(
            f"  eval RMSE: selected={eval_metrics['overall_selected_rmse']:.4f} dB, "
            f"full={eval_metrics['overall_full_rmse']:.4f} dB, "
            f"|res|={eval_metrics['overall_residual_abs']:.4f} dB"
        )
        print(
            f"  latent: z_std={latent_diag['z_std_mean']:.4f}, "
            f"dead={latent_diag['dead_dims']}/{latent_diag['latent_dim']}"
        )
        return

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
        print(f"  mean |residual| : {r['res_abs']:.3f} dB")

    print(f"\n  Overall selected-point Return-3 RMSE")
    print(f"    TOTAL: {eval_metrics['overall_selected_rmse']:.4f} dB")
    print(f"\n  Overall interpolated full-grid Return-3 RMSE")
    print(f"    TOTAL: {eval_metrics['overall_full_rmse']:.4f} dB")
    print(f"\n  Overall mean |residual|")
    print(f"    TOTAL: {eval_metrics['overall_residual_abs']:.4f} dB")

    print(f"\n  Latent diagnostic")
    print(f"    z shape       : {latent_diag['z_shape']}")
    print(f"    z_std mean    : {latent_diag['z_std_mean']:.4f}")
    print(f"    z_std median  : {latent_diag['z_std_median']:.4f}")
    print(f"    z_std min/max : {latent_diag['z_std_min']:.4f} / {latent_diag['z_std_max']:.4f}")
    print(f"    ||z|| mean    : {latent_diag['z_norm_mean']:.4f}")
    print(f"    dead dims     : {latent_diag['dead_dims']} / {latent_diag['latent_dim']}")


# ══════════════════════════════════════════════════════════════
# Train (surrogate only)
# ══════════════════════════════════════════════════════════════
def train_surrogate(
    cfg, dataset, type_names, train_idx, val_idx, val_idx_per_type,
    common_curve, device,
):
    set_seed(cfg.seed)

    interp_w = torch.tensor(
        dataset.interp_matrix, dtype=torch.float32, device=device,
    )

    section("RUN START — surrogate-only")
    if log_is_compact():
        print(
            f"  preset={cfg.preset}, epochs={cfg.epochs}, bs={cfg.batch_size}, "
            f"d_model={cfg.d_model}, latent={cfg.latent}, freq={cfg.n_freq}/{cfg.raw_n_freq}"
        )
    else:
        print(f"  preset             : {cfg.preset}")
        print(f"  use_vicreg          : {cfg.use_vicreg}")
        print(f"  w_var / w_cov       : {cfg.w_var} / {cfg.w_cov}")
        print(f"  var_target          : {cfg.vicreg_var_target}")
        print(f"  epochs / bs         : {cfg.epochs} / {cfg.batch_size}")
        print(f"  d_model / latent    : {cfg.d_model} / {cfg.latent}")

    train_set = SubsetDataset(dataset, train_idx)
    val_set = SubsetDataset(dataset, val_idx)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
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

    # Optional ckpt load (skip training)
    loaded_from_ckpt = False
    ckpt_best_metric = None
    ckpt_load_path = getattr(cfg, "ckpt_load_path", "")
    if ckpt_load_path and os.path.exists(ckpt_load_path):
        try:
            ckpt = torch.load(ckpt_load_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            saved_map = ckpt.get("role_name_to_id", {})
            cur_map = getattr(dataset, "role_name_to_id", {})
            if (saved_map or cur_map) and saved_map != cur_map:
                raise ValueError(
                    f"role_name_to_id mismatch — saved={saved_map}, current={cur_map}"
                )
            ckpt_best_metric = ckpt.get("best_val_metric", None)
            loaded_from_ckpt = True
            print(f"\n  ✓ loaded ckpt ← {ckpt_load_path}")
            print(f"    → training will be skipped (0 epochs)\n")
        except Exception as e:
            print(f"  ⚠ failed to load ckpt {ckpt_load_path}: {e} — training from scratch")
    elif ckpt_load_path:
        print(f"  ⚠ ckpt_load_path={ckpt_load_path} not found — training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    encoder_params = (
        sum(p.numel() for p in model.cmd_emb.parameters())
        + sum(p.numel() for p in model.param_embs.parameters())
        + sum(p.numel() for p in model.param_proj.parameters())
        + sum(p.numel() for p in model.emb_norm.parameters())
        + sum(p.numel() for p in model.encoder.parameters())
        + sum(p.numel() for p in model.to_z.parameters())
    )
    surrogate_params = sum(p.numel() for p in model.surrogate.parameters())

    section("BUILD MODEL")
    if log_is_compact():
        print(
            f"  params total={n_params:,} | "
            f"encoder≈{encoder_params:,} | surrogate={surrogate_params:,}"
        )
    else:
        print(f"  total params        : {n_params:,}")
        print(f"  encoder + pool + z  : {encoder_params:,}")
        print(f"  surrogate MLP       : {surrogate_params:,}")
        print(f"  decoder             : (none, surrogate-only)")
        print(f"  Surrogate form      : pred = common_curve + residual(z)")
        print(f"  VICReg              : w_var={cfg.w_var}, w_cov={cfg.w_cov}, "
              f"target={cfg.vicreg_var_target}")
        print(f"  selected output dim : {cfg.n_freq * 3}")
        print(f"  full target dim     : {cfg.raw_n_freq * 3}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr_ae, weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
    )

    def lr_sched(ep):
        if ep < cfg.warmup:
            return (ep + 1) / max(1, cfg.warmup)
        return 0.5 * (
            1.0 + math.cos(
                math.pi * (ep - cfg.warmup) / max(1, cfg.epochs - cfg.warmup)
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    best_metric = float(ckpt_best_metric) if ckpt_best_metric is not None else float("inf")
    best_state = None

    hist = {
        "tr_total": [], "va_total": [],
        "tr_full": [], "va_full": [],
        "tr_sel": [], "va_sel": [],
        "res_abs": [],
        "z_std": [], "var": [], "cov": [],
    }
    log_every = max(1, cfg.epochs // 8)

    section("SURROGATE TRAINING")
    if log_is_compact():
        print(
            f"  loss = w_sparam·sp + w_var·var + w_cov·cov  "
            f"(w_sparam={cfg.w_sparam}, w_var={cfg.w_var}, w_cov={cfg.w_cov})"
        )
        print(f"\n  {'ep':>4s} | {'tr_full':>8s} {'va_full':>8s} {'best':>8s} | "
              f"{'ep_t':>5s} {'eta':>6s} | {'lr':>8s}")
        print("  " + "-" * 65)
    else:
        print(f"  loss = w_sparam·sparam + w_var·var + w_cov·cov")
        print(f"  lr={cfg.lr_ae:.2e}, wd={cfg.weight_decay}")
        print(f"\n  {'ep':>4s} | {'tr_total':>9s} {'tr_full':>8s} {'tr_sel':>8s} "
              f"{'var':>7s} {'cov':>7s} {'zstd':>6s} | "
              f"{'va_total':>9s} {'va_full':>8s} {'va_sel':>8s} | "
              f"{'ep_t':>5s} {'eta':>6s} | {'lr':>8s}")
        print("  " + "-" * 130)

    t_loop_start = time.time()
    n_epochs_eff = 0 if loaded_from_ckpt else cfg.epochs

    for ep in range(1, n_epochs_eff + 1):
        t_ep = time.time()
        tr = run_epoch_surrogate(
            model, train_loader, optimizer, device, cfg,
            interp_w=interp_w, train_mode=True,
        )
        va = run_epoch_surrogate(
            model, val_loader, optimizer, device, cfg,
            interp_w=interp_w, train_mode=False,
        )
        scheduler.step()
        ep_time = time.time() - t_ep
        elapsed = time.time() - t_loop_start
        eta = elapsed / ep * (n_epochs_eff - ep)

        val_metric = va["rmse_db_full"]

        hist["tr_total"].append(tr["total"])
        hist["va_total"].append(va["total"])
        hist["tr_full"].append(tr["rmse_db_full"])
        hist["va_full"].append(va["rmse_db_full"])
        hist["tr_sel"].append(tr["rmse_db_sel"])
        hist["va_sel"].append(va["rmse_db_sel"])
        hist["res_abs"].append(va["res_abs"])
        hist["z_std"].append(tr["z_std"])
        hist["var"].append(tr["var"])
        hist["cov"].append(tr["cov"])

        if val_metric < best_metric:
            best_metric = val_metric
            best_state = {
                k: v.detach().clone().cpu()
                for k, v in model.state_dict().items()
            }

        if ep == 1 or ep % log_every == 0 or ep == cfg.epochs:
            if log_is_compact():
                print(
                    f"  {ep:4d} | "
                    f"{tr['rmse_db_full']:8.3f} {va['rmse_db_full']:8.3f} "
                    f"{best_metric:8.3f} | {ep_time:5.1f} {eta:6.0f}s | "
                    f"{optimizer.param_groups[0]['lr']:8.2e}"
                )
            else:
                print(
                    f"  {ep:4d} | "
                    f"{tr['total']:9.4f} {tr['rmse_db_full']:8.3f} {tr['rmse_db_sel']:8.3f} "
                    f"{tr['var']:7.4f} {tr['cov']:7.4f} {tr['z_std']:6.3f} | "
                    f"{va['total']:9.4f} {va['rmse_db_full']:8.3f} {va['rmse_db_sel']:8.3f} | "
                    f"{ep_time:5.1f} {eta:6.0f}s | "
                    f"{optimizer.param_groups[0]['lr']:8.2e}"
                )

    t_loop_total = time.time() - t_loop_start
    print(f"\n  Training loop elapsed: {t_loop_total:.1f} s  ({t_loop_total / 60:.1f} min)")
    print(f"  Best full-grid val RMSE dB: {best_metric:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ── Checkpoint save (self-contained: model + cfg + source + freqs + common_curve) ──
    ckpt_save_path = getattr(cfg, "ckpt_save_path", "")
    if (ckpt_save_path and getattr(cfg, "save_ckpt_after_train", True)
            and not loaded_from_ckpt):
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
            cc_np = (
                common_curve.detach().cpu().numpy()
                if hasattr(common_curve, "detach")
                else np.asarray(common_curve)
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "role_name_to_id": getattr(dataset, "role_name_to_id", {}),
                "role_id_to_name": getattr(dataset, "role_id_to_name", {}),
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
            print(f"  ⚠ failed to save ckpt {ckpt_save_path}: {e}")

    section("EVALUATION")

    eval_metrics = evaluate_sparam_predictions(
        ae=model, mlp=model.surrogate,
        dataset=dataset, val_idx_per_type=val_idx_per_type,
        type_names=type_names, cfg=cfg, device=device,
    )

    latent_diag = diagnose_latent_simple(
        ae=model, dataset=dataset,
        indices=val_idx if len(val_idx) > 0 else train_idx,
        device=device,
    )

    print_surrogate_eval_diagnostics(eval_metrics, latent_diag, type_names)

    if cfg.show_figures:
        subsection("Training curves")
        try:
            plot_training_curves(hist)
        except Exception as e:
            print(f"  ⚠ plot_training_curves failed: {type(e).__name__}: {e}")

        subsection("Latent space analysis")
        try:
            viz_idx = val_idx if len(val_idx) > 0 else train_idx
            z_all, tids = collect_latents(model, dataset, viz_idx, device)
            analyze_latent_space(z_all, tids, type_names)
        except Exception as e:
            print(f"  ⚠ analyze_latent_space failed: {type(e).__name__}: {e}")

        subsection("S-param prediction figures")
        try:
            n_sp = min(3, len(val_idx)) if len(val_idx) > 0 else 0
            if n_sp > 0:
                visualize_sparam_predictions(
                    ae=model, mlp=model.surrogate,
                    dataset=dataset, val_indices=val_idx, device=device,
                    n_samples=n_sp, seed=cfg.seed,
                )
        except Exception as e:
            print(f"  ⚠ visualize_sparam_predictions failed: {type(e).__name__}: {e}")

    result = {
        "best_val": best_metric,
        "eval_full": eval_metrics["overall_full_rmse"],
        "eval_sel": eval_metrics["overall_selected_rmse"],
        "res_abs": eval_metrics["overall_residual_abs"],
        "z_std": latent_diag["z_std_mean"],
        "z_norm": latent_diag["z_norm_mean"],
        "dead": latent_diag["dead_dims"],
        "latent_dim": latent_diag["latent_dim"],
    }

    section("FINAL SUMMARY — surrogate-only")
    if log_is_compact():
        print(
            f"  best_val={result['best_val']:.4f} dB | "
            f"eval_full={result['eval_full']:.4f} dB | "
            f"eval_sel={result['eval_sel']:.4f} dB | "
            f"dead={result['dead']}/{result['latent_dim']}"
        )
    else:
        print(f"  best_val     : {result['best_val']:.4f} dB")
        print(f"  eval_full    : {result['eval_full']:.4f} dB")
        print(f"  eval_sel     : {result['eval_sel']:.4f} dB")
        print(f"  res_abs      : {result['res_abs']:.4f} dB")
        print(f"  z_std        : {result['z_std']:.4f}")
        print(f"  z_norm       : {result['z_norm']:.4f}")
        print(f"  dead dims    : {result['dead']} / {result['latent_dim']}")

    return model, result


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    from datetime import datetime as _dt
    run_tag = _dt.now().strftime("%Y%m%d_%H%M%S")

    t0 = time.time()

    LOG_VERBOSITY = "compact"
    set_log_verbosity(LOG_VERBOSITY)

    USE_TYPES = [1, 2, 3]
    RAW_N_FREQ = 401
    SELECTED_N_FREQ = 81
    FREQ_SELECT_MODE = "linspace"
    SAMPLES_PER_TYPE = (800, 800, 800)
    SHOW_FIGURES = True

    USE_VICREG = True
    VICREG_W_VAR = 0.2
    VICREG_W_COV = 0.02
    VICREG_VAR_TARGET = 0.7

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

    if len(SAMPLES_PER_TYPE) != len(USE_TYPES):
        raise ValueError(
            f"SAMPLES_PER_TYPE length mismatch: "
            f"{len(SAMPLES_PER_TYPE)} vs USE_TYPES {len(USE_TYPES)}"
        )

    print(
        f"[CONFIG] preset={PRESET}, types={USE_TYPES}, "
        f"freq={RAW_N_FREQ}->{SELECTED_N_FREQ}, log={LOG_VERBOSITY}"
    )

    cfg = CFG(
        preset=PRESET,
        seed=7,
        run_name="surrogate_only",
        log_verbosity=LOG_VERBOSITY,
        show_figures=SHOW_FIGURES,

        npy_dirs=npy_dirs,
        sparam_globs=sparam_globs,
        type_names=type_names,
        n_samples_per_type=SAMPLES_PER_TYPE,
        sample_mode="random",

        raw_n_freq=RAW_N_FREQ,
        n_freq=SELECTED_N_FREQ,
        freq_select_mode=FREQ_SELECT_MODE,
        freq_start=1.0,
        freq_end=5.0,

        clip_db_enable=False,

        lr_ae=3e-4,
        lr_mlp=1e-3,
        weight_decay=1e-3,
        grad_clip=1.0,
        val_ratio=0.15,

        # ★ surrogate-only: cmd/prm/aux loss 비활성화
        w_cmd=0.0,
        w_prm=0.0,
        w_aux=0.0,
        w_sparam=1.0,

        db_loss_scale=20.0,

        use_vicreg=USE_VICREG,
        w_var=VICREG_W_VAR,
        w_cov=VICREG_W_COV,
        vicreg_var_target=VICREG_VAR_TARGET,

        aux_numeric=False,

        mlp_hidden_mult=2.0,
        mlp_dropout=0.3,

        residual_scale=1.0,
        zero_init_residual=True,

        n_preview=0,
        ckpt_save_path="ckpt/surrogate_last.pt",
    )

    try:
        apply_preset(cfg)

        cfg.use_vicreg = USE_VICREG
        cfg.w_var = VICREG_W_VAR
        cfg.w_cov = VICREG_W_COV
        cfg.vicreg_var_target = VICREG_VAR_TARGET

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        section("Surrogate-only training (encoder + S-param MLP)")
        dev_msg = str(device) + (
            f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
        )
        print(f"  env: python={_sys.version.split()[0]}, torch={torch.__version__}, device={dev_msg}")

        section("LOAD DATA")
        dataset, npy_files, type_ids, type_names, sparam_db, max_len = load_multitype_data(
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

        model, result = train_surrogate(
            cfg=cfg, dataset=dataset, type_names=type_names,
            train_idx=train_idx, val_idx=val_idx,
            val_idx_per_type=val_idx_per_type,
            common_curve=common_curve, device=device,
        )

        section("DONE")
        print(f"  Final eval_full RMSE : {result['eval_full']:.4f} dB")
        print(f"  figures: {len(plt.get_fignums())} (backend={_MPL_BACKEND})")

        elapsed = time.time() - t0
        h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = elapsed % 60
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
        h = int(elapsed // 3600); m = int((elapsed % 3600) // 60); s = elapsed % 60
        print(f"\n  경과 시간 (실패 시점): {h}h {m}m {s:.1f}s")


# ═══════════════════════════════════════════════════════════════
#  ★ 여기서 선택 ★    "tiny"  /  "small"  /  "full"
# ═══════════════════════════════════════════════════════════════
PRESET = "small"


if __name__ == "__main__":
    main()
