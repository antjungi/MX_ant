#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepCAD AE  —  v3 (ROLE token + same-role merge + canonical sketch reorder)
============================================================================

v2 의 모든 동작 +
★ Sketch 순서를 canonical role 순으로 재정렬:

    frame → S1_port → S2_port → S3_port → S1_gnd → S2_gnd → S3_gnd → camera

  - camera 처럼 sample 마다 있을 수도/없을 수도 있는 component 는 끝에 배치
    → autoregressive 분기점이 단순해짐 ("frame 뒤에 camera 또는 EOS")
  - 같은 role 이 여러 개여도 canonical 위치에 모이게 됨 → 그 후 v2 의 merge 가
    coplanar+coextrude 면 1 chunk(multi-loop) 으로 합침
  - canonical 에 없는 role 은 끝에 (camera 보다 뒤) 정렬

처리 순서 (dataset 로딩 시):
  1) JSON metadata.solids[].name 으로 chunk role 식별
  2) ★ canonical role 순으로 Sketch+Extrude pair 재정렬 (v3 신규)
  3) 같은 role + coplanar + same extrude → 1 sketch(multi-loop) 병합 (v2)
  4) 재토큰화
  5) 각 chunk 첫 SOL 앞에 ROLE 토큰 삽입 (v1)
  6) 병합/재정렬된 JSON 캐시 → load_json() 일관성 유지

CFG:
  reorder_sketches_by_role: bool   = True
  canonical_role_order    : tuple  = ("frame", "S1_port", "S2_port", "S3_port",
                                       "S1_gnd", "S2_gnd", "S3_gnd", "camera")
  merge_same_role_chunks  : bool   = True   (v2)
  use_json_names          : bool   = True   (v1)
"""

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

    # ★ ROLE 토큰화 + 같은 role coplanar chunk 자동 병합
    use_json_names: bool = True              # JSON metadata names → ROLE 토큰 삽입
    merge_same_role_chunks: bool = True      # 같은 role + 같은 plane + 같은 extrude → 1 chunk(multi-loop)
    reorder_sketches_by_role: bool = True    # ★ canonical role 순서로 Sketch+Extrude pair 재정렬
    canonical_role_order: tuple = (
        "frame",
        "S1_port", "S2_port", "S3_port",
        "S1_gnd",  "S2_gnd",  "S3_gnd",
        "camera",                            # 없을 수도 있는 component → 마지막
    )

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
def save_inverse_design_outputs(inv_result, save_dir, run_tag,
                                channel_target_freqs=None,
                                bandwidth_ghz=None,
                                deep_db=None):
    """Inverse design 결과를 'inversed/' 폴더에 저장.

    1) inverse_<tag>_tokens.txt — 디코딩된 시퀀스 (가독성 좋게)
    2) inverse_<tag>_sparam.csv — 예상 S-param (freq, S11, S22, S33 in dB)
    3) inverse_<tag>_fig*.png    — 인버스 디자인 관련 figure 들
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = []

    # ── 1) 시퀀스 토큰 → txt
    tokens = inv_result.get("decoded_tokens")
    if tokens is not None and len(tokens) > 0:
        tok_path = os.path.join(save_dir, f"inverse_{run_tag}_tokens.txt")
        try:
            with open(tok_path, "w", encoding="utf-8") as f:
                f.write(f"# Inverse design decoded tokens  (run_tag={run_tag})\n")
                if channel_target_freqs is not None:
                    f.write(
                        f"# Target: S11@{channel_target_freqs[0]:.2f}GHz, "
                        f"S22@{channel_target_freqs[1]:.2f}GHz, "
                        f"S33@{channel_target_freqs[2]:.2f}GHz"
                    )
                    if bandwidth_ghz is not None:
                        f.write(f", BW=±{bandwidth_ghz / 2 * 1000:.0f}MHz")
                    if deep_db is not None:
                        f.write(f", spec ≤ {deep_db:.0f}dB")
                    f.write("\n")
                f.write(f"# n_tokens = {len(tokens)}\n")
                f.write(f"# best_match_loss = {inv_result.get('best_loss', float('nan')):.6f}\n")
                f.write(f"# Columns: idx | cmd_name | param0..param15 (PAD=-1)\n\n")
                for i, row in enumerate(tokens):
                    c = int(row[0])
                    cmd_name = CMD_NAME.get(c, f"UNK({c})")
                    params = " ".join(f"{int(p):5d}" for p in row[1:])
                    f.write(f"[{i:4d}] {cmd_name:7s} | {params}\n")
            saved.append(tok_path)
        except Exception as e:
            print(f"  ⚠ tokens txt 저장 실패: {type(e).__name__}: {e}")

    # ── 2) 예상 S-param → csv
    pred = inv_result.get("best_pred_full")
    freqs = inv_result.get("freqs_full")
    if pred is not None and freqs is not None:
        sparam_path = os.path.join(save_dir, f"inverse_{run_tag}_sparam.csv")
        try:
            with open(sparam_path, "w", encoding="utf-8") as f:
                f.write("freq_GHz,S11_dB,S22_dB,S33_dB\n")
                for i in range(len(freqs)):
                    f.write(
                        f"{float(freqs[i]):.6f},"
                        f"{float(pred[i, 0]):.4f},"
                        f"{float(pred[i, 1]):.4f},"
                        f"{float(pred[i, 2]):.4f}\n"
                    )
            saved.append(sparam_path)
        except Exception as e:
            print(f"  ⚠ S-param csv 저장 실패: {type(e).__name__}: {e}")

    # ── 3) 인버스 디자인 figure 들 → png
    #   inverse 또는 decoded 와 관련된 suptitle 을 가진 figure 만 저장.
    fig_idx = 0
    for fn in plt.get_fignums():
        try:
            fig = plt.figure(fn)
            suptitle = ""
            if getattr(fig, "_suptitle", None) is not None:
                suptitle = fig._suptitle.get_text() or ""
            stl = suptitle.lower()
            if not ("inverse" in stl or "decoded" in stl or "top view" in stl):
                continue
            # 의미 있는 단서 분류
            tag_part = "fig"
            if "top view" in stl and "annotated" in stl:
                tag_part = "topview_annotated"
            elif "top view" in stl and "clean" in stl:
                tag_part = "topview_clean"
            elif "top view" in stl:
                tag_part = "topview"
            elif "decoded" in stl:
                tag_part = "decoded"
            elif "inverse design" in stl:
                tag_part = "sparam_curve"
            fig_idx += 1
            fig_path = os.path.join(
                save_dir, f"inverse_{run_tag}_{tag_part}_{fig_idx}.png",
            )
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            saved.append(fig_path)
        except Exception as e:
            print(f"  ⚠ figure fig{fn} 저장 실패: {type(e).__name__}: {e}")

    print(f"  ✓ inversed 폴더에 {len(saved)} 개 파일 저장:")
    for p in saved:
        print(f"    · {os.path.basename(p)}")


# ══════════════════════════════════════════════════════════════
# Geometry visualization
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
            all_extents.append(op["extent_one"]["distance"])
            all_extents.append(op["extent_two"]["distance"])

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
            e1 = op["extent_one"]["distance"]
            e2 = op.get("extent_two", {}).get("distance", 0.0)
            q_e1 = _q01_tok(_norm_tok(e1, 0.0, max_ext))
            q_e2 = _q01_tok(_norm_tok(e2, 0.0, max_ext))
            q_bool = _BOOL_MAP_TOK.get(op.get("operation", "NewBody"), 0)
            q_etype = _ETYPE_MAP_TOK.get(op.get("extent_type", "OneSide"), 0)
            rows.append(row(EXT, q_sk_scale, q_ox, q_oy, q_oz,
                            q_e1, q_e2, q_bool, q_etype))
    rows.append(row(EOS))
    return np.asarray(rows, dtype=np.int32)


def reorder_sketches_by_canonical_role(json_data, names, canonical_order,
                                        verbose=False):
    """JSON 의 Sketch+Extrude pair 들을 canonical role 순으로 재정렬.

    canonical_order 에 없는 role 은 끝쪽(canonical 다음 우선순위)에 모이고,
    이름 없는 chunk 는 그 뒤에. 같은 role 끼리는 원본 등장 순서 유지 (stable).

    Returns:
        new_json_data, new_names, changed_flag
    """
    seq = json_data["sequence"]
    pairs = []
    extra_ops = []
    i = 0
    while i < len(seq):
        if (seq[i].get("type") == "Sketch" and i + 1 < len(seq)
                and seq[i + 1].get("type") == "Extrude"):
            idx = len(pairs)
            nm = names[idx] if (names is not None and idx < len(names)) else None
            pairs.append((seq[i], seq[i + 1], nm))
            i += 2
        else:
            extra_ops.append(seq[i])
            i += 1

    canon = list(canonical_order)
    canon_index = {r: k for k, r in enumerate(canon)}
    N_CANON = len(canon)

    def _key(item):
        orig_idx, (_sk, _ex, nm) = item
        if nm is None:
            return (N_CANON + 2, orig_idx)
        if nm in canon_index:
            return (canon_index[nm], orig_idx)
        return (N_CANON + 1, orig_idx)   # canonical 에 없는 알려진 role

    indexed = list(enumerate(pairs))
    indexed_sorted = sorted(indexed, key=_key)

    sorted_pairs = [p for _, p in indexed_sorted]
    new_names = [p[2] for p in sorted_pairs]
    changed = any(orig != new for new, (orig, _) in enumerate(indexed_sorted))

    new_seq = list(extra_ops)
    for sk, ex, _nm in sorted_pairs:
        new_seq.append(sk)
        new_seq.append(ex)
    new_json = dict(json_data)
    new_json["sequence"] = new_seq

    if verbose and changed:
        orig_order = [nm for _, _, nm in pairs]
        new_order = [nm for _, _, nm in sorted_pairs]
        print(f"    reorder: {orig_order} → {new_order}")

    return new_json, new_names, changed


def merge_coplanar_same_role_sketches(json_data, names,
                                       plane_tol=1e-3, extent_tol=1e-3,
                                       verbose=False):
    """JSON sequence 에서 연속된 Sketch+Extrude pair 중
    같은 role + coplanar + 같은 extrude 인 것들을 1개의 Sketch (multi-loop) + 1개의 Extrude 로 병합.

    Args:
        json_data: original JSON dict
        names: list of role names per (Sketch, Extrude) pair, 길이 = n_pairs

    Returns:
        new_json_data: 병합 결과 (원본 unchanged)
        new_names: 병합 후 pair 들의 role name 리스트
        n_merged: 줄어든 chunk 수 (양수면 merge 발생)
    """
    seq = json_data["sequence"]
    pairs = []
    extra_ops = []
    i = 0
    while i < len(seq):
        if (seq[i].get("type") == "Sketch" and i + 1 < len(seq)
                and seq[i + 1].get("type") == "Extrude"):
            idx = len(pairs)
            nm = names[idx] if (names is not None and idx < len(names)) else None
            pairs.append((seq[i], seq[i + 1], nm))
            i += 2
        else:
            extra_ops.append(seq[i])
            i += 1

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
        return True

    merged = []
    new_names = []
    j = 0
    while j < len(pairs):
        sk, ex, nm = pairs[j]
        loops = list(sk.get("profile", {}).get("children", []))
        k = j + 1
        while k < len(pairs):
            sk2, ex2, nm2 = pairs[k]
            if (nm is not None and nm == nm2
                    and planes_match(sk["plane"], sk2["plane"])
                    and extrudes_match(ex, ex2)):
                for lp in sk2.get("profile", {}).get("children", []):
                    new_lp = dict(lp)
                    new_lp["is_outer"] = True   # 이 단계의 모든 loop = 별도 outer
                    if "children" in lp:
                        new_lp["children"] = list(lp["children"])
                    loops.append(new_lp)
                k += 1
            else:
                break

        if k > j + 1:
            fixed_loops = []
            for lp in loops:
                new_lp = dict(lp)
                new_lp["is_outer"] = True
                if "children" in lp:
                    new_lp["children"] = list(lp["children"])
                fixed_loops.append(new_lp)
            new_sk = dict(sk)
            new_sk["profile"] = dict(sk.get("profile", {}))
            new_sk["profile"]["children"] = fixed_loops
            merged.append((new_sk, ex, nm))
            if verbose:
                print(f"    merged role={nm!r}: {k - j} chunks → 1 chunk "
                      f"({len(fixed_loops)} loops total)")
        else:
            merged.append((sk, ex, nm))
        new_names.append(nm)
        j = k

    new_seq = list(extra_ops)
    for sk, ex, _nm in merged:
        new_seq.append(sk)
        new_seq.append(ex)
    new_json = dict(json_data)
    new_json["sequence"] = new_seq

    n_merged = len(pairs) - len(merged)
    return new_json, new_names, n_merged


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
            extent_one = float(ex["extent_one"]["distance"])
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


def _dequant(q, lo, hi):
    if q < 0:
        return (lo + hi) / 2.0
    return lo + (q / QMAX) * (hi - lo)


def tokens_to_real_sketches(tokens: np.ndarray, dq_meta: dict) -> list:
    sk_metas = dq_meta["sketches"]
    max_ext = dq_meta["max_ext"]
    sketches = []
    cur_loops = []
    cur_pts = []
    sketch_idx = 0
    cur_2d = (0.0, 0.0)

    for row in tokens:
        cmd = int(row[0])
        p = row[1:]

        if cmd == SOL:
            if cur_pts:
                cur_loops.append(cur_pts)
            cur_pts = []
            cur_2d = (0.0, 0.0)
        elif cmd == LINE:
            if p[0] < 0 or p[1] < 0:
                continue
            if sketch_idx < len(sk_metas):
                sm = sk_metas[sketch_idx]
                ex = _dequant(p[0], sm["xy_min"][0], sm["xy_max"][0])
                ey = _dequant(p[1], sm["xy_min"][1], sm["xy_max"][1])
                cur_pts.append(sm["origin"] + ex * sm["x_axis"] + ey * sm["y_axis"])
                cur_2d = (ex, ey)
        elif cmd == ARC:
            if any(p[i] < 0 for i in range(4)):
                continue
            if sketch_idx < len(sk_metas):
                sm = sk_metas[sketch_idx]
                mx = _dequant(p[0], sm["xy_min"][0], sm["xy_max"][0])
                my = _dequant(p[1], sm["xy_min"][1], sm["xy_max"][1])
                ex = _dequant(p[2], sm["xy_min"][0], sm["xy_max"][0])
                ey = _dequant(p[3], sm["xy_min"][1], sm["xy_max"][1])
                sx, sy = cur_2d
                a2d = _arc_pts(sx, sy, mx, my, ex, ey)
                for ax2, ay2 in a2d[1:]:
                    cur_pts.append(sm["origin"] + ax2 * sm["x_axis"] + ay2 * sm["y_axis"])
                cur_2d = (ex, ey)
        elif cmd == CIRCLE:
            if any(p[i] < 0 for i in range(3)):
                continue
            if sketch_idx < len(sk_metas):
                sm = sk_metas[sketch_idx]
                cx = _dequant(p[0], sm["xy_min"][0], sm["xy_max"][0])
                cy = _dequant(p[1], sm["xy_min"][1], sm["xy_max"][1])
                r = (p[2] / QMAX) * sm["sk_scale"]
                for x2, y2 in _circle_pts(cx, cy, r):
                    cur_pts.append(sm["origin"] + x2 * sm["x_axis"] + y2 * sm["y_axis"])
        elif cmd == EXT:
            if cur_pts:
                cur_loops.append(cur_pts)
            cur_pts = []
            if cur_loops and sketch_idx < len(sk_metas):
                sm = sk_metas[sketch_idx]
                e1_val = _dequant(p[P_E1], 0, max_ext)
                e2_val = _dequant(p[P_E2], 0, max_ext) if int(p[P_E2]) >= 0 else 0.0
                z_axis_raw = sm["z_axis"]
                normal = z_axis_raw / (np.linalg.norm(z_axis_raw) + 1e-12)
                vl = [lp for lp in cur_loops if len(lp) >= 2]
                if vl:
                    sketches.append({
                        "loops_3d": vl,
                        "normal": normal,
                        "extent": e1_val,
                        "z_axis_raw": z_axis_raw,
                        "origin_z": float(sm["origin"][2]),
                        "extent_one": float(e1_val),
                        "extent_two": float(e2_val),
                    })
            cur_loops = []
            sketch_idx += 1
        elif cmd == EOS:
            break

    return sketches



def tokens_to_sketches_normalized(tokens: np.ndarray) -> list:
    sketches = []
    cur_loops = []
    cur_pts = []
    cur_2d = (0.0, 0.0)
    for row in tokens:
        cmd = int(row[0])
        p = row[1:]
        if cmd not in CMD_NAME:
            continue
        if cmd == SOL:
            if cur_pts:
                cur_loops.append(cur_pts)
            cur_pts = []
            cur_2d = (0.0, 0.0)
        elif cmd == LINE:
            if p[0] < 0 or p[1] < 0:
                continue
            ex, ey = p[0] / QMAX, p[1] / QMAX
            cur_pts.append((ex, ey))
            cur_2d = (ex, ey)
        elif cmd == ARC:
            if any(p[i] < 0 for i in range(4)):
                continue
            mx, my = p[0] / QMAX, p[1] / QMAX
            ex, ey = p[2] / QMAX, p[3] / QMAX
            sx, sy = cur_2d
            a2d = _arc_pts(sx, sy, mx, my, ex, ey)
            for ax2, ay2 in a2d[1:]:
                cur_pts.append((ax2, ay2))
            cur_2d = (ex, ey)
        elif cmd == CIRCLE:
            if any(p[i] < 0 for i in range(3)):
                continue
            cx, cy = p[0] / QMAX, p[1] / QMAX
            r = (p[2] / QMAX) * 0.45
            for x2, y2 in _circle_pts(cx, cy, r):
                cur_pts.append((x2, y2))
        elif cmd == EXT:
            if cur_pts:
                cur_loops.append(cur_pts)
            cur_pts = []

            def gp(i):
                return p[i] / QMAX if p[i] >= 0 else 0.5

            if cur_loops:
                sketches.append({
                    "loops_2d": cur_loops,
                    "pos": (gp(P_PX), gp(P_PY), gp(P_PZ)),
                    "scale": max(gp(P_SCALE), 0.05),
                    "e1": max(gp(P_E1), 0.03),
                })
            cur_loops = []
        elif cmd == EOS:
            break
    return sketches


# ── 3D rendering helpers ──
def render_3d_real(ax, sketches: list) -> list:
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

            def outline(pts, lw=1.6, la=0.95):
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                ax.plot(
                    xs + [xs[0]], ys + [ys[0]], zs + [zs[0]],
                    color=color, lw=lw, alpha=la,
                    solid_capstyle="round", solid_joinstyle="round",
                )

            outline(bot, lw=1.8, la=0.95)
            outline(top, lw=1.2, la=0.75)

            step = max(1, len(bot) // 12)
            for i in range(0, len(bot), step):
                b, t = bot[i], top[i]
                ax.plot(
                    [b[0], t[0]], [b[1], t[1]], [b[2], t[2]],
                    color=color, lw=0.8, alpha=0.50,
                )
            if len(bot) >= 3:
                bl = [b.tolist() for b in bot]
                tl = [t.tolist() for t in top]
                for face, a in [(bl, 0.35), (tl, 0.40)]:
                    pf = Poly3DCollection([face], alpha=a)
                    pf.set_facecolor(color)
                    pf.set_edgecolor("none")
                    ax.add_collection3d(pf)
                ss = max(1, len(bot) // 20)
                for i in range(0, len(bot), ss):
                    j = (i + ss) % len(bot)
                    ps = Poly3DCollection(
                        [[bot[i].tolist(), bot[j].tolist(),
                          top[j].tolist(), top[i].tolist()]],
                        alpha=0.22,
                    )
                    ps.set_facecolor(color)
                    ps.set_edgecolor("none")
                    ax.add_collection3d(ps)
    return all_pts


def render_3d_normalized(ax, sketches: list) -> list:
    all_pts = []
    for idx, sk in enumerate(sketches):
        color = PAL[idx % len(PAL)]
        px, py, pz = sk["pos"]
        size = sk["scale"] * 2.2
        e1 = sk["e1"] * 1.8
        ox = (px - 0.5) * 2
        oy = (py - 0.5) * 2
        oz = pz * 2
        for loop_pts in sk["loops_2d"]:
            if len(loop_pts) < 2:
                continue
            bot = [((x - 0.5) * size + ox, (y - 0.5) * size + oy, oz) for x, y in loop_pts]
            top = [(b[0], b[1], b[2] + e1) for b in bot]
            all_pts.extend(bot)
            all_pts.extend(top)

            def outline(pts, lw=1.6, la=0.95):
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                zs = [p[2] for p in pts]
                ax.plot(
                    xs + [xs[0]], ys + [ys[0]], zs + [zs[0]],
                    color=color, lw=lw, alpha=la,
                    solid_capstyle="round", solid_joinstyle="round",
                )

            outline(bot, lw=1.8, la=0.95)
            outline(top, lw=1.2, la=0.75)
            step = max(1, len(bot) // 12)
            for i in range(0, len(bot), step):
                b, t = bot[i], top[i]
                ax.plot(
                    [b[0], t[0]], [b[1], t[1]], [b[2], t[2]],
                    color=color, lw=0.8, alpha=0.50,
                )
            if len(bot) >= 3:
                for face, a in [(bot, 0.35), (top, 0.40)]:
                    pf = Poly3DCollection([face], alpha=a)
                    pf.set_facecolor(color)
                    pf.set_edgecolor("none")
                    ax.add_collection3d(pf)
                ss = max(1, len(bot) // 20)
                for i in range(0, len(bot), ss):
                    j = (i + ss) % len(bot)
                    ps = Poly3DCollection([[bot[i], bot[j], top[j], top[i]]], alpha=0.22)
                    ps.set_facecolor(color)
                    ps.set_edgecolor("none")
                    ax.add_collection3d(ps)
    return all_pts


def _equal_axes(ax):
    """데이터의 실제 비율 그대로 box aspect 설정.
    (1,1,1) 강제하면 길쭉한 구조도 정육면체처럼 보여서 왜곡됨."""
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    extents = lims[:, 1] - lims[:, 0]
    # 0/음수 방지 (flat sketch 등)
    max_ext = float(extents.max()) if extents.size else 1.0
    floor = max(max_ext * 0.02, 1e-3)
    extents = np.maximum(extents, floor)
    try:
        ax.set_box_aspect(tuple(extents))
    except Exception:
        pass


def _style(ax, title):
    # ★ Orthographic projection — perspective 왜곡 제거
    #   기본은 'persp' 라 top/side view 에서 평면이 사다리꼴로 비틀려 보임.
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title(title, color="#222", fontsize=9.5, fontweight="normal", pad=8)


def _set_axes_and_render(ax, sketches, use_real):
    pts = render_3d_real(ax, sketches) if use_real else render_3d_normalized(ax, sketches)
    if pts:
        arr = np.array(pts)
        # ★ 더 넉넉한 padding (확대해도 잘려나가는 느낌 줄임)
        ext = arr.max(axis=0) - arr.min(axis=0)
        pad = max(float(ext.max()) * 0.18, 0.1)
        ax.set_xlim(arr[:, 0].min() - pad, arr[:, 0].max() + pad)
        ax.set_ylim(arr[:, 1].min() - pad, arr[:, 1].max() + pad)
        ax.set_zlim(arr[:, 2].min() - pad, arr[:, 2].max() + pad)
        _equal_axes(ax)
    else:
        ax.text2D(
            0.5, 0.5, "기하 없음",
            color="#888", ha="center", va="center",
            fontsize=12, transform=ax.transAxes,
        )
    return pts


def _compute_sketch_centers(sketches, use_real):
    """각 sketch 의 3D 중심점 (extruded body 의 가운데).
    real: bot vertices 평균 + normal * extent/2
    normalized: render_3d_normalized 와 동일한 변환으로 중심 계산.
    """
    centers = []
    for sk in sketches:
        if use_real:
            pts_all = []
            for loop in sk.get("loops_3d", []):
                pts_all.extend([np.asarray(p, dtype=float) for p in loop])
            if not pts_all:
                centers.append(None)
                continue
            arr = np.stack(pts_all, axis=0)
            normal = np.asarray(sk.get("normal", [0, 0, 1]), dtype=float)
            extent = float(sk.get("extent", 0.0))
            bot_c = arr.mean(axis=0)
            c = bot_c + normal * (extent / 2.0)
        else:
            pts2d = []
            for loop in sk.get("loops_2d", []):
                pts2d.extend(loop)
            if not pts2d:
                centers.append(None)
                continue
            arr2d = np.array(pts2d, dtype=float)
            px, py, pz = sk["pos"]
            size = sk["scale"] * 2.2
            e1 = sk["e1"] * 1.8
            ox = (px - 0.5) * 2.0
            oy = (py - 0.5) * 2.0
            oz = pz * 2.0
            cx_2d = arr2d[:, 0].mean()
            cy_2d = arr2d[:, 1].mean()
            cx = (cx_2d - 0.5) * size + ox
            cy = (cy_2d - 0.5) * size + oy
            cz = oz + e1 / 2.0
            c = np.array([cx, cy, cz], dtype=float)
        centers.append(c)
    return centers


def _annotate_centers(ax, centers, with_coords=True, with_label=True):
    """각 중심점을 marker + 좌표 라벨로 표시."""
    for i, c in enumerate(centers):
        if c is None:
            continue
        color = PAL[i % len(PAL)]
        ax.scatter(
            [float(c[0])], [float(c[1])], [float(c[2])],
            s=55, c=color, marker="o",
            edgecolors="black", linewidths=0.8,
            zorder=2000, depthshade=False,
        )
        if with_coords or with_label:
            if with_coords:
                txt = f"  S{i+1} ({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})"
            else:
                txt = f"  S{i+1}"
            ax.text(
                float(c[0]), float(c[1]), float(c[2]),
                txt,
                fontsize=6.5, color="#111",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc=(1, 1, 1, 0.82),
                    ec=(0.6, 0.6, 0.6, 0.6),
                    lw=0.3,
                ),
                zorder=2000,
            )


def _print_sketch_centers(centers, use_real):
    if not centers:
        return
    unit = "mm" if use_real else "norm"
    print(f"\n  Sketch centers ({len(centers)} entries, {unit}):")
    for i, c in enumerate(centers):
        if c is None:
            print(f"    [sketch {i+1}] (empty)")
        else:
            print(
                f"    [sketch {i+1}] center = "
                f"({c[0]:8.3f}, {c[1]:8.3f}, {c[2]:8.3f}) {unit}"
            )


def _describe_sketches_z(sketches, use_real):
    """sketch별 z축 진단 문자열 (현재 미사용 — 호출부가 패치로 제거됨)."""
    lines = []
    if not sketches:
        lines.append("  (no sketches)")
        return lines
    return lines


def show_comparison(orig_tok, recon_tok, fname, save_path,
                    json_data=None):
    """2×2: (3D View / Top View) × (Original, Recon).
    ★ 패치: 터미널에 z축 정보 출력 부분 제거됨.
    plt.show() 호출하지 않음."""
    orig_tok = trim_after_eos(orig_tok)
    recon_tok = trim_after_eos(recon_tok)
    use_real = json_data is not None

    if use_real:
        orig_sk = json_to_real_sketches(json_data)
        dq_meta = extract_dequant_meta(json_data)
        recon_sk = tokens_to_real_sketches(recon_tok, dq_meta)
    else:
        orig_sk = tokens_to_sketches_normalized(orig_tok)
        recon_sk = tokens_to_sketches_normalized(recon_tok)

    if use_real:
        nl_o = sum(len(s["loops_3d"]) for s in orig_sk)
        nl_r = sum(len(s["loops_3d"]) for s in recon_sk)
    else:
        nl_o = sum(len(s["loops_2d"]) for s in orig_sk)
        nl_r = sum(len(s["loops_2d"]) for s in recon_sk)
    ne_o = len(orig_sk)
    ne_r = len(recon_sk)

    coord_label = "real coord (mm)" if use_real else "normalized"

    # ── 2행 × 2열 layout ──
    fig = plt.figure(figsize=(15, 14), facecolor="white")
    fig.suptitle(
        f"Geometry Comparison [{coord_label}]\n{fname}",
        fontsize=11, fontweight="normal", color="#222", y=0.985,
    )

    views = [
        (0, 0, "Original — 3D View",          orig_sk,  28, -55),
        (0, 1, "Reconstruction — 3D View",    recon_sk, 28, -55),
        (1, 0, "Original — Top View",         orig_sk,  89.9, -90.0),
        (1, 1, "Reconstruction — Top View",   recon_sk, 89.9, -90.0),
    ]
    for row, col, label, sk, elev, azim in views:
        ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
        if not sk:
            ax.text2D(
                0.5, 0.5, "(empty)",
                color="#888", ha="center", va="center",
                fontsize=11, transform=ax.transAxes,
            )
            _style(ax, label)
            ax.view_init(elev=elev, azim=azim)
            continue
        _set_axes_and_render(ax, sk, use_real)
        ne = ne_o if "Original" in label else ne_r
        nl = nl_o if "Original" in label else nl_r
        tok = orig_tok if "Original" in label else recon_tok
        _style(ax, f"{label}\n{ne} extrude · {nl} loop · {tok.shape[0]} token")
        ax.view_init(elev=elev, azim=azim)

    n_leg = max(ne_o, ne_r)
    if n_leg:
        fig.legend(
            handles=[mpatches.Patch(color=PAL[i % len(PAL)], label=f"Sketch {i+1}")
                     for i in range(n_leg)],
            loc="lower center", ncol=min(n_leg, 6),
            facecolor="white", edgecolor="#ddd", labelcolor="#444",
            fontsize=9, framealpha=0.95,
        )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    print(f"  ✓ 비교 figure 생성: {fname}")


@torch.no_grad()
def visualize_reconstruction(ae, dataset, indices, device, max_samples=2):
    ae.eval()
    sel = list(indices)[:max_samples]
    if not sel:
        print("  ⚠ visualize_reconstruction: 표시할 sample 없음")
        return []

    figs = []
    for k, idx in enumerate(sel):
        x = torch.tensor(
            dataset.padded[idx],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        z = ae.encode(x)
        gen = ae.generate(z, max_gen_len=dataset.max_len)

        recon = gen[0].detach().cpu().numpy().astype(np.int32)
        orig = dataset.raw[idx]

        try:
            t_id = int(dataset.type_ids[idx])
            t_name = dataset.type_names[t_id]
        except Exception:
            t_name = "?"

        try:
            base = os.path.basename(dataset.npy_files[idx])
        except Exception:
            base = f"sample_{idx}"
        fname = f"[{t_name}] {base}"

        json_data = None
        try:
            json_data = dataset.load_json(idx)
        except Exception:
            json_data = None

        try:
            show_comparison(
                orig_tok=orig,
                recon_tok=recon,
                fname=fname,
                save_path="",
                json_data=json_data,
            )
            figs.append(plt.gcf())
            mode = "real" if json_data is not None else "normalized"
            print(
                f"  · recon viz [{k+1}/{len(sel)}]: idx={idx} type={t_name} "
                f"mode={mode}  orig_tok={len(trim_after_eos(orig))} "
                f"recon_tok={len(trim_after_eos(recon))}"
            )
        except Exception as e:
            import traceback as _tb
            print(
                f"  ⚠ recon viz [{k+1}/{len(sel)}] failed "
                f"(idx={idx}): {type(e).__name__}: {e}"
            )
            _tb.print_exc()

    return figs


# ══════════════════════════════════════════════════════════════
# S-param loading
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
        merge_same_role_chunks=True,
        reorder_sketches_by_role=True,
        canonical_role_order=(
            "frame",
            "S1_port", "S2_port", "S3_port",
            "S1_gnd",  "S2_gnd",  "S3_gnd",
            "camera",
        ),
    ):
        self.npy_files = list(npy_files)
        self.max_len = max_len
        self.merge_same_role_chunks = bool(merge_same_role_chunks)
        self.reorder_sketches_by_role = bool(reorder_sketches_by_role)
        self.canonical_role_order = tuple(canonical_role_order)
        self.merged_json_cache = [None] * len(self.npy_files)

        self.json_files = []
        for f in self.npy_files:
            jf = f.replace("_tokens.npy", "_deepcad.json")
            self.json_files.append(jf if os.path.exists(jf) else None)

        # ★ JSON metadata.solids[i].name → global role_id 매핑 구축
        per_sample_names = []
        for jf in self.json_files:
            if jf is None or not os.path.exists(jf):
                per_sample_names.append(None)
                continue
            try:
                import json as _json
                with open(jf, "r", encoding="utf-8") as f:
                    jd = _json.load(f)
                solids_meta = jd.get("metadata", {}).get("solids", [])
                names = [str(s.get("name", "")) for s in solids_meta]
                per_sample_names.append(names if any(names) else None)
            except Exception:
                per_sample_names.append(None)

        all_names_set = set()
        for names in per_sample_names:
            if names is None:
                continue
            for n in names:
                if n:
                    all_names_set.add(n)
        self.role_name_to_id = {n: i for i, n in enumerate(sorted(all_names_set))}
        self.role_id_to_name = {i: n for n, i in self.role_name_to_id.items()}
        n_with_names = sum(1 for n in per_sample_names if n is not None)
        if self.role_name_to_id:
            print(f"  [roletoken] {len(self.role_name_to_id)} unique role(s) "
                  f"from {n_with_names}/{len(self.npy_files)} samples with JSON names")
            for n, i in sorted(self.role_name_to_id.items(), key=lambda kv: kv[1]):
                print(f"    role_id={i:>2d}  {n!r}")
        else:
            print(f"  [roletoken] no JSON names — ROLE tokens not inserted (base behavior)")

        self.raw = []
        self.padded = []
        n_inserted_total = 0
        n_merge_total = 0
        n_samples_merged = 0
        n_samples_reordered = 0
        unknown_roles_seen = set()

        for i, (f, names) in enumerate(zip(self.npy_files, per_sample_names)):
            jf = self.json_files[i]
            t = None
            updated_names = names
            jd_current = None    # 변경 누적용

            if (jf is not None and os.path.exists(jf)
                    and (self.reorder_sketches_by_role or self.merge_same_role_chunks)
                    and names is not None):
                try:
                    import json as _json
                    with open(jf, "r", encoding="utf-8") as fh:
                        jd_current = _json.load(fh)

                    # ★ STEP 1: canonical role 순서로 Sketch+Extrude pair 재정렬 (v3 신규)
                    if self.reorder_sketches_by_role:
                        # 알 수 없는 role 추적 (canonical 에 없는 것)
                        for nm in updated_names:
                            if nm and nm not in self.canonical_role_order:
                                unknown_roles_seen.add(nm)
                        jd_current, updated_names, changed = (
                            reorder_sketches_by_canonical_role(
                                jd_current, updated_names,
                                canonical_order=self.canonical_role_order,
                                verbose=(i < 3),
                            )
                        )
                        if changed:
                            n_samples_reordered += 1

                    # ★ STEP 2: same-role coplanar+coextrude chunk 병합 (v2)
                    if self.merge_same_role_chunks:
                        jd_current, updated_names, n_merged = (
                            merge_coplanar_same_role_sketches(
                                jd_current, updated_names, verbose=(i < 3),
                            )
                        )
                        if n_merged > 0:
                            n_merge_total += n_merged
                            n_samples_merged += 1
                            if i < 3:
                                print(f"    sample {i}: merged {n_merged} chunk(s)")

                    # ★ STEP 3: 변경됐으면 재토큰화 + JSON 캐시 (시각화 일관성)
                    t = json_to_tokens(jd_current)
                    self.merged_json_cache[i] = jd_current
                except Exception as e:
                    print(f"    [warn] sample {i}: reorder/merge failed: {e}")
                    t = None
                    self.merged_json_cache[i] = None

            if t is None:
                t = np.load(f).astype(np.int32)

            t, n_inserted = self._insert_role_tokens(t, updated_names)
            n_inserted_total += n_inserted
            t = ensure_eos_when_truncated(t, max_len)
            self.raw.append(t)
            self.padded.append(self._pad(t))

        if self.role_name_to_id:
            print(f"  [roletoken] inserted {n_inserted_total} ROLE tokens "
                  f"across {len(self.npy_files)} samples")
        if self.reorder_sketches_by_role:
            print(f"  [roletoken] reordered sketches in {n_samples_reordered}/"
                  f"{len(self.npy_files)} samples "
                  f"(canonical: {list(self.canonical_role_order)})")
            if unknown_roles_seen:
                print(f"    ⚠ roles NOT in canonical order (placed at end): "
                      f"{sorted(unknown_roles_seen)}")
        if self.merge_same_role_chunks:
            print(f"  [roletoken] merged {n_merge_total} chunks across "
                  f"{n_samples_merged}/{len(self.npy_files)} samples "
                  f"(same role + coplanar + same extrude → multi-loop)")

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

    def _insert_role_tokens(self, tokens, names):
        """각 sketch chunk 의 첫 SOL 앞에 [ROLE, role_id, PAD, ..., PAD] 토큰 삽입.

        chunk 의 정의: SOL 로 시작하고 EXT 로 끝나는 구간.
        multi-loop chunk (SOL ... SOL ... EXT) 의 경우 첫 SOL 앞에만 1개 삽입.
        names 없거나 매핑 안 되는 chunk 는 ROLE 안 끼움 (base 와 동일).
        """
        if not self.role_name_to_id or not names:
            return tokens, 0
        out_rows = []
        chunk_idx = -1
        expecting_first_sol = True   # 시작점 또는 EXT 직후
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
        # ★ merge 가 적용된 sample 은 캐시된 병합 JSON 반환 (재토큰화와 일관)
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


class DeepCADBaselineAE(nn.Module):
    def __init__(
        self,
        max_len,
        d_model,
        d_param,
        nhead,
        n_enc,
        n_dec,
        d_ff,
        latent,
        mem_tokens,
        dropout,
        n_pool,
        n_freq_bands,
        aux_numeric,
        aux_hidden_mult,
    ):
        super().__init__()

        assert d_model % nhead == 0

        self.max_len = max_len
        self.d_model = d_model
        self.d_param = d_param
        self.latent = latent
        self.mem_tokens = mem_tokens
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

        self.pos_enc = SinPosEnc(
            d_model, max_len=max_len + 16, dropout=dropout,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_enc, norm=nn.LayerNorm(d_model),
        )

        if self.n_pool >= 2:
            self.pool_queries = nn.Parameter(torch.randn(self.n_pool, d_model) * 0.02)
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

        self.from_z = nn.Linear(latent, mem_tokens * d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_dec, norm=nn.LayerNorm(d_model),
        )

        self.bos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.cmd_head = nn.Linear(d_model, N_CMD)
        self.param_head = nn.Linear(d_model, N_ARGS * N_QUANT)

        if aux_numeric:
            aux_h = max(int(latent * aux_hidden_mult), 64)
            aux_out = max_len * N_ARGS
            self.aux_numeric_head = nn.Sequential(
                nn.Linear(latent, aux_h),
                nn.LayerNorm(aux_h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(aux_h, aux_out),
                nn.Sigmoid(),
            )
        else:
            self.aux_numeric_head = None

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

    def _memory_from_z(self, z):
        return self.from_z(z).view(
            z.size(0), self.mem_tokens, self.d_model,
        )

    def decode_teacher(self, z, teacher):
        B, L, _ = teacher.shape

        memory = self._memory_from_z(z)
        bos = self.bos.expand(B, 1, self.d_model)

        prev_emb, prev_pad = self._content_embed(teacher[:, :-1, :])
        tgt = self.pos_enc(torch.cat([bos, prev_emb], dim=1))

        tgt_pad = torch.cat([
            torch.zeros((B, 1), dtype=torch.bool, device=teacher.device),
            prev_pad,
        ], dim=1)

        out = self.decoder(
            tgt=tgt, memory=memory,
            tgt_mask=causal_mask(L, teacher.device),
            tgt_key_padding_mask=tgt_pad,
        )

        cmd_logits = self.cmd_head(out)
        prm_logits = self.param_head(out).view(B, L, N_ARGS, N_QUANT)

        return cmd_logits, prm_logits

    def aux_numeric_predict(self, z):
        if self.aux_numeric_head is None:
            return None
        B = z.size(0)
        out = self.aux_numeric_head(z)
        return out.view(B, self.max_len, N_ARGS)

    @torch.no_grad()
    def generate(self, z, max_gen_len=None):
        if max_gen_len is None:
            max_gen_len = self.max_len

        B = z.size(0)
        device = z.device

        memory = self._memory_from_z(z)
        bos = self.bos.expand(B, 1, self.d_model)

        steps = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_gen_len):
            if not steps:
                tc = bos
                tp = torch.zeros((B, 1), dtype=torch.bool, device=device)
            else:
                prev = torch.stack(steps, dim=1).float()
                pe, pp = self._content_embed(prev)
                tc = torch.cat([bos, pe], dim=1)
                tp = torch.cat([
                    torch.zeros((B, 1), dtype=torch.bool, device=device),
                    pp,
                ], dim=1)

            ti = self.pos_enc(tc)
            Lc = ti.size(1)

            out = self.decoder(
                tgt=ti, memory=memory,
                tgt_mask=causal_mask(Lc, device),
                tgt_key_padding_mask=tp,
            )

            last = out[:, -1, :]
            cmd_logits = self.cmd_head(last)
            prm_logits = self.param_head(last).view(B, N_ARGS, N_QUANT)

            next_cmd = torch.argmax(cmd_logits, dim=-1)
            next_prm = torch.argmax(prm_logits, dim=-1)

            next_tok = torch.full((B, 17), PAD_V, dtype=torch.long, device=device)
            next_tok[:, 0] = next_cmd

            for cid in range(N_CMD):
                n_valid = VALID_PAR.get(cid, 0)
                if n_valid <= 0:
                    continue
                mask = next_cmd == cid
                if mask.any():
                    next_tok[mask, 1:1 + n_valid] = next_prm[mask, :n_valid]

            if finished.any():
                next_tok[finished] = PAD_V
                next_tok[finished, 0] = EOS

            steps.append(next_tok)
            finished = finished | (next_cmd == EOS)

            if bool(finished.all()):
                break

        if not steps:
            return torch.zeros((B, 0, 17), dtype=torch.long, device=device)

        rt = torch.stack(steps, dim=1)
        rn = rt.detach().cpu().numpy().astype(np.int32)
        cleaned = []

        for i in range(rn.shape[0]):
            s = trim_after_eos(rn[i])
            pad_len = rn.shape[1] - s.shape[0]
            if pad_len > 0:
                s = np.concatenate([
                    s, np.full((pad_len, 17), PAD_V, dtype=np.int32),
                ], axis=0)
            cleaned.append(s)

        return torch.tensor(
            np.stack(cleaned, axis=0), dtype=torch.long, device=device,
        )

    def forward(self, x):
        z = self.encode(x)
        cmd_logits, prm_logits = self.decode_teacher(z, x)
        aux = self.aux_numeric_predict(z)
        return cmd_logits, prm_logits, aux, z


# ══════════════════════════════════════════════════════════════
# Common-curve residual surrogate
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# Epoch
# ══════════════════════════════════════════════════════════════
def run_epoch(ae, mlp, loader, optimizer, device, cfg, interp_w, train_mode=True):
    if train_mode:
        ae.train(); mlp.train()
    else:
        ae.eval(); mlp.eval()

    acc = {
        "total": 0.0, "ae": 0.0, "sp": 0.0,
        "rmse_db_full": 0.0, "rmse_db_sel": 0.0,
        "cmd": 0.0, "prm": 0.0, "aux": 0.0,
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
            cmd_logits, prm_logits, aux, z = ae(batch_tok)

            ae_loss, ae_comp = ae_loss_fn(
                cmd_logits, prm_logits, aux, batch_tok, cfg,
            )

            pred_sel_db, residual = mlp(z, return_parts=True)

            sp_loss, sp_comp, _pred_full_db = sparam_full_interp_loss(
                pred_sel_db=pred_sel_db,
                true_sel_db=batch_sel_db,
                true_full_db=batch_full_db,
                interp_w=interp_w, cfg=cfg,
            )

            total = ae_loss + cfg.w_sparam * sp_loss

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
                params = list(ae.parameters()) + list(mlp.parameters())
                gn = nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                optimizer.step()
                acc["grad_norm"] += float(gn)

        with torch.no_grad():
            acc["total"] += float(total.item())
            acc["ae"] += ae_comp["ae_total"]
            acc["sp"] += float(sp_loss.item())
            acc["rmse_db_full"] += sp_comp["rmse_db_full"]
            acc["rmse_db_sel"] += sp_comp["rmse_db_sel"]
            acc["cmd"] += ae_comp["cmd"]
            acc["prm"] += ae_comp["prm"]
            acc["aux"] += ae_comp["aux"]
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


# ══════════════════════════════════════════════════════════════
# Diagnostics and evaluation
# ══════════════════════════════════════════════════════════════
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
def diagnose_reconstruction(ae, dataset, indices, device, max_samples=8):
    ae.eval()
    sel = indices[:max_samples]

    cmd_correct = 0; cmd_total = 0
    struct_match = 0
    prm_abs = 0.0; prm_count = 0

    for idx in sel:
        x = torch.tensor(dataset.padded[idx], dtype=torch.float32).unsqueeze(0).to(device)
        z = ae.encode(x)
        gen = ae.generate(z, max_gen_len=dataset.max_len)
        recon = trim_after_eos(gen[0].cpu().numpy().astype(np.int32))
        orig = trim_after_eos(dataset.raw[idx])

        L = min(len(orig), len(recon))
        if L > 0:
            cmd_correct += int((orig[:L, 0] == recon[:L, 0]).sum())
            cmd_total += L

            op = orig[:L, 1:]
            rp = recon[:L, 1:]
            valid = (op >= 0) & (rp >= 0)
            if valid.any():
                prm_abs += float(
                    np.abs(op[valid].astype(np.int64) - rp[valid].astype(np.int64)).sum()
                )
                prm_count += int(valid.sum())

        if int((orig[:, 0] == EXT).sum()) == int((recon[:, 0] == EXT).sum()):
            struct_match += 1

    cmd_acc = cmd_correct / max(cmd_total, 1)
    prm_mae = prm_abs / max(prm_count, 1)

    print(f"  Recon diagnostic:")
    print(f"    samples evaluated    : {len(sel)}")
    print(f"    cmd accuracy aligned : {cmd_acc * 100:.2f}%")
    print(f"    struct match         : {struct_match}/{len(sel)}")
    print(f"    param MAE quant      : {prm_mae:.3f}")
    print(f"    param MAE normalized : {prm_mae / QMAX:.5f}")

    return {
        "cmd_acc": cmd_acc, "struct_match": struct_match, "prm_mae": prm_mae,
    }


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


def print_eval_diagnostics(eval_metrics, latent_diag, recon_diag, type_names):
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

    print(f"\n  Reconstruction diagnostic")
    print(f"    cmd_acc       : {recon_diag['cmd_acc'] * 100:.2f}%")
    print(f"    prm_MAE       : {recon_diag['prm_mae']:.3f}")
    print(f"    struct_match  : {recon_diag['struct_match']}")


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

    # ★ ROLE 토큰 chunk 당 1개씩 추가됨 — 최대 chunk 수 만큼 buffer 확보
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
        merge_same_role_chunks=getattr(cfg, "merge_same_role_chunks", True),
        reorder_sketches_by_role=getattr(cfg, "reorder_sketches_by_role", True),
        canonical_role_order=getattr(cfg, "canonical_role_order", (
            "frame",
            "S1_port", "S2_port", "S3_port",
            "S1_gnd",  "S2_gnd",  "S3_gnd",
            "camera",
        )),
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
# Train
# ══════════════════════════════════════════════════════════════
def train_fixed_medium_var(
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
        dataset.interp_matrix, dtype=torch.float32, device=device,
    )

    section("RUN START — fixed medium_var")

    print(f"  preset             : {cfg.preset}")
    print(f"  use_vicreg          : {cfg.use_vicreg}")
    print(f"  w_var / w_cov       : {cfg.w_var} / {cfg.w_cov}")
    print(f"  var_target          : {cfg.vicreg_var_target}")
    print(f"  epochs / bs         : {cfg.epochs} / {cfg.batch_size}")
    print(f"  d_model / latent    : {cfg.d_model} / {cfg.latent}")
    print(f"  selected_n_freq     : {cfg.n_freq}")
    print(f"  full_n_freq         : {cfg.raw_n_freq}")

    train_set = SubsetDataset(dataset, train_idx)
    val_set = SubsetDataset(dataset, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
    )

    ae = DeepCADBaselineAE(
        max_len=dataset.max_len,
        d_model=cfg.d_model,
        d_param=cfg.d_param,
        nhead=cfg.nhead,
        n_enc=cfg.n_enc,
        n_dec=cfg.n_dec,
        d_ff=cfg.d_ff,
        latent=cfg.latent,
        mem_tokens=cfg.mem_tokens,
        dropout=cfg.dropout,
        n_pool=cfg.n_pool,
        n_freq_bands=cfg.n_freq_bands,
        aux_numeric=cfg.aux_numeric,
        aux_hidden_mult=cfg.aux_hidden_mult,
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

    section("BUILD MODEL")
    print(f"  AE params           : {ae_params:,}")
    print(f"  Residual MLP params : {mlp_params:,}")
    print(f"  Total params        : {ae_params + mlp_params:,}")
    print(f"  Surrogate form      : pred = common_curve + residual(z)")
    print(f"  VICReg fixed        : w_var={cfg.w_var}, w_cov={cfg.w_cov}, target={cfg.vicreg_var_target}")
    print(f"  Aux numeric         : {cfg.aux_numeric}")
    print(f"  selected output dim : {cfg.n_freq * 3}")
    print(f"  full target dim     : {cfg.raw_n_freq * 3}")

    optimizer = torch.optim.AdamW(
        [
            {"params": list(ae.parameters()), "lr": cfg.lr_ae},
            {"params": list(mlp.parameters()), "lr": cfg.lr_mlp},
        ],
        weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
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

    best_metric = float("inf")
    best_ae_state = None
    best_mlp_state = None

    hist = {
        "tr_total": [], "va_total": [],
        "tr_full": [], "va_full": [],
        "tr_sel": [], "va_sel": [],
        "res_abs": [],
        "z_std": [], "var": [], "cov": [],
    }

    log_every = max(1, cfg.epochs // 8)

    section("JOINT TRAINING — AE + Common-Residual MLP")

    print(f"  epochs={cfg.epochs}, batch_size={cfg.batch_size}")
    print(f"  lr_ae={cfg.lr_ae:.2e}, lr_mlp={cfg.lr_mlp:.2e}, wd={cfg.weight_decay}")
    print(f"  loss = cmd({cfg.w_cmd}) + prm({cfg.w_prm}) + aux({cfg.w_aux})")
    print(f"       + sparam_full_interp({cfg.w_sparam})")
    print(f"       + var({cfg.w_var}) + cov({cfg.w_cov})")
    print(f"  db_loss_scale={cfg.db_loss_scale}")

    print(
        f"\n  {'ep':>4s} | "
        f"{'tr_tot':>8s} {'cmd':>6s} {'prm':>6s} {'aux':>7s} "
        f"{'tr_full':>8s} {'tr_sel':>8s} {'res':>7s} "
        f"{'var':>7s} {'cov':>7s} {'zstd':>6s} "
        f"{'va_tot':>8s} {'va_full':>8s} {'va_sel':>8s} {'va_res':>7s} | {'lr':>8s}"
    )
    print("  " + "-" * 154)

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(
            ae, mlp, train_loader, optimizer, device, cfg,
            interp_w=interp_w, train_mode=True,
        )
        va = run_epoch(
            ae, mlp, val_loader, optimizer, device, cfg,
            interp_w=interp_w, train_mode=False,
        )

        scheduler.step()
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
            best_ae_state = {
                k: v.detach().clone().cpu() for k, v in ae.state_dict().items()
            }
            best_mlp_state = {
                k: v.detach().clone().cpu() for k, v in mlp.state_dict().items()
            }

        if ep == 1 or ep % log_every == 0 or ep == cfg.epochs:
            print(
                f"  {ep:4d} | "
                f"{tr['total']:8.4f} {tr['cmd']:6.3f} {tr['prm']:6.3f} {tr['aux']:7.4f} "
                f"{tr['rmse_db_full']:8.3f} {tr['rmse_db_sel']:8.3f} {tr['res_abs']:7.3f} "
                f"{tr['var']:7.4f} {tr['cov']:7.4f} {tr['z_std']:6.3f} "
                f"{va['total']:8.4f} {va['rmse_db_full']:8.3f} {va['rmse_db_sel']:8.3f} {va['res_abs']:7.3f} | "
                f"{optimizer.param_groups[0]['lr']:8.2e}"
            )

    print(f"\n  Best full-grid interpolated val RMSE dB: {best_metric:.4f}")

    if best_ae_state is not None:
        ae.load_state_dict({k: v.to(device) for k, v in best_ae_state.items()})
    if best_mlp_state is not None:
        mlp.load_state_dict({k: v.to(device) for k, v in best_mlp_state.items()})

    section("EVALUATION")

    recon_diag = diagnose_reconstruction(
        ae, dataset,
        val_idx if len(val_idx) > 0 else train_idx,
        device,
        max_samples=min(8, len(val_idx) or len(train_idx)),
    )

    # ── ★ 구조 복원 시각화 (encoder 입력 vs decoder 복원) ──
    if cfg.show_figures:
        subsection("Reconstruction structure visualization")
        # type 별로 한 개씩 우선 뽑고, n_preview 더 많으면 나머지 채움
        pool_source = list(val_idx) if len(val_idx) > 0 else list(train_idx)
        n_show = max(1, int(cfg.n_preview))

        stratified = []
        seen_types = set()
        for idx in pool_source:
            try:
                t_id = int(dataset.type_ids[idx])
            except Exception:
                t_id = -1
            if t_id not in seen_types:
                stratified.append(idx)
                seen_types.add(t_id)

        # 모든 type 1 개씩은 무조건 보이고, n_show 가 더 크면 추가로 채움
        viz_pool = list(stratified)
        if n_show > len(viz_pool):
            for idx in pool_source:
                if idx not in viz_pool:
                    viz_pool.append(idx)
                    if len(viz_pool) >= n_show:
                        break
        max_to_show = max(n_show, len(stratified))

        print(
            f"  recon viz: showing {min(len(viz_pool), max_to_show)} samples "
            f"(stratified by type, n_types={len(seen_types)})"
        )

        try:
            visualize_reconstruction(
                ae, dataset, viz_pool, device, max_samples=max_to_show,
            )
        except Exception as e:
            import traceback as _tb
            print(f"  ⚠ visualize_reconstruction failed: {type(e).__name__}: {e}")
            _tb.print_exc()

    eval_metrics = evaluate_sparam_predictions(
        ae, mlp, dataset, val_idx_per_type, type_names, cfg, device,
    )

    latent_diag = diagnose_latent_simple(
        ae, dataset,
        val_idx if len(val_idx) > 0 else train_idx,
        device,
    )

    print_eval_diagnostics(eval_metrics, latent_diag, recon_diag, type_names)

    # ── training curves / latent analysis / sparam prediction figures ──
    if cfg.show_figures:
        subsection("Training curves")
        try:
            plot_training_curves(hist)
        except Exception as e:
            print(f"  ⚠ plot_training_curves failed: {type(e).__name__}: {e}")

        subsection("Latent space analysis")
        try:
            viz_idx = val_idx if len(val_idx) > 0 else train_idx
            z_all, tids = collect_latents(ae, dataset, viz_idx, device)
            analyze_latent_space(z_all, tids, type_names)
        except Exception as e:
            print(f"  ⚠ analyze_latent_space failed: {type(e).__name__}: {e}")

        subsection("S-param prediction figures")
        try:
            n_sp = min(3, len(val_idx)) if len(val_idx) > 0 else 0
            if n_sp > 0:
                visualize_sparam_predictions(
                    ae, mlp, dataset, val_idx, device,
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

    section("FINAL SUMMARY — fixed medium_var")

    print(f"  VICReg       : ON")
    print(f"  w_var        : {cfg.w_var}")
    print(f"  w_cov        : {cfg.w_cov}")
    print(f"  var_target   : {cfg.vicreg_var_target}")
    print(f"  best_val     : {result['best_val']:.4f} dB")
    print(f"  eval_full    : {result['eval_full']:.4f} dB")
    print(f"  eval_sel     : {result['eval_sel']:.4f} dB")
    print(f"  res_abs      : {result['res_abs']:.4f} dB")
    print(f"  z_std        : {result['z_std']:.4f}")
    print(f"  z_norm       : {result['z_norm']:.4f}")
    print(f"  dead dims    : {result['dead']} / {result['latent_dim']}")

    return ae, mlp, result


# ══════════════════════════════════════════════════════════════
# Inverse design (latent optimization via surrogate)
# ══════════════════════════════════════════════════════════════
def make_target_db_curve(
    freqs_full,
    channel_target_freqs,
    bandwidth_ghz,
    deep_db=-20.0,
    flat_db=0.0,
):
    """채널별 target 주파수에서 bandwidth 안만 deep_db, 외부는 flat_db."""
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


def inverse_design_optimize(
    ae,
    mlp,
    dataset,
    channel_target_freqs,
    bandwidth_ghz,
    device,
    n_starts=32,
    n_iters=2000,
    lr=5e-2,
    in_band_weight=10.0,
    out_band_weight=0.0,
    z_prior_weight=1e-3,
    z_prior_weight_end=1e-5,
    deep_db=-20.0,
    flat_db=0.0,
    seed=0,
    verbose_every=100,
    # Tier 1
    cosine_lr=True,
    lr_min_ratio=0.05,
    early_stop_patience=200,
    # Tier 2(a) — random restart on stagnation
    restart_patience=80,
    restart_frac=0.25,
    restart_noise=0.3,
    max_restarts=10,
):
    """Multi-start latent optimization with cosine LR + prior decay + random restart.

    Tier1:
      - cosine LR schedule (lr → lr * lr_min_ratio)
      - z_prior_weight 선형 감소 (초반엔 분포 안 유지, 후반엔 풀어줘서 fine search)
      - early stopping (early_stop_patience iter 정체)

    Tier2(a):
      - 정체(restart_patience iter 동안 best 개선 없음) 시 worst restart_frac
        만큼을 학습 latent 무작위 샘플 + noise 로 교체 + Adam moments 리셋
    """
    ae.eval()
    mlp.eval()

    all_indices = list(range(len(dataset)))
    z_prior, _ = collect_latents(ae, dataset, all_indices, device)
    z_prior_t = torch.tensor(z_prior, dtype=torch.float32, device=device)
    z_mean = z_prior_t.mean(dim=0)
    z_std = z_prior_t.std(dim=0).clamp(min=1e-3)

    target_full, masks = make_target_db_curve(
        dataset.freqs_full,
        channel_target_freqs,
        bandwidth_ghz,
        deep_db=deep_db,
        flat_db=flat_db,
    )
    target_full_t = torch.tensor(target_full, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks, dtype=torch.bool, device=device)

    interp_w = torch.tensor(
        dataset.interp_matrix, dtype=torch.float32, device=device,
    )

    rng = np.random.default_rng(seed)
    n_starts = min(int(n_starts), z_prior_t.size(0))
    init_idx = rng.choice(z_prior_t.size(0), size=n_starts, replace=False)
    z_init = z_prior_t[init_idx].clone()

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    best_z = None
    best_loss = float("inf")
    best_pred_full = None
    best_in_band = None
    best_start_idx = -1

    in_band_count = masks_t.float().sum(dim=0).clamp(min=1.0).unsqueeze(0)
    out_band_count = (~masks_t).float().sum(dim=0).clamp(min=1.0).unsqueeze(0)

    # ★ z trajectory 추적 (PCA 시각화 용)
    track_every = max(1, n_iters // 50)
    z_trajectory = [z.detach().cpu().numpy().copy()]
    track_iters = [0]

    last_improve_iter = 0
    last_restart_iter = -10 ** 9
    restart_count = 0
    iters_used = n_iters
    early_stopped = False

    print(
        f"  multi-start optimization | "
        f"n_starts={n_starts}, n_iters={n_iters}, lr={lr} "
        f"({'cosine→' + f'{lr * lr_min_ratio:.1e}' if cosine_lr else 'constant'})"
    )
    print(
        f"  loss weights: in_band={in_band_weight}, out_band={out_band_weight}"
    )
    print(
        f"  prior weight schedule: {z_prior_weight:.1e} → {z_prior_weight_end:.1e}"
    )
    print(
        f"  random restart: worst {int(restart_frac * 100)}% replaced after "
        f"{restart_patience} stagnant iters (max {max_restarts}x)"
    )
    print(
        f"  early stop: after {early_stop_patience} stagnant iters"
    )
    print(f"  target: deep_db={deep_db}, flat_db={flat_db}")

    for it in range(n_iters):
        # ── Tier1: schedules ──
        progress = it / max(n_iters - 1, 1)
        if cosine_lr:
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            cur_lr = lr * (lr_min_ratio + (1.0 - lr_min_ratio) * cos_factor)
        else:
            cur_lr = lr
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        cur_prior_w = (
            z_prior_weight
            + (z_prior_weight_end - z_prior_weight) * progress
        )

        optimizer.zero_grad()

        pred_sel = mlp(z)
        pred_full = interpolate_selected_to_full_torch(pred_sel, interp_w)

        diff = pred_full - target_full_t.unsqueeze(0)
        in_band_violation_sq = F.relu(diff) ** 2
        out_band_sq = diff ** 2

        in_band = (
            in_band_violation_sq * masks_t.unsqueeze(0).float()
        ).sum(dim=1) / in_band_count
        out_band = (
            out_band_sq * (~masks_t).unsqueeze(0).float()
        ).sum(dim=1) / out_band_count

        match_per = (
            in_band_weight * in_band + out_band_weight * out_band
        ).mean(dim=-1)

        z_reg_per = (((z - z_mean) / z_std) ** 2).mean(dim=-1)
        total_per = match_per + cur_prior_w * z_reg_per
        loss = total_per.sum()

        loss.backward()
        optimizer.step()

        # ★ z snapshot for PCA trajectory
        if (it + 1) % track_every == 0:
            z_trajectory.append(z.detach().cpu().numpy().copy())
            track_iters.append(it + 1)

        with torch.no_grad():
            bi = int(torch.argmin(match_per).item())
            cur = float(match_per[bi].item())
            if cur < best_loss - 1e-8:
                best_loss = cur
                best_z = z[bi:bi + 1].detach().clone()
                best_pred_full = pred_full[bi].detach().cpu().numpy()
                best_in_band = in_band[bi].detach().cpu().numpy()
                best_start_idx = bi
                last_improve_iter = it

        # ── Tier2(a): random restart on stagnation ──
        if (
            restart_count < max_restarts
            and it - last_improve_iter >= restart_patience
            and it - last_restart_iter >= restart_patience
        ):
            with torch.no_grad():
                n_replace = max(1, int(round(n_starts * restart_frac)))
                worst_idx = torch.topk(
                    match_per, k=n_replace, largest=True
                ).indices

                new_pick = rng.choice(
                    z_prior_t.size(0), size=n_replace, replace=False,
                )
                new_init = z_prior_t[new_pick].clone()
                if restart_noise > 0:
                    new_init = new_init + restart_noise * torch.randn_like(new_init)
                z.data[worst_idx] = new_init

                # Adam moments 리셋 (해당 슬롯만)
                state = optimizer.state.get(z, {})
                if "exp_avg" in state:
                    state["exp_avg"][worst_idx] = 0
                    state["exp_avg_sq"][worst_idx] = 0
                if "step" in state:
                    # tensor 형태 step (PyTorch >=1.7) 대비
                    pass

            restart_count += 1
            last_restart_iter = it
            last_improve_iter = it  # 새 init 에 patience 부여
            print(
                f"    iter {it + 1:4d}/{n_iters} | RESTART worst {n_replace}/{n_starts} "
                f"(restart #{restart_count})"
            )

        # ── Tier1: early stop ──
        if (
            it - last_improve_iter >= early_stop_patience
            and it - last_restart_iter >= early_stop_patience
        ):
            iters_used = it + 1
            early_stopped = True
            print(
                f"    iter {it + 1:4d}/{n_iters} | EARLY STOP "
                f"(no improvement for {early_stop_patience} iters)"
            )
            break

        # ── verbose logging ──
        if it == 0 or (it + 1) % verbose_every == 0 or it == n_iters - 1:
            with torch.no_grad():
                avg = float(match_per.mean().item())
                zr = float(z_reg_per.mean().item())
                print(
                    f"    iter {it + 1:4d}/{n_iters} | best={best_loss:.4f} | "
                    f"avg={avg:.4f} | z_reg={zr:.2f} | "
                    f"lr={cur_lr:.2e} | prior_w={cur_prior_w:.1e}"
                )

    print(
        f"\n  optimization stats: iters_used={iters_used}, "
        f"restarts={restart_count}, early_stopped={early_stopped}"
    )

    return {
        "best_z": best_z,
        "best_loss": best_loss,
        "best_pred_full": best_pred_full,
        "best_in_band_mse": best_in_band,
        "best_start_idx": best_start_idx,
        "target_full": target_full,
        "masks": masks,
        "freqs_full": np.asarray(dataset.freqs_full, dtype=np.float32),
        "channel_target_freqs": list(channel_target_freqs),
        "bandwidth_ghz": float(bandwidth_ghz),
        "deep_db": float(deep_db),
        "flat_db": float(flat_db),
        "iters_used": iters_used,
        "restart_count": restart_count,
        "early_stopped": early_stopped,
        "z_trajectory": z_trajectory,
        "track_iters": track_iters,
    }


def visualize_inverse_z_on_pca(
    z_train, type_ids_train, type_names,
    z_trajectory, track_iters, best_start_idx,
):
    """학습 z 의 PCA / t-SNE 위에 inverse design z 궤적 시각화.

    Args:
        z_train: (N, D) training latents
        type_ids_train: (N,) type IDs
        type_names: list of type names
        z_trajectory: list of (n_starts, D) numpy arrays, snapshot per iter
        track_iters: list of iteration indices for each snapshot
        best_start_idx: which start was best (highlighted)
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  ⚠ sklearn 없음 — inverse z trajectory viz skip")
        return
    if not z_trajectory:
        print("  ⚠ z_trajectory 비어있음 — skip")
        return

    z_train_np = np.asarray(z_train, dtype=np.float32)
    type_ids_np = np.asarray(type_ids_train, dtype=np.int64)
    N, D = z_train_np.shape

    # PCA fit on training z (analyze_latent_space 와 동일 축)
    pca = PCA(n_components=2).fit(z_train_np)
    z_train_pca = pca.transform(z_train_np)
    ev2 = pca.explained_variance_ratio_

    # Project trajectories
    n_snaps = len(z_trajectory)
    n_starts = z_trajectory[0].shape[0]
    # traj_pca shape: (n_snaps, n_starts, 2)
    traj_pca = np.stack(
        [pca.transform(z_snap) for z_snap in z_trajectory], axis=0,
    )

    # t-SNE: training z 로만 fit (parametric 아니라 inverse z project 못 함)
    z_tsne_train = None
    try:
        from sklearn.manifold import TSNE
        if N >= 10:
            perp = min(30, max(5, N // 4))
            z_tsne_train = TSNE(
                n_components=2, perplexity=perp,
                random_state=0, init="pca", learning_rate="auto",
            ).fit_transform(z_train_np)
    except Exception:
        z_tsne_train = None

    # 1개 figure, 1~2 panel (PCA + 옵션 t-SNE for reference only)
    if z_tsne_train is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")
    else:
        fig, ax_only = plt.subplots(1, 1, figsize=(10, 8), facecolor="white")
        axes = [ax_only]
    fig.suptitle(
        "Inverse design latent trajectory  —  "
        f"projected onto training PCA  ({n_starts} starts, {n_snaps} snapshots)",
        fontsize=11, color="#222",
    )

    colors_t = ["#4C9BE8", "#E05C5C", "#6DBF67", "#F4A340", "#A57CC1", "#4ECBCB"]
    n_types = len(type_names) if type_names is not None else int(type_ids_np.max()) + 1

    # ── Panel 1: PCA ──
    ax = axes[0]
    for ti in range(n_types):
        m = (type_ids_np == ti)
        if not m.any():
            continue
        tname = type_names[ti] if type_names and ti < len(type_names) else f"type{ti}"
        ax.scatter(
            z_train_pca[m, 0], z_train_pca[m, 1],
            c=colors_t[ti % len(colors_t)], s=15, alpha=0.35,
            label=f"train {tname}", edgecolor="none",
        )

    # 각 multi-start trajectory 표시
    for k in range(n_starts):
        xs = traj_pca[:, k, 0]
        ys = traj_pca[:, k, 1]
        is_best = (k == best_start_idx)
        if is_best:
            ax.plot(xs, ys, color="#FF1744", lw=1.8, alpha=0.95, zorder=10)
            ax.scatter(xs[0], ys[0], marker="o", s=60, color="#FF1744",
                       edgecolor="black", linewidths=0.8, zorder=11,
                       label="best start")
            ax.scatter(xs[-1], ys[-1], marker="*", s=300, color="#FF1744",
                       edgecolor="black", linewidths=1.2, zorder=12,
                       label="best end")
        else:
            ax.plot(xs, ys, color="#666666", lw=0.5, alpha=0.4, zorder=5)
            ax.scatter(xs[0], ys[0], marker="o", s=15, color="#666666",
                       alpha=0.5, zorder=6)
            ax.scatter(xs[-1], ys[-1], marker="X", s=25, color="#333333",
                       alpha=0.7, zorder=7)
    ax.set_title(f"PCA (var={ev2[0]:.2f}+{ev2[1]:.2f})", fontweight="normal")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)

    # ── Panel 2: t-SNE (reference only) ──
    if z_tsne_train is not None:
        ax = axes[1]
        for ti in range(n_types):
            m = (type_ids_np == ti)
            if not m.any():
                continue
            tname = type_names[ti] if type_names and ti < len(type_names) else f"type{ti}"
            ax.scatter(
                z_tsne_train[m, 0], z_tsne_train[m, 1],
                c=colors_t[ti % len(colors_t)], s=15, alpha=0.5,
                label=tname, edgecolor="none",
            )
        # inverse z 의 NN training sample 찾아서 그 좌표에 마커
        z_best_final = z_trajectory[-1][best_start_idx]
        dists = np.linalg.norm(z_train_np - z_best_final[None, :], axis=1)
        nn_idx = int(np.argmin(dists))
        ax.scatter(
            z_tsne_train[nn_idx, 0], z_tsne_train[nn_idx, 1],
            marker="*", s=300, color="#FF1744",
            edgecolor="black", linewidths=1.2, zorder=15,
            label=f"NN (idx={nn_idx})",
        )
        ax.annotate(
            "best z NN",
            xy=(z_tsne_train[nn_idx, 0], z_tsne_train[nn_idx, 1]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=10, color="#FF1744",
            arrowprops=dict(arrowstyle="->", color="#FF1744", lw=1),
        )
        ax.set_title("t-SNE  (best z 의 NN training sample 표시)", fontweight="normal")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="best", framealpha=0.85)

    plt.tight_layout()
    print(
        f"  ✓ inverse z trajectory figure 생성 "
        f"(n_starts={n_starts}, n_snaps={n_snaps})"
    )


def visualize_inverse_design_curve(result):
    freqs = result["freqs_full"]
    pred = result["best_pred_full"]
    target_freqs = result["channel_target_freqs"]
    bw = result["bandwidth_ghz"]
    deep_db = result["deep_db"]

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.5), facecolor="white", sharey=False,
    )
    fig.suptitle(
        f"Inverse design — surrogate prediction vs spec (≤ {deep_db:.0f} dB in band)",
        fontsize=11, color="#222", y=1.0,
    )

    pred_min = float(pred.min())
    pred_max = float(pred.max())
    ymin = min(pred_min - 3.0, deep_db - 5.0)
    ymax = max(pred_max + 2.0, 3.0)

    band_color = "#3F6E5C"
    spec_color = "#2E4172"
    pred_color = "#E07B5B"
    ref_color = "#888888"
    ref_db = -10.0

    for c, lbl in enumerate(RETURN_LABELS):
        ax = axes[c]
        f0 = target_freqs[c]
        f_lo, f_hi = f0 - bw / 2.0, f0 + bw / 2.0

        # -10 dB reference (전 주파수 점선)
        ax.axhline(
            ref_db, color=ref_color, ls=":", lw=0.8, alpha=0.7,
            label=f"{ref_db:.0f} dB ref",
        )

        # surrogate 예측 (얇게)
        ax.plot(
            freqs, pred[:, c],
            color=pred_color, lw=1.0, alpha=0.9,
            label="surrogate pred",
        )

        # spec 한계선: in-band 안에서만 ≤ deep_db
        ax.hlines(
            deep_db, f_lo, f_hi,
            colors=spec_color, linestyles="-", lw=1.0,
            label=f"spec ≤ {deep_db:.0f} dB",
        )

        # band 경계: 얇은 dashed 수직선만
        for fx in (f_lo, f_hi):
            ax.axvline(fx, color=band_color, ls="--", lw=0.6, alpha=0.55)

        # 위쪽 brackets + bw label
        bracket_y = ymax - (ymax - ymin) * 0.04
        ax.annotate(
            "",
            xy=(f_lo, bracket_y), xytext=(f_hi, bracket_y),
            arrowprops=dict(
                arrowstyle="|-|, widthA=0.3, widthB=0.3",
                color=band_color, lw=0.7, shrinkA=0, shrinkB=0,
            ),
        )
        ax.text(
            f0, bracket_y - (ymax - ymin) * 0.025,
            f"{bw * 1000:.0f} MHz",
            ha="center", va="top",
            fontsize=8, color=band_color,
        )

        # in-band worst
        band_mask = (freqs >= f_lo) & (freqs <= f_hi)
        if band_mask.any():
            worst = float(pred[band_mask, c].max())
            margin = deep_db - worst  # >0 이면 spec 만족
            ok = "✓" if margin >= 0 else "✗"
            title = (
                f"{lbl}  f={f0:.2f} GHz  "
                f"worst={worst:.1f} dB  margin={margin:+.1f}  {ok}"
            )
        else:
            title = f"{lbl}  f={f0:.2f} GHz"

        ax.set_title(title, fontsize=9, fontweight="normal")
        ax.set_xlabel("frequency [GHz]")
        ax.set_ylabel("|S| [dB]")
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(True, alpha=0.15, lw=0.4)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

    plt.tight_layout()
    print(f"  ✓ inverse design curve figure 생성")


def extract_decoded_dimensions(tokens, dq_meta):
    """Decoded tokens → 실좌표(mm) 단위의 dimension list.
    - tokens_to_real_sketches() 와 달리 ARC/CIRCLE 을 polyline 화 하지 않고
      각 곡선의 진짜 파라미터(length / radius / 직경 / extrude height) 를 보존."""
    if dq_meta is None:
        return []

    sk_metas = dq_meta["sketches"]
    max_ext = dq_meta["max_ext"]

    dims = []
    cur_2d = (0.0, 0.0)
    loop_first_2d = None
    sketch_idx = 0

    for row in tokens:
        cmd = int(row[0])
        p = row[1:]

        if sketch_idx >= len(sk_metas):
            if cmd == EOS:
                break
            if cmd == EXT:
                sketch_idx += 1
            continue

        sm = sk_metas[sketch_idx]

        if cmd == SOL:
            cur_2d = (0.0, 0.0)
            loop_first_2d = None
        elif cmd == LINE:
            if p[0] < 0 or p[1] < 0:
                continue
            ex = _dequant(p[0], sm["xy_min"][0], sm["xy_max"][0])
            ey = _dequant(p[1], sm["xy_min"][1], sm["xy_max"][1])
            sx, sy = cur_2d
            if loop_first_2d is None:
                loop_first_2d = (sx, sy)
            p_s_3d = sm["origin"] + sx * sm["x_axis"] + sy * sm["y_axis"]
            p_e_3d = sm["origin"] + ex * sm["x_axis"] + ey * sm["y_axis"]
            length = float(np.linalg.norm(p_e_3d - p_s_3d))
            dims.append({
                "kind": "line",
                "sketch_idx": sketch_idx,
                "p_start_3d": p_s_3d,
                "p_end_3d": p_e_3d,
                "length": length,
            })
            cur_2d = (ex, ey)
        elif cmd == ARC:
            if any(p[i] < 0 for i in range(4)):
                continue
            mx = _dequant(p[0], sm["xy_min"][0], sm["xy_max"][0])
            my = _dequant(p[1], sm["xy_min"][1], sm["xy_max"][1])
            ex = _dequant(p[2], sm["xy_min"][0], sm["xy_max"][0])
            ey = _dequant(p[3], sm["xy_min"][1], sm["xy_max"][1])
            sx, sy = cur_2d

            mid_3d = sm["origin"] + mx * sm["x_axis"] + my * sm["y_axis"]
            p_s_3d = sm["origin"] + sx * sm["x_axis"] + sy * sm["y_axis"]
            p_e_3d = sm["origin"] + ex * sm["x_axis"] + ey * sm["y_axis"]

            # 원의 중심·반지름 (2D 평면 안에서 계산)
            D = 2 * (sx * (my - ey) + mx * (ey - sy) + ex * (sy - my))
            if abs(D) < 1e-9:
                # 거의 직선 — line 으로 취급
                length = float(np.linalg.norm(p_e_3d - p_s_3d))
                dims.append({
                    "kind": "line",
                    "sketch_idx": sketch_idx,
                    "p_start_3d": p_s_3d,
                    "p_end_3d": p_e_3d,
                    "length": length,
                })
            else:
                ux = ((sx ** 2 + sy ** 2) * (my - ey)
                      + (mx ** 2 + my ** 2) * (ey - sy)
                      + (ex ** 2 + ey ** 2) * (sy - my)) / D
                uy = ((sx ** 2 + sy ** 2) * (ex - mx)
                      + (mx ** 2 + my ** 2) * (sx - ex)
                      + (ex ** 2 + ey ** 2) * (mx - sx)) / D
                r = math.sqrt((sx - ux) ** 2 + (sy - uy) ** 2)
                # arc sweep angle (대략)
                a1 = math.atan2(sy - uy, sx - ux)
                a2 = math.atan2(ey - uy, ex - ux)
                am = math.atan2(my - uy, mx - ux)
                # use _arc_pts logic to get correct sweep
                def _fix(a, ref):
                    while a - ref > math.pi:
                        a -= 2 * math.pi
                    while a - ref < -math.pi:
                        a += 2 * math.pi
                    return a
                am_f = _fix(am, a1)
                a2_f = _fix(a2, a1)
                if not (min(a1, a2_f) <= am_f <= max(a1, a2_f)):
                    a2_f = a2_f - 2 * math.pi if a2_f > a1 else a2_f + 2 * math.pi
                sweep = abs(a2_f - a1)
                arc_len = r * sweep
                dims.append({
                    "kind": "arc",
                    "sketch_idx": sketch_idx,
                    "p_start_3d": p_s_3d,
                    "p_mid_3d": mid_3d,
                    "p_end_3d": p_e_3d,
                    "radius": float(r),
                    "sweep_deg": float(math.degrees(sweep)),
                    "arc_length": float(arc_len),
                })
            cur_2d = (ex, ey)
        elif cmd == CIRCLE:
            if any(p[i] < 0 for i in range(3)):
                continue
            cx = _dequant(p[0], sm["xy_min"][0], sm["xy_max"][0])
            cy = _dequant(p[1], sm["xy_min"][1], sm["xy_max"][1])
            r = (p[2] / QMAX) * sm["sk_scale"]
            c_3d = sm["origin"] + cx * sm["x_axis"] + cy * sm["y_axis"]
            dims.append({
                "kind": "circle",
                "sketch_idx": sketch_idx,
                "center_3d": c_3d,
                "radius": float(r),
                "diameter": float(2 * r),
            })
        elif cmd == EXT:
            e1_val = _dequant(p[P_E1], 0, max_ext)
            e2_val = (
                _dequant(p[P_E2], 0, max_ext)
                if int(p[P_E2]) >= 0 else 0.0
            )
            z_raw = sm["z_axis"]
            normal = z_raw / (np.linalg.norm(z_raw) + 1e-12)
            ref_3d = sm["origin"]
            top_3d = ref_3d + normal * e1_val
            dims.append({
                "kind": "extrude",
                "sketch_idx": sketch_idx,
                "p_start_3d": ref_3d,
                "p_end_3d": top_3d,
                "label_pos_3d": (ref_3d + top_3d) / 2.0,
                "length": float(e1_val),
                "extent_one": float(e1_val),
                "extent_two": float(e2_val),
            })
            sketch_idx += 1
            cur_2d = (0.0, 0.0)
            loop_first_2d = None
        elif cmd == EOS:
            break

    return dims


def _format_dim_label(d):
    k = d["kind"]
    if k == "line":
        return f"{d['length']:.1f}"
    if k == "circle":
        return f"Ø{d['diameter']:.1f}"
    if k == "arc":
        return f"R{d['radius']:.1f}"
    if k == "extrude":
        return f"h={d['length']:.1f}"
    return ""


def _annotate_dimensions(ax, dims):
    """3D axes 에 각 dimension 라벨을 표시."""
    if not dims:
        return
    label_kwargs = dict(
        fontsize=6.5,
        color="#111",
        ha="center", va="center",
        zorder=1000,
    )
    bbox_kw = dict(
        boxstyle="round,pad=0.15",
        fc=(1, 1, 1, 0.78),
        ec=(0.7, 0.7, 0.7, 0.6),
        lw=0.3,
    )

    for d in dims:
        k = d["kind"]
        txt = _format_dim_label(d)
        if not txt:
            continue
        if k == "line":
            pos = (d["p_start_3d"] + d["p_end_3d"]) / 2.0
        elif k == "circle":
            pos = d["center_3d"]
        elif k == "arc":
            pos = d["p_mid_3d"]
        elif k == "extrude":
            pos = d["label_pos_3d"]
        else:
            continue
        ax.text(
            float(pos[0]), float(pos[1]), float(pos[2]),
            txt, bbox=bbox_kw, **label_kwargs,
        )


def _print_decoded_dimensions(dims):
    if not dims:
        print("  (no dimensions extracted)")
        return
    # sketch 별로 묶어서 출력
    by_sk = {}
    for d in dims:
        by_sk.setdefault(d["sketch_idx"], []).append(d)
    print(f"\n  Decoded dimensions ({len(dims)} entries, mm):")
    for sk_i in sorted(by_sk.keys()):
        print(f"    [sketch {sk_i + 1}]")
        for d in by_sk[sk_i]:
            k = d["kind"]
            if k == "line":
                print(f"      LINE     length = {d['length']:7.3f} mm")
            elif k == "circle":
                print(f"      CIRCLE   Ø = {d['diameter']:7.3f} mm   (R={d['radius']:.3f})")
            elif k == "arc":
                print(
                    f"      ARC      R = {d['radius']:7.3f} mm   "
                    f"sweep={d['sweep_deg']:6.1f}°   arc_len={d['arc_length']:.3f}"
                )
            elif k == "extrude":
                print(
                    f"      EXTRUDE  height = {d['extent_one']:7.3f} mm   "
                    f"(ext2={d['extent_two']:.3f})"
                )


@torch.no_grad()
def _find_nn_dequant_meta(dataset, ae, z_target, device):
    """latent 공간에서 z_target 에 가장 가까운 training sample 의
    JSON 으로 dequant_meta 를 만든다. 실패하면 (None, -1, inf)."""
    all_indices = list(range(len(dataset)))
    z_all, _ = collect_latents(ae, dataset, all_indices, device)

    z_t = z_target.detach().cpu().numpy()
    if z_t.ndim == 2:
        z_t = z_t[0]

    dists = np.linalg.norm(z_all - z_t[None, :], axis=1)
    order = np.argsort(dists)

    for cand in order:
        ci = int(cand)
        jd = dataset.load_json(ci)
        if jd is None:
            continue
        try:
            dq = extract_dequant_meta(jd)
            if dq["sketches"]:
                return dq, ci, float(dists[ci])
        except Exception:
            continue

    return None, -1, float("inf")


def _extend_dequant_meta(dq_meta, n_needed):
    """sketch meta 가 부족하면 마지막 항목 복제로 확장."""
    if dq_meta is None:
        return None
    sk = list(dq_meta["sketches"])
    if not sk:
        return dq_meta
    while len(sk) < n_needed:
        sk.append(dict(sk[-1]))
    dq_meta = dict(dq_meta)
    dq_meta["sketches"] = sk
    return dq_meta


@torch.no_grad()
def visualize_decoded_structure(
    ae, z, dataset, device, title="Decoded structure",
    separate_windows=False,
):
    """latent z → AE.generate() → 앞쪽 reconstruction viz 와 동일한 스타일로 표시.
    NN training sample 의 JSON 을 빌려서 real-coord(mm) 모드로 렌더.

    separate_windows=True 면 4 개 view 를 각각 다른 figure (큰 창) 로 띄움.
    확대해서 자세히 보고 싶을 때 유용.
    """
    ae.eval()

    gen = ae.generate(z, max_gen_len=dataset.max_len)
    recon = gen[0].detach().cpu().numpy().astype(np.int32)
    recon_trim = trim_after_eos(recon)

    # nearest-neighbor 의 JSON 으로 실좌표 dequant meta 확보
    dq_meta, nn_idx, nn_dist = _find_nn_dequant_meta(dataset, ae, z, device)
    n_ext = int((recon_trim[:, 0] == EXT).sum())

    if dq_meta is not None and n_ext > 0:
        dq_meta = _extend_dequant_meta(dq_meta, max(n_ext, len(dq_meta["sketches"])))
        sketches = tokens_to_real_sketches(recon_trim, dq_meta)
        use_real = True
        coord_label = "real coord (mm)"
        try:
            nn_name = os.path.basename(dataset.npy_files[nn_idx])
            t_id = int(dataset.type_ids[nn_idx])
            t_name = dataset.type_names[t_id]
            meta_note = f"NN ref: [{t_name}] {nn_name}  (z-dist={nn_dist:.2f})"
        except Exception:
            meta_note = f"NN ref idx={nn_idx}  (z-dist={nn_dist:.2f})"
    else:
        sketches = tokens_to_sketches_normalized(recon_trim)
        use_real = False
        coord_label = "normalized"
        meta_note = "no JSON metadata available — fallback to normalized"

    if use_real:
        nl_total = sum(len(s["loops_3d"]) for s in sketches)
        dims = extract_decoded_dimensions(recon_trim, dq_meta)
        _print_decoded_dimensions(dims)
    else:
        nl_total = sum(len(s["loops_2d"]) for s in sketches)
        dims = []
        print("  (dimensions not annotated — normalized mode, no real-coord meta)")

    # ★ 각 sketch 중심점 (한 번만 계산해 모든 view 에서 재사용)
    centers = _compute_sketch_centers(sketches, use_real)
    _print_sketch_centers(centers, use_real)

    views = [
        ("Decoded — 3D View",        28,   -55),
        ("Decoded — 3D View (alt)",  20,    45),
        ("Decoded — Top View",       89.9, -90.0),
        ("Decoded — Side View",       5,   -90),
    ]

    ne = len(sketches)
    nl = (
        sum(len(s["loops_3d"]) for s in sketches) if use_real
        else sum(len(s["loops_2d"]) for s in sketches)
    )
    info_line = (
        f"{recon_trim.shape[0]} tokens · {ne} extrude · {nl_total} loop"
    )

    def _draw_one(ax, label, elev, azim, annotate=True):
        if not sketches:
            ax.text2D(
                0.5, 0.5, "(empty)",
                color="#888", ha="center", va="center",
                fontsize=11, transform=ax.transAxes,
            )
            _style(ax, label)
            ax.view_init(elev=elev, azim=azim)
            return
        _set_axes_and_render(ax, sketches, use_real=use_real)
        if annotate:
            if use_real and dims:
                _annotate_dimensions(ax, dims)
            # ★ 중심점 마커 + 좌표 라벨
            _annotate_centers(ax, centers, with_coords=use_real, with_label=True)
        _style(
            ax,
            f"{label}\n{ne} extrude · {nl} loop · {recon_trim.shape[0]} token",
        )
        ax.view_init(elev=elev, azim=azim)

    if separate_windows:
        # ── (옵션) 4 개 view 를 각각 별도 figure 로 ──
        for label, elev, azim in views:
            fig = plt.figure(figsize=(10, 9), facecolor="white")
            fig.suptitle(
                f"{title} [{coord_label}]\n{info_line}\n{meta_note}",
                fontsize=10, fontweight="normal", color="#222", y=0.985,
            )
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            _draw_one(ax, label, elev, azim)
            if sketches:
                fig.legend(
                    handles=[
                        mpatches.Patch(color=PAL[i % len(PAL)], label=f"Sketch {i + 1}")
                        for i in range(len(sketches))
                    ],
                    loc="lower center", ncol=min(len(sketches), 6),
                    facecolor="white", edgecolor="#ddd", labelcolor="#444",
                    fontsize=9, framealpha=0.95,
                )
            plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        n_figs_created = 4
    else:
        # ── 기본: Top View 2 창 (숫자 포함 / 구조만) ──
        top_label, top_elev, top_azim = "Decoded — Top View", 89.9, -90.0

        def _make_top_fig(annotate, suffix):
            fig_top = plt.figure(figsize=(12, 11), facecolor="white")
            fig_top.suptitle(
                f"{title}  —  TOP VIEW {suffix}[{coord_label}]\n{info_line}\n{meta_note}",
                fontsize=11, fontweight="normal", color="#222", y=0.985,
            )
            ax_top = fig_top.add_subplot(1, 1, 1, projection="3d")
            _draw_one(ax_top, top_label, top_elev, top_azim, annotate=annotate)
            if sketches:
                fig_top.legend(
                    handles=[
                        mpatches.Patch(color=PAL[i % len(PAL)], label=f"Sketch {i + 1}")
                        for i in range(len(sketches))
                    ],
                    loc="lower center", ncol=min(len(sketches), 6),
                    facecolor="white", edgecolor="#ddd", labelcolor="#444",
                    fontsize=10, framealpha=0.95,
                )
            plt.tight_layout(rect=[0, 0.05, 1, 0.94])

        # (1) 숫자/중심점 포함 — dimension 확인용
        _make_top_fig(annotate=True, suffix="(annotated) ")
        # (2) 구조만 — 깔끔한 형상 확인용
        _make_top_fig(annotate=False, suffix="(clean) ")
        n_figs_created = 2
    print(
        f"  ✓ decoded structure figure 생성 "
        f"(tokens={recon_trim.shape[0]}, mode={'real' if use_real else 'normalized'}, "
        f"nn_idx={nn_idx}, nn_dist={nn_dist:.2f}, figs={n_figs_created})"
    )

    return recon_trim, sketches, (dq_meta if use_real else None)


def run_inverse_design_pipeline(
    ae,
    mlp,
    dataset,
    device,
    channel_target_freqs=(2.0, 3.0, 4.0),
    bandwidth_ghz=0.1,
    n_starts=32,
    n_iters=2000,
    lr=5e-2,
    in_band_weight=10.0,
    out_band_weight=0.0,
    z_prior_weight=1e-3,
    z_prior_weight_end=1e-5,
    deep_db=-20.0,
    seed=0,
    cosine_lr=True,
    early_stop_patience=200,
    restart_patience=80,
    restart_frac=0.25,
    restart_noise=0.3,
    max_restarts=10,
    separate_windows=False,
):
    section("INVERSE DESIGN — latent search + decode")

    print(f"  Targets")
    for c, lbl in enumerate(RETURN_LABELS):
        f0 = channel_target_freqs[c]
        print(
            f"    {lbl}: f={f0:.2f} GHz, band=±{bandwidth_ghz / 2 * 1000:.0f} MHz, "
            f"deep_db={deep_db}"
        )

    result = inverse_design_optimize(
        ae=ae,
        mlp=mlp,
        dataset=dataset,
        channel_target_freqs=channel_target_freqs,
        bandwidth_ghz=bandwidth_ghz,
        device=device,
        n_starts=n_starts,
        n_iters=n_iters,
        lr=lr,
        in_band_weight=in_band_weight,
        out_band_weight=out_band_weight,
        z_prior_weight=z_prior_weight,
        z_prior_weight_end=z_prior_weight_end,
        deep_db=deep_db,
        cosine_lr=cosine_lr,
        early_stop_patience=early_stop_patience,
        restart_patience=restart_patience,
        restart_frac=restart_frac,
        restart_noise=restart_noise,
        max_restarts=max_restarts,
        seed=seed,
    )

    print(f"\n  Optimization result")
    print(f"    best match loss  : {result['best_loss']:.4f}")
    print(f"    best start index : {result['best_start_idx']}")
    print(f"    in-band MSE      : "
          f"S11={result['best_in_band_mse'][0]:.3f}  "
          f"S22={result['best_in_band_mse'][1]:.3f}  "
          f"S33={result['best_in_band_mse'][2]:.3f}")

    subsection("Inverse design — S-param curve figure")
    try:
        visualize_inverse_design_curve(result)
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ visualize_inverse_design_curve failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    # ★ Inverse z trajectory on training PCA (latent space 위치 추적)
    subsection("Inverse design — z trajectory on training PCA")
    try:
        all_indices = list(range(len(dataset)))
        z_train_all, tids_train_all = collect_latents(ae, dataset, all_indices, device)
        type_names_local = list(dataset.type_names) if hasattr(dataset, "type_names") else None
        visualize_inverse_z_on_pca(
            z_train=z_train_all,
            type_ids_train=tids_train_all,
            type_names=type_names_local,
            z_trajectory=result.get("z_trajectory", []),
            track_iters=result.get("track_iters", []),
            best_start_idx=int(result.get("best_start_idx", 0)),
        )
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ visualize_inverse_z_on_pca failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    subsection("Inverse design — decoded structure figure")
    try:
        recon_trim, sketches, _ = visualize_decoded_structure(
            ae, result["best_z"], dataset, device,
            title="Inverse-designed structure (decoded from optimal z)",
            separate_windows=separate_windows,
        )
        result["decoded_tokens"] = recon_trim
        result["decoded_n_sketches"] = len(sketches)
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ visualize_decoded_structure failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    return result


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    from datetime import datetime as _dt
    run_tag = _dt.now().strftime("%Y%m%d_%H%M%S")

    t0 = time.time()

    # ---------------------------------------------------------
    # 실행 설정
    # ---------------------------------------------------------
    PRESET = "tiny"   # tiny / small / full / custom
    LOG_VERBOSITY = "simple"

    USE_TYPES = [1, 2, 3]

    RAW_N_FREQ = 401
    SELECTED_N_FREQ = 81
    FREQ_SELECT_MODE = "linspace"

    # ★ SAMPLES_PER_TYPE는 PRESET="custom"일 때만 반영됨.
    #   tiny/small/full preset 사용 시 preset이 정의한 n_samples_each가 우선 적용된다:
    #     tiny  → 300/type
    #     small → 800/type
    #     full  → 0(전체)/type
    #   per-type 다른 개수가 필요하면 PRESET="custom" 으로 두고 아래 값 조정.
    SAMPLES_PER_TYPE = (800, 800, 800)

    SHOW_FIGURES = True

    # ★ Inverse design 켜고/끄기.
    #   True  : 학습 + 평가 후 latent 최적화 → 디코딩 → spec 매칭 figure 까지 표시
    #   False : 학습 + 평가만 (AE 단독 모드. 기존 AE.py 와 동등)
    RUN_INVERSE_DESIGN = True

    # ★ 학습 끝나고 띄울 구조 복원 figure 개수 (sample 1개당 창 1개)
    N_PREVIEW = 2

    # ---------------------------------------------------------
    # fixed medium_var VICReg
    # ---------------------------------------------------------
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

    if len(SAMPLES_PER_TYPE) != len(USE_TYPES):
        raise ValueError(
            f"SAMPLES_PER_TYPE length mismatch: "
            f"{len(SAMPLES_PER_TYPE)} vs USE_TYPES {len(USE_TYPES)}"
        )

    print(f"[CONFIG] USE_TYPES={USE_TYPES} → {type_names}")
    print(f"[CONFIG] preset={PRESET}")
    print(f"[CONFIG] raw_n_freq={RAW_N_FREQ}, selected_n_freq={SELECTED_N_FREQ}")
    print(f"[CONFIG] samples_per_type={SAMPLES_PER_TYPE}")
    print(f"[CONFIG] surrogate = common_curve + residual(z)")
    print(f"[CONFIG] loss basis = selected output → interpolated full-grid")
    print(f"[CONFIG] fixed VICReg medium_var")
    print(f"         use_vicreg={USE_VICREG}")
    print(f"         w_var={VICREG_W_VAR}")
    print(f"         w_cov={VICREG_W_COV}")
    print(f"         var_target={VICREG_VAR_TARGET}")
    print(f"[CONFIG] n_preview (recon viz windows)={N_PREVIEW}")

    cfg = CFG(
        preset=PRESET,
        seed=7,
        run_name="medium_var_fixed",
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
        clip_db_min=-50.0,
        clip_db_max=5.0,

        lr_ae=3e-4,
        lr_mlp=1e-3,
        weight_decay=1e-3,
        grad_clip=1.0,
        val_ratio=0.15,

        w_cmd=1.0,
        w_prm=1.0,
        w_aux=2.0,
        w_sparam=5.0,

        db_loss_scale=20.0,

        use_vicreg=USE_VICREG,
        w_var=VICREG_W_VAR,
        w_cov=VICREG_W_COV,
        vicreg_var_target=VICREG_VAR_TARGET,

        aux_numeric=True,
        aux_hidden_mult=2.0,

        mlp_hidden_mult=2.0,
        mlp_dropout=0.3,

        residual_scale=1.0,
        zero_init_residual=True,

        n_preview=N_PREVIEW,
    )

    try:
        apply_preset(cfg)

        # preset 적용 후에도 fixed medium_var 값 재고정
        cfg.use_vicreg = USE_VICREG
        cfg.w_var = VICREG_W_VAR
        cfg.w_cov = VICREG_W_COV
        cfg.vicreg_var_target = VICREG_VAR_TARGET
        cfg.n_preview = N_PREVIEW

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        section("DeepCAD AE + Return-3 Common-Curve Residual Surrogate — fixed medium_var")
        print(f"  Python : {_sys.version.split()[0]}")
        print(f"  Torch  : {torch.__version__}")
        print(f"  Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
        print(f"  Preset : {cfg.preset}")
        print(f"  Backend: {_MPL_BACKEND}")

        section("LOAD DATA")
        dataset, npy_files, type_ids, type_names, sparam_db, max_len = load_multitype_data(
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
            dataset=dataset,
            train_idx=train_idx,
            val_idx_per_type=val_idx_per_type,
            type_names=type_names,
        )

        ae, mlp, result = train_fixed_medium_var(
            cfg=cfg,
            dataset=dataset,
            type_names=type_names,
            train_idx=train_idx,
            val_idx=val_idx,
            val_idx_per_type=val_idx_per_type,
            common_curve=common_curve,
            device=device,
        )

        # ── ★ Inverse design: target S11/S22/S33 = 2/3/4 GHz, BW 100 MHz ──
        INV_CHANNEL_TARGET_FREQS = (2.0, 3.0, 4.0)
        INV_BANDWIDTH_GHZ = 0.1
        INV_DEEP_DB = -15.0

        # Tier 1 + Tier 2(a) — stronger optimizer
        INV_N_STARTS = 32              # 8 → 32 (4배)
        INV_N_ITERS = 2000             # 500 → 2000 (early stop 으로 줄어들 수 있음)
        INV_LR = 5e-2
        INV_IN_BAND_WEIGHT = 10.0
        INV_OUT_BAND_WEIGHT = 0.0
        INV_Z_PRIOR_WEIGHT = 1e-3      # 초기값
        INV_Z_PRIOR_WEIGHT_END = 1e-5  # 종료값 (선형 감소)
        INV_COSINE_LR = True           # cosine LR schedule
        INV_EARLY_STOP_PATIENCE = 200  # 200 iter 정체 시 중단
        INV_RESTART_PATIENCE = 80      # 80 iter 정체 시 worst restart
        INV_RESTART_FRAC = 0.25        # 하위 25% 교체
        INV_RESTART_NOISE = 0.3        # restart 시 noise 크기
        INV_MAX_RESTARTS = 10          # 최대 restart 횟수

        # ★ 디코딩된 구조 figure 를 별도 4 개 창으로 띄울지
        #   False: 한 figure 안에 2×2 (overview, 기본)
        #   True : 4 개 view 각각 별도 큰 창 (확대해서 자세히 볼 때 유용)
        INV_SEPARATE_WINDOWS = False

        if cfg.show_figures and RUN_INVERSE_DESIGN:
            try:
                inv_result = run_inverse_design_pipeline(
                    ae=ae,
                    mlp=mlp,
                    dataset=dataset,
                    device=device,
                    channel_target_freqs=INV_CHANNEL_TARGET_FREQS,
                    bandwidth_ghz=INV_BANDWIDTH_GHZ,
                    n_starts=INV_N_STARTS,
                    n_iters=INV_N_ITERS,
                    lr=INV_LR,
                    in_band_weight=INV_IN_BAND_WEIGHT,
                    out_band_weight=INV_OUT_BAND_WEIGHT,
                    z_prior_weight=INV_Z_PRIOR_WEIGHT,
                    z_prior_weight_end=INV_Z_PRIOR_WEIGHT_END,
                    deep_db=INV_DEEP_DB,
                    seed=cfg.seed,
                    cosine_lr=INV_COSINE_LR,
                    early_stop_patience=INV_EARLY_STOP_PATIENCE,
                    restart_patience=INV_RESTART_PATIENCE,
                    restart_frac=INV_RESTART_FRAC,
                    restart_noise=INV_RESTART_NOISE,
                    max_restarts=INV_MAX_RESTARTS,
                    separate_windows=INV_SEPARATE_WINDOWS,
                )
            except Exception as e:
                import traceback as _tb
                print(f"\n[INVERSE DESIGN] failed: {type(e).__name__}: {e}")
                _tb.print_exc()
                inv_result = None

            # ★ 결과 파일 저장: tokens.txt, sparam.csv, figures.png
            if inv_result is not None:
                subsection("Saving inverse design outputs to inversed/")
                inversed_dir = os.path.join(script_dir, "inversed")
                try:
                    save_inverse_design_outputs(
                        inv_result, inversed_dir, run_tag,
                        channel_target_freqs=INV_CHANNEL_TARGET_FREQS,
                        bandwidth_ghz=INV_BANDWIDTH_GHZ,
                        deep_db=INV_DEEP_DB,
                    )
                except Exception as e:
                    import traceback as _tb
                    print(f"  ⚠ save_inverse_design_outputs failed: {type(e).__name__}: {e}")
                    _tb.print_exc()

        section("DONE")
        print(f"  Final eval_full RMSE : {result['eval_full']:.4f} dB")
        print(f"  figures: {len(plt.get_fignums())} (backend={_MPL_BACKEND})")

        # ★ 패치: figure 표시 전에 경과 시간 lock ──
        elapsed = time.time() - t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = elapsed % 60
        print(f"\n  경과 시간 (figure 표시 전): {h}h {m}m {s:.1f}s")

        # plt.show() 는 blocking — 사용자가 창 닫을 때까지 대기.
        # 이 대기 시간은 위에서 측정한 elapsed 에 안 들어감.
        # ★ Ctrl+C 로 즉시 종료 가능하도록 SIGINT 를 기본(프로세스 kill)으로 복원.
        #   matplotlib GUI mainloop 가 SIGINT 를 가로채서 Python 에 안 넘기는 백엔드
        #   (특히 TkAgg) 에서 Ctrl+C 가 먹지 않는 문제를 해결.
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

    finally:
        pass
