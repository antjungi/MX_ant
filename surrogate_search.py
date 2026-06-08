#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surrogate-based parameter sweep + target search
================================================

학습된 surrogate ckpt 를 이용해, base 시퀀스의 특정 param 들을 바꿔가며
S-param 을 빠르게 예측하고 target spec 에 가장 잘 맞는 후보를 찾는다.

흐름:
  1) ckpt 로 SurrogateModel 복원
  2) base 토큰 시퀀스 1개 선택 (dataset 의 한 sample, 또는 임의)
  3) N개의 후보 생성:
       - random_perturb: 모든 (또는 특정) 슬롯에 ±delta 노이즈
       - 또는 grid sweep (user 지정 슬롯)
  4) 한 번에 batch forward → N×n_freq×3 S-param 예측
  5) target spec (channel/freq/dB depth) 으로 score 매기고 best 정렬
  6) top-K 출력 + best vs target 곡선 figure
  7) inversed/ 폴더에 결과 저장 (옵션)

모든 설정은 아래 ★ CONFIG ★ 블록에서. CLI 인자 없음.
"""

import os
import sys
import tempfile
import importlib
import importlib.util


# ═══════════════════════════════════════════════════════════════
#  ★ CONFIG  ★    (수정은 여기에서만)
# ═══════════════════════════════════════════════════════════════

# ── 필요한 파일/폴더 경로 ─────────────────────────────────────
CKPT_PATH         = "auto"             # "auto" → 파일 다이얼로그
CKPT_INITIAL_DIR  = "ckpt"

LOCAL_PY_PATH     = ""                 # surrogate.py 경로 (빈 문자열 = 자동 검색/ckpt embed)
DATA_ROOT         = ""                 # "" = script 폴더 기준

# ── Base 시퀀스 ─────────────────────────────────────────────────
# dataset 안의 어느 sample 을 기준 (base) 로 쓸지
#   integer        : 그 idx 사용
#   "random"       : 매 실행 다른 sample
#   ("type", N)    : 특정 type 의 N번째 (예: ("type1", 0))
BASE_SAMPLE       = "random"

# ── Target spec ────────────────────────────────────────────────
# 각 채널의 target. None 이면 그 채널 score 에 영향 X.
TARGET_S11        = 2.4                # GHz
TARGET_S22        = 3.5
TARGET_S33        = 5.8

BANDWIDTH_GHZ     = 0.1                # ±bw/2 안에서 deep_db 이하 만족이 목표
DEEP_DB           = -15.0

# ── Sweep 설정 ─────────────────────────────────────────────────
SWEEP_MODE        = "random_perturb"   # 현재 지원: "random_perturb"
N_CANDIDATES      = 1000

# random_perturb 강도: 각 param 을 base ± PERTURB_QUANT 안에서 균등 샘플
#   (param vocab 은 0..1023 이라 100 정도가 ~10% 변화)
PERTURB_QUANT     = 80

# 어떤 슬롯을 흔들지: "all_param" / "ext_only" / "line_arc_circle"
PERTURB_TARGET    = "all_param"

SEED              = 0

# ── Display / save ─────────────────────────────────────────────
TOPK_PRINT        = 10                 # 콘솔에 top-K 출력
TOPK_PLOT         = 3                  # figure 에 best K 곡선 겹치기
SAVE_OUTPUTS      = True
SAVE_DIR          = "surrogate_search"

# ── Preset fallback (ckpt 에 cfg_dict 없을 때만) ───────────────
FALLBACK_PRESET   = None


# ═══════════════════════════════════════════════════════════════
#  (아래 일반적으로 수정 불필요)
# ═══════════════════════════════════════════════════════════════


def _pick_ckpt_gui(initial_dir):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print(f"  ⚠ tkinter not available ({e})")
        return None
    init_abs = (
        os.path.abspath(initial_dir)
        if initial_dir and os.path.isdir(initial_dir)
        else os.getcwd()
    )
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askopenfilename(
            title="Select surrogate ckpt (.pt / .pth)",
            initialdir=init_abs,
            filetypes=[("PyTorch ckpt", "*.pt *.pth"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception as e:
        print(f"  ⚠ Tk filedialog failed: {e}")
        return None
    return path if path else None


def select_ckpt(ckpt_path_cfg, initial_dir):
    if ckpt_path_cfg and ckpt_path_cfg.lower() != "auto":
        if not os.path.exists(ckpt_path_cfg):
            print(f"  ✗ ckpt not found: {ckpt_path_cfg}")
            sys.exit(1)
        return ckpt_path_cfg
    print(f"  → opening file picker (initial dir: {initial_dir})")
    path = _pick_ckpt_gui(initial_dir)
    if not path:
        print("  ✗ no ckpt selected")
        sys.exit(1)
    return path


def _load_module_from_file(path, mod_name=None):
    if not path or not os.path.exists(path):
        return None, None
    if mod_name is None:
        mod_name = os.path.splitext(os.path.basename(path))[0]
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return None, None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        d = os.path.dirname(os.path.abspath(path))
        if d not in sys.path:
            sys.path.insert(0, d)
        spec.loader.exec_module(mod)
        print(f"  ✓ module '{mod_name}' loaded from {path}")
        return mod, mod_name
    except Exception as e:
        print(f"  ✗ failed to load module from {path}: {e}")
        return None, None


def _try_local_import():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.getcwd()):
        if p not in sys.path:
            sys.path.insert(0, p)
    for name in ("surrogate", "AE_main"):
        try:
            return importlib.import_module(name), name
        except ModuleNotFoundError:
            continue
    return None, None


def _import_from_ckpt_embedded(ckpt_path):
    import torch as _torch
    try:
        raw = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ✗ failed to read ckpt: {e}")
        return None, None
    src = raw.get("source_code")
    src_name = raw.get("source_filename")
    if not src or not src_name:
        print("  ✗ no source_code embedded in ckpt")
        return None, None
    tmp_dir = tempfile.mkdtemp(prefix="surr_src_")
    tmp_file = os.path.join(tmp_dir, src_name)
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(src)
    sys.path.insert(0, tmp_dir)
    mod_name = os.path.splitext(src_name)[0]
    try:
        mod = importlib.import_module(mod_name)
        print(f"  ✓ module '{mod_name}' loaded from ckpt-embedded source ({tmp_dir})")
        return mod, mod_name
    except Exception as e:
        print(f"  ✗ failed to import embedded source: {e}")
        return None, None


def _resolve_module(ckpt_path):
    if LOCAL_PY_PATH:
        m, n = _load_module_from_file(LOCAL_PY_PATH)
        if m is not None:
            return m, n
        print(f"  ⚠ LOCAL_PY_PATH failed: {LOCAL_PY_PATH}")
    m, n = _try_local_import()
    if m is not None:
        return m, n
    return _import_from_ckpt_embedded(ckpt_path)


def _restore_cfg(mod, ckpt_dict, fallback_preset):
    saved = ckpt_dict.get("cfg_dict")
    cfg_cls = mod.CFG
    if saved:
        import dataclasses as _dc
        valid = {f.name for f in _dc.fields(cfg_cls)}
        cfg = cfg_cls(**{k: v for k, v in saved.items() if k in valid})
        print(f"  ✓ cfg restored from ckpt "
              f"(preset={getattr(cfg, 'preset', '?')}, "
              f"d_model={getattr(cfg, 'd_model', '?')}, "
              f"latent={getattr(cfg, 'latent', '?')})")
        return cfg
    cfg = cfg_cls()
    if fallback_preset:
        cfg.preset = fallback_preset
    if hasattr(mod, "apply_preset"):
        mod.apply_preset(cfg)
    print(f"  ⚠ no cfg_dict in ckpt — fresh CFG (preset={cfg.preset})")
    return cfg


# ═══════════════════════════════════════════════════════════════
# Candidate generation
# ═══════════════════════════════════════════════════════════════
def make_random_perturb_candidates(
    base_tokens, n_candidates, perturb_quant, target,
    n_quant, pad_v, valid_par_dict, cmd_role,
    rng,
):
    """base 시퀀스를 N번 복제하면서 param 값들에 ±perturb_quant 노이즈 추가.

    PAD (-1) 자리는 흔들지 않고, role_id (ROLE.param[0]) 도 건드리지 않음.
    """
    import numpy as np
    base = base_tokens.astype(np.int32)
    L = base.shape[0]
    N = n_candidates
    out = np.broadcast_to(base, (N, L, 17)).copy()

    # 어느 (row, col) 자리에 노이즈를 더할지 마스크 만들기
    cmd = base[:, 0]
    is_valid_row = cmd >= 0
    perturb_mask_col = np.zeros((L, 16), dtype=bool)   # row, param_col
    for r in range(L):
        c = int(cmd[r])
        if not is_valid_row[r]:
            continue
        n_valid_param = valid_par_dict.get(c, 0)
        if n_valid_param <= 0:
            continue
        if target == "ext_only" and c != 4:           # EXT cmd id = 4
            continue
        if target == "line_arc_circle" and c not in (0, 1, 2):  # LINE/ARC/CIRCLE
            continue
        # ROLE 의 param[0] (role_id) 는 흔들지 않음
        start = 1 if c == cmd_role else 0
        for col in range(start, n_valid_param):
            if base[r, 1 + col] >= 0:
                perturb_mask_col[r, col] = True

    # 노이즈 적용
    for r in range(L):
        for col in range(16):
            if not perturb_mask_col[r, col]:
                continue
            noise = rng.integers(-perturb_quant, perturb_quant + 1, size=N)
            new_vals = out[:, r, 1 + col] + noise
            out[:, r, 1 + col] = np.clip(new_vals, 0, n_quant - 1)

    return out


# ═══════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════
def score_against_target(
    pred_full_db, freqs_full,
    target_freqs, channel_active, bandwidth_ghz, deep_db,
):
    """pred_full_db: (N, n_freq, 3). 활성 채널에서 in-band violation 의 MSE 평균."""
    import numpy as np
    N = pred_full_db.shape[0]
    scores = np.zeros(N, dtype=np.float32)
    n_active = sum(channel_active)
    if n_active == 0:
        return scores

    half = bandwidth_ghz / 2.0
    for c, (active, f0) in enumerate(zip(channel_active, target_freqs)):
        if not active:
            continue
        band_mask = (freqs_full >= f0 - half) & (freqs_full <= f0 + half)
        if not band_mask.any():
            continue
        # pred 가 deep_db 보다 위에 있으면 위반
        in_band = pred_full_db[:, band_mask, c]       # (N, n_band)
        violation = np.maximum(in_band - deep_db, 0)
        scores += (violation ** 2).mean(axis=1) / max(n_active, 1)

    return scores


def in_band_worst_db(pred_full_db, freqs_full, target_freqs, bandwidth_ghz):
    """채널별 in-band worst (= max) dB 값. (N, 3) 반환."""
    import numpy as np
    N = pred_full_db.shape[0]
    worst = np.zeros((N, 3), dtype=np.float32)
    half = bandwidth_ghz / 2.0
    for c, f0 in enumerate(target_freqs):
        band_mask = (freqs_full >= f0 - half) & (freqs_full <= f0 + half)
        if band_mask.any():
            worst[:, c] = pred_full_db[:, band_mask, c].max(axis=1)
        else:
            worst[:, c] = np.nan
    return worst


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    import time
    t0 = time.time()

    # 1) 채널 활성화
    user_targets = [TARGET_S11, TARGET_S22, TARGET_S33]
    channel_active = [t is not None for t in user_targets]
    if not any(channel_active):
        print("Error: 적어도 한 채널은 타겟 지정 필요 (TARGET_S11/S22/S33)")
        sys.exit(1)
    default_freqs = [2.0, 3.0, 4.0]
    target_freqs = tuple(
        t if t is not None else df
        for t, df in zip(user_targets, default_freqs)
    )

    # 2) Ckpt 선택
    ckpt_path = select_ckpt(CKPT_PATH, CKPT_INITIAL_DIR)

    # 3) 모듈 (surrogate.py) 로드
    mod, mod_name = _resolve_module(ckpt_path)
    if mod is None:
        print("  ✗ unable to obtain surrogate module — abort")
        sys.exit(1)

    import torch
    import numpy as np

    mod.section("SURROGATE SEARCH — sweep + target match")
    print(f"  module        : {mod_name}")
    print(f"  ckpt          : {ckpt_path}")
    print(f"  base sample   : {BASE_SAMPLE}")
    print(f"  sweep         : {SWEEP_MODE}, N={N_CANDIDATES}, "
          f"perturb=±{PERTURB_QUANT}, target_slots={PERTURB_TARGET}")
    print(f"  spec          : bw=±{BANDWIDTH_GHZ / 2 * 1000:.0f} MHz, deep_db={DEEP_DB}")
    for c, lbl in enumerate(mod.RETURN_LABELS):
        if channel_active[c]:
            print(f"    {lbl}: {target_freqs[c]:.3f} GHz   ★ ACTIVE")
        else:
            print(f"    {lbl}: (ignored)")

    # 4) Ckpt 로드 + cfg 복원
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _restore_cfg(mod, ckpt, FALLBACK_PRESET)
    mod.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5) 데이터셋 빌드 (base sample 가져오려고 필요)
    mod.section("LOAD DATA")
    script_dir = (
        os.path.abspath(DATA_ROOT) if DATA_ROOT
        else os.path.dirname(os.path.abspath(__file__))
    )
    dataset, _, type_ids, type_names, _, _ = mod.load_multitype_data(cfg, script_dir)
    train_idx, _val_idx, val_idx_per_type = mod.make_stratified_split(
        cfg, dataset, type_ids, type_names,
    )
    common_curve = mod.build_common_curve_and_print_baseline(
        dataset=dataset, train_idx=train_idx,
        val_idx_per_type=val_idx_per_type, type_names=type_names,
    )

    # 6) SurrogateModel build + 가중치 로드
    mod.section("BUILD SURROGATE")
    model = mod.SurrogateModel(
        max_len=dataset.max_len,
        d_model=cfg.d_model, d_param=cfg.d_param, nhead=cfg.nhead,
        n_enc=cfg.n_enc, d_ff=cfg.d_ff, latent=cfg.latent,
        dropout=cfg.dropout, n_pool=cfg.n_pool, n_freq_bands=cfg.n_freq_bands,
        n_freq=cfg.n_freq, common_curve=common_curve,
        mlp_hidden_mult=cfg.mlp_hidden_mult, mlp_dropout=cfg.mlp_dropout,
        residual_scale=cfg.residual_scale,
        zero_init_residual=cfg.zero_init_residual,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  ✓ loaded weights from {ckpt_path}")

    # 7) Base 시퀀스 선택
    if isinstance(BASE_SAMPLE, int):
        base_idx = int(BASE_SAMPLE)
    elif BASE_SAMPLE == "random":
        rng_base = np.random.default_rng(SEED)
        base_idx = int(rng_base.integers(0, len(dataset)))
    elif isinstance(BASE_SAMPLE, (tuple, list)) and len(BASE_SAMPLE) == 2:
        tname, k = BASE_SAMPLE
        try:
            ti = list(dataset.type_names).index(str(tname))
        except ValueError:
            print(f"  ✗ unknown type {tname}")
            sys.exit(1)
        candidates = [i for i in range(len(dataset)) if int(dataset.type_ids[i]) == ti]
        base_idx = candidates[int(k) % len(candidates)] if candidates else 0
    else:
        base_idx = 0
    base_tokens = dataset.raw[base_idx].astype(np.int32)
    try:
        tname = dataset.type_names[int(dataset.type_ids[base_idx])]
    except Exception:
        tname = "?"
    print(f"  base sample idx={base_idx}  type={tname}  tokens={base_tokens.shape[0]}")

    # 8) 후보 생성
    mod.section("GENERATE CANDIDATES")
    rng = np.random.default_rng(SEED)
    cands = make_random_perturb_candidates(
        base_tokens=base_tokens,
        n_candidates=N_CANDIDATES,
        perturb_quant=PERTURB_QUANT,
        target=PERTURB_TARGET,
        n_quant=mod.N_QUANT,
        pad_v=mod.PAD_V,
        valid_par_dict=mod.VALID_PAR,
        cmd_role=mod.ROLE,
        rng=rng,
    )
    print(f"  generated {cands.shape[0]} candidates  (shape={cands.shape})")
    print(f"  base + {N_CANDIDATES} candidates → total {N_CANDIDATES + 1} predictions")

    # base 도 함께 평가 (idx 0 = base, 1.. = perturbed)
    all_seqs = np.concatenate([base_tokens[None, :, :], cands], axis=0)

    # padding 처리 (max_len 까지)
    max_len = dataset.max_len
    if all_seqs.shape[1] < max_len:
        pad = np.full(
            (all_seqs.shape[0], max_len - all_seqs.shape[1], 17),
            mod.PAD_V, dtype=np.int32,
        )
        all_seqs = np.concatenate([all_seqs, pad], axis=1)
    elif all_seqs.shape[1] > max_len:
        all_seqs = all_seqs[:, :max_len, :]

    # 9) Batch forward
    mod.section("PREDICT (batch)")
    batch_size = 64
    pred_full_list = []
    t_pred = time.time()
    interp_w = torch.tensor(
        dataset.interp_matrix, dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        for i in range(0, all_seqs.shape[0], batch_size):
            chunk = all_seqs[i:i + batch_size]
            x = torch.tensor(chunk, dtype=torch.float32).to(device)
            pred_sel, _z = model(x)                     # (b, n_sel, 3)
            pred_full = mod.interpolate_selected_to_full_torch(pred_sel, interp_w)
            pred_full_list.append(pred_full.cpu().numpy())
    pred_full_db = np.concatenate(pred_full_list, axis=0)
    print(f"  predicted {pred_full_db.shape[0]} curves "
          f"in {time.time() - t_pred:.1f} s "
          f"({pred_full_db.shape[0] / max(time.time() - t_pred, 1e-6):.0f}/s)")

    # 10) Score
    mod.section("SCORE")
    freqs_full = np.asarray(dataset.freqs_full, dtype=np.float32)
    scores = score_against_target(
        pred_full_db, freqs_full,
        target_freqs, channel_active, BANDWIDTH_GHZ, DEEP_DB,
    )
    worst_db = in_band_worst_db(
        pred_full_db, freqs_full, target_freqs, BANDWIDTH_GHZ,
    )

    # 11) Top-K 출력
    order = np.argsort(scores)
    base_rank = int(np.where(order == 0)[0][0]) + 1   # base 의 rank (1-based)
    base_score = float(scores[0])
    best_idx = int(order[0])
    print(f"\n  base score = {base_score:.4f}  →  rank {base_rank} / {len(scores)}")
    print(f"  best candidate idx = {best_idx} (0=base), score = {scores[best_idx]:.4f}")
    print(f"\n  Top-{TOPK_PRINT}:")
    print(f"  {'rank':>4s} {'cand':>5s} {'score':>9s}   "
          f"{'S11_worst':>10s} {'S22_worst':>10s} {'S33_worst':>10s}")
    print("  " + "-" * 60)
    for r in range(min(TOPK_PRINT, len(order))):
        idx = int(order[r])
        is_base = "(base)" if idx == 0 else ""
        print(
            f"  {r + 1:>4d} {idx:>5d} {scores[idx]:>9.4f}   "
            f"{worst_db[idx, 0]:>10.2f} {worst_db[idx, 1]:>10.2f} "
            f"{worst_db[idx, 2]:>10.2f}  {is_base}"
        )

    # 12) Figure: best K vs target
    import matplotlib.pyplot as plt
    mod.subsection(f"Best top-{TOPK_PLOT} S-param vs target")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")
        fig.suptitle(
            f"Surrogate parameter sweep — best {TOPK_PLOT} of {len(scores)} "
            f"(base idx={base_idx}, type={tname})",
            fontsize=11, color="#222",
        )
        colors = ["#E07B5B", "#3F6E5C", "#A57CC1", "#F4A340", "#4C9BE8"]
        for c, lbl in enumerate(mod.RETURN_LABELS):
            ax = axes[c]
            # base
            ax.plot(
                freqs_full, pred_full_db[0, :, c],
                color="#666666", lw=1.2, ls="--", alpha=0.85,
                label="base",
            )
            # top-K
            for r in range(min(TOPK_PLOT, len(order))):
                idx = int(order[r])
                if idx == 0:
                    continue
                ax.plot(
                    freqs_full, pred_full_db[idx, :, c],
                    color=colors[r % len(colors)], lw=1.4,
                    alpha=0.92,
                    label=f"#{r + 1} cand{idx}",
                )
            # target band
            if channel_active[c]:
                f0 = target_freqs[c]
                ax.hlines(
                    DEEP_DB, f0 - BANDWIDTH_GHZ / 2, f0 + BANDWIDTH_GHZ / 2,
                    colors="#2E4172", linestyles="-", lw=1.0,
                    label=f"spec ≤ {DEEP_DB:.0f} dB",
                )
                for fx in (f0 - BANDWIDTH_GHZ / 2, f0 + BANDWIDTH_GHZ / 2):
                    ax.axvline(fx, color="#2E4172", ls=":", lw=0.6, alpha=0.5)
            ax.axhline(-10, color="#888", ls=":", lw=0.6, alpha=0.5)
            ax.set_title(f"{lbl}  target={target_freqs[c]:.2f} GHz "
                         f"{'(active)' if channel_active[c] else '(off)'}",
                         fontsize=9)
            ax.set_xlabel("freq [GHz]"); ax.set_ylabel("|S| [dB]")
            ax.grid(True, alpha=0.2)
            if c == 0:
                ax.legend(fontsize=7, loc="best", framealpha=0.85)
        plt.tight_layout()
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ figure failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    # 13) Save outputs
    if SAVE_OUTPUTS:
        os.makedirs(SAVE_DIR, exist_ok=True)
        tag_parts = [
            f"{lbl}_{tf:.2f}GHz"
            for lbl, tf, ac in zip(mod.RETURN_LABELS, target_freqs, channel_active)
            if ac
        ]
        run_tag = "sweep_" + "_".join(tag_parts)

        # top-K 시퀀스 + score csv
        try:
            csv_path = os.path.join(SAVE_DIR, f"{run_tag}_topk.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("rank,cand_idx,score,S11_worst_dB,S22_worst_dB,S33_worst_dB,is_base\n")
                for r in range(len(order)):
                    idx = int(order[r])
                    f.write(
                        f"{r + 1},{idx},{float(scores[idx]):.6f},"
                        f"{float(worst_db[idx, 0]):.4f},"
                        f"{float(worst_db[idx, 1]):.4f},"
                        f"{float(worst_db[idx, 2]):.4f},"
                        f"{1 if idx == 0 else 0}\n"
                    )
            print(f"  ✓ saved score table → {csv_path}")
        except Exception as e:
            print(f"  ⚠ csv save failed: {e}")

        # best 시퀀스 1개 (token txt)
        try:
            tok_path = os.path.join(SAVE_DIR, f"{run_tag}_best_tokens.txt")
            best_seq = all_seqs[best_idx]
            with open(tok_path, "w", encoding="utf-8") as f:
                f.write(f"# Best candidate idx={best_idx}, score={scores[best_idx]:.6f}\n")
                f.write(f"# base_idx={base_idx}, base_score={base_score:.6f}\n")
                f.write(
                    f"# in-band worst: "
                    f"S11={worst_db[best_idx, 0]:.2f} dB, "
                    f"S22={worst_db[best_idx, 1]:.2f} dB, "
                    f"S33={worst_db[best_idx, 2]:.2f} dB\n\n"
                )
                f.write(f"# Columns: idx | cmd_name | param0..param15 (PAD=-1)\n")
                for i, row in enumerate(best_seq):
                    c = int(row[0])
                    cmd_name = mod.CMD_NAME.get(c, f"UNK({c})")
                    params = " ".join(f"{int(p):5d}" for p in row[1:])
                    f.write(f"[{i:4d}] {cmd_name:7s} | {params}\n")
                    if c == mod.EOS:
                        break
            print(f"  ✓ saved best tokens → {tok_path}")
        except Exception as e:
            print(f"  ⚠ tokens save failed: {e}")

        # best S-param csv
        try:
            sp_path = os.path.join(SAVE_DIR, f"{run_tag}_best_sparam.csv")
            with open(sp_path, "w", encoding="utf-8") as f:
                f.write("freq_GHz,S11_dB,S22_dB,S33_dB\n")
                pf = pred_full_db[best_idx]
                for i in range(len(freqs_full)):
                    f.write(
                        f"{float(freqs_full[i]):.6f},"
                        f"{float(pf[i, 0]):.4f},"
                        f"{float(pf[i, 1]):.4f},"
                        f"{float(pf[i, 2]):.4f}\n"
                    )
            print(f"  ✓ saved best S-param → {sp_path}")
        except Exception as e:
            print(f"  ⚠ sparam save failed: {e}")

        # figure
        try:
            fig_path = os.path.join(SAVE_DIR, f"{run_tag}_topk.png")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"  ✓ saved figure → {fig_path}")
        except Exception as e:
            print(f"  ⚠ figure save failed: {e}")

    elapsed = time.time() - t0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"\n  total elapsed: {h}h {m}m {s:.1f}s")

    plt.show()


if __name__ == "__main__":
    main()
