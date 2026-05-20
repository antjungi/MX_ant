#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse design from saved v3 checkpoint
========================================

학습된 AE_inverse_roletoken_v3 ckpt 를 불러와서 사용자 지정 target frequency 로
인버스 설계만 빠르게 돌리는 standalone 스크립트.

모든 설정은 아래 ★ CONFIG ★ 블록에서 수정. CLI 인자 없음.
"""

import os
import sys
import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  ★ CONFIG  ★    (수정은 여기에서만)
# ═══════════════════════════════════════════════════════════════

# ── Checkpoint ──────────────────────────────────────────────────
# "auto" 또는 ""  → tkinter 파일 선택 다이얼로그 띄움
# 또는 정확한 경로 직접 지정. 예: "ckpt/v3_last.pt"
CKPT_PATH         = "auto"
CKPT_INITIAL_DIR  = "ckpt"             # 다이얼로그가 처음 열 디렉토리

# ── Target frequencies per channel (GHz) ─────────────────────────
# None = 그 채널 loss 무시 (optimizer 가 그 채널 신경 안 씀)
TARGET_S11        = 2.4
TARGET_S22        = 3.5
TARGET_S33        = 5.8

BANDWIDTH_GHZ     = 0.1          # ± bw/2 안에서 deep_db 도달이 목표
DEEP_DB           = -15.0

# ── Optimization knobs ──────────────────────────────────────────
N_STARTS          = 32
N_ITERS           = 2000
LR                = 5e-2
IN_BAND_WEIGHT    = 10.0
OUT_BAND_WEIGHT   = 0.0
Z_PRIOR_WEIGHT    = 1e-3         # 초기 prior 가중치 (학습 분포 근처 유지)
Z_PRIOR_WEIGHT_END = 1e-5        # 최종 prior 가중치 (선형 감소)
SEED              = 0

# Tier 2(a) random restart on stagnation
COSINE_LR         = True
EARLY_STOP_PATIENCE = 200
RESTART_PATIENCE  = 80
RESTART_FRAC      = 0.25
RESTART_NOISE     = 0.3
MAX_RESTARTS      = 10

# ── Display / save ──────────────────────────────────────────────
SEPARATE_WINDOWS  = False        # decoded 4-view 를 각각 별도 창에 띄울지
SAVE_DIR          = "inversed"   # 결과 저장 폴더
SAVE_OUTPUTS      = True         # False 면 figure 만, 파일 안 떨굼

# ── Preset (학습 때와 동일해야 함) ─────────────────────────────
PRESET            = None         # None = CFG default, 또는 "small"/"medium"/"tiny"/"test"

# ═══════════════════════════════════════════════════════════════
#  (아래는 일반적으로 수정 불필요)
# ═══════════════════════════════════════════════════════════════

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import AE_inverse_roletoken_v3 as v3
from AE_inverse_roletoken_v3 import (
    CFG, apply_preset, set_seed,
    load_multitype_data, make_stratified_split,
    build_common_curve_and_print_baseline,
    DeepCADBaselineAE, SparamCommonResidualMLP,
    inverse_design_optimize,
    visualize_inverse_design_curve, visualize_decoded_structure,
    visualize_inverse_z_on_pca,
    save_inverse_design_outputs,
    collect_latents,
    section, subsection,
    RETURN_LABELS,
)


def _pick_ckpt_gui(initial_dir):
    """tkinter 파일 다이얼로그로 .pt 선택. 실패 시 None 반환."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print(f"  ⚠ tkinter not available ({e})")
        return None

    init_abs = os.path.abspath(initial_dir) if initial_dir and os.path.isdir(initial_dir) else os.getcwd()

    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askopenfilename(
            title="Select trained ckpt (.pt / .pth)",
            initialdir=init_abs,
            filetypes=[("PyTorch ckpt", "*.pt *.pth"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception as e:
        print(f"  ⚠ Tk filedialog failed: {e}")
        return None

    return path if path else None


def select_ckpt(ckpt_path_cfg, initial_dir):
    """CKPT_PATH 가 auto/"" 면 파일 다이얼로그, 아니면 그 경로 그대로 사용."""
    if ckpt_path_cfg and ckpt_path_cfg.lower() != "auto":
        if not os.path.exists(ckpt_path_cfg):
            print(f"  ✗ ckpt not found: {ckpt_path_cfg}")
            sys.exit(1)
        return ckpt_path_cfg

    print(f"  → opening file picker (initial dir: {initial_dir})")
    path = _pick_ckpt_gui(initial_dir)
    if not path:
        print("  ✗ no ckpt selected (or picker unavailable)")
        sys.exit(1)
    if not os.path.exists(path):
        print(f"  ✗ selected file does not exist: {path}")
        sys.exit(1)
    return path


def _print_config_summary(ckpt_path, channel_active, channel_target_freqs):
    section("INVERSE-ONLY — load v3 ckpt + custom targets")
    print(f"  ckpt          : {ckpt_path}")
    print(f"  preset        : {PRESET}")
    print(f"  target spec   : bw=±{BANDWIDTH_GHZ / 2 * 1000:.0f} MHz, deep_db={DEEP_DB}")
    for c, lbl in enumerate(RETURN_LABELS):
        if channel_active[c]:
            print(f"    {lbl}: {channel_target_freqs[c]:.3f} GHz   ★ ACTIVE")
        else:
            print(f"    {lbl}: (ignored)")
    print(f"  optimization  : n_starts={N_STARTS}, n_iters={N_ITERS}, lr={LR}")
    print(f"                  in_band_w={IN_BAND_WEIGHT}, out_band_w={OUT_BAND_WEIGHT}")
    print(f"                  z_prior {Z_PRIOR_WEIGHT:.1e} → {Z_PRIOR_WEIGHT_END:.1e}")
    print(f"  save          : {'ON' if SAVE_OUTPUTS else 'OFF'} → {SAVE_DIR}/")


def main():
    # ── 채널 활성화 결정 ──
    user_targets = [TARGET_S11, TARGET_S22, TARGET_S33]
    channel_active = [t is not None for t in user_targets]
    if not any(channel_active):
        print("Error: 적어도 한 채널은 타겟 지정 필요 (TARGET_S11/S22/S33)")
        sys.exit(1)

    # 비활성 채널의 default freq (mask 가 비활성이라 실제 영향 없음)
    default_freqs = [2.0, 3.0, 4.0]
    channel_target_freqs = tuple(
        t if t is not None else df
        for t, df in zip(user_targets, default_freqs)
    )

    # ── Ckpt 선택 ──
    ckpt_path = select_ckpt(CKPT_PATH, CKPT_INITIAL_DIR)

    _print_config_summary(ckpt_path, channel_active, channel_target_freqs)

    # ── CFG / preset ──
    cfg = CFG()
    if PRESET:
        cfg.preset = PRESET
    apply_preset(cfg)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data + build dataset ──
    section("LOAD DATA")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset, npy_files, type_ids, type_names, sparam_db, max_len = (
        load_multitype_data(cfg, script_dir)
    )

    # split required for common_curve baseline (MLP needs it)
    train_idx, val_idx, val_idx_per_type = make_stratified_split(
        cfg, dataset, type_ids, type_names,
    )
    common_curve = build_common_curve_and_print_baseline(
        dataset=dataset, train_idx=train_idx,
        val_idx_per_type=val_idx_per_type, type_names=type_names,
    )

    # ── Build models ──
    section("BUILD MODELS + LOAD CKPT")
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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ae.load_state_dict(ckpt["ae_state_dict"])
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    saved_max_len = ckpt.get("max_len")
    saved_role_map = ckpt.get("role_name_to_id", {})
    cur_role_map = getattr(dataset, "role_name_to_id", {})
    if saved_max_len is not None and saved_max_len != dataset.max_len:
        print(f"  ⚠ max_len mismatch: ckpt={saved_max_len} dataset={dataset.max_len}")
    if saved_role_map and saved_role_map != cur_role_map:
        print(f"  ⚠ role_name_to_id mismatch:")
        print(f"      saved  : {saved_role_map}")
        print(f"      current: {cur_role_map}")
    print(f"  ✓ loaded ckpt ← {ckpt_path}")
    bvm = ckpt.get("best_val_metric")
    if bvm is not None:
        print(f"    best val RMSE dB at save: {bvm:.4f}")

    # ── Monkey-patch make_target_db_curve 로 비활성 채널 mask 제거 ──
    _orig_make_target = v3.make_target_db_curve

    def _masked_make_target(freqs_full, ctfs, bw, deep_db=-20.0, flat_db=0.0):
        target, masks = _orig_make_target(freqs_full, ctfs, bw, deep_db, flat_db)
        for c, active in enumerate(channel_active):
            if not active:
                masks[:, c] = False
                target[:, c] = flat_db
        return target, masks

    v3.make_target_db_curve = _masked_make_target

    # ── Run inverse optimization ──
    section("INVERSE DESIGN — latent search")
    result = inverse_design_optimize(
        ae=ae, mlp=mlp, dataset=dataset,
        channel_target_freqs=channel_target_freqs,
        bandwidth_ghz=BANDWIDTH_GHZ,
        device=device,
        n_starts=N_STARTS,
        n_iters=N_ITERS,
        lr=LR,
        in_band_weight=IN_BAND_WEIGHT,
        out_band_weight=OUT_BAND_WEIGHT,
        z_prior_weight=Z_PRIOR_WEIGHT,
        z_prior_weight_end=Z_PRIOR_WEIGHT_END,
        deep_db=DEEP_DB,
        seed=SEED,
        cosine_lr=COSINE_LR,
        early_stop_patience=EARLY_STOP_PATIENCE,
        restart_patience=RESTART_PATIENCE,
        restart_frac=RESTART_FRAC,
        restart_noise=RESTART_NOISE,
        max_restarts=MAX_RESTARTS,
    )

    print(f"\n  best match loss : {result['best_loss']:.4f}")
    print(f"  in-band MSE:")
    for c, lbl in enumerate(RETURN_LABELS):
        tag = "★" if channel_active[c] else "(ignored)"
        print(f"    {lbl}: {result['best_in_band_mse'][c]:.3f}  {tag}")

    # ── Figures (v3 의 그림 9, 10 스타일) ──
    subsection("S-param curve figure")
    try:
        visualize_inverse_design_curve(result)
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ visualize_inverse_design_curve failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    subsection("Decoded structure figure")
    recon_trim = None
    try:
        recon_trim, sketches, _ = visualize_decoded_structure(
            ae, result["best_z"], dataset, device,
            title="Inverse-designed structure (from loaded ckpt)",
            separate_windows=SEPARATE_WINDOWS,
        )
        result["decoded_tokens"] = recon_trim
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ visualize_decoded_structure failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    subsection("z trajectory on training PCA")
    try:
        z_train_all, tids_train_all = collect_latents(
            ae, dataset, list(range(len(dataset))), device,
        )
        visualize_inverse_z_on_pca(
            z_train=z_train_all,
            type_ids_train=tids_train_all,
            type_names=list(dataset.type_names),
            z_trajectory=result.get("z_trajectory", []),
            track_iters=result.get("track_iters", []),
            best_start_idx=int(result.get("best_start_idx", 0)),
        )
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ visualize_inverse_z_on_pca failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    # ── Save ──
    if SAVE_OUTPUTS and recon_trim is not None:
        subsection(f"Saving outputs → {SAVE_DIR}/")
        try:
            tag_parts = []
            for lbl, ct, ac in zip(
                RETURN_LABELS, channel_target_freqs, channel_active,
            ):
                if ac:
                    tag_parts.append(f"{lbl}_{ct:.2f}GHz")
            run_tag = "infer_" + "_".join(tag_parts)
            save_inverse_design_outputs(
                result, SAVE_DIR, run_tag,
                channel_target_freqs=channel_target_freqs,
                bandwidth_ghz=BANDWIDTH_GHZ,
                deep_db=DEEP_DB,
            )
        except Exception as e:
            import traceback as _tb
            print(f"  ⚠ save failed: {type(e).__name__}: {e}")
            _tb.print_exc()

    section("DONE")
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
