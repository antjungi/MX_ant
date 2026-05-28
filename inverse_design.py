#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse design from saved AE_main checkpoint
============================================

학습된 AE_main ckpt 하나만 있으면 인버스 설계만 따로 돌릴 수 있는 standalone 스크립트.

지원 동작:
  - tkinter 파일 다이얼로그로 ckpt 선택
  - cfg / architecture 정보를 ckpt 안에서 자동 복원 (preset mismatch 0)
  - AE_main.py 가 같이 없어도 됨 → ckpt 안의 embed 된 source_code 자동 추출/import
  - 데이터 폴더 위치도 경로로 override 가능
  - 채널별 (S11/S22/S33) 타겟 주파수를 켜고/끌 수 있음

모든 설정은 아래 ★ CONFIG ★ 블록에서 수정. CLI 인자 없음.
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
# Ckpt 파일: 학습 후 저장된 .pt / .pth
#   "auto" 또는 ""  → tkinter 파일 다이얼로그 띄움
#   특정 경로       → 그 파일 사용. 예: r"G:\jg\MX_AI_Code\ckpt\AE_main_last.pt"
CKPT_PATH         = "auto"
CKPT_INITIAL_DIR  = "ckpt"                # 다이얼로그가 처음 열 디렉토리

# AE_main.py 파일 (이름 자유 — 예: AE_main.py / 다른 이름):
#   ""  → ckpt 안의 embed 된 source_code 자동 사용
#   특정 경로 → 그 파일 직접 load
LOCAL_PY_PATH     = ""

# 데이터 폴더 root (hfss_results/ 의 상위):
#   ""  → 이 스크립트가 있는 폴더 기준
#   특정 경로 → 그 폴더 기준으로 cfg.npy_dirs / sparam_globs 해석
DATA_ROOT         = ""

# ── Target frequencies per channel (GHz) ─────────────────────────
# None = 그 채널 loss 무시 (optimizer 가 그 채널 신경 안 씀)
TARGET_S11        = 2.4
TARGET_S22        = 3.5
TARGET_S33        = 5.8

BANDWIDTH_GHZ     = 0.1                   # ± bw/2 안에서 deep_db 도달이 목표
DEEP_DB           = -15.0

# ── Optimization knobs ──────────────────────────────────────────
N_STARTS          = 32
N_ITERS           = 2000
LR                = 5e-2
IN_BAND_WEIGHT    = 10.0
OUT_BAND_WEIGHT   = 0.0
Z_PRIOR_WEIGHT    = 1e-3
Z_PRIOR_WEIGHT_END = 1e-5
SEED              = 0

COSINE_LR         = True
EARLY_STOP_PATIENCE = 200
RESTART_PATIENCE  = 80
RESTART_FRAC      = 0.25
RESTART_NOISE     = 0.3
MAX_RESTARTS      = 10

# ── Display / save ──────────────────────────────────────────────
SEPARATE_WINDOWS  = False                 # decoded 4-view 를 각각 별도 창에
SAVE_DIR          = "inversed"
SAVE_OUTPUTS      = True

# ── Preset fallback (ckpt 에 cfg_dict 없을 때만 사용) ───────────
FALLBACK_PRESET   = None

# ═══════════════════════════════════════════════════════════════
#  (아래는 일반적으로 수정 불필요)
# ═══════════════════════════════════════════════════════════════


# ── ckpt 선택 ───────────────────────────────────────────────────
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


# ── 모듈 로드 (3 단계 fallback) ────────────────────────────────
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
    for name in ("AE_main", "AE_inverse"):
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
        print("  ✗ no source_code embedded in this ckpt (older ckpt without embed)")
        return None, None
    tmp_dir = tempfile.mkdtemp(prefix="ae_src_")
    tmp_file = os.path.join(tmp_dir, src_name)
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(src)
    except Exception as e:
        print(f"  ✗ failed to write temp source: {e}")
        return None, None
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
    """우선순위 (a) LOCAL_PY_PATH → (b) 로컬 자동검색 → (c) ckpt embedded."""
    if LOCAL_PY_PATH:
        m, n = _load_module_from_file(LOCAL_PY_PATH)
        if m is not None:
            return m, n
        print(f"  ⚠ LOCAL_PY_PATH failed: {LOCAL_PY_PATH}")
    m, n = _try_local_import()
    if m is not None:
        return m, n
    print("  ⚠ no local AE module — trying ckpt-embedded source")
    return _import_from_ckpt_embedded(ckpt_path)


# ── CFG 복원 ────────────────────────────────────────────────────
def _restore_cfg(ae_mod, ckpt_dict, fallback_preset):
    saved = ckpt_dict.get("cfg_dict")
    cfg_cls = ae_mod.CFG
    if saved:
        import dataclasses as _dc
        valid_keys = {f.name for f in _dc.fields(cfg_cls)}
        filtered = {k: v for k, v in saved.items() if k in valid_keys}
        cfg = cfg_cls(**filtered)
        print(f"  ✓ cfg restored from ckpt "
              f"(preset={getattr(cfg, 'preset', '?')}, "
              f"d_model={getattr(cfg, 'd_model', '?')}, "
              f"latent={getattr(cfg, 'latent', '?')})")
        return cfg
    cfg = cfg_cls()
    if fallback_preset:
        cfg.preset = fallback_preset
    if hasattr(ae_mod, "apply_preset"):
        ae_mod.apply_preset(cfg)
    print(f"  ⚠ no cfg_dict in ckpt — using fresh CFG (preset={cfg.preset})")
    return cfg


def _print_config_summary(ckpt_path, mod_name, channel_active,
                          channel_target_freqs, ae_mod):
    ae_mod.section("INVERSE-ONLY — load ckpt + custom targets")
    print(f"  module        : {mod_name}")
    print(f"  ckpt          : {ckpt_path}")
    print(f"  data root     : {DATA_ROOT or '(script dir default)'}")
    print(f"  target spec   : bw=±{BANDWIDTH_GHZ / 2 * 1000:.0f} MHz, deep_db={DEEP_DB}")
    for c, lbl in enumerate(ae_mod.RETURN_LABELS):
        if channel_active[c]:
            print(f"    {lbl}: {channel_target_freqs[c]:.3f} GHz   ★ ACTIVE")
        else:
            print(f"    {lbl}: (ignored)")
    print(f"  optimization  : n_starts={N_STARTS}, n_iters={N_ITERS}, lr={LR}")
    print(f"                  in_band_w={IN_BAND_WEIGHT}, out_band_w={OUT_BAND_WEIGHT}")
    print(f"                  z_prior {Z_PRIOR_WEIGHT:.1e} → {Z_PRIOR_WEIGHT_END:.1e}")
    print(f"  save          : {'ON' if SAVE_OUTPUTS else 'OFF'} → {SAVE_DIR}/")


def main():
    # ── 1) 채널 활성화 결정 ──
    user_targets = [TARGET_S11, TARGET_S22, TARGET_S33]
    channel_active = [t is not None for t in user_targets]
    if not any(channel_active):
        print("Error: 적어도 한 채널은 타겟 지정 필요 (TARGET_S11/S22/S33)")
        sys.exit(1)
    default_freqs = [2.0, 3.0, 4.0]
    channel_target_freqs = tuple(
        t if t is not None else df
        for t, df in zip(user_targets, default_freqs)
    )

    # ── 2) Ckpt 선택 ──
    ckpt_path = select_ckpt(CKPT_PATH, CKPT_INITIAL_DIR)

    # ── 3) AE 모듈 확보 ──
    ae_mod, mod_name = _resolve_module(ckpt_path)
    if ae_mod is None:
        print("  ✗ unable to obtain AE module — abort")
        sys.exit(1)

    import torch
    import numpy as np   # noqa: F401

    _print_config_summary(ckpt_path, mod_name, channel_active,
                          channel_target_freqs, ae_mod)

    # ── 4) Ckpt dict 로드 + CFG 복원 ──
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _restore_cfg(ae_mod, ckpt, FALLBACK_PRESET)
    ae_mod.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 5) Data root + dataset ──
    ae_mod.section("LOAD DATA")
    script_dir = (
        os.path.abspath(DATA_ROOT) if DATA_ROOT
        else os.path.dirname(os.path.abspath(__file__))
    )
    print(f"  data script_dir = {script_dir}")
    dataset, npy_files, type_ids, type_names, sparam_db, max_len = (
        ae_mod.load_multitype_data(cfg, script_dir)
    )
    train_idx, val_idx, val_idx_per_type = ae_mod.make_stratified_split(
        cfg, dataset, type_ids, type_names,
    )
    common_curve = ae_mod.build_common_curve_and_print_baseline(
        dataset=dataset, train_idx=train_idx,
        val_idx_per_type=val_idx_per_type, type_names=type_names,
    )

    # ── 6) 모델 build + ckpt weights 로드 ──
    ae_mod.section("BUILD MODELS + LOAD WEIGHTS")
    ae = ae_mod.DeepCADBaselineAE(
        max_len=dataset.max_len,
        d_model=cfg.d_model, d_param=cfg.d_param,
        nhead=cfg.nhead, n_enc=cfg.n_enc, n_dec=cfg.n_dec,
        d_ff=cfg.d_ff, latent=cfg.latent, mem_tokens=cfg.mem_tokens,
        dropout=cfg.dropout, n_pool=cfg.n_pool,
        n_freq_bands=cfg.n_freq_bands,
        aux_numeric=cfg.aux_numeric, aux_hidden_mult=cfg.aux_hidden_mult,
    ).to(device)
    mlp = ae_mod.SparamCommonResidualMLP(
        latent_dim=cfg.latent, n_freq=cfg.n_freq,
        common_curve=common_curve,
        hidden_mult=cfg.mlp_hidden_mult, dropout=cfg.mlp_dropout,
        residual_scale=cfg.residual_scale,
        zero_init_residual=cfg.zero_init_residual,
    ).to(device)
    ae.load_state_dict(ckpt["ae_state_dict"])
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    ae.to(device); mlp.to(device)

    saved_max_len = ckpt.get("max_len")
    if saved_max_len is not None and saved_max_len != dataset.max_len:
        print(f"  ⚠ max_len mismatch: ckpt={saved_max_len} dataset={dataset.max_len}")
    print(f"  ✓ loaded weights ← {ckpt_path}")
    bvm = ckpt.get("best_val_metric")
    if bvm is not None:
        print(f"    best val RMSE dB at save: {bvm:.4f}")

    # ── 7) 비활성 채널 mask 제거 (per-channel target 선택) ──
    _orig_make_target = ae_mod.make_target_db_curve

    def _masked_make_target(freqs_full, ctfs, bw, deep_db=-20.0, flat_db=0.0):
        target, masks = _orig_make_target(freqs_full, ctfs, bw, deep_db, flat_db)
        for c, active in enumerate(channel_active):
            if not active:
                masks[:, c] = False
                target[:, c] = flat_db
        return target, masks

    ae_mod.make_target_db_curve = _masked_make_target

    # ── 8) Inverse optimization ──
    ae_mod.section("INVERSE DESIGN — latent search")
    result = ae_mod.inverse_design_optimize(
        ae=ae, mlp=mlp, dataset=dataset,
        channel_target_freqs=channel_target_freqs,
        bandwidth_ghz=BANDWIDTH_GHZ,
        device=device,
        n_starts=N_STARTS, n_iters=N_ITERS, lr=LR,
        in_band_weight=IN_BAND_WEIGHT, out_band_weight=OUT_BAND_WEIGHT,
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
    for c, lbl in enumerate(ae_mod.RETURN_LABELS):
        tag = "★" if channel_active[c] else "(ignored)"
        print(f"    {lbl}: {result['best_in_band_mse'][c]:.3f}  {tag}")

    # ── 9) Figures ──
    ae_mod.subsection("S-param curve figure")
    try:
        ae_mod.visualize_inverse_design_curve(result)
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ visualize_inverse_design_curve failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    ae_mod.subsection("Decoded structure figure")
    recon_trim = None
    try:
        ret = ae_mod.visualize_decoded_structure(
            ae, result["best_z"], dataset, device,
            title="Inverse-designed structure (from loaded ckpt)",
            separate_windows=SEPARATE_WINDOWS,
        )
        if isinstance(ret, tuple) and len(ret) >= 1:
            recon_trim = ret[0]
        result["decoded_tokens"] = recon_trim
    except Exception as e:
        import traceback as _tb
        print(f"  ⚠ visualize_decoded_structure failed: {type(e).__name__}: {e}")
        _tb.print_exc()

    if hasattr(ae_mod, "visualize_inverse_z_on_pca"):
        ae_mod.subsection("z trajectory on training PCA")
        try:
            z_train_all, tids_train_all = ae_mod.collect_latents(
                ae, dataset, list(range(len(dataset))), device,
            )
            ae_mod.visualize_inverse_z_on_pca(
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

    # ── 10) Save ──
    if SAVE_OUTPUTS and recon_trim is not None:
        ae_mod.subsection(f"Saving outputs → {SAVE_DIR}/")
        try:
            tag_parts = []
            for lbl, ct, ac in zip(
                ae_mod.RETURN_LABELS, channel_target_freqs, channel_active,
            ):
                if ac:
                    tag_parts.append(f"{lbl}_{ct:.2f}GHz")
            run_tag = "infer_" + "_".join(tag_parts)
            ae_mod.save_inverse_design_outputs(
                result, SAVE_DIR, run_tag,
                channel_target_freqs=channel_target_freqs,
                bandwidth_ghz=BANDWIDTH_GHZ,
                deep_db=DEEP_DB,
            )
        except Exception as e:
            import traceback as _tb
            print(f"  ⚠ save failed: {type(e).__name__}: {e}")
            _tb.print_exc()

    ae_mod.section("DONE")
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
