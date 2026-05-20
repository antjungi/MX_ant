#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse design from saved v3 checkpoint
========================================

학습된 AE_inverse_roletoken_v3 ckpt 를 불러와서 사용자 지정 target frequency 로
인버스 설계만 빠르게 돌리는 standalone 스크립트.

채널별 타겟을 독립적으로 켜고/끌 수 있음:
  --s11 2.4                           # S11 만 2.4 GHz 에 dip (S22/S33 무시)
  --s22 3.5                           # S22 만
  --s33 5.8                           # S33 만
  --s11 2.4 --s22 3.5 --s33 5.8       # 세 채널 모두

지정 안 한 채널은 loss 에 포함 안 됨 (optimizer 가 그 채널 신경 안 씀).

결과:
  - Fig: S-param curve   (v3 그림 9 스타일)
  - Fig: decoded 구조 4-view  (v3 그림 10 스타일)
  - Fig: z trajectory on training PCA
  - inversed/ 폴더에 tokens / sparam / figures 저장 (--no_save 로 끄기)
"""

import os
import sys
import argparse
import torch
import numpy as np

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


def main():
    parser = argparse.ArgumentParser(
        description="Inverse design from v3 ckpt with per-channel target selection",
    )
    parser.add_argument("--ckpt", default="ckpt/v3_last.pt",
                        help="v3 학습 후 저장된 ckpt 경로")
    parser.add_argument("--preset", default=None,
                        help="preset name (default = CFG default)")

    parser.add_argument("--s11", type=float, default=None,
                        help="S11 target freq (GHz). 안 주면 S11 무시")
    parser.add_argument("--s22", type=float, default=None,
                        help="S22 target freq (GHz). 안 주면 S22 무시")
    parser.add_argument("--s33", type=float, default=None,
                        help="S33 target freq (GHz). 안 주면 S33 무시")
    parser.add_argument("--bw", type=float, default=0.1,
                        help="대역폭 (GHz, target ± bw/2)")
    parser.add_argument("--deep_db", type=float, default=-15.0,
                        help="band 안에서 도달하려는 dB 깊이")

    parser.add_argument("--n_starts", type=int, default=32)
    parser.add_argument("--n_iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--in_band_weight", type=float, default=10.0)
    parser.add_argument("--out_band_weight", type=float, default=0.0)
    parser.add_argument("--z_prior_weight", type=float, default=1e-3)
    parser.add_argument("--z_prior_weight_end", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--separate_windows", action="store_true",
                        help="구조 figure 4 view 를 각각 별도 창으로")
    parser.add_argument("--save_dir", default="inversed",
                        help="결과 저장 폴더")
    parser.add_argument("--no_save", action="store_true",
                        help="파일 저장 끔 (figure 만 띄움)")
    args = parser.parse_args()

    # ── 채널 선택 ──
    user_targets = [args.s11, args.s22, args.s33]
    channel_active = [t is not None for t in user_targets]
    if not any(channel_active):
        print("Error: 적어도 한 채널은 타겟 지정 필요 (--s11/--s22/--s33)")
        sys.exit(1)

    default_freqs = [2.0, 3.0, 4.0]
    channel_target_freqs = tuple(
        t if t is not None else df
        for t, df in zip(user_targets, default_freqs)
    )

    section("INVERSE-ONLY — load v3 ckpt + custom targets")
    print(f"  ckpt          : {args.ckpt}")
    print(f"  target spec   : bw=±{args.bw / 2 * 1000:.0f} MHz, deep_db={args.deep_db}")
    for c, lbl in enumerate(RETURN_LABELS):
        if channel_active[c]:
            print(f"    {lbl}: {channel_target_freqs[c]:.3f} GHz   ★ ACTIVE")
        else:
            print(f"    {lbl}: (ignored)")

    # ── CFG / preset ──
    cfg = CFG()
    if args.preset:
        cfg.preset = args.preset
    apply_preset(cfg)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data + build dataset ──
    section("LOAD DATA")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset, npy_files, type_ids, type_names, sparam_db, max_len = (
        load_multitype_data(cfg, script_dir)
    )

    # split is required for common_curve baseline (MLP needs it)
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

    if not os.path.exists(args.ckpt):
        print(f"  ✗ ckpt not found: {args.ckpt}")
        sys.exit(1)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
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
    print(f"  ✓ loaded ckpt ← {args.ckpt}")
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
        bandwidth_ghz=args.bw,
        device=device,
        n_starts=args.n_starts,
        n_iters=args.n_iters,
        lr=args.lr,
        in_band_weight=args.in_band_weight,
        out_band_weight=args.out_band_weight,
        z_prior_weight=args.z_prior_weight,
        z_prior_weight_end=args.z_prior_weight_end,
        deep_db=args.deep_db,
        seed=args.seed,
        cosine_lr=True,
        early_stop_patience=200,
        restart_patience=80,
        restart_frac=0.25,
        restart_noise=0.3,
        max_restarts=10,
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
            separate_windows=args.separate_windows,
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

    # ── 저장 ──
    if not args.no_save and recon_trim is not None:
        subsection(f"Saving outputs → {args.save_dir}/")
        try:
            tag_parts = []
            for lbl, ct, ac in zip(
                RETURN_LABELS, channel_target_freqs, channel_active,
            ):
                if ac:
                    tag_parts.append(f"{lbl}_{ct:.2f}GHz")
            run_tag = "infer_" + "_".join(tag_parts)
            save_inverse_design_outputs(
                result, args.save_dir, run_tag,
                channel_target_freqs=channel_target_freqs,
                bandwidth_ghz=args.bw,
                deep_db=args.deep_db,
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
