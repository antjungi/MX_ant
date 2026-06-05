#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surrogate-only training: data → encoder → latent → S-param MLP
==============================================================

AE_main 의 전체 AE (encoder + decoder + 보조 head) 중에서
★ encoder + surrogate MLP 만 ★ 떼내서 학습하는 standalone 스크립트.

목적:
  - "안테나 구조 (토큰 시퀀스) → S-param 예측" 만 하는 성능 추론기
  - decoder / cmd-prm reconstruction loss / aux loss / inverse 다 없음
  - 학습 loss = S-param dB RMSE (+ optional VICReg)

장점:
  - 더 작고 빠름 (decoder 관련 ~10M 파라미터 제거)
  - 학습 신호 단일화 (S-param 만)
  - 인코더가 "S-param 예측" 에만 특화 → recon 보단 성능 예측 정확도가 더 중요할 때

CFG 기본은 AE_main 의 small preset 과 동일. 학습 후 `ckpt/surrogate_last.pt` 저장.

데이터 로딩 / S-param 파싱 / VICReg / interp matrix 같은 인프라는 AE_main 에서 import.
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import AE_main as aem
from AE_main import (
    # CFG / preset
    CFG, apply_preset, set_seed,
    # Constants
    N_CMD, N_ARGS, N_QUANT, N_BIT, QMAX, PAD_V,
    PAD_CMD_INDEX, PAD_PARAM_INDEX,
    RETURN_LABELS,
    # Data
    load_multitype_data, make_stratified_split,
    build_common_curve_and_print_baseline,
    # Submodules
    SinPosEnc, SparamCommonResidualMLP,
    # Losses
    sparam_full_interp_loss, vicreg_z_loss,
    interpolate_selected_to_full_torch,
    # Logging
    section, subsection,
)


# ═══════════════════════════════════════════════════════════════
#  Surrogate Model:  tokens → encoder → latent z → MLP → S-param
# ═══════════════════════════════════════════════════════════════
class SurrogateModel(nn.Module):
    """Encoder + S-param surrogate MLP. Decoder 없음.

    구조:
      tokens (B, L, 17)
        → cmd_emb + param_embs(+fourier) → param_proj → emb_norm
        → pos_enc → Transformer encoder ×n_enc → h (B, L, d_model)
        → PMA attention pooling (n_pool queries) → flatten → to_z
        → z (B, latent)
        → SparamCommonResidualMLP (= common_curve + residual(z))
        → S-param (B, n_freq, 3)
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

        # ── Embedding ──
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

        # ── Transformer Encoder ──
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_enc, norm=nn.LayerNorm(d_model),
        )

        # ── Pooling (PMA-style if n_pool>=2, else mean) ──
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

        # ── Surrogate MLP: z → S-param ──
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
        """tokens (B,L,17) → emb (B,L,d_model), pad_mask (B,L). AE_main 의 동일 로직."""
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
        """tokens → latent z."""
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
        """tokens → (S-param pred, z)."""
        z = self.encode(x)
        sparam_pred_sel = self.surrogate(z)
        return sparam_pred_sel, z


# ═══════════════════════════════════════════════════════════════
#  Training loop  (loss = S-param + optional VICReg)
# ═══════════════════════════════════════════════════════════════
def run_epoch(model, loader, optimizer, scheduler, device, cfg,
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


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
def train_surrogate(cfg, dataset, train_idx, val_idx, common_curve, device):
    set_seed(cfg.seed)

    from torch.utils.data import Subset
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if len(val_idx) > 0 else Subset(dataset, train_idx[:1])

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
    print(f"    encoder + pool + to_z: encoder path only (no decoder)")
    print(f"    surrogate MLP        : SparamCommonResidualMLP(latent={cfg.latent})")
    print(f"  Loss: S-param dB RMSE  + VICReg({cfg.use_vicreg})")

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

    section("JOINT TRAINING — Encoder + Surrogate MLP (S-param only)")
    print(f"  {'ep':>4s} | {'tr_total':>9s} {'tr_sp':>8s} {'tr_full':>8s} {'tr_sel':>8s} "
          f"{'var':>7s} {'cov':>7s} | "
          f"{'va_total':>9s} {'va_full':>8s} {'va_sel':>8s} | {'lr':>8s}")
    print("  " + "-" * 120)

    best_metric = float("inf")
    best_state = None
    log_every = max(1, cfg.epochs // 30)

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scheduler, device, cfg,
                       interp_w, train_mode=True)
        va = run_epoch(model, val_loader, None, None, device, cfg,
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
    if ckpt_save_path:
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
            torch.save({
                "model_state_dict": model.state_dict(),
                "max_len": dataset.max_len,
                "type_names": list(getattr(dataset, "type_names", [])),
                "common_curve": common_curve.detach().cpu().numpy()
                                if hasattr(common_curve, "detach") else np.asarray(common_curve),
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


def main():
    cfg = CFG()
    cfg.run_name = "surrogate_only"
    apply_preset(cfg)
    cfg.ckpt_save_path = "ckpt/surrogate_last.pt"
    # surrogate 만이라 decoder 관련 loss 가중치 무시됨. S-param weight 만 의미.
    cfg.w_sparam = 1.0
    cfg.w_cmd = 0.0
    cfg.w_prm = 0.0
    cfg.w_aux = 0.0

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    section("SURROGATE-ONLY TRAINING")
    print(f"  Device : {device}" + (f" ({torch.cuda.get_device_name(0)})"
                                     if device.type == "cuda" else ""))
    print(f"  Preset : {cfg.preset}")

    section("LOAD DATA")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset, _npy, type_ids, type_names, _sp, _ml = aem.load_multitype_data(
        cfg, script_dir,
    )

    section("SPLIT")
    train_idx, val_idx, val_idx_per_type = aem.make_stratified_split(
        cfg, dataset, type_ids, type_names,
    )

    common_curve = aem.build_common_curve_and_print_baseline(
        dataset=dataset, train_idx=train_idx,
        val_idx_per_type=val_idx_per_type, type_names=type_names,
    )

    train_surrogate(cfg, dataset, train_idx, val_idx, common_curve, device)

    section("DONE")


if __name__ == "__main__":
    main()
