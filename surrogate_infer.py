#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surrogate inference viewer
===========================

surrogate.py 로 학습/저장한 ckpt (예: ckpt/surrogate_last.pt) 를 로드하고,
파일 선택 창에서 토큰 npy 를 고르면 그 입력에 대한 S-param 예측 곡선을 보여줌.

ckpt 는 self-contained (source_code + cfg_dict + common_curve 내장) 라서
surrogate.py 가 없어도 ckpt 만으로 동작. 우선순위:
  1) ckpt 안에 내장된 source_code  (학습 당시 architecture 와 100% 일치 보장)
  2) 같은 폴더의 surrogate.py      (fallback)

입력 npy 형식: (L, 17) int — [cmd, p0..p15], param 0~1023, PAD=-1
출력: S11/S22/S33 dB 곡선 (figure + 콘솔 요약)

실행:  python surrogate_infer.py
       (창 1: ckpt 선택 → 창 2: npy 선택 (여러 개 가능) → figure)
"""

import os
import sys
import tempfile
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════
#  ★ CONFIG ★
# ═══════════════════════════════════════════════════════════════

# ckpt 기본 경로. 존재하면 파일 선택 창 없이 바로 사용. 없으면 창 띄움.
DEFAULT_CKPT = "ckpt/surrogate_last.pt"

# 한 figure 에 곡선 겹쳐 그릴지 (True), npy 마다 행 분리할지 (False)
OVERLAY = False

# ═══════════════════════════════════════════════════════════════


def pick_file(title, patterns, initialdir=".", multiple=False):
    """tkinter 파일 선택 창. GUI 불가 환경이면 콘솔 입력으로 fallback."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if multiple:
            paths = filedialog.askopenfilenames(
                title=title, filetypes=patterns, initialdir=initialdir)
            root.destroy()
            return list(paths)
        path = filedialog.askopenfilename(
            title=title, filetypes=patterns, initialdir=initialdir)
        root.destroy()
        return [path] if path else []
    except Exception as e:
        print(f"  (GUI 사용 불가: {e}) — 경로를 직접 입력하세요.")
        raw = input(f"  {title}: ").strip()
        return [p for p in raw.split() if p] if raw else []


def load_module_for_ckpt(ckpt, script_dir):
    """SurrogateModel class 가 정의된 모듈 확보.
    1순위: ckpt 내장 source_code (학습 시점 코드 그대로)
    2순위: 로컬 surrogate.py
    """
    src = ckpt.get("source_code")
    if src:
        try:
            tmp = tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8")
            tmp.write(src)
            tmp.close()
            spec = importlib.util.spec_from_file_location(
                "surrogate_from_ckpt", tmp.name)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, f"ckpt 내장 source ({ckpt.get('source_filename', '?')})"
        except Exception as e:
            print(f"  ⚠ ckpt 내장 source 로드 실패: {e} — 로컬 surrogate.py 시도")

    sys.path.insert(0, script_dir)
    import surrogate as mod
    return mod, "로컬 surrogate.py"


def build_model_from_ckpt(ckpt, mod, device):
    cfg = ckpt.get("cfg_dict")
    if not cfg:
        raise RuntimeError("ckpt 에 cfg_dict 가 없음 — architecture 재현 불가")
    model = mod.SurrogateModel(
        max_len=ckpt["max_len"],
        d_model=cfg["d_model"], d_param=cfg["d_param"], nhead=cfg["nhead"],
        n_enc=cfg["n_enc"], d_ff=cfg["d_ff"], latent=cfg["latent"],
        dropout=cfg["dropout"], n_pool=cfg["n_pool"],
        n_freq_bands=cfg["n_freq_bands"],
        n_freq=cfg["n_freq"], common_curve=ckpt["common_curve"],
        mlp_hidden_mult=cfg["mlp_hidden_mult"], mlp_dropout=cfg["mlp_dropout"],
        residual_scale=cfg["residual_scale"],
        zero_init_residual=cfg["zero_init_residual"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def parse_tokens_txt(path, mod):
    """surrogate.py 의 show_sample_input_sequences 가 저장한 txt 형식 파서.

    한 줄 예: "[   0] | ROLE   |     0    -1    -1 ..." (param 16개)
    '#' 로 시작하는 줄은 주석. 빈 줄도 무시.
    """
    name_to_cmd = {v: k for k, v in mod.CMD_NAME.items()}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln_no, raw in enumerate(f, 1):
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) != 3:
                raise ValueError(f"{path}:{ln_no}: '|' 3분할 안 됨: {raw!r}")
            cname = parts[1].upper()
            if cname not in name_to_cmd:
                raise ValueError(f"{path}:{ln_no}: 알 수 없는 cmd '{cname}'")
            params = parts[2].split()
            if len(params) != 16:
                raise ValueError(
                    f"{path}:{ln_no}: param 개수 {len(params)} != 16")
            row = [name_to_cmd[cname]] + [int(p) for p in params]
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: 토큰 행이 하나도 없음")
    return np.array(rows, dtype=np.int32)


def load_tokens_any(path, mod):
    """확장자 보고 npy 또는 txt 로 로드. 둘 다 (L, 17) int 반환."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        if path.endswith("_tokens_float.npy"):
            raise ValueError(
                "_tokens_float.npy 는 정규화된 float 데이터라 모델 입력으로 안 됨. "
                "_tokens.npy 를 골라.")
        t = np.load(path).astype(np.int32)
    elif ext == ".txt":
        t = parse_tokens_txt(path, mod)
    else:
        raise ValueError(f"지원 안 함: {ext} (.npy 또는 .txt 만)")
    if t.ndim != 2 or t.shape[1] != 17:
        raise ValueError(f"shape 이상: {t.shape} (기대: (L, 17))")
    return t


def prep_tokens(npy_path, max_len, mod):
    """입력 → (1, max_len, 17) float32 tensor. EOS 보장 + PAD(-1) 패딩."""
    t = load_tokens_any(npy_path, mod)
    t = mod.ensure_eos_when_truncated(t, max_len)
    L = t.shape[0]
    if L < max_len:
        pad = np.full((max_len - L, 17), mod.PAD_V, dtype=np.int32)
        t = np.concatenate([t, pad], axis=0)
    return torch.tensor(t, dtype=torch.float32).unsqueeze(0), L


def token_summary(path, mod):
    t = load_tokens_any(path, mod)
    cmds = t[:, 0]
    n = {name: int((cmds == code).sum()) for code, name in mod.CMD_NAME.items()}
    return (f"L={t.shape[0]}  ROLE={n['ROLE']} SOL={n['SOL']} EXT={n['EXT']} "
            f"LINE={n['LINE']} ARC={n['ARC']} CIRCLE={n['CIRCLE']}")


@torch.no_grad()
def predict(model, x, device):
    pred_sel, _z = model(x.to(device))
    return pred_sel.cpu().numpy()[0]            # (n_freq, 3)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── 1) ckpt 선택 ──
    default_ckpt = os.path.join(script_dir, DEFAULT_CKPT)
    if os.path.exists(default_ckpt):
        ckpt_path = default_ckpt
        print(f"  ckpt: {ckpt_path} (기본 경로 자동 사용)")
    else:
        picks = pick_file(
            "Surrogate ckpt (.pt) 선택",
            [("PyTorch ckpt", "*.pt"), ("All files", "*.*")],
            initialdir=script_dir)
        if not picks:
            print("  ✗ ckpt 미선택 — 종료")
            sys.exit(0)
        ckpt_path = picks[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mod, mod_src = load_module_for_ckpt(ckpt, script_dir)
    print(f"  model code: {mod_src}")

    model = build_model_from_ckpt(ckpt, mod, device)
    n_params = sum(p.numel() for p in model.parameters())
    freqs = np.asarray(ckpt["freqs_sel"], dtype=np.float32)
    max_len = int(ckpt["max_len"])
    best = ckpt.get("best_val_metric")
    print(f"  ✓ model loaded: {n_params:,} params, max_len={max_len}, "
          f"n_freq={len(freqs)}"
          + (f", val RMSE dB={best:.3f}" if best is not None else ""))
    if ckpt.get("type_names"):
        print(f"  trained types: {ckpt['type_names']}")

    # ── 2) npy 선택 (여러 개 가능) ──
    npy_paths = pick_file(
        "입력 토큰 npy / txt 선택 (여러 개 가능)",
        [("Token npy/txt", ("*.npy", "*.txt")),
         ("Token npy", "*.npy"), ("Token txt", "*.txt"),
         ("All files", "*.*")],
        initialdir=script_dir, multiple=True)
    if not npy_paths:
        print("  ✗ npy 미선택 — 종료")
        sys.exit(0)

    # ── 3) 추론 + 출력 ──
    labels = ["S11", "S22", "S33"]
    colors = ["#2E4172", "#8B5A3C", "#3F6E5C"]
    results = []
    for p in npy_paths:
        try:
            x, L = prep_tokens(p, max_len, mod)
            pred = predict(model, x, device)            # (n_freq, 3)
            results.append((p, L, pred))
            print(f"\n  [{os.path.basename(p)}]  {token_summary(p, mod)}")
            for si, lab in enumerate(labels):
                c = pred[:, si]
                fmin = freqs[int(np.argmin(c))]
                print(f"    {lab}: min={c.min():7.2f} dB @ {fmin:.2f} GHz   "
                      f"max={c.max():7.2f} dB   mean={c.mean():7.2f} dB")
        except Exception as e:
            print(f"  ✗ {os.path.basename(p)}: {type(e).__name__}: {e}")

    if not results:
        print("  ✗ 추론 성공한 npy 없음 — 종료")
        sys.exit(1)

    # ── 4) figure ──
    if OVERLAY:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="white")
        for si, (ax, lab) in enumerate(zip(axes, labels)):
            for (p, L, pred) in results:
                ax.plot(freqs, pred[:, si], lw=1.4,
                        label=os.path.basename(p))
            ax.set_title(lab, fontsize=10)
            ax.set_xlabel("Freq (GHz)", fontsize=9)
            ax.set_ylabel("dB", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(f"Surrogate prediction — {os.path.basename(ckpt_path)}",
                     fontsize=11)
    else:
        nrow = len(results)
        fig, axes = plt.subplots(nrow, 3, figsize=(14, 3.6 * nrow),
                                 facecolor="white", squeeze=False)
        for ri, (p, L, pred) in enumerate(results):
            for si, lab in enumerate(labels):
                ax = axes[ri][si]
                ax.plot(freqs, pred[:, si], lw=1.5, color=colors[si])
                ax.set_title(f"{os.path.basename(p)} — {lab}", fontsize=9)
                ax.set_xlabel("Freq (GHz)", fontsize=8)
                ax.set_ylabel("dB", fontsize=8)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=8)
        fig.suptitle(
            f"Surrogate prediction — ckpt: {os.path.basename(ckpt_path)}",
            fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
