#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze structural variants across ALL types
=============================================

각 type 폴더에 대해 analyze_type_variants.py 와 동일한 그룹화를 돌려서
"이 type 안에 골격이 몇 개나 있나" 를 한꺼번에 보여줌.

옵티마이저로 한 type 골라서 좌표만 search 하려면 그 type 의 골격이
가능한 한 1개로 좁혀져 있어야 함. 이 스크립트로 빠르게 확인:

  - n_variants == 1  →  ✓  완전한 단일 골격 (좌표 search 만으로 충분)
  - n_variants >  1  →  ⚠  여러 골격 섞임 → 가장 흔한 variant 1개를
                            template 으로 박거나 variant 별로 따로 search

출력:
  - 콘솔: 각 type 별 variant 표 + 마지막에 summary 표
  - txt : all_types_variants_summary.txt
  - png : all_types_variants_top1.png (각 type 의 가장 흔한 variant 1개 3D)

실행:  python analyze_type_variants2.py
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# v1 의 검증된 helper 들을 그대로 재사용 (signature, rendering 모두)
from analyze_type_variants import (
    USE_JSON_NAMES,
    sample_signature,
    sig_to_str,
    json_to_real_sketches,
    _render_3d_real_simple,
    _style_struct_ax,
    _set_axes_struct,
)


# ═══════════════════════════════════════════════════════════════
#  ★ CONFIG  ★
# ═══════════════════════════════════════════════════════════════

# 검사할 type 폴더들 (label, dir).  dir 은 상대경로면 이 스크립트 기준.
TYPE_DIRS = [
    ("type1", "hfss_results/step_test"),
    ("type2", "hfss_results/step_test1"),
    ("type3", "hfss_results/step_test2"),
]

# 결과 저장 폴더
SAVE_DIR = "type_variants"

# 그림 표시 / 저장
SHOW_FIGURES = True

# 콘솔에 type 별 표시할 상위 variant 개수
TOP_N_PER_TYPE = 5


# ═══════════════════════════════════════════════════════════════


def analyze_one_type(type_label, type_dir):
    """한 type 의 모든 sample → signature grouping → groups_sorted 반환."""
    if not os.path.isdir(type_dir):
        print(f"  ✗ [{type_label}] type dir not found: {type_dir}")
        return None, []

    npy_files = sorted(glob.glob(os.path.join(type_dir, "*_tokens.npy")))
    if not npy_files:
        print(f"  ✗ [{type_label}] no *_tokens.npy in {type_dir}")
        return None, []

    samples = []
    for npy in npy_files:
        try:
            tokens = np.load(npy).astype(np.int32)
        except Exception as e:
            print(f"    [warn] {os.path.basename(npy)}: {e}")
            continue

        names = None
        if USE_JSON_NAMES:
            jp = npy.replace("_tokens.npy", "_deepcad.json")
            if os.path.exists(jp):
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        jd = json.load(f)
                    solids = jd.get("metadata", {}).get("solids", [])
                    nm = [str(s.get("name", "")) for s in solids]
                    names = nm if any(nm) else None
                except Exception:
                    names = None

        sig = sample_signature(tokens, names)
        samples.append({"npy": npy, "sig": sig, "names": names})

    sig_groups = {}
    for s in samples:
        sig_groups.setdefault(s["sig"], []).append(s)
    groups_sorted = sorted(sig_groups.items(), key=lambda kv: -len(kv[1]))

    return samples, groups_sorted


def print_type_report(type_label, type_dir, samples, groups_sorted):
    print(f"\n  [{type_label}]  dir = {type_dir}")
    if samples is None or not samples:
        print(f"    (no data)")
        return
    n_var = len(groups_sorted)
    print(f"    samples = {len(samples)}   unique variants = {n_var}")
    n_show = min(TOP_N_PER_TYPE, n_var)
    print(f"    {'rank':>4s} | {'count':>5s} | structure signature")
    print(f"    -----+-------+" + "-" * 60)
    for r in range(n_show):
        sig, group = groups_sorted[r]
        print(f"    {r + 1:>4d} | {len(group):>5d} | {sig_to_str(sig)[:60]}")
    if n_var > n_show:
        print(f"    ...  ({n_var - n_show} more variants)")


def print_summary_table(per_type_results):
    print("\n" + "═" * 76)
    print("  SUMMARY — 각 type 의 골격 개수")
    print("═" * 76)
    print(f"  {'type':<10s} | {'samples':>7s} | {'variants':>8s} | "
          f"{'top1 share':>10s} | single skeleton?")
    print("  " + "-" * 74)
    for type_label, type_dir, samples, groups in per_type_results:
        if not samples:
            print(f"  {type_label:<10s} | {'-':>7s} | {'-':>8s} | "
                  f"{'-':>10s} | (no data)")
            continue
        n_sam = len(samples)
        n_var = len(groups)
        top1 = len(groups[0][1]) if groups else 0
        top1_pct = 100.0 * top1 / max(1, n_sam)
        if n_var == 1:
            tag = "✓  YES — 단일 골격"
        elif top1_pct >= 95.0:
            tag = f"~  거의 단일 (top1 {top1_pct:.1f}%)"
        else:
            tag = f"⚠  NO — {n_var} variants 섞임"
        print(f"  {type_label:<10s} | {n_sam:>7d} | {n_var:>8d} | "
              f"{top1_pct:>9.1f}% | {tag}")
    print("═" * 76)


def save_summary_txt(per_type_results, save_path):
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("# Variant analysis across all types\n\n")
            for type_label, type_dir, samples, groups in per_type_results:
                f.write(f"== {type_label}  ({type_dir}) ==\n")
                if not samples:
                    f.write("  (no data)\n\n")
                    continue
                f.write(f"  samples         : {len(samples)}\n")
                f.write(f"  unique variants : {len(groups)}\n")
                for r, (sig, group) in enumerate(groups):
                    f.write(f"  Variant {r + 1}: {len(group)} samples\n")
                    f.write(f"    signature: {sig_to_str(sig)}\n")
                    f.write(f"    detail   : {sig}\n")
                f.write("\n")
        print(f"  ✓ summary saved → {save_path}")
    except Exception as e:
        print(f"  ⚠ failed to save summary: {e}")


def make_top1_figure(per_type_results, save_path):
    valid = [(lbl, grps[0][0], grps[0][1])
             for lbl, _, sam, grps in per_type_results
             if sam and grps]
    if not valid:
        return
    ncol = len(valid)
    fig = plt.figure(figsize=(5 * ncol, 5), facecolor="white")
    fig.suptitle(
        "Most common variant of each type (Variant 1)",
        fontsize=11, color="#222",
    )
    for k, (type_label, sig, group) in enumerate(valid):
        ax = fig.add_subplot(1, ncol, k + 1, projection="3d")
        rep = group[0]
        jp = rep["npy"].replace("_tokens.npy", "_deepcad.json")
        sketches = []
        if os.path.exists(jp):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    jd = json.load(f)
                sketches = json_to_real_sketches(jd)
            except Exception:
                sketches = []
        if sketches:
            pts = _render_3d_real_simple(ax, sketches)
            _set_axes_struct(ax, pts)
            ax.view_init(elev=25, azim=-55)
        else:
            ax.text2D(0.5, 0.5, "(no JSON)",
                      ha="center", va="center", color="#888",
                      fontsize=11, transform=ax.transAxes)
        sig_short = sig_to_str(sig)
        if len(sig_short) > 60:
            sig_short = sig_short[:57] + "..."
        _style_struct_ax(
            ax,
            f"{type_label}  ·  {len(group)} samples\n{sig_short}",
        )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    try:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ figure saved → {save_path}")
    except Exception as e:
        print(f"  ⚠ failed to save figure: {e}")
    plt.show()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, SAVE_DIR), exist_ok=True)

    per_type_results = []
    for type_label, type_dir in TYPE_DIRS:
        type_dir_abs = (type_dir if os.path.isabs(type_dir)
                        else os.path.join(script_dir, type_dir))
        samples, groups = analyze_one_type(type_label, type_dir_abs)
        per_type_results.append((type_label, type_dir_abs, samples, groups))
        print_type_report(type_label, type_dir_abs, samples, groups)

    print_summary_table(per_type_results)

    save_summary_txt(
        per_type_results,
        os.path.join(script_dir, SAVE_DIR, "all_types_variants_summary.txt"),
    )

    if SHOW_FIGURES:
        make_top1_figure(
            per_type_results,
            os.path.join(script_dir, SAVE_DIR, "all_types_variants_top1.png"),
        )


if __name__ == "__main__":
    main()
