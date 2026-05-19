#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sketch role labeling tool
=========================

각 type 폴더의 sample 들을 읽어서:
  1) 구조적으로 동일한 sample 들을 묶어 unique "structural variant" 발견
     (signature = 각 sketch 의 (n_lines, n_arcs, n_circles, n_loops) 튜플 in order)
  2) 각 variant 마다 대표 sample 1개의 모든 sketch 를 3D + Top view 로 그리고
     번호 라벨 (0,1,2,...) 표시
  3) 콘솔에서 sketch 마다 role 입력 받음 (frame / port_1 / gnd_1 / camera / ...)
     여러 sketch 가 같은 role 가질 수 있음 (예: 6개 모두 frame 가능)
     특정 role 없으면 "skip" 입력 (예: camera 없으면 skip)
  4) 같은 variant 에 속한 모든 sample 에 동일한 role list 를 `*_roles.json`
     sidecar 로 작성

사용:
    python label_sketches.py
    python label_sketches.py --types type1 type3       # 특정 type 만
    python label_sketches.py --skip-existing            # 이미 _roles.json 있으면 건너뜀
"""

import os
import sys
import glob
import json
import argparse
import numpy as np

import matplotlib
for _b in ("TkAgg", "Qt5Agg", "Qt6Agg", "wxAgg", "MacOSX"):
    try:
        matplotlib.use(_b)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt

# 같은 디렉토리의 AE_inverse_role_variants 에서 helper import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AE_inverse_role_variants import (
    _split_sketch_chunks,
    _chunk_topology_signature,
    json_to_real_sketches,
)


# ───────────────────────────────────────────────────────────────
# Structural variant discovery
# ───────────────────────────────────────────────────────────────
def per_sample_signature(tokens):
    """sample 구조 signature = tuple of chunk_sig (per sketch, in original order)."""
    chunks, _tail = _split_sketch_chunks(tokens)
    return tuple(_chunk_topology_signature(ch) for ch in chunks)


def group_samples_by_variant(npy_files):
    """npy 리스트를 signature 별로 그룹. dict {signature: [npy_path, ...]}."""
    groups = {}
    for f in npy_files:
        try:
            tokens = np.load(f).astype(np.int32)
        except Exception as e:
            print(f"  ! failed to load {f}: {e}")
            continue
        sig = per_sample_signature(tokens)
        groups.setdefault(sig, []).append(f)
    return groups


# ───────────────────────────────────────────────────────────────
# Plotting
# ───────────────────────────────────────────────────────────────
def render_variant_for_labeling(json_data, fig_title):
    """sketch 들을 3D + Top view 로 그리고 sketch index 를 노란 원으로 표시."""
    sketches = json_to_real_sketches(json_data)
    if not sketches:
        return None, [], []

    fig = plt.figure(figsize=(15, 7.5))
    fig.suptitle(fig_title, fontsize=13)

    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2, projection="3d")

    centers = []
    all_pts_list = []
    bbox_info = []
    for sk in sketches:
        pts = []
        for loop in sk.get("loops_3d", []):
            for p in loop:
                pts.append(np.asarray(p, dtype=float))
        if pts:
            arr = np.stack(pts, axis=0)
            centers.append(arr.mean(0))
            all_pts_list.append(arr)
            ext = (arr.max(0) - arr.min(0)).tolist()
            sorted_ext = sorted(ext, reverse=True)
            bbox_info.append({
                "bbox_max": float(sorted_ext[0]),
                "bbox_mid": float(sorted_ext[1]) if len(sorted_ext) > 1 else 0.0,
                "bbox_min": float(sorted_ext[2]) if len(sorted_ext) > 2 else 0.0,
                "area_approx": float(sorted_ext[0] * (sorted_ext[1] if len(sorted_ext) > 1 else 0.0)),
            })
        else:
            centers.append(np.zeros(3))
            all_pts_list.append(np.zeros((1, 3)))
            bbox_info.append({"bbox_max": 0, "bbox_mid": 0, "bbox_min": 0, "area_approx": 0})

    cmap = plt.get_cmap("tab20")
    for ax in (ax_3d, ax_top):
        for i, (sk, c) in enumerate(zip(sketches, centers)):
            color = cmap(i % 20)
            for loop in sk.get("loops_3d", []):
                lp = np.asarray(loop)
                if len(lp) >= 2:
                    ax.plot(lp[:, 0], lp[:, 1], lp[:, 2],
                            color=color, linewidth=2.2)
            ax.text(c[0], c[1], c[2], f"{i}",
                    color="black", fontsize=15, weight="bold",
                    ha="center", va="center", zorder=10,
                    bbox=dict(boxstyle="circle,pad=0.35",
                              fc="yellow", ec="black", lw=1.5))

    all_arr = np.concatenate(all_pts_list, axis=0)
    bmin = all_arr.min(0)
    bmax = all_arr.max(0)
    span = max((bmax - bmin).max() * 0.6, 1.0)
    mid = (bmax + bmin) / 2
    for ax in (ax_3d, ax_top):
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")

    ax_3d.set_title("3D view")
    ax_3d.view_init(elev=25, azim=-55)
    ax_top.set_title("Top view")
    ax_top.view_init(elev=89.9, azim=-90.0)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, centers, bbox_info


# ───────────────────────────────────────────────────────────────
# Console labeling
# ───────────────────────────────────────────────────────────────
def prompt_labels(n_sketches, centers, bbox_info, suggestions):
    print(f"\n  ← Suggested role names:")
    print(f"     {', '.join(suggestions)}")
    print(f"     (자유 입력 가능. 빈 Enter = 'skip')\n")

    while True:
        roles = []
        for i in range(n_sketches):
            c = centers[i]
            bi = bbox_info[i]
            ans = input(
                f"    sk{i:>2}  center=({c[0]:7.2f},{c[1]:7.2f},{c[2]:7.2f})mm"
                f"  bbox=({bi['bbox_max']:6.2f}×{bi['bbox_mid']:6.2f}×{bi['bbox_min']:6.2f})"
                f"  → role: "
            ).strip()
            if not ans:
                ans = "skip"
            roles.append(ans)

        print(f"\n  ★ Labels so far:")
        for i, r in enumerate(roles):
            print(f"     sk{i:>2}: {r}")
        ok = input("  Confirm? (y/n, Enter=y): ").strip().lower()
        if ok in ("y", "yes", ""):
            return roles
        print("  re-entering labels...\n")


# ───────────────────────────────────────────────────────────────
# Write sidecar
# ───────────────────────────────────────────────────────────────
def write_roles_json(sample_npy_path, roles, source_rep_path=None):
    out_path = sample_npy_path.replace("_tokens.npy", "_roles.json")
    payload = {
        "roles": list(roles),
        "n_sketches": len(roles),
    }
    if source_rep_path is not None:
        payload["source_rep"] = os.path.basename(source_rep_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


# ───────────────────────────────────────────────────────────────
# Per-type loop
# ───────────────────────────────────────────────────────────────
def label_type(type_name, npy_dir, suggestions, skip_existing):
    print("\n" + "=" * 64)
    print(f"  TYPE: {type_name}    dir: {npy_dir}")
    print("=" * 64)

    npy_files = sorted(glob.glob(os.path.join(npy_dir, "*_tokens.npy")))
    if not npy_files:
        print("  (no *_tokens.npy files found)")
        return

    if skip_existing:
        before = len(npy_files)
        npy_files = [
            f for f in npy_files
            if not os.path.exists(f.replace("_tokens.npy", "_roles.json"))
        ]
        print(f"  {len(npy_files)} samples need labels  (skipped {before - len(npy_files)} already-labeled)")
    else:
        print(f"  {len(npy_files)} samples found")

    if not npy_files:
        return

    groups = group_samples_by_variant(npy_files)
    print(f"  → {len(groups)} unique structural variant(s)")

    # 가장 흔한 variant 부터
    sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    for var_idx, (sig, members) in enumerate(sorted_groups, start=1):
        print(f"\n  --- Variant {var_idx}/{len(groups)}  ({len(members)} sample(s)) ---")
        print(f"      sketch signatures (Line, Arc, Circle, SOL=loop):")
        for k, s in enumerate(sig):
            print(f"        sk{k:>2}: L{s[0]:>2}  A{s[1]:>2}  C{s[2]:>2}  S{s[3]:>2}")

        rep_npy = members[0]
        rep_json = rep_npy.replace("_tokens.npy", "_deepcad.json")
        if not os.path.exists(rep_json):
            print(f"      ! JSON missing for rep {os.path.basename(rep_npy)} — skip")
            continue
        with open(rep_json, "r", encoding="utf-8") as f:
            jd = json.load(f)

        fig, centers, bbox_info = render_variant_for_labeling(
            jd, fig_title=f"{type_name} · variant {var_idx}/{len(groups)} · rep: {os.path.basename(rep_npy)}",
        )
        if fig is None:
            print("      ! no sketches in rep — skip")
            continue
        plt.show(block=False)
        plt.pause(0.4)

        roles = prompt_labels(len(centers), centers, bbox_info, suggestions)
        plt.close(fig)

        n_written = 0
        for m in members:
            try:
                tk = np.load(m).astype(np.int32)
            except Exception:
                print(f"      ! failed to load {os.path.basename(m)} — skip")
                continue
            n_sk_m = len(per_sample_signature(tk))
            if n_sk_m != len(roles):
                print(f"      ! {os.path.basename(m)}: n_sk={n_sk_m} != n_roles={len(roles)} — skip")
                continue
            write_roles_json(m, roles, source_rep_path=rep_npy)
            n_written += 1
        print(f"      ✓ wrote {n_written}/{len(members)} *_roles.json")


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────
DEFAULT_SUGGESTIONS = [
    "frame",
    "port_1", "port_2", "port_3",
    "gnd_1", "gnd_2", "gnd_3",
    "camera",
    "skip",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--types", nargs="+", default=["type1", "type2", "type3"],
        help="라벨링할 type 들 (default: type1 type2 type3)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="이미 *_roles.json 있는 sample 은 건너뜀",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    type_dirs = {
        "type1": os.path.join(here, "hfss_results", "step_test"),
        "type2": os.path.join(here, "hfss_results", "step_test1"),
        "type3": os.path.join(here, "hfss_results", "step_test2"),
    }

    for tname in args.types:
        if tname not in type_dirs:
            print(f"  (unknown type {tname}, skip)")
            continue
        tdir = type_dirs[tname]
        if not os.path.isdir(tdir):
            print(f"  (skip {tname}: dir not found {tdir})")
            continue
        label_type(tname, tdir, DEFAULT_SUGGESTIONS, args.skip_existing)

    print("\n  ★ All done.")


if __name__ == "__main__":
    main()
