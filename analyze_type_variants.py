#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze structural variants within a single type
=================================================

한 type 폴더 (예: type1 = hfss_results/step_test) 안의 모든 sample 을 읽어,
구조적으로 어떤 변형들이 섞여 있는지 자동 그룹화 + 리포트 + 시각화.

변형 식별 방법:
  각 sample 을 "chunk signature 의 튜플" 로 표현:
    - chunk 마다: (role_name, n_LINE, n_ARC, n_CIRCLE, n_SOL_loops)
    - sample = 그 chunk signature 들의 ordered tuple

  → 같은 signature 끼리 묶으면 한 그룹 = 같은 토폴로지의 변형 (수치만 다름)
     다른 signature = 토폴로지 자체가 다른 변형 (loop 수, edge 수 등)

출력:
  - 콘솔: rank | count | signature
  - txt: type1_variants.txt — 모든 그룹 + sample 파일명
  - png: type1_variants.png — top N 그룹의 대표 3D 구조

CONFIG 는 파일 맨 위. 실행:  python analyze_type_variants.py
"""

import os
import sys
import json
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ═══════════════════════════════════════════════════════════════
#  ★ CONFIG  ★    (수정은 여기에서만)
# ═══════════════════════════════════════════════════════════════

# 분석할 type 폴더 (token npy + deepcad json 들이 있는 곳)
TYPE_DIR        = "hfss_results/step_test"     # type1

# 결과 저장 폴더
SAVE_DIR        = "type_variants"

# 시각화
SHOW_FIGURES    = True
TOP_N_GROUPS    = 6        # figure 에 보여줄 가장 흔한 변형 개수

# JSON 의 role name 사용 (None 으로 두면 token 만 보고 분류 — 더 거친 그룹화)
USE_JSON_NAMES  = True

# token signature 에 param 의 통계도 포함할지 (True 면 같은 토폴로지여도 param 분포 다르면 분리됨 — 보통 False)
INCLUDE_PARAM_HISTOGRAM = False


# ═══════════════════════════════════════════════════════════════

# token 상수 (surrogate.py 와 동일)
LINE = 0
ARC = 1
CIRCLE = 2
SOL = 3
EXT = 4
EOS = 5
ROLE = 6


def split_chunks(tokens):
    """tokens → list of chunks (SOL...EXT 구간) + tail.

    ROLE 토큰은 chunk 의 일부가 아니라 헤더로 취급. 여기선 chunk 시그니처용으로만 사용.
    """
    chunks = []
    cur = []
    in_chunk = False
    for row in tokens:
        c = int(row[0])
        if c < 0 or c == EOS:
            break
        if c == ROLE:
            # 다음 SOL 까지의 헤더, 다음 chunk 의 role 정보는 별도로 호출자가 처리
            continue
        if c == SOL and not in_chunk:
            in_chunk = True
            cur = [row.copy()]
        elif in_chunk and c == EXT:
            cur.append(row.copy())
            chunks.append(np.stack(cur, axis=0))
            cur = []
            in_chunk = False
        elif in_chunk:
            cur.append(row.copy())
    return chunks


def extract_role_sequence(tokens):
    """ROLE 토큰의 param[0] 값들을 순서대로 (chunk 별 role id, JSON name 매핑 X)."""
    seq = []
    for row in tokens:
        c = int(row[0])
        if c < 0 or c == EOS:
            break
        if c == ROLE and len(row) > 1 and int(row[1]) >= 0:
            seq.append(int(row[1]))
    return seq


def chunk_signature(chunk):
    """chunk → (n_LINE, n_ARC, n_CIRCLE, n_SOL_loops). loop 수는 SOL 개수."""
    cmds = chunk[:, 0]
    return (
        int((cmds == LINE).sum()),
        int((cmds == ARC).sum()),
        int((cmds == CIRCLE).sum()),
        int((cmds == SOL).sum()),
    )


def sample_signature(tokens, role_names):
    """sample → tuple of (role_name, chunk_signature) per chunk."""
    chunks = split_chunks(tokens)
    sig = []
    for i, ch in enumerate(chunks):
        rs = chunk_signature(ch)
        role = (role_names[i] if role_names and i < len(role_names) else None)
        sig.append((role, rs))
    return tuple(sig)


def sig_to_str(sig):
    parts = []
    for role, (l, a, c, sol) in sig:
        rname = role if role else "?"
        parts.append(f"{rname}(L{l}A{a}C{c}S{sol})")
    return " | ".join(parts)


# ══════════════════════════════════════════════════════════════
# 3D rendering helpers (surrogate.py 의 검증된 구현 그대로 복사)
# ══════════════════════════════════════════════════════════════
_PAL = [
    "#2E4172",   # deep slate blue
    "#8B5A3C",   # sienna brown
    "#3F6E5C",   # deep teal-green
    "#6B5B7A",   # muted mauve
    "#555555",   # charcoal grey
    "#8B7E3D",   # dark gold
]


def _v3(d):
    return np.array([d["x"], d["y"], d["z"]], dtype=float)


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


def json_to_real_sketches(json_data):
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
                        for cx2, cy2 in _circle_pts(cp["x"], cp["y"], r):
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
        color = _PAL[idx % len(_PAL)]
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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_dir_abs = TYPE_DIR if os.path.isabs(TYPE_DIR) else os.path.join(script_dir, TYPE_DIR)

    if not os.path.isdir(type_dir_abs):
        print(f"  ✗ type dir not found: {type_dir_abs}")
        sys.exit(1)

    npy_files = sorted(glob.glob(os.path.join(type_dir_abs, "*_tokens.npy")))
    if not npy_files:
        print(f"  ✗ no *_tokens.npy in {type_dir_abs}")
        sys.exit(1)

    type_name = os.path.basename(type_dir_abs.rstrip(os.sep))
    print(f"\n  Analyzing type folder: {type_dir_abs}")
    print(f"  found {len(npy_files)} samples\n")

    # ── 1) 모든 sample 읽고 signature 계산 ──
    samples = []
    for npy in npy_files:
        try:
            tokens = np.load(npy).astype(np.int32)
        except Exception as e:
            print(f"    [warn] failed to load {npy}: {e}")
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
        samples.append({"npy": npy, "tokens": tokens, "names": names, "sig": sig})

    # ── 2) signature 별 그룹화 ──
    sig_groups = {}
    for s in samples:
        sig_groups.setdefault(s["sig"], []).append(s)

    groups_sorted = sorted(sig_groups.items(), key=lambda kv: -len(kv[1]))

    # ── 3) 콘솔 출력 ──
    print(f"  {len(groups_sorted)} unique structural variants found:\n")
    print(f"  {'rank':>4s} | {'count':>5s} | structure signature")
    print("  -----+-------+" + "-" * 60)
    for r, (sig, group) in enumerate(groups_sorted):
        print(f"  {r + 1:>4d} | {len(group):>5d} | {sig_to_str(sig)}")

    # ── 4) 리포트 txt 저장 ──
    os.makedirs(SAVE_DIR, exist_ok=True)
    report_path = os.path.join(SAVE_DIR, f"{type_name}_variants.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Variant analysis of {type_dir_abs}\n")
            f.write(f"# total samples       : {len(samples)}\n")
            f.write(f"# unique variants     : {len(groups_sorted)}\n")
            f.write(f"# JSON role names used: {USE_JSON_NAMES}\n\n")
            for r, (sig, group) in enumerate(groups_sorted):
                f.write(f"== Variant {r + 1}: {len(group)} samples ==\n")
                f.write(f"  signature: {sig_to_str(sig)}\n")
                f.write(f"  detail   : {sig}\n")
                f.write(f"  samples:\n")
                for s in group[:20]:
                    f.write(f"    {os.path.basename(s['npy'])}\n")
                if len(group) > 20:
                    f.write(f"    ... +{len(group) - 20} more\n")
                f.write("\n")
        print(f"\n  ✓ report saved → {report_path}")
    except Exception as e:
        print(f"  ⚠ failed to save report: {e}")

    # ── 5) 시각화: 상위 N 그룹의 대표 sample 3D ──
    if SHOW_FIGURES:
        n_show = min(TOP_N_GROUPS, len(groups_sorted))
        if n_show > 0:
            ncol = min(n_show, 3)
            nrow = (n_show + ncol - 1) // ncol
            fig = plt.figure(figsize=(5 * ncol, 5 * nrow), facecolor="white")
            fig.suptitle(
                f"Structural variants in [{type_name}]  "
                f"— {len(groups_sorted)} unique, top {n_show} shown  "
                f"({len(samples)} total samples)",
                fontsize=11, color="#222",
            )

            for k in range(n_show):
                sig, group = groups_sorted[k]
                rep = group[0]
                ax = fig.add_subplot(nrow, ncol, k + 1, projection="3d")
                jp = rep["npy"].replace("_tokens.npy", "_deepcad.json")
                if os.path.exists(jp):
                    try:
                        with open(jp, "r", encoding="utf-8") as f:
                            jd = json.load(f)
                        sketches = json_to_real_sketches(jd)
                    except Exception:
                        sketches = []
                else:
                    sketches = []

                if sketches:
                    pts = _render_3d_real_simple(ax, sketches)
                    _set_axes_struct(ax, pts)
                    ax.view_init(elev=25, azim=-55)
                else:
                    ax.text2D(0.5, 0.5, "(no JSON)",
                              ha="center", va="center", color="#888",
                              fontsize=11, transform=ax.transAxes)

                # 짧은 signature 라벨
                sig_short = sig_to_str(sig)
                if len(sig_short) > 70:
                    sig_short = sig_short[:67] + "..."
                _style_struct_ax(
                    ax,
                    f"Variant {k + 1}  ·  {len(group)} samples\n{sig_short}",
                )

            plt.tight_layout(rect=[0, 0, 1, 0.95])

            fig_path = os.path.join(SAVE_DIR, f"{type_name}_variants.png")
            try:
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                print(f"  ✓ figure saved → {fig_path}")
            except Exception as e:
                print(f"  ⚠ failed to save figure: {e}")

            plt.show()


if __name__ == "__main__":
    main()
