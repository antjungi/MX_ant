#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze structural variants per type (1개 이상의 type 폴더)
============================================================

TYPE_DIRS 의 각 type 폴더를 순회하며, 폴더 안의 모든 sample 을 읽어
구조적으로 어떤 변형들이 섞여 있는지 자동 그룹화 + 리포트 + 시각화.
마지막에 모든 type 의 골격 개수를 한 표로 요약.

변형 식별 방법:
  각 sample 을 "chunk signature 의 튜플" 로 표현:
    - chunk 마다: (role_name, n_LINE, n_ARC, n_CIRCLE, n_SOL_loops)
    - sample = 그 chunk signature 들의 ordered tuple

  → 같은 signature 끼리 묶으면 한 그룹 = 같은 토폴로지의 변형 (수치만 다름)
     다른 signature = 토폴로지 자체가 다른 변형 (loop 수, edge 수 등)

출력 (type 별):
  - 콘솔: rank | count | signature
  - txt : {type_label}_variants.txt — 모든 그룹 + sample 파일명
  - png : {type_label}_variants.png — top N 그룹의 대표 3D 구조
출력 (전체):
  - 콘솔: SUMMARY 표 — 각 type 의 골격 개수 / single skeleton 여부
  - png : all_types_top1.png — 각 type 의 가장 흔한 variant 1개 비교

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

# 분석할 type 폴더들.  (type_label, dir).  dir 은 상대경로면 이 스크립트 기준.
# 1개만 넣으면 단일 type 분석, 여러 개면 비교 + 요약 표.
TYPE_DIRS = [
    ("type1", "hfss_results/step_test"),
    ("type2", "hfss_results/step_test1"),
    ("type3", "hfss_results/step_test2"),
]

# 결과 저장 폴더
SAVE_DIR        = "type_variants"

# 시각화
SHOW_FIGURES    = True
TOP_N_GROUPS    = 6        # figure 에 보여줄 가장 흔한 변형 개수
TOP_N_CONSOLE   = 5        # 콘솔에 type 별 표시할 상위 variant 개수

# JSON 의 role name 사용 (None 으로 두면 token 만 보고 분류 — 더 거친 그룹화)
USE_JSON_NAMES  = True

# 주의: 파라미터 (좌표값) 는 grouping 에서 항상 무시됨. cmd 종류와 개수, 그리고 chunk 순서만 비교.

# ── ★ Variant 후처리 머지 ★ ──
# exact 그룹화 후, "사실상 같은 골격" 인 variant 들을 합쳐줌.
# 두 variant 가 머지되는 조건:
#   1) chunk 개수와 순서가 같고, 각 chunk 의 role 이름이 같음
#   2) 각 chunk 의 SOL(loop) 개수가 정확히 같음 — loop 수 다르면 진짜 다른 골격
#   3) 각 chunk 의 LINE/ARC/CIRCLE 개수가 ±COUNT_TOLERANCE 안
# 예: tolerance=2 면 L30A5C0S1 와 L32A5C0S1 가 합쳐짐 (양자화/edge 병합 노이즈).
#     0 으로 두면 exact 동작 (머지 안 함).
COUNT_TOLERANCE = 2

# ── ★ Canonicalize Preview ★ ──
# True 면 각 type 의 exact variant 1 (가장 흔한) 을 template 으로 잡고,
# variant 2 의 sample 1개를 template 의 LINE 개수에 맞춰 강제 변환했을 때
# geometry 가 얼마나 왜곡되는지 2D 로 비교 (sample 파일은 안 건드림).
# 결과: {type_variants/}{type_label}_canonicalize_preview.png
CANONICALIZE_PREVIEW = True

# ── ★ Canonicalize Apply (sample 파일 실제 수정) ★ ──
# True 면 각 merged group 안의 dominant exact-skeleton 을 template 으로 잡고,
# 같은 group 안 다른 sample 들의 _tokens.npy 를 template 의 LINE 개수에 맞춰
# 토큰 수준에서 강제 변환해 저장. 원본 _tokens.npy 는 자동 백업.
#
# 주의:
#   - _deepcad.json / _tokens_float.npy 는 안 건드림 → tokens.npy 와 약간 불일치
#     하지만 surrogate 학습 입력은 _tokens.npy 라 영향 없음. 분석/시각화는
#     _deepcad.json 기반이라 그것도 그대로.
#   - 백업은 {type_dir}{CANONICALIZE_BACKUP_SUFFIX}/ 폴더에 같은 파일명으로 저장.
APPLY_CANONICALIZE = False
CANONICALIZE_BACKUP_SUFFIX = "_canon_backup"

# True 면 COUNT_TOLERANCE / SOL 조건 무시하고 "타입 안 모든 sample" 을 한 그룹으로
# 취급해서 dominant exact-skeleton 으로 통일.  결과: 타입별 정확히 1개 골격.
# 주의: 너무 다른 topology (예: 1 loop vs 2 loop) 까지 강제로 합치면 일부 sample 은
# canonicalize 가 깨끗하게 안 될 수 있음. preview 보고 OK 일 때만 사용.
FORCE_SINGLE_SKELETON_PER_TYPE = False

# ── ★ Variant 제외 ★ ──
# type 별로 따로 지정.  키 = TYPE_DIRS 의 label, 값 = 제외할 rank 리스트.
# 빈 dict {} 또는 키가 없는 type 은 아무 것도 안 함 (기본).
# 예시:
#   EXCLUDE_VARIANTS_PER_TYPE = {"type1": [3]}            → type1 의 variant 3
#   EXCLUDE_VARIANTS_PER_TYPE = {"type1": [3], "type2": [2, 5]}
EXCLUDE_VARIANTS_PER_TYPE = {}

# 어떻게 제외할지:  "move" (안전, 되돌릴 수 있음)  /  "delete" (영구)
#   "move" → {type_dir}_excluded/ 폴더로 이동.
#            나중에 다시 쓰려면 그 폴더에서 원래 자리로 복사하면 됨.
EXCLUDE_MODE = "move"
EXCLUDE_FOLDER_SUFFIX = "_excluded"   # 이동될 폴더 이름의 suffix


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
    """chunk → (n_LINE, n_ARC, n_CIRCLE, n_SOL_loops).  파라미터 값은 무시."""
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


def _can_merge_sigs(sig_a, sig_b, tol):
    """두 sample signature 가 머지 가능한 지 검사."""
    if len(sig_a) != len(sig_b):
        return False
    for (role_a, ca), (role_b, cb) in zip(sig_a, sig_b):
        if role_a != role_b:
            return False
        la, aa, ca_c, sa = ca
        lb, ab, cb_c, sb = cb
        if sa != sb:                                # loop 수는 정확히 같아야 함
            return False
        if abs(la - lb) > tol or abs(aa - ab) > tol or abs(ca_c - cb_c) > tol:
            return False
    return True


def merge_close_groups(groups_sorted, tolerance):
    """exact 그룹 → tolerance 내 인접한 그룹들을 합쳐서 새 groups 반환.

    Returns:
        merged_groups : [(rep_sig, samples_list)]   가장 큰 그룹 순서
        merge_info    : [(rep_sig, [original_ranks_merged])]   리포트용
    """
    if tolerance <= 0 or len(groups_sorted) <= 1:
        info = [(sig, [i + 1]) for i, (sig, _) in enumerate(groups_sorted)]
        return list(groups_sorted), info

    n = len(groups_sorted)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            if _can_merge_sigs(groups_sorted[i][0], groups_sorted[j][0], tolerance):
                union(i, j)

    clusters = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)

    merged = []
    info = []
    for _, idxs in clusters.items():
        idxs.sort(key=lambda i: -len(groups_sorted[i][1]))   # 큰 거 먼저
        rep_sig = groups_sorted[idxs[0]][0]
        all_samples = []
        for i in idxs:
            all_samples.extend(groups_sorted[i][1])
        merged.append((rep_sig, all_samples))
        info.append((rep_sig, [i + 1 for i in idxs]))

    pack = sorted(zip(merged, info), key=lambda p: -len(p[0][1]))
    merged_sorted = [m for m, _ in pack]
    info_sorted = [inf for _, inf in pack]
    return merged_sorted, info_sorted


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


# ══════════════════════════════════════════════════════════════
# Canonicalize: variant 2 의 sample 을 variant 1 의 LINE 개수에 맞춰 강제 변환
# (sample 파일은 안 건드리고 토큰만 변환해서 시각화)
# ══════════════════════════════════════════════════════════════
def _chunk_lines_indices(chunk_rows):
    """chunk (numpy 2D) 안에서 LINE 행들의 인덱스 + 첫 loop 의 (s, e) 범위."""
    sol_idxs = [i for i, r in enumerate(chunk_rows) if int(r[0]) == SOL]
    if not sol_idxs:
        return [], (0, 0)
    s = sol_idxs[0] + 1
    if len(sol_idxs) >= 2:
        e = sol_idxs[1]
    else:
        ext_idx = next((i for i, r in enumerate(chunk_rows) if int(r[0]) == EXT), len(chunk_rows))
        e = ext_idx
    line_idxs = [i for i in range(s, e) if int(chunk_rows[i][0]) == LINE]
    return line_idxs, (s, e)


def canonicalize_chunk_lines(chunk, target_n_line):
    """chunk 의 첫 loop 안에서 LINE 개수를 target_n_line 에 맞춤.
    over → 인접한 LINE 두 개를 하나로 병합 (중간점 drop), under → 가장 긴 LINE 을 중간에서 split.
    """
    rows = [r.copy() for r in chunk]
    while True:
        line_idxs, (s, e) = _chunk_lines_indices(rows)
        cur_n = len(line_idxs)
        if cur_n == target_n_line or cur_n < 2:
            break

        if cur_n > target_n_line:
            # 가장 "병합해도 무해" 한 LINE 한 개 drop
            # heuristic: 인접한 두 LINE 의 끝점 사이가 가장 짧은 쪽 (그쪽이 거의 일직선)
            best_drop = line_idxs[1]
            best_d = float("inf")
            for k in range(1, len(line_idxs)):
                ip = line_idxs[k - 1]
                ic = line_idxs[k]
                px, py = int(rows[ip][1]), int(rows[ip][2])
                cx, cy = int(rows[ic][1]), int(rows[ic][2])
                d = (cx - px) ** 2 + (cy - py) ** 2
                if d < best_d:
                    best_d = d
                    best_drop = ic
            del rows[best_drop]
        else:
            # 가장 긴 LINE 을 둘로 split (중간점 끼워넣기)
            best_split = None
            best_len = -1
            for k in range(1, len(line_idxs)):
                ip = line_idxs[k - 1]
                ic = line_idxs[k]
                px, py = int(rows[ip][1]), int(rows[ip][2])
                cx, cy = int(rows[ic][1]), int(rows[ic][2])
                d = (cx - px) ** 2 + (cy - py) ** 2
                if d > best_len:
                    best_len = d
                    best_split = (ip, ic)
            if best_split is None:
                break
            ip, ic = best_split
            px, py = int(rows[ip][1]), int(rows[ip][2])
            cx, cy = int(rows[ic][1]), int(rows[ic][2])
            mx = (px + cx) // 2
            my = (py + cy) // 2
            new_row = np.full(17, -1, dtype=rows[ic].dtype)
            new_row[0] = LINE
            new_row[1] = mx
            new_row[2] = my
            rows.insert(ic, new_row)

    return np.stack(rows, axis=0) if rows else chunk


def canonicalize_sample_to_template(sample_tokens, template_chunk_sigs):
    """sample 의 각 chunk 의 LINE 개수를 template chunk_sigs 의 LINE 개수에 맞춤.
    role 행, ARC, CIRCLE, SOL, EXT 는 그대로 둠.
    """
    out_rows = []
    cur_chunk = []
    chunk_idx = 0
    in_chunk = False
    for row in sample_tokens:
        c = int(row[0])
        if c < 0:
            break
        if c == EOS:
            out_rows.append(row.copy())
            break
        if c == ROLE:
            out_rows.append(row.copy())
            continue
        if c == SOL and not in_chunk:
            in_chunk = True
            cur_chunk = [row.copy()]
        elif in_chunk and c == EXT:
            cur_chunk.append(row.copy())
            chunk_arr = np.stack(cur_chunk, axis=0)
            if chunk_idx < len(template_chunk_sigs):
                tgt_l = template_chunk_sigs[chunk_idx][1][0]   # (role, (L, A, C, SOL))
                chunk_arr = canonicalize_chunk_lines(chunk_arr, tgt_l)
            for r in chunk_arr:
                out_rows.append(r)
            cur_chunk = []
            in_chunk = False
            chunk_idx += 1
        elif in_chunk:
            cur_chunk.append(row.copy())
    return np.stack(out_rows, axis=0)


# ══════════════════════════════════════════════════════════════
# Token-only 2D 렌더 (canonicalize 결과는 JSON 이 없으므로 토큰 직접 그림)
# ══════════════════════════════════════════════════════════════
def render_tokens_2d(ax, tokens, color="#2E4172", title=""):
    """tokens 의 모든 chunk × 모든 loop 의 LINE 끝점들을 2D 폐곡선으로 그림.
    좌표축은 0~1023 토큰 스케일."""
    in_chunk = False
    cur_loop_pts = None
    chunk_pal = ["#2E4172", "#8B5A3C", "#3F6E5C", "#6B5B7A", "#555555", "#8B7E3D"]
    ci = 0
    for row in tokens:
        c = int(row[0])
        if c < 0 or c == EOS:
            break
        if c == ROLE:
            continue
        if c == SOL:
            if cur_loop_pts and len(cur_loop_pts) >= 2:
                xs = [p[0] for p in cur_loop_pts] + [cur_loop_pts[0][0]]
                ys = [p[1] for p in cur_loop_pts] + [cur_loop_pts[0][1]]
                ax.plot(xs, ys, color=chunk_pal[ci % len(chunk_pal)], lw=1.3)
                ax.fill(xs, ys, color=chunk_pal[ci % len(chunk_pal)], alpha=0.18)
            cur_loop_pts = []
            in_chunk = True
            continue
        if c == EXT:
            if cur_loop_pts and len(cur_loop_pts) >= 2:
                xs = [p[0] for p in cur_loop_pts] + [cur_loop_pts[0][0]]
                ys = [p[1] for p in cur_loop_pts] + [cur_loop_pts[0][1]]
                ax.plot(xs, ys, color=chunk_pal[ci % len(chunk_pal)], lw=1.3)
                ax.fill(xs, ys, color=chunk_pal[ci % len(chunk_pal)], alpha=0.18)
            cur_loop_pts = None
            in_chunk = False
            ci += 1
            continue
        if in_chunk and c == LINE:
            cur_loop_pts.append((int(row[1]), int(row[2])))
        elif in_chunk and c == ARC:
            cur_loop_pts.append((int(row[3]), int(row[4])))   # mid
            cur_loop_pts.append((int(row[1]), int(row[2])))   # end
        elif in_chunk and c == CIRCLE:
            cx, cy, r = int(row[1]), int(row[2]), int(row[3])
            ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
            xs = cx + r * np.cos(ang)
            ys = cy + r * np.sin(ang)
            ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]),
                    color=chunk_pal[ci % len(chunk_pal)], lw=1.3)
            ax.fill(xs, ys, color=chunk_pal[ci % len(chunk_pal)], alpha=0.18)
    ax.set_aspect("equal")
    ax.set_xlim(-50, 1080); ax.set_ylim(-50, 1080)
    ax.set_xticks([0, 512, 1023]); ax.set_yticks([0, 512, 1023])
    ax.tick_params(labelsize=7, colors="#666")
    ax.set_title(title, fontsize=9)
    for s in ax.spines.values():
        s.set_color("#bbb")


def preview_canonicalize(type_label, exact_groups, save_path):
    """exact variant 1 을 template 으로 잡고 variant 2 → template skeleton 으로 강제 변환,
    그 결과를 2D 로 비교 (3 panel).  sample 파일은 안 건드림.
    """
    if len(exact_groups) < 2:
        print(f"  [{type_label}] (canonicalize preview skip: only {len(exact_groups)} exact variant)")
        return
    sig1, group1 = exact_groups[0]
    sig2, group2 = exact_groups[1]
    template_sigs = sig1                  # tuple of (role, (L,A,C,SOL))
    s1 = group1[0]
    s2 = group2[0]
    tok1 = np.asarray(s1["tokens"] if "tokens" in s1 else np.load(s1["npy"]).astype(np.int32))
    tok2 = np.asarray(s2["tokens"] if "tokens" in s2 else np.load(s2["npy"]).astype(np.int32))
    tok2_canon = canonicalize_sample_to_template(tok2, template_sigs)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor="white")
    fig.suptitle(
        f"[{type_label}] canonicalize preview  —  v2 sample → v1 skeleton "
        f"(token-only 2D view)",
        fontsize=10,
    )
    render_tokens_2d(axes[0], tok1,
                     title=f"v1 template  ({sig_to_str(sig1)})\n{len(group1)} samples · L={template_sigs[0][1][0]}")
    render_tokens_2d(axes[1], tok2,
                     title=f"v2 original  ({sig_to_str(sig2)})\n{len(group2)} samples · L={sig2[0][1][0]}")
    n_line_after = sum(int(r[0]) == LINE for r in tok2_canon)
    render_tokens_2d(axes[2], tok2_canon,
                     title=f"v2 rewritten → v1 skeleton\nL={n_line_after}  (target {template_sigs[0][1][0]})")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    try:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ canonicalize preview → {save_path}")
    except Exception as e:
        print(f"  ⚠ failed to save canonicalize preview: {e}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Canonicalize 실제 적용: 각 merged group 안 sample 의 _tokens.npy 를
# dominant exact-skeleton 으로 강제 변환해 저장 (원본은 백업)
# ══════════════════════════════════════════════════════════════
def apply_canonicalize_to_disk(exact_groups, merged_groups, type_dir_abs,
                               type_label, backup_suffix):
    """
    각 merged group:
      - dominant = 그 group 안에서 가장 sample 수가 많은 exact-sig
      - 같은 group 안 sig != dominant 인 sample 들 → tokens 를 dominant 의
        LINE 개수에 맞춰 canonicalize → _tokens.npy 덮어쓰기 (원본 백업 후).
    """
    import shutil
    backup_dir = type_dir_abs.rstrip(os.sep) + backup_suffix
    os.makedirs(backup_dir, exist_ok=True)

    # exact sig → sample 수 lookup
    exact_count = {sig: len(group) for sig, group in exact_groups}

    n_total_rewrite = 0
    n_total_skip = 0
    n_failed = 0

    print(f"\n  [{type_label}] applying canonicalize → in-place rewrite")
    print(f"    backup dir: {backup_dir}")

    for mr, (merged_sig, merged_group) in enumerate(merged_groups):
        # merged_group 안의 exact sig 별 분포
        sigs_here = {}
        for s in merged_group:
            sigs_here.setdefault(s["sig"], []).append(s)
        if not sigs_here:
            continue
        # dominant = 가장 많은 exact-sig (전체 exact_count 기준)
        dominant_sig = max(sigs_here.keys(),
                           key=lambda sg: exact_count.get(sg, len(sigs_here[sg])))
        n_keep = len(sigs_here.get(dominant_sig, []))
        targets = [(sg, samples) for sg, samples in sigs_here.items() if sg != dominant_sig]
        n_to_rw = sum(len(s) for _, s in targets)

        print(f"    merged-rank {mr + 1}: {len(merged_group)} samples → "
              f"dominant {sig_to_str(dominant_sig)[:50]}  "
              f"(keep {n_keep}, rewrite {n_to_rw})")

        if not targets:
            n_total_skip += n_keep
            continue

        # template_sigs = dominant 의 chunk_sigs (= dominant_sig 그 자체)
        template_sigs = dominant_sig
        for sg, samples in targets:
            for s in samples:
                npy_path = s["npy"]
                try:
                    tokens = np.load(npy_path).astype(np.int32)
                    new_tokens = canonicalize_sample_to_template(tokens, template_sigs)
                    # 백업 (이미 있으면 덮어쓰지 않음 — 첫 원본 보존)
                    bak_path = os.path.join(backup_dir, os.path.basename(npy_path))
                    if not os.path.exists(bak_path):
                        shutil.copy2(npy_path, bak_path)
                    np.save(npy_path, new_tokens.astype(tokens.dtype))
                    n_total_rewrite += 1
                except Exception as e:
                    print(f"      ✗ {os.path.basename(npy_path)}: {e}")
                    n_failed += 1
        n_total_skip += n_keep

    print(f"  ✓ rewrote {n_total_rewrite} samples, kept {n_total_skip}"
          + (f", {n_failed} failed" if n_failed else ""))
    print(f"    원복: {backup_dir} 의 파일을 다시 {type_dir_abs} 로 복사")


# ══════════════════════════════════════════════════════════════
# Variant 제외 (선택된 variant 의 sample 파일들을 옮기거나 삭제)
# ══════════════════════════════════════════════════════════════
def do_exclude_variants(groups_sorted, type_dir_abs,
                        ranks_to_exclude, mode="move",
                        folder_suffix="_excluded"):
    """ranks_to_exclude 의 variant 들에 속한 모든 sample 파일을 처리.

    한 sample 당 같이 옮길/지울 파일:
      - {basename}_tokens.npy
      - {basename}_deepcad.json
      - {basename}_tokens_float.npy   (있을 때만)

    mode="move":   {type_dir_abs}{folder_suffix}/ 폴더로 이동.
                   학습 코드 (surrogate.py) 가 더 이상 이 폴더 안 보니까
                   자동으로 학습 대상에서 제외됨.
    mode="delete": 영구 삭제. 되돌릴 수 없음.
    """
    import shutil

    rank_set = set(int(r) for r in ranks_to_exclude)
    target_groups = []
    for r0, (sig, group) in enumerate(groups_sorted):
        rank = r0 + 1
        if rank in rank_set:
            target_groups.append((rank, sig, group))

    if not target_groups:
        print(f"\n  ⚠ EXCLUDE_VARIANTS={ranks_to_exclude} 중 매칭되는 그룹 없음 (전체 {len(groups_sorted)} 개)")
        return

    n_samples = sum(len(g) for _, _, g in target_groups)
    print(f"\n  ★ EXCLUDE {len(target_groups)} variant(s) → {n_samples} sample(s) "
          f"  mode={mode}")
    for rank, sig, group in target_groups:
        print(f"    - Variant {rank}: {len(group)} samples  ({sig_to_str(sig)[:60]})")

    if mode == "move":
        excluded_dir = type_dir_abs.rstrip(os.sep) + folder_suffix
        os.makedirs(excluded_dir, exist_ok=True)
        print(f"    → moving to: {excluded_dir}/")

    n_moved = 0
    n_failed = 0
    for rank, sig, group in target_groups:
        for s in group:
            npy_path = s["npy"]
            json_path = npy_path.replace("_tokens.npy", "_deepcad.json")
            float_path = npy_path.replace("_tokens.npy", "_tokens_float.npy")

            files_to_handle = [npy_path]
            if os.path.exists(json_path):
                files_to_handle.append(json_path)
            if os.path.exists(float_path):
                files_to_handle.append(float_path)

            for fp in files_to_handle:
                try:
                    if mode == "delete":
                        os.remove(fp)
                    elif mode == "move":
                        dst = os.path.join(excluded_dir, os.path.basename(fp))
                        shutil.move(fp, dst)
                    else:
                        raise ValueError(f"Unknown EXCLUDE_MODE: {mode}")
                    n_moved += 1
                except Exception as e:
                    print(f"    ✗ failed {os.path.basename(fp)}: {e}")
                    n_failed += 1

    print(f"  ✓ {mode}d {n_moved} file(s)" + (f", {n_failed} failed" if n_failed else ""))
    if mode == "move":
        print(f"    되돌리려면: {excluded_dir} 의 파일들을 다시 {type_dir_abs} 로 복사.")


def analyze_one_type(type_label, type_dir_abs):
    """한 type 폴더 전체 분석: 로딩 → grouping → txt/png 저장 → 제외 처리.
    Returns:
        (samples, groups_sorted)  또는 (None, []) 데이터 없을 때.
    """
    if not os.path.isdir(type_dir_abs):
        print(f"  ✗ [{type_label}] type dir not found: {type_dir_abs}")
        return None, []

    npy_files = sorted(glob.glob(os.path.join(type_dir_abs, "*_tokens.npy")))
    if not npy_files:
        print(f"  ✗ [{type_label}] no *_tokens.npy in {type_dir_abs}")
        return None, []

    print(f"\n  Analyzing [{type_label}]: {type_dir_abs}")
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

    # ── 2) signature 별 그룹화 (exact) ──
    sig_groups = {}
    for s in samples:
        sig_groups.setdefault(s["sig"], []).append(s)

    exact_groups = sorted(sig_groups.items(), key=lambda kv: -len(kv[1]))

    # ── 2.5) tolerance 머지 ──
    groups_sorted, merge_info = merge_close_groups(exact_groups, COUNT_TOLERANCE)
    n_exact = len(exact_groups)
    n_merged = len(groups_sorted)

    # ── 3) 콘솔 출력 ──
    if COUNT_TOLERANCE > 0 and n_exact != n_merged:
        print(f"  {n_exact} exact variants → {n_merged} after tolerance "
              f"merge (±{COUNT_TOLERANCE}):\n")
    else:
        print(f"  {n_merged} structural variants:\n")
    print(f"  {'rank':>4s} | {'count':>5s} | structure signature")
    print("  -----+-------+" + "-" * 60)
    for r, (sig, group) in enumerate(groups_sorted):
        line = f"  {r + 1:>4d} | {len(group):>5d} | {sig_to_str(sig)}"
        if COUNT_TOLERANCE > 0:
            orig_ranks = merge_info[r][1]
            if len(orig_ranks) > 1:
                line += f"   ← merged from exact ranks {orig_ranks}"
        print(line)

    # ── 4) 리포트 txt 저장 ──
    os.makedirs(SAVE_DIR, exist_ok=True)
    report_path = os.path.join(SAVE_DIR, f"{type_label}_variants.txt")
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

    # ── 4.5) EXCLUDE_VARIANTS 처리: 선택한 variant 들의 sample 옮기기/삭제 ──
    excl = EXCLUDE_VARIANTS_PER_TYPE.get(type_label, [])
    if excl:
        do_exclude_variants(
            groups_sorted, type_dir_abs,
            ranks_to_exclude=excl,
            mode=EXCLUDE_MODE,
            folder_suffix=EXCLUDE_FOLDER_SUFFIX,
        )

    # ── 4.6) Canonicalize preview: v1 template vs v2 original vs v2 rewritten ──
    if CANONICALIZE_PREVIEW and len(exact_groups) >= 2:
        preview_path = os.path.join(SAVE_DIR, f"{type_label}_canonicalize_preview.png")
        try:
            preview_canonicalize(type_label, exact_groups, preview_path)
        except Exception as e:
            print(f"  ⚠ canonicalize preview failed: {type(e).__name__}: {e}")

    # ── 4.7) Canonicalize APPLY: 실제 _tokens.npy 변환 + 원본 백업 ──
    if APPLY_CANONICALIZE and len(exact_groups) >= 2:
        if FORCE_SINGLE_SKELETON_PER_TYPE:
            # 모든 exact group 의 sample 을 한 그룹으로 묶고 dominant exact 를 template 로
            all_samples = [s for _, group in exact_groups for s in group]
            dominant_sig = exact_groups[0][0]   # exact_groups 는 큰 거 먼저 정렬돼 있음
            apply_target = [(dominant_sig, all_samples)]
            print(f"  [{type_label}] FORCE_SINGLE_SKELETON_PER_TYPE=True → "
                  f"merging {len(exact_groups)} exact variants into 1")
        else:
            apply_target = groups_sorted
        try:
            apply_canonicalize_to_disk(
                exact_groups, apply_target, type_dir_abs,
                type_label=type_label,
                backup_suffix=CANONICALIZE_BACKUP_SUFFIX,
            )
        except Exception as e:
            print(f"  ⚠ apply_canonicalize failed: {type(e).__name__}: {e}")

    # ── 5) 시각화: 상위 N 그룹의 대표 sample 3D ──
    if SHOW_FIGURES:
        n_show = min(TOP_N_GROUPS, len(groups_sorted))
        if n_show > 0:
            ncol = min(n_show, 3)
            nrow = (n_show + ncol - 1) // ncol
            fig = plt.figure(figsize=(5 * ncol, 5 * nrow), facecolor="white")
            fig.suptitle(
                f"Structural variants in [{type_label}]  "
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

            fig_path = os.path.join(SAVE_DIR, f"{type_label}_variants.png")
            try:
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                print(f"  ✓ figure saved → {fig_path}")
            except Exception as e:
                print(f"  ⚠ failed to save figure: {e}")

    return samples, groups_sorted


# ══════════════════════════════════════════════════════════════
# Cross-type summary + figure
# ══════════════════════════════════════════════════════════════
def print_summary_table(per_type_results):
    print("\n" + "═" * 78)
    print("  SUMMARY — 각 type 의 골격 개수 (옵티마이저 search 용 single-skeleton 체크)")
    print("═" * 78)
    print(f"  {'type':<10s} | {'samples':>7s} | {'variants':>8s} | "
          f"{'top1 share':>10s} | single skeleton?")
    print("  " + "-" * 76)
    for type_label, _, samples, groups in per_type_results:
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
    print("═" * 78)


def make_all_types_top1_figure(per_type_results, save_path):
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
        print(f"  ✓ all-types figure saved → {save_path}")
    except Exception as e:
        print(f"  ⚠ failed to save all-types figure: {e}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, SAVE_DIR), exist_ok=True)

    if not TYPE_DIRS:
        print("  ✗ TYPE_DIRS is empty.")
        sys.exit(1)

    per_type_results = []
    for type_label, type_dir in TYPE_DIRS:
        type_dir_abs = (type_dir if os.path.isabs(type_dir)
                        else os.path.join(script_dir, type_dir))
        samples, groups = analyze_one_type(type_label, type_dir_abs)
        per_type_results.append((type_label, type_dir_abs, samples, groups))

    print_summary_table(per_type_results)

    if SHOW_FIGURES:
        make_all_types_top1_figure(
            per_type_results,
            os.path.join(script_dir, SAVE_DIR, "all_types_top1.png"),
        )
        plt.show()


if __name__ == "__main__":
    main()
