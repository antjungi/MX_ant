#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_d4.py  —  single-file (no external project modules).

Randomly generates N D4-symmetric press-type planar antenna designs and
opens N interactive Matplotlib windows.  Set SEED below to reproduce a
run; leave it None for fresh random samples each time.

What is randomly drawn for each design:
  * radiator outline  : square | octagonal | side-notched | plus | star
                        (with random sub-parameters within safe ranges)
  * slot pattern      : none | cross | ring | concentric rings | cross+ring
                        (slot widths are either 0.5 mm or 1.5 mm)
  * extra fold tabs   : none | 4 inner UP-tabs | 4 inner DOWN-tabs

All designs satisfy the locked physical / DFM spec:
  plate    50 x 50 mm    thickness 1 mm    grid 0.5 mm
  kerf     0.5 mm        bend radius 1 mm
  metal/gap min width    0.5 mm (both polarities)
  4-connectivity         no floating pieces (ring slots use 4 cardinal
                         bridges; combination constraints below keep the
                         bridges intact)
  D4 symmetry            four-fold rotation + mirrors across both diagonals
"""
import os, sys, random
import numpy as np
import matplotlib.pyplot as plt           # interactive backend left to the user
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ════════════════════════════════════════════════════════════════════════════
# Locked physical / DFM constants
# ════════════════════════════════════════════════════════════════════════════
PLATE   = 50.0    # plate side (mm)
GRID    = 0.5     # grid pitch (mm)
THICK   = 1.0     # sheet thickness (mm)
KERF    = 0.5     # cut kerf (mm)
BEND_R  = 1.0     # min bend radius (mm)
VIS_EPS = 0.05    # visualisation-only z gap so PCB top doesn't z-fight legs

STEEL = np.array([0.66, 0.70, 0.78])
TABC  = np.array([0.55, 0.78, 0.55])
FR4   = np.array([0.14, 0.45, 0.22])
LIGHT = np.array([0.4, 0.5, 0.78]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMB, DIF = 0.42, 0.58
MTN, VAL = "#d62728", "#1f77b4"


# ════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ════════════════════════════════════════════════════════════════════════════
def rot_axis(d, deg):
    d = np.asarray(d, float); d = d / np.linalg.norm(d); x, y, z = d
    th = np.radians(deg); c, s = np.cos(th), np.sin(th); C = 1 - c
    return np.array([[c+x*x*C,   x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C,   y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def rect_minus(r, h):
    a0, a1, b0, b1 = r; c0, c1, d0, d1 = h
    ix0, ix1 = max(a0, c0), min(a1, c1)
    iy0, iy1 = max(b0, d0), min(b1, d1)
    if ix0 >= ix1 or iy0 >= iy1: return [r]
    out = []
    if b0 < iy0: out.append((a0, a1, b0, iy0))
    if iy1 < b1: out.append((a0, a1, iy1, b1))
    if a0 < ix0: out.append((a0, ix0, iy0, iy1))
    if ix1 < a1: out.append((ix1, a1, iy0, iy1))
    return out


def _subtract_intervals(intervals, to_subtract):
    result = list(intervals)
    for s_lo, s_hi in to_subtract:
        if s_hi <= s_lo: continue
        new = []
        for lo, hi in result:
            if s_hi <= lo or s_lo >= hi:
                new.append((lo, hi))
            else:
                if lo < s_lo: new.append((lo, s_lo))
                if s_hi < hi: new.append((s_hi, hi))
        result = new
    return result


def _boundary_segments(rects, eps=1e-9):
    """Union-boundary of axis-aligned rectangles. ('V', x, y_lo, y_hi) / ('H', y, x_lo, x_hi)."""
    segs = []
    for i, (x0, x1, y0, y1) in enumerate(rects):
        others = [(b0, b1) for j, (a0, a1, b0, b1) in enumerate(rects)
                  if j != i and a0 < x0 - eps < a1]
        for a, b in _subtract_intervals([(y0, y1)], others):
            if b - a > eps: segs.append(('V', x0, a, b))
        others = [(b0, b1) for j, (a0, a1, b0, b1) in enumerate(rects)
                  if j != i and a0 < x1 + eps < a1]
        for a, b in _subtract_intervals([(y0, y1)], others):
            if b - a > eps: segs.append(('V', x1, a, b))
        others = [(a0, a1) for j, (a0, a1, b0, b1) in enumerate(rects)
                  if j != i and b0 < y0 - eps < b1]
        for a, b in _subtract_intervals([(x0, x1)], others):
            if b - a > eps: segs.append(('H', y0, a, b))
        others = [(a0, a1) for j, (a0, a1, b0, b1) in enumerate(rects)
                  if j != i and b0 < y1 + eps < b1]
        for a, b in _subtract_intervals([(x0, x1)], others):
            if b - a > eps: segs.append(('H', y1, a, b))
    return segs


# ════════════════════════════════════════════════════════════════════════════
# Design class : (faces, folds) -> 3D after applying hinge rotations
# ════════════════════════════════════════════════════════════════════════════
class Design:
    def __init__(self, faces, folds, root=0):
        self.faces = faces; self.folds = folds; self.root = root
        self.kids = {}
        for p, c, line, th in folds:
            self.kids.setdefault(p, []).append((c, line, th))
        self.tf = [None] * len(faces)
        self._assign(root, np.eye(3), np.zeros(3))

    def _assign(self, i, M, t):
        self.tf[i] = (M, t)
        for c, (Ax, Ay, Bx, By), th in self.kids.get(i, []):
            A = M @ np.array([Ax, Ay, 0.0]) + t
            B = M @ np.array([Bx, By, 0.0]) + t
            R = rot_axis(B - A, th)
            self._assign(c, R @ M, R @ (t - A) + A)

    def _map(self, i, u, v):
        M, t = self.tf[i]
        return M @ np.array([u, v, 0.0]) + t

    def _normal(self, i):
        M, _ = self.tf[i]
        n = M @ np.array([0, 0, 1.0])
        return n / np.linalg.norm(n)

    def quads(self, thick=THICK):
        out = []
        for i, f in enumerate(self.faces):
            rect = f['rect']
            holes = f.get('holes', [])
            hole_walls = f.get('hole_walls', holes)
            rgb = f.get('col', STEEL)
            alpha = f.get('alpha', 1.0)
            n = self._normal(i)
            off = 0.5 * thick * n

            def top(u, v): return self._map(i, u, v) + off
            def bot(u, v): return self._map(i, u, v) - off

            tiles = [rect]
            for h in holes:
                nt = []
                for r in tiles: nt += rect_minus(r, h)
                tiles = nt
            for (x0, x1, y0, y1) in tiles:
                if x1 - x0 <= 1e-9 or y1 - y0 <= 1e-9: continue
                out.append(([top(x0, y0), top(x1, y0), top(x1, y1), top(x0, y1)], rgb, alpha))
                out.append(([bot(x0, y0), bot(x1, y0), bot(x1, y1), bot(x0, y1)], rgb, alpha))

            def walls(x0, x1, y0, y1):
                for (p, qq) in [((x0, y0), (x1, y0)),
                                ((x1, y0), (x1, y1)),
                                ((x1, y1), (x0, y1)),
                                ((x0, y1), (x0, y0))]:
                    out.append(([top(*p), top(*qq), bot(*qq), bot(*p)], rgb, alpha))

            perim = f.get('perim', None)
            if perim is None:
                walls(*rect)
            else:
                for k in range(len(perim)):
                    p0 = perim[k]; p1 = perim[(k+1) % len(perim)]
                    out.append(([top(*p0), top(*p1), bot(*p1), bot(*p0)], rgb, alpha))

            for h in hole_walls:
                walls(*h)
        return out


# ════════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ════════════════════════════════════════════════════════════════════════════
def _newell(v):
    n = np.zeros(3); m = len(v)
    for i in range(m):
        a = np.asarray(v[i], float); b = np.asarray(v[(i+1) % m], float)
        n[0] += (a[1]-b[1])*(a[2]+b[2])
        n[1] += (a[2]-b[2])*(a[0]+b[0])
        n[2] += (a[0]-b[0])*(a[1]+b[1])
    nn = np.linalg.norm(n)
    return n / nn if nn > 1e-12 else n


def draw_crease(ax, design, sym_axes=True, pad=4.0):
    m = design.meta
    for f in design.faces:
        green = np.allclose(f.get('col', STEEL), TABC)
        fc = "#d7efd7" if green else "#dfe6f2"
        if 'perim' in f:
            ax.add_patch(Polygon(f['perim'], closed=True, fc=fc, ec='none', zorder=1))
        else:
            x0, x1, y0, y1 = f['rect']
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc=fc, ec='none', zorder=1))
    xs = [p[0] for p in m['perim']]; ys = [p[1] for p in m['perim']]
    if sym_axes:
        mx = max(max(map(abs, xs)), max(map(abs, ys))) * 1.12
        ax.plot([-mx, mx], [0, 0], color="#9a9a9a", lw=0.9, ls=(0, (5, 3, 1, 3)), zorder=0)
        ax.plot([0, 0], [-mx, mx], color="#9a9a9a", lw=0.9, ls=(0, (5, 3, 1, 3)), zorder=0)
    ax.add_patch(Polygon(m['perim'], closed=True, fill=False, ec='k', lw=2.4, zorder=4))

    cuts = list(m.get('cuts', []))
    for (x0, x1, y0, y1) in cuts:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc='white', ec='none', zorder=5))
    for seg in _boundary_segments(cuts):
        kind, c, a, b = seg
        if kind == 'V': ax.plot([c, c], [a, b], color='k', lw=1.0, zorder=6)
        else:           ax.plot([a, b], [c, c], color='k', lw=1.0, zorder=6)

    # true polygon cuts (e.g. diagonal X-slot) -- drawn as actual parallelograms
    for poly in m.get('poly_cuts', ()):
        ax.add_patch(Polygon(poly, closed=True, fc='white', ec='k', lw=1.0, zorder=5))

    for (Ax, Ay, Bx, By), mv in m['folds2d']:
        col = MTN if mv == 'M' else VAL
        ls  = (0, (6, 4)) if mv == 'M' else (0, (1, 2.5))
        lw  = 1.8 if mv == 'M' else 2.4
        ax.plot([Ax, Bx], [Ay, By], color=col, lw=lw, ls=ls, zorder=6)

    ax.set_xlim(min(xs)-pad, max(xs)+pad)
    ax.set_ylim(min(ys)-pad, max(ys)+pad)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def _box_quads(b, rgb, alpha=1.0):
    x0, x1, y0, y1, z0, z1 = b
    F = [[(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
         [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
         [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
         [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
         [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
         [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)]]
    return [(f, rgb, alpha) for f in F]


def pcb_box(plate_size=PLATE, top_z=0.0, thick=1.6, color=FR4):
    P = plate_size / 2.0
    return _box_quads((-P, P, -P, P, top_z-thick, top_z), color)


def _shade_quads(quads):
    polys, cols, pts = [], [], []
    for q in quads:
        if len(q) == 2: verts, rgb = q; alpha = 1.0
        else:           verts, rgb, alpha = q
        f = AMB + DIF * abs(float(np.dot(_newell(verts), LIGHT)))
        s = np.clip(np.asarray(rgb, float) * f, 0, 1)
        cols.append((float(s[0]), float(s[1]), float(s[2]), float(alpha)))
        polys.append(verts)
        pts += [np.asarray(p, float) for p in verts]
    return polys, cols, pts


def _set_axes_equal_from_pts(ax, pts, view):
    P = np.array(pts); lo = P.min(0); hi = P.max(0)
    c = (lo + hi) / 2; r = (hi - lo).max() / 2 * 1.08
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()


def render_assembly(ax, design, extras=(), fillets=(), view=(26, -52), thick=THICK):
    main_quads  = list(design.quads(thick)) + list(fillets)
    extra_quads = list(extras)
    all_pts = []
    for quads in [extra_quads, main_quads]:
        if not quads: continue
        polys, cols, pts = _shade_quads(quads)
        all_pts += pts
        pc = Poly3DCollection(polys, facecolors=cols, edgecolor='none')
        pc.set_zsort('average')
        ax.add_collection3d(pc)
    _set_axes_equal_from_pts(ax, all_pts, view)


# ════════════════════════════════════════════════════════════════════════════
# Lance (kerf-aware U cut) + bend fillets
# ════════════════════════════════════════════════════════════════════════════
def lance(direction, inner, length, width, kerf=KERF):
    hw = width / 2.0
    if direction == '+x':
        return ((inner, inner+length, -hw, hw),
                (inner, inner+length+kerf, -hw-kerf, hw+kerf),
                (inner, -hw, inner, hw), -90)
    if direction == '-x':
        return ((-inner-length, -inner, -hw, hw),
                (-inner-length-kerf, -inner, -hw-kerf, hw+kerf),
                (-inner, -hw, -inner, hw), +90)
    if direction == '+y':
        return ((-hw, hw, inner, inner+length),
                (-hw-kerf, hw+kerf, inner, inner+length+kerf),
                (-hw, inner, hw, inner), +90)
    if direction == '-y':
        return ((-hw, hw, -inner-length, -inner),
                (-hw-kerf, hw+kerf, -inner-length-kerf, -inner),
                (-hw, -inner, hw, -inner), -90)
    raise ValueError(direction)


def bend_fillet_extras(design, r=None, t=None, N=10):
    if r is None: r = BEND_R
    if t is None: t = THICK
    extras = []
    for parent, child, line, theta in design.folds:
        Ax, Ay, Bx, By = line
        Mp, tp = design.tf[parent]
        A3 = Mp @ np.array([Ax, Ay, 0.0]) + tp
        B3 = Mp @ np.array([Bx, By, 0.0]) + tp
        u_axis = B3 - A3
        hd = np.array([Bx-Ax, By-Ay, 0.0], float); hd /= np.linalg.norm(hd)
        perp = np.array([-hd[1], hd[0], 0.0])
        pr = design.faces[parent]['rect']
        cf = np.array([(pr[0]+pr[1])/2 - (Ax+Bx)/2,
                       (pr[2]+pr[3])/2 - (Ay+By)/2, 0.0])
        v_par_flat = perp * (1.0 if np.dot(cf, perp) > 0 else -1.0)
        v_par_3d = Mp @ v_par_flat
        R = rot_axis(u_axis, theta)
        n_bend_3d = R @ (Mp @ (-v_par_flat))
        bA = A3 + r * v_par_3d + r * n_bend_3d
        bB = B3 + r * v_par_3d + r * n_bend_3d
        ath = np.radians(abs(theta))
        ro = r + t / 2.0
        ri = max(r - t / 2.0, 0.01)
        col   = design.faces[child].get('col', design.faces[parent].get('col', STEEL))
        alpha = design.faces[parent].get('alpha', 1.0)
        for i in range(N):
            a0 = ath * i / N
            a1 = ath * (i + 1) / N
            o0 = -n_bend_3d * np.cos(a0) - v_par_3d * np.sin(a0)
            o1 = -n_bend_3d * np.cos(a1) - v_par_3d * np.sin(a1)
            extras.append(([tuple(bA + ro*o0), tuple(bB + ro*o0),
                            tuple(bB + ro*o1), tuple(bA + ro*o1)], col, alpha))
            extras.append(([tuple(bA + ri*o0), tuple(bB + ri*o0),
                            tuple(bB + ri*o1), tuple(bA + ri*o1)], col, alpha))
            extras.append(([tuple(bA + ri*o0), tuple(bA + ro*o0),
                            tuple(bA + ro*o1), tuple(bA + ri*o1)], col, alpha))
            extras.append(([tuple(bB + ri*o0), tuple(bB + ro*o0),
                            tuple(bB + ro*o1), tuple(bB + ri*o1)], col, alpha))
    return extras


# ════════════════════════════════════════════════════════════════════════════
# Base design : 50x50 plate, 4 down-folded U-legs to a PCB
# ════════════════════════════════════════════════════════════════════════════
def planar_on_pcb(plate=PLATE, leg_inner=14.0, leg_len=8.0, leg_w=5.0, bend_r=BEND_R):
    P = plate / 2.0
    hw = leg_w / 2.0
    holes, legfaces, folds, cuts, folds2d = [], [], [], [], []
    for k, dirn in enumerate(['+x', '-x', '+y', '-y']):
        tab, hole, line, th_up = lance(dirn, leg_inner, leg_len, leg_w)
        cuts += rect_minus(hole, tab)
        if dirn == '+x':
            tab2 = (tab[0]+bend_r, tab[1], tab[2], tab[3])
            bend_ext = (leg_inner-bend_r, leg_inner, -hw, hw)
        elif dirn == '-x':
            tab2 = (tab[0], tab[1]-bend_r, tab[2], tab[3])
            bend_ext = (-leg_inner, -leg_inner+bend_r, -hw, hw)
        elif dirn == '+y':
            tab2 = (tab[0], tab[1], tab[2]+bend_r, tab[3])
            bend_ext = (-hw, hw, leg_inner-bend_r, leg_inner)
        else:
            tab2 = (tab[0], tab[1], tab[2], tab[3]-bend_r)
            bend_ext = (-hw, hw, -leg_inner, -leg_inner+bend_r)
        holes.append(hole); holes.append(bend_ext)
        legfaces.append(dict(rect=tab2, col=TABC))
        folds.append((0, 1+k, line, -th_up))                # NEGATE -> fold DOWN
        folds2d.append((line, 'V'))                          # valley = down to PCB
    faces = [dict(rect=(-P,P,-P,P), holes=holes, hole_walls=cuts)] + legfaces
    d = Design(faces, folds)
    d.meta = dict(perim=[(-P,-P),(P,-P),(P,P),(-P,P)], cuts=cuts, folds2d=folds2d,
                  leg_drop=leg_len, bend_r=bend_r, plate=plate)
    return d


def base(leg_inner=14.0):
    d = planar_on_pcb(plate=PLATE, leg_inner=leg_inner, leg_len=8.0, leg_w=5.0)
    d.faces[0]['alpha'] = 0.45
    return d


# ════════════════════════════════════════════════════════════════════════════
# Radiator-outline modifiers (D4-symmetric)
# ════════════════════════════════════════════════════════════════════════════
def chamfer_radiator(d, chamfer=4.0):
    """Octagonal-ish: 4 small corner cuts."""
    h, c = d.meta['plate'] / 2.0, chamfer
    perim = [
        (-h+c, -h),  (h-c, -h),  (h-c, -h+c), ( h,   -h+c),
        ( h,   h-c), (h-c,  h-c),(h-c,  h),    (-h+c,  h),
        (-h+c, h-c), (-h,   h-c),(-h,  -h+c), (-h+c, -h+c),
    ]
    corner_cuts = [
        (-h, -h+c, -h, -h+c), ( h-c, h, -h, -h+c),
        (-h, -h+c,  h-c, h),  ( h-c, h,  h-c, h),
    ]
    d.faces[0]['perim'] = perim
    d.faces[0]['holes'] = list(d.faces[0]['holes']) + corner_cuts
    d.meta['perim'] = perim
    return d


def plus_radiator(d, arm_hw=10.0):
    """Plus shape: 4 large corner cuts leave 4 arms of width 2*arm_hw."""
    h, a = d.meta['plate'] / 2.0, arm_hw
    perim = [
        (-a, -h), (a, -h), (a, -a), (h, -a),
        ( h,  a), (a,  a), (a,  h), (-a, h),
        (-a,  a), (-h, a), (-h, -a), (-a, -a),
    ]
    corner_cuts = [
        (-h, -a, -h, -a), ( a, h, -h, -a),
        (-h, -a,  a, h),  ( a, h,  a, h),
    ]
    d.faces[0]['perim'] = perim
    d.faces[0]['holes'] = list(d.faces[0]['holes']) + corner_cuts
    d.meta['perim'] = perim
    return d


def notched_radiator(d, notch_hw=4.0, notch_d=3.0):
    """Square with 4 mid-side indents going inward."""
    h, nw, nd = d.meta['plate'] / 2.0, notch_hw, notch_d
    perim = [
        (-h, -h),    (-nw, -h),    (-nw, -h+nd), (nw, -h+nd), (nw, -h),
        ( h, -h),    ( h, -nw),    ( h-nd, -nw), ( h-nd,  nw), ( h,  nw),
        ( h,  h),    ( nw, h),     ( nw, h-nd),  (-nw, h-nd), (-nw, h),
        (-h,  h),    (-h,  nw),    (-h+nd, nw),  (-h+nd, -nw),(-h, -nw),
    ]
    notches = [
        (-nw, nw, -h, -h+nd), (-nw, nw,  h-nd, h),
        (-h, -h+nd, -nw, nw), ( h-nd, h, -nw, nw),
    ]
    d.faces[0]['perim'] = perim
    d.faces[0]['holes'] = list(d.faces[0]['holes']) + notches
    d.meta['perim'] = perim
    return d


# ════════════════════════════════════════════════════════════════════════════
# Slot patterns (D4-symmetric)
# ════════════════════════════════════════════════════════════════════════════
def cross_slot(width, length=10.0):
    hw = width / 2.0
    return [
        (-length,  length, -hw,     hw),
        (-hw,      hw,      hw,     length),
        (-hw,      hw,     -length, -hw),
    ]


def broken_ring(radius, width, gap=1.5):
    """Square ring with 4 cardinal `gap`-wide bridges (so the inside stays connected)."""
    r0 = radius; r1 = radius + width; g = gap / 2.0
    return [
        (-r1, -g,  r0, r1), ( g, r1,  r0, r1),       # top edge halves
        (-r1, -g, -r1, -r0), ( g, r1, -r1, -r0),     # bottom edge halves
        (-r1, -r0,  g, r1), (-r1, -r0, -r1, -g),     # left edge halves
        ( r0,  r1,  g, r1), ( r0,  r1, -r1, -g),     # right edge halves
    ]


def x_slot_polygons(half_extent, width):
    """Two TRUE 45-degree parallelogram slots forming a D4-symmetric 'X' shape.
       half_extent  : distance from centre to corner of the slot centreline
       width        : perpendicular slot width  (>= 0.5 mm by DFM)
    Returns a list of two polygons (one per diagonal) as lists of (x,y) verts."""
    d = width / (2.0 * 1.4142135623730951)
    # y = x diagonal slot
    poly1 = [
        (-half_extent + d, -half_extent - d),
        ( half_extent + d,  half_extent - d),
        ( half_extent - d,  half_extent + d),
        (-half_extent - d, -half_extent + d),
    ]
    # y = -x diagonal slot
    poly2 = [
        (-half_extent - d,  half_extent - d),
        ( half_extent - d, -half_extent - d),
        ( half_extent + d, -half_extent + d),
        (-half_extent + d,  half_extent + d),
    ]
    return [poly1, poly2]


def make_poly_slot_extras(d, thick=THICK):
    """Visualise polygon slots in 3D: a dark top/bottom patch slightly outside
    the radiator thickness plus 4 walls along the polygon edges. The polygon
    cuts themselves live in d.meta['poly_cuts']."""
    polys = d.meta.get('poly_cuts', ())
    extras = []
    alpha_radiator = d.faces[0].get('alpha', 1.0)
    dark = STEEL * 0.35                          # darker shade -> reads as 'cut'
    for poly in polys:
        top = [(x, y,  thick/2.0 + VIS_EPS) for x, y in poly]
        bot = [(x, y, -thick/2.0 - VIS_EPS) for x, y in poly]
        extras.append((top, dark, alpha_radiator))
        extras.append((bot, dark, alpha_radiator))
        for i in range(len(poly)):
            (x0, y0) = poly[i]; (x1, y1) = poly[(i+1) % len(poly)]
            verts = [(x0, y0,  thick/2.0),
                     (x1, y1,  thick/2.0),
                     (x1, y1, -thick/2.0),
                     (x0, y0, -thick/2.0)]
            extras.append((verts, STEEL, alpha_radiator))
    return extras


# ════════════════════════════════════════════════════════════════════════════
# Adding slots / extra lances to an already-built design
# ════════════════════════════════════════════════════════════════════════════
def add_slots(d, slot_rects):
    d.faces[0]['holes']      = list(d.faces[0]['holes'])      + list(slot_rects)
    d.faces[0]['hole_walls'] = list(d.faces[0]['hole_walls']) + list(slot_rects)
    d.meta['cuts']           = list(d.meta['cuts'])           + list(slot_rects)
    return d


def add_extra_lance(d, direction, inner, length, width, fold_dir, bend_r=BEND_R):
    tab, hole, line, th_up = lance(direction, inner, length, width)
    kerf_strips = rect_minus(hole, tab)
    hw = width / 2.0
    if direction == '+x':
        tab2 = (tab[0]+bend_r, tab[1], tab[2], tab[3])
        bend_ext = (inner-bend_r, inner, -hw, hw)
    elif direction == '-x':
        tab2 = (tab[0], tab[1]-bend_r, tab[2], tab[3])
        bend_ext = (-inner, -inner+bend_r, -hw, hw)
    elif direction == '+y':
        tab2 = (tab[0], tab[1], tab[2]+bend_r, tab[3])
        bend_ext = (-hw, hw, inner-bend_r, inner)
    else:
        tab2 = (tab[0], tab[1], tab[2], tab[3]-bend_r)
        bend_ext = (-hw, hw, -inner, -inner+bend_r)
    theta = th_up if fold_dir == 'up' else -th_up
    mv    = 'M'   if fold_dir == 'up' else 'V'
    d.faces[0]['holes']      += [hole, bend_ext]
    d.faces[0]['hole_walls'] += kerf_strips
    new_idx = len(d.faces)
    d.faces.append(dict(rect=tab2, col=TABC))
    d.folds.append((0, new_idx, line, theta))
    d.kids.setdefault(0, []).append((new_idx, line, theta))
    d.meta['cuts']    += kerf_strips
    d.meta['folds2d'].append((line, mv))
    return d


def finalize(d):
    d.tf = [None] * len(d.faces)
    d._assign(d.root, np.eye(3), np.zeros(3))
    return d


def four_taps(d, inner, length, width, fold_dir):
    for dirn in ['+x', '-x', '+y', '-y']:
        add_extra_lance(d, dirn, inner, length, width, fold_dir)
    return finalize(d)


# ════════════════════════════════════════════════════════════════════════════
# RANDOM D4-symmetric design generator  (diverse mode)
# ════════════════════════════════════════════════════════════════════════════
# Everything below is randomly sampled and (mostly) independent:
#   * plate size              40 / 45 / 50 / 55 / 60 mm
#   * leg geometry            leg_inner, leg_len, leg_w  (chosen so 4 outer
#                              legs fit inside the plate with margin)
#   * radiator transparency   alpha in 0.3..0.7
#   * radiator outline        square / octagonal / notched / plus / star
#   * slot features (INDEPENDENT rolls -- a design can have several stacked):
#       - center cross         widths 0.5..2.5 mm
#       - broken ring          widths 0.5..1.5 mm, radius respects legs/ring/tab
#       - 4 corner-diagonal holes
#       - 4 cardinal small holes
#       - 4 edge slits         (square radiator only)
#       - center square hole
#   * extra inner lanced tabs  4 cardinal UP or DOWN tabs, optional
#   * view angle               random elev/azim per design (visual variety)
#
# Per-feature safety rules avoid the failure modes we hit before:
#   - cross length is limited to (leg_inner - 1.5) so it never crosses a leg
#   - when a ring is present, cross/tab/hole radii are kept clear of the
#     ring's 4 cardinal bridges
#   - tab inner radius respects ring radius -2 so tab openings never
#     destroy ring bridges
#   - notched radiator depth must leave room for the legs
# ────────────────────────────────────────────────────────────────────────────

PLATES   = [40.0, 45.0, 50.0, 55.0, 60.0]
LEG_W    = [3.0, 4.0, 5.0, 6.0]
LEG_LENS = [5.0, 7.0, 9.0, 11.0, 13.0]
ALPHAS   = [0.3, 0.4, 0.5, 0.6, 0.7]
SHAPE_WEIGHTS = [('square', 30), ('octagonal', 15), ('notched', 15),
                 ('plus', 20), ('star', 20)]


def _weighted_pick(rng, weighted):
    total = sum(w for _, w in weighted)
    pick = rng.uniform(0, total)
    s = 0
    for v, w in weighted:
        s += w
        if pick <= s: return v
    return weighted[-1][0]


def four_corner_holes(r, size):
    """4 small square holes at the 4 diagonal corner positions (D4-sym)."""
    h = size / 2.0
    return [
        ( r-h,  r+h,  r-h,  r+h),
        (-r-h, -r+h,  r-h,  r+h),
        ( r-h,  r+h, -r-h, -r+h),
        (-r-h, -r+h, -r-h, -r+h),
    ]


def four_cardinal_holes(r, size):
    """4 small square holes on the cardinal axes (D4-sym)."""
    h = size / 2.0
    return [
        ( r-h,  r+h, -h,    h),
        (-r-h, -r+h, -h,    h),
        (-h,    h,    r-h,  r+h),
        (-h,    h,   -r-h, -r+h),
    ]


def four_edge_slits(plate, edge_inset, slit_len, slit_w):
    """4 slits parallel to and inset from each plate edge (D4-sym)."""
    P = plate / 2.0
    e = P - edge_inset
    sh = slit_len / 2.0
    sw = slit_w / 2.0
    return [
        (-sh, sh,  e-sw,  e+sw),       # top
        (-sh, sh, -e-sw, -e+sw),       # bottom
        (-e-sw, -e+sw, -sh, sh),       # left
        ( e-sw,  e+sw, -sh, sh),       # right
    ]


def random_design(rng):
    # ---------- 1. plate + legs ----------------------------------------
    plate   = rng.choice(PLATES)
    P       = plate / 2.0
    leg_w   = rng.choice(LEG_W)
    leg_len = rng.choice(LEG_LENS)

    inner_max = P - leg_len - KERF - 1.5
    inner_min = max(7.0, leg_w + 2.5)
    # if the chosen leg_len is too long for this plate, shorten it
    while inner_min > inner_max and leg_len > 4.0:
        leg_len -= 1.0
        inner_max = P - leg_len - KERF - 1.5
    inner_choices = [round(inner_min + 0.5*i, 1)
                     for i in range(int((inner_max - inner_min) * 2) + 1)]
    inner_choices = [v for v in inner_choices if v <= inner_max]
    leg_inner = rng.choice(inner_choices) if inner_choices else inner_min

    alpha = rng.choice(ALPHAS)
    d = planar_on_pcb(plate=plate, leg_inner=leg_inner,
                      leg_len=leg_len, leg_w=leg_w)
    d.faces[0]['alpha'] = alpha

    desc = [f"P{int(plate)}", f"leg({leg_inner:g},L{leg_len:g},w{leg_w:g})", f"a{alpha}"]

    # ---------- 2. radiator outline ------------------------------------
    shape = _weighted_pick(rng, SHAPE_WEIGHTS)
    if shape == 'octagonal':
        c = rng.choice([2.0, 3.0, 4.0, 5.0])
        chamfer_radiator(d, chamfer=c)
        desc.append(f"oct(c{c:g})")
    elif shape == 'notched':
        nw = rng.choice([3.0, 4.0, 5.0])
        nd = rng.choice([2.0, 3.0, 4.0])
        # notch must not eat into leg-opening region
        if P - nd > leg_inner + leg_len + KERF + 1.0:
            notched_radiator(d, notch_hw=nw, notch_d=nd)
            desc.append(f"ntc({nw:g},{nd:g})")
        else:
            shape = 'square'
    elif shape == 'plus':
        opts = [a for a in [8.0, 10.0, 12.0] if a >= leg_w/2.0 + 3.0]
        if opts:
            a = rng.choice(opts); plus_radiator(d, arm_hw=a)
            desc.append(f"plus(a{a:g})")
        else:
            shape = 'square'
    elif shape == 'star':
        opts = [a for a in [5.0, 6.0, 7.0] if a >= leg_w/2.0 + 2.0]
        if opts:
            a = rng.choice(opts); plus_radiator(d, arm_hw=a)
            desc.append(f"star(a{a:g})")
        else:
            shape = 'square'

    # ---------- 3. independent slot features ---------------------------
    feats   = []
    has_ring = False
    ring_r   = None

    # ring slot
    if rng.random() < 0.35:
        r_max = leg_inner - 2.5
        opts  = [r for r in [6.0, 7.0, 8.0, 9.0, 10.0] if r <= r_max]
        if opts:
            ring_r = rng.choice(opts)
            w = rng.choice([0.5, 1.0, 1.5])
            add_slots(d, broken_ring(ring_r, w, gap=2.0))
            has_ring = True
            feats.append(f"ring(r{ring_r:g},w{w})")

    # cross slot
    if rng.random() < 0.40:
        w = rng.choice([0.5, 1.0, 1.5, 2.0, 2.5])
        L_max = ring_r - 2.0 if has_ring else leg_inner - 1.5
        L_opts = [L for L in [4.0, 6.0, 8.0, 10.0, 12.0] if 3.0 <= L <= L_max]
        if L_opts:
            L = rng.choice(L_opts)
            add_slots(d, cross_slot(w, length=L))
            feats.append(f"+(w{w},L{L:g})")

    # diagonal X-slot (TRUE 45-deg parallelogram slots, D4-symmetric -- no grid)
    if rng.random() < 0.30:
        w = rng.choice([0.5, 1.0, 1.5, 2.0])
        if has_ring:
            he_max = ring_r / 1.4142 - w/2.0 - 0.5
        else:
            he_max = leg_inner - 2.0 - w/2.0
        he_opts = [h for h in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0] if h <= he_max]
        if he_opts:
            he = rng.choice(he_opts)
            d.meta.setdefault('poly_cuts', []).extend(x_slot_polygons(he, w))
            feats.append(f"X(he{he:g},w{w})")

    # 4 corner holes
    if rng.random() < 0.25:
        r_max = leg_inner - 3.0
        opts  = [r for r in [3.0, 4.0, 5.0, 6.0, 7.0]
                 if r <= r_max and (not has_ring or abs(r - ring_r) > 1.5)]
        if opts:
            r = rng.choice(opts); s = rng.choice([1.0, 1.5, 2.0])
            add_slots(d, four_corner_holes(r, s))
            feats.append(f"4cor(r{r:g},s{s})")

    # 4 cardinal small holes
    if rng.random() < 0.20:
        r_max = leg_inner - 2.5
        opts  = [r for r in [3.0, 4.0, 5.0, 6.0]
                 if r <= r_max and (not has_ring or abs(r - ring_r) > 1.5)]
        if opts:
            r = rng.choice(opts); s = rng.choice([1.0, 1.5])
            add_slots(d, four_cardinal_holes(r, s))
            feats.append(f"4card(r{r:g})")

    # 4 edge slits (only on plain square radiator -- the cleanest fit)
    if rng.random() < 0.18 and shape == 'square':
        inset = rng.choice([1.5, 2.0, 2.5])
        e_pos = P - inset
        if e_pos > leg_inner + leg_len + KERF + 1.5:
            slit_w = rng.choice([0.5, 1.0])
            slit_L = rng.choice([6.0, 8.0, 10.0, 12.0])
            slit_L = min(slit_L, plate - 8.0)
            add_slots(d, four_edge_slits(plate, inset, slit_L, slit_w))
            feats.append(f"edges(L{slit_L:g})")

    # center square hole
    if rng.random() < 0.10 and not any('X(' in f for f in feats):
        s = rng.choice([2.0, 3.0, 4.0])
        if not has_ring or s/2.0 < ring_r - 1.5:
            add_slots(d, [(-s/2.0, s/2.0, -s/2.0, s/2.0)])
            feats.append(f"cH({s:g})")

    # ---------- 4. extra inner lanced tabs -----------------------------
    extras = "-"
    if rng.random() < 0.35:
        if has_ring:
            inner_max = ring_r - 2.0
            t_in  = rng.choice([2.0, 2.5])
            t_len = rng.choice([2.5, 3.0])
            t_w   = rng.choice([1.5, 2.0])
            ok    = t_in < inner_max
        else:
            inner_max = leg_inner - 5.0
            ok = inner_max >= 3.0
            if ok:
                t_in  = rng.choice([v for v in [3.0, 4.0, 5.0, 6.0] if v <= inner_max])
                t_len = rng.choice([3.0, 4.0, 5.0])
                t_w   = rng.choice([2.0, 2.5, 3.0])
        if ok:
            fold_dir = rng.choice(['up', 'down'])
            four_taps(d, inner=t_in, length=t_len, width=t_w, fold_dir=fold_dir)
            extras = f"4{fold_dir}({t_in:g},L{t_len:g})"

    feat_str = " + ".join(feats) if feats else "plain"
    title = " | ".join(desc) + f" | {feat_str} | tabs:{extras}"
    return d, title


# ════════════════════════════════════════════════════════════════════════════
# Main : build N random designs, open N interactive windows
# ════════════════════════════════════════════════════════════════════════════
N    = 20            # how many designs to draw
SEED = None          # set to an int (e.g. 0) for reproducible runs

if __name__ == "__main__":
    rng = random.Random(SEED)
    legend = [
        Line2D([0], [0], color='k',  lw=2.4,                   label="CUT (slot / lance)"),
        Line2D([0], [0], color=MTN, lw=1.8, ls=(0,(6,4)),     label="MOUNTAIN +90 (UP)"),
        Line2D([0], [0], color=VAL, lw=2.4, ls=(0,(1,2.5)),   label="VALLEY -90 (DOWN)"),
    ]
    # pre-generate all N designs first so we can pair them across figures
    drawn = [random_design(rng) for _ in range(N)]

    # 2 designs per figure : top row = case A, bottom row = case B
    for fig_idx in range(0, N, 2):
        fig = plt.figure(num=f"page {fig_idx//2 + 1}", figsize=(15, 11))
        for sub in range(2):
            idx = fig_idx + sub
            if idx >= N: break
            d, title = drawn[idx]
            pcb     = pcb_box(plate_size=d.meta['plate'],
                              top_z=-d.meta['leg_drop'] - VIS_EPS)
            fillets = bend_fillet_extras(d)
            poly_x  = make_poly_slot_extras(d)
            view = (rng.uniform(14.0, 34.0), rng.uniform(-65.0, -30.0))

            axc = fig.add_subplot(2, 2, sub*2 + 1)
            draw_crease(axc, d)
            axc.set_title(f"#{idx+1}  {title}", fontsize=9, fontweight="bold")

            ax3 = fig.add_subplot(2, 2, sub*2 + 2, projection="3d")
            render_assembly(ax3, d, extras=list(pcb) + poly_x,
                            fillets=list(fillets), view=view)
            ax3.set_title(f"Folded 3D  (view {view[0]:.0f},{view[1]:.0f})",
                          fontsize=10, fontweight="bold")

        fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9,
                   frameon=True, bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(f"Random D4 designs — page {fig_idx//2 + 1} "
                     f"(cases {fig_idx+1} & {min(fig_idx+2, N)})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
