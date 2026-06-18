#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_d4_v2.py  —  single-file (no external project modules).

v2 vs v1 (random_d4.py):
  * NEW radiator outline 'tooth' (castellated D4-sym): a long-perimeter
    outline carved with N inward teeth per side -- mirror-symmetric within
    each side, leg-attach region preserved
  * NEW _design_metrics(d) -> (outline_length_mm, plate^2 * leg_drop_mm^3),
    used to bias the random sampler toward long edge / small fold volume
  * Each title now leads with 'L=... V=... FoM=L/V^(1/3) | ...' so the
    perimeter-vs-volume figure of merit is visible per design
  * Robustness: outline-shaping holes (any hole touching the plate edge)
    are skipped from the metal-difference and walls -- they're already in
    the perim and re-subtracting them broke GEOS for tooth designs

Randomly generates N D4-symmetric press-type planar antenna designs and
opens N interactive Matplotlib windows.  Set SEED below to reproduce a
run; leave it None for fresh random samples each time.

What is randomly drawn for each design:
  * radiator outline  : square | octagonal | side-notched | plus | star
                        | tooth (castellated)
                        (with random sub-parameters within safe ranges)
  * slot pattern      : none | cross | ring | concentric rings | cross+ring
                        | corner brackets | diagonal slits | X-slot
  * extra fold tabs   : none | 4 inner UP-tabs | 4 inner DOWN-tabs

All designs satisfy the locked physical / DFM spec:
  plate    40-60 mm side  thickness 1 mm    grid 0.5 mm
  kerf     0.5 mm         bend radius 1 mm
  metal/gap min width    1.0 mm everywhere on the patch (morphological
                         erosion check + pairwise feature distance)
  4-connectivity         no floating pieces (ring slots use 4 cardinal
                         bridges; combination constraints below keep the
                         bridges intact)
  D4 symmetry            four-fold rotation + mirrors across both diagonals
"""
import os, sys, random
import numpy as np
import matplotlib.pyplot as plt           # interactive backend left to the user
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, PathPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# shapely is used to (a) merge overlapping/adjacent slots into single shapes for
# 2D drawing and (b) verify that every random design leaves at least MIN_METAL
# mm of metal between any two slot features. Falls back to non-merged drawing
# and skips validation if shapely is not installed.
try:
    from shapely.geometry import box as _sh_box, Polygon as _ShPoly, Point as _ShPoint
    from shapely.ops import unary_union as _sh_union, triangulate as _sh_triangulate
    HAS_SHAPELY = True
    try:
        from shapely import constrained_delaunay_triangles as _sh_cdt
        HAS_SH_CDT = True
    except ImportError:
        HAS_SH_CDT = False
except ImportError:
    HAS_SHAPELY = False
    HAS_SH_CDT  = False


# ════════════════════════════════════════════════════════════════════════════
# Locked physical / DFM constants
# ════════════════════════════════════════════════════════════════════════════
PLATE     = 50.0    # plate side (mm)
GRID      = 0.5     # grid pitch (mm)
THICK     = 1.0     # sheet thickness (mm)
KERF      = 0.5     # cut kerf (mm)
BEND_R    = 1.0     # min bend radius (mm)
VIS_EPS   = 0.05    # visualisation-only z gap so PCB top doesn't z-fight legs
MIN_METAL = 1.0     # minimum metal feature width / slot-to-slot gap / slot-to-edge gap (mm)

STEEL = np.array([0.66, 0.70, 0.78])
TABC  = np.array([0.55, 0.78, 0.55])
FR4   = np.array([0.62, 0.83, 0.58])    # PCB substrate (lighter green)
LIGHT = np.array([0.4, 0.5, 0.78]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMB, DIF = 0.42, 0.58
MTN, VAL = "#d62728", "#1f77b4"
FOLD_LS = (0, (1, 1.2))                 # dense dotted (used for BOTH M and V)
FOLD_LW = 2.2


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
        for i in range(len(self.faces)):
            out.extend(self._face_quads(i, thick))
        return out

    def _face_quads(self, i, thick=THICK):
        out = []
        f = self.faces[i]
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

    # Merge ALL slot features into a single shape so overlapping slots read
    # as one blob (not multiple overlapping rectangles + parallelograms).
    # Clip the union to the radiator perimeter so a slot whose tip pokes
    # past the patch (e.g. long cross on plus-shape) does NOT draw a
    # black-outlined 'frame' segment out in empty space.
    union = _design_cuts_union(design)
    if union is not None and not union.is_empty and HAS_SHAPELY:
        union = union.intersection(_ShPoly(m['perim']))
    if union is not None and not union.is_empty:
        geoms = ([union] if union.geom_type == 'Polygon'
                 else [g for g in getattr(union, 'geoms', []) if g.geom_type == 'Polygon'])
        for poly in geoms:
            verts, codes = [], []
            ext = list(poly.exterior.coords)
            verts.extend(ext)
            codes.append(Path.MOVETO)
            codes.extend([Path.LINETO] * (len(ext) - 2))
            codes.append(Path.CLOSEPOLY)
            for hole in poly.interiors:
                inn = list(hole.coords)
                verts.extend(inn)
                codes.append(Path.MOVETO)
                codes.extend([Path.LINETO] * (len(inn) - 2))
                codes.append(Path.CLOSEPOLY)
            ax.add_patch(PathPatch(Path(verts, codes), fc='white', ec='k', lw=1.0, zorder=5))
    else:
        # shapely missing -> fallback: render cuts + polygons separately
        cuts = list(m.get('cuts', []))
        for (x0, x1, y0, y1) in cuts:
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc='white', ec='none', zorder=5))
        for seg in _boundary_segments(cuts):
            kind, c, a, b = seg
            if kind == 'V': ax.plot([c, c], [a, b], color='k', lw=1.0, zorder=6)
            else:           ax.plot([a, b], [c, c], color='k', lw=1.0, zorder=6)
        for poly in m.get('poly_cuts', ()):
            ax.add_patch(Polygon(poly, closed=True, fc='white', ec='k', lw=1.0, zorder=5))

    for (Ax, Ay, Bx, By), mv in m['folds2d']:
        col = MTN if mv == 'M' else VAL
        ax.plot([Ax, Bx], [Ay, By], color=col, lw=FOLD_LW, ls=FOLD_LS, zorder=6)

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


def _face0_shapely_quads(d, thick=THICK):
    """Face-0 (radiator) quads with TRUE Boolean subtraction. The radiator
    polygon has the union of all slot shapes (rects + polygons) carved out
    of it (rad.difference(union(slots))), then the metal region is
    triangulated to feed Poly3DCollection. The slot WALLS are drawn along
    the union's BOUNDARY -- so overlapping or adjacent slot edges that sit
    *inside* the merged shape are not rendered. Only the outer + inner
    silhouette of the combined slot is left, like an HFSS subtract."""
    if not HAS_SHAPELY:
        return None
    f = d.faces[0]
    if not f.get('holes') and not d.meta.get('poly_cuts'):
        return None       # no cuts -> defer to the rect-only engine

    rgb   = f.get('col',  STEEL)
    alpha = f.get('alpha', 1.0)

    # ---- radiator outline ---------------------------------------------
    if 'perim' in f:
        rad = _ShPoly(f['perim'])
    else:
        x0, x1, y0, y1 = f['rect']
        rad = _ShPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    # ---- metal = radiator MINUS union(all interior holes + polygon cuts) -
    # Outline-shaping cuts (those touching the plate edge) are skipped --
    # they're already part of the perim, and re-subtracting can cause GEOS
    # to choke on coincident boundaries.
    P = d.meta['plate'] / 2.0
    EPS = 0.1
    # Folded arms have to be removed from the radiator metal even though
    # their rects touch the plate edge -- the arm metal is rendered by the
    # arm's own child face (after the 90 deg fold), so leaving it in face[0]
    # would double-render the unfolded arm.
    folded_arms = set(tuple(r) for r in d.meta.get('folded_arm_rects', []))
    all_cuts = []
    for h in f.get('holes', []):
        if tuple(h) in folded_arms:
            all_cuts.append(_sh_box(h[0], h[2], h[1], h[3]))
            continue
        if (h[0] <= -P + EPS or h[1] >= P - EPS or
            h[2] <= -P + EPS or h[3] >= P - EPS):
            continue
        all_cuts.append(_sh_box(h[0], h[2], h[1], h[3]))
    for p in d.meta.get('poly_cuts', ()):
        all_cuts.append(_ShPoly(p))
    metal = rad.difference(_sh_union(all_cuts)) if all_cuts else rad
    if metal.is_empty:
        return []

    out = []
    parts = [metal] if metal.geom_type == 'Polygon' else list(metal.geoms)

    # Triangulate the metal (square minus union(slots)). We need a CONSTRAINED
    # Delaunay -- the slot edges have to be in the triangle edges so no
    # triangle crosses a hole boundary. Shapely 2.1+ does this natively.
    # Fall back to unconstrained Delaunay + centroid-inside filter (which is
    # what we used before but visibly misses triangles for non-trivial holes).
    for part in parts:
        if HAS_SH_CDT:
            tris = _sh_cdt(part)
            tri_iter = list(tris.geoms) if hasattr(tris, 'geoms') else [tris]
        else:
            tri_iter = [t for t in _sh_triangulate(part) if part.contains(t.centroid)]
        for tri in tri_iter:
            if tri.is_empty or tri.geom_type != 'Polygon':
                continue
            xy = list(tri.exterior.coords)[:3]
            top = [(x, y,  thick/2.0) for x, y in xy]
            bot = [(x, y, -thick/2.0) for x, y in xy]
            out.append((top, rgb, alpha))
            out.append((bot, rgb, alpha))

    # outer-perimeter walls (always drawn -- the radiator's outline)
    perim = f.get('perim', None)
    if perim is None:
        x0, x1, y0, y1 = f['rect']
        perim = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    # If the arms are folded, the perim of face[0] is the unfolded plus
    # outline -- drawing those walls would put vertical strips floating in
    # mid-air where the arms USED to be (they're now bent down).  The arm
    # child faces draw their own walls instead, and the fold-line edges of
    # the central square are covered by the bend fillets.
    if 'arm_central_hw' not in d.meta:
        for k in range(len(perim)):
            p0 = perim[k]; p1 = perim[(k+1) % len(perim)]
            out.append(([(p0[0], p0[1],  thick/2.0),
                         (p1[0], p1[1],  thick/2.0),
                         (p1[0], p1[1], -thick/2.0),
                         (p0[0], p0[1], -thick/2.0)], rgb, alpha))

    # ---- slot walls = boundary of union(EVERY interior cut) ------------
    # Include lance openings + slots (rects) + polygon cuts together so
    # they merge into one outline; the kerf strips end up absorbed inside
    # the lance opening so there are no stray walls inside the leg holes.
    # Corner cuts (radiator-outline shapers) and bend_exts (fillet relief)
    # are excluded because their boundaries are either on the plate edge
    # or already represented by the fillet's curved surfaces.
    P = d.meta['plate'] / 2.0
    EPS = 0.1
    bend_set = set(tuple(r) for r in d.meta.get('bend_exts', []))
    wall_shapes = []
    for h in f.get('holes', []):
        if tuple(h) in folded_arms:
            continue                         # arm child face draws its own walls
        if (h[0] <= -P + EPS or h[1] >= P - EPS or
            h[2] <= -P + EPS or h[3] >= P - EPS):
            continue                         # corner cut -> shape goes to perim
        if tuple(h) in bend_set:
            continue                         # bend region -> fillet renders it
        wall_shapes.append(_sh_box(h[0], h[2], h[1], h[3]))
    for p in d.meta.get('poly_cuts', ()):
        wall_shapes.append(_ShPoly(p))

    if wall_shapes:
        wall_union = _sh_union(wall_shapes)
        # Clip slot walls to the radiator outline. Otherwise a slot whose tip
        # sticks out past the radiator (e.g. a long cross on a plus-shape
        # patch) would draw 'frame' walls floating in empty space.
        wall_union = wall_union.intersection(rad)
        if not wall_union.is_empty:
            wparts = ([wall_union] if wall_union.geom_type == 'Polygon'
                      else [g for g in getattr(wall_union, 'geoms', []) if g.geom_type == 'Polygon'])
            rad_b = rad.exterior
            for wp in wparts:
                boundaries = [list(wp.exterior.coords)]
                for inter in wp.interiors:
                    boundaries.append(list(inter.coords))
                for coords in boundaries:
                    for k in range(len(coords) - 1):
                        x0, y0 = coords[k]; x1, y1 = coords[k+1]
                        # Skip segments that coincide with the radiator's own
                        # outline -- those walls are already drawn by the
                        # perim-wall loop above.
                        if rad_b.distance(_ShPoint((x0+x1)/2.0, (y0+y1)/2.0)) < 1e-4:
                            continue
                        out.append(([(x0, y0,  thick/2.0),
                                     (x1, y1,  thick/2.0),
                                     (x1, y1, -thick/2.0),
                                     (x0, y0, -thick/2.0)], rgb, alpha))
    return out


def render_assembly(ax, design, extras=(), fillets=(), view=(26, -52), thick=THICK,
                    pcb_frame=None):
    """Render the whole assembly into ONE Poly3DCollection so depth-sorting is
    consistent across radiator, legs, fillets, and extras. Polygon slots in
    face 0 are TRUE subtractions (shapely Polygon.difference + constrained
    Delaunay triangulation) -- the same operation HFSS does when you select
    two shapes and subtract one from the other.

    pcb_frame: optional (plate_size, top_z, pcb_thick) -> draws the PCB as a
    plain wireframe of its bounding box (no fill, no colour); the corners are
    fed into the bbox so the camera still includes the PCB volume."""
    face0 = _face0_shapely_quads(design, thick)
    if face0 is not None:
        main_quads = list(face0)
        for i in range(1, len(design.faces)):
            main_quads.extend(design._face_quads(i, thick))
    else:
        main_quads = list(design.quads(thick))

    all_quads = list(extras) + main_quads + list(fillets)
    polys, cols, pts = _shade_quads(all_quads)

    if pcb_frame is not None:
        plate, top_z, pcb_thick = pcb_frame
        P = plate / 2.0
        z0, z1 = top_z - pcb_thick, top_z
        corners = [(-P,-P,z0),(P,-P,z0),(P,P,z0),(-P,P,z0),
                   (-P,-P,z1),(P,-P,z1),(P,P,z1),(-P,P,z1)]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        segs = [(corners[a], corners[b]) for a, b in edges]
        ax.add_collection3d(Line3DCollection(segs, colors='#888888', linewidths=0.9))
        pts += corners

    pc = Poly3DCollection(polys, facecolors=cols, edgecolor='none')
    pc.set_zsort('average')
    ax.add_collection3d(pc)
    _set_axes_equal_from_pts(ax, pts, view)


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
    holes, legfaces, folds, cuts, folds2d, bend_exts = [], [], [], [], [], []
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
        bend_exts.append(bend_ext)
        legfaces.append(dict(rect=tab2))
        folds.append((0, 1+k, line, -th_up))                # NEGATE -> fold DOWN
        folds2d.append((line, 'V'))                          # valley = down to PCB
    faces = [dict(rect=(-P,P,-P,P), holes=holes, hole_walls=cuts)] + legfaces
    d = Design(faces, folds)
    d.meta = dict(perim=[(-P,-P),(P,-P),(P,P),(-P,P)], cuts=cuts, folds2d=folds2d,
                  leg_drop=leg_len, bend_r=bend_r, plate=plate, bend_exts=bend_exts)
    return d


def base(leg_inner=14.0):
    d = planar_on_pcb(plate=PLATE, leg_inner=leg_inner, leg_len=8.0, leg_w=5.0)
    d.faces[0]['alpha'] = 0.45
    return d


def planar_no_legs(plate=PLATE, leg_drop=8.0):
    """Like planar_on_pcb but with NO cardinal lance legs.  Used when the
    folded-arm structure (fold_plus_arms below) provides the vertical
    support: the arms themselves drop down to the PCB."""
    P = plate / 2.0
    faces = [dict(rect=(-P, P, -P, P), holes=[], hole_walls=[])]
    d = Design(faces, folds=[])
    d.meta = dict(perim=[(-P, -P), (P, -P), (P, P), (-P, P)],
                  cuts=[], folds2d=[], leg_drop=leg_drop, bend_r=BEND_R,
                  plate=plate, bend_exts=[])
    return d


def fold_plus_arms(d, arm_hw, fold_dir='down', bend_r=BEND_R):
    """Convert a plus-shape radiator into central square + 4 foldable arms.
    Each arm bends 90 deg about the arm-base edge of the central square.
    The folded footprint is just the central square (2*arm_hw on a side),
    and the bounding-box volume drops to (2*a)^2 * max(leg_drop, arm_len).

    Pre-conditions:
      - d.faces[0]['perim'] is already the plus polygon (call plus_radiator
        first, or set it manually)
      - d has no leg lances that overlap the arm regions (use planar_no_legs)
    """
    plate = d.meta['plate']
    h = plate / 2.0
    a = arm_hw

    # Per-direction lance-style parameters:
    #   (arm_rect, fold_line, tab2_post-bend, bend_ext, th_up)
    table = {
        '+x': ((a, h, -a, a),
               (a, -a, a, a),
               (a + bend_r, h, -a, a),
               (a,           a + bend_r, -a, a),
               -90),
        '-x': ((-h, -a, -a, a),
               (-a, -a, -a, a),
               (-h, -a - bend_r, -a, a),
               (-a - bend_r, -a, -a, a),
               +90),
        '+y': ((-a, a, a, h),
               (-a, a, a, a),
               (-a, a, a + bend_r, h),
               (-a, a, a, a + bend_r),
               +90),
        '-y': ((-a, a, -h, -a),
               (-a, -a, a, -a),
               (-a, a, -h, -a - bend_r),
               (-a, a, -a - bend_r, -a),
               -90),
    }
    d.meta.setdefault('folded_arm_rects', [])
    for dirn, (arm, line, tab2, bend_ext, th_up) in table.items():
        theta = th_up if fold_dir == 'up' else -th_up
        mv = 'M' if fold_dir == 'up' else 'V'
        d.faces[0]['holes'].append(arm)
        d.faces[0]['holes'].append(bend_ext)
        d.meta['folded_arm_rects'].append(arm)
        d.meta.setdefault('bend_exts', []).append(bend_ext)
        new_idx = len(d.faces)
        d.faces.append(dict(rect=tab2))
        d.folds.append((0, new_idx, line, theta))
        d.kids.setdefault(0, []).append((new_idx, line, theta))
        d.meta['folds2d'].append((line, mv))
    d.meta['arm_central_hw'] = a
    d.meta['arm_length']     = h - a
    return finalize(d)


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


def tooth_radiator(d, n_per_half=2, tooth_w=2.0, tooth_d=2.0,
                   leg_margin=4.0, corner_margin=2.0):
    """Castellated D4-symmetric outline: n_per_half teeth per half-side
    (so 2*n_per_half teeth per side, 8*n_per_half total). Teeth dip inward
    by `tooth_d` and are `tooth_w` wide along the edge. They are mirror-
    symmetric about each side's midpoint and stay `leg_margin` away from it
    so the lance-tab attachment region is untouched.

    This is the 'long-outline' generator: increases the radiator's outer
    perimeter length without changing the bounding plate dimensions, which
    is useful when the design objective is 'long electrical edge, small
    folded volume'."""
    h = d.meta['plate'] / 2.0
    # Corner margin must be at least tooth_d + tooth_w/2 + 1 so the outermost
    # tooth on one edge and the first tooth on the perpendicular edge can't
    # carve into each other's space at the corner (which would create a
    # self-intersecting perim).
    corner_margin = max(corner_margin, tooth_d + tooth_w / 2.0 + 1.0)
    span = (h - corner_margin) - leg_margin
    if n_per_half < 1 or span <= 0: return d
    step = span / n_per_half
    if step < tooth_w + 1.0: return d                # teeth would touch / overlap
    offs = [leg_margin + step * (i + 0.5) for i in range(n_per_half)]
    tw = tooth_w / 2.0; td = tooth_d
    centers = sorted([-o for o in offs] + offs)

    perim, notches = [], []
    perim.append((-h, -h))
    for cx in centers:
        perim.extend([(cx - tw, -h),     (cx - tw, -h + td),
                      (cx + tw, -h + td),(cx + tw, -h)])
        notches.append((cx - tw, cx + tw, -h, -h + td))
    perim.append(( h, -h))
    for cy in centers:
        perim.extend([( h,      cy - tw),( h - td, cy - tw),
                      ( h - td, cy + tw),( h,      cy + tw)])
        notches.append(( h - td, h, cy - tw, cy + tw))
    perim.append(( h,  h))
    for cx in reversed(centers):
        perim.extend([(cx + tw,  h),     (cx + tw,  h - td),
                      (cx - tw,  h - td),(cx - tw,  h)])
        notches.append((cx - tw, cx + tw, h - td, h))
    perim.append((-h,  h))
    for cy in reversed(centers):
        perim.extend([(-h,      cy + tw),(-h + td, cy + tw),
                      (-h + td, cy - tw),(-h,      cy - tw)])
        notches.append((-h, -h + td, cy - tw, cy + tw))

    d.faces[0]['perim'] = perim
    d.faces[0]['holes'] = list(d.faces[0]['holes']) + notches
    d.meta['perim'] = perim
    return d


def _design_metrics(d):
    """(outline_length_mm, bounding_volume_mm3).
    outline_length : length of the radiator outer perimeter (the thick
                     black line in the 2D crease). Higher = longer
                     electrically-active edge.
    bounding_vol   : plate * plate * leg_drop, the volume of the 6-face
                     hexahedron that the folded design fits inside
                     (top radiator + 4 side legs + open bottom).
                     Lower = more compact."""
    perim = d.meta['perim']
    n = len(perim)
    L = sum(((perim[i][0] - perim[(i+1) % n][0])**2 +
             (perim[i][1] - perim[(i+1) % n][1])**2) ** 0.5
            for i in range(n))
    # If the plus arms are folded down, the footprint shrinks to the central
    # square (2*arm_hw on a side) and the height is max(leg_drop, arm_length).
    if 'arm_central_hw' in d.meta:
        a  = d.meta['arm_central_hw']
        al = d.meta['arm_length']
        V  = (2.0 * a) ** 2 * max(d.meta.get('leg_drop', 0.0), al)
    else:
        V  = d.meta['plate'] ** 2 * d.meta.get('leg_drop', 0.0)
    return L, V


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


def _slot_shapes_drawn(d):
    """Cuts shown in the 2D crease pattern: kerf strips + added slot rects +
    polygon cuts. We DON'T include the lance-opening rects from face['holes']
    -- those would white-out the whole tab area; instead just the kerf
    around the tab is drawn, leaving the lance tab visible as part of the
    patch metal. bend_exts and corner cuts are also implicitly skipped."""
    if not HAS_SHAPELY: return []
    out = []
    for r in d.meta.get('cuts', []):
        out.append(_sh_box(r[0], r[2], r[1], r[3]))
    for p in d.meta.get('poly_cuts', ()):
        out.append(_ShPoly(p))
    return out


def _slot_shapes_for_valid(d):
    """Features used by is_valid's pairwise distance check.
    Includes lance openings (so slots can't be placed inside them) but
    skips bend_exts (user-exempted) and corner cuts (radiator outline)."""
    if not HAS_SHAPELY: return []
    P = d.meta['plate'] / 2.0
    EPS = 0.1
    bend_set = set(tuple(r) for r in d.meta.get('bend_exts', []))
    out = []
    seen = set()
    for r in d.meta.get('cuts', []):
        seen.add(tuple(r))
        # Kerf strips that touch the plate edge belong to a perim-tab lance --
        # the 'edge' they sit on IS the cut, not patch metal, so the plate
        # edge distance check would always trip on them. Skip those.
        if r[0] <= -P + EPS or r[1] >= P - EPS or r[2] <= -P + EPS or r[3] >= P - EPS:
            continue
        out.append(_sh_box(r[0], r[2], r[1], r[3]))
    for r in d.faces[0].get('holes', []):
        if tuple(r) in seen: continue
        if r[0] <= -P + EPS or r[1] >= P - EPS or r[2] <= -P + EPS or r[3] >= P - EPS:
            continue
        if tuple(r) in bend_set: continue
        out.append(_sh_box(r[0], r[2], r[1], r[3]))
    for p in d.meta.get('poly_cuts', ()):
        out.append(_ShPoly(p))
    return out


def _design_cuts_union(d):
    """shapely union of all DRAWN slot shapes (used for the 2D crease)."""
    shapes = _slot_shapes_drawn(d)
    if not shapes: return None
    try:
        return _sh_union(shapes)
    except Exception:
        return None


def _metal_polygon(d):
    """The radiator's actual metal region (rad MINUS every interior hole,
    including bend_exts which physically have no metal even though the
    bend rule exempts them for the min-width check). Corner / outline-shaping
    cuts (rects that TOUCH the plate edge) are skipped -- those are already
    absorbed into the perim polygon, and subtracting them again would create
    a GEOS topology conflict on exact boundary edges."""
    if not HAS_SHAPELY: return None
    f = d.faces[0]
    if 'perim' in f:
        rad = _ShPoly(f['perim'])
    else:
        x0, x1, y0, y1 = f['rect']
        rad = _ShPoly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    P = d.meta['plate'] / 2.0
    EPS = 0.1
    folded_arms = set(tuple(r) for r in d.meta.get('folded_arm_rects', []))
    cuts = []
    for h in f.get('holes', []):
        if tuple(h) in folded_arms:
            cuts.append(_sh_box(h[0], h[2], h[1], h[3])); continue
        if (h[0] <= -P + EPS or h[1] >= P - EPS or
            h[2] <= -P + EPS or h[3] >= P - EPS):
            continue                                # outline cut -> in perim
        cuts.append(_sh_box(h[0], h[2], h[1], h[3]))
    for p in d.meta.get('poly_cuts', ()):
        cuts.append(_ShPoly(p))
    if not cuts: return rad
    return rad.difference(_sh_union(cuts))


def _metal_has_min_width(metal, min_w=MIN_METAL, eps=0.05, lost_tol=0.5):
    """Morphological erosion check: the metal must keep at least `min_w` mm
    of width everywhere. We erode by ~(min_w/2 - eps) and then dilate back;
    if any area is lost, some part of the metal was thinner than min_w."""
    if metal is None or metal.is_empty: return True
    d = min_w / 2.0 - eps
    try:
        eroded = metal.buffer(-d, join_style=2)
    except Exception:
        return True
    if eroded.is_empty:
        return False
    try:
        recovered = eroded.buffer(d, join_style=2)
        lost = metal.difference(recovered).area
    except Exception:
        return True
    return lost < lost_tol


def is_valid(d, min_metal=MIN_METAL):
    """Two conditions must hold:
       (a) every pair of non-overlapping slot features is >= min_metal apart
           and every slot is >= min_metal from the plate edge (pairwise check)
       (b) the radiator metal polygon has width >= min_metal everywhere
           (morphological erosion check -- catches thin necks pairwise misses)."""
    if not HAS_SHAPELY: return True
    feats = _slot_shapes_for_valid(d)
    if feats:
        n = len(feats)
        for i in range(n):
            for j in range(i+1, n):
                if feats[i].intersects(feats[j]):
                    continue
                if feats[i].distance(feats[j]) < min_metal:
                    return False
        P = d.meta['plate'] / 2.0
        plate_b = _sh_box(-P, -P, P, P).exterior
        for f in feats:
            if plate_b.distance(f) < min_metal:
                return False
    if not _metal_has_min_width(_metal_polygon(d), min_metal):
        return False
    return True


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
    d.meta.setdefault('bend_exts', []).append(bend_ext)
    new_idx = len(d.faces)
    d.faces.append(dict(rect=tab2))
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
# Perimeter bend-tabs (volume-reduction: bend the outline metal down instead
# of carving it off; outer face of each tab is the plate edge -- no kerf).
# ════════════════════════════════════════════════════════════════════════════
def perim_lance(side, cx, length, width, plate, kerf=KERF):
    """Lance a bend-down tab on the OUTER plate edge.  Differs from `lance`
    in that the outer side of the tab IS the plate edge: the hole adds kerf
    only on the two side-edges, never on the outer (plate-edge) side.
    Returns (tab, hole, line, th_up) compatible with the existing fold engine."""
    h = plate / 2.0
    hw = width / 2.0
    if side == '-y':
        return ((cx-hw, cx+hw, -h, -h+length),
                (cx-hw-kerf, cx+hw+kerf, -h, -h+length),
                (cx-hw, -h+length, cx+hw, -h+length), -90)
    if side == '+y':
        return ((cx-hw, cx+hw, h-length, h),
                (cx-hw-kerf, cx+hw+kerf, h-length, h),
                (cx-hw, h-length, cx+hw, h-length), +90)
    if side == '-x':
        return ((-h, -h+length, cx-hw, cx+hw),
                (-h, -h+length, cx-hw-kerf, cx+hw+kerf),
                (-h+length, cx-hw, -h+length, cx+hw), +90)
    if side == '+x':
        return ((h-length, h, cx-hw, cx+hw),
                (h-length, h, cx-hw-kerf, cx+hw+kerf),
                (h-length, cx-hw, h-length, cx+hw), -90)
    raise ValueError(side)


def add_perim_tab(d, side, cx, length, width, fold_dir='down', bend_r=BEND_R):
    """Lance one edge tab and register it as a child face that bends about
    the inner edge of the tab.  This is the perim-tab version of
    add_extra_lance -- intended for the 'bend the teeth down' compactness
    feature."""
    plate = d.meta['plate']
    tab, hole, line, th_up = perim_lance(side, cx, length, width, plate)
    kerf_strips = rect_minus(hole, tab)
    if side == '-y':
        tab2     = (tab[0], tab[1], tab[2],         tab[3]-bend_r)
        bend_ext = (tab[0], tab[1], tab[3],         tab[3]+bend_r)
    elif side == '+y':
        tab2     = (tab[0], tab[1], tab[2]+bend_r,  tab[3])
        bend_ext = (tab[0], tab[1], tab[2]-bend_r,  tab[2])
    elif side == '-x':
        tab2     = (tab[0],         tab[1]-bend_r,  tab[2], tab[3])
        bend_ext = (tab[1],         tab[1]+bend_r,  tab[2], tab[3])
    else:  # '+x'
        tab2     = (tab[0]+bend_r,  tab[1],         tab[2], tab[3])
        bend_ext = (tab[0]-bend_r,  tab[0],         tab[2], tab[3])
    theta = th_up if fold_dir == 'up' else -th_up
    mv = 'M' if fold_dir == 'up' else 'V'
    d.faces[0]['holes']      += [hole, bend_ext]
    d.faces[0]['hole_walls'] += kerf_strips
    d.meta.setdefault('bend_exts', []).append(bend_ext)
    new_idx = len(d.faces)
    d.faces.append(dict(rect=tab2))
    d.folds.append((0, new_idx, line, theta))
    d.kids.setdefault(0, []).append((new_idx, line, theta))
    d.meta['cuts']    += kerf_strips
    d.meta['folds2d'].append((line, mv))
    return d


def eight_perim_tabs(d, cx_off, length, width, fold_dir='down'):
    """8 D4-symmetric bend-down tabs on the outer perimeter: each plate
    side has 2 tabs at +-cx_off from its midpoint, mirror-symmetric.  All
    four sides identical -> full D4 symmetry."""
    for side in ['-y', '+y', '-x', '+x']:
        for cx in [-cx_off, +cx_off]:
            add_perim_tab(d, side, cx, length, width, fold_dir=fold_dir)
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
ALPHAS   = [0.75, 0.85, 0.95]
SHAPE_WEIGHTS = [('square',     30), ('octagonal',   8), ('notched',  7),
                 ('plus',        6), ('star',        4), ('tooth',   30),
                 ('plus_fold',  15)]


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


def corner_brackets(d, length, width):
    """4 corner '⌐' brackets on the diagonals -- each bracket has a horizontal
    arm and a vertical arm meeting at (sx*d, sy*d), pointing back toward the
    centre.  The two arms of one bracket are mirror images across the
    diagonal, and the 4 brackets form a full D4-symmetric set."""
    w = width / 2.0
    rects = []
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        cx, cy = sx * d, sy * d
        if sx > 0:
            rh = (cx - length, cx,        cy - w,      cy + w)
        else:
            rh = (cx,          cx + length, cy - w,    cy + w)
        if sy > 0:
            rv = (cx - w,      cx + w,    cy - length, cy)
        else:
            rv = (cx - w,      cx + w,    cy,          cy + length)
        rects.append(rh); rects.append(rv)
    return rects


def diagonal_slits(plate_half, length, width, inset):
    """4 short slits parallel to each diagonal (45 deg), placed symmetrically
    near the corners. Returns a list of polygon vertex lists."""
    d = width / (2.0 * 1.4142135623730951)
    L = length / (2.0 * 1.4142135623730951)
    polys = []
    P = plate_half - inset
    # centres on the diagonals at +-(P, P), +-(P, -P) etc.
    for sx, sy in [(1, 1), (-1, -1)]:           # parallel to y=x diagonal
        cx, cy = sx * (P - L), sy * (P - L)
        polys.append([
            (cx - L + d, cy - L - d),
            (cx + L + d, cy + L - d),
            (cx + L - d, cy + L + d),
            (cx - L - d, cy - L + d),
        ])
    for sx, sy in [(1, -1), (-1, 1)]:           # parallel to y=-x diagonal
        cx, cy = sx * (P - L), sy * (P - L)
        polys.append([
            (cx - L - d, cy + L - d),
            (cx + L - d, cy - L - d),
            (cx + L + d, cy - L + d),
            (cx - L + d, cy + L + d),
        ])
    return polys


def random_design(rng, max_retries=25):
    """Reject-resample wrapper: keep drawing until is_valid passes, so every
    returned design has >= MIN_METAL mm of metal between any two slot features
    and to the plate edge. Falls back to the last attempt with a [!] tag if
    no valid sample is found within `max_retries`."""
    last = None
    for _ in range(max_retries):
        d, title = _build_random_design(rng)
        last = (d, title)
        if is_valid(d):
            return d, title
    return last[0], last[1] + "  [!min-metal]"


def _build_random_design(rng):
    # ---------- 0. compactness mode (perim bend-tabs) ------------------
    # When this is on, the radiator's outline metal is BENT down at 8
    # D4-symmetric positions instead of being carved off.  This adds
    # vertical edges to the 3D structure at the perimeter (more 'box-
    # like' antenna shape, smaller leg_drop is enough to keep it stable),
    # so we also bias the leg-length shorter to actually reduce the
    # bounding-volume plate^2 * leg_drop.
    want_perim_tabs = (rng.random() < 0.35)
    # ---------- 1. plate + legs ----------------------------------------
    plate   = rng.choice(PLATES)
    P       = plate / 2.0
    leg_w   = rng.choice(LEG_W)
    leg_len = rng.choice([5.0, 6.0, 7.0]) if want_perim_tabs else rng.choice(LEG_LENS)

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
    elif shape == 'tooth':
        # Castellated outline -- long perimeter, small bounding volume.
        nh = rng.choice([1, 2, 3, 4])
        tw = rng.choice([1.5, 2.0, 2.5])
        td = rng.choice([1.5, 2.0, 2.5, 3.0])
        lm = max(leg_w/2.0 + 2.5, rng.choice([4.0, 5.0, 6.0]))
        tooth_radiator(d, n_per_half=nh, tooth_w=tw, tooth_d=td, leg_margin=lm)
        desc.append(f"tooth(n{nh},w{tw:g},d{td:g})")
    elif shape == 'plus_fold':
        # "Crumpled" plus: 4 arms fold 90 deg down to form a small box.
        # Bounding volume drops to (2a)^2 * arm_length -- much smaller than
        # plate^2 * leg_drop. Arms are the structural support; no leg lances.
        a_opts = [v for v in [8.0, 10.0, 12.0, 14.0]
                  if v >= leg_w/2.0 + 3.0 and (P - v) >= 5.0]
        if a_opts:
            a = rng.choice(a_opts)
            arm_length = P - a
            d = planar_no_legs(plate=plate, leg_drop=arm_length)
            d.faces[0]['alpha'] = alpha
            plus_radiator(d, arm_hw=a)
            fold_plus_arms(d, arm_hw=a, fold_dir='down')
            desc = [f"P{int(plate)}", f"plusfold(a{a:g},Larm{arm_length:g})",
                    f"a{alpha}"]
            feats = []
            L, V = _design_metrics(d)
            fom = L / (V ** (1.0/3.0)) if V > 0 else 0.0
            title = (f"L={L:.0f}mm V={V:.0f}mm³ FoM={fom:.2f} | "
                     + " | ".join(desc) + " | minimal | tabs:-")
            return d, title
        else:
            shape = 'square'


    # ---------- 3. independent slot features ---------------------------
    feats     = []
    ring_rs   = []                                  # all ring radii placed (sorted)
    used_radii = []                                 # radii of corner/cardinal holes etc.

    def _clear_of_rings(r, m=1.5):
        return all(abs(r - rr) > m for rr in ring_rs)

    # ---- broken rings (1 to 3 concentric) -----------------------------
    if rng.random() < 0.55:
        n_rings = rng.choices([1, 2, 3], weights=[60, 30, 10])[0]
        r_max = leg_inner - 2.5
        candidates = [r for r in [4.0, 5.5, 7.0, 8.5, 10.0] if r <= r_max]
        rng.shuffle(candidates)
        for r in candidates[:n_rings]:
            if any(abs(r - rr) < 2.5 for rr in ring_rs): continue
            w = rng.choice([0.5, 1.0, 1.5])
            add_slots(d, broken_ring(r, w, gap=2.0))
            ring_rs.append(r)
            feats.append(f"ring(r{r:g},w{w})")
        ring_rs.sort()

    has_ring  = bool(ring_rs)
    ring_min  = min(ring_rs) if has_ring else None
    ring_max  = max(ring_rs) if has_ring else None

    # ---- cross slot (twice -- big + small are allowed to stack) -------
    for trial, p in enumerate([0.60, 0.30]):
        if rng.random() < p:
            w = rng.choice([0.5, 1.0, 1.5, 2.0, 2.5])
            L_max = (ring_min - 2.0) if has_ring else (leg_inner - 1.5)
            L_opts = [L for L in [3.0, 4.0, 6.0, 8.0, 10.0, 12.0] if 2.5 <= L <= L_max]
            if trial == 1: L_opts = [L for L in L_opts if L <= 5.0]   # 2nd cross is small
            if L_opts:
                L = rng.choice(L_opts)
                add_slots(d, cross_slot(w, length=L))
                feats.append(f"+(w{w},L{L:g})")

    # ---- diagonal X-slot (true 45-deg parallelogram) ------------------
    if rng.random() < 0.45:
        w = rng.choice([0.5, 1.0, 1.5, 2.0])
        he_max = (ring_min/1.4142 - w/2.0 - 0.5) if has_ring else (leg_inner - 2.0 - w/2.0)
        he_opts = [h for h in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0] if h <= he_max]
        if he_opts:
            he = rng.choice(he_opts)
            d.meta.setdefault('poly_cuts', []).extend(x_slot_polygons(he, w))
            feats.append(f"X(he{he:g},w{w})")

    # ---- 4 corner-bracket '⌐' slots -----------------------------------
    if rng.random() < 0.35:
        w_br  = rng.choice([0.5, 1.0])
        max_d = (leg_inner - 1.5) / 1.4142
        d_opts = [v for v in [4.0, 5.0, 6.0, 7.0] if v <= max_d and _clear_of_rings(v*1.4142, 1.0)]
        if d_opts:
            db = rng.choice(d_opts)
            L_max = db - 1.0
            L_opts = [L for L in [2.0, 3.0, 4.0, 5.0] if L <= L_max]
            if L_opts:
                Lb = rng.choice(L_opts)
                add_slots(d, corner_brackets(db, Lb, w_br))
                feats.append(f"brk(d{db:g},L{Lb:g})")

    # ---- diagonal slits (parallel to the 4 diagonals, in corners) -----
    if rng.random() < 0.25:
        sl_w = rng.choice([0.5, 1.0])
        sl_L = rng.choice([3.0, 4.0, 5.0])
        inset = rng.choice([3.0, 4.0, 5.0])
        if P - inset - sl_L/2.0 > leg_inner + leg_len + KERF + 1.0:
            d.meta.setdefault('poly_cuts', []).extend(
                diagonal_slits(P, sl_L, sl_w, inset))
            feats.append(f"dslit(L{sl_L:g})")

    # ---- 4 corner holes -----------------------------------------------
    if rng.random() < 0.40:
        r_max = leg_inner - 3.0
        opts  = [r for r in [3.0, 4.0, 5.0, 6.0, 7.0]
                 if r <= r_max and _clear_of_rings(r) and r not in used_radii]
        if opts:
            r = rng.choice(opts); s = rng.choice([1.0, 1.5, 2.0])
            add_slots(d, four_corner_holes(r, s))
            used_radii.append(r)
            feats.append(f"4cor(r{r:g},s{s})")

    # ---- 4 cardinal small holes ---------------------------------------
    if rng.random() < 0.35:
        r_max = leg_inner - 2.5
        opts  = [r for r in [3.0, 4.0, 5.0, 6.0]
                 if r <= r_max and _clear_of_rings(r) and r not in used_radii]
        if opts:
            r = rng.choice(opts); s = rng.choice([1.0, 1.5])
            add_slots(d, four_cardinal_holes(r, s))
            used_radii.append(r)
            feats.append(f"4card(r{r:g})")

    # ---- 4 edge slits (only on plain square radiator) -----------------
    if rng.random() < 0.30 and shape == 'square':
        inset = rng.choice([1.5, 2.0, 2.5])
        e_pos = P - inset
        if e_pos > leg_inner + leg_len + KERF + 1.5:
            slit_w = rng.choice([0.5, 1.0])
            slit_L = rng.choice([6.0, 8.0, 10.0, 12.0])
            slit_L = min(slit_L, plate - 8.0)
            add_slots(d, four_edge_slits(plate, inset, slit_L, slit_w))
            feats.append(f"edges(L{slit_L:g})")

    # ---- center square hole -------------------------------------------
    if rng.random() < 0.15 and not any('X(' in f for f in feats):
        s = rng.choice([2.0, 3.0, 4.0])
        if not has_ring or s/2.0 < ring_min - 1.5:
            add_slots(d, [(-s/2.0, s/2.0, -s/2.0, s/2.0)])
            feats.append(f"cH({s:g})")

    # ---------- 3b. perim bend-tabs (compactness mode) -----------------
    # 8 D4-symmetric edge tabs lanced into the plate edge and folded DOWN.
    # Compatible only with non-castellated outlines (tooth shape already
    # carves its own notches into the perim; mixing would collide).
    ptab_desc = ""
    if want_perim_tabs and shape in ('square', 'octagonal', 'notched'):
        # cx_off measured from each side's midpoint.  Must clear leg_w/2 +
        # kerf on the inside and stay 1mm away from the corner on the outside.
        cx_max = P - 1.5 - 1.0                                  # corner clearance
        cx_min = leg_w / 2.0 + 2.0                              # leg clearance
        cx_opts = [v for v in [7.0, 9.0, 11.0, 13.0, 15.0] if cx_min <= v <= cx_max]
        if cx_opts:
            cx_off = rng.choice(cx_opts)
            t_len  = rng.choice([2.0, 3.0, 4.0])
            t_w    = rng.choice([1.5, 2.0, 2.5])
            # Tab must fit inside the patch on the inner side (not crash into
            # the slot region near the centre).
            inner_clear = leg_inner - t_len - 1.0
            if inner_clear > 0:
                try:
                    eight_perim_tabs(d, cx_off=cx_off, length=t_len,
                                     width=t_w, fold_dir='down')
                    ptab_desc = f"ptab(off{cx_off:g},L{t_len:g},w{t_w:g})"
                    feats.append(ptab_desc)
                except Exception:
                    pass

    # ---------- 4. extra inner lanced tabs -----------------------------
    extras = "-"
    if rng.random() < 0.50:
        if has_ring:
            inner_max = ring_min - 2.0
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
    L, V = _design_metrics(d)
    # L  = outer-perimeter length (longer ⇒ longer electrical edge)
    # V  = plate^2 * leg_drop (smaller ⇒ more compact 6-face fold)
    # L/V^(1/3) is the dimensionless 'edge-per-compactness' figure of merit.
    fom = L / (V ** (1.0/3.0)) if V > 0 else 0.0
    title = (f"L={L:.0f}mm V={V:.0f}mm³ FoM={fom:.2f} | "
             + " | ".join(desc) + f" | {feat_str} | tabs:{extras}")
    return d, title


# ════════════════════════════════════════════════════════════════════════════
# Main : build N random designs, open N interactive windows
# ════════════════════════════════════════════════════════════════════════════
N    = 20            # how many designs to draw
SEED = None          # set to an int (e.g. 0) for reproducible runs

if __name__ == "__main__":
    rng = random.Random(SEED)
    legend = [
        Line2D([0], [0], color='k',  lw=2.4,                            label="CUT (slot / lance)"),
        Line2D([0], [0], color=MTN, lw=FOLD_LW, ls=FOLD_LS,             label="MOUNTAIN +90 (UP)"),
        Line2D([0], [0], color=VAL, lw=FOLD_LW, ls=FOLD_LS,             label="VALLEY -90 (DOWN)"),
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
            fillets = bend_fillet_extras(d)
            # if shapely is unavailable, fall back to the dark-overlay extras
            # for polygon slots; otherwise render_assembly subtracts them itself
            slot_extras = [] if HAS_SHAPELY else make_poly_slot_extras(d)
            view = (rng.uniform(14.0, 34.0), rng.uniform(-65.0, -30.0))

            axc = fig.add_subplot(2, 2, sub*2 + 1)
            draw_crease(axc, d)
            axc.set_title(f"#{idx+1}  {title}", fontsize=9, fontweight="bold")

            ax3 = fig.add_subplot(2, 2, sub*2 + 2, projection="3d")
            render_assembly(ax3, d, extras=slot_extras,
                            fillets=list(fillets), view=view,
                            pcb_frame=(d.meta['plate'],
                                       -d.meta['leg_drop'] - VIS_EPS, 1.6))
            ax3.set_title(f"Folded 3D  (view {view[0]:.0f},{view[1]:.0f})",
                          fontsize=10, fontweight="bold")

        fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9,
                   frameon=True, bbox_to_anchor=(0.5, 0.005))
        fig.suptitle(f"Random D4 designs — page {fig_idx//2 + 1} "
                     f"(cases {fig_idx+1} & {min(fig_idx+2, N)})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
