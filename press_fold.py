#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
press_fold.py
=============
Rigid-origami (press-type antenna) fold engine + DFM grid/kerf model.

Physical spec (locked):
  plate      100 x 100 mm
  thickness  1 mm
  grid       0.5 mm pitch  -> 200 x 200 cells
  min feature 0.5 mm  on BOTH polarities (metal >= 0.5  AND  gap >= 0.5)
  cut kerf   0.5 mm  (a cut removes a 0.5 mm-wide strip; a U-lance frees a tab
                      whose opening = tab footprint + kerf on the 3 cut sides)
  folds      90 deg only, on grid edges

A design is just (faces, folds):
  faces[i] = dict(rect=(x0,x1,y0,y1), holes=[...], col=rgb)
  folds    = [(parent, child, (Ax,Ay,Bx,By), theta_deg), ...]   line in FLAT coords
The engine folds the flat pattern by rotating each child face about its hinge.
"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---- locked physical constants (mm) ----
PLATE = 100.0
GRID  = 0.5
THICK = 1.0
KERF  = 0.5

STEEL = np.array([0.66, 0.70, 0.78])
TABC  = np.array([0.55, 0.78, 0.55])
LIGHT = np.array([0.4, 0.5, 0.78]); LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMB, DIF = 0.42, 0.58
MTN, VAL = "#d62728", "#1f77b4"   # mountain / valley fold colours (2D crease)


def q(v):
    """snap a length to the 0.5 mm grid."""
    return round(v / GRID) * GRID


def rot_axis(d, deg):
    d = np.asarray(d, float); d = d / np.linalg.norm(d); x, y, z = d
    th = np.radians(deg); c, s = np.cos(th), np.sin(th); C = 1 - c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def rect_minus(r, h):
    a0, a1, b0, b1 = r; c0, c1, d0, d1 = h
    ix0, ix1 = max(a0, c0), min(a1, c1); iy0, iy1 = max(b0, d0), min(b1, d1)
    if ix0 >= ix1 or iy0 >= iy1:
        return [r]
    out = []
    if b0 < iy0: out.append((a0, a1, b0, iy0))
    if iy1 < b1: out.append((a0, a1, iy1, b1))
    if a0 < ix0: out.append((a0, ix0, iy0, iy1))
    if ix1 < a1: out.append((ix1, a1, iy0, iy1))
    return out


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
        M, t = self.tf[i]; return M @ np.array([u, v, 0.0]) + t

    def _normal(self, i):
        M, _ = self.tf[i]; n = M @ np.array([0, 0, 1.0]); return n / np.linalg.norm(n)

    def quads(self, thick=THICK):
        out = []
        for i, f in enumerate(self.faces):
            rect = f['rect']; holes = f.get('holes', []); rgb = f.get('col', STEEL)
            n = self._normal(i); off = 0.5 * thick * n
            top = lambda u, v: self._map(i, u, v) + off
            bot = lambda u, v: self._map(i, u, v) - off
            tiles = [rect]
            for h in holes:
                nt = []
                for r in tiles: nt += rect_minus(r, h)
                tiles = nt
            for (x0, x1, y0, y1) in tiles:
                if x1-x0 <= 1e-9 or y1-y0 <= 1e-9: continue
                out.append(([top(x0,y0), top(x1,y0), top(x1,y1), top(x0,y1)], rgb))
                out.append(([bot(x0,y0), bot(x1,y0), bot(x1,y1), bot(x0,y1)], rgb))
            def walls(x0, x1, y0, y1):
                for (p, qq) in [((x0,y0),(x1,y0)), ((x1,y0),(x1,y1)),
                                ((x1,y1),(x0,y1)), ((x0,y1),(x0,y0))]:
                    out.append(([top(*p), top(*qq), bot(*qq), bot(*p)], rgb))
            walls(*rect)
            for h in holes: walls(*h)
        return out

    def bbox_xy(self):
        """flat-pattern bounding box (for fit checking)."""
        xs, ys = [], []
        for f in self.faces:
            x0, x1, y0, y1 = f['rect']; xs += [x0, x1]; ys += [y0, y1]
        return min(xs), max(xs), min(ys), max(ys)


def _newell(v):
    n = np.zeros(3); m = len(v)
    for i in range(m):
        a = v[i]; b = v[(i+1) % m]
        n[0] += (a[1]-b[1])*(a[2]+b[2]); n[1] += (a[2]-b[2])*(a[0]+b[0]); n[2] += (a[0]-b[0])*(a[1]+b[1])
    nn = np.linalg.norm(n); return n/nn if nn > 1e-12 else n


def plus_perim(a, hw, L):
    """outline of a Greek-cross net: square half-size a, arm half-width hw, arm length L."""
    return [(-hw,-a-L),(hw,-a-L),(hw,-a),(a,-a),(a,-hw),(a+L,-hw),(a+L,hw),(a,hw),(a,a),
            (hw,a),(hw,a+L),(-hw,a+L),(-hw,a),(-a,a),(-a,hw),(-a-L,hw),(-a-L,-hw),(-a,-hw),(-a,-a),(-hw,-a)]


def draw_crease(ax, design, sym_axes=True, pad=4.0):
    """Draw the 2D flat pattern (crease pattern) for a design with .meta."""
    from matplotlib.patches import Rectangle, Polygon
    m = design.meta
    for f in design.faces:                       # face fills (full rect, ignore holes)
        x0, x1, y0, y1 = f['rect']
        green = np.allclose(f.get('col', STEEL), TABC)
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0,
                     fc=("#d7efd7" if green else "#dfe6f2"), ec='none', zorder=1))
    xs = [p[0] for p in m['perim']]; ys = [p[1] for p in m['perim']]
    if sym_axes:
        mx = max(max(map(abs,xs)), max(map(abs,ys))) * 1.12
        for seg in ([[-mx,mx],[0,0]], [[0,0],[-mx,mx]]):
            ax.plot(seg[0], seg[1], color="#9a9a9a", lw=0.9, ls=(0,(5,3,1,3)), zorder=0)
    ax.add_patch(Polygon(m['perim'], closed=True, fill=False, ec='k', lw=2.4, zorder=4))
    for (x0,x1,y0,y1) in m.get('cuts', []):      # kerf strips / slots (removed)
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fc='white', ec='k', lw=1.0, zorder=5))
    for (Ax,Ay,Bx,By), mv in m['folds2d']:        # fold lines
        col = MTN if mv == 'M' else VAL
        ls = (0,(6,4)) if mv == 'M' else (0,(6,2,1,2))
        ax.plot([Ax,Bx], [Ay,By], color=col, lw=1.8, ls=ls, zorder=6)
    ax.set_xlim(min(xs)-pad, max(xs)+pad); ax.set_ylim(min(ys)-pad, max(ys)+pad)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def render(ax, design, view=(26, -52), thick=THICK):
    quads = design.quads(thick); polys, cols, pts = [], [], []
    for verts, rgb in quads:
        f = AMB + DIF * abs(float(np.dot(_newell(verts), LIGHT)))
        polys.append(verts); cols.append(tuple(np.clip(rgb*f, 0, 1))); pts += verts
    pc = Poly3DCollection(polys, facecolors=cols, edgecolor='none'); pc.set_zsort('average')
    ax.add_collection3d(pc)
    P = np.array(pts); lo = P.min(0); hi = P.max(0); c = (lo+hi)/2; r = (hi-lo).max()/2*1.08
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=view[0], azim=view[1]); ax.set_axis_off()


# =====================================================================
#  Kerf-aware U-lance:  free a tab; the plate opening = tab + kerf (3 sides)
# =====================================================================
def lance(direction, inner, length, width, kerf=KERF):
    """Return (tab_rect, plate_hole_rect, fold_line, theta) for a tab folded up.
    direction in {'+x','-x','+y','-y'}; inner = fold distance from centre."""
    hw = width / 2.0
    if direction == '+x':
        tab = (inner, inner+length, -hw, hw)
        hole = (inner, inner+length+kerf, -hw-kerf, hw+kerf)
        return tab, hole, (inner, -hw, inner, hw), -90
    if direction == '-x':
        tab = (-inner-length, -inner, -hw, hw)
        hole = (-inner-length-kerf, -inner, -hw-kerf, hw+kerf)
        return tab, hole, (-inner, -hw, -inner, hw), +90
    if direction == '+y':
        tab = (-hw, hw, inner, inner+length)
        hole = (-hw-kerf, hw+kerf, inner, inner+length+kerf)
        return tab, hole, (-hw, inner, hw, inner), +90
    if direction == '-y':
        tab = (-hw, hw, -inner-length, -inner)
        hole = (-hw-kerf, hw+kerf, -inner-length-kerf, -inner)
        return tab, hole, (-hw, -inner, hw, -inner), -90
    raise ValueError(direction)


# =====================================================================
#  Example designs (real mm, grid-quantised, fit in 100 x 100)
# =====================================================================
def closed_box(a=20.0, h=20.0):
    F = [dict(rect=(-a,a,-a,a)), dict(rect=(a,a+h,-a,a)), dict(rect=(-a-h,-a,-a,a)),
         dict(rect=(-a,a,a,a+h)), dict(rect=(-a,a,-a-h,-a))]
    fo = [(0,1,(a,-a,a,a),-90), (0,2,(-a,-a,-a,a),+90),
          (0,3,(-a,a,a,a),+90), (0,4,(-a,-a,a,-a),-90)]
    d = Design(F, fo)
    d.meta = dict(perim=plus_perim(a,a,h), cuts=[],
                  folds2d=[(f[2],'M') for f in fo])
    return d

def cross_fins(a=20.0, hw=5.0, h=25.0):
    F = [dict(rect=(-a,a,-a,a)), dict(rect=(a,a+h,-hw,hw)), dict(rect=(-a-h,-a,-hw,hw)),
         dict(rect=(-hw,hw,a,a+h)), dict(rect=(-hw,hw,-a-h,-a))]
    fo = [(0,1,(a,-hw,a,hw),-90), (0,2,(-a,-hw,-a,hw),+90),
          (0,3,(-hw,a,hw,a),+90), (0,4,(-hw,-a,hw,-a),-90)]
    d = Design(F, fo)
    d.meta = dict(perim=plus_perim(a,hw,h), cuts=[],
                  folds2d=[(f[2],'M') for f in fo])
    return d

def crown(a=20.0, wall=15.0, tip=8.0):
    L = wall + tip
    F = [dict(rect=(-a,a,-a,a)),
         dict(rect=(a,a+wall,-a,a)), dict(rect=(-a-wall,-a,-a,a)),
         dict(rect=(-a,a,a,a+wall)), dict(rect=(-a,a,-a-wall,-a)),
         dict(rect=(a+wall,a+L,-a,a),col=TABC), dict(rect=(-a-L,-a-wall,-a,a),col=TABC),
         dict(rect=(-a,a,a+wall,a+L),col=TABC), dict(rect=(-a,a,-a-L,-a-wall),col=TABC)]
    fo = [(0,1,(a,-a,a,a),-90), (0,2,(-a,-a,-a,a),+90), (0,3,(-a,a,a,a),+90), (0,4,(-a,-a,a,-a),-90),
          (1,5,(a+wall,-a,a+wall,a),+90), (2,6,(-a-wall,-a,-a-wall,a),-90),
          (3,7,(-a,a+wall,a,a+wall),-90), (4,8,(-a,-a-wall,a,-a-wall),+90)]
    d = Design(F, fo)
    d.meta = dict(perim=plus_perim(a,a,L), cuts=[],
                  folds2d=[(f[2],'M' if i < 4 else 'V') for i,f in enumerate(fo)])
    return d

def lanced_tabs(plate=PLATE, inner=10.0, length=15.0, width=10.0):
    P = plate / 2.0
    holes, tabfaces, folds, cuts, folds2d = [], [], [], [], []
    for k, ddirec in enumerate(['+x','-x','+y','-y']):
        tab, hole, line, th = lance(ddirec, inner, length, width)
        holes.append(hole)
        tabfaces.append(dict(rect=tab, col=TABC))
        folds.append((0, 1+k, line, th))
        cuts += rect_minus(hole, tab)      # the 3 kerf strips of the U-lance
        folds2d.append((line, 'M'))
    faces = [dict(rect=(-P,P,-P,P), holes=holes)] + tabfaces
    d = Design(faces, folds)
    d.meta = dict(perim=[(-P,-P),(P,-P),(P,P),(-P,P)], cuts=cuts, folds2d=folds2d)
    return d


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    cases = [
        ("Closed box\ncenter 40x40, wall 20mm", closed_box(), (26,-52)),
        ("Cross fins\ncenter 40x40, fin 10x25mm", cross_fins(), (24,-52)),
        ("Crown / top-loaded\nwall 15, tip 8mm", crown(), (24,-52)),
        ("Lanced 4-tab on 100x100\ntab 10x15mm, kerf 0.5", lanced_tabs(), (34,-54)),
    ]
    fig = plt.figure(figsize=(16, 8.6))
    for j, (name, d, vw) in enumerate(cases):
        axc = fig.add_subplot(2, 4, j+1)                       # row 1: crease (전개도)
        draw_crease(axc, d)
        axc.set_title(name, fontsize=10.5, fontweight="bold")
        ax3 = fig.add_subplot(2, 4, j+5, projection="3d")      # row 2: folded 3D
        render(ax3, d, view=vw)
        x0,x1,y0,y1 = d.bbox_xy()
        fits = "fits" if (x1-x0<=PLATE and y1-y0<=PLATE) else "TOO BIG"
        print(f"{name.splitlines()[0]:18s} flat bbox = {x1-x0:.1f} x {y1-y0:.1f} mm  [{fits}]")
    leg = [Line2D([0],[0],color='k',lw=2.4,label="CUT / blank outline"),
           Line2D([0],[0],color=MTN,lw=1.8,ls=(0,(6,4)),label="MOUNTAIN +90"),
           Line2D([0],[0],color=VAL,lw=1.8,ls=(0,(6,2,1,2)),label="VALLEY +90")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5,0.005))
    fig.suptitle("press_fold.py  —  flat pattern (crease) + folded 3D  "
                 "(100x100mm, t=1mm, 0.5mm grid, kerf 0.5mm)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0,0.04,1,0.97])
    fig.savefig("/tmp/press_fold_demo.png", dpi=130, bbox_inches="tight")
    print("saved /tmp/press_fold_demo.png")
