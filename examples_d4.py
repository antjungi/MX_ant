#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
examples_d4.py
==============
20 D4-symmetric press-type planar antenna variations on a 50x50 mm radiator.

Constraints respected for every case:
  - D4 symmetry  : invariant under 90 deg rotation AND mirrors across both
                   diagonals (the two-axis mirror requirement)
  - 4-connectivity: no floating metal islands. Ring slots use 4 cardinal
                   bridges so the inner patch stays attached to the outer
                   ground via thin metal strips.
  - 0.5 mm grid  : every dimension is an integer multiple of 0.5 mm,
                   minimum metal width and minimum gap both >= 0.5 mm.

What varies across the 20 cases:
  - 5 radiator outlines  : square, octagonal (chamfered), side-notched,
                           plus-shape, and "star-like" (plus with very
                           narrow 12 mm arms)
  - 2 slot widths        : thin (0.5 mm) vs thick (1.5 mm)
  - feature mix          : center cross, broken ring, concentric rings,
                           extra inner UP-tabs (mountain folds), extra
                           inner DOWN-tabs (valley folds), and combinations

Run locally:
    python3 examples_d4.py
plt.show() at the bottom opens 20 interactive windows; no PNGs are written.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt   # interactive backend chosen by the user
from matplotlib.lines import Line2D

import press_fold as pf


# ============================================================================
# helper : slot patterns (D4-symmetric)
# ============================================================================
def cross_slot(width, length=10.0):
    """Three non-overlapping rects forming a '+' slot of given arm width."""
    hw = width / 2.0
    return [
        (-length,  length, -hw,     hw),       # horizontal bar
        (-hw,      hw,      hw,     length),   # top vertical
        (-hw,      hw,     -length, -hw),      # bottom vertical
    ]


def broken_ring(radius, width, gap=1.5):
    """Square ring slot, broken into 4 'L' arcs by `gap`-wide cardinal bridges."""
    r0 = radius
    r1 = radius + width
    g  = gap / 2.0
    return [
        # top edge -- left and right halves separated by the bridge
        (-r1, -g, r0, r1), ( g, r1, r0, r1),
        # bottom edge
        (-r1, -g, -r1, -r0), ( g, r1, -r1, -r0),
        # left edge -- top and bottom halves
        (-r1, -r0,  g, r1), (-r1, -r0, -r1, -g),
        # right edge
        ( r0,  r1,  g, r1), ( r0,  r1, -r1, -g),
    ]


# ============================================================================
# helper : non-square radiator outlines (all D4-symmetric)
# ============================================================================
def chamfer_radiator(d, chamfer=4.0):
    """Octagonal-ish: 4 small corner cuts of (chamfer x chamfer)."""
    h, c = d.meta['plate'] / 2.0, chamfer
    perim = [
        (-h+c, -h),  (h-c, -h),  (h-c, -h+c), ( h,   -h+c),
        ( h,   h-c), (h-c,  h-c),(h-c,  h),    (-h+c,  h),
        (-h+c, h-c), (-h,   h-c),(-h,  -h+c), (-h+c, -h+c),
    ]
    corner_cuts = [
        (-h,    -h+c,   -h,    -h+c),   # bottom-left
        ( h-c,   h,     -h,    -h+c),   # bottom-right
        (-h,    -h+c,    h-c,   h),     # top-left
        ( h-c,   h,      h-c,   h),     # top-right
    ]
    d.faces[0]['perim'] = perim
    d.faces[0]['holes'] = list(d.faces[0]['holes']) + corner_cuts
    d.meta['perim'] = perim
    return d


def plus_radiator(d, arm_hw=10.0):
    """Plus-shape: 4 large corner cuts leave 4 arms each (2*arm_hw) wide."""
    h, a = d.meta['plate'] / 2.0, arm_hw
    perim = [
        (-a, -h), (a, -h), (a, -a), (h, -a),
        ( h,  a), (a,  a), (a,  h), (-a, h),
        (-a,  a), (-h, a), (-h, -a), (-a, -a),
    ]
    corner_cuts = [
        (-h, -a, -h, -a),    # bottom-left
        ( a,  h, -h, -a),    # bottom-right
        (-h, -a,  a, h),     # top-left
        ( a,  h,  a, h),     # top-right
    ]
    d.faces[0]['perim'] = perim
    d.faces[0]['holes'] = list(d.faces[0]['holes']) + corner_cuts
    d.meta['perim'] = perim
    return d


def notched_radiator(d, notch_hw=4.0, notch_d=3.0):
    """Square with 4 mid-side notches that indent the outline inward."""
    h, nw, nd = d.meta['plate'] / 2.0, notch_hw, notch_d
    perim = [
        (-h, -h),    (-nw, -h),    (-nw, -h+nd), (nw, -h+nd), (nw, -h),
        ( h, -h),    ( h, -nw),    ( h-nd, -nw), ( h-nd,  nw), ( h,  nw),
        ( h,  h),    ( nw, h),     ( nw, h-nd),  (-nw, h-nd), (-nw, h),
        (-h,  h),    (-h,  nw),    (-h+nd, nw),  (-h+nd, -nw),(-h, -nw),
    ]
    notches = [
        (-nw,    nw,     -h,      -h+nd),    # bottom
        (-nw,    nw,      h-nd,    h),       # top
        (-h,    -h+nd,   -nw,      nw),      # left
        ( h-nd,  h,      -nw,      nw),      # right
    ]
    d.faces[0]['perim'] = perim
    d.faces[0]['holes'] = list(d.faces[0]['holes']) + notches
    d.meta['perim'] = perim
    return d


# ============================================================================
# helpers : adding slots, extra lances, finalizing
# ============================================================================
def add_slots(d, slot_rects):
    d.faces[0]['holes']      = list(d.faces[0]['holes'])      + list(slot_rects)
    d.faces[0]['hole_walls'] = list(d.faces[0]['hole_walls']) + list(slot_rects)
    d.meta['cuts']           = list(d.meta['cuts'])           + list(slot_rects)
    return d


def add_extra_lance(d, direction, inner, length, width, fold_dir,
                    bend_r=pf.BEND_R):
    tab, hole, line, th_up = pf.lance(direction, inner, length, width)
    kerf_strips = pf.rect_minus(hole, tab)
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
    d.faces.append(dict(rect=tab2, col=pf.TABC))
    d.folds.append((0, new_idx, line, theta))
    d.kids.setdefault(0, []).append((new_idx, line, theta))
    d.meta['cuts']    += kerf_strips
    d.meta['folds2d'].append((line, mv))
    return d


def finalize(d):
    """Recompute transforms after extra lances were added post-construction."""
    d.tf = [None] * len(d.faces)
    d._assign(d.root, np.eye(3), np.zeros(3))
    return d


def four_taps(d, inner, length, width, fold_dir):
    """Lance 4 D4-symmetric cardinal tabs and rebuild the transform tree."""
    for dirn in ['+x', '-x', '+y', '-y']:
        add_extra_lance(d, dirn, inner, length, width, fold_dir)
    return finalize(d)


def base(leg_inner=14.0):
    """The shared 50x50 starting design: 4 down-folded legs, alpha=0.45."""
    d = pf.planar_on_pcb(plate=50.0, leg_inner=leg_inner, leg_len=8.0, leg_w=5.0)
    d.faces[0]['alpha'] = 0.45
    return d


# ============================================================================
# 20 D4-symmetric cases
# ============================================================================
THIN, THICK = 0.5, 1.5

cases = []

# --- square radiator, slot-only (1-5) ---------------------------------------
cases.append(("(01)  square  +  cross slot,  w = 0.5  (thin)",
              add_slots(base(), cross_slot(THIN))))

cases.append(("(02)  square  +  cross slot,  w = 1.5  (thick)",
              add_slots(base(), cross_slot(THICK))))

cases.append(("(03)  square  +  broken ring slot,  w = 0.5",
              add_slots(base(), broken_ring(8.0, THIN))))

cases.append(("(04)  square  +  2 concentric ring slots,  thin",
              add_slots(base(),
                  broken_ring(4.0, THIN) + broken_ring(9.0, THIN))))

cases.append(("(05)  square  +  cross  +  ring  (combined slots)",
              add_slots(base(),
                  cross_slot(THIN, length=6.0) + broken_ring(8.0, THIN))))

# --- square radiator, slot + extra folds (6-9) ------------------------------
_d = add_slots(base(), cross_slot(THIN))
four_taps(_d, inner=4.0, length=4.0, width=2.5, fold_dir='up')
cases.append(("(06)  square  +  thin cross  +  4 inner UP-tabs", _d))

_d = add_slots(base(), cross_slot(THICK))
four_taps(_d, inner=4.0, length=3.0, width=2.5, fold_dir='down')
cases.append(("(07)  square  +  thick cross  +  4 inner DOWN-tabs", _d))

_d = add_slots(base(), broken_ring(8.0, THIN))
four_taps(_d, inner=4.0, length=4.0, width=2.5, fold_dir='up')
cases.append(("(08)  square  +  ring slot  +  4 inner UP-tabs", _d))

_d = add_slots(base(),
               cross_slot(THIN, length=6.0) + broken_ring(8.0, THIN))
four_taps(_d, inner=2.5, length=3.0, width=2.0, fold_dir='up')
cases.append(("(09)  square  +  cross  +  ring  +  4 inner UP-tabs", _d))

# --- octagonal (chamfered) radiator (10-12) ---------------------------------
cases.append(("(10)  octagonal  +  cross slot,  w = 0.5",
              add_slots(chamfer_radiator(base()), cross_slot(THIN))))

cases.append(("(11)  octagonal  +  cross slot,  w = 1.5",
              add_slots(chamfer_radiator(base()), cross_slot(THICK))))

_d = add_slots(chamfer_radiator(base()),
               cross_slot(THIN, length=6.0) + broken_ring(8.0, THIN))
four_taps(_d, inner=2.5, length=3.0, width=2.0, fold_dir='up')
cases.append(("(12)  octagonal  +  cross  +  ring  +  4 UP-tabs", _d))

# --- side-notched radiator (13-15) ------------------------------------------
cases.append(("(13)  side-notched square  +  cross slot,  thin",
              add_slots(notched_radiator(base(leg_inner=11.0)),
                        cross_slot(THIN))))

cases.append(("(14)  side-notched square  +  broken ring slot",
              add_slots(notched_radiator(base(leg_inner=11.0)),
                        broken_ring(8.0, THIN))))

_d = add_slots(notched_radiator(base(leg_inner=11.0)),
               cross_slot(THIN, length=6.0) + broken_ring(7.5, THIN))
four_taps(_d, inner=2.5, length=3.0, width=2.0, fold_dir='up')
cases.append(("(15)  side-notched  +  cross  +  ring  +  4 UP-tabs", _d))

# --- plus-shape radiator (16-18) --------------------------------------------
cases.append(("(16)  plus-shape  +  cross slot,  thin",
              add_slots(plus_radiator(base()), cross_slot(THIN))))

cases.append(("(17)  plus-shape  +  broken ring slot",
              add_slots(plus_radiator(base()), broken_ring(7.5, THIN))))

_d = add_slots(plus_radiator(base()), cross_slot(THIN, length=6.0))
four_taps(_d, inner=2.5, length=3.0, width=2.0, fold_dir='up')
cases.append(("(18)  plus-shape  +  cross  +  4 UP-tabs", _d))

# --- star-like (extreme: very narrow plus arms, 12 mm wide) (19-20) ---------
cases.append(("(19)  star-like  +  cross slot,  thin  (arms 12 mm)",
              add_slots(plus_radiator(base(), arm_hw=6.0),
                        cross_slot(THIN, length=5.0))))

_d = add_slots(plus_radiator(base(), arm_hw=6.0),
               cross_slot(THIN, length=5.0)
               + broken_ring(7.5, THIN, gap=1.0))
four_taps(_d, inner=2.0, length=3.0, width=1.5, fold_dir='up')
cases.append(("(20)  star-like  +  cross  +  ring  +  4 UP-tabs  (extreme)", _d))


# ============================================================================
# build figures (NO savefig, NO close — plt.show() opens 20 windows at end)
# ============================================================================
legend_handles = [
    Line2D([0], [0], color='k',    lw=2.4,                       label="CUT (slot / lance)"),
    Line2D([0], [0], color=pf.MTN, lw=1.8, ls=(0, (6, 4)),       label="MOUNTAIN +90 (fold UP)"),
    Line2D([0], [0], color=pf.VAL, lw=2.4, ls=(0, (1, 2.5)),     label="VALLEY -90 (fold DOWN)"),
]

for title, d in cases:
    pcb     = pf.pcb_box(plate_size=d.meta['plate'],
                         top_z=-d.meta['leg_drop'] - pf.VIS_EPS)
    fillets = pf.bend_fillet_extras(d)

    fig = plt.figure(num=title, figsize=(14, 7))
    axc = fig.add_subplot(1, 2, 1)
    pf.draw_crease(axc, d)
    axc.set_title("Flat pattern  (D4-symmetric)", fontsize=12, fontweight="bold")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    pf.render_assembly(ax3, d, extras=list(pcb), fillets=list(fillets),
                       view=(22, -48))
    ax3.set_title("Folded 3D  (alpha = 0.45 radiator)",
                  fontsize=12, fontweight="bold")

    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

plt.show()
