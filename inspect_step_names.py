#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 파일의 MANIFOLD_SOLID_BREP 이름을 type 별로 쭉 출력하는 진단 툴.

각 type 폴더의 STEP 파일을 읽어 각 solid 의 이름 (HFSS object name) 을 표시한다.
- 의미 있는 이름이면 (Frame, Port_1, GND, ...) → 그대로 role 로 매핑 가능
- 무의미한 default 이름이면 (Solid_1, Solid_2, ...) → HFSS 에서 이름부터 정리 필요

사용:
    python inspect_step_names.py
    python inspect_step_names.py --types type1 type3
    python inspect_step_names.py --max-files 3        # type 당 첫 N개만
    python inspect_step_names.py --summary            # 파일별 안 보이고 unique 이름만
"""

import os
import re
import glob
import argparse
from collections import Counter


def load_entities(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    m = re.search(r"DATA;(.*?)ENDSEC;", content, re.DOTALL)
    if not m:
        return {}
    flat = re.sub(r"[\r\n\t]+", " ", m.group(1))
    ents = {}
    for seg in flat.split(";"):
        seg = seg.strip()
        m2 = re.match(r"#(\d+)\s*=\s*(.+)$", seg, re.DOTALL)
        if m2:
            ents[int(m2.group(1))] = m2.group(2).strip()
    return ents


def find_solid_names(ents):
    """STEP entities 에서 MANIFOLD_SOLID_BREP 의 이름 리스트 반환 (등장 순)."""
    names = []
    # 등장 순서 보존을 위해 entity id 오름차순으로
    for eid in sorted(ents.keys()):
        s = ents[eid]
        if not s.startswith("MANIFOLD_SOLID_BREP"):
            continue
        m = re.match(r"MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'", s)
        if m:
            names.append(m.group(1))
    return names


def inspect_type(type_name, type_dir, max_files=None, summary_only=False):
    print("\n" + "=" * 70)
    print(f"  TYPE: {type_name}    dir: {type_dir}")
    print("=" * 70)

    if not os.path.isdir(type_dir):
        print("  (dir not found)")
        return

    step_files = []
    for ext in ("*.step", "*.stp", "*.STEP", "*.STP"):
        step_files.extend(glob.glob(os.path.join(type_dir, "**", ext), recursive=True))
        step_files.extend(glob.glob(os.path.join(type_dir, ext)))
    step_files = sorted(set(step_files))

    if not step_files:
        print("  (no STEP files found)")
        return
    print(f"  {len(step_files)} STEP file(s) found")

    if max_files is not None:
        step_files = step_files[:max_files]

    all_solid_lists = []        # list of name-tuple per file
    name_counter = Counter()    # name -> n_files containing it
    files_with_n_solids = Counter()  # n_solids -> n_files

    for f in step_files:
        try:
            ents = load_entities(f)
            names = find_solid_names(ents)
        except Exception as e:
            print(f"    [err] {os.path.basename(f)}: {e}")
            continue

        all_solid_lists.append(tuple(names))
        files_with_n_solids[len(names)] += 1
        for nm in set(names):
            name_counter[nm] += 1

        if not summary_only:
            print(f"\n  [{os.path.relpath(f, type_dir)}]   {len(names)} solid(s)")
            for i, nm in enumerate(names):
                print(f"      sk{i:>2}: {nm!r}")

    # ───── summary ─────
    print(f"\n  ─ summary for {type_name} ─")
    print(f"    n_solid distribution: {dict(files_with_n_solids)}")

    print(f"    unique name set (count = files containing it):")
    for nm, cnt in sorted(name_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        ratio = cnt / max(len(step_files), 1)
        tag = "★" if ratio >= 0.95 else ""
        print(f"      {ratio*100:5.1f}% ({cnt:>3}/{len(step_files)})  {nm!r}  {tag}")

    # 같은 순서 (tuple) 인 파일 그룹
    order_groups = Counter(all_solid_lists)
    print(f"    distinct order patterns: {len(order_groups)}")
    for pat, cnt in order_groups.most_common(5):
        print(f"      {cnt:>3} file(s):  {list(pat)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--types", nargs="+", default=["type1", "type2", "type3"],
    )
    parser.add_argument("--max-files", type=int, default=None,
                        help="type 당 첫 N개 파일만 검사")
    parser.add_argument("--summary", action="store_true",
                        help="파일별 출력 생략, 요약만")
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
        inspect_type(tname, type_dirs[tname],
                     max_files=args.max_files,
                     summary_only=args.summary)


if __name__ == "__main__":
    main()
