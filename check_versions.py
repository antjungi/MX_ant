# -*- coding: utf-8 -*-
"""
check_versions.py

이 repo 의 스크립트들 (surrogate.py / surrogate_infer.py / AE_main.py /
analyze_type_variants.py / inverse_design.py) 이 사용하는 외부 라이브러리들의
설치 버전을 한눈에 출력한다.

사용법:
    python check_versions.py
"""
import importlib
import platform
import sys

# (표시 이름, import 모듈명) — repo 전체 import 문 기준
LIBS = [
    ("numpy",        "numpy"),
    ("torch",        "torch"),
    ("matplotlib",   "matplotlib"),
    ("pandas",       "pandas"),
    ("scikit-learn", "sklearn"),
    ("cadquery",     "cadquery"),   # AE_main.py 의 STEP export 에서만 사용 (optional)
]


def get_version(module_name):
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(mod, "__version__", "(version attribute 없음)")


def main():
    print("=" * 60)
    print("환경 정보")
    print("=" * 60)
    print(f"  Python     : {sys.version.split()[0]} ({sys.executable})")
    print(f"  Platform   : {platform.platform()}")

    print()
    print("=" * 60)
    print("라이브러리 버전")
    print("=" * 60)
    for display, module_name in LIBS:
        ver = get_version(module_name)
        status = ver if ver is not None else "NOT INSTALLED"
        print(f"  {display:<14}: {status}")

    # tkinter 는 stdlib 이지만 환경에 따라 빠져 있을 수 있어 별도 확인
    try:
        import tkinter
        print(f"  {'tkinter':<14}: {tkinter.TkVersion} (Tk)")
    except ImportError:
        print(f"  {'tkinter':<14}: NOT INSTALLED")

    # torch 가 있으면 CUDA / GPU 상세 정보 추가 출력
    try:
        import torch
    except ImportError:
        return
    print()
    print("=" * 60)
    print("torch / CUDA 상세")
    print("=" * 60)
    print(f"  torch.version.cuda : {torch.version.cuda}")
    print(f"  cuDNN              : {torch.backends.cudnn.version()}")
    print(f"  CUDA available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}              : {torch.cuda.get_device_name(i)}")


if __name__ == "__main__":
    main()
