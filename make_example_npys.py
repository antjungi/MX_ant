"""
Surrogate .pt 모델 입력용 예시 npy 3개 (type1/2/3) 생성 + 전체 내용 출력.
값은 학습 분포가 아니라 '규칙을 만족하는 최소한의 valid 토큰' 예시임.
"""
import numpy as np

# cmd codes
LINE, ARC, CIRCLE, SOL, EXT, EOS, ROLE = 0, 1, 2, 3, 4, 5, 6
PAD = -1
N_ARGS = 16


def row(cmd, params):
    """[cmd, p0..p15] 한 줄 생성. params 는 길이 0~16."""
    r = [cmd] + list(params) + [PAD] * (N_ARGS - len(params))
    assert len(r) == 17
    return r


def build_type1():
    """type1: 사각형 frame 1개 + 사각형 patch radiator 1개 (2 chunk)"""
    seq = [
        # ── chunk #1: outer rectangular frame (role_id=0) ──
        row(ROLE, [0]),
        row(SOL,  []),
        row(LINE, [200, 200]),    # (200,200)
        row(LINE, [800, 200]),
        row(LINE, [800, 800]),
        row(LINE, [200, 800]),
        row(LINE, [200, 200]),    # 폐합
        row(EXT,  [512, 500, 500, 500, 540, 500, 0, 0]),
        # ── chunk #2: rectangular patch (role_id=1) ──
        row(ROLE, [1]),
        row(SOL,  []),
        row(LINE, [350, 350]),
        row(LINE, [650, 350]),
        row(LINE, [650, 650]),
        row(LINE, [350, 650]),
        row(LINE, [350, 350]),
        row(EXT,  [512, 500, 500, 540, 560, 500, 0, 0]),
        # ── end ──
        row(EOS,  []),
    ]
    return np.array(seq, dtype=np.int64)


def build_type2():
    """type2: frame 1개 + 두 개의 split patch (3 chunk)"""
    seq = [
        # ── chunk #1: frame ──
        row(ROLE, [0]),
        row(SOL,  []),
        row(LINE, [180, 180]),
        row(LINE, [860, 180]),
        row(LINE, [860, 860]),
        row(LINE, [180, 860]),
        row(LINE, [180, 180]),
        row(EXT,  [512, 500, 500, 500, 540, 500, 0, 0]),
        # ── chunk #2: left patch ──
        row(ROLE, [1]),
        row(SOL,  []),
        row(LINE, [300, 350]),
        row(LINE, [480, 350]),
        row(LINE, [480, 700]),
        row(LINE, [300, 700]),
        row(LINE, [300, 350]),
        row(EXT,  [512, 500, 500, 540, 560, 500, 0, 0]),
        # ── chunk #3: right patch ──
        row(ROLE, [1]),                # 같은 role (radiator)
        row(SOL,  []),
        row(LINE, [560, 350]),
        row(LINE, [740, 350]),
        row(LINE, [740, 700]),
        row(LINE, [560, 700]),
        row(LINE, [560, 350]),
        row(EXT,  [512, 500, 500, 540, 560, 500, 0, 0]),
        # ── end ──
        row(EOS,  []),
    ]
    return np.array(seq, dtype=np.int64)


def build_type3():
    """type3: frame 1개 + 원형 슬롯이 뚫린 patch (multi-loop chunk)"""
    seq = [
        # ── chunk #1: frame ──
        row(ROLE, [0]),
        row(SOL,  []),
        row(LINE, [200, 200]),
        row(LINE, [820, 200]),
        row(LINE, [820, 820]),
        row(LINE, [200, 820]),
        row(LINE, [200, 200]),
        row(EXT,  [512, 500, 500, 500, 540, 500, 0, 0]),
        # ── chunk #2: patch 외곽 + 원형 hole (멀티 loop) ──
        row(ROLE, [1]),
        row(SOL,  []),                  # 외곽 loop
        row(LINE, [350, 350]),
        row(LINE, [670, 350]),
        row(LINE, [670, 670]),
        row(LINE, [350, 670]),
        row(LINE, [350, 350]),
        row(SOL,  []),                  # 내부 hole loop
        row(CIRCLE, [510, 510, 80]),    # 중심(510,510), 반지름 80
        row(EXT,  [512, 500, 500, 540, 560, 500, 0, 0]),
        # ── end ──
        row(EOS,  []),
    ]
    return np.array(seq, dtype=np.int64)


CMD_NAME = {0: "LINE", 1: "ARC", 2: "CIRCLE", 3: "SOL", 4: "EXT", 5: "EOS", 6: "ROLE"}


def pretty_print(arr, name):
    print(f"\n{'='*72}\n {name}   shape={arr.shape}   dtype={arr.dtype}\n{'='*72}")
    print(f"{'row':>3} {'cmd':>6}  p0   p1   p2   p3   p4   p5   p6   p7   "
          f"p8   p9   p10  p11  p12  p13  p14  p15")
    print("-" * 110)
    for i, r in enumerate(arr):
        cmd = int(r[0])
        cmdname = CMD_NAME.get(cmd, "?")
        params = " ".join(f"{int(v):>4}" for v in r[1:])
        print(f"{i:>3} {cmdname:>6}  {params}")


def main():
    builders = {"type1": build_type1, "type2": build_type2, "type3": build_type3}
    for name, fn in builders.items():
        arr = fn()
        np.save(f"/home/user/MX_ant/example_{name}.npy", arr)
        pretty_print(arr, name)
        print(f"  → saved /home/user/MX_ant/example_{name}.npy")


if __name__ == "__main__":
    main()
