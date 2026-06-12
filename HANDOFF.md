# HANDOFF — DeepCAD EM Surrogate 프로젝트 컨텍스트

> 이 문서는 이전 Claude 세션의 작업 컨텍스트 요약본.
> 새 세션에서 이 파일을 읽으면 진행 상황을 이어받을 수 있음.
> 브랜치: `claude/create-deepcad-viz-patched-wYhk2`

---

## 1. 프로젝트 개요

안테나 CAD 구조 (DeepCAD 토큰 시퀀스) → S-parameter dB 곡선 (S11/S22/S33) 을
예측하는 **EM surrogate 모델**. 최종 목적:

1. surrogate `.pt` 를 만들고 (완료)
2. 상위 옵티마이저가 이 `.pt` 를 호출해서 "타입 1/2/3 중 뭐가 target S-param 에
   가장 적합한지 + 그 타입의 최적 수치" 를 찾게 함 (다른 사람이 작업 예정)

## 2. 파일 구성 (모두 standalone — 서로 import 하지 않음)

| 파일 | 역할 |
|---|---|
| `AE_main.py` | 전체 AE (인코더+디코더) + surrogate 학습. 원조 코드 |
| `surrogate.py` | **인코더 + S-param MLP 만** (디코더 없음). 학습 후 self-contained ckpt 저장 |
| `surrogate_infer.py` | ckpt 로드 → npy/txt 파일 선택 창 → S-param 예측 시각화 |
| `analyze_type_variants.py` | 타입별 구조 variant 분석 + 머지 + canonicalize + 제외 |
| `inverse_design.py` | AE_main ckpt 용 inverse 추론 (별도) |

## 3. 데이터

- 위치: `hfss_results/step_test` (type1), `step_test1` (type2), `step_test2` (type3)
- sample 당 3개 파일: `{base}_tokens.npy` (양자화 정수 — **모델 입력**),
  `{base}_deepcad.json` (mm 원본), `{base}_tokens_float.npy` (정규화 float, 디버그용 — 미사용)
- S-param 정답: `hfss_results/sparam/` (타입별 글롭 패턴 `[12].*` / `3.*` / `4.*`)
- 토큰과 S-param 은 파일명 (sample id) 으로 짝 매칭

## 4. 토큰 포맷 (모델 입력)

- shape `(L, 17)`, int: `[cmd, p0..p15]`
- cmd: `LINE=0 ARC=1 CIRCLE=2 SOL=3 EXT=4 EOS=5 ROLE=6`
- param: 0~1023 (10-bit 양자화), 미사용 슬롯 = `-1`
- 슬롯 사용: LINE p0,p1 (끝점) / ARC p0..p3 (끝점+중점) / CIRCLE p0..p2 (중심+반지름)
  / EXT p0..p7 (scale, origin xyz, extents, bool, etype) / ROLE p0 (role_id)
- 문법: `ROLE → SOL → curves+ → [SOL → curves+]* → EXT` (chunk), 마지막 `EOS`
- **주의**: per-sketch bbox 별로 축 독립 정규화라 토큰만으론 종횡비 소실.
  양자화 범위 밖 (0~1023 외) 값은 embedding 에러

## 5. ckpt 구조 (self-contained)

`ckpt/surrogate_last.pt` 에 포함: `model_state_dict`, `cfg_dict`, `source_code`
(학습 당시 surrogate.py 전문), `common_curve`, `freqs_sel/full`, `max_len`,
`role_name_to_id`, `type_names`, `best_val_metric`.
→ `.pt` 파일 하나만으로 어디서든 모델 복원 가능 (surrogate_infer.py 가 그렇게 함)

- 마지막 학습: 14,125,427 params, max_len=94, n_freq=81 (1~5 GHz), val RMSE 0.381 dB
- 모델: cmd/param embedding (+Fourier) → Transformer encoder → PMA pool → z
  → common_curve + residual MLP → (81, 3) dB

## 6. 최근 주요 결정/작업 (시간순)

1. surrogate.py 에서 디코더/미사용 코드 제거, `@torch.no_grad()` 데코레이터 복원
2. 타입별 입력 시퀀스 예시 출력 + `example_{type}.npy` 자동 저장 기능 추가
3. `analyze_type_variants.py`: 타입 내 구조 variant 자동 그룹화
   - signature = chunk 별 (role, LINE/ARC/CIRCLE/SOL 개수), **파라미터 값은 비교 안 함**
   - `COUNT_TOLERANCE=2`: 개수 ±2 차이는 같은 골격으로 머지 (L30 vs L32 등)
   - `EXCLUDE_VARIANTS_PER_TYPE`: 특정 variant 의 sample 들을 `_excluded/` 로 이동
   - **canonicalize**: variant 들을 dominant 골격으로 토큰 강제 변환
     (`APPLY_CANONICALIZE=True` + `FORCE_SINGLE_SKELETON_PER_TYPE=True` 로 실행하면
     타입별 정확히 1개 골격으로 통일, 원본은 `{dir}_canon_backup/` 백업)
   - 사용자가 preview 확인 후 "왜곡 없음, 허용 가능" 판정 → 실제 적용함
4. canonicalize 된 데이터로 surrogate 재학습 → 새 `surrogate_last.pt`
5. `surrogate_infer.py` 추가: ckpt 자동 로드 (기본 `ckpt/surrogate_last.pt`)
   → npy/txt 파일 선택 창 → 예측 곡선 figure + 콘솔 요약
   - txt 파서는 4가지 포맷 자동 감지 (| 구분 / 공백+이름 / 공백+정수 / CSV)
   - `_tokens_float.npy` 는 명시적으로 거부
6. `surrogate_search.py` 삭제됨

## 7. 옵티마이저 설계 방침 (다른 사람에게 전달된 스펙)

- 타입 = 고정 골격 (canonicalize 후 타입별 1개). cmd 시퀀스/길이/ROLE 고정
- 옵티마이저는 자유 슬롯 (LINE/ARC/CIRCLE 좌표, EXT scale/extents) 만 search
- 값 = 정수 0~1023. 연속 최적화 후 round 권장 (CMA-ES / Bayesian / Nelder-Mead)
- search 범위는 학습 분포 percentile (5~95%) 로 clip 해야 OOD 방지
- loop closure (첫점=끝점) / 자기교차 validity 체크 필요

## 8. 미해결 / 다음 단계 후보

- 사용자 환경 (`G:/jg/MX_AI_code/`) 의 `1_tokens.txt` 가 어떤 포맷인지 확인 중이었음
  — 최신 파서 (commit fdbb000) 로 재시도 예정이었음. 실패 시 에러에 skipped 줄
  예시 3개가 출력되니 그걸 보고 파서 추가 대응
- 학습 데이터 슬롯별 분포 (percentile) 추출 헬퍼 스크립트 (옵티마이저용) — 제안만 한 상태
- canonicalize 후 surrogate 재학습 성능 비교

## 9. 사용자 작업 환경

- 실제 데이터/실행: Windows (`G:/jg/MX_AI_code/`), GPU (cuda)
- 이 repo 샌드박스에는 `hfss_results/` 데이터 없음 → 코드 수정/검증은 합성 데이터로
- 커밋 메시지는 영어, 사용자와의 대화는 한국어
