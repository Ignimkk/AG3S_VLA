# ATS-CLOUD-DIAG — V11AB perception record 에서 `cloud_target` 이 비는 원인 진단 (진단만)

> writer: lead · 읽는 쪽: implementer H (진단, Opus) → 필요하면 verifier · 작성 2026-10-24 KST
> 사용자 지시 2026-10-24: "네 진행하세요".
> 근거: `BASELINE-EVAL-V1.verify.json` issues — SL4 perception run `20261012T225820_46a08d6ff644_sl4-perception-dev` 에서 `cloud_target` 이 has_target 179 call 중 134 call 에서 비어 있다. 그래서 cloud-labeling 의 ATS 행이 45 call 로만 계산됐다.
> - 보충 행(`target_points` 좌표 일치)은 151 call 이다. `target_points` 가 cloud 점과 맞지 않는 call 은 28 / 17 이다.

## 질문

1. 왜 `cloud_target` 이 비는가?
   - `benchmark/runners/perception_replay.py` 의 기록 경로인가(어느 cloud 에 flag 를 다는지, `--no-clouds`, downsample 이후의 index 대응)?
   - 아니면 V11AB `AG3S.process_multi_debug` 의 산출물 구조 때문인가(target 이 manipulated object 의 last geometry 에서 오는 경우: occluded / held, `GroundingStatus`)?
2. `target_points` 는 있는데 cloud 점과 일치하지 않는 call(28 / 17)은 무엇인가? 다른 cloud(fused vs filtered), 다른 frame(held geometry), 좌표 변환?
3. 공정 비교에 쓸 올바른 ATS 표현은 무엇인가? 각 대안의 장단점을 적는다:
   - (a) `cloud_target`
   - (b) `target_points` 를 cloud 에 대응시킨 것
   - (c) perception 을 다시 돌려 point index 를 기록하는 것

## 범위

- **code 수정 금지 — 진단만 한다.**
  - 기록된 npz 와 code 를 읽고, 작은 분석 script 는 `outputs/ats_cloud_diag/` 에 둔다.
  - perception 재실행이 필요하다는 결론이면, 필요한 최소 변경(flag, 기본 off)을 `파일:줄` 로 제안만 한다.
- 정성 진단과 함께 call 별 분류표(원인 범주 × call 수)를 남긴다. 숫자는 이후 verifier 가 재확인한다.
- 산출물은 `ATS-CLOUD-DIAG-H.impl.md` 다. 다음을 쓴다:
  - 원인
  - 증거(`파일:줄`, npz key, call 예시)
  - 공정 비교에 쓸 ATS 표현 추천
  - 그 표현으로 cloud-labeling 을 다시 계산하는 명령(기존 `eval/cloud_labeling.py` 옵션으로 되는지, 아니면 새 옵션이 필요한지)
- GPU 금지. CPU worker 1개. commit 금지. 다른 session 의 파일은 건드리지 않는다.

## 후속 (lead, 2026-10-24) — 진단 결과에 따른 정식 재계산

H 의 진단(`ATS-CLOUD-DIAG-H.impl.md`): `cloud_target` 이 비는 것은 V11AB 가 stand-in(이전 frame geometry)을 target 으로 쓰기 때문이다. 기록 버그가 아니다.

**결정:** 공정 비교의 ATS 주 표현은 **(b+) exclusion geometry 를 같은 cloud 에 5 mm 로 대응시킨 것**이다.
- (a) `cloud_target` 은 "fresh-only" 진단 행으로 남긴다.
- exact / 10 mm 는 sensitivity 행이다.
- 사용자 지시("진행하세요")의 범위 안이다. 기본값을 지금과 같게 두는 옵션 추가이므로 기존 수치는 바뀌지 않는다.

1. **Implementer J:**
   - `eval/cloud_labeling.py` 에 `--ats-repr {cloud_target,target_points,exclusion}` (기본 `cloud_target`) 과 `--ats-tol` (기본 0.005) 을 추가한다. H 의 최소 변경안 그대로다.
   - test 두 가지:
     - 기본 옵션이 V1 결과와 byte 동일하다.
     - `exclusion` 이 H 의 `recompute_ats_arms.py` 결과와 같다.
   - 산출물: `ATS-CLOUD-DIAG-J.impl.md`
2. **Verifier V4 (CPU):**
   - 실행: exclusion 의 exact / 5 mm / 10 mm 와 cloud_target 을 모두 돌리고, segmenter arm 과의 paired bootstrap 을 낸다.
   - 범주별 표: fresh / anchor / occluded / 제외 call 의 수와 지표.
   - stale geometry 와 GT target 의 이동 거리 분포를 낸다. V11AB 가 ESDF 에서 빼는 영역이 실제 물체 위치와 얼마나 어긋나는지가 closed-loop 해석에 필요하다.
   - figure: `figures/baselines/ats_v11ab/` 에 `v4-` 접두어로 둔다.
     - 실제 장면: stale geometry vs 현재 GT
     - 핵심 기능: call 범주 시계열
     - 표
   - 산출물: `ATS-CLOUD-DIAG-V4.verify.json`
