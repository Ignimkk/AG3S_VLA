# 야간 수행 요약 (2026-11-04 04:46 ~ 11:30 KST)

- 판정 목록: [OVERNIGHT_AUTONOMOUS_DECISIONS_20261104_KO.md](OVERNIGHT_AUTONOMOUS_DECISIONS_20261104_KO.md). ★ 표시를 먼저 보세요: **C4 · C6 · C11 · C14**(AG3S HSR6 probe를 이 기계에서 재현해 사용), **C13**(queue 사고와 중복 run 삭제), **C18 · C19**(순서 변경과 AG3S LII 중단 → pb7 재현 점검), **C17**(LIBERO-Safety 설치 편차).
- 결과 문서: [BASELINE_VS_AG3S_SAME_MACHINE_RESULTS_20261104_KO.md](BASELINE_VS_AG3S_SAME_MACHINE_RESULTS_20261104_KO.md).
- Figure와 표: `outputs/nb1104/agg/` (git 제외).

## 0. 시작 시 확인한 것

| 항목 | 내용 |
|---|---|
| 실험 문서 위치 | 맞습니다: `/mnt/dev/AG3S_VLA/doc/active/experiments/`(26개 문서). 계획서는 `doc/active/plans/AG3S_EXPERIMENT_PLAN_20261030_KO.md`, 기록 정본은 `AG3S/ag3s/docs/AG3S_BASELINE_LOG.md`(BL) · `AG3S_BENCHMARK_LOG.md`(SL) |
| 기계 | **새 GPU container**였다. 이전 기계의 run 기록(sl5f VLA 단독 기록, policy_records, 이전 baseline 결과)과 `outputs/`, baseline weights, LIBERO-Safety 환경이 모두 없었다. 그래서 **비교 표 전체를 이 기계에서 다시 쟀다** |
| 이전 문서 기준 미완료 항목 | AEGIS 2 arm, 같은 20 episode에서 AG3S(TrajOpt) vs CBF, LIBERO-Safety rollout, offline 분할의 원인 분리, 외부 mask를 AG3S에 주입한 closed loop(F2) 등 |

## 1. 끝난 것

| 항목 | 결과 | 한 줄 요약 |
|---|---|---|
| 환경 | `.venv-baselines`(GDINO · SAM · SAM2 + CUDA op · YOLOE weights), `.venv-libero-safety` + asset 15.5 GB | 새 기계에서 baseline · 3번째 benchmark를 돌릴 수 있게 됐다 |
| AG3S HSR6-Panda probe 재현 | VB3 통과 (macro-F1 0.822 / 원래 0.834, grasp recall 0.879 / 0.851) | label 띠 figure: `outputs/nb1104/probe/vb2_bands_remade.png` (아침에 눈 확인 권장, C11) |
| SafeLIBERO spatial **Level I** dev 20 ep × **9 arm** | VLA 단독 · AEGIS / KNOWS / multi-link × (detector, GT 장애물) · AG3S 최종 · AG3S pb7 | 아래 §2 |
| SafeLIBERO spatial **Level II** dev 20 ep × **8 arm** | VLA 단독 · 같은 6 filter arm · AG3S 최종 | 아래 §2 |
| **LIBERO-Safety** VLA 단독: L0 3 suite, L1 · L2 2 suite (각 10 ep) | 공식 위반 L0 0 · 0 · 1, L1 0 · 0, **L2 obstacle_avoidance 6/10** · human_safety 0 | L0 · L1은 위반이 드물어 비교 가치가 낮고, **L2 obstacle_avoidance가 safety layer 비교 후보**. 공식 위반은 policy step만 보고 접촉을 놓칠 수 있다(L0 full-trace 위반 5/10) |
| AG3S 재현 점검 | pb7 LI 2 · 9 · 2 (SL5i 3 · 8 · 3) | 이 기계에서도 AG3S Panda가 이전과 같은 수준으로 동작 |

## 2. 핵심 결과 (SS = 공식 성공 ∧ 충돌 없음, n=20 dev)

| arm | LI TSR | LI 충돌 | **LI SS** | LII TSR | LII 충돌 | **LII SS** |
|---|---:|---:|---:|---:|---:|---:|
| VLA 단독 (π0.5) | 14 | 18 | **2** | 15 | 16 | **4** |
| AEGIS (detector) | 10 | 12 | **7** | 10 | 11 | **4** |
| AEGIS (GT 장애물) | 12 | 10 | **10** | 16 | 10 | **10** |
| KNOWS (detector) | 4 | 19 | **1** | 6 | 17 | **1** |
| KNOWS (GT 장애물) | 16 | 10 | **10** | 14 | 13 | **7** |
| Multi-link CBF (detector) | 8 | 15 | **3** | 10 | 9 | **5** |
| Multi-link CBF (GT 장애물) | 14 | 7 | **10** | 15 | 6 | **12** |
| AG3S 최종 (v3-e0to, 재현 probe) | 0 | 10 | **0** | 2 | 11 | **2** |
| AG3S V11AB-pb7 | 2 | 9 | **2** | — | — | — |

1. **장애물 인식이 결과를 가른다.** 같은 filter에서 GT 장애물을 주면 SS 7–12/20(VLA 대비 불일치 쌍 3–9 대 0–1; KNOWS LII를 뺀 5개는 McNemar p ≤ 0.031). detector 장애물에서는 1–7/20이고 LII에서는 VLA 단독(4/20)과 구별되지 않는다. 11-03 야간과 같은 방향. 원인을 가른 실험은 아니다.
2. **AG3S(Panda 현재 구현)는 두 level 모두 작업 성공을 크게 잃는다**(LI TSR 0/20, LII 2/20, VLA만 성공한 쌍 13–14 대 0). 충돌은 VLA보다 적지만 충돌 없는 episode의 대부분이 미완료(멈춤)다. 실패 층: task 0–2는 쥐기(`held`)까지 가지만 놓기에서 끝나지 않고, task 3은 쥐지 못한다(SL6-H 진단과 같은 층). **현재 숫자로는 "AG3S가 기존 filter보다 safe success가 높다"는 주장을 할 수 없다.**
3. 비교 공정성 주의: AG3S는 장애물을 스스로(attention + depth) 인식하고, filter의 "GT" arm은 privileged 입력이다. 같은 인식 조건의 비교(계획서 F 표의 F2 · F6 cell: 외부 mask 또는 GT mask를 AG3S 서버에 넣는 경로)는 아직 구현되지 않았다.

## 3. 못 한 것 / 아침에 정할 것

1. **P5 offline 분할 재현**(새 VLA 기록 위 GSAM2 · SAM · YOLOE vs ATS): 입력(GT 재렌더 20/20, bit 일치)까지만 만들었다. 다시 돌리려면 `outputs/nb1104/queue.txt`의 주석 줄을 살려 `experiments/segmentation/nb1104_queue.sh`로 돌린다(C21).
2. **AG3S 최종 실패 원인**: 놓기 단계 하강 반전(SL6-H)과 task 3 미파지. SL6-H(`-pc`) · SL6-I(target 교체)는 사용자 판정 대기로 보류된 상태다. AG3S가 baseline과 경쟁하려면 이 층부터 고쳐야 한다는 것이 이번 결과의 함의다.
3. **F2 · F6 cell**(같은 AG3S 제약에 외부 / GT target mask): 기본 꺼짐 flag 구현이 필요하다(계획서 §5 F 전제 작업).
4. **LIBERO-Safety**: L2 obstacle_avoidance(VLA 위반 6/10)에서 AG3S · filter arm을 돌릴지. MagickWand `.so.7` preload를 client 코드에 넣을지(지금은 `LD_LIBRARY_PATH`로 우회).
5. guidance arm(OmniGuide · CFM · NUDGE · Lan-o3dp)과 다른 suite(object · goal · long), test split은 돌리지 않았다.
6. 재현 probe를 받아들일지(C11, figure 확인). 거부하면 AG3S 최종 run 2개를 무효로 한다.

## 4. 정리 상태

- 내 GPU queue는 마지막 job 뒤 `STOP`으로 lock을 놓고 끝나도록 해 두었다. 다른 세션은 없었다.
- **commit하지 않았다.** 새 파일: `experiments/segmentation/nb1104_*.{sh,py}`, 이 문서들. 수정: 11-03 스크립트 6개에 `AG3S_SL_SOURCE_RUN` env override(기본값 불변), `environments/libero-safety/validated-freeze.txt`(설치 스크립트가 덮어씀, 원본은 scratchpad).
