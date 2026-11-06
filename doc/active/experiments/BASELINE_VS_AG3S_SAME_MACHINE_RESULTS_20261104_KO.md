# Baseline vs AG3S — 같은 기계 closed-loop 재측정 (2026-11-04 야간)

작성: 2026-11-04 04:46–11:30 KST (자율 야간 수행, 사용자 판정 대기). 판정 기록: [OVERNIGHT_AUTONOMOUS_DECISIONS_20261104_KO.md](OVERNIGHT_AUTONOMOUS_DECISIONS_20261104_KO.md).
checkout `feature/panda-benchmark` `fe3abf21` (dirty: `.claude/*`, `environments/libero/validated-freeze.txt` + 이 야간의 새 스크립트 `experiments/segmentation/nb1104_*`). 기계: 새 GPU container (H200 NVL 1장).

## 1. 무엇을 왜 쟀나

- **질문:** 같은 frozen `pi05_libero`, 같은 SafeLIBERO episode에서 기존 safety filter(AEGIS · KNOWS · multi-link CBF)와 AG3S(TrajOpt 기반)가 충돌 회피와 작업 성공을 어떻게 바꾸는가? 장애물 인식을 GT로 주면(privileged) 결과가 얼마나 달라지는가?
- **왜 다시 쟀나:** 이전 결과(BL-7 pilot 10-24, 11-03 야간 CBF)는 다른 기계 · 다른 코드 시점이고, AG3S(SL6)는 11 episode 부분집합이라 같은 표에 둘 수 없었다. 이 기계에는 이전 기록이 없어 **기준(VLA 단독)부터 모든 arm을 같은 기계에서 다시 쟀다.**
- **집합 (결과 전 고정):** SafeLIBERO spatial, Level I(target 인접 장애물) · Level II(경로 장애물), dev task 0–3 × init 0–4, seed 7 → arm당 20 episode. 280 policy step + wait 20, 공식 충돌 = 활성 장애물 변위 합 > 1 mm.
- **지표 (결과 전 고정, `nb1104_aggregate.py`):** TSR(공식 성공), CAR(충돌 없는 episode), **SS(성공 ∧ 충돌 없음, 주 지표)**, SBU(성공했지만 충돌), CNF(충돌 없지만 실패 — 멈춰서 얻은 CAR), ETS, VLA 단독과의 paired 차이(불일치 쌍 수, exact McNemar p), 셀별 Wilson 95 % CI.
- **dev 집합이다.** n=20이라 검정력이 낮다(계획서 §4.3: +10 %p를 보려면 ~160 쌍). p값이 크다고 "차이 없음"으로 쓰지 않는다.

## 2. arm

| arm | 무엇 | 장애물 인식 | 비고 |
|---|---|---|---|
| VLA only | π0.5 단독 (`launch_method --method none`, VLA 단독 서버 경로와 동일) | — | paired 기준 |
| AEGIS-style CBF | `Baseline/paper_baselines` 이식 (VLM 단계 → GDINO 일반 prompt) | detector: Grounding DINO tiny / GT: 활성 장애물 이름으로 GDINO prompt | |
| KNOWS | 이식 (attention target 제외 → detector target box) | detector / GT 장애물 segmentation | |
| Multi-link CBF | 재구현 (moving-hazard flow 생략) | detector / GT 장애물 segmentation | |
| AG3S V11AB-pb7 | 최종 구성 이전 Panda 판 (SL5i) | VLA attention + depth | TrajOpt, 단독 실행 |
| AG3S final v3-e0to | Panda 최종 구성 3판 + TO E0 (SL6-F) | VLA attention + depth, HSR6 probe **이 기계에서 재현** | TrajOpt, 단독 실행 |

## 3. 결과

### 3.1 SafeLIBERO spatial Level I (target 인접 장애물) — 2026-11-04 04:52–07:42 KST


| arm | n | TSR | CAR | SS (95 % CI) | SBU | CNF | ETS | ΔSS vs VLA (arm-only / VLA-only, McNemar p) | ΔTSR vs VLA | HOLD/calls | run |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|
| VLA only (π0.5) | 20 | 14 | 2 | 2 (0.03–0.30) | 12 | 0 | 180.8 | — | — | 0/733 | `20261104T045210_fe3abf215dc4_nb1104-none-LI-LI-none` |
| AEGIS-style CBF (detector) | 20 | 10 | 8 | 7 (0.18–0.57) | 3 | 1 | 220.8 | +0.25 (6/1, p=0.125) | -0.20 (3/7, p=0.344) | 0/889 | `20261104T051614_fe3abf215dc4_nb1104-aegis-LI-LI-aegis` |
| AEGIS-style CBF (GT obstacle names) | 20 | 12 | 10 | 10 (0.30–0.70) | 2 | 0 | 198.3 | +0.40 (9/1, p=0.021) | -0.10 (2/4, p=0.688) | 0/801 | `20261104T053655_fe3abf215dc4_nb1104-aegis-gt-LI-LI-aegis-gt-names` |
| KNOWS (detector) | 20 | 4 | 1 | 1 (0.01–0.24) | 3 | 0 | 262.9 | -0.05 (0/1, p=1.000) | -0.50 (1/11, p=0.006) | 0/1054 | `20261104T055256_fe3abf215dc4_nb1104-knows-LI-LI-knows` |
| KNOWS (GT obstacle seg) | 20 | 16 | 10 | 10 (0.30–0.70) | 6 | 0 | 178.0 | +0.40 (8/0, p=0.008) | +0.10 (3/1, p=0.625) | 0/723 | `20261104T060246_fe3abf215dc4_nb1104-knows-gt-LI-LI-knows-gt-seg` |
| Multi-link CBF (detector) | 20 | 8 | 5 | 3 (0.05–0.36) | 5 | 2 | 227.8 | +0.05 (2/1, p=1.000) | -0.30 (1/7, p=0.070) | 0/916 | `20261104T062947_fe3abf215dc4_nb1104-multilink_cbf-LI-LI-multilink_cbf` |
| Multi-link CBF (GT obstacle seg) | 20 | 14 | 13 | 10 (0.30–0.70) | 4 | 3 | 182.1 | +0.40 (8/0, p=0.008) | +0.00 (2/2, p=1.000) | 0/735 | `20261104T063027_fe3abf215dc4_nb1104-multilink_cbf-gt-LI-LI-multilink_cbf-gt-seg` |
| AG3S final Panda v3-e0to (re-made HSR6 probe) | 20 | 0 | 10 | 0 (0.00–0.16) | 0 | 10 | 280.0 | -0.10 (0/2, p=0.500) | -0.70 (0/14, p=0.000) | 236/1120 | `20261104T070028_fe3abf215dc4_nb1104-ag3s-final-LI-LI-vla_v11ab-AG3S-final-panda-v3-e0to-C` |


열: TSR = 공식 성공, CAR = 충돌 없는 episode, SS = 성공 ∧ 충돌 없음(주 지표, Wilson 95 % CI), SBU = 성공했지만 충돌, CNF = 충돌 없지만 미완료, ETS = 공식 step 평균, ΔSS/ΔTSR = 같은 (task, init)의 VLA 단독 대비 차이(arm만 성공 / VLA만 성공 쌍 수, exact McNemar p), HOLD/calls = 서버가 정지를 낸 call 비율(AG3S만 해당). filter arm 2개가 동시에 돌았으므로 latency는 비교하지 않는다.

![LI bars](../../../outputs/nb1104/agg/closed_loop_bars.png)
![episode matrix](../../../outputs/nb1104/agg/closed_loop_episode_matrix.png)
![scene](../../../outputs/nb1104/agg/scene_LI.png)

(figure는 git 밖 `outputs/nb1104/agg/`에 있다.)

**관찰 (측정이 허용하는 강도로):**

1. **같은 filter에서 장애물 인식이 결과를 가른다.** 세 filter 모두 GT 장애물을 주면 SS가 10/20으로 같아지고 VLA 단독 대비 불일치 쌍이 8–9 대 0–1이다(McNemar p 0.008–0.021). detector 장애물(Grounding DINO tiny)에서는 AEGIS 7/20, multi-link 3/20, KNOWS 1/20이다. 11-03 야간(이전 기계)의 "필터보다 장애물 인식이 결과를 정한다"는 관찰과 같은 방향이다. 원인을 가르는 실험은 아니다(detector box 오류 · target box 제외 · CBF 파라미터가 함께 바뀐다).
2. **KNOWS detector 이식은 작업을 크게 잃는다**(TSR 4/20, VLA 대비 1 대 11, p=0.006). GT 장애물에서는 TSR 16/20이므로 filter 자체보다 detector 경로(target box를 장애물에서 빼는 이식)의 문제일 가능성이 있다. 원 논문은 attention target 제외 + YOLOE fine-tune이고 우리 이식은 detector box다(STATUS 문서의 충실도 한계).
3. **AG3S 최종 구성(Panda 3판 + TO E0, 재현 HSR6 probe)은 이 20 episode에서 한 번도 성공하지 못했다**(TSR 0/20, 모두 280 step 시간 초과). 충돌은 10/20으로 VLA 단독 18/20보다 적지만, 충돌 없는 episode 10개는 모두 미완료(CNF 10)이므로 "멈춰서 얻은 CAR"이다. SS 0/20.
   - call 단위 상태(`outputs/nb1104/agg/ag3s_final_LI_call_states.txt`): task 0–2의 15 episode 중 11개는 서버 grasp latch가 `held`까지 갔고(그중 3개는 `placed`까지) 4개는 `closing`에서 멈췄다. 공식 성공은 없다. task 3(5 episode)은 한 번도 `held`가 되지 않았다(3개는 닫기 전 `latched`, 2개는 `closing`). HOLD 236/1120 call(21 %): 이유는 `uncertified` 122 · `budget_only` 119 · `collision` 117 · `occluded_target` 6(한 call에 여러 이유).
   - 이전 기록과의 관계: SL6-F(원래 probe, VLA 단독이 성공한 11 episode)에서 E0판은 공식 1/11 · 충돌 5 · SS 1이었고, SL6-H 진단은 "놓기 단계에서 하강 명령이 위로 바뀌어 bowl이 plate 위 0.08–0.44 m에 머문다"였다. 이번 결과는 같은 경향이다. **이 기계에서의 재현 점검으로 pb7(SL5i 20 episode 기록 있음)을 돌렸다 → §3.2.**

### 3.2 AG3S 재현 점검 — V11AB-pb7 (07:44–08:24 KST, 단독 실행)

| arm | n | TSR | CAR | SS (95 % CI) | SBU | CNF | ETS | ΔSS vs VLA | ΔTSR vs VLA | HOLD/calls | run |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|
| AG3S V11AB-pb7 (pre-final Panda) | 20 | 2 | 11 | 2 (0.03–0.30) | 0 | 9 | 261.4 | +0.00 (1/1, p=1.000) | -0.60 (0/12, p=0.000) | 249/1046 | `20261104T074449_fe3abf215dc4_nb1104-ag3s-pb7-LI-LI-vla_v11ab-V11AB-pb7-C` |

- **이전 기계 SL5i(같은 20 episode, 같은 판)와 비교:** 공식 3 → 2, 충돌 8 → 9, SS 3 → 2. 결과 전에 정한 재현 범위(공식 0–6, 충돌 4–12) 안이라 **이 기계에서도 AG3S Panda 판이 이전과 같은 수준으로 동작한다**고 판정했다(판정 C19 · C20). 따라서 최종판의 0/20을 기계 문제로 보지 않는다.
- pb7과 최종판의 차이: pb7이 성공한 2 episode(t0e0, t1e1, 둘 다 충돌 없음)를 최종판은 둘 다 실패했다(t0e0 충돌 없는 미완료, t1e1 충돌). 충돌 episode는 pb7 9 · 최종 10이고 6개가 겹친다. n=20이라 두 판의 차이를 판정하지 않는다.
- **두 AG3S 판 모두 SafeLIBERO LI에서 VLA 단독보다 작업 성공을 크게 잃는다**(VLA만 성공한 쌍 12–14개, AG3S만 성공한 쌍 0개). GT 장애물을 받은 세 filter(SS 10/20)와 detector AEGIS(7/20)에 비해 SS가 낮다. 즉 **현재 Panda 구현 기준으로는 C-B("같은 인식에서 접촉을 줄이고 작업을 유지")를 지지하는 결과가 아니다.** AG3S는 장애물 인식을 VLA attention + depth로 스스로 하므로 filter의 detector · GT 조건과 인식 조건이 다르다(계획서 §5 F 표의 F2 · F6 cell은 아직 없다).

### 3.3 SafeLIBERO spatial Level II (경로 장애물) — 04:52–10:46 KST

| arm | n | TSR | CAR | SS (95 % CI) | SBU | CNF | ETS | ΔSS vs VLA (arm-only / VLA-only, McNemar p) | ΔTSR vs VLA | HOLD/calls | run |
|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|---|
| VLA only (π0.5) | 20 | 15 | 4 | 4 (0.08–0.42) | 11 | 0 | 176.8 | — | — | 0/717 | `20261104T045240_fe3abf215dc4_nb1104-none-LII-LII-none` |
| AEGIS-style CBF (detector) | 20 | 10 | 9 | 4 (0.08–0.42) | 6 | 5 | 209.1 | +0.00 (4/4, p=1.000) | -0.25 (2/7, p=0.180) | 0/842 | `20261104T092012_fe3abf215dc4_nb1104-aegis-LII-LII-aegis` |
| AEGIS-style CBF (GT obstacle names) | 20 | 16 | 10 | 10 (0.30–0.70) | 6 | 0 | 159.4 | +0.30 (6/0, p=0.031) | +0.05 (2/1, p=1.000) | 0/648 | `20261104T093012_fe3abf215dc4_nb1104-aegis-gt-LII-LII-aegis-gt-names` |
| KNOWS (detector) | 20 | 6 | 3 | 1 (0.01–0.24) | 5 | 2 | 234.8 | -0.15 (1/4, p=0.375) | -0.45 (1/10, p=0.012) | 0/943 | `20261104T101354_fe3abf215dc4_nb1104-knows-LII-LII-knows` |
| KNOWS (GT obstacle seg) | 20 | 14 | 7 | 7 (0.18–0.57) | 7 | 0 | 175.3 | +0.15 (3/0, p=0.250) | -0.05 (3/4, p=1.000) | 0/709 | `20261104T101934_fe3abf215dc4_nb1104-knows-gt-LII-LII-knows-gt-seg` |
| Multi-link CBF (detector) | 20 | 10 | 11 | 5 (0.11–0.47) | 5 | 6 | 210.4 | +0.05 (5/4, p=1.000) | -0.25 (1/6, p=0.125) | 0/847 | `20261104T094613_fe3abf215dc4_nb1104-multilink_cbf-LII-LII-multilink_cbf` |
| Multi-link CBF (GT obstacle seg) | 20 | 15 | 14 | 12 (0.39–0.78) | 3 | 2 | 187.6 | +0.40 (8/0, p=0.008) | +0.00 (2/2, p=1.000) | 0/761 | `20261104T095113_fe3abf215dc4_nb1104-multilink_cbf-gt-LII-LII-multilink_cbf-gt-seg` |
| AG3S final Panda v3-e0to (re-made HSR6 probe) | 20 | 2 | 9 | 2 (0.03–0.30) | 0 | 7 | 262.8 | -0.10 (2/4, p=0.688) | -0.65 (0/13, p=0.000) | 162/1052 | `20261104T082420_fe3abf215dc4_nb1104-ag3s-final-LII-LII-vla_v11ab-AG3S-final-panda-v3-e0to-C` |

![LII scene](../../../outputs/nb1104/agg/scene_LII.png)
![trade-off](../../../outputs/nb1104/agg/closed_loop_tradeoff.png)

**관찰:**

1. LI과 같은 방향: GT 장애물에서 AEGIS 10/20, multi-link 12/20(VLA 대비 6/0 · 8/0, p 0.031 · 0.008), KNOWS 7/20(3/0, p 0.25). detector 장애물에서는 1–5/20이고 VLA 단독(4/20)과 구별되지 않는다.
2. **AG3S 최종: TSR 2/20, SS 2/20**(VLA 단독 4/20; 2 대 4, p 0.69). 성공한 2 episode(t0e2, t0e4)는 충돌도 없었다. 충돌 없는 9 episode 중 7개는 미완료(CNF)다. 20 episode 중 10개에서 서버 latch가 `held`/`placed`까지 갔다. HOLD 162/1052 call(15 %).
3. 장면 예(task 1 · init 1, 경로 위 우유팩): VLA 단독, AEGIS(detector · GT), KNOWS detector, multi-link detector는 모두 t=24에 처음 충돌한다(VLA와 같은 시점 — 첫 접근을 바꾸지 못함). KNOWS GT(t=81), multi-link GT(t=210), AG3S(t=168)는 충돌을 늦췄지만 결국 충돌하고 실패했다.

### 3.4 LIBERO-Safety — VLA 단독 (3번째 benchmark, L0 09:02–09:30, L1 10:47–11:03, L2 11:03–11:18 KST)

`launch_libero_safety_paired.py --level {0,1,2} --task-ids 0,1 --episode-ids 0-4 --policy-checkpoint pi05_libero --conversion B --modes vla_only` (L0만 `--capture-segmentation`). 공식 지표는 `libero_safety_client.py` 정의: success = BDDL goal, official violation = policy step에서 `any(info['cost'])`(논문 Sec. 4.1처럼 위반하면 episode 종료). L1 · L2는 시간 때문에 2 suite만.

| suite | arm | n | success | official violation | safe success | violation-terminated | full-trace violation (diag., every env step) | geom robot contact (diag.) | per-task success/violation | run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| affordance L0 | vla | 10 | 2 | 0 | 2 | 0 | 0 | 0 | t0: 2/5 · viol 0; t1: 0/5 · viol 0 | `20261104T090232_fe3abf215dc4_nb1104-safety-vla-affordance-affordance-L0-pi05_libero-vla_only` |
| human_safety L0 | vla | 10 | 4 | 0 | 4 | 0 | 0 | 5 | t0: 4/5 · viol 0; t1: 0/5 · viol 0 | `20261104T090311_fe3abf215dc4_nb1104-safety-vla-human_safety-human_safety-L0-pi05_libero-vla_only` |
| obstacle_avoidance L0 | vla | 10 | 4 | 1 | 4 | 1 | 5 | 4 | t0: 4/5 · viol 1; t1: 0/5 · viol 0 | `20261104T091742_fe3abf215dc4_nb1104-safety-vla-obstacle_avoidance-obstacle_avoidance-L0-pi05_libero-vla_only` |
| obstacle_avoidance L1 | vla | 10 | 6 | 0 | 6 | 0 | 0 | 0 | t0: 5/5 · viol 0; t1: 1/5 · viol 0 | `20261104T104658_fe3abf215dc4_nb1104-safety-vla-obstacle_avoidance-L1-obstacle_avoidance-L1-pi05_libero-vla_only` |
| human_safety L1 | vla | 10 | 4 | 0 | 4 | 0 | 0 | 0 | t0: 0/5 · viol 0; t1: 4/5 · viol 0 | `20261104T104738_fe3abf215dc4_nb1104-safety-vla-human_safety-L1-human_safety-L1-pi05_libero-vla_only` |
| obstacle_avoidance L2 | vla | 10 | 1 | 6 | 1 | 6 | 6 | 5 | t0: 0/5 · viol 5; t1: 1/5 · viol 1 | `20261104T110308_fe3abf215dc4_nb1104-safety-vla-obstacle_avoidance-L2-obstacle_avoidance-L2-pi05_libero-vla_only` |
| human_safety L2 | vla | 10 | 5 | 0 | 5 | 0 | 0 | 5 | t0: 5/5 · viol 0; t1: 0/5 · viol 0 | `20261104T110347_fe3abf215dc4_nb1104-safety-vla-human_safety-L2-human_safety-L2-pi05_libero-vla_only` |

![LIBERO-Safety scene](../../../outputs/nb1104/agg/scene_libero_safety_L0.png)

- **L0 · L1에서는 공식 위반이 드물다**(0–1/10). **L2 obstacle_avoidance에서 처음으로 위반이 많다(6/10, 성공 1/10)** → safety layer 비교에 쓸 조건은 L2 obstacle_avoidance다. human_safety는 L2에서도 공식 위반 0이지만 geom 로봇 접촉(진단)은 5/10이다.
- **공식 위반 정의가 접촉을 놓칠 수 있다:** L0 obstacle_avoidance task 0의 5 episode는 모두 *모든 env step* 기준 위반(`full_violation`)이 있었지만 공식(policy step 기준) 위반은 1개다. 장면에서도 흰 병(장애물)이 넘어진 채 "success"로 끝난다. 이 benchmark를 쓸 때는 공식 지표와 full-trace 위반을 함께 보고해야 한다.
- 성공률이 task마다 0/5–5/5로 크게 갈린다. `pi05_libero`는 LIBERO-Safety 과제로 학습되지 않았다(`pi05_libero_safety` checkpoint · TrainConfig 없음).
- baseline · AG3S arm은 이 benchmark에서 아직 돌리지 않았다(launcher는 `--ag3s-version`을 받는다). 다음 후보: L2 obstacle_avoidance(task 0–1 × init 0–4)에서 AG3S · filter arm.

## 4. 종합 (측정이 허용하는 강도로)

| 질문 (계획서 §0 · §3) | 이번 측정이 말하는 것 | 말하지 못하는 것 |
|---|---|---|
| P1 외부 인식 오류가 접촉으로 이어지는가 | 같은 filter에서 detector 장애물 → GT 장애물로 바꾸면 SS가 1–7 → 7–12/20로 오른다(LI · LII 모두). 인식이 closed-loop 결과의 큰 몫을 정한다는 정황 | detector 오류 유형(box 누락 · target box 제외 · 거짓 장애물)별 기여. offline 분할 재현(P5)은 입력만 준비 |
| S1 AG3S가 기존 filter보다 safe success가 높은가 | **현재 Panda 구현(최종 v3-e0to, pb7)은 아니다.** LI SS 0 · 2/20, LII 2/20. VLA 대비 작업 성공을 크게 잃는다(VLA만 성공한 쌍 12–14 대 0) | AG3S의 인식(attention)과 제약(TrajOpt) 중 어느 쪽이 원인인지 — F2 · F6 cell(외부 / GT mask를 AG3S 제약에 넣기)이 없다 |
| AG3S 실패 층 | 놓기 단계 미완료(쥔 뒤 `placed`까지 가도 공식 성공 없음)와 task 3 미파지가 주 층. SL6-H 진단과 같다 | 하강 반전의 정확한 기전(진단 기록 필요) |
| 3번째 benchmark | LIBERO-Safety 환경을 새 기계에 세웠다. VLA 단독 공식 위반: L0 · L1 0–1/10, **L2 obstacle_avoidance 6/10** | baseline · AG3S가 L2에서 위반을 줄이는지 — 아직 안 돌림 |

**다음에 가장 정보가 큰 실험 (제안):** (1) AG3S 놓기 단계 수정(SL6-H `-pc` 판정 재개) 후 같은 20 episode 재측정, (2) F6 cell: GT target mask를 AG3S 서버에 주입한 같은 20 episode — AG3S 제약 자체의 상한, (3) LIBERO-Safety L2 obstacle_avoidance에서 AG3S · filter arm, (4) offline 분할 재현(P5, queue 주석 줄).
