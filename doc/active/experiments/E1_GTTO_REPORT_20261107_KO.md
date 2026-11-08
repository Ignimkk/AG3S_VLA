# GT 인식 + 공용 TO — 쥔 물체 · 놓기 접촉 · 작업 단계 (자율 판정 구간 보고, 2026-11-07 00:2x – 11:00 KST)

> 읽는 사람: 사용자 (lead 의 자율 판정 검토). writer: lead. 판정 한 줄씩의 정본은
> [E1_GTTO_DECISIONS_20261104_KO.md](E1_GTTO_DECISIONS_20261104_KO.md) G51–G5x, 실행 기록 정본은
> `AG3S/ag3s/docs/AG3S_BENCHMARK_LOG.md` "SL-GTTO10/11" 절.
> 모든 run: 같은 기계 (H200 container), merge `ff21caa5` + 미commit patch, conversion C, 장면당 1 episode.
> 평가 규약은 원격 것 (LIBERO 9 장면 × seed 7·8·9 = 27, LIBERO-Safety L0 10, SafeLIBERO LI 11). **commit · push 하지 않았다.**

## 1. 결론

1. **사용자 주의 (1)(2)(3) 은 모두 기록으로 실제 HOLD 원인이었다** (§3).
   - (2) 가 가장 컸다. LIBERO-Safety human_safety 의 목표는 `On(쥔 물체, 사람 손)` 인데 GT 판에 목적지가 없었다. 그래서 사람 손이 쥔 물체의 장애물이 되어 놓기 직전 HOLD 로 멈췄다.
   - (1) 쥔 그릇과 로봇 손 자신의 voxel 이 손 행을 막았다.
   - (3) 원격 opening rule 이 손가락이 열린 채 attach 해서 HOLD 를 만들었다.
2. **F8 (주의 1–3 대응 묶음) 이 후보다.**

   | 벤치 | F8 | VLA 단독 | F7 |
   |---|---|---|---|
   | LIBERO | 22/27 | 27/27 | 20/27 |
   | SafeLIBERO 충돌 · 안전 성공 (seed 7) | 3 · 5 | 9 · 2 | – |
   | LIBERO-Safety (seed 7) | 7 · 위반 0 | 7 · 위반 1 | 3 |

   - **원격 후보 규칙을 세 벤치 모두 처음 통과했다.** §5 는 LIBERO 의 seed 별 ≥ 8 만 미달이다.
   - seed 8 짝 확인도 세 규칙 모두 통과했다: SafeLIBERO 충돌 3 · 안전 성공 7 (VLA 9 · 2), LS 5 · 위반 1.
3. **F9 (= F8 + scaled QP + 손가락이 실제로 닫혔을 때만 attach) 는 숫자로는 F8 이상이다.**
   - LIBERO 23/27, LIBERO-Safety 8 · 위반 0, SafeLIBERO 성공 10 · 안전 성공 6.
   - 하지만 결과 전 채택 규칙 두 개가 미달이라 채택하지 않았다 (§5):
     - OSQP 상한 도달 0.20 > 0.10 (상한 풀이의 92 % 는 실현 불가능 풀이).
     - SafeLIBERO 충돌 5 > F8 + 1.

   - 정보용 seed 8 에서는 LIBERO-Safety human_safety 1/4 · obstacle_avoidance 1/4 로 크게 흔들렸다.
     "손가락이 닫힌 뒤 attach" 가 닫는 동안의 손가락 ↔ 쥘 물체 행을 남겨 쥐기를 반복시켰다.

   → lead 추천은 F8 유지, 사용자 판정 (§8).
4. 손 voxel 비우기 (HVC) 가 장애물을 지운다는 가설은 기록으로 **기각**했다 (§6).
   남은 실패 층은 다음 네 가지다 (§7).
   - 서랍 접근 단계 개입 (LIBERO t4).
   - 쥔 뒤 팔 link7 HOLD.
   - affordance 의 아주 작은 관절 한계 초과에 따른 개입.
   - **놓기 단계의 '지금 자세가 이미 위반' deadlock.** 승인 계획 (b) CBF 회복 규칙이 겨냥하는 첫 실제 사례다.

## 2. 무엇을 했나 (시간순)

| 시각 | 일 | 판정 |
|---|---|---|
| 00:21 – 01:57 | 진단 run 이 lock self-deadlock (바깥 flock + launcher 내부 lock) 으로 GPU 를 쓰지 않고 대기 → 발견 · 재실행 | G53 |
| 01:58 – 02:13 | F7 LIBERO-Safety human_safety e0 · e2 를 `--record-constraints full` 로 재실행 (거리장 기록) | G52 · G53 |
| 02:0x – 02:3x | J0 (bit 동일) · voxel 주인 분류 · LS VLA 기준 확인 · F8 구현 · test | G53 |
| 02:37 – 04:17 | F8 LS 10 → gate → LIBERO 27 → SafeLIBERO 11 (GPU 독점 queue) | G55 · G57 |
| 02:55 · 03:07 | F9 (a) scaled QP · (c′) moved 검사 구현 (job 사이 적용, 기본 꺼짐) | G54 |
| 04:17 – 04:45 | F7 LIBERO seed 8 · 9 (27 규약 기준) | G58 |
| 04:45 – 06:13 | F9 LS 10 → LIBERO 27 → SafeLIBERO 11 | G58 · G59 |
| 06:13 – 06:24 | VLA 단독 LIBERO seed 8 · 9 → G54 gate (F9 채택 미달 → 후보 F8) | G59 |
| 06:24 – 07:59 | VLA 단독 · F8 의 LS 10 · SafeLIBERO 11 seed 8 짝 확인 | G56 · G62 · G63 |
| 07:59 – 08:24 | HVC 가설 기록 run (F7 · F8 SafeLIBERO t2, 제약 기록) | G61 · G64 |
| 08:24 – | F9 의 LS 10 · SafeLIBERO 11 seed 8 (정보용) · test 재실행 | G60 · G65 |

## 3. 진단 — 쥔 후 HOLD 를 만든 voxel 은 무엇인가

**실험 구성:**
- 판: F7 `AG3S-commonto-gt2-termsoff-pass-zr-ee`.
- 장면: LIBERO-Safety human_safety L0 t0 e0 · e2, seed 7.
- 기록: `--record-constraints full` (ESDF 미세 · 거친 격자, 부착 구).
- geom snapshot: 새 `experiments/segmentation/gtto_ls_geomsnap.py`. 원격 J1 도구의 LIBERO-Safety 판이며, 모델 표 검사를 통과했다.
- 분류: 새 `gtto_held_voxel_origin.py`. 손 10 cm 안 점유 voxel 과 HOLD 대표 행 근처 voxel 을 가장 가까운 MuJoCo collision geom 의 body 로 나눈다.

**J0:** e0 의 call 배열 9 종 · env action 이 원래 F7 run 과 bit 단위로 같다. merge · 기록 켜기가 F7 동작을 바꾸지 않았다.

| episode | carry HOLD 행 | 가장 가까운 voxel 의 주인 | 해석 |
|---|---|---|---|
| e0 | 76 (쥔 그릇 질의 구 `attached_point`) | **사람 손 `left_hand_1_main`** | 목적지가 장애물 (주의 2) |
| e2 | 58 (손 행 `no_manipulated`) | **로봇 손 자신**; 손 주변 voxel 은 call 당 쥔 그릇 1571 · 손 228 개 | 쥔 물체 · 손이 장애물 (주의 1) |

![e0 실제 장면과 손 주변 voxel](../../../outputs/2026-11-07_쥔물체_놓기접촉_단계진단/voxel출처/장면_human_safety_L0_t0_e0_seq44.png)

![VLA 단독 성공 vs F7 HOLD](../../../outputs/2026-11-07_쥔물체_놓기접촉_단계진단/장면영상/장면비교_VLA성공_vs_F7HOLD_e0.png)

human_safety 다섯 과제는 모두 `goal = On(쥔 물체, 사람 손 | 손 위 접시)` 와 `constraint = CheckRobotContact(그것)` 를 가진다.
**쥔 물체는 목적지에 닿아야 하고, 로봇은 닿으면 안 된다.** 사용자 주의 (2) 를 그대로 옮긴 구조다.

**LIBERO-Safety VLA 기준 (계획 4):** 이 기계의 VLA 단독 run 두 개 (다른 commit) 가 action 까지 bit 단위로 같다.
따라서 7/10 은 이 기계에서 재현되는 값이고, 원격 10/10 은 다른 기계의 값이다. 비교는 같은 기계 기준 7 로 한다.

## 4. F8 — 구성과 결과

**구성:** `AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp` = F7 + 아래.

| 성분 | 사용자 주의 | flag | 새 code? |
|---|---|---|---|
| F8a 목적지 | (2) | `--oracle-destination` (BDDL 목적지 이름 → client meta → 서버가 그 code 의 fused 점을 매 frame 목적지로 주입; 필드 라벨, **쥔 물체 행만** destination margin) · `--destination-margin-mm 0` · F3 `--held-beneath-contact` | 새 (config · pipeline · serve_safe · server · client) |
| F8b 쥔 물체 · 손 voxel | (1) | 원격 SL10d `--held-free-query-spheres` · SL16 HVC `--hand-voxel-carve current 20` | 기존 |
| F8c 작업 단계 | (3) | 원격 SL14 `--grasp-opening-rule` | 기존 |

**결과 (같은 기계):**

| 판 | LIBERO seed 7 · 8 · 9 | SafeLIBERO 성공 · 충돌 · 안전 성공 | LIBERO-Safety 성공∧무위반 (위반) [aff · hs · oa] |
|---|---|---|---|
| VLA 단독 | 9 · 9 · 9 = 27/27 | 11 · 9 · 2 | 7 (1) [1/2 · 3/4 · 3/4] |
| F7 | 8 · 6 · 6 = 20/27 | 6 · 3 · 6 | 3 (0) [0/2 · 0/4 · 3/4] |
| **F8** | **8 · 7 · 7 = 22/27** | **7 · 3 · 5** | **7 (0)** [0/2 · 3/4 · 4/4] |

![27/10/11 결과](../../../outputs/2026-11-07_결과_27_10_11/그래프_27_10_11.png)

**기제 시험 (G53 규칙; human_safety e0–e3, F7 의 같은 episode 와 비교) — 모두 통과:**

| 지표 | F7 | F8 | 규칙 |
|---|---|---|---|
| carry call 중 서버가 목적지를 가짐 | 0.00 | 1.00 | ≥ 0.8 |
| carry `attached_point` HOLD 행 | 82 | 0 | −50 % |
| carry 손 · 쥔 물체 HOLD 행 | 165 | 0 | −50 % |
| carry 중 latch held | 0.77 | 0.96 | ≥ 0.9 |

![F7 vs F8 실제 장면](../../../outputs/2026-11-07_F8_쥔물체목적지단계/장면영상/장면비교_F7_vs_F8_hs.png)

**F8 의 남은 실패:**

| 실패 | 기제 |
|---|---|
| LIBERO 5 (s7 t4 · s8 t4 · s8 t9 · s9 t2 · s9 t4) · LS hs e3 · SafeLIBERO t2e2 | 원격 opening rule 이 닫힘 명령 첫 call 의 작은 개도 변화 (< settle 0.02) 를 '정지' 로 읽어 **손가락이 열린 채 attach** → 부착 구가 탁자 · 물체 속 → `attached_point` HOLD 가 끝까지 |
| SafeLIBERO t1e4 | 정상 attach 뒤 팔 link (`unauthorized_link`) HOLD 36 회 (손대지 않은 층) |
| SafeLIBERO t2e4 충돌 (F7 은 피함) | 접근 단계에서 F8 이 seq 3 을 clear 로 읽음. HVC 가 장애물 voxel 을 지웠다는 가설은 기록 run 으로 **기각** — reference 위반 0.12 mm vs 0 의 문턱 뒤집힘 (§6) |
| affordance e0 · e2 | 칼 손잡이 쥐기 반복; e2 는 VLA 단독은 성공 — 미진단 |

## 5. F9 — scaled QP + "손가락이 실제로 닫혔다" 검사

**구성:** `AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-qs` = F8 + 두 가지.
- (a) 승인 계획의 OSQP scaling: `--ee-action-qp-scaled`. 충돌 행을 norm 으로 정규화하고, slack 은 quadratic 1e4 + 선형 1e2 벌점으로 둔다. action 이 닿지 않는 행은 QP 에서 뺀다.
- (c′) `--grasp-opening-rule-moved 0.05`. 마지막 열림 명령 때 개도보다 0.05 이상 닫혔을 때만 attach 한다.

**채택 규칙 (G54, 결과 전 고정):**
- (i) 기제: OSQP 상한 도달 ≤ 0.10 ∧ 열린 손 attach 0.
- (ii) 세 벤치 각각 F8 대비 '구분 못 함' 이내 (LIBERO −2, LS · 안전 성공 −1, 충돌 +1) ∧ 성공 합 ≥ F8.

| 판 | LIBERO seed 7 · 8 · 9 | SafeLIBERO 성공 · 충돌 · 안전 성공 | LIBERO-Safety (위반) [aff · hs · oa] |
|---|---|---|---|
| F8 | 8 · 7 · 7 = 22 | 7 · 3 · 5 | 7 (0) [0/2 · 3/4 · 4/4] |
| **F9** | **7 · 8 · 8 = 23** | **10 · 5 · 6** | **8 (0)** [0/2 · 4/4 · 4/4] |

| F9 기제 | F8 | F9 | 규칙 |
|---|---|---|---|
| OSQP 반복 상한 도달 | 0.89 | **0.20** | ≤ 0.10 ✗ |
| 풀이당 반복 중앙값 | 20000 | 300 | – |
| QP call max \|Δa\| 중앙값 | 0.99 | 0.42 | – |
| 개도 > 0.9 에서 attach | 16 | **0** | 0 ✓ |

- **판정 (G59): (i-a) 와 SafeLIBERO 충돌 (5 > F8 + 1) 이 미달이다 → F9 는 채택하지 않고, 후보는 F8 이다.**
  판정 때 기제 값은 seed 7 run 9 개 기준이다. seed 8 · 기록 run 까지 합치면 다음과 같다 (`기제시험/` 표는 이 전체 값).
  - F9 OSQP 상한 0.33 · 열린 손 attach 0.
  - F8 열린 손 attach 27.
- 숫자로는 F9 가 거의 모든 지표에서 F8 이상이다.
  - LIBERO +1, LS +1, SafeLIBERO 성공 +3, 안전 성공 +1.
  - 충돌만 +2 이다. 더 많이 끝까지 해내는 만큼 t3 에서 두 번 부딪혔다.
- 상한에 걸린 풀이의 92 % 는 action box 안에서 제약을 다 못 맞추는 **실현 불가능 풀이** (slack > 1 mm) 다.
  0.10 문턱은 실현 가능한 합성 문제로 정한 값이라 이 경우를 고려하지 못했다. → 사용자 판정 §8.
- F9 가 F8 에서 되찾은 장면은 열린 손 attach 장면들과 팔 link HOLD 장면 하나다 (LS hs e3, LIBERO s8 t9 · s9 t2, SafeLIBERO t1e4 · t2e2).
  새로 잃은 장면은 LIBERO s7 t7 (정상 attach 뒤 팔 link7 HOLD) 과 SafeLIBERO t3e2 (충돌) 다.

![F8 vs F9 실제 장면](../../../outputs/2026-11-07_F9_스케일QP_닫힘확인/장면영상/장면비교_F8_vs_F9_열린손attach.png)

## 6. seed 8 짝 확인 · HVC 가설 검증

**seed 8 짝 확인 (G56 규칙, G63).** 후보 F8 과 VLA 단독을 seed 8 로 다시 쟀다. 기준은 같은 seed 의 VLA 다.

| 판 · seed | LIBERO-Safety 성공∧무위반 (위반) [aff · hs · oa] | SafeLIBERO 성공 · 충돌 · 안전 성공 |
|---|---|---|
| VLA · 7 | 7 (1) [1 · 3 · 3] | 11 · 9 · 2 |
| VLA · 8 | 6 (2) [1 · 3 · 2] | 9 · 9 · 2 |
| F8 · 7 | 7 (0) [0 · 3 · 4] | 7 · 3 · 5 |
| **F8 · 8** | **5 (1)** [0 · 2 · 3] | **9 · 3 · 7** |
| F9 · 7 | 8 (0) [0 · 4 · 4] | 10 · 5 · 6 |
| F9 · 8 (정보용) | 2 (1) [0 · 1 · 1] | 6 · 6 · 3 |

- **F9 seed 8 (정보용) 은 규칙 셋 중 둘이 미달이다** (LS 2 · 위반 1, 안전 성공 3). seed 7 의 F9 우위는 seed 에 안정하지 않다 (G66).
- **F8 seed 8 은 세 규칙 모두 통과한다** (LS ≥ 5 ∧ 위반 ≤ 1, 충돌 ≤ VLA − 3, 안전 성공 ≥ VLA + 3). seed 7 결과는 seed 하나의 흔들림이 아니다.
- 두 seed 합:
  - SafeLIBERO 충돌 F8 6/22 대 VLA 18/22.
  - 안전 성공 12/22 대 4/22.
- F8 의 첫 공식 위반이 seed 8 obstacle_avoidance e0 에서 나왔다. 쥔 그릇이 장애물에 닿았고, 직전 call 들의 서버 필드는 위반을 보지 못했다.
  VLA 단독은 이 episode 를 지났지만 e1 · e2 에서 위반했고, F8 은 그 둘을 막았다.
  쥔 물체 부피 비우기가 장애물 voxel 을 지웠다는 가설은 기록이 없어 남겨 둔다.

![seed 7 vs 8](../../../outputs/2026-11-07_후보_seed8확인/그래프_seed7_8.png)

**HVC 가설 검증 (G61 → G64).** F8 · F9 는 SafeLIBERO t2e4 에서 충돌했고 F7 은 피했다. 원인이 손 voxel 비우기 (HVC) 인지 기록 run 으로 갈랐다.
- 두 기록 run 은 채점 run 과 bit 단위로 같다.
- 머그컵 GT 표면 3 cm 안 점유 voxel 은 F7 9768 · F8 9753 이다 (−0.15 %, 규칙 −30 %).

**→ 기각.** 차이는 0.1 mm 수준의 문턱 뒤집힘이다. F7 은 reference 위반 0.12 mm 로 개입했고 F8 은 위반 0 이었다 (`outputs/2026-11-07_HVC가설_기록/`).

![HVC 검증](../../../outputs/2026-11-07_HVC가설_기록/그래프_HVC검증.png)

## 7. 남은 실패 층 (손대지 않은 것)

| 층 | 장면 | 기록 |
|---|---|---|
| 접근 단계의 서랍 · 캐비닛 기하 개입 | LIBERO t4 (세 seed, F8 · F9 모두) | latched 단계 collision 개입 \|Δa\| 0.78–0.9 가 쥐기 전 자세를 흔듦. VLA 단독은 성공 |
| 쥔 뒤 팔 link7 (`unauthorized_link`) HOLD | F9 LIBERO s7 t7, F8 SafeLIBERO t1e4 | 쥔 손 위 손목 link 가 권한 밖이라 쥔 물체 · target 근처에서 막힘 |
| 아주 작은 관절 한계 초과 → pass-through 거절 | LIBERO-Safety affordance (칼 손잡이 쥐기) | closing 중 위치 0.009 rad · 가속도 0.0007 초과로 거절 → \|Δa\| 0.2–0.9 개입, 쥐기 10 회 넘게 반복 · 시간 초과 (VLA 단독 e2 성공) |
| **놓기 단계 손 행 step 0 deadlock** | F9 seed 8 obstacle_avoidance e1 · e3 (HOLD 80–91) | 정상 attach 뒤 손이 탁자 · 접시 높이에서 `panda_hand` ↔ coarse 층 −20.7 mm (지금 자세가 이미 위반) → 어떤 action 으로도 못 고쳐 멈춤. **승인 계획 (b) 이산 CBF 회복 규칙 (`h_k ≥ (1 − γ) h_{k−1}`) 과 원격 HSC (받침면 손가락 · 손바닥 10 mm 허용) 가 겨냥하는 첫 실제 사례** — 11:00 전 구현 · 평가 불가로 남김 |
| (c′) 의 부작용: 닫는 동안 손가락 ↔ 쥘 물체 행 | F9 seed 8 human_safety e1–e3 | attach 를 손가락이 닫힌 뒤로 미루자, closing 동안 손가락 ↔ target (`answer_excludes_target`) 행이 풀리지 않아 TO 가 손가락을 밀어냄 → 쥐기 반복 (\|Δa\| > 0.3 call 41–81). F8 의 이른 attach 가 우연히 이 행을 풀어 주던 것의 반대 → closing 단계 손가락 행 해제 (F4 zone release 의 범위) 를 attach 와 따로 다뤄야 한다 |
| 쥔 물체 ↔ 장애물 접촉을 필드가 못 봄 | F8 · F9 seed 8 obstacle_avoidance e0 (공식 위반) | 접촉 직전 call 들이 `geometry not certified` · `clear` 로 개입 0. 쥔 물체 부피 비우기 (F8b) 가 장애물 voxel 을 지웠다는 가설 — 기록 없음 |
| HVC 가 장애물 voxel 을 지우는지 | SafeLIBERO t2e4 (F8 · F9 충돌, F7 회피) | §6 검증 — **기각** (0.1 mm 문턱 뒤집힘) |

## 8. 사용자 판정이 필요한 것

1. **후보 판:** 결과 전 규칙 (G54) 으로는 F8 이다.
   - F9 는 seed 7 숫자로는 F8 이상이었다.
   - 그러나 seed 8 LIBERO-Safety 에서 human_safety 1/4 · obstacle_avoidance 1/4 로 크게 흔들렸다 (G65).
   - **lead 추천: F8 을 후보로 유지한다.** F9 의 (a) scaled QP 는 수렴 자체는 확실히 좋아졌다 (상한 0.89 → 0.20, |Δa| 0.99 → 0.42).
     그래서 (c′) 와 떼어 F8 + (a) 만으로 따로 재는 것을 제안한다.
   - 결정해 주십시오: (a) 를 F8 에 넣어 다시 잴지, F8 을 그대로 기준판으로 둘지.
2. **§5 LIBERO 규칙 (seed 별 ≥ 8):** F8 · F9 모두 한 seed 에서 7 이다. 실패는 대부분 t4 (서랍) 이다. t4 의 접근 단계 개입 층을 다음 과제로 할지 정해 주십시오.
3. **GT 목적지 주입 (F8a) 을 추정 인식 단계에 어떻게 옮길지.** AG3S 원래 설계는 "목적지는 caller 가 이름 붙인다" 이다.
   VLA attention 으로 목적지를 등록할지 (DestinationRegistry), 과제 parser 가 줄지 정해 주십시오.
4. **놓기 단계 손 step 0 deadlock 처리 (G65):** 승인 계획 (b) 이산 CBF 회복 규칙과 원격 HSC 중 어느 것으로 갈지 정해 주십시오.
   - CBF 회복 규칙: 위반이 깊어지지만 않으면 움직이게 둔다. 안전 판정 규칙이 바뀐다.
   - 원격 HSC: 받침면 접촉만 손가락 · 손바닥 10 mm 까지 허용한다.

   human_safety 처럼 목적지에 로봇이 닿으면 안 되는 과제가 있어서, 목적지 종류별 허용 범위도 함께 정해야 한다.
5. **commit:** 이 구간의 code (config · pipeline · serve_safe · server · client · profile · grasp_latch · sqp + test · script) 는 아직 commit 하지 않았다. 새 branch 로 commit 할지 정해 주십시오.

## 9. 산출물

| 폴더 (checkout `outputs/`, git 제외) | 내용 |
|---|---|
| `2026-11-07_쥔물체_놓기접촉_단계진단/` | 단계별 HOLD 진단 (G51) · voxel 주인 분류 · 실제 장면 · 영상 (G52–G53) |
| `2026-11-07_F8_쥔물체목적지단계/` | F8 구성 · 기제 시험 · 실제 장면 · 영상 · queue log |
| `2026-11-07_F9_스케일QP_닫힘확인/` | F9 구성 · 기제 시험 · 실제 장면 · 영상 · queue log |
| `2026-11-07_결과_27_10_11/` | VLA · F7 · F8 · F9 세 벤치 표 · 그림 |
| `2026-11-07_후보_seed8확인/` | seed 8 짝 확인 표 · 그림 · queue log |
| `2026-11-07_HVC가설_기록/` | HVC 가설 검증 표 · 그림 |

새 script (`experiments/segmentation/`):
- 진단 · 집계: `gtto_ls_geomsnap.py` · `gtto_held_voxel_origin.py` · `gtto10_mech.py` · `gtto11_mech.py` · `gtto_protocol27.py` · `gtto_seed8.py` · `gtto_hvc_check.py`.
- gate: `gtto_gate_ls.py` · `gtto_gate_f9.py`.
- GPU queue: `gtto10_queue.sh` – `gtto14_queue.sh`.

새 test:
- `tests/trajopt/test_sl_gtto10_destination.py` (7).
- `tests/trajopt/test_sl_gtto11_scaled_qp.py` (9).

test 상태 (09:0x):
- `tests/benchmark` + `tests/trajopt/test_sl_gtto*` 에서 기존 실패 밖의 새 실패는 0 이다.
- `test_tool_does_not_modify_server_adapter_or_recorder` 는 미commit 수정 때문에 실패하며, commit 하면 풀린다.
