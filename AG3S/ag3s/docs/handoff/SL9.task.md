# SL9 — development 수정안 A · B ablation (Panda, LIBERO · SafeLIBERO · LIBERO-Safety, closed loop)

> writer: lead (benchmark checkout), 2026-11-04 18:xx KST. **결과를 보기 전에 고정.**
> 지시: development 이식 문서 `BENCHMARK_T96_T96E_PALM_TRANSFER_20261104_KO.md`(development main `4ecf8d61`)의 사용자 지시
> "3 가지를 ablation 으로 세 벤치마크에서 시험, 결과로 P0m 유지 / 수정안 반영을 고른다".
> benchmark 사용자 판정 (2026-11-04 17:3x–17:5x KST):
> 1. 기준 판 둘 다 — P0m, 그리고 Panda 후보 쪽.
> 2. LIBERO-Safety 포함. 한계는 표시한다.
> 3. SL8b는 별도로 본다.
> 4. fb 기준에서는 A가 거절되므로 "둘 다": fb 위 B, 그리고 SL7e `-pc-surface` 위 A×B.
> 5. ATS 제안은 worktree에서 병행한다.
>
> 채택은 아무것도 하지 않는다. 최종 선택은 사용자다.

## 1. 수정안

| 기호 | 내용 | 서버 인자 |
|---|---|---|
| A | target을 닫힌 물체로 지우기 (T96 · T96c · T96d) | `− --surface-hull-fill − --surface-contact-allowance-mm 10` `+ --surface-solid-owned --surface-solid-protect-mm 5 --surface-solid-dilate-mm 0 --surface-solid-dn-ever-target --surface-solid-shadow-mm 80` |
| B | 지운 seed 부호 수정 (T96e) | `+ --surface-removed-sign-free` |
| C | 실제 크기 손바닥 | **Panda에서 해당 없음 (측정, §4)** |

T97(`--surface-crease-protrusion*`, `--surface-crease-hand-freeze-mm`)은 꺼 둔다. Panda 서버에서 미이식(거절)으로 분류한다.

## 2. 판 (10개, 모두 같은 commit · 같은 장면 · seed 7 · conversion C)

| 기준 | 판 이름 | A | B |
|---|---|:-:|:-:|
| P0m | `AG3S-final-panda-p0m` | | |
| | `AG3S-final-panda-p0m-B` | | ✓ |
| | `AG3S-final-panda-p0m-A` | ✓ | |
| | `AG3S-final-panda-p0m-AB` | ✓ | ✓ |
| pc판 + 표면, 기권 시 공 (SL7g) | `AG3S-final-panda-v3-e0to-pc-surface-fb` | | |
| | `AG3S-final-panda-v3-e0to-pc-surface-fb-B` | | ✓ |
| pc판 + 표면, keep-last · no-ball (SL7e) | `AG3S-final-panda-v3-e0to-pc-surface` | | |
| | `AG3S-final-panda-v3-e0to-pc-surface-B` | | ✓ |
| | `AG3S-final-panda-v3-e0to-pc-surface-A` | ✓ | |
| | `AG3S-final-panda-v3-e0to-pc-surface-AB` | ✓ | ✓ |

- 기준 판도 이번 commit에서 다시 돈다. 앞 결과(SL7 · SL7e · SL7g · SL7h)는 참고로만 쓴다.
- D1(이웃 허용 끔)은 그대로다.

## 3. 장면 · 지표

| 벤치마크 | 장면 | 주 지표 | 금지 접촉 |
|---|---|---|---|
| LIBERO | libero_spatial task 0–4 · 6–9 × init 0 (9, VLA 단독 9/9) | 공식 성공 | (LIBERO에 공식 접촉 판정 없음 → 기록만: 실행 행동 포화 · 큰 변경) |
| SafeLIBERO | spatial Level I dev, VLA 단독 공식 성공 11 (t1e1 · t1e4 · t2e0–e4 · t3e0 · t3e2–e4) | 안전 성공 (공식 성공 ∧ 충돌 없음) | 장애물 충돌 (`safelibero.collided`) |
| LIBERO-Safety | level 0 dev, VLA 단독 공식 성공 10 (affordance t0e0 · t0e2, human_safety t0e0–e3, obstacle_avoidance t0e0–e3) | 공식 성공 ∧ 공식 위반 없음 | 공식 위반 (`libero_safety.official_violation`) |

**LIBERO-Safety 한계 (결과마다 표시):**
- policy는 `pi05_libero`와 libero → safety action adapter다. 전용 `pi05_libero_safety`는 openpi TrainConfig가 없다(H1).
- Panda 변환은 robosuite 1.4.1 OSC(kp 150)를 모델로 하는데, fork는 kp 750이다(H2).
- 그래서 같은 조건의 VLA 단독과 짝 비교만 하고, 논문 수치와는 비교하지 않는다.

## 4. 사전 확인 (실행 전, 결과 숫자 아님)

1. **C 해당 여부:** `panda_hand` collision mesh(`hand.obj` = LIBERO 그리퍼 형상, 40,000 표면 표본) 중 덮지 못한 표본을 셌다.

   | 덮개 | 못 덮은 표본 | 최대 부족 |
   |---|---:|---:|
   | franka 구만 | 34,802 | 15.5 mm |
   | franka 구 + Panda palm shell (여유 0) | 2,457 | 4.98 mm |
   | franka 구 + Panda palm shell (Panda 판이 쓰는 여유 5 mm) | 0 | 0 |

   RB-Y1에서는 palm shell이 27 mm 부족했다. Panda에는 그런 부족이 없으므로 C는 해당 없음이다(이식 문서 §2-2).
2. **flag 분류:**
   - A · B flag 7개는 `PORTED_FINAL_CONFIG_DESTS`로 분류한다.
   - T97 3개와 `palm_cover_margin`은 `UNPORTED_FINAL_CONFIG_DESTS`로 분류한다.
3. **기록 키 smoke:** 새 판마다 1 episode를 돈다(시간 보고 없음).
   - A: `target_exclusion.attention_surface.solid`가 있고, 받침면 투영 점 수(`R⊥`) > 0인 요청이 있어야 한다.
   - B: `oracle_split.removed_sign_free.applied == true`.
   - 손 층 미세 voxel 크기를 기록한다.
   - 키가 없거나 늘 0이면 실행하지 않고 보고한다.

## 5. 판정 규칙 (결과 전 고정)

1. **기준별로 따로 판정한다.** P0m 쪽은 P0m 대비, fb 쪽은 fb 대비, `-pc-surface` 쪽은 `-pc-surface` 대비다.
2. **수정안 판이 기준을 대신할 후보가 되는 조건** (development 제안 그대로):
   - 세 벤치마크 모두 주 지표 ≥ 기준이고 금지 접촉 ≤ 기준이다.
   - 그리고 적어도 한 벤치마크에서 주 지표 차이 > 2 episode다.
   - 아니면 기준을 유지한다.
3. **효과 읽기:** 효과를 가산 요약으로 적는다. 2 episode 이하 차이는 "구분 못 함"이다.
   - B 단독 효과: 1→2 (각 기준)
   - A 단독 효과: 1→3
   - A와 B의 상호작용: 1 · 2 · 3 · 4
4. **SafeLIBERO에서는 target 옆 장애물 접촉을 따로 센다.** B가 접촉 여유로 지운 이웃 seed도 양수로 만들 위험이 있기 때문이다(이식 문서 §1 B 위험).
5. 실패마다 원인 층(ATS · HSR · LTE · CCR)을 적는다.
6. 기준 판 셋 사이의 비교(P0m 대 fb 대 `-pc-surface`)는 이 STEP의 판정이 아니다. 기록만 한다.

## 6. 실행

- `benchmark/runners/run_parallel_gpu.sh`로 동시 2 job을 돌린다(exclusive lock을 한 번 잡음).
  - LIBERO · SafeLIBERO는 `launch_libero_paired.py`를 쓴다.
  - LIBERO-Safety는 같은 스크립트에 launcher를 `launch_libero_safety_paired.py`로 바꿔 돈다. lock 동작은 같다.
- 실행 중에는 이 checkout에 commit하지 않는다. benchmark-a6 세션과 합의했다.
- 시작과 끝을 development 세션에 알린다.
- 원시 기록: storage `results/benchmark/<날짜_commit>_sl9-*`
- 보기: `outputs/2026-11-04_SL9_수정안AB_ablation_3벤치/`
  - 표 · 그래프
  - 판별 성공 grid
  - 대표 영상: 기준 대 수정안이 갈린 episode
  - README
