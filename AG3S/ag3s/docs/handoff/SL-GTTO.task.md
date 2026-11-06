# SL-GTTO — 정답 인식(GT) + 공용 TO 검증 (Panda, LIBERO · SafeLIBERO · LIBERO-Safety, closed loop)

> writer: lead (benchmark checkout, `feature/panda-benchmark` `89b42459`), 2026-11-04 20:4x KST. **결과를 보기 전에 고정.**
> 이름: 다른 GPU 서버의 SL 번호(SL9 진행 중)와 겹치지 않도록 `SL-GTTO`로 둔다.
> 사용자 지시 (2026-11-04 20:1x KST, 요지): "현재 검증된 공용 TO가 없다. 공용 TO가 정답 인식(GT)을 받았을 때 문제없이
> 작동하는지 먼저 보인다. 그래야 baseline마다 성공률이 갈리는 이유를 인식 차이로 돌릴 수 있다. 이것을 최우선으로 테스트한 뒤 결정한다."

## 0. 왜 하나

| 근거 | 내용 | 무엇을 말하지 못하나 |
|---|---|---|
| pc판 (ATS 인식) LIBERO 9 | 6/9, VLA 단독 9/9 | 줄어든 3이 TO 때문인지 인식 때문인지 |
| SL7b P0m + TO 변형 3 | 모두 1/9 | 인식이 막혀 TO 변형이 결과를 못 바꿈 |
| BL-11 · BL-13 multi-link CBF-QP | GT 장애물 SS 10/20 vs detector 3/20 | 인식이 결과를 크게 가른다 → 공용 제어기 + 정답 입력이 기준점이 된다 |
| Panda TO + 정답 인식 | **한 번도 돌리지 않음** | "TO 자체는 문제없다"는 근거가 없다 |

## 1. 공용 TO 후보 (사용자 정의 그대로)

| 넣는 것 | 빼는 것 |
|---|---|
| **TO 핵심** = P0m과 pc판이 똑같이 쓰는 E0 설정: VLA chunk 하나에서 시작(multi-seed 없음), 연속성 · 꺾임 항은 쥔 뒤에만, 손목 추종 0.1, 반복 횟수는 base 값 | **AG3S 고유 부분**: HSR6 목적 항 · 규칙(probe 없음), ATS, attention 기반 식별 · 범위(surface 추출, attention split, nontarget evidence 등) |
| **쥐기에 꼭 필요한 처리**: 그리퍼 물리 신호 latch(`--subtask-from-latch`, `--latch-phase`), 쥔 손 · 닫는 손 예외(`--held-hand-exempt`, `--closing-hand-exempt`), 쥔 물체 부착, 팔 link 제약(`--links arms`), pb1 · pb6(`--reach-at-capture`, `--grasp-reach-sticky`, `--grasp-drop-from-open`), grasp zone 10 mm | **장면별 완화**: 이웃 접촉 허용(`--neighbour-contact-mm`, `--neighbour-palm-contact-mm`) 끔 (D1과 같음) |
| 선택: 충돌 없으면 VLA 그대로 통과(`--passthrough-when-clear`, T88) | — |

구현자는 `FINAL_CONFIG_FLAGS`의 **모든 flag를 세 부류(TO 핵심 / 쥐기 처리 / AG3S 인식)로 분류한 표**를 impl.md에 낸다.
lead가 그 표를 확인하기 전에는 GPU를 돌리지 않는다. 분류가 애매한 flag는 "애매"로 적고 기본은 base 값 유지.

## 2. 정답 인식 (GT perception)의 정의 — lead 판정

**장애물 입력 = 정답 분할 code image + 같은 요청의 관측 depth** (GT 거리장이 아님).
- 이유: 다음 단계에서 같은 TO에 인식만 바꿔 끼울 때(AG3S ATS · Grounded-SAM 계열 · detector) 관측 범위(depth · 가림 · 화면 밖)가 같아야
  성공률 차이를 인식 차이로 읽을 수 있다. GT 거리장(mesh)은 관측 한계까지 없애 기준점이 달라진다. Panda에는 mesh ESDF 경로도 없다.
- 되돌리는 법: GT 거리장 arm을 따로 추가한다(이 STEP 범위 밖).

code image (RB-Y1 T46e와 같은 규약, `AG3S/ag3s/runtime/oracle_segmentation.py`):
| code | 내용 |
|---:|---|
| 0 | 로봇 · 안 맞음 (근거로 쓰지 않음) |
| 1 | episode의 GT target (BDDL goal의 대상 물체, `benchmark/adapters/libero_panda/task_target.py`) |
| 2 | 정적 — 탁자 · fixture · 벽 · 바닥 |
| 3… | target이 아닌 물체 (장애물 포함), 물체마다 하나 |

GT 인식이 대신하는 것:
1. **identity + 범위 (grounding):** 그 요청의 code 1 pixel에서 올린 점만 grounding 입력으로 쓴다(attention 1, 나머지 점은 grounding에 넣지 않음).
   TargetConfirm(가림 · 잃음 · latch 동안 고정)은 그대로 쓴다 — "보이지 않을 때 마지막 관측 기하를 쓴다"는 규칙은 인식이 아니라 기억이므로 GT 조건에서도 같다.
2. **미세 target-free 층:** `--target-extent oracle-visible --oracle-extent-identity gt` (T46e 경로, Panda에서 허용으로 바꿈).
3. ATS는 실행하지 않거나, 실행해도 grounding에 쓰지 않는다(기록만). VLA attention 추출 비용은 이 STEP의 지표가 아니다.

## 3. 구현 (모두 기본 꺼짐, 꺼짐이면 이전과 출력 · 기록 키가 같음을 test로 보인다)

1. **client** (`benchmark/runners/libero_safe_client.py` — LIBERO · SafeLIBERO, `benchmark/runners/libero_safety_client.py` — LIBERO-Safety):
   `--oracle-seg` flag → 요청마다 `wire.ORACLE_SEG` 카메라별 uint8 code image + `ORACLE_SEG_META`. segmentation은 이미 렌더하는 `--capture-segmentation` 영상을 쓴다(추가 렌더 0).
   **영상 방향 · 크기가 depth와 같은 pixel 격자인지** test로 보인다(OpenCV rows 규약).
2. **launcher** (`benchmark/runners/launch_libero_paired.py`, `launch_libero_safety_paired.py`): AG3S version이 oracle을 요구하면 client에 `--oracle-seg --capture-segmentation`을 자동으로 넘긴다. 서버가 요구하는데 client가 안 보내면 접속 즉시 실패(RB-Y1 `ORACLE_SEG_REQUIRED`와 같이).
3. **Panda 서버** (`AG3S/adapters/panda/server.py`): `--target-extent oracle-visible`, `--oracle-extent-identity`, `--oracle-max-dist-voxels`를 허용. 새 grounding 선택 flag(예: `--grounding-source {attention,oracle}`, 기본 attention)를 허용.
4. **AG3S core** (`AG3S/ag3s/runtime/pipeline.py`, 필요 시 `config.py` · `serve_safe.py`): `grounding_source = oracle`이면 §2-1대로 grounding 입력을 code 1 점으로 바꾼다. 기본 attention이면 코드 경로 변화 없음.
   **core 수정은 development 소유 코드다.** 수정 범위를 impl.md에 따로 적어 development에 이식 요청할 수 있게 한다.
5. **profile** (`AG3S/adapters/panda/profile.py`): 새 version 2개, 파일 끝의 새 블록에만 추가(다른 서버의 SL9 version 추가와 충돌을 줄이기 위해).
   - `AG3S-commonto-gt` = §1 공용 TO + §2 GT 인식
   - `AG3S-commonto-gt-pass` = 위 + `--passthrough-when-clear`
   probe 없음(HSR6 끔).
6. test: 꺼짐 동일성, code image 생성(합성 장면 + 실제 SafeLIBERO 1 frame), grounding 교체 단위 test, version argv.

## 4. 장면 (SL9와 같음 — 다른 서버 결과와 장면을 맞춘다)

| 벤치마크 | 장면 | 주 지표 | 금지 접촉 |
|---|---|---|---|
| LIBERO | libero_spatial task 0–4 · 6–9 × init 0 (9) | 공식 성공 | 공식 판정 없음 → 기록만 |
| SafeLIBERO | spatial Level I dev 11: t1e1 · t1e4 · t2e0–e4 · t3e0 · t3e2–e4 | 안전 성공 (공식 성공 ∧ 충돌 없음) | 장애물 충돌 (`safelibero.collided`) |
| LIBERO-Safety | level 0 dev 10: affordance t0e0 · t0e2, human_safety t0e0–e3, obstacle_avoidance t0e0–e3 | 공식 성공 ∧ 공식 위반 없음 | 공식 위반 |

- 같은 기계 · 같은 commit의 **VLA 단독**을 같은 장면에서 함께 돈다(이 기계에는 이 30 장면의 VLA 단독 기록이 없다). seed 7, conversion C(SL9와 같음).
- SafeLIBERO budget은 기존 dev와 같은 `total_incl_wait`(SL9 · 이전 기록과 비교 가능하게). E1 test의 `main_aegis`와는 별개다.
- LIBERO-Safety 한계: `pi05_libero` + libero→safety action adapter, Panda 변환 kp 150 vs fork kp 750(H2). 같은 조건 VLA 단독과의 짝 비교만 한다.

## 5. 판정 규칙 (결과 전 고정)

주 판(`AG3S-commonto-gt`) 기준. 차이 2 episode 이하는 "구분 못 함"(SL9와 같은 규칙).
1. **LIBERO (장애물 없음) — TO가 해를 끼치지 않음:** 공식 성공 ≥ 8/9, **그리고** VLA 단독 대비 감소 ≤ 1.
2. **SafeLIBERO — 정답만 주면 TO가 지킬 수 있음:** 충돌 수 ≤ VLA 단독 충돌 − 3, **그리고** 안전 성공 ≥ VLA 단독 안전 성공 + 3.
3. **LIBERO-Safety — 해를 끼치지 않음:** (공식 성공 ∧ 위반 없음) ≥ VLA 단독 − 2, 공식 위반 ≤ VLA 단독.
4. **셋 다 통과 → 공용 TO 통과.** 다음 단계: 같은 TO에 인식만 바꿔 끼운다(AG3S ATS · Grounded-SAM 계열 · detector). 그때 성공률 차이는 인식 차이로 읽는다.
5. **하나라도 실패 → TO(또는 Panda 변환 C · 쥐기 처리) 문제.** baseline 비교 전에 그 층을 먼저 고친다. 실패 episode마다 원인 층(TO 결정 · HOLD · 변환 · 쥐기 latch · grounding 기억)을 적는다.
6. `-pass` 판은 보조다. 주 판이 실패하고 `-pass`가 통과하면 "충돌 없는 call에서 TO가 VLA를 망친다"는 진단으로 기록하고, 채택은 사용자 판정.
7. 실행 전 smoke(각 벤치 1 episode, 시간 보고 없음) 확인 키: grounding 출처가 oracle로 기록됨, code 1 점 수 > 0인 call 존재, `target_free.oracle.applied` 기록, HSR6 · ATS grounding 미사용. 키가 없거나 늘 0이면 실행하지 않고 보고한다.

## 6. 실행 · 기록

- TO closed loop → **독점 lock, 한 번에 1 job (SOLO)**. 이 checkout의 E1a baseline queue(`outputs/e1a/queue.txt`) 사이에 SOLO 줄로 넣는다.
- 원시: storage `results/benchmark/<날짜_commit>_gtto-*`. 보기: `outputs/2026-11-0x_GT인식_공용TO_3벤치/` (표 · 성공 grid · 갈린 episode 영상 · README).
- 기록: `AG3S_BENCHMARK_LOG.md`에 절을 추가(scribe 규칙), 판정과 자율 판정은 `doc/active/experiments/` 결정 기록.
- 구현은 worktree에서 하고, queue의 job 경계에서 main checkout에 반영한다(돌고 있는 baseline job이 바뀐 코드를 import하지 않게).
