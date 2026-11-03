# development main a0b16d0a → feature/panda-benchmark merge 결과 (2026-11-03)

작성: benchmark checkout lead (세션 benchmark-44), 2026-11-03 KST. 사전 비교 · 시험 merge · 판정 과정은 [DEV_P0_MERGE_PREVIEW_20261103_KO.md](DEV_P0_MERGE_PREVIEW_20261103_KO.md).

## 0. 한눈에

| 항목 | 값 |
|---|---|
| merge commit | `93e2ae8d` (`git merge --no-ff`, parent `76568b60` · `a0b16d0a`), push 안 함 |
| merge 직전 | `76568b60` (tag `pre-dev-main-merge-20261103`) |
| 가져온 것 | development main `a0b16d0a` = 이전 main `3ff415d5` 이후 development 개선 전부: `8ab3c7ef` (T67) · `ae842524` (T72–T90, P0 표면 추출 포함) · `63e98cc8` (T91–T93) 와 그 문서 · 이식 지시 |
| 이 merge의 원칙 | 코드만 들어오고 **기존 Panda 판 동작은 그대로**. 새 option은 모두 기본 꺼짐이며, Panda 서버는 이식 전까지 거절한다 |
| 검증 | 이 checkout에서 tests/benchmark 329 passed / 3 skipped, 핵심 묶음 561 passed. trajopt · ag3s 전체는 시험 merge(scratch worktree)에서 merge 전 HEAD와 비교해 새 회귀 없음 |

## 1. 이전 구성과 변경 후 구성

### 1.1 benchmark Panda 서버 판 (실제로 돌리는 구성)

**merge 전과 후가 같다.** `AG3S/adapters/panda/profile.py`의 판 목록과 flag는 바뀌지 않았고, RB-Y1 golden(`_sl5a_golden`)과 Panda 최종 구성 테스트(`test_sl6a_panda_final` · `test_sl6h_panda_placement_contact`)가 그대로 통과한다.

| 판 | 무엇을 쓰나 (요약) | merge 후 |
|---|---|---|
| V11AB, V11AB-pb1 ~ pb7 | merge 전(2026-10-29 이전) 서버 그대로, ROI carve · BLAS 1 thread 끔 | 같음 |
| AG3S-final-panda (1판) | RB-Y1 최종 구성(A8)의 Panda 이식: target 범위 = 공(ball-owned) + 비 target 표면 증거, 명사 attention `noun_target:L8H6`, HSR6 규칙 4개(Panda probe), 손가락 20 mm · 손바닥 10 mm 이웃 접촉 허용, TrajOpt 여러 초기값 16 | 같음 |
| -g1a4 / -v2 / -v3 / -v3-e0to / -v3-e0to-pc | 1판 + G1·A4 / + pb1·pb6·pb7 · action attention / + pb7 전체 · 연속성 꺾임 항 끔 / + E0 TrajOpt / + placement 단계 쥔 물체 접촉 허용 | 같음 |

### 1.2 RB-Y1 최종 구성 (development 정의, 이번 merge로 코드가 들어옴)

| 구성 요소 | 이전 (A8 최종 구성, 2026-10-29) | 변경 후 (P0m, 사용자 판정 2026-11-03, development d2 세션 전달) |
|---|---|---|
| target 범위 (ATS) | 공(ball-owned) + `--nontarget-evidence surface` | **표면 추출** `--target-extent attention-surface` + 겹침 검사 · 직전 영역 유지 · 쥐는 중 고정 · 공 끔 · GPU (+ hull 채우기 · 접촉 여유 10 mm는 development T92b 결과 대기) |
| attention head | `noun_target:L10H7` (RB-Y1) | 같음 (Panda는 `noun_target:L8H6`, 사용자 판정) |
| 단계 · 쥠 (HSR) | HSR6 규칙 4개 + 3단계 학습 분류기 gate(`kv_L4_v1`) | HSR6 끔, **3단계 gate label을 grasp latch에서**(`--subtask-from-latch`, T92) |
| TrajOpt (CCR) | GPU 여러 초기값 16 | E0 (여러 초기값 끔) |
| G1 · A4 | 끔 | 끔 (사용자가 켜라 할 때까지) |
| 이웃 접촉 허용 | 손가락 20 mm · 손바닥 10 mm | RB-Y1 데모는 같음. **benchmark profile은 둘 다 끄고 target(쥘 때) · 목적지(놓을 때) 접촉만 허용**(D1, d2 세션 전달 — 이 checkout 사용자 확인 전) |

### 1.3 Panda에서 쓸 수 있는 option (merge가 가져온 것)

| 구분 | option | 상태 |
|---|---|---|
| 이식 예정 (허용 판정) | 표면 추출과 그 검사(`--target-extent attention-surface`, `--surface-target-check`, `--surface-keep-last`, `--surface-grasp-hold`, `--surface-no-ball`), GPU(`--attention-surface-device`), 단계 출처 latch(`--subtask-from-latch`) | Panda 서버 거절 목록에 있음 → Panda 연결 · 테스트 · 오프라인 재생 뒤 해제 |
| 조건부 (Panda 값 재설정 + 측정 후) | hull 채우기, 충돌 없으면 통과(`--passthrough-when-clear`), target 유지 기억, 손목 depth 단계(`--subtask-from-phase-machine`), VLA flow 유도(`--fm-guide*`, EE delta → IK 사상 필요) | 거절 |
| 거절 유지 | 5단계 분류기(`--hsr5`), hidden probe(`--attention-source hidden_probe`, RB-Y1 학습), 정답 단계(`--hsr-gt-stage`), 손가락 통로 · grasp 후보(RB-Y1 그리퍼), 표면 분할 · 꺾임 끊기, 접촉 여유(`--surface-contact-allowance-mm`), 지지면 보정, ④ 창 넓히기 | 거절 |

거절 목록: `AG3S/adapters/panda/server.py` `UNPORTED_FINAL_CONFIG_DESTS` (merge가 더한 41개) + hidden probe source 검사. 테스트: `tests/benchmark/test_merge_dev_main_20261103_panda.py`.

## 2. 충돌과 해결

| 파일 | 충돌 | 해결 |
|---|---:|---|
| `AG3S/trajopt/safe_policy.py` | 13 | 양쪽 추가(benchmark 로봇 주입 · SL6-A A9 scope · SL6-H / development T72 · T82 · T83 · T92 · T91c · T93)를 모두 남김. **의미 충돌 1개:** `_hsr6_objective_ok`를 `(q_now, manipulators, *, term)`으로 합침(development 테스트가 위치 인자로 `(q_now, manipulators)`를 넘김). class 기본값 `_hands = None` 추가(T91c test double용) |
| `AG3S/trajopt/serve_safe.py` | 5 | 양쪽 추가 유지. `--hsr6-objective-window` + `--hsr6-objective-scope nbc`를 함께 주면 시작 시 거절(사용자 판정) |
| `AG3S/trajopt/grasp_latch.py` | 3 | V11AB-pb6 sticky reach · drop 기록 + T82 closing release, 둘 다 |
| `AG3S/trajopt/hsr6.py` | 2 | SL6-H ⑤ placement + T67 ② objective window, 필드 순서 = 기존 4개 → placement_contact → objective_window |
| `AG3S/trajopt/linearize.py` · `sqp.py` | 2 · 2 | SL6-H placement 기록 + T67 ① 지지면 보정 기록, 둘 다 |

테스트 쪽 수정: `tests/trajopt/test_sl6a_injections.py`(`term=` 호출 4줄), `tests/trajopt/data/ag3s_final_merge_surface.json`(RB-Y1 golden이 무시하는 "merge가 추가한 기본값" 목록에 새 namespace key 41개 — 기본값이 아닌 값이 되면 다시 실패한다).

## 3. 검증

| 묶음 | 위치 | 결과 |
|---|---|---|
| tests/benchmark 전체 | 이 checkout | 329 passed / 3 skipped |
| 핵심 묶음: Panda 최종 구성 · SL5a · SL5j · SL4 · SL6-A · SL6-H · HSR6 · T67 · T72 · T82 · T91 · T91b · T91c · T92 · T93 · merge 거절 | 이 checkout | 561 passed / 4 skipped (RB-Y1 golden 목록 갱신 전 1 failed → 갱신 후 통과) |
| benchmark · ag3s 실패 파일 20개 재실행 | 시험 merge vs merge 전 HEAD | 새 실패 1개 `test_t86_patch_probe`(CPU venv에 jax 없음 = 환경) |
| tests/trajopt 122 파일(파일별) | 시험 merge | 실패 4개 = legacy fixture 없음 3 + `test_t43t_replay`(merge 전 HEAD에서도 같은 실패). `test_t46f_a14_wrist_cover`는 OOM 뒤 단독 재실행 30 passed |
| tests/ag3s | 시험 merge | 4 failed / 1985 passed (위 20개 파일 비교에 포함) |

## 4. 다음

1. P0 표면 추출의 Panda 이식(head `noun_target:L8H6`, `seed_map_source_ok` 완화) → SafeLIBERO 20 episode 오프라인 재생으로 target 지움 · 장애물 잘못 지움을 먼저 잰다.
2. 단계 출처 latch(T92)의 Panda 이식 · 시험(benchmark-84 세션과 분담 협의 중).
3. D1(이웃 접촉 허용 끄기)은 새 Panda P0m 판에서 반영 — 이 checkout 사용자 확인 후. LIBERO pilot에서 끈 비용 먼저.
