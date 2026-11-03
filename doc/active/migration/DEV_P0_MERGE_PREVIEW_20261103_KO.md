# development → main → feature/panda-benchmark merge 사전 비교 (2026-11-03 KST)

작성: benchmark-44 세션 (lead). merge 전 비교이며 아무것도 merge·commit하지 않았다. 판정은 사용자에게 요청한다.

## 0. 기준

| 항목 | 값 |
|---|---|
| benchmark | `feature/panda-benchmark` HEAD `96fda38a` (SL6-H). 지난 merge `dd23560f` (main `3ff415d5`, 2026-10-29) 뒤로 commit 12개 |
| development | `feature/ag3s-development` HEAD `8ab3c7ef` (T67, commit 1개) + **commit 안 된 작업 트리**: 수정 28 파일, 새 파일 239개 (T68–T90, P0 표면 추출 포함) |
| 3-way merge base | main `3ff415d5` (= development `5f7a99f2` 와 겹치는 6개 파일 내용이 같음을 확인) |
| 시험 방법 | 겹치는 파일마다 `git merge-file --diff3` (base = `3ff415d5`, ours = benchmark HEAD, theirs = development 작업 트리), scratchpad에서만 실행 |

## 1. 양쪽이 모두 바꾼 파일 (6개, 모두 `AG3S/trajopt/`)

| 파일 | benchmark 쪽 변경 (dd23560f 이후 + 이전 pb 계열) | development 쪽 변경 | 자동 merge | 충돌 성격 |
|---|---|---|---|---|
| `attention_source.py` | SL6-B: LIBERO target 명사 규칙(`LiberoNoun`, `libero_prompt_noun`) | T73·T90: hidden state, `HiddenProbeSource`, `seed_map_source_ok` | **충돌 없음**, 합친 파일 compile 됨 | — |
| `hsr6.py` | SL6-H ⑤ `placement_contact` (PLACEMENT phase, flag 필드) | T67 ② `objective_window` (OBJECTIVE_WINDOW_PHASE) | 충돌 3 | 둘 다 같은 자리에 새 필드·상수 추가 → 둘 다 남기면 됨. 단 `names()` 순서 주석("placement_contact가 마지막 필드")을 정해야 함 |
| `linearize.py` | SL6-H `placement_contact` 여유, `_last_grasp_contact` 기록 | T67 ① `support_surface_offset` | 충돌 2 | 추가 · 추가 → 둘 다 남김 |
| `sqp.py` | SL6-H placement 기록 | T67 ① support 기록, T72 reach box 선택 | 충돌 2 | 추가 · 추가 → 둘 다 남김 |
| `serve_safe.py` | pb1 `reach_at_capture` 경고, SL6-H `--hsr6-placement-contact`, SL6-A A9 `--hsr6-objective-scope` 등 | T81 vacancy, T67 `--hsr6-objective-window`·`--support-surface-offset-mm`, T82 hsr5 기록 등 (+634 줄) | 충돌 4 | 추가 · 추가 → 둘 다 남김 |
| `safe_policy.py` | SL5a·SL6-A 로봇 주입(`hands`, `arm_joint_names`, `neighbour_finger_links`, `hold_site_offset_m`, `palm_links`, `geo_gripper`), A9 scope, SL6-H | T72 grasp set, T82 hsr5, T83 corridor, T67 ② window | 충돌 12 | 10개는 추가 · 추가. **2곳은 의미 충돌**: 아래 1.1 |

### 1.1 의미 충돌: `SafePolicy._hsr6_objective_ok`
- benchmark (SL6-A A9): `_hsr6_objective_ok(term="nbc")` — `--hsr6-objective-scope nbc`이면 A4·G1 목표는 막지 않고 기록만 남긴다.
- development (T67 ②): `_hsr6_objective_ok(q_now=None, manipulators=None)` — `--hsr6-objective-window`이면 approach 단계에서도 A6 창이 열려 있으면 허락한다.
- 호출 3곳(g1 · a4 · nbc)이 양쪽에서 다르게 바뀌었다. 합치려면 `_hsr6_objective_ok(term="nbc", q_now=None, manipulators=None)`처럼 두 인자를 모두 받게 새로 써야 한다. 두 flag 모두 기본 꺼짐이라 꺼진 상태 동작은 그대로지만, **둘을 함께 켰을 때의 규칙(scope와 window 중 어느 것을 먼저 적용하나)은 아무도 정하지 않았다.**

## 2. 겹치지 않지만 benchmark에 영향이 있는 development 변경

| # | 내용 | benchmark 영향 |
|---|---|---|
| 2.1 | **P0 표면 추출(`--target-extent attention-surface` + `--surface-*`)이 head를 `noun_target:L10H7`(RB-Y1)로 고정**한다 (`seed_map_source_ok`: L10H7 또는 T90 hidden probe만 허용). Panda 최종 구성은 `noun_target:L8H6` | Panda에서 P0를 그대로 켜면 서버가 거절한다. Panda에서 쓰려면 이 고정을 풀거나 Panda head를 허용하는 규칙이 필요하다 |
| 2.2 | Panda 서버의 `UNPORTED_FINAL_CONFIG_DESTS`(이식 안 된 flag를 `--robot panda`에서 거절)에는 `target_extent`만 있고, development의 새 flag(`surface_*`, `hsr5`, T81 vacancy, T72 grasp set, T83 corridor, T67 support offset·objective window, T88 passthrough, T90 hidden probe 등)는 **목록에 없다** | 그대로 merge하면 Panda 서버가 이 flag들을 **조용히 받아들여** RB-Y1 기준 코드(RB-Y1 학습 분류기 포함)를 Panda에서 실행할 수 있다 |
| 2.3 | 3단계 subtask gate(`kv_L4_v1`, RB-Y1 학습)가 서버 기본 on (P0 문서 §5) | benchmark Panda 판들이 이미 이 gate를 어떻게 다루는지 merge 뒤 회귀로 확인 필요. 기본값 자체가 바뀐 것은 아니다(같은 base) |
| 2.4 | `third_party/openpi` (gemma.py · pi0.py · pi0_config.py): `return_hidden` 추가, 기본 False | 기본값에서 계산·반환이 같다고 적혀 있다. benchmark VLA 단독 재현성은 merge 뒤 회귀로 확인 |
| 2.5 | `.claude/agents/ag3s-implementer.md` 수정, `.claude/worktrees/` 새 폴더 | agent 정의는 모든 checkout이 공유. `.claude/worktrees/`는 commit에서 빼야 할 것으로 보임 |
| 2.6 | development 작업 트리에 `papers/`(논문 PDF)가 untracked로 있음. benchmark에도 같은 이름 PDF가 들어 있는 untracked `papers/`가 있음 | development가 `papers/`를 commit하면 benchmark merge 때 "untracked file would be overwritten"으로 멈춘다. PDF는 commit에서 빼는 것을 제안 |
| 2.7 | sqp multi-seed 선택: `violation`에 T72 reach 초과를 더함 | `reach is None`(T72 꺼짐)이면 그대로. Panda 1판(multi-seed 16)도 영향 없음으로 읽었다 |

## 3. benchmark 작업 트리 상태 (merge 전에 정리해야 함)

| 경로 | 상태 | 소유 (추정) |
|---|---|---|
| `AG3S/ag3s/docs/handoff/SL6.task.md` | 수정 (+1 줄) | benchmark lead (SL6) |
| `paper/notes/SAFELIBERO_PLAN.md`, `paper/tables/table05_prior/table.tex` | 수정 | 논문 작업 세션 (확인 필요) |
| `benchmark/docs/AG3S_FINAL_CONFIG_PANDA_KO.md`, `doc/active/plans/*` (3개), `main.pdf`, `papers/` | untracked | SL6 · 계획 세션 (확인 필요) |
| `experiments/segmentation/`, `tests/segmentation/`, `doc/active/experiments/`, 이 문서 | untracked | **benchmark-44 (이 세션)** — 야간 분할·CBF 실험 |

untracked 파일은 merge 대상과 경로가 겹치지 않으면 merge를 막지 않는다(2.6 제외).

## 4. merge 뒤 확인할 것 (제안)
1. 합친 결과로 benchmark 회귀 테스트: `tests/benchmark/test_sl6a_panda_final.py`, `test_sl6h_panda_placement_contact.py`, `test_panda_sl4.py`, `test_panda_sl5j_recorder.py`, `tests/trajopt/test_sl6a_injections.py`, `test_sl6h_placement_contact.py`, `_sl5a_golden.py`를 쓰는 V11AB golden.
2. development 쪽 새 테스트(T67–T90) 전체.
3. 실제 merge 전에 scratch worktree에서 시험 merge + 위 테스트 (작업 트리를 건드리지 않음).

## 5. development main(02fe59e6)이 새로 가져오는 기능 — 하는 일로 정리 (사용자 판정용)

출처는 development 문서(P0 이식 명세, DECISIONS_20261103 · dev07, 규칙 목록, 각 impl/task)와 `serve_safe.py` argparse diff입니다. 코드를 읽어 정리했으며 실행해 확인한 것은 아닙니다. 모든 새 flag는 기본 꺼짐입니다.

| # | 기능 | 하는 일 | development 상태 | Panda에 그대로 쓸 수 있나 |
|---|---|---|---|---|
| 1 | 표면 영역 검사 · 유지 · 쥐는 중 고정 · 공 끔 | target 범위를 depth 표면 위에서 고른 영역으로 정하는 기존 표면 추출의 후처리. ATS target 점과 겹침이 50% 이상(10 mm 안)인 영역만 채택하고, 영역이 없으면 직전 영역을 유지하며, 닫는 중 · 쥔 동안 다시 고르지 않는다 | **P0 채택** | 규칙은 로봇 무관. 다만 head가 L10H7로 고정돼 있어 Panda head를 허용해야 함(판정 3 = Panda head 사용) |
| 2 | 표면 선택 GPU 계산 | 같은 계산을 CUDA로 | **P0 채택** (487 요청 CPU와 IoU 1.0, p95 768 → 83–105 ms) | 무관 |
| 3 | 쥐기 직전 hull 채우기 | 닫는 중이거나 손바닥이 영역 중심 80 mm 안이면 손 층에서 마지막 영역 convex hull + 5 mm를 지움 | **P0 채택** (재생 사전 기준은 미통과) | 80 · 5 mm는 RB-Y1 그리퍼 기준 |
| 4 | 손 층 target 접촉 여유 | target 표본 10 mm 안 · 지지면 위 seed를 손 층에서 지움 | **P0 채택** (사전 기준 미통과). SafeLIBERO는 target 옆 장애물 접촉을 충돌로 세므로 development도 범위를 좁히자고 제안(D1) | 10 mm는 RB-Y1 값 |
| 5 | 표면 분할 · 조각 붙이기 | 그리퍼 폭보다 넓은 영역을 attention으로 다시 가름 | **폐기** | — |
| 6 | 오목 꺾임 끊기 | 표면 graph에서 오목하게 꺾인 edge를 끊어 맞닿은 물체를 가름 | 실험 (P0 금지) | 기하만 씀 |
| 7 | target 유지 기억 | 한 번 target이 된 기억 칸을 ray가 깎기 전까지 target으로 유지 | 실험, 사용자 판정 대기 (P0 제외) | 무관 |
| 8 | 옛 target 위치가 비면 다시 고름 | 쥐기 전 옛 target 점을 depth에 투영해 2 요청 연속 비면 놓고 200 mm 안에서 재선택 | 실험 (P0 끔) | depth 기반, 문턱은 RB-Y1 장면 |
| 9 | 손가락 사이 통로 carve | 쥐기 전 손가락 사이 · 앞 60 mm에서 장애물 seed를 지움 | **철회** | RB-Y1 그리퍼 기하만 지원 |
| 10 | 기하 grasp 후보 + 도달 상자 | 충돌을 고려한 grasp 후보를 골라 TrajOpt에 ±5 mm 도달 상자 · 회전 항 | 미채택 | RB-Y1 그리퍼 spec |
| 11 | 5단계 분류기 | VLA hidden(layer 4, 카메라 3대)으로 접근 · 쥐기 · 운반 · 놓기 · 물러남을 분류해 gate · 규칙을 정함 | 문서끼리 상태가 다름 (dev-07 "채택", P0 "끔") | **RB-Y1 체크포인트 학습 asset · 카메라 3대 고정 — 불가** |
| 12 | 정답 단계 공급 | 분류기 대신 MuJoCo 정답 단계를 씀 | 분석용 | **시뮬레이터 특권 정보** |
| 13 | HSR6 ④ 창 넓히기 | approach에서도 A6 창이 열리면 이웃 허용 · A4 · G1 허락 | 미채택 | RB-Y1 HSR6 probe (Panda는 별도 probe). scope nbc와 함께는 거절(판정 1) |
| 14 | 지지면 거리 보정 | 지지면으로 분류된 로봇 행의 정밀 계층 거리에 B mm를 더함 | 미채택 | 4 mm는 RB-Y1 장면 TSDF bias |
| 15 | `--neighbour-contact-mm 0`의 의미 변경 | 예전엔 시작 오류, 이제 "켜짐 · 겹침 0" | 미채택 | 무관 |
| 16 | 충돌 없으면 그대로 통과 | VLA 행동에 위반이 없으면 TrajOpt를 거치지 않음 | 미채택 (P1 7/10 · 금지 접촉 3 vs P0 9/10 · 0). 동기는 장애물 없는 LIBERO에서 VLA 성공률 유지 | 무관 |
| 17 | hidden probe target 신호 | image token hidden에 학습한 probe로 target 지도를 만들어 attention 지도 대신 씀 | 실험 (closed loop 결과 없음) | **RB-Y1 학습 asset · 카메라 3대 — 불가** |

모든 flag를 꺼도 동작이 바뀌는 곳은 코드 읽기로는 찾지 못했다(각 단계의 "꺼짐 = bit 동일" test 통과 기록 있음). 예외: 15번, 그리고 RB-Y1 학습 3단계 gate가 서버 기본 켜짐인 점(새 기능은 아님).

## 6. 시험 merge 진행 기록 (scratch worktree, 실제 checkout은 그대로)

| 시각 (KST) | 내용 |
|---|---|
| 11-03 13:1x | main 02fe59e6을 HEAD 96fda38a 위에 시험 merge했다. 충돌 6개 파일 25곳: grasp_latch 3 · hsr6 2 · linearize 2 · sqp 2 · serve_safe 4 · safe_policy 12. 추가끼리 겹친 것은 둘 다 남겼다. `_hsr6_objective_ok`는 `(term, q_now, manipulators)`로 합쳤고, 판정 1에 따라 `--hsr6-objective-window` + `--hsr6-objective-scope nbc` 조합을 시작할 때 거절하게 했다. 합친 6개 파일은 compile된다 |
| 13:5x–14:38 | 전체 테스트를 한 번에 돌리면 pod OOM(cgroup oom_kill)으로 죽어서 디렉터리별로 나눠 돌렸다. tests/ag3s: 4 failed / 1985 passed |
| 15:2x | benchmark·ag3s에서 실패가 난 20개 파일을 시험 merge와 merge 전 HEAD(20bdf7a4)에서 같은 조건으로 다시 돌렸다. merge: 11 failed + 20 errors, HEAD: 10 failed + 20 errors. **merge에서만 실패한 것은 1개**(`test_t86_patch_probe`, `.venv-ag3s`에 jax가 없음 = 환경). 공통 실패 30개는 worktree 환경 때문이다(golden이 checkout 경로를 기록 · worktree에 `.venv-libero` 등이 없음) |
| 15:3x | tests/trajopt을 파일 하나씩 다시 돌리는 중 |

## 7. 사용자 판정 (2026-11-03) 과 시험 merge 반영

| 판정 | 내용 | 시험 merge 반영 |
|---|---|---|
| 1 | scope nbc와 window를 함께 켜면 시작할 때 거절 | `serve_safe.reject_hsr6_flags`. `_hsr6_objective_ok(q_now, manipulators, *, term="nbc")`: development 테스트가 위치 인자로 `(q_now, manipulators)`를 넘겨서, `term`은 keyword로만 받게 했다. benchmark 테스트 `test_sl6a_injections.py` 4줄은 `term=`으로 고쳤다 |
| 2 | 새 기능은 **허용**: 표면 추출 · 검사(1), GPU(2) / **조건부**(Panda 값 재설정 + 오프라인 측정 후): hull 채우기(3), 충돌 없으면 통과(16), target 유지 기억(7), neighbour 0 의미(15, 영향 없음) / 나머지는 **거절** | Panda 서버는 AG3S 설정과 SafePolicy를 직접 만들기 때문에, 허용 목록에 넣기만 하면 flag를 받고도 적용하지 않는다. 그래서 merge commit에서는 **merge가 가져온 option 34개를 모두 거절 목록에 넣고**, RB-Y1 학습 hidden probe attention source도 거절한다. 허용 · 조건부 기능은 다음 단계에서 Panda에 실제로 연결하고 테스트한 뒤, 오프라인 재생으로 target 지움 · 장애물 잘못 지움을 재고 나서 하나씩 목록에서 뺀다 |
| 3 | P0 표면 추출은 Panda용 attention head · layer를 쓴다 | 다음 단계(이식)에서 `seed_map_source_ok`를 Panda head(`noun_target:L8H6`)도 받게 고친다. merge commit에서는 표면 추출이 Panda에서 거절된 상태라 바꾸지 않는다 |
| 4 | 시험 merge → 테스트 → 보고 → 실제 merge는 다시 승인 | 진행 중 |

새 테스트 `tests/benchmark/test_merge_02fe59e6_panda.py`(option 34개 각각 거절, hidden probe 거절, 조합 거절). 관련 묶음 339 passed / 5 skipped(benchmark Panda 최종 구성 · SL6-A · SL6-H · HSR6 · T67 · T72 · T82 · 새 테스트).

## 8. 시험 merge 회귀 결과 (최종, 2026-11-03 16:3x KST)

| 묶음 | 시험 merge | merge 전 HEAD와 비교 |
|---|---|---|
| tests/benchmark · tests/ag3s에서 실패가 난 20개 파일 | 11 failed + 20 errors | HEAD 10 failed + 20 errors. 공통 30개는 worktree 환경(golden이 checkout 경로 기록 · venv 연결). merge에서만: `test_t86_patch_probe` 1개(`.venv-ag3s`에 jax 없음) |
| tests/ag3s 전체 | 4 failed / 1985 passed | 4개는 위 20개 파일에 포함 |
| tests/trajopt 122개 파일 (파일별 실행) | 통과하지 못한 것: legacy fixture 필요 3개(환경), `test_t43t_replay` 1 failed, `test_t46f_a14_wrist_cover`(전체 실행에서 OOM) | `test_t43t_replay`는 HEAD에서도 같은 실패. `test_t46f_a14_wrist_cover`는 혼자 돌리면 30 passed(최대 RSS 31.4 GiB) |
| 판정 반영 관련 묶음 | 339 passed / 5 skipped | — |

**merge 때문에 생긴 코드 회귀는 찾지 못했다.** 실제 checkout의 미커밋 · untracked 경로 가운데 merge가 바꾸는 파일과 겹치는 것은 없다(경로 대조).
