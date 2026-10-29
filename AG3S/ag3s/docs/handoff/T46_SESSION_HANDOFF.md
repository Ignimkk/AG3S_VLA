# T46 세션 인계 (2026-10-13 KST) — 새 세션은 여기부터 읽는다

> writer: lead (A0, T46 세션). 사용자 지시 (2026-10-13): 컨텍스트 부족으로 **현재 작업까지만 수행, 판정은 새 세션에서.**
> 이 문서 이후 내려진 판정은 없다. 아래 "판정 대기" 는 전부 사용자 몫이다.

## 0. 먼저 읽을 것 (순서)
1. `CLAUDE.md` · `.claude/ag3s-rules.md` · `doc/active/plans/develope_PLAN.md` §0.10 (방향의 정본 — **작업 트리에 사용자의 미커밋 수정이 있다, 건드리지 말 것**).
2. 로그: `AG3S/ag3s/docs/AG3S_T47_LOG.md` (최신 정본, T46b 이후 T46 접미사 STEP 포함) → `AG3S_T0T6_LOG.md` (T46 까지).
3. task: `T46.task.md` (끝의 "T47 세션과의 조율" — port · commit · log · W1 규칙), `T46c.task.md` (F1 채택 · 한계 L1–L11 · W1 판정), `T46d.task.md`, `T46e.task.md`, **`T46f.task.md` (DX-04 LTE 검증 블록 — 다음 작업의 정본)**.
4. 사용자 보고 형식 (memory): 목적 + 이전 (V11AB = C0 · C1a) 대비 before → after, develope_PLAN 의 어느 표 · 그림에 쓰이는지, LTE · F1 등 제안 방법의 우수성을 전제하지 않음 (사전 고정 기준 · 실제 MuJoCo contact 주 지표 · 정지 ≠ 보호 · 사후 기준 변경 금지).

## 1. 이 세션에서 끝난 것 (commit, push 안 함)
| STEP | 결과 요지 | 산출 |
|---|---|---|
| CL2 · CL-A · D-SF · D-TF | closed-loop 재실행과 원인 진단 | `T44e-CL2/CLA.verify.json`, `T45SF/TF.verify.json` |
| T46 (F1/F2) · CL3 | F1 = non-target 근거에서 target 표면 점 제외. CL3: grasp 직전 HOLD run C1a 12 → F1 6, 이웃 과일 접촉 run 4 → 1, task 14 → 13 (유의 아님) | `T46.verify.json`, `T46-CL3.verify.json` |
| T46b | close 이후 HOLD 증가 = placed `no_seed` 인증 공백 (F1 간접 영향) | `T46b.verify.json` |
| T46c | **F1 채택 (한계 L1–L11)**, **W1** (`--placed-no-seed-certify`, 기본 off) 구현 · 검증 PASS (commit `6dfaf7bb`). 사용자 판정: Ablation 모든 arm 에서 W1 on, 18 쌍 유지 | `T46c.impl/verify.json` |
| T46d | gripper scope 에서 LTE vs global 비교 퇴화, arm_5 추가 차이는 형상 근사 오류의 영향 (사용자 지정 문구) | `T46d.verify.json` |
| 환경 | cuRobo editable install (`7339fa4d`), legacy index 경로 rebase (`c906cda2`), flock 사용법 문서 (`2cf87e92`) | — |

## 2. 진행 중 / 미적용 (격리 worktree — 메인 트리 아님)
| 작업 | 위치 | 상태 |
|---|---|---|
| **T46e** visible-target oracle extent (기본 off) | worktree `.claude/worktrees/agent-a770cbc1ab3162e54` (base 15314ba3, 미커밋 11 파일 + `T46e.impl.md` · `T46e.verify.json` · `figures/t46/t46e-*`) | 구현 완료 · **V-T46e 사전 기준 4/4 PASS** · **메인 미적용**. 추가 발견: grasp 직전 162 chunk 전체에서 off/F1/F1GT 가 0 행이던 7 chunk 에 oracle 만 143 행을 새로 만듦 (원인 미진단). raw: `results/development/20261013T012537+0900_15314ba39dbd_8f15850f0784/verify/T46e` |
| **T46f A14** 형상 보정 (FT sensor `wrist_cover`, arm_5 보정, `--links gripper_wrist`, `--target-field-policy none`, coverage audit) | worktree `.claude/worktrees/agent-a48b7e7cad456d7a9` (base ab9b3fba), 진행 기록 `AG3S/ag3s/docs/handoff/T46f-A14.progress.md` | 구현 · 새 테스트 30 통과, 전체 pytest 진행 중 (이 문서 작성 시점). audit run2: arm_6 L 0.97 mm crevice 하나로 all_pass false. run: `results/development/20261013T091400+0900_ab9b3fba7316_a2f876957965/impl/T46f-A14/` — **결과는 §5 에 덧붙인다** |

worktree 의 diff 를 메인에 적용할 때: T47 세션에 먼저 알린다 (T47 의 T53 rollout · 추출 종료 알림 뒤), 기본 off · bit-identical 확인, pathspec commit. 두 worktree 가 `serve_safe.py` · `config.py` · `safe_policy.py` · `ta_offline.py` 를 함께 고치므로 순서대로 적용하며 충돌을 해결한다.

## 3. 판정 대기 (사용자)
1. **T46e oracle 의 새 충돌 7 chunk** 원인 진단 여부 (lead 추천: 진단).
2. **oracle 의 지지면 층** 을 다른 arm 처럼 지울지 (lead 추천: 지움 — arm 간 처리 동일).
3. T46e 메인 적용 시점 (lead 추천: 1 · 2 해결 · 재검증 뒤).
4. identity 전체 GT 교체 (DX-01 조건 A) — lead 추천: 지금은 안 함, `identity_matches_gt` 기록으로 대체.
5. `AG3S_T47_LOG.md` 진행 현황 표의 이전 T46c 행 ("W1 진행 중") 에 취소선을 칠지.
6. T46f A14 결과 (audit · 테스트) 를 보고 단계 2 (진단 장면, S1–S4) 로 넘어갈지 — `T46f.task.md` §8 중단 기준.
7. CL4 (DT-02 extent closed-loop) 는 보정 형상 뒤로 미룸 (T46f §3).

## 4. 다른 세션과의 조율 (현재 유효)
- **T47 세션** (`T47 Attention head 고르는 방법론`): T47–T57 예약 (T53 rollout 진행 중, GPU lock 보유, port 8290·8291 이번만). 이후 port **T46 = 8290–8291, T47 = 8292–8293**. 로그 쓰기 전 · 후 서로 알림, pathspec commit 만. T55 · T57 은 W1 값을 arm 간 고정. T46e 메인 적용은 T53 종료 알림 뒤.
- **benchmark 세션** (`ag3s-vla-benchmark-60` · `-74`): 공용 GPU lock 규칙을 benchmark CLAUDE.md 에 반영 (`41339944`, 형식 수정 `d53d8e07`). SL4b GPU 작업 ~11:30 KST 까지.
- GPU: `( flock -w N 9 || exit 13; CMD ) 9>/tmp/ag3s-vla-gpu-validation.lock` + `nvidia-smi` 확인. TO 는 wall-clock 마감이라 closed-loop 중 GPU 공유 금지.
- 작업 트리의 다른 세션 미커밋 파일 (T47–T57 · `doc/active/plans/develope_PLAN.md` · `.claude/worktrees/`) 은 commit 하지 않는다.

## 5. A14 결과 (2026-10-13 KST, 구현 종료 — 독립 검증 · 판정 전)
worktree `.claude/worktrees/agent-a48b7e7cad456d7a9` (base ab9b3fba, 미커밋). 보고서 `AG3S/ag3s/docs/handoff/T46f-A14.impl.md` (worktree 안).
적용용 patch: `results/development/20261013T091400+0900_ab9b3fba7316_a2f876957965/impl/T46f-A14/diff_final_with_new.patch` (새 파일 포함). raw: 같은 폴더 `audit.json` · `pytest_all.log`.
- **변경 (기본 off, 기본 flag 에서 모델 bit 동일):** `--links gripper_wrist` = gripper + 양손 arm_5 · arm_6 · FT sensor. arm_5 URDF capsule (r 75 mm) → MJCF mesh 몸통 구, arm_6 도 같은 방식으로 보정, FT sensor 덮개 구 (role `wrist_cover`, URDF link `FT_sensor_<S>`). 비권한 구는 권한 손 MJCF 껍질을 keep-out 으로. FT 는 `clearance.PROTECTED_LINKS` (어떤 설정에서도 비권한). `--target-field-policy none` (예외 없음) 추가.
- **audit (구현자 측정, 독립 검증 전):** arm_5 손 geom 침범 20.0 mm → 0 (손 geom 까지 ≥ 67 mm); 겉면 구 밖 비율 arm_5 98 % → 0.02 %, arm_6 96 % → 0.05–0.07 %, FT 100 % → 0.68 % (최대 1.9 mm, 손바닥 옆 틈); 넘침 ≤ 5.04 mm. **사전 기준 all_pass = false** — arm_6 L 표본 2 개가 0.97 mm (기준 0.5 mm, 사후 변경 안 함).
- 테스트: 새 30 passed, 전체 CPU pytest 3075 passed / 11 skipped / 0 failed (CUDA test 8 개 미실행 — GPU 를 다른 세션이 사용). 구 수 1,122 (gripper 270 의 4.2 배), TO 시간 · closed-loop · 회귀 기준선 미측정.
- **구현자가 남긴 판정 필요 7 건** (impl.md): ① FT 0.68 % 가 §8 "형상 누락" 인가 ② arm_6 L 기준 초과 (inset 0.5 mm 로 해결 가능, 구 84 → 99) ③ arm_6 보정은 task 범위 밖 추가 ④ 구 1,122 개 비용 (overflow 10 mm 면 ~686) ⑤ `none` 이 verdict 허용 접촉까지 없앰 ⑥ exclude_all 에서 FT 도 target-free 층을 읽음 ⑦ audit 침범 기준 0 → 0.001 mm (본 측정 전, FK 부동소수).
- 다음 (새 세션 판정 뒤): V-T46f-A14 독립 검증 → 메인 적용 (T46e 와 충돌 해결, T47 에 알림) → T46f 단계 2 (진단 장면).
