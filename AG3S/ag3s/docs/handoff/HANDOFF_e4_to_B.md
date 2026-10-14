# 인계 — 재검증 세션 (e4) → B (identity · phase source 분석) · A (메인)

> writer: e4 (재검증 작업 T44~T56), 2026-10-14 KST. 역할 정본: [`SESSIONS_A_B.md`](SESSIONS_A_B.md).
> 받는 세션: A = `ag3s-vla-development-95` (d2 를 대신함), B = `ag3s-vla-development-c4`.
> 이 세션은 인계 후 종료한다. **T61 의 선택 · held-out 판정은 열지 않았다** (사전 고정 [`T61.task.md`](T61.task.md) 그대로).
> 맨 아래 "인계 시점 상태" 절에 commit hash · 남은 process 를 적었다.

## (a) STEP 별 상태

| STEP | 상태 | 근거 |
|---|---|---|
| **T44–T56 재검증** (감사 문서 [`doc/reviews/TERMINATION_SCOPE_AUDIT_20261013_KO.md`](../../../../doc/reviews/TERMINATION_SCOPE_AUDIT_20261013_KO.md) 의 1–3 단계) | **완료** — T60 으로 수행 | 아래 T60 |
| **T60** 종료 구간 P · H 술어, CL2 · CL3 · T53 · T54 재집계, T47 · T49 · T51 · T48a 선택 재계산 | **완료 · 판정 반영** | [`T60.task.md`](T60.task.md) · [`T60.verify.json`](T60.verify.json) · 로그 "T60" 절 · figures `t60/` · code `t60_*.py` (commit `e8e5bae1`) |
| **T60b** H 미도달 27 run 진단 | **완료 · 결론 채택 (2026-10-14)** — "복귀 미도달은 TO 차단보다 정책의 복귀 제안 부재" | [`T60b.verify.json`](T60b.verify.json) · 로그 "T60b" 절 · figures `t60b/` · `t60b_return.py` |
| **T60c** T51b (명사 · combined Ω) 종료 창 재집계 | **완료** | [`T60c.verify.json`](T60c.verify.json) · 로그 "T60c" 절 · figures `t60c/` · `t60_t51b.py` |
| **T61** 사전 고정 head 선택 + 새 held-out 16 episode | **진행 중 (partial)** — 사전 고정 commit `95b6586c`; calibration 선택 chain 실행; held-out rollout · 추출 실행 (아래 "인계 시점 상태"). **선택 결과 해석 · held-out 평가 · 판정은 B** | [`T61.task.md`](T61.task.md) · [`T61.verify.json`](T61.verify.json) (partial) · `t61_select.py` · `t61_records.py` · `t61_build_verify.py` · `tests/ag3s/test_t61_select.py` |

## (b) 받아 둔 사용자 판정

| 날짜 (KST) | 판정 | 문서 위치 |
|---|---|---|
| 2026-10-13 | P = in-crate ∧ hand–target contact 없음 ∧ target 속도 ≤ 0.03 m/s, 8 row. H = P 이후 사용 팔 관절 max\|q−q0\| ≤ 0.35 rad ∧ TCP ≤ 0.12 m, 8 row. H 미도달 = H 기준 실패 (P 성공 별도). 범위 2 + 3 단계 | `T60.task.md` "사용자 판정" |
| 2026-10-14 | ① **P 주 작업 지표 + H 필수 보고** (develope_PLAN 평가 규칙 문장, commit `53f3911d`). H 완료 후 대기는 기본 지표에서 제외. 미도달 run 은 full horizon. ② 복귀 미도달 기록 진단 (→ T60b). ③ L8H2 를 관행적으로 유지하지 않고 사전 고정 알고리즘으로 선택 · 실행 중 고정 (→ T61), L8H2 는 기준선. ④ **이미 고정된 실험 (T54b · T54c · T55 · T56) 은 기존 주 판정 유지 + P · H 보조** (T55 는 placement 절단이 전체 기능 평가를 대신하지 않음). 새로 설계하는 실험 (T57 부터) 에 ① 적용. ⑤ T51b 재집계 진행, HSR demo 재시뮬 보류. ⑥ commit: 자기 파일 pathspec, 공유 로그는 세 세션 확인 후 한 세션이 한 번에 | `T60.task.md` "사용자 판정 2" · 로그 T60 §9 |
| 2026-10-14 | T60b 결론 채택 ("정책의 복귀 제안 부재"). 왜 특정 scene 에서 제자리 유지인지는 미확정, 추가 rollout 으로 바로 확대하지 않음 | 로그 "T60b" 절 끝 |
| 2026-10-14 | **T61 사전 고정 (판정 3):** held-out = 새 16 episode (기존 4 record 는 개발 · 참고); calibration ep0 · ep500 · ep1200 유지, **`long` 제외 (입력 배선 — T56 과 같은 이유)**; 강제 후보 없음 (L8H2 · T47 oracle 도 일반 규칙, L8H2 강제는 기준선 평가에만, filter threshold 바꾸지 않음); 명사 query 포함 · query 별 고정 배분 (6 개씩); **선택 단위 = query · denoise · aggregation · layer · head** (mean 이 선택되면 serving 도 mean); k 는 CV one-SE (fold 안 순위); bootstrap 은 보고만, L8H2 복귀 규칙 없음 | `T61.task.md` "사용자 판정 3" · "사전 고정" |

## (c) 판정 대기 질문

1. **T61 선택 구성의 serving 구현 (A 와 함께)** — 선택이 `mean` · `first` aggregation 이나 명사 query 면 serving 에 그 경로를 구현해야 한다 (판정 3-4). 구현 STEP 을 A 에 요청할 시점 · 범위.
2. **T61 held-out 결과 보고 방식** — 사전 고정은 "선택 구성 대 L8H2 paired 차이 + CI". 결과가 L8H2 와 비슷할 때 serving 을 바꿀지 (판정 3: "반드시 우월할 필요는 없다") 는 결과 뒤 사용자 판정.
3. **T60b 후속** — 특정 scene (ep110 · 123 · 3) 의 제자리 유지 원인 (offline counterfactual 로 볼지). 사용자는 "추가 rollout 으로 바로 확대하지 않음" 까지만 판정 (→ A).
4. **HSR demo (T48a val · T48b · T50) 의 P · H** — 재시뮬레이션 보류 판정. 다시 열지.
5. **T49 의 바뀐 label-free (secondary) 명사 · basket 집합** 은 held-out 평가를 하지 않았다 (exploratory).
6. **`doc/reviews/` 의 감사 문서 3 파일** (`TERMINATION_SCOPE_AUDIT_20261013_KO.md` · `.json` · `termination_scope_audit_20261013.py`) 은 untracked. 이 세션이 만든 것이 아니라 (끝에 후속 링크만 덧붙임) commit 하지 않았다. commit 여부는 사용자 판정.

## (d) 파일 · process · port · snapshot · run dir

- **run dir**: T60 = `/mnt/dev/AG3S_VLA-storage/results/development/20261013T223348+0900_cf38c9dbca7b_8aef81831126` (`verify/T60` · `T60b` · `T60c`). T61 = `/mnt/dev/AG3S_VLA-storage/results/development/20261014T183529+0900_53f3911df197_dd2fad8756a6` (`verify/T61/{select,runs,p,extract_logs,logs}`, `snapshot/`).
- **snapshot**: `95b6586c` git archive → `<T61 run>/snapshot` (서빙 파일 6 개 md5 = T54c snapshot; A14c `8df34591` 이전). 서버 · client · code md5 검사가 모두 snapshot 안에서 돈다.
- **port**: 8292 · 8293 (B 몫). rollout 이 끝나면 비어 있다.
- **T60 mask**: `<T60 run>/verify/T60/step1/masks/<SET>__<NAME>.npz` (CL2 · CL3 · T53 의 keep_rows / keep_frames P · H), `<T60 run>/verify/T60/step3/records/<key>.npz` (T47 · T51 record).
- 미커밋 · process 는 맨 아래 "인계 시점 상태".

## (e) 넘길 것

### A 로

- **T60 P · H 술어 (DX-00 평가 정의)** — 정의: `T60.task.md` 표 · 로그 T60 §2. 코드: `AG3S/ag3s/experiments/studies/t60_termination.py` (`Params` 기본값 = 판정값, `endpoints` row 단위, `endpoints_sampled` inference 시점 근사 — 142 run 에서 P 0.9977 · H 1.0 일치, `keep_rows` / `keep_frames`, `series_from_record` replay npz + traj.npz loader). 새 closed-loop 실험의 기본 지표로 쓰려면 이 함수로 run 마다 t_P · t_H 를 계산하고 P 창 지표를 주, H 창 지표를 필수 보고로 낸다 (`t60_closed_loop.py` 가 그 예: success · success_ct · HOLD · 완료 시간).
- runtime 종료 감지기 (감사 문서 4 단계) 는 구현하지 않았다. 필요하면 `endpoints` 를 client 에 넣는 STEP (서빙 = A).
- **T60b 후속** — (c) 3. 진단 코드 `t60b_return.py` (사전 고정 분류 `classify`, 정책 원래 출력은 planning row 의 `actions_reference`).

### B 로 — T61 분석 순서 (사전 고정을 바꾸지 않는다)

1. **선택 결과 고정 확인.** `<T61 run>/verify/T61/select/combined/gt/ksel/ksel__scene.json` 의 `selected` 가 사전 고정 절차의 최종 구성이다 (k = one-SE). `T61.verify.json` `selection_outputs_sha256` 와 같은지 확인하고 `T61.selection.json` (구성 · sha256) 으로 따로 저장한 **뒤에** held-out 을 연다. query 별 결과 (action 만 · 명사만) 와 bootstrap 은 보고만.
2. **held-out rollout 결과 정리.** `<T61 run>/verify/T61/runs/C1a/R/` 16 run. `p/` 의 `analyze_cl.py` · `run_replays_cl.sh` (replay_contacts) · `agg_cl.py` 로 success · HOLD · contact (T54c 와 같음).
3. **held-out 의 P · H endpoint.** replay npz 가 생기면 `t60_termination.series_from_record` → `endpoints` (row 단위, 판정값) 로 episode 별 t_P · t_H, `keep_frames` 로 inference frame mask.
4. **추출물.** `<T61 run>/verify/T61/{attention,noun,gt}/` (t61_records 의 `t53_extract` wrapper). feat 는 생략했다 (T61 평가에 쓰지 않음).
5. **held-out 3D row.** 선택 구성 · L8H2 (기준선 강제) · T51 full 선택 `last:L12H4` · T54 N `noun_target:L10H7` 를 16 episode 에 replay (T51 replay 옵션 그대로, `t51_replay.run_sources` 를 t61 registry 로). P 창이 주, full · H 보조.
6. **평가.** episode 평균 GT `identity_correct`, purity, coverage, paired 차이 (선택 − L8H2) episode bootstrap 95 % CI (B 10000, seed 6102), moving block 5 민감도. 표현 제한은 `T61.task.md` "표현".
7. 결과 보고 → 사용자 판정 → (serving 구현은 A 에 task 로 요청).

## (f) 알려진 함정

- **이름으로 저장되는 cache.** T51 의 특수 source `paper3_s1` row 는 이름이 같아도 내용 (T47 paper top-3) 이 `--t47-dir` 에 따라 바뀐다. 다른 T47 출력으로 만든 row 를 복사했다면 `rows/*/paper3_s1.jsonl` 을 지워야 한다 (T60 에서 실제로 0.000 이 나왔다 → 0.644 정정). `t60_t51b.py` 는 시작 시 이를 검사한다.
- **`predicted_chunks` 는 TO 뒤의 chunk 다** (`--safe-remote`). 정책 원래 출력은 frames.jsonl planning row 의 `actions_reference`.
- **`long` · `ep1800` 은 X2 수정 전 head 0 녹화** (cam_high 가 벽). calibration · dev 에서 제외 (T56 · T61).
- **기존 held-out 4 record 는 여러 번 열렸다** — T61 평가에 쓰지 않는다 (개발 · 참고).
- **T61 calibration 순위를 일부 보았다.** e4 는 score3d 진행 확인 중 calibration 3D 1 위 (`mean:L10H7`, scene 평균 0.698) 를 보았다. held-out 은 열지 않았다.
- **`resolve_output_path`** 는 run dir 이 `results/development` 바로 아래여야 한다 — 창별 출력은 wrapper 에서 base 를 바꿨다 (`t60_t51b.py`).
- **`t51_study.out_root` 는 `verify/T51` 고정** — 창별 root 는 `t60_t51.py` 처럼 patch.
- **세션 재시작이 nohup 없이 띄운 chain 을 죽였다** (T61 ksel_gt 도중). `resume_select.sh` 로 재개했고 끝난 단계 (filter · score2d · score3d) 의 출력과 cache row 는 그대로 재사용했다. 긴 작업은 `nohup setsid` 로 띄운다.
- **pkill / kill 패턴** 이 자기 bash 명령 문자열에 걸려 그 명령을 죽인 적이 있다 (exit 144). pid 로 죽인다.
- **(apple, left) 미사용 test episode 가 없다** — T61 held-out 은 (apple, right) 4 개. 과일 × 팔로 일반화하지 않는다.
- **pod 메모리**: T61 선택 chain ≈ 4 GB, rollout 서버 2 + client ≈ 20 GB. T56 test (≈ 28 GB) 와 겹치지 않게 순서를 맞췄다.
- **calibration 5 scene** 이라 CV SE 가 크다 — one-SE 가 작은 k 를 고르기 쉽다 (규칙은 바꾸지 않는다).

## 인계 시점 상태

(인계 직전에 채운다)

> **(세션 B 가 채움, 2026-10-14 22:5x KST)** e4 는 이 절을 채우지 않고 종료했다.
> - 확인한 상태: T61 held-out rollout 16/16 (`*.done`), 추출 단계 `ALL DONE` (13:16 UTC), 선택 chain `chain done` (13:34 UTC). T61 관련 process 는 남아 있지 않다.
> - B 의 처리:
>   - `t61_build_verify.py` 로 `T61.verify.json` (partial) 을 만들었다. 선택 출력 sha256 8 파일은 B 가 고정한 `T61.selection.json` (commit `b49f5f6c`) 의 같은 파일과 일치한다.
>   - 이 문서 · `t61_build_verify.py` · `T61.verify.json` 을 A 의 요청에 따라 B 가 pathspec commit 한다.
