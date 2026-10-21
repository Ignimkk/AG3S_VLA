# IFG-CL3-run — 선별 없는 paired 재평가: C1aN vs SURF, 40 장면 × seed 1 (verifier)

작성: 2026-10-18 KST, lead. 근거: `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md` 2026-10-18 14:30 절 (사용자 지시 전문 요약).
선행 틀: `IFG-CL1-run.task.md` 의 run `20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1/p/` (모든 `LEAD_AMENDMENTS.md` 수정 포함)
과 `IFG-HURDLE` run 의 `p/bench_watch.py` (lead 수정판 — Claude tool shell 제외).

## 사용자 판정 (2026-10-18)

- **규모:** 40 장면 × seed 1 × arm 2 = 80 run. **arm: C1aN vs SURF** (IFG-CL1 정의 그대로, snapshot tree `f0d8046b` = 현재 SURF. 하이브리드 · feature 수정 넣지 않음).
- **구성:** clear 14 · adjacent 13 · place 13.
- **선정:** 어느 arm 의 성공 여부도 보지 않고 **metadata 로만**. 규칙과 목록 (episode · seed · stratum · 장애물 profile) 을 **rollout 전에** `seeds.json` 으로 고정하고 sha256 을 기록한다.
- **우선:** 설계 · 진단에 쓰지 않은 장면. 제외 목록 = IFG `$RUN/used_episodes_prior.json` (`20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba`) ∪ held-out collector 의 모든 episode (`20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout/state.json` 의 episodes, 성공 · 실패 · reference 모두) ∪ T43 허들 7 episode. 제외 후 후보가 부족해 기존 장면에 새 seed 를 쓰게 되면 그 행을 `reused_scene=true` 로 표시.
- **실패 포함:** no-grasp · timeout · 접촉 실패 · 서버/client 오류 아닌 모든 실패를 평가에 남긴다. 성공 장면으로 교체 금지. (인프라 오류로 run 이 완료되지 않은 경우만 같은 seed 로 재시도, 재시도 횟수 기록.)
- **seed 를 자동 추가하지 않는다.**

## 장면 선정 규칙 (사전 고정)

1. 후보 = 16-D randomized dataset meta 의 episode (`meta/randomized_episodes.jsonl` 등 — collector `p/hc_plan.py` · `pool.json` 이 쓴 같은 원천) 중 task 가 place_one 이고 위 제외 목록에 없는 것.
2. 순서 = `sha256("IFG-CL3:" + str(episode))` 오름차순 (결과를 보기 전 고정). 앞에서부터 stratum 에 배정: 먼저 clear 14, 다음 adjacent 13, 다음 place 13 — 각 stratum 은 그 순서에서 장애물 배치 (아래) 가 **기하 조건으로** 가능한 episode 만 받는다. 불가능하면 그 episode 는 다음 stratum 후보로 넘기지 않고 건너뛰며 (`skipped_geometry` 기록) 다음 episode 로.
3. seed = `episode * 10 + 3` (collector 의 `*10+1` 과 겹치지 않게). used_arm (side) 은 dataset meta.

## 장애물 배치 — 기하 규칙만 (사용자 판정: rollout 없음)

collector `p/hc_search.py` 의 검사 틀을 쓰되 **기준 rollout 궤적 검사 (b) 는 쓰지 않는다**. reset 상태 (warmup 후 초기 자세) 와 scene metadata 만 쓴다.
- adjacent: `post_0` (반지름 0.015, 높이 0.10) 를 target 중심에서 수평 5–10 cm. 후보 위치는 고정 순서 (각도 0°, 45°, …, 315° × 거리 5, 7.5, 10 cm) 로 시도.
- place: `crate_post_0` 를 바구니 rim 옆. 후보 고정 순서.
- 조건 (모두 reset 상태): (a) 모든 robot geom 이 장애물에서 ≥ 0.03 m, (c) 장애물 표면과 target 표면 수평 간격 ≥ 0.04 m, 장애물이 다른 과일 · 바구니와 겹치지 않음, (d) adjacent 는 head camera (`zed_left`) 의 target 가림 ≤ 5 % 이고 보이는 target pixel ≥ 50 (224², hc_search 규약).
- 첫 번째로 조건을 만족하는 후보를 쓴다. obstacle profile JSON 을 장면마다 저장하고 검사 수치를 기록.

## 실행

- IFG-CL1 틀 그대로 (C1aN · SURF arm, server log guard, check_run, smoke gate (층이 만들어진 chunk 기준 적용 비율 ≥ 0.3)). 짝마다 두 arm 을 포트에 번갈아 배정.
- **launcher 는 만들고 사전 검사 (flag · import · address · legacy baseline · seeds.json 고정) 까지 하고 띄우지 않는다.** benchmark 세션이 GPU 를 쓰는 중이다 (~15:20 KST 종료 예정). lead 가 그 종료 알림 뒤에 직접 띄운다. launcher 는 그래도 lead 수정판 `bench_watch.py` 조건 (Claude tool shell 제외, 10 분 quiet) + blocking lock 을 둔다.

## 보고 (집계 스크립트는 지금 만들고 IFG-CL1 기록으로 동작 확인)

`AG3S/ag3s/docs/handoff/IFG-CL3.verify.json` (수치만): 주 지표 = 접촉 없는 작업 성공 (IFG-CL1 `agg_cl1.py` 정의). 
- paired 성공 차이 (SURF − C1aN) 와 95 % CI (episode 쌍 bootstrap 10,000, seed 20261018 + exact McNemar p 는 참고).
- 2×2: 둘 다 성공 · C1aN 만 · SURF 만 · 둘 다 실패. stratum 별.
- grasp 실패 · failure_mode 분포 · 접촉 (비허용 link–target, 장애물, 이웃 과일) · HOLD chunk · 완료 시간.
- 요청 지연 (client 왕복, 서버 단계별 median · p95, 533 ms 초과 비율) 과 계산 비용 (attention_surface 단계 ms, render ms).
- `reused_scene` · `skipped_geometry` · 재시도 기록.
동등성 · "영향 없음" 같은 판정 문구는 쓰지 않는다.

## 하지 말 것 · 넘길 것

코드 (`AG3S/` · `tests/` 의 `*.py`) 수정 금지, commit 금지, 다른 세션 process 금지, lock 없이 GPU 금지. 새 run (`python -m common.run_output`), run 밖을 가리키는 symlink 금지.
보고: run 경로, seeds.json (40 행 표: episode · stratum · seed · side · obstacle profile · reused 여부) 과 sha256, skipped 목록, 배치 검사 수치, launcher 경로와 띄우는 명령, 예상 소요.
