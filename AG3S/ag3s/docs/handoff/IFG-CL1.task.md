# IFG-CL1 — serving 에 attention-surface target extent 넣기 (closed-loop 1 단계, CPU 구현 · unit test 만)

작성: 2026-10-16 KST, lead (IFG 분석 session). STEP 번호 없음 (사용자 지시: IFG 작업은 `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md` 에만 기록).

## 목적

IFG dev 에서 고정한 target extent 방법 (`top1-only-ns+rm:N`, `$RUN/LOCK_heldout.json` → `arms.candidate_extent`) 을 serving 에서
돌려, 현재 serving (C1a: `--target-exclusion ball-owned`) 과 **같은 cuRobo ESDF · 같은 TrajOpt · 같은 flag** 에서 closed-loop 로 비교한다.
사용자 판정 (2026-10-16): 단계적 — 1 단계는 **표면 영역 + rendered-robot 표면 제거만**. feature 기억 · 경보는 2 단계 (이 task 아님).

`$RUN = /mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba`

## 설계 (lead 결정 — 바꿀 필요가 있으면 구현 전에 SendMessage 로 lead 에게 묻는다)

이미 있는 `--target-extent oracle-visible` 경로 (`AG3S/ag3s/runtime/oracle_segmentation.py`, `curobo_builder.oracle_visible_split`) 는
**카메라별 code 영상** (0 = 없음/로봇, 1 = target, 2 = static, 3.. = 다른 잡을 물체) 을 받아 target-free 미세 계층의 target 범위를 정한다.
지금은 그 영상을 client 의 시뮬레이터 GT segmentation 에서 받는다. 이번 단계는 **같은 code 영상을 서버가 attention + depth 로 직접 만든다.**
그 뒤의 memory · carving · split · 계층 · 권한은 oracle 경로를 그대로 쓴다 (target 범위의 출처만 다르다).

1. **새 값** `esdf.target_extent = "attention_surface"` (CLI `--target-extent attention-surface`). `TARGET_EXTENTS` 에 추가. 기본 (`estimated`) 동작은 비트 단위로 그대로.
   `--attention-source noun_target:L10H7` 가 없으면 시작 시 에러 (fail loudly).
2. **서버 안 code 영상 생성** (새 모듈 `AG3S/ag3s/runtime/attention_surface.py`):
   - 입력 (요청마다): 3 카메라 depth · K · T_base_cam (서버가 이미 받는 것), noun attention map (768 = 3 × 16 × 16, `attention_source` 가 이미 계산), robot mask (아래 3).
   - 처리 = LOCK 의 `candidate_extent` 그대로: `ifg_graph.GraphConfig()` 기본값 (voxel 5 mm, range_max 2.0, jump_gate 10 mm, roi 0.15 m …),
     `ifg_round_ns.eligible(geo, cfg, True)` (≥ 60 non-robot non-support px), `ifg_round_ns.g_and_top1` (seed-mass top-1 region, τ 0.5).
   - **잠금 코드 (`AG3S/ag3s/experiments/studies/ifg_*.py`) 는 수정하지 않는다** (LOCK 에 sha256 이 있다). import 해서 쓴다. GT 인자 (`class_map`, `classes`) 는
     평가 전용이므로 더미를 넘기되, 그 더미가 선택 결과를 바꾸지 않음을 test 로 보인다.
   - 출력: 선택된 region node 가 나온 pixel → code 1, 그 밖의 유효 non-robot pixel → code 2, robot · 무효 → 0. 카메라 해상도는 depth 원본 해상도.
   - 영역이 없으면 (seed 없음 · region 없음) 그 frame 은 oracle 경로와 같은 규약으로 `estimated` 로 짓고 이유를 `stats["target_free"]["attention_surface"]` 에 남긴다.
   - 시간 (ms) 을 stats 에 남긴다.
3. **rendered-robot 표면 제거** (새 flag `--robot-mask render`, 기본 = 지금 그대로):
   - 서버의 `--model-xml` 로봇 모델을 요청의 관절 상태로 posing → 3 카메라 (요청의 K · T) 로 robot-only depth 렌더 → 관측 depth 와 차 ≤ 10 mm 인 pixel = robot.
     offline 기준 구현은 `ifg_render_mask.py` (MuJoCo replay). 서버는 snapshot scene XML 이 없으므로 로봇 모델만 렌더한다.
   - 이 mask 로 (a) 2 의 graph 입력 robot mask, (b) TSDF 에 넣는 점에서 로봇 pixel 을 뺀다. sphere self-filter 를 끌지 / 같이 둘지는
     현재 serving 의 self-filter 경로를 읽고 lead 에게 제안한다 (offline R6 는 render mask 만 썼다).
4. **기록:** 기존 `--record-constraints` chunk 에 code 영상 (또는 target pixel 수 · node 수) · fallback 이유 · ms 를 남긴다.

## 수락 조건 (unit test, CPU, `.venv-ag3s` 또는 `.venv-openpi-live`)

- **parity:** dev frame 5 개 이상 (`$RUN/s0/replay/<frame>.npz` 의 depth/K/T, `$RUN/rounds/R6/robot_mask/<frame>.npz`, dump `derived.npz` 의
  `noun_target_L10H7`) 로 새 모듈이 고른 node 집합이 offline `top1-only-ns+rm:N` 과 같다 (node IoU ≥ 0.99; 다르면 이유).
- code 영상의 code-1 pixel 을 다시 올린 점이 선택 node 의 voxel 에 들어간다.
- 기본 설정 (`estimated`, `--robot-mask` 미지정) 에서 기존 test 전부 통과, 동작 비트 동일.
- `attention_surface` 인데 attention source 가 없으면 시작 에러.
- render robot mask: 합성 상태 1 개 이상에서 offline `ifg_render_mask` 결과와 pixel 일치율 보고.

## 하지 말 것

- GPU · server · rollout 실행 금지 (collector 가 GPU 사용 중; closed-loop 실행은 lead 가 나중에 한다).
- `AG3S/ag3s/docs/` · figure · log 수정 금지. 잠금 `ifg_*.py` 수정 금지.
- 다른 세션 process 종료 금지. worker ≤ 4.

## 넘길 것

`AG3S/ag3s/docs/handoff/IFG-CL1.impl.md`: 바꾼 파일 · 새 flag · test 결과 (parity 수치) · 서버 1 요청당 추가 ms 추정 · self-filter 처리 제안 · 남은 위험.
