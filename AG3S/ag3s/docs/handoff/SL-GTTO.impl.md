# SL-GTTO — 구현: 정답 인식 (GT) + 공용 TO 판 (Panda, 기본 꺼짐)

> writer: ag3s-implementer · 2026-11-04 KST · 읽는 쪽: lead (flag 표 승인), verifier, scribe
> task: [SL-GTTO.task.md](SL-GTTO.task.md) §3 (§1 · §2 가 새 판의 내용).
> 코드 기준: 격리 worktree `/mnt/dev/AG3S_VLA/.claude/worktrees/agent-a5af2649e9e869962`, branch `worktree-agent-a5af2649e9e869962`, base `89b42459` (feature/panda-benchmark).
> worktree 는 처음에 main (`3ff415d5`) 위에 만들어져 있어서, 작업 전에 내 branch 를 `89b42459` 로 옮겼다 (`git reset --hard`, 깨끗한 tree). 커밋하지 않았다. 주 checkout 은 건드리지 않았다.
> import 확인: `AG3S.__file__` = `…/agent-a5af2649e9e869962/AG3S/__init__.py`, client 모듈 = `…/agent-a5af2649e9e869962/benchmark/runners/libero_safe_client.py` (worktree 를 `PYTHONPATH` 맨 앞, cwd = worktree).
> **GPU 는 쓰지 않았다.** lock 도 잡지 않았다. 모든 검증은 `CUDA_VISIBLE_DEVICES=''`, process 2 개 이하.

## 무엇을 했나 (평이한 요약 먼저)

**목적.** 공용 TO 가 정답 인식을 받았을 때 문제없이 작동하는지 보기 위한 Panda 판 2 개를 만들었다.
- `AG3S-commonto-gt` = 공용 TO (§1) + 정답 인식 (§2).
- `AG3S-commonto-gt-pass` = 위 + `--passthrough-when-clear`.

**정답 인식이 대신하는 것 (§2).**
1. **grounding 입력 (identity + 범위)** — 새 option `clustering.grounding_source = "oracle"` (`--grounding-source oracle`).
   - 그 요청의 code 1 pixel 에서 올린 fused 점만 attention 1 로 grounding 에 들어간다. 나머지 점은 연결에서 빠진다.
   - 그래서 target 에 닿은 이웃으로 cluster 가 자라지 않는다.
   - code 1 점이 없으면 grounding 은 기하를 못 본다 (`NO_GEOMETRY`). TargetConfirm 이 늘 하던 대로 마지막 기하를 유지하고 나이를 먹인다.
2. **미세 target-free 층** — T46e 경로 (`--target-extent oracle-visible --oracle-extent-identity gt`) 를 Panda 서버에서 허용했다.
3. **ATS** — grounding 에 쓰지 않는다. attention 은 기본 cell (`action:last:L8H2`) 로 계속 뽑히고 기록에만 남는다.

**client** 두 곳 (LIBERO · SafeLIBERO, LIBERO-Safety) 은 `--oracle-seg` 이면 요청마다 카메라별 uint8 code 영상과 meta 를 싣는다.
- 영상은 이미 렌더하는 `--capture-segmentation` geom-id 영상에서 만든다 (추가 렌더 0).
- target 은 BDDL goal 의 물체다 (`task_target.goal_roles`, gt_oracle 과 같은 규칙).
- launcher 는 GT 판이면 `--oracle-seg --capture-segmentation` 을 client 에 자동으로 붙인다.
- 서버가 요구하는데 client 가 안 보내면 접속 즉시 죽는다.

**기본은 모두 꺼짐이다.** 꺼짐이면 argv · 요청 · 응답 · 기록 키 · dry-run 출력이 base 와 같다 (test 로 보였다). 하나 예외가 있다. AG3S config dict 에 `clustering.grounding_source: "attention"` 이 한 줄 생긴다 — 아래 "verifier 가 알아야 할 것".

## 1. flag 분류표 — **lead 승인 필요 (GPU 실행 전)**

근거 열의 `serve_safe.py:N` 은 worktree `AG3S/trajopt/serve_safe.py` 의 `add_argument` 줄이다 (help 문구).
"두 판" = `AG3S-commonto-gt` 와 `-pass` (둘은 `--passthrough-when-clear` 만 다르다).

### 1a. `FINAL_CONFIG_FLAGS` (41 flag)

| flag (값) | 부류 | 두 판에서 | 왜 (근거) |
|---|---|---|---|
| `--to-fast-eval` | TO 핵심 | 유지 | FK · ESDF memo, "Bit-identical output" (`serve_safe.py:2070`). 결과를 안 바꾸는 TO 구현 |
| `--esdf-host-mirror` | TO 핵심 | 유지 | 거리장 grid 를 host 로 한 번 복사, "same values" (`:2075`) |
| `--target-exclusion ball-owned` | AG3S 인식 (추정 범위) → **애매** | 유지 | 공 안 seed 의 ownership 분할 (`:2406`). oracle 이 적용되는 frame 에서는 미세 층 분할을 oracle 이 대신한다 (T46e). oracle 근거가 없는 frame (target 표본이 아직 없음) 의 fallback 이 이 값이다. 빼면 fallback 이 `ball` (이웃까지 지움) 로 바뀌므로 base 값을 유지했다 |
| `--owned-max-dist-voxels 3` | 위와 같음 → 애매 | 유지 | ball-owned 의 unowned 거리 (`:2413`). 위와 같은 이유 |
| `--attention-split-radius-mm 12` | AG3S 인식 | **제거** | attention split 재군집 반경 (`:1993`). 제거하면 config 기본 15 mm (`config.py:248`). GT 입력은 attention 이 0/1 이라 ratio gate 가 code 1 cluster 를 가르지 않는다 |
| `--attention-split-ratio-source raw` | AG3S 인식 | **제거** | split ratio gate 가 읽는 attention (`:1996`) |
| `--target-permission-strict` | **애매** (제약 배정) | 유지 | target 예외 (tail drop · margin 면제 · sweep 행 제외) 를 권한 행에만 준다 (`:1982`). 인식이 아니라 "누가 target 예외를 받나" 의 TO 쪽 규칙이다. P0m · pc판 둘 다 쓴다 |
| `--attention-source noun_target:L8H6` | AG3S 인식 (ATS) | **제거** | ATS cell (`:2959`). 빼면 기본 `action:last:L8H2` 로 계속 뽑히지만 grounding 은 GT 입력만 본다 (§2-3 "실행해도 grounding 에 쓰지 않는다"). per-run `attention_source` 도 거절한다 |
| `--attention-split-partition` | AG3S 인식 | **제거** | 한 덩어리 trigger cluster 를 attention 으로 가른다 (`:2002`) |
| `--split-supersedes-anchor` | AG3S 인식 | **제거** | split 의 target 조각이 anchor 를 대체 (`:2005`) |
| `--wrist-track-scale 0.1` | TO 핵심 | 유지 | task §1 "손목 추종 0.1" (`:2119`) |
| `--neighbour-contact-mm 20` | 장면별 완화 | **제거** | 손가락 이웃 겹침 허용 (`:1954`). task §1 (D1) |
| `--nontarget-evidence surface` | AG3S 인식 | **제거** | ball-owned F1 근거 (`:2462`). oracle-visible 과 함께 주면 serve_safe 가 거절한다 (`:4098`) |
| `--placed-no-seed-certify` | **애매** (gate · 인증) | 유지 | PLACED 뒤 home 에서 `no_seed` frame 을 인증 (`:2837`). latch 단계 (`--subtask-from-latch`) 와 묶인 HOLD 인증 규칙이다. GT 입력에서 `no_seed` 는 거의 안 나온다 (code 1 이 보이면 seed 가 있다) |
| `--kept-geometry-follow` | **애매** (TargetConfirm 기억) | 유지 | 쥐기 직전 · 닫는 중 유지한 기하를 이번 frame 의 지지면 아닌 점에 맞춰 옮긴다 (`:2887`). TargetConfirm 은 그대로 쓴다 (§2-1). GT 입력에서는 "쓸 수 있는 점" = code 1 점뿐이라 GT target 만 따라간다 |
| `--roi-carve` | **애매** (관측 층) | 유지 | 물체 · 목적지 · 손 경로 상자 안 TSDF 자유 공간 갱신 (`:2443`). depth 관측의 유지보수라 어느 인식을 끼워도 같은 관측을 쓴다 (§2 이유) |
| `--roi-carve-footprint` · `-pad-px 0` · `-robot ignore` · `-device cuda` | 위와 같음 | 유지 | `:2451` · `:2453` · `:2455` · `:2457` |
| `--blas-threads 1` | TO 핵심 (실행 환경) | 유지 | OpenBLAS thread 수, "bit-identical commands" (`:2079`) |
| `--to-multi-seed 16` | TO (E0 아님) | **제거** | multi-seed TrajOpt (`:2347`). task §1 "multi-seed 없음" |
| `--to-gpu-seeds curobo` | TO (E0 아님) | **제거** | `:2353` |
| `--to-gpu-seeds-check first-pass` | TO (E0 아님) | **제거** | `:2359` |
| `--palm-shell on` | TO 핵심 (제약 로봇 모형) | 유지 | 손바닥 바깥 껍질 구 (`:2726`). 로봇 형상이다. P0m · pc판 둘 다 쓴다 |
| `--palm-shell-margin 5` | TO 핵심 | 유지 | 껍질 행만의 obstacle 여유 (`:2733`) |
| `--hsr6-attach-assist` | HSR6 | **제거** | `:2209`. task §1 "HSR6 끔" |
| `--hsr6-held-gate` | HSR6 | **제거** | `:2212` |
| `--hsr6-target-exempt` | HSR6 | **제거** | `:2217` |
| `--hsr6-objective` | HSR6 | **제거** | `:2221` |
| `--neighbour-palm-contact-mm 10` | 장면별 완화 | **제거** | 손바닥 이웃 겹침 허용 (`:1959`). task §1 (D1) |
| `--hsr6-objective-scope nbc` | HSR6 | **제거** | `:2229` |
| `--continuity-after-grasp` | TO 핵심 | 유지 | task §1 "연속성 항은 쥔 뒤에만" (`:2140`) |
| `--deflection-rate-after-grasp` | TO 핵심 | 유지 | task §1 "꺾임 항은 쥔 뒤에만" (`:2264`) |

- 제거 목록 = `profile.GT_REMOVED_FLAGS` (16 flag). 나머지 `FINAL_CONFIG_FLAGS` 는 값까지 그대로다.
- test (`test_versions_are_the_classified_final_flags`) 가 판의 token 을 하나씩 적어 고정한다.

### 1b. `P0M_PANDA_FIX_FLAGS` · `P0M_SUBTASK_FLAGS` (추가)

| flag (값) | 부류 | 두 판에서 | 근거 |
|---|---|---|---|
| `--reach-at-capture` | 쥐기 처리 (pb1) | 추가 | reach 관문을 그 capture 의 상태와 짝짓는다 (`:3045`) |
| `--links arms` | 쥐기 처리 (팔 link 제약) | 추가 | 팔 link 를 제약에 넣는다 (`:1926`). base argv 의 `--links gripper` 를 덮는다 (P0m-pandafix 와 같음) |
| `--held-hand-exempt` | 쥐기 처리 | 추가 | HELD 동안 쥔 손의 쥔 물체 행 제외 (`:3014`) |
| `--latch-phase` | 쥐기 처리 (latch) | 추가 | 접촉 phase 를 latch 에서 (`:3038`) |
| `--grasp-zone-mm 10` | 쥐기 처리 (grasp zone) | 추가 | LATCHED / CLOSING 의 10 mm 근접 허용 (`:3018`) |
| `--grasp-reach-sticky` | 쥐기 처리 (pb6) | 추가 | `:3024` |
| `--grasp-drop-from-open` | 쥐기 처리 (pb6) | 추가 | `:3029` |
| `--closing-hand-exempt` | 쥐기 처리 (pb7) | 추가 | `:3034` |
| `--subtask-from-latch` | 쥐기 처리 (latch → gate label) | 추가 | 3 단계 gate label 을 latch phase 에서 (`:2800`) |

### 1c. 정답 인식 · 선택 (추가)

| flag | 두 판에서 | 근거 |
|---|---|---|
| `--target-extent oracle-visible` | 추가 | T46e 미세 target-free 층 (`:2491`) |
| `--oracle-extent-identity gt` | 추가 | 지울 물체 = GT target (`:2506`) |
| `--grounding-source oracle` | 추가 | 새 flag (SL-GTTO, `:2516`) |
| `--oracle-max-dist-voxels` | 안 줌 (config 6.0) | `:2511` |
| `--passthrough-when-clear` | `-pass` 만 | T88 (`:2365`) |

### 1d. 참고 — base argv (V11AB 서버 flag, 모든 판 공통) 는 바꾸지 않았다

`start_server.sh` 의 V11AB flag 는 모든 판이 같이 쓰는 base 다. 이번 분류 범위 밖이라 그대로 두었다.
- TO · 형상: `--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --target-field-policy exclude-authorized --esdf-margin 0 --obstacle-margin 0.010 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20`.
- 쥐기 · 기억: `--held-body-cover --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3`.
- TargetConfirm 의 identity 규칙: `--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60`.
  - 이 셋은 TargetConfirm 의 기억 · admissibility 규칙이다. task §2-1 "TargetConfirm 은 그대로" 에 따라 유지했다.
- `--hsr6-probe` 는 주지 않는다 (P0M_VERSIONS 처럼). per-run `--attention-source` 도 거절한다.

## 2. grounding 교체 — 고른 규칙

| 단계 | 규칙 | 이유 |
|---|---|---|
| code 1 점 | fused 점 (`FusedPointCloud` CSR block) 의 관측 (카메라 + pixel `(u, v)`) 중 그 요청의 code 영상 위에 있는 것이 **모두** code 1 이고 하나 이상이면 code 1 점 | 한 voxel 에 이웃 · table · 로봇 pixel 이 섞이면 target 이 아니다 → 이웃 경계 voxel 이 빠진다 |
| grounding cloud | 같은 cloud (같은 index), attention = raw attention = code 1 이면 1 아니면 0 | `point_indices` 를 읽는 하류 (camera provenance, `_nontarget_points`, 기록) 가 그대로다 |
| exclude mask | code 1 이 아닌 점 전부. **지지면 mask 는 grounding 에 쓰지 않는다** | GT 가 target 이라고 한 점이다. 지지면 맞춤이 평평한 target (접시 · 책) 을 먹어도 grounding 은 GT 를 따른다. 기록 `n_code1_on_support` |
| 없음 | segmentation 없음 · code 1 점 없음 · fusion 없음 → 빈 cloud | grounding = `NO_GEOMETRY`, TargetConfirm 이 늘 하던 대로 유지 · 나이 먹임 |
| segmentation | 읽기만 하고 소비하지 않는다 | 같은 frame 의 8b 단계 (`_oracle_update`) 가 memory 에 접는다 |
| 기록 | `metrics["grounding_source"]` = `{source, n_cloud, n_code1, n_mixed, n_unlabelled, n_code1_on_support, cameras, target, why, status, n_target_points}` | on 일 때만 키. Panda 응답 `panda/gt_perception` 으로 client `*_calls.jsonl` 에 남는다 |

## 3. code 영상 (client) — 고른 규칙

| code | LIBERO body |
|---:|---|
| 0 | geom 없음 (-1) · 로봇 (`robot0_*` · `gripper0_*` · `mount0_*` body, `eef_marker`) |
| 1 | BDDL goal 의 대상 물체 (`task_target.goal_roles`, gt_oracle 규칙) — `<name>_*` body 전부 |
| 2 | 나머지 정적 body — table · fixture (cabinet, stove, …) · 벽 · 바닥 · world |
| 3, 4, … | target 이 아닌 BDDL object (BDDL 순서) → 그다음 BDDL 에 없는 free-joint root body (body id 순). 자식 body 는 root 의 code |

- **affordance fallback.** LIBERO-Safety affordance 의 goal 은 `(CheckGripperContactPart knife_1 (…))` 라서 `goal_roles` 가 실패한다 (L0 5 task 전부).
  - 그래서 client 만 fallback 을 쓴다: "obj_of_interest 에 작용하는 첫 goal predicate 의 첫 인자 (인자 수 무관)" → `knife_1`.
  - SL3 / gt_oracle 의 `task_target.py` 는 바꾸지 않았다. 기록 `oracle_seg.rule` 에 어느 규칙인지 남는다.
- **다중 target task** (`put_both_moka_pots…` · `put_both_the_alphabet_soup…`) 는 첫 물체만 code 1 이고 둘째는 장애물 code 다.
  - SL-GTTO 장면 (obstacle_avoidance t0) 에는 없다. 판정 요청 2 참고.
- segmentation 은 `env.step` 반환 순간에 렌더된다. depth 는 image 순간이라 한 substep 차이가 있다 (gt_oracle 과 같음, 보고만).

## 바뀐 파일

### development 소유 core (`AG3S/ag3s/**`, `AG3S/trajopt/**`) — development 이식 요청용

줄은 worktree 기준이다. 모두 `# SL-GTTO` 표시가 붙어 있다.

| 파일:줄 | 무엇이 |
|---|---|
| `AG3S/ag3s/config.py:421-431` | `ClusteringConfig.grounding_source: str = "attention"` (+ 설명) |
| `AG3S/ag3s/config.py:493` | `GROUNDING_SOURCES = ("attention", "oracle")` |
| `AG3S/ag3s/config.py:639-641` | validate: 목록 밖 값 거절 |
| `AG3S/ag3s/runtime/pipeline.py:224-226` | `self._grounding_frame = None` (init) |
| `AG3S/ag3s/runtime/pipeline.py:482` | `reset()` 이 `_grounding_frame` 을 비운다 |
| `AG3S/ag3s/runtime/pipeline.py:785-851` | 새 method `_oracle_grounding_input(attention_cloud, support_mask, fusion)` (§2 규칙) |
| `AG3S/ag3s/runtime/pipeline.py:2672-2699` | step 5: `grounding_source == "oracle"` 이면 `ground_target` 에 GT cloud · mask (아니면 예전 객체 그대로), provenance 에 같은 cloud, 기록에 status · 점 수 |
| `AG3S/ag3s/runtime/pipeline.py:3084-3085` | `metrics["grounding_source"]` (on 일 때만) |
| `AG3S/trajopt/serve_safe.py:1102-1103` | `t43t_identity_section` 이 `grounding_source` 를 clustering 키로 (attention 이면 키 없음) |
| `AG3S/trajopt/serve_safe.py:2515-2523` | `--grounding-source {attention,oracle}` (기본 attention) |
| `AG3S/trajopt/serve_safe.py:3956-3963` | 거절: oracle 인데 `--target-extent oracle-visible` 아님, 또는 `--no-safe` / `--no-perception` |

`wire.py` · `client.py` · `safe_policy.py` · `oracle_segmentation.py` · `curobo_builder.py` 는 바꾸지 않았다.
- SafePolicy 는 이미 oracle-visible 이면 metadata `oracle_segmentation_required` 를 싣고 요청의 code 영상을 `set_oracle_segmentation` 으로 넘긴다 (`safe_policy.py:617-621`, `:1275-1278`).

### benchmark 소유

| 파일:줄 | 무엇이 |
|---|---|
| `AG3S/adapters/panda/profile.py:466-502` (새 블록, 연속) | `GT_REMOVED_FLAGS` · `GT_PERCEPTION_FLAGS` · `COMMONTO_GT_FLAGS` · `GT_VERSIONS` · 두 판 등록 · `version_requires_oracle_segmentation` |
| `…/profile.py:519-520` · `:534-538` · `:592-595` | `is_final_version` 에 GT 판, `version_server_flags` 는 probe · attention cell 거절, `version_flag_values` 는 GT 전용 dest 를 켜졌을 때만 기록 (다른 판의 기록 키 그대로) |
| `AG3S/adapters/panda/server.py:403` · `:453-458` | `oracle_extent_identity` · `oracle_max_dist_voxels` 를 unported → ported, `grounding_source` ported, `PORTED_TARGET_EXTENTS` |
| `…/server.py:484-489` | target-extent 허용에 `oracle-visible` |
| `…/server.py:345-380` · `:301-303` | `gt_perception_call_record` → 응답 `panda/gt_perception` (on 일 때만) |
| `…/server.py:841-862` · `:932-933` | `gt_perception_record` → config dump `final_config.gt_perception` (on 일 때만) |
| `benchmark/adapters/libero_panda/oracle_codes.py` (새, py3.8) | code 표 · `code_image` · `OracleCoder` · BDDL target 규칙 (+ affordance fallback) |
| `benchmark/runners/libero_safe_client.py:194-249` | `--oracle-seg` helper (`add_oracle_arguments` · `check_oracle_arguments` · `check_server_oracle` · `oracle_coder_for` · `oracle_record`) |
| `…/libero_safe_client.py:264-268` · `:290-291` · `:326-328` · `:406-413` | `SafeClient(oracle=None)`: 첫 frame 에 code 표, 요청에 code 영상, row `oracle_seg` (code 1 pixel 수) · `gt_perception` (on 일 때만) |
| `…/libero_safe_client.py:467` · `:475` · `:513-521` · `:539-540` | parser, 인자 검사, 접속 검사, episode 별 coder, jsonl `oracle_seg` |
| `benchmark/runners/libero_safety_client.py:580` · `:597` · `:615-622` · `:629-632` | 같은 것 (LIBERO-Safety) |
| `benchmark/runners/launch_libero_paired.py:128-157` · `:305-315` · `:479-481` | `oracle_client_argv` · `check_oracle_client_argv`, `run_mode` 이 GT 서버의 client 에만 두 flag + settings `oracle_segmentation`, dry-run 에 `client_argv_by_mode` 한 줄 (GT 판일 때만) |
| `benchmark/runners/launch_libero_safety_paired.py:198-200` | dry-run 한 줄 (run 은 `lp.run_mode` 를 타므로 자동) |

### test

| 파일 | 무엇이 |
|---|---|
| `tests/ag3s/test_sl_gtto_oracle_grounding.py` (새, 14) | config · CLI, code 1 점 규칙 (섞인 voxel 제외), 닿은 이웃 box 로 안 자람 (대조: 같은 seed 로 attention 경로는 자람), 없음 → `NO_GEOMETRY` + 유지 · 나이, pipeline (`process_multi`, 3 카메라 + 팔): attention 이 이웃을 가리켜도 oracle 은 GT 구, 꺼짐이면 `ground_target` 이 같은 객체를 받고 metrics 동일 |
| `tests/benchmark/test_sl_gtto_gt_perception.py` (새, 32) | 판 argv (token 고정) · probe 거절 · read-back · 다른 판 = base commit, 서버 ported · CPU build · metadata · dump · per-call record, launcher (run_mode · dry-run · LIBERO-Safety wrapper, 기본은 base 와 byte 동일), client (code 표 · wire 왕복 · BDDL 규칙 · 접속 검사 · flag 없으면 요청 동일), 실제 장면 5 개 |
| `tests/benchmark/libero_gtto_scene_checks.py` (새) | 실제 장면 검사 (benchmark interpreter 에서 subprocess) |
| `tests/ag3s/test_t46e_oracle_extent.py:716-717` | CL3 기록이 없는 host 에서 빈 parametrize 의 `ids` 가 수집 오류를 내던 것 (기존 문제) — id 함수만 고쳤다 |
| `tests/benchmark/test_sl7_panda_p0m.py:230-232` · `:252` | SL7 이 "아직 거절" 로 고정하던 oracle 두 dest · `oracle-visible` 을 뺐다 (이번에 이식) |
| `tests/benchmark/test_sl6h_panda_placement_contact.py:59-61` | HEAD 대비 새 판 목록에 `GT_VERSIONS` |
| `tests/trajopt/_sl5a_golden.py:411-413` | `PB_DEFAULT_OFF` 에 `grounding_source: "attention"` (SL8b 와 같은 방식) |

## 단위 검증

```bash
# wt.sh = cd <worktree>; PYTHONPATH=<worktree>; CUDA_VISIBLE_DEVICES=''; JAX_PLATFORMS=cpu; MUJOCO_GL=osmesa;
#         AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage; PYTHONDONTWRITEBYTECODE=1; BLAS threads 2; nice 19
# 실제 장면: GTTO_VENV_ROOT=/mnt/dev/AG3S_VLA, LIBERO / SAFELIBERO / LIBERO_SAFETY _CONFIG_PATH = 주 checkout 의 .runtime/*,
#           GTTO_LIBERO_SAFETY_ROOT=/mnt/dev/AG3S_VLA/benchmark/LIBERO-Safety (worktree 에 그 checkout 이 없어서)
wt.sh .venv-ag3s/bin/python -m pytest -q tests/ag3s/test_sl_gtto_oracle_grounding.py tests/benchmark/test_sl_gtto_gt_perception.py
#   46 passed (실제 장면 5 개 포함, skip 0)
wt.sh .venv-ag3s/bin/python -m pytest -q tests/ag3s/test_t46e_oracle_extent.py tests/benchmark/test_sl8b_identity_versions.py tests/benchmark/test_sl7_panda_p0m.py
#   70 passed, 12 skipped (CL3 · C1A storage 기록 없음, GPU, stored rollout 없음)
wt.sh .venv-ag3s/bin/python -m pytest -q tests/benchmark          # base 사본과 비교
scratchpad/chunked.sh wt.sh …   # tests/ag3s · tests/trajopt · tests/test_*.py 를 12 file 씩 순차 (한 번에 process 하나)
# (처음에 worktree · base 두 전체 run 을 동시에 돌렸다가 pod 의 cgroup 에서 base run 이 OOM kill 됐다 (oom_kill 1).
#  내 run 만 죽었고 남은 내 run 도 멈췄다. 그 뒤로는 process 하나씩, chunk 로 나눠 RSS 를 줄였다.)
wt.sh .venv-ag3s/bin/python -m tests.trajopt._sl5a_golden --check-serve-safe … / --check …          # base 사본과 비교
```

결과 (worktree 대 base 사본 — `git archive 89b42459` 를 scratch 에 풀고 같은 환경으로 돌림):

| 묶음 | worktree | base 사본 | 새 실패 |
|---|---|---|---|
| `tests/benchmark` | 303 passed · 11 failed · 4 errors · 72 skipped | 270 passed · 13 failed · 4 errors · 104 skipped | **0** (worktree 실패 ⊂ base 실패) |
| `tests/ag3s` + `tests/trajopt` + `tests/test_*.py` (12 file 씩 순차, 273 file) | 4602 passed · 2 failed · 3 collection errors · 295 skipped | 실패 · 오류 5 개 모두 base 에서도 실패 (`test_t61_select::test_heldout_registry_matches_rule`, `test_run_artifacts::test_v11_server_records_actual_flags_without_gpu`, legacy fixture 없는 `test_t41b` · `test_t41c` · `test_t43q` 수집 오류) | **0** |
| SL5a golden `--check-serve-safe` | 24 차이 | 24 차이 | 0 (`grounding_source` 를 `PB_DEFAULT_OFF` 에 넣은 뒤) |
| SL5a golden `--check` (응답 npz) | 12 차이 | 12 차이 | 0 |

- base 와 같은 실패의 원인 (이 container · worktree 환경):
  - `.venv-ag3s` 에 `openpi_client` 없음 (8).
  - test 가 `ROOT/.venv-libero` · `ROOT/.venv-ag3s` 를 찾는데 worktree 에는 venv 가 없다.
  - `benchmark/LIBERO/requirements.txt` · LIBERO-Safety `robosuite-1.4` 가 worktree 에 없다 (submodule).
  - golden 은 base 에서도 24 / 12 차이다: T96 · T97 merge 의 새 arg 7 종과 기록 경로 문자열 (`/mnt/dev/AG3S_VLA-benchmark/…` 대 이 checkout).
- base 사본은 git 이 아니라서 HEAD 비교 test 가 skip 된다 (skip 104 대 72). 그 test 들은 worktree 에서는 돌았고 통과했다 (`test_sl6h…keeps_its_argv` 는 새 판 목록만 더해서 통과).
- worktree 환경 caveat:
  - submodule 디렉토리 (`third_party/curobo`, `benchmark/LIBERO`, `benchmark/SafeLIBERO`, `benchmark/LIBERO-Safety`) 가 비어 있었다.
  - test 동안만 그 안에 주 checkout 내용을 가리키는 **자식 symlink** 를 두었다 (`…/curobo/curobo`, `…/LIBERO/libero`, `…/SafeLIBERO/{safelibero,main}`, `…/LIBERO-Safety/libero`). 끝나고 지웠다.
  - git 은 submodule 경로라 이것을 보지 않는다. bytecode 는 쓰지 않았다 (`PYTHONDONTWRITEBYTECODE=1`). 주 checkout 의 파일은 읽기만 했다.

### 실제 장면 (CPU, 각 benchmark interpreter, init 0, wait 뒤 1 frame)

| 장면 | python | target (규칙) | head code 1 px | wrist code 1 px | = gt_oracle mask | 투영 pixel code (뒤집으면) | 역투영 중앙 거리 (뒤집으면) |
|---|---|---|---:|---:|---|---|---|
| LIBERO spatial t0 | 3.8.20 | akita_black_bowl_1 (goal_roles) | 769 | 2281 | 둘 다 같음 | 1 (2) · 1 (2) | 55 mm (298) · 45 mm (247) |
| SafeLIBERO spatial LI t0 | 3.8.20 | akita_black_bowl_1 (goal_roles) | 769 | 2288 | 둘 다 같음 | 1 (2) · 1 (2) | 55 mm (298) · 45 mm |
| LIBERO-Safety affordance L0 t0 | 3.10.22 | knife_1 (**fallback**) | 333 | 1262 | 둘 다 같음 | 1 (1) · 1 (1) | 61 mm (70) · 51 mm (106) |
| LIBERO-Safety human_safety L0 t0 | 3.10.22 | akita_black_bowl_1 (goal_roles) | 1175 | 0 (안 보임) | 둘 다 같음 | 1 (2) · — | 51 mm (2222) |
| LIBERO-Safety obstacle_avoidance L0 t0 | 3.10.22 | akita_black_bowl_1 (goal_roles) | 542 | 2430 | 둘 다 같음 | 1 (2) · 1 (2) | 58 mm (331) · 45 mm (211) |

- 모든 장면에서 로봇 pixel 은 0, table 은 2 였다. code 영상 shape = depth shape (256 × 256, uint8) 이었다.
- SafeLIBERO legend: target 1, 둘째 bowl 3, cookies 4, ramekin 5, plate 6, 장애물 7–12 (moka pot … yellow book), box_base 13.
- 역투영 중앙 거리는 body 원점 (bowl 바닥 쪽) 까지의 거리라 0 이 아니다. 뒤집은 영상은 수백 mm 떨어진다.
- affordance 칼은 화면 가운데 줄 근처에 길게 누워 있어서, 뒤집어도 투영 pixel 이 칼 위에 놓인다. 그래도 역투영 거리는 뒤집은 쪽이 멀다.

## verifier 가 알아야 할 것

- **새 flag**
  - 서버: `--grounding-source {attention,oracle}` (기본 attention). oracle 은 `--target-extent oracle-visible` 과 함께만이다.
  - client: `--oracle-seg` (`--capture-segmentation` 필요).
  - launcher: GT 판이면 두 flag 를 자동으로 붙인다.
  - 기본값 변경은 없다.
- **옛 기록과 호환.** 꺼짐이면 argv · 요청 · 응답 · jsonl row · metrics · stats · server metadata · manifest settings · dry-run 출력이 base 와 같다.
  - 하나 예외: `AG3SConfig.to_dict()` (manifest 의 `panda_config` 등) 에 `clustering.grounding_source: "attention"` 한 키가 생긴다. SL8b · T81 이 config 에 기본 꺼짐 field 를 더할 때와 같은 방식이다.
- **smoke 확인 키 (task §5-7)** — client `episodes/<stem>_calls.jsonl`:
  - row `gt_perception.grounding.source == "oracle"`, `gt_perception.grounding.n_code1 > 0` 인 call 이 있어야 한다.
  - `gt_perception.oracle.applied` (미세 층 oracle 적용) · `why` 를 본다.
  - row `oracle_seg` = 카메라별 code 1 pixel 수다.
  - HSR6 · ATS grounding 미사용은 `panda_server_config.json` `final_config.hsr6 = null`, `final_config.attention` (기본 cell), `final_config.gt_perception` 에서 본다.
  - `--record-constraints` 를 켜면 chunk `summary_json.esdf_stats.target_free.oracle` 에도 남는다 (T46e 와 같음).
- **비용.** T46e 측정 기준으로 oracle memory 갱신 + 분할에 서버 CPU 가 chunk 당 ~200 ms 더 든다. client 는 code 영상 2 × 64 kB/요청이고 렌더 추가는 0 이다 (segmentation 렌더는 `--capture-segmentation` 이 이미 한다).
  - `--capture-segmentation` 은 episode `_geom.npz` 에 `segmentation_geom/*` 도 쌓는다 (int16 256² × frame 수).
- **재생산이 필요한 산출물:** 없음.

## 판정이 필요한 것 (lead)

1. **§1 flag 표 승인.** 애매로 남기고 base 값을 유지한 것:
   - `--target-exclusion ball-owned` / `--owned-max-dist-voxels 3` — oracle 근거 없는 frame 의 fallback.
   - `--target-permission-strict`, `--placed-no-seed-certify`, `--kept-geometry-follow`, `--roi-carve*`.
   - base argv 의 TargetConfirm 규칙 3 개 (`--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60`).
   - attention split 은 flag 를 빼도 config 기본 (15 mm, T31b) 으로 켜져 있다. GT 입력은 0/1 attention 이라 영향이 없을 것으로 보지만 측정하지 않았다.
2. **target 규칙.**
   - affordance 는 client fallback (`CheckGripperContactPart` 의 물체) 이다.
   - 다중 target task (`put_both_*`) 의 둘째 물체는 장애물 code 로 남는다. SL-GTTO 장면에는 없지만 장면을 넓히면 정해야 한다.
3. **GT grounding 은 지지면 mask 를 무시한다** (§2). attention 경로와 같은 지지면 제외를 원하면 한 줄로 바꿀 수 있다.
4. **`--grounding-source oracle` 은 `oracle-visible` 을 요구한다.** segmentation 을 저장하는 것이 그 extent 뿐이라서다. "GT grounding × 추정 범위" arm 이 필요하면 `set_oracle_segmentation` · SafePolicy 의 저장 조건을 넓혀야 한다 (core 수정).
5. **VLA 단독 짝.** launcher 는 vla_only client 에는 `--oracle-seg --capture-segmentation` 을 붙이지 않는다 (VLA 단독 기록 = 이전과 같은 argv). segmentation 렌더는 정책 영상에 영향이 없다고 보지만 확인하지 않았다.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->

- 꺼짐: 차이 0 (위 표 그대로).
- smoke: GT 판의 call 대부분에서 `grounding.n_code1` 이 수백 이상일 것이다 (head 에서 bowl 769 px 정도).
  - 손이 bowl 을 가리는 grasp 직전 call 에서는 0 일 수 있고, 그때 `status = no_geometry` 와 TargetConfirm 의 `occluded` 기하가 남을 것이다.
  - `oracle.applied` 는 pre-grasp 에서 대부분 True 일 것이다.
- closed loop (재지 않았다): SafeLIBERO 에서 grounding 이 장애물 · 둘째 bowl 로 가는 일은 없을 것이다. 결과 차이는 TO · 변환 · latch 쪽에서 나올 것이다.
