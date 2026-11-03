# SL7 — Panda P0m 판: development 최종 구성(P0m)을 Panda 서버에 이식

> writer: lead (benchmark checkout, 세션 benchmark-6e/44), 2026-11-03 KST.
> 사용자 지시: development d2 세션 전달(2026-11-03 18:4x KST) + 이 checkout 사용자 확인(같은 날) — "변경된 최종 구성(P0m)을 Panda로 LIBERO · SafeLIBERO · LIBERO-Safety에서 시험". 이 task는 그 전 단계인 **서버 이식 + CPU 테스트**만 다룬다(GPU · rollout 없음).
> 정본: development main `80898def`의 `AG3S/ag3s/docs/handoff/P0_FINAL_CONFIG_FOR_BENCHMARK_20261103_KO.md` (이 checkout에는 merge `93e2ae8d` 시점 판이 있음; 갱신본은 `git show devlocal/main:AG3S/ag3s/docs/handoff/P0_FINAL_CONFIG_FOR_BENCHMARK_20261103_KO.md`). merge 기록: `doc/active/migration/DEV_MAIN_MERGE_20261103_REPORT_KO.md`.

## 0. 확정된 판정 (바꾸지 않는다)

| # | 판정 |
|---|---|
| 0.1 | Panda P0m 판 = P0 표면 추출 + `--subtask-from-latch`(T92). HSR6 규칙 4개 · `--hsr5` · G1 · A4 · TrajOpt 여러 초기값(`--to-multi-seed` · `--to-gpu-seeds*`) **끔**(E0) |
| 0.2 | attention 출처는 **Panda head `noun_target:L8H6`** (L10H7 아님) |
| 0.3 | **D1:** `--neighbour-contact-mm` · `--neighbour-palm-contact-mm` **끔**. 접촉 허용은 target(쥘 때) · 목적지(놓을 때)만 |
| 0.4 | 판 2개를 등록한다: **`AG3S-final-panda-p0m`** (hull 채우기 `--surface-hull-fill` + 접촉 여유 `--surface-contact-allowance-mm 10` 포함 = development P0m 그대로) 와 **`AG3S-final-panda-p0m-nohull`** (그 둘만 뺌, development T92b 대조) |
| 0.5 | `--nontarget-evidence surface` · `--surface-split` · `--surface-crease-cut` · `--surface-memory-target-sticky` · `--grasp-corridor-*` · `--passthrough-when-clear` 금지(P0m에 없음) |
| 0.6 | 그 밖의 P0 값(손가락 덮개, 손바닥 껍질, ROI carve, 장애물 여유, 손목 추종 약화 0.1, 쥔 뒤 연속성 · 꺾임 변화율, hold-follow 등)은 **Panda 판의 기존 값을 유지**하고 P0와 다른 것만 표로 기록한다(값 변경은 사용자 판정 사항) |
| 0.7 | `PANDA_HSR6_PROBE`(`AG3S/adapters/panda/profile.py:334`)는 `/mnt/dev/AG3S_VLA-storage` 고정 경로 대신 `AG3S_STORAGE_ROOT` 기준으로 찾는다(SHA-256 검사 유지). 기존 판의 실제 경로 · 동작은 같아야 한다 |
| 0.8 | 새 기능은 기본 꺼짐. **V11AB · V11AB-pb1~pb7 · AG3S-final-panda 1판~pc판의 서버 argv · SafePolicy 인자 · AG3S config · `info`는 bit 단위로 그대로**. RB-Y1 서버 경로도 그대로(`tests.trajopt._sl5a_golden` 통과) |

## 1. 해야 할 것

1. **profile:** `AG3S/adapters/panda/profile.py`에 두 판을 등록(이름 0.4). flag는 `FINAL_CONFIG_FLAGS`에서 0.1 · 0.3 · 0.5에 해당하는 것을 빼고 P0 표면 추출 flag(`--target-extent attention-surface --surface-target-check overlap --surface-keep-last --surface-grasp-hold --surface-no-ball --attention-surface-device cuda` [+ `--surface-hull-fill --surface-contact-allowance-mm 10`]) 와 `--subtask-from-latch`를 더한다. 판 정의 옆에 무엇을 바꿨는지 주석. `--hsr6-probe`는 이 판들에서 요구하지 않는다(HSR6 끔).
2. **Panda 서버 연결** (`AG3S/adapters/panda/server.py` · `profile.py`): Panda 서버는 AG3S config와 SafePolicy를 직접 만든다(`build_safe_server`, `profile.build_panda_ag3s(kw=v11ab_build_kwargs(args))`). 위 flag들이 **실제로 적용되도록** 연결한다:
   - 표면 추출 flag → AG3S config(`esdf.target_extent = attention_surface`, `surface_*`, `attention_surface_device`, hull fill, contact allowance). RB-Y1은 `serve_safe` 의 esdf override(대략 `serve_safe.py:1083-1112`)와 `ag3s_from_args`로 한다 — 같은 해석을 재사용할 것.
   - `--subtask-from-latch` → SafePolicy 인자(`serve_safe.subtask_from_latch_kwargs(args)`), 시작 announce. **Panda 경로에서 3단계 gate가 실제로 동작하는지** 확인(지금 Panda AttentionPolicy는 `subtask_probe=None` — latch label이 gate에 들어가는 경로와 `target_grounding` 의 `subtask_gate` 설정을 확인하고, 동작하지 않으면 연결).
   - 연결한 dest만 `UNPORTED_FINAL_CONFIG_DESTS`에서 빼 `PORTED_FINAL_CONFIG_DESTS`로 옮긴다: `target_extent`, `surface_target_check`, `surface_keep_last`, `surface_grasp_hold`, `surface_no_ball`, `attention_surface_device`, `surface_hull_fill`, `surface_contact_allowance_mm`, `subtask_from_latch`. 나머지(`surface_split`, crease, memory sticky, vacancy, grasp set/corridor, hsr5, gt stage, objective window, support offset, passthrough, phase machine, fm_guide*)는 거절 유지. `tests/benchmark/test_merge_dev_main_20261103_panda.py`의 목록도 맞춘다.
3. **Panda head 허용:** `AG3S/trajopt/attention_source.py` `seed_map_source_ok`(표면 추출 · confidence gate가 요구하는 seed 지도)는 지금 `noun_target:L10H7`(+ hidden probe)만 받는다. **Panda 서버에서만** Panda head(`noun_target:L8H6`)를 받도록 한다. RB-Y1 서버의 L10H7 고정은 그대로 둔다(예: Panda adapter가 허용 label을 넘기는 방식 — 설계는 implementer가 정하고 impl.md에 이유를 적는다).
4. **Panda에서 다른 점 확인(코드 읽기 + CPU 테스트):**
   - 표면 추출 코드(`AG3S/ag3s/runtime/attention_surface*.py`, `pipeline._attention_surface_update`)의 카메라 이름 · 순서(`CAMERA_ORDER`), robot mask 방식(`ARM_RENDER` 렌더 mask vs `ARM_SPHERES` 구), depth 입력이 Panda 관측(`perception_replay.build_observations` / Panda adapter 카메라 alias)에서 그대로 동작하는가. 다르면 Panda 쪽 입력을 맞추고 impl.md에 적는다.
   - hull 채우기의 손바닥 거리 창(80 mm)과 손바닥 site: Panda hold site(`hand_geometry.PANDA_HOLD_SITE_OFFSET_M`)가 쓰이는가.
   - latch 문턱은 Panda 보정(`panda_latch_config`, `gripper_calibration.json`) 그대로.
5. **profile 경로(0.7)** 수정.
6. **테스트 (CPU, `tests/benchmark/test_sl7_panda_p0m.py` 신설):**
   - 두 판의 flag = 위 정의(1판 대비 뺀 것 · 더한 것을 명시적으로 검사), launcher argv, read-back(`match_ag3s_version`).
   - CPU build(`gpu_ok` fixture 방식, `test_sl6a_panda_final.py` 참고): AG3S config에 `target_extent == "attention_surface"` 와 surface 키, hull fill · allowance가 nohull 판에서는 꺼짐; SafePolicy가 latch gate를 받음; `hsr6 is None`; multi-seed 끔; neighbour contact 깊이 0.
   - 서버 refusal: 연결하지 않은 option은 여전히 거절; P0m 판은 `check_unported_flags` 통과(probe 없이).
   - 표면 추출 한 프레임 smoke: 저장된 SafeLIBERO 기록 한 프레임(예: `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only/episodes/safelibero_spatial_t0_e0*.npz` + noun rows `.../20261102T110500_*_safelibero-seg-dev-noun-rows-20261102/noun_rows/`)로 P0m AG3S의 `process_multi_debug`가 표면 영역을 만들고 `exclusion_trace`가 비어 있지 않은지(CPU 장치로; `experiments/segmentation/eval_real_ats_safelibero.py`가 같은 재생을 한다). 파일이 없으면 skip.
   - 회귀: `tests/benchmark/test_sl6a_panda_final.py`, `test_sl6h_panda_placement_contact.py`, `test_panda_sl5j_recorder.py`, `test_panda_sl5a.py`(RB-Y1 golden), `tests/trajopt/test_t92_subtask_from_latch.py`, `test_sl6a_injections.py` 통과. 기존 판 build 결과 bit 동일.
7. **impl.md** `AG3S/ag3s/docs/handoff/SL7.impl.md`: 바꾼 파일:줄, 연결 방식, Panda와 RB-Y1이 다른 점(카메라 · mask · 손바닥 site · gate), P0와 Panda 판 값이 다른 항목 표(0.6), 테스트 결과(passed 수), 남은 위험.

## 2. 하지 말 것

- GPU 실행, rollout, storage에 새 run 쓰기(테스트 tmp 제외).
- 0.6의 값 변경, 다른 판(V11AB 계열 · 1판~pc판) 동작 변경, RB-Y1 경로 동작 변경.
- `doc/`, `AG3S/ag3s/docs/*.md`(impl.md 제외), figures 수정. commit은 lead가 한다(하지 말 것).
- 공용 pod 규칙: CPU worker ≤ 2, 큰 pytest 묶음을 한 번에 돌리지 말고 파일별로(이 pod은 전체 실행 시 OOM 이력이 있다).
