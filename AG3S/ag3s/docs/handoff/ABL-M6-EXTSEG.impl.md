# ABL-M6-EXTSEG — 구현 (본문 ablation 행 6 "AG3S with external segmenter")

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · 2026-11-16 · branch `wip/aud-next`
> base: `wip/sl-aud` 를 merge (`280f53dc`; 충돌은 TAB3 code 행 등록뿐 — 아래 "merge" 절)
> 정의 (사용자, 계획서 §9.45): target 표면 = instruction 명사로 prompt 한 Grounded-SAM-2 mask 를 같은 depth 로 3D 로 올린 것.
> 접촉 규칙 · 물체 부착은 AG3S full (행 5 = X-fwd = `pf.SOTA_FINAL_X_FWD`) 그대로.
> 근거 지도: `/mnt/dev/AG3S_VLA/outputs/2026-11-15_tab3_ablation_map/README.md` 행 3

## 무엇을 했나 (평이한 요약 먼저)

X-fwd 에서 **target 표면의 출처만** 바꾼 판 `…-gh156a-fwd-m6-tro` 를 만들었다. GT mask 경로 (`--target-extent oracle-visible
--oracle-extent-identity pipeline`) 는 client 가 보낸 시뮬레이터 GT code 영상을 같은 depth 로 올려 "보인 target 표면" 만 미세
target-free 층에서 지운다. 새 flag `--external-target-mask grounded-sam2` (기본 꺼짐) 를 켜면 그 code 영상을 **서버가** 만든다:
요청마다 head · wrist policy RGB 와 instruction 명사 (baseline 과 같은 규칙) 를 `.venv-baselines` sidecar 의 Grounded-SAM-2 에
보내 mask 를 받고, code 1 = mask, 2 = 그 밖 depth, 0 = 로봇 (AG3S 의 TSDF robot mask) 으로 `wire.ORACLE_SEG` 자리에 넣는다.
그 뒤 (3D lift · `OracleMemory` · extent · 접촉 규칙 · 부착 · TO) 는 GT 경로 code 그대로다. 꺼짐이면 argv · 동작 · 기록 key 가
이전과 같다 (모든 1,392 판의 `version_flag_values` 가 변경 전과 key · 값까지 같음, 아래 단위 검증 (5)).

## 판 (profile 맨 끝, `SOTA_TAB3_ID_VERSIONS` 뒤에 추가만; `GT15_VERSIONS` 에 포함)

`AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs-gh156a-fwd-m6-tro`
(tuple `pf.SOTA_ABL_M6_VERSIONS`)

### X-fwd 와 다른 flag (정확히 이것뿐)

| flag | X-fwd | -m6 | 왜 |
|---|---|---|---|
| `--nontarget-evidence` | `surface` | **없음** (config 기본 `proximity`) | oracle-visible 은 ball-owned 미세 층의 분할을 통째로 대신한다 — F1 (`surface`) 은 그 분할에만 걸려 inert 이고, `serve_safe` 가 이 조합을 거절한다 (`serve_safe.py:4704`). one-component 를 위해 지울 수밖에 없는 flag |
| `--target-extent` | 없음 (`estimated`) | `oracle-visible` | target 표면 = code 1 표본 (GT 경로와 같은 code) |
| `--oracle-extent-identity` | 없음 | `pipeline` | 지울 물체 = AG3S 의 manipulated 기하가 놓인 물체 (identity 는 X-fwd 의 AG3S 그대로; `gt` 를 쓰면 identity 까지 분할기가 정한다) |
| `--external-target-mask` | 없음 | `grounded-sam2` | code 영상 출처 = 분할기 (client GT 아님) |

나머지 (`--target-exclusion ball-owned --owned-max-dist-voxels 3`, `--target-field-policy exclude-authorized` (base), `--links arms`,
latch · grasp-zone · attach · held 규칙, `--attention-inpass`, optimizer) 는 X-fwd 그대로. `version_flag_values` 차이 = 새 key
`external_target_mask` 하나 + 값이 바뀐 `nontarget_evidence` · `target_extent` · `oracle_extent_identity` (CPU 확인, test).
`--external-target-mask-mode` 는 주지 않는다 (= `per_frame`, 아래 설계 4). 기본 preset = Grounding DINO base + SAM 2.1 hiera-large
(`weights.PRESETS["grounded_sam2"]["default"]`), 임계값 0.4 / 0.3 (공식 demo), top-1 box.

**ball-owned 가 남는 이유:** oracle-visible 은 `--target-exclusion` 의 공 · 창 · 권한 · 거친 층을 그대로 쓰고 (help `serve_safe.py:2651`),
oracle evidence 가 없는 frame (`identity_unresolved` · `no_visible_sample` · memory 없음) 에는 그 frame 의 **estimated extent**
(= ball-owned, F1 없음) 로 돌아간다. GT 판 (`-pc-oraclegt` 등) 과 같은 규칙이다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/adapters/panda/external_mask.py` (새) | `prompt_from_instruction` (= `wire_obs.target_phrase_from_instruction`), `code_image(mask, depth)`, `ExternalTargetMask` (sidecar 시작 · `segmenter_init` · 요청마다 `apply(obs)` · `reset` · `close`, 기록 · metadata) | 행 6 의 서버 쪽 전부 |
| `AG3S/adapters/panda/server.py:288-292` | `PandaSafeServer(external_mask=None)` | 꺼짐: 생성자 인자 그대로 |
| `server.py:305-314` | `metadata`: mask 가 있으면 `oracle_segmentation_required` 를 빼고 `external_target_mask` 추가 | client 가 GT seg 를 요구받지 않게 (안 빼면 `libero_safe_client.check_server_oracle` 가 접속을 거절) |
| `server.py:393-398` | `infer`: `SafePolicy` 앞에서 `external_mask.apply(obs)` — 요청에 `ag3s/oracle_seg/<cam>` · `ag3s/oracle_seg_meta` 를 넣음. client 가 GT code 영상을 보내면 거절 | GT 와 분할기가 섞이지 않게 |
| `server.py:437-443` | 응답 `panda/oracle["external_mask"]` (+ AG3S 가 센 `robot_masked_px`) | 기록 (client 는 `panda/oracle` 를 이미 call row 의 `oracle` 로 저장 — client 수정 없음) |
| `server.py:806-808` | `PORTED_FINAL_CONFIG_DESTS` += `external_target_mask`, `external_target_mask_mode` | `final_config.options_on` 에 보임 |
| `server.py:1162-1172, 1283-1292, 1315-1317` | `build_safe_server(external_mask=None)`: flag 가 있으면 `ExternalTargetMask.from_args(args)` (sidecar 시작) 또는 test 가 준 것, `info["external_target_mask"]` | 꺼짐: 아무것도 안 만듦; flag 없이 주면 거절 |
| `server.py:1447-1450` | `final_config.oracle_visible.needs_client_flag` = "none: the server makes the code images …" (flag 일 때만) | 기록이 거짓 client flag 를 말하지 않게 |
| `server.py:1523-1525` | `main` 의 `finally`: sidecar close | 서버 종료와 함께 멈춤 |
| `AG3S/trajopt/serve_safe.py:1093-1114` | `EXTERNAL_TARGET_MASKS`, `reject_external_mask_flags` (mode 만 · oracle-visible 없이 · `--grounding-source oracle` · `--oracle-target-volume` 과 거절) | GT 로 읽히거나 안 읽히는 조합 차단. GT grounding 의 F1/F2/F3… flag 는 모두 `--grounding-source oracle` 을 요구하므로 함께 막힌다 |
| `serve_safe.py:2700-2711` | `--external-target-mask {grounded-sam2}`, `--external-target-mask-mode {per_frame,track}` (기본 None) | |
| `serve_safe.py:4380-4384` | `reject_panda_only_flags`: RB-Y1 서버 거절 | |
| `serve_safe.py:5091` | `reject_bad_flag_combinations` 에서 호출 | |
| `AG3S/ag3s/runtime/pipeline.py:879-916` | `_oracle_update(observations, cameras=None)`: `meta["robot_mask"] == "server"` 일 때만 `_oracle_server_robot_codes` — 그 frame 의 TSDF robot mask (`cameras[i].robot_mask`) pixel 을 code 0 으로 | 시뮬레이터 code 영상은 로봇이 0 이다. 분할기 영상에는 로봇 code 가 없어, 안 하면 로봇 표면이 memory 의 "다른 물체" 표본이 되어 손가락 옆 target seed 를 지키는 (지우지 않는) 쪽으로 extent 가 바뀐다. client 영상 (key 없음) 은 이전과 같다 |
| `pipeline.py:3541` | 호출부에 `cameras` 전달 | |
| `AG3S/adapters/panda/profile.py:770-771` | `version_requires_oracle_segmentation`: `--external-target-mask` 판은 False | launcher 가 client 에 `--capture-segmentation --oracle-seg` 를 붙이지 않게 (붙이면 서버가 거절) |
| `profile.py:1573-1586` | `ABL_M6_DROP`, `ABL_M6_FLAGS`, `SOTA_ABL_M6_VERSIONS`, 판 등록, `GT15_VERSIONS` += , `TAB3_CODE_DESTS` += 두 dest | 새 dest 는 켰을 때만 기록 (TAB3 code 행과 같은 장치) |
| `Baseline/paper_baselines/closed_loop/sidecar.py:16-17, 145, 188-197, 229-258` | op `segmenter_init` (`provider` = 등록 이름 또는 `module:attr`), `target_mask` (`TargetQuery(prompt_mode="phrase")`, `validate_results`, mask + `TargetMaskResult.summary`), `reset` 이 segmenter 도 reset | 기존 op 는 그대로 |
| `tests/benchmark/test_abl_m6_external_mask.py` (새), `tests/benchmark/abl_m6_stub_segmenter.py` (새) | 아래 단위 검증 | |
| `tests/benchmark/test_sl_sota18_hand_sized.py`, `tests/trajopt/test_sl_gtto11_scaled_qp.py`, `tests/trajopt/test_sl_sota11_opening_rule_reach_sticky.py` | 판 고정 test 의 제외 목록 += `SOTA_ABL_M6_VERSIONS` | 이전 commit 들과 같은 방식 |

## 설계

1. **prompt 규칙 = baseline 규칙 그대로.** `Baseline/paper_baselines/closed_loop/wire_obs.py:47-66` `target_phrase_from_instruction`
   (KNOWS · AEGIS sidecar 가 `obs.target_phrase` 로 detector 에 주는 문장): 소문자 · 공백 정리 · 끝 마침표 제거 뒤 동사 규칙
   8 개 (`pick up the X …`, `put both the X and …`, `put the X on/in …`, `place …`, `push the X …`, `open/close the (top|middle|bottom) X …`,
   `turn on/off the X …`, `stack/move/lift/grab/take the X …`) 의 첫 일치 group. 예: "pick up the black bowl between the plate and the
   ramekin and place it on the plate" → "black bowl". 규칙이 안 맞으면 None → sidecar 호출 없이 모든 camera not found (기록 `why=no_prompt`).
   **registry (`benchmark/configs/suites.json`, in-scope 8 suite 72 instruction, language = LIBERO `grab_language_from_filename`) 전부
   일치 (None 0)**. SafeLIBERO 결과: Spatial 4 → "black bowl", Object → "orange juice" · "chocolate pudding" · "milk" · "bbq sauce",
   Goal → "bowl" ×3 · "drawer" (L-I t3 "open the top drawer and put the bowl inside") · "cream cheese" (L-II t3),
   Long → "alphabet soup" ×2 · "white mug" ×2 (test 에 표로 고정).
2. **RGB.** `observation/image` (head) · `observation/wrist_image` (left_wrist) policy 영상 (224, `raw[::-1, ::-1]` resize) 을
   `wire_obs.policy_rgb_to_opencv` 로 depth grid (256, OpenCV row) 로 되돌린다 (열 뒤집기 + bilinear) — baseline sidecar 가 segment 하는
   영상과 같다. mask pixel 과 같은 (row, col) 의 depth pixel 이 같은 광선. (GT 경로와 같은 두 camera.)
3. **code 영상.** 1 = mask ∧ depth > 0, 2 = 그 밖 depth > 0, 0 = depth 0. 로봇 pixel 은 AG3S 안에서 0 (위 pipeline). GT 표 (`oracle_segmentation`
   docstring) 와 달리 **다른 잡을 물체 code (≥ 3) 가 없다** — 분할기는 target 만 준다. 그래서 `pipeline` identity 는 "AG3S 기하가 mask
   표본 위에 있나" 만 판정한다: 위에 있으면 code 1 표본을 지우고, 다른 물체 위면 (`no_object_sample`) 그 frame 은 estimated extent.
   meta = `{"version": 1, "target": prompt, "codes": {prompt: 1}, "source": "external_segmenter", "provider", "robot_mask": "server"}`.
4. **mode = per_frame.** baseline offline 측정 (SL3 dev, `AG3S_BASELINE_LOG.md:1247-1252`): phrase prompt 3D soft IoU per_frame 0.511 vs
   track 0.391, wrong-object 3D frame 0.253 vs 0.321 → per_frame 이 낫다. track 은 `--external-target-mask-mode track` 으로 가능 (episode
   첫 요청의 `ag3s/reset` 에서 sidecar `reset`).
5. **못 찾음.** 그 camera code 1 없음 (code 2 · 0 만 fold). memory 는 앞 요청의 target 표본을 carving 규칙대로 유지 — GT 경로에서 target 이
   안 보일 때와 같다. 기록 `found=false`, `why` (`no_box_above_threshold` · `empty_sam_mask` · `track_lost`).
6. **sidecar 수명.** `build_safe_server` 가 `SidecarClient` (`<checkout>/.venv-baselines/bin/python -m Baseline.paper_baselines.closed_loop.sidecar`,
   cwd = checkout, env = 서버 env (`JAX_PLATFORMS` · `XLA_*` 제거, `HF_HUB_OFFLINE=1`, cache = `.cache/paper_baselines`)) 를 시작해
   `ping` · `segmenter_init` 까지 끝낸 뒤 websocket 을 연다 (모델 load 가 port 열기 전). 서버 env 의 GPU 를 쓴다 (device 미지정 →
   `default_device()` = cuda). 종료: `main` 의 `finally` 가 `close`; 서버가 kill 되어도 pipe EOF 로 sidecar 가 스스로 끝난다.
   log = `$AG3S_RUN_DIR/logs/external_mask_sidecar.log`. sidecar 오류는 요청을 실패시킨다 (조용히 mask 없이 돌면 행 6 으로 읽힌다).
7. **기록.** call row `oracle.external_mask` = `{provider, mode, prompt, cameras: {cam: {found, score, mask_pixels, code1_pixels,
   latency_ms, why?, source}}, found, sidecar_ms, total_ms, robot_masked_px}`; server config `external_target_mask` (provider · kwargs ·
   provenance (weights sha) · sidecar versions · log · 규칙 문장); metadata `external_target_mask`. 모두 flag 가 켜졌을 때만.

## 단위 검증 (CPU, `CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`, `nice -n 10`, `.venv-ag3s`)

```bash
source scripts/setup/env.sh; CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 nice -n 10 \
  /mnt/dev/AG3S_VLA/.venv-ag3s/bin/python -m pytest -q tests/benchmark/test_abl_m6_external_mask.py
```

| # | test | 무엇을 |
|---|---|---|
| 1 | `test_prompt_rule_on_every_registry_instruction` | registry 72 instruction 모두 noun phrase, baseline 함수와 같음, SafeLIBERO 17 문장 표 고정 |
| 2 | `test_code_image_convention`, `test_pipeline_zeroes_robot_pixels_only_for_server_code_images` | code 규약; robot mask → 0 (mask 없는 camera 는 그대로) |
| 2 | `test_sidecar_ops_round_trip` | **실제 sidecar process** (framing · `segmenter_init` · `target_mask` · `reset`) + stub provider (알려진 box mask) |
| 2 | `test_server_path_feeds_the_oracle_extent` | `build_safe_server` 로 -m6 판 서버 (실제 AG3S · SafePolicy 구성, ESDF build 만 CPU 용으로 대체) + 실제 sidecar (stub) + Tab. 1 기록 frame (`tab1sdm-ag3s-sl-spatial-LI`, t0 e13 call 0): `PandaSafeServer.infer` → `SafePolicy` 가 받는 요청에 uint8 code 영상 (depth 크기) · meta → 실제 `wire.unpack_oracle_segmentation` → `set_oracle_segmentation` → `process_multi_debug` → `_oracle_frame.segmentation=True`, `robot_masked_px` (wrist > 0), memory 의 code-1 표본 > 0, memory code ⊆ {1, 2} (로봇 표본 없음), `oracle_evidence(identity=pipeline)` 이 mask 위 기하면 code 1 적용 · 먼 기하면 `identity_unresolved`; 응답 `panda/oracle.external_mask` key; GT code 영상을 보낸 client 거절 |
| 4 | `test_version_is_x_fwd_with_only_the_target_surface_source_replaced` | argv = X-fwd − `--nontarget-evidence surface` + 3 flag, 맨 끝 등록, record key 차이, `final_options_on`, client seg 불필요 |
| 4 | `test_version_passes_the_panda_checks_and_validates_its_config` | `reject_bad_flag_combinations` · `reject_panda_only_flags` · `check_unported_flags` · `check_inpass_flags` · `check_record_flags` · TO config · `build_panda_ag3s` config validate |
| 4 | `test_refusals` ×4, `test_rby1_server_refuses_and_mode_choices` | 거절 |
| 4 | `test_launcher_dry_run_has_no_client_segmentation_flag` | `launch_libero_paired.server_argv` 에 flag, client argv 에 `--oracle-seg` 없음 |
| 5 | `test_flag_off_server_has_no_mask_and_refuses_one`, `test_earlier_versions_keep_their_record_keys` | 꺼짐: 서버 · info · metadata 그대로 |

추가 (scratch script, test 밖): 변경 전 (`280f53dc`) 과 뒤의 **모든 판 1,392 + 기본 argv** 의 `version_flag_values` 를 JSON 으로 떠서
비교 → 새 판 하나 외 **차이 0** (key · 값).

결과 (2026-11-16): `test_abl_m6_external_mask.py` **15 passed** (server path test ~195 s, AG3S 구성이 CPU 에서 느림). 함께 돈 묶음 (이 file + `test_tab3_ablation_flags` · `test_ablation_main_versions` · 판 고정 3 개 · `test_tab3_identity_grounding` · `test_sl12_oracle_panda` · `test_t46e_oracle_extent` · Baseline `test_wire_and_framing`): **103 passed, 9 skipped, 4 failed** — 4 개는 모두 환경: worktree 에 LIBERO BDDL 없음 (`test_sl12…target_choice…`, `…fixture_only…`) 2, `hsr6_panda_kv_L4.npz` 없음 (`test_sl12…build_on_cpu…` ×2, 알려진 base 실패와 같은 원인).

## verifier 가 알아야 할 것

- 새 flag: `--external-target-mask {grounded-sam2}`, `--external-target-mask-mode {per_frame,track}` (Panda 서버만, 기본 꺼짐).
- client 쪽 변경 없음. launcher 는 이 판에 GT seg client flag 를 붙이지 않는다.
- 재생산 산출물: 없음. 옛 기록 호환: 깨지지 않음 (모든 판 record 같음).
- GPU: sidecar 가 서버와 같은 GPU 에 Grounding DINO base + SAM 2.1 large 를 올린다 (~3 GB [P]). TrajOpt wall-clock 마감과 GPU 를 나눠 쓴다 —
  분할기 호출은 `SafePolicy` 앞에서 끝나므로 TO 와 **동시에** 돌지는 않는다 (같은 요청 안에서 순차).
- 요청당 추가 시간 [D]: baseline offline 측정 `grounded_sam2:per_frame` median 246 ms / frame (2 camera, H200, BL-2) ~ 381 ms (BL-1) +
  pipe 왕복 (2 × 256² RGB, 수 ms) → **요청당 ~0.25–0.4 s**. Panda 요청 0.25 s 간격 (5 step) 의 서버 시간에 더해진다 (client 는 기다림 —
  sim 은 멈춰 있으므로 결과에는 영향 없음, wall 만 늘어남). 서버 시작 +10–20 s (모델 load) [P].
- 제한 (open question 후보):
  1. Long 의 두 물체 task ("put both the alphabet soup and the cream cheese box …") 에서 prompt 는 첫 물체뿐이다 (baseline 규칙).
     두 번째 물체를 집는 동안 AG3S identity 는 mask 밖 → 그 frame 은 estimated extent (ball-owned, F1 없음).
  2. Goal L-I t3 ("open the top drawer and put the bowl inside") 의 prompt 는 "drawer" — bowl 을 집는 동안 같은 fallback.
  3. 분할기에 다른 물체 code 가 없어 `pipeline` identity 의 "다른 물체 위" 판정이 GT 보다 약하다 (항상 fallback).
  4. fallback (estimated) 은 X-fwd 의 ball-owned + F1 이 아니라 ball-owned (F1 없음) 이다 — oracle-visible 과 F1 이 공존 불가라서.

## 내가 기대하는 결과

(verifier 는 측정 전 읽지 않는다.) [P] GT mask 판보다 extent 정확도는 낮고 (offline 3D soft IoU 0.51), 다른 물체를 target 으로 잡은
frame (offline 20–25 %) 은 pipeline identity 와 어긋나 fallback 으로 간다 → 행 5 와 SSR 이 비슷하거나 약간 낮을 것.

## merge (`280f53dc`, wip/sl-aud → wip/aud-next)

두 branch 가 TAB3 code 행을 따로 등록했다 (`wip/aud-next` 의 `65881b5a` -ab1 · -ab4 는 `wip/sl-aud` 에 없음). **`wip/sl-aud` 의 registry
순서를 그대로** 두고 (-ab7, -m3, -m2), -ab1 · -ab4 는 main-table block 뒤에 `SOTA_TAB3_ID_VERSIONS` 로 추가 등록, `SOTA_TAB3_CODE_VERSIONS`
= (-ab7, -ab1, -ab4). `serve_safe.py` · `server.py` 는 `wip/aud-next` 쪽 (상위 집합: -ab7 code 는 양쪽이 같은 commit). -m6 은 그 뒤 맨 끝.
