# IFG-S2SERVE — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-19 KST
> task: `AG3S/ag3s/docs/handoff/IFG-S2SERVE.task.md` · lead 결정 (2026-10-19, 구현 중 SendMessage): target 중심 = (b), feature = 별도 jit pass, 보류 = exclusion-gate refusal 경로 · 쥔 동안 미적용

## 무엇을 했나 (평이한 요약 먼저)

서버에 두 기능을 flag 로 붙였다. 둘 다 기본 off 이고, off 이면 새 코드에 들어가지 않는다.

1. **안전 경보 (`--confidence-gate on`)** — 매 요청마다 서버가 attention + depth 로 고른 표면 영역(top1-only-ns:N, FK 구 mask)의
   VLA prefix layer-8 feature 를 직전 frame 의 기억(chain)과 비교한다. 점수가 잠근 문턱 0.5904 아래면 그 frame 은 **target 접촉
   예외를 보류**한다(target 을 장애물로 둔다). 점수 계산은 잠근 offline 코드(`ifg_memory`)를 그대로 import 해 쓰고, held-out 12 frame
   에서 offline `conf.jsonl` 과 **소수점 끝까지 같은 값**(|Δ| = 0)을 냈다.
2. **학습 없는 3D 쥠 규칙 (`--held-3d-rule on`)** — grasp latch 가 attach(CLOSING → HELD)를 걸 때 기존 증거에 `grip < 0.9 ∧ (target
   없음 ∨ hand_d ≤ 0.1313 m)` 를 AND 로 더한다. 손 대리점 FK 는 offline 정의와 held-out frame 에서 0.006 mm 이내로 같다. revoke · 놓기
   · slip 은 그대로다.

로봇 렌더 mask 는 어디에서도 쓰지 않는다. 잠근 `experiments/studies/ifg_*.py` 는 수정하지 않았다(import 만).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/runtime/confidence_gate.py` (새 파일) | `ConfidenceGate.step` (:97) · `unit_patch_features` (:52) · `candidate_regions` (:60) · `dump_rounding` (:41) · `surface_selection` (:163) · `GATE_THRESHOLD` | 경보 점수. `ifg_memory.episode(center=True)` 의 feature 정규화 · 후보 영역 · chain 기억 갱신(δ 0.1 switch)을 같은 줄로 옮겼고 `ifg_memory.unit_rows` · `MIN_NODES` · `MAX_SIDE` · `ifg_round_mass.regions` 는 import |
| `AG3S/ag3s/runtime/attention_surface.py:158,228` | `select_target_surface(..., codes=True)` 인자 | 경보 경로는 code image(pixel→node 매핑)를 건너뛴다. 기본 True = 기존 그대로 (IFG-CL1/CL2 test 통과) |
| `AG3S/ag3s/runtime/pipeline.py:146` | `AG3S(confidence_gate=None)` | 기본 None = 경보 코드에 들어가지 않음 |
| `AG3S/ag3s/runtime/pipeline.py:537,543,548` | `set_gate_feature` · `confidence_gate_record` · `_confidence_gate_step` | 요청 feature 받기, 기록, 보류 판단 |
| `AG3S/ag3s/runtime/pipeline.py:1799,2119,2312` · reset `:459` | `_run` 시작에서 feature 소비, ESDF 단계에서 `manipulated_geometry` 를 쓰는 곳들(nontarget · `_build_esdf` · `_constraints`) **앞**에서 경보, `metrics["confidence_gate"]` | 보류는 이 frame 의 `manipulated_geometry = None` 하나로 끝난다 |
| `AG3S/trajopt/attention_source.py:348–375` | `RESID_LAYER = 8` · `IFG_PREFIX_L8_KEY` · `resid_block_impl` · `prefix_resid_loop` | held-out collector(`hc_dump.py`)의 layer loop |
| `AG3S/trajopt/attention_source.py:396,428,455,485,499` | `JaxPrefixBackend._resid_setup` · `resid_l8` · `run(want_resid=False)` | 서버 VLA forward 쪽 feature 추출 (아래 "feature 위치") |
| `AG3S/trajopt/attention_policy.py:75,95–96,190,228–233` | `AttentionPolicy(prefix_feature=False)` | 켜면 extended 경로에서 `want_resid=True`, 결과에 dump 와 같은 float16 반올림 tensor. noun_tokens 검사의 `attention_source is None` 경우 보호 (`prefix_feature` 만 켠 경우) |
| `AG3S/trajopt/safe_policy.py:157,409,487,524,1636` | `SafePolicy(held_3d=None)` · `_confidence_gate_on` · feature pop · `ag3s.set_gate_feature` · summary `confidence_gate` | feature 는 응답에 싣지 않는다 (pop). 켰을 때만 metadata · summary 키 |
| `AG3S/trajopt/safe_policy.py:1204,1361–1369,~1432` | `_held_3d_target_centre` · `_advance_grasp` 에서 규칙 평가 → `observe_grasp(extra_checks={"held_3d": …})` · grasp record `held_3d` | 3D 규칙의 latch 연결 |
| `AG3S/trajopt/grasp_latch.py:515,596` | `observe_grasp(extra_checks=None)` | CLOSING 의 attach checks 에만 AND. None = 기존 그대로 |
| `AG3S/trajopt/held_3d.py` (새 파일) | `held_3d_rule` · `HandProxyFK` · `Held3DRule.evaluate` · 상수 `HELD_3D_MAX_M` / `HELD_3D_GRIP_CLOSED` (`ifg_stage2_fk.D_HOLD` / `GRIP_CLOSED` import) | 3D 쥠 규칙 |
| `AG3S/trajopt/serve_safe.py:1964–1980` | `--confidence-gate {off,on}` · `--confidence-gate-threshold` · `--held-3d-rule {off,on}` · `--held-3d-max-m` | 새 flag |
| `AG3S/trajopt/serve_safe.py:2067,2077,2139,2174,2199,1219,3046,3235,3245` | `confidence_gate_from_args` · `held_3d_from_args` · `reject_ifg_s2serve_flags` · `ifg_s2serve_record_meta` · 배선 | flag → 객체, 시작 시 거절, 기록 meta |
| `tests/ag3s/test_ifg_s2serve_gate.py` · `tests/trajopt/test_ifg_s2serve_held3d.py` · `tests/trajopt/test_ifg_s2serve_resid.py` (새 파일) | 단위 test | 아래 |

## 새 flag (모두 기본 off)

| flag | 기본 | 뜻 | 거절되는 조합 |
|---|---|---|---|
| `--confidence-gate {off,on}` | off | 경보 on | `--attention-source noun_target:L10H7` 가 없거나 다름 · `--no-attention` · `--no-safe` · `--no-perception` |
| `--confidence-gate-threshold S` | None = 0.5904207590672415 (LOCK) | 문턱 | gate off 에서 주면 거절 · 유한하지 않으면 거절 |
| `--held-3d-rule {off,on}` | off | 3D 규칙 on | `--no-safe` · `--no-perception` |
| `--held-3d-max-m M` | None = 0.1313 (0.0813 + 0.05, IFG-S2FK) | 거리 상한 | rule off 에서 주면 거절 · 양수가 아니면 거절 |

기록 meta (`--record-constraints`): 켠 flag 만 `confidence_gate` · `held_3d_rule` 키 (`ifg_s2serve_record_meta`). server metadata 도 켰을 때만.

## 1. 경보 — feature 위치와 텐서 일치 근거

**잡는 텐서:** prefix block 0 … 8 을 지난 residual stream 의 image 행 0–767 (`x[0, :768]`, 768 × 2048). held-out collector
`…/20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout/p/hc_dump.py` `dump_step` 의 `r8` 과 같은 위치다:
`embed_prefix → x = tokens.astype(embed_dtype) → for l in range(9): x = Block(configs, return_probs=False).apply(layer_p[l], [x, None],
None, pos, mask4, [None, None], True)[0] → x[0, :768]`, 저장은 float16 (유한할 때) — 서버도 `dump_rounding` 으로 같은 반올림을 한다.

**왜 forward "hook" 이 아니라 별도 pass 인가:** openpi gemma 는 18 layer 를 `nn.scan` 안에서 돌려 중간 layer 출력을 꺼낼 자리가 없고,
`third_party/openpi` 수정은 내 소유가 아니다. 그래서 collector 와 같은 loop 를 `JaxPrefixBackend.resid_l8`
(`attention_source.py:428`) 로 따로 돌린다 (lead 승인). 서빙 attention 의 jit executable(`_jit_noun`, `_jit_prefix`)은 건드리지 않는다.
같은 전처리된 observation(noun 경로의 `obs`)을 쓴다. 차이 하나: collector 는 layer 파라미터를 jit 밖에서 잘랐고, 여기는 jit 안에서
traced index 로 자른다 (한 executable). 아래 test 1 이 둘이 비트 단위로 같음을 보인다.

| 근거 | 결과 |
|---|---|
| `test_ifg_s2serve_resid.py::test_traced_slice_loop_is_bit_identical_to_the_collectors_loop` (tiny openpi gemma, JAX CPU) | collector loop 와 `np.array_equal` |
| `…::test_full_depth_loop_plus_final_norm_is_the_model_forward` | loop 전체 + final norm = openpi `Module` forward (nn.scan + remat), rtol 2e-5 → loop 가 모델 자신의 residual stream 이다 |
| `…::test_intermediate_is_the_truncated_models_stream[1,2,3]` | block k 뒤 값 = k+1 layer 로 자른 모델의 stream |
| 실제 checkpoint, JAX **CPU**, held-out HO_adjacent_ep333 (`JaxPrefixBackend.run(want_resid=True)` = 서버 코드 그대로) vs GPU dump `prefix_L8.npz` | 아래 표. CPU 와 GPU 의 bf16 수치 차이라 비트 일치는 아니다 |

| 항목 (HO_adjacent_ep333 f000–f011, 12 frame) | 값 |
|---|---|
| shape | (768, 2048) — dump 와 같다 |
| token 별 cosine (CPU vs GPU dump) | 최소 0.99974, frame 중앙값의 중앙값 0.99999 |
| 상대 L2 ‖cpu − dump‖ / ‖dump‖ | 최대 0.0067 (max |Δ| 10, dump |max| 748 — float16 한 칸 수준) |
| 같은 pass 의 noun map (L10H7) vs dump `noun_target_L10H7` | argmax 12/12 같음, max |Δ| 0.0098 |
| 경보 점수: CPU feature vs dump feature (같은 영역, chain 기억 각각) | |Δ| 최대 0.0055 (11 frame), 경보 frame 같음 (f011 하나, 0.5768 vs 0.5775) |
| CPU 시간 | ≈ 243 s / frame (noun pass + layer-8 pass, JAX CPU — GPU 지연과 무관) |

산출물: `/mnt/dev/AG3S_VLA-storage/results/development/20261019T135704+0900_f15e587290f6_803d9134428f/s2serve_cpu_resid/`
(`cpu_resid_check.py` · `cpu_resid_check.jsonl` · `cpu_resid_check.log`). 정확한 비트 일치는 GPU 서버에서만 볼 수 있다 — 같은 jit 구성,
같은 dtype 이라 GPU 에서는 이보다 가깝거나 같을 것이지만 **재지 않았다**.

## 1b. 경보 — 영역 · 점수 일치 근거 (서버 경로 = offline 함수)

| 근거 | 결과 |
|---|---|
| `test_ifg_s2serve_gate.py::test_server_gate_equals_the_offline_score_on_heldout_frames` — held-out HO_adjacent_ep333 f000–f011, replay depth + FK 구 mask + dump noun map + dump `prefix_L8` | 12/12 frame 에서 `surface_selection` 영역 == 잠근 `ifg_memory.r6_choice` 영역 (`np.array_equal`), 점수 == `IFG-S2FK` `R11b/conf.jsonl` `q_mem_chain:prefix_L8` (abs ≤ 1e-12; 실측 차 0.0), f000 무기억, 경보는 f011 (0.5775) 하나 |
| `…::test_gate_first_frame_score_formula_and_threshold_boundary` | 첫 frame `no_memory`, 점수 = `unit_rows(w, U1) · unit_rows(w, U0)`, 경보는 score < 문턱 (같으면 경보 없음, `nextafter` 위면 경보) |
| `…::test_missing_feature_or_region_keeps_the_memory_and_reset_empties_it` · `…no_candidate…` · `…switch_at_delta_0_1…` | `no_feature` · `no_region` 은 기억 유지, 후보 없음은 점수 없음 + 기억 = 이번 선택 (offline 과 같음), δ 0.1 switch 는 기억을 best 후보로, `reset` 이 비운다 |
| `test_threshold_is_the_lock_value` | `GATE_THRESHOLD == LOCK_heldout.json arms.confidence_gate.abstain_if_below` |

영역 선택이 같은 이유: `select_target_surface` 는 `ifg_round_ns.g_and_top1` 을, offline 은 `ifg_memory.r6_choice` 를 부른다. 차이는
`region_gate` 의 `px` 인자 하나이고 `top1_only=True` 에서는 prototype 점수 `q` 에만 쓰여 선택 영역은 같다 — 위 test 가 12 frame 에서 확인.

## 2. 경보 — 보류 경로

`AG3S._confidence_gate_step` (`pipeline.py:548`): 경보 frame 에서 `manipulated_geometry = None`. 이것은 **exclusion gate 가 refuse 할
때 쓰는 경로**(`_exclusion_gate` → `(None, record)`: "no ball, no target-free layer, no guard, no contact permission this frame")와
같은 변수이고, 그 아래 소비자(`_nontarget_points` · `_build_esdf` 의 target-free 계층 · `TARGET_LABEL` · `build_constraint_set` 의
`manipulated_link_margin` / `target_field_exclude`)가 전부 그것을 읽는다. 다른 점:

- grounding target (`constraint_set.target`) 은 **그대로** — latch 의 identity 잠금 · `metrics["manipulated"]` 가 흔들리지 않는다.
- `_gated_manipulated` (다음 frame 의 self-filter target guard) 는 그대로 — guard 는 target 점을 지우지 않게 하는 쪽(보수적)이다.
- `invariant_violation` 은 기록하지 않는다 (불변량 위반이 아니다). 대신 note 한 줄과 `metrics["confidence_gate"].applied = true`.
- **쥔 동안(`_attached is not None`) 은 적용하지 않는다**: `skip = "attached"`, `episode_counts.skipped_attached` 로 센다. 쥔 물체는
  로봇 쪽(attached 경로)이라 되돌리면 자기 자신과 충돌한다. 예외가 원래 없던 frame 은 `skip = "no_exemption"`.

같은 frame 의 부수 효과 (refusal 경로와 같다): fine window 가 target box 를 합치지 않는다 · `_last_object_fit`(held freeing 용)을 그
frame 에 갱신하지 않는다 · `ball_owned` 의 nontarget 근거를 만들지 않는다.

경보는 ESDF 단계 안(`collision_backend` esdf/both)에서만 돈다. 그 단계가 없으면 `gate_why = "no_depth_cameras"`.

**기록** — `summary_json.confidence_gate` (`SafePolicy._exec_summary`) 와 `constraint_set.metrics["confidence_gate"]`:
`gate_score` · `gate_alert` · `gate_why` (`no_memory` · `no_region` · `no_feature` · `no_candidates` · `no_depth_cameras`) · `threshold` ·
`had_memory` · `switched` · `best_candidate_cos` · `n_candidates` · `applied` · `skip` · `surface` (선택 why · node 수 · centroid · ms) ·
`ms` (`surface` · `gate_features` · `gate_regions` · `gate_score` · `gate_total` · `total`) · `episode_counts`
(`frames` · `alerts` · `applied` · `skipped_attached` · `skipped_no_exemption`).

## 3. 3D 쥠 규칙 — latch 연결 지점

- 위치: `SafePolicy._advance_grasp` (`safe_policy.py:1361–1369`) 가 `GraspLatch.observe_grasp(extra_checks={"held_3d": …})` 로 넘기고,
  latch 는 CLOSING 의 `_grasp_checks` 결과에 AND 로 더한다 (`grasp_latch.py:596`). attach 를 **거는** 조건에만 들어간다. HELD 뒤의
  revoke (`_hold_checks`) · 놓기 (`open_streak` / placed) · slip 은 extra_checks 를 보지 않는다 (`test_extra_check_does_not_touch_release_once_held`).
  `pipeline.set_grasp_latched` (LATCHED/CLOSING 신호) 는 바꾸지 않았다.
- `grip` = 이번 요청 16D state 의 두 gripper 열 최소 (`SafePolicy.gripper_columns` = (7, 15), offline `min(raw_state[7], raw_state[15])`).
- 손 대리점 = `link_{left,right}_arm_6` 아래 body 평균 (base frame). `HandProxyFK` 는 `--model-xml` 을 요청의 20 관절 `robot_state` +
  측정 손가락 개도(`_measured_openings`)로 놓는다 — 그 posing 코드는 IFG-CL1 의 `RobotRenderMask.pose` 를 그대로 쓴다 (**렌더 · mask
  메서드는 부르지 않는다**). held-out 30 frame 에서 offline(전체 sim qpos) 대리점과 최대 0.006 mm (test 는 5 frame, < 0.05 mm).
- target 중심 = **직전 요청의 grounding target centroid** (`_held_3d_target_centre`, lead 결정 (b)). attach 판단이 이번 요청의 grounding
  전에 돌기 때문이다 → offline (같은 frame 의 영역 centroid) 보다 **한 요청(≈ 8 control step) 늦다**. 직전 frame target 없음 →
  gripper 조건만. FK 실패 → `held_3d = False` (fail-closed: attach 하지 않음). grip 모름 → not held.
- 기록: `summary_json.grasp.held_3d` = `{grip, hand_d, held_gripper, held_3d, target_centre, proxies, why}`, attach 판정에 쓴 값은
  `grasp.evidence.checks.held_3d`.

## 단위 검증

```bash
# CPU, .venv-ag3s
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
  tests/ag3s/test_ifg_s2serve_gate.py tests/trajopt/test_ifg_s2serve_held3d.py
# JAX CPU, .venv-openpi-live (tiny gemma; .venv-ag3s 에는 jax 가 없어 skip 된다)
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=/mnt/dev/AG3S_VLA-development .venv-openpi-live/bin/python -m pytest -q \
  -p no:cacheprovider tests/trajopt/test_ifg_s2serve_resid.py
# 기존 회귀 (off 경로)
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
  tests/trajopt/test_t57a_attention_source.py tests/trajopt/test_grasp_latch.py tests/trajopt/test_grasp_evidence.py \
  tests/trajopt/test_safe_policy.py tests/ag3s/test_ifg_cl1_attention_surface.py tests/ag3s/test_ifg_cl2_hybrid_extent.py \
  tests/trajopt/test_t34_attach_revoked.py tests/trajopt/test_t46g_reach_hand_relative.py
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
  tests/ag3s/test_pipeline.py tests/ag3s/test_multicamera.py tests/ag3s/test_degradation_reason.py \
  tests/trajopt/test_subtask_wiring.py tests/trajopt/test_record_and_hold_state.py tests/trajopt/test_t43z3_held_slip.py \
  tests/trajopt/test_verdict_reasons.py tests/trajopt/test_exec_feedback.py tests/ag3s/test_frame_record.py
```

결과 (2026-10-19 KST):

| 묶음 | 마지막 줄 |
|---|---|
| 새 test 2 파일 (.venv-ag3s, `nice -n 19`, 단일 thread) | `37 passed in 44.12s` (gate 25 + held3d 12) |
| `test_ifg_s2serve_resid.py` (.venv-openpi-live, JAX CPU) | `5 passed, 115 warnings in 28.93s` |
| 기존 회귀 1 (8 파일) | `205 passed, 1 skipped in 105.02s` |
| 기존 회귀 2 (9 파일) | `209 passed, 14 warnings in 476.04s` |
| 전체 `tests/ag3s tests/trajopt` | **돌리지 않았다** — 시작했으나 같은 pod 의 closed-loop 실행 부하 때문에 lead 요청으로 중단 (중단 시점 ≈ 8 %, 실패 0). 최종 전체 실행은 lead 승인 뒤, `nice -n 19` · 단일 thread 로 |

**flag off = 출력 불변 근거:** (a) `test_default_pipeline_never_reaches_the_gate` — 기본 AG3S 는 `surface_selection` 을 부르지 않고
`metrics` · profile 에 키가 없다; (b) `test_safe_policy_off_passes_no_extra_check` — off 면 `observe_grasp` 에 `extra_checks` 인자 자체가
없다, grasp record 에 `held_3d` 없음; (c) `test_attention_policy_carries_the_dump_rounded_feature_only_with_the_flag` — off 면 backend 에
`want_resid` 를 넘기지 않고 결과 키 없음 (T57-A fake backend 가 그 kwarg 없이 그대로 돈다); (d) `test_flags_default_off_and_record_meta_only_when_on`
— 기본 parse 에서 meta `{}`, gate 객체 None; (e) `select_target_surface(codes=True)` 기본 — IFG-CL1/CL2 test 그대로 통과; (f) 기존 회귀
test 전부 통과; (g) regression-baseline legacy (아래).

**regression-baseline (legacy, 단독 실행, 수정 전 / 후):**

| | 위반으로 시작 | has_target | frame0 clearance_before (mm) | 15 frame clearance_before sha1 |
|---|---|---|---|---|
| 수정 전 (같은 tree, 내 수정 전) | 14/15 | 15/15 | −29.031048280806342 | `40798fb0a4d2…` |
| 수정 후 | 14/15 | 15/15 | −29.031048280806342 | `40798fb0a4d2…` (같다) |

기준선 정본(`regression-baseline` skill legacy: 14/15 · 15/15 · −29.031048280806342)과 같다. cuRobo 기준선은 GPU 가 필요해 돌리지 않았다.

## 측정한 CPU 비용 (이 machine, 다른 job 과 겹친 상태 — verifier 가 단독으로 다시 잰다)

| 단계 | 값 | 출처 |
|---|---|---|
| 경보 영역 선택 (`surface_selection`: `build_geometry` + solve, code image 생략) | 0.88–1.03 s / 요청 (held-out 12 frame, 3 × 480×640) | parity 실행 로그 |
| 경보 점수 (feature 정규화 + 후보 영역 + cos) | 20–104 ms / 요청 | 같은 실행 |
| 3D 규칙 (`Held3DRule.evaluate`, MJCF FK) | ≈ 2 ms / 요청 | 200 회 평균 |
| layer-8 feature pass (GPU) | **재지 않았다** (GPU 금지). SigLIP 3 장 + gemma 9/18 layer 한 번 더. lead: closed-loop 실행에서 잰다 | — |

## verifier 가 알아야 할 것

- 새로 생긴 플래그·기본값 변경: 위 4 개 flag (기본 off). 기본값 변경 없음.
- 재생산이 필요한 산출물(npz 등): 없음. CPU feature 대조 산출물:
  `/mnt/dev/AG3S_VLA-storage/results/development/20261019T135704+0900_f15e587290f6_803d9134428f/s2serve_cpu_resid/` (`cpu_resid_check.py` · `.jsonl` · `.log`).
- 같은 pod 에 떠 있던 `serve_safe` pid 2810253 (port 8292, `--no-safe`) 은 **내 것이 아니다** — IFG-APPLE `run_chain.sh` (run `20261019T132719+0900_f15e587290f6_11408fd6d27f`) 의 서버. 건드리지 않았다.
- 옛 기록과 호환이 깨지는가: 없음 (켰을 때만 새 키).
- cuRobo regression-baseline 은 GPU 가 필요해 돌리지 않았다 (legacy 만).

## 남은 위험

1. **CPU 지연** — 경보 하나에 ≈ 1 s/요청 (영역 선택). SURF 의 attention_surface (≈ 1.7 s) 와 같은 graph 를 짓는다. 533 ms 예산을 넘는다.
   C1aN + 경보 서버의 client 왕복은 그만큼 늘 것이다.
2. **GPU 지연 · 메모리** — feature pass 는 SigLIP + 9 layer 를 한 번 더 돈다. 미측정. 층 파라미터는 복사하지 않는다 (stacked params 를
   jit 인자로 넘기고 안에서 자름).
3. **mask 차이** — offline held-out FK mask 는 기본 `PointCloudConfig` 의 `depth_robot_mask` 였다. 서버의 구 mask 는 서빙 self-filter
   설정(inflation · target guard · 쥔 물체 구)을 따른다 — 영역이 frame 에 따라 다를 수 있다. 같은 위험이 IFG-CL1 SURF 에도 있다.
4. **feature 수치** — 서버 GPU 의 bf16 계산이 dump 를 만든 GPU 계산과 비트 단위로 같다고 보장하지 않는다 (embed 를 별도 jit 으로
   돌린다; collector 는 embed 를 전체 prefix jit 안에서 했다). CPU 대조에서 token cos ≥ 0.9999, 점수 차 ~1e-4 수준 (위 표).
5. **3D 규칙의 target 중심이 한 요청 늦다** — 손이 빠르게 다가가는 마지막 요청에서 hand_d 가 실제보다 클 수 있다 → attach 가 한 요청
   늦어질 수 있다 (안전 쪽은 아님: 쥔 물체가 그만큼 장애물로 남는다).
6. **경보와 latch 의 상호작용** — 경보 frame 은 target 이 장애물로 남아 closing 직전에 HOLD 가 늘 수 있다. 쥔 동안은 적용하지 않는다.
7. 경보는 `--target-extent attention-surface` 와 같이 켜면 같은 영역 선택을 두 번 한다 (C1aN 에는 해당 없음).

## 내가 기대하는 결과

<verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다>
held-out offline 수치(IFG-S2FK.verify.json `memory_gate.fk.gate_all`: 3064 frame 중 경보 408, ≈ 13 %)에 가까운 경보 비율과, 경보 frame 의 대부분이
target 을 잘못 고른 frame 일 것. 3D 규칙은 빈손 닫힘 · 먼 물체에서 attach 를 막고 정상 파지는 한 요청 안에서 attach 할 것.
