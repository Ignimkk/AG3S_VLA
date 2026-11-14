# TAB2-L12H3-ARMSTRICT — 구현

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · 작성 2026-11-15 02:2x KST · branch `wip/aud-next` (worktree `/mnt/dev/AG3S_VLA-wt-next`)
>
> 이름 약속: P = `AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs`,
> X = `P-gh156a-tro` (논문 최종 동결 판, 2026-11-14), XF = `P-gh156a-fwd-tro` (X 의 FWDATT twin).

## 무엇을 했나 (평이한 요약 먼저)

1. **Tab. 2 (LIBERO-Safety, 공식 `pi05_libero_safety` checkpoint):** X 와 XF 에 SL-LS1 의 `-lsnative` 를 붙인 판 2 개를 등록했다.
   `--attention-inpass` 는 response model 과 겹치지 않는다 (아래 §Task 1). 거절하는 code 가 없고 launcher dry-run 이 server argv 를 그대로 낸다.
2. **L12H3 를 최종 판에:** `P-gh156a-a12h3-tro` (= X + `--attention-source action:last:L12H3`) 와 FWDATT twin `P-gh156a-a12h3-fwd-tro`.
3. **새 flag `--passthrough-arm-strict` (기본 꺼짐):** T88 통과 판정에서 팔 link (Panda `panda_link1..7`) 행은 가장 가까운 면이 무엇이든
   장애물 물체 행의 허용치 (`--passthrough-obstacle-tolerance-mm`, X 는 0) 로 본다. 어기면 사유 `arm` 으로 거절하고, 그 call 은
   `obstacle` 거절과 똑같이 기존 경로 (action QP / SQP) 로 간다. 판 4 개 (`-pas`) 등록.

모든 새 동작은 기본 꺼짐이다. 이전 판의 argv · config dump · record 키는 그대로다 (test 로 보임).

## 최종 판 이름 (8 개, `profile.SOTA_FINAL_NEXT_VERSIONS`)

| 묶음 | 이름 | = |
|---|---|---|
| Tab. 2 | `P-gh156a-tro-lsnative` | X + `--response-model libero_safety_fork_native` |
| Tab. 2 | `P-gh156a-fwd-tro-lsnative` | XF + `--response-model libero_safety_fork_native` |
| L12H3 | `P-gh156a-a12h3-tro` | X + `--attention-source action:last:L12H3` |
| L12H3 | `P-gh156a-a12h3-fwd-tro` | `P-gh156a-a12h3-tro` + `--attention-inpass` |
| ARMSTRICT | `P-gh156a-pas-tro` | X + `--passthrough-arm-strict` |
| ARMSTRICT | `P-gh156a-a12h3-pas-tro` | `P-gh156a-a12h3-tro` + `--passthrough-arm-strict` |
| ARMSTRICT | `P-gh156a-fwd-pas-tro` | XF + `--passthrough-arm-strict` |
| ARMSTRICT | `P-gh156a-a12h3-fwd-pas-tro` | `P-gh156a-a12h3-fwd-tro` + `--passthrough-arm-strict` |

`-lsnative` 는 기존 규칙 (`X + LSNATIVE_SUFFIX`, 이름 끝에 붙음) 을 따랐고, 나머지는 `<base without -tro><segment>-tro`.
8 개 모두 `GT15_VERSIONS` 에만 넣었다 (`SOTA_VERSIONS` 는 family 별 개수가 test 에 고정돼 있어 SL-AUD 처럼 뺐다).
기존 `SOTA_LSNATIVE_VERSIONS` 는 `(frozen-lsnative,)` 로 test 에 고정돼 있어 건드리지 않고 새 tuple `SOTA_TAB2_LSNATIVE_VERSIONS` 를 만들었다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/adapters/panda/profile.py:1474-1505` | `SOTA_AUDIT_FWD_BASES` block 바로 뒤에 새 block (append only): `SOTA_FINAL_X` · `SOTA_FINAL_X_FWD` · `SOTA_TAB2_LSNATIVE_VERSIONS` · `SOTA_FINAL_ATT_VERSIONS` · `ARMSTRICT_SEGMENT/FLAGS` · `SOTA_ARMSTRICT_VERSIONS` · `SOTA_FINAL_NEXT_VERSIONS` | Task 1-3 의 판 등록. 앞 block 은 수정 없음 |
| `AG3S/trajopt/config.py:1029-1036` | `SqpConfig.passthrough_arm_strict: bool = False` | 새 flag 의 config field |
| `AG3S/trajopt/config.py:1097-1101` | validate: bool 이어야 하고, 켜면 `passthrough_when_clear` 와 `passthrough_obstacle_tolerance` 가 필요 | 허용치가 없으면 무엇에 맞출지 없다 |
| `AG3S/trajopt/config.py:1688` | `_OMIT_WHEN_DEFAULT` 에 `("sqp","passthrough_arm_strict"): False` | 꺼짐이면 config dump 에 키 없음 |
| `AG3S/trajopt/sqp.py:42` | `import re` | link 이름 pattern |
| `AG3S/trajopt/sqp.py:86-102` | `_ARM_LINK_RE` (`^(panda_link[1-7]\|link_(left\|right)_arm_[0-9]+)$`) · `arm_link_row_mask(sphere_link_names, n_spheres)` | 행 → link 대응 (아래 §행과 link) |
| `AG3S/trajopt/sqp.py:1428-1429` | `_clear_passthrough` 의 elif 사슬: `obstacle` 다음, `path` 앞에 `elif self._arm_row_refuses(...): reason = "arm"` | 거절 사유 `arm` |
| `AG3S/trajopt/sqp.py:1509-1551` | `_arm_row_refuses` (꺼짐이면 곧바로 `False`, 기록 없음) | 판정과 `arm_rows` 기록 |
| `AG3S/trajopt/serve_safe.py:2804-2809` | argparse `--passthrough-arm-strict` (`# SL-AUD — end` 바로 뒤) | CLI |
| `AG3S/trajopt/serve_safe.py:4979-4981` | `reject_bad_flag_combinations`: `--passthrough-when-clear` · `--passthrough-obstacle-tolerance-mm` 없이는 거절 | config validate 와 같은 규칙을 한 층 위에서 |
| `AG3S/trajopt/serve_safe.py:5198-5199` | `passthrough_overrides`: `out["passthrough_arm_strict"] = True` | `trajopt_config_from_args` 경로 (SL-AUD2 와 같음) |
| `AG3S/adapters/panda/server.py:774-776` | `PORTED_FINAL_CONFIG_DESTS` 에 `'passthrough_arm_strict'` | Panda server 가 거절하지 않게 |
| `tests/trajopt/test_arm_strict_passthrough.py` (새 파일, 15 test) | 아래 §단위 검증 | |
| `tests/benchmark/test_sl_sota18_hand_sized.py:80` · `tests/trajopt/test_sl_gtto11_scaled_qp.py:59` · `tests/trajopt/test_sl_sota11_opening_rule_reach_sticky.py:122` | 예외 목록에 `pf.SOTA_FINAL_NEXT_VERSIONS` 추가 | 기존 family 를 넣던 방식 그대로 (새 판이 `--graspable-hand-sized-mm` · `--ee-action-qp-scaled` · `-rs-` 를 물려받음) |
| `tests/benchmark/test_sl_ls1_fork_native.py:195` · `tests/benchmark/test_sl11_fork_response.py:295` | `--response-model` 을 가진 판 목록 예외에 `pf.SOTA_TAB2_LSNATIVE_VERSIONS` 추가 | 새 `-lsnative` 2 개 |

**identity section 에 대해:** `--passthrough-arm-strict` 는 sqp key 라 `t43t_identity_section` 이 아니라 TrajOpt config dump
(`sqp.passthrough_arm_strict`, 켜졌을 때만) 에 실린다. SL-AUD2 (`282e22d9`) 의 `ee_action_limit_passthrough` 와 같은 경로라 identity section 은 손대지 않았다.

## Task 1 — `--attention-inpass` 와 `-lsnative` 는 같이 쓸 수 있나

같이 쓸 수 있다. 근거:

- `AG3S/adapters/panda/server.py:827` `check_inpass_flags` 가 거절하는 것은 `--no-attention`/`--no-safe`, action 이 아닌 `--attention-source`, HSR6 뿐이다. response model 은 보지 않는다.
- `AG3S/trajopt/attention_policy.py:415` `_check_inpass_args` 도 task probe · sampler · backend · prefix feature · HSR5/6 · subtask probe · non-action source 만 거절한다.
- `server.py` `build_safe_server`: `AttentionPolicy(policy, …, inpass=True, model_config=get_config(args.config).model)` 를 만든 다음 그것을 `ChunkToJointPolicy(served_policy, kin, response=pa.response_for(response_name))` 가 감싼다. 즉 fork-unit 나눗셈 (`NATIVE_UNIT`) 은 in-pass sampling **뒤** 단계라 서로 독립이다.
- `InpassSampler` (`AG3S/trajopt/inpass_attention.py`) 는 policy 의 `_input_transform` / `_output_transform` 과 `Pi0Config` 를 쓴다. `pi05_libero_safety` TrainConfig (`third_party/openpi/src/openpi/training/config.py:790`) 의 model 은 `pi05_libero` 와 같은 `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` 이다 (norm stats 만 다름).
- launcher `launch_libero_safety_paired.check` 는 `native_fork_version` 과 adapter 짝만 본다 — 두 판 모두 통과 (test 로 확인, adapter `fork_native_unit` 이 자동으로 설정됨).

남는 위험 (code 로는 확인 불가): in-pass sampler 가 `pi05_libero_safety` 에서 Tab. 1 의 FWDATT 처럼 action 이 bit-identical 인지는 GPU 에서 한 번 재야 한다 (FWDATT 의 765/765 는 `pi05_libero` 에서 잰 것).

### launcher dry-run

```bash
source scripts/setup/env.sh; export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
.venv-ag3s/bin/python benchmark/runners/launch_libero_safety_paired.py --conversion C --seed 7 --data-split test \
  --violation-termination official --policy-checkpoint pi05_libero_safety --modes vla_v11ab --record-constraints full \
  --ag3s-version <V> --task-suite-name affordance --level 2 --task-ids 0 --episode-ids 19 --run-id dryrun-tab2 --dry-run
```

두 판 모두 `check()` 를 통과하고 server argv JSON 과 client argv 를 출력한 뒤, 마지막 `runtime_schedule` 에서
`FileNotFoundError: benchmark/LIBERO/libero/libero/envs/env_wrapper.py` 로 끝났다 (이 worktree 에는 LIBERO submodule 이 없다; rc=1).
출력된 server argv 를 parse 한 결과:

| V | argv 길이 | `--config` | `--checkpoint` | `--response-model` | `--attention-inpass` | `--graspable-hand-sized-mm` | client `--action-adapter` |
|---|---|---|---|---|---|---|---|
| `P-gh156a-tro-lsnative` | 160 | `pi05_libero_safety` | `…/checkpoints/panda/pi05_libero_safety` | `libero_safety_fork_native` | 없음 | 156 | `fork_native_unit` |
| `P-gh156a-fwd-tro-lsnative` | 161 | `pi05_libero_safety` | `…/checkpoints/panda/pi05_libero_safety` | `libero_safety_fork_native` | 있음 | 156 | `fork_native_unit` |

(원문: `/tmp/claude-0/-mnt-dev/f02a3e30-9681-4dde-90d4-595d58047b97/scratchpad/tab2_dry_X-lsnative.txt`, `…/tab2_dry_Xfwd-lsnative.txt` — 임시 파일.)
GPU 에서 돌리기 전에 main checkout (LIBERO submodule 있음) 에서 dry-run 을 한 번 더 끝까지 돌려 `runtime_schedule` 줄을 확인할 것.

## Task 3 — 행과 link 의 대응

- 통과 판정이 읽는 행은 `(step, query)` 쌍이다. `query < linearizer.n_spheres` 는 로봇 구, 그 뒤는 쥔 물체 점 (`attached:<link>[i]`).
- 로봇 구 → link 는 robot model 의 `sphere_link_names` (구 하나에 이름 하나, FK 순서) — `linearizer._query_name` 이 기록에 쓰는 그 목록이다 (`obstacle_rows.worst.link`, `reference_worst.link` 의 값).
- `arm_link_row_mask(names, n_spheres)` 는 이 목록에서 `_ARM_LINK_RE` 에 맞는 구만 True. Panda `--links arms` 모델 (최종 판) 에서 59 구 중 37 구:
  link1 4 · link2 4 · link3 4 · link4 4 · link5 13 · link6 3 · link7 5. `panda_hand` 18 · finger 4 는 아니다. `panda_link0` 는 constraint 모델에 없다.
  쥔 물체 행은 index 로 빠진다 (`[:, :n_spheres]`). 목록 길이가 구 개수와 다르면 판단할 수 없으므로 fail closed (사유 `arm`, `arm_rows.error`).
- 행의 clearance 는 다른 모든 검사가 쓰는 같은 숫자: candidate · plane · ESDF block 의 최소 (`_clearances_from` + `_esdf_clearance`;
  ESDF block 은 실제로 `(H, n_query, 1)` 이라 `(H, n_query)` 로 읽는다 — SL-AUD1 이 틀렸던 그 모양).
- 창은 장애물 물체 검사와 같다: `passthrough_obstacle_full_horizon` 이면 전체 horizon (X 는 켜짐), 아니면 실행 창 (`lookahead_from`).
- 판정: 팔 행 중 하나라도 `clearance < -passthrough_obstacle_tolerance` 이면 거절. X 는 tolerance 0 이므로 floor (ESDF 행은 10 mm obstacle margin 포함) 아래로 조금이라도 들어가면 거절. margin (`-om15`) 은 팔 행에는 적용하지 않았다 (요청은 "tolerance").
- 순서: `collision` (τ) → `obstacle` → **`arm`** → `path` → `limit`. 장애물 물체 행이 먼저 결정되므로 그 동작은 그대로다 (test).
- 기록 (`to.passthrough_clear.arm_rows`, flag 켜짐 + 앞 두 검사를 통과해 이 검사가 돌았을 때만):
  `{"n_violating": int, "worst_m": float|None, "link": str|None, "step": int|None, "tolerance_m": float, "n_arm_spheres": int}`
  (`n_violating` 은 (step, 구) 쌍 수; 요청한 세 키 + 세 키).

### 실제 기록에서 얼마나 걸릴까 (offline, 하한)

`…/20261113T173311_820535cd7cd5_sub-a12h3sdm-e1a-spatial-LI-t1-LI-…-a12h3-tro-C` 의 모든 `*_calls.jsonl`: 통과 판정 490 call 중 applied 226,
그중 `reference_worst.clearance_m < 0` 50, 그 worst 행이 팔 link 인 것 27 (`panda_link7` 23, `panda_link5` 4). 이 27 call 은 새 flag 로 거절된다
(`reference_worst` 는 전체 최악 행 하나만 실려 있어 하한이다; 그리고 X 의 tolerance 0 기준). 예: `safelibero_spatial_t1_e11` call 22 — `reference_worst` `panda_link7` −15.2 mm (applied → 새 flag 로 `arm`).
**주의:** 걸리는 것의 대부분이 `panda_link7` (hand 바로 위 flange) 이다. 집는 동안 link7 이 target 에 붙는 chunk 도 이제 거절되어 action QP 로 간다 — 아래 action QP 동작 참고. link7 을 팔로 볼지는 lead 판단 (요청대로 link1-7 로 넣었다).

## action QP 는 `arm` 거절 뒤 그 행을 어떻게 다루나 (변경 없음, 보고만)

`arm` 거절은 `obstacle` 거절과 같이 `solve` 의 다음 단계로 간다 (SL-AUD2 `limit` 통과는 사유가 `limit` 일 때만이라 해당 없음). action map 이 있으면 `_solve_ee_action`:

- 팔 행은 대개 장애물 물체 행이 **아니다** (그래서 통과됐던 것). SL-SOTA1 (`-om15p10`) 의 `_linearize_with_obstacle_floor` 에서 그 행의 floor 는 `other = -relief` = 0 →
  QP 행 `value + G·Da + s >= backoff` 로 **+backoff (5 mm) 까지 고친다.** slack 가중은 장애물 물체 행의 1/10 (priority 10 은 장애물 물체 행에만).
- SL-SOTA4 `-gr` 가 켜지면 그 행은 `-passthrough_tolerance` 까지 풀린다 (고치지 않음).
- SL-AUD1 `-rf` 가 켜지면 그 행의 floor 는 `min(0, c_ref − backoff)` → policy chunk 의 깊이 그대로 두고 **고치지 않는다** (Da = 0). 즉 `-rf` 와 `-pas` 를 같이 쓰면 팔 행에 대해서는 `-pas` 가 action 을 바꾸지 못한다 (거절 → QP → 같은 chunk). 최종 verdict 는 `violation_tolerance` (X 100 mm) 안이라 실행된다.
- SL-SOTA2 진행 선호 (`--ee-action-qp-progress 0.4`) 는 `obstacle_rows.violation_m` 에 비례하는데 `arm` 거절에서는 0 이라 꺼진다.
- `-qs` 의 `translation-only` (fixed dims) 는 그대로 적용된다 — 회전 성분으로 팔을 비키는 수정은 못 한다.

실제 Panda 모델 + 실제 ESDF 로 잰 값 (`test_real_rows_the_action_qp_after_an_arm_refusal`): link5 −15 mm, action map 0.05 rad/unit 일 때
기본 · `-om15p10` → max |Da| 0.511, 결과 최악 행 +5.0 mm (link5); `-rf` → max |Da| 0, 결과 최악 행 −15 mm.

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-wt-next && source scripts/setup/env.sh && cd /mnt/dev/AG3S_VLA-wt-next
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
nice -n 10 /mnt/dev/AG3S_VLA/.venv-ag3s/bin/python -m pytest -q -p no:cacheprovider <files>
```
(`AG3S.__file__` = worktree 확인.)

- 새 파일 `tests/trajopt/test_arm_strict_passthrough.py`: **15 passed.**
  - 꺼짐: config/CLI 키 없음, 새 code 미도달 (`arm_link_row_mask` 를 raise 로 바꿔도 같은 결과), 명시적 False 와 bit 동일, record 동일.
  - 켜짐 (RB-Y1 fixture, candidate block): 팔 구가 τ 안 33 mm 로 들어간 chunk → 사유 `arm`, 결과는 pass-through 를 끈 solve 와 bit 동일.
  - 켜짐 + clear chunk: record 에 `arm_rows` 하나만 추가, 나머지 동일.
  - 장애물 물체 행: 먼저 결정되고 켜짐/꺼짐 record 동일, `arm_rows` 없음.
  - fail closed (`sphere_link_names` 길이 불일치).
  - **실제 linearizer · 실제 행:** Panda `--links arms` constraint 모델 + `CollisionLinearizer` + 실제 `EsdfField` (analytic), linearizer 는 아무것도 바꾸지 않음.
    link5 −15 mm → `arm`, `arm_rows.link == "panda_link5"`, `n_arm_spheres == 37`; 같은 깊이의 `panda_leftfinger` · `panda_hand` → `clear`.
  - action QP 동작 (위 절).
  - CLI · port · 8 판 (argv = base + segment, parse + `reject_bad_flag_combinations` + `check_unported_flags` + `check_inpass_flags` + `validate`), LIBERO-Safety launcher `check()`.
- 지정 회귀 묶음 + 관련 (`test_t88_passthrough_when_clear`, `test_sl_aud_action_qp`, `test_sl_aud_versions`, `test_sota_fwd_versions`,
  `test_sota_clearance_versions`, `test_sl_sota18_hand_sized`, `test_sl_gtto11_scaled_qp`, `test_sl_sota11_opening_rule_reach_sticky`,
  `test_fwdatt_inpass`, `test_sl_gtto9_ee_action_qp`, + `test_sl_ls1_fork_native`, `test_sl11_fork_response`, `test_sl_gtto16_obstacle_tolerance`,
  `test_sl_sota1_obstacle_row_margin`, `test_sl_sota7_hand_beneath`): **203 passed, 7 skipped.**
- `pytest tests/benchmark tests/trajopt -k "version or flag" --ignore=…t41b… --ignore=…t41c… --ignore=…t43q…`: **10 failed, 435 passed, 42 skipped** (19 min). 10 개 모두 base 에서 알려진 실패다: 7 = storage 파일 `sl6b-hsr6-panda-labels/probe/hsr6_panda_kv_L4.npz` 없음 (`test_sl10_action_box` 1 · `test_sl10b_termsoff_versions` 1 · `test_sl10d_held_free_query` 1 · `test_sl12_oracle_panda` 2 · `test_sl13_oracle_volume` 2, FileNotFoundError 확인), 1 = `test_sl_gtto_gt_perception::test_only_the_gt_versions_require_the_segmentation` (새 판 8 개 중 GT segmentation 을 요구하는 것은 0 — 이 실패와 무관), 2 = `test_launcher_record_constraints` (worktree 에 LIBERO submodule 없음). 그 밖의 실패 없음.

## verifier 가 알아야 할 것

- 새 flag: `--passthrough-arm-strict` (`sqp.passthrough_arm_strict`, 기본 False). 꺼짐이면 config dump · record 키 · 결과 동일.
- 재생산이 필요한 산출물: 없음. closed loop 은 돌리지 않았다 (CPU only).
- 옛 기록과 호환: 깨지지 않음. 새 record 키 `to.passthrough_clear.arm_rows` 와 새 사유 `"arm"` 은 flag 켜짐에서만.
- 하지 않은 것: GPU 측정 · closed loop · `-lsnative` FWDATT 의 action 동일성 측정 · main checkout 에서의 dry-run 끝까지 (LIBERO submodule 이 이 worktree 에 없음).
- 열린 질문: (1) `panda_link7` 을 팔로 둘지 (거절의 대부분); (2) `-pas` 를 `-rf` 와 같이 쓰면 팔 행은 고쳐지지 않음 — 조합할 거면 action QP 쪽 규칙이 따로 필요;
  (3) 팔 행에 장애물 margin (15 mm) 을 줄지 (지금은 tolerance 만).

## 내가 기대하는 결과

(verifier 는 측정 전에 읽지 않는다.) L12H3 판에서 `clear` 로 통과되던 chunk 중 팔이 target 옆 장애물을 스치는 경우가 action QP 로 넘어가
팔 접촉 충돌이 줄 것. 대가로 link7 이 target 에 붙는 집기 chunk 가 수정되어 집기 성공이 조금 줄 수 있다.
