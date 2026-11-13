# FWDATT — 구현

> writer: ag3s-implementer (A1, sub-agent 대행) · 읽는 쪽: verifier, scribe, lead
> 작성: 2026-11-13 18:04 KST (`date`로 확인) · worktree `/mnt/dev/AG3S_VLA-wt-fwdattn`, branch `wip/fwd-attn`, base `820535cd`

## 무엇을 했나 (평이한 요약 먼저)

서빙 action chunk를 만드는 **그 sampling 한 번**에서 step 0 suffix pass의 attention probs를 함께 꺼내는 sampler를 만들었다
(`AG3S/trajopt/inpass_attention.py`). 두 번째 checkpoint 사본도, `noise_seed=0` 고정 noise도, 10번 도는 별도 suffix loop도 없다.
`AttentionPolicy(inpass=True)`와 서버 flag `--attention-inpass`(기본 꺼짐, Panda 서버만)로 켠다. 꺼짐이면 코드 경로가 전과 같다.

CPU tiny 모델(실제 `pi05_libero` Pi0 class, `pi05=True`, random init)에서 in-pass action은 `Pi0.sample_actions`와 **bit 단위로 같았고**
(max abs diff 0.0), step-0 cell probs도 `AttentionSampler._jit_prefix/_jit_suffix`와 같았다 (0.0).

## 어떻게 (코드 근거)

- **graphdef / state:** policy의 `nnx.split(policy._model)` (prefix pass와 step 1..9)와, `dataclasses.replace(model_config,
  return_attn_probs=True)`로 만든 attention graphdef (step 0만)를 **같은 state**에 `nnx.merge`한다 (`inpass_attention.py:100-106`).
  `attention_tap.py`와 같은 방식이고, state tree 구조가 다르면 거부한다 (`:104-105`).
  prefix pass를 policy graph로 돌리는 이유: attention graph로 돌리면 prefix probs `[L,B,K,G,T,T]`가 계산 graph에 들어간다.
- **jit 함수 하나:** `_sample_noise(state, observation, noise) -> (x_0, cell)` (`:133-168`). `Pi0.sample_actions`
  (`third_party/openpi/src/openpi/models/pi0.py:222-303`)와 같은 식: preprocess → prefix pass(2-tuple) → step 0을 loop 밖에서
  (`t0 = jnp.asarray(1.0)`, loop carry와 같은 weak float32) attention graph로 한 번 → `x_1 = noise + dt*v_0` →
  `while_loop(time >= -dt/2)`로 나머지. cell = `probs[layer, 0, 0, head, -H:, :768]` float32 = `[action tokens, 768]`.
- **noise는 jit 밖에서** `normal(sample_rng, (1,H,D))`로 뽑는다 (`:170-179`, task 문구 그대로). jit 안에서 뽑는 변형을 먼저
  만들었더니 CPU에서 chunk가 `sample_actions`와 **2.384e-7** 달랐다 (같은 noise를 밖에서 넣으면 0.0; 진단 기록 아래 "남은 위험").
- **infer 감싸개:** `Policy.infer`(`policy.py:69-107`) 순서 그대로 (`:182-208`): input transform → batch → `_rng, sample_rng = split(_rng)`
  → sample → `x[0]` → output transform → `policy_timing`. `noise=` kwarg도 `Policy.infer`처럼 받는다.
- **거부:** `sample_kwargs`에 `guidance_fn` / `guidance_xv_fn`(값이 None이 아니면) 또는 `num_steps` 외의 키 → `InpassUnsupported`
  (`ValueError` subclass) (`:59-72`). 생성 때와 **매 infer 때**(RNG split 전) 둘 다 확인한다 — 뒤에 hook을 꽂아도 거부된다.
  PyTorch policy, openpi `Policy`가 아닌 객체, 범위 밖 (layer, head)도 거부.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/inpass_attention.py` (새, 213줄) | `InpassSampler`, `check_sample_kwargs`, `InpassUnsupported` | 위 "어떻게" |
| `AG3S/trajopt/attention_policy.py:36-41` | docstring FWDATT 절 | |
| `AG3S/trajopt/attention_policy.py:86-95` | `inpass: bool = False`, `model_config=None` 인자; 켜짐이면 거부 검사 후 `sampler=False` (두 번째 model 없음) | 기본 꺼짐 flag |
| `AG3S/trajopt/attention_policy.py:136` | `... and not inpass` — 켜짐일 때 `JaxPrefixBackend`를 만들지 않음 | 꺼짐이면 조건 값이 전과 같다 |
| `AG3S/trajopt/attention_policy.py:150-154` | 켜짐이면 `self._inpass = InpassSampler(...)` (꺼짐: `None`) | |
| `AG3S/trajopt/attention_policy.py:216-224` | `inpass` property; `infer` 첫 줄에서 `self._inpass is not None`일 때만 새 경로 | 꺼짐이면 그 다음 줄부터 전과 같다 |
| `AG3S/trajopt/attention_policy.py:238-255` | `_infer_inpass`: action+block → `AGGREGATIONS[agg]` → `cell_maps` → `result["attention"]`; action source면 `result["attention_source"]` (확장 경로와 같은 record) | 형식 유지 |
| `AG3S/trajopt/attention_policy.py:414-444` | `_check_inpass_args` (noun / hidden-probe source, task_probe, prefix_feature, hsr5/hsr6, subtask_probe≠None, sampler, backend → `ValueError`), `_build_inpass` | 범위 밖 조합 거부 |
| `AG3S/trajopt/serve_safe.py:2646-2650` | `--attention-inpass` (store_true, 기본 off) | 서버와 test가 같은 parser |
| `AG3S/trajopt/serve_safe.py:4209-4211` | RB-Y1 서버는 `--attention-inpass` 거부 (`reject_panda_only_flags`) | 아래 "task와 다른 점" 1 |
| `AG3S/adapters/panda/server.py:819-845` | `check_inpass_flags` (`--no-attention`/`--no-safe`/noun·hidden-probe source/HSR6와 함께면 거부), `attention_inpass_record` | checkpoint 올리기 전 거부 |
| `AG3S/adapters/panda/server.py:1107,1125-1135` | `build_safe_server(..., model_config=None)`; 켜짐이면 `AttentionPolicy(..., inpass=True, model_config=...)` | |
| `AG3S/adapters/panda/server.py:1201-1202,1220` | 켜짐일 때만 `info["attention"]["inpass"]`, 최상위 `info["attention_inpass"] = True` | 서버 config 기록 (`panda_server_config.json`) |
| `AG3S/adapters/panda/server.py:1407,1416-1428` | `main`: `check_inpass_flags`; 켜짐이면 `load_attention_model`(두 번째 사본)을 **부르지 않음** | |
| `tests/trajopt/fwdatt_tiny_inpass.py` (새) | tiny 모델 subprocess helper (수치를 JSON 한 줄로) | `sl3_tiny_pi05.py` 방식 |
| `tests/trajopt/test_fwdatt_inpass.py` (새) | 29 test | 아래 |

`third_party/openpi/`는 고치지 않았다.

## 단위 검증

tiny 모델: `pi05_libero` config에 `paligemma_variant="dummy"`, `action_expert_variant="dummy"`, test 안에서만 gemma dummy
depth 4→10 (V11AB cell L8이 있어야 함), SigLIP So400m/14→mu/14 (224 px에서 같은 16×16 token). `return_attn_probs`는 policy 쪽 **꺼짐**.
action_horizon 10, action_dim 32 → output transform 뒤 `[10, 7]`. random init, bf16. jax 0.5.3, CPU.

```bash
cd /mnt/dev/AG3S_VLA-wt-fwdattn
export PYTHONPATH=$PWD:$PWD/integration/rby1/manipulation/src:$PWD/third_party/openpi/src:$PWD/third_party/openpi/packages/openpi-client/src
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa
# FWDATT test (tiny 모델 포함) — .venv-openpi-live
taskset -c 0-1 /mnt/dev/AG3S_VLA/.venv-openpi-live/bin/python -m pytest tests/trajopt/test_fwdatt_inpass.py -q -s -p no:cacheprovider
# 꺼짐 회귀 — .venv-ag3s (jax 없음: tiny 모델 test 4개는 skip)
taskset -c 0-1 /mnt/dev/AG3S_VLA/.venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_fwdatt_inpass.py \
  tests/trajopt/test_t57a_attention_source.py tests/trajopt/test_t90_hidden_probe.py tests/trajopt/test_t82_hsr5.py \
  tests/trajopt/test_core_eval_hsr6.py tests/ag3s/test_ifg_s2serve_gate.py tests/ag3s/test_subtask_gate.py \
  tests/ag3s/test_t55_attention.py tests/trajopt/test_to_only_policy.py tests/benchmark/test_sl_att1_head_only.py
```

import 확인: `AG3S.trajopt.attention_policy.__file__` = `/mnt/dev/AG3S_VLA-wt-fwdattn/AG3S/trajopt/attention_policy.py`,
`openpi.__file__` = `/mnt/dev/AG3S_VLA-wt-fwdattn/third_party/openpi/src/openpi/__init__.py` (main checkout의 openpi 5개 파일과 `cmp` 동일).

| # | 시험 | 결과 (실제 수치) |
|---|---|---|
| 1 | in-pass action vs `Pi0.sample_actions` (policy의 `module_jit` 호출), 같은 rng, 요청 3개 | max abs diff **0.0, 0.0, 0.0** (bit equal 3/3; action 크기 max 3.63). `Policy.infer` 경로 output transform 뒤 0.0, 0.0. `num_steps=5` 0.0 |
| 2 | in-pass step-0 cell probs `[10, 768]` vs `AttentionSampler._jit_prefix/_jit_suffix` step 0, 같은 noise, 요청 3개 | max abs diff **0.0, 0.0, 0.0** |
| 2b | `AttentionPolicy(inpass=True, noise=seed-0 noise)` vs 기본 경로 (`noise_seed=0`) | `result["attention"]` 3 camera `[16,16]` float32, diff **0.0**; action diff 0.0; 반환 키 같음 `[actions, attention, policy_timing]` |
| 2c | `attention_source="action:mean:L3H5"` | 기본 경로 `AttentionSampler.attention` mean pooling과 diff **0.0**; `result["attention_source"]` record = 확장 경로 형식 |
| 3 | `inner._rng`: in-pass infer 2번 vs `Policy.infer` 2번 (같은 시작 key) | **같음** (key_data equal); 반환 키 `[actions, policy_timing]`·dtype·shape 같음, action diff 0.0, 0.0 |
| 4 | 꺼짐: `inpass` 기본 False, `ap.inpass is None`, 기본·확장 경로 반환 키 전과 같음 + 기존 AttentionPolicy test들 | FWDATT 29 passed (openpi-live); 꺼짐 회귀 묶음 **339 passed, 22 skipped** (.venv-ag3s) |
| 5 | guidance 거부 | `guidance_fn`, `guidance_xv_fn` 각각 `InpassSampler`와 `AttentionPolicy(inpass=True)` 모두 `InpassUnsupported`; 생성 뒤 꽂은 hook도 infer에서 거부, 그때 `_rng` 그대로 |
| + | server flag | parser 기본 False; RB-Y1 `reject_panda_only_flags` → SystemExit; Panda `check_inpass_flags`: action source 허용, `--no-attention`·`--no-safe`·noun source·HSR6 거부; 꺼짐이면 no-op; record 키는 켜짐일 때만 |

Panda 서버 회귀 (.venv-ag3s; `test_panda_sl5a`, `test_sl6a_panda_final`, `test_sl7_panda_p0m`, `test_panda_sl5j_recorder`,
`test_merge_dev_main_20261103_panda`, `tests/trajopt/test_sl6a_injections`): **103 passed, 8 skipped, 29 failed, 13 errors**.
실패·error 42개는 **모두** worktree 환경 때문이다 — 40개 `FileNotFoundError: .../AG3S_VLA-wt-fwdattn/third_party/curobo/curobo/content/configs/robot/franka.yml`
(이 worktree에는 `third_party/curobo` submodule이 초기화되어 있지 않다; main checkout에는 있다), 2개는 worktree에 없는 `.venv-libero` /
`.venv-ag3s` interpreter 경로. symlink로 맞추지 않았다 (agent 규칙). **verifier는 이 묶음을 curobo가 있는 checkout에서 다시 돌려야 한다.**
꺼짐 경로의 실제 모델 test `tests/benchmark/test_panda_sl3.py::test_attention_reader_on_a_tiny_pi05_libero_model`
(V11AB AttentionPolicy cell을 bit 단위로 비교) 는 .venv-openpi-live에서 1 passed.

## verifier 가 알아야 할 것

- 새로 생긴 플래그·기본값 변경: `--attention-inpass` (serve_safe parser, 기본 off, `--robot panda`만). `AttentionPolicy(inpass=False,
  model_config=None)`. 기본값 변경 없음. `PORTED_FINAL_CONFIG_DESTS` / version 표(`profile.py`)에는 넣지 않았다 — version
  이름을 붙일지는 lead가 정한다 (`match_ag3s_version`은 이 flag를 보지 않는다).
- 켜짐일 때 기록: `panda_server_config.json` 최상위 `"attention_inpass": true`, `attention.inpass` 문자열. 꺼짐이면 두 키 없음.
- 켜짐이면 서빙 chunk는 `Policy.infer`의 `sample_actions` 실행 파일이 아니라 in-pass 실행 파일에서 나온다. 첫 요청에서 새로 compile된다.
- 재생산이 필요한 산출물(npz 등): 없음.
- 옛 기록과 호환이 깨지는가: 없음 (꺼짐이면 키·경로 동일).

## 남은 위험

1. **GPU에서 chunk가 rounding으로 움직일 수 있다 (CPU test로는 안 보인다).** in-pass 실행 파일은 `sample_actions`와 다른 HLO다
   (step 0이 while loop 밖, 그 suffix pass가 probs를 추가 출력). CPU에서는 bit 동일했지만, GPU XLA는 fusion 경계·cuBLAS/cuDNN
   algorithm(autotuning) 선택·FMA contraction이 실행 파일마다 달라질 수 있다. 실제로 CPU에서도 noise를 jit **안에서** 뽑는
   변형은 fusion이 바뀌어 chunk가 2.384e-7 (float32 1 ulp 근처, action 크기 ~3.6) 달라졌다 — 같은 수식이라도 program 모양이 바뀌면
   rounding이 움직인다는 직접 증거다. bf16 matmul이 10 step Euler를 거치므로 GPU 차이는 이보다 클 수 있다. closed loop에서는
   1 ulp 차이도 episode 궤적을 갈라 SSR을 seed 수준으로 흔들 수 있다. verifier B2: 같은 요청·같은 `policy_seed`로 flag on/off chunk
   max abs diff를 GPU에서 재고 (`XLA_FLAGS=--xla_gpu_deterministic_ops=true` 유무 둘 다), 0이 아니면 그 크기를 기록한다.
2. **같은 요청에서도 flag off 서버 두 번 시작 사이의 차이**가 기준선이다 (autotuning이 시작마다 달라질 수 있음). 1의 차이는 이것과 비교해야 한다.
3. attention 값은 이제 서빙 sample의 noise에서 나오므로 flag off (`noise_seed=0`)와 **다른 값이 정상**이다 (같은 noise를 주면 0.0 —
   test 2b). 따라서 attention을 쓰는 AG3S 결정(target 선택 등)도 바뀐다; 이것은 기대된 변화이지 bug가 아니다.
4. `model_config`는 서버가 `get_config(args.config).model`로 다시 만든다. checkpoint와 config가 다른 구조면 state tree 비교에서
   거부된다 (`inpass_attention.py:104-105`); 같은 구조에서 dtype만 다른 경우는 policy state의 dtype을 그대로 쓴다.
5. 메모리: 두 번째 checkpoint 사본(≈ pi0.5 한 벌)이 없어진다. step 0 probs `[L, 1, 1, 8, 10, ~1000]` bf16 한 번만 추가.
6. RB-Y1 서버는 지원하지 않는다 (아래 1).

## task와 다른 점

1. **RB-Y1 `serve_safe` 진입점은 flag를 거부한다.** RB-Y1 서버는 `AttentionPolicy(subtask_probe=args.subtask_probe or "default")`로
   kv_L4 subtask label을 항상 싣는데, subtask feature는 task가 이번 범위 밖·거부 대상으로 정했다. 그래서 parser에는 flag를 넣고
   (Panda 서버가 이 parser를 쓴다) RB-Y1에서는 `reject_panda_only_flags`로 거절한다.
2. **noise는 jit 밖에서 뽑는다** (task 문구와 같음). 처음엔 `Pi0.sample_actions`를 흉내 내 jit 안에서 뽑았는데 CPU에서 2.4e-7 차이가
   나서 바꿨다. 결과적으로 jit 함수 signature는 task대로 `(state, observation, noise) -> (x_0, cell_probs)`.
3. 반환 block은 `probs[..., -action_horizon:, :768]` (action token만). pi05는 suffix가 정확히 action token이라 `AttentionSampler`의
   전체 suffix slice와 같다 (test 2 = 0.0). pi0(non-pi05)라면 state token이 빠지는 점만 다르다.
4. `action:` `attention_source`가 있으면 `result["attention_source"]` record도 단다 (확장 경로가 같은 source에서 다는 것과 같은 형식).
