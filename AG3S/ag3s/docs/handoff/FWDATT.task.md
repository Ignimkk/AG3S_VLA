# FWDATT — action-query attention을 서빙 forward pass 안에서 읽기 (task, 2026-11-13)

> 요청: 사용자 (2026-11-13 17:xx). "forward pass로 하는 것을 반드시 해봅시다."
> lead: sdm-ahs 세션 (paper/action-head 실험). 구현: `ag3s-implementer`. 검증: `ag3s-verifier`.
> 계획 기록: `outputs/2026-11-13_action_head_intent_selection/README.md`의 §10 B0–B3 (main checkout, git 제외 폴더).

## 왜

지금 V11AB 서버는 attention을 **별도 복제본**에서 읽는다. 문제는 세 가지다.

- **복제본을 따로 띄운다.** checkpoint를 한 벌 더 올린다 (`AG3S/adapters/panda/server.py:1367-1370`).
- **noise가 실행된 sample과 다르다.** 복제본은 `noise_seed=0`으로 고정된 noise로 계산한다. 그래서 attention이 실행된 sample에서 나오지 않는다 (`AG3S/trajopt/attention_policy.py:337-361`).
- **필요 없는 계산을 한다.** step 0만 쓰는데도 suffix를 10번 모두 돈다 (`AG3S/ag3s/experiments/sources/pi05_attention.py` `_sample`, 중간에 멈추지 않음).
  - docstring의 "one prefix+suffix pass"는 코드와 다르다.

목표는 **서빙 action을 만드는 바로 그 sampling에서 step 0 attention을 함께 꺼내는 것**이다. 그러면 위 세 가지가 함께 없어진다.

## 참고 (먼저 읽을 것)

- **`Baseline/paper_baselines/closed_loop/attention_tap.py` (KNOWS F4a):**
  - attention용 graphdef(`return_attn_probs=True`)를 정책의 **state와 공유**한다 (`nnx.merge`, checkpoint 복사 없음).
  - `policy._rng`를 읽어 **실행된 noise**를 다시 만든다.
  - 이 파일은 action 경로를 바꾸지 않고 step 0만 **다시 계산**한다. 이번 task는 다시 계산하지 않고 **한 번의 sampling에서 action과 attention을 함께** 낸다. 다른 점은 이것뿐이다.
- **`third_party/openpi/src/openpi/models/pi0.py` `sample_actions` (:222-303):**
  - while_loop, `dt`, `cond: time >= -dt/2`, noise 생성 방식 (`normal(rng, (b, H, D))`)을 볼 것.
  - `sample_prefix` / `sample_velocity`(T93)도 볼 것.
- **`third_party/openpi/src/openpi/models/gemma.py`:** `return_probs=True`이면 `PaliGemma.llm`이 3-tuple을 돌려준다. probs 모양은 `[L, B, K, G, T, S]`이다.
- **`third_party/openpi/src/openpi/policies/policy.py` `Policy.infer` (:69-107):**
  - 순서: input transform → batch → `self._rng, sample_rng = split(self._rng)` → sample → `x[0]` → output transform → `policy_timing`.
- **`AG3S/trajopt/attention_policy.py`:** `AttentionPolicy.infer`와 `_attention_and_feature`. 돌려주는 `result["attention"]`의 형식은 그대로 유지해야 한다.

## 무엇을 만드는가

1. **새 module `AG3S/trajopt/inpass_attention.py`**
   - **graphdef와 state:** 정책 config에 `return_attn_probs=True`를 준 graphdef를 만들고 `policy._model`의 state를 공유한다 (AttentionTap과 같은 방식).
   - **jit 함수 하나:** `(state, observation, noise) -> (x_0, cell_probs)`.
     - prefix pass를 한다. 3-tuple로 받는다.
     - Euler loop는 `Pi0.sample_actions`와 같은 식으로 짠다.
     - step 0만 loop 밖으로 펼쳐 그 suffix pass의 probs를 받는다.
     - 지정한 (layer, head)의 `[action_tokens, n_image_tokens]`만 돌려준다. 전송량을 줄이기 위해서다.
     - 나머지 step은 probs를 버린다.
   - **guidance 거부:** guidance hook(`guidance_fn`, `guidance_xv_fn`)이 정책 `sample_kwargs`에 있으면 **명시적으로 거부**한다 (raise). 이번 범위에서 지원하지 않는다.
   - **infer 감싸개:** openpi `Policy.infer`와 **같은 순서와 같은 RNG 진행**으로 action을 만든다.
     - `inner._rng`를 split하고, `normal(sample_rng, (1, H, D))`로 noise를 만든다.
     - output transform을 거친다.
     - 반환 dict의 키와 dtype은 `Policy.infer`와 같게 한다.
2. **`AttentionPolicy`에 기본 꺼짐 flag `inpass: bool = False`**
   - **켜짐일 때:**
     - action과 `result["attention"]`을 모두 위 module에서 얻는다.
     - 복제 sampler는 쓰지 않는다.
     - `result["attention"]` 형식은 기존 기본 경로와 같게 한다: 카메라별 `[16, 16]` raw softmax mass, `_AGG` pooling, `_LAYER` / `_HEAD`. 단, `attention_source`로 다른 cell이 지정되면 그 cell을 쓴다.
     - 확장 경로(T57-A의 `attention_source` noun, `task_probe`, `hsr5`/`hsr6`, subtask feature)는 이번 범위가 아니다. 켜짐과 함께 쓰이면 거부한다.
   - **꺼짐일 때:** 지금과 **byte 단위로 같은 동작**이어야 한다.
3. **서버 flag `--attention-inpass` (기본 꺼짐)**
   - 대상: `AG3S/adapters/panda/server.py` (`serve_safe.py`에도 같은 진입점이 있으면 그곳도).
   - 켜짐이면 두 번째 checkpoint를 **올리지 않는다**.
   - 서버 config 기록에 `attention_inpass: true`를 남긴다.

## 단위 시험 (CPU, `tests/`)

모델은 `paligemma_variant="dummy"`, `action_expert_variant="dummy"`인 작은 Pi0을 random init으로 쓴다. `gemma.py:55` Variant에 "dummy"가 있다.

1. **action 일치:** 같은 rng에서 in-pass action과 `Pi0.sample_actions` action을 비교한다.
   - `max_abs_diff`를 보고한다.
   - bit 단위로 같지 않아도 된다. 목표는 ≤ 1e-5이고, 실제 값을 그대로 적는다.
2. **attention 일치:** 같은 noise에서 in-pass step 0 cell probs와, `AttentionSampler`의 `_jit_prefix` / `_jit_suffix`로 step 0 한 번을 계산한 값을 비교한다.
3. **RNG 진행:** in-pass infer를 연속 두 번 호출한 뒤의 `inner._rng`가, `Policy.infer`를 두 번 호출한 뒤와 같아야 한다.
4. **flag 꺼짐:** `AttentionPolicy` 기존 시험이 그대로 통과해야 한다. 꺼짐일 때 반환 키가 이전과 같은지도 확인한다.
5. **guidance 거부:** guidance가 켜진 정책을 넘기면 거부하는지 확인한다.

## 범위 밖 (하지 않는다)

- GPU 실행, 시간 측정, closed-loop. 이것은 verifier가 한다 (README §10 B2 · B3).
- `third_party/openpi/` 수정. 필요해 보이면 멈추고 lead에게 묻는다.
- main checkout(`/mnt/dev/AG3S_VLA`) 수정, commit, pull. Tab. 1 queue가 돌고 있다.
- 로그와 계획서 문서 수정 (scribe의 몫).

## 환경 주의

- 작업 위치: worktree `/mnt/dev/AG3S_VLA-wt-fwdattn` (branch `wip/fwd-attn`, base 820535cd).
- `scripts/setup/env.sh`는 main checkout으로 `cd`한다. 시험은 worktree에서 `PYTHONPATH=/mnt/dev/AG3S_VLA-wt-fwdattn`로 돌린다. 바꾼 module의 `__file__`이 worktree를 가리키는지 확인한다.
- CPU interpreter: `/mnt/dev/AG3S_VLA/.venv-openpi-live/bin/python`. jax CPU에서 dummy 모델을 쓴다. 이 venv의 jax가 CPU로 돌 수 있는지 먼저 확인하고, `JAX_PLATFORMS=cpu`를 준다.
- GPU를 쓰지 않는다. Tab. 1과 SL-ATT1이 돌고 있다.
- commit은 worktree branch `wip/fwd-attn`에만 한다.

## 넘길 것

- `AG3S/ag3s/docs/handoff/FWDATT.impl.md` (템플릿: `_TEMPLATE.impl.md`).
  - 바꾼 파일과 줄, 시험 결과(수치 포함), 남은 위험을 적는다.
  - 특히 XLA rounding으로 action이 달라질 수 있는 정도를 적는다.
- 끝나면 lead에게 알린다.
