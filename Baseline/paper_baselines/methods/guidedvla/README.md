# guidedvla — GuidedVLA [13] (fetch + README 만)

| 항목 | 내용 |
|---|---|
| 원 논문 | Jia et al., *GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization* (RSS 2026), arXiv:2605.12369 |
| 출처 repo | `GuidedVLA/GuidedVLA@04be059e0d6bd448be5cb45fdbafc775f7eb5e38` (openpi 확장, PyTorch 학습 pipeline. submodule `third_party/LIBERO-plus` 는 fetch 하지 않았다) |
| clone 위치 | `Baseline/paper_baselines/external/GuidedVLA` (untracked) |
| license | **Apache-2.0** (+ `LICENSE_GEMMA.txt`). checkpoint 의 model card 에는 license 가 없다 |
| 상태 | **실행하지 않는다.** fetch 와 이 README 까지만 했다 |

## 방법과 입력

- π0 action expert 의 attention head 일부를 ControlNet 식 branch (zero-init) 로 특화시킨다.
  - object head: GT object mask 에 attention 을 맞추는 NLL loss. `object_head_indices=[0, 1]`.
  - skill head: sub-skill 분류 KL loss. `skill_head_indices=[6, 7]`, 4 class.
  - depth head: frozen **Depth Anything V3 Small** feature 를 key / value 로 넣는다.
- 추론 입력은 π0 LIBERO 와 같다 (agentview + wrist RGB, 8D state, prompt). object mask 와 skill label 은 **학습할 때만** 필요하다 (`ybwowen/libero` dataset 의 `agentview_attention_object_mask` 등).
- 보고 benchmark: **LIBERO-Plus 만** 있다 (π0 68.2 → 75.4). 표준 LIBERO 와 SafeLIBERO 결과는 없다.

## checkpoint

- HF `ybwowen/pi0-libero-object-depth-skill`, model sha `524da8325864da21cd7dfd7c6456d72d84fed090`.
  - config `pi0_libero_object_depth_skill`.
  - `model.safetensors` 와 `assets/ybwowen/libero/norm_stats.json` 이 들어 있다.
- **π0 기반이다** (우리 policy 는 π0.5). DA3-Small checkpoint 를 따로 받아야 한다.

## 우리 protocol 에 넣으려면

1. 별도 환경을 만든다: GuidedVLA 의 `uv sync` 와 transformers patch 복사. 이 checkout 의 openpi 와 venv 에는 섞지 않는다.
2. checkpoint 와 DA3-Small 을 `AG3S_STORAGE_ROOT` 에 받는다.
3. LIBERO 와 SafeLIBERO client 가 이 server 를 부르게 한다 (`examples/libero/main.py` 와 같은 websocket protocol).
4. grounding provider 로 쓰려면 object head (index 0, 1) 의 attention map 을 읽는 hook 을 붙인다. 그 head 의 token layout 은 이 checkpoint 에서 다시 확인해야 한다.
5. 같은 policy 위의 비교가 아니다 (π0 vs π0.5). 비교표에서 그 점을 명시해야 한다.
