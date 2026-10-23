# BM-SETUP-A — LIBERO 전체 suite · SafeLIBERO 전체 suite 평가 환경 셋업 (CPU)

> writer: lead · 읽는 쪽: implementer · 작성 2026-10-23 KST
> 사용자 지시 2026-10-23: "LIBERO, SafeLIBERO, LIBERO-Safety 등 benchmark에서 실험할 환경을 최대한 셋업해 놓고, 최종 방법론이 도착하면 바로 테스트할 수 있도록 하자. 진단을 위한 GPU 사용은 잠시 멈춘다."
> **GPU 금지.** 이 단계는 CPU에서 확인할 수 있는 것만 한다. VLA를 넣은 closed-loop smoke는 GPU가 필요하므로 목록으로만 만들어 둔다(아래 5).
> LIBERO-Safety는 BM-SETUP-B가 따로 맡는다. 같은 file을 고치지 않는다. 공용 file(`launch_libero_paired.py`, `libero_safe_client.py`)은 이 task(A)만 고친다.

## 지금 상태 (lead 확인)

- runner(`benchmark/runners/launch_libero_paired.py` + `libero_safe_client.py`)로 실제로 돌려 본 것은 `libero_spatial`과 `safelibero_spatial` Level I/II뿐이다.
- client에는 `SAFELIBERO_TOTAL_STEPS`에 4 suite(spatial · object · goal · long)가 있다. LIBERO 쪽은 `base.MAX_STEPS`를 쓴다.
- 확정된 dev/test 분리: LIBERO Spatial init 0–4 / 5–49, SafeLIBERO Spatial task 0–3 × Level I/II × episode 0–4 / 5–49. 그 밖의 suite는 아직 정하지 않았다(plan §6.0: "해당 단계 시작 전에 정한다").

## 무엇을 하나

1. **LIBERO 4 suite(spatial · object · goal · 10) 확인 (CPU, policy 없음)**
   - task마다 env 생성 → init state 0 reset → dummy action 몇 step → 관측을 adapter 형식으로 변환한다. image 방향, depth, K / extrinsics, eef state, gripper가 SL1–SL2와 같은 규약인지 본다.
   - suite별 max_steps와 num_steps_wait의 출처를 기록한다(공식 openpi LIBERO example 기준).
   - adapter 가정 중 spatial에만 맞는 것이 있는지 찾는다. 예: fixture / reset 이력(SL4b), segmentation id, 물체 이름 규칙, target 추출 규칙. 표로 낸다.
   - `libero_90`은 확인만 하고 범위 밖으로 표시한다.
2. **SafeLIBERO 4 suite × Level I/II 확인 (CPU)**
   - task마다 env 생성 → reset → obstacle 배치와 이름 → 충돌 판정기(SL5c, README 규칙)가 해당 suite에서 동작하는지 본다. dummy action으로 일부러 obstacle에 부딪혀 판정이 서는지도 본다. 가능하면 suite당 1 task.
   - total_steps / wait를 공식 `main_aegis.py`와 대조한다.
3. **runner를 모든 suite에 열기**
   - launcher와 client가 위 8개 suite × level을 인자로 받아 돌 수 있게 한다.
   - suite별 protocol(steps · wait · split · 성공 판정)을 한 곳에 표로 둔다. 예: `benchmark/configs/suites.json` 또는 기존 상수 정리.
   - **spatial 기존 동작은 byte 단위로 그대로 둔다.** settings, protocol, argv 동일성 test를 고정한다.
4. **"최종 방법론이 오면 바로 돌리는" 평가 recipe**
   - `benchmark/runners/eval_matrix.py`(이름은 바꿔도 된다): 입력은 version 이름(AG3S_VERSIONS key), 변환, suite × level × task × episode 목록, split(dev/test).
   - 이것으로 `run_parallel_gpu.sh` jobs 파일을 만들고, 끝난 뒤 결과를 집계한다.
   - 집계 항목: suite별 official success, SafeLIBERO의 공식 지표(CAR / TSR / ETS 등은 README / 공식 code의 정의 그대로 확인)와 우리 공통 지표(collided, safe success), paired 전이(VLA-only 대비), episode 단위 CI.
   - VLA-only 결과 재사용 규칙(SL5g pairing basis 방식)을 recipe에 넣는다.
   - **새 method를 version으로 등록하는 절차**(profile key 추가 → recipe 실행)를 README 한 장으로 남긴다: `benchmark/docs/EVAL_RECIPE.md`.
5. **GPU smoke 목록**: suite × level마다 VLA-only 1 episode, V11AB 1 episode를 jobs 파일로 만들어 두고 **실행하지 않는다.** 예상 시간도 적는다.
6. **dev/test 분리 제안**: spatial 외 suite에 대해 plan §6.0과 같은 방식(init / episode 0–4 dev, 5–49 test)으로 **제안**만 표로 낸다. 확정은 사용자가 한다.

## 넘겨줄 것

- `AG3S/ag3s/docs/handoff/BM-SETUP-A.impl.md`:
  - suite별 확인 표(생성 / reset / 관측 / 충돌 판정 / protocol 출처)
  - adapter 가정 표
  - 바뀐 file과 spatial 동일성 근거
  - recipe 사용법
  - GPU smoke 목록
  - split 제안
- raw: `outputs/verify/BM-SETUP-A/`
- commit하지 않는다.

## 규칙

- GPU 금지. CPU worker ≤ 2. pod working set이 62 GiB를 넘으면 멈춘다.
- 다른 세션 process는 건드리지 않는다.
- 외부 download가 필요하면(asset 등) 무엇을 얼마나 받는지 먼저 적고 진행한다. 저장 위치는 `AG3S_STORAGE_ROOT` 아래로 한다.
