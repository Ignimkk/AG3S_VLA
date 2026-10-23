# BM-SETUP-B — LIBERO-Safety 평가 환경 셋업 (CPU)

> writer: lead · 읽는 쪽: implementer · 작성 2026-10-23 KST
> 사용자 지시 2026-10-23: benchmark 실험 환경을 최대한 셋업해 두고, 최종 방법론이 오면 바로 테스트한다. 진단용 GPU는 잠시 멈춘다.
> **GPU 금지.** 공용 runner file(`launch_libero_paired.py`, `libero_safe_client.py`)은 BM-SETUP-A가 고친다. 이 task는 **새 file**만 만든다(아래 3). 공용 file에 hook이 꼭 필요하면 고치지 말고 impl.md에 요청으로 적는다.

## 대상

- `benchmark/LIBERO-Safety`: submodule, commit `19ec8df`, ECCV 2026.
- 구성: suite 5개(affordance · human_safety · obstacle_avoidance · obstacle_avoidance_human · reasoning_safety) × task 5 × level L0–L2.
- README에 따르면 LIBERO와 drop-in 호환이고(`pip install -e .`), asset은 HuggingFace `LIBERO-Safety/libero_safety_assets`, π0.5 fine-tuned weight는 `LIBERO-Safety/pi05_libero_safety`에 있다.

## 무엇을 하나

1. **공식 protocol audit.** 공식 code와 문서에서 다음을 표로 정리한다. 줄 번호와 출처를 함께 적는다.
   - suite / task / level map(`vla_safety_task_map.py`)
   - step 수, wait, controller, 카메라
   - init state 수
   - 성공 판정
   - **안전 지표의 정의**(분자 · 분모 · 집계)
   - 평가에 쓴 policy와 checkpoint
2. **설치 방식 결정과 설치.** `.venv-libero`(LIBERO 공식)와 package 이름이 겹친다(`libero`). 그래서 **별도 venv `.venv-libero-safety`**를 만드는 것을 기본안으로 한다.
   - Python 버전과 requirements를 공식 문서대로 맞춘다.
   - asset download는 용량을 먼저 확인해 impl.md에 적고 `AG3S_STORAGE_ROOT/benchmarks/libero-safety/` 아래에 둔다(symlink 허용).
   - checkpoint `pi05_libero_safety`는 **download하지 않는다**(사용 여부를 사용자가 정한다). 용량과 위치만 조사한다.
3. **adapter / runner (새 file)**
   - `benchmark/runners/libero_safety_client.py`: `libero_safe_client.py`의 구조를 따른다. 같은 websocket 계약, 같은 episode 기록 형식, 같은 `safe_client` field. LIBERO-Safety env를 쓰고 공식 지표를 그대로 계산해 기록한다.
   - launcher 연결: 공용 launcher를 고치지 않는다. 필요한 인터페이스(client script 경로, suite 상수)를 impl.md에 적어 lead / A와 맞춘다. 임시로 쓸 수 있는 얇은 wrapper를 새 file로 둬도 된다.
   - Panda camera / action 규약이 LIBERO와 같은지 확인한다(SL1–SL2 규약, image 방향, depth, K).
   - human 모델 같은 새 물체가 AG3S 관측(segmentation id, 물체 이름)에서 어떻게 보이는지 확인한다.
4. **CPU 확인 (policy 없음):** suite × level마다 1 task 이상, env 생성 → reset → dummy action → 관측 변환 → 공식 안전 판정이 서는지 본다. 일부러 위반하는 dummy 동작을 넣어도 된다.
5. **GPU smoke 목록:** VLA-only 1 episode / V11AB 1 episode를 suite × level마다 jobs로 만들어 두고 **실행하지 않는다.**
6. **dev/test 분리 제안:** 확정은 사용자가 한다.

## 넘겨줄 것

- `AG3S/ag3s/docs/handoff/BM-SETUP-B.impl.md`
  - protocol audit 표, 설치 기록(venv · 용량 · 경로), CPU 확인 표, 공용 runner에 필요한 hook 요청
  - checkpoint 선택지와 비교: `pi05_libero`를 쓸지 `pi05_libero_safety`를 쓸지. 공식 결과와 비교할 수 있는지가 기준이다.
  - GPU smoke 목록과 split 제안
- raw: `outputs/verify/BM-SETUP-B/`
- commit하지 않는다.

## 규칙

- GPU 금지. CPU worker ≤ 2. 다른 세션 process는 건드리지 않는다.
- 이전 `/mnt/dev/work` checkout을 쓰지 않는다.
