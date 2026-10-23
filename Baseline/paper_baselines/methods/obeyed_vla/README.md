# obeyed_vla — OBEYED-VLA [24] (= DRAGON-VLA perception module, fetch + README 만)

| 항목 | 내용 |
|---|---|
| 원 논문 | Vo et al., *OBEYED-VLA* (arXiv:2512.22519 v2, 2026-04-24). 같은 eprint 를 project page 는 "DRAGON-VLA: Distraction-Robust …" 로 인용한다 |
| 출처 repo | `UARK-AICV/DRAGON_VLA@bdf25556b65728588080cd5a7245b76d39b24591`. GitHub 설명 = "OBEYED-VLA: Clutter-Resistant …" |
| clone 위치 | `Baseline/paper_baselines/external/DRAGON_VLA` (untracked. `weights/yolo11l-seg.pt` 를 포함해 약 233 MB) |
| license | **license unknown** (LICENSE 파일이 없고 GitHub license API 는 404) → **code 를 tracked 파일로 복사하지 않는다.** 읽기와 참조만 한다 |
| 상태 | **실행하지 않는다.** fetch 와 이 README 까지만 했다 |

## 방법과 입력

- repo 에는 **perception grounding module 만** 있다. robot interface 와 action policy 는 없다 (README).
- 처리 단계:
  - YOLO11l-seg 가 instance mask 후보를 만든다.
  - **Qwen3-VL 8B-Instruct** (vLLM 서빙) 가 instruction 에 맞는 mask 를 고른다.
  - Cutie 가 mask 를 추적한다.
  - DepthAnythingV2 (metric, ViT-L) 가 masked relative depth 를 만든다.
- 출력은 grounded overlay 영상이다 (`overlay_base.npy`, `overlay_wrist.npy`). 이 영상으로 clean single-object demo 에서 fine-tune 한 π0 / π0-FAST 를 conditioning 한다.
- 진입점: `perception_service_batch.py` (학습 data 전처리), `perception_service_fastapi.py` (배포).
- 평가: 실물 UR10e 만 있다. LIBERO 와 SafeLIBERO 결과는 없다.

## checkpoint

- policy weight: **없다** (release 언급 없음).
- perception: `weights/yolo11l-seg.pt` 는 repo 에 있다. DepthAnythingV2 metric 과 Cutie weight 는 따로 받아야 한다. Qwen3-VL 8B 는 GPU 1장 (A6000 급) 이 더 든다.

## 우리 protocol 에 넣으려면

1. "external segmenter + VLM 선택" grounding baseline 으로만 쓸 수 있다. 우리 `TargetMaskProvider` interface 로 같은 구성을 **새로 짜야** 한다 (license 가 없어 code 를 들여올 수 없다).
   - 구성: YOLO-seg 후보 → local VLM 선택. 외부 유료 API 는 쓰지 않는다.
2. policy 부분 (grounded overlay 로 fine-tune 한 π0) 은 weight 가 없어서 학습부터 해야 한다. frozen π0.5 setting 과 다르다.
3. 저자에게 license 를 문의하기 전에는 이 clone 의 code 를 실행 경로에 넣지 않는다.
