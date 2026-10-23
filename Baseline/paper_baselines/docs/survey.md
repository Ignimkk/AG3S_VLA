# Survey — 구현하지 않는 비교 대상 8편 (공개 code · license · checkpoint · LIBERO 수치 · 통합 비용)

> writer: Implementer B (PAPER-BASELINES) · 확인 2026-10-23 KST · 인용 번호는 `main.pdf` (CVPR 2027 draft) References.
> **primary source 만 썼다.** arXiv abstract/HTML, 저자 project page, 저자 GitHub repo (`git ls-remote` HEAD, GitHub
> REST API `license` · 파일 목록), Hugging Face model API. 수치는 논문 HTML 의 표에서 옮겼다. 표 추출은 WebFetch 요약을
> 거쳤으므로, 논문 본문에 인용하기 전에 PDF 의 표와 한 번 더 대조해야 한다 (§4).
> 이 문서의 어떤 항목도 구현하지 않았다.

## 1. 한눈에

| # | 논문 | 공개 code (2026-10-23) | HEAD SHA | license | checkpoint | SafeLIBERO / LIBERO 보고 | 우리 protocol 통합 비용 |
|---|---|---|---|---|---|---|---|
| [1] | Agarwal & Raghunathan, Multi-link safety filtering… (arXiv:2609.40007 v1, 2026-09-30) | **없음.** project page 에 "Code (coming soon)". `yathAg/multilink-safety-filter` 는 project page 만 담은 repo | (page repo) `cef126dd…` | page repo 에 LICENSE 없음 | 자체 weight 없음. `pi05_libero` 공개 checkpoint 사용 | **SafeLIBERO 있음** (32 scene, 정적) + 자체 moving-hazard 6 조건 | 높음 (재구현 필요) |
| [5] | English, Zheng, Ewetz, Neuro-symbolic safety guidance… constrained flow matching (arXiv:2607.01378 v1, 2026-07-01) | **있음, 단 논문·project page 에서 링크되지 않음.** `Dubascudes/VLA-Safety-ConstrainedFlowMatching` (1저자 계정. README 가 이 논문의 code 라고 밝힌다. 생성 2026-10-14) | `b91153ad…` | **없음.** LICENSE 파일이 "license … has not yet been selected" 라고 적는다 | 자체 weight 없음. `pi05_libero` 사용 (Sec. 5.1) | **SafeLIBERO 있음** (4 suite × Level I/II) | 중간 (license 미정이라 vendoring 불가) |
| [9] | Guo & Kavraki, Zero-shot reactive obstacle avoidance (NUDGE) (arXiv:2609.35231 v2, 2026-09-29) | **없음** (논문·abstract 에 URL 없음. GitHub 검색 `NUDGE` 0건, 1저자 `WeihangGuo` 공개 repo 에 해당 repo 없음) | — | — | 자체 weight 없음. "the fine-tuned π0.5 checkpoint" (이름 미기재) | **LIBERO-Object + 주입 장애물** (자체 변형). SafeLIBERO 없음 | 높음 (재구현 필요) |
| [11] | Huang et al., RoboGround (arXiv:2504.21530, CVPR 2025) | **있음.** `ZzZZCHS/RoboGround` (project page 에서 링크) + 하위 `ZzZZCHS/GR1`, `ZzZZCHS/groundingLMM` | `25a86263…` (GR1 `16b63e57…`, groundingLMM `3958ffdb…`) | MIT (RoboGround). GR1 · groundingLMM 은 GitHub license 미표기 | **없음** (README · project page 어디에도 weight 링크 없음). data 만 HF `ZzZZCHS/RoboGround_Data` | **없음.** RoboCasa 기반 자체 benchmark | 매우 높음 (GLaMM 학습 + GR-1 정책 학습) |
| [13] | Jia et al., GuidedVLA (arXiv:2605.12369, RSS 2026) | **있음.** `GuidedVLA/GuidedVLA` (README · paper 링크 일치) | `04be059e…` | Apache-2.0 (+ `LICENSE_GEMMA.txt`) | **있음.** HF `ybwowen/pi0-libero-object-depth-skill` (sha `524da832…`, π0 기반, card 에 license 없음) | **LIBERO-Plus 만** (표준 LIBERO · SafeLIBERO 없음) | 중간–높음 (학습된 정책 교체형) |
| [15] | Li et al., Lan-o3dp (arXiv:2407.00451 v3, ICRA 2025) | **있음.** `qqianfeng/lan-o3dp` (공저자 Qian Feng 계정. README 가 arXiv 링크) | `b89c2898…` | MIT | **없음** (README 는 학습 경로만 있다) | **없음.** RLBench 21 task + 실물 | 높음 (task 별 diffusion policy 학습 + RLBench → LIBERO 이식) |
| [19] | Song et al., OmniGuide (arXiv:2603.10052 v1) | **있음.** `daniilidis-group/OmniGuide` (project page 에서 링크). 실제 code 는 submodule 두 개 | `d649d698…` (openpi `663f1cb2…`, gr00t `ef38d07e…`) | 상위 repo 미표기. `OmniGuide-Code/omniguide-openpi` Apache-2.0. `omniguide-gr00t` NOASSERTION | 자체 weight 없음. π0.5 **DROID** · GR00T N1.6 | **없음.** RoboCasa (GR00T) + IsaacLab DROID / 실물 (π0.5) | 중간 (π0.5 guidance 경로는 있으나 DROID 용) |
| [24] | Vo et al., OBEYED-VLA (arXiv:2512.22519 v2, 2026-04-24) | **부분.** `UARK-AICV/DRAGON_VLA` (GitHub 설명 = OBEYED-VLA 제목. project page `uark-aicv.github.io/DRAGON_VLA` 가 같은 eprint 를 인용). **perception grounding module 만** 있고 정책·robot interface 는 없다 (README) | `bdf25556…` | **없음** (LICENSE 파일 없음, API 404) | 정책 weight 없음. 포함: `weights/yolo11l-seg.pt` | **없음.** 실물 UR10e 만 | 중간 (perception 만 쓰는 grounding baseline) |

SHA 는 2026-10-23 KST 의 `git ls-remote <repo> HEAD` 값이다. 앞 8자리만 적었고 전체는 §3 에 있다.

## 2. 항목별

### [1] Multi-link safety filtering for VLA policies around moving hazards

- **source:** arXiv:2609.40007 v1 (abs, HTML). project page https://yathag.github.io/multilink-safety-filter/
- **code:** 없음. page 에 "Code (coming soon)" 만 있다.
  - page repo `github.com/yathAg/multilink-safety-filter` 에는 `index.html`, `static/`, `README.md` 만 있다.
  - 논문은 baseline AEGIS [8] · OSCBF [13] 를 "authors' public code" 로 돌렸다고 적는다 (Sec. IV).
- **방법 요지:**
  - 학습 없는 safety shield. gripper · wrist · forearm 을 ellipsoid 5개로 덮는다.
  - reset 때 RGB-D 로 hazard keep-out ellipsoid 를 맞춘다. hazard 이름은 Qwen3-VL 로, 검출은 Grounding DINO 로 한다.
  - hazard 는 sparse optical flow 로 추적한다. 해법은 OSQP (CVXPY).
- **checkpoint:** `pi05_libero` 공개 checkpoint (Sec. IV). 자체 weight 는 없다.
- **SafeLIBERO 보고 (Table I(b), 정적, 32 scene 집계):**

  | method | collision ↓ | success ↑ | safe-success ↑ |
  |---|---|---|---|
  | π0.5 | 83.44 | 60.00 | 15.06 |
  | AEGIS | 30.75 | 64.88 | 49.75 |
  | OSCBF | 66.88 | 60.56 | 27.88 |
  | ours | 27.00 | 60.31 | 51.69 |

  moving-hazard 6 조건 평균 (Table I(a), abstract 수치): collision 65.62 → 27.27, safe-success 29.35 → 50.43.
- **통합 비용: 높음.**
  - code 가 없어 ellipsoid 5개 shield + hazard 추적을 재구현해야 한다.
  - 같은 `pi05_libero` · Panda · SafeLIBERO 라서 protocol 은 맞다. 단 32 scene 의 선택 기준을 맞춰야 한다.
  - 논문 수치를 그대로 옮겨 쓰는 비교만 지금 가능하다.

### [5] Neuro-symbolic safety guidance via constrained flow matching

- **source:** arXiv:2607.01378 v1 (abs, HTML). project page https://willenglish.tech/SafetyGuidedFlowMatching/ (동영상만. page repo `Dubascudes/SafetyGuidedFlowMatching` 는 page template)
- **code:** `github.com/Dubascudes/VLA-Safety-ConstrainedFlowMatching` @ `b91153adc2f40d0ecfd2ae9ece4b6a2aed31dc7e`
  - 1저자 William English 의 계정이다 (GitHub profile 이름 · blog `willenglish.tech`).
  - README 첫 줄이 "Code for [this paper]" 이고, π0.5 + SafeLIBERO 통합 · live viewer · tests 를 담는다.
  - **논문(v1)과 project page 는 이 repo 를 링크하지 않는다.** 공식성은 저자 계정과 README 로만 확인된다.
- **license:** 없음. LICENSE 파일 본문: "A license for this project's original contributions has not yet been selected. This file does not grant a new license to those contributions." → **가져와 쓰거나 vendoring 할 수 없다.**
- **checkpoint:** `pi05_libero` (Sec. 5.1 "We use the π0.5-LIBERO checkpoint as the base VLA model"). 자체 weight 는 없다.
- **SafeLIBERO 보고 (Table 1, Level I/II 평균, %):**

  | method | metric | Spatial | Goal | Object | Long | Avg |
  |---|---|---|---|---|---|---|
  | π0.5 | CAR | 15.25 | 23.75 | 23.00 | 12.75 | 18.69 |
  | π0.5 | TSR | 59.75 | 54.25 | 53.75 | 35.75 | 50.88 |
  | AEGIS | CAR | 75.50 | 81.50 | 74.75 | 79.63 | 77.85 |
  | AEGIS | TSR | 73.25 | 75.25 | 80.25 | 43.75 | 68.13 |
  | ours | CAR | 76.50 | 88.25 | 84.00 | 82.50 | 82.81 |
  | ours | TSR | 76.75 | 87.25 | 85.75 | 76.75 | 81.62 |

  CAR = collision avoidance rate (충돌 없는 episode 비율). 추출 메모: Table 2 (level 별) 의 overall 값 (84.00 / 81.62) 이 Table 1 과 다르다. episode 수는 본문에 없다.
- **통합 비용: 중간.**
  - 같은 `pi05_libero` + SafeLIBERO 라서 repo 의 `materialize.py` (pinned upstream + patch) 를 따로 돌리면 재현은 가능해 보인다.
  - 다만 license 가 정해지지 않았다. 이 checkout 에 code 를 넣는 것은 저자 허락 또는 license 결정 전에는 하지 않는다.
  - 우리 SafeLIBERO harness (`benchmark/SafeLIBERO`) 와 metric 정의 (CAR vs 우리 collision 기준) 를 맞추는 일이 추가로 든다.

### [9] Zero-shot reactive obstacle avoidance for generative robot policies (NUDGE)

- **source:** arXiv:2609.35231 v2 (abs, HTML)
- **code:** 없음.
  - abstract · 논문에 URL 이 없다.
  - GitHub repo 검색 `NUDGE obstacle avoidance` · `nudge differentiable geometry` 0건 (2026-10-23).
  - 1저자 `WeihangGuo` 의 공개 repo 에도 없다.
- **방법 요지:**
  - SDF gradient 를 diffusion / flow-matching 정책의 sampling 에 주입한다 (학습 없음).
  - differentiable joint-trajectory decoder 로 joint / EE 행동을 지원한다.
- **checkpoint:** "the fine-tuned π0.5 checkpoint" (LIBERO-Object 실험). 이름이 적혀 있지 않다.
- **LIBERO 보고:** LIBERO-Object 10 task 에 구형 장애물을 넣은 **자체 변형** (Table III, task 평균).

  | 조건 | method | success | contact rate |
  |---|---|---|---|
  | 장애물 1 | unguided | 30 | 13.15 |
  | 장애물 1 | OmniGuide | 30 | 12.52 |
  | 장애물 1 | RAIL | 1 | 0.00 |
  | 장애물 1 | NUDGE | 44 | 2.30 |
  | 장애물 3 | unguided | 4 | 38.26 |
  | 장애물 3 | NUDGE | 13 | 14.01 |

  추출 메모: Sec. V 본문은 장애물 1 조건을 41% vs 16% 로 적어 Table III 와 다르다. SafeLIBERO 는 없다.
- **통합 비용: 높음.**
  - 재구현이 필요하다.
  - 우리 cuRobo ESDF 를 SDF 로 그대로 쓸 수 있어서 개념적 접점은 크다. 그러나 action space 변환 (EE delta ↔ joint) 과 guidance 강도 ρ 를 다시 맞춰야 한다.

### [11] RoboGround (CVPR 2025)

- **source:** arXiv:2504.21530 (abs, HTML), project page https://robo-ground.github.io/
- **code:** `github.com/ZzZZCHS/RoboGround` @ `25a862634398bc0407bc7b36b5d8b7c61d7a502b` (MIT)
  - data 생성 code 와 robosuite / robocasa / robomimic 포함.
  - 학습은 하위 repo 에서 한다: `ZzZZCHS/groundingLMM` (GLaMM LoRA, `3958ffdb…`, license 미표기), `ZzZZCHS/GR1` (정책, `16b63e57…`, license 미표기).
- **checkpoint:** 없음. data 만 HF `ZzZZCHS/RoboGround_Data` 에 있다.
- **LIBERO 보고:** 없음. RoboCasa 기반 자체 benchmark (Sec. 5.1). pick-and-place 성공률 (Table 1):
  - GR-1: Easy 42.8 / Appearance 13.8 / Spatial 16.3 / Common-sense 11.5
  - RoboGround: 43.3 / 30.5 / 33.5 / 30.0
- **통합 비용: 매우 높음.**
  - grounding mask 를 **정책 입력**으로 쓰는 방법이라 frozen π0.5 를 바꾸지 않는 우리 setting 과 구조가 다르다.
  - 비교하려면 GLaMM grounding 만 떼어 offline grounding provider 로 쓰는 길이 있다. 그래도 weight 가 공개되지 않아 학습부터 해야 한다.

### [13] GuidedVLA (RSS 2026)

- **source:** arXiv:2605.12369 (abs, HTML). project page https://guidedvla.github.io/project_page/
- **code:** `github.com/GuidedVLA/GuidedVLA` @ `04be059e0d6bd448be5cb45fdbafc775f7eb5e38`
  - Apache-2.0. Gemma 조항은 `LICENSE_GEMMA.txt`.
  - openpi 를 확장한 PyTorch 학습 pipeline 이다.
  - 논문 본문은 "code and dataset will be open-sourced after acceptance" 라고 적었고, 현재 repo 는 공개돼 있다.
- **checkpoint:** HF `ybwowen/pi0-libero-object-depth-skill` (model sha `524da8325864da21cd7dfd7c6456d72d84fed090`, 2026-05-13)
  - **π0** 기반이고 `model.safetensors` + norm stats 를 담는다. model card 에 license 가 없다.
  - Depth Anything V3 Small 이 추가로 필요하다.
- **LIBERO 보고:** **LIBERO-Plus 만** (Table I. 표준 LIBERO · SafeLIBERO 없음).
  - π0: 77.7 / 74.1 / 61.4 / 60.1, total 68.2
  - GuidedVLA: 84.0 / 80.9 / 70.8 / 66.2, total 75.4 (Spatial / Object / Goal / Long)
- **우리와의 관계:**
  - action attention head 를 GT (Qwen3-VL point → SAM2 propagate → 사람 확인) object mask 로 **학습시켜** 특화한다.
  - 우리는 frozen head 를 읽기만 한다. "VLA 내부 attention 으로 grounding" 의 학습형 대응물이다.
- **통합 비용: 중간–높음.**
  - checkpoint 가 π0 이고 우리 정책은 π0.5 다. 같은 정책 위의 비교가 아니다.
  - object head 의 attention 을 grounding provider 로 읽으려면 그 head 의 index 와 token layout 을 이 checkpoint 에서 다시 확인해야 한다.

### [15] Lan-o3dp (ICRA 2025)

- **source:** arXiv:2407.00451 v3 (abs, HTML)
- **code:** `github.com/qqianfeng/lan-o3dp` @ `b89c2898d8548f7dfde63c44eeb733a59f0c5222`
  - MIT. 공저자 Qian Feng 계정이고 README 가 arXiv 를 링크한다. arXiv 쪽에서는 repo 를 링크하지 않는다.
- **방법 요지 (Sec. III-B, IV-B1):**
  - GPT-4 가 instruction 에서 target · obstacle 이름을 뽑는다.
  - GroundingDINO box → SAM mask → Cutie 추적 → object point cloud.
  - diffusion policy 가 target point cloud 를 조건으로 받고, 장애물 cost 를 denoising 중 추정 clean trajectory 에 걸어 회피한다 (학습 없음).
- **checkpoint:** 없음 (README 는 학습 · 평가 명령만 있다).
- **LIBERO 보고:** 없음. RLBench 21 task 평균 성공률: object-centric 3D 68.7 / 3D scene 43.6 / 2D 39.3.
- **통합 비용: 높음.**
  - task 별 diffusion policy 를 학습해야 하고 RLBench (CoppeliaSim) 기반이다.
  - 다만 "Grounded-SAM 류 segmenter 로 target / obstacle point cloud 를 만든다" 는 perception 절반은 이 STEP 의 `grounded_sam` provider (Implementer A) 가 이미 같은 구성이다.

### [19] OmniGuide

- **source:** arXiv:2603.10052 v1 (abs, HTML). project page https://omniguide.github.io/
- **code:** `github.com/daniilidis-group/OmniGuide` @ `d649d69849bfc0c56bc822cdbe1f5f8a9fad2706` (project page 에서 링크)
  - 상위 repo 는 submodule 두 개를 pin 한다 (license 미표기):
    - `OmniGuide-Code/omniguide-openpi` @ `663f1cb2021282cf673fdfd737800b04ba80c893`, Apache-2.0. π0.5 DROID serving + IsaacLab DROID client + collision / VLM-pointer guidance
    - `OmniGuide-Code/omniguide-gr00t` @ `ef38d07ead28eb3ee91b9fdbe1d7df7fa2b6bfaf`, GitHub license `NOASSERTION`. RoboCasa PnP
- **checkpoint:** 자체 weight 없음. π0.5 **DROID** (PyTorch 변환) · GR00T N1.6-3B 를 쓴다.
- **LIBERO 보고:** 없음.
  - RoboCasa (GR00T, 자체 clutter 변형) + 실물 (π0.5).
  - 도입부 수치: success 24.2 → 92.4, collision avoidance 7.0 → 93.5 (어느 설정의 집계인지는 Fig. 5 수치로 확인해야 한다. 추출된 본문에 표 값이 없다).
  - [9] 가 OmniGuide 를 LIBERO-Object 변형에서 재현한 수치가 있다 (위 [9] 표).
- **통합 비용: 중간.**
  - guidance 경로가 openpi π0.5 위에 있어 우리 serving 과 가깝다.
  - 그러나 DROID config · IsaacLab 용이다. `pi05_libero` + LIBERO MuJoCo 로 옮기는 일 (action space · observation key · 장애물 표현) 이 필요하다.
  - license 는 openpi 쪽 Apache-2.0 이라 vendoring 이 가능하다.

### [24] OBEYED-VLA (arXiv v2 2026-04-24, v1 2025-12-27)

- **source:** arXiv:2512.22519v2 (abs, HTML)
  - 논문이 적은 project page `uark-aicv.github.io/OBEYED_VLA` 는 **404** (2026-10-23).
  - GitHub repo 의 homepage 필드도 그 주소다.
  - 살아 있는 page `uark-aicv.github.io/DRAGON_VLA/` 는 같은 eprint 2512.22519 를 "Distraction-Robust …" 제목으로 인용하고 code 를 링크한다.
- **code:** `github.com/UARK-AICV/DRAGON_VLA` @ `bdf25556b65728588080cd5a7245b76d39b24591`
  - GitHub 설명 = "OBEYED-VLA: Clutter-Resistant …".
  - README: "official code for the perception grounding module". 실물 robot interface 와 action policy 는 들어 있지 않다.
  - 구성: Qwen3-VL 8B (vLLM 서빙) 로 mask 선택, YOLO11l-seg, Cutie, DepthAnythingV2.
- **license:** 없음 (repo 에 LICENSE 파일 없음, GitHub license API 404) → **vendoring 불가.**
- **checkpoint:** 정책 weight 없음. `weights/yolo11l-seg.pt` 가 repo 에 들어 있다. 논문은 π0 / π0-FAST 를 fine-tune 했다고 적고 release 언급이 없다.
- **LIBERO 보고:** 없음 (실물 UR10e 만).
- **통합 비용: 중간 (perception 만).**
  - VLM 이 고른 object mask 를 "external segmenter + VLM 선택" grounding baseline 으로 쓸 수는 있다.
  - license 가 없어 code 를 들여올 수 없다. 같은 구성 (YOLO-seg + VLM 선택) 을 우리 interface 로 따로 짜야 하고, Qwen3-VL 8B GPU 1장이 더 든다.

## 3. 전체 SHA 와 확인 명령

```
yathAg/multilink-safety-filter            cef126dde2bc53c28ea20b674771ad4f97f1bb68   (project page only)
Dubascudes/VLA-Safety-ConstrainedFlowMatching b91153adc2f40d0ecfd2ae9ece4b6a2aed31dc7e
Dubascudes/SafetyGuidedFlowMatching       894bca62b18ca1e92be922089d4fb5086eccd8f3   (project page only)
ZzZZCHS/RoboGround                        25a862634398bc0407bc7b36b5d8b7c61d7a502b
ZzZZCHS/GR1                               16b63e577e09d1c2025d996d6d891003e790882d
ZzZZCHS/groundingLMM                      3958ffdbd0f2f20bb5bfa9fc2a43cb2320135218
GuidedVLA/GuidedVLA                       04be059e0d6bd448be5cb45fdbafc775f7eb5e38
qqianfeng/lan-o3dp                        b89c2898d8548f7dfde63c44eeb733a59f0c5222
daniilidis-group/OmniGuide                d649d69849bfc0c56bc822cdbe1f5f8a9fad2706
OmniGuide-Code/omniguide-openpi           663f1cb2021282cf673fdfd737800b04ba80c893
OmniGuide-Code/omniguide-gr00t            ef38d07ead28eb3ee91b9fdbe1d7df7fa2b6bfaf
UARK-AICV/DRAGON_VLA                      bdf25556b65728588080cd5a7245b76d39b24591
HF ybwowen/pi0-libero-object-depth-skill  524da8325864da21cd7dfd7c6456d72d84fed090
```

`git ls-remote https://github.com/<repo> HEAD`, `curl https://api.github.com/repos/<repo>` (license),
`curl https://api.github.com/repos/<repo>/contents/` (파일 목록), `curl https://huggingface.co/api/models/<id>`.
code 가 없다고 판정한 항목은 다음을 확인했다.
- 논문 abstract / Comments / HTML 본문
- project page 의 링크
- GitHub repository 검색 API
- 1저자 공개 repo 목록

## 4. 주의

- 표 수치는 arXiv HTML 을 WebFetch 로 요약 추출한 것이다. 논문 원문에도 자기모순이 있다 ([5] Table 1 vs 2, [9] Table III vs Sec. V). 논문 Tab. 에 옮기기 전에 PDF 로 다시 대조한다.
- SafeLIBERO 를 보고한 것은 [1] · [5] 둘뿐이다. 둘 다 `pi05_libero` 를 쓴다. 우리 protocol 과 직접 비교할 수 있는 후보는 이 둘이다. 단 [1] 은 code 가 없고 [5] 는 license 가 없다.
- arXiv 2609.xxxxx 처럼 최근 번호는 v2 이상이 나올 수 있다. 확인한 version 을 위 표에 적었다.
