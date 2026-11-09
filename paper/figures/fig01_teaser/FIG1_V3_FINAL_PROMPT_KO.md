# Fig. 1 v3 최종 제작 프롬프트 (기존 Fig. 1 + Fig. 2 병합)

- 작성: 2026-11-09 KST. 이 문서 전체를 새 Claude Code 세션에 그대로 붙여 넣어 쓴다.
- 작업 경로: `/mnt/dev/AG3S_VLA` (branch `feature/panda-benchmark`). 논문 규칙은 `.claude/rules/cvpr-workspace.md`와 `paper/CLAUDE.md`를 따른다.

---

## 0. 너의 역할과 범위

CVPR 2027 논문 "AG3S: Grounding Target Surfaces for Selective Contact in Vision-Language-Action Policies"의 **Fig. 1을 최종 형태로 만든다.** 기존 Fig. 1(teaser)과 기존 Fig. 2(세 축 개념 비교)를 한 장으로 합치는 작업이다. 그림 PDF · PNG · sidecar JSON, 합성 스크립트, LaTeX wrapper와 caption까지 만든다.

하지 않는 것:
- 기존 연구 코드(`AG3S/`, `experiments/`, `benchmark/`, `tests/`)와 `outputs/`는 읽기만 한다. 새 코드는 `paper/tools/plotting/`에만 쓴다.
- 기존 그림 파일(`paper/figures/cvpr_v1/*`)은 덮어쓰지 않는다. 새 산출물은 `paper/figures/fig01_teaser/v3/`에 둔다.
- GPU 작업, 새 rollout, segmenter 재실행은 하지 않는다(필요하면 §9에 적고 사용자에게 넘긴다).
- `cvpr.sty`, margin, font size는 바꾸지 않는다.

## 1. 이 그림의 목적

**한 문장 메시지:** 하나의 frozen VLA가 자기 내부 표현으로 "무엇을 만져도 되는지(identity), 그 표면이 어디서 끝나는지(extent), 어느 link가 언제 닿아도 되는지(permission)"를 정하고, 그래서 기존 방식이 놓치는 접촉을 막는다.

독자가 그림 한 장에서 가져가야 할 것:
1. **문제:** 정책 단독은 장애물을 밀고 지나가며 task를 끝낸다(위쪽 줄).
2. **차이:** 기존 안전 방식은 세 축 각각에서 실패한다. 외부 segmenter는 다른 물체를 고르거나 로봇을 따라가고, ellipsoid는 target과 이웃을 한 덩어리로 덮고, 손 끝 ellipsoid만 보는 filter는 쥔 물체를 모른다.
3. **우리 방식:** 같은 세 축에서 정책 내부 attention, depth로 경계 지은 표면, link · 단계별 접촉 허용으로 이를 해결한다.

왜 Fig. 1이 이 역할인가: CVPR 2026 참고 논문 4편(ACoT-VLA, AVA-VLA, ActiveVLA, ForceVLA2)의 Fig. 1은 모두 "기존 방식 vs 우리 방식" 개념 비교이고, 결과 수치를 넣지 않는다. 방법 구조는 Fig. 2 한 장이 맡는다. AVA-VLA Fig. 1(b)처럼 실제 장면의 정성 비교를 붙이는 것도 관례 안이다. 근거 표는 `paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md` §1.

세 축은 논문의 논증 단위와 1:1이다. 독자가 이 그림의 틀로 이후 절과 표를 읽게 된다.

| 축 | 방법 절 | 검증 |
|---|---|---|
| 1 Identity | 3.2 seeds, 3.3 memory | Tab. 3 seeds · memory 행, Tab. 4 target-ID · switch |
| 2 Extent | 3.2 depth-continuous surface | Tab. 3 extent 행, Tab. 4 precision, Tab. 1 external 행 |
| 3 Permission | 3.4 stage · contact links · attached object | Tab. 3 policy 행, Tab. 2 AAG, Tab. 1 AEGIS 잔여 충돌 |

이 그림이 **하지 말아야 할 것**: 결과 수치(IoU, 성공률, %)를 넣지 않는다. 파이프라인 흐름도를 넣지 않는다(새 Fig. 2 몫). 실제로 일어나지 않은 장면을 합성하지 않는다.

## 2. 병합의 정의: 기존 두 그림에서 무엇을 남기고 무엇을 버리나

| 출처 | 파일 | 요소 | 처리 |
|---|---|---|---|
| 기존 Fig. 1 | `paper/figures/cvpr_v1/fig01_teaser_noCL.pdf` (wrapper `paper/figures/fig01_teaser/figure_ifg_textwidth.tex`) | VLA only vs VLA + AG3S 장면(RB-Y1 hurdle) | **역할만 유지:** 위쪽 장면 줄로, SafeLIBERO 최종 run 장면으로 교체 |
| 기존 Fig. 1 | 같음 | 아래 4단계(seeds → surface → features → contact links) | 삭제. 새 Fig. 2(파이프라인)가 담당 |
| 기존 Fig. 1 | 같음 | 오른쪽 "Measured" 막대(0.773 / 0.891 / −92 %) | 삭제. 수치는 표가 담당 |
| 기존 Fig. 2 | `paper/figures/cvpr_v1/fig02_overview.pdf` (wrapper `paper/figures/fig02_overview/figure_cvpr_v1.tex`) | 위쪽 흐름 줄(RGB → frozen VLA → … → execute / hold) | 삭제. 새 Fig. 2와 중복 |
| 기존 Fig. 2 | 같음 | 아래 세 칸(identity · extent · permission, Prior vs AG3S), 칸 제목과 부제, badge 체계 | **뼈대로 유지.** 이미지만 §4로 교체 |
| 기존 Fig. 3 | `paper/figures/cvpr_v1/fig03_pipeline.pdf` | 방법 파이프라인 | 손대지 않음. 번호만 새 Fig. 2가 됨 |

## 3. 레이아웃 사양

```
 "pick up the black bowl ... and place it on the plate"   (SafeLIBERO Level I · same initial state and seed)
 [VLA only: bottle pushed over ✗] [VLA + AG3S: bottle untouched ✓] [VLA + AG3S: bowl placed ✓]     ← 위쪽 줄 (선택)

 ┌ 1 What: identity ───────┐ ┌ 2 Where it ends: extent ┐ ┌ 3 Who and when: permission ┐
 │ the noun's attention    │ │ depth continuity sets   │ │ contact links only, in the │
 │ names the object        │ │ the boundary            │ │ allowed stage              │
 │ Prior: external segmenter│ │ Prior: ellipsoid / mask │ │ Prior: end-effector filter │
 │ [이미지]  ✗ badge        │ │ [이미지]  ✗ badge        │ │ [이미지]  ✗ badge           │
 │ AG3S (ours)             │ │ AG3S (ours)             │ │ AG3S (ours)                │
 │ [이미지]  ✓ badge        │ │ [이미지]  ✓ badge        │ │ [이미지]  ✓ badge           │
 └─────────────────────────┘ └─────────────────────────┘ └────────────────────────────┘
```

- 폭 6.875 in(CVPR text width, 축소 없이 이 크기로 저작). 높이 목표: 위쪽 줄 포함 ≤ 2.9 in, 제외 ≤ 2.5 in. 두 변형을 모두 만들어 사용자가 고르게 한다(`top_strip=True/False`).
- 1쪽 제목 블록 아래 전폭(현재 `main.tex`의 `\g@addto@macro\@maketitle` 방식 유지).
- 글자 최소 7 pt(인쇄 크기 기준). 벡터 PDF, raster 패널 400 dpi 삽입, 폰트 TrueType 내장(`pdf.fonttype = 42`).
- 색 규약(기존 `cvpr_v1`과 동일): 주황 = 선택 target, 파랑 = 보호 이웃, 금색 = seed / contact link, 보라 = attached object. Prior badge 빨강 ✗, Ours badge 초록 ✓.
- badge는 7단어 이내. 긴 설명은 LaTeX caption으로.
- **모든 패널은 정사각형 crop, 같은 크기.** 잘라낸 Prior 패널에 남은 원본 라벨(물체 이름 태그, 범례 글씨)은 7 pt 미만이면 가리거나 crop에서 제외한다. 남길 라벨은 벡터로 다시 쓴다(`cvpr_v1/NOTES.md` 4차 패스의 inpaint 방식 참고).
- 저장 시 `layout_audit()` 위반 0(`paper/tools/plotting/cvpr_v1_compose.py`의 `Fig.layout_audit`, `CVPR_V1_STRICT=1`).

## 4. 칸별 이미지와 위치

모든 자산은 `/mnt/dev/AG3S_VLA/paper/figures/fig01_teaser/v3/assets/`에 있다(아래는 그 폴더 기준 파일 이름). 각 파일의 출처 run · frame · chunk · 원본 sha256은 같은 폴더의 `assets.json`. 자산 생성 스크립트는 `paper/tools/plotting/fig1_v3_assets.py`(다른 episode가 필요하면 `--only render --task T --ep E`).

### 위쪽 줄 (선택) — 최종 run, 실제 장면

장면: SafeLIBERO spatial Level I, task 2, episode 9, seed 7, initial state 9, 장애물 와인병. VLA 단독 run `results/benchmark/*qe-vla-e1a-spatial-LI-t2-*`, 최종 AG3S run(pe123458, GT 없음) `results/benchmark/*qe-pe123458-e1a-spatial-LI-t2-*`.

| 슬롯 | 1순위 | 대안 | 내용 |
|---|---|---|---|
| VLA only | `top_vla_step111.png` | `top_vla_step146.png` | step 86에 첫 공식 충돌, step 111에 병이 쓰러져 접시 위 |
| VLA + AG3S | `top_ag3s_step111.png` | `top_ag3s_step086.png` | 같은 step, 병은 서 있고 그릇을 운반 |
| VLA + AG3S (결과) | `top_ag3s_step139.png` | — | 그릇을 접시에 놓음, 안전 성공 |

같은 조건(VLA 단독 밀고 성공 · AEGIS 충돌 · AG3S 안전 성공)을 만족하는 대안 episode: task 2 episode 11, task 3 episode 5.

### 칸 1 Identity

| 행 | 1순위 | 대안 | 내용 · 주의 |
|---|---|---|---|
| Prior | `crop_grounded_sam2_track__v2-sl3-track-strip_r0c0.png` | `..._r0c1.png`–`..._r0c6.png`(같은 episode의 다른 시점), `crop_2026-11-01_segmentation-baseline__fixed_segmentation_examples_r2c1.png` | Grounded-SAM-2(per-frame, 지시문 prompt)가 같은 종류의 다른 그릇(주황 윤곽)을 target으로 칠함(파랑). LIBERO spatial t2 e0, 11-01–03 이전 컨테이너 run |
| Prior (가림 · 로봇) | `crop_sl4b__v2-sl4b-scene-overlay_r1c3.png` | `..._r1c5.png`, `..._r1c6.png` | segmenter 마스크(파랑)가 로봇 팔로 번짐. 칸 1 Prior를 두 개 쓸 공간이 없으면 위 1순위만 |
| Ours | `c1_ours_attention.png` | — | 점에 붙은 noun attention을 head 카메라에 투영한 열지도 + seed 점(금색). 최종 run t2 e9 call 2(target latched) |

### 칸 2 Extent

| 행 | 1순위 | 대안 | 내용 · 주의 |
|---|---|---|---|
| Prior | `crop_aegis__v1-aegis-scene-detection-ellipsoid_r0c1.png` | `..._r0c0.png`, `..._r0c2.png` | AEGIS 검출 ellipsoid(주황)가 target 그릇과 옆 cabinet을 한 덩어리로 덮음. LIBERO 장면(장애물 없음) |
| Prior (SafeLIBERO 대안) | `crop_2026-11-03_safelibero-seg-ats__fig4_erase_keep_maps_r1c0.png` | `..._r0c0.png`, `..._r2c0.png` (SAM-2), `..._c1` (Grounded-SAM) | SafeLIBERO 실제 frame에서 segmenter 마스크 기반 erase/keep. 색 규약이 다름(초록 = target 지움, 빨강 = target 남음, 검정 = 장애물을 잘못 지움, 주황 = 장애물 보존) → 쓰면 범례를 우리 색 규약으로 다시 칠하거나 caption에 명시 |
| Ours | `c2_ours_extent.png` | — | target-free field에서 지운 voxel(주황, 그릇 표면 2114개)과 바로 옆에서 보존한 비-target voxel(파랑, 567개). 최종 run t2 e9 call 6(파지 직전) |

### 칸 3 Permission

| 행 | 1순위 | 대안 | 내용 · 주의 |
|---|---|---|---|
| Prior | `crop_knows__v1-knows-scene-ellipsoids_r0c0.png` | `crop_multilink_cbf__v1-multilink-scene-link-ellipsoids_r0c0.png` | 손 끝 ellipsoid(초록)와 물체 ellipsoid만 있고 쥔 물체 개념이 없음. multi-link 대안은 link별 ellipsoid(파랑). LIBERO 장면 |
| Ours | `c3_ours_links.png` | — | 손 · 손가락 sphere(금색 = contact link)와 쥔 그릇의 attached sphere(보라 80개). 최종 run t2 e9 call 14(운반 중) |

### 참고용 (그림에 쓰지 않음)

- 현재 초안 시안: `paper/figures/fig01_teaser/v3/mock_fig1_v3.png` (위 1순위로 묶은 배치 확인용, 라벨 · 크기 미조정)
- 기존 두 그림의 자산(교체 대상): `paper/figures/cvpr_v1/assets/`, `paper/figures/paperbanana/figure2/assets/`, `paper/figures/ifg_overview/assets/` — 모두 RB-Y1 개발 run · 예시 렌더

## 5. 제작 방법

1. `paper/tools/plotting/cvpr_v3_fig1_compose.py`를 새로 만든다. `cvpr_v1_compose.py`의 `fig02()` 카드 레이아웃, badge 함수, `Fig.layout_audit()`, 폰트 · DPI 설정을 import하거나 복사해 쓴다(원본 수정 금지).
2. 함수 `fig01_v3(top_strip: bool)`: §3 레이아웃, §4 1순위 이미지. 출력은 `paper/figures/fig01_teaser/v3/fig01_v3.{pdf,png,json}`과 `fig01_v3_notop.{pdf,png,json}`.
3. sidecar JSON에 패널마다 원본 파일 경로, sha256, crop 상자, 출처 run · episode · step · chunk(`assets.json`에서 복사), 패널 성격(`recorded run` / `previous-container figure crop`)을 적는다. 그림에 인쇄된 글자 목록과 최소 글자 크기(`fonts.min_pt`)도 기록한다.
4. 렌더 후 PNG를 직접 열어 확인한다: 7 pt 이하 글자 없음, badge가 패널을 가리지 않음, 세 칸 높이 같음, 색 규약 일치.

## 6. LaTeX 반영

1. 새 wrapper `paper/figures/fig01_teaser/figure_v3.tex` (기존 `figure_ifg_textwidth.tex`와 같은 minipage 형식, `\label{fig:teaser}`).
2. `paper/latex/main.tex`: 제목 블록의 `\input{../figures/fig01_teaser/figure_ifg_textwidth}`를 `figure_v3`로 바꾸고, `\input{../figures/fig02_overview/figure_cvpr_v1}` 줄을 지운다. **먼저 사용자의 최신 main.tex를 확인한다**(1차 초안 PDF `paper/AG3S_ Grounding Target Surfaces ... (2).pdf`는 이미 기존 Fig. 2를 Fig. 1 자리에 둔 상태라, 로컬 main.tex와 다를 수 있다).
3. 참조 정리: `sections/01_introduction.tex` 10행의 `\ref{fig:overview}`와 `\ref{fig:teaser}`를 `fig:teaser` 하나로, `sections/03_method.tex` 58행의 `\ref{fig:overview}`(identity 칸의 cosine 예)는 칸 라벨에서 점수를 지웠으므로 "(Fig.~\ref{fig:teaser}, identity)"로 문장 조정, `exp_section/04_experiments.tex` 159행 "(Fig.~1, panel~d)" → "(Fig.~\ref{fig:teaser}, identity)". `exp_section/build_experiments.tex`의 `\setcounter{figure}{3}` → `{2}`.
4. 빌드 후 쪽수와 Fig. 번호(새 Fig. 2 = 파이프라인, 실험 실로봇 그림 = Fig. 3)를 확인한다.

Caption 초안(실제 패널에 맞춰 다듬는다):

> **Figure 1. One frozen VLA decides what may be touched, where that surface ends, and by which link, when.** *Top:* on a SafeLIBERO scene with a bottle beside the bowl, the frozen π0.5 completes "place the bowl on the plate" by knocking the bottle over; with AG3S-VLA, on the same initial state and seed, it carries the bowl past the bottle and places it. *Bottom, each against the closest prior practice.* **Identity:** an external segmenter prompted with the instruction's noun marks a similar bowl; the policy's own noun attention, lifted to the observed points, seeds the bowl it acts on. **Extent:** an ellipsoid around the detected object covers the target together with what touches it; depth continuity bounds the bowl's observed surface (orange) and keeps the adjacent geometry protected (blue). **Permission:** a filter on an end-effector ellipsoid has no notion of the carried object; AG3S-VLA frees the target surface only for the contact links (gold) and, once grasped, protects the bowl as attached geometry (purple). Prior panels are outputs of the baseline implementations on LIBERO scenes; the top row and our panels come from one recorded SafeLIBERO episode.

## 7. 정직성 제약 (반드시 지킴)

- Prior 패널은 11-01–11-03 이전 컨테이너 run의 그림에서 잘라낸 것이고 위쪽 줄과 **다른 장면**이다. caption에 그렇게 쓴다("same scene"이라고 쓰지 않는다).
- 칸 2 · 3 Prior의 ellipsoid 그림은 장애물이 없는 LIBERO 장면이다.
- 칸 1 Ours는 점별 attention 값이다. 기록된 16×16 `attention_head` 지도는 점별 값과 상관이 없어(|r| < 0.03, 원인 미확인) 쓰지 않았다. 이 지도를 쓰지 않는다.
- 위쪽 줄의 VLA 단독 run(11-08)과 AG3S run(11-09)은 commit이 다르다. seed 7과 initial state 9는 같다. sidecar에 두 run 이름을 모두 기록한다.
- 카메라 원본 해상도는 256 px다. 3배 확대본(768 px)을 쓰며, 그 이상 선명하게 보정하지 않는다.
- 그림 안에 수치를 넣지 않는다.

## 8. 완료 기준

- [ ] `fig01_v3.pdf` · `fig01_v3_notop.pdf`와 각 PNG · JSON 생성, `layout_audit` 위반 0, 최소 글자 ≥ 7 pt
- [ ] 패널 9개(또는 위쪽 줄 제외 6개) 모두 §4의 파일에서 왔고 sidecar에 출처 · sha256 기록
- [ ] caption이 §7 제약을 모두 반영
- [ ] main.tex · 참조 수정 후 전체 원고 빌드 성공, undefined reference 0, 그림 번호 확인
- [ ] 높이(in)와 원고 쪽수 변화를 실측해 보고 (구 Fig. 1 + Fig. 2 = 5.97 in 대비)
- [ ] `paper/figures/fig01_teaser/TEASER_V2_SAFELIBERO_PLAN_KO.md` §5.1과 `paper/evidence/PAPER_LOG.md`에 결과 기록

## 9. 사용자에게 물을 것 (작업은 기본값으로 진행하고 마지막에 보고)

1. 위쪽 장면 줄을 넣을지(기본: 두 변형 모두 제작, 선택은 사용자).
2. Prior를 다른 장면의 이전 run 그림으로 둘지, 위쪽 줄과 같은 episode에서 baseline segmenter · ellipsoid를 offline으로 다시 돌릴지(기본: 이전 run 그림. 재실행은 GPU 작업이라 이 세션에서 하지 않음).
3. 칸 2 Prior를 AEGIS ellipsoid(LIBERO)로 할지 SafeLIBERO erase/keep 지도로 할지(기본: AEGIS ellipsoid, 색 규약 충돌이 없어서).
