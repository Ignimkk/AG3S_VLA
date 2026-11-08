# Fig. 1 v3 — 개념 비교 그림 (기존 Fig. 1 + Fig. 2 병합) 구성안

- 작성: 2026-11-08 KST(v2), **2026-11-09 KST v3로 개정**. 상태: `PLANNED` (제작 전).
- v3 결정 (2026-11-09, 사용자 동의): 기존 Fig. 1(RB-Y1 hurdle teaser, `cvpr_v1/fig01_teaser_noCL`)과 기존 Fig. 2(세 축 개념 비교, `cvpr_v1/fig02_overview`)를 **한 장의 Fig. 1로 합친다.** 기존 Fig. 3(파이프라인, `cvpr_v1/fig03_pipeline`)은 Fig. 2가 된다. 실험 전 그림이 세 장에서 두 장으로 줄어든다.
- v2(π0.5 / AEGIS / AG3S 세 열 + AG3S 내부 패널 (d))는 폐기한다. 그 장면 선택 기준과 자료 목록은 v3의 위쪽 줄과 각 칸의 "Prior" 이미지에 재사용한다(§4, §5).

## 1. 근거: CVPR 2026 참고 논문 4편의 그림 구성

2026-11-09에 `papers/cvpr2026/*.pdf`에서 caption과 쪽 배치를 직접 확인했다.

| 논문 | Fig. 1 (1쪽) | Fig. 2 | 실험 절 그림 |
|---|---|---|---|
| ACoT-VLA | 언어 CoT · 시각 CoT · 우리 action CoT 비교 (단일 열) | 전체 구조 (전폭) | 실로봇 막대그래프, 실로봇 장면 |
| AVA-VLA | (a) 기존 VLA vs 우리 구조, (b) 같은 작업에서 baseline vs 우리 attention 정성 비교 (단일 열) | 전체 구조 (전폭) | 실로봇 장면 + 막대그래프, attention 가중치 변화 |
| ActiveVLA | 고정 카메라로 가려 실패하는 기존 방식 vs 시점을 바꿔 성공하는 우리 방식 (전폭) | 파이프라인 (전폭) | 단계별 정성, 실로봇, hyperparameter 그래프 |
| ForceVLA2 | 왼쪽 기존 입력(시각 · 상태) vs 오른쪽 힘 인식을 더한 우리 방식 (전폭) | 전체 구조 (전폭) | 데이터셋, 수집 장치, baseline 대비 정성, 추가 시험 |

공통 규칙:
1. 실험 전 그림은 **두 장**이다. Fig. 1 = "기존 방식이 왜 실패하고 우리는 무엇이 다른가"의 개념 비교, Fig. 2 = 방법 구조 하나.
2. Fig. 1의 비교 축은 그 논문의 **핵심 주장 축**을 따른다. AVA-VLA처럼 실제 baseline 출력을 정성 패널로 붙이기도 한다.
3. **결과 수치를 Fig. 1에 넣은 논문은 없다.** 수치는 실험 절의 표와 그림이 맡는다.

우리 원고(2026-11-08 17:47 빌드)는 실험 전 그림이 세 장이고 역할이 겹친다.

| 현재 그림 | 내용 | CVPR 기준 역할 | v3 처리 |
|---|---|---|---|
| Fig. 1 위 2×2 장면 | VLA only vs VLA + AG3S, pick / carry | 문제 제시 | **위쪽 줄로 축소해 유지(선택)** |
| Fig. 1 아래 4단계 | seeds → surface → features → contact links | 방법 구조 | 삭제 (새 Fig. 2가 담당) |
| Fig. 1 오른쪽 "Measured" 막대 | IoU · leakage 수치 | 결과 | 삭제 (수치는 Tab. 1 · Tab. 4) |
| Fig. 2 위쪽 흐름 줄 | RGB → frozen VLA → … → execute / hold | 방법 구조 | 삭제 (새 Fig. 2와 중복) |
| Fig. 2 아래 세 칸 | identity · extent · permission, Prior vs AG3S | **CVPR Fig. 1의 역할 그 자체** | **새 Fig. 1의 뼈대** |
| Fig. 3 | 파이프라인 | CVPR Fig. 2의 역할 | 새 Fig. 2로 번호만 변경 |

## 2. 왜 세 칸을 뼈대로 삼는가

세 축은 논문의 논증 단위와 1:1로 대응한다. 그래서 Fig. 1을 본 독자가 이후 절과 표를 같은 틀로 읽는다.

| 축 | 질문 | 기여 (1장) | 방법 절 | 검증 (4장) |
|---|---|---|---|---|
| 1 Identity | 무엇이 target인가 | 기여 1 · 2 | 3.2 seeds, 3.3 memory | Tab. 3 seeds · memory 행, Tab. 4 target-ID · switch, 지시문 변경 scene pair |
| 2 Extent | target 표면은 어디서 끝나는가 | 기여 1 | 3.2 depth-continuous surface | Tab. 3 extent 행, Tab. 4 precision, Tab. 1 external 행 |
| 3 Permission | 어느 link가 언제 닿아도 되는가 | 기여 2 | 3.4 stage · contact links · attached object | Tab. 3 policy 3행, Tab. 2 AAG vs TSA/FSHOA, Tab. 1 AEGIS 잔여 충돌 분해 |

## 3. 레이아웃 (text width 6.875 in, 1쪽 상단 전폭)

```
 "Put the bowl on the plate."  (SafeLIBERO Level I: obstacle beside the target)            ← 지시문 1줄
 [ VLA only: 장애물을 밀고 지나감 ✗ ]   [ VLA + AG3S: target만 잡고 장애물 보존 ✓ ]        ← 위쪽 줄 (선택)

 ┌ 1 What: identity ────────┐ ┌ 2 Where it ends: extent ─┐ ┌ 3 Who and when: permission ┐
 │ the noun's attention     │ │ depth continuity sets    │ │ contact links only, in the  │
 │ names the object         │ │ the boundary             │ │ allowed stage               │
 │ Prior: external segmenter│ │ Prior: ellipsoid / mask  │ │ Prior: end-effector         │
 │ [실제 baseline 출력]      │ │ [실제 baseline 출력]      │ │ ellipsoid, fixed role       │
 │ ✗ wrong object / follows │ │ ✗ neighbor freed with    │ │ [실제 baseline 출력]         │
 │   the robot              │ │   the target             │ │ ✗ carried object hits       │
 │ AG3S (ours)              │ │ AG3S (ours)              │ │ AG3S (ours)                 │
 │ [seeds] [wrong → alert]  │ │ [kept apart] [fruit kept]│ │ [contact links] [attached]  │
 └──────────────────────────┘ └──────────────────────────┘ └─────────────────────────────┘
```

- 높이 목표: 위쪽 줄 포함 ≈ 2.9 in, 제외 ≈ 2.5 in. 현재 두 그림의 합(fig01_noCL 2.62 in + fig02 3.35 in = 5.97 in)보다 약 3 in 낮다. caption 축소분을 합쳐 **약 0.4–0.5쪽 절약**(추정, 빌드 후 실측).
- **위쪽 줄은 선택 사항.** 세 칸만으로 주장이 전달되므로, 쪽수가 빠듯하면 빼고 지시문 줄만 남긴다. 남기면 독자가 첫 장면에서 문제를 보지만 약 0.1쪽 늘어난다.
- 각 칸의 Prior 이미지 아래 빨간 badge(✗ + 7단어 이내), AG3S 이미지 아래 초록 badge(✓). 현재 `fig02_overview`의 badge 체계를 그대로 쓴다.
- 색 규약(`cvpr_v1`과 동일): 주황 = 선택 target, 파랑 = 보호 이웃, 금색 = seed / contact link, 보라 = attached object.
- 글자 최소 7 pt, 벡터 PDF, raster 400 dpi, `layout_audit()` 위반 0(`cvpr_v1/NOTES.md` 4차 규칙).
- **수치는 넣지 않는다**(§1 규칙 3). 각 칸 라벨 안의 점수(예: `target 0.61`)도 삭제한다.

## 4. 각 칸의 "Prior" 이미지: 실제 baseline 출력으로 교체

현재 `fig02_overview`의 Prior 행은 대체 자료다. caption 주석이 "segmenter mask = simulator label as a stand-in; volume exemption, fixed role = recorded runs"라고 밝히고 있다. 최종본은 AVA-VLA Fig. 1(b)처럼 **4장 표에 들어간 baseline의 실제 출력**이어야 한다.

| 칸 | 보여줄 실패 | 출처 baseline (4장 행) | 장면 선택 기준 | 4장 연결 문장 |
|---|---|---|---|---|
| 1 Identity | 지시문 noun으로 prompt한 segmenter가 비슷한 다른 물체를 고르거나, 로봇이 target을 가릴 때 mask가 로봇을 따라감 | external segmentation (Grounded-SAM-2, 지시문 noun prompt) — Tab. 1 · Tab. 4 | Tab. 4 held-out grounding 장면 중 비슷한 물체가 있는 장면, 또는 파지 직전 가림 frame | 4.4 "when the robot occludes the target the mask follows the robot" |
| 2 Extent | target에 닿은 장애물이 target과 함께 풀리거나, ellipsoid가 target을 삼켜 파지가 막힘 | AEGIS(공식 코드) ellipsoid 또는 external mask — Tab. 1 · Tab. 4 precision | SafeLIBERO **Level I**(장애물이 target에 닿음), 위쪽 줄과 **같은 장면**이면 가장 좋음 | 4.2 "an ellipsoid or an external mask either swallows the target … or frees the obstacle with it" |
| 3 Permission | 운반 중 attached object나 end-effector 밖 link가 이름 없는 clutter에 부딪힘 | AEGIS 잔여 충돌(그중 61 %가 다른 link · 운반 물체) — Tab. 1 | Level I 또는 II의 transport 단계 첫 접촉 step (공식 1 mm 규칙 발동 step) | 4.2 "collisions of the other hand links and of the carried object with unnamed clutter remain" |

AG3S 행은 같은 장면 · 같은 initial state · 같은 seed의 최종 구성 run에서 같은 step(또는 가장 가까운 step)을 쓴다. 각 칸 두 장면 중 하나는 내부 표현(seeds, surface, contact links)을 보여주는 렌더, 하나는 agent-view 결과 frame으로 한다. 현재 `fig02_overview`의 구성과 같다.

위쪽 줄(선택)의 장면 기준은 v2와 같다: SafeLIBERO Level I, 같은 initial state에서 π0.5가 **장애물을 밀며 성공**(TSR에는 들어가고 SSR에는 안 들어가는 episode)하고 AG3S가 **safe success**. 접촉 순간은 공식 1 mm 변위 규칙이 처음 발동한 step.

후보는 칸마다 3개씩 contact sheet로 고르고, 선택 근거(run ID, episode, step, 거리)를 sidecar JSON에 기록한다. 그림에 쓴 episode는 4장 표 집계에 포함된 **eval 분할** episode여야 한다.

## 5. 지금 있는 자료와 쓸 수 있는 범위

| 자료 | 경로 | 쓸 수 있나 |
|---|---|---|
| 현재 Fig. 2 세 칸 자산과 합성 코드 | `paper/figures/cvpr_v1/assets/`, `paper/tools/plotting/cvpr_v1_compose.py` `fig02()` | **layout 뼈대로 재사용.** 위쪽 흐름 줄을 뺀 `fig01_v3()`를 새로 만든다. AG3S 행 자산은 RB-Y1 / 개발 episode라 SafeLIBERO 최종 run으로 교체 |
| 현재 Fig. 1 hurdle 장면 | `paper/figures/cvpr_v1/assets/{vla_pick,vla_carry_fail,ag3s_pick,ag3s_carry}.png` | 위쪽 줄 시안용으로만. 최종은 SafeLIBERO 장면 |
| 같은 장면 baseline 영상 (Level I Spatial t0 e0, t1 e1; none / aegis / aegis-gt / multilink_cbf-gt / ag3s-final / ag3s-pb7) | `outputs/2026-11-04_baseline_vs_AG3S_같은기계_SafeLIBERO_LIBERO-Safety/videos/*.mp4` | 위쪽 줄과 칸 2 · 3 Prior **시안용**. dev initial state(0–4)이고 AG3S는 11-04 구성 |
| 장면 개요 그림 | 같은 폴더 `scene_LI.png`, `closed_loop_episode_matrix.png` | 장면 선택 참고 |
| AG3S 내부 6-panel 진단 렌더 | `outputs/2026-11-08_GT없는_인식_정량/AG3S_기능장면/s8_t{3,4,7,9}/*.png` + `*_index.json` | 칸 1 · 2 AG3S 행 내부 렌더의 **렌더 코드와 데이터 형식** 재사용. LIBERO 무장애물 · 개발 run이라 그림에 직접 쓰지 않음 |
| 최신 E1a Spatial LI run 영상 (B5 GT 구성, t0–t3) | `AG3S_VLA-storage/results/benchmark/20261108T175826_*qe-gt23t100-e1a-spatial-LI-t0*/videos/` 외 t1–t3 | GT perception 구성이라 oracle에 해당. 최종 non-GT 구성 run으로 교체 |
| 접촉 원인 자료 | `outputs/2026-11-08_SafeLIBERO_충돌_원인/contacts_*.json` | 칸 3 "첫 접촉 step"과 접촉 link 산출 방식 참고 |

결론: **지금 자료로는 layout 시안만 만들 수 있다.** 최종 그림은 E1a(SafeLIBERO main) eval run과 E2(grounding held-out) run 뒤에 제작한다.

## 6. 제작 절차

1. **시안 (지금 가능).** `cvpr_v1_compose.py`의 `fig02()`를 복사해 `paper/tools/plotting/cvpr_v3_fig1_compose.py`의 `fig01_v3()`로 만든다. 위쪽 흐름 줄을 지우고, 위쪽 장면 줄을 선택 인자(`top_strip=True/False`)로 둔다. 기존 자산으로 두 변형을 렌더해 높이와 쪽수 절약을 실측한다.
2. **후보 추출 (E1a · E2 뒤).** 집계 script의 episode 단위 결과(`end_reason`, `collision`, `success`, 접촉 link)에서 §4 기준에 맞는 후보를 칸마다 3개 뽑아 `paper/figures/fig01_teaser/v3/candidates.json`에 쓴다.
3. **frame 추출.** 각 후보 run의 기록된 obs에서 step index로 agent / wrist frame을 뽑는다(영상 재인코딩 아님). AG3S run의 같은 step에서 서버 dump(attention map, target surface, link field)를 `AG3S_기능장면` 렌더 코드로 렌더한다. baseline의 ellipsoid · mask는 그 baseline의 기록 출력으로 겹친다.
4. **합성과 검사.** 모든 패널에 source run ID · step · sha256 sidecar. 글자 ≥ 7 pt, 색 규약, `layout_audit()` 위반 0.
5. **원고 대조.** caption(§7)과 1장 본문, 4.2 · 4.4 문장이 같은 실패를 같은 말로 부르는지 확인.

## 7. Caption 초안 (결과 뒤 확정)

> **Figure 1. One frozen VLA decides what may be touched, where that surface ends, and by which link, when.** *Top (optional):* on "put the bowl on the plate" with a mug touching the bowl (SafeLIBERO Level I), the frozen π0.5 completes the task by pushing the mug; with AG3S-VLA it grasps the bowl and leaves the mug in place. *Bottom, each against the closest prior practice.* **Identity:** an external segmenter prompted with the instruction's noun picks a similar object, or its mask follows the robot that occludes the target; the noun attention of the policy seeds the object it acts on, and a feature memory flags a change of object. **Extent:** an ellipsoid or an image mask frees the touching mug together with the bowl; depth continuity bounds the bowl's observed surface, so the mug stays protected. **Permission:** a filter on the end-effector ellipsoid leaves the other links and the carried object unprotected; AG3S-VLA frees the target surface for the contact links in the grasp stage only and protects the attached object during transport. Prior rows are outputs of the baselines in Tab. 1 on the same episodes. No external segmenter, no change to the policy.

## 8. 원고에서 함께 바꿀 참조

| 위치 | 현재 | 변경 |
|---|---|---|
| `latex/sections/01_introduction.tex` 10행 | `Fig.~\ref{fig:overview}` (현재 Fig. 2), `Fig.~\ref{fig:teaser}` | 둘 다 새 Fig. 1을 가리키므로 label을 하나로 통일(`fig:teaser` 권장). "In the example of Fig. 1, the fingers may enter…" 문장은 위쪽 줄 또는 칸 3과 맞춘다 |
| `latex/sections/03_method.tex` 58행 | `Fig.~\ref{fig:overview}` (identity 칸의 cosine ≈ 0.97 예) | 새 Fig. 1의 칸 1을 가리키도록 유지하거나, 칸 1 라벨에서 점수를 지웠으므로 문장을 "(Fig. 1, identity)"로 |
| Method의 파이프라인 참조 | 현재 Fig. 3 | 새 Fig. 2 (`\ref` 사용 시 자동) |
| `latex/exp_section/04_experiments.tex` 159행 | "(Fig.~1, panel~d)" | v2의 패널 (d)가 없어지므로 "(Fig.~1, identity)"로. 가림 실패는 칸 1 Prior가 보여준다 |
| `latex/exp_section/build_experiments.tex` | `\setcounter{figure}{3}` | `{2}` (실험 절 실로봇 그림이 Fig. 3이 됨) |
| `paper/notes/SEC1-3_REVISION_REQUESTS_20261108_KO.md` C1 · C2 | C1 "teaser v2로 교체", C2 "Fig. 2 삭제 권고" | v3 병합으로 갱신 (2026-11-09 반영) |
| `paper/evidence/AG3S-VLA_CVPR2027_FINAL_EXPERIMENT_PLAN_20261108_KO.md` §6 Fig. 1 행, §13 | teaser v2 | v3 병합으로 갱신 (2026-11-09 반영) |
