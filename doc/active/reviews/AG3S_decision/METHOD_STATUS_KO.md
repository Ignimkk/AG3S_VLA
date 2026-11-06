# 방법론 현황 — 논문 설명과 실제 구현의 대응 (ATS · HSR · LTE · CCR)

작성: 2026-10-31 19:43 KST (`date -u` 10:42 + 9 h) · 기준 commit `8ab3c7ef775737b628f5563a74ffdd13db40d023` (`feature/ag3s-development`) + **commit 하지 않은 작업 트리** (수정 15 개 파일: `AG3S/ag3s/config.py` · `grasp/candidates.py` · `runtime/{attention_surface,oracle_segmentation,pipeline}.py` · `AG3S/trajopt/{config,grasp_anchor,safe_policy,serve_safe,sqp}.py` · `ADOPTED_KO.md` · `paper/evidence/PAPER_LOG.md` · `third_party/openpi/.../{gemma,pi0,pi0_config}.py`; 새 파일 `runtime/attention_surface_gpu.py` · `runtime/attention_surface_post.py` 등). T78 구현이 이 작업 트리에서 진행 중이라 **이 문서의 "T78" 항목은 코드가 있어도 검증 전**이다.

- 모듈별 방법과 상태 목록의 정본: [ADOPTED_KO.md §A9](ADOPTED_KO.md) (A9.1 모듈별 표 · A9.2 HSR 5 단계 · A9.3 남은 일). 이 문서는 그 표를 "어떻게 동작하나" 로 풀고 논문 설명과 나란히 놓는다.
- 논문 설명 = [`paper/METHOD_GUIDE_KO.md`](../../../../paper/METHOD_GUIDE_KO.md) (의도한 방법). 이 문서에서 "논문" 은 그것이다.
- 이 문서는 코드를 읽어 쓴 것이다. 실험은 돌리지 않았다. 코드에서 확인하지 못한 것은 "코드에서 확인 못 함" 으로 적었다.
- 수치는 원래 문서에 있는 것만 옮기고 출처 경로를 옆에 단다.

## 0. 먼저 읽을 것 — 세 가지 구성을 구분한다

| 이름 | 뜻 | 정본 |
|---|---|---|
| **논문 설명** | METHOD_GUIDE 가 쓴 방법. 실제로 도는 시스템과 다른 곳이 있다 | `paper/METHOD_GUIDE_KO.md` |
| **A8 = 평가한 최종 구성** | 2026-10-29 확정, commit `cc0d527f`. benchmark 비교에 쓰려던 구성. 서버 인자 한 줄이 `AG3S/ag3s/docs/handoff/AG3S-CORE-EVAL-A6.impl.md` §3 에 있다 | [ADOPTED_KO.md A8](ADOPTED_KO.md) |
| **A9 = 수정 최종 구성** | A8 에서 바꾸는 점을 적용한 것 (2026-10-31 사용자 판정). **확정 일부 + 진행 중 일부**. benchmark 로는 이 구성을 보낸다 | [ADOPTED_KO.md A9](ADOPTED_KO.md) |

- 표기: "T숫자" 는 개발 STEP 번호 (`AG3S/ag3s/docs/handoff/T숫자.task.md`), "IFG-…" 는 target 표면 추출 · 추적 연구 시리즈의 STEP 이름이다.
- 상태 열: **그대로** (A8 과 같음) · **바뀜** (A9 에서 바뀜, 확정) · **추가 예정** (A9 에 넣기로 했고 진행 중) · **없음** (A8 · A9 모두 꺼짐 또는 구현 없음).

## 1. 한 페이지 요약 — A9 에서 실제로 도는 데이터 흐름

```text
RGB + 지시 + 로봇 상태 ──> frozen VLA(pi0.5, RBY1 16D) ──> 원래 action chunk
                            │ 같은 forward 에서: 명사 attention(layer 10 head 7), KV(layer 4), [A9] hidden(layer 4)
depth + 카메라 + 로봇 구 mask ─> ATS: target 정체(identity) + 범위(extent)
KV/hidden + 쥠 증거 ───────────> HSR: 작업 단계(phase) + 쥠 상태(latch)
ATS + HSR ─────────────────────> LTE: 링크별 거리장 (target 을 지운 거리장 / 안 지운 거리장)
chunk + 거리장 ────────────────> CCR: SQP 수정 → 앞 8 행 검사 → 실행 또는 HOLD
```

- **ATS (목표 물체 찾기 · 추적).**
  - **정체:** 명사 attention 의 상위 점에서 3D 로 자라는 덩어리 (cluster) 를 후보로 골라 target 으로 잠근다. 3 프레임 연속 이긴 물체만 바꾼다.
  - **범위 (A8):** target 덩어리에 구 (ball) 를 맞추고, 구 안에서 "target 점이 더 가까운 voxel" 을 지운다 (ball-owned).
  - **범위 (A9):** attention 을 시작점 (seed) 으로 depth 연속 표면 graph 에서 표면 영역을 얻어 그 영역을 지운다. 공 제거. 코드는 작업 트리에 있고 검증 전이다 (T78).
  - **시간 추적:** 3D 위치 · 크기 규칙 기반이다. VLM hidden 특징으로 다시 찾기는 offline 실험 중이다 (T79).
- **HSR (작업 단계 · 쥠 상태).**
  - **A8:** 3 단계 분류기 (pick · place · home) 가 "새 조작 대상을 받을지" 만 정한다. 6 단계 분류기 (HSR6) 가 이웃 접촉 허용 · 쥔 물체 허용 · target 권한 거둠을 켜고 끈다. 쥠 상태는 규칙 기반 latch 가 따로 정한다.
  - **A9:** 3 단계 + 6 단계 분류기를 **5 단계 분류기 하나** (VLM hidden layer 4 + 물리 정답 라벨) 로 바꾼다. 분류기는 offline 평가까지 했고 serving 연결 · 규칙 재연결은 구현 전이다.
- **LTE (링크별 거리장).**
  - 제약을 거는 링크는 양손 손바닥 + 손가락뿐이다.
  - 쥐는 손의 그 링크들만 target 을 뺀 거리장을 본다. 다른 링크는 제약이 없다.
  - 쥔 뒤에는 쥔 물체가 손에 붙은 충돌 구가 된다.
  - 이웃 접촉 20 mm (손가락) · 10 mm (손바닥) 허용, 손바닥 껍질 구, 관심 영역 TSDF 비우기가 있다.
- **CCR (chunk 수정 · 검증).**
  - 앞 16 행을 SQP 로 풀고 (이웃이 20 mm 안일 때만 뒤 8 행 look-ahead), 앞 8 행을 실행한다.
  - 목적함수는 VLA 추종 + 부드러움 + 꺾임 변화율 · 연속성.
  - **A9:** 꺾임 변화율 · 연속성은 쥔 뒤에만 켜고, 여러 초기값 · GPU 초기값 풀이와 쥐기 목표 항을 뺀다 (T74 Q1, 확정).
  - 검증은 waypoint 에서의 실제 거리 재계산이다. 위반이 남으면 HOLD (측정된 팔 자세 유지) 한다.

## 2. ATS — 실제로 어떻게 동작하나

### 2.1 요청마다 일어나는 일 (A8 = 평가한 구성)

| # | 무엇이 일어나나 | 서버 flag | 주요 코드 |
|---|---|---|---|
| 1 | VLA forward 중 prompt 의 target 명사 token 이 image patch 를 보는 attention (layer 10, head 7) 을 꺼낸다. 이 요청에는 patch 해상도 (카메라당 16×16) 의 지도만 있다 | `--attention-source noun_target:L10H7` | `AG3S/trajopt/attention_source.py` |
| 2 | depth 를 3 카메라에서 3D 점으로 바꾸고 TSDF 에 적분한다. 로봇 몸은 **로봇 구 (self-filter sphere) 의 depth mask** 로 지운다 (렌더한 로봇 depth 와 비교하는 방식은 쓰지 않는다) | (`--robot-mask` 없음) | `AG3S/ag3s/stages/robot_filter.py` · `stages/reconstruction.py` |
| 3 | RANSAC 으로 받침면 (탁자 등) 평면을 찾아 후보에서 뺀다 | (기본) | `stages/support_surface.py` |
| 4 | attention 을 3D 점에 올리고 (lift), 상위 5 % 점을 seed 로 3D 연결 성장 + DBSCAN 으로 덩어리 (cluster) 를 만든다. 받침면 점은 제외한다 | (기본: `attention.seed_percentile 95`, `clustering.method dbscan`) | `stages/attention_lifting.py` · `stages/target_grounding.py` (`ground_target`) |
| 5 | 후보를 거른다: 손가락이 벌어지는 폭 안에 들어가는 물체, 목적지(바구니) 아님, 가장 낮은 점이 받침면 위 60 mm 를 넘는 것(허들 막대 등)은 제외 | `--supported-max-bottom-mm 60` | `stages/admissibility.py` |
| 6 | 맞닿은 과일이 한 덩어리로 합쳐졌으면 12 mm 반경으로 다시 군집하고 raw attention 비율로 target 쪽 조각을 고른다. 쥐기 전에는 그 조각이 anchor 를 대체한다 (IFG-SPLIT: 맞닿은 이웃 분리 STEP) | `--attention-split-radius-mm 12 --attention-split-ratio-source raw --attention-split-partition --split-supersedes-anchor` | `stages/attention_split.py` |
| 7 | **시간 유지.** 다른 물체가 3 프레임 연속 앞서야 target 이 바뀐다 (`TargetConfirm`). 사과가 안 보이면 마지막으로 본 기하를 유지하고(occluded) 오래 안 보이면 lost. 쥐는 동안에는 "안 보임" 만으로 정체가 바뀌지 않는다. 닫는 동안 닫기 전 기하를 고정하고, 손에 밀린 과일을 따라간다 | `--latched-identity-hold --closing-keep-geometry --kept-geometry-follow` | `stages/target_grounding.py` (`TargetConfirm`) · `AG3S/trajopt/safe_policy.py` |
| 8 | target 기하에 **구 (ball) 를 맞춘다.** 구 안 voxel 마다 가장 가까운 target 점 거리 `d_t` 와 가장 가까운 "target 아닌 관측점" 거리 `d_o` 를 비교해 target 쪽이 확실한 것만 지운다 (이웃 쪽이거나 애매하면 남긴다) → LTE 의 "target 을 뺀 거리장" 이 된다 | `--target-exclusion ball-owned --owned-max-dist-voxels 3 --nontarget-evidence surface` | `AG3S/ag3s/fields/curobo_builder.py` (`TargetBall` · `_ball_owned_split`) |
| 9 | 놓은 뒤 home 에서 seed 가 없어도(전부 받침면 위) 인증한다 | `--placed-no-seed-certify` | `AG3S/ag3s/runtime/pipeline.py` |

- **A8 의 "target 범위" 는 공이다.** 구는 target 덩어리에 맞춘 둥근 상한이고, 그 안에서 거리 비교로 이웃을 보존한다. depth 연속 표면으로 경계를 정하지 않는다. 이웃이 구 안에 있고 target 점과 구별이 안 되는 경우가 알려진 한계다 (`ADOPTED_KO.md` A8 "알려진 한계": target 공이 사과보다 커서 붙은 이웃 표면 일부를 지움).
- **정체 (identity) 의 근거는 attention 의 3D 덩어리다.** 중심화한 VLM feature 로 같은 물체인지 확인하지 않는다 (§2.3).

### 2.2 A9 에서 바뀌는 점 — 표면 추출 (T78, 추가 예정)

사용자 판정 (2026-10-31): "공으로 지우는 건 좋은 방법론이 아니다 … 반드시 최종 구성에 포함" (`AG3S/ag3s/docs/handoff/T78.task.md` §0). 요청마다 다음이 달라진다.

| # | 새로 일어나는 일 | 서버 flag | 주요 코드 |
|---|---|---|---|
| a | 표면 graph: 3 카메라 depth 를 5 mm voxel node 로 묶고, 이웃 pixel 의 depth 차가 10 mm 를 넘으면 edge 를 만들지 않는다 (depth 불연속을 넘지 않는다). node 마다 normal (PCA) 이 있고 edge weight 에 위치 · normal 항이 들어간다 | `--target-extent attention-surface` | `AG3S/ag3s/runtime/attention_surface.py` → 고정된 연구 코드 `AG3S/ag3s/experiments/studies/ifg_graph.py` |
| b | seed: 명사 attention 상위 5 patch 의 질량을 patch 안 pixel 비율대로 node 에 나눈다. 받침면 node 는 음의 증거다 | 위와 같음 | `studies/ifg_round_ns.py` (`g_and_top1`) · `ifg_graph.seed_unary` |
| c | 퍼뜨리기: box 제약 Dirichlet 에너지를 풀어 node 의 target 소속도를 얻는다. 증거가 없는 연결 성분은 unknown 으로 남는다. 소속도 ≥ 0.5 영역 중 seed 질량이 가장 큰 것을 고른다. feature 는 graph 에 섞지 않는다 | 위와 같음 | `ifg_graph.solve` |
| d | 맞닿은 이웃을 attention 분할로 한 번 더 자른다 | `--surface-split` | `runtime/attention_surface_post.py` |
| e | 고른 영역이 이번 요청의 ATS target 덩어리와 겹치지 않으면 그 요청은 기권 | `--surface-target-check overlap` | 같은 파일 |
| f | 기권이면 직전에 맞게 고른 영역을 유지한다 (처음부터 없으면 아무것도 지우지 않는다. 공으로 대신하지 않는다). 쥐는 중에는 표면을 새로 고르지 않는다 | `--surface-keep-last --surface-grasp-hold` | `runtime/pipeline.py` |
| g | 공 제거 (표면 영역에서 만든 범위를 쓴다) · 표면 선택을 GPU 로 | `--surface-no-ball` · `--attention-surface-device cuda` | `runtime/pipeline.py` · `runtime/attention_surface_gpu.py` |

- 위 flag 는 작업 트리의 `serve_safe.py` 에 있다. **재생 평가 · closed-loop 은 아직이다** (판정 조건은 `T78.task.md` §3).
- 근거 STEP: T76 (고정 입력 재생 비교, `outputs/2026-10-30_T76_depth연속경계_대_공방식_재생비교/`). 표면 방식은 맞닿은 이웃을 합치거나 (고른 영역 최대 145–183 mm, 사과 약 68 mm) 엉뚱한 물체 (바구니) 를 고르는 요청이 있었고 선택 중앙 697 ms 였다. 그래서 T78 이 위 d–g 를 더한다 (`T78.task.md` §1).
- `--target-extent attention-surface` 에서도 정체 (어느 덩어리가 target 인가) 를 정하는 것은 §2.1 의 4–7 단계다. 표면은 **범위** 만 바꾼다.

### 2.3 시간 추적 — 논문의 feature 기반 추적은 어디에 있나

- **A8 의 시간 추적:** §2.1 의 7 (3 프레임 이긴 물체만 교체, 마지막 기하 유지, 밀린 과일 따라가기). 3D 위치 · 크기 규칙이고 feature 를 쓰지 않는다.
- **구현은 있으나 꺼진 것:** `--confidence-gate on` (기본 off, A8 인자에 없음, A9 표에도 없음). 코드: `AG3S/ag3s/runtime/confidence_gate.py`, 호출은 `runtime/pipeline.py` 의 `_confidence_gate_step`.
  - 서버가 attention 표면 영역을 따로 한 번 더 고르고, 그 영역의 VLA prefix layer 8 image feature (프레임 평균을 빼고 정규화) 를 만든다.
  - 직전 프레임 선택의 feature (chain 기억) 와 cosine 이 0.5904 (`LOCK_heldout.json` dev 점수 10 백분위, 코드 상수) 아래면 **경보** 로 보고 그 프레임의 target 예외(target 을 뺀 거리장 · 접촉 권한) 를 거둔다. 쥔 동안에는 적용하지 않는다.
  - 기억 갱신 때 후보 중 기억과 더 비슷한 영역(차이 ≥ 0.1)으로 "switch" 를 **기록** 하지만, 코드에서 읽은 범위에서는 서버가 쓰는 영역을 그 후보로 **교체하지는 않는다.**
- **T79 (진행 중, offline):** target 이 밀리거나 가려진 뒤 VLM hidden patch 특징으로 진짜 사과를 다시 찾을 수 있는지 평가한다. 동기는 회귀 장면 211 에서 사과가 밀린 뒤 target 점이 옛 자리에 남은 실패 (T75). flag 없음 (`T79.task.md`).

## 3. HSR — 실제로 어떻게 동작하나

### 3.1 A8 (평가한 구성)

| # | 무엇이 일어나나 | 서버 flag | 주요 코드 |
|---|---|---|---|
| 1 | VLM prefix 의 layer 4 V cache (head 1 개) 를 [text 평균 \| image 평균] 512 차원으로 줄여 **3 단계 분류기** (pick · place · home; 표준화 + 로지스틱 회귀, 시연 라벨) 를 적용한다. 같은 forward 에서 꺼내므로 추가 forward 가 없다 | (기본 on) `kv_L4_v1` | `AG3S/ag3s/stages/subtask_probe.py` |
| 2 | 같은 label 이 3 프레임 연속 나와야 "확정" 으로 본다. 확정 place · home 이면 **새 조작 대상을 받지도 바꾸지도 않는다**. 놓은 뒤 home 이면 놓인 대상을 푼다 | (기본 on) `clustering.subtask_gate` | `subtask_probe.SubtaskDebounce` · `stages/target_grounding.py` |
| 3 | **6 단계 분류기 HSR6** (접근 · 쥐기 · 들기 · 운반 · 놓기 · 물러남; 같은 layer 4 KV, 시연 라벨, 요청 하나의 argmax) 가 네 규칙을 켜고 끈다. ① 쥠 인정 보조 (들기·운반·놓기 단계 ∧ 닫힘 ∧ 막힘이면 attach 인정). ② 쥔 물체 들어올림 허용은 접근 · 물러남에서만 거둔다. ③ 물러남에서는 쥐지 않은 target 의 예외를 거둔다. ④ 이웃 접촉 허용 · 쥐기 목표 항(A4) · 기하 보정(G1) 은 쥐기 단계에서만 | `--hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective` | `AG3S/trajopt/hsr6.py` · `safe_policy.py` (`_hsr6_*`) |
| 4 | **쥠 상태는 분류기가 아니라 규칙 상태 기계 (latch) 가 정한다.** searching → latched(대상 확정) → closing(닫는 명령이 실제로 적용됨) → held(쥠 확인) → placed(놓음). held 가 되는 증거: 닫는 명령 적용 ∧ 측정 개도가 멈춤 ∧ 개도가 줄어 문턱 아래 ∧ 명령보다 위에서 막힘(빈손이 아님) ∧ target 이 손에서 reach(120 mm) 안. attach 뒤 일정 청크 안에 반증이 있으면 되돌린다. 닫힌 채 빠지면 놓음으로 본다 | `--held-slip-detach` (그 밖은 기본) | `AG3S/trajopt/grasp_latch.py` · `safe_policy.py` (`_advance_grasp`) |

- **client 가 주는 phase 는 고정이다.** client 인자 `--safe-phase approach` (`ADOPTED_KO.md` A8 "client") 이므로 phase 규칙은 client 가 단계별로 바꿔 주지 않는다. 단계 구분은 위 latch 와 HSR6 에서 온다.
- 이 구성의 한계: 학습 라벨이 전문가 시연이라 "닫았지만 빈손" 예가 없고, HSR6 의 빈손 → 운반 · 놓기 오판이 28 % 였다 (`ADOPTED_KO.md` A9.2 표).
- A8 에서 기하 보정(G1) · 쥐기 목표 항(A4) 은 켜져 있지만 규칙 ④ 가 "쥐기 단계에서만" 이라 닫기 전 요청(분류기는 접근으로 읽음) 에 작동하지 않았다. 측정된 성공은 두 항 없이 나왔다 (`ADOPTED_KO.md` A8 확정 5).

### 3.2 A9 에서 바뀌는 점 — 5 단계 분류기 (T77, 확정 · serving 구현 전)

- 단계 수 = 제약 묶음이 갈리는 수 = **5** (접근 · 쥐기 · 운반 · 놓기 · 물러남; 들기 · 운반은 제약이 같아 합침). 3 단계와 6 단계 두 분류기를 하나로 바꾼다.
- 입력: VLM hidden layer 4 의 카메라 3 대 image token 평균 + 마지막 유효 text token = 8192 차원 (같은 forward). 라벨: 전문가 시연 + closed-loop 기록 427 run 의 MuJoCo 물리 정답 (실제로 들지 않았으면 운반이 아님).
- 학습에 쓰지 않은 장면 (211 · 954 · 122 · 119) 평가: 빈손인데 운반 · 놓기라고 읽는 오판 28 % → 2.2 %, 5 단계 정확도 0.47 → 0.72 (HSR6 → 채택안). 출처 `ADOPTED_KO.md` A9.2, `AG3S/ag3s/docs/handoff/T77.task.md`.
- **남은 일:** serving 구현 (hidden pooling · 분류기 asset), 3 단계 gate 와 규칙 ①–④ 를 5 단계에 다시 잇기, 꺾임 변화율 · 연속성을 켜는 시점을 "운반 · 놓기" 에 묶을지 정하기, closed-loop 확인 (`ADOPTED_KO.md` A9.3-3).
- 쥠 상태 latch 는 A9 에서도 규칙 기반 그대로다 (`ADOPTED_KO.md` A9.1).

## 4. LTE — 실제로 어떻게 동작하나

### 4.1 요청마다 일어나는 일 (A8 · A9 공통, 바뀌는 곳은 표시)

| # | 무엇이 일어나나 | 서버 flag | 주요 코드 |
|---|---|---|---|
| 1 | 거리장 3 계층: 거친 20 mm · 미세 5 mm · TSDF 5 mm (cuRobo ESDF). GPU 거리장을 요청마다 host 로 한 번 복사한다 | `--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --esdf-host-mirror` | `AG3S/ag3s/fields/curobo_builder.py` · `curobo_field.py` |
| 2 | 관심 영역 (조작 대상 · 목적지 · 손 실행 경로 상자) 안에서 이번 depth 광선이 지나간 TSDF voxel 을 비운다 (보이는 빈 공간의 옛 표면 제거; 가장자리 보존, GPU) | `--roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda` | `fields/` · `serve_safe.py` |
| 3 | **제약을 거는 링크 = 양손 손바닥 + 손가락 넷.** 팔뚝 · 손목 · 몸통 · 반대 팔은 제약 밖이다 (부딪혀도 막지 않는다). 자기 충돌 제약도 끈다 | `--links gripper --no-self-collision` | `AG3S/ag3s/robot_models/` · `serve_safe.py` |
| 4 | **쥐는 손의 손바닥 · 손가락 링크만 "target 을 뺀 거리장"** 을 본다 (target 에 닿는 것이 과제이기 때문). 어느 링크가 target 에 닿아도 되는지는 쥐는 손(client `--safe-manipulators`) 의 허용 목록이다. target 예외는 그 링크의 행에만 준다 | `--target-field-policy exclude-authorized --target-permission-strict --esdf-margin 0` | `AG3S/ag3s/constraints/clearance.py` (`authorized_links`) |
| 5 | "target 을 뺀 거리장" 이 지우는 범위: **A8 = §2.1 의 ball-owned 범위, A9 = 표면 영역 (공 제거)** | A8 `--target-exclusion ball-owned …` / A9 `--target-extent attention-surface …` | `curobo_builder.py` |
| 6 | 로봇 충돌 구: capsule 반지름 0.05 배 · 간격 0.4 · capsule 당 최대 32 개 + 손가락별 몸통 구 + **손바닥 껍질 구** (손바닥 mesh 겉면에 5 mm 넘치게, 껍질 행의 비타겟 여유는 5 mm). 장애물 여유는 10 mm | `--capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --finger-cover both --palm-shell on --palm-shell-margin 5 --obstacle-margin 0.010` | `AG3S/ag3s/robot_models/` |
| 7 | **이웃 가벼운 접촉 허용:** 쥐기 전 손가락이 target 120 mm 안 이웃과 20 mm 까지, 쥐는 순간 손바닥이 10 mm 까지 겹쳐도 된다. 켜는 조건은 HSR 규칙 ④ (쥐기 단계) 와 쥐는 순간 창 | `--neighbour-contact-mm 20 --neighbour-palm-contact-mm 10` | `safe_policy.py` (`_neighbour_contact_gate`) |
| 8 | **쥔 물체 = 손의 일부.** held 가 되면 쥔 과일 몸통 전체를 덮는 충돌 구를 손바닥 frame 에 붙인다 (반지름은 관측 반폭 + 10 mm 이내). 그 물체는 환경 거리장에서 빠진다. 놓음 확인(placed) 후에는 다시 장면 물체다 | `--held-body-cover --held-cover-cap-mm 10 --held-slip-detach` | `AG3S/ag3s/robot_models/held_object.py` · `runtime/pipeline.py` |

- **논문과 가장 다른 곳:** 논문은 손목 · 반대 팔 등이 "target 을 포함한 전체 장면 거리장" 을 본다. 구현 (A8 · A9 모두) 은 그 링크들을 **제약에서 뺀다** (`--links gripper`). 손목을 포함하는 `gripper_wrist` 선택지가 있으나 A8 인자에 없다 (`serve_safe.py` `--links` help).
- 팔 충돌 구는 A8 에 넣지 않았다 (`--arm-spheres`, 사용자 판정 대기 `OPEN_ISSUES_KO.md` O7).
- 쥐기 때 꼭지 덮개(`--target-stem-cap`)와 렌더 로봇 mask 는 꺼져 있다 (`ADOPTED_KO.md` A8 "넣지 않은 것").

## 5. CCR — 실제로 어떻게 동작하나

| # | 무엇이 일어나나 | 서버 flag | 주요 코드 |
|---|---|---|---|
| 1 | VLA chunk 를 받아 앞 **16 행** 을 계획 창으로 풀고 앞 8 행만 실행한다 (`execution_length` 기본 8). 뒤 8 행은 이웃 장애물이 20 mm 안일 때만 look-ahead 로 풀되 인증하지 않는다 | `--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20` | `AG3S/trajopt/sqp.py` · `refiner.py` · `config.py` |
| 2 | 목적함수 = VLA 추종 (손목 관절은 0.1 배, 쥐기 전 target 150 mm 안) + 부드러움 (w 0.05) + 꺾임 변화율 (w 20) + 연속성 (w 0.5, 이전 chunk 와의 겹침). 제약 = 모든 구 · waypoint 의 `거리 − 반지름 − 여유 ≥ 0` + 관절 한계 | `--wrist-track-scale 0.1 --w-deflection-rate 20` | `AG3S/trajopt/problem.py` · `linearize.py` |
| 3 | SQP: 반복마다 거리 제약을 선형화하고 OSQP 로 푼다 (정확도 1e-4). 계산은 한 번씩 평가, BLAS thread 1 (결과 bit 동일) | `--qp-eps 1e-4 --to-fast-eval --blas-threads 1` | `sqp.py` · `qp.py` |
| 4 | **A8 만:** 위반이 남으면 cuRobo 로 초기값 16 개를 GPU 에서 한 번에 풀어 가장 나은 것을 SQP 시작점으로 쓴다 | `--to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass` | `gpu_seeds.py` · `multi_seed.py` |
| 5 | **A8 만:** 쥐기 목표 항 — VLA 가 처음 닫기를 계획한 순간의 손바닥 위치를 얼리고 닫는 행에서 당긴다 (무게 2000). A8 에서 실제 작동 0 요청이었다 | `--grasp-anchor 2000 --grasp-anchor-path-weight 0` | `grasp_anchor.py` |
| 6 | **검증:** 풀이가 돌려준 궤적의 모든 waypoint 행을 Jacobian 없이 실제 거리로 다시 계산한다 (`_finish`). 허용 오차를 넘는 위반이 남으면 VIOLATED. 사유별로 나눈 verdict 가 하나라도 HOLD 면 chunk 를 실행하지 않는다 | client `--safe-gate reasons` | `sqp.py` (`_finish`) · `safe_policy.py` (`_verdict`) · `wire.py` (`gate_decision`) |
| 7 | **HOLD:** client 가 HOLD 에 들어가는 순간의 측정 팔 관절을 잡아 그 자리에 서고 gripper 는 마지막 명령을 유지한다. 충돌 사유만 남았고 모든 행이 margin 안쪽이며 현재보다 나빠지지 않으면 HOLD 대신 수정한 chunk 를 최대 3 청크 실행한다 | client `--safe-hold-mode measured` · 서버 `--hold-follow-to 3` | `integration/rby1/bringup/pi05_infer.py` · `safe_policy.py` (`_follow_to`) |

**A9 에서 바뀌는 점 (확정, T74 Q1; `ADOPTED_KO.md` A9.1 CCR 표):**

| 바뀜 | 방법 | 효과 (T74 Q1, 10 run 같은 seed) |
|---|---|---|
| 꺾임 변화율 · 연속성 항을 쥐기 전에는 끄고 쥔 뒤(held 부터)에만 켠다 | `--deflection-rate-after-grasp --continuity-after-grasp` 추가 | 금지 접촉 없는 성공 4/10 → 8/10 |
| chunk 마다 표준 SQP 풀이 하나 (여러 초기값 · GPU 초기값 batch 없음) | 위 4 의 flag 를 뺀다 | 위와 같음 |
| 쥐기 목표 항 뺌 | 위 5 의 flag 를 뺀다 | 실제 작동 0 (T68) |

- 출처: `outputs/2026-10-30_T74_E0TrajOpt_목적함수조합/` (`ADOPTED_KO.md` A9.1 아래 T74 Q1 결과).
- 기하 grasp 보정(G1) flag 는 A9 에도 남지만 규칙 ④ 가 막아 **작동 0** 이다. HSR 5 단계 연결 뒤 작동하는지 확인이 필요하다 (`ADOPTED_KO.md` A9.1).
- 사용자 표현 "시작할 때는 꺾임 · 연속성을 끄고 place 할 때만 켜기" 의 실제 동작은 "latch 가 held 가 된 뒤 켜기" 다. 5 단계로 바꾸면 "운반 · 놓기" 에 묶을지 정해야 한다.

## 6. 논문 설명 대 실제 구현

열 설명: **구현?** = 코드가 있나 (flag · 파일). **A8 on?** = 평가한 최종 구성에서 켜져 있나. **A9** = 그대로 / 바뀜 / 추가 예정 / 없음.

### 6.1 ATS

| 논문 (METHOD_GUIDE) | 구현? | A8 on? | A9 |
|---|---|---|---|
| §4.1 로봇을 **렌더한 depth 와 비교** 해 로봇 pixel 만 제거 | 있음: `--robot-mask render`, `runtime/robot_render_mask.py` | **꺼짐.** 구 mask 만 쓴다 (A7: 거리장 · ATS · 성공을 못 바꾸고 요청마다 중앙 77 ms 추가, 실물 mesh 확보 어려움) | 없음 (미채택 N12) |
| §4.2 depth 연속 surface graph (edge 는 이웃 pixel + depth 차 + 3D 근접, 위치 · normal weight) | 있음: `--target-extent attention-surface` → `studies/ifg_graph.py` (jump gate 10 mm, normal PCA) | **꺼짐 — target 범위는 공 방식(ball-owned)** | **추가 예정** (T78 구현 진행, 검증 전) |
| §4.2 받침면 RANSAC 을 negative evidence 로 | 있음: 받침면 RANSAC (`stages/support_surface.py`) — A8 에서는 후보에서 빼는 용도, 표면 방식에서는 음의 unary | 후보 제외로만 on | 추가 예정 (표면 방식 안) |
| §4.3 명사 attention 을 patch 질량 보존으로 seed 화 | 있음: 상위 5 patch, 질량을 pixel 비율로 분배 (`ifg_graph.seed_unary`) | 꺼짐 (표면 방식 안에 있음). A8 은 상위 5 % 점을 seed 로 3D 성장 | 추가 예정 |
| §4.4 선형 시스템으로 소속도, 증거 없는 성분은 unknown, seed 질량 최대 영역 | 있음: box 제약 Dirichlet solve (`ifg_graph.solve`), 소속도 ≥ 0.5 중 seed 질량 최대 | 꺼짐 | 추가 예정 |
| §4.4 feature 를 graph 에 섞지 않음 | 일치 (`edge_weights(…, feat=None)`) | — | — |
| (논문에 없음) 맞닿은 물체 attention 분할 · target 겹침 검사 · 기권 시 직전 영역 유지 · 쥔 뒤 고정 · 공 제거 · GPU | 작업 트리: `--surface-split` `--surface-target-check` `--surface-keep-last` `--surface-grasp-hold` `--surface-no-ball` `--attention-surface-device` | 꺼짐 (없었음) | 추가 예정 (T78) |
| (논문에 없음) **공 + 소유 비교로 지우기** | 있음, 이것이 A8 의 방식 | **켜짐** | **바뀜 예정** → 표면 영역 |
| §5.1 프레임 평균을 뺀 patch feature · region descriptor | 있음: `runtime/confidence_gate.py` (prefix layer 8 image feature) | **꺼짐** (`--confidence-gate off`) | 없음 (A9 표에 없음) |
| §5.2 처음 유효한 영역의 descriptor 를 reference 로 저장하고 margin 이상 비슷한 후보로 **재연결(re-link)** | 부분: 기억 = 직전 프레임 선택(chain)이고 reference(처음 영역) 가 아니다. switch 는 기록만 하고 서버가 쓰는 영역을 바꾸는 코드는 읽지 못함. 연구용 offline 평가 코드 `studies/ifg_memory.py` 는 있음 | 꺼짐 | 없음. 대신 hidden patch 로 다시 찾기를 offline 평가 중 (T79) |
| §5.3 직전 승인 영역과의 causal 일관성 점수 < 문턱이면 alert → 추적은 계속, 새 접촉 예외만 차단 | 있음: `confidence_gate.py` (문턱 0.5904, 경보 프레임에 target 예외 보류, 쥔 동안 미적용) | **꺼짐** | 없음 |
| §5.2 attached 가 되면 재연결 중지, identity 고정 | 있음 (`--latched-identity-hold` · latch) | 켜짐 | 그대로 |
| (논문에 없음) 3D 위치 · 크기 규칙 시간 추적 (`TargetConfirm`, 마지막 기하 유지, 밀린 과일 따라가기) | 있음 | 켜짐 | 그대로 |

### 6.2 HSR

| 논문 | 구현? | A8 on? | A9 |
|---|---|---|---|
| §6.1 VLA prefix pooled KV 에 linear probe → pick · place · home | 있음: `stages/subtask_probe.py` (`kv_L4_v1`) | **켜짐** (기본) | **바뀜** → 5 단계 하나 |
| §6.1 두 번째 regularized classifier (probe log-prob + gripper 개도 + target–손 거리 + 받침면 위 높이 + region 크기) | **코드에서 확인 못 함** (관련 문자열 검색에서 못 찾음) | — | — |
| §6.1 여러 연속 관측이 지속되어야 phase 확정 | 있음: 3 프레임 연속 (`SubtaskDebounce`) | 켜짐 (3 단계 분류기). HSR6 은 debounce 없는 argmax 하나 | 5 단계 쪽 debounce 규칙은 구현 전 |
| §6.1 confirmed pick: 새 목표 채택 · 재연결 / place: identity 고정 / home: 실제 놓음 확인 후 identity 제거 | 있음 (place · home 에서 새 대상 안 받음, placed 뒤 home 이면 해제) | 켜짐 | 5 단계에 다시 연결 필요 |
| (논문에 없음) 6 단계 분류기 HSR6 + 규칙 ①–④ | 있음: `trajopt/hsr6.py` | **켜짐** | **바뀜** → 5 단계로 대체 (규칙 재연결 구현 전) |
| §6.2 쥠 상태 free · held · uncertain, 증거는 측정 개도 · 실행 명령 · 목표–손 동행 | 부분: 규칙 상태 기계 searching · latched · closing · held · placed (`trajopt/grasp_latch.py`). 증거는 닫는 명령 적용 · 개도 멈춤 · 개도 감소 · 명령보다 위에서 막힘 · reach, 반증 회수, 닫힌 채 빠짐 검출. **"uncertain" 상태는 코드에서 확인 못 함** (grasp_latch.py 에 없음) | 켜짐 | 그대로 |
| §6.2 target–손 동행 증거 | `--held-3d-rule on` 이 target 중심–손 거리 + gripper 로 쥠을 판단하는 규칙을 attach 조건에 AND | **꺼짐** (기본 off) | 없음 |
| §6.2 신호가 충돌하면 uncertain → 새 permissive transition 차단 | **코드에서 확인 못 함** | — | — |

### 6.3 LTE

| 논문 | 구현? | A8 on? | A9 |
|---|---|---|---|
| §7.1 제외 sample = 현재 관측된 target 표면 근처 ∧ 이웃 표면보다 명확히 target 쪽 (`d_T ≤ ρ`, `d_T + τ < d_N`), 구 · bounding volume 은 계산 범위일 뿐 | **근사 구현:** ball-owned. 공 안 seed 중 `d_t` 가 `d_o` 보다 tie(기본 1 voxel) 이상 가까운 것만 지움, 둘 다 3 voxel 보다 멀면 unowned. 단 공 자체가 상한이며 target 점은 grounding 덩어리 점이다 (표면 소속도가 아님) | **켜짐** | **바뀜 예정:** target 점을 표면 영역으로 (T78). 공 제거(`--surface-no-ball`) 포함 |
| §7.1 애매한 sample 은 장애물로 남김 | 일치 (ambiguous 는 기본 keep) | 켜짐 | 그대로 |
| §7.1 매 request 현재 sample 로 다시 만들어 철회가 즉시 반영 | **부분 확인:** 거리장은 TSDF 누적이며 ROI 비우기로 보이는 빈 공간을 정리. 표면 방식의 "직전 영역 유지" 는 의도적으로 지난 영역을 쓴다. 세부는 코드에서 확인 못 함 | — | — |
| §7.2 접촉 링크 `L` 만 target 을 뺀 거리장, **나머지 링크는 target 포함 전체 장면** | 다름: 제약은 손바닥 + 손가락만 걸고 **나머지 링크는 제약이 없다** | `--links gripper` 켜짐 | 그대로 |
| §7.2 예외 켜짐 조건 `e_t = v_t ∧ phase ∈ S_ok` | 있음: grounded target + 쥐는 손 허용 목록 + phase 규칙 (HSR6 ③ 물러남에서 거둠) | 켜짐 | 5 단계에 다시 연결 필요 |
| §7.2 local field 밖은 전체 장면 거리장으로 fallback, unknown 에는 예외 없음 | 창(window) 안에서만 target-free 계층을 합성한다는 설명이 flag help 에 있음. 세부는 코드에서 확인 못 함 | — | — |
| §7.3 grasp 확인되면 닫기 직전 target 기하를 손 frame 의 attached 구로 | 있음: 몸통 전체 덮개, 닫는 동안 기하 고정 | 켜짐 | 그대로 |
| §7.3 release 확인되면 장면으로 되돌림, uncertain 이면 permissive 차단 | placed 확인 시 복귀는 있음. uncertain 처리는 코드에서 확인 못 함 | 켜짐 | 그대로 |
| §7.3 목적지에서 물체–받침 접촉 허용, 로봇 링크–받침 접촉은 불허 | 목적지(바구니) 등록과 규칙이 있음. 세부는 코드에서 확인 못 함 | 켜짐 | 그대로 |
| (논문에 없음) 손바닥 껍질 구 · 이웃 접촉 허용 · ROI TSDF 비우기 | 있음 | 켜짐 | 그대로 |

### 6.4 CCR

| 논문 | 구현? | A8 on? | A9 |
|---|---|---|---|
| §8 앞 `J` 행 최적화, 앞 `K ≤ J` 행만 실행 | 있음: J = 16, K = 8, look-ahead 는 이웃 20 mm 안일 때만 | 켜짐 | 그대로 |
| §8 목적함수 tracking + smoothness + continuity | 있음 (+ 논문에 없는 꺾임 변화율 항) | 켜짐 | **바뀜:** 꺾임 변화율 · 연속성은 쥔 뒤에만 |
| §8 SQP + OSQP, 선형화한 clearance | 있음 | 켜짐 | 그대로 |
| (논문에 없음) GPU 여러 초기값 · 쥐기 목표 항 | 있음 | 켜짐 | **빠짐** |
| §8 prefix waypoint **와 그 사이 보간 자세** 모두 검사 | 부분: waypoint 행은 전수 재계산 (`_finish`). waypoint 사이 보간 검사(`--sweep-check`, `--servo-model`, `trajopt/sweep.py`) 는 구현돼 있으나 A8 인자에 없음 | **꺼짐** | 없음 (A9 표에 없음) |
| §8 모든 robot sphere · attached sphere 의 비선형 실제 거리 재계산 | 있음 (구 + 쥔 물체 구가 같은 행 집합) | 켜짐 | 그대로 |
| §8 측정된 현재 pose 에서 시작 | 서버는 요청의 로봇 상태에서 시작. "측정" 세부는 코드에서 확인 못 함 | — | — |
| §8 refiner 와 같은 geometry snapshot 으로 검사 | 같은 요청의 같은 scene 으로 계획 · verdict 를 계산하나 snapshot 동일성은 코드에서 확인 못 함 | — | — |
| §8 하나라도 실패하면 hold: 측정된 팔 pose 유지 + gripper 명령 유지 | 있음: client `--safe-hold-mode measured` | 켜짐 | 그대로 |
| (논문에 없음) 충돌 사유만 남은 HOLD 를 수정 chunk 실행으로 대체 | 있음 | `--hold-follow-to 3` 켜짐 | 그대로 |

## 7. 혼동하기 쉬운 것

1. **논문의 "depth 연속 표면 추출" 은 A8 에 없다.** 코드(`--target-extent attention-surface`)는 있지만 꺼져 있고, A8 의 target 제거는 공 + 소유 비교다. A9 는 이를 넣기로 했고 구현 · 검증이 진행 중이다.
2. **"ball-owned" 는 segmentation 이 아니다.** 구 안에서 거리로 이웃을 보존하는 규칙이다. 표면 경계를 얻지 못해 붙은 이웃을 같이 지우는 한계가 있다.
3. **`attention-surface` 와 `attention-surface-hybrid` 는 다르다.** hybrid 는 지우는 범위는 공 그대로 두고 표면 선택을 보호에만 쓴다 (`serve_safe.py` `--target-extent` help). A9 로 정한 것은 `attention-surface` 다.
4. **feature 기반 시간 추적 (논문 §5) 은 serving 에서 꺼져 있다.** 구현된 것은 경보 계산(`--confidence-gate`)이고 기본 off 이며 A8 · A9 표에 없다. 영역을 feature 로 재연결하는 코드는 읽지 못했다. hidden patch 로 다시 찾기는 offline 실험(T79) 이다.
5. **HSR 이라는 이름이 셋을 가리킨다:** 3 단계 분류기(`kv_L4_v1`, 새 대상 gate) · 6 단계 분류기(HSR6, 규칙 ①–④) · 5 단계 분류기(T77, A9 채택, offline). 그리고 **쥠 상태는 어느 분류기도 아니고 규칙 latch 다.** 논문의 "phase 와 grasp state 분리" 는 구조상 맞지만, grasp state 에 "uncertain" 은 없다.
6. **번호 충돌:** `ADOPTED_KO.md` 의 A1–A9 (채택 항목), `T78.task.md` 의 A1–A5 (표면 후처리 단계), HSR 규칙 ①–④, 쥐기 목표 항 "A4" 가 서로 다르다. `A8` = 최종 구성이다.
7. **기하 grasp 보정(G1) · 쥐기 목표 항(A4):** A8 인자에 있지만 작동 0 이었다. 성공은 두 항 없이 나왔고 논문에 기여를 주장하지 않는다. A9 는 A4 를 뺀다.
8. **`--safe-phase approach` 는 client 고정값이다.** phase 를 client 가 단계별로 주지 않는다.
9. **제약은 손바닥 + 손가락만이다 (`--links gripper`).** 논문의 "손목 · 팔은 전체 장면을 본다" 와 다르다. 손목 · 팔은 제약 밖이다.
10. **이웃 접촉 허용은 두 가지다.** 이미 켜진 `--neighbour-contact-mm 20` (손가락) · `--neighbour-palm-contact-mm 10` (손바닥, 쥐는 순간 창) 과, 꺼진 A6 원안 "쥐기 자세를 고를 때 이웃 접촉 허용" (`ADOPTED_KO.md` A8 "이웃 접촉 허용 — 지금 상태와 켜는 규칙").
11. **렌더 로봇 mask 는 쓰지 않는다.** 논문 §4.1 의 렌더 비교 제거는 구현돼 있으나 미채택이다 (A7).
12. **waypoint 사이 보간 충돌 검사(논문 §8) 는 구현돼 있으나 꺼져 있다.** 검증은 waypoint 에서의 실제 거리 재계산이다.
13. **T78 flag 는 작업 트리에 있으나 commit 되지 않았고 검증 전이다.** A9 의 "추가 예정" 을 평가된 사실로 인용하지 않는다.

## 8. 이 문서에서 코드로 확인하지 못한 것

- HSR 논문 §6.1 의 두 번째 분류기 (gripper · 거리 · 높이 · 크기 입력) 존재 여부, 그리고 grasp state 의 uncertain 처리.
- `--confidence-gate on` 에서 switch(re-link) 가 서버 영역을 실제로 교체하는지 (읽은 범위에서는 교체 없음).
- LTE: 매 request 현재 sample 로 target-free 가 다시 만들어지는 범위, local window 밖 fallback 의 정확한 구현, 목적지 접촉 규칙.
- CCR: 검사 시작 자세가 "측정된" 자세인지, refiner 와 verdict 가 같은 snapshot 을 쓰는지.
- T78 flag 가 작업 트리 코드에서 실제로 동작하는지 (재생 · closed-loop 결과 없음).
