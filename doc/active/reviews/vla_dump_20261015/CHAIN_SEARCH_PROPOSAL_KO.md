# 최종 실험 사슬(①–④) 탐색 — 설계 제안과 사용자 확인 사항

작성: 2026-10-18 KST (IFG 후속 세션, CL3 대기 중). 기준: [FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md](FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md) §3.
**상태: 제안. 아래 "확인 사항" 에 사용자 판정이 내려진 뒤 task 문서를 쓴다.** 숫자는 모두 기존 run 기록에서 읽은 것이다.

## 0. 목적을 먼저 적는다

이 탐색은 **시연 · 논문 figure 용 시나리오를 고르는 것**이다. 성공 여부를 보고 장면을 고르므로 성공률 평가(CL3 방식)와 **목적이 다르고 수치를 성능 근거로 쓰지 않는다.**
보고할 것은 "후보 N 장면 중 네 단계를 모두 통과한 장면 k 개" 라는 통과율과, 통과 장면의 영상 · figure 다.

## 1. 네 단계의 정의 (같은 episode · seed, client argv 는 단계 flag 만 다르다)

| 단계 | 조건 | client 구성 | 서버 구성 | 이미 있는 기록 |
|---|---|---|---|---|
| ① | VLA checkpoint 단독 | `--safe-*` flag 없음 (T43 의 E0 규약), `--obstacle-profile clear` | serve_safe 는 그대로 떠 있어도 됨 (E0 요청은 seed 만 벗겨 정책만 호출) | T43 E0 (허들 7 episode, 옛 snapshot) |
| ② | AG3S, 장애물 없음 | CL3 clear 행과 같은 argv | CL3 arm flag (C1aN 또는 SURF) | **CL3 clear 14 장면** (C1aN · SURF 모두, 오늘 끝남) |
| ③ | 허들 + AG3S | `overbar_0` (T43 P2 규칙) | 같음 | **T43 허들 7 episode** (C1a · C1aN · SURF, IFG-HURDLE run) |
| ④ | 허들 + pick 방해 장애물 + AG3S | ③ profile 에 `post_0` 등 pick 장애물을 **같은 profile 안에** 추가 (manager 는 여러 slot 동시 활성 지원) | 같음 | 없음 |

- 허들(③) 배치: T43 P2 search 를 재사용한다. P2 는 **① 과 ② 의 기록된 운반 경로가 모두 막대에 걸리는 위치**(criterion c) 를 고르므로, ① · ② rollout 이 끝난 뒤에 CPU 로 계산한다. 그 script 는 옛 경로(`/mnt/dev/work/...`) 를 박아 두었으므로 storage 경로로 옮겨 쓴다 (코드 수정 = implementer).
- pick 장애물(④) 배치 — **새 규칙** (계획 문서 §2):
  - (a) head camera(`zed_left`) 의 target 가림률: 접근 구간(첫 gripper close 직전 8 step) median 이 **5–60 %**, **30 % 에 가까운 후보를 먼저**. 가림률 계산은 held-out collector `hc_search.py` 의 segmentation render 를 그대로 쓴다.
  - (b) 궤적 간섭: ② 기록의 pre-grasp 구간에서 robot geom 과 장애물의 최소 거리가 **≤ 0 m** (그 경로를 그대로 가면 닿는다).
  - 공통 전제(reset 상태): robot ≥ 3 cm, 다른 과일 · 바구니와 겹치지 않음. **grasp room(target 표면과 4 cm)** 은 가림 30 % 와 양립하기 어려우므로 ④ 에서는 요구하지 않는다 (확인 사항 3).
  - 후보 순서는 hc_search 의 adjacent 격자(거리 6–10 cm, 30° 간격, 높이 3 종, 반지름 2 종) 를 쓰되 **거리 범위를 4–12 cm 로 넓힌다**. 첫 번째로 (a) 또는 (b) 를 만족하는 후보가 아니라, **(a) 통과 후보 중 |가림 − 30 %| 최소** 를 고른다. (a) 통과가 없으면 (b) 통과 중 reset robot 거리 최대.

## 2. 후보 장면 풀 — 선택지

| 풀 | 장면 | 장점 | 비용 (단계별 새 rollout) |
|---|---|---|---|
| **P1 (권장)** | CL3 clear 14 + T43 허들 7 = **21** | ② 14 개 · ③ 7 개가 이미 있다. 허들 episode 는 P2 배치가 이미 있다 | ① 21, ② 7, ③ 14 (P2 search 뒤), ④ ≤ 21 |
| P2 | CL3 40 전부 | 풀이 크다 | P1 + adjacent · place 26 장면의 ② 26, ③ 26, … (약 2배) |
| P3 | CL3 해시 순서 41 번째부터 새 장면 | 어떤 설계에도 쓰지 않은 장면 | 네 단계 모두 처음부터 (장면당 4 run) |

- seed: CL3 장면은 `episode·10+3` (CL3 와 같음), 허들 episode 는 T43 seed `·1`. 장면당 seed 1.
- ① 을 허들 7 episode 에서 T43 E0 기록으로 대신할지, 현재 snapshot 으로 다시 돌릴지 → 확인 사항 4 (권장: 다시 돈다. 같은 snapshot · 같은 client 로 네 단계를 맞춘다).

## 3. 시간 추정 (기존 run 의 wall time 중앙값)

| 구성 | run 당 wall | 출처 |
|---|---|---|
| C1aN (noun attention + 공 영역) | 5.0–5.6 분 | IFG-HURDLE 302 s, IFG-CL3 PHASE A 316–358 s |
| SURF (+ 표면 영역 + robot 렌더) | 7.2 분 | IFG-HURDLE 434 s, IFG-CL3 PHASE A 377–471 s |
| VLA 단독 | 측정 없음 (C1aN 보다 짧다고 가정하지 않는다; 5.5 분으로 잡는다) | — |

P1 · arm 1개(C1aN) · 서버 2개 병렬: ① 21 + ② 7 + ③ 14 + ④ ≤ 21 = **≤ 63 run ≈ 5.8 h GPU 직렬, 병렬 약 3 h** + P2 search · ④ 배치 CPU.
arm 을 둘(C1aN · SURF) 다 돌리면 ②–④ 가 두 배다.

CPU 비용 (GPU lock 없이): T43 P2 허들 탐색은 episode 당 후보 9,180 개 · 1 process 에 394–466 s (`T43/P2/raw/p2search_*.json` `elapsed_s`) → clear 14 장면을 worker 4 로 약 30 분. ④ 배치(가림률 렌더 포함)는 hc_search 와 같은 규모(후보 수백 개)라 장면당 수 분으로 본다 (측정 뒤 갱신).

## 4. 통과 판정 (단계마다 같은 정의)

- 성공 = CL3 의 "접촉 없는 작업 성공" (analyze_cl 성공 ∧ 비허용 link–target 접촉 0 ∧ 장애물 접촉 0, MuJoCo replay). ①(VLA 단독) 도 같은 replay 로 접촉을 센다.
- 단계 통과 여부를 장면 × 단계 표로 모두 기록한다. 중간에 실패한 장면도 다음 단계를 **돌리지 않는다** (사슬이므로). 단, ④ 배치가 기하적으로 불가능한 장면은 `skipped_geometry` 로 기록.

## 5. 산출물

- run: label 붙은 새 run (`ifg-chain-search`, IFG-RUNLABEL 뒤) 또는 그 전이면 `RUN_INDEX_IFG.md` 에 한 줄.
- 표: 장면 × 단계 통과 표 (figure + `.json`). 통과 장면마다 네 단계 영상(`third_person_front.mp4`) 과 손끝 경로 비교 figure (계획 문서 §5 · §6 의 재료).
- 가림률 · 간섭 거리 등 ④ 배치 검사 수치 표.

## 6. 확인 사항 (사용자 판정 필요)

1. **풀:** P1 (21 장면) 로 시작할까, P2/P3 로 갈까.
2. **arm:** 사슬의 AG3S 를 C1aN 하나로 할까, SURF 도 같은 장면에 돌려 figure 에서 나란히 보일까. (CL3 결과를 본 뒤 정한다.)
3. **④ 규칙:** (a) 가림 30 % 우선 → (b) 궤적 간섭 보조, grasp room 4 cm 요구 안 함 — 이 순서와 범위(5–60 %, 거리 4–12 cm)가 맞는가.
4. **① 재실행:** 허들 7 episode 의 VLA 단독도 현재 snapshot 으로 다시 돌린다 (권장) / T43 E0 기록을 쓴다.
5. **규모 상한:** GPU 약 3 h (병렬) 를 한 번에 쓰는 것이 괜찮은가, benchmark 세션과 시간대를 나눌까.
