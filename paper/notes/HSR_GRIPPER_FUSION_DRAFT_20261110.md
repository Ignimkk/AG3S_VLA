# HSR + gripper feedback fusion — manuscript / supplementary draft (2026-11-10 KST, dev-38)

> 상태: `DRAFT` — 원고 (`latex/sections/*.tex`, `supplementary/supplementary.tex`) 는 고치지 않았다.
> 근거: offline 기록 재생 1 건 (decision G174). closed loop 에는 이 결합기가 없다.
> 결과 출처: `outputs/2026-11-10_HSR6_gripper_재생/replay.json` · `close_runs.json` · `README.md`
> script: `experiments/segmentation/hsr6_gripper_replay.py` (2026-11-10 00:10 KST 실행, CPU)
> Claim · Result ID 는 **제안**이며 CLAIMS · RESULTS 에 등록하지 않았다 (아래 §6).

## 0. 무엇을 주장할 수 있고 무엇을 못 하나

| 문장 유형 | 쓸 수 있나 | 상태 표시 |
|---|---|---|
| Panda 에서 gripper 개도만으로는 쥔 그릇과 빈손이 ~3 mm 로만 갈린다 (sim 측정) | 예 | `MEASURED` (dev split, 40 ep) |
| HMM 결합이 probe · latch 단독보다 단계를 더 잘 읽는다 (offline) | 예, 범위를 붙여서 | `EXPLORATORY_DESCRIPTIVE` (재생, proxy label, 1 판) |
| hidden 경보가 빈 파지를 더 빨리 · 덜 틀리게 잡는다 (offline) | 예, 범위를 붙여서 | `EXPLORATORY_DESCRIPTIVE` |
| 목적지 오등록이 줄어든다 | **아니오** | `NOT_ESTABLISHED`: registry 기록이 없는 판이다. 등록 '시점' 정확도만 쟀다 |
| 성공률 · 안전 성공 향상 | **아니오** | `NOT_MEASURED`: closed loop 미시험 |
| Method 의 구성 요소로 서술 | **채택 · closed loop 검증 뒤에만** | 지금은 `PLANNED` |

---

## 1. Method — 두 가지 쓰는 법

### 1A. 결합기를 closed loop 에 넣고 검증한 뒤 (`PLANNED`, 지금 쓰면 안 됨)

`03_method.tex` §HSR 두 번째 문단 ("A phase prediction is intent, not contact. …") 뒤에 붙인다:

```latex
\paragraph{Fusing intent and attachment.}
The two estimates fail in complementary ways. On a parallel gripper holding a thin rim, the measured opening separates a
held object from an empty closure by only a few millimetres, so $z_t$ alone lags or flips; the probe, in turn, confuses
neighbouring stages of the same motion. We therefore filter the phase with a hidden Markov model whose emission is the
product of the probe posterior and the likelihood of the current grasp state,
\begin{equation}
 b_t(s)\;\propto\;p_t(s)\;P(z_t\mid s)\sum_{s'}T_{s's}\,b_{t-1}(s'),
 \label{eq:hsr-fusion}
\end{equation}
where $T$ allows only the task order, a return to approach (a failed or lost grasp), and placement$\to$transport.
$P(z_t\mid s)$ is counted on development episodes. A disagreement between the two sources is itself a signal:
a closing gripper while the policy has returned to approach ($p_t(\mathrm{approach})\geq0.7$ on two consecutive
requests) raises a failing-grasp event that revokes the pending attachment, and a destination may be registered only
when $z_t=\textsc{held}$ and the filtered phase is transport or placement.
```

- 수식 기호는 원고 (`p_t`, `z_t`, `s_t`) 를 따랐다. 단계 이름은 Panda 5 단계다. RB-Y1 원고의 pick / place / home 과 다르므로, 넣을 때 둘의 대응 (approach · grasp → pick, lift · transport · placement → place) 을 한 줄 더 적어야 한다.
- 문턱 0.7, 2 요청 연속, 전이 확률은 replay 에서 고정한 값이다 (`hsr6_gripper_replay.py` docstring). 채택하면 supplementary Tab. `supp-params` 에 추가한다.

### 1B. 채택하지 않을 때 (지금 쓸 수 있는 형태)

Method 는 그대로 두고, `04_experiments.tex` §SafeLIBERO 의 다음 문장을 고친다.

> 현재: "…and the sub-task probe is replaced by the gripper rule because LIBERO demonstrations carry no phase labels."

**이 문장은 기록과 맞지 않는다.** 근거:
- Panda probe 는 simulator phase label 로 학습했다 (SL6-B, 2026-10-29; nb1104 재현, 5-fold macro-F1 0.834).
- 11-04 의 final-panda-v3 판은 closed loop 에서 HSR6 규칙을 켰다.
- 지금 평가 판 (B5 · pe123458) 이 gripper 규칙 (`--subtask-from-latch`, P0m) 을 쓰는 것은 사실이다. 하지만 이유는 "label 이 없어서" 가 아니라 사용자 판정 (P0m) 이다.

수정안 (`DRAFT`):

```latex
… which CCR refines through the robot's Jacobian. The evaluated configuration takes the task stage from the grasp state
alone; a sub-task probe fitted on simulator labels of $\pi_{0.5}$-only LIBERO episodes is analysed offline in
Supp.~\ref{sec:supp-hsr-panda}.
```

---

## 2. Supplementary — 새 절 초안 (LaTeX, `DRAFT`)

`supplementary.tex` 의 "Development ablation tables" 뒤에 둔다. 수치는 모두 `replay.json` 의 값이다.

```latex
\section{Phase and attachment on a parallel gripper: offline replay}
\label{sec:supp-hsr-panda}
\textbf{Question.} On the Panda, does the policy's hidden state add to gripper feedback when reading the task stage and
detecting failed grasps? \textbf{Data.} 40 closed-loop SafeLIBERO Spatial episodes (tasks 0--3 $\times$ development
initial states 0--4, Levels I and II, one seed; 2172 policy requests) of an earlier configuration that logged, per request,
both the sub-task probe posterior (prefix layer-4 key/value, five stages, fitted on $\pi_{0.5}$-only LIBERO episodes not
included here) and the grasp state with its gripper evidence. No test initial state is used. \textbf{Truth.}
Simulator poses: the target is \emph{held} when it is $\geq$2\,cm above its rest height and within 8\,cm of the
end-effector; a closing run that reaches held is \emph{grasp} until then; \emph{lift}/\emph{transport} split at 5\,cm
horizontal travel; \emph{placement} is within 6\,cm of the destination after the first lift; everything else is
\emph{approach}. Contacts are approximated by distances, so the grasp--lift boundary is approximate. Rules and thresholds
were fixed before the replay; the probe was not refitted.

\begin{table}[h]
\centering
\caption{\textbf{Reading the stage (2172 requests).} Three-level = approach / grasp / carrying.
Fusion R lets the grasp state choose the level and the probe the stage inside it; fusion H is Eq.~\eqref{eq:hsr-fusion}
with leave-one-task-out emission counts and a hand-set transition matrix.}
\label{tab:supp-hsr-panda-phase}
\small
\begin{tabular}{lccc}
\toprule
Reader & 3-level acc. & 5-level acc. & 5-level macro-F1\\
\midrule
Grasp state only & .639 & -- & --\\
Probe only & .861 & .711 & .451\\
Rule fusion R & .764 & .682 & .376\\
HMM fusion H & \textbf{.875} & \textbf{.744} & \textbf{.478}\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{\textbf{Why the gripper alone is not enough.} Requests by logged grasp state (rows) and simulator stage (columns).
A held rim leaves a finger gap of 4.8--5.0\,mm, an empty closure 1.5--2.4\,mm.}
\label{tab:supp-hsr-panda-latch}
\small
\begin{tabular}{lrrrrr}
\toprule
Grasp state & appr. & grasp & lift & transp. & place.\\
\midrule
searching & 268 & 11 & 4 & 13 & 0\\
latched & 695 & 5 & 3 & 7 & 0\\
closing & 394 & 47 & 23 & \textbf{185} & 13\\
held & \textbf{56} & 43 & 25 & 254 & 59\\
placed & 27 & 0 & 0 & 0 & 40\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{\textbf{Failing-grasp alarm and destination gate.} Closing runs: 26 reach a hold, 74 close empty, 40 neither.
Latency in environment steps from the start of the run (median). Gate: requests at which a destination could be
registered, against simulator transport/placement; the HMM gate was added after the first replay.}
\label{tab:supp-hsr-panda-events}
\small
\begin{tabular}{lccc}
\toprule
Alarm & Empty caught & Latency & Alarms on held runs\\
\midrule
Gripper (closed to empty) & 48/74 & 11.5 & 8/26\\
Hidden (closing $\land$ policy back to approach) & 32/74 & 9.0 & 2/26\\
Either & 53/74 & 9.0 & 8/26\\
\midrule
Gate & Precision & Recall & Open, target not carried\\
\midrule
Grasp state held & .716 & .548 & 99\\
Held $\land$ HMM in transport/placement & .883 & .448 & 33\\
\bottomrule
\end{tabular}
\end{table}

\textbf{Reading.} The grasp state stays \emph{closing} on 221 of the 626 carrying requests and reports \emph{held} on
56 approach requests, because the rim leaves the fingers only $\sim$3\,mm apart from an empty closure. Trusting it as
the coarse level (fusion R) is therefore worse than the probe alone; using it as one likelihood (fusion H) is better than
either. Seven of the eight gripper alarms on held runs fire while the target is carried; of the two hidden alarms, one
coincides with the target slipping out of the hand and one follows the release. Neither alarm reacts to
closures blocked by something other than the target except the hidden one (7/40).
\textbf{Limits.} One configuration, one seed, development states only; distance-based truth; offline replay of
recorded signals, so the effect on task or safe success is not measured.
```

---

## 3. Supplementary — 그림 · 자료 목록

| ID (제안) | 무엇 | 지금 있는 파일 | 논문에 넣으려면 |
|---|---|---|---|
| SF-H1 | 4 episode 단계 타임라인 (정답 · probe · latch · 결합 R + 손가락 간격) | `outputs/2026-11-10_HSR6_gripper_재생/그래프_단계_타임라인.png` | `/cvpr-figure` 로 다시 그림: 결합 H 행 추가, 영어 축 · 범례, sidecar json, `paper/figures/` 에 저장 |
| SF-H2 | 요약 막대 (3 단계 정확도 · 경보 · 목적지 문) | `…/그래프_요약.png` | 표 3 개와 중복되므로 본문 대신 project page 용으로 쓴다 |
| SF-H3 | 손가락 간격 분포 (쥔 가장자리 vs 빈손) | 없음 | `close_runs.json` 의 `min_gap_mm` 와 HELD run 간격으로 histogram 을 그린다. Tab. latch 의 근거 그림 |
| SV-H1 | 영상: LI t0 e1 (빈 닫기 3 회) · LII t0 e0 (운반 중 빠짐, hidden 경보 obs 270 · gripper 275) | 원 run 의 `videos/` | `gtto_dual_video.py` 로 agentview \| 손목 영상에 단계 · 경보 자막을 넣는다 (CPU) |
| ST-H | 표 3 개 (§2) | `replay.json` 키: `readers.*` · `latch_vs_truth` · `close_runs` · `destination_gate` | 그대로 |
| SP-H | 고정 parameter (문턱 · 전이 확률 · 정답 규칙) | script docstring | 채택하면 `tab:supp-params` 에 행 추가 |

## 4. 원고와 어긋나는 점 (고칠 때 같이 볼 것)

1. §2 의 `04_experiments.tex` SafeLIBERO 문장 (위 1B).
2. `03_method.tex` HSR 은 RB-Y1 기준 (pick / place / home + LR 2 단 결합) 이다. Panda 는 5 단계 probe 를 쓰고 2 단 LR 이 없다. 본문에 Panda 를 쓰려면 이 차이를 한 문장으로 밝혀야 한다.
3. Fig. 2 ④ (오늘 다른 세션이 PAPER_LOG 에 기록한 "hidden state + gripper feedback" 도식) 는 이 결합의 개념도와 맞는다. 다만 그 run 에는 probe 출력이 없다는 메모가 이미 있다. 그림 caption 에 SF-H1 을 연결할 수 있다.
4. supp `tab:supp-phase` (RB-Y1 LR 0.939) 와 이 절 (Panda 0.875) 은 label · 단계 수 · 평가 단위가 다르다. 한 표에 섞지 않는다.

## 5. 이 draft 를 본문 주장으로 올리려면 필요한 것

| 필요 | 왜 | 비용 |
|---|---|---|
| 결합기 + 경보를 Panda 서버에 구현 · closed loop dev 짝 비교 | Method 1A 를 쓰려면 | 공유 파일 수정 (dev-4a 조율) + GPU |
| init 15–49 headline | E1a 5–14 는 개발에 쓴 분할 (G171) | GPU |
| 접촉 기반 정답 (`phase_export` 의 SafeLIBERO 지원) | 거리 근사의 grasp ↔ lift 경계 오차 | CPU, 공유 code 수정 |
| 목적지 registry 가 있는 판 (pe123458) 의 hidden 추출 | "오등록 감소" 를 직접 재려면 | GPU (공유 단계) |
| seed 2 개 이상 | 단일 seed | GPU |

## 6. 제안 ID (등록 안 함)

- **C021** `EXPLORATORY_DESCRIPTIVE`: "On Panda SafeLIBERO development episodes, fusing the hidden-state sub-task probe with the grasp state in an HMM reads the 3-level stage at .875 (probe .861, grasp state .639) in offline replay." → EXP-H1 → R027.
- **R027**: `replay.json /readers/{L latch,P probe,R rule fusion,H HMM fusion}/acc3` = .6390 / .8610 / .7643 / .8748 (n = 2172). 그 밖의 값은 `latch_vs_truth`, `close_runs`, `destination_gate` 에 있다.
- **EXP-H1**: offline replay. run `nb1104-ag3s-final-LI/LII` (fe3abf215dc4). script md5 는 등록할 때 기록한다.

---

## 7. 성공을 가정한 작성안 (사용자 질문 2026-11-10 00:2x, `HYPOTHETICAL` — closed loop 수치는 모두 `\tbd{}`)

> 이 절은 "결합기가 closed loop 에서 성공했다면" 의 **글 구조**다. `\tbd{}` 안의 수치를 결과 전에 채우지 않는다.
> 실제 값은 §1–§3 의 offline 재생 값 (G174) 뿐이고, 아래에서 `[offline]` 으로 표시했다.

### 7.0 '성공' 의 정의 — 결과 전에 고정할 것

| 판정 | 기준 (제안) | 분할 |
|---|---|---|
| 주 판정 | 결합 판 vs grasp-state-only 판, SafeLIBERO SSR 짝 비교에서 결합 판만 성공 > 반대 (McNemar, 단측 아님) | init 15–49 (G171) |
| 보존 | LIBERO 27 (seed 7 · 8 · 9) 성공 감소 ≤ 1 | 평가 seed |
| 기제 | 잘못된 attach 취소 (그릇을 들고 있는데 취소) 횟수 ≤ grasp-state-only 판 | 같은 run |
| 기제 | 목적지 등록 중 '그릇 미보유' 시점 등록 수 감소 | registry 기록 |

기제 두 줄이 없으면 성공해도 "왜" 를 쓸 수 없다. 그러면 Q3 문단을 쓰지 않는다.

### 7.1 어디에 넣나 — 새 contribution 이 아니라 Contribution 2 의 한 조각

원고의 Contribution 2 ("Evidence separation … physical evidence decides attachment") 가 이미 이 이야기의 자리다. 새 bullet 을 만들지 않고 한 구절만 늘린다:

```latex
\item \textbf{Evidence separation for identity, extent, permission, and stage.} Geometry decides membership, centered
internal features decide re-association and consistency, physical evidence decides attachment, and the policy's hidden
state decides the task stage; where the last two disagree, a grasp has failed, and a controlled analysis shows where
internal features help and where they do not.
```

Abstract 에는 한 절만 넣는다 (기존 문장 끝):

```latex
… and the policy's hidden state, read against gripper feedback, tells a failed grasp from a held object
\tbd{N} steps earlier than the gripper alone.
```

### 7.2 Method (§HSR) — 1A 를 다듬은 판

```latex
\paragraph{Intent against attachment.}
The sub-task probe reports what the policy intends; the grasp state reports what the hand holds. Each fails where the
other does not: on a thin rim the measured opening separates a held object from an empty closure by a few millimetres,
while the probe confuses adjacent stages of one motion. We filter the stage with a hidden Markov model whose emission
multiplies the probe posterior by the likelihood of the current grasp state (Eq.~\eqref{eq:hsr-fusion}); the transition
matrix permits only the task order, a return to approach, and placement$\to$transport. Their disagreement is used
directly. A closing gripper while the policy has already returned to approach marks a failed grasp and revokes the
pending attachment before the opening settles; a destination is registered only while the object is held and the
filtered stage is transport or placement.
```

- 첫 문장 대구 (intends / holds) 가 이 절의 요지다. 리뷰어가 이 한 줄만 읽어도 기제를 알 수 있다.
- "probe 가 좋다" 가 아니라 "둘이 서로 다른 곳에서 실패한다" 를 앞에 둔다. 그래야 결합할 이유가 생긴다.

### 7.3 Experiments Q3 ("which signal decides what") — 새 문단

```latex
\textbf{Intent and attachment disagree where grasps fail.} On a parallel gripper the grasp state alone reads the coarse
stage at 0.639 and the sub-task probe at 0.861; their HMM fusion reaches 0.875, while letting the grasp state choose the
level drops to 0.764 [offline, Supp.~\ref{sec:supp-hsr-panda}]. The reason is a 3\,mm margin: the grasp state stays
\emph{closing} on 35\% of carrying requests. Used as an alarm, the probe catches \tbd{x}/\tbd{n} empty closures
\tbd{k} steps before the gripper and fires on \tbd{m} held grasps, against \tbd{m'} for the gripper threshold
[closed loop, held-out]. Closed loop, the fused configuration turns \tbd{a} grasp-state-only failures into safe
successes and loses \tbd{b} (McNemar $p=\tbd{}$, Tab.~\ref{tab:ablation}); \tbd{c} of the gained episodes contain a
revoked empty grasp and \tbd{d} a destination that the grasp state alone would have registered before the object was lifted.
```

- 구조: (1) offline 의 정확도 → (2) 왜 그런가 (3 mm) → (3) 경보로 쓸 때 → (4) closed loop 짝 → (5) **얻은 episode 가 실제로 그 기제를 거쳤나**. (5) 가 없으면 상관만 있는 결과다.
- 진 episode (`\tbd{b}`) 도 반드시 쓴다. 0 이 아니면 어떤 경우인지 한 구절 붙인다 (예: hidden 이 늦게 확인해서 목적지 등록이 늦어짐 → recall 0.548 → 0.448 [offline] 의 대가).

### 7.4 Ablation 표 (Q1 또는 SafeLIBERO 표에 행 추가)

```latex
% rows: same checkpoint, same seeds, held-out init 15--49
Stage source              & TSR & CAR & SSR & false revocations & early destinations\\
Grasp state only (P0m)    & \tbd{} & \tbd{} & \tbd{} & \tbd{} & \tbd{}\\
Sub-task probe only       & \tbd{} & \tbd{} & \tbd{} & \tbd{} & \tbd{}\\
HMM fusion (ours)         & \tbd{} & \tbd{} & \tbd{} & \tbd{} & \tbd{}\\
HMM fusion, no alarm      & \tbd{} & \tbd{} & \tbd{} & \tbd{} & \tbd{}\\
HMM fusion, no dest. gate & \tbd{} & \tbd{} & \tbd{} & \tbd{} & \tbd{}\\
```

- 마지막 두 행은 경보와 목적지 문을 따로 끈 것이다. 어느 쪽이 성공을 만들었는지 가른다. GPU 비용이 2 arm 늘어난다.
- "probe only" 행은 지금 offline 에서 0.861 로 이미 강하다. closed loop 에서도 결합과 차이가 작으면, 정직한 결론은 "hidden 이 주역이고 gripper 는 보조" 다. 그 경우 7.2 의 첫 문장을 그 방향으로 바꾼다.

### 7.5 Limitations 한 줄

```latex
The stage model is fitted on simulator labels of one policy checkpoint and one gripper; its emission table and
alarm threshold transfer to another gripper only after recounting on that hardware's development episodes.
```

### 7.6 Supplementary — 성공 시 추가할 것

- §2 의 offline 절을 그대로 두고 아래 세 가지를 추가한다.
  - closed loop 짝 비교의 episode 목록
  - 얻은 · 잃은 episode 의 기제 분류 표: 빈 파지 취소 / 목적지 지연 / 기타
  - 대표 영상 2 편: 얻은 episode 1 · 잃은 episode 1 (agentview | 손목, 단계 · 경보 자막)
- 고정 parameter 표에 3 행을 넣는다: 전이 확률, 경보 문턱 0.7 × 2 요청, emission 집계 분할.
