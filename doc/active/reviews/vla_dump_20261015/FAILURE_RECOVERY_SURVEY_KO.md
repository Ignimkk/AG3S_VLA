# 정책 기반 제어에서 추가 학습 없이 실패를 회복 · 재시도하는 방법 — 문헌 조사

작성: 2026-10-21 KST.

- 사용자 지시: "VLA 혹은 diffusion policy와 같은 정책 기반 제어에서 추가 학습 없이 작업 실패를 회복하여 재시도 하는 방식이 무엇이 있는지 찾아."
- 조사: literature-analyst sub-agent(웹 조사만, 우리 저장소 파일은 읽지 않음)
- 출처 확인 범위
  - arXiv abs/HTML 페이지와 프로젝트 페이지를 직접 열어 확인했다
  - PDF 본문까지 읽은 것은 Sentinel의 p.1–6뿐이다
  - 2026년 논문은 대부분 peer review 전 preprint다
  - venue를 확인하지 못한 것은 `UNVERIFIED`로 표시했다
- 이 문서는 조사 보고를 옮긴 것이다. 수치는 각 논문이 보고한 값이며 우리가 재현하지 않았다.

## 결론

우리 실패 사례(빈 grasp 뒤 VLA가 place로 넘어감)에 가장 가까운 training-free 선행연구는 두 편이다.

- **ProbeAct (arXiv 2606.09740)**
  - hidden-state probe와 kinematic state machine으로 빈 grasp를 탐지하고 회복한다
  - 다만 probe는 시뮬레이터 정답 위치로 **학습**한다
- **Sheng et al. (arXiv 2610.06921)**
  - frozen π0.5에서 실패를 감지하면 몇 초 전 joint 자세로 후퇴(retreat)시킨 뒤 다시 정책에 맡긴다
  - 이 단순한 후퇴와 학습된 corrector 사이에 차이가 검출되지 않았다

## 1. 방법 표

| method | venue/year | policy | detection | recovery | training-free? | eval / 수치 | link |
|---|---|---|---|---|---|---|---|
| Retreat vs learned corrector (Sheng, …, Tang) | arXiv 2026-10 (venue 없음) | frozen π0.5 | 시간 threshold + stall monitor(손으로 정한 상수 5개). held-out 실패 208개 중 놓친 것 0개, 성공 episode의 18%에서 오경보 | 120 step 전 joint 자세로 선형 보간해 후퇴한 뒤 정책에 반환. 비교 대상인 learned corrector(MLP)는 cuRobo teacher가 만든 rescue 데이터로 학습 | retreat 방식은 완전히 training-free | RoboTwin 2.0 4개 task. bbh에서 timer+retreat가 +15.3pp, 학습 pipeline과 차이 +0.7pp (90% CI [−2.6, +4.0]). 나머지 3개 task는 gain 없음. 같은 seed 재실행만으로도 bbh 실패 episode의 43.2%가 성공 | [arXiv](https://arxiv.org/html/2610.06921) |
| ProbeAct (Zhang et al.) | arXiv 2026-06, under review | OpenVLA-OFT | layer 8 hidden state에서 물체 3D 위치를 예측하는 MLP probe + kinematic state machine (hard / soft empty grasp, drop) | 첫 실패는 push-back 후 VLA에 다시 맡김. 같은 위치에서 반복 실패하면 그 지점에 CBF safe zone을 만들어 pre-grasp 동작을 보정 | policy는 frozen. **probe는 학습**(시뮬레이터 정답으로 만든 50k 쌍) | LIBERO-plus 69.6 → 74.1%. sim만 평가 | [arXiv](https://arxiv.org/abs/2606.09740) |
| CoRe (Zhang et al.) | arXiv 2026-08 | π0, π0.5, GR00T, X-VLA | action expert suffix latent의 Mahalanobis 거리(K = 8 pass). threshold는 성공 rollout의 0.95 quantile, 3회 연속이면 trigger | 최근의 viable anchor에서 이어질 장면을 counterfactual로 합성 → 필요한 최소 물체만 복원 → arm을 rejoin 자세로 → 기록을 anchor까지 잘라냄 | policy · monitor 모두 학습 없음 | LIBERO-Long 교란 12.8–43.5 → 69.8–88.2%. Arm-only 변형도 65.0–81.9%. 실물 π0.5: bare ≤ 9/20, CoRe 17–19/20. 회복 1회에 약 9.7 s | [arXiv](https://arxiv.org/html/2608.14822) |
| Causal-History TTS (Li et al.) | arXiv 2026-09 | autoregressive world-action model | 진행 정체를 보는 progress-aware trigger | KV history의 신뢰 가능한 prefix를 복원하고, 세 가설을 비교해 선택 | 주장 | 수치는 abstract에 없음, UNVERIFIED | [arXiv](https://arxiv.org/abs/2609.18016) |
| B2FF (Shin et al.) | arXiv 2026-06 | foresight VLA | 실패를 주입한 시점에 trigger | 미리 상상한 milestone을 visual goal로 고정 | low-level 학습 없음, 기타는 UNVERIFIED | 실패 주입 LIBERO 56.3 → 74.0% | [arXiv](https://arxiv.org/abs/2606.09258) |
| ActFovea (Yu et al.) | arXiv 2026-07 | π0 | 시각 motion과 kinematics · action의 일치 검사 | 후보 observation 생성 → action chunk 검증 후 채택. 회복 불가면 safe-fail | 주장 | LIBERO overlay 교란 49.3 → 90.3% | [arXiv](https://arxiv.org/abs/2607.29169) |
| Sentinel (Agia et al.) | CoRL 2024, PMLR 270 | Diffusion Policy | STAC(chunk 분포 거리 누적합, conformal threshold) + zero-shot VLM 진행 판정 | **회복 없음**. 경보 시 정지 | 성공 rollout으로 calibration만 | 실패 > 97% 탐지 | [PMLR](https://proceedings.mlr.press/v270/agia25a.html) |
| BID (Liu et al.) | arXiv v4 2025-04 (ICLR 2025 workshop은 UNVERIFIED) | DP, VQ-BeT | 없음(예방형) | N = 16 sample 중 backward coherence + forward contrast로 선택 | policy 학습 없음, 단 weak checkpoint 필요 | sim 7 task 상대 +32%, 실물 약 2× | [arXiv](https://arxiv.org/html/2408.17355v4) |
| FOREWARN (Wu et al.) | arXiv 2025 (RSS는 UNVERIFIED) | DP | VLM이 미래를 상상해 평가 | 100 sample → 6 mode → VLM 선택 | **world model · LoRA VLM 학습** | 실물 3 task, 0.1–0.5 → 0.6–0.8 | [arXiv](https://arxiv.org/html/2502.01828v3) |
| VLS (Liu et al.) | arXiv 2026-02 | diffusion / flow | — | VLM이 만든 미분 가능한 reward로 denoising을 guide | 예 | CALVIN +31%, LIBERO-PRO +13% (기존 steering 대비) | [arXiv](https://arxiv.org/abs/2602.03973) |
| VLA-Pilot (Li et al.) | arXiv 2025 (RA-L accept는 UNVERIFIED) | VLA | — | MLLM + evolutionary diffusion steering | 주장 | 실물 6 task, 수치 UNVERIFIED | [arXiv](https://arxiv.org/abs/2511.14178) |
| Code-as-Monitor (Zhou et al.) | CVPR 2025 (arXiv 주석 기준) | open-loop policy | VLM이 constraint 검사 code 생성 | closed-loop 재계획 | UNVERIFIED | 교란 조건 success +28.7%, 시간 −31.8% | [arXiv](https://arxiv.org/abs/2412.04455) |
| ReKep (Huang et al.) | arXiv 2024 (CoRL은 UNVERIFIED) | VLM keypoint 최적화(VLA 아님) | constraint 추적 | stage를 넘나드는 재계획 | task별 데이터 없음 | 실물 | [page](https://rekep-robot.github.io/) |
| Inner Monologue / REFLECT | arXiv 2022 (UNVERIFIED) / CoRL 2023 | LLM planner + skill | success detector, LLM 실패 설명 | skill 재계획 · 재시도 | LLM 추가 학습 없음 | UNVERIFIED | [IM](https://arxiv.org/abs/2207.05608), [REFLECT](https://arxiv.org/abs/2306.15724) |
| CorrectVLA (Kwon et al.) | arXiv 2026-08 | VLA | 사람이 판단 | 언어 교정 → action 크기의 additive 보정 | 예 | LIBERO-90 + xArm7, UNVERIFIED | [arXiv](https://arxiv.org/abs/2608.29967) |

training-free가 **아니어서** 표에서 뺀 것은 다음과 같다.

- ReCoVLA(2606.09630): residual RL 학습
- RoboMonkey(2506.17811): VLM verifier 학습
- V-GPS(CoRL 2024): offline RL value 학습
- Hide-and-Seek: detector 학습
- FailSafe(2510.01642): 원문 미확인, UNVERIFIED

instruction만 다시 넣어 재시도하는 방식을 π0.5 계열에서 검증한 1차 문헌은 찾지 못했다.

## 2. 우리 설정에 적용할 우선순위 (조사 agent의 제안, 검증 전)

1. **Grasp latch로 trigger → 후퇴 → 재접근** (Sheng et al. + ProbeAct의 detector 정의)
   - 탐지: 빈 grasp를 우리 latch로 잡는다
     - hard: gripper 최소 폭인데 attach 없음
     - soft: EE는 올라가는데 target extent는 그대로
     - 모순 trigger: phase probe는 place인데 latch는 attach가 아님
   - 회복: pick 진입 때 pre-grasp joint 자세를 anchor로 저장한다. 실패하면 gripper를 열고 anchor로 후퇴한 뒤 π0.5에 새로 질의한다
   - 우리 쪽 차별점: 후퇴 경로를 TrajOpt/ESDF로 통과시킨다. Sheng et al.은 scripted retreat로 파손이 placebo 대비 +12.6pp 늘었다
   - 필요한 것: 같은 seed 재실행 placebo 대조군. 같은 seed의 base 실행끼리도 39–60% seed에서 결과가 갈렸다
2. **CoRe의 Arm-only 변형 (anchor rollback)**
   - 탐지: π0.5 action expert hidden state에 Mahalanobis OOD를 적용한다. threshold는 성공 rollout의 quantile이고 신경망 학습은 없다
   - 회복: arm만 anchor 자세로 되돌린다
   - 필요한 것: 성공 episode calibration set. 우리 serving이 history를 쓰는지 확인해야 한다
3. **반복 실패 시 재접근 위치 보정**
   - ProbeAct의 학습된 probe 대신 attention + RGB-D target 3D를 쓴다
   - 같은 위치에서 반복 실패하면 그 지점을 SQP 제약 · cost로 넣는다
4. **Sampling + geometric verifier (BID / FOREWARN-lite)**
   - π0.5 noise seed를 바꿔 chunk 여러 개를 뽑는다
   - 일관성 + ESDF clearance + target 접근도로 고른다
   - latency가 늘어난다
5. **Sentinel STAC + VLM 진행 판정**: latch가 놓치는 stall 실패를 잡는 보조 monitor다. 성공 rollout 약 50개로 calibration한다

## 3. Novelty 영향

- "hidden-state probe + kinematic grasp-failure 탐지 + frozen VLA 회복"은 ProbeAct가, "frozen π0.5 + cuRobo + retreat"는 Sheng et al.이 이미 보고했다. 재시도를 기여로 내세우려면 둘 다 인용하고 차별점을 밝혀야 한다
- 차별 축 후보: attention으로 target 선택, target을 제외한 ESDF 위의 TrajOpt 후퇴 경로, 실물 평가. 사전에 우월하다고 가정하지 않는다
- UNVERIFIED 항목: BID · FOREWARN · ReKep · Inner Monologue의 venue, VLA-Pilot · CorrectVLA · B2FF · Causal-History TTS · Code-as-Monitor의 상세 수치, Code-as-Monitor의 training-free 여부, FailSafe의 내용
