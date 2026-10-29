# Research Step R1 — 기존 PDF 3편 분석

선정 상태: `CANDIDATE_USER_SELECTION_PENDING`.
사용자는 선행연구를 아직 선정하지 않았으며 추후 후보를 간추려 제공한다.
이 문서는 기존 로컬 자료의 탐색적 분석이고, 최종 related-work/baseline 선정 결과가 아니다.
현재 원고 인용도 임시 초안이다. 사용자 후보 목록을 받은 뒤 분석·포함 범위를 조정한다.

분석일: 2026-10-06 KST. 범위: 로컬 PDF의 아래 핵심 section과 primary 서지 자료.
전체 선행연구를 망라한 survey·novelty 판정·baseline 구현 검증은 아직 수행하지 않았다.
PDF 페이지는 뷰어 기준 1부터 센다. 원본 PDF와 연구 코드·결과는 수정하지 않았다.
파일 hash·버전·읽은 페이지는 [REFERENCE_PROVENANCE_R1](REFERENCE_PROVENANCE_R1.json)에 있다.

## RW001 — LoHoVLA

- Citation key: `yang2025lohovla`.
- 저자: Yi Yang, Jiaxuan Sun, Siqi Kou, Yihan Wang, Zhijie Deng.
- 제목: *LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks*.
- 확인한 공개본: [arXiv:2506.00411v1](https://arxiv.org/abs/2506.00411v1),
  최초 공개 2025-05-31. 로컬 첫 페이지의 ID·저자·제목과 일치한다.
- 인용 유형: 2025 arXiv preprint. 확인한 arXiv record에는 venue가 표시되지 않는다.
  이것으로 peer-reviewed 출판이 없다고 단정하지 않으며 CVPR acceptance를 기재하지 않는다.

| 축 | 실제 내용·근거 | 우리 연구에 주는 의미 |
|---|---|---|
| 문제·표현 | §3.1–3.3, pp.3–5: RGB/depth top-down reconstruction과 언어 goal로 다음 sub-task와 pick/place action을 순차 생성 | Task-conditioned VLA는 이미 존재한다. Language sub-task와 내부 attention의 3D 접지 후보는 구분해야 한다 |
| 모델·학습 | §3.3–3.4, pp.5–6: PaliGemma backbone, action을 1,024 bins로 이산화; text/action loss, 두 단계 학습 | 우리 16-D joint action chunk와 같은 출력·제어 문제라고 가정할 수 없다 |
| 폐루프 | Algorithm 1, p.5 및 §4.3, p.8: 환경 reward·done 신호와 실패 횟수로 sub-task 재계획, action 재예측 | Reward 접근 조건과 replanning 비용을 맞추지 않으면 공정한 직접 baseline이 아니다 |
| 평가 | §4.1–4.4, pp.6–9: Ravens/UR5e suction, pose/zone task score와 success, closed-loop/training ablation | Task score를 collision/safe-success로 바꾸어 인용하지 않는다. Planning success는 LLM 판정도 사용한다(§4.4, p.9; Appendix B, p.17) |
| 저자 한계 | §5, p.10: 이산 action의 정밀도와 sub-task가 한 timestep에 끝난다는 가정 | Long-horizon success를 실제 고주파 joint-space refinement의 증거로 일반화할 수 없다 |

Overlap: 목표에 따라 action을 생성하고 폐루프로 오류에 대응한다.
확인한 §3–4에서는 attention→multi-view 3D relevance→ESDF contact constraint라는
우리 후보와 동일한 절차를 확인하지 못했다. 이는 모든 VLA 문헌에 그 절차가 없다는 뜻이 아니다.
직접 실행 baseline보다 task reasoning/closed-loop 문맥의 reference가 우선이다.
공식 code/checkpoint 공개 주소는 이번 primary-source 확인 범위에서 검증하지 못했다.

## RW002 — cuRoboV2

- Citation key: `sundaralingam2026curobov2`.
- 저자: Balakumar Sundaralingam, Adithyavairavan Murali, Stan Birchfield.
- 제목: *cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots*.
- 확인한 공개본: [arXiv:2603.05493v2](https://arxiv.org/abs/2603.05493v2).
  v1 최초 공개 2026-03-05, v2 개정 2026-04-16. 로컬 PDF의 첫 페이지에도 v2가 명시된다.
  PDF 내부 표지 날짜 2026-04-17과 arXiv version timestamp는 별개로 기록한다.
- [저자 공식 technical reports](https://nvlabs.github.io/curobo/latest/technical_reports.html)는
  technical report와 [공식 code](https://github.com/NVlabs/curobo)를 제공한다.
  인용 유형은 2026 technical report이며 peer-reviewed venue를 추가하지 않는다.

| 축 | 실제 내용·근거 | 우리 연구에 주는 의미 |
|---|---|---|
| 최적화 | §3–4, pp.6–9: cubic B-spline control points, inverse dynamics/RNEA, torque·joint·collision 관련 비용/제약; L-BFGS update | ESDF 기반 optimization 자체는 새로운 기여로 주장할 수 없다. 우리 SQP/OSQP와 solver·변수·목적이 다르다 |
| 거리장 | §5.1–5.4, pp.9–13: depth와 primitive를 block-sparse TSDF에 융합하고 PBA+ dense ESDF 생성; TSDF/ESDF 해상도 분리 | Depth fusion·dense ESDF·task-specific resolution은 명시적인 prior art다 |
| 관측 한계 | §5.4, pp.12–13: thin surface seeding 누락 가능; depth truncation band 밖 sign의 한계와 미관측 voxel positive 기본값 | Dense query가 가능하다고 hidden obstacle까지 완전 관측한 것은 아니다. Unknown-space 의미·GT coverage가 필요하다 |
| 물리 범위 | §6 및 §7.1, pp.14–20: whole-body/self-collision, inverse dynamics와 3 kg payload의 torque feasibility | Payload mass dynamics와 운반 물체의 collision geometry는 별도로 확인해야 한다. 우리 snapshot의 coverage를 대신 입증하지 않는다 |
| 거리장 평가 | §7.6, pp.30–32: Redwood depth, nvblox와 시간/메모리/collision recall 비교; recall은 nvblox valid-distance 영역에 제한 | 이 recall을 전 workspace physical collision GT나 우리 safety result로 인용하지 않는다 |
| 실제 폐루프 | §7.7/Fig.23–24, pp.32–33: YAM, ZED Mini depth, robot sphere projection segmentation, live TSDF/ESDF+MPC | Depth-driven reactive collision avoidance는 실제 prior work다. Robot removal은 task-target 접촉 허용과 다른 처리다 |
| 저자 한계·향후 방향 | §9, p.38: depth-error에 민감한 segmentation, single-camera partial coverage; multi-camera fusion과 learned-policy safety layer 제안 | 넓은 아이디어도 선행 제안과 겹친다. 후속 제안과 구현/실험 완료를 구분한다 |

Overlap가 큰 engineering reference다. 후보 차별성은 VLA 내부 relevance의 추출·검증,
phase별 target/payload/obstacle 규칙, action chunk의 task–safety trade-off에 좁혀 검증한다.
cuRoboV2 MPC/geometry refinement는 향후 비교 후보지만 동일 robot/limits/observations/
contact rule/latency budget에서 실제 adapter를 확인하기 전 baseline 완료로 표시하지 않는다.
현재 B002는 cuRobo ESDF를 소비하는 프로젝트 SQP/OSQP이며 cuRoboV2 MotionGen/MPC가 아니다.

## RW003 — cuRobo minimum-jerk technical report

- Citation key: `sundaralingam2023curobo`.
- 저자: Balakumar Sundaralingam, Siva Kumar Sastry Hari, Adam Fishman, Caelan Garrett,
  Karl Van Wyk, Valts Blukis, Alexander Millane, Helen Oleynikova, Ankur Handa,
  Fabio Ramos, Nathan Ratliff, Dieter Fox.
- 제목: *cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*.
- 공개 record: [arXiv:2310.17274](https://arxiv.org/abs/2310.17274),
  v1 2023-10-26, v2 2023-11-03(62 pages).
  로컬은 62 pages지만 arXiv version stamp가 없다. `curobo_report_v1.pdf`의 `v1`은
  파일명이며 version 확정 근거가 아니다. 로컬·공개 PDF의 byte 동일성은 검증하지 않았다.
- [공식 technical reports](https://nvlabs.github.io/curobo/latest/technical_reports.html)는
  minimum-jerk report와 ICRA 2023 초기 논문을 구분한다. 초기 논문 제목은
  *CuRobo: Parallelized Collision-Free Robot Motion Generation*이며 minimum-jerk 항목이 없다.
  초기 논문의 venue/DOI를 확장 report에 붙이지 않는다.

| 축 | 실제 내용·근거 | 우리 연구에 주는 의미 |
|---|---|---|
| 입출력·solver | §2/Fig.2/§4, pp.6–7,10–11: start joint configuration+goal pose→joint trajectory; parallel IK seeds, particle optimization, L-BFGS, 필요 시 geometric plan | Goal-pose MotionGen과 VLA action-chunk 보정은 입력·objective·실행 방식이 다르다 |
| geometry | §3, pp.7–10: robot spheres, self/world collision, signed-distance queries, sweep approximation·speed-weighted 비용 | Robot spheres·ESDF gradient·constraint refinement라는 일반 조합은 prior art다 |
| collision 조건 | §3.4, p.9: waypoint 사이 sphere가 선형 이동한다고 근사; §4, p.10: hard constraints를 큰 soft-cost weight로 근사 | Collision-free라는 제목을 무조건적인 physical safety theorem으로 쓰지 않는다 |
| 평가 | §6.1, pp.13–15: motion-planning benchmarks, collision-free+joint limits+goal tolerance; p.15 cuboid representation으로 일부 geometry 근사 | Planning success와 manipulation 성공·독립 contact GT를 구분해야 한다 |
| depth 연결 | §6.4, p.27: nvblox ESDF와 Realsense를 사용한 UR10e obstacle-avoidance deployment | Depth-camera ESDF+cuRobo 결합을 우리 최초 기여로 주장할 수 없다 |
| 저자 한계 | §8.1, pp.32–33: reactive planning tail latency, contact-rich/full dynamics, partial-world sensing 문제 | 원 report의 한계를 cuRoboV2에도 그대로 적용하지 않는다. V2의 개선과 나란히 비교한다 |

공식 code 공개는 확인했지만 프로젝트가 사용한 cuRobo commit/API와 report 버전의 대응은
다음 구현 snapshot audit에서 확인한다. 라이브러리 사용만으로 report의 모든 기능을
우리 pipeline에서 사용한다고 쓰지 않는다.

## 이번 Step의 수정 결론

1. 확인한 preprint/technical-report 서지로 3개 citation을 등록하고 English Related Work에 연결한다.
2. 넓은 novelty 후보를 검증 대상으로 유지하되, 이미 구현되거나 제안된 부분을 명시한다.
3. 동일 환경의 geometry-only refinement와 relevance/contact ablation을 향후 비교 후보로 둔다.
4. ACoT-VLA/ActiveVLA/AVA-VLA/OptimusVLA/SaPaVe와 attention/3D grounding/safety-shield
   문헌이 남아 있다. 이름·venue·결과를 채워 넣지 않는다.
5. 최종 novelty·task preservation·physical safety는 `UNVERIFIED`/hypothesis 상태를 유지한다.

후속 순서는 [TODO](TODO.md)를 따른다. R1은 실험을 실행하거나 기존 claim/result의 상태를
승격하는 작업이 아니다. 문헌의 성과를 우리 결과로 옮기지 않는다.
