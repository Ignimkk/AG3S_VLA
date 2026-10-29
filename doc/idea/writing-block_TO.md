# TrajOpt 충돌 회피 및 Grasp Goal 유지 개선

## 1. 현재 문제

현재 VLA가 생성한 궤적을 TrajOpt로 보정할 때, 장애물과의 충돌은 회피하지만 **원래 목표인 Grasp를 수행하지 못하는 문제**가 발생한다.

예상 원인:
- **Goal Drift:** 충돌 비용을 줄이는 과정에서 원래 Grasp Goal이 유지되지 않음.
- **Local Minimum:** 단일 초기 궤적만 사용하기 때문에 장애물의 적절한 우회 경로를 찾지 못함.
- **Task Awareness 부족:** 충돌 회피와 실제 작업 목표 달성이 독립적으로 처리됨.
- **Contact Constraint 문제:** Grasp를 위해 필요한 Target 접촉까지 충돌로 처리할 가능성.

## 2. 추천 방법 4가지

### 방법 1. Goal-Constrained TrajOpt

**목적:** 충돌 회피 과정에서도 Grasp Goal을 유지한다.

적용 방법:
- VLA 궤적은 Reference Trajectory로 사용한다.
- Grasp Pose 또는 현재 Subtask Goal을 명시적으로 정의한다.
- Goal Position/Orientation Error를 제약조건 또는 강한 비용으로 설정한다.
- Collision Cost와 Smoothness를 최소화하되 Goal 도달 여부를 최종 검증한다.
- 50-step Action Chunk가 Grasp 단계에 도달하지 않는다면 최종 Grasp를 강제로 적용하지 않는다.

### 방법 2. Multi-Seed Trajectory Optimization

**목적:** Local Minimum 문제를 완화하고 다양한 장애물 우회 경로를 탐색한다.

적용 방법:
- 기존 VLA 궤적 이외에 여러 초기 궤적을 생성한다.
- 장애물 좌우 우회, IK 기반 궤적, 노이즈 기반 초기값을 활용한다.
- GPU에서 여러 후보를 병렬 최적화한다.
- 충돌 여부, Goal 도달 오차, 궤적 길이, Smoothness를 기준으로 최적 후보를 선정한다.

### 방법 3. GPU MPPI + L-BFGS / Graph Planner

**목적:** 충돌 회피 경로 탐색과 궤적 정밀 최적화를 분리한다.

적용 방법:
- MPPI를 사용하여 여러 경로 후보를 탐색한다.
- GPU L-BFGS를 사용하여 후보 궤적을 정밀 최적화한다.
- 장애물로 인해 우회 경로를 발견하지 못한다면 RRT/PRM 기반 Graph Planner로 초기 경로를 생성한다.
- 기존 CPU IPOPT/SQP를 Baseline으로 유지하여 성능을 비교한다.

### 방법 4. Subtask-Aware Selective Contact

**목적:** Grasp에 필요한 접촉과 금지된 충돌을 구분한다.

적용 방법:
- 작업을 Approach → Grasp → Transport로 구분한다.
- Approach: Target 및 Non-target 충돌 회피.
- Grasp: 지정된 Finger–Target 접촉만 허용.
- Transport: 파지한 물체의 형상을 고려하여 주변 장애물 회피.
- 작업 단계와 Robot Link에 따라 Collision Constraint를 동적으로 변경한다.

## 3. 먼저 수행할 실험

**기존 구현의 실패 원인을 먼저 분석하고, 방법을 한 번에 적용하지 말고 순차적으로 검증한다.**

| 실험 | 설정 | 검증 목적 |
|---|---|---|
| E0. Baseline | 기존 TrajOpt | 현재 실패 유형 분석 |
| E1. Goal Constraint | 기존 솔버 + Goal 제약 | Goal Drift 여부 확인 |
| E2. Multi-Seed | E1 + 16/32/64 Seeds | Local Minimum 문제 확인 |
| E3. GPU Planning | E2 + MPPI/L-BFGS | 탐색 성능 및 최적화 시간 비교 |
| E4. Graph Planning | E3 실패 사례 + RRT/PRM | 복잡한 장애물 우회 가능성 확인 |
| E5. Selective Contact | 가장 우수한 Planner + 단계별 접촉 제약 | 실제 Grasp 성공 및 안전성 확인 |

### 핵심 평가지표

- Grasp Success Rate (%)
- Safe Grasp Success Rate (%)
- Goal Position Error (cm)
- Goal Orientation Error (deg)
- Forbidden Collision Rate (%)
- Optimization Time (ms)
- Trajectory Length / Smoothness

### 실험 원칙

1. 동일한 VLA 출력, 초기 로봇 상태, 장애물 환경을 사용한다.
2. 먼저 VLA의 Grasp Goal 자체가 유효하고 IK로 도달 가능한지 확인한다.
3. 최적화 전후 Goal Error를 비교한다.
4. **E1에서 문제가 해결되면 불필요한 Planner 교체는 보류한다.**
5. E1에서도 충돌 회피와 목표 도달을 동시에 만족하지 못하면 E2~E4를 수행한다.
6. 최종적으로 안전 제약과 Grasp 성공률을 함께 검증한다.

## 4. 구현 요청

현재 코드 구조와 TrajOpt 목적함수, 제약조건, Gradient 계산 및 Solver 구성을 먼저 분석하라.

위 4가지 방법 중 현재 실패 원인을 해결할 수 있는 최소한의 변경부터 제안하고 구현하라.

기존 기능을 유지하면서 각 실험을 독립적으로 ON/OFF할 수 있도록 구성하고, 실험 결과를 정량적으로 비교하라.

**최종 목표는 Collision-Free Trajectory가 아니라, Grasp Goal을 유지하면서 Forbidden Collision 없이 작업을 성공시키는 Trajectory를 생성하는 것이다.**