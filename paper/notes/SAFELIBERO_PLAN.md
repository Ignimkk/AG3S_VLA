# SafeLIBERO 실행 계획 — 제안 · 주의 · 구현 (핵심)

작성: 2026-10-07 KST · 상태: `PLANNED` (조사·구현·실행 전)
목적: VLSA·KNOWS·WBAG와 **같은 벤치마크에서** AG3S VLA를 비교한다. MuJoCo(RB-Y1)는 메커니즘·ablation 전용으로 역할을 나눈다.
근거: [REFERENCE_ANALYSIS_R2](REFERENCE_ANALYSIS_R2.md) (VLSA·KNOWS 원문 정리), [R3](REFERENCE_ANALYSIS_R3.md) (WBAG), [develope_PLAN](../../develope_PLAN.md).

---

## 1. 확정된 실험 구성 (사용자 판정, 2026-10-07)

**용어.** *Benchmark* = 시험 환경과 평가 규칙(SafeLIBERO, LIBERO-Safety, 우리 MuJoCo 장면).
*Baseline* = 비교 대상 방법(pi0.5 단독, VLSA, KNOWS, WBAG). VLSA는 방법(baseline)이고, 그 논문이 만든 SafeLIBERO는 benchmark다.

| 환경 | 비교 | baseline 수치를 얻는 방법 | 논문 위치 |
|---|---|---|---|
| **SafeLIBERO** | pi0.5 + AG3S 대 pi0.5 단독 · VLSA · KNOWS · WBAG (2–3개) | pi0.5 단독과 AG3S는 직접 실행. 세 baseline은 각 논문 수치를 **그 논문의 지표 정의대로** 인용(조건이 다르면 별도 블록), 코드가 있으면 재실행 | 주 결과 표 |
| **MuJoCo (RB-Y1, 자체 과제)** | AG3S 대 baseline 3–4개 | GitHub 코드가 있으면 사용, 없으면 핵심을 재구현하고 "re-implemented"로 표시(KNOWS가 VLSA를 다룬 방식) | 자체 과제 결과 표 |
| **실물 RB-Y1** | AG3S만 | — | 실행 가능성 시연(개선 주장 없음) |

**MuJoCo baseline 우선순위** (모두 end effector 하나를 타원체 CBF로 보호 → 양팔 joint 공간 RB-Y1로 옮기는 adapter 필요; 코드 공개 여부 미확인):

| 순위 | baseline | 필요한 것 | 비용 |
|---|---|---|---|
| 1 | KNOWS | YOLOE segmenter(우리 물체에 fine-tune 필요 가능) + attention 순위 + 타원체 CBF | 중간 — 우리와 가장 가까움 |
| 2 | VLSA | VLM(GLM-4.5V) + GroundingDINO + 타원체 CBF | 중간–큼 |
| 3 | WBAG | 시뮬레이터 정답 identity + BP-SDF + whole-body CBF | 큼(정답 정보 사용을 명시) |

마감을 고려해 2개를 먼저 완성하고, 여유가 있으면 3–4개로 늘린다.

**Ablation (MuJoCo, 부품 대 부품).** 논문 전체와 비교하지 않는다. 우리 시스템에서 부품 하나만 바꾸고, 대안으로 다른 논문의 방식을 넣는다.

| 핵심 모듈 | 대안 (나머지 동일) | 답하는 질문 |
|---|---|---|
| **ATS** | ① 외부 segmenter(VLSA·KNOWS식) ② 기하 규칙(손에 가장 가까운 물체) ③ 시뮬레이터 정답 target(참고) | perception model 없이도 target을 충분히 찾는가 |
| **HSR** | ① gate 없음 ② 강한 gripper/grasp 상태 규칙 ③ 정답 phase(참고) | 학습한 read-out이 단순 규칙보다 나은가 |
| **LTE** | ① 면제 없음 ② 모든 링크에서 target 제거(MoManipVLA식) | 링크별 처리가 필요한가 |
| **attached object** | 끔 / 켬 | 잡은 물체를 충돌 계산에 넣어야 하는가 |

ATS와 HSR의 대안 비교가 ablation 표의 가장 중요한 행이다.

## 2. 확인된 벤치마크 사실 (원문 정리 노트 기준)

| 항목 | 내용 | 출처 |
|---|---|---|
| SafeLIBERO 구성 | LIBERO 4 suite × 16 task, Level I(target 근처 장애물) / Level II(경로를 막는 장애물) = 32 scenario × 50 episode = 1,600 episode | VLSA p2·p5 |
| 기본 정책 | pi0.5-LIBERO (VLSA는 OpenVLA-OFT도 baseline) | VLSA p5 |
| 제어 | Franka Panda, OSC_POSE(end-effector 변위), 20 Hz | VLSA p5 |
| VLSA 지표 | CAR(collision avoidance rate), TSR(task success rate), ETS | VLSA p5–6 |
| KNOWS 확장 | 4 suite × 3 level(자체 Level III: 이동 장애물), task당 50 episode; 지표 SR·CR·SSR; VLSA 대신 GT segmentation 타원체를 쓰는 "Naive" 재구현과 비교 | KNOWS p5–6 |
| KNOWS attention | pi0.5 LIBERO에서 layer 12, head 3 (오프라인 profiling) | KNOWS p5 |
| WBAG | SafeLIBERO, frozen pi0.5, **target·goal·support identity를 시뮬레이터에서 받음**, whole-body + attached geometry | WBAG §V-A (arXiv HTML) |

**미확인 (첫날 확인):** SafeLIBERO 환경 코드·asset 공개 여부와 버전 / VLSA·KNOWS·WBAG 코드 공개 여부 /
pi0.5-LIBERO 체크포인트가 세 논문과 동일한지 / ETS 등 지표의 정확한 정의.

## 3. 주의사항

1. **지표가 논문마다 다르다.** VLSA(CAR·TSR·ETS), KNOWS(SR·CR·SSR), WBAG(Scene Safety·Safe Success).
   같은 이름이라도 정의가 같다고 가정하지 않는다. 우리는 **세 정의를 모두 계산할 수 있게** 원자료(접촉 event, 성공 판정)를 남긴다.
2. **보고 수치와 재실행 수치를 섞지 않는다.** 체크포인트·level·episode 수·지표 정의가 같을 때만 같은 열에 둔다. 다르면 "reported" 블록을
   분리하고 †·* 같은 표시로 출처를 밝힌다(ACoT·GeoPredict·OptimusVLA의 관행).
3. **WBAG는 시뮬레이터 정답 identity를 쓴다.** privileged input으로 표에 명시한다(우리는 attention + depth).
4. **행동 공간이 다르다.** LIBERO pi0.5는 end-effector 변위를 낸다. 우리 CCR은 joint 공간 chunk를 정제한다 → adapter 필요(4절).
   adapter는 방법의 변형이므로 논문에 그대로 적는다.
5. **HSR은 label이 없다.** LIBERO 시연에는 phase 주석이 없다. 기본안은 SafeLIBERO에서 **HSR 대신 강한 규칙**을 쓰고 HSR은 MuJoCo에서만
   평가한다. gripper 이벤트로 label을 만들면 "규칙과 같지 않냐"는 질문이 더 강해진다.
6. **attention head를 다시 정한다.** 우리 head(L8·H2)는 RB-Y1 16-D 체크포인트 기준이다. KNOWS의 L12·H3을 출발점으로 하되 **개발 split에서만**
   선택한다(test로 고르지 않는다).
7. **검사 범위를 맞춰 쓴다.** 우리가 SafeLIBERO에서 gripper만 검사하면 VLSA·KNOWS(end effector)와는 같은 범위, WBAG(whole body)보다는 좁은
   범위다. 표에 "protected links" 열을 둔다.
8. **SafeLIBERO ≠ LIBERO-Safety.** 이름이 비슷한 별도 벤치마크다. 인용·표기에서 혼동하지 않는다.
9. **실행 비용.** 1,600 episode × 조건 수. 서버 ≤ 2, 공용 pod 제한을 지킨다. 먼저 suite 하나·level 하나로 pilot.

## 4. 구현 항목

| ID | 항목 | 내용 | 완료 확인 |
|---|---|---|---|
| SL-0 | 환경 확보 | SafeLIBERO 코드·asset, pi0.5-LIBERO 체크포인트, openpi LIBERO client 연결 | VLA 단독 1 episode가 공식 성공 판정으로 끝남 |
| SL-1 | 관측 adapter | agent-view·wrist 카메라의 RGB + **depth**, intrinsics·extrinsics를 AG3S wire 형식으로 | depth back-projection이 장면 기하와 일치(overlay) |
| SL-2 | 로봇 모델 | Franka gripper(palm·finger) query sphere, self-filter용 전신 sphere | mesh–sphere overlay, 로봇 픽셀 제거 확인 |
| SL-3 | attention 추출 | pi0.5-LIBERO의 action→image attention 추출(별도 pass), head 선택(개발 split) | 대상 물체 위 peak 비율을 개발 split에서 보고 |
| SL-4 | ATS·LTE 연결 | 다중 view 융합, target cluster, target-free ESDF (cuRobo Mapper) | target·장애물 분리 overlay |
| SL-5 | CCR adapter | **end-effector pose 공간 정제**: gripper 구 위치는 EE pose로 결정되므로 EE 변위 chunk를 직접 최적화하거나, IK로 joint 공간에 옮겨 정제 후 되돌림. 두 안 중 하나를 smoke test로 선택 | 정제된 chunk가 OSC 제어기로 실행되고 clearance 재검사 통과 |
| SL-6 | phase 처리 | 강한 규칙(gripper 명령·개도·grasp latch) 기본, HSR은 선택 | gate 로직 동일, 출처만 교체 가능 |
| SL-7 | 평가기 | SafeLIBERO 공식 성공 판정 + 접촉 event 원자료 → CAR·TSR·SR·CR·SSR·Safe Success 계산 | 공식 baseline(pi0.5) 수치를 우리 평가기로 재현 |
| SL-8 | baseline | pi0.5 단독 재실행(필수). VLSA·KNOWS·WBAG는 코드 공개 시 재실행, 아니면 보고 수치 인용(조건 일치 시) | 표의 각 셀에 출처 표시 |

## 5. 일정과 판정 기준

| 날짜 | 작업 | 판정 |
|---|---|---|
| 10/8 | SL-0, 코드·체크포인트·지표 정의 확인 | 공개 여부에 따라 baseline 방식(재실행/인용) 결정 |
| 10/9–10/10 | SL-1 ~ SL-5 최소 경로 | **10/10: AG3S로 1 episode 끝까지 돌면 진행**, 아니면 원인과 남은 일수로 재판정 |
| 10/11–10/14 | SL-6 ~ SL-8, suite 1개 · Level I/II pilot | pilot에서 pi0.5 공식 수치 재현 여부 확인 |
| 10/15–10/28 | 전체 SafeLIBERO 실행 (필요시 KNOWS Level III) | 10/21 주장 고정과 함께 실행 범위 확정 |

중단 기준: 10/10까지 1 episode가 끝까지 돌지 않거나, 10/14 pilot에서 pi0.5 공식 수치를 재현하지 못하면 원인을 기록하고
SafeLIBERO를 축소(suite 일부)하거나 MuJoCo held-out 장면을 주 결과로 전환한다.

## 6. 논문에 쓰는 방식

- 주 결과 표: 행 = pi0.5, VLSA, KNOWS, WBAG, AG3S VLA / 열 = 성공·충돌·안전 성공(각 논문 정의 병기), protected links, privileged input.
- 각 셀에 "re-run" 또는 "reported [cite]" 표시. 조건이 다른 보고 수치는 별도 블록.
- adapter(EE 공간 정제), HSR 미사용 또는 대체, attention head 선택 방식을 Experimental setup에 명시.
