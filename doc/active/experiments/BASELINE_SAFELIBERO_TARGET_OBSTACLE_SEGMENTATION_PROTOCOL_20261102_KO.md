# SafeLIBERO에서 외부 segmenter baseline의 target·obstacle 분할 측정 규약

작성 2026-11-02 KST, 결과를 보기 전. 사용자 지시(2026-11-02): ① ATS 비교를 SafeLIBERO로 확장한다 ② 쥠 판정은 HSR probe가 아니라 그리퍼 물리 신호로 한다 ③ **외부 segmenter의 target·obstacle 분할 성능을 충분히 확인한 뒤에** 거리장 voxel을 지우는 능력을 잰다. 이 문서는 ③의 앞부분(분할 성능)과 그 비교 대상 ATS 측정을 다룬다. voxel 지우기는 이 문서의 범위 밖이며 분할 결과를 확인한 뒤 별도 규약으로 정한다.

## 표본과 GT

| 항목 | 고정값 |
|---|---|
| 기록 | 기존 SafeLIBERO spatial Level I dev, VLA 단독 rollout `20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only` (π0.5, task 0–3 × init 0–4, seed 7, 20 episode). 새 rollout을 하지 않고 성공/실패와 무관하게 전부 쓴다. dev init 0–4만 쓰며 test(init 5 이상)는 건드리지 않는다 |
| 평가 frame | 기록된 policy call 전부(obs 20부터 5 step 간격, 공식 성공으로 일찍 끝난 episode는 짧다), **20 episode 합계 796 frame × 2 camera(head, wrist)**. 재렌더 GT는 모든 기록 관측(episode당 최대 301)에서 만들었고 두 camera RGB 일치 검사를 통과했다 |
| GT | 저장된 `sim_state_image`로 매 관측을 재렌더한 geom id 분할. 두 camera의 재렌더 RGB가 저장 관측과 **bit 단위로 같아야** 하며, 하나라도 다르면 실행이 중단된다. depth·K·T는 같은 기록의 값이다 |
| class | target(과제의 black bowl), destination(plate), **obstacle**(SafeLIBERO가 놓은 active 장애물: 이름에 obstacle이 있고 대기 후 책상 위에 있는 것), other_object(다른 bowl·ramekin·cookies 등), fixture, support(table), robot, mount |
| 한계 | 20 episode = 4 task × 5 init, task 안에서는 장면이 거의 같다. frame은 독립 표본이 아니다. VLA 단독 rollout이라 정책이 충돌하는 장면이 많다(이 기록의 공식 성공 11/20 — 선택 기준이 아니라 기록된 사실. 충돌 판정은 이전 기록의 값을 그대로 쓰며 이 문서에서 다시 계산하지 않는다) |

## 비교 방법 (모델 입력은 RGB와 지시문에서 뽑은 문구뿐)

| 방법 | 설정 |
|---|---|
| Grounded-SAM v1 | 기본값 (box 0.3, text 0.25), frame별 |
| Grounded-SAM-2 frame | 기본값 (box 0.4, text 0.3), frame별 |
| Grounded-SAM-2 track | 첫 검출로 시작, 이후 causal 추적. **모든 기록 frame(0부터 마지막 call까지)을 시간순으로** 입력, 저장은 call frame만 |
| YOLOE | 기본 confidence 0.25. 민감도로 0.05를 같이 돌리며 두 값 중 좋은 쪽을 고른 것으로 취급하지 않는다 |
| 문구 | `phrase`(예: "black bowl between the plate and the ramekin")와 `instruction`(문장 전체) 두 모드. 정규식 실패는 오류로 남기고 정답 이름으로 대체하지 않는다. 관계 표현이 들어 있으므로 Spatial의 "같은 bowl 둘" 구분이 실제로 시험된다 |
| 제외 | ATS는 이 문서의 외부 baseline 표에 섞지 않는다. TokenCut·localization heads(VLA attention 입력)는 이번 표본에서 실행하지 않았다 |

## 지표

1. **target 분할(2D, camera별)**: target이 보이는 frame의 IoU · recall · precision. target이 보이는데 빈 mask이면 0점이다. 보이지 않는 frame의 비어 있지 않은 출력은 따로 센다.
2. **obstacle 처리(핵심)**: 예측 mask가 obstacle 픽셀을 얼마나 포함하는가(obstacle 포함 픽셀, 예측 중 비율)와 obstacle 3D voxel이 target 집합에 들어간 수(`obstacle_fp`). 이 값이 크면 이후 거리장에서 obstacle을 지우는 방향의 위험이다. other_object·destination 포함도 같은 방식으로 센다.
3. **3D(두 camera depth의 5 mm voxel 합집합)**: soft IoU · recall · precision, 이웃(obstacle + other_object + destination) FP.
4. **대상 선택**: 3D precision ≥ 0.5이면 대상 물체를 고른 것으로 센다(이전 Object 실험과 같은 규칙). obstacle이 target 근처에 있는 frame(obstacle–target 최소 거리 구간별)은 층화해서 본다.
5. 집계: frame → episode → task → 4 task 동일 가중 평균. paired bootstrap은 4 task라 구간을 만들지 않고 episode 20개 단위 분포만 기술한다. 오류·parser 실패는 분모에서 빼지 않고 개수를 보고한다.

## 해석의 한계 (결과 전 기록)

- 4 task × 5 init, VLA 단독 한 정책의 궤적이다. 일반화·순위 주장은 하지 않는다. 임계값은 기본값이며 최적화하지 않는다.
- 외부 segmenter는 target 하나를 찾는 모델이다. obstacle을 따로 분할하는 능력은 측정 대상이 아니고, **target mask가 obstacle을 얼마나 침범하는지**로 본다.
- Grounded-SAM v1과 SAM2 계열은 detector 임계값이 달라 mask decoder 버전의 효과를 분리하지 못한다.
- 이 단계는 인식 성능만 잰다. 거리장에서 voxel을 지우는 능력, 접촉, 성공은 측정하지 않는다.

## 같은 표본에서 실제 ATS (비교 대상, 이 문서의 baseline 표와 섞지 않음)

| 항목 | 고정값 |
|---|---|
| ATS | Panda 최종 구성 서버 flag(`AG3S-final-panda`), attention `noun_target:L8H6`(Panda에서 Spatial 개발 404 요청으로 고른 head, 이번 표본으로 바꾸지 않음). 같은 policy 요청에서 target noun row를 읽는다. 이 run들에는 policy_records가 없으므로 요청은 저장된 관측·policy_state·지시문으로 client 전처리 그대로 재구성한다 |
| 단계·쥠 신호 | **HSR probe를 쓰지 않는다.** 그리퍼 물리 신호만 쓴다. call k마다 직전 chunk의 마지막 실행 gripper 명령이 "닫힘"이면 `closed_cmd`, 측정 개도 변화 < 0.02이면 `settled`, `closed_cmd ∧ settled ∧ 0.0253 < 개도 < 0.9`이면 `holding`(Panda 빈 손 보정 0.0253, 성공 기준으로 조정하지 않음). `closed_cmd`가 켜진 동안 `set_grasp_active`·`set_grasp_latched`를 켜서 identity 고정을 작동시키고, 명령이 다시 열림이면 푼다. 물체·접촉 GT는 쓰지 않는다 |
| 대조 조건 | `none`: 쥠 신호 없음(Object 재생과 같은 조건). `physical` 대 `none` 차이가 쥠 신호의 효과다 |
| 검증 | `holding`이 GT의 "target이 첫 frame 위치에서 20 mm 이상 움직임"과 얼마나 일치하는지 보고한다(규칙 조정 근거로 쓰지 않음) |
| 출력 두 가지 | target cluster(`grounding.target.points`)와 서버가 거리장에서 실제로 지우는 voxel(`field.exclusion_trace['removed_ijk']`, erased extent). 둘 다 같은 class-aware 3D 지표로 채점하며 `obstacle_fp`(지운 범위에 든 obstacle voxel)를 함께 본다. 이것은 분할 성능 단계의 채점이며, 거리장 자체를 지우는 효과(접근 경로의 target-free 질의)는 다음 단계다 |
| 한계 | 쥠 전 identity 잠금(LATCHED)은 물리 신호만으로는 알 수 없어 닫힘 명령 이후에만 건다. 서버의 증거 기반 latch(물체 reach·manipulated 확인)를 그대로 재현한 것이 아니다. 쥔 물체 attach(held body)는 재생하지 않는다 |
