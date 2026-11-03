# 외부 baseline의 연속 rollout 분할 및 가림 분석

## 목적과 공통 평가

정적 초기 영상과 불연속 4-frame 스트레스 시험에 이어, **로봇이 움직이는 연속 시뮬레이션 관측에서 외부 segmenter가 목표 객체를 찾고, 보이는 부분을 분할하며, 가림 이후 어떤 출력을 내는지** 측정한다. ATS는 입력·평가·비교에 포함하지 않는다. 네 arm은 Grounded-SAM v1 프레임별, Grounded-SAM2 프레임별, Grounded-SAM2 causal track, YOLOE 프레임별이다. 언어 지시문만으로 고정 정규식이 추출한 phrase와 RGB를 제공하고, GT 위치·mask·객체명은 모델에 제공하지 않는다.

시뮬레이터의 목표 객체 instance를 GT로 하고 224×224 영상에서 visible target 2D IoU를 계산한다. 대상이 보일 때 빈 mask는 0점이다. 대상이 완전히 가려지거나 화면 밖에 있으면 IoU를 계산하지 않으며 nonempty 출력을 별도로 센다. 연속 frame은 같은 물체·경로의 반복 관찰이므로 **독립 표본이 아니다**. 여기에 제시하는 수치는 소수의 선택된 탐색 실행에 관한 기술 통계다.

| 구분 | 연속 시퀀스 | 프레임 | 동작 출처 |
|---|---:|---:|---|
| LIBERO | 6 | 707 | 이전에 저장된 task 정책 rollout, Spatial/Object 각 task 0–2, init 0 |
| SafeLIBERO | 3 | 771 | 이전에 저장된 VLA 단독 정책 rollout, Spatial Level I task 0–2, init 0 |
| LIBERO-Safety | 3 | 286 | 다른 LIBERO task의 저장된 action을 옮겨 실행한 스트레스 시험 |

마지막 줄은 해당 Safety task를 수행하도록 만들어진 정책의 rollout이 아니며 과제 성공률을 뜻하지 않는다. 각 데이터셋의 방법과 고정 표본은 [LIBERO 규약](BASELINE_CONTINUOUS_ROLLOUT_PROTOCOL_20261102_KO.md), [SafeLIBERO 규약](BASELINE_SAFELIBERO_ROLLOUT_PROTOCOL_20261102_KO.md), [Safety action 전이 규약](BASELINE_LIBERO_SAFETY_ACTION_TRANSFER_PROTOCOL_20261102_KO.md), [SAM v1 사후 확장](BASELINE_GROUNDED_SAM_CONTINUOUS_EXTENSION_20261102_KO.md)에 기록했다. 기존 결과를 본 뒤 진행한 탐색적 분석이며, v1 추가도 기존 연속 결과를 확인한 뒤 정했다.

SafeLIBERO는 저장된 이미지 시점 simulator state로 프레임마다 원본을 재렌더하고, head RGB **771/771 픽셀 단위 일치**를 확인했다. 같은 상태에서 다른 geom을 일시적으로 숨겨 목표만 다시 렌더해 화면 안 가림을 계산했다. 따라서 화면 밖으로 잘린 대상은 가림에 포함하지 않는다. Safety도 실제 `env.step(action)` 전후로 순차 수집한 동일 상태 GT를 사용했으나 action이 다른 task에서 왔다. LIBERO 원본은 동일 시점 저장 instance mask만 있고 target-only 렌더가 없어 가림률을 계산하지 않는다. RGB 리사이즈와 GT 최근접 변환의 차이로 경계에 약 1–2 pixel 불확실성이 생길 수 있다.

## LIBERO: 대상 식별과 추적

**검증 질문:** 조작 중인 대상의 실제 identity를 언어 기반 모델이 선택하고 유지하는가? 707개 모든 프레임에서 대상의 일부가 보였다.

| 모델 | visible frame 평균 IoU | episode별 IoU 평균 후 평균 | 빈 mask | 다른 객체가 예측 영역의 ≥50% |
|---|---:|---:|---:|---:|
| Grounded-SAM v1 | 0.108 | 0.148 | 184 | 427 |
| Grounded-SAM2 per-frame | 0.044 | 0.075 | 320 | 352 |
| Grounded-SAM2 track | 0.081 | 0.141 | 217 | 417 |
| YOLOE | 0.001 | 0.001 | 706 | 0 |

다른 객체 판정은 해당 mask 픽셀의 절반 이상이 GT instance map의 동일한 다른 scene 객체에 속하는 경우다. 이 기준은 출력 픽셀 점유에 관한 것이며 작은 mask가 다른 물체에 살짝 걸친 모든 경우를 세지 않는다. 장면에 유사한 다른 black bowl이 있으며, 프레임별 SAM2의 다른 객체 우세 352프레임 중 205프레임이 `akita_black_bowl_2`였다. 연속 frame이므로 205개의 독립된 오류가 아니다. [객체별 진단 결과](../../../outputs/continuous_rollout_20261102/libero_instance_diagnostics/summary.json).

Grounded-SAM2 per-frame이 선택한 detector box의 GT 가시 target box IoU가 **0.5 이상인 프레임은 31/707**이었다. 이 31프레임의 마스크 IoU는 평균 **0.937**이고 2-pixel 허용 경계 F1은 **0.990**이었다. 이는 *정답을 보고 잘 맞은 detector box만 사후 선택한 조건부 진단*이다. 전체 점수 0.044와 같은 성능 수치로 비교할 수 없지만, 이 표본에서 목표 위치 선택이 주요 병목이라는 해석을 뒷받침한다. 경계 F1은 빈 예측을 0점으로 포함하고, 모델별 detector 설정은 고정 기본값을 사용했다.

SAM2 track은 첫 프레임 IoU ≥0.5인 episode가 **1/6**이었다. 아래 그림은 그 1개 episode의 실제 연속 실행 중 고정 프레임 0, 22, 45, 67이다. 로봇이 bowl을 가리기 시작한 뒤 per-frame은 빈 마스크를 냈지만, 정확한 초기화에서 출발한 track은 보이는 목표 부분을 계속 분할했다. 다른 5개 episode의 초기 실패를 포함한 전체 수치는 위 표와 같다. 단일 사례는 전체 성능의 대표 근거가 아니다.

![연속 LIBERO 정책 실행에서 프레임별 분할과 추적 비교](../../../outputs/continuous_rollout_20261102/libero_summary/sequence_example.png)

색은 초록 GT만, 빨강 예측만, 노랑 일치다. [초기부터 다른 인스턴스를 선택한 예시](../../../outputs/continuous_rollout_20261102/libero_summary/identity_failure_example.png)도 별도로 보존했다.

## SafeLIBERO: 연속 가림, 탐지와 재등장

**검증 질문:** 실제 정책 행동 중 대상이 점차 가려지고 다시 보일 때, 외부 baseline의 visible mask와 추적 결과가 어떻게 달라지는가? 전체 771프레임 중 대상이 일부라도 보이는 프레임은 755개다.

| 화면 안 가림 정도 | visible frame | SAM v1 IoU | SAM2 per-frame IoU | SAM2 track IoU | YOLOE IoU |
|---|---:|---:|---:|---:|---:|
| 0–<10% | 87 | 0.491 | 0.504 | 0.000 | 0.000 |
| 10–<50% | 237 | 0.006 | 0.003 | 0.015 | 0.000 |
| 50–<100% | 431 | 0.002 | 0.001 | 0.282 | 0.000 |
| 전체 visible | 755 | 0.060 | 0.060 | 0.166 | 0.000 |
| episode별 평균 후 평균 | 3 episodes | 0.051 | 0.051 | 0.142 | 0.000 |

가림 구간별 프레임은 같은 세 실행의 서로 다른 시점이다. 장면과 시간, 로봇·장애물 배치도 함께 바뀌므로 표를 가림만의 인과 효과로 해석하지 않는다. 추적의 0–<10% 점수가 0인 이유를 포함해 초기화가 중요하다. **SAM2 tracking은 세 episode 모두 최초 프레임에서 IoU가 0**이었다. 이후 어떤 프레임에서 target과 겹쳤다고 해서 처음부터 올바른 instance identity가 유지되었다고 단정할 수 없다.

완전히 가려진 경우는 총 **16프레임, 3구간(7·3·6프레임)**이었다. 저장된 simulator 관측 간격은 대략 0.05초다. IoU가 정의되지 않는 이 16프레임에서 비어 있지 않은 mask 수는 SAM v1 **16**, SAM2 per-frame **13**, SAM2 track **16**, YOLOE **6**이었다. 이 mask가 틀린 객체나 이전 위치에 해당할 가능성이 있지만, 하나의 숫자만으로 잔상과 다른 객체 선택을 구분하지 않는다.

3개 완전 가림 구간 바로 직전의 SAM2 per-frame IoU는 각각 약 0.0005, 0, 0이었다. 따라서 이후 즉각적인 실패를 모두 가림에 의한 상실로 볼 수 없다. 목표가 다시 한 픽셀 이상 보인 후 첫 프레임의 SAM2 per-frame IoU도 세 구간 모두 0이었다. SAM2 track은 첫 구간에서 재등장 후 83프레임이 지나 IoU ≥0.5에 도달했으나 그 구간 직전과 episode 시작에서 이미 잘못된 대상을 추적했기 때문에 **정체성 복구 성공으로 세지 않는다**. 남은 두 구간은 다음 비가시 구간 또는 기록 종료 전 IoU ≥0.5에 도달하지 않았다. 이벤트별 수치는 [전체 결과](../../../outputs/continuous_rollout_20261102/safelibero_summary/summary.json)에 있다.

![SafeLIBERO 실제 실행에서 가림 전후의 객체 분할](../../../outputs/continuous_rollout_20261102/safe_paired_interim/full_occlusion_example.png)

같은 연속 episode에서 왼쪽부터 50, 66, 68, 74번째 관측이다. 68번째에는 target이 완전히 가려져 IoU가 정의되지 않는다. 프레임별 모델은 이 예시에서 가림 물체를 분할했고, 추적 모델은 처음부터 다른 bowl을 따라갔다. [가림률과 시간별 IoU 그래프](../../../outputs/continuous_rollout_20261102/safe_paired_interim/occlusion_timeline.png).

## LIBERO-Safety: action 전이 스트레스 장면

**검증 질문:** 칼과 그릇 등 다른 시뮬레이터 장면의 연속 관측에서도 외부 모델이 목표를 선택하고 분할하는가? 앞서 설명했듯 행동은 다른 LIBERO task의 정책 action을 그대로 옮긴 것이며 Safety 전용 정책이 아니다. 세 시퀀스에서 target은 286/286프레임에서 일부 보였고, 가림이 10% 이상인 프레임은 3개뿐이었다. 완전 가림은 없었다. 따라서 이 데이터로 가림 복구 성능을 주장하지 않는다.

| 모델 | visible frame 평균 IoU | episode별 평균 후 평균 |
|---|---:|---:|
| Grounded-SAM v1 | 0.203 | 0.278 |
| Grounded-SAM2 per-frame | 0.197 | 0.274 |
| Grounded-SAM2 track | 0.194 | 0.272 |
| YOLOE | 0.324 | 0.346 |

세 실행의 길이와 대상이 달라 두 평균의 의미가 다르다. 예를 들어 SAM2 per-frame의 episode 평균 IoU는 칼 대상 `affordance`에서 0.815, 두 그릇 대상 장면에서 0.008과 0.000이다. 이 세 장면과 전이된 행동만으로 모델의 Safety benchmark 순위를 주장하지 않는다. 세 모델과 네 번째 v1의 설정·프레임별 지표는 [집계](../../../outputs/continuous_rollout_20261102/libero_safety_summary/summary.json)에 있다.

[Safety 칼 장면의 연속 분할 예시](../../../outputs/continuous_rollout_20261102/safety_three_interim/sequence_example.png). 칼은 이 사례에서 로봇이 움직이는 동안 대부분 제자리에 있고, 사진은 특정한 한 실행을 설명하기 위해 고른 것이다.

## 해석, 검증, 다음에 필요한 baseline 실험

- Grounded-SAM v1의 Grounding DINO box/text threshold는 0.3/0.25, SAM2 조합은 0.4/0.3이다. 다른 detector 설정 때문에 각 arm의 차이를 SAM mask decoder 버전의 효과로 분리할 수 없다. YOLOE도 긴 phrase와 고정 confidence 설정에 영향을 받을 수 있다. 동일 검출 box를 공유하는 조건 및 개발·검증 데이터 분리 후의 threshold/phrase 민감도 실험이 필요하다.
- 이 실험에서 분명히 구분해야 할 것은 **목표 instance 선택, 선택된 대상의 마스크 경계, 가림 중 부재 판단, 재등장 시 identity 회복**이다. 특히 추적이 초기에 틀린 인스턴스를 선택하면 나중의 IoU 상승만으로 올바른 정체성을 유지·회복했다고 할 수 없다. 별도의 instance별 추적 이력과 초기화 성공/실패별 성능이 필요하다.
- 실제 Safety 전용 정책이 생성한 연속 rollout, 여러 task·초기화·카메라, 가림 길이와 가림 물체를 같은 scene에서 달리한 짝 실험을 보강해야 한다. 현재 SafeLIBERO 전체는 세 Spatial Level I episode이고, 강한 가림 프레임이 긴 소수 시퀀스에 집중되어 있다. 손목 카메라는 이번 정책 rollout에서 동기화된 정답으로 평가하지 않았다.
- 경계 F1은 224×224 영상에서 거리 2 pixel 허용이다. 특히 칼처럼 좁은 물체에서는 허용 폭이 상대적으로 커서 점수가 높아질 수 있다. 같은 원본 해상도에서 경계 F1, IoU, 크기별 통계를 함께 확인하는 편이 타당하다.

검증: 저장된 mask와 GT로 LIBERO **707×4**, SafeLIBERO **771×4**, Safety **286×4**의 TP/FP/FN과 IoU를 전수 재계산했고, target-only가 있는 경우 visible 포함관계와 가림률도 전수 확인했다. box IoU와 경계 F1은 매 25번째 frame을 독립 계산으로 표본 확인했다. [LIBERO 검증](../../../outputs/continuous_rollout_20261102/libero_verification_full.json), [SafeLIBERO 검증](../../../outputs/continuous_rollout_20261102/safelibero_verification.json), [Safety 검증](../../../outputs/continuous_rollout_20261102/libero_safety_verification.json). 새 경계/box 단위 테스트와 기존 mask/causal tracking 테스트는 총 **13 passed**다. 모델 실행 오류는 없다.

초기 v1 평가 스크립트의 manifest는 LIBERO 실행에도 SafeLIBERO 규약 경로와 target-only 설명을 잘못 적었다. 이는 실제 이미지·모델 입력이나 점수 계산에는 영향을 주지 않았고, 입력 파일 해시는 정확히 기록되어 있다. 실행 당시 [스크립트 보존 사본](../../../experiments/segmentation/eval_grounded_sam_video_v1.py)을 남겼고 이후 실행 코드의 dataset별 metadata를 수정했다. 각 조건에 적용된 실제 규약은 이 문서 첫 부분의 링크와 같다. [실험 코드 및 산출물 해시](../../../outputs/continuous_rollout_20261102/provenance.json).
