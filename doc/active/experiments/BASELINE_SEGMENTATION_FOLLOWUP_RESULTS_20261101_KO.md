# 외부 baseline 추가 측정: SAM2 위치 단서와 가림 추적

## 범위

사용자 지시에 따라 **ATS는 평가하지 않았다**. 앞선 [다중 benchmark 분할 실험](MULTIBENCH_SEGMENTATION_OCCLUSION_RESULTS_20261101_KO.md)의 같은 80개 자연 장면과 16개 통제 가림 장면에서 외부 baseline만 추가 측정했다. 대상은 현재 camera에서 **보이는 2D target 표면**이다. 추가 실험들은 이전 결과를 이미 본 뒤 정한 탐색적 분석이다.

## 실험 1 — Grounded-SAM v1 추가 측정

**무엇을 검증했는가:** 같은 시뮬레이션 이미지와 지시문에서 Grounding DINO + SAM v1의 분할 결과를 기존 Grounded-SAM2 및 YOLOE와 함께 관찰한다. [추가 측정 규약](BASELINE_GROUNDED_SAM_EXTENSION_20261101_KO.md)의 고정된 공개 adapter 기본 설정을 사용했다. 세 모델 모두 모델 입력은 RGB와 지시문에서 추출한 phrase이며 simulator 정답 이름·정답 mask는 제공하지 않았다. 모델 추론 오류는 없었다.

| 벤치마크 | 대상이 보이는 phrase 평가 영상 | Grounded-SAM v1 | Grounded-SAM2 | YOLOE |
|---|---:|---:|---:|---:|
| LIBERO | 29 | 0.472 | 0.371 | 0.067 |
| SafeLIBERO | 30 | 0.307 | 0.225 | 0.127 |
| LIBERO-Safety | 87 | 0.497 | 0.380 | 0.226 |
| 전체 | 146 | 0.453 | 0.347 | 0.174 |

숫자는 **2D IoU의 영상 평균**이며 camera·같은 task의 level/초기 상태가 독립 표본은 아니다. 고정 parser가 지시문을 해석하지 못한 4장면·8 camera는 세 모델의 phrase 조건 모두에서 제외됐고, 화면에서 대상이 보이지 않는 6영상도 IoU 평균에서 제외했다. 지시문 전체를 주는 조건은 80장면 모두 실행했다. 이 데이터에서 v1이 높았지만, Grounded-SAM v1의 detector box/text threshold는 **0.3/0.25**, SAM2 조합은 **0.4/0.3**이므로 차이를 SAM 버전만의 효과로 해석할 수 없다. 전체 설정과 가려짐별 결과는 [세 baseline 결과](../../../outputs/multibench_segmentation_20261101/summary_natural_three_baselines/summary.json)에 있다.

| 실제 화면 내 가려짐 | target이 보이는 영상 | Grounded-SAM v1 IoU | Grounded-SAM2 IoU | YOLOE IoU |
|---|---:|---:|---:|---:|
| 0–<10% | 120 | 0.498 | 0.406 | 0.212 |
| 10–<50% | 22 | 0.292 | 0.085 | 0.000 |
| 50–<100% | 4 | 0.002 | 0.000 | 0.000 |

가려짐 구간별 scene과 대상이 다르고 강한 자연 가림은 4영상뿐이다. 그 차이를 가려짐만의 인과 효과나 모집단 성능으로 단정하지 않는다. SafeLIBERO에서 3D 상자를 의도적으로 옮긴 별도 통제 장면에서는, head camera의 부분 가림 8개에 대해 Grounded-SAM v1의 IoU가 원본 평균 **0.365 → 0.000**이었다. 완전 가림 8개에는 IoU 대신 비어 있지 않은 예측 **6/8**을 기록했다([짝별 결과](../../../outputs/multibench_segmentation_20261101/summary_controlled_grounded_sam/summary.json)). 원본 상태에서 이미 잘못된 선택이 있던 frame도 포함한다.

## 실험 2 — SAM2에 정답 위치를 알려 준 분할 진단

**무엇을 검증했는가:** 낮은 언어 기반 결과가 대상 탐지와 마스크 생성 중 어디에서 나타나는지 좁힌다. 같은 SAM2.1 Hiera-Large checkpoint를 사용해, 지시문 기반 Grounding DINO의 box 대신 **정답에서 만든 box 또는 정답 내부 point**를 SAM2에 넣었다. [실험 규약](BASELINE_SAM2_PROMPT_DIAGNOSTIC_PROTOCOL_20261101_KO.md). 정답에서 만든 box/point는 **특권 입력**이며 Grounded-SAM2와 동등한 실사용 성능 비교가 아니다. 정답 mask 전체는 모델에 주지 않았다.

| 같은 target-visible 영상 | 영상 수 | 자동 Grounded-SAM2 IoU | SAM2 + GT box IoU | SAM2 + GT point IoU |
|---|---:|---:|---:|---:|
| 자연 장면 전체 | 146 | 0.347 | 0.973 | 0.719 |
| 자연 장면, 50% 이상 부분 가림 | 4 | 0.000 | 0.959 | 0.936 |
| 통제 장면, 50% 이상 부분 가림 | 10 | 0.000 | 0.925 | 0.882 |

자연 장면 146개 중 자동 조건은 **44개에서 빈 mask**, **50개에서 비어 있지 않지만 target과 겹치지 않는 mask**를 냈다. GT box 조건은 146/146에서 IoU ≥ 0.5였다. 여기서는 정확한 위치를 알려 주면 SAM2가 보이는 표면을 잘 분리한 경우가 많았으나, 작은 탐색 집합의 GT 기반 box는 실제 detector의 불확실성을 포함하지 않는다. Target이 전혀 보이지 않는 16 camera 영상에는 GT box/point가 정의되지 않아 점수를 만들지 않았다. 전체 지표는 [일치 영상별 비교](../../../outputs/multibench_segmentation_20261101/sam2_prompt_comparison/summary.json), [모든 정답 프롬프트 결과](../../../outputs/multibench_segmentation_20261101/sam2_oracle_prompts/summary.json)에 있다.

![같은 영상에서 언어 기반 탐지와 정답 위치 단서 비교](../../../outputs/multibench_segmentation_20261101/sam2_prompt_comparison/example.png)

그림의 둘째 행은 target이 59.3% 가려진 같은 장면이다. 자동 Grounded-SAM2는 다른 bowl을 선택했지만, GT box 또는 point를 받은 SAM2는 보이는 target을 분할했다. 청록 box·자홍 point는 simulator GT에서 만든 **특권 단서**다. 초록은 GT만, 빨강은 예측만, 노랑은 겹침이다. 고정된 한 예시이며 전체 성능을 대표하지 않는다.

## 실험 3 — 부분 가림·완전 가림·재등장에서 외부 추적 baseline

**무엇을 검증했는가:** 초기 분할 후 가림이 생겼을 때 per-frame 재탐지와 causal SAM2 tracking의 출력 차이를 확인한다. [규약](BASELINE_OCCLUSION_TRACKING_PROTOCOL_20261101_KO.md)에 따라 SafeLIBERO 원본 8장면마다 **원본 → 부분 가림 → 완전 가림 → 원본 재등장**의 4개 관측을 입력했다. 이는 정적 simulator 이미지를 이어 붙인 **불연속 스트레스 시퀀스**이고 실제 연속 영상이 아니다. 재등장은 최초 RGB와 bit 동일하다.

| 외부 baseline | 초기 2D IoU | 부분 가림 IoU | 완전 가림 중 비어 있지 않은 mask | 재등장 IoU |
|---|---:|---:|---:|---:|
| Grounded-SAM2 per-frame | 0.365 | 0.000 | 6/8 | 0.365 |
| Grounded-SAM2 causal track | 0.365 | 0.344 | 3/8 | 0.364 |
| SAM2 track + 빈 mask 후 재검출 | 0.365 | 0.344 | 3/8 | 0.365 |
| YOLOE per-frame | 0.238 | 0.000 | 2/8 | 0.238 |

IoU는 대상이 보이는 8개 frame의 평균이다. 완전 가림에서 대상 mask는 비므로 IoU는 계산하지 않고 nonempty mask만 센다. 최초 IoU ≥ 0.5인 SAM2 시퀀스는 **3/8**이었으며, causal track은 이 3개 모두 부분 가림과 재등장에서 IoU ≥ 0.5였다. 나머지 5개는 처음부터 대상을 정확히 찾지 못했다. 따라서 이 결과는 **올바른 초기화가 있는 소수 장면에서의 추적 진단**이지, 추적기가 전반적으로 대상 구별을 해결한다는 뜻이 아니다. Per-frame 재등장 결과는 최초 이미지와 같은 이미지를 사용한 기존 예측을 재사용했으므로 일반적인 재탐지 회복 능력의 독립 증거가 아니다. [전체 프레임 및 초기화별 결과](../../../outputs/multibench_segmentation_20261101/baseline_occlusion_tracking/summary.json).

![동일한 네 시뮬레이터 관측에서 per-frame과 tracking 결과 비교](../../../outputs/multibench_segmentation_20261101/baseline_occlusion_tracking/example.png)

그림은 초기화가 맞은 고정 예시 하나다. 초록은 GT만, 빨강은 예측만, 노랑은 일치다. 부분 가림에서 per-frame은 다른 bowl을 골랐고 causal track은 보이는 대상 부분을 유지했다. 이 한 장면으로 일반적 tracking 성능을 주장하지 않는다.

## 검증 및 다음 baseline 측정

Grounded-SAM v1, oracle-prompt SAM2, 추적의 예측 실행 오류는 없었다. [전수 검증](../../../outputs/multibench_segmentation_20261101/baseline_extension_verification.json)에서 oracle prompt·점수 **352개**, 적용 불가 **32개**, Grounded-SAM 점수 **376개**와 phrase parser 실패 **8개**, tracking **128개 행**을 원본 GT 및 저장 mask로 다시 확인했다. 초기 SAM2 tracking 결과와 per-frame 결과는 8/8에서 점수가 정확히 같았다. 관련 기존 2D 지표 테스트는 **6 passed**다. [실행·입력·결과 해시](../../../outputs/multibench_segmentation_20261101/baseline_extension_provenance.json).

다음 baseline 측정은 실제 연속 rollout에 대해 **동일 기준의 detector box 품질, 잘못된 instance 선택, 마스크 경계, 가림 지속 시간 및 재등장 복구**를 함께 기록하는 것이 타당하다. 동일 인식 모델이라도 threshold와 초기화 정책에 따라 결과가 달라지므로 개발 설정과 독립 검증 설정을 분리해야 한다. ATS와의 비교는 이 단계에서 비워 둔다.
