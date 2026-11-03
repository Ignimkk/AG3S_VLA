# LIBERO 계열: 객체 분할과 가려짐 평가

## 실험 목적과 구성

**검증 질문:** 외부 세그멘터가 LIBERO, SafeLIBERO, LIBERO-Safety의 여러 시뮬레이션 장면에서 지시된 객체의 **현재 보이는 부분**을 분할할 수 있는가? 다른 물체가 대상을 가릴 때 어떤 예측을 내는가? 이 작업은 외부 segmenter의 분할 기준을 확인하는 단계이며 ATS 전체와의 비교나 충돌 회피 실험은 아니다.

[초기 규약](MULTIBENCH_SEGMENTATION_OCCLUSION_PROTOCOL_20261101_KO.md)의 40개 init 0 장면에, 가려짐이 심한 사례가 적은 것을 확인한 후 [init 1 탐색 확장](MULTIBENCH_INIT1_EXTENSION_20261101_KO.md) 40개를 더했다. 각각 head/wrist 두 camera이므로 **자연 초기 배치 80장면·160개 camera 영상**이다.

| benchmark | 장면 | 구성 |
|---|---:|---|
| LIBERO | 16 | Spatial/Object 각 4 task × init 0·1 |
| SafeLIBERO | 16 | Spatial/Object × Level I·II × 2 task × init 0·1 |
| LIBERO-Safety | 48 | 4 suite × Level 0·1·2 × 2 task × init 0·1 |

각 장면을 simulator의 한 고정 상태에서 RGB와 instance segmentation으로 렌더하고, 같은 상태에서 대상 geom만 보이게 재렌더했다. 다른 geom을 복원한 뒤 RGB와 target mask가 원래대로 돌아오는지, 원래 보이는 target mask가 target-only mask에 포함되는지 모든 camera에서 확인했다. **가려짐 비율 = 1 − visible target pixel / target-only in-frame target pixel**이다. 이 지표는 화면 안에서 다른 물체·로봇·장면에 의해 가려진 비율이며, 카메라 화면 밖으로 잘린 부분이나 물체 뒷면은 포함하지 않는다. Target-only 결과는 GT 진단용이며 모델에 넣지 않는다.

Grounded-SAM-2 per-frame과 YOLOE-11L을 기존 공개 demo의 기본 설정으로 실행했다. 각 모델에는 RGB와 **지시문에서만 추출한 대상 구절** 또는 전체 지시문을 준다. 정답 객체 이름·mask·target-only 영상은 모델 입력이 아니다. 각 모델은 detector의 최고 점수 인스턴스 하나를 선택한다. 모델 실행 오류는 없었다. 최초 고정 정규식은 `get ...`/`grasp ...` 형태의 LIBERO-Safety **4장면**에서 실패해 phrase 조건의 8개 camera 결과가 없다. 이 실패를 정답 이름으로 대체하지 않았으며 전체 지시문 조건은 80장면 모두 평가했다. 각각 [자연 장면 결과](../../../outputs/multibench_segmentation_20261101/summary_natural/summary.json), [실행 코드](../../../experiments/segmentation/eval_multibench_static.py).

## 실험 1 — 다양한 자연 초기 장면의 2D 분할

**검증 내용:** 실제 장면과 카메라가 바뀌었을 때 외부 모델이 지시된 대상의 가시 표면과 얼마나 겹치는가? 아래는 **대상 구절 조건**, target이 보이는 camera 영상의 평균 2D IoU다. target이 보이지만 예측이 비어 있으면 0이다. 화면에 target이 아예 없는 영상은 IoU에서 제외하고 아래에 별도로 센다. 두 camera와 같은 task의 init·level은 서로 독립 표본이 아니다.

| benchmark | target이 보이는 영상 수 | Grounded-SAM-2 IoU | YOLOE IoU |
|---|---:|---:|---:|
| LIBERO | 29 | 0.371 | 0.067 |
| SafeLIBERO | 30 | 0.225 | 0.127 |
| LIBERO-Safety | 87 | 0.380 | 0.226 |

여기서 LIBERO-Safety phrase 조건은 파싱 실패 4장면·8 camera를 제외한 **44장면·88개 camera**에서 계산했다. 전체 지시문 조건은 LIBERO/SafeLIBERO/LIBERO-Safety 각각 29/30/95개의 target-visible 영상에서 SAM2 IoU **0.235/0.159/0.370**, YOLOE IoU **0.002/0.000/0.131**이었다. Phrase와 instruction의 LIBERO-Safety 분모가 다르므로 이 수치만으로 prompt 효과를 단정하지 않는다. 정답 대상이 target-only 렌더에서도 화면에 나타나지 않은 phrase 조건의 영상 6개 중 SAM2는 **4개**, YOLOE는 **1개**에서 비어 있지 않은 예측을 냈다. 이는 target이 이 화면에 보이지 않을 때의 오검출 진단이다.

아래 가려짐 표도 phrase 파싱 성공 영상 기준이다. 추가 8개 파싱 실패 영상은 모두 `<10%` 구간이고, 전체 지시문 조건에는 포함된다.

| 가려짐 구간 | target-visible camera 영상 수 | Grounded-SAM-2 IoU | YOLOE IoU |
|---|---:|---:|---:|
| 0–<10% | 120 | 0.406 | 0.212 |
| 10–<50% | 22 | 0.085 | 0.000 |
| 50–<100% | 4 | 0.000 | 0.000 |

상대적으로 많이 가려진 소수 사례의 점수가 낮았다. 하지만 구간별 물체, 시점, 대상 크기, task가 다르고 특히 50% 이상은 4개뿐이므로 위 표만으로 가려짐의 인과 효과나 전체 benchmark 성능을 일반화할 수 없다. 전체 160개 영상의 가려짐 분포는 `<10%` 128, `10–<50%` 22, `50–<100%` 4, target-only 영상에서도 화면 안 대상 없음 6이다.

[세 benchmark의 실제 시뮬레이션 분할 예시](../../../outputs/multibench_segmentation_20261101/summary_natural/examples_init0.png). 각 benchmark에서 phrase 예측이 있고 대상이 보이는 영상 중 가려짐이 가장 큰 사례를 고른 **설명용 그림**이므로 대표 표본이 아니다. 초록은 GT만, 빨강은 예측만, 노랑은 일치, 파랑 경계는 다른 geom을 숨긴 target-only의 화면 내 윤곽이다.

## 실험 2 — 같은 장면에서 3D 상자로 대상 가리기

**검증 내용:** 장면·대상·지시문을 유지한 상태에서 다른 3D 물체가 시야를 가리면 분할 결과가 어떻게 바뀌는가? [별도 규약](MULTIBENCH_CONTROLLED_OCCLUSION_20261101_KO.md)에 따라 SafeLIBERO 원본 8장면을 재생성해 기존 RGB·GT와 정확히 일치하는지 검사했다. 각 장면의 실제 simulator 상자를 head camera–대상 사이의 고정된 두 위치로 옮겨 **16개 통제 장면**을 만들었다. 상자가 공중에 있거나 다른 물체와 겹칠 수 있는 인위적 스트레스 조건이다. 자연 분포 평균에 합치지 않는다. 원본과 가림 장면의 target-only silhouette도 32/32 camera에서 일치했다.

Head camera의 한 위치에서는 대상이 **59.3–80.9%** 가려졌지만 일부 보였고(8개), 다른 위치에서는 **완전히 가려졌다**(8개). Phrase 조건 결과는 다음과 같다.

| 모델 | 부분 가림 전 → 후 2D IoU | 부분 가림 후 다른 곳이라도 예측한 수 | 완전 가림에서 비어 있지 않은 예측 |
|---|---:|---:|---:|
| Grounded-SAM-2 | 0.365 → 0.000 | 6/8 | 6/8 |
| YOLOE | 0.238 → 0.000 | 2/8 | 2/8 |

부분 가림 전에도 SAM2가 IoU 0보다 큰 원본은 3/8, YOLOE는 2/8이었으므로 8개 모두가 가림 때문에 성공에서 실패로 바뀐 것은 아니다. **완전히 가려진 경우 2D IoU는 정의하지 않으며**, 비어 있지 않은 mask 수만 보고한다. 원본 8장면은 같은 task의 level을 공유하고, 두 가림 위치도 같은 원본을 반복하므로 독립적인 16개 task 실험이 아니다. [짝별 데이터](../../../outputs/multibench_segmentation_20261101/summary_controlled/summary.json)에는 전체 지시문 조건과 wrist 결과도 있다.

![동일한 SafeLIBERO 장면에서 가림 전과 후의 분할](../../../outputs/multibench_segmentation_20261101/summary_controlled/example.png)

그림은 고정한 첫 SafeLIBERO 장면이다. 첫 행은 원본, 둘째는 상자로 59.3% 가린 경우, 셋째는 완전히 가린 경우다. 첫 행에서 SAM2는 정답 bowl을 분할했지만, 둘째와 셋째에서는 다른 bowl을 선택했다. 파란 윤곽은 평가를 위해 상자를 숨겼을 때 대상이 투영되는 위치이며 **모델에 제공하지 않았다**. 이 그림 하나를 전체 통계로 해석하지 않는다.

## 한계와 다음 단계

- 정적 시작 장면만 평가했다. 로봇이 움직이며 가림이 생기거나 사라지는 연속 영상, tracking·identity 유지, 다른 초기 상태의 더 넓은 표본은 아직 검증하지 않았다.
- 실제 RGB-D와 camera pose를 이 정적 수집에서 함께 보존하지 않았으므로, 새 결과에는 3D IoU를 붙이지 않았다. 이전 시간 정합 RGB-D 결과와 혼합하지 않는다.
- 두 모델의 기본 threshold와 top-1 선택을 사용한 탐색 평가다. 다른 prompt·threshold·instance 선택 기준의 최적화나 실제 로봇 성능으로 일반화하지 않는다.
- 다음 ATS 비교에서는 같은 장면·정답과 지시문-only 입력을 사용해 **대상 식별·visible 2D mask·공통 3D 표면·가림 후 identity 유지**를 각각 비교한다. 완전 가림은 2D IoU 대신 잘못된 identity 선택과 복구 여부를 측정한다.

구현과 검증: [동기화 렌더](../../../experiments/segmentation/capture_multibench_static.py), [통제 가림 렌더](../../../experiments/segmentation/capture_controlled_occlusion.py), [모델 평가](../../../experiments/segmentation/eval_multibench_static.py), [자연 장면 집계](../../../experiments/segmentation/summarize_multibench.py), [가림 짝 비교](../../../experiments/segmentation/summarize_controlled.py). 초기 init 0 수집에 사용한 원본 코드는 [보존 사본](../../../experiments/segmentation/capture_multibench_static_v1.py)이다. 분할 지표 기존 테스트는 **6 passed**. 별도 [전수 검증](../../../outputs/multibench_segmentation_20261101/verification.json)에서 자연·통제 96장면의 저장 mask와 GT로 **752개 점수 행**을 재계산하고 나머지 **16개 행**의 phrase parsing 실패를 확인했다. 새 코드·결과와 각 capture/provider manifest의 해시는 [provenance](../../../outputs/multibench_segmentation_20261101/provenance.json)에 기록한다.
