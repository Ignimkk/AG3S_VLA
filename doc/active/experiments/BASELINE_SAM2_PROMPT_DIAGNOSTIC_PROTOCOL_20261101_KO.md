# 외부 SAM2의 정답 프롬프트 분할 진단

## 질문과 목적

ATS 비교를 보류한 상태에서, 이전 다중 benchmark의 낮은 Grounded-SAM2 성능을 **언어 기반 탐지 + SAM2 분할 전체**와 **정답 위치를 준 SAM2 분할**로 분해한다. 이를 통해 가려진 대상의 정확한 위치를 알려 줘도 모델이 관측된 부분을 분할할 수 있는지 확인한다.

## 실험 설정

- 기존 동기화된 자연 장면 80개(160 camera)와 통제 가림 16개(32 camera)를 그대로 쓴다. 장면·모델 예측 및 가려짐 분포를 이미 보았고 그 뒤 추가한 탐색 진단이다.
- SAM2.1 이미지 모델에 이미지와 ① 보이는 target GT mask의 최소 외접 **box**, ② 보이는 target 안에서 mask 경계까지 거리가 가장 먼 픽셀 하나를 양성 **point**로 준다. point 거리 동률은 NumPy의 첫 row-major 픽셀을 쓴다. 두 입력은 simulator 정답에서 만들었으므로 **특권 oracle-prompt 진단**이며 지시문 기반 end-to-end 성능이나 실제 배포 성능이 아니다. Mask 전체를 SAM2에 넘기지 않는다.
- 원래 RGB와 대상 visible GT는 이전 평가와 같다. 두 조건 모두 **대상이 이 camera에 보이는 경우에만** 유효하다. target이 완전히 가려졌거나 target-only에서도 화면 내에 없다면 GT box·point를 정의할 수 없으므로 결과는 `not_applicable`로 표기한다. 해당 frame을 IoU 0으로 채우지 않는다.
- point는 기본 `multimask_output=True`, 예측 confidence가 가장 높은 mask를 고르고 box는 `multimask_output=False`인 기존 SAM2 backend를 쓴다. 각 camera 독립, 동일 이미지·동일 가려짐 구간, 2D IoU·recall과 인원수(영상 수)를 보고한다. 자동 text grounding과 비교할 때 정답 입력에 의한 이점을 반드시 밝힌다.
- 통제 가림은 자연 benchmark와 합치지 않는다. 정답 prompt 성능의 차이를 순수 원인으로 단정하지 않는다. 실제 segmentation 일반화는 별도 테스트가 필요하다.
