# 외부 baseline: 가림 및 재등장 순서 실험

## 목적

ATS 비교는 수행하지 않는다. 앞선 [통제 가림 실험](MULTIBENCH_SEGMENTATION_OCCLUSION_RESULTS_20261101_KO.md)의 같은 8개 SafeLIBERO 원본 장면을 이용해, 외부 baseline이 부분 가림·완전 가림·재등장에 어떤 mask를 내는지 측정한다. 특히 프레임별 Grounded-SAM-2와 causal SAM2 tracking의 차이를 살펴본다.

## 고정 조건

- 각 시퀀스는 같은 original head RGB → 3D 상자 `stress_00` 부분 가림 → `stress_06` 완전 가림 → original RGB 재등장의 **4개 관측**이다. 기존 저장된 RGB·동기화 GT를 그대로 사용하고 대상·지시문·camera는 바꾸지 않는다. 원본 RGB와 재등장 RGB는 bit 동일하며, target-only silhouette는 앞선 평가에서 32/32 동일했다.
- 이는 **시뮬레이터에서 상자를 두 위치로 순간 이동한 정적 영상을 연결**한 인공 4-frame stress sequence다. 영상 사이 보간·물리적인 연속 속도·실시간 주기는 없고, 일반적인 로봇 rollout·비디오 추적 성능이나 장시간 가림에서의 복구를 주장하지 않는다.
- 지시문만으로 추출한 같은 대상 phrase를 사용한다. 입력 위치의 GT나 GT mask는 제공하지 않는다. 매 시퀀스마다 tracking state를 초기화한다.
- 비교군은 기존 공개 기본값: Grounded-SAM-2 **per-frame detector + SAM2 mask**, **causal track**, **causal track + mask가 비었을 때 다음 프레임에서 재검출**, YOLOE-11L per-frame이다. 마지막 설정은 구현의 `redetect_on_loss=True`이며 검출이 틀렸지만 mask가 비지 않은 경우 자동 교정하지 않는다. 프레임별 저장 mask를 재사용하는 arm의 원래 동일 입력을 검증한다.
- 정답 대상이 보이는 초기·부분 가림·재등장에서는 2D IoU·recall 및 비어 있지 않은 출력 수를 기록한다. **완전 가림에서는 IoU를 계산하지 않고** 비어 있지 않은 출력 수를 기록한다. 비어 있지 않은 mask는 wrong-object일 수도 있고 마지막 위치의 잔상일 수도 있으므로 추가 instance GT 없이 이 둘을 단정하지 않는다.
- 초기 프레임 IoU ≥ 0.5를 임의의 사전 고정 설명 기준으로 하여 초기 성공·실패를 나눠 재등장 결과를 함께 낸다. 이는 초기 실패를 제외해 전체 성능을 올리려는 것이 아니라 최초 실패와 추적 손실을 구분하는 진단이다. 8개 원본 중 level과 task가 공유되는 경우가 있어 8개 독립 task의 통계가 아니다.
- 에러·부재·마스크 shape 오류 및 초기화 실패는 누락하지 않고 기록하며, 원본과 재등장 프레임이 같은 데이터라도 tracking arm은 서로 별도로 추론한다. 현재 실험은 두 모델의 baseline 측정만 수행한다.
