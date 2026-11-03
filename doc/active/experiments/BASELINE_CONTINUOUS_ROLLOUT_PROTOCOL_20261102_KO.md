# 외부 분할 baseline: 실제 정책 연속 rollout 탐색 규약

## 목적과 범위

ATS를 사용하거나 비교하지 않는다. 저장된 LIBERO 정책 rollout에서 같은 시점의 RGB와 simulator instance 정답을 이용해, 외부 baseline의 대상 선택·2D 분할·시간적 유지 성능을 측정한다. 기존 정적 결과를 확인한 후 마련한 탐색적 분석이다.

## 입력과 고정 표본

기존 `datasets/workspace/knows_p0b`의 LIBERO Spatial/Object 각각 task 0, 1, 2의 episode 0, 총 6개 파일을 결과를 보기 전에 선택한다. `collect_p0b.py`는 실제 policy가 시뮬레이터에 action을 실행할 때 매 관측의 RGB와 instance segmentation을 저장했다. 대기 단계를 제외한 저장 프레임을 **빠짐없이 시간순**으로 입력한다. 과거 원본의 성공/실패와 프레임 수는 기록하되 결과에 따른 선택은 하지 않는다. 카메라는 정답 instance segmentation이 저장된 head만 사용한다.

RGB는 저장된 224×224 정책 입력, GT는 동시 저장된 256×256 instance map을 수집 시 이미지와 같은 180도 변환 후 제공된 기존 변환식 `floor(i*256/224)`으로 최근접 리샘플한 224×224 mask다. RGB 보간과 GT 최근접 변환은 경계 한두 픽셀의 불확실성을 남긴다. 조작 객체 instance는 GT 평가만을 위해 저장된 manipulated 및 id2name에서 정확히 찾는다. 모델은 RGB와 지시문에서 고정 정규식으로 추출한 phrase만 사용하며 GT instance, 위치, 마스크를 보지 않는다. 다중 대상 또는 phrase 해석 실패는 정답을 보충하지 않고 별도 오류로 기록한다.

## 방법과 지표

- Grounded-SAM2 per-frame, Grounded-SAM2 causal track(첫 검출로 초기화, 이후 재검출 없음), YOLOE per-frame. 기존 기본 설정을 유지하고 각 시퀀스마다 상태를 reset한다. ATS는 포함하지 않는다.
- 가시 target이 존재할 때 2D IoU, precision, recall, 검출 결과 상자와 GT 가시 영역의 bounding-box IoU, 예측/GT 경계 2픽셀 허용 F1을 기록한다. IoU 0 이상 예측을 감추지 않고 모든 visible 프레임을 계산한다. 경계 F1도 빈 마스크를 0으로 취급한다.
- mask가 비어 있지 않지만 GT와 겹치지 않으면 'target 비중첩 예측'으로 기록한다. 이 표현 자체가 다른 객체를 선택했다는 증거는 아니며, 다른 instance가 차지하는 픽셀도 별도 기록한다.
- frame 평균과 episode별 평균 후의 평균을 함께 기록한다. frame은 시간적으로 강하게 상관되므로 독립 표본으로 통계적 유의성을 주장하지 않는다. 초기 IoU 0.5 이상 episode의 추적 결과를 진단적으로 구분한다.
- GT가 화면에서 사라진 연속 구간과 다시 나타나는 시점을 기록한다. target-only 렌더/카메라 경계 정보는 이 기록에 없으므로, 이러한 구간은 가림, 화면 이탈, 작은 대상의 리샘플 소실을 구분할 수 없다. '완전 가림' 또는 가려진 시간으로 단정하지 않는다. 후속 정합된 재렌더가 확인되면 따로 분석한다.
- 원래 episode의 실제 행동 주기와 저장 간격은 수집 당시 설정을 확인해 기록한다. 재등장 검사 시 마지막 구간이 시퀀스 끝에서 종료되면 복구를 판정하지 않는다.

## 재현 및 추가 근거

원본 데이터·수집 코드와 평가 코드의 해시, 모델 provenance, 전체 프레임 예측/평가를 저장한다. 실제 정책이 생성한 rollout이지만 본 실험은 오프라인 시각 평가이며 로봇 제어 성능이나 실시간 처리 지연 비교는 아니다. 별도의 SafeLIBERO policy 상태 파일을 이용한 정합 재렌더 가능성은 위 고정 LIBERO 표본과 독립적으로 점검하고, 원래 RGB 정합에 실패하면 성능 통계에 넣지 않는다.
