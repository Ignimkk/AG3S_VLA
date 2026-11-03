# SafeLIBERO 연속 정책 rollout의 가림 평가 추가 규약

## 질문 및 표본

기록된 실제 VLA 단독 정책 실행 중 장면 내 가림 정도와 외부 분할 baseline의 결과가 어떻게 변하는지 확인한다. ATS의 출력·비교는 사용하지 않는다. 이전 독립 정합 probe(동일 episode의 4시점×2 camera에서 RGB 완전 일치) 이후 성능 결과를 보기 전에, `20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only`의 SafeLIBERO Spatial Level I task 0, 1, 2 episode 0을 고정한다. 20개 대기 단계 포함 전체 관측을 빠짐없이 순서대로 평가하며, 원래 기록의 성공 여부와 프레임 수를 보고한다. 이미 존재하는 VLA 단독 정책을 새로 실행하거나 성공 대상으로 표본을 골라내지 않는다.

## 정합 및 가림 진단

각 task의 원래 seed 7, 첫 reset, init 0 환경을 복원하고 매 이미지 시점의 저장된 `sim_state_image`를 적용한 뒤 head RGB가 저장 영상과 **픽셀 단위로 일치해야** 한다. 모델 body/geom 이름 및 ID 표를 검사하고 일치하지 않는 episode 또는 영상은 점수에 포함하지 않는다. 저장 RGB에 original 정책 전처리 180도 회전 및 224 rescale을 적용한다. 같은 상태의 instance GT와 target만 표시한 GT 렌더를 같은 orientation으로 256에서 기록한 후 nearest-neighbor로 평가 해상도로 변환한다. 숨겼던 geom은 복원하며, visible GT가 target-only 안에 포함되는지, 렌더 전후 물리 상태가 바뀌지 않는지 검사한다.

화면 내 가림 = `1 - visible target pixel / target-only in-frame target pixel`. target-only에서도 대상이 0이면 화면 밖/렌더 범위 밖으로 별도 처리하고, target-only는 존재하나 visible이 0이면 완전 가림으로 기록한다. 이 비율은 이미지 프레임 밖으로 잘린 부분을 가림으로 계산하지 않는다. 같은 상태에 대한 두 GT 렌더는 점수 산출 전용이고 모델 입력이 아니다.

## 평가

별도 [연속 rollout 규약](BASELINE_CONTINUOUS_ROLLOUT_PROTOCOL_20261102_KO.md)과 같은 세 외부 baseline과 phrase 입력, 동일 IoU·경계·선택 진단을 사용한다. 추가로 `<10%`, `10–<50%`, `50–<100%` 가림 구간과 완전 가림 구간의 지속 프레임 및 이후 가시 target의 복구까지 프레임 수를 기록한다. 각 episode는 연속 관측으로 분석하며 frame은 독립 표본이 아니다. 이 세 개의 동일 suite·level episode는 SafeLIBERO 전체를 대표하지 않는다. 본 작업은 open-loop 시각 분석이며 원래 로봇 행동이나 성패를 수정하지 않는다.
