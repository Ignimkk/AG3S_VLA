# Grounded-SAM v1 외부 baseline 추가 측정

ATS 비교는 수행하지 않는다. 기존 다중 benchmark에서 Grounded-SAM2와 YOLOE 결과를 확인한 뒤, SAM v1을 사용하는 공개 Grounded-SAM도 같은 **자연 80장면 + 통제 가림 16장면**에서 추가 측정한다. 이미 다른 baseline 결과를 본 후 추가한 탐색 비교다.

기존 scene과 동일한 head/wrist RGB, task instruction, instruction-only 규칙으로 추출한 phrase를 사용한다. BDDL 정답 이름은 모델 입력에 사용하지 않는다. 구현은 프로젝트의 공식 Grounded-SAM adapter이며 Grounding DINO box threshold 0.3, text threshold 0.25와 top-1 box, SAM v1 mask를 사용한다. 기존 Grounded-SAM2의 box/text threshold 0.4/0.3과 다르므로 결과 차이를 SAM 버전만의 효과로 해석하지 않는다. 각 조건의 결과·실패·카메라와 가려짐별 sample 수를 기록하며 통제 장면은 자연 장면과 합치지 않는다.
