# 연속 rollout에 Grounded-SAM v1 확장

초기 세 arm의 LIBERO 결과를 확인한 뒤, 앞선 정적 실험에 포함한 Grounded-SAM v1도 연속 데이터에서 누락되지 않도록 추가한다. 이는 결과를 본 뒤 계획한 탐색 확장이다. 표본과 모델 입력, 가시 GT·가림·오류 처리 및 요약법은 각 dataset의 앞선 연속 실험 규약을 그대로 사용한다. Grounded-SAM v1의 detector box/text threshold는 0.3/0.25이며 SAM2 조합의 0.4/0.3과 달라 모델 버전 효과를 단독으로 분리하지 못한다. 각 프레임을 독립 탐지해 최고 점수 객체 하나를 SAM v1으로 분할한다. ATS는 사용하지 않는다.
