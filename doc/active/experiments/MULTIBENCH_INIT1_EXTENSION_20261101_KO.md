# 다중 benchmark 분할: init 1 탐색 확장

기존 40개 init 0 장면을 수집한 결과 80개 camera 중 가려짐 50% 이상이 2개에 그쳤다. 따라서 [기존 규약](MULTIBENCH_SEGMENTATION_OCCLUSION_PROTOCOL_20261101_KO.md)의 같은 benchmark·suite·level·task 목록 전체를 새 env에서 init 1로 반복해 40 장면을 추가한다. 이 확장은 기존 **가려짐 분포를 본 뒤** 결정됐으므로 사전 등록된 독립 확인 실험이 아니라 탐색적 표본 확장이다. 성능을 보고 task를 고르지 않고 선정된 40개를 모두 실행한다.

동일한 seed 7, 실제 scene과 같은 상태의 target-only 렌더, 두 camera와 모델·prompt·지표·가려짐 분류를 사용한다. init 1은 같은 task의 재배치이며 새로운 task나 실제 로봇이 아니다. 모델별 오류·GT 실패·조건별 n을 보고한다.
