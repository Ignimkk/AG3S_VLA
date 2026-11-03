# SafeLIBERO 실제 3D 물체를 이용한 통제 가림 — 탐색 확장

기존 init 0 장면의 camera 80개 중 대상이 50% 이상 가려진 경우가 2개뿐이어서, 가려짐 조건을 명시적으로 바꾼 짝 비교를 추가한다. 원본 데이터의 가려짐 분포를 본 뒤 정한 탐색 확장으로, 최종 독립 검증이 아니다.

대상은 기존 SafeLIBERO Spatial/Object × Level I/II × task 0/2, init 0의 8 장면이다. 원본을 재생성해 이전 RGB·target mask와 정확히 일치하는지 확인한 후, 같은 simulator의 `white_storage_box_obstacle_1` 자유관절을 대상과 head camera 사이로 옮긴다. camera 기준 시선에 대한 가로 offset 0 m와 0.06 m의 두 고정 위치를 사용한다. 목표의 simulator 위치는 가림 장면을 **생성하기 위해서만** 사용하고 분할 모델에는 전달하지 않는다. 상태 변경 후 물리 step 없이 RGB·GT·target-only를 같은 상태에서 렌더한다. 박스가 공중에 있거나 다른 물체와 겹칠 수 있으므로 실제 배치 확률이나 현실 장면으로 해석하지 않는다.

원본·가림 두 조건에 동일 공개 모델·입력 문구와 2D 지표를 적용하고, camera별 target-only 대비 실제 보이는 비율과 원본 대비 IoU 변화량을 짝지어 보고한다. target이 완전히 가려지면 IoU를 정의하지 않고 빈 예측 여부를 따로 평가한다. 이 장면을 자연 분포 80개 장면과 합쳐 단순 benchmark 평균으로 제시하지 않는다.
