# Object 다른 초기 상태의 분할 비교 — 수집 전 고정 규약

목적: Object init 0 개발 실험에서 선택된 고정 설정이 같은 10개 task의 다른 초기 배치에서도 유지되는지 확인한다. π0.5가 학습한 LIBERO 내의 초기 상태 이동이며 unseen object, 새로운 task, 독립적 최종 test 또는 실제 로봇 일반화가 아니다. 이 문서는 rollout 수집 및 평가 결과 확인 전에 작성한다. init 0의 전체 분할 결과와 이 turn의 일부 factorial smoke 프레임은 이미 관찰했으므로 엄밀한 사전등록 확증 실험이라고 부르지 않는다.

- LIBERO Object task 0–9, init 5와 6 각각 1회, 총 20 episode, π0.5-LIBERO 기존 checkpoint, seed 7; 정책 요청의 모든 frame. 요청 실패/GT 수집 실패는 감추지 않고 전체 분모를 밝힌다.
- Head/wrist RGB-D와 이미지 시점 상태를 저장하고, 각 frame을 재렌더해 RGB 일치 검사를 통과한 이미지 시점 target segmentation을 평가 정답으로 쓴다. 실패 시 해당 결과는 미완료로 보고한다.
- 동일 프레임에서 기존 Grounded-SAM-2 per-frame의 target phrase 및 full instruction, 원래 noun L8H6 surface adapter의 robot 없음과 기존 Panda FK sphere robot mask를 비교한다. 추가로 simulator image-time robot mask와 target patch는 특권 진단으로 따로 실행할 수 있다. 2-camera 연구 adapter이며 전체 AG3S serving 또는 동작 성능이 아니다.
- 기존 baseline provider default confidence와 graph 설정을 바꾸지 않는다. 개발에서 볼 수 있었던 성능을 근거로 task·frame·특정 실패를 제외하지 않고 baseline을 추가하는 경우에도 별도로 밝힌다. 본 결과를 본 후 설정을 수정하면 이를 탐색적 분석으로 명시한다.
- 5 mm soft-voxel IoU, target recall, 이웃 FP, abstention을 동일 프레임으로 계산한다. frame→episode→task→10 task 균등 평균, paired task bootstrap 10,000회 seed 20261101을 쓴다. task당 2 episode로 불확실성이 크며 카메라별 target 가시성, robot mask의 target 삭제도 따로 확인한다.
