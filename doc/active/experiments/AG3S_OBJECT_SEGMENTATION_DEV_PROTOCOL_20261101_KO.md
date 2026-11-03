# LIBERO Object 개발 데이터의 target 분할 비교 규약

목적: Spatial 개발 집합은 task가 모두 검은 bowl을 사용한다. 동일 정책과 고정된 Spatial 선택 head가 다른 target 물체 10종에도 쓸 수 있는지, 외부분할과 3D 영역 복원에서 얼마나 차이 나는지 살펴본다. 이는 개발 목적의 suite 이동 평가이며 공식 held-out test가 아니다. π0.5-LIBERO 자체가 LIBERO 데이터를 사용하므로 unseen-object 또는 zero-shot VLA 일반화로 주장하지 않는다.

수집: π0.5-LIBERO checkpoint, `libero_object` 10 task 전부, task별 init 0 한 episode, seed 7, RGB·depth·camera·recorded policy request·simulator state·geom segmentation을 함께 저장한다. 기준 목록은 task 0–9 모두이며 결과를 보고 task를 제외하지 않는다. 현재 수집 중인 run: `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261101T035128_96fda38abd1b_segmentation-object-dev-10-20261101/`. episode당 1회라 작업 성공률이나 일반화 성능을 통계적으로 주장하지 않는다.

대상 이름: BDDL의 goal이 움직이는 object를 simulator GT 식별에만 사용한다. 추출된 10개 이름과 명령문을 점검하고, ambiguity나 geom 매칭 실패는 제외하지 않고 오류로 기록한다. 예측 모델에는 해당 실험에서 허용하는 instruction 또는 target phrase만 전달한다.

평가 프레임: 성공/실패나 예측값에 무관하게 저장된 policy call 전부. source RGB와 동일한 image-instant simulator state를 재렌더해 GT segmentation을 얻고, 재렌더 RGB와 원 기록 두 카메라가 bit-identical인지 검사한다. 충돌·안전 benchmark가 아닌 기본 LIBERO Object이며, 추가 장애물을 넣지 않는다.

후보와 비교군: Grounded-SAM 및 Grounded-SAM-2 per-frame/causal-track (target phrase와 full instruction을 모두 보고), YOLOE (phrase와 instruction), 동일 rollout에서의 localization heads/TokenCut, simulator GT mask. 검출 실패·마스크 없음·provider exception의 각 분모와 원인을 숨기지 않는다. 모든 method가 같은 depth·camera·frame을 쓰며 provider version/weights/threshold를 기록한다.

우리 후보: 동일 frame과 head/wrist noun attention `L8H6`을 고정해 기존 attention_surface graph를 호출하는 **탐색적 2-camera adapter**. 로봇 제거 없음의 비특권 결과를 기본으로 하고, simulator GT robot mask 및 GT target patch 조건은 별도의 특권 진단으로 낸다. 세 번째 camera는 robot mask로 완전히 제외한 dummy이며 실제 Panda serving parity를 주장하지 않는다. 객체 이름의 마지막 토큰을 찾지 못하면 오류/불능 비율을 포함해 보고하며 임의의 target을 선택하지 않는다. 후보가 abstain하면 빈 예측으로 평가하고, 완성된 AG3S의 fallback으로 해석하지 않는다.

지표: image-instant GT 기준으로 카메라별 2D IoU/precision/recall, 두 view 5 mm soft-voxel 3D IoU/precision/recall과 이웃 FP, 선택 실패율. frame 평균 후 episode 평균 및 10 task의 동일 가중 평균. 빈 예측의 precision이 정의되지 않는 점을 함께 보고한다. paired task 부트스트랩은 10 task에 대한 탐색적 구간으로 제시하고 우월성·비열등성의 확정 검정으로 취급하지 않는다. phase 분류를 쓸 수 있으면 물리 규칙을 기록하고 그 정확성 한계를 병기한다.

이 문서는 수집 run을 시작한 직후에 작성했으며, 수집된 일부 episode의 환경 성공 로그를 볼 수 있었다. 분할 성능·baseline 예측 결과를 보기 전에 대상 task·init·frame·비교 항목을 고정했다. 결과 기반으로 이 문서를 소급하여 preregistered test라고 표현하지 않는다.
