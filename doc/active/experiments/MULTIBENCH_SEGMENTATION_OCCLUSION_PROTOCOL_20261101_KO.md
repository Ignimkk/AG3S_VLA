# LIBERO 계열 장면의 분할 및 가려짐 평가 — 탐색 규약

질문: 외부 세그멘터가 다른 시뮬레이션 장면과 실제 기하에 의해 가려진 대상의 **보이는 표면**을 얼마나 정확히 분할하는가? 지시문에서 관계를 포함한 대상 구절을 전달했을 때와 전체 지시문을 전달했을 때 차이가 있는가?

고정할 소규모 장면: 정적 init 0, seed 7, head/wrist 256×256. LIBERO Spatial/Object 각각 task 0·3·6·9 = 8 장면. SafeLIBERO Spatial/Object 각각 Level I·II, task 0·2 = 8 장면. LIBERO-Safety affordance/human_safety/obstacle_avoidance/obstacle_avoidance_human 각각 Level 0·1·2, task 0·1 = 24 장면. 모두 40 장면·80 camera image를 계획하며 환경을 만들지 못하거나 GT를 만들 수 없는 장면도 실패로 보고한다. LIBERO-Safety는 fork에서 별도 프로세스로 생성한다. 이 장면들은 전체 benchmark 대표 표본이나 성공률 평가가 아니다.

관측과 GT: 대기 동작 이후 `sim.forward`한 **한 상태**에서 RGB와 segmentation을 연속 렌더한다(중간에 step 없음). 목표 geom id는 benchmark 정답 goal을 평가 정답에만 사용한다. 같은 상태에서 다른 geom을 시각적으로 숨긴 target-only segmentation을 별도 렌더하고, 원래 보이던 target pixel이 target-only mask에 포함되는지 검사한다. 각 카메라에서 **가려짐 = 1 − 원래 visible target pixel 수 / target-only in-frame target pixel 수**다. 이는 시야 안의 target silhouette에 대한 다른 물체·로봇·환경의 차폐로, 화면 바깥으로 잘린 부분과 물체의 보이지 않는 뒷면은 측정하지 않는다. target-only silhouette가 없으면 화면 밖/미가시로 따로 기록하며 가려짐 0으로 치지 않는다. solo 렌더는 정답 진단에만 사용하며 모델 입력이 아니다.

세그멘터: 고정된 공개 Grounded-SAM-2 per-frame과 YOLOE-11L, 기존 기본 confidence/box threshold. 각 모델은 같은 RGB만 사용한다. ① 기록된 전체 instruction, ② **instruction만 읽는 고정 정규식**에서 추출한 target phrase를 별도 조건으로 평가한다. 문구 추출 실패는 누락을 숨기지 않고 기록하며 GT 객체 이름으로 대체하지 않는다. 실험에서는 해당 regex가 일부 언어 형식을 지원하지 않을 수 있다. 원문 보고 수치와 이 데이터의 수치를 직접 순위 비교하지 않는다.

지표: 기본은 target이 보이는 이미지에서 2D IoU·precision·recall; empty prediction은 IoU·recall 0, precision undefined. **실제 시야 내 가려짐 정도**는 0–<10%, 10–<50%, 50–100%로 기술하고 각 구간에서 모델/카메라별 n을 함께 표시한다. target이 화면에 전혀 보이지 않는 경우 비어 있지 않은 예측 비율을 별도 표기한다. 목표가 동일한 두 camera 결과는 독립 scene이 아니며 task·suite별 결과와 각 scene을 보존한다. 가려짐 구간에 장면과 물체 종류가 섞여 있으므로 단독 인과 효과로 해석하지 않는다. full-scene depth/camera가 정합되지 않은 상태의 3D IoU는 보고하지 않는다.
