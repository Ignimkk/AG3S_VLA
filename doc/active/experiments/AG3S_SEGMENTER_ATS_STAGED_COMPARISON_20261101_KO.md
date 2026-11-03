# 외부 segmenter 기준 확인 → ATS 대체 비교

## 연구 질문과 순서

사용자 연구 방향: 먼저 SAM 계열 또는 Grounded-SAM 계열을 사용해 **객체 분할 성능과 평가 방법을 확인**한다. 다음 단계에서 **외부 segmenter를 썼을 때와 AG3S ATS를 썼을 때** 무엇이 달라지는지 분석한다. 충돌 회피는 분할의 후속 응용 중 하나이며 첫 번째 검증 목표가 아니다.

1. **분할 기준 확인:** 원문의 지표와 조건을 정확히 인용하고, 이 프로젝트의 동일 영상·depth에서 외부 segmenter의 2D target 분할 및 관측 3D 표면 분리를 평가한다. GT는 평가용으로만 쓴다. 비어 있는 mask, 틀린 대상, 경계 오류, 가림 및 추적 실패를 따로 기록한다. 원문 수치와 LIBERO 수치를 같은 순위표에서 직접 비교하지 않는다.
2. **ATS 대체 비교:** 같은 이미지·depth·지시·정책 rollout을 고정하고 외부 segmenter 경로와 **실제 ATS 경로**를 비교한다. 먼저 입력과 출력 형태 차이를 통제한 offline 인식 실험을 한다. 그 후 필요할 때 같은 AG3S 후속 모듈에 두 인식 결과를 각각 공급하는 교체 실험으로 확장한다. 현재의 작업은 이 두 번째 단계의 실행을 완료했다는 뜻이 아니다.

## 1단계: 무엇을 확인하는가

| 질문 | 측정과 조건 |
|---|---|
| 지시된 물체를 구별했는가? | target identity 정확도, 다른 물체 선택, 무검출. 동일 물체 종류가 여러 개이면 위치·관계 표현도 포함한다. |
| 보이는 부분을 제대로 분할했는가? | 카메라별 2D IoU·precision·recall 및 대상 미가시 frame의 비어 있지 않은 출력. 빈 예측의 분모를 밝힌다. |
| 3D로 옮겨도 유지되는가? | 같은 관측 depth·camera calibration·voxel 또는 공통 점군에서 target IoU·recall, 이웃·robot 오포함. 가려진 전체 물체의 3D 완전 복원 성능으로 부르지 않는다. |
| 시간과 시야 변화에 견디는가? | 동일 episode의 per-frame 및 causal track: identity 전환, 가림 뒤 재발견, 지연, 시작 프롬프트와 실패/복구 횟수. 서로 다른 tracking 정보나 초기 정답 제공은 명시한다. |

현재 분석 자료: [분할 baseline 및 원문](AG3S_SEGMENTATION_BASELINES_20261101_KO.md), [Object 분할 및 로봇 제거 진단](AG3S_SEGMENTATION_FACTORIAL_RESULTS_20261101_KO.md). 이들 가운데 Object의 SAM2 phrase는 **실행 코드가 BDDL 정답 이름을 참조**했다. 사후 audit에서 Object 10개 개발 episode와 init 5·6의 20개 episode는, 지시문만으로 추출한 전체 대상 문구와 실제 전달된 문구가 문자 단위로 같았다. 반면 Spatial은 40개 episode 모두 달랐다. 예컨대 `black bowl between the plate and the ramekin` 대신 `akita black bowl`을 입력했기 때문에 관계 표현을 검증한 결과로 쓸 수 없다([audit](../../../outputs/segmentation_baseline_20261101/phrase_source_audit_spatial.json)). 이 audit은 실제 기존 실행의 정보 출처를 바꾸지 않으며, 지시문-only parser가 일반 task에서도 잘 동작한다는 증명도 아니다.

## 2단계: ATS와의 공정한 비교

실험 전에 실제 ATS 버전·attention head·클러스터·추적/latch·fallback과 외부 모델 체크포인트·prompt·tracking·confidence를 기록하고 선택을 고정한다. 모든 방법의 대상 문구는 **정답 객체 이름이나 simulator GT를 참조하지 않고 작업 지시문에서 추출**한다. 전체 지시문, 지시문 추출 phrase, 정답 이름을 쓰는 진단은 구분한다. 파서가 실패하면 임의의 정답으로 바꾸지 않고 실패로 남긴다. GT mask 또는 GT 이름을 모델에 주는 경우 별도 특권 행이다.

| 비교 수준 | 고정하는 것 | 바꾸는 것과 확인할 차이 |
|---|---|---|
| 인식만, offline | 동일 정책 기록·frame·RGB-D·GT·카메라·평가 단위 | 외부 mask와 실제 ATS의 target identity, extent, 누락·이웃 혼입·가림 및 추적 차이. ATS가 반환하는 대상 점/영역과 dense mask는 공통 점군 또는 동일 3D voxel로 변환해 비교하며 변환 오차도 기록한다. |
| 통합, 후속 단계 | 정책, 장면, 초기 상태, 로봇 마스크, 후속 target 처리·제약·controller·실행 예산 | ATS 또는 외부 segmenter 입력을 교체했을 때 거리장·작업·접촉·개입 차이. 외부 mask → 실제 target 입력 경로 및 ATS tracking·fallback과의 대응을 먼저 구현하고 검증한다. 관측 지표와 행동 지표는 따로 보고한다. |
| 비용 | 같은 기기·측정 구간·동일 프레임 | 첫 실행과 안정화 후 latency, 메모리·모델 크기, 프레임 처리율. 외부 모델 추론·depth lift와 ATS attention 추출·재사용·후처리를 각각 포함하고 캐시 및 동기화 조건을 밝힌다. |

공정성은 prompt 문자열을 기계적으로 똑같이 주는 것만으로 해결되지 않는다. 한쪽의 tracking·latch를 켰다면 다른 쪽도 그 차이를 독립된 조건으로 측정하고, 모델별 threshold와 prompt의 개발 예산·선택 시점을 기록한다. 같은 종류의 다른 물체를 구별하는 지시, 작은 물체, 접촉·가림 상황은 필수로 나눈다. 실패한 frame을 빼지 않는다. 충분한 episode와 suite에 대해 사전에 고정한 독립 test를 준비한다. 기존 계획에 적힌 init 범위가 이번에 관찰한 LIBERO Object init 5·6과 같거나 파생될 가능성도 확인한 뒤 겹침을 제외한다.

## 기존 결과를 이 질문에 쓰는 범위

- Spatial 40 episode의 [공통 점군 비교](../../../outputs/segmentation_baseline_20261101/cloud_dev40_v2/task_summary.md)는 **이전 V11AB ATS, action-query L8H2**와 SAM2를 같은 점에서 본 개발 진단이다. ATS point IoU 0.290, SAM2 instruction 0.467이지만 task·표현·5 mm 매칭 근사 및 개발 데이터의 한계가 있다. 최종 ATS 우열이나 전체 파이프라인 결과가 아니다.
- Object 552 frame의 0.318 대 0.207은 **독립 frame의 noun attention-surface 연구용 adapter + FK mask** 대 SAM2 phrase다. 이 adapter는 실제 ATS의 전체 상태·latch·fallback과 배포 parity를 검증하지 않았으므로 **실제 ATS 대 외부 segmenter의 최종 비교가 아니다**. 0.207의 phrase는 위와 같이 실행 시 BDDL 이름에서 만들어졌지만 Object에서는 지시문-only 규칙과 문자열이 일치했다.
- 기존 제약 factorial 계획 [실험 F](../plans/AG3S_EXPERIMENT_PLAN_20261030_KO.md)는 두 번째 단계의 행동 실험 후보로 보관한다. 외부 segmentation을 실제 AG3S server의 target 인식에 주입하는 경로는 아직 검증되지 않았으므로 이 문서에서는 실험 결과로 취급하지 않는다.
