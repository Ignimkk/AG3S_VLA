# 대상 단서 × 로봇 제거: Object 개발 데이터 진단

> **입력 조건 정정:** 아래 외부 segmenter의 `phrase`는 실행 당시 BDDL goal에서 얻은 정답 객체 이름을 문구로 바꿔 제공했다. 정답 마스크를 제공한 것은 아니지만 **정답 대상 이름을 참조한 조건**이다. 반면 noun attention은 정책 지시문에서 대상 noun을 추출한다. 이후 지시문만 읽는 별도 규칙으로 전체 대상 구절을 추출해 대조하니, Object 개발 **10/10 episode**, init 5·6 **20/20 episode**에서 실제 외부 모델에 제공된 문자열과 정확히 같았다([개발 audit](../../../outputs/segmentation_baseline_20261101/object_dev/phrase_source_audit.json), [init 5·6 audit](../../../outputs/segmentation_baseline_20261101/object_init56/phrase_source_audit.json)). 이 Object 기록에서는 같은 문자열을 정답 이름 없이도 만들 수 있었지만, 당시 파이프라인이 실제로 정답 이름을 참조한 사실은 그대로다. 이 사후 검사는 다른 task에서의 일반성을 입증하지 않는다. `instruction` 행은 기록된 작업 지시문을 입력한다. 실행 코드: [query_for](../../../Baseline/paper_baselines/eval/offline_grounding.py), [BDDL target과 이름 변환](../../../Baseline/paper_baselines/providers/gt_oracle.py).

## 실험 목적과 조건

이전 결과에서 특권 GT 로봇 마스크와 GT 대상 단서를 넣었을 때 성능이 크게 달라졌다. 따라서 **대상을 가리키는 신호와 로봇 제거가 각각 결과에 어떻게 관련되는지**, 실제 관절 상태로 계산하는 로봇 구 마스크에서 어떤 절충이 생기는지 확인했다. [실행 전 규약](AG3S_SEGMENTATION_FACTORIAL_DEV_20261101_KO.md)대로 기존 LIBERO Object 10 task, 각 init 0의 281 policy-call frame을 사용했다. 이미 살펴본 개발 자료이므로 독립 검증이 아니다.

동일한 두 camera RGB-D와 graph에서 단서는 π0.5 noun attention `L8H6` 또는 **특권** image-time GT target의 16×16 patch coverage를, robot mask는 없음·Panda FK sphere·**특권** image-time GT robot을 썼다. FK는 이미지 시점 관절값, depth, camera calibration, 기존 Panda self-filter 기하 모델을 사용하며 GT segmentation은 사용하지 않는다. 다만 기존 기본 설정의 offline 2-camera adapter이고 정식 serving parity는 확인하지 않았다. GT를 모델에 넣은 조건은 실사용 성능이 아니다. Abstain은 빈 target 예측으로 점수화한다.

## 결과: 3D target surface

5 mm soft-voxel 기준이며 frame → episode → task → task 평균이다. 이웃 FP는 주변 물체의 잘못 포함된 voxel mass로, 비율이 아니다. 모든 조건의 분모는 281 frame이다.

| 대상 단서 | robot mask | IoU ↑ | recall ↑ | 이웃 FP ↓ | abstain ↓ |
|---|---|---:|---:|---:|---:|
| noun attention | 없음 | 0.189 | 0.434 | 167.3 | 17 |
| noun attention | FK sphere | 0.359 | 0.374 | 176.6 | 23 |
| noun attention | GT robot, 특권 | 0.410 | 0.431 | 173.9 | 22 |
| GT target patch, 특권 | 없음 | 0.360 | 0.784 | 2.4 | 1 |
| GT target patch, 특권 | FK sphere | 0.656 | 0.684 | 1.5 | 1 |
| GT target patch, 특권 | GT robot, 특권 | 0.743 | 0.783 | 1.5 | 1 |

Noun 조건에서 FK mask − mask 없음은 IoU **+0.170**, 10 task paired bootstrap 95% 탐색 구간 **[+0.108, +0.232]**이다. IoU는 좋아졌지만 recall은 0.434에서 0.374로 낮아졌다. GT target patch 조건에서도 FK mask − 없음은 **+0.296 [+0.277, +0.316]**이고 recall은 0.784에서 0.684로 감소했다. FK mask 조건에서 GT patch − noun은 **+0.297 [+0.188, +0.411]**이다. 따라서 이 개발 데이터에서는 대상 단서와 로봇 제거 모두 진단할 가치가 있지만, 특권 GT patch의 효과를 배포 가능한 모델의 개선량이나 완전한 상한으로 해석하면 안 된다. 한 task당 한 episode인 구간은 확증 통계가 아니다.

Noun 조건의 3D robot FP voxel mass는 task 평균 **475.1 → 0.2**로 줄었고, target FN voxel mass는 **215.1 → 237.6**으로 늘었다. 즉 이 데이터의 전체 IoU 향상은 로봇을 target으로 잘못 포함하는 오류 감소와 함께 나타났고, 대상 누락 증가라는 비용이 있다. 두 값 모두 평가 voxel의 class 혼합을 반영하는 mass이며 실제 부피나 독립 사건 수가 아니다.

동일 개발 프레임의 기존 Grounded-SAM-2 per-frame 대상 phrase baseline은 IoU **0.205**, recall **0.279**였다. FK 후보 − baseline의 task 평균 IoU 차이는 **+0.153**, paired task bootstrap 탐색 구간 **[+0.012, +0.282]**이다([과제별 차이와 입력 해시](../../../outputs/segmentation_baseline_20261101/factorial_summary/baseline_contrast.json)). 다만 task 6에서는 후보가 더 낮았고, 이 결과는 이전 baseline 결과를 이미 본 뒤 수행한 개발 비교이므로 확정적인 우월성 주장이 아니다. [기존 baseline의 전체 조건](AG3S_SEGMENTATION_BASELINES_20261101_KO.md)은 prompt 조건에 따라 큰 차이를 보였다.

**2D 대상 분할을 camera별로 따로 평가한 IoU**도 다음과 같다. 정답 대상이 해당 camera에 보이지 않는 frame은 IoU 평균에서 제외하며, 보이지만 예측이 빈 frame은 0이다. 두 camera에 같은 물체가 보이는 크기와 가림은 서로 다르므로 camera 자체의 효과로 해석하지 않는다. [2D 전체 결과](../../../outputs/segmentation_baseline_20261101/factorial_summary/score_2d.json)에는 target이 보이지 않는 frame에서 비어 있지 않은 예측 수와 precision/recall도 기록했다.

| 대상 단서 | robot mask | head 2D IoU | wrist 2D IoU |
|---|---|---:|---:|
| noun attention | 없음 | 0.165 | 0.333 |
| noun attention | FK sphere | 0.349 | 0.371 |
| noun attention | GT robot, 특권 | 0.410 | 0.506 |
| GT target patch, 특권 | 없음 | 0.321 | 0.617 |
| GT target patch, 특권 | FK sphere | 0.652 | 0.671 |
| GT target patch, 특권 | GT robot, 특권 | 0.749 | 0.894 |

[두 camera 분할 예시](../../../outputs/segmentation_baseline_20261101/factorial_summary/fixed_examples.png)는 task 0·4·9의 첫 프레임을 보여준다. 초록은 GT만, 빨강은 예측만, 노랑은 겹치는 부분이며 설명용 예시로 전체 분포를 대표하지 않는다.

## 로봇 마스크 자체의 오류

FK mask와 이미지 시점 GT robot/target을 camera별 pixel로 비교했다. 아래는 모든 frame의 pixel을 먼저 합친 **micro 비율**이다. 따라서 큰 물체와 길게 관측되는 frame의 비중이 크며 task 평균이나 독립 표본 비율이 아니다. `target 삭제율`은 GT target pixel 중 FK mask에 포함된 비율이다.

| camera | robot mask precision | robot mask recall | target 삭제율 |
|---|---:|---:|---:|
| head | 0.795 | 0.992 | 0.106 |
| left wrist | 0.797 | 0.550 | 0.300 |

위 전체 영상 기준에서 손목 robot recall이 0.550인 것은 **graph 입력에서의 로봇 누락률을 뜻하지 않는다**. 별도로 graph와 같은 유효 depth 범위(z 0.05–3 m, camera 기준 radial 거리 2 m 이하)만 집계하면 head robot recall은 **0.9963**, wrist는 **0.9998**이다. 손목의 전체 영상 robot FN 1,511,031 pixel 중 graph 유효 범위에 속한 것은 375 pixel이다. 대부분은 graph가 depth 범위에서 이미 제외한다. 그 범위 안에서 FK mask가 지운 target의 비율은 head **10.6%**, wrist **30.1%**였다([전체 영상·유효 depth 비교](../../../outputs/segmentation_baseline_20261101/factorial_summary/robot_validity.json)). 이 비율도 frame을 합친 micro 비율이다.

즉 이 관측 조건의 핵심 문제는 graph 입력에서 robot을 대량으로 놓치는 것보다 **대상 표면을 함께 제거하는 것**이다. 카메라 가림, 접촉·파지, 로봇 기하의 피복 및 calibration 오차를 아직 분해하지 않았으므로 그 구체적 원인은 확정하지 않는다. Task별 전체 영상 pixel 합계 기반 비율은 [원시 결과](../../../outputs/segmentation_baseline_20261101/factorial_summary/summary.json)에 있다. 로봇 픽셀 recall만 극대화하면 대상까지 삭제할 수 있으므로 두 오류를 동시에 보고해야 한다.

## 검증과 해석 범위

6개 조건의 frame 집합은 모두 281/281 일치했다. Noun + robot 없음 조건은 이전 결과와 **281/281 frame, 두 camera에서 pixel 단위 mask가 일치**했고 IoU·recall·이웃 FP도 같은 값이다. dummy camera에 selector 출력이 없음을 매 실행 검사했다. Panda 손가락 관절·self-filter 관련 기존 테스트는 **2 passed**, selector와 연결 경로 테스트는 의존성이 설치된 `.venv-ag3s`에서 **23 passed**다. 후자는 최초에 baseline 전용 환경에서 실행했을 때 CasADi·MuJoCo 부재로 6개 실패했고 올바른 환경에서 전체 재실행했다. [집계 결과](../../../outputs/segmentation_baseline_20261101/factorial_summary/summary.json), [조건별 표](../../../outputs/segmentation_baseline_20261101/factorial_summary/summary.md), [실행 당시 코드 사본](../../../experiments/segmentation/factorial_robot_seed_development_v1.py)에 각 실행 입력과 해시가 연결된다. 검증 집합에 GT seed만 실행할 수 있게 확장한 코드는 [현재 코드](../../../experiments/segmentation/factorial_robot_seed.py)다.

이 실험은 로봇 마스크와 대상 단서가 같은 graph의 입력으로 상호작용하는 진단이다. 특정 입력을 바꾼 효과를 다른 환경에 자동으로 일반화하거나, 관측되지 않은 물체 전체 3D 형상을 복원했다고 주장하지 않는다. 최종 AG3S fallback, temporal tracking, 작업 성공 및 접촉은 실행하지 않았다.

다음으로, 별도 초기 상태에서 고정 조건을 확인하기 위해 [Object init 5·6 검증 규약](AG3S_OBJECT_INIT56_VALIDATION_20261101_KO.md)을 수집 전에 작성했다. 이 역시 동일 LIBERO task 내의 초기 상태 이동이며 최종 확증 test가 아니다.

## 다른 초기 상태에서 로봇 마스크 확인

고정 규약대로 10 task × init 5·6 = **20 episode / 552 policy-call frame**을 수집했다. 이미지 시점 GT 재렌더 RGB는 head와 wrist 모두 **552/552** 일치했고, scene·phase·원본의 요청 frame 목록도 일치했다. GPU와 무관하게 먼저 계산할 수 있는 FK mask의 결과는 다음과 같다([원시 결과](../../../outputs/segmentation_baseline_20261101/object_init56/robot_validity.json)). 이 값들은 frame을 모두 합친 camera별 pixel micro 비율이다.

| camera | graph 유효 depth 내 robot recall | graph 유효 depth 내 target 삭제율 | 전체 이미지 내 robot recall |
|---|---:|---:|---:|
| head | 0.9965 | 0.1151 | 0.9923 |
| left wrist | 0.9998 | 0.3264 | 0.5503 |

다른 초기 배치에서도 graph에 들어가는 손목 target pixel의 약 32.6%가 FK mask에 포함됐다. [Task별 값](../../../outputs/segmentation_baseline_20261101/object_init56/robot_validity_by_task.json)은 두 초기 상태의 pixel을 task마다 합산했을 때 **18.5–47.5%**이며 10 task 모두에서 나타났다. 개발 집합의 30.1%와 비슷한 현상이 관찰되지만 같은 task·simulator·VLA의 서로 다른 두 초기 상태에 대한 관찰로 한정한다.

추가로 이미지 시점 MuJoCo geom에서 그리퍼–target 접촉이 기록됐는지에 따라 손목 영상의 대상 삭제율을 분리했다. 아래는 조건별 frame을 먼저 모아 target pixel을 합산한 **전체 영상 pixel micro 비율**이다. 접촉 여부는 정답 기반 분석 label일 뿐 모델 입력이 아니며, 단일 접촉이 안정적인 파지를 뜻하지 않는다.

| 데이터 | 비접촉 frame, target 삭제율 | 접촉 frame, target 삭제율 |
|---|---:|---:|
| 개발 init 0 | 122 frame, 8.3% | 159 frame, 36.0% |
| 새 init 5·6 | 225 frame, 6.9% | 327 frame, 38.4% |

접촉 프레임에 삭제가 집중됐으므로 손가락 주변 target 표면의 보존 여부를 따로 시험할 근거가 생겼다. 다만 frame이 같은 episode 안에서 상관되어 있고 카메라 각도·거리·가림도 함께 변하므로, 이 표는 접촉의 인과 효과가 아니다. 각 task 내부의 비율과 head 카메라 값은 [개발 자료](../../../outputs/segmentation_baseline_20261101/factorial_summary/contact_stratified.json), [새 초기 상태](../../../outputs/segmentation_baseline_20261101/object_init56/contact_stratified.json)에 기록했다.

추가 탐색 대조로 YOLOE-11L 대상 phrase, 공개 demo 기본 confidence 0.25를 같은 552 frame에서 CPU로 실행했다. Provider 오류는 없었고 촬영 시점 GT 기준 3D soft IoU **0.057**, recall **0.061**, head/wrist 2D IoU **0.039/0.053**이었다([전체 결과](../../../outputs/segmentation_baseline_20261101/object_init56/yoloe_image_instant/summary.json)). 양 camera 중 하나라도 비어 있지 않은 후보는 **41/552 frame**이며 이것은 정답 검출 수가 아니다. 빈 예측은 recall과 IoU에 0으로 포함하고 precision 계산에서는 제외한다. 이 설정은 YOLOE의 최적 성능을 보장하지 않고, 원 논문의 다른 benchmark 수치와 직접 비교할 수 없다.

기본 confidence 결과를 본 뒤, 기존 Object 개발에서 시험한 **0.05**도 추가 적용했다. 3D IoU **0.116**, recall **0.133**, head/wrist 2D IoU **0.104/0.126**, 한 camera 이상 비어 있지 않은 출력 **125/552**였다([결과](../../../outputs/segmentation_baseline_20261101/object_init56/yoloe_conf005_image_instant/summary.json)). 새 검증 결과를 이미 본 뒤 추가한 탐색적 민감도 분석이고 두 임계값 중 결과가 좋은 것을 최종 평가치로 고른 것으로 취급하지 않는다.

같은 552 frame에서 **특권 GT target patch를 graph에 제공한 진단**도 실행했다([3D 결과](../../../outputs/segmentation_baseline_20261101/object_init56/gt_seed_summary/summary.json), [2D 결과](../../../outputs/segmentation_baseline_20261101/object_init56/gt_seed_summary/score_2d.json)). FK sphere를 사용한 graph의 3D IoU는 **0.651**, recall은 **0.684**, head/wrist 2D IoU는 **0.649/0.655**였다. GT patch + robot mask 없음은 3D IoU 0.389, GT patch + 이미지 시점 GT robot mask는 0.750이다. 이는 정답 위치를 이미 입력하는 조건이므로 실제 candidate 성능이나 YOLOE에 대한 공정 비교가 아니다. 3개 조건 모두 552 frame을 처리했고 각 조건에서 abstain은 1회였다.

동일 검증 데이터의 noun `L8H6` attention은 **552/552 frame** 추출됐고, 2-camera 연구 adapter의 결과는 다음과 같다. frame → episode → task → task 평균이다. 각 camera의 2D IoU는 해당 camera의 target이 보이지 않는 frame을 제외한다. [전체 3D](../../../outputs/segmentation_baseline_20261101/object_init56/factorial_summary/summary.json), [2D](../../../outputs/segmentation_baseline_20261101/object_init56/factorial_summary/score_2d.json).

| 입력 | 3D IoU | 3D recall | head / wrist 2D IoU | abstain |
|---|---:|---:|---:|---:|
| noun attention, robot mask 없음 | 0.164 | 0.387 | 0.149 / 0.304 | 50/552 |
| noun attention + FK sphere | 0.318 | 0.333 | 0.320 / 0.325 | 58/552 |
| noun attention + GT robot, 특권 | 0.373 | 0.392 | 0.376 / 0.455 | 56/552 |

FK sphere − 없음의 3D IoU task 평균 차이는 **+0.155**, task bootstrap 탐색 구간 **[+0.094, +0.223]**이고 recall은 감소했다. 이 데이터에서도 10 task, task당 2 episode의 결과이므로 독립 task 일반화 또는 확증적 우월성으로 해석하지 않는다.

### 동일 프레임의 외부 baseline 비교

Grounded-SAM-2 per-frame 대상 phrase와 전체 instruction도 각각 552 frame을 실행했다. Provider 생성 및 예측 오류는 없었으며 후보·baseline마다 **정확히 같은 552개 frame key**를 확인하고 image-instant GT로 집계했다. YOLOE 0.05는 앞서 설명한 결과 관찰 후의 탐색 조건이다. 각 행은 frame → episode → task 평균이며, GT robot mask는 특권 조건으로만 표시한다.

| 방법 | 3D IoU | recall | 이웃 FP | head / wrist 2D IoU |
|---|---:|---:|---:|---:|
| image-time GT oracle, 특권 | 0.950 | 1.000 | 1.6 | 1.000 / 1.000 |
| noun + GT robot, 특권 | 0.373 | 0.392 | 197.8 | 0.376 / 0.455 |
| noun + FK robot | 0.318 | 0.333 | 199.7 | 0.320 / 0.325 |
| Grounded-SAM-2, 대상 phrase | 0.207 | 0.274 | 282.2 | 0.060 / 0.316 |
| noun + robot mask 없음 | 0.164 | 0.387 | 194.4 | 0.149 / 0.304 |
| Grounded-SAM-2, 전체 instruction | 0.129 | 0.247 | 1216.7 | 0.003 / 0.306 |
| YOLOE phrase, confidence 0.05, 사후 탐색 | 0.116 | 0.133 | 36.9 | 0.104 / 0.126 |
| YOLOE phrase, confidence 0.25 | 0.057 | 0.061 | 3.9 | 0.039 / 0.053 |

FK 후보 − SAM2 phrase의 3D IoU task 평균 차이는 **+0.111**, paired task bootstrap 탐색 구간 **[+0.010, +0.207]**이다. 10 task 중 3개에서 후보가 더 낮았다. FK 후보 − YOLOE 0.25는 **+0.261 [+0.191, +0.337]**이고, 사후 추가한 0.05에 대해서는 **+0.202 [+0.100, +0.304]**이다. 모든 구간은 10개 task의 기술적 재표집이며 task당 2 episode다. 동일 task·환경의 초기 상태 이동으로 얻은 수치이고 추가 prompt 최적화, 다른 task/suite 및 실제 로봇 검증 전의 결과다. 특히 SAM2의 전체 instruction 대 phrase 결과가 달라 prompt 설정에 따른 민감도를 함께 보고해야 한다.

전체 비교와 과제별 paired 차이는 [표](../../../outputs/segmentation_baseline_20261101/object_init56/combined_summary/summary.md), [JSON](../../../outputs/segmentation_baseline_20261101/object_init56/combined_summary/summary.json)에 있다. 특권 GT seed 조건은 이 경쟁 비교에 넣지 않았다. 별도 실행한 GT seed 예측은 552/552 frame, 두 camera 모두에서 전체 factorial 재실행과 bit 단위로 일치했다.

검증 집합의 실행 입력·코드·주요 결과 해시는 [provenance](../../../outputs/segmentation_baseline_20261101/object_init56/provenance_validation.json)에 기록했다. 외부 mask 재평가기는 이전 Object 개발 결과의 3D IoU·recall·precision·이웃 FP와 일치함을 별도 확인했다.
