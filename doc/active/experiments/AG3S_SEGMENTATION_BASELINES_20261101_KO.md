# AG3S target 분할: baseline 및 개발 실험 (2026-11-01 KST)

> **입력 조건 정정:** 아래 외부 segmenter의 `phrase`는 BDDL goal의 정답 객체 이름을 변환해 전달한 **oracle-assisted target-name 조건**이다. GT 마스크를 입력하지는 않았다. 사후에 지시문만으로 전체 대상 구절을 추출해 대조하면 Object 개발 10/10 및 init 5·6 20/20 episode는 전달된 문자열이 같았으나, **Spatial 40/40 episode는 달랐다**. 예를 들어 지시문의 `black bowl between the plate and the ramekin` 대신 기존 코드는 `akita black bowl`을 전달해 관계 표현을 빼고 이름을 추가했다([Spatial audit](../../../outputs/segmentation_baseline_20261101/phrase_source_audit_spatial.json)). `instruction`은 기록된 작업 지시문을 사용하며, noun attention은 그 지시문에서 noun을 찾는다. 따라서 Spatial phrase 결과를 지시문만 사용한 end-to-end 비교로 해석하지 않는다. [query_for](../../../Baseline/paper_baselines/eval/offline_grounding.py), [target-name 변환](../../../Baseline/paper_baselines/providers/gt_oracle.py).

## 결론과 범위

연구의 첫 번째 질문을 **지시된 target을 주변 물체와 분리하고 관측된 3D 표면을 복원할 수 있는가**로 둔다. 충돌 회피나 성공률은 이 인식 결과가 행동에 어떤 영향을 주는지 별도로 시험한다. 아래는 개발 데이터에 대한 진단이며, AG3S의 확정 구성 또는 held-out test의 결과가 아니다.

2026-10-24 기록을 재사용한 LIBERO Spatial 10 task × init 1–4 = 40 episode, head/wrist 2 camera, 총 873 policy-call frame. 작업 대상은 같은 종류의 검은 bowl이다. 하나의 frozen π0.5 rollout을 모든 방법이 사용한다. simulator target geom과 image-instant re-rendered segmentation은 **평가 정답**이며 배포 입력이 아니다. 정답 이미지와 depth는 같은 촬영 시점으로 맞춘다. 그래도 데이터는 모두 개발에 사용됐고, 카메라·scene geometry가 제한되어 최종 주장을 지지하기에는 부족하다.

## 실험 A — 외부분할과 후보 surface의 동일 RGB-D 비교

**검증 질문:** 같은 frame에서 외부 segmenter와 attention + depth surface가 어느 정도 target 표면을 복원하며, 이웃 표면을 얼마나 잘못 포함하는가?

5 mm voxel grid에서 target class의 비율을 쓰는 soft IoU와 recall, 예측 영역 안의 비-target 중 다른 물체·destination의 voxel mass(이웃 FP)를 계산했다. 각 지표는 frame → episode → task → task 평균 순서로 집계했다. 이웃 FP는 비율이나 mm³가 아니며 예측 크기와 표본의 표면 구성에 영향을 받는다. 95% 구간은 10개 task cluster를 재표집한 탐색적 bootstrap이다. 40 episode를 서로 독립인 40개 task로 취급하지 않았다.

| 비교 방법 | 3D soft IoU ↑ | target recall ↑ | 이웃 FP ↓ | 구분 |
|---|---:|---:|---:|---|
| 이미지 시점 GT mask | 0.970 | 1.000 | 1.3 | simulator 특권 정답; voxel 경계 혼합으로 IoU가 1 미만 |
| Grounded-SAM, instruction | 0.535 | 0.745 | 360.6 | 기존 공개 모델/adapter의 저장 mask |
| Grounded-SAM-2, per-frame instruction | 0.528 | 0.699 | 303.4 | 기존 공개 모델/adapter의 저장 mask |
| Grounded-SAM-2, causal track instruction | 0.450 | 0.569 | 220.4 | 기존 저장 mask |
| YOLOE, target phrase | 0.355 | 0.458 | 255.5 | 기존 저장 mask |
| **Surface 후보, robot mask 없음** | **0.263** | **0.442** | **46.2** | 이번 2-camera 탐색적 adapter, head L8H6 |
| Surface 후보, GT robot mask | 0.412 | 0.437 | 41.8 | **특권 진단**; 실제 배포 성능 아님 |
| Surface 후보, GT target patch + GT robot mask | 0.479 | 0.507 | 0.5 | **특권 진단**; target mask 전체를 직접 사용하는 oracle과 다름 |
| Localization heads + SAM point | 0.216 | 0.403 | 195.6 | 같은 dev에서 head 선택, in-sample |
| TokenCut foreground | 0.078 | 0.746 | 466.7 | object discovery adapter |

전체 [machine-readable 결과](../../../outputs/segmentation_baseline_20261101/image_instant_all873/summary.json)와 [전체 표](../../../outputs/segmentation_baseline_20261101/image_instant_all873/summary.md). 동일 task의 paired difference: surface(no robot mask) − Grounded-SAM-2(instruction) IoU = **−0.265**, task bootstrap 95% CI **[−0.423, −0.071]**. 로봇 마스크 GT 진단은 **−0.117 [−0.307, +0.114]**. Task가 10개뿐이고 선택·수정에 사용된 dev이므로 확정적인 일반화 검정이 아니다.

**2D 분할 자체도 따로 보면**, 촬영 시점 GT 기준 task 평균 IoU는 아래와 같다. 해당 카메라에 target GT가 보이지 않는 프레임의 IoU는 정의하지 않고 평균에서 제외한다. 따라서 미검출 또는 빈 mask이지만 target이 보이는 프레임은 0으로 포함한다. 두 카메라의 유효 프레임 집합이 다르므로 숫자 차이를 카메라만의 효과로 단정하지 않는다. 예컨대 wrist GT에 target이 전혀 없는 31프레임 중 SAM2 per-frame은 29프레임, surface 무 robot mask는 5프레임에서 비어 있지 않은 mask를 냈다. 이 횟수는 episode 독립 표본이 아니다.

| 방법 | head 2D IoU | wrist 2D IoU |
|---|---:|---:|
| Grounded-SAM-2 per-frame, instruction | 0.207 | 0.844 |
| Grounded-SAM-2 track, instruction | 0.613 | 0.240 |
| Surface 후보, robot mask 없음 | 0.221 | 0.352 |
| Surface 후보, GT robot mask (특권 진단) | 0.408 | 0.446 |

카메라별 검출 실패, 가림, 초기 추적 실패와 target 크기 분포를 분해해야 한다. Track과 per-frame이 서로 다른 camera에서 좋게 나온 결과만으로 어느 쪽이 전체적으로 항상 우수하다고 결론 내리지 않는다.

**실험적인 adapter의 실제 내용:** `runtime.attention_surface.select_target_surface`를 그대로 호출하되 RB-Y1용 세 카메라 인터페이스에 Panda head/wrist만 채운다. 세 번째 카메라는 유효 depth 1 pixel을 제공한 뒤 전부 robot mask로 제외한다(기존 backprojection이 완전 빈 카메라를 허용하지 않기 때문). `n_labelled_px[right_wrist] == 0`을 확인했다. 입력 attention은 기록된 `noun_target:L8H6`을 policy orientation에서 OpenCV orientation으로 바꿨다. 이 head는 같은 40개 dev episode에서 선택돼 **in-sample**이다. RB-Y1 기본 graph 설정을 Panda에 직접 사용했으며 Panda 최종 배포 pipeline과 parity를 검증하지 않았다. 각 frame 독립 평가이며 temporal identity 유지·latch는 실행하지 않는다. selector가 abstain하면 해당 frame의 target 예측을 빈 mask로 점수화한다. 실제 서버의 다른 fallback을 실행하지 않으므로 **통합 AG3S 성능이 아니다**.

GT robot mask는 기록의 return-time 로봇 segmentation에서 왔으므로 image-instant robot GT와도 차이가 날 수 있다. GT target patch 진단은 image-instant target mask를 16×16 patch coverage로 바꿔 seed로 전달한다. 이는 정답을 완전히 넣은 oracle mask가 아니며 배포 불가능한 상한 실험도 아니다. 표면 graph의 seed와 robot 제거가 좋은 조건일 때도 어디에서 놓치는지 확인하는 진단이다.

Surface selector 적용/abstain: no robot mask 735/138, GT robot mask 690/183, GT patch+robot mask 643/230 (873 frame). GT patch+robot 조건의 abstain은 모두 `no_region`, 즉 점수 ≥ 0.5인 최종 region이 없는 경우였다. 접근 단계에서는 404 frame 중 193회다. 추가 원인 진단으로 `no_region`이 있었던 task마다 앞의 최대 3건씩 총 29건을 다시 실행했다([frame별 결과](../../../outputs/segmentation_baseline_20261101/no_region_diagnostic/rows.jsonl)). 29건 모두 GT target에 해당하는 graph node가 있었지만, 최종 임계값 이전의 최대 점수는 0.060–0.485(중앙값 0.295)로 모두 0.5 미만이었다. GT target node 중 support로 분류된 비율의 중앙값은 0.361이었다. 이 29건은 의도적으로 고른 실패 사례이며, **이것만으로 원인이 target 크기·support 검출·threshold 중 무엇인지는 단정할 수 없다.** 해석 시 abstain을 제외한 precision만 보면 선택하지 못한 frame을 숨기므로 분모를 함께 낸다. 모든 조건은 객체가 들린 시점도 포함하고, 접근 단계는 기존 MuJoCo 규칙 기반 label이므로 완벽한 인간 주석이 아니다.

촬영 시점 GT와 기존 return-time GT에서 target pixel이 다른 frame은 head **563/873**, wrist **593/873**이다. 같은 이미지 시점 GT로 return-time oracle mask를 점수화한 IoU는 0.963이며 image-time oracle의 0.970보다 낮다. 차이는 이 데이터의 평균에서 작지만 경계 정확도 평가에서는 시간 정합을 요구해야 한다. 원본 40 episode의 npz와 geometry hash는 기존 V11AB replay와 segmenter 입력에서 **40/40 일치**한다.

## 실험 B — 같은 3D 점에서 V11AB와 baseline 비교

**검증 질문:** 대표점 수가 적은 V11AB를 dense mask의 voxel과 직접 비교할 때 생기는 표현 차이를 없애도 결론이 유지되는가?

동일한 40 episode·873 frame에서 V11AB가 기록한 filtered cloud를 평가 단위로 고정했다. simulator의 image-instant mesh 표면과의 10 mm 이내 최근접 instance를 점별 GT로, segmenter mask는 관측 카메라의 투영 결과로 point별 예측을 정한다. V11AB는 실제 exclusion geometry에서 5 mm 이내인 cloud 점을 target 예측으로 본다. 이 5 mm 대응은 평가용 근사다. 비교 결과(episode 평균): V11AB **point IoU 0.290, recall 0.300**, Grounded-SAM-2 per-frame instruction **0.467, 0.641**. task 단위 paired difference(V11AB − SAM2) = **−0.177 [−0.316, −0.044]**. 완전 원시 결과와 arm별 수치는 [동일 점군 표](../../../outputs/segmentation_baseline_20261101/cloud_dev40_v2/task_summary.md), [JSON](../../../outputs/segmentation_baseline_20261101/cloud_dev40_v2/task_summary.json).

주의: 결과 파일의 오래된 arm 이름 `ats_v11ab|noun_attention`은 잘못된 라벨이다. **이 replay가 실제 사용한 것은 action-query L8H2이며 noun attention이 아니다.** 출력 표에는 action L8H2로 바로잡았다. 이는 이전 V11AB이고 수정 중인 surface 또는 최종 AG3S와 다르다. Point IoU는 실험 A의 voxel IoU와 다른 지표이므로 수치를 서로 뺄 수 없다. oracle point IoU도 0.849로, mesh 근접 거리와 카메라 투영 때문에 완전한 1이 아니다.

## 실험 C — mask 경계 오류가 관측 3D 영역에 미치는 영향

**검증 질문:** 2D mask가 과대·과소 분할되면 3D target 누락과 잘못 지워질 수 있는 비-target 표면은 어떻게 달라지는가? 기존 40 episode에서 episode당 고정 간격의 5 frame, 총 200 frame을 선택했다. 동일 camera·depth·voxel grid로 simulator GT mask에 2/4/8 pixel 팽창 또는 침식을 준 통제 실험이다. 이 실험의 기준 mask와 GT는 기존 return-time 기록을 사용하므로 촬영 시점 GT 결과로 표기하지 않는다. ground-truth mask 행의 3D IoU는 기존 결과와 **200/200 frame에서 정확히 일치**한다.

| 오류 | 3D IoU | target recall | 비-target FP voxel mass | target FN voxel mass | 이웃 FP voxel mass |
|---|---:|---:|---:|---:|---:|
| 오류 없음 | 0.971 | 1.000 | 20.9 | 0.0 | 1.9 |
| 2 px 팽창 | 0.747 | 1.000 | 236.3 | 0.0 | 25.8 |
| 4 px 팽창 | 0.630 | 1.000 | 410.4 | 0.0 | 46.8 |
| 8 px 팽창 | 0.473 | 1.000 | 785.2 | 0.0 | 95.7 |
| 2 px 침식 | 0.840 | 0.844 | 2.9 | 103.3 | 0.2 |
| 4 px 침식 | 0.702 | 0.703 | 1.2 | 198.8 | 0.1 |
| 8 px 침식 | 0.481 | 0.481 | 0.1 | 353.0 | 0.0 |

[전체 결과](../../../outputs/segmentation_baseline_20261101/perturbation_dev40/summary.json)는 카메라 하나의 target mask가 빠지는 진단도 포함한다. 팽창 시 FP, 침식 시 FN의 frame별 단조성은 모두 확인했다. 이 실험은 **관측 3D 영역의 오류만 측정**하며 충돌이나 작업 실패를 입증하지 않는다.

## 원문 수치의 역할

다른 논문의 기존 segmentation benchmark 숫자는 출처 확인·구현 검토에는 쓰되, 환경과 지표가 달라 이 실험의 LIBERO 3D 수치와 순위를 매기지 않는다.

- [Grounding DINO](../../../papers/Grounding%20DINO%20Marrying%20DINO%20with%20Grounded%20Pre-Training%20for%20Open-Set%20Object%20Detection.pdf), 초록: COCO zero-shot **box AP 52.5**. 언어 조건 검출이다.
- [YOLOE](../../../papers/YOLOE%20Real-Time%20Seeing%20Anything.pdf), Table 2: LVIS val zero-shot **YOLOE-11-L mask AP 22.6 (text prompt)**. 검출 AP나 우리 3D IoU와 다르다.
- [TokenCut](../../../papers/Self-Supervised%20Transformers%20for%20Unsupervised%20Object%20Discovery%20using%20Normalized%20Cut.pdf), Table 1: VOC07/VOC12/COCO20K **CorLoc 68.8/72.1/58.8**, DINO ViT-S/16. 물체 아무거나 찾는 class-agnostic 지표이므로 지시된 target의 IoU와 다르다.
- [SAM 2](../../../papers/SAM%202%20Segment%20Anything%20in%20Images%20and%20Videos.pdf)와 [localization heads](../../../papers/Your%20Large%20Vision-Language%20Model%20Only%20Needs%20A%20Few%20Attention%20Heads%20For%20Visual%20Grounding.pdf)는 각각 video J&F·image segmentation과 referring-expression cIoU/Acc@0.5를 사용한다. Grounded-SAM-2는 Grounding DINO와 SAM 2를 조합한 별도 provider다.

원문 분할 성능의 구체적인 기준점은 다음과 같다. 이는 아래 LIBERO 실험과 데이터, 입력 및 지표가 달라 수치를 직접 비교하거나 같은 순위표에 넣을 수 없다.

| 원문 실험 | 검증 내용과 논문 수치 |
|---|---|
| SAM 2, Table 4 | 첫 영상 프레임에 주어진 정답 기반 프롬프트를 후속 프레임으로 전파하는 zero-shot 영상 객체 분할. 17개 비디오 데이터셋 평균 정확도는 1/3/5 클릭에서 64.7/75.3/77.6, 박스에서 74.4, 정답 마스크에서 79.3이다. 원문에 명시된 데이터셋별 지표로 평균했으며 LIBERO의 자유 언어 지시 분할이 아니다. |
| Localization heads, Table 2 | 지시 표현이 가리키는 대상을 영상에서 분할하는 RES. RefCOCO validation의 cIoU는 LLaVA-1.5-7B 74.2, LLaVA-1.5-13B 76.1이다. 논문에서 선택한 세 개의 localization head와 SAM을 사용하며, 본 연구의 π0.5 attention head 실험과 모델 및 head 선택 절차가 다르다. |

## 다음 실험 제안

1. **인식 분해:** target identity, target extent와 robot masking을 분리한다. 특히 같은 source에 대해 실제 배포 가능한 robot sphere/render mask, GT robot mask, mask 없음과 GT seed/attention seed를 교차 비교한다. GT mask는 특권 진단으로 따로 둔다.
2. **작은 원거리 target:** 접근 단계에서 GT seed가 있어도 `no_region`이 많은 원인을 top-5 patch의 target 점유율, support 평면 오인, graph node/edge와 0.5 threshold 각각으로 분해한다. threshold 변경은 dev에서만 하고 동일 고정 예산으로 segmenter도 조정한다.
3. **언어 구별:** 같은 프레임에서 지시된 명사나 관계 표현을 바꾸고 같은 종류 물체를 서로 구분하는지 검증한다. 단순 saliency 및 무작위 head를 대조하고 언어 변경이 실제 token에 반영됐는지 확인한다.
4. **독립 일반화:** Spatial bowl dev로 head를 선택했으므로 LIBERO Object 등의 다른 대상 물체에서 고정된 설정을 먼저 평가하고, 최종 성능은 init 5 이상 및 다른 suite의 사전 등록 test에서 검증한다. 관측 가능한 표면에 대한 주장과 가려진 물체 전체 3D 복원 주장을 구분한다.
5. **행동 연결:** 분할이 입증되면 동일 인식 아래 동일 제약을 사용해 거리장 오류·접촉·작업 성공을 별도로 측정한다. 현재 실험으로 접촉 감소를 주장하지 않는다.

## 재현성과 검증

분석 코드와 입력/환경 해시는 [초기 provenance](../../../outputs/segmentation_baseline_20261101/provenance.json), [최종 코드 및 결과 해시](../../../outputs/segmentation_baseline_20261101/final_provenance.json)와 각 실행 manifest에 기록했다. 기준 mask 오류 주입 재계산 200/200 일치, 저장 SAM2 mask의 기존 IoU 대조 3/3 일치, surface 전체 200-frame 재실행 IoU 200/200 bit-level 숫자 일치, dummy 카메라에 algorithm pixel 0, 공통 point 비교 arm마다 873개의 동일 frame을 확인했다. 기존 평가기의 관련 테스트 `test_metrics.py`, `test_lift.py`, `test_cloud_labeling.py`, `test_offline_grounding_speed.py`는 **33 passed**. 재현 이미지: [설명용 3개 task의 예](../../../outputs/segmentation_baseline_20261101/fixed_segmentation_examples.png). task 0·4·9 첫 frame의 설명용 사례이며 대표 표본으로 해석하지 않는다. 색상은 green=GT, red=prediction, yellow=overlap; target이 카메라에 보이지 않으면 2D IoU는 정의하지 않는다.

## 실험 D — 새로운 물체의 개발용 suite 이동 진단

**검증 질문:** 검은 bowl로 head를 고른 후보가 soup·cheese·sauce 등 다른 물체에도 target 영역을 복원하는가? 앞서 기록한 [Object 개발 규약](AG3S_OBJECT_SEGMENTATION_DEV_PROTOCOL_20261101_KO.md)에 따라 π0.5로 LIBERO Object task 0–9, init 0 각 1회, 총 10 episode/281 policy-call frame을 새로 수집했다. 정적 LIBERO Object이며 SafeLIBERO의 별도 장애물 조건이 아니다. 이 suite에서 BDDL 목적지는 `basket_1_contain_region`이라 scene class registry상 basket 본체는 destination이 아니라 other_object로 들어간다. 둘 다 이웃 FP에 포함하지만 세부 class 해석은 주의한다. 10개의 task와 한 번의 초기 상태뿐이므로 population 성능이나 테스트 통계로 사용하지 않는다. π0.5-LIBERO 자체는 LIBERO 데이터를 사용하므로 VLA의 unseen-object 일반화 평가가 아니다.

- 10개 episode의 281개 명사-query를 추출해 noun lookup **281/281 성공**. Spatial dev에서 고른 L8H6과 같은 RB-Y1 graph 설정을 Object에서 바꾸지 않았다.
- simulator image-instant state에서 head/wrist RGB를 각각 재생했을 때 **281/281 frame bit 동일**을 확인하고 그때 렌더한 target mask로 평가했다. 원 기록의 return-time target mask는 head 162/281, wrist 186/281 frame에서 image-instant mask와 적어도 1 pixel 다르다.
- 동일한 두 카메라의 depth와 촬영 시점 정답, 모든 frame으로 다음 예비 결과를 얻었다. [전체 원시 수치](../../../outputs/segmentation_baseline_20261101/object_dev/candidate_image_instant/summary.json).

| Object 후보 입력 | 3D soft IoU ↑ | target recall ↑ | 이웃 FP voxel mass ↓ | 특권 여부 |
|---|---:|---:|---:|---|
| image-time GT mask | 0.953 | 1.000 | 1.4 | simulator GT oracle |
| 후보: attention, robot mask 없음 | 0.189 | 0.434 | 167.3 | 배포 가능 입력에 가까운 2-camera **시험 adapter**, 정식 구현 아님 |
| 후보: attention + GT robot mask | 0.409 | 0.431 | 174.0 | **특권 진단** |
| 후보: GT target patch + GT robot mask | 0.742 | 0.783 | 1.5 | **특권 진단**, 완전한 GT mask oracle이 아님 |

후보는 독립 프레임의 `no_region`에서 빈 예측을 점수화했으며 실제 시스템 fallback·tracking·동작은 실행하지 않았다. 로봇 GT mask는 return-time segmentation이라는 시간 불일치도 남는다. 마지막 두 행은 배포 성능이 아니라, 같은 설정의 target cue 또는 robot mask를 바꿨을 때의 진단이다. GT target patch + GT robot 대 attention + GT robot의 차이는 IoU **+0.333** (10 task bootstrap 구간 **[+0.202, +0.467]**); attention + GT robot 대 attention + robot mask 없음은 **+0.221 [+0.141, +0.299]**. 구간은 과제당 1 episode인 탐색적 기술 통계다. 이 결과는 target 선택과 robot 영역 제거를 별도 모듈로 평가할 필요성을 보여주지만, 각 실패의 구체 원인은 추가 확인해야 한다.

YOLOE-11L의 공개 demo 기본 confidence 0.25, target phrase 설정은 281 frame 중 10 frame에서 적어도 한 camera에 비어 있지 않은 후보 mask를 내고, image-time GT의 3D IoU는 **0.031**이다. 결과를 본 뒤 baseline을 약하게 두지 않기 위한 탐색적 민감도 진단으로 confidence 0.05를 추가하니 비어 있지 않은 후보는 42/281 frame, 3D IoU는 **0.080**이었다. 각각 [기본 설정 결과](../../../outputs/segmentation_baseline_20261101/object_dev/yoloe_comparison/summary.json), [낮은 threshold 결과](../../../outputs/segmentation_baseline_20261101/object_dev/yoloe_conf005_comparison/summary.json)다. 이 수치는 provider의 top-1 규칙과 선택된 phrase·임계값에 의존하며, 최적 설정이나 YOLOE 원 논문의 성능이 아니다. 검출 수는 올바른 대상 검출 수가 아니라 비어 있지 않은 후보 출력 수다. 두 threshold의 task별 결과는 [표](../../../outputs/segmentation_baseline_20261101/object_dev/per_task_pilot.md)에 있다. 낮은 threshold의 평균 향상은 orange juice task의 상승이 큰 비중을 차지하므로 이를 모든 물체에서의 개선으로 해석하지 않는다.

**외부 모델과 같은 프레임 비교:** Grounded-SAM, Grounded-SAM-2, YOLOE와 TokenCut을 위 281개 프레임 전체에 실행하고 같은 image-instant GT로 다시 계산했다. 모델별 기록은 각 281개이며 provider 생성 및 추론 오류는 없었다. Phrase는 대상 문구, instruction은 전체 작업 지시문을 입력한 조건이다. 같은 기술통계 방식으로 얻은 결과는 다음과 같다([전체 표와 task bootstrap](../../../outputs/segmentation_baseline_20261101/object_dev/external_image_instant/summary.md), [JSON](../../../outputs/segmentation_baseline_20261101/object_dev/external_image_instant/summary.json)).

| 방법 | 3D soft IoU ↑ | target recall ↑ | 이웃 FP ↓ |
|---|---:|---:|---:|
| 이미지 시점 GT mask, 특권 oracle | 0.953 | 1.000 | 1.4 |
| Grounded-SAM-2, per-frame 대상 phrase | 0.205 | 0.279 | 281.3 |
| Surface 후보, robot mask 없음 | 0.189 | 0.434 | 167.3 |
| Localization heads + SAM point | 0.133 | 0.349 | 278.1 |
| Grounded-SAM-2, per-frame instruction | 0.082 | 0.224 | 1321.0 |
| Grounded-SAM-2, track instruction | 0.072 | 0.212 | 1597.8 |
| Grounded-SAM, instruction | 0.050 | 0.347 | 2178.6 |
| YOLOE, phrase, 기본 confidence | 0.031 | 0.032 | 0.0 |
| Localization heads, threshold | 0.031 | 0.896 | 1276.2 |
| TokenCut foreground | 0.031 | 0.612 | 1283.8 |
| TokenCut, attention overlap | 0.024 | 0.532 | 2047.5 |

Surface − SAM2 phrase의 IoU 차이는 **−0.017**, 10 task paired bootstrap 95% 구간 **[−0.141, +0.090]**이다. Surface − SAM2 instruction은 **+0.106 [+0.052, +0.163]**이고, SAM2 phrase − instruction은 **+0.123 [+0.019, +0.237]**이다. 후자의 차이는 이 개발 데이터에서 prompt 선택이 매우 중요함을 보여준다. [대상별 결과](../../../outputs/segmentation_baseline_20261101/object_dev/external_image_instant/per_task.md)에서 보듯 평균 뒤에 큰 task별 차이도 있다. 각 task에 단 한 episode인 데다 개발용 결과이므로 이 구간을 독립적인 최종 검증이나 방법의 일반적 우열로 해석하지 않는다. TokenCut은 지시 대상이 아니라 foreground를 발견하는 대조군이다. 모든 baseline 및 후보에 동등한 개발 예산으로 prompt·threshold를 정하고 이후 독립 test에서 고정해야 한다.

Action attention을 이용하는 localization heads 경로는 최초 추출이 스토리지 I/O에서 장시간 정지해 중단됐으나 재시도로 **281/281 frame**을 추출했다. Localization heads 두 조건과 TokenCut attention overlap도 각각 281 frame, provider 오류 없이 완료했다([동일 GT 재평가](../../../outputs/segmentation_baseline_20261101/object_dev/internal_image_instant/summary.json)). Localization heads에는 Spatial dev에서 고정한 π0.5 head를 사용했으므로 원문 LLaVA 모델의 재현 수치가 아닌 적용 실험이다. Threshold 조건은 recall이 높지만 이웃 FP도 크므로 recall만으로 분할 성공을 판단할 수 없다.
