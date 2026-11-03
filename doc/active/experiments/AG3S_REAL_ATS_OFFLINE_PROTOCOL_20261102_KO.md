# 실제 ATS(Panda 최종 구성) 대 외부 segmenter: offline 인식 비교 규약

작성: 2026-11-02 KST, 결과를 보기 전. [외부 segmenter → ATS 대체 비교](AG3S_SEGMENTER_ATS_STAGED_COMPARISON_20261101_KO.md)의 **2단계 중 인식만 offline** 비교다. 행동·접촉·작업 성공은 측정하지 않는다.

## 무엇을 비교하는가

기존 Object init 5·6 검증 표본(10 task × init 5·6 = 20 episode, **552 policy-call frame**, head/wrist RGB-D, 이미지 시점 GT, 재렌더 RGB 일치 552/552)에서 이미 실행한 외부 baseline 결과와, **실제 Panda 최종 구성 ATS**의 결과를 같은 frame key·같은 GT·같은 3D 지표로 비교한다. 외부 baseline은 새로 돌리지 않고 저장된 결과를 재사용한다.

| 항목 | 고정값 |
|---|---|
| ATS | `AG3S-final-panda` 서버 flag(`profile.version_server_flags`)로 만든 AG3S. `--attention-source noun_target:L8H6`, `--target-exclusion ball-owned`, `--attention-split-partition`, `--split-supersedes-anchor`, `--nontarget-evidence surface` 등 최종 구성 그대로. Panda에서 head를 다시 고른 결과이며(SL6-B, Spatial 개발 404 요청, L8H6 0.659 대 RB-Y1에서 옮긴 L10H7 0.634) **Object 결과를 보고 바꾸지 않는다** |
| attention 입력 | 같은 policy 요청에서 저장한 noun query row(`noun_rows`)의 layer 8 head 6. 서버가 읽는 slice와 같은 함수로 추출된 값이다 |
| 입력 | 저장된 head/wrist RGB-D, calibration, 이미지 시점 로봇 관절. GT 위치·mask·객체명은 AG3S 호출에 들어가지 않는다 |
| 상태 | 에피소드 시작에서 `AG3S.reset()`, 이후 frame을 obs_index 순서로 처리해 시간 상태(추적 등)를 이어 간다. grasp·latch 신호는 no-grasp 값이므로 **쥔 뒤 identity 고정(`latched-identity-hold`)은 이 실험에서 작동하지 않는다** |
| 외부 baseline(재사용) | Grounded-SAM-2 대상 phrase, Grounded-SAM-2 전체 instruction, YOLOE confidence 0.25, YOLOE 0.05(사후 탐색 조건), 이전 연구 adapter `noun L8H6` + FK sphere |
| 사용하지 않음 | 특권 GT 조건은 참고행으로만 표시한다. 실제 ESDF·제약·TrajOpt는 평가하지 않는다 |

## 지표 (이전 비교와 동일)

1. 주 지표: ATS가 반환한 **target 점 집합**(`grounding.target.points`)을 5 mm voxel로 바꿔 이미지 시점 GT와 비교한 3D soft IoU, recall, precision, 이웃 물체 FP. 외부 mask와 같은 `voxel_metrics`를 쓴다.
2. 선택 실패율: target을 반환하지 않으면(abstain) 빈 예측으로 0점이며 별도로 센다.
3. identity: 예측 voxel의 precision이 0.5 이상이면 "대상 물체 선택"으로 센다(2D mask 불필요). 이 기준은 결과 전에 정했고 임계값을 바꾸지 않는다.
4. 집계: frame → episode → task → task 평균, 10 task 동일 가중. paired task bootstrap 구간은 탐색용이다.
5. 비용: frame당 wall time(첫 호출 제외). GPU를 다른 job과 공유하므로 **시간은 참고값이며 비교·주장하지 않는다**.

2D IoU는 ATS가 pixel mask가 아닌 3D 점을 반환하므로 이 실험의 주 지표가 아니다. 점→pixel 변환 오차가 들어가므로 추가하지 않는다.

## 해석의 한계(결과 전 기록)

- 이 표본은 이미 외부 baseline과 이전 adapter의 결과를 본 검증 표본이다. 실제 ATS는 이번이 첫 실행이며 head·flag를 이 표본으로 고르지 않는다. 그래도 **확증 test가 아니다.** 10 task × 2 init 외부 일반화를 주장하지 않는다.
- 정책이 π0.5-LIBERO이고 LIBERO 데이터로 학습했으므로 unseen-object 일반화가 아니다.
- 이전 adapter는 attention surface graph이고, 실제 최종 구성은 ball-owned 추출을 쓰므로 두 ATS 수치는 서로 다른 방법이다. 외부 baseline과의 차이를 모듈별(ATS 세부) 기여로 해석하지 않는다.
- no-grasp 상태 가정 때문에 쥔 뒤 프레임에서는 실제 서버보다 identity가 흔들릴 수 있다. 쥔 뒤(접촉 frame) 구간을 따로 보고한다.
- 명령의 명사가 대부분 서로 다른 물체이므로(Object suite) Spatial의 "같은 bowl 둘" 문제는 이 표본에서 나타나지 않는다. Spatial 결과로 일반화하지 않는다.
- 개발 smoke(2 episode × 3 frame)는 연결 확인용이며 결과에 넣지 않는다.

## 수정 1 (2026-11-02, 연결 확인 smoke 2 episode × 3 frame을 본 뒤, 전체 552 frame 실행 전)

smoke는 파이프라인이 동작함을 보였고, `grounding.target.points`가 약 50개(GT 약 450–520 voxel)로 작다는 것도 보였다. 최종 구성에서 거리장에서 실제로 지워지는 범위는 이 seed cluster가 아니라 **공 + ball-owned 분할이 지운 관측 표면 voxel**(`field.exclusion_trace['removed_ijk']`)이다. 그래서 target 점만 채점하면 서버가 실제로 쓰는 ATS 출력을 과소평가한다. 전체 실행 전에 다음을 확정한다.

- **보조 채점 추가:** 지워진 voxel(extent)을 같은 `voxel_metrics`로 채점한다. 비교표에는 두 행을 모두 싣는다. `ATS target cluster`(원 규약의 주 지표)와 `ATS 지워진 범위`.
- 어느 쪽을 "주"로 부를지 결과를 보고 고르지 않는다. 외부 mask와의 직접 비교는 두 행 모두 보고하고, 서버가 target으로 취급하는 범위는 지워진 범위라는 점을 함께 적는다.
- smoke의 6 frame 결과는 연결 확인용이다. 통계에 넣지 않는다.

## 수정 2 (2026-11-02, 전체 552 frame 결과를 본 뒤 추가한 층화)

전체 결과와 장면 그림에서 물체가 들린 뒤 ATS가 원래 책상 위 위치에 남는 것을 보고, GT 대상 중심이 첫 평가 frame에서 20 mm 이상 움직였는지로 frame을 나눈 층화를 **결과를 본 뒤** 추가했다(임계값은 층화 계산 전에 고정). 탐색적 층화이며 규약의 주 지표는 전체 552 frame 결과다.

## 결과 요약 (실행 완료 2026-11-02 10:17 KST)

원시 기록과 그림, 표: `outputs/2026-11-02_실제ATS_vs_외부segmenter_offline_ObjectInit56/` (README.md 참고), storage run `20261102T101200_96fda38abd1b_segmentation-real-ats-object-init56-20261102`.

- 전체 552 frame, 3D soft IoU: 실제 ATS target cluster 0.101, 지워진 범위 0.088, Grounded-SAM-2 phrase 0.207, 이전 adapter 0.318. 실행 오류와 abstain 없음(552/552 target 반환).
- 대상이 움직이기 전 117 frame: ATS 선택률 0.925(SAM2 phrase 0.128), IoU 0.217(SAM2 0.098). 움직인 뒤 435 frame: ATS 0.068(선택 0.233).
- 해석 한계: 재생이 grasp · latch를 고정했으므로 움직인 뒤 구간은 실제 서버 성능을 대표하지 않는다. 그 구간이 실제 서버에서 어떻게 되는지는 grasp 신호를 공급하는 별도 실험이 필요하다.
