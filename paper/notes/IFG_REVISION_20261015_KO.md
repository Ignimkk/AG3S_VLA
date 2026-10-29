# IFG 중간 결과와 novelty 검토를 반영한 논문 개정

작성: 2026-10-15. 이번 개정은 사용자가 지정한 두 실험 보고서와 `Novelty_contribution.pdf`를 확인한 뒤 실제 영문 원고에 반영한 작업이다. 연구 코드, serving 설정, 덤프 수집 작업은 수정하지 않았다.

## 현재 원고의 중심 문장

**Frozen VLA의 noun attention으로 대상의 의미를 정하고, 관측된 depth surface로 경계를 복원하며, 접촉 권한은 별도 규칙으로 제한한다. 내부 feature는 경계에 무조건 융합하지 않고 identity consistency에 적합한지 따로 검증한다.**

제목: **AG3S: Grounding Target Surfaces for Selective Contact in Vision-Language-Action Policies**

이전의 identity / extent / permission 구분은 유지했다. CVPR 독자가 판단할 수 있도록 관측 표면 표현, 구현 가능한 수식, 동일 조건 비교, 실패 원인 분석을 앞으로 옮겼다. 새로운 수학적 solver, 최초의 attention grounding, whole-body safety 자체를 novelty로 주장하지 않는다. 현재 기여는 특정 surface construction과 그 설계 근거이며, 이것만으로 학회 수준 novelty가 확정됐다는 의미는 아니다.

## 수정 전 / 수정 후

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 제목 | Where to Touch / Task-Conditioned 3D Contact Geometry | Grounding Target Surfaces / Selective Contact; 실제 시각적 산출물을 명시 |
| 초록 | 개선 방법이 완성됐다고 가정한 prospective draft; 실험 전부 예정 | 구현된 surface graph와 동일 mask 개발 결과 0.837→0.893을 제시; 검증 범위 명시 |
| Contribution | 경계 보존·시간 누적·접촉 통합의 예정 기능 중심 | 관측 surface construction / 신호별 역할을 분리한 controlled analysis / contact interface와 분리 평가 |
| Attention + Feature + 3D | feature 융합이 개선에 필요한 것처럼 읽힐 수 있음 | feature edge/unary/growing의 재검증 결과를 공개; primary extent는 Attention + 3D |
| Method | visibility weight, occlusion memory, persistent membership 등 미검증 개선 가정 | R6/R9 구현의 robot-depth mask, 5 mm node, depth-continuous edge, patch mass, support unary, 선형계, top-1 region |
| 시간 기억 | 지속적 surface membership 보존을 구현된 기능처럼 기술 | 실제 구현된 feature descriptor 기억으로 한정; chain의 과거 선택/0.1 switch 의존성 명시 |
| 정량 표 | 대부분 미측정 placeholder | 동일 mask 2×2 비교, feature ablation, supervised phase 표 |
| 그림 | 합성 시연과 예시 성능 막대가 포함된 prospective 이미지 | 편집 가능한 벡터 개념/구조도, 원자료 기반 결과 그래프, 실제 replay 관측 예시 |
| 안전성 | 통합 pipeline의 개선을 전제로 설명 | surface FP는 proxy; permission builder 및 closed-loop 결과와 구분 |
| 외부 baseline | KNOWS 등 일반 비교 계획 | 외부 segmenter+다중뷰/추적과 같은 downstream contact 조건을 사용; perception × contact handling 분리 |
| held-out | 광범위 unseen/occlusion 검증 계획 | 이미 잠근 설정, 새 episode, 성공 rollout 조건부 평가; 실패·가림은 별도 보고 |

기존 그림 원본은 삭제하지 않았다. 수정 전 원고/그림 wrapper/evidence는 작업 중 `/tmp/ag3s-paper-before-ifg-20261015/`에 보존했다. `/tmp` 보관은 영구 버전 관리 대체물이 아니다.

## PDF 내용의 채택·교정

| PDF 내용 | 판정과 반영 |
|---|---|
| 정확한 segmentation만으로 안전한 manipulation이 보장되지 않음 | 채택. 단, 다른 논문의 실패를 충돌 제약 탓으로 단정하지 않음 |
| RoboGround 예측 mask 64%, GT mask 68% | 원문 Table 3의 Easy success 열과 일치. 표는 grounding/success 쌍이며 이 숫자는 후자다. 이 비교가 AG3S의 충돌 감소를 입증하지 않으므로 본문에 공격적인 근거로 사용하지 않음 |
| AFI는 3D affordance를 이용한 VLA 개입 연구 | 채택, CVPR 2026 원문 확인. 개입 목적이 다름을 기술하고 실패율·실험 횟수를 우리의 상대 성능 근거로 가져오지 않음 |
| KNOWS가 내부 attention을 이미 사용 | 채택. 최초 attention 재사용 주장 제거. 기존의 외부 객체 geometry와 우리 observed-surface construction의 차이에 집중 |
| OBEYED-VLA에서 한 view의 grounding 오류가 문제 | 원문 v2의 limitation과 부합. 우리도 multi-view 모순을 모두 해결했다고 주장하지 않음 |
| GuidedVLA는 supervised attention specialization | 원문에서 확인. frozen policy의 inference-time surface construction과 구분 |
| 외부 모델 없이 segmentation을 한다는 것 자체가 충분한 novelty | 축소. 설계 차이이며 정확도·지연·안전 우위는 동일 조건 실험이 필요 |
| Attention과 feature를 반드시 함께 사용해야 하는 이유 | 현재 R7 결과와 불일치. 필수 융합 주장은 제외하고 feature의 역할을 재설정 |
| uncertainty-aware safety | 조건부. 현재 cosine/unknown은 calibrated risk가 아니며 confidence gate는 offline empty-mask 분석; 물리적 안전 보증으로 쓰지 않음 |
| link-specific / attached-object 제약이 독자적 novelty | 제외. WBAG/AutoMoMa 등과 중복; 우리 surface output의 소비 인터페이스로 기술 |
| GT mask는 성능 상한 | '진단용 oracle'로 변경. 고정 controller에서는 mask가 정확해도 성공률이 단조 증가한다는 보장이 없음 |
| EXAONE-VLA 등 기타 후보 | 이번 중심 논지에 직접 필요하지 않아 신규 인용하지 않음. 그 논문의 서지/상세 주장을 검증 완료했다고 표시하지 않음 |

검토한 primary sources:

- [RoboGround, 원문 Table 3](https://arxiv.org/html/2504.21530v1)
- [AFI, CVPR 2026 원문](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_Affordance_Field_Intervention_Enabling_VLAs_to_Escape_Memory_Traps_in_CVPR_2026_paper.pdf)
- [AFI, 저자·초록](https://arxiv.org/abs/2512.07472)
- [OBEYED-VLA v2](https://arxiv.org/html/2512.22519v2)
- [GuidedVLA](https://arxiv.org/abs/2605.12369)
- [KNOWS](https://arxiv.org/abs/2606.09749)
- [TokenCut, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Self-Supervised_Transformers_for_Unsupervised_Object_Discovery_Using_Normalized_Cut_CVPR_2022_paper.html)

RoboGround는 확인된 arXiv 서지로 인용했다. GuidedVLA RSS acceptance는 arXiv comments에서 확인했으며 현재 bib는 arXiv 레코드를 사용한다. PDF에 적힌 모든 논문별 지연·실험 숫자를 전수 재검증했다는 뜻은 아니다.

## 원자료 대조에서 확인한 중요한 차이

1. **동일 robot mask 비교:** 0.893−0.837=약 0.057. 기존 mask와 최종 후보를 직접 비교한 약 +0.108을 'extent 방법만의 효과'로 쓰지 않는다. 이 +0.057도 voxel 크기·seed·graph·선택 규칙 전체의 차이이며 solver 하나의 효과는 아니다.
2. **Feature 경보의 분모:** 요약의 0.998은 147프레임, 0.914는 179프레임이다. 저장된 per-frame score를 같은 147프레임으로 다시 집계하면 0.997636 대 0.899527이다. 본문은 반올림해 **0.998 대 0.900**을 사용한다. 오류는 6개뿐이다.
3. **보류 효과:** 15/147 frame 보류에 따른 FP mass 제거 81.6%, TP mass 손실 6.0%다. 충돌 감소율이나 safe success가 아니다.
4. **Phase 정확도:** 0.878→0.939는 pooled frame accuracy. 별도 episode-mean paired gain은 +0.067이다. 같은 집계인 것처럼 숫자를 빼서 설명하지 않는다. 기존 HSR probe 학습과 upstream dev 설계까지 포함한 nested held-out 검증은 아니다.
5. **Precision 결측:** empty prediction의 precision은 정의되지 않아 평균에서 제외된다. Table 1 행별 제외 수는 5/1/3/1/3; IoU와 recall은 180프레임 모두를 사용한다. 본문과 caption에 명시했다.
6. **Memory chain의 실제 의존성:** 직전 R6 선택만 단순 저장하는 구조가 아니다. `Mem-chain:...:d0.1` 전환이 발생하면 그 선택의 descriptor로 다음 기억을 갱신한다. '순수한 동일 mask + 무상태 confidence'로 기술하지 않았다. 후속 collector/evaluator는 이 의존성도 재현해야 한다.
7. **현재 장면:** 32개 source meta 모두 `rby1_randomized_pick_place_16d`, `obstacle_profile=clear`다. 타깃은 apple/banana/orange/pear이고 양팔 사례가 포함된다. 새 장애물·가림 robustness로 주장하지 않는다.
8. **연속 시간·가림:** sparse key frame의 기억 실험을 continuous tracking/occlusion recovery로 일반화하지 않는다.

## Evidence 등록과 재현

- 등록: [EXP002 IFG manifest](../experiments/manifests/ifg/EXP002_IFG_development.json)
- 상세 audit: [EXP002_IFG_audit.json](../experiments/summaries/EXP002_IFG_audit.json)
- 검증: `python3 paper/tools/analysis/audit_ifg_evidence.py`
- 원자료: `/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba`

180개 dump의 실제 raw state (16), action (50×16), 완료·S0 정합 기록, 32 episode, 실행 소스 hash를 확인했다. R9의 표 값 및 paired bootstrap은 per-frame rows에서 다시 집계했다. R11b의 confidence 비교도 동일 프레임으로 재집계했다. R7/R12/rootcause는 원본 summary와 계산 코드를 대조하고 hash로 연결했다. 모델 추론·시뮬레이터·rollout을 다시 실행하지 않았다.

기존 EXP001 validator는 T39의 `trajectory.npz`와 run-check 구조에 묶여 있어 IFG의 per-request NPZ와 형식이 다르다. 기존 검사를 완화하지 않고 IFG용 명시적 manifest 종류와 전용 validator를 추가했다. 이번 허용 범위는 **OFFLINE_DEVELOPMENT**뿐이다. 기존 T39의 사라진 `benchmark/ag3s/...` source path 문제는 별도 이전 문제이며 이번 결과의 출처로 쓰지 않는다.

## 제출 전 필요한 증거

- 잠근 extent 설정의 독립 episode 평가. 성공 rollout 조건부 평가라는 모집단을 명시.
- 같은 147프레임 comparator와 동일한 memory update를 재현하는 confidence 평가; 오류 수 확충.
- 외부 segmenter+다중뷰/추적을 같은 contact interface에 연결한 강한 baseline.
- 새 surface output을 실제 permission builder에 연결한 false-exemption 측정과 모든 시도 episode를 포함한 closed-loop 평가.
- 실제 센서 depth, end-to-end latency, 관측되지 않은 표면과 물체 부착 처리 검증.

[CVPR 2026 공식 지침](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)의 본문 8페이지+참고문헌 기준을 임시 적용한다. 2027 Author Guidelines는 이번 접근에서도 내용을 확보하지 못했으므로 2027 확정 규정으로 표시하지 않는다. Official style·margin·본문 font는 변경하지 않았다. 최종 논문 준비 완료나 full submission audit 완료를 의미하지 않는다.
