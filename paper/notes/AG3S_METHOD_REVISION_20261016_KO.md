# AG3S 전체 방법론을 유지한 1–3장 개정

사용자 최신 지시: 개선 내용만으로 논문을 대체하지 않고 기존 방법론에 개선을 통합한다. 이번 작성 범위는 Introduction, Related Work, Method이며 실험·결과는 후속 작성한다.

## 핵심 메시지

**VLA가 의도한 대상을 찾는 데서 끝나지 않고, 관측된 표면을 어느 링크가 언제 접촉할 수 있는지 명시하는 3D 제약으로 바꾸고, 그 제약 아래에서 action chunk를 보정한다.**

Identity(어떤 물체), extent(관측 표면의 범위), permission(누가 언제 접촉 가능한가)을 구분한다. ATS–HSR–LTE–CCR 전체 흐름을 유지한다. 네 모듈 각각을 독립적인 최초 발명으로 주장하지 않는다.

## 장별 내용

1. **Introduction:** 사과를 잡을 때 손가락은 닿아야 하지만 손목은 피해야 하며, 운반 중에는 사과 자체도 보호해야 한다는 문제로 시작한다. 정확한 대상 식별만으로 접촉 영역을 정할 수 없다는 논리에서 표면 grounding, 시간축 identity, 접촉 제약 연결이라는 세 기여를 제시한다. 결과 없는 성능 주장은 삭제했다.
2. **Related Work:** VLA grounding, attention/feature graph, 기하 기반 실행 보정, 내부 표현의 작업 상태 추론을 비교한다. KNOWS의 attention 활용, TokenCut의 feature graph, MoManipVLA의 기하 보정, WBAG의 attached geometry를 인정한다. 차별점은 이 요소 자체의 최초성이 아니라 관측된 target extent를 링크·상태별 contact geometry로 구성하는 방법에 둔다.
3. **Method:** 전체 계약 → ATS 표면 → ATS temporal memory → HSR/physical state → LTE → CCR 순서다. 장면에서 제약과 실행까지 끊기지 않게 정의했다.

## 기존 방법과 개선의 결합

| 기존 축 | 유지한 기능 | 통합한 개선 |
|---|---|---|
| ATS | 내부 attention으로 target을 찾고 depth로 3D화 | robot-depth 표면 일치 mask, depth-continuous graph, patch mass seed, support negative, region membership |
| ATS identity | 대상의 지속성과 갱신 | centered feature first-reference switch(B), 별도 causal chain consistency |
| HSR | pick/place/home 및 contact lifetime | gripper + target–hand distance + height + region size; physical attachment와 phase 분리 |
| LTE | contact link에만 target exemption, carry/release | ball 전체가 아닌 supported surface만 제외, competing/support veto, unknown 보존, attachment/release ownership |
| CCR | tracking/smoothness/continuity와 제약 최적화 | 실행 feedback으로 overlap 정렬, nonlinear prefix check, 실패 시 HOLD; 통합 검증 필요 상태 명시 |

## 구현과 통합 설계의 경계

- `FINAL_REPORT_KO.md`, `METHODOLOGY_EVOLUTION_KO.md` 및 이전 AG3S 원고/개발 계획을 결합했다. 원문 보고서의 표현을 그대로 성능 주장으로 옮기지 않았다.
- 새 surface/memory/state 구성은 offline development 결과가 있다. 새 surface를 LTE 및 전체 closed-loop에 연결한 결과는 아직 없다. Method 첫 부분의 **Scope of this method draft**에 이를 집중해 명시했다.
- B의 first-reference switch와 confidence chain은 서로 다른 memory다. Offline B switch에는 HSR phase gate가 없으므로 phase-gated integration을 기존 실험으로 가장하지 않았다.
- Confidence 실험은 base selection을 평가한다. 전환된 영역의 validity를 자동 승계하지 않고 재검사해야 한다는 통합 요건을 추가했다. 결합 threshold/calibration은 미완료이며 현 component 성능으로 보증하지 않았다.
- Feature memory는 identity 정보이며 occluded surface completion 또는 오래된 공간의 접촉 허용을 제공하지 않는다.
- Support-relative height는 현재 floor median 기준이라는 구현 사실을 명시했다. Table-relative 일반화를 주장하지 않는다.
- Grasp confirmation, surface ownership, strict prefix check는 완성형 통합 설계의 요구사항이다. 모든 기능이 현재 serving에 검증 완료됐다는 뜻이 아니다.
- Confidence threshold와 permit/attachment 임계값을 포함한 최종 통합 설정, query coverage, 지연, held-out 및 closed-loop 결과는 평가 revision에서 확정해야 한다.
- Related Work는 기존 검증 참고문헌을 재사용했다. 이번 편집으로 포괄적인 신규 선행연구 조사나 최초성 검증을 완료한 것은 아니다.

## 원고와 검증 범위

- 활성 PDF: `paper/latex/build/main.pdf`. 초록, 1–3장, 개념도 2개, 참고문헌만 포함한다.
- `04_experiments.tex`, `05_conclusion.tex`, `06_limitations.tex`, 실험 figure/부록 source는 삭제하지 않고 활성 main에서 제외했다. 이번 지시 이후 추가 실험·결과 수정은 하지 않았다.
- 실험 그림은 현 단계에서 Method의 실증 그림처럼 사용하지 않았다. Figure 1은 identity/extent/permission, Figure 2는 전체 ATS–HSR–LTE–CCR 구조를 설명한다.
- 현재 재현 명령: `python3 paper/tools/latex/build_method.py`.
- 이 검사는 활성 원고의 범위, 공식 style hash, 개념도 출처 hash, 수식/인용 참조, overfull box, PDF 페이지 수를 확인한다. 실험 evidence 검증이나 최종 CVPR 제출 적합성 검사를 대신하지 않는다.
- `build_ifg.py`는 이전 전체 IFG 실험 원고용이다. 현재 1–3장 전용 원고에는 결과 수치를 요구하는 해당 명령을 사용하지 않는다. 기존 EXP001 validator의 경로 문제도 해결했다고 주장하지 않는다.
- 기존 CVPR template/font/margins와 연구·serving 코드는 변경하지 않았다.
