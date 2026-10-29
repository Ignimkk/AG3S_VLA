# 개선 완료를 가정한 논문 방향 검토용 초안

작성일: 2026-10-13. 사용자가 요청한 **개선 작업이 완료되었다는 가정**에 따라 작성했다. 방법 절의 현재형은 완성형 연구 설계의 서술이며, 현재 serving 코드의 구현 완료나 성능을 인증하지 않는다. 원고 첫 페이지에도 이 상태를 표시했다. 기존 실험·claim registry의 완료 상태는 변경하지 않았다.

- 원고: [main.tex](../latex/main.tex)
- 검토용 PDF: [main.pdf](../latex/build/main.pdf)
- 기준 계획: [develope_PLAN.md](../../doc/active/plans/develope_PLAN.md), 특히 0.10 및 10.4–10.6절

## 연구 중심과 기여의 배치

제목: **AG3S: Grounding Where to Touch with Task-Conditioned 3D Contact Geometry**.

핵심 질문은 “타깃이 어디인가?”를 넘어 **“관측한 표면 중 어디까지 해당 타깃이라고 인정하고, 어느 링크에 언제 접촉을 허용할 것인가?”**이다. Identity / extent / permission을 서론·수식·실험에서 일관되게 구분했다. Where to Touch는 접촉 허용 영역을 뜻하며 최적 grasp point, 힘 제어, 학습된 affordance를 뜻하지 않는다.

1. 중심 방법 기여: 타깃 identity와 surface membership을 분리하고, 다중 시점·시간 증거로 경계를 수정하면서 인접 장애물을 보존하는 3D grounding.
2. 연결 기여: 관측 extent를 링크별 field와 attachment 전환, 실제 실행 prefix의 검사까지 연결.
3. 실증 기여 자리: identity/extent/permission의 효과를 분리하는 paired evaluation. 아직 결과가 없으므로 완료된 발견 대신 평가 설계와 향후 입증할 내용을 서술했다. 최종 논문에서는 가장 강한 실험 발견으로 교체한다.

Attention target selection, allowed collision, attached object, ESDF, SQP를 각각 최초 발명이라고 주장하지 않는다. HSR은 강한 rule baseline보다 downstream 효과가 확인될 때 독립 기여로 승격한다. 충돌 sphere coverage 보완은 필요한 기반 수정이며 grounding의 novelty로 전용하지 않는다.

## 이번 원고에서 완료를 가정한 설계

다음은 구현 현황표가 아니다. 이전에 토론한 개선 방향을 재현 가능한 방법으로 구체화한 **목표 명세**다. 특히 가시성 결합식, membership predicate, stale geometry 처리와 엄격한 gate는 최종 코드와 동등성 검사가 필요하다.

| 설계 | 원고 반영 | 최종 확인해야 할 사항 |
|---|---|---|
| Visibility-aware multi-view fusion | 유효 depth·투영·가려짐 검사로 view별 가중치 결정. Head occlusion 시 유효 wrist evidence 이용 | 가중치/정규화/동기화, 완전 가림에서 abstention |
| Identity와 extent 분리 | 타깃 identity 유지 중에도 voxel 소속 수정·복구 가능 | 초기 target–post merge를 memory가 고착시키지 않는지 |
| Boundary-preserving exclusion | Ball은 탐색 상한. Target support·neighbor distance·validity 조건을 만족한 seed만 제외 | rho/tau/age 등 개발 split에서 고정; neighbor label 오류에 대한 실험 |
| Occlusion memory | 최근 확실한 표면만 제한된 기간 유지; 모순되거나 오래된 표면은 exemption 철회 | 사라진 물체/움직이는 물체에서 ghost geometry와 과도한 HOLD |
| Phase / physical state 분리 | HSR은 identity update, grasp feedback은 attach/release | 독립 phase label, strong rule 비교, slip 검출 |
| Geometry lifecycle | Attach 시 소유가 확인된 환경 geometry만 제거; release 시 보관 geometry 즉시 복귀 | 자기 물체 중복 검사와 주변 장애물 제거가 없는지 |
| Hand/arm collision coverage | Palm/fingers/wrist와 과제 관련 arm link의 sphere model audit | 실제 mesh 대비 자세별 coverage, 평가 link list, attached-object coverage |
| Executed-prefix gate | Margin이 이미 포함된 residual >= 0, interpolation 검사, 유효하지 않으면 HOLD | 기존 collision-only fallback과 혼동 금지. 새 명세를 코드로 맞춰야 함 |
| Completion endpoints | P=release+stable basket residence, R=P 이후 준비자세 허용영역 안정 진입 | 기존 단일 placed flag나 HSR home을 완료 GT로 사용하지 않음 |

기존 진단에서 양의 clearance residual은 이미 margin을 뺀 값일 수 있다. 이번 원고는 `h = distance - radius - margin`을 명시했으며 “4.7 mm < 10 mm이므로 gate 위반” 같은 중복 margin 해석을 하지 않는다. 새 엄격한 gate 명세와 현재 구현의 fallback 여부는 별도 사안이다.

위 표의 threshold와 최종 head/query 선택, probe split, 정확한 query-link 목록을 확정하기 전에는 이 문서를 재현성 명세의 최종본으로 보지 않는다. 과거 검증되지 않은 completion 선언을 만들지 않기 위해 연구 코드와 evidence registry를 수정하지 않았다.

## Figure 1–3 사용과 남은 그림 문구 정리

사용자가 지정한 PNG는 수정하지 않았다. LaTeX에서 배치·crop하고 출처 해시를 figure sidecar에 기록했다.

- Figure 1: `variant_A_performance.png`의 두 작업 × OFF/ON 장면을 teaser로 사용. 예시 숫자 performance 열과 하단의 중복 방법 strip은 제외했다. 그 열의 `no extra training` 문구도 함께 제외된다. 원본 장면은 검증된 paired 성공 사례로 인증하지 않고 의도한 동작의 시각화로 명시했다. 최종 제출에서는 scene/run/step/contact evidence가 있는 장면으로 확정해야 한다.
- Figure 2: `reference_montage_v2.png`를 개념도로 사용. “one frozen VLA decides ... by which link”라는 제목 줄은 crop했다. 현재 설계에서 link permissions는 supplied configuration이다. `Prior`는 모든 선행연구를 지칭하지 않는다는 점, RGB와 depth의 실제 입력 경로, C1–C3가 세 독립 novelty라는 뜻이 아니라는 점을 caption에서 명시했다.
- Figure 3: `reference_montage.png`를 방법 pipeline으로 사용. 별도 extraction pass, target boundary, attachment와 executed prefix를 설명한다. 아래 긴 home timeline과 경로는 성능 결과가 아닌 설명 도식이며 종료 이후 추론 지속을 평가 프로토콜로 채택하지 않는다고 명시했다.

후속 그림 개정 시 Figure 2/3의 `RGB-D → VLA` 도식을 RGB→VLA / depth→geometry로 분리하고, Figure 3에 visibility weighting·membership correction·실제 종료 marker를 반영하면 본문과 더 직접적으로 일치한다. Phase curve, CCR path, 성공/실패 표시는 실제 로그 출처를 연결한 뒤 empirical figure로 승격한다. 이번 작업은 새 그림 생성 요청이 아니므로 PNG를 다시 그리지 않았다.

## 실험을 채우는 순서

1. Snapshot·sphere coverage·독립 contact label·종료 평가기의 신뢰성을 먼저 고정.
2. 동일 identity에서 ball/cluster/proposed/oracle extent 비교, identity×extent oracle 진단.
3. 동일 geometry와 controller에서 no/global/link-specific exemption 및 attachment 비교.
4. Head/wrist occlusion·인접 장애물 간격·depth noise와 memory recovery 검증. 기존 실패 scene을 regression으로 유지하고 별도 held-out set 확보.
5. SafeLIBERO 주 비교, 자체 MuJoCo ablation, RB-Y1 실물 feasibility의 역할 유지.
6. HSR strong-rule 비교와 prefix/gate 검증, 추가 extraction pass를 포함한 전체 runtime.

Primary outcome은 release 후 안정 placement 종료를 사용한다. 준비자세 복귀까지 보는 결과는 별도 protocol로 보고한다. 확인 interval이 끝난 순간까지의 observation/contact는 보존하며 이후 inference는 중단한다. 완료하지 못한 episode를 유리하게 잘라내지 않는다. 동일 실행이 R까지 이어졌다면 P-prefix 분석도 병기할 수 있으나 P 조건의 실제 조기종료 실행과 혼동하지 않는다.

실험 부분은 전부 미래형·계획형이다. 샘플 수, trial seed 목록, CI, 실제 수치, 최종 baseline parity는 수행 전에 고정하고 이후 채운다. 예전 파일의 누적 success나 T58 pilot을 이번 개선판의 성능으로 옮기지 않았다.

## 문서 범위

이번 개정은 main manuscript, Figure 1–3의 LaTeX wrapper/출처 sidecar, 이 설명 문서에 한정한다. Supplementary는 이전 scaffold를 유지하므로 이번 완성형 method의 appendix로 제출하면 안 된다. 참고문헌은 기존 원고의 key와 이전 primary-source 검토 기록을 재사용했고 새 포괄적 선행연구 조사를 수행하지 않았다. 공식 2027 kit가 아닌 저장소의 pinned provisional CVPR kit를 그대로 사용했다.

빌드 및 PDF 검토 결과는 아래에 기록한다.

## 빌드 및 검토 결과

- Tectonic 0.17.0의 cached resources로 main PDF 컴파일 성공. 총 9쪽, 본문은 8쪽 내 종료하고 참고문헌은 8–9쪽에 배치된다.
- 미정의 인용·참조, `??`, overfull/underfull box 없음. Figure 1/2/3은 각각 2/4/6쪽에 정상 배치되며 PDF 1/2/4/6/8쪽을 이미지로 확인했다.
- Figure/table/algorithm sidecar의 렌더 소스·입력 SHA-256 검사 통과. 지정 PNG 원본은 변경하지 않았다.
- 공식 kit와 working style 해시 일치, 수정 파일의 `git diff --check` 통과.
- 저장소 전체 `paper/tools/latex/build.py` 검사는 기존 EXP001 manifest의 `benchmark/ag3s/docs/handoff/T39.verify.json` 경로 부재로 중단된다. 이번 원고의 LaTeX 오류가 아니며 과거 증거 파일을 이동하거나 manifest를 재등록하지 않았다. 직접 compiler 호출로 PDF 빌드를 별도로 확인했다.
- 패키지 `lineno.sty`, `algorithm.sty`의 cached source에 대한 UTF-8 경고는 남아 있다. 원고의 unresolved citation 또는 layout warning은 없다.
- 검증 메타데이터: [COMPLETED_METHOD_DRAFT_20261013_BUILD.json](COMPLETED_METHOD_DRAFT_20261013_BUILD.json).
