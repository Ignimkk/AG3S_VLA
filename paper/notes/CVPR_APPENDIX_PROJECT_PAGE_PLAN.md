# CVPR 2027 — Appendix와 프로젝트 페이지 계획

작성·부분 규정 확인: 2026-10-07 19:18 KST  
상태: `PLANNED`  
상위 계획: [develope_PLAN.md](../../develope_PLAN.md), 특히 PAPER-FIRST 0절  
목적: 핵심 기여의 시각적 설명을 강화하고, 같은 실험 자료로 본문·부록·영상·웹페이지를 완성한다.

## 1. 확인된 규정과 적용 범위

이번 확인은 appendix/supplementary, 프로젝트 웹사이트, 학회 scope에 한정한다. 실제 제출 PDF·영상·사이트의 compliance를 판정한 것은 아니다.

| 항목 | 상태 | 확인 결과 / 계획에 적용할 내용 | 출처 |
|---|---|---|---|
| CVPR 2027 scope | `CONFIRMED_2027` | CFP 공개. Embodied vision and robotics, Physical AI/VLA, 3D geometry 포함 | [2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) |
| 2027 세부 Author Guidelines | `UNKNOWN` | 확인한 URL의 내용을 확보하지 못함. 세부 형식이 2026과 같다고 단정하지 않음 | [확인 대상](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines) |
| Main/appendix 분리 | `PROVISIONAL_2026` | 본문 8페이지, 참고문헌 별도. Appendix는 별도 supplementary PDF로 준비 | [공식 kit 본문 안내](https://raw.githubusercontent.com/cvpr-org/author-kit/main/sec/1_intro.tex), [supplement 분리 안내](https://raw.githubusercontent.com/cvpr-org/author-kit/main/sec/X_suppl.tex) |
| 부록 PDF | `PROVISIONAL_2026` | 추가 그림·표·분석 허용. 정돈된 읽기 쉬운 형식; CVPR 스타일 사용 가능 | [2026 Author Guidelines / FAQ](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) |
| 부록의 지위 | `PROVISIONAL_2026` | 검토 의무 없음. 추가 tuning/training으로 개선한 방법의 결과나 수정된 본문 PDF는 넣지 않음 | [2026 Reviewer Guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines) |
| 공개 사이트 | `PROVISIONAL_2026` | arXiv 논문 관련 project website 허용. 공개 사이트에 CVPR 투고 중/예정 표시하지 않음 | [2026 Author FAQ](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) |
| 심사용 외부 링크 | `PROVISIONAL_2026` | 추가 기여 내용을 확장하는 외부 영상·페이지 링크 불허. 익명 URL도 예외가 아님 | [2026 Author FAQ](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) |
| 2027 부록 용량·파일 형식·영상 제한 | `UNKNOWN` | 임의의 허용 용량·분량을 약속하지 않고 공식 안내/제출 시스템에서 재확인 | [기존 supplementary 정책](../guidelines/CVPR_SUPPLEMENTARY_POLICY.md) |

**현재 판단:** 2026 기준으로 appendix를 작성하며 준비를 진행할 수 있다. 2027 최종 제출 형식의 확정은 아니다. 기존 [supplementary scaffold](../supplementary/supplementary.tex)를 사용하고 본문에서는 Appendix A/B 등의 위치를 참조한다. 공개 preprint에서 본문과 부록을 합치는 것은 별도 배포 구성으로 처리한다.

## 2. CVPR 독자에게 무엇을 보여줄 것인가

다음은 공식 채택 기준의 해석이나 채택 보장이 아니라, 현재 구현과 CFP를 바탕으로 한 집필 전략이다.

> Frozen VLA에서 읽은 시각적 대상 정보를 관측 depth와 결합해 작업 대상의 3D 기하를 얻고, 이를 작업 단계·로봇 링크에 맞는 접촉 제약으로 바꾼다.

Abstract와 Introduction에서 외부 segmentation model 없이 이 연결을 만든다는 점을 선명하게 한다. 기존 VLA는 collision avoidance를 위해 추가 학습하지 않는다고 설명하고, HSR probe의 학습과 추가 attention pass는 method에 정확히 구분한다.

시각 인식과 제약 구성의 연결이 본문에서 보이도록 다음을 우선한다.

1. **대상 선택:** 같은 장면에서 지시 대상이 바뀌었을 때 attention과 target geometry가 어떻게 바뀌는가.
2. **기하적 구체화:** Heatmap의 위치뿐 아니라 target의 3D 범위가 올바른 접촉 예외를 만드는가.
3. **행동상의 효과:** 동일 controller에서 이 정보가 어떤 잘못된 회피 또는 충돌을 줄이는가.

이것은 P0-B의 작은 메커니즘 비교와 Figure 3에 통합한다. 새 대규모 vision benchmark를 기본 과제로 추가하지 않는다. Controller 세부는 필요한 만큼 본문에 남기고 수치 solver·상태 전이의 전체 명세는 부록으로 이동한다. 핵심 novelty나 핵심 ablation을 부록에만 두지 않는다.

## 3. Appendix 구성안

아래 목차는 작성 슬롯이다. 모든 항목을 채우기 위해 새 실험을 만들 필요는 없다. 실제 주장과 확보한 자료에 따라 합치거나 줄인다. A–F는 문서 구조 제안이며 공식 요구 형식이 아니다.

| 절 | 내용 | 필요한 Figure / Table | 우선순위 |
|---|---|---|---|
| A. Experimental protocol | Task·scene·split·seed·success/contact 정의, baseline의 입력과 설정 | Table A1: 실험 조건·sample 수·baseline 구성 | P0: 주 결과를 해석하는 데 필요한 세부 |
| B. Target geometry extraction | Attention 선택·depth 연결·fusion/clustering·target 범위와 threshold | Figure A1: RGB→attention→3D target; 설정은 Table A2에 통합 | P0: ATS 재현, 그림은 Figure 3의 확장 |
| C. Phase and contact rules | Probe feature·label·학습 split, phase gate, link exemption, attach/detach | Table A2: 단계×링크 규칙 및 주요 파라미터; 필요시 Figure A2 timeline | P0: 본문의 frozen VLA 설명과 일치 |
| D. Refinement and implementation | ESDF·geometry coverage·CCR objective/constraint·solver·hardware·timing 경계 | Table A3: runtime breakdown/구현 설정 | P0: 기록된 설정·측정 재사용 |
| E. Extended results and analysis | 조건별 결과, 선택한 ablation, target 오류와 실행 실패의 연결 | Table A4: 조건별 결과; Figure A3: 대표 실패와 원인 | P1: 핵심 비교 이후 남은 분석 |
| F. Reproduction and video guide | 실행 명령·config·snapshot·영상별 task/조건 설명 | 간결한 설정 목록과 video index | P1: 실제 제공 자료의 범위에서 작성 |

본문에는 비교 조건의 핵심, metric 정의, 실제 사용한 입력과 학습 범위, 결론의 범위를 바꾸는 한계를 남긴다. 부록에는 그 세부와 추가 근거를 둔다. 모든 내부 개발 로그나 시행착오를 정리해 싣는 것은 목표가 아니다.

## 4. GitHub Pages 구성안

**목표:** 첫 화면에서 문제와 개선을 이해하고, 1–2분 안에 핵심 메커니즘과 비교 결과를 파악할 수 있는 연구 소개 페이지.

참고 사례:

- [NVIDIA FoundationStereo](https://nvlabs.github.io/FoundationStereo/): 자료 링크, method overview, 시각적/정량적 비교, 3D 결과 구성 참고.
- [Google Research·UW Nerfies](https://nerfies.github.io/): 짧은 설명, 영상 중심 전개, 직접 조작하는 시각적 비교 구성 참고.

두 사이트의 표현 방식을 참고하되 AG3S VLA의 실제 자료로 구성한다. 템플릿 코드를 재사용할 경우 해당 license·attribution을 확인하고 불필요한 analytics를 제거한다.

| 페이지 순서 | 제안 내용 | 재사용할 자료 |
|---|---|---|
| 1. 첫 화면 | 제목·한 문장 설명·15–30초 무음 대표 영상·Paper/Code/Video 링크 | Figure 1의 대표 실행; 링크는 실제 공개된 자료만 활성화 |
| 2. 핵심 문제 | 필요한 contact와 피해야 할 collision을 짧은 비교 영상으로 제시 | 같은 초기 조건의 baseline/ours |
| 3. 방법 이해 | RGB/attention→target point cloud→link-specific constraint를 순서대로 표시 | Figure 2–3의 패널과 동일 raw frame |
| 4. 직접 비교 | 같은 장면의 다른 지시 대상, grasp/carry/release 단계별 탭 또는 slider | P0-B와 필요한 HSR 진단; 결과가 확보된 범위만 |
| 5. 주요 결과 | 간결한 결과 표 또는 plot, metric·평가 규모·조건 설명 | Main Table과 동일 결과 파일 |
| 6. 더 보기 | 짧은 overview 영상, 전체 rollout, 추가 사례, 설명력 있는 실패 1–2개 | Supplementary video와 Appendix 분석 |
| 7. 자료 | Paper·supplementary·code·설정/데이터·BibTeX | 공개 가능한 최종 artifact |

대표 영상은 한 가지 장면만으로 넓은 일반성을 암시하지 않도록 task와 조건을 표시한다. 핵심 접촉 순간을 편집으로 가리지 않고 속도 변경과 reset을 표시한다. 웹사이트만의 새로운 수치나 별도 성능 주장을 만들지 않는다.

GitHub Pages는 HTML/CSS/JavaScript 정적 파일을 제공하므로 영상·이미지 중심 소개에 적합하다. 초기 구현은 backend 없이 구성한다. [GitHub Pages 공식 설명](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

- 예정 source 경로: `paper/project_page/` — 이번 작업에서 아직 생성하지 않음.
- 예정 주소 형태: `https://<owner>.github.io/<project>/` — 계정/저장소 미정이며 실제 주소가 아님.
- 최소 구현: 단일 페이지, 반응형 배치, lazy-loaded 영상, poster image, 자막·대체 설명, 실제 자료 링크.
- 선택 구현: 미리 계산한 point cloud viewer. 실시간 VLA 추론이나 새 GPU 서비스는 범위 밖.
- 콘텐츠는 영어를 기본으로 하고 내부 storyboard와 제작 체크는 한국어로 유지.

## 5. 실험과 산출물의 연결

| 실험/확인 | 함께 확보할 최소 원자료 | 본문 | Appendix / 영상 | 웹페이지 |
|---|---|---|---|---|
| P0-A: main comparison | 동일 task/seed의 실행 영상·outcome·contact event·설정 | Figure 1, Main Table | 조건별 결과, 전체 rollout | 대표 비교, 주요 결과 |
| P0-B: target/constraint 비교 | 동일 RGB/depth, instruction, attention, 3D target, 적용 link/constraint, 실제 궤적 | Figure 3, Ablation Table | Figure A1, target 오류 분석 | 방법 단계별 시각화, 지시 대상 비교 |
| P1-A: HSR/rule 진단 | phase·grasp state·exemption 전환·release 시점 | 필요시 Figure 3 timeline/ablation 행 | Figure A2, 단계별 영상 | grasp/carry/release 탭 |
| P0-C: 구현/runtime 확인 | 실제 pass별 timing, hardware, geometry 범위·설정 | 짧은 비용/구현 설명 | Table A2–A3 | Methods/Resources의 짧은 설명 |
| P1-B: 선택한 확장 평가 | 기존 평가와 구분되는 조건·sample·동일 metric | 필요시 Main Table 확장 | Table A4, 대표 장면 | 추가 환경 데모 |

기존 로그와 rendering 기능부터 확인한다. 저장되지 않는 필수 신호만 보완하며, 모든 run에서 모든 tensor를 저장하는 infrastructure는 만들지 않는다. 정량 결과 집계와 시각화용 episode 선정을 구분한다. 대표 episode는 메커니즘을 명확히 보여주는 사례로 선택하고 전체 성능은 정해진 평가 집합으로 계산한다.

작은 artifact manifest에 `artifact_id`, `run_id`, `snapshot/config`, `task/seed`, `frame/time`, `metric_source`, `output_paths`, `caption`을 기록한다. 본문·부록·영상·웹 결과를 같은 원자료와 연결한다. 이 manifest가 기존 evidence registry를 대체하지는 않는다.

## 6. 실행 순서와 완료 기준

| 기존 단계 | 추가할 작업 | 완료 기준 |
|---|---|---|
| PA-1 | Appendix 목차와 Figure/video/site storyboard 연결 | 어떤 실험이 어떤 패널·표·영상을 채우는지 정리됨 |
| PA-2 | 필수 시각 자료의 기존 export 확인 및 누락 보완 | 추가 연구 없이 선택한 실행의 그림·영상을 생성할 수 있음 |
| PA-3 | 실험 결과와 함께 본문·부록을 작성하고 영상 컷 생성 | 핵심 표·그림과 익명 영상 초안이 같은 결과를 설명함 |
| PA-4 | reviewer가 제기할 큰 의문 하나 보완; 웹 정적 초안 | 본문 근거를 확보한 뒤 기존 자료로 웹 스토리 구성 |
| PA-5 | 본문·부록·영상 대조, 사이트 모바일/링크/재생 확인 | 제출 파일이 자체적으로 완결되고, 웹 초안의 내용이 일치함 |

우선순위는 본문 → 필요한 부록/심사용 영상 → 공개 페이지 완성이다. 웹 디자인을 위해 핵심 실험이나 원고 완성을 미루지 않는다. 대표 영상·그림부터 재사용하면 전문적인 소개 페이지를 만들기 위한 추가 작업은 제한할 수 있다.

심사용 자료는 익명 PDF·영상으로 완결하고, 공개 페이지에는 배포 시점에 맞는 저자·자료 정보를 사용한다. 웹 공개를 무조건 채택 이후로 미루는 규칙은 추가하지 않는다. 공개 시점은 preprint 준비와 2027의 최신 규정을 함께 확인해 정한다.

## 7. 현재 상태

- 완료: 관련 공식 자료 확인, CVPR 기여 배치 제안, appendix/site 목차, 실험별 산출물과 기존 실행 계획 연결.
- 미실행: 추가 실험, Appendix 본문 집필, Figure/video 제작, 사이트 구축·배포, 실제 제출물 compliance 검사.
- 다음 작업: PA-1에서 Main/Ablation Table skeleton과 Figure 1–3 storyboard를 작성하며 본 계획의 자료 배치를 함께 확정.
