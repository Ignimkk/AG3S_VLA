# AG3S 선행 baseline 측정 현황 (2026-10-31)

이 문서는 기존 검증 기록을 정리한 것이며 새로운 성능 실험 결과가 아니다. AG3S 최종 구성이 확정되기 전에 baseline의 기준점을 파악한다. 정식 test 결과나 원 논문의 보고 수치로 해석하지 않는다.

## 실험: SafeLIBERO spatial Level I 개발 파일럿

무엇을 검증했나: 동일한 frozen π0.5 policy와 장애물 환경에서 기존 safety filter 및 sampling guidance를 붙였을 때 작업 성공과 충돌 없는 성공이 어떻게 바뀌는지 확인하는 예비 실험이다. Level I은 target 근처의 장애물을 다룬다. task 0–3 × init 0–4, seed 7, arm당 20 episode이며, 280 policy step + wait 20 조건이다. 공식 충돌 판정은 장애물 변위 1 mm 기준이다. 이 표본은 개발에 사용됐으며 작은 표본이므로 정식 성능 순위로 사용하지 않는다.

| 방법 | 작업 성공 TSR | 충돌 없는 비율 CAR | 안전 성공 SS | ETS (step) |
|---|---:|---:|---:|---:|
| π0.5 단독 | 11/20 (55%) | 2/20 (10%) | 2/20 (10%) | 197.25 |
| AEGIS 이식 | 4/20 (20%) | 8/20 (40%) | 3/20 (15%) | 258.00 |
| KNOWS 이식 | 4/20 (20%) | 6/20 (30%) | 1/20 (5%) | 265.90 |
| multi-link CBF 재구현 | 6/20 (30%) | 9/20 (45%) | 4/20 (20%) | 246.85 |
| OmniGuide 이식 | 11/20 (55%) | 2/20 (10%) | 2/20 (10%) | 201.30 |
| constrained flow matching 재구현 | 0/20 (0%) | 15/20 (75%) | 0/20 (0%) | 280.00 |
| NUDGE 부분 재구현 | 0/20 (0%) | 8/20 (40%) | 0/20 (0%) | 280.00 |
| Lan-o3dp guidance 이식 | 0/20 (0%) | 8/20 (40%) | 0/20 (0%) | 280.00 |

TSR은 공식 작업 성공 비율, CAR은 충돌이 없었던 episode 비율, SS는 공식 성공과 충돌 없음이 동시에 성립한 비율, ETS는 공식 기준의 episode step 수다. 안전하게 정지하는 것만으로 CAR이 올라갈 수 있으므로 SS와 TSR을 함께 봐야 한다. 위 수치는 [BL-7 검증 로그](../../../AG3S/ag3s/docs/AG3S_BASELINE_LOG.md) 표 B-1과 [검증 원본](../../../AG3S/ag3s/docs/handoff/CLOSED-LOOP-FIX-V5.verify.json)에서 옮겼다.

주의: BL-7 이후 BL-8에서 장애물 cloud의 depth trim과 CFM action box 등 기본 동작이 바뀌었다. 따라서 이 표는 현재 구현의 측정치가 아니다. 원 논문 재현도 판정할 수 없다. 당시 CFM의 clip 전 action 값은 최대 7.284였고, 두 job 병렬로 측정해 latency도 단일 job 지연으로 쓰면 안 된다. AEGIS는 원래 VLM 단계를 대체했으며, KNOWS의 closed-loop adapter는 attention target exclusion 대신 detector box를 사용한다. OmniGuide는 EE-delta 이식이고 multi-link와 NUDGE는 부분 재구현이다. 상세 차이는 [closed-loop README](../../../Baseline/paper_baselines/closed_loop/README.md)와 [catalog](../../../Baseline/paper_baselines/README.md)에 있다.

## 후속 실험과 비교 규칙

1. 현재 설정을 dev에서 재측정하고, 같은 benchmark·checkpoint로 보고된 AEGIS, multi-link, CFM의 privileged 조건과 원 논문 조건 차이를 대조한다. 원 논문 수치와 재실행 수치는 별도 표로 둔다. 원 조건의 차이가 남으면 충실도 한계를 명시한다.
2. 정식 test에 앞서 episode 목록, seed, commit, checkpoint, argv와 입력 권한, 보호 link 및 지표 정의를 고정한다. 같은 task/init/seed를 모든 arm에 쓰고 결과는 episode 단위로 보존한다. dev init 0–4와 test init 5 이상을 분리한다. 계획서의 test 표본 및 gate는 제안 상태이므로 이 문서에서 확정하지 않는다.
3. 개선된 AG3S가 준비되면 동일 조건에서 SS, TSR, CAR, ETS, 성공했지만 충돌한 비율, 충돌 없이 미완료한 비율을 함께 보고한다. paired 차이와 불확실성을 보고하고, 실제로 보호하지 않는 link의 접촉을 구분한다. 상이한 설정으로 측정한 BL-7 수치와 정식 test 결과를 직접 순위화하지 않는다.
4. 인식과 제약의 효과를 구분하는 F 실험에서는 각 비교쌍이 정말 같은 mask provider, depth, target 제외 및 장애물 입력을 사용하는지 확인한다. 현재 closed-loop KNOWS/AEGIS는 Grounded-SAM-2 mask를 사용하는 F1이 아니며, `:gt`는 문서상 GT 장애물 입력이지 자동으로 GT target mask가 아니다. F1/F2와 F5/F6을 같은 인식 조건으로 취급하기 전에 adapter와 입력 기록의 검증이 필요하다.

## 논리 보강 제안

- 충실도 점검은 같은 모델·suite·수준·episode 수·충돌 정의·step budget까지 맞는지 비교표를 먼저 만든다. 원 논문과 차이가 있으면 구현 성능과 논문 수치의 차이를 원인 미확정으로 보고한다.
- 표본 수는 episode 수만으로 검정력을 판단하지 않는다. task별 상관과 paired 불일치 비율을 고려한 사전 검정력/정밀도 분석을 수행한다. 비열등성은 사전 지정 margin과 신뢰구간으로 판정하고, 불확실한 결과를 동등성이나 실패 원인의 증명으로 해석하지 않는다.
- 계획서의 F1·F2, F5·F6은 mask와 장애물 입력까지 일치해야 인식 축을 고정한 비교가 된다. controller 외의 차이(예: CBF와 CCR의 최적화 방식·link coverage)도 함께 달라지므로 각 구성요소의 기여는 별도 ablation이 필요하다.
- 금지 접촉을 벤치마크별 공식 충돌 판정과 별도로 정의한다. 1 mm 장애물 변위는 모든 로봇 접촉을 직접 측정하는 값이 아니며, 접촉 지속·침투 깊이는 센서/로그가 실제로 제공할 때만 산출한다.

## 현재 코드 smoke (2026-10-31)

π0.5 단독으로 spatial Level I dev task 0, init 0, seed 7의 1 episode를 실행했다. 서버 기동 및 client 종료는 정상(client return code 0)이었고 공식 성공 0/1, 충돌 1/1, 안전 성공 0/1, ETS 280이었다. 실행 경로 확인을 위한 n=1이며 성능 추정이나 BL-7과의 우열 비교가 아니다. 서버 준비 130.1초, client 128.5초가 걸렸다. 실행 원본과 manifest는 `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261031T192004_96fda38abd1b_baseline-20261031-dev-smoke-none-LI-none/`에 있다. launcher의 출력 root는 `AG3S_RESULTS_ROOT`가 아니라 `AG3S_STORAGE_ROOT`를 사용하므로 결과가 공유 storage에 생성됐다. checkpoint는 `pi05_libero`, 코드 HEAD는 `96fda38abd1b`다. 나머지 baseline의 현재 설정 다중 episode 재측정과 원 논문 조건 충실도 점검은 아직 수행하지 않았다.
