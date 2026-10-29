# T44–T56 종료 구간 영향 감사 — 2026-10-13 KST

## 결론과 이번 확인 범위

종료 후 추론이 원본 관측 자체를 소급해 망가뜨린 것은 아니다. 그러나 완료 이후 frame이 평균, 실패 수, head 선택, 학습·개발 목적함수에 들어가면 연구가 측정하는 대상이 달라진다. 특히 완료가 빠른 성공 run에서 무의미한 뒷부분이 커져, 필요한 작업 구간의 품질이 희석될 수 있다.

이번에는 T44–T56 task/verify와 분석 코드를 점검하고, **T53/T54의 기존 물리 `placed` 표시를 이용한 잠정 재집계**를 수행했다. 새 VLA 추론·학습·MuJoCo rollout은 수행하지 않았다. 원본 결과·공용 로그·실행 코드는 변경하지 않았다.

- 재현 코드: [termination_scope_audit_20261013.py](termination_scope_audit_20261013.py)
- 별도 진단 결과: [TERMINATION_SCOPE_AUDIT_20261013.json](TERMINATION_SCOPE_AUDIT_20261013.json)
- 검증: 기존 T54의 current/hold × identity/no_target/purity/coverage × 8 arm 평균을 1e-6 이내로 재현한 뒤 절단했다. episode bootstrap B=10000, seed=5407 유지.

## 두 종료 규칙을 구분해야 한다

**규칙 P — placement 완료:** target이 지정 용기에 놓이고 손에서 분리돼 안정적으로 남는 첫 시점. 완료 확인에 필요한 관측은 terminal evidence로 보존하고, 그 이후의 task 추론·행동·지표를 제외한다. 단순 `in_crate`, latch의 `PLACED`, HSR의 `home`은 서로 같은 정답이 아니다. 안정성 유지시간을 쓰면 그것이 충족된 시점이 causal 종료 시점이다.

**규칙 H — 준비자세 복귀 완료:** 성공적인 placement 이후 준비자세 부근으로 복귀한 첫 시점. 정확한 관절값 일치나 정밀한 home 위치 복원은 요구하지 않는다. 재현 가능한 평가를 위해 준비자세 기준과 넉넉한 자세 허용범위를 정하고, 움직이며 잠깐 통과한 경우를 구분할 짧은 확인 구간을 사용할 수 있다. 초기 자세가 준비자세와 비슷하다는 이유로 t=0에서 종료하면 안 된다. place→준비자세 이동은 이 규칙에서 유효 작업이며 그 구간의 충돌·HOLD는 평가 대상이다.

**사용자 확인 (2026-10-13 KST, 최초 보고 후):** P는 “놓여서 안정적으로 남은 시점”이다. H에는 기존 수치 기준이 없으며 “그냥 준비자세 정도로 돌아오면 됩니다. 오차가 발생해도 괜찮습니다.”라고 명시했다. 두 종료 규칙의 영향 비교를 유지한다. 자세·속도 허용오차와 유지시간의 구체적인 수치는 아직 검증·확정하지 않았으며, 위의 짧은 확인 구간은 구현 제안이다.

P의 정식 후속 판정은 물체가 지정 용기에 있고 손에서 분리됐는지와 함께, 위치 변화·속도 등으로 안정성을 확인해야 한다. H는 그 뒤 준비자세 부근 복귀를 확인한다. 고정된 step 수나 HSR class만으로 대신하지 않는다. 허용오차는 준비자세 기록과 대표 장면을 이용해 정하고, 방법별 점수가 유리해지는 방향으로 조정하지 않는다. 이전 401 frame 절단 및 T54 수치는 기존 물리 `placed` 표시를 이용한 잠정 분석으로 그대로 보존한다.

현재 확인한 평가 경로에는 위 H를 만족하는 공통 물리 완료 timestamp가 없다. 따라서 H의 정확한 제거 frame 수와 수정 수치는 이번에 계산하지 않았다. HSR `home`은 복귀 단계의 class이며 실제 도착 확인으로 쓰면 안 된다.

두 규칙 모두 완료하지 못한 run은 원래 horizon까지 유지한다. 실패 episode를 버리지 않는다. 정책별 유리한 cutoff를 고르지 않고, 동일한 술어를 모든 arm에 적용한다. Offline 동일 입력 비교는 동일 원본 episode의 공통 cutoff를, closed-loop 비교는 각 arm의 실제 완료 시점을 같은 술어로 판정한다.

## 실측: T53/T54에는 큰 배치 후 구간이 있다

T53 물리 `grasp_state`에서 16 episode × 75 frame = 1200 frame 중 **401 frame (33.42%)**이 `placed`다. 9 episode에서 42–48 frame씩 나타나며, 그 9개에서는 첫 `placed` 이후 관측된 label이 끝까지 `placed`로 유지됐다. 나머지 7 episode는 제거하지 않았다.

T53의 `placed` 정의는 **target 중심이 crate 내부이고 held가 아님**이다. 별도의 안정성 dwell을 확인하지 않으므로 정식 규칙 P와 완전히 같다고 단정할 수 없다. 아래 결과는 첫 `placed` frame에서 새 추론을 시작하지 않는다는 가정으로 그 frame부터 제외한 **종료 정의 민감도 분석**이다.

T54의 GT target 가시 조건(3 camera 중 하나에서 200 pixel 이상)을 만족한 평가 frame은 1178→777로 줄었다. episode 16개 동일 가중, 기존 head·arm 고정, `current` 조건이다.

| arm | 기존 전체 identity | 첫 placed 이전 identity | 기준 L8H2와의 차이: 기존 → 절단 |
|---|---:|---:|---:|
| L8H2 | 70.93% | 78.30% | — |
| N | 78.08% | 88.17% | +7.15 → +9.88 %p |
| B | 78.00% | 87.78% | +7.07 → +9.48 %p |
| A | 67.85% | 76.61% | −3.08 → −1.68 %p |

N−L8H2 절단 후 차이의 episode bootstrap 95% CI는 [+5.28, +14.93] %p, B는 [+5.23, +14.27] %p다. 이는 기존 고정 후보의 잠정 평가이지 새로운 선택·채택 결과가 아니다.

**판정에 실제 영향이 있다.** A의 current identity 하락은 −3 %p 허용폭 바깥에서 안쪽으로 들어온다. 그러나 purity 차이는 −1.72 %p, coverage 차이는 −1.83 %p이고, 기존 hold 재생은 attach fidelity 문제가 있다. 따라서 A가 최종 기준을 통과했다거나 채택해야 한다는 결론은 낼 수 없다. 기존 “identity −3.08 %p여서 탈락” 문장은 종료 정의를 고정하고 다시 판정해야 한다.

기존 hold 조건에서 L8H2/N/B 전체 identity 43.67%는 절단 후 64.62%다. A는 42.26→61.19%로 변한다. **시간적 재생이 실제 attach를 재현하지 못한 기존 제한은 절단으로 해결되지 않는다.** T54b의 충실도 검증은 별개로 필요하다.

근거: [T53.verify.json](../../../AG3S/ag3s/docs/handoff/T53.verify.json) `extraction`; [t53_extract.py](../../../AG3S/ag3s/experiments/studies/t53_extract.py) `grasp_state_sequence`; [T54.verify.json](../../../AG3S/ag3s/docs/handoff/T54.verify.json); [t54_stats.py](../../../AG3S/ag3s/experiments/studies/t54_stats.py).

## STEP별 영향과 필요한 재작업

| STEP / 결과 | 규칙 P | 규칙 H | 필요한 작업 |
|---|---|---|---|
| T44 I0/I1/I2 등의 구현·권한 일관성·단위/회귀 테스트 | 기능적 사실 유지 | 동일 | 종료 규칙 때문에 테스트 전체를 다시 할 필요 없음 |
| T44 E2 demonstration pre-grasp 기하 비교 | 범위 안, 직접 영향 없음 | 동일 | 새 cutoff보다 앞이라는 확인만. 파지 전 geometry 결과 유지 |
| T44d CL / T44e CL2 / T46 CL3 | full-run HOLD·접촉·시간·최종 성공 및 release 기반 지표 재계산 | home 도착 후 부분만 제거; 도착 전 복귀 충돌/HOLD 포함 | 원본 control/contact 기록을 새 종료 술어로 재집계. 기존 궤적부터 전부 재수집할 필요 없음 |
| T45 mesh coverage / palm-cover 구성 | 정적인 형상 누락 진단 유지 | 동일 | 실제 contact 사례·빈도·phase 평균만 cutoff 대조 |
| T46 F1/F2 held-out pre8+close 비교, 파지 직전 HOLD 12→6 | 첫 정상 placement 이전인 해당 파지 구간은 유지 | 동일 | geometry 효과와 pre-grasp 근거를 완료 후 현상과 분리 |
| T46b / T46c-W1 놓은 뒤 no_seed·uncertified HOLD | 실제 placement 완료 이후라면 주 작업 평가 밖. W1의 주 성능 개선 근거에서 제외 | home 도착 이전이면 유효한 복귀 장애, 이후면 평가 밖 | 실제 placement/home 시점과 HOLD onset 교차표. W1 코드가 잘못됐다는 의미는 아님 |
| T46d/f 손목·팔 형상 audit, T46e oracle pre-grasp 분석 | 형상과 pre-grasp 결론 유지 | 동일 | 전체 planning 자세를 이용한 coverage audit는 여전히 기하 검증으로 유효. 실행 지표는 새 window 적용 |
| T47/47b attention head selection·전체 grounding·sink/entropy 집계 | calibration과 held-out 모두 재절단 필요 | 도착 후 frame 제거 필요 | 저장 attention으로 선택 점수·필터·bootstrap·평가 재실행. 후보가 바뀌면 해당 3D replay 추가 |
| T48a HSR 주 dataset 실험 | max-step tail 원인의 직접 손상은 아님. 새 작업 범위에서는 home 제외/분리 집계 필요 | demo의 home 이동 포함, 실제 도착 후만 제외 | 기존 feature/prediction을 새 window로 평가. image/state donor도 새 범위 준수 여부 점검 |
| T48a closed-loop ep1807 보조 평가 | tail 포함 요약 재계산 | 도착 후 제거 | 기존 보조 record를 절단 |
| T48b empty/slip/phantom 물체 개입 | E1/E2는 held, E3는 pre-grasp여서 첫 placement 전 host라면 주요 개입 근거 유지 | 동일 | host 시점 대조; E0 전체 정확도·phase 평균은 재집계. image parity 자체는 유효 |
| T49/49b 명사 query 비교 | 전체 및 post-grasp 묶음 결과 재집계 | 도착 후 제거 | 저장 map/GT로 평가. 선택을 다시 하면 calibration도 재계산. post-grasp와 post-place를 구분 |
| T50 HSR representation / counterfactual probe | 주 입력은 demo이므로 max-step 손상 아님. place+home을 묶은 C1_instr와 전체 민감도는 새 범위와 다름 | demo 복귀 구간은 유효, 도착 이후는 구분 | 기존 probe 고정 평가와 새 범위용 probe 재학습을 별도 실험으로 구분. train/CV 범위를 바꾸면 CPU 재학습 필요 |
| T51/51b head·k 선택 protocol | P1 필터, GT calibration, 순위 안정성, k 선택, 평가 재실행 | 도착 후 제외 후 동일 | self label의 event 이전 frame은 직접 영향 작음. 하지만 후보 필터와 전체 GT 품질 목표는 영향 있음. LIBERO 결과는 RB-Y1 tail 문제만으로 무효 아님 |
| T52 stage별 실패 분해 | 집계 비율·phase 구성 재계산 | 도착 후 제거 | 보존 frame의 단계별 기전은 그대로. 끝난 뒤 실패를 작업 중 실패로 세지 않도록 수정 |
| T53 평가 데이터·기준선 | 401 frame 제거 규모를 잠정 확인. 원본 보존, benchmark window 수정 | actual home timestamp 필요 | 성공·접촉·HOLD·시간과 downstream manifest 갱신 |
| T54 / T54b / T54c | 비교 수치·진입 판단에 실제 영향 확인 | actual home cutoff 적용 후 재평가 | 현재 후보 고정 재평가부터. hold_attach 충실도 수정은 별도. 새 평가 집합의 종료 정의도 통일 |
| T55 물리 파지 / task gate | placed 빈손이 negative 평가를 과도하게 차지할 수 있음 | 복귀 중 구간은 유효, 도착 후는 제외 | dev/test window, 오류 분모, event·지연 censoring 재설정. empty/slip intervention이 placement를 막으면 원본 E0의 완료 시점으로 잘라서는 안 됨 |
| T56 self score 선택 | S0/S1 pre-event 점수는 보통 유지되나 전체 GT regret·dev 최적화는 변함. S2도 HELD 종료 대조 필요 | 도착 후 GT 구간을 제외 | dev grid/순위/regret/CI와 prereg 버전 갱신. 아직 최종 검증된 개선으로 세지 않음 |

## 기존 성공 지표는 마지막 frame에 의존한다

T44e-CL2 및 T46-CL3의 명시된 정의:

- task success: target을 들어 올렸고 **마지막 planning row**에서 target이 crate 내부.
- `release_t`: **전체 run의 마지막 hand–target contact row + 1**.
- `success_rel`: 그 release 이후 분리 상태가 끝까지 유지되고 마지막에 crate 내부.
- `success_ct`: 위 성공 조건과 `t <= release_t`의 비허용 접촉 부재.

따라서 새 종료 규칙에서는 단순히 HOLD만 잘라서는 안 된다. 완료 이후 재접촉·재파지·밀어내기가 있으면 마지막 접촉과 성공 판정까지 달라진다. 기존 `pre-release` 수치도 원래 정의의 release가 정말 첫 정상 placement와 맞는지 확인해야 한다.

home 복귀를 task에 포함하면 `success_ct`가 release까지만 접촉을 보는 것도 부족할 수 있다. place→home의 비허용 접촉까지 같은 독립 접촉 정책으로 포함해야 한다. 이미 있던 전 구간 contact trace를 활용한다.

근거: [T44e-CL2.verify.json](../../../AG3S/ag3s/docs/handoff/T44e-CL2.verify.json) `definitions`; [T46-CL3.verify.json](../../../AG3S/ag3s/docs/handoff/T46-CL3.verify.json) `definitions.aggregation`.

## W1과 F1을 섞어 결론 내리지 않는다

T46b의 phase는 **서버 `grasp.state`** 기준이다. `placed`는 release 후 place+home을 포함하며, 물리적으로 안정적인 성공 placement를 인증한 label이 아니다. 기록상 C1a/C1aF1/C1aF1F2의 placed HOLD는 각각 55/130/118 chunk이고 그중 uncertified는 37/130/118이다. 이 수치를 곧바로 규칙 P의 제거량으로 사용해서는 안 된다.

다만 다음 실제 기록은 종료 후 꼬리의 영향을 강하게 시사한다: C1aF1 ep11_s111은 기록 `place_t=184`, `no_seed` onset t=280 뒤 40 chunk가 끝까지 HOLD; ep30_s301은 `place_t=224`, onset t=312 뒤 36 chunk. **실제 home에 이미 도착했는지는 별도 판정이 필요**하다.

규칙 P에서는 정상 placement 이후 W1 효과가 주 작업 범위 밖이다. 규칙 H에서는 W1이 home 도착을 가능하게 하는지, 도착 뒤 불필요한 추론만 줄이는지 구분해야 한다. 도착하지 못하고 HOLD한 run을 이미 완료했다고 제외해서는 안 된다.

근거: [T46b.verify.json](../../../AG3S/ag3s/docs/handoff/T46b.verify.json) `design.phase_definition`, `numbers.phase_table`, `numbers.no_seed_onset_vs_place`.

## 무엇부터 다시 해야 하는가

1. **정식 P/H 종료 술어와 timestamp를 만든다.** 관측·control·contact·planning chunk의 시간 정렬을 확인한다. 경계를 가로지르는 chunk는 행동과 접촉을 실제 cutoff까지만 포함하고, 완료 확인용 관측과 완료 후 새 추론을 구분한다.
2. **기존 후보·가중치를 고정한 재평가:** T44/CL2/CL3/T53의 성공·safe success·HOLD·접촉·시간, T54 current/hold_attach를 우선한다. P/H를 함께 보고 결론의 민감도를 확인한다. 이번 JSON은 그 전 단계의 proxy다.
3. **선택/학습 목적함수 재계산:** T47/T49/T51 계열 calibration부터, T55/T56 dev/test까지 cutoff를 적용한다. 필터나 shortlist가 달라지면 새 후보에 대한 replay가 필요하다. 저장된 feature/map을 우선 재사용한다.
4. **새 rollout은 필요한 경우에만:** 종료 감지기를 실제 runtime에 넣었을 때의 작동 확인, 변경된 head/extent/형상/gate의 closed-loop 검증, 기록에 완료 판정 정보가 부족한 경우. 미래 tail을 지운다는 이유만으로 동일 정책의 모든 기존 prefix를 다시 실행할 필요는 없다.
5. **변경 이력과 평가 집합 관리:** 원본을 삭제하지 말고 endpoint mask와 protocol version을 저장한다. 기존 결과는 과거 full-horizon 평가로 남기고 수정 수치를 별도 보고한다. 결과를 보고 유리한 종료 규칙을 고르지 않는다. 이미 개발에 사용한 T53을 새 후보의 독립 test로 복원할 수는 없다.

추가 혼동 방지: 과거 `long`/`ep1800`의 head camera 자세 오류, T54 hold의 attach 누락은 이번 종료 구간 문제와 별개다. 해당 기록의 원래 제한도 계속 표시해야 한다. T58의 첫 obstacle contact 실패는 placement 이전에 발생했으므로 두 종료 규칙으로 사라지지 않는다.

## 후속 — T60 수행 결과 (2026-10-14 KST 추가, 위 본문은 고치지 않음)

위 "무엇부터 다시 해야 하는가" 1–3 을 정식 술어 (P: 8 row 안정 placement, H: 관절 0.35 rad ∧ TCP 0.12 m 복귀, 사용자 판정
2026-10-13) 로 수행했다. 결과 · 한계 · 판정 대기는 [`AG3S_T47_LOG.md` 의 T60 절](../../../AG3S/ag3s/docs/AG3S_T47_LOG.md),
수치는 [`T60.verify.json`](../../../AG3S/ag3s/docs/handoff/T60.verify.json). 이 문서의 잠정 수치 (첫 physical `placed`, 401 frame)
는 그대로 두고, 정식 P 는 T53 에서 362 frame 을 제거한다. 위 표에서 "T52 재계산 필요" 는 T52 가 placement 전에 끝나는
held-out 15 frame record 만 써서 해당 없음으로 확인됐다.
