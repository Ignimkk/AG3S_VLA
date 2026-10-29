# 새 세션 시작 안내 (IFG 작업 인계)

> **2026-10-23 최신 연구 결정:** 먼저 [연구 범위 한 페이지](plans/AG3S_RESEARCH_SCOPE_20261023_KO.md), [읽은 문서와 판단 근거](reviews/AG3S_SCOPE_DECISION_20261023_KO.md), [평가 구성·비교·종료 기준](../../AG3S/ag3s/docs/handoff/AG3S-CORE-EVALUATION.task.md)을 읽는다. 다음 작업은 그 task §8의 CPU 평가 준비다. 아래 10-18 인계는 당시 이력으로 보존하며, 실패 장면별 규칙 추가를 자동 재개하지 않는다.

작성: 2026-10-18 18:20 KST. 이전 세션(IFG 분석, ag3s-vla-development-52)이 만료되기 전에 남긴 인계 문서다. **이 문서를 가장 먼저 읽는다.**

## 1. 읽는 순서

1. 이 문서
2. [reviews/vla_dump_20261015/FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md](reviews/vla_dump_20261015/FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md) — **다음 할 일의 기준**(최종 실험 사슬, 논문 figure)
3. [reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md](reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md)의 **마지막 4개 절**(10-18 13:10 이후). 특히 14:30 "정정" 절은 결론을 쓸 때 지켜야 할 강도 기준이다
4. 필요할 때만 읽는다: [README.md](README.md)(active 문서 목록), FINAL_REPORT, METHODOLOGY_EVOLUTION(IFG dev 결과와 채택안 B)

## 2. 지금 돌고 있는 작업: 선별 없는 paired 재평가

> **2026-10-19 01:09 KST 완료.** 결과는 WORKLOG 10-19 01:15 절과 `AG3S/ag3s/docs/handoff/IFG-CL3.verify.json`, figure는 `figures/ifg-cl3/ifg-cl3-*.png`. 아래 "끝나면 할 일" 1–4는 모두 했다. 다음은 3절(사용자 확인).

| 항목 | 값 |
|---|---|
| 무엇 | 40 장면 × seed 1. **기존 방식(C1aN) 대 표면 영역 방식(SURF)** 짝 비교, 80 run |
| 장면 | clear 14 · adjacent 13 · place 13. 성공 여부를 보지 않고 metadata(해시 순서)로 골랐다. `seeds.json` sha256 `88f6b37e…190f2b` |
| run | `/mnt/dev/AG3S_VLA-storage/results/development/20261018T135457+0900_f15e587290f6_2cd4b0b6acad/verify/IFG-CL3` |
| launcher | `p/cl3_launch.sh`, log `cl3_launch.out`. 18:19 KST 기준 C1aN 20/40, SURF 18/40 |
| 예상 종료 | rollout 약 20:20 KST, 이어서 CPU 분석 약 10분(자동) |
| 결과 파일 | `agg.json`(성공 · 접촉 · HOLD · 완료 시간 · bootstrap CI), `timing.json`(지연 · 단계별 계산 비용) |
| task | `AG3S/ag3s/docs/handoff/IFG-CL3-run.task.md` |

**끝나면 할 일:**

1. `cl3_launch.out`에서 `ANALYSIS DONE`과 `STOPPED`가 없는지 확인한다
2. **benchmark 세션에 GPU 반납을 알린다.** `ListAgents`로 현재 이름을 확인한다(10-18 기준 `ag3s-vla-benchmark-cb`). 소켓 주소는 바뀔 수 있다
3. 사용자 지시 형식으로만 보고한다
   - 주 지표: 접촉 없는 작업 성공의 paired 차이(SURF − C1aN)와 95% CI
   - 2×2 표: 둘 다 성공 / C1aN만 / SURF만 / 둘 다 실패. stratum별로도 낸다
   - 잡기 실패 · 접촉 · HOLD · 완료 시간
   - 요청 지연과 계산 비용
   - **동등성이나 "영향 없음"을 판정하지 않는다. p값이 유의하지 않다고 "차이 없음"이라 쓰지 않는다. seed를 자동으로 추가하지 않는다**
   - SURF는 표면 extent와 render mask를 **함께** 바꾼 구성이다. extent 하나의 효과로 설명하지 않는다
4. WORKLOG에 결과 절을 추가하고, figure와 `.json` sidecar를 만든다(규칙 A)

## 3. 그다음 순서 (사용자 확인을 받고 진행)

1. **run 폴더 label 구현:** `common/run_output.py`에 선택적 label을 넣는다(하위 호환). 공용 모듈이라 implementer가 test와 함께 구현한다. 그 전까지는 storage의 `results/development/RUN_INDEX_IFG.md`에 새 run을 한 줄씩 추가한다
2. **최종 실험 사슬 탐색:** 같은 장면에서 다음 네 단계를 모두 통과하는 시나리오를 찾는다(계획 문서 3절)
   - ① VLA 단독 성공
   - ② 장애물 없이 AG3S 성공
   - ③ 허들 + AG3S 성공
   - ④ 허들 + pick 방해 장애물 + AG3S 성공
   - pick 장애물은 target 가림 약 30%(5–60%)이거나 원래 궤적과 간섭할 때만 쓴다
   - 탐색 규모와 장면 풀을 사용자에게 먼저 묻는다
3. **논문 figure**
   - Fig. 1 teaser: 일부 가려진 target의 정확한 segmentation과 선정
   - pick · place 궤적 수정: 그래프와 실제 장면
   - task phase 시각화

## 4. 사용자 규칙 (반드시 지킨다)

- **보고할 때 실험 코드명(C1a, SURF, H0 …)을 설명 없이 쓰지 않는다.** "우리 영역" 같은 모호한 말도 쓰지 않는다. 쓰려면 아래 표처럼 뜻을 함께 적는다
- 성공률 **평가**에서는 어떤 arm의 성공으로도 장면을 고르지 않는다. 시연과 figure용 시나리오 **탐색**과 구분한다
- 결론의 강도는 측정이 허용하는 만큼만 쓴다(WORKLOG 10-18 14:30 정정 절의 표)
- 논문 figure는 성능만큼 중요하다. 실제 장면 중심으로, 아름답고 독창적으로 만든다
- 판단이 필요한 설계 결정은 실행 전에 사용자에게 묻는다. commit과 push는 사용자가 요청할 때만 한다(push는 사용자가 직접 한다)
- GPU: 다른 checkout(benchmark)과 같은 lock(`/tmp/ag3s-vla-gpu-validation.lock`)을 쓴다. 시작 전과 종료 후에 benchmark 세션과 알림을 주고받는다

## 5. 실험 방법 이름 (보고할 때 풀어 쓸 것)

| 이름 | 뜻 |
|---|---|
| C1a | 현재 serving. action attention으로 target을 고르고, target 주변 **공 모양 영역**을 잡는 손의 충돌 지도에서 지운다. robot은 FK 구 부피로 지운다 |
| C1aN | C1a에서 target 선택만 **noun attention**(명령의 과일 단어)으로 바꾼 것 |
| SURF | C1aN + 공 영역 대신 **attention + depth로 복원한 target 표면**만 지움 + robot을 **렌더한 표면**으로 지움 |
| ORAC · ORACS · SURFR · SURFS | 진단용: SURF의 표면을 시뮬레이터 정답으로 바꾼 것(ORAC), 그것을 FK 구 mask로 바꾼 것(ORACS), SURF 재실행(SURFR), SURF를 FK 구 mask로 바꾼 것(SURFS) |
| hybrid (CL2 · CL2b) | 공 영역으로 지우되, 표면 영역으로 이웃 물체를 보호하는 수정안. offline 재생만 했고 closed-loop는 아직 없다 |

## 6. 지금까지의 핵심 결과 (자세한 내용은 WORKLOG)

- **offline(정답과 표면 겹침, held-out 34 episode):**
  - 기준 graph 0.773 → 표면 영역 + robot 렌더 0.876 → + 기억 전환 0.891
  - robot 렌더 효과 +0.044, 34/0
- **closed-loop(C1a 성공 장면 34개, 성공 조건부):** C1aN 34, SURF 28
  - 이 결과는 기준선 성공 사례가 보존되는지를 보여 준다. 선별 없는 전체 장면의 성능 차이는 추정하지 못한다
- **허들 14:** 막대 접촉 없는 성공 C1a 11, C1aN 11, SURF 7. 불리한 관측이지만 불확실성이 크다. 허들 seed 확대는 보류했다
- **SURF 실패의 물리적 원인은 미확정이다.** GT 표면으로도 4/6 실패, TrajOpt 수정량은 작음, 그리퍼 닫을 때 HOLD 여부와 상관은 있으나 기전은 없음
- **실시간성:** client 왕복 median C1aN 1.1–1.3 s, SURF 2.9–3.3 s. attention_surface 계산 약 1.7 s가 병목이다. 두 방식 모두 533 ms 예산을 넘는다

## 7. commit 안 된 작업 (사용자가 요청할 때만 commit)

- **serving 구현:** `AG3S/ag3s/runtime/attention_surface.py`, `robot_render_mask.py`, `attention_surface_hybrid.py`, 그리고 `config.py` · `pipeline.py` · `curobo_builder.py` · `constraint_record.py` · `serve_safe.py` · `safe_policy.py`의 변경
- **test:** `tests/ag3s/test_ifg_cl1_attention_surface.py`, `test_ifg_cl2_hybrid_extent.py`
- **분석 스크립트:** `AG3S/ag3s/experiments/studies/`의 `ifg_heldout*.py`, `ifg_cl1_*.py`, `cl1_parity.py`, `cl2_hybrid_replay.py`, `ifg_ppt_figures.py`
- **문서와 figure:** `doc/` 재편(active / archive), handoff 문서(`IFG-*.task.md`, `IFG-CL1.impl.md`, `IFG-CL2.impl.md`), figure(`figures/ppt-1015`, `ifg-cl3`, `ifg/ifg-heldout-locked.*`)
- 진행 중인 재평가 run은 snapshot tree `f0d8046b`를 쓰므로 위 작업과 무관하게 돈다
