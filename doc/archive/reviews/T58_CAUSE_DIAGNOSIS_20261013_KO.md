# T58 실패 원인 진단 — 2026-10-13

## 범위와 결론

T58 원본 chunk NPZ, 진단 JSON, 실행 스크립트, 고정 코드 스냅샷을 읽어 대조했다. 새 rollout, MuJoCo contact replay 재실행, palm-cover 효과 검증은 수행하지 않았다. 접촉 결과는 기존 T58 replay 기록에 근거한다.

**핵심은 안전 판정이 보는 환경과 로봇 형상이 실제 접촉을 충분히 표현하지 못한다는 것이다.** 1830에서는 초기 target 혼합과 그 기하의 유지가 기둥 제거로 이어진다. 세 episode 모두 손바닥 모델 누락이 관측된다. 다만 “clearance 4.7–5.2 mm < margin 10 mm인데 실행했다”는 기존 해석은 잘못됐다. 해당 metric은 이미 margin을 뺀 제약 잔차다.

- 원본 run: `/mnt/dev/AG3S_VLA-storage/results/development/20261013T161809+0900_cf38c9dbca7b_6ee6157e1d6c`
- 이하 `V` = 위 run의 `verify/T58`.
- 실행 코드: run의 `snapshot`, HEAD `cf38c9dbca7b2c6d78935a67a1edc6fc2ed6d3ac`.
- `linearize.py`, `safe_policy.py`, `curobo_builder.py`는 점검 시 작업 트리와 snapshot이 바이트 동일했다.
- 원본 요약: [T58.verify.json](../../../AG3S/ag3s/docs/handoff/T58.verify.json).
- 핵심 NPZ: `V/runs/server_ON_8295_20261013T074151/server_constraints/run_0000/chunk_00010.npz` ~ `chunk_00016.npz` (1830).

## 1. 1830: exclusion보다 앞에서 target 점 집합이 오염된다

첫 planning frame `t=0`의 기록:

- target 점 262개 중 167개가 기둥 중심축 반경 26 mm, 높이 0.82–1.08 m의 ROI에 있다. 패딩 없이 반경 20 mm로 세어도 161개다. 이는 GT instance segmentation precision이 아니라 알려진 기둥 위치에 대한 공간 중첩 측정이다.
- target principal extents는 **53.8 × 121.0 × 258.3 mm**이다.
- `attention_recluster`는 실제로 두 성분을 찾았다. 큰 성분은 187점, 크기 39.7 × 40.4 × 248.2 mm로 기둥에 부합한다. 작은 성분은 75점, 크기 27.9 × 60.0 × 61.3 mm로 과일에 부합한다.
- 기둥 성분의 attention 합계는 127.1111, 과일은 68.1868이다. 반대로 평균 attention은 각각 0.6797, 0.9092이며 raw 평균도 0.0540, 0.0653으로 과일 쪽이 높다.
- 구현은 **attention 합계로 후보를 먼저 정하고**, raw 평균 비율로 그 후보를 승인한다. 따라서 기둥을 먼저 고른 뒤 `ratio=0.827 < 2`로 분리를 거절한다. 결과는 `attention_not_discriminative`, 원래의 혼합 cluster 유지다.
- 좁은 축 53.81 mm가 gripper opening 71.46 mm보다 작으므로 `admissible=True`이다. 이 검사는 잡을 수 있는 폭에 대한 조건이며 단일 물체라는 보증이 아니다.

근거 코드: [attention_split.py](../../../AG3S/ag3s/stages/attention_split.py), `split_by_attention`; [admissibility.py](../../../AG3S/ag3s/stages/admissibility.py), `graspable`.

**진단:** 기하적 분리 자체가 불가능했던 경우가 아니다. 후보 선택 통계와 승인 통계가 다르고, 분리 불확실성이 혼합 물체 전체의 exclusion 허용으로 이어진다. 평균 attention으로 바꾸기만 하면 해결된다는 뜻도 아니다. 첫 frame에서 과일/기둥 raw 평균비는 약 1.21로 현재 기준 2에 여전히 못 미친다.

## 2. 오염된 기하가 큰 ball과 잘못된 ownership으로 전파된다

1830의 시간별 기록은 다음과 같다. 기둥 ROI는 위 반경 26 mm 조건이다.

| planning t | target 점 수 | 기둥 ROI에 든 target 점 | exclusion ball 반지름 (mm) | F1 적용 | manipulated 상태 |
|---|---:|---:|---:|---|---|
| 0 | 262 | 167 | 148.601 | false | visible |
| 8 | 259 | 165 | 148.304 | false | visible |
| 16 | 258 | 167 | 147.374 | false | visible |
| 24 | 275 | 166 | 153.485 | false | visible |
| 32 | 278 | 163 | 155.018 | false | visible |
| 40 | 278 | 163 | 155.018 | false | occluded |
| 48 | 278 | 163 | 155.018 | false | occluded |

구 표면 fitting은 첫 frame에서 residual 44.4 mm > 허용 4 mm로 실패한다. 그러나 exclusion 자체가 중단되는 대신 centroid 기반 ball로 물러난다. 그 반지름이 약 15 cm라 기둥을 포함한다.

이어 `_nontarget_points`는 target 점을 non-target 근거에서 먼저 뺀다. 초기에 이 경로는 정확한 point index를 사용하지만 **잘못된 target 집합의 index**이므로 기둥의 보호 근거까지 제거한다. `ball_owned`는 이 오염된 target/non-target 증거로 ownership을 판정한다. 따라서 ownership을 켰다는 사실만으로 인접 장애물이 보존되지는 않는다.

`owned_unowned=remove`도 적용돼 있다. t=0 전체 제거 seed 3165개 중 2200개는 target-owned, 965개는 unowned이며 t=48은 각각 1747개와 983개다. 이 수치는 전체 ball 집계이므로 기둥 제거량을 두 원인으로 분해한 수치로 읽으면 안 된다. 증거가 부족한 seed를 제거하는 기본값은 추가 점검 대상이다.

**F1 설정은 켜져 있지만 1830의 7개 chunk 모두 `applied=false`다.** F1은 `surface_fit` 성공을 요구한다. 따라서 이 episode의 직접 원인을 “F1이 기둥 점을 새로 삭제했다”로 설명하면 틀린다. 기존 혼합 target 처리의 실패이며 F1이 해결하는 범위를 벗어났다. W1은 배치 후 처리이므로 이 파지 전 실패의 직접 해결책도 아니다.

근거 코드: [pipeline.py](../../../AG3S/ag3s/runtime/pipeline.py), `_nontarget_points`; [curobo_builder.py](../../../AG3S/ag3s/fields/curobo_builder.py), `_ball_owned_split`, `nontarget_surface_evidence`; [config.py](../../../AG3S/ag3s/config.py), `owned_unowned`.

## 3. 후반의 좋은 분리가 기존 오염을 복구하지 못한다

t=40, 48에는 attention split이 성공한다. 과일에 부합하는 98점 성분이 분리되고 ratio는 각각 222.189, 290.394다. 그럼에도 실제 manipulated geometry는 t=32의 혼합된 278점 집합을 유지한다.

기록상 association 후보는 약 39 × 41 × 250 mm의 기둥형 성분이며, 기존 혼합 기하 대비 middle extent 비율 0.346 / 0.350 때문에 거절된다. 결과는 `occluded`, `geometry_frame=4` 유지다. **좋은 split 출력이 나와도 그것이 실제 exclusion에 사용되는 대상 기하로 전달되지 않는 것**까지 확인된다. 과일 성분 자체가 association에서 거절됐다고 단정할 수는 없다. 후보 매칭과 challenger 전환 경로를 함께 점검해야 한다.

근거 코드: [target_grounding.py](../../../AG3S/ag3s/stages/target_grounding.py), `select`, `_associate`, `observe`.

개선 시 identity의 시간적 안정성과 extent의 정제를 구분해야 한다. 같은 물체 identity를 유지하면서, 처음 섞였던 이웃 성분은 나중의 명확한 증거로 제거할 수 있어야 한다. 단순히 latch를 끄거나 모든 geometry 축소를 허용하면 기존 가림 대응이 손상될 수 있다.

## 4. 세 episode 공통: 실제 손바닥 표면이 제약 구 밖에 있다

`V/p/start_server.sh`는 `--finger-cover both`를 켜지만 `--palm-cover on`은 사용하지 않는다. `serve_safe.py`의 기본 palm-cover는 off다. 일반 capsule에는 `--capsule-radius-scale 0.05`가 적용된다. finger cover의 정확한 구 반지름에는 이 축소가 적용되지 않는다.

기존 cover 진단에서 첫 contact 시점의 **실제 손–기둥 최근접 표면점**은 모델 구 밖에 다음만큼 남는다.

| episode | contact t | 실제 최근접 body | 구 union 밖 거리 (mm) |
|---|---:|---|---:|
| 1925 | 75 | EE_BODY_L | 14.26 |
| 1830 | 55 | EE_BODY_L | 32.12 |
| 1623 | 94 | EE_BODY_R | 26.02 |

보고된 12.7–38.8 mm는 접촉 직전 여러 시점까지 포함한 범위이며 전부 첫 contact 시점의 값은 아니다. 진단은 planning 시점 구를 palm body에 고정해 옮기는 방식이므로 gripper opening이 고정된 구간이라는 전제도 유지해야 한다.

1925에서는 기둥 ROI 제거 seed가 대부분 0이고 마지막 13개, 1623에서는 모두 0이다. 이 두 episode의 공통 설명으로 1830의 대규모 exclusion 실패를 그대로 적용하면 안 된다. 손 형상 누락이 우선 점검 대상이다. 다만 알려진 두 누락을 고치면 반드시 회피 성공한다는 인과 결론은 아직 없으며, 추종 오차·시간 사이 충돌·지도 오차가 남는지도 재검증해야 한다.

`--palm-cover on`은 측정된 EE_BODY 누락에 직접 대응하는 기존 기능이다. 그러나 기둥이 제거된 환경 지도에서는 더 정확한 손 모델도 장애물을 볼 수 없다. palm-cover 단독의 1830 개선을 보장할 수 없다.

근거: `V/diag/cover_ON_ep*.json`; [urdf_sphere_chain.py](../../../AG3S/ag3s/robot_models/urdf_sphere_chain.py), `PALM_BODY_ROLE`; [serve_safe.py](../../../AG3S/trajopt/serve_safe.py), `palm_cover_options`.

## 5. 정정: 4.7–5.2 mm는 10 mm margin 적용 후의 잔차다

`linearize._esdf_clearance_core`의 정의는 다음과 같다.

```text
constraint_clearance = d_esdf - sphere_radius - effective_margin
```

`sqp._finish`의 `to.metrics.clearance_m`는 이 제약들을 평가한 최악 행의 값이다. 따라서 **양수 4.7 mm를 다시 10 mm와 비교해 위반이라고 판단하면 margin을 두 번 적용하는 셈**이다. 해당 행이 obstacle로 분류되어 10 mm를 받았다면 모델 표면 간 거리는 약 14.7 mm라는 의미다. target/support 분류는 margin이 다르므로 정확한 원거리 복원은 그 행의 분류와 effective margin까지 확인해야 한다.

`budget_only`도 무조건 실행을 뜻하는 timeout 우회가 아니다. 실제 1830 마지막 chunk는 iteration cap 3에 도달했지만 최종 제약 검사를 통과했다는 이유로 실행됐다. 현재 증거는 **표현이 불완전해서 검사상 안전했던 것**을 뒷받침하며, 음수 위반을 gate가 무시했다는 증거는 아니다.

후속 기록에는 `d_esdf`, `radius`, `effective_margin`, `constraint_clearance`, `margin_class`, `link`, `tier`, `step`을 함께 남기는 것이 좋다. 기둥 행이 target/support로 오분류되어 margin을 면제받는 별도 문제가 있는지도 이 기록으로 구분한다.

근거: [linearize.py](../../../AG3S/trajopt/linearize.py), `_esdf_clearance_core`, `_obstacle_rows`; [sqp.py](../../../AG3S/trajopt/sqp.py), `_finish`; [safe_policy.py](../../../AG3S/trajopt/safe_policy.py), `_reasons`.

## 6. 다음 검증 순서

1. **기존 기록 정정:** margin 설명을 고치고 F1이 1830에서 미적용이었다는 사실을 추가한다. T58은 실패 사례로 보존할 가치가 있다.
2. **같은 기록 입력으로 원인 분리:** 현재 설정, palm-cover만, 정확한 target extent만, 둘 다의 2×2 진단. 정확한 extent는 가능하면 GT segmentation으로 만들되 진단 상한으로만 사용한다. 원본에 필요한 segmentation이 없으면 동일 조건에서 별도 수집하고 입력 차이를 명시한다. 지도 복원은 목표 identity뿐 아니라 실제 제거 영역을 통제해야 한다.
3. **혼합 target 경로 수정:** 불확실한 multi-component target을 전체 제거로 연결하지 않기, 후보 선택/승인 통계의 일관성 점검, 후속 split을 이용한 동일 identity의 extent 복구, unowned 처리의 독립 ablation. 구 fitting 실패만으로 모든 물체를 차단하는 방식은 비구형 target 회귀가 있어 피한다.
4. **동일 3 episode/seed/Rule B로 closed-loop 재검증:** 실제 접촉, task success, HOLD 및 지속시간을 함께 측정한다. HOLD 증가만으로 회피하며 작업도 성공했다는 주장을 해서는 안 된다. no-obstacle 회귀도 포함한다.
5. **남은 오차 확인:** 수정한 구가 실제 손 mesh를 덮는지 먼저 확인하고, 그 다음 지도 잔차·실행 추종·시간 사이 충돌을 점검한다. margin/iteration 증가만으로 원인을 대체하지 않는다.

Rule A로 옮기면 두 문제가 사라진다는 주장은 현재 선택 기록으로 입증되지 않는다. `V/raw/select_t58.json`에서 1830은 A/B의 XY가 같고 높이만 0.10/0.26 m로 다르며, 1623은 A/B가 같은 위치·높이다. 기존 Rule B를 유지한 수정 검증이 우선이다. 더 쉬운 조건은 필요하면 별도의 난이도 분석으로 다룬다.

논문에서는 이 3/3을 일반 성능 추정으로 사용하지 않는다. 현재 가장 정확한 설명은 **초기 target 혼합이 접촉 허용 영역을 오염시키고, 불완전한 손 형상이 실제 접촉을 숨기는 failure case**다. identity–extent–permission의 분리를 강화해야 한다는 연구 동기는 분명하지만, 개선의 contribution은 후속 비교로 입증해야 한다.

이번 작업은 본 진단 문서만 추가했다. runtime, T47 공용 로그, T58 원본 결과는 변경하지 않았으며 commit이나 다른 세션에 대한 통지는 수행하지 않았다.
