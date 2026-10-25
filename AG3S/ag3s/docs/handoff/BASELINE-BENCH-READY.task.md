# BASELINE-BENCH-READY — baseline 을 benchmark 실험에 넣을 수 있게: 인식 경로 수정 · CFM 범위 · privileged 변형 · job 생성

> writer: lead · 읽는 쪽: implementer L / M → verifier V6 · 작성 2026-10-25 KST
> 사용자 지시 2026-10-25: "권장하는 순서로 진행. baseline 방법들은 benchmark 에서 실험할 예정."
> 선행: `3412861c` (CLOSED-LOOP-FIX V5 pilot, 로그 BL-7).

## 권장 순서 (lead)

| 순서 | 무엇 | 왜 |
|---|---|---|
| 1 | **(a) 인식 경로 수정 + CFM action 범위 진단** | pilot 수치를 흐리는 구현 문제다. 검출 상자에 depth trim 이 없어 ellipsoid 가 최대 0.77 m 이고, CFM 은 executed \|xyz\| 가 7.28 까지 나왔다. benchmark 전체 실험에 들어가기 전에 고쳐야 한다 |
| 2 | **(c) privileged 변형 (GT 장애물)** | 원 논문 조건(AEGIS 의 VLM 이름, CFM 의 GT sphere 등)과 같은 행이 있어야 Tab. 4 "reported" 행과 비교할 수 있다. non-privileged 행과 나란히 둔다 |
| 3 | **(b) benchmark job 생성 연결** | 실제 suite × level 실험은 benchmark 단계에서 한다(사용자). 여기서는 `suites.json` / `suite_registry` / `eval_matrix` 로 모든 suite · level 의 job 을 만들 수 있게 하고, 각 조합의 1 episode smoke 까지만 한다 |

## Implementer L — 순서 1 (methods 쪽)

1. **box crop depth trim** — `methods/safety_common.py`
   - 검출 상자 안 점 가운데 상자 중심 물체의 depth 군집만 남긴다. 예: 상자 중앙 영역의 depth median ± k, 또는 연결 성분.
   - head 와 wrist 의 같은 이름 물체는 3D 거리로 association 한 뒤에 합친다.
   - 원 AEGIS 의 처리(`main_aegis.py` 의 crop · DBSCAN)와 맞춘다.
   - 옛 동작은 option 으로 남긴다 (`crop_mode="box"` / `"depth_trim"`). 기본값은 `depth_trim` 이다. V5 결과를 재현할 수 있게 V5 기준을 manifest 에 남긴다.
   - test:
     - 합성 장면(상자 안에 앞뒤 두 물체)에서 ellipsoid 가 앞 물체만 덮는다.
     - V5 smoke 의 call 0 반축 0.77 m 사례가 줄어든다 (CPU replay).
2. **CFM 의 executed \|xyz\| 7.28 진단과 수정** — `methods/constrained_flow_matching/`
   - V5 `pilot/served_range_by_call.txt` 의 call 을 CPU replay 로 재현한다.
   - 원인을 확인한다: QP 해에 action box 가 없는가, scale, depth cloud 장애물 크기 등.
   - 원 논문 / 원 repo 에 action 범위 제약이 있으면 그대로 넣는다. 없으면 lan 과 같은 방식(clean chunk 가 action box 안에 남도록)으로 막고, 원 방법과의 차이로 기록한다.
3. 고친 arm 외에는 served chunk 가 bit 동일해야 한다 (V5 replay 입력).
4. 산출물: `BASELINE-BENCH-READY-L.impl.md`. 수정 범위는 `methods/{safety_common.py,aegis,knows,multilink_cbf,constrained_flow_matching}/**` 와 tests 이다.

## Implementer M — 순서 2 · 3 (closed_loop 쪽)

1. **privileged 변형** (GT 장애물):
   - client 가 이미 아는 obstacle 정보를 server 로 넘기는 경로를 만든다. `libero_safe_client.py` 의 `ObstacleTracker` 는 obstacle 이름과 위치를 가지고 있다. 필요하면 SafeLIBERO env 의 obstacle geom 크기와 GT segmentation 도 넘긴다.
   - **client 수정은 flag 로만 한다** (`--emit-privileged-obstacles`, 기본 off, off 일 때 argv · 출력 byte 동일).
   - 수정 전에 같은 checkout 의 다른 benchmark session 에 알리고, 진행 중인 run 이 없는지 확인한다. session 이름은 `ListAgents` 로 찾는다. 지금은 `ag3s-vla-benchmark-90` 이며, 이름은 재시작마다 바뀐다.
   - server 는 wire 의 privileged key 를 `SceneObservation.extras["privileged/..."]` 로 넘긴다. adapter 들의 GT 변형이 그것을 읽는다.
   - 변형 이름 예시: `aegis:gt_obstacles`, `cfm:gt_spheres` (원 논문의 SafeLIBERO GT sphere 설정), `knows:gt`, `multilink_cbf:gt`, `nudge:gt`, `omniguide:gt`.
   - manifest 에 `privileged=true` 를 남긴다.
2. **benchmark job 생성** — `closed_loop/jobs.py`
   - `benchmark/configs/suites.json` + `suite_registry` 에서 suite × level × task × episode × arm job 목록을 만든다. 대상은 LIBERO 4 suite, SafeLIBERO 4 suite × L I / L II 이다. LIBERO-Safety 는 그쪽 hook 이 생기면 붙일 자리만 둔다.
   - 출력은 `run_parallel_gpu.sh` 형식이다.
   - split(dev / test) 규칙은 registry 를 따른다.
   - 공식 지표 집계(`closed_loop.metrics`)를 suite · level 별로 묶는 summary 를 만든다.
   - `benchmark/runners/eval_matrix.py` 를 import 해서 재사용할 수 있으면 재사용한다. 수정은 하지 않는다.
3. 산출물: `BASELINE-BENCH-READY-M.impl.md`. 수정 범위는 `Baseline/paper_baselines/closed_loop/**` 와 tests 다. client 는 위 flag 하나만 고칠 수 있다.

## Verifier V6 (L · M 이후, GPU)

1. 고친 arm(aegis · knows · multilink_cbf · CFM)의 smoke (SafeLIBERO spatial L I t0e0).
2. privileged 변형의 smoke, 각 1 ep.
3. `jobs.py` 로 만든 **suite × level 조합마다 1 episode smoke**. arm 은 `none` 과 대표 1–2 개로 한다. 전체 benchmark 실험은 하지 않는다.
4. figure (`figures/baselines/bench_ready/`):
   - crop 전후 ellipsoid 장면
   - CFM 범위 전후
   - privileged vs non-privileged 장면
   - suite × level smoke 표
5. 산출물: `BASELINE-BENCH-READY-V6.verify.json`
- GPU 규칙과 다른 session 알림은 이전과 같다. 이름은 `ListAgents` 로 찾는다.
