# SL4 — Panda 관측을 V11AB perception → ESDF에 연결 (ATS · LTE · ESDF), offline

> writer: lead (A0) · 읽는 쪽: implementer, verifier · 작성 2026-10-12 KST
> branch `feature/panda-benchmark`, 시작 HEAD `d1605030`. 계획: `doc/develope_PLAN_SUMMARY.md` §2.1 SL-4, §2.2.
> 사용자 판정 2026-10-12: SL3 확정, Panda attention cell = V11AB L8H2 유지(재선택 없음). LIBERO dev = init 0–4 (§6.0).

## 왜 지금 이것인가

SL1–SL3으로 Panda의 camera(depth·K·T), robot geometry(self-filter·constraint), attention(L8H2, 좌우 반전 해제)이 각각 V11AB 형식으로
준비됐다. SL4는 이것들을 **V11AB의 실제 perception 코드**(`AG3S.process_multi_debug`)에 넣어, ATS(target 추출) → target 제외 layer(LTE) →
cuRobo ESDF가 Panda/LIBERO scene에서 맞게 동작하는지 recorded dev frame으로 확인한다. Closed-loop(TrajOpt·action 변환)은 SL-5다.

```
recorded dev frame (wire npz: depth mm, K, T_base_cam, robot_state 9-D) + attention(L8H2, un-mirror)
  → split robot_state (q7 + finger) → AG3S.set_finger_joints → CameraObservation ×2 (head, left_wrist)
  → AG3S(robot_model=Panda self-filter, constraint_robot_model=Panda gripper, attached_parent_links=Panda, config=V11AB + Panda override)
  → process_multi_debug(phase="approach", active_manipulators=["left"])
  → grounding(ATS) · target-free layer(exclude-authorized) · CuroboEsdfField · CollisionConstraintSet
```

## 확인된 사실 (audit, read-only — 2026-10-12)

| 항목 | 근거 |
|---|---|
| entry: `AG3S.process_multi_debug` → `(CollisionConstraintSet, debug)`; serve_safe도 이 경로(`SafePolicy._scene_fn`) | `AG3S/ag3s/runtime/pipeline.py:1308`, `AG3S/trajopt/safe_policy.py:837` |
| V11AB build 설정: voxel 0.020, max_distance 0.4, fine 0.005, tsdf 0.005, `exclude_support_surfaces=False`, backend curobo, `attached_sign_threshold_voxels=1.5`, `range_max=2.0`, `--links gripper`, `--target-field-policy exclude-authorized` | `AG3S/trajopt/serve_safe.py:891-897`, `scripts/benchmark/v11ab/start_server.sh` |
| V11AB client는 `--safe-phase approach --safe-manipulators $SIDE`를 보낸다 → grasp phase 없음, exclude-authorized는 active hand에 적용 | `scripts/benchmark/v11ab/run_client.sh:12`, `safe_policy.py:135,839` |
| 주입 가능: `AG3S(robot_model, constraint_robot_model, attached_parent_links)`, config `contact.contact_links`, `esdf.bounds_lower/upper`, `set_finger_joints` | `pipeline.py:135-148, 492`, `config.py:865, 1524` |
| wire는 `head`/`left_wrist` 이름을 그대로 `CameraID`로 받는다 | `AG3S/trajopt/wire.py:538-548`, `types.py:207-224` |
| robot_state 길이 = model nq(7) 검사 → 9-D는 split 필요 | `AG3S/ag3s/runtime/multiview.py:145-147, 163-165`; `AG3S/adapters/panda/model.py:269` |
| ESDF 기본 box (−0.3,−0.9,0.0)–(1.2,0.9,1.6) base frame; LIBERO table top = base z −0.012 m → table이 잘림 | `AG3S/ag3s/fields/esdf.py:777-790`; `SL1.verify.json` |
| attention: camera별 raw softmax → fusion에서 voxel별 max → fused cloud 전체 percentile(5,99) 정규화 → seed = 95th percentile | `multiview.py:378-399`, `stages/attention_lifting.py:240-286`, `stages/target_grounding.py:269-297` |
| subtask label 없음(Panda) → subtask gate 비활성(ungated) | `stages/subtask_probe.py:153-196` |
| online server의 attention alias·flip, robot model, chunk layout은 code 변경이 필요 → **SL-5** | `serve_safe.py:839-849, 1037-1058, 2346`, `safe_policy.py:256` |

## 구현에 요청하는 것 (→ implementer)

**원칙: AG3S core(`AG3S/ag3s/**`, `AG3S/trajopt/**`) 파일은 수정하지 않는다.** 모든 Panda 차이는 생성자 인자·config dict·adapter 함수로 넣는다.

1. **Panda AG3S profile** (`AG3S/adapters/panda/profile.py`): V11AB server와 같은 AG3S를 만드는 함수.
   - config는 **serve_safe의 V11AB 설정에서 유도**한다: `scripts/benchmark/v11ab/start_server.sh`의 flag를 serve_safe 자신의 argparse로 parse하고,
     `build_ag3s`가 쓰는 것과 같은 값으로 config dict를 만든다. 복제가 불가피한 값은 test가 `build_ag3s` 경로와 field별로 비교한다.
   - Panda override는 다음만: robot_model(self-filter 기본 cover), constraint_robot_model(`--links gripper` = `PANDA_GRIPPER_LINKS`),
     `attached_parent_links`(Panda grasp/hold link), `contact.contact_links = {"left": PANDA_CONTACT_LINKS}`, **ESDF `bounds_lower` z만 −0.10 m**
     (나머지 bound는 V11AB 기본값). 각 override의 근거(좌표/robot 차이)를 docstring과 manifest에 기록.
   - `diff_against_v11ab()`: V11AB config와의 차이 목록을 반환(위 override 외에는 비어야 함).
2. **Offline harness** (`benchmark/runners/perception_replay.py`, `.venv-openpi-live`, GPU):
   - 입력: SL3 dev rollout run(geom npz) + SL3 attention audit run(per-call L8H2). policy call 시점 frame(221)마다:
     wire camera 관측 → `unpack_camera_observations` → robot_state split → `set_finger_joints` → attention un-mirror를 `CameraObservation.attention`에 부착 →
     `process_multi_debug(phase="approach", active_manipulators=["left"])`.
   - `--attention-mapping {unmirror,mirror}`로 비교 실행 가능(진단용).
   - 저장(새 run, `create_run`, source run id·SHA 기록): frame별 grounding 결과(status, target points/centroid, seed camera 기여), fused/filtered cloud 요약,
     ESDF를 GT 점·sphere 위치에서 query한 값(main layer, target-free layer), constraint set 요약, stage별 시간. 큰 field 자체는 저장하지 않는다(필요 시 slice만).
   - **GPU 단계는 `flock /tmp/ag3s-vla-gpu-validation.lock` 안에서**, 시작 전 `nvidia-smi --query-compute-apps` 확인(CLAUDE.md).
3. **GT 도구** (py3.8 쪽 export는 기존 SL3 GT export 재사용 가능): frame별 물체(target, plate, bowl_2, cookies, ramekin, stove, cabinet, table) 표면 sample 점(base frame)과
   MuJoCo 기준 gripper constraint sphere–물체 정확 거리(`mj_geomDistance` 양수 구간 또는 해석적 box 거리). SL2 verifier의 방법을 따른다.
4. **Test (CPU)**: profile config vs V11AB diff, robot_state split·finger 적용, attention 부착 방향, ESDF bound 적용, harness dry-run.

**손대지 않을 것:** `AG3S/ag3s/**`·`AG3S/trajopt/**` 기존 파일, V11AB 설정값(override 목록 외), submodule, storage input.
serve_safe online 연결은 SL-5. Core 변경이 꼭 필요하다고 판단되면 구현하지 말고 impl.md에 근거와 함께 적는다.

## 측정에 요청하는 것 (→ verifier)

Dev 데이터만 사용(SL3 dev run, init 0). 설정을 고르지 않는다.

| # | 무엇을 재나 | 어떤 숫자로 | 통과 기준 |
|---|---|---|---|
| 1 | V11AB 동일성 | core diff 0줄; profile config의 V11AB 대비 차이 목록; RB-Y1 legacy 16D + **cuRobo 16D** baseline | 차이 = 선언한 override만; baseline 둘 다 일치 |
| 2 | ATS (target 추출) | frame별 has_target, target 점의 GT target 표면 대비 precision/recall(거리 ≤ 10 mm), centroid 오차(mm); phase·task별 | 숫자만 |
| 3 | camera 기여 | seed 점 중 head/left_wrist 비율, head attention이 seed를 만든 frame 비율 | 숫자만 (SL3 열린 항목: head block mass) |
| 4 | wrist 반전 진단 | phase별로 un-mirror/mirror mapping의 attention mass가 target·plate(destination)·gripper·기타에 어떻게 나뉘는지; mirror mapping으로 harness를 돌렸을 때 #2 비교 | 숫자만 (SL3 열린 항목) |
| 5 | 장애물 보존 · LTE | 비target 물체(plate, bowl_2, cookies, ramekin, stove, cabinet, table) GT 표면 점의 ESDF 값: main layer와 target-free layer 각각 occupied(≤ fine voxel) 비율; target GT 점의 target-free layer 거리. 특히 target ball 제거가 plate·bowl_2를 지우는지 | 비target 보존율, target 제거율 |
| 6 | field 정확도 | gripper constraint sphere 중심에서 ESDF 값 − MuJoCo 정확 거리(sphere 반경 반영): 두 layer, 0.4 m 이내 구간 | median/p95 오차(mm) |
| 7 | ESDF 범위 | bounds z −0.10 적용 전/후 table 표면 표현, box 밖으로 나간 점 비율 | 숫자만 |
| 8 | mount 잔여 | self-filter 후 남은 mount pedestal 점이 field에 들어가는지, gripper/arm sphere까지 거리 | 숫자만 |
| 9 | 비용 | stage별(self-filter 4271 sphere, fusion, grounding, ESDF) 시간 p50/p95(첫 호출 제외), GPU memory | 측정만 (E10) |
| 10 | 회귀 | `scripts/testing/run_cpu_regression.sh`, SL0 경로 보존 | 기존과 동일 |

- figure 3종(`sl4-*`): (1) 실제 scene — 대표 frame(phase별)의 fused cloud + ATS target 점 + GT target, ESDF slice(위치 배치도 포함) main vs target-free,
  (2) graph — frame별 ATS precision/recall·centroid 오차, 장애물 보존율, field 오차 분포, stage 시간, (3) table — override 목록, 지표 요약.
- raw: `outputs/verify/SL4/`. GPU는 lock 안에서, 측정 job은 foreground, CPU RSS ≤ 3 GB, worker ≤ 2.

## 이 STEP 이 답하는 질문

V11AB perception 코드를 수정하지 않고 Panda 입력만 바꿨을 때, LIBERO scene에서 target bowl을 추출하고(ATS), target만 제외한 layer를
만들면서 다른 장애물은 보존하고(LTE), gripper 주변 거리장이 실제 거리와 mm 수준으로 맞는가?

## 되돌아올 지점

- #2에서 ATS가 target을 충분히 잡지 못하면 → 원인 분리(attention vs 기하). attention이 원인이면 dev(init 0–4) calibration variant를 사용자에게 제안(SL3 판정).
- #5에서 target ball이 plate 등 인접 장애물을 지우면 → V11AB `_target_ball` 반경/규칙이 Panda scene 스케일에 맞는지 확인. 변경은 variant이므로 사용자 판정.
- #1에서 baseline이 어긋나면 즉시 중단(core는 바뀌지 않았어야 한다).
- ESDF box를 z 외에도 바꿔야 하면 근거(측정)와 함께 사용자 판정.
