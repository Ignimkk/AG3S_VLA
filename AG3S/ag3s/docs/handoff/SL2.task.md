# SL2 — Panda geometry: sphere chain · self-filter · protected/contact link (+ SL1 후속 2건)

> writer: lead (A0) · 읽는 쪽: implementer, verifier · 작성 2026-10-12 KST
> branch `feature/panda-benchmark`, 시작 HEAD `6c7ce76f`. 계획: `doc/develope_PLAN_SUMMARY.md` §2.1 SL-2, §2.2.
> 사용자 판정 2026-10-12: SL1 확정. adapter K principal point = pixel-centre `(W−1)/2`; `robot_state`를 image 시점으로 수정.

## 왜 지금 이것인가

SL1으로 LIBERO camera 관측이 V11AB wire 형식(OpenCV, robot base frame, metric depth)으로 정확히 옮겨졌다.
V11AB는 이 depth에서 **robot 자신을 지우고**(self-filter, `DepthRobotMask`), 남은 점으로 ESDF를 만들고,
**robot의 sphere chain**으로 clearance constraint를 건다. 둘 다 robot geometry가 필요한데 V11AB의 기본값은 RB-Y1이다.
SL2는 Panda geometry를 V11AB가 쓰는 같은 구조로 만들고, 그것이 실제 simulator의 Panda와 맞는지 측정한다.
Action 변환(SL-5), attention(SL-3), ESDF 연결(SL-4)은 이번 범위가 아니다.

```
robot_state(q7 + finger2, image 시점) ─FK─▶ Panda UrdfSphereChain
        ├─ self-filter model (전신 + 덮개 capsule)  → DepthRobotMask → 로봇 점 제거
        └─ constraint model (arm + gripper link)   → clearance constraint (SL-4/5에서 연결)
```

## 확인된 사실 (audit)

| 항목 | 근거 |
|---|---|
| sphere chain은 범용: `UrdfSphereChain(model, joint_names, extra_capsules=..., link_filter=..., fixed_joint_values=...)` | `AG3S/ag3s/robot_models/urdf_sphere_chain.py:546-560` |
| URDF `<capsule>`만 읽음 → franka URDF는 mesh뿐이라 0 sphere | `urdf_sphere_chain.py:315-347` |
| RB-Y1 self-filter = URDF capsule + gap/덮개 capsule(MJCF mesh fit); constraint model은 덮개 없이 `link_filter` | `AG3S/ag3s/experiments/reports/grounding_report.py:71-101, 146` |
| link 집합 정본: `PALM_LINKS`, `FINGER_LINKS`, `GRIPPER_LINKS = PALM+FINGER`(= `--links gripper`, V11AB 기본), `ARM_LINKS` | `grounding_report.py:104-127` |
| target 접촉 허용 link: `DEFAULT_CONTACT_LINKS = {right: (ee_right, ee_finger_r1, r2), left: (...)}` | `AG3S/ag3s/config.py:826-832` |
| Panda 자산: cuRobo `franka.yml`(sphere), `franka_panda.urdf`; robosuite Panda MJCF + PandaGripper | `third_party/curobo/curobo/content/...`, `.venv-libero/.../robosuite/models/assets/robots/panda/` |

## 구현에 요청하는 것 (→ implementer)

### 0. SL1 후속 (사용자 확정)
1. `benchmark/adapters/libero_panda/camera.py`: K principal point를 `(W−1)/2, (H−1)/2`로. 규약 이름과 값을 `conventions()`·manifest에 기록.
2. `robot_state`를 **image 시점** q로: SL1 verifier 측정상 image는 `forward(mj_integratePos(qpos, qvel, −dt))` 상태와 bit-identical,
   기록된 robot_state는 `sim_state.qpos`(1 substep 늦음). image 시점 q(7 arm + 2 finger)를 기록하도록 고친다. Test: 이 q로 MuJoCo forward한
   wrist camera pose가 기록된 `T_world_cam`과 일치.

### A. Panda robot model (Python 3.11, `AG3S/adapters/panda/` — 새 파일. 기존 `AG3S/ag3s/**` algorithm 파일은 수정하지 않는다)
3. Kinematics: LIBERO가 실제로 쓰는 robosuite Panda(+PandaGripper, LIBERO `MountedPanda`)와 같은 FK를 내는 URDF를 정한다.
   후보는 pinned cuRobo `franka_panda.urdf`(submodule, 수정 금지 — 경로 참조)이고, base(`robot0_base` ↔ `panda_link0`)·hand 장착 offset·finger joint
   대응이 다르면 adapter 쪽 URDF(`robots/panda/description/`에 provenance 주석과 함께)로 보정한다. joint 이름 순서와 robosuite qpos 대응표를 만든다.
4. Geometry, V11AB와 **같은 구조**로:
   - constraint model: cuRobo `franka.yml` collision sphere를 zero-length `UrdfCapsule`로 `extra_capsules`에 넣는다(출처·sphere 수 기록).
   - self-filter model: 위 + robosuite Panda **collision mesh**에서 덮지 못한 표면을 덮는 capsule(RB-Y1 `self_filter_covering_capsules`와 같은 원칙,
     덮개는 constraint model에 넣지 않음). finger 개도는 parameter joint(`robot_state`의 finger qpos).
   - link 집합: `PANDA_PALM_LINKS`, `PANDA_FINGER_LINKS`, `PANDA_GRIPPER_LINKS`, `PANDA_ARM_LINKS`(구동 관절 link1–7 + gripper; 고정 link0 제외 — RB-Y1이
     결정 변수 아닌 link를 constraint에서 뺀 원칙과 같다), contact-allowed links. RB-Y1 집합과의 역할 대응표를 module docstring에 둔다.
5. Manipulator side: Panda 단일 팔을 **`left`** 로 둔다. 근거: SL1에서 wrist camera를 pi05_libero token 슬롯에 맞춰 `left_wrist`로 정했으므로
   camera 장착 팔과 contact-link side가 일치해야 한다; V11AB 16-D layout `[L7, Lg, R7, Rg]`의 L 슬롯에 대응(SL-5에서 사용). manifest에 기록.
6. Builder: `build_panda_self_filter_model()`, `build_panda_constraint_model(link_filter=...)` + `DepthRobotMask`에 넣어 쓸 수 있는지 확인하는 얇은 연결 함수.
   V11AB server 연결(`serve_safe.build_ag3s`)은 하지 않는다 — RB-Y1 hardcode 지점 목록만 impl.md에 적는다(SL-4/5 입력).

### B. Test (CPU)
7. FK: 3–4개 고정 q에서 Panda chain link frame vs robosuite MuJoCo body pose(MuJoCo는 `.venv-libero` subprocess로 fixture 생성, 작은 npz)가 일치.
8. link 집합 무결성(오타 시 error), contact link ⊂ gripper ⊂ arm, sphere 수/반지름 > 0.

**손대지 않을 것:** `AG3S/ag3s/**`·`AG3S/trajopt/**` 기존 파일(V11AB algorithm·config), `robots/rby1/**`, submodule(cuRobo, LIBERO 등) source, storage input.
불가피하게 AG3S core에 주입 지점이 필요하면 **구현하지 말고** impl.md에 필요 사항으로 적는다(lead·사용자 판정).

## 측정에 요청하는 것 (→ verifier)

새 replay run(`launch_libero_replay.py --capture-geometry --capture-float-depth --capture-segmentation`, SL0 run의 t0/e0)을 만든 뒤:

| # | 무엇을 재나 | 어떤 숫자로 | 통과 기준 |
|---|---|---|---|
| 1 | SL1 후속 | (a) image 시점 robot_state로 forward한 wrist pose vs `T_world_cam`, (b) pixel-centre K로 SL1 #2·#4 재측정 | (a) ≈ 0, (b) SL1 표와 나란히 |
| 2 | FK 일치 | 114 frame × link(link1–7, hand, fingers) 위치 오차(mm)·회전 오차(deg) max/p95 | ≤ 1 mm / ≤ 0.1° 목표, 초과 시 link별 원인 |
| 3 | sphere coverage | robosuite collision mesh 표면 sample 중 sphere union 안 비율(link별, 두 model), 밖 최대 거리; sphere가 mesh 밖으로 나간 두께 p95 | 숫자만 (판정은 lead·사용자) |
| 4 | self-filter | SL1 depth에 `DepthRobotMask`(Panda self-filter model): robot segmentation pixel 제거율(camera별), non-robot pixel 오제거(물체별 — 특히 grasp 중 target bowl, table) | 숫자만; target 오제거 frame은 따로 표시 |
| 5 | contact 장면 | SL0 trajectory의 MuJoCo contact(link × 물체) vs constraint sphere의 물체 geometry 대비 signed distance: 접촉 frame에서 sphere 침투 여부, 비접촉 frame의 최소 clearance. 추가로 금지 접촉 fixture(관절 설정으로 link5–7/hand를 장애물에 넣은 정적 장면 3개 이상) | link별 confusion table (MuJoCo contact × sphere 침투) |
| 6 | 회귀 | `scripts/testing/run_cpu_regression.sh` + legacy 16D baseline; SL0 경로 보존(flags off replay = SL0 npz) | 기존과 동일 |

- figure 3종(`sl2-*`, 기존 이름과 겹치지 않게): (1) 실제 scene — robosuite Panda mesh 위 sphere overlay(두 model, 정면/측면), self-filter 전후 point cloud,
  contact fixture 장면, (2) graph — FK 오차 추이, coverage 분포, self-filter 제거율, (3) table — link 집합 대응(RB-Y1 ↔ Panda), sphere 출처, confusion table.
- raw: `outputs/verify/SL2/`.

## 이 STEP 이 답하는 질문

V11AB와 같은 구조(self-filter 전신 + constraint gripper/arm)의 Panda sphere model이, LIBERO simulator의 실제 Panda 위치·형상과 맞아서
robot 점을 지우되 target을 지우지 않고, 실제 접촉을 sphere 침투로 나타내는가?

## 되돌아올 지점

- FK 불일치가 장착 offset으로 설명되지 않으면 → cuRobo URDF 대신 robosuite MJCF에서 URDF를 생성하는 방식으로 전환.
- `franka.yml` sphere coverage가 robosuite collision mesh와 크게 다르면 → MJCF mesh fit sphere로 constraint model을 만드는 variant를 별도 이름으로 비교(V11AB main에는 하나만).
- self-filter가 grasp 중 target을 지우면 → RB-Y1 T17(손목 capsule이 사과를 삼킨 건)과 같은 원인인지 확인, 덮개 반경·finger parameter 재검토.
- Manipulator side `left` 선택은 SL-3/SL-5에서 V11AB의 camera↔arm 결합 audit 결과와 다르면 adapter 표만 바꾼다.
