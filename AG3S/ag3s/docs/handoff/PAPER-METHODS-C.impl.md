# PAPER-METHODS-C — 구현 (safety filter 계열: interface · KNOWS · VLSA/AEGIS · multi-link)

> writer: ag3s-implementer (Implementer C) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-23 (KST 15:50 경)
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, 선행 commit `bf1403d5`. commit 하지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

`Baseline/paper_baselines/methods/` 에 공통 interface 를 먼저 만들고 (D 가 이 spec 으로 동시에 작업), safety filter 세 개를 그
interface 뒤에 두었다. KNOWS 는 `git mv` 로 옮기고 adapter 만 붙였다. VLSA/AEGIS 는 SafeLIBERO submodule 의 공식 함수를 수정 없이
import 하고, script 안에 있는 per-step QP 는 원 줄 번호를 달아 옮겼다. VLM(GLM API) 단계는 쓰지 않고 privileged 이름 / GT 분할 /
Grounding DINO 로 바꿨다. Multi-link filter [1] 는 공개 code 가 없어 논문(arXiv HTML v1)대로 구현했다 — AEGIS barrier 를 Panda 의
5 개 link ellipsoid (adapter sphere model 에서 fit) 로 확장하고 `G_k = J_k J_ee^+` 로 EE twist 에 연결했다.
세 filter 모두 recorded SL3 frame 의 기록된 action chunk 를 끝까지 처리해 수정된 chunk 를 낸다 (CPU smoke, 아래).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/methods/interface.py:36-84` | `SceneObservation` · `FilterResult` · `SafetyFilter` · `SamplingGuidance` (task spec 그대로) | 공통 interface (D 와 공유) |
| `interface.py:86-131` | `validate_filter_result`, `unnormalize_actions` / `normalize_actions` (openpi `Normalize` 와 같은 식: q01/q99 → quantile, 아니면 mean/std; `xp=np|jnp`, pad 차원 유지) | contract 검사, guidance 의 norm 처리 helper |
| `interface.py:152-200` | registry `get_method` / `register_method` / `available_methods`. 조회 순서: 등록 → `BUILTIN_METHODS`(knows/aegis/multilink_cbf) → 관례 `methods.<name>` 또는 `.adapter` import (등록 또는 `make_method`) | D 가 interface.py 를 고치지 않고 등록 가능 |
| `interface.py:30` | `PRIVILEGED_PREFIX = "privileged/"` — extras 의 privileged key 접두사 | privileged 입력을 grep 가능하게 |
| `methods/__init__.py` | 신규 (1 줄) | package |
| `Baseline/knows_vla/** → methods/knows/**` | `git mv` (28 개 rename, history 보존) | task 1 |
| `methods/knows/*.py`, `docs/*.md`, `README.md` | `Baseline.knows_vla` → `Baseline.paper_baselines.methods.knows`, `Baseline/knows_vla` → 새 경로 (import, `--camera-params` 기본값, `LIBERO_CONFIG_PATH` 안내) | 옮긴 뒤 import 경로 |
| `Baseline/knows_vla/__init__.py` (신규, 47 줄) | **호환 alias package**: `Baseline.knows_vla.<sub>` → 새 module 과 **같은 module 객체** (MetaPathFinder) | 수정 금지 파일이 옛 이름을 import 한다: `AG3S/ag3s/stages/geometry.py:188` (`mvee` 재사용; 없으면 조용히 covariance fit 으로 바뀜), `benchmark/__init__.py:12`, `scripts/setup/check_workspace.py:19` |
| `methods/knows/adapter.py` (144 줄) | `KnowsSafetyFilter` — `cbf/filter.py` 의 `SafetyFilter` 를 interface 로 감쌈 (offline_filter 규칙: EE ellipsoid 0.04/0.04/0.07, 물체 MVEE, shape 고정 + centroid 추적, 0.05 m / 0.5 rad 단위) | task 1 adapter |
| `methods/knows/README.md` | 맨 앞에 PAPER-METHODS 절 추가 (원 README 는 아래에 보존) | method README |
| `methods/safety_common.py` (427 줄) | SO(3) helper, `propagate_ee` (실측 gain), `backproject`, `ObstacleCloudSource` (`given` / `gt_seg` / `detector` + robot self-filter), `load_recorded_call` (SL3 `_geom.npz` → `SceneObservation`) | 세 filter 공용 · smoke 입력 |
| `methods/aegis/official.py` (82 줄) | 공식 `main/utils.py` 를 그대로 import (robosuite 는 import 동안만 stub, `__pycache__` 를 submodule 에 쓰지 않음) | 공식 함수 재사용, submodule 무수정 |
| `methods/aegis/perception.py` (172 줄) | `filtering_points` 이식 (utils.py:257-316; open3d DBSCAN → scipy), 공식 `fit_ellipse` 로 MVEE, `perceive_obstacles` | AEGIS 물체 기하 |
| `methods/aegis/adapter.py` (200 줄) | `AegisSafetyFilter` — main_aegis.py:149-164, 184-233, 301-354, 378-388 이식 (줄 번호 주석) | task 2 |
| `methods/aegis/README.md` | 출처 · license · 차이 8 개 · 실행 명령 | method README |
| `methods/multilink_cbf/robot.py` (197 줄) | `PandaGuardedBodies`: gripper / wrist / forearm×3 ellipsoid (adapter sphere model 의 MVEE, coverage 보정, fit cache), FD Jacobian | task 3 |
| `methods/multilink_cbf/adapter.py` (187 줄) | `MultiLinkCBFFilter` — 논문 Eq. 2 (W=I/25, W_z=I, α=10, OSQP), `barrier_rows_world` (공식 AEGIS 계수 → base frame) | task 3 |
| `methods/multilink_cbf/README.md` | 논문 대비 표, 생략한 것 (moving-hazard optical flow) | method README |
| `methods/smoke_safety_filters.py` (185 줄) | recorded-frame smoke (json · npz · png · summary.json) | task 5 |
| `methods/tests/test_safety_interface.py`, `aegis/tests/test_aegis_filter.py`, `multilink_cbf/tests/test_multilink_filter.py`, `knows/tests/test_knows_adapter.py` | unit tests (아래) | task 5 |
| `setup/requirements-baselines.in` / `.txt` | `cvxpy==1.7.3`, `osqp==1.1.3` + 의존 6 개 추가 (기존 줄 변경 없음) | `.venv-baselines` 설치 기록 |

수정하지 않음: `AG3S/**`, `benchmark/**` (SafeLIBERO 포함), `third_party/**`, A/B 의 `providers` · `eval` · `interface.py`, `doc/migration/validation/*.json`.

### `.venv-baselines` 설치 (C 만)

```bash
# constraints = 설치 직전 freeze 의 numpy==2.2.6 scipy==1.16.1 torch==2.7.1 torchvision==0.22.1 transformers==4.53.2
$AG3S_STORAGE_ROOT/tools/uv pip install --python .venv-baselines/bin/python -c <constraints> "cvxpy==1.7.3" "osqp==1.1.3"
```
결과: 신규 8 개 (`cffi 2.1.1, clarabel 0.11.1, cloudpickle 3.1.2, cvxpy 1.7.3, joblib 1.6.0, osqp 1.1.3, pycparser 3.12, scs 3.3.1`),
변경 · 재설치 0 (freeze 전후 diff = 추가 8 줄뿐). **C 의 설치는 cvxpy/osqp 신규 8 개뿐, sam2 무관** (lead 확인 2026-10-23).
setup_venv.sh · `-r` · `--reinstall` 은 실행하지 않았다. 이후 추가 설치 없음.

## 단위 검증

KNOWS 이동 전후 (`.venv-ag3s`, 같은 명령, 경로만 다름):
```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest <knows>/tests -q -rA
```
| | test_cbf.py | test_mujoco_bridge.py | 합계 |
|---|---|---|---|
| 이동 전 `Baseline/knows_vla/tests` | 42 passed | 7 passed | **49 passed** |
| 이동 후 `methods/knows/tests` (원 2 파일) | 42 passed | 7 passed | **49 passed** — test id 집합 동일 (경로 치환 후 diff 없음) |

`.venv-baselines` (mujoco 없음 → mujoco_bridge 7 개 skip, osqp 설치 전에는 collection error):
```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest Baseline/paper_baselines/methods/tests \
  Baseline/paper_baselines/methods/aegis/tests Baseline/paper_baselines/methods/multilink_cbf/tests \
  Baseline/paper_baselines/methods/knows/tests -q
```
→ **74 passed, 1 skipped** (C 의 신규 32 + KNOWS 원 42; skip = test_mujoco_bridge module).

그 밖 (alias 확인): `tests/ag3s/test_geometry.py` 47 passed (`.venv-ag3s`; `fit_ellipsoid_mvee` 가 alias 를 통해 KNOWS `mvee` 를
실제로 import 함을 확인), `Baseline/paper_baselines/tests/test_interface.py` 8 passed, `scripts/setup/check_workspace.py` 의
`Baseline.knows_vla` 항목 통과.

C 신규 test 가 보이는 것:

| file | 확인 |
|---|---|
| `test_safety_interface.py` (11) | registry 3 filter 해석 · Protocol 만족 · 미등록 KeyError; normalize round-trip (quantile/z, pad 유지); contract 위반 검출; SO(3); `propagate_ee`; pinhole; recorded call 의 EE 위치 = Panda FK grip site (<2 mm); gt_seg 에서 target·destination·table 제외; detector crop 경로 (fake detector) + robot self-filter 후 남은 점이 모든 sphere 밖 |
| `test_aegis_filter.py` (10) | 공식 함수가 submodule 파일에서 옴; 공식 gradient (a_v, a_w, a_uz) = 유한차분 (body frame, `@R^T` → base); QP = 닫힌형 반공간 투영; **장애물 1 개 + 직진: 위반 chunk 는 수정 (a_x ≤ 10h, h>0 단조 감소), 안전 chunk 는 bit 동일**; 장애물 없음 → identity; reset 전까지 장애물 고정; 키 큰 물체 Q1; 합성 상자 filtering+fit; recorded frame end-to-end |
| `test_multilink_filter.py` (6) | 5 body 이름 · 모든 sphere 중심이 ellipsoid 안; Jacobian = FK 차분; **constraint row 가 h 변화량을 예측** (`a_u G u dt`); 위반 → 수정 · 반대 방향 → bit 동일; **forearm 위 hazard: multi-link 은 개입, EE-only AEGIS 는 개입 없음**; recorded frame end-to-end |
| `test_knows_adapter.py` (5) | relaxed/fixed 모두 위반 chunk 수정, e-stop 없음, gripper 불변; fixed 에서 이산 CBF `h1 ≥ (1-γ)h0`; 안전 chunk bit 동일; no-obstacle; recorded frame 에서 shape 고정 |

## Recorded-frame smoke (CPU, 1 process, OMP 1 thread, peak RSS 2.4 GB)

```bash
source Baseline/paper_baselines/setup/paper_baselines_env.sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD .venv-baselines/bin/python -m \
  Baseline.paper_baselines.methods.smoke_safety_filters --run 20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0 \
  --stem libero_spatial_t0_e0 --calls 0 12 --arms knows aegis multilink_cbf aegis:detector_names aegis:detector_all
```
출력: `outputs/paper_baselines_smoke/methods/` — `libero_spatial_t0_e0_c{0,12}_<arm>.json|npz`, `libero_spatial_t0_e0_c{0,12}.png`,
`summary.json`. 입력 = 그 call 의 기록된 pi0.5 chunk (10×7). 모든 arm 에서 contract ok, gripper 차원 불변.

| call (obs_index) | arm | obstacle source | ms | intervened | max \|Δa_xyz\| | min h | obstacles |
|---|---|---|---|---|---|---|---|
| 0 (10) | knows | GT seg (priv.) | 497 | 0/10 | 0.000 | 0.130 | 5 |
| 0 (10) | aegis | GT seg (priv.) | 1288 | 0/10 | 0.000 | 0.104 | 5 |
| 0 (10) | multilink_cbf | GT seg (priv.) | 918 | 0/10 | 0.000 | 0.093 | 5 |
| 0 (10) | aegis:detector_names | GDINO-tiny + LIBERO 이름 (priv.) | 57072 | 0/10 | 0.000 | 0.140 | 1 |
| 0 (10) | aegis:detector_all | GDINO-tiny "object" − target | 21336 | 0/10 | 0.000 | 0.140 | 1 |
| 12 (70) | knows | GT seg | 392 | 10/10 | 1.118 | −0.051 | 5 |
| 12 (70) | aegis | GT seg | 1094 | 9/10 | 0.435 | 0.035 | 5 |
| 12 (70) | multilink_cbf | GT seg | 666 | 8/10 | 0.400 | 0.041 | 5 |
| 12 (70) | aegis:detector_names | GDINO-tiny + 이름 | 58645 | 10/10 | 0.777 | −0.032 | 4 |
| 12 (70) | aegis:detector_all | GDINO-tiny "object" | 17821 | 10/10 | 0.713 | −0.022 | 1 |

(Δa 단위 = LIBERO action unit; min h 는 각 method 자기 barrier 의 m 값 — method 끼리 정의가 달라 직접 비교하지 않는다.
ms 의 대부분은 첫 call 의 장애물 fit · cvxpy import · CPU GDINO.) call 12 는 bowl 을 plate 와 ramekin 사이에서 집으러 내려가는 구간이며
AEGIS 의 binding row 는 ramekin. 별도 scan (t0_e0 21 call, t3_e0 18 call, 연속 episode 상태)에서도 개입은 접근 구간 (t0: call 11-13,
t3: call 6-8) 에 모였다.

## 원 방법과의 차이 (요약; 상세는 각 README)

| method | 차이 |
|---|---|
| KNOWS | 공식 code 없음 (우리 재구현, 이동만). target 제외는 attention (Eq. 4) 대신 privileged BDDL target+destination (`exclude="task_objects"`; `"phase_target"` = offline_filter 규칙). 물체 분할 = GT seg. chunk 의 뒤쪽 action 은 실측 gain 0.0115 m/unit 으로 예측한 EE 자세에서 filter |
| VLSA/AEGIS | **GLM-4.5V API 미사용** → (a) privileged 장애물 이름 + GDINO, (b) GDINO 일반 prompt − target box, (GT) seg mask. 장애물 여러 개면 row·z 를 장애물마다. box crop 에서 Panda sphere(+2 cm) 안의 점 제거 (원본에 없음). open3d DBSCAN → scipy. 원본은 agentview+backview 1024 px, 우리는 head 256 px. QP 실패 시 원본은 NameError 로 episode 종료 → 여기선 nominal + log. 비활성 row 면 nominal 을 정확히 반환. 공식 code 의 회전 규약(OSC world-frame 회전을 body-frame ω 로 사용)은 그대로 둠 |
| Multi-link [1] | 공개 code 없음 → 논문 Eq. 2 구현. SO-101 5 ellipsoid → Panda gripper / wrist / forearm×3 (sphere model MVEE). **moving-hazard optical-flow 이동 생략** (`track_centers=True` 는 매 call centroid 재측정으로 대체; 기본 off = 논문의 "frozen at reset"). VLM(Qwen3-VL) 대신 AEGIS 와 같은 대체. scale factor (논문 미기재) = AEGIS 의 5. damped pinv λ=1e-3 |

## verifier 가 알아야 할 것

- 새로 생긴 플래그·기본값: 없음 (rollout runner 와 연결 안 함). 각 filter 의 주요 kwarg: `obstacle_source` (`gt_seg` 기본 · `detector` · `given` · `ellipsoids`), `exclude` (`task_objects` 기본), `detector_prompt`, AEGIS/multilink `skip_inactive=True`, multilink `track_centers=False`.
- `SceneObservation.ee_pose` 는 robosuite `robot0_eef_quat` 자세다 (AEGIS 가 읽는 것). adapter URDF `panda_grip_site` 와 원점은 같고 z 축 −90° 회전 차이 (`test_recorded_call_frames_agree_with_panda_fk` 가 확인).
- `Baseline.knows_vla` 는 alias 로 계속 import 된다 (같은 module 객체). 옛 경로를 쓰는 기록 (`doc/migration/validation/*.json`) 은 그대로.
- GPU 는 쓰지 않았다. GDINO 는 CPU (`grounding-dino-tiny`, 캐시된 weight, `HF_HUB_OFFLINE=1`).
- 재생산이 필요한 산출물: 없음. 옛 기록과 호환이 깨지는가: import 이름은 alias 로 유지; `Baseline/knows_vla/` 파일 경로를 직접 여는 script 는 깨진다 (찾은 것 없음).

## 확인하지 못한 것 / 남은 일

- closed-loop rollout (SafeLIBERO Level I/II) 연결 — 다음 STEP. per-step (T=1) 호출이 원본과 같은 timing 이다.
- SafeLIBERO recorded frame 에서의 smoke 는 하지 않았다 (segmentation 기록 run 이 SL3 LIBERO 뿐). SafeLIBERO 에서는 `privileged/obstacle_names` 를 env 의 `*obstacle*` joint 이름으로 채워야 한다 (main_aegis.py:238-248).
- GDINO-tiny 256 px 의 box 는 크고 부정확했다 (call 12: "ramekin"/"stove" box 가 이미지의 1/3, 장애물 ellipsoid semi-axis 0.30 m). AEGIS 원 해상도 1024 px · backview 와 같은 조건은 재현하지 않았다.
- KNOWS 의 paper 기본 relaxed normal 은 실제 barrier 를 `(1-γ)h` 보다 빨리 줄이는 경우가 있다 (KNOWS docs/16 의 기존 측정과 같은 현상, 합성 test 에서도 재현).
- 공식 AEGIS 의 ω frame 규약이 robosuite OSC 와 일치하는지는 원본 그대로 두었고 검증하지 않았다.
- `Baseline/paper_baselines/README.md` (A 소유) 의 KNOWS 행 ("not moved; `Baseline/knows_vla`") 은 이제 틀리다 — 소유자가 `methods/knows` 로 고쳐야 한다. `methods/README.md` (method 목록) 는 C/D 공유라 만들지 않았다.
- smoke figure 는 smoke 산출물이다 (`outputs/...`). 규칙 A 의 문서용 figure 는 verifier 몫.

## 내가 기대하는 결과

closed-loop 에서 세 filter 모두 SafeLIBERO 장애물 접촉을 줄이되, AEGIS (EE 큰 ellipsoid) 와 KNOWS 는 target 근처 접근을 늦춰
성공률을 떨어뜨릴 수 있다. multi-link 는 forearm 접촉 사례에서만 AEGIS 와 차이가 날 것이다. detector 경로는 GDINO box 품질
때문에 GT seg 경로보다 과잉 개입할 것이다.
