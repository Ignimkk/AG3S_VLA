# PAPER-METHODS — 논문 비교 방법(safety filter · sampling guidance)을 paper_baselines 에 구성

> writer: lead · 읽는 쪽: implementer C / D · 작성 2026-10-23 KST
> 사용자 지시 2026-10-23: "논문의 내용도 동일한 폴더에 구성. KNOWS 는 해 둔 것을 그대로 가져오고,
> 나머지는 구현하거나 git 에 있으면 가져온다. 완벽하지 않아도 되고 **기능만 잘 작동하면 된다.** SEAM 은 하지 않는다."
> 선행: PAPER-BASELINES (commit `bf1403d5`) — `Baseline/paper_baselines/` 의 grounding provider · harness.

## 무엇을 만드나

`Baseline/paper_baselines/methods/<name>/` 에 논문(`main.pdf`)의 비교 · 관련 방법을 하나씩 둔다.

**"기능이 작동한다"의 기준:** recorded Panda/LIBERO frame 에서 그 방법이 끝까지 실행되어 **수정된 action chunk 또는 guidance 결과**를 낸다. unit test 와 CPU smoke 로 보인다.

- closed-loop rollout 연결은 다음 STEP 이다. `benchmark/runners` 는 지금 다른 session 이 쓰고 있다.
- 각 method 폴더에 `README.md` 를 둔다. 내용: 원 논문 · 출처 repo@SHA · license · 원 방법과 다른 점 · 실행 명령 · 상태.

### 공통 interface — `Baseline/paper_baselines/methods/interface.py` (C 가 만든다)

```python
@dataclass
class SceneObservation:            # 한 policy call 의 관측
    rgb: dict[str, np.ndarray]; depth: dict[str, np.ndarray]           # camera -> HxWx3 uint8 / HxW float m (OpenCV frame)
    intrinsics: dict[str, np.ndarray]; T_base_cam: dict[str, np.ndarray]
    q: np.ndarray                  # joint position (Panda 7)
    ee_pose: np.ndarray            # 4x4 base frame
    gripper: float
    instruction: str
    target_phrase: str | None = None
    extras: dict = field(default_factory=dict)   # GT seg, attention grid, obstacle 이름 등 (privileged 여부를 key 로 표시)

@dataclass
class FilterResult:
    actions: np.ndarray            # 입력과 같은 shape/규약 (LIBERO: 7D EE delta + gripper)
    intervened: np.ndarray         # (T,) bool
    info: dict                     # QP 상태, barrier 값, 검출 물체 등

class SafetyFilter(Protocol):      # 실행 직전에 action 을 고치는 방법 (KNOWS, AEGIS, multi-link)
    name: str
    def provenance(self) -> dict: ...
    def reset(self) -> None: ...
    def filter(self, obs: SceneObservation, actions: np.ndarray) -> FilterResult: ...

class SamplingGuidance(Protocol):  # flow-matching denoising 안에서 작동하는 방법 (OmniGuide, CFM, Lan-o3dp adapted, NUDGE)
    name: str
    def provenance(self) -> dict: ...
    def make_guidance_fn(self, obs: SceneObservation, norm: dict) -> Callable:  # openpi pi0.sample_actions 의 guidance_fn(x_next, t_next)
        ...
```

- **`guidance_fn` hook:** `third_party/openpi/src/openpi/models/pi0.py:226-283` 에 있다. 기본값 None 이면 baseline 과 bit 단위로 같다. **openpi 는 수정하지 않는다.**
  - action 은 normalized 공간이다. `norm` 으로 unnormalize / normalize 한다.
  - pure JAX 로 작성해 jit 안에서 돌게 한다.
- registry: `get_method(name)`.
- smoke 공통 입력: SL3 dev run `20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0` 의 `_geom.npz` 에서 call 1~2개.
  - 그 call 의 기록된 action chunk 를 입력으로 쓴다.
  - 읽는 법은 `Baseline/paper_baselines/eval/offline_grounding.py` 를 참고한다.

## Implementer C — safety filter 계열

1. **KNOWS [17] — 그대로 가져온다.**
   - `git mv Baseline/knows_vla Baseline/paper_baselines/methods/knows` (history 보존).
   - 내부 import `Baseline.knows_vla` → `Baseline.paper_baselines.methods.knows` 로 바꾸고, 경로 문자열(`libero_config` 등)도 고친다.
   - `doc/migration/validation/*.json` 의 옛 경로는 provenance 기록이므로 **고치지 않는다.**
   - 기존 tests 가 옮긴 뒤에도 전부 통과해야 한다. 옮기기 전과 후의 결과를 impl.md 에 쓴다.
   - 기존 `cbf/filter.py` 의 `SafetyFilter` 를 위 interface 로 감싸는 `adapter.py` 를 추가한다.
2. **VLSA / AEGIS [10] — 공식 code 사용.**
   - 공식 code 는 `benchmark/SafeLIBERO/main/main_aegis.py` · `utils.py` (submodule, **수정 금지**)다.
   - per-step safety layer(perception → 물체 기하 → cvxpy CBF-QP)를 interface 로 감싼다. 공식 함수를 import 할 수 있으면 import 하고, script 구조라 안 되면 그 로직을 옮겨 쓰되 원 줄 번호를 주석으로 단다.
   - **외부 API 금지.** 공식 code 의 VLM 단계(zhipuai / zai-sdk GLM 호출)는 쓰지 않는다. 대체는 둘 중 하나다:
     - (a) SafeLIBERO env metadata 의 obstacle 이름 (privileged 로 표시)
     - (b) A 의 Grounding DINO 로 찾은 비-target 물체 전부
   - Grounding DINO 는 A 의 `providers/grounding_dino.py` 를 재사용한다. 원 방법과의 차이를 README 에 쓴다.
3. **Multi-link safety filter [1] — 공개 code 없음, 구현.**
   - 논문(arXiv:2609.40007)을 읽고 핵심을 구현한다: 여러 robot link 의 CBF 제약을 둔 QP.
   - Panda link sphere 는 `AG3S/adapters/panda` 의 sphere model 을 **읽어서** 쓴다. obstacle 은 depth point cloud 또는 AEGIS 와 같은 물체 기하로 표현한다.
   - EE delta → joint 는 Jacobian 으로 연결한다. moving hazard 부분은 생략할 수 있다. 생략한 것은 README 에 쓴다.
4. 필요한 package(cvxpy 등)는 `.venv-baselines` 에 설치할 수 있다. 설치하면 `setup/requirements-baselines.in` / `.txt` 도 갱신한다. **C 만 설치한다.**
5. tests: 각 filter 를 synthetic 장면(장애물 1개, 직진 action)에서 확인한다.
   - barrier 를 위반하는 action 은 수정되고, 안전한 action 은 그대로인지 본다.
   - recorded frame smoke 는 `outputs/paper_baselines_smoke/methods/` 에 남긴다.

## Implementer D — sampling guidance 계열 + 외부 repo 가져오기

1. **external fetch 확장 (D 만 수정):** `external/sources.json`, `fetch_sources.py` 에 아래를 pinned SHA 로 추가한다.

   | cite | 방법 | repo @ HEAD |
   |---|---|---|
   | [19] | OmniGuide | `daniilidis-group/OmniGuide` @ `d649d698` |
   | [5] | constrained flow matching | `Dubascudes/VLA-Safety-ConstrainedFlowMatching` @ `b91153ad` |
   | [15] | Lan-o3dp | `qqianfeng/lan-o3dp` @ `b89c2898` |
   | [11] | RoboGround | `ZzZZCHS/RoboGround` @ `25a86263` |
   | [13] | GuidedVLA | `GuidedVLA/GuidedVLA` @ `04be059e` |
   | [24] | OBEYED-VLA | `UARK-AICV/DRAGON_VLA` @ `bdf25556` |

   - SHA 는 B 의 `docs/survey.md` 에서 full SHA 로 확인한다.
   - clone 본체는 git 이 추적하지 않는다(`external/.gitignore`).
   - **license 가 없는 repo([5], [24])의 code 는 tracked 파일로 복사하지 않는다.** import 나 참조만 하고 README 에 "license unknown" 이라고 쓴다.
2. **OmniGuide [19]** — 공식 repo 에서 energy field(attractor/repeller)와 action guidance 계산을 찾아 `SamplingGuidance` 로 옮긴다.
   - repeller 는 obstacle(비-target) 기하, attractor 는 target 이다. target 은 A 의 grounding provider 결과 또는 ATS 를 쓴다.
   - LIBERO EE-delta action 의 누적 위치에 적용한다.
3. **Constrained flow matching [5]** — 공식 repo 의 constrained 갱신 규칙을 `SamplingGuidance` 로 감싼다. 이 논문은 SafeLIBERO 결과를 보고했으니, 그 설정(제약 · 파라미터)을 README 에 기록한다.
4. **Lan-o3dp [15] (adapted)** — 원 policy 는 RLBench 용이고 checkpoint 가 없다.
   - obstacle cost guidance 만 pi0.5 `guidance_fn` 으로 옮긴다.
   - 이름은 `lan_o3dp_guidance_adapted` 로 하고, 원 method 결과가 아니라는 것을 명시한다 (develope_PLAN_SUMMARY §3.1).
5. **NUDGE / zero-shot reactive avoidance [9]** — 공개 code 가 없으니 논문(arXiv:2609.35231)을 읽고 핵심을 구현한다. sampling guidance 가 아니면 `SafetyFilter` 로 둔다.
6. **[11] [13] [24]** (학습이 필요한 grounding policy) — fetch 와 README(입력 · checkpoint · 우리 protocol 에 넣으려면 필요한 것)만 한다. 실행하지 않는다.
7. guidance 는 pure JAX/numpy 로 쓴다. **`.venv-openpi-live` 에 설치하지 않는다.** test 는 `JAX_PLATFORMS=cpu .venv-openpi-live/bin/python` 으로 돌린다.
   - synthetic x_t 에서: repeller 근처의 waypoint 가 밀려나는지, guidance 를 끄면 identity 인지(bit 단위) 확인한다.
   - pi0.5 model 을 붙인 smoke 는 GPU 가 필요하므로 다음 STEP verifier 에게 넘긴다. 그 실행 명령을 README 에 준비해 둔다.

## 둘 다 지킬 것

- 수정 금지: `AG3S/**`, `benchmark/**`(SafeLIBERO submodule 포함), `integration/**`, `third_party/**`, 루트 `.gitignore`, `.gitmodules`. 읽기와 import 는 된다.
- **예외:** C 의 `git mv Baseline/knows_vla` 와 그에 따른 import 수정.
- 수정 금지: 이미 있는 `Baseline/paper_baselines` 의 A/B 파일(providers, eval, interface.py). 재사용만 한다. 고칠 것이 보이면 impl.md 에 적는다.
- GPU 금지(`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`). 같은 session 의 verifier 가 GPU lock 을 잡고 측정 중이다.
- CPU worker 는 각자 1개. RSS 를 작게 유지한다. 외부 유료 API 호출 금지. git commit 금지.
- 끝나면 `AG3S/ag3s/docs/handoff/PAPER-METHODS-<C|D>.impl.md` 를 쓴다. 내용: 바꾼 파일, 실제 test 출력, smoke, 원 방법과의 차이, 확인하지 못한 것.
