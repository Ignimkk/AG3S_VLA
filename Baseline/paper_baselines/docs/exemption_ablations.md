# Exemption ablation 설계 — "No exemption" · "All-link exemption" (code 수정 없음)

> writer: Implementer B (PAPER-BASELINES) · 작성 2026-10-23 KST · 기준 commit `8cd02b1e` (feature/panda-benchmark)
> 이 문서는 조사·설계만 한다. **아무 code 도 고치지 않았다.** 실제 수정은 run 이 없을 때 다음 STEP 에서 한다.
> 줄 번호는 commit `8cd02b1e` 기준이다. 작성 중 다른 session 이 `benchmark/runners/launch_libero_paired.py` 를 고치고 있었다 (BM-SETUP-A, 미commit).
> 그 working tree 에서는 `server_argv` 가 `:64-75`, `--ag3s-version` 이 `:375` 다. version 기구 (`AG3S_VERSIONS` → `version_server_flags`) 는 바뀌지 않았다.
> 확인 방법: `.venv-ag3s` 에서 `AG3S.adapters.panda.profile` 의 `parse_v11ab_args` · `v11ab_build_kwargs` ·
> `panda_ag3s_config` · `diff_against_v11ab` · `match_ag3s_version` 을 **읽기 전용으로** 호출했다 (아래 §4 출력).

## 1. 요약

| ablation | 뜻 (논문) | 현재 V11AB argv 로 표현? | 정확한 argv (V11AB argv 뒤에 덧붙임) | 남는 문제 |
|---|---|---|---|---|
| **No exemption** | 모든 observed surface 가 obstacle. accepted target 도 어느 link 에서도 빠지지 않는다 | **예** | `--target-field-policy relax` | (a) launcher 가 받지 못한다 — `AG3S_VERSIONS` 항목이 없다. (b) 기록상 version 이름이 `V11AB` 로 잘못 남는다 (§3) |
| **All-link exemption** | accepted target 을 **모든** constraint link 에서 뺀다 | **예** (flag 는 있다) | `--target-field-policy exclude-all` | V11AB 는 `--links gripper` 라 constraint link 가 전부 contact link 다. 그래서 **V11AB 위에서는 target 을 빼는 sphere 집합이 V11AB 와 같다** (§2.2). 의미 있는 비교는 `--links arms` 판(pb3 이상) 위에서만 된다 |

결론: 두 ablation 모두 **새 알고리즘 code 없이 기존 flag 하나로 표현된다.** 필요한 최소 변경은
`AG3S/adapters/panda/profile.py:204-237` 의 `AG3S_VERSIONS` 에 항목을 추가하는 것뿐이다 (§3).
V11AB 와 기존 version 의 argv 는 바뀌지 않는다. 한 가지 예외가 있다. 새 항목이 `--target-field-policy` 를 쓰면
`version_flag_values` 의 비교 key 에 `target_field_policy` 가 하나 늘어난다. 그래서 server 기록의 `version_flags`
dict 에 key 가 하나 더 생긴다 (값은 V11AB 와 같다).

## 2. 현재 구조 (읽은 code)

### 2.1 V11AB 가 target 을 빼는 방식

| 항목 | 값 | 출처 |
|---|---|---|
| V11AB server flag | `--target-field-policy exclude-authorized` (그 밖에 `--links gripper`, `--esdf-margin 0`, `--obstacle-margin 0.010` 등) | `scripts/benchmark/v11ab/start_server.sh` (`profile.v11ab_server_flags()` 출력, §4) |
| 정책 선택지 | `("relax", "exclude_authorized", "exclude_all")`, 첫 항목이 config 기본값 | `AG3S/ag3s/config.py:1234`, `:1296` |
| CLI | `--target-field-policy {relax,exclude-authorized,exclude-all}`, 기본 `relax` | `AG3S/trajopt/serve_safe.py:1340` |
| target 없는 계층 생성 | 정책이 `relax` 가 아니고 쥔 것이 없을 때만 target-free fine layer 를 만든다 | `AG3S/ag3s/runtime/pipeline.py:2596-2611` |
| 어느 sphere 가 그 계층에 묻나 | `exclude_all` → 모든 constraint sphere (`np.ones`). `exclude_authorized` → `clearance_policy.authorized_links(ctx)` 에 든 link 의 sphere | `AG3S/ag3s/constraints/to_adapter.py:226-243` |
| 권한 집합 | `contact.contact_links[manipulator]`, context 의 `active_manipulators` 합집합 | `AG3S/ag3s/constraints/clearance.py:206-215`, `AG3S/ag3s/config.py:1193` |
| Panda 권한 집합 | `{"left": ("panda_hand", "panda_leftfinger", "panda_rightfinger")}` | `AG3S/adapters/panda/model.py:112`, `profile.py` `panda_overrides` |
| Panda 의 `--links gripper` | `PANDA_GRIPPER_LINKS` = palm + 두 finger (22 sphere) | `AG3S/adapters/panda/model.py:102`, `:115-116` |
| Panda 의 `--links arms` | `panda_link1..7` + gripper (59 sphere) | `AG3S/adapters/panda/model.py:105` |
| LIBERO client 가 보내는 context | `ag3s/phase=approach`, `ag3s/active_manipulators=["left"]` (매 요청) | `benchmark/runners/libero_safe_client.py:38-39`, `:175` |
| 정책의 margin 쪽 효과 (T30 F1) | `exclude_all` 이면 `target_free_exempt = mask & (per_link < max)`. `exclude_authorized` 이면 `None` (= mask 전체 면제) | `AG3S/trajopt/linearize.py:2004-2012`, `:1466-1480`, `:932-978` |
| target 행 margin (relax) | target 이 최근접인 행 (`|d − d_object| ≤ voxel`) 은 `manipulated_link_margin` 을 받는다 = `geometry.safety_margin 0.05 × phase scale`. approach 에서 권한 link 는 0.4 → 20 mm, 권한 없는 link 는 50 mm | `AG3S/trajopt/linearize.py:963-978`, `AG3S/ag3s/config.py:826-834` (phase rule), `clearance.py:170` |

### 2.2 핵심 관찰 — V11AB(Panda) 에서는 "authorized" 와 "all" 이 같은 sphere 집합이다

- V11AB 는 `--links gripper` 다. constraint model 의 link 는 `panda_hand`, `panda_leftfinger`, `panda_rightfinger` 셋뿐이다.
- 이 셋은 모두 `contact_links["left"]` 에 있다.
- client 는 매 요청 `active_manipulators=["left"]` 를 보낸다.
- 그래서 `to_adapter.py:236-241` 의 `exclude_authorized` mask 는 22 sphere 전부 `True` 가 된다. `exclude_all` 의 `np.ones` 와 같다.

**V11AB 위에서 "All-link exemption" 은 target 을 빼는 sphere 집합으로는 V11AB 자신과 같다.** 남는 차이는 하나다.
`linearize.py:2010-2012` 에서 `exclude_all` 은 `target_free_exempt = mask & relaxed` 를 쓴다.
`relaxed = per_link < max(per_link)` 인데, 권한 link 만 있으면 모든 sphere 의 margin 이 같다 (approach 0.02 m).
그래서 이 mask 는 전부 `False` 다. 결과적으로 target-free 계층에 묻는 행에도 T30 F1 이전 규칙이 적용된다.
최근접 거리가 우연히 사라진 target 거리와 voxel 안으로 같으면 20 mm 가 붙는다 (`linearize.py:932-940` 주석의 E3b 사례).
이것은 "모든 link 에서 뺀다" 와 무관한 margin 부작용이다. **V11AB 위의 `exclude-all` run 을 All-link exemption 의 증거로 쓰면 안 된다.**

의미 있는 All-link 비교는 비-contact link 가 constraint model 에 있을 때만 성립한다.
즉 `--links arms` 를 쓰는 판(pb3 이상)이다. 거기서는 `panda_link1..7` 의 37 sphere 가 target 을 보는지 여부가 정확히 ablation 축이 된다.
- 기준: `V11AB-pb3` (`exclude-authorized`, 링크 arms)
- 비교: 같은 flag + `--target-field-policy exclude-all`

현재 최신판 pb7 에서도 같은 짝을 만들 수 있다.

### 2.3 "No exemption" 이 `relax` 로 정확한가

`relax` 에서는 target-free 계층이 만들어지지 않는다 (`pipeline.py:2598`).
`to_adapter.py:228` 의 첫 조건에서 `target_field_exclude = None` 이 된다.
그래서 **모든 sphere 가 target 이 든 fine layer 에 거리를 묻는다.** observed target surface 가 모든 link 에게 obstacle 이다.

남는 target 특유의 처리는 margin 뿐이고, 이것은 exemption 이 아니다 (margin ≥ 0 이라 표면 안으로는 못 들어간다, `config.py:1281-1286` T7a).

| 행 | `relax` 에서 받는 margin (approach, V11AB 값) |
|---|---|
| 권한 link (gripper) 의 target 행 | `0.05 × 0.4` = 20 mm (`manipulated_link_margin`) |
| 권한 없는 link (arms 판에서 link1..7) 의 target 행 | 50 mm (full margin) |
| 일반 obstacle 행 | `max(esdf-margin 0, obstacle-margin 10 mm)` = 10 mm |

pb5 이상(`--latch-phase`)에서는 latch 가 pre_grasp/grasp phase 를 준다. 이때 권한 link 의 target 행은 5 mm / 0 mm 로 내려간다.
이것도 "닿을 수는 있지만 들어갈 수는 없다" 이므로 No exemption 정의와 맞는다. 0 mm 로 내려간다고 surface 가 사라지지 않는다.

더 엄격한 변형("target 에 phase 완화도 없음")은 `contact.phase_aware=False` 다 (`config.py:1179`, `rule_for` 가 항상 margin_scale 1.0).
이 값은 serve_safe flag 가 없고, Panda 경로에서는 `build_panda_ag3s(extra_overrides=...)` 로만 바꿀 수 있다.
그 경로는 진단용으로 assert 를 건너뛴다 (`profile.py` `panda_ag3s_config`). 논문의 No exemption 정의(observed surface 를 빼지 않음)에는 필요하지 않으므로 **제안하지 않는다.**

**쥔 뒤(attach)는 이 ablation 범위 밖이다.**
- 쥔 물체를 field 에서 빼고 attached sphere 로 싣는 것은 `target_field_policy` 와 무관하다. `to_adapter.py:228` 의 `attached is None` 조건 때문에 쥔 뒤에는 정책이 돌지 않는다.
- 이것은 LTE 의 target exemption 이 아니라 HSR 의 carry 다.
- `relax` 에서도 V11AB 와 같이 동작한다. 쥔 물체까지 obstacle 로 두면 손이 늘 물체 안에 있어 infeasible 이 된다. 그래서 별도 ablation 으로 하지 않는다.

## 3. 최소 code 변경안 (다음 STEP, run 이 없을 때)

### 3.1 왜 argv 만으로는 부족한가

- paired launcher 는 server argv 를 `profile.v11ab_server_argv() + ["--robot", "panda"] + profile.version_server_flags(version)` 으로 만든다 (`benchmark/runners/launch_libero_paired.py:57-67`). `--ag3s-version` 은 `AG3S_VERSIONS` 의 이름만 받는다 (`:348`, `profile.py:240-243`). **임의의 추가 flag 를 넣을 통로가 없다.**
- 수동으로 server 를 띄워 flag 를 덧붙이면 동작은 하지만 기록이 틀린다. server 는 `profile.match_ag3s_version(args)` 로 version 이름을 기록한다 (`AG3S/adapters/panda/server.py:345`). 비교 key 는 `AG3S_VERSIONS` 항목이 쓰는 dest 뿐이다 (`profile.py:246-252`). 지금은 어느 항목도 `--target-field-policy` 를 쓰지 않는다. 그래서 `relax` / `exclude-all` run 이 **`V11AB` 로 기록된다** (§4 출력의 `match_version=V11AB`).

### 3.2 제안 diff (기본 off, 기존 version bit-identical)

`AG3S/adapters/panda/profile.py:236` (`"V11AB-pb7": ...` 항목 끝) 과 `:237` (`}`) 사이에 항목을 추가한다.
이름은 바꾸는 내용을 말하게 짓는다 (memory: descriptive step names).

```python
    # Exemption ablations (PAPER-BASELINES, paper Tab. ablation). Same algorithm, only the target-field policy changes.
    # No exemption: the observed target stays in the field for every link (margins only; serve_safe --target-field-policy).
    "V11AB-no-target-exemption": ("--target-field-policy", "relax"),
    # All-link exemption on the gripper-only V11AB is the same sphere set as V11AB (all constraint links are contact
    # links; docs/exemption_ablations.md 2.2) -- kept only as a control; the meaningful pair is on --links arms below.
    "V11AB-all-link-target-exemption": ("--target-field-policy", "exclude-all"),
    "V11AB-pb3-no-target-exemption": ("--reach-at-capture", "--minimal-intervention-mm", "10", "--links", "arms",
                                       "--target-field-policy", "relax"),
    "V11AB-pb3-all-link-target-exemption": ("--reach-at-capture", "--minimal-intervention-mm", "10", "--links", "arms",
                                             "--target-field-policy", "exclude-all"),
```

(pb7 위의 짝이 필요하면 `AG3S_VERSIONS["V11AB-pb7"] + ("--target-field-policy", "relax" | "exclude-all")` 로 같은 모양의 항목 둘을 더한다.)

| 변경 후 확인할 것 | 어떻게 |
|---|---|
| 기존 version 의 argv 불변 | `version_server_flags(v)` 를 기존 9 개 이름에 대해 변경 전후로 비교 |
| version 이름 기록 | `tests/benchmark/test_panda_sl5j_recorder.py:164-166` 이 모든 `AG3S_VERSIONS` 에 대해 `match_ag3s_version(_args(v)) == v` 를 이미 검사한다. 새 항목도 자동으로 들어간다. V11AB 의 `target_field_policy` 는 `"exclude-authorized"` 라 새 항목과 구분된다 |
| `version_flags` 기록 key 증가 | 새 항목이 `target_field_policy` 를 비교 dest 에 넣으므로, 이후 run 의 `version_flags` dict 에 `target_field_policy` key 가 생긴다. 옛 run 과 dict 를 그대로 비교하는 code 가 있으면 그 key 를 무시해야 한다. 현재 그런 비교는 `match_ag3s_version` 하나뿐이다. 그 함수는 매번 같은 시점의 dest 집합으로 양쪽을 계산하므로 영향이 없다 |
| config 차이 | `diff_against_v11ab` 가 `constraint.target_field_policy` 한 행을 `reason: "UNDECLARED"` 로 낸다 (§4). manifest 의 `override_reasons` 는 Panda override 용이고 version 차이는 `version_flags` 가 기록하므로 그대로 둔다 |
| runtime | `relax` 에서는 target-free 계층이 없다. 그 계층을 읽는 곳이 `None` 을 견디는지 확인한다. 1-episode dry run 으로 본다. 정적으로 읽은 곳: `to_adapter.py:228`, `safe_policy.py:1949`, `:2173`, `linearize.py:1475` 는 모두 mask `None` 분기를 가진다 |

## 4. 확인 출력 (읽기 전용 호출, 2026-10-23 KST)

`.venv-ag3s`, `PYTHONPATH=<checkout>`. 각 줄은 V11AB argv 뒤에 열거한 flag 를 덧붙여 `parse_v11ab_args` → `v11ab_build_kwargs` → `panda_ag3s_config` 를 돌린 결과다.
`diff_vs_v11ab_panda` 는 Panda override 2 행을 뺀 config 차이다.

```
V11AB              OK policy=exclude_authorized links=gripper match_version=V11AB diff_vs_v11ab_panda=[]
NE relax           OK policy=relax links=gripper match_version=V11AB diff_vs_v11ab_panda=['constraint.target_field_policy']
ALL exclude-all    OK policy=exclude_all links=gripper match_version=V11AB diff_vs_v11ab_panda=['constraint.target_field_policy']
pb3                OK policy=exclude_authorized links=arms match_version=V11AB-pb3 diff_vs_v11ab_panda=[]
pb3+exclude-all    OK policy=exclude_all links=arms match_version=V11AB-pb3 diff_vs_v11ab_panda=['constraint.target_field_policy']
pb7+exclude-all    OK policy=exclude_all links=arms match_version=V11AB-pb7 diff_vs_v11ab_panda=['constraint.target_field_policy']
pb7+relax          OK policy=relax links=arms match_version=V11AB-pb7 diff_vs_v11ab_panda=['constraint.target_field_policy']
```

- parser 와 `reject_bad_flag_combinations` 는 모든 조합을 받는다.
- `match_version` 열이 §3.1 의 기록 문제를 보여 준다. `relax` / `exclude-all` 을 덧붙여도 원래 version 이름이 나온다.

V11AB server flag 전체 (`profile.v11ab_server_flags()`):

```
--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized
--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32
--obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60
--held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles
--lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20
```

## 5. 확인하지 못한 것

- runtime 동작: server 를 띄우지 않았다 (GPU 금지, run 진행 중). `relax` 에서 target-free 계층을 기대하는 기록·진단 경로가 빈 값으로 기록되는지는 1-episode dry run 으로 본다.
- RB-Y1 (논문 Tab. 의 실제 집합, development checkout) 에서는 `GRIPPER_LINKS` 에 **양쪽 손**이 들어 있다. client 가 한 손만 active 로 보내면 `exclude_authorized` 와 `exclude_all` 의 sphere 집합이 다르다 (반대 손). 이 문서의 §2.2 결론은 Panda 단일 팔에만 해당한다.
