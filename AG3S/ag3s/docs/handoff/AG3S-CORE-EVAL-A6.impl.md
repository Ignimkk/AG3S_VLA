# AG3S-CORE-EVAL-A6 — 구현 (쥐는 순간 손바닥 이웃 접촉 허용 A6 + HSR 규칙 ② 수정)

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe · 작성 2026-10-29 11:50 KST (`date -u` 02:45 + 9 h)
> task: [AG3S-CORE-EVAL-A6.task.md](AG3S-CORE-EVAL-A6.task.md) §2–4 (§5 closed-loop 은 lead) · 바탕: HEAD `bb85d9f8` (작업 트리 깨끗한 상태에서 시작)
> 원시 기록 · 도구: `$AG3S_STORAGE_ROOT/results/development/20261029T111648+0900_bb85d9f88675_52cc38430638/verify/AG3S-CORE-EVAL-A6/` (아래 `$R`)

## 무엇을 했나 (평이한 요약 먼저)

- **A6:** 새 flag `--neighbour-palm-contact-mm MM` (기본 꺼짐, 최종 구성 10).
  - "쥐는 순간" (latch latched · closing, HELD 아님, 손바닥 site 가 target 중심 60 mm 안) 에만 켜진다.
  - 이때 쥐는 손의 손바닥 구 (`ee_<side>` link 의 손 몸체 구 + 손바닥 껍질 구) 가 이웃 장애물과 10 mm 까지 겹쳐도 된다. 여유 기준이 "≥ 5 mm (껍질 구)" 에서 "≥ −10 mm" 로 바뀐다.
  - 탁자 (지지면), 바구니 (목적지 label), target, 쥔 물체 행은 풀지 않는다.
  - HSR6 ④ 로 거두지 않는다.
  - 요청마다 `summary_json.grasp.neighbour_palm_contact` 에 창 여부를 남긴다. `to.metrics.neighbour_palm_contact` 에는 풀린 행 수와 그 행의 최소 여유를 남긴다.
  - TrajOpt 와 최종 판정 (`sqp._finish` → verdict) 은 같은 `SceneSnapshot` 의 같은 `_esdf_clearance` 를 읽는다. 그래서 solver 와 인증이 같은 규칙을 쓴다.
- **② 수정:** `--hsr6-held-gate` 의 쥔 물체 들어올림 허용은 이제 HSR6 ∈ {approach, retreat} 일 때만 거둔다. HSR6 = grasp ∧ HELD 면 유지한다. lift · transport · placement 는 그대로다.
- **결과:**
  - A1 단위 test 통과: 새 test 42 개, 전체 2248 통과, 실패 1 = 알려진 기존 실패.
  - A2 꺼짐 동등성: 기록 요청 181 개 재생에서 refined **181/181 bit 동일**. SafePolicy 기록 · config dump 도 같다.
  - A3 장면 119: A6 창이 요청 11–15 에서 켜진다. 요청 11 의 손바닥 이동이 9.67 → 4.19 mm 로 줄었다.

## 0. 규칙을 코드로 옮기며 정한 것 (결과 전, lead 확인 필요 1 건)

| 항목 | task 문장 | 코드 | 이유 |
|---|---|---|---|
| **창의 거리 기준점 (확인 필요)** | "손바닥 (hold link site) 이 target 중심에서 60 mm 안" | hold link `ee_<side>` frame 의 **site** = (0, 0, −0.100) m 점 (`safe_policy.hold_site_position`). MJCF `left_ee` / `right_ee` site 와 같은 점이다 (test 가 FK 로 확인, 차 < 1 µm) | 60 mm 의 근거는 진단의 "손바닥 – 사과 30–55 mm" 이다. 이 값은 MJCF site `left_ee` 로 잰 것이다 (`diag119.py`). hold link **원점**은 site 보다 100 mm 뒤에 있다. 원점으로 재면 같은 요청 11–14 가 97–132 mm 가 되어 창이 한 번도 켜지지 않는다 (기록 `grasp.neighbour_contact.palm_to_target_mm` 96.7–132.3 mm). IFG-NBC (150 mm) 는 원점 기준 그대로 두었다 |
| 120 mm 범위 | "그 표면이 target 중심 120 mm 안 (IFG-NBC 와 같은 범위)" | IFG-NBC 코드와 같은 식: **구 중심**이 target 중심 120 mm 안 | IFG-NBC 의 실제 규칙이 구 중심 기준이다 (`config.py` NBC 주석, `linearize.py` NBC 블록) |
| 이웃 = 장애물 | target 이 아닌 과일 · 기둥 · 막대 | `_obstacle_rows` 분류 `obstacle` (target · 지지면 · 쥔 물체가 아님) **∧ 목적지 label 이 아님** | 바구니를 빼는 것은 IFG-NBC 에는 없던 조건이다. 거리장 label 층으로 뺀다. 목적지 label 이 있는데 거리장이 label 을 못 읽으면 아무 행도 풀지 않는다 (fail-closed) |
| 손바닥 구 | `ee_<side>` 손 몸체 구 + 껍질 구 | link 이름이 정확히 hold link (`ee_left` / `ee_right`) 인 구 전부 (`safe_policy.palm_contact_rows`). A8 모델에서 손마다 껍질 구 168 개 + 손 몸체 · 틈 채움 구 23 개 = 191 개. 손가락 link 는 들지 않는다 | 껍질 구도 link `ee_<side>` 에 붙는다 (`gripper_cover.palm_shell_capsules`, test 로 확인) |
| ② 의 grasp ∧ ¬HELD | 명시 없음 ("approach · retreat 일 때만 거둔다") | grasp 는 lift 와 같은 조건 (`blocked` ∨ HELD) 으로 허용 | "approach · retreat 일 때만 거둔다" 를 문자 그대로 옮겼다. 쥔 물체는 HELD (또는 ①) attach 뒤에만 있다. 그래서 grasp ∧ CLOSING 에서는 허용 값이 쓰이지 않는다 |
| ② 의 label 없음 (None) | 명시 없음 | 그대로 거둔다 (엄격한 쪽, 수정 전 규칙과 같음) | HSR6 impl §2 의 "label 없음 = 허용 없음" 유지 |

## 1. 바뀐 파일 · md5 (메인 작업 트리, 이 STEP 끝, commit 하지 않음)

HEAD `bb85d9f8` 대비 `git diff --stat`: 코드 6 파일 +251 −14 (test 포함). 새 파일 1 개.

| 파일 | md5 | 무엇이 |
|---|---|---|
| `AG3S/trajopt/config.py` | `b6ee08089aac699ca2fa41646948b175` | `collision.neighbour_palm_contact_{depth 0.0, radius 0.12, gate 0.06}` (`:1025-1037`), 검증 (`:1116-1121`), 꺼짐이면 `to_dict` 에서 키 빠짐 (`_OMIT_WHEN_DEFAULT`, `:1319-1323`) |
| `AG3S/trajopt/linearize.py` | `53e11e95e03f29e1a95092e322c5039d` | `SceneSnapshot.neighbour_palm_contact` (`:288-296`). `_esdf_clearance_impl` 에서 IFG-NBC 블록 바로 뒤에 호출 (`:1052-1056`). `_neighbour_palm_contact` 메서드: 행 선택 · −depth · 기록 (`:1081-1136`). memo side attr 에 `_last_neighbour_palm_contact` 추가 (`:120`) |
| `AG3S/trajopt/sqp.py` | `5cda3fab09d25297ebdb1c9fa2f660f5` | `_finish`: 반환 궤적의 최종 검사가 푼 행을 `metrics.neighbour_palm_contact` 에 싣는다. 장면에 A6 dict 가 있을 때만 생긴다 (`:1480-1484`, `:1586`) |
| `AG3S/trajopt/safe_policy.py` | `4e900950b2e9abbe393f261adc53a7af` | `HOLD_LINK_SITE_OFFSET_M` · `hold_site_position` · `palm_contact_rows` (`:73-94`). `_neighbour_palm_contact_gate` · `_neighbour_palm_contact_result` (`:2004-2062`). `_scene_fn` 이 snapshot 에 붙인다 (`:1132-1139`, 판정 `_last_snapshot` 도 같은 것). refine 뒤 기록 병합 (`:632-633`). 초기화 · reset · class 기본값 (`:170-171`, `:385`, `:519`). `_hsr6_held_gate` docstring (`:1250`) |
| `AG3S/trajopt/hsr6.py` | `b2d0738823afdc254b2ef6312e1cfb79` | ② 수정: `HELD_WITHDRAW_PHASES = {approach, retreat}`, `held_allowance` (`:124-135`). 모듈 표 · 설명 (`:15`, `:21-26`) |
| `AG3S/trajopt/serve_safe.py` | `c9cd8c487ea668385feb2bb70912e4e4` | `--neighbour-palm-contact-mm` (`:1717-1721`). `obstacle_overrides` → `neighbour_palm_contact_depth` (`:3343-3348`, 0 이하는 거절). 켰을 때만 시작 줄 (`:3938-3945`). `--hsr6-held-gate` help (`:1943`) |
| `tests/trajopt/test_core_eval_a6.py` (**새**) | `fa159ec4d2a362b174e54588401d5d68` | 42 test (§2 A1) |
| `tests/trajopt/test_core_eval_hsr6.py` | `ccf38d7bcd7e71f63662f2adaaa0cf76` | ② 기대값 2 곳을 수정 규칙으로 바꿈 (grasp ∧ HELD → True). approach ∧ HELD 경우 추가 |
| `tests/trajopt/test_scm_palm_shell_margin.py` | `ee5614f0248cba71a31485f3c5e7f2c7` | `to_dict` 키 집합 test 에서 A6 키 3 개를 뺌 (꺼짐이면 키가 없는 규칙이 같음) |

손대지 않은 것:
- `gpu_seeds.py`: GPU seed 세계는 A6 를 모른다. IFG-NBC 와 같은 gap 이다 (아래 "알아야 할 것").
- `pipeline.py`: ② 는 SafePolicy 가 넘기는 값만 바뀐다.
- 문서 · figure.

## 2. 검증 (task §4, 결과 전 고정)

모두 CPU · 한 번에 1 process · `nice -n 10` · `.venv-ag3s` · 메인 checkout. 로그 첫 줄 `trees:` 로 module 위치를 확인했다 (worktree 를 쓰지 않음).

| # | 검사 | 결과 | 판정 |
|---|---|---|---|
| A1 | 단위 test | 새 42/42. 관련 4 파일 113/113. 전체 `tests/trajopt` + geo 2 + subtask gate + config roundtrip: **2248 passed, 34 skipped, 1 failed** (= 알려진 `test_t43t_replay::test_off_is_bit_identical_to_the_pre_fix_commit`, G1 · HSR6 impl 와 같음) | **통과** |
| A2 | 꺼짐 동등성 | A6 flag 없음, HEAD `bb85d9f8` 대 메인: refined **181/181 bit 동일** (core 106 + 결합 시험 75). 기록은 시간 필드만 다르다. SafePolicy 경로 · config dump 도 8/8 같다 | **통과** |
| A3 | 장면 119 재풀이 | A6 창이 요청 11–15 에서 켜진다 (요청 10 은 site 69 mm 라 꺼짐). 요청 11 손바닥 이동 9.67 → **4.19 mm**. 12 는 7.48 → 6.71. 13–15 는 ±0.4 mm | **기전 확인** (통과 기준 아님) |

### A1 — 단위 test

```bash
cd /mnt/dev/AG3S_VLA-development
MUJOCO_GL=osmesa PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s/test_geo_grasp.py \
  tests/ag3s/test_geo_grasp_obstacles.py tests/ag3s/test_subtask_gate.py tests/ag3s/test_config_roundtrip.py
```

- 전체: 2248 passed · 34 skipped · 1 failed (기존), 24 분. 로그 `$R/A1/a1_full_pytest.log`.
- 관련 4 파일 (a6 · hsr6 · nbc · scm): 113 passed. 로그 `$R/A1/a1_targeted.log`.
  - SCM 고치기 전 1 차 로그는 `a1_targeted_before_scm_test_fix.log` 다. 실패 1 개는 to_dict 키 집합 test 였고, A6 키를 뺀 기대값으로 고쳤다.

`test_core_eval_a6.py` 가 고정하는 것:

| 묶음 | 경우 |
|---|---|
| 꺼짐 | `None` · depth 0 이면 행 값이 bit 동일하고 기록 attr 이 없다. `SceneSnapshot` 기본 None |
| 대상 행 | 손바닥 mask ∧ 120 mm 안 ∧ obstacle 행만 −10 mm (막대 행: 0.008 − r − 10 mm → 0.008 − r + 10 mm). mask 밖 · 범위 밖 행은 같다 |
| 대상 물체 제외 | 탁자 (support) · 사과 (target) · 쥔 물체 행은 그대로. **바구니** = 목적지 label 이면 그대로 (`n_destination_excluded` 1). label 을 못 읽으면 아무 행도 풀지 않는다 (`reason`) |
| 껍질 구 여유 | `palm_shell_margin` 5 mm 행이 −10 mm 로 (차 15 mm). `n_rows_below_old_margin` 1 |
| −10 mm 경계 | 겹침 9.5 · 10.0 mm → 여유 ≥ 0 (허용), 10.5 · 11 mm → 음수 (위반). 기록 `min_clearance_mm` = −겹침 |
| 손가락 · 손바닥 독립 | IFG-NBC (손가락) 와 A6 (손바닥) 를 함께 줘도 서로의 행을 바꾸지 않는다 |
| 창 참 | latched · closing, site 30 mm → 켜짐. mask = `ee_left` 구만 (손가락 · 오른손 아님). `grasp.neighbour_palm_contact` 기록 |
| 창 경계 | site 59 · 59.9 mm 켜짐, 60.1 · 100 mm 꺼짐 ("palm outside the gate") |
| 창 기준점 | hold link 원점에 target 을 두면 (site 에서 100 mm) 꺼진다 |
| 창 거짓 | searching · held · placed · attach 회수 · target 없음 · flag 꺼짐 (기록 `{}`, grasp 기록에 키 없음) |
| HSR6 ④ | phase approach + `--hsr6-objective`: IFG-NBC 는 거두어지고 (`hsr6_withdrawn`), A6 는 켜진다 |
| 기록 병합 | refine 뒤 `to.metrics.neighbour_palm_contact` 가 `rows` 로 들어간다. 없으면 None |
| 실제 서버 모델 | A8 모델 (`--palm-shell on`) 에서 `ee_left` / `ee_right` mask 는 껍질 구 168 + 손 몸체 구. 손가락 0. site = MJCF `left_ee` / `right_ee` (FK 3 자세, 차 < 1 µm) |
| ② 수정 | grasp ∧ HELD → 유지, grasp ∧ blocked → 유지, grasp ∧ 둘 다 아님 → 거둠, approach · retreat → 거둠, lift · transport · placement 그대로, None · 모르는 label → 거둠. SafePolicy 가 AG3S 에 grasp ∧ HELD = True 를 넘김. flag 꺼짐이면 묻지 않음 |
| config · CLI | 기본 0 · 키 없음, 켜면 키 · roundtrip, 나쁜 값 거절, `--neighbour-palm-contact-mm 0` 거절. **최종 argv (§3) 가 parse · 조합 검사 · config 를 통과**: 10 mm · 60 mm · 120 mm, NBC 20, 껍질 5, HSR6 4 flag |
| 기록 chunk | 장면 119 요청 13 (`CB_HSR_k0` 기록) 을 A6 off / on 으로 풂. off 는 metrics 키 없음. on 의 `metrics.neighbour_palm_contact` 는 반환 궤적을 같은 장면으로 다시 잰 값과 같다 (look-ahead gate closed → window optimizer). `clearance_m` · `max_violation` 도 같다 = **solver 의 최종 검사와 기록이 같은 규칙** |

### A2 — 꺼짐 동등성

- 도구:
  - `$R/tools/a2_replay.py`: HSR6 H2 `h2_replay.py` (= G1 V2) 사본에 entry 하나를 더했다.
    - 재생 경로는 `tp_timing.run_timing` server mode 다. 기록 명령줄은 그대로 쓴다 (A6 flag 없음).
    - CPU 이므로 GPU seed 만 끈다. CPU 여러 초기값 16 은 켠다.
    - A4 anchor 는 `e1_goal` 로 다시 만들고, IFG-TOW 배율은 기록값을 재생한다.
  - `$R/tools/a2b_off_records.py`: HSR6 H2b 사본에 config dump 비교를 더했다.
- 두 트리:
  - 변경 전: `git archive HEAD` (AG3S · common · tests, docs 제외) 를 scratch `a6/pre_tree` 에 풀었다. 7 파일 md5 가 HEAD 와 같은 것을 확인했다. 비코드 폴더 (robots 등) 는 링크했다.
  - 변경 후: 메인 작업 트리.

| entry | 요청 | refined bit 동일 (전 = 후) | 기록 같음 | 다른 기록 키 | A4 켜짐 | 여러 초기값 발동 |
|---|---|---|---|---|---|---|
| core REG ep211 | 75 | **75** | 73 | seq 15 · 16 의 `multi_seed.*_ms` (시간) 만 | 27 | 2 |
| core OCC 121 | 31 | **31** | 31 | — | 6 | 0 |
| 결합 시험 HSR k0 장면 119 (명령줄에 G1 + `--hsr6-*` 4 개) | 75 | **75** | 75 | — | 18 | 0 |

- A2b (재생이 닿지 않는 경로, 두 트리 JSON 비교) 는 8/8 같다. 다음을 비교했다:
  - grasp latch 순서열 3 개 (chunk 기록 17 개, `_exec_summary` 전체), AttentionPolicy 키, AG3S held 구 2 경우.
  - 최종 명령줄에서 A6 만 뺀 config dump (`to_dict` = `--record-constraints` meta).
  - 기본 config dump.
- ② 수정은 `--hsr6-held-gate` 가 켜질 때만 `held_allowance` 를 부른다 (`safe_policy.py` `if self.hsr6 is not None and self.hsr6.held_gate`).
  - 위 재생은 TrajOpt 만 다시 푼다. 결합 시험 entry 도 ② 를 거치지 않는다.
  - 그래서 ② 의 동작 변화는 A1 test 로 고정했다.
- 로그: `$R/A2/a2_pre.log`, `a2_main.log`, `a2_compare.json`, `a2b_compare.log`.

### A3 — 장면 119 offline 재풀이 (A6 off 대 on)

- 도구: `$R/tools/a3_scene119.py`.
  - 입력: 결합 시험 HSR k0 `회귀119_k0` (spec `module/spec_cb_HSR.json`, chunk 203 + j = 요청 j). 요청 6–16 을 server mode 로 연속해서 풀었다.
  - 재생 방식은 A2 와 같다. 두 설정을 요청마다 번갈아 푼다.
  - IFG-NBC · A6 dict 는 **서빙 gate 메서드 그대로** 만든다 (`_neighbour_contact_gate` · `_neighbour_palm_contact_gate`).
    - 이 메서드를 기록의 latch · manipulated centroid · HSR6 label · `q_now` 를 담은 SafePolicy 껍데기에서 부른다.
    - NBC 활성이 기록과 같은지 확인했다 (11 요청 모두 같음).
  - 손바닥 이동 = refined − VLA reference 의 MuJoCo site `left_ee` 이동이다. 실행 8 행 중 가장 큰 행을 쓴다 (진단 `diag119.py` 와 같은 정의).
- 재생 충실도: A6 off 재생과 기록의 손바닥 이동 차는 요청 7–15 에서 ≤ 0.1 mm 다 (아래 표 "기록" 열).

| 요청 | latch | HSR6 | site–target (mm) | A6 창 | 손바닥 이동 기록 (mm) | **off** | **on** | on − off | 풀린 행 (계획 전체 / 실행 창) | 풀린 행 최소 여유 (mm) | 이전 여유를 어겼을 행 | 최악 행 off → on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | latched | approach | 103.8 | 꺼짐 | 1.28 | 1.35 | 1.35 | 0 (bit 동일) | — | — | — | finger_r2 30.5 mm |
| 10 | latched | approach | 68.9 | 꺼짐 | 5.64 | 5.64 | 5.64 | 0 (bit 동일) | — | — | — | ee_left 24.8 mm (look-ahead gate open) |
| **11** | latched | approach | 40.6 | **켜짐** | 9.78 | **9.67** | **4.19** | **−5.48** | 2061 / 1031 | −0.47 / 실행 창 +1.88 | 29 | ee_left 5.0 mm → finger_l1 5.0 mm |
| 12 | latched | approach | 13.7 | 켜짐 | 7.45 | 7.48 | 6.71 | −0.77 | 1021 | 12.1 | 0 | ee_left 9.4 → finger_l1 12.6 |
| 13 | latched | approach | 14.8 | 켜짐 | 1.40 | 1.42 | 1.44 | +0.02 | 920 | 13.7 | 0 | ee_left 8.7 → finger_l1 8.9 |
| 14 | latched | approach | 14.0 | 켜짐 | 0.92 | 0.92 | 1.29 | +0.37 | 1002 | 10.4 | 0 | ee_left 5.3 → finger_l1 8.7 |
| 15 | closing | approach | 15.4 | 켜짐 | 1.12 | 1.12 | 1.28 | +0.16 | 939 | 18.9 | 0 | finger_l1 13.4 → 13.4 |
| 16 | held | approach | — | 꺼짐 (held) | 19.92 | 18.81 | 18.87 | +0.06 | — | — | — | 쥔 구 3.8 mm |

- 요청 11 이동의 성분 (target 쪽 / 위 / 옆, mm): off −7.09 / +6.38 / 1.61 → on −2.84 / +3.08 / 0.10. 사과에서 멀어지는 쪽과 위로 가는 이동이 각각 절반 이하로 줄었다.
- 모든 요청이 `feasible`, 위반 0, 여러 초기값 발동 없음 (양쪽 같음).
- HSR6 ④ 가 켜진 기록 구성에서도 A6 창이 켜졌다. 이 구간의 HSR6 는 11 요청 모두 approach 였고, IFG-NBC 는 ④ 로 거두어져 꺼져 있었다.
- 읽는 법 · 한계:
  - **요청 12 의 남은 6.7 mm 는 손바닥–이웃 행 때문이 아니다.** off 에서도 최악 행 여유가 9.4 mm 로 묶이지 않았다.
    - 이 재생의 `q_now` · 실행 이력 (꺾임률 항의 `deflection_history`) 은 기록 (A6 off 로 실행된 자세) 이다.
    - 그래서 요청 11 에서 줄어든 이동이 다음 요청의 시작 자세 · 이력으로 이어지지 않는다. 누적 효과는 closed-loop 에서만 보인다.
  - 요청 10 (6–10 mm 이동 구간의 첫 요청) 은 창 밖이다 (site 69 mm > 60 mm).
    - 이동 5.6 mm 는 look-ahead 꼬리 (gate open) 에서 온 것이다.
    - 진단의 GT 거리로도 83 mm 이므로 task 가 정한 창 (11–14) 밖이다.
  - 재생하지 않은 것 (두 설정 같음): G1 보정 (slot 에는 A4 만), GPU seed, AG3S 지각 (거리장 · target 은 기록값).
  - 요청 11 에서 계획 전체로는 손바닥–이웃이 0.47 mm 겹친다. 실행 8 행에서는 +1.9 mm 떨어져 있다. 허용 10 mm 중 일부를 꼬리에서만 썼다.
- 결과: `$R/A3/a3_scene119.json` (요청별 전체), `refined_{OFF,A6}_seq*.npy`, 로그 `$R/A3/a3.log`.

## 3. 최종 서버 argv (한 줄)

= ADOPTED_KO.md §A8 flag + G1 4 flag (G1 impl §5) + HSR6 4 flag + `--neighbour-palm-contact-mm 10`.
`<…>` 는 lead 가 채운다: checkpoint, snapshot, port, run dir, `--record-constraints`. 이 줄의 parse · 조합 검사 · config 는 A1 test `test_final_argv_parses_to_the_fixed_values` 가 확인한다 (GPU 가용성 probe 는 통과로 둠). GPU 기동은 lead 몫이다.

```text
python -m AG3S.trajopt.serve_safe --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint <16D 30k ckpt> --model-xml <snapshot>/robots/rby1/description/models/rby1a/mujoco/model_transport.xml --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass --palm-shell on --palm-shell-margin 5 --grasp-anchor 2000 --grasp-anchor-path-weight 0 --geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off --hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective --neighbour-palm-contact-mm 10
```

- client flag 는 A8 과 같다 (A6 · ② 는 client 를 바꾸지 않는다).
- **closed-loop 에서 셀 기록** (`summary_json`, 요청마다):
  - `grasp.neighbour_palm_contact`:
    - 창: `active`, `site_to_target_mm`, `reason` (꺼진 이유), `n_spheres`
    - `rows` = 반환 궤적의 최종 검사가 푼 행: `n_rows` · `n_spheres` · `n_rows_below_old_margin` · `min_clearance_mm` (d − r, 음수 = 겹침), look-ahead 를 쓰면 `n_rows_window` · `min_clearance_window_mm`, 목적지 label 로 뺀 `n_destination_excluded`
  - 같은 값의 원본: `to.metrics.neighbour_palm_contact`
  - ② 수정의 확인: `hsr6.detail.held_gate.allowed` 와 `hsr6.changed.held_gate`. 쥔 직후 HSR6 = grasp 요청에서 이제 `allowed: true` 여야 한다.
  - 서버 시작 줄: `TO neighbour palm contact (AG3S-CORE-EVAL-A6): ...`

## verifier · lead 가 알아야 할 것

- **새 flag (기본 꺼짐):** `--neighbour-palm-contact-mm MM` (> 0, 아니면 시작 거절).
  - config `collision.neighbour_palm_contact_{depth, radius 0.12, gate 0.06}`. radius · gate 는 CLI 로 바꿀 수 없다 (task 고정값).
- **동작이 바뀐 기존 flag:** `--hsr6-held-gate` (② 수정). 이 flag 를 쓴 이전 run (결합 시험 HSR) 과 새 run 은 쥔 직후 grasp 요청에서 다르다.
- **기록 키:** 꺼짐이면 새 키가 없다 (summary · metrics · config dump, A2 로 확인). 켜면 위 키가 생긴다.
- **재생산할 산출물:** 없음. 옛 기록 호환은 깨지지 않는다.
- **lead 확인 필요 — 창 기준점:** §0 첫 줄. "hold link site" 를 MJCF `left_ee` site (hold link 원점에서 100 mm 앞) 로 구현했다.
  - 원점으로 재면 장면 119 에서 창이 켜지지 않는다.
  - 원점이 의도였다면 `HOLD_LINK_SITE_OFFSET_M` 한 줄을 (0, 0, 0) 으로 바꾸면 된다. 그 경우 test 2 개 (기준점 · 경계) 를 고쳐야 한다.
- **GPU 여러 초기값과의 gap (IFG-NBC 와 같음):**
  - cuRobo seed 세계 (`gpu_seeds.py` B0m) 는 A6 를 모른다. 손바닥 껍질 행이 GPU seed 쪽에서는 여유 10 mm (`obstacle_margin`) 를 받는다.
  - GPU 결과는 우리 CPU 최종 검사 (A6 포함) 로 다시 재고, 표준 풀이 (A6 포함) 와 경쟁한다. 그래서 안전 판정은 일관된다.
  - 다만 여러 초기값이 발동한 chunk 에서는 GPU 해가 손바닥을 더 비켜 둘 수 있다. GPU 기록 `WORLD_GAPS` 문구 ("neighbour-contact relaxation absent") 는 바꾸지 않았다.
- **평가:** 금지 접촉 정의는 바꾸지 않았다. 실제 이웃 접촉은 그대로 실패로 센다 (task §2).
- **figure · outputs 보기:** 만들지 않았다 (verifier · lead 몫). A3 의 요청별 숫자는 `$R/A3/a3_scene119.json` 에 있다.
- worktree 는 쓰지 않았다. commit 하지 않았다. storage snapshot · 실행 중 서버는 건드리지 않았다. 새 출력은 위 `$R` run 아래에만 썼다.

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)

- closed-loop 장면 119 에서 요청 11 의 손바닥 이동이 줄어든다. 그러면 사과가 손가락 사이 중심에 더 가깝게 들어가 쥐기 성공이 늘 것이다. 다만 요청 10 (창 밖) 의 look-ahead 이동은 남는다.
- A6 창은 쥐는 장면마다 닫기 직전 2–5 요청에서 켜질 것이다. 실제 이웃 접촉은 거의 늘지 않을 것이다 (요청 11 실행 창 최소 +1.9 mm).
- ② 수정 뒤 쥔 직후 HSR6 = grasp 요청의 `lift_withdrawn` 이 사라진다. 세션 B 가 본 6–10° 꺾임과 multi-seed 발동도 줄 것이다.
