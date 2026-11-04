# SL8b — 구현

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · 2026-11-04 · 계약: `SL8b.task.md` §2

## 무엇을 했나 (평이한 요약 먼저)
`TargetConfirm`에 "처음 고른 물체의 고정"을 푸는 규칙 둘을 **기본 꺼짐**으로 넣었다.
(1) 창 투표: 같은 challenger가 연속 3번 1위여야 하던 것을, 최근 W번의 `select` 호출 중 M번 표를 얻으면 교체하는 것으로 바꾼다.
(2) attention 해제: 쥐기 전에 seed attention이 유지 물체 밖에 K번 연속 머무르면 고정을 풀고, seed 질량의 50% 이상을 가진 cluster로 옮기거나, 없으면 target 없음으로 둔다.
`serve_safe` flag와 Panda 판 셋(`-pc-idwin`, `-pc-idattn`, `-pc-idboth`)을 등록했다. 끈 판의 결정 · 기록 · config 값은 이전과 같다.

## 바뀐 파일
| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/config.py:402-420` (필드), `:607-630` (검증) | `ClusteringConfig.target_switch_window`(None) · `target_switch_votes`(3) · `identity_attention_release`(None) · `identity_attention_off_frac`(0.7) · `identity_attention_radius`(0.015 m) 와 범위 검사 | 규칙 2.1 · 2.2의 config |
| `AG3S/ag3s/stages/target_grounding.py:163-175` | 상수 `WINDOW_VOTE` · `ATTENTION_RELEASE` · `ATTENTION_RELEASE_NONE` · `ATTENTION_RELEASE_MIN_SHARE`(0.5) | decision reason |
| `target_grounding.py` `TargetConfirm.__init__`, `reset` | 상태: `_votes`, `_attn_shares`, `_seeds`, `_attn_pending`, `identity_record` | 두 규칙의 cross-frame 상태 |
| `target_grounding.py` `note_seeds` · `identity_rules_on` (`:1253-1275`) | `note_seeds(points, attention, indices=None)`: 이번 frame seed. 해제가 꺼져 있으면 바로 return | `ground_target`이 `select` 전에 넘김 |
| `target_grounding.py` `select` (`:1300-1420`) | 해제 단계 호출(held일 때, `held is None` 분기 앞) · pending `first` 조건 · 창 투표 count/교체 · keep 호출의 표 push | 2.1 · 2.2 본체 |
| `target_grounding.py` `_vote` · `_seed_shares` · `_seed_majority` · `_attention_release_step` (`:1849-1935`) | 창 투표 계산, seed 질량 share, 해제 판정 | 위와 같음 |
| `target_grounding.py` `ground_target` | `confirm.note_seeds(points[seeds], attention[seeds], seeds)` 를 `select` 직전에 호출. target metrics에 `identity_window_vote` · `identity_attention_release` · `identity_attention_off_share` (규칙이 켜졌을 때만) | seed 전달 + 소비자가 reason을 볼 경로 |
| `AG3S/ag3s/runtime/pipeline.py:2901-2903` | `_admissibility_frame_record`의 `decision`에 `identity` 키 (규칙이 켜졌을 때만) | 기존 decision record 경로 (`mode`/`reason`은 원래 거기 있음) |
| `AG3S/trajopt/serve_safe.py` `identity_rules_section` · `reject_identity_rules_flags` (`:1022-1060`), argparse (`:2762-2780`), `t43t_identity_section` 에 병합, `t43t_ag3s` 목록 | `--identity-switch-window W` `--identity-switch-votes M` `--identity-attention-release K` `--identity-attention-off-frac F` `--identity-attention-radius-mm R`; 문턱만 주면 거절, 범위 밖이면 거절 | 2.3. `assert_ag3s_config`가 키 생존을 확인 |
| `AG3S/adapters/panda/profile.py:453-466` | `ID_WINDOW_FLAGS` (W=5, M=3) · `ID_ATTENTION_FLAGS` (K=4, F=0.7, R=15) 와 세 판, 모두 `PC_LATCH_VERSIONS`에 | 2.3 |
| `tests/trajopt/_sl5a_golden.py` (`PB_DEFAULT_OFF`) | 새 flag 5개의 꺼진 값 | serve_safe golden 비교에서 꺼진 option은 제외하는 기존 방식 |
| `tests/ag3s/test_sl8b_identity_rules.py` (신규, 18개) | 단위 테스트 | 2.4 |
| `tests/benchmark/test_sl8b_identity_versions.py` (신규, 8개) | 세 판 parse · clustering override · server build | 2.4 |

## 정확한 의미 (모호한 곳을 어떻게 풀었는지 포함)

### 2.1 창 투표 (`target_switch_window = W`, `target_switch_votes = M`)
- 표를 얻는 조건은 지금의 `counts`와 같다: 1위 admissible 후보가 있고, frozen 아님, 손 가림 아님, score ≥ `target_switch_min_score`, latched identity hold · subtask gate가 막지 않음.
- `held_rank`가 아닌 1위가 표를 얻는 호출은 그 centroid를 창에 넣고, 창(최근 W 호출) 안에서 이 centroid의 `target_identity_tolerance` 안에 있는 표 수를 센다. 이번 호출 것을 포함한다. ≥ M이면 `switch`(reason `window_vote`)이고 새 id로 채택, 창을 비운다.
- 표를 못 얻은 호출(`counts` 거짓, **그리고 held가 1위인 `keep` 호출**)은 `None`을 창에 넣는다. count를 0으로 만들지 않고, W 호출이 지나면 밀려 난다. `select`를 거치지 않는 `unseen()` 호출은 창에 넣지 않는다(예전에도 streak을 건드리지 않았다).
- 서로 다른 장소(tolerance 밖)의 표는 합쳐지지 않는다.
- 기록: `ConfirmDecision.streak` = 이번 호출의 표 수, `ConfirmDecision.frames` = M. (`switch`에서는 예전처럼 채택 뒤 0.)
- 꺼짐(`None`): 기존 연속 streak 코드 경로 그대로. `_vote`는 호출되지 않는다.
- 해석: `freeze()`는 예전처럼 `_streak`만 지우고 창은 건드리지 않는다. frozen 호출은 `counts` 거짓이라 `None`이 들어가 창에서 자연히 밀려 난다.

### 2.2 attention 해제 (`identity_attention_release = K`, `off_frac = F`, `radius = R`)
- 적용 호출: 유지 중인 물체가 있고 `frozen` 아님, `note_latched` 아님, 마지막 관측 형상(`_geometry`)이 있고, seed가 전달되었고 seed attention 합 > 0, subtask gate가 place/home으로 막지 않음일 때.
  - frozen 또는 latched 호출에서는 모인 share를 **지운다**(쥐기 시작 전의 증거를 이어 쓰지 않는다 — 스펙에 없는 해석). 그 밖에 적용되지 않는 호출(형상 없음, seed 없음, 합 0)은 share를 그대로 둔다.
- 각 호출: off share = (유지 물체 형상 점에서 R **밖**에 있는 seed의 attention 합) / (전체 seed attention 합). 최근 K개를 보관하고, K개가 차고 평균 ≥ F이면 해제한다. 해제 뒤 share는 비운다.
- 해제 시 admissible cluster별 share = 그 cluster에 속한 seed의 attention 합 / 전체 seed 합 (cluster 소속은 `ground_target`이 넘기는 seed의 cloud index로, index가 없으면 `select`의 `points`와 좌표 일치로 정한다).
  - 최대 share ≥ 0.5: 그 cluster로 `switch`(reason `attention_release`, 새 id).
  - 단, 그 cluster가 유지 물체 자신의 cluster(`_nearest`)이면(형상이 낡았을 뿐 seed는 같은 물체) 해제하지 않고 기록 `decision = "same_object"`로 계속 진행한다. **스펙에 없는 해석**: 같은 물체로 id를 바꾸는 일을 피한다.
  - 최대 share < 0.5 또는 없음: 유지 물체를 놓는다(`_forget_manipulated`). 이 호출은 mode `no_admissible`, reason `attention_release_none`, target 없음(`GroundingStatus.NO_ADMISSIBLE`, vacancy release와 같은 경로). `_attn_pending = True`.
- pending 동안의 `first`: admissible cluster 중 seed 질량의 최대 share가 ≥ 0.5인 cluster만 채택한다(reason `attention_release`). 아니면 mode `no_admissible`, reason `attention_release_none`. seed가 없으면 거절. 다음 채택(`_adopt`)에서 pending이 풀린다. 이 규칙은 vacancy re-pick(`reselect`)보다 뒤다(reselect가 있으면 적용 안 함).
- 해제 경로의 `switch`는 일반 `_adopt` 경로라 `observe`가 source `switch`로 새 anchor를 만든다.
- 기록: decision record (`metrics`의 `admissibility` frame record `decision.identity`)에 `{rule, applies, off_share, mean_off_share, n_calls, decision, best_share, why}`; `ConfirmDecision.reason`; target metrics `identity_attention_release`(0/1), `identity_attention_off_share`.
- 꺼짐(`None`): `note_seeds`는 즉시 return, `select`는 해제 단계를 호출하지 않는다.

### 켜졌을 때만 생기는 키 (꺼진 판 = 이전과 같음)
- `ClusteringConfig` 필드 5개는 기본값으로 항상 존재한다(다른 T81 · T43 필드와 같은 방식). 꺼진 판의 서버 config override 딕셔너리에는 이 키가 들어가지 않는다.
- pipeline decision record `identity` 키, target metrics `identity_*` 키는 규칙이 하나라도 켜졌을 때만 생긴다.

## 단위 검증
```bash
source scripts/setup/env.sh
MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest tests/ag3s/test_sl8b_identity_rules.py tests/benchmark/test_sl8b_identity_versions.py -q
MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest tests/ag3s -q -p no:cacheprovider
MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-openpi-live/bin/python -m pytest tests/benchmark -q -p no:cacheprovider
```
- 새 테스트: 18 + 8 = 26 passed.
- `tests/ag3s` 전체 (`.venv-ag3s`): 2002 passed, 52 skipped, **3 failed** — `test_t55_perception.py::test_real_t53_prefix_regrounded_with_serving_attention`, `test_t55_resim.py::test_out_root_refusals`, `test_t55_resim.py::test_real_t53_prefix_parity_and_park`. 셋 모두 `Output must be inside .../results/benchmark: .../results/development/...` (development 기록을 이 checkout의 출력 규칙이 거절) 이며 이번 변경과 무관한 환경 문제다.
- `tests/benchmark` (`.venv-openpi-live`): 390 passed, 1 failed → `test_panda_sl5a.py::test_rby1_golden_identity`: 새 argparse dest 5개가 serve_safe golden의 namespace에 "only in current"로 나옴. `tests/trajopt/_sl5a_golden.py`의 `PB_DEFAULT_OFF`에 꺼진 값(None)을 추가해 고쳤다(다른 꺼진 option과 같은 방식; 켜지면 여전히 차이로 보인다). 고친 뒤 `tests/benchmark/test_panda_sl5a.py` 14 passed, golden `identical: true` (RB-Y1 path · serve_safe 둘 다).
- `tests/trajopt` (`.venv-ag3s`, 전체): 2701 passed, 48 skipped, 1 failed → `test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit`. 변경 전 commit `4957327d`의 worktree(모듈 `__file__`이 worktree를 가리키는 것을 확인)에서도 같은 test가 실패하므로 이번 변경과 무관하다(저장된 TO 궤적 npz와의 `array_equal`).
- `--help`에 `50%`가 있어 argparse가 깨지던 것(`tests/trajopt/test_no_collision_ablation.py::test_the_help_says_it_is_not_no_safe`)을 `50%%`로 고쳤고, `--help` 렌더 test를 추가했다 (신규 테스트 총 18 + 8).
- default-off bit 동일 확인 (커밋 안 한 일회성 스크립트): HEAD의 `target_grounding.py`와 작업본의 `TargetConfirm`에 같은 무작위 열(300 시퀀스 × 60 호출, 점수 · freeze · latched 무작위, mode first/hold/keep/occluded/switch 모두 등장)을 넣어 `(mode, rank, streak, frames, reason, state, manipulated_id, age, frozen, n_admissible, latched_hold)` 를 비교: 불일치 0/300.

## verifier 가 알아야 할 것
- 새 플래그 (모두 기본 꺼짐): `--identity-switch-window` `--identity-switch-votes` `--identity-attention-release` `--identity-attention-off-frac` `--identity-attention-radius-mm`. 기본값 변경 없음.
- Panda 판: `AG3S-final-panda-v3-e0to-pc-idwin` (W=5, M=3), `-idattn` (K=4, F=0.7, R=15 mm), `-idboth`. `PC_LATCH_VERSIONS` 소속이라 HSR6 probe를 받는다.
- 재생산이 필요한 산출물: 없음.
- 옛 기록과 호환: 깨지지 않는다. 새 `ClusteringConfig` 필드는 기본값이 있어 옛 config dict가 그대로 읽힌다.
- offline replay(`sl8_ats_replay.py`)가 `TargetConfirm`을 `ground_target` 밖에서 직접 쓴다면 `note_seeds`를 직접 불러야 attention 해제가 작동한다(`ground_target`을 거치면 자동).
- 알려진 한계: 해제는 유지 물체의 **마지막 관측 형상**에서의 거리만 본다. 물체가 밀려 R(15 mm)보다 멀리 가면 seed가 그 물체 위에 있어도 "밖"으로 센다. 그 경우 seed cluster가 유지 물체의 cluster이면 `same_object`로 해제하지 않고(위), 그 cluster가 아니면 교체한다.

## 내가 기대하는 결과
(verifier는 측정 뒤에 읽는다.) 창 투표는 SL8 (a) 유형(번갈아 1위)만 고치고, 해제는 (b) 유형에서 "틀린 물체를 놓음"까지만 기대한다. 계약 §1의 기대와 같다.
