# TAB3-ABLATION-FLAGS — 구현 (paper Tab. 3 행 1 · 4 · 7)

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · 2026-11-16 · branch `wip/aud-next` (base 9b454081)
> commit: 행 7 `171d21de` (lead 우선순위: 본문 표는 행 7 만) → 행 1 · 4 는 다음 commit (supp)
> 근거 지도: `/mnt/dev/AG3S_VLA/outputs/2026-11-15_tab3_ablation_map/README.md` (행 1 · 4 · 7 의 file:line lead)

## 무엇을 했나 (평이한 요약 먼저)

Tab. 3 에서 새 code 가 필요한 세 행을 기본 꺼짐 flag 셋으로 넣고, 각각 X-fwd (`pf.SOTA_FINAL_X_FWD`) 에 그 flag 하나만
더한 판 셋을 등록했다. 행 1 `--seed-source hand-nearest` 는 쥐기 전 call 의 grounding seed · peak · cluster 순위를 attention 대신
손에서의 거리로 정해 "손에 가장 가까운 물체" 를 대상으로 삼는다. 행 4 `--identity-memory off` 는 잡는 중 · 쥔 동안이 아닌
모든 call 에서 `TargetConfirm` 이 이전 identity 를 잊게 한다. 행 7 `--no-attached-query` 는 쥔 물체의 질의 구를 TO · 최종 검사
snapshot 에서 뺀다 (latch · field 제외 · self-filter 는 그대로). 셋 다 꺼짐이면 argv · 동작 · 기록 key 가 이전과 같다.

## 판 이름 (profile 맨 끝, `SOTA_TAB3_VERSIONS` block 뒤에 추가만; `GT15_VERSIONS` 에 포함)

| 행 | 판 | argv |
|---|---|---|
| 1 Seeds: attention → nearest object | `AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs-gh156a-fwd-ab1-tro` | X-fwd + `--seed-source hand-nearest` |
| 4 Identity: memory off | `…-gh156a-fwd-ab4-tro` | X-fwd + `--identity-memory off` |
| 7 Policy: no object attachment | `…-gh156a-fwd-ab7-tro` | X-fwd + `--no-attached-query` |

tuple `pf.SOTA_TAB3_CODE_VERSIONS`. base argv (start_server.sh V11AB FLAGS) 는 그대로다 — `-ab4` 도 `--latched-identity-hold`
를 base 에서 빼지 않고 code 에서 무력화한다 (아래 설계 2).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/config.py:224-239` | `ClusteringConfig.seed_source` (`"attention"` \| `"hand_nearest"`) · `identity_memory: bool = True` | 행 1 · 4 의 config key |
| `AG3S/ag3s/config.py:638, 676-682` | `SEED_SOURCES`, 검증 (값 · bool · `seed_source` 와 `grounding_source oracle` 배타) | |
| `AG3S/ag3s/stages/target_grounding.py:330-364` | `HAND_NEAREST_SCALE_M = 0.05`, `hand_proximity(points, hand, exclude_mask)` = `exp(-(d − d_min)/s)`, d = 손 구 표면까지 거리, 받침면 점 0 | 행 1 seed 신호 |
| `target_grounding.py:2162-2249` (`ground_target`) | kwarg `seed_signal=None`: 주면 flat 검사 · seed (percentile, `seed_threshold` 미적용) · peak · cluster score (`member_attention`) · SL8b `note_seeds` 가 그 신호를 읽음. split (`_split_groups`) 과 `confirm.observe(attention=…)` 는 cloud 의 attention 그대로 | 행 1: seed 출처만 교체 |
| `target_grounding.py:785-790, 1221-1231, 1391, 1586` | `TargetConfirm.identity_memory` · `memory_forgot`, `_forget_without_memory()` 를 `select` · `unseen` 첫머리 (hand gap 전) 에서 호출: off ∧ not frozen 이면 `_forget_manipulated()` | 행 4 |
| `target_grounding.py:2399, 2540-2546` | `_memory_metrics`: off 일 때만 `identity_memory_off` · `identity_memory_forgot` metrics | 기록 (off 에만 key) |
| `AG3S/ag3s/runtime/pipeline.py:234-236, 521` | `_seed_source_frame` (init · reset) | 행 1 기록 |
| `pipeline.py:632-661` | `AG3S._hand_nearest_seed_signal(cloud, exclude, hand)`: attached · frozen · 이번 call close seed 적용 · 손 없음 · 점 없음이면 None (`skipped` 기록) | 행 1 적용 조건 |
| `pipeline.py:3259-3260, 3283-3296, 3308-3314, 3733-3734` | `_hand_for_grounding` 결과를 `hand_now` 로 한 번 계산해 `note_hand` 와 공유 (동작 같음); `seed_source == hand_nearest` 일 때만 `seed_signal` 을 `ground_target` 에 넘김; `metrics["seed_source"]` (applied · skipped · status · n_target_points · target_hand_gap_mm) | 행 1 |
| `AG3S/trajopt/config.py:1387-1392, 1466-1467, 1706` | `CollisionBackendConfig.attached_query: bool = True`, 검증, `_OMIT_WHEN_DEFAULT` (꺼짐이면 config dump key 그대로) | 행 7 |
| `AG3S/trajopt/linearize.py:2928-2936, 3043, 3052` (`scene_from_constraint_set`) | `holding = attached_points is not None`; `attached_query` False 면 `attached_points / radii / parent_link = None`. held capture volume 두 곳은 `holding` 으로 (꺼짐이면 같은 값) | 행 7: 질의 구만 끔 |
| `AG3S/trajopt/safe_policy.py:3276-3277` (`_finish_grasp_record`) | attach 중이고 flag 켜짐이면 `grasp.attached.queried = False` | 행 7 call 별 기록 |
| `AG3S/trajopt/serve_safe.py:2859-2875` | argparse `--seed-source {attention,hand-nearest}` · `--identity-memory {on,off}` · `--no-attached-query` | |
| `serve_safe.py:1093-1134` | `reject_tab3_flags` (아래 조합 거절), `tab3_inert` (`-ab4` 에서 `--latched-identity-hold` inert 기록) | |
| `serve_safe.py:1154-1157` | `t43t_identity_section`: `seed_source: "hand_nearest"` · `identity_memory: False` (켜짐에만 key) → `build_ag3s(identity=…)` · `assert_ag3s_config` 경로 | RB-Y1 · Panda 공통 |
| `serve_safe.py:5050, 5427-5431, 5710-5711` | `reject_bad_flag_combinations` 에서 `reject_tab3_flags` 호출; `attached_query_overrides` 를 `trajopt_config_from_args` collision 에 | |
| `AG3S/adapters/panda/server.py:778-781` | `PORTED_FINAL_CONFIG_DESTS` += `seed_source`, `identity_memory`, `no_attached_query` | Panda 서버가 받음 |
| `server.py:857-868, 1221-1223, 1252` | `tab3_record(args)` → `info["tab3_ablation"]` (켜짐에만); `ss.tab3_inert` 를 `inert_on_panda` 에 | 서버 기록 |
| `AG3S/adapters/panda/profile.py:1534-1548` | `SOTA_TAB3_CODE_VERSIONS` (`-ab1` · `-ab4` · `-ab7`), `TAB3_CODE_DESTS` | 판 등록 |
| `profile.py:1792, 1797-1801` (`version_flag_values`) | `gt16` 에서 `TAB3_CODE_DESTS` 제외; 그 dest 는 **기본값이 아닐 때 그 key 하나만** 기록에 더함 | 이전 판 (X-fwd 포함) 기록 key 불변 |
| `tests/benchmark/test_sl_sota18_hand_sized.py:81`, `tests/trajopt/test_sl_gtto11_scaled_qp.py:59`, `tests/trajopt/test_sl_sota11_opening_rule_reach_sticky.py:122` | 판 목록 고정 test 에 `+ pf.SOTA_TAB3_CODE_VERSIONS` (e3d71f8a 와 같은 방식) | |
| `tests/ag3s/test_tab3_identity_grounding.py` (새, 12) · `tests/trajopt/test_tab3_ablation_flags.py` (새, 16) | 아래 | |

## 설계 선택

1. **행 1 — seed 출처만 바꾼다.** X 에서 attention 을 읽는 곳은 (a) grounding seed · peak · cluster score, (b) T31b attention split
   (`--attention-split-*`, raw ratio), (c) SL-SOTA18b hand-sized gate (cluster `mean_attention` 1 위), (d) SL-PE1 close seed
   (VLA chunk 의 닫는 점 — attention 이 아니라 action; X 에 `--close-seed-attention-ratio` 없음), (e) release split 의 기하
   attention, (f) 쥔 동안 destination 등록 (`--destination-held-leader`: score 1 위).
   - (a) 교체 — 이것이 행의 뜻. score 의 attention 항이 proximity 평균이 되므로 순위 = 손과의 거리.
   - (b) **그대로 attention**: split 은 "붙은 물체를 따로 본다" 는 segmentation (extent) 이고, 쪼갠 뒤 성분 사이의 대상 선택은
     (a) 의 score 가 한다. proximity 로 바꾸면 extent 까지 바뀐다 (두 요소).
   - (c) **seed 신호를 따름** (code 변경 없이: gate 가 `ClusterInfo.mean_attention` 을 읽고 그것이 proximity 이므로).
     gate 는 "어느 물체가 대상인가" 의 규칙이라 attention 으로 남기면 attention 이 identity 에 남는다 (ablation 이 attention 의
     기여를 덜 재게 됨). 효과: 손 크기 후보는 가장 가까울 때만 대상 — 사실상 (a) 와 같은 순위라 거의 inert.
   - (d) **유지**: close seed 가 적용된 call 은 hand-nearest 를 쓰지 않는다 (`skipped: close_seed`). close seed 는 attention 이
     아니라 VLA action 이므로 빼면 두 요소 교체 (README 행 1 권장과 같음). 원고에 "close seed 유지" 를 적어야 한다.
   - (e) attention 그대로 (release split 은 extent). (f) attach 중 · frozen 에는 hand-nearest 를 쓰지 않으므로 destination
     등록 · 쥔 동안의 score 는 attention 그대로 (`skipped: attached / frozen`).
   - scale 50 mm, `d_min` 정규화 (가장 가까운 점 = 1). `seed_threshold` (attention 값) 는 proximity 에 적용하지 않음 (Panda config
     는 None 이라 X 에서는 차이 없음). flat-attention 검사도 proximity 로 하므로 attention 이 평평해도 grounding 된다.
2. **행 4 — 기억만 끈다, freeze 는 남긴다.** off 이면 not frozen call 의 `select` / `unseen` 첫머리에서 `_forget_manipulated()` →
   `_held is None` 경로가 그 call 의 1 위 admissible cluster 를 `first` 로 채택 (매 call 새 id). 따라서 3-frame hysteresis ·
   occluded 마지막 기하 · T30 association · anchor · IFG-FOLLOW · T43 3(a) latched hold 가 unfrozen call 에서 읽히지 않는다.
   **frozen (closing → held → release) 은 그대로**: grasp latch 의 물체 lock 이고 방법 절 l.71 "suspended once attachment
   locks" 와 같다 (reach-sticky 의 같은 `manipulated_id`, closing-keep-geometry 도 이 안이라 영향 없음).
   `--latched-identity-hold` 는 base argv 라 판 flag 로 지울 수 없지만, 그 규칙은 `counts` (not frozen 필요) 안에서만 돌고
   unfrozen call 은 매번 잊으므로 **code 상 inert** — `inert_on_panda` 에 `--latched-identity-hold` 로 기록한다.
   destination registry 의 frame hysteresis 는 대상 identity 가 아니라 목적지라 그대로 (기록 필요 시 원고에 명시).
3. **행 7 — 질의 구만 끈다.** `scene_from_constraint_set` 한 곳 (TO · 최종 검사 · pass-through 가 같은 snapshot,
   `safe_policy.py:1423`). `constraint_set.attached` 는 남아 holding-state 규칙 (strict held permission rows, target volume 없음,
   nonauthorized rows 없음, look-ahead 의 held capture volume) 은 그대로다. 쥔 물체 행을 쓰는 X 규칙 (`--held-body-cover` ·
   `--held-point-model-mm` · `--held-beneath-contact` · `--hold-follow-to` 의 쥔 물체 행, destination margin) 은 행이 없어 inert —
   attachment 의 일부로 본다 (README 행 7 (b)). `fm_guide` 는 scene 의 `attached_points` 를 읽으므로 자동으로 같음 (X 에서 off).
4. **기록 규칙.** 켜짐에만 key: `metrics.seed_source` (call 별), target `metrics.identity_memory_off/_forgot`,
   `grasp.attached.queried`, 서버 `info.tab3_ablation`, `version_flag_values` 의 해당 dest 하나. clustering override 도 켜짐에만.
   예외 (이전 flag 들과 같은 관례): `ClusteringConfig` · `CollisionBackendConfig` 의 dataclass field 가 늘었다. AG3S config 전체
   dump (`AG3SConfig.to_dict`) 에는 `seed_source: attention` · `identity_memory: true` 가 새로 보인다 (AG3S config 에 omit 장치가
   없음 — SL-SOTA18b · SL-SOTA19 때와 같음). TrajOpt config dump 는 `_OMIT_WHEN_DEFAULT` 로 그대로.
5. **flag 조합 거절** (`reject_tab3_flags`): hand-nearest × (`--no-safe` / `--no-perception`, `--grounding-source oracle`,
   `--target-extent attention-surface*`, `--attention-peak-seed`, `--identity-attention-release`); memory off ×
   (`--no-safe` / `--no-perception`, `--identity-switch-window`, `--identity-attention-release`, `--target-vacancy-release`,
   `--refused-identity-release`); `--no-attached-query` × (`--no-safe` / `--no-perception`). X-fwd 에는 이 중 하나도 없다.

## 단위 검증 (CPU, `JAX_PLATFORMS=cpu`, 한 process, nice 10)

```bash
pytest -q tests/ag3s/test_tab3_identity_grounding.py tests/trajopt/test_tab3_ablation_flags.py
pytest -q tests/benchmark tests/trajopt -k "version or flag" --ignore=tests/trajopt/test_t41b_rest_start.py \
       --ignore=tests/trajopt/test_t41c_hold_measured.py --ignore=tests/trajopt/test_t43q_replay.py
pytest -q tests/ag3s -k "confirm or identity or grounding"
```

결과 (2026-11-16):
- 새 test 2 개 + 판 고정 test 4 개: **52 passed** (`test_tab3_identity_grounding` 12 · `test_tab3_ablation_flags` 16 ·
  `test_tab3_ablation_versions` · `test_sl_sota18_hand_sized` · `test_sl_gtto11_scaled_qp` · `test_sl_sota11_…`).
  행 7 만의 commit (171d21de) 시점: 행 7 test + 판 고정 + `test_t32_held_query` = 37 passed.
- `tests/benchmark tests/trajopt -k "version or flag"` (3 ignore): **453 passed, 10 failed, 42 skipped** — 10 개 모두 알려진 base 실패:
  7 = `sl6b-hsr6-panda-labels/probe/hsr6_panda_kv_L4.npz` 없음 (FileNotFoundError 확인: sl10 abox · sl10b termsoff · sl10d hfq ·
  sl12 ×2 · sl13 ×2), 1 = `test_sl_gtto_gt_perception::test_only_the_gt_versions_require_the_segmentation`, 2 =
  `test_launcher_record_constraints` (worktree 에 LIBERO submodule 없음). 그 밖 실패 없음.
- `tests/ag3s -k "confirm or identity or grounding"`: **199 passed, 9 skipped** (새 file 포함, 기존 TargetConfirm · T43T · SL8b ·
  SL-PE · split test 그대로 통과 = 꺼짐 동작 불변).

새 test 가 보이는 것 (모두 실제 함수):
- 행 1: `hand_proximity` 값 (가장 가까운 점 1, 받침면 0, 없음 → None); `ground_target` 이 attention (배 0.9) 이 아니라 손 쪽 사과를
  대상으로, 평평한 attention 에서도 grounding; `TargetConfirm` 와 함께 `first`; **실제 `AG3S.process`** (RB-Y1 hand model, 점군
  입력 — attention 없음) 에서 off = 대상 없음 · `metrics.seed_source` 없음, on = 손 아래 공이 대상 · 기록 `applied` · frozen 에서
  `skipped: frozen`.
- 행 4: on = T31b 규칙 (occluded, occluded, switch) · key 없음; off = 즉시 `first` (새 id) · 다음 call 다시 1 위 · 군집 없는 call 에
  identity 없음; latched hold 가 off 에서 무력; frozen 동안 identity 유지 → release 후 잊음; AG3S 가 config 를 TargetConfirm 에 전달.
- 행 7: config 기본 · dump · 검증; 실제 `scene_from_constraint_set` 에서 off = 질의 구 2 개, on = None, held capture volume 은 둘 다
  유지, 안 쥐었으면 차이 없음, SL-GTTO17 point model 도 빠짐; `_finish_grasp_record` 의 `queried` key; **X-fwd argv → config →
  snapshot** 과 `-ab7` argv → snapshot.
- 판: argv = X-fwd + flag 하나, registry 끝, `parse_v11ab_args` (= `reject_bad_flag_combinations`) · `check_unported_flags` ·
  `check_inpass_flags` · `trajopt_config_from_args().validate()`, `version_flag_values` 가 X-fwd 의 key · 값을 그대로 두고 key
  하나만 더함, `final_options_on` 차이 = 그 dest, identity section → `AG3SConfig.with_overrides` · `assert_ag3s_config`,
  `-ab2/-ab5/-ab6` · X-fwd 기록에 새 dest 없음; 거절 조합 7 + 2 개.

## verifier 가 알아야 할 것

- 새 flag (모두 기본 꺼짐): `--seed-source`, `--identity-memory`, `--no-attached-query`. 기본값 변경 없음.
- 재생산이 필요한 산출물: 없음.
- 옛 기록과 호환: X-fwd 및 이전 판의 argv · `version_flag_values` · 서버 info key 불변. AG3S config 전체 dump 에 clustering key 2 개
  추가 (설계 4). `match_ag3s_version` 은 판마다 같은 함수로 비교하므로 영향 없음 (느려서 test 에서 부르지 않음).
- 서버 기록에서 확인할 곳: 행 1 `metrics.seed_source.applied` 비율 (쥐기 전 call 중 close seed 가 아닌 것), 행 4 target
  `metrics.identity_memory_forgot`, `inert_on_panda` 의 `--latched-identity-hold`, 행 7 `grasp.attached.queried == false`.

## 한계 · 열린 질문

- 행 1: proximity scale 50 mm 는 고정 상수 (flag 아님, 근거는 물체 한 폭). close seed 를 남긴 것 · split 을 attention 으로 둔 것은
  원고에 적어야 한다 (lead 판단 필요 시 flag 하나로 close seed 도 끌 수 있음 — 지금은 구현하지 않음).
- 행 1: 손 위치는 constraint model 의 palm + finger 구 (T32b `hand_spheres`) — 손끝 중점이 아니다. 손이 물체에서 멀 때 (episode
  시작) 도 가장 가까운 물체가 대상이 되며, attention 이 없는 call 에서도 grounding 된다 (off 는 NO_ATTENTION).
- 행 4: 매 call 새 `manipulated_id` 가 생긴다. id 로 묶는 규칙 중 X 에서 unfrozen 에 도는 것은 확인한 범위 (latch reach-sticky:
  CLOSING 안, IFG-HSR pin: closing-kept) 에 없지만, closed-loop 기록의 id 연속성 통계는 이 판에서 뜻이 없다.
- 행 7: 쥔 물체가 field 에서 빠지고 질의도 없으므로 운반 중 쥔 물체-장애물 충돌은 TO 가 보지 않는다 (의도). 최종 검사도 같은
  snapshot 이라 그 충돌은 verdict 에도 안 잡힌다 — SafeLIBERO 의 위반 판정 (simulator 접촉) 으로만 보인다.
- GPU closed-loop 은 돌리지 않았다 (CPU 단위 검증만).
