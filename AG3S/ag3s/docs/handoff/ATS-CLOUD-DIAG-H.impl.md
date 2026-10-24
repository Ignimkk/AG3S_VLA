# ATS-CLOUD-DIAG — 구현 (진단만, Implementer H)

> writer: ag3s-implementer (H) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-24 (UTC 05:55 = 14:55 KST)
> checkout `/mnt/dev/AG3S_VLA-benchmark` · branch `feature/panda-benchmark` · HEAD `3c677553` · **code 수정 없음, commit 없음**
> 대상 run: `20261012T225820_46a08d6ff644_sl4-perception-dev` (commit `46a08d6f`, dirty false, `save_clouds: true`)
> CPU worker 1개 · `.venv-ag3s` · GPU 사용 안 함. 분석 script 와 산출물은 모두 `outputs/ats_cloud_diag/` 아래에 있다.

## 무엇을 했나 (평이한 요약 먼저)

`cloud_target` 이 비는 것은 기록 버그가 아니다. V11AB grounding 은 일부 frame 에서 target 을 **이번 frame cloud 에서
새로 찾은 cluster** 대신 **이전 frame 에 관측한 geometry(stand-in)** 로 보고한다. 이 stand-in 은 설계상
`point_indices` 가 비어 있다. 그 점들은 이번 cloud 의 점이 아니기 때문이다. `perception_replay` 는 `cloud_target` 을
`point_indices` 로만 세우므로 stand-in call 에서는 flag 가 0 개가 된다.
221 call 은 다섯 범주로 나뉜다. fresh 45, anchor stand-in 62, occluded stand-in 72, has_target=False 이지만 ESDF 가
여전히 manipulated geometry 를 받는 call 38, 아무것도 없는 call 4 이다.

`target_points` 가 cloud 와 맞지 않는 28 / 17 call 은 모두 stand-in 이다. 그 geometry 를 관측한 뒤 물체가 움직였거나
(들어 올림·운반, GT 이동 41~471 mm), wrist camera 가 움직여 fused 점 좌표가 조금 달라진 경우다. 다른 cloud, 다른
좌표계, 변환 오류는 아니다.

공정 비교에는 **(b+) "ESDF 가 실제로 받은 target geometry"** 를 추천한다. 이는 `target_points` 에, has_target=False 이면서
exclusion 이 active 인 38 call 에서는 직전 geometry 를 이어 붙인 것이다. 이것을 같은 cloud 에 **5 mm**
(V11AB fusion voxel 10 mm 의 절반) 로 대응시킨다. 이 표현은 기록된 npz 만으로 정확히 재구성되므로
perception 재실행 (c) 은 필요 없다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| (없음) | code 수정 없음. 진단 task 범위 | — |
| `outputs/ats_cloud_diag/diag_calls.py` (신규, 분석) | call 별 범주 분류, provenance(byte 비교), cloud 대응, GT 이동 측정 → `calls_diag.jsonl`, `summary_diag.json`, `diag_calls.log` | Q1·Q2 |
| `outputs/ats_cloud_diag/recompute_ats_arms.py` (신규, 분석) | F 의 `cloud_labeling` 함수를 그대로 써서 ATS 표현 7 종을 같은 점·같은 GT class 로 다시 채점 → `recompute/summary_ats_arms.json`, `recompute/calls_ats_arms.jsonl` | Q3 (예비 수치) |
| `outputs/ats_cloud_diag/fig_diag.py` (신규, 진단 그림) | `outputs/ats_cloud_diag/figures/diag-{call-categories,scene-stale,arms}.png` + 같은 이름 `.json` sidecar | 규칙 A. 공식 figure 는 verifier 가 만든다. `docs/figures/` 에는 쓰지 않았다 |

## Q1 — 왜 `cloud_target` 이 비는가

### 결론 (확인된 사실)

**원인은 V11AB `process_multi_debug` 의 산출물 구조다.** `perception_replay` 의 기록 경로는 이 run 에서 정상이다.

| 점검 항목 | 결과 | 증거 |
|---|---|---|
| `--no-clouds` 였나 | 아니다 | `run_manifest.json` `"save_clouds": true`, npz 에 `cloud_points`/`cloud_target` 이 있다 |
| 다른 cloud 에 flag 를 달았나 | 아니다 | grounding 은 `AttentionPointCloud(cloud, …)` 위에서 돈다 (`pipeline.py@46a08d6f:1475`, `ground_target(` `:1513`). replay 는 같은 `debug["filtered_cloud"]` 를 기록한다 (`perception_replay.py@46a08d6f:385` 가 `pipeline.py@46a08d6f:1852` 의 `"filtered_cloud": cloud` 를 읽는다) |
| downsample 이후 index 가 어긋났나 | 아니다 | fresh 45 call 전부에서 `nonzero(cloud_target) == target_indices`, `target_points == cloud_points[target_indices]` (float32 byte 동일) 가 성립한다. `summary_diag.json` `fresh_consistency` = 45/45/45 |
| flag 를 세우는 규칙 | `point_indices` 만 쓴다 | `perception_replay.py@46a08d6f:420-422` (HEAD `:629-631`): `target_flag[target.point_indices] = True`. `:404` (HEAD `:610`) `target_indices` 도 같은 배열이다 |
| stand-in 의 `point_indices` | **설계상 비어 있다** | `target_grounding.py:1959` (`_anchor_result`: `point_indices=np.zeros(0)`), `:1995` (`_unobserved_result`: 같음), `:1056` (release split, 이 run 에서는 0 회). docstring `:1966-1971`: "point_indices is empty (none of *this* cloud's points are it)" |
| stand-in 으로 가는 분기 | `ground_target` | `target_grounding.py:1764-1777`. `decision.cluster is None` 이면 `_unobserved_result` (occluded 또는 lost), `decision.reason in (SUBSET_KEPT_ANCHOR, UNSPLIT_KEPT_ANCHOR, CLOSING_KEPT_GEOMETRY)` 이면 `_anchor_result` 다 |
| 부수 증상 | stand-in 은 `target_seed_camera` 가 비어 있다 | `pipeline.py@46a08d6f:2956-2975` `_with_camera_provenance`: index 가 비면 `seed_camera=None` 이다. frames.jsonl `target_seed_camera is None` ↔ index 0 이 179 call 전부에서 일치한다 |

verifier audit (`outputs/verify/BASELINE-EVAL-V1/ats_cloud_target_audit.txt`) 의 `ti_valid: 0` 은 위 결과와 같은 사실을
가리킨다. 134 call 모두 `target_indices` 길이가 0 이다 (`summary_diag.json` `standin_provenance.n_target_indices_nonzero = 0`).

### stand-in 이 "이전 frame 의 cluster" 라는 증거 (확인된 사실)

- stand-in 134 call 전부에서 `target_points` 는 `manipulated.anchor.geometry_frame` call 의 `target_points` 와
  **byte 단위로 같다** (134/134, `standin_provenance.tp_byte_equal_geom_frame`). geometry frame 은 모두 fresh call 이다.
- has_target 179 call 전부에서 `mean(target_points)` 와 npz `manipulated_centroid` 의 차이는 0.01 mm 미만이다.
  따라서 `target_points` 가 곧 ESDF 에 들어간 manipulated geometry 다. ESDF 는 `manipulated_geometry` 를 받고
  (`pipeline.py@46a08d6f:1542`, `:1679`), 그 `target.points` 로 `TARGET_LABEL` 과 target-free layer 를 만든다 (`:2511`, `:2547`, `:2566`).
- has_target=False 인 42 call 중 38 call 에서는 `manipulated_state = occluded`, `exclusion.active = true`, `has_target_free = true`,
  ESDF `label_names ∋ 'target'` 이다. note 는 *"contact permission and target exclusion keep its last observed geometry"*
  (예: t0 call 18) 이다. 이 38 call 에서도 ATS 는 cuRobo 에 stale geometry 를 넘긴다. cloud-labeling 은 이를 "예측 없음" 으로 셌다.
  직전 has_target call 의 `target_points` 를 이어 붙인 것은 geometry-frame cluster 와 byte 동일하고(38/38),
  centroid 는 `manipulated_centroid` 와 4.6e-6 mm 안에서 같다.
- `observe()` 의 `_keep_anchor` 분기(`target_grounding.py:1269-1270`) 는 fresh target 과 ESDF geometry 를 다르게 만들 수 있다.
  이 run 에서는 fresh 45 call 모두 centroid 가 같아 이 분기가 발생하지 않았다.

### call 별 원인 범주 × call 수 (221 call, `summary_diag.json` `by_cat_detail`)

| 범주 | 코드 경로 | 세부 (status · manipulated state · association.outcome) | call | `cloud_target` > 0 | `target_points` > 0 | ESDF target-free |
|---|---|---|---:|---:|---:|---:|
| fresh | `target_grounding.py:1777-1834` | ok·visible·(accepted 28 / 첫 frame·switch 등 None 14 / rejected 3) | **45** | 45 | 45 | 45 |
| anchor stand-in | `_anchor_result` `:1935-1961` | ok·visible·subset_kept_anchor 61, ok·visible·accepted 1 (t1 call 8, geometry frame 7) | **62** | 0 | 62 | 62 |
| occluded stand-in | `_unobserved_result` `:1964-1997` | ok·occluded·None 56, ok·occluded·rejected 16 | **72** | 0 | 72 | 72 |
| has_target=False, geometry 는 계속 제외 | `_unseen` (`target_grounding.py:1929`) + `pipeline.py@46a08d6f:1597-1602` no-target 분기 note | no_cluster·occluded 27, no_seed·occluded 11 | **38** | 0 | 0 | 38 |
| target 없음, 제외도 없음 | — | no_admissible (t4 call 0-3, manipulated 없음) | **4** | 0 | 0 | 0 |
| 합계 | | | 221 | 45 | 179 | 217 |

`association.outcome` 은 frames.jsonl `manipulated.association` 의 값이다. no-target frame 에서는 이전 frame 의 기록이 남아 있을
수 있으므로 occluded 의 세부 이유 (inadmissible_match / association_rejected / 손 가림) 를 가르는 데는 쓰지 않는다.
`target.metrics["manipulated_occluded"]` 등 stand-in 표지는 npz 에 기록되지 않는다. 그래서 범주는
`target_indices`·`manipulated_state` 로 나눴다.

정성 예시:

- t0 call 1 (obs 15): `subset_kept_anchor` 이고 coverage 0.81, containment 0.94 다. 이번 frame cluster 는 anchor 의 부분집합이라
  frame 0 의 154 점이 그대로 target 이 된다 (`target_n_points` 154 → `cloud_target` 0).
- t1 call 20 (obs 110): occluded 이고 geometry frame 은 7 이다. 그 사이 GT target 이 326 mm 이동했다
  (`figures/diag-scene-stale.png`: 주황 점은 call 7 의 bowl 위치에 남아 있고, GT bowl 은 x 0.75 m · z 0.08 m 로 들려 있다).
- t0 call 18 (obs 100): no_cluster 이고 has_target=False 지만, ESDF note 는 "target exclusion keep its last observed geometry" 다.

## Q2 — `target_points` 가 cloud 점과 맞지 않는 28 / 17 call

### 결론

다른 cloud 도 아니고 좌표 변환도 아니다. **stale geometry 다.** 28 / 17 call 은 모두 stand-in 이다. 그 geometry 는
과거 frame 의 cluster 이고(byte 동일), 지금 cloud 는 물체가 움직인 뒤의 장면이다.

| 집합 | call | 범주 | 그 geometry frame 이후 GT target 이동 | 이번 cloud 에 GT target 점이 있는가 |
|---|---:|---|---|---|
| exact (1e-6 m) 대응 0 점 | 28 | anchor stand-in 6, occluded stand-in 22 | 1-5 mm 2, 5-10 mm 1, 10-50 mm 7, ≥ 50 mm 18 | 28/28 있음 (`n_gt` 127~202) |
| 5 mm 대응 0 점 | 17 | occluded stand-in 17 | 10-50 mm 1 (41 mm), ≥ 50 mm 16 (최대 471 mm) | 17/17 있음 |

call 목록은 `summary_diag.json` `supp_mismatch.*.calls` 에 있다. 형식은 (stem, k, 범주, geometry 이후 call 수, GT 이동 mm,
target_points→cloud 거리 중앙값 mm, n_gt) 이다.

범주별 대응률 (`summary_diag.json` `match_by_cat`, has geometry 인 call):

| 범주 | call | 대응 ≥1 점 exact / 5 mm / 10 mm | target_points 중 cloud 와 대응하는 비율 중앙값 exact / 5 mm / 10 mm | GT 이동 > 10 mm | geometry 이후 call 수 중앙값 |
|---|---:|---|---|---:|---:|
| fresh | 45 | 45 / 45 / 45 | 1.00 / 1.00 / 1.00 | 0 | 0 |
| anchor stand-in | 62 | 56 / 62 / 62 | 0.60 / 0.86 / 0.93 | 12 | 4 |
| occluded stand-in | 72 | 50 / 55 / 64 | 0.56 / 0.67 / 0.74 | 38 | 5 |
| has_target=False 제외 (이어 붙인 geometry) | 38 | 14 / 15 / 31 | 0.00 / 0.00 / 0.07 | 38 | 12 |

- **확인된 사실**: V11AB fused cloud 는 1 cm 격자 위의 점이 아니다. 격자 corner 나 centre 위에 있는 점의 비율은 0.0 이다
  (`summary_diag.json` `grid_check`). 따라서 정지 물체에서 exact 일치가 나오는 것은 격자 양자화 때문이 아니다.
- **부분 확인된 가설**: 정지 물체(GT 이동 < 1 mm)의 anchor stand-in 에서 exact 일치한 cloud 점은 camera mask 가 head(1) 1773 점,
  head+wrist(3) 2672 점이었다. wrist 단독(2) 점은 0 점이다. 5 mm 안이지만 exact 가 아닌 점은 wrist 단독 1388 점, head 430 점,
  둘 다 277 점이다. 가설은 고정된 head camera 와 결정적인 sim 렌더링이 같은 fused 점을 다시 만들고, 움직인 wrist camera 가
  기여한 점만 mm 단위로 바뀐다는 것이다. 그래서 exact 대응은 물체가 정지해 있어도 부분적이며, 좌표 일치 판정에는 tolerance 가 필요하다.
- t4 · t5 · t9 일부 call 에서 ATS geometry 는 BDDL target 이 아니다. geometry-frame 의 fresh cluster 에서 GT target 점이 절반 미만이었고,
  이는 SL4c 의 오선택과 같다. 이런 call 에서는 위의 "GT target 이동" 이 ATS 가 잡은 물체의 이동을 뜻하지 않는다.
  wrong-object geometry 를 쓰는 call 수는 fresh 21, anchor 7, occluded 19, 제외 9 다.

## Q3 — 공정 비교에 쓸 ATS 표현

### 예비 수치 (H 측정, verifier 재확인 필요)

`recompute_ats_arms.py` 는 F 의 `point_metrics`·`gt_classes`·`summarize` 를 수정 없이 사용한다. 같은 점과 같은 GT class 로
채점했고, F 의 ATS 행과 221/221 동일하다 (`identity_check_vs_F_ats_rows.all_equal = true`). 값은 episode 평균(10 episode)이다.

| ATS 표현 | found (pred > 0) | IoU | precision (n call) | recall | neighbour_frac | pooled IoU / P / R |
|---|---:|---:|---|---:|---:|---|
| (a) `cloud_target` (= F 의 ATS 행) | 45/221 | 0.082 | 0.693 (45) | 0.084 | 0.301 | 0.082 / 0.534 / 0.088 |
| (b) `target_points` exact | 151/221 | 0.190 | 0.636 (151) | 0.192 | 0.352 | 0.174 / 0.546 / 0.204 |
| (b) `target_points` 5 mm | 162/221 | 0.258 | 0.649 (162) | 0.261 | 0.338 | 0.230 / 0.594 / 0.272 |
| (b) `target_points` 10 mm | 171/221 | 0.321 | 0.596 (171) | 0.329 | 0.322 | 0.279 / 0.614 / 0.339 |
| (b+) exclusion geometry exact | 165/221 | 0.190 | 0.621 (165) | 0.192 | 0.367 | 0.169 / 0.499 / 0.204 |
| **(b+) exclusion geometry 5 mm (추천)** | 177/221 | 0.259 | 0.633 (177) | 0.262 | 0.353 | 0.224 / 0.551 / 0.273 |
| (b+) exclusion geometry 10 mm | 202/221 | 0.324 | 0.524 (202) | 0.333 | 0.327 | 0.273 / 0.572 / 0.343 |
| ATS seeds (참고) | 210/221 | 0.207 | 0.569 (210) | 0.232 | 0.379 | 0.205 / 0.567 / 0.243 |
| GSAM2 per-frame · instruction | 219/221 | 0.486 | 0.593 (219) | 0.655 | 0.331 | 0.432 / 0.550 / 0.668 |
| GSAM · instruction | 221/221 | 0.493 | 0.554 (221) | 0.684 | 0.385 | 0.423 / 0.519 / 0.695 |

(b) 의 exact 와 5 mm 행은 verifier 보충값(`numbers.cloud_labeling_supplement_verifier`)과 같다. 다만 (a) 와 (b) 의 차이는
verifier 의 found 151 / 162 와 표현 정의가 같아서 생긴 것이다. 이 값은 독립 재현이 아니다.

(b+) 5 mm 의 범주별 pooled 값 (`recompute/summary_ats_arms.json` `per_category.ats_ex_5mm`):

| 범주 | call | found | pooled precision | pooled recall | neighbour_frac |
|---|---:|---:|---:|---:|---:|
| fresh | 45 | 45 | 0.536 | 0.415 | 0.460 |
| anchor stand-in | 62 | 62 | 0.864 | 0.525 | 0.136 |
| occluded stand-in | 72 | 55 | 0.297 | 0.113 | 0.641 |
| has_target=False 제외 | 38 | 15 | 0.027 | 0.006 | 0.826 |

anchor stand-in 은 fresh 보다 오히려 정확하다. 정지한 물체의 온전한 anchor 를 유지하기 때문이다. 오차는 occluded 와 제외 call 에
몰려 있다. 물체가 떠난 자리이거나 wrong-object 다. (a) 는 이 두 성질을 모두 지운다.

### 대안별 장단점

| 대안 | 장점 | 단점 | 판정 |
|---|---|---|---|
| (a) `cloud_target` | index 가 정확하고 tolerance parameter 가 없다 | ATS 가 cuRobo 에 target 을 넘긴 217 call 중 172 call 을 "예측 없음" 으로 센다. precision 은 fresh 45 call 에서만 계산되어 선택 편향이 있다. 측정하는 것은 "이번 frame 에 새 cluster 가 났는가" 이지 ATS 의 출력이 아니다 | 진단 행으로만 남긴다 ("fresh-only") |
| (b) `target_points` → cloud (tol) | `grounding.target` 그 자체이고 ESDF geometry 와 centroid 가 같다 (179/179). 재실행이 필요 없다. stale geometry 가 그대로 오차로 잡힌다 | tolerance 를 정해야 한다 (exact 151 · 5 mm 162 · 10 mm 171 found). has_target=False 인데 제외가 걸린 38 call 이 빠진다 | 차선 |
| **(b+) exclusion geometry → cloud (5 mm)** | **cuRobo 가 실제로 받은 target 영역** 이다 (target-free layer active 217/221). 프로젝트 목적과 같은 대상을 잰다. npz 만으로 정확히 재구성된다 (이어 붙인 geometry 가 geometry-frame cluster 와 38/38 byte 동일하고, centroid 가 `manipulated_centroid` 와 같다). 재실행이 필요 없다 | 재구성은 "unseen frame 동안 geometry 가 바뀌지 않는다" 는 불변식에 기댄다. 이 run 에서 release split 은 0 회였고 centroid 검사로 매 call 확인할 수 있다. tolerance 는 (b) 와 같은 문제다 | **추천** |
| (c) perception 재실행 + index 기록 | 재구성 가정이 없어진다 | stand-in 점은 정의상 이번 cloud 의 index 를 갖지 않는다. 재실행해도 (a) 가 stand-in 을 덮게 되지 않는다. 얻는 것은 (b+) 와 같은 geometry 의 명시 기록뿐이고 GPU 재실행 · 새 run id · 동일성 확인 비용이 든다. "이번 frame 의 관측 subset cluster" index 를 따로 기록할 수는 있지만, 그것은 ATS 가 cuRobo 에 넘기는 것이 아니다 | 지금은 불필요. 다음 perception run 을 위한 기록 옵션으로만 제안 |

tolerance 근거: V11AB fusion voxel 은 `fusion.fusion_voxel_size_m = 0.01` 이다 (221/221). 5 mm 는 그 반경, 곧 한 fused 점이 대표하는
voxel 의 반 폭이다. segmenter 는 이번 frame mask 로 같은 점을 칠하므로, ATS 에 voxel 반 폭 이상의 여유를 주면 ATS 에 유리해진다.
exact 와 10 mm 는 sensitivity 행으로 같이 보고할 것을 권한다. ESDF voxel(20 mm) 기준의 footprint 는 이 비교의 대상이 아니다.

## 다시 계산하는 명령

### 지금 바로 (code 변경 없이, 분석 script)

```bash
cd /mnt/dev/AG3S_VLA-benchmark
# 1. call 별 분류와 provenance (≈ 10 s)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa PYTHONPATH=$PWD \
  .venv-ag3s/bin/python outputs/ats_cloud_diag/diag_calls.py
# 2. ATS 표현 7 종 재채점 (F 의 calls.jsonl 의 segmenter 행 + 새 ATS 행, ≈ 100 s, RSS < 0.3 GB)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=$PWD \
  .venv-ag3s/bin/python outputs/ats_cloud_diag/recompute_ats_arms.py
# 3. 진단 그림
PYTHONPATH=$PWD .venv-ag3s/bin/python outputs/ats_cloud_diag/fig_diag.py
```

입력은 다음과 같다. ATS 는 `…/20261012T225820_46a08d6ff644_sl4-perception-dev`, F 의 행은
`…/20261024T035953_5752c64c7006_baseline-eval-cloud-sl3dev/calls.jsonl`, scene GT 는 `outputs/verify/SL4/raw/scene` 이다.
paired bootstrap 의 기준은 `ref_ats_ex_5mm` 이다 (`recompute/summary_ats_arms.json`).

### 정식 경로 — `eval/cloud_labeling.py` 에 옵션이 필요하다 (현재 옵션으로는 안 된다)

현재 `PerceptionEpisode.call` 은 ATS 를 `cloud_target & has_target` 로 고정한다 (`Baseline/paper_baselines/eval/cloud_labeling.py:205`).
최소 변경안은 아래와 같다 (제안만, 적용하지 않았다). 기본값은 지금과 같다.

| 파일:줄 | 변경 |
|---|---|
| `cloud_labeling.py:190-191` | `keep` 에 `target_points`, `target_points_offsets`, `manipulated_state`, `manipulated_centroid` 추가 (없는 run 이면 건너뛴다) |
| `cloud_labeling.py:201-208` `call()` | `"target_points": ragged(P, "target_points", k)`, `"manip_state": str(P["manipulated_state"][k])`, `"manip_centroid": …` 반환 |
| `cloud_labeling.py:283-298` episode loop | `--ats-repr` 에 따라 ATS flag 를 만든다. `cloud_target` 은 현행이다. `target_points` 는 `cKDTree(target_points)` 로 `≤ tol` 인 cloud 점이며 has_target call 에만 적용한다. `exclusion` 은 `target_points` 와 같되, has_target=False 이고 `manip_state != ""` 이면 episode 안에서 직전 has_target call 의 `target_points` 를 쓴다. 이때 `‖mean − manipulated_centroid‖ < 1e-5 m` 를 assert 한다. 선택한 표현은 `ATS_ARM` 행으로 두어 paired 기준과 figure script 를 그대로 쓴다. `--ats-repr` 가 `cloud_target` 이 아니면 현행 flag 를 `ats_v11ab_cloud_target|noun_attention` 행으로 함께 낸다 |
| `cloud_labeling.py:266` settings | `ats_repr`, `ats_tol_m` 기록 |
| `cloud_labeling.py:360` 부근 parser | `--ats-repr {cloud_target,target_points,exclusion}` (default `cloud_target`), `--ats-tol` (default `0.005`) |
| `Baseline/paper_baselines/tests/test_cloud_labeling.py` | 합성 episode 1 개를 둔다. fresh·stand-in·no-target-excluded 각 1 call 의 flag 수와 carry-forward assert 를 검사한다 |

옵션이 생긴 뒤의 명령 (V1 과 같은 환경):

```bash
source Baseline/paper_baselines/setup/paper_baselines_env.sh; CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.eval.cloud_labeling \
  --ats-run 20261012T225820_46a08d6ff644_sl4-perception-dev \
  --grounding-run 20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack \
  --ats-repr exclusion --ats-tol 0.005 --dump-figure-data --dump-per-episode 2 --run-id ats-cloud-excl5mm-sl3dev
# sensitivity: --ats-tol 1e-6 / 0.010, --ats-repr target_points
```

### (c) 를 택할 경우의 최소 변경 (제안만, 지금은 비추천)

| 파일:줄 (HEAD) | 변경 |
|---|---|
| `benchmark/runners/perception_replay.py:546` 부근 `FrameOptions` | `record_ats_geometry: bool = False` |
| `perception_replay.py:614` 뒤 (`manipulated = cs.metrics.get("manipulated")`) | flag 가 켜져 있으면 다음을 ragged 로 기록한다. `ats_excl_points` = `ag3s._gated_manipulated.geometry.points` (ESDF 가 받은 그 geometry, `pipeline.py:1589`), `ats_target_source` = fresh / anchor / occluded / none (`grounding.target.metrics` 의 `manipulated_subset_kept_anchor` · `manipulated_occluded` 와 `point_indices` 길이), `ats_observed_cluster_indices` = `ag3s._target_confirm.last.cluster.point_indices` (anchor·association 의 이번 frame cluster, 없으면 빈 배열) |
| `perception_replay.py:1089` 부근 parser, `:1030` `FrameOptions(...)` | `--record-ats-geometry` (default off) 를 전달 |

flag 가 꺼져 있으면 기록이 bit 동일해야 한다. 켜도 AG3S 동작은 바뀌지 않는다. 실행은 SL4 와 같은 GPU 경로이며 GPU lock 이 필요하다.

## 단위 검증

code 를 바꾸지 않았으므로 새 unit test 는 없다. 분석 script 의 자체 검사 결과는 다음과 같다.

```
identity_check_vs_F_ats_rows: n 221, all_equal true          (recompute_ats_arms.log:11)
carry_forward_centroid_check: n 38, all_ok true
fresh_consistency: 45 / flag==indices 45 / points==cloud 45  (summary_diag.json)
standin_provenance: 134 / byte-equal geometry frame 134 / cloud_target>0 0 / target_indices>0 0
(b) exact·5 mm found 151 · 162 = verifier supp_ats 와 같음
```

## verifier 가 알아야 할 것

- 새로 생긴 플래그·기본값 변경: 없음 (제안만 있다. 위 `cloud_labeling.py` · `perception_replay.py` 표).
- 재생산이 필요한 산출물: 없음. (b+) 는 기존 npz 로 계산된다. perception 재실행은 필요하지 않다.
- 옛 기록과 호환이 깨지는가: 없음.
- 재확인할 숫자: 범주 표(45/62/72/38/4), provenance(134 byte 동일, 38 carry-forward), 28/17 의 범주·이동 bin,
  Q3 표의 (b+) 5 mm 행과 paired 비교. paired 결과는 `recompute/summary_ats_arms.json` `ref_ats_ex_5mm.paired_vs_ats` 에 있다.
  `ats_v11ab|noun_attention` 은 (b+) 5 mm 대비 IoU −0.177 [−0.262, −0.091] 이다.
- 주의 1: `gt_target_shift_since_geom_mm` 은 BDDL target 의 이동이다. wrong-object episode(t4·t5·t9 일부)에서는 ATS 물체의 이동이 아니다.
- 주의 2: frames.jsonl `manipulated.association.outcome` 은 no-target frame 에서 이전 frame 값일 수 있다.
  occluded 의 세부 이유를 나누는 데는 쓰지 않았다.
- 규칙 A 진단 그림은 `outputs/ats_cloud_diag/figures/` 에만 두었다 (`docs/figures/` 는 verifier 소유).

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)
(b+) 5 mm 로 다시 재면 ATS 행의 found 는 177/221, IoU 는 약 0.26 일 것이다. segmenter 와의 IoU 차이는 V1 의 (a) 기준 약 +0.40 에서
약 +0.23 으로 줄지만 부호는 유지될 것이다. 따라서 "ATS recall 이 낮다" 는 V1 의 결론 일부는 표현 효과(fresh-only)였다.
그래도 남는 차이는 occluded stand-in 의 stale geometry 와 wrong-object (SL4c) 에서 온다.
