# AG3S-PRE-CORE-TO LTE — target 없는 거리장에 남는 "사과" 표면: 원인 측정과 설계안 (설계만, 구현 전)

> writer: lead 의 설계 위임 (sub-agent) · 작성 2026-10-26 20:30–02:12 KST (설계 위임 sub-agent; what-if 숫자는 lead가 2026-10-27 02:20 KST에 sub-agent 최종 보고에서 옮김) (`date -u` + 9 h)
> 사용자 판정 2026-10-26 20:06 KST: "SC target 없는 층의 사과 잔여 먼저" — LTE 설계를 SCM closed-loop 보다 먼저 보고한다. **구현 여부는 사용자 판정.**
> 측정은 기록 데이터 위의 read-only offline 측정만 (CPU, 동시 worker ≤ 2, `nice -n 19`). 코드 · 기존 파일은 바꾸지 않았다.
> 원시 · 스크립트: `RUN/verify/AG3S-PRE-CORE-TO/LTE/` (`RUN = /mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581`)

이 문서에서 쓰는 짧은 이름 (처음 한 번 밝혀 둔다):

- **target 없는 층** = 권한 있는 손 (쥐는 손의 손가락 + 손바닥 `ee_<side>`) 만 읽는 미세 (5 mm) 거리장. 사과를 지운 거리장이다.
- **껍질 구** = SC 에서 만든 손바닥 바깥 껍질 구 (`--palm-shell on`).
- **새 위반 행** = SC 의 "기존 성공 쥐기" 측정 (SC impl §3 ⓑ, 이하 "SC 쥐기 측정") 에서 껍질 구에만 생긴 TrajOpt 위반 행 (값 < 0). 20 run · 닫기 1 s 전 창 59 요청 · 1,663 행.
- **사과 꼭지 · 잎** = MuJoCo 모델의 `apple_stem` · `apple_leaf` geom. **충돌이 없는 visual geom** (contype 0) 이라 지금까지의 GT (충돌 geom 표본) 에 없었다. depth camera 는 visual geom 을 그리므로 거리장에는 들어간다.

## 0. 요약 (평이하게)

- **"target 없는 층에 사과 표면이 남는다" 는 절반만 맞다.** 껍질 구의 새 위반 행 1,663 개가 실제로 읽은 표면 (가장 가까운 seed voxel, 정확 계산) 을 GT 로 다시 가르면:
  - **사과 꼭지 · 잎 (사과의 visual 부분) 797 행 (47.9 %)** + 꼭지 근처 (꼭지 9 mm) 31 행 (1.9 %),
  - **이웃 과일 표면 519 행 (31.2 %)** — 오렌지 (ep119) · 배 (ep1668 · ep126). 진짜 장애물이다,
  - **쥐는 손 자신의 표면이 거리장에 들어간 것 316 행 (19.0 %)** — 거의 한 run (ep1668, 손이 배에 닿은 자리),
  - **사과 몸체 (충돌 geom) 표면: 0 행.**
- SC 문서의 "사과 868 행 (25 mm 안)" 귀속은 **사과 몸체가 15–23 mm 떨어진 꼭지 · 잎** 을 몸체로 센 것이다 (8 mm 기준으로는 41 행뿐이었다).
- **사과 몸체 자체는 손 근처에서 거의 다 지워져 있다.** 닫기 1 s 창의 관측된 사과 몸체 표면 표본 31,629 개 중 target 없는 층에 남은 것 3,183 (10.1 %) — 그중 2,729 (85.7 %) 가 **지지면 + 5 mm 아래 띠** (테이블에 구멍을 내지 않으려고 일부러 남기는 띠, T13), 454 가 ambiguous 보존. **손 (권한 구 표면) 20 mm 안에서는 118 / 5,526 = 2.1 %.**
- **꼭지 · 잎이 남는 이유 (숫자):** 꼭지 · 잎 seed 는 target 공 (반지름 53–56 mm) 의 **경계에 붙어 있다** (공 표면까지 −8.9 … +1.0 mm, 중앙 0.0). (i) 공 밖으로 ≤ 1 mm 나간 seed 533 행 — 후보가 아니라 그대로 남음, (ii) 공 안 seed 264 행 — **ambiguous 로 보존**. 꼭지 · 잎 위의 cloud 점 (요청당 4–15 개) 은 target 점이 아니고 (0–2 개만 target), 꼭지 · 잎의 28–37 % 가 F1 의 "target 자기 표면" 반경 (맞춘 반지름 + 10 mm = 43–44 mm) 밖이라 **non-target 근거로 남는다** → 공 안 꼭지 seed 의 `d_t ≈ d_o`.
- **설계안 3 개 (§5)** — 안 1 "꼭지 근거 정리" (공 안 ambiguous 만), 안 2 "꼭지 덮개" (공 위쪽 좁은 원기둥까지 후보 확장), 안 3 "손 자국 막기" (쥐는 손의 거리장 침입). offline what-if (§4.4): 19 key · 56 요청에서 새 위반 1,624 행 → 안 1 상한(cap_in) 1,329 · 안 2 상한(cap_all) 835 · GT 꼭지 · 잎 상한 867, 세 규칙 모두 이웃 과일 · 테이블 · 장애물 seed 0 개
- **어떤 안도 이웃 과일 519 행은 줄이지 않는다 (줄이면 안 된다).** 이 행은 SCM (껍질 행 여유) · TrajOpt 쪽 문제다.

## 1. 질문과 측정 정의 (측정 전 고정, 2026-10-26 20:14 KST)

**질문:** SC 쥐기 측정의 닫기 창에서 사과 표면이 왜 target 없는 층에 남는가 — target 점에 없음 / 공 반경 밖 / non-target 으로 보존 / unowned 보존 / ambiguous / ROI carve 미적용 중 무엇인가.

**대상:** SC 쥐기 측정과 같은 20 run · 59 요청 (`SC/b1/jobs_gt.json`, `SC/b1/b1_rows_gt.json`; 회귀 14 = E3 S-REG 금지 접촉 없는 성공, 가림 6 = E3 case1 L1 · L3). 서버 명령줄 = 기록 그대로 (가림은 `E1/occ_index/<run>/server_proc.txt` — 직접 읽음, `make_links.sh` 미사용): `--target-field-policy exclude-authorized --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --nontarget-evidence surface --kept-geometry-follow --roi-carve --roi-carve-footprint ...`.

| 측정 | 정의 |
|---|---|
| 격자 | 기록 chunk 의 주 미세 층 `esdf_distance` (D) · target 없는 층 `esdf_free_distance` (F), 같은 원점 (첫 voxel 중심), 5 mm. **seed = 값이 정확히 0 인 voxel.** 확인: F 의 seed = D 의 seed − `target_free_removed_ijk` 가 **59 / 59 요청에서 bit 단위로 같다** |
| GT | `task_dynamic_map_eval.scene_truth` (SC 쥐기 측정과 같음) 를 그 요청의 policy step qpos 로. 충돌 geom 표본 5 mm + 로봇 충돌 geom (`RobotSurface`). 덧붙여 (측정 뒤, 아래 §3.2 참고) visual geom: 사과 꼭지 · 잎 (2 mm 표본), 로봇 visual mesh |
| A 사과 몸체 표면 잔여 | 사과 충돌 표면 표본 중 D ≤ 5 mm (관측되어 점유) 인 것 → 그중 F ≤ 5 mm 이면 "잔여". 잔여 표본마다 가장 가까운 F seed → 경로 분류 |
| C 새 위반 행이 읽는 표면 | SC 의 행 값 계산 (`sc_common.rows`, `--palm-shell on`) 으로 껍질 구 위반 행을 다시 만들고 (1,663 행 — SC 와 같은 수), 구 중심에서 **가장 가까운 seed** (그 행이 읽는 층; 1,663 행 모두 권한 행 → F) 를 KD 로 찾는다. 거리장 값 = 가장 가까운 seed 까지 거리이므로 이것이 정확한 귀속이다 (KD 거리 대 격자 값 차 ≤ 3.8 mm, 격자 반올림) |
| seed 경로 | 같은 격자의 trace: `removed` / `nontarget_kept` / `ambiguous` / `unowned_kept`, trace 에 없으면 공 밖 (`|c − 중심| > r`) / `z_min` 아래 / 공 안인데 trace 없음 |
| seed GT 범주 | 8 mm 안의 가장 가까운 GT (5 mm seed 띠 4.5 mm + visual 대 collision ~2–3 mm). 8–20 mm 는 "근처" 로 따로 |
| 현재 관측 | seed 를 각 camera (기록 raw depth) 에 투영: 그 pixel 의 depth 가 seed 깊이 ± 10 mm (지금 거기 표면이 보임) / 더 멀다 (지금 뚫고 보임) / 더 가깝다 · 화면 밖 |

원시: `LTE/lte_reg.json` · `lte_occ.json` (A · B · C, `lte_measure.py`), `lte_seed_detail.json` (C 의 서로 다른 seed 71 개, `lte_seed_detail.py`), `final_rows.json` (`final_rows.py`), `visual_check.json` (`visual_check.py`), `stem_check.json` (`stem_check.py`), 요약 `summary_lte.json` (`aggregate.py`), what-if `whatif_*.json` · `summary_whatif.json` (`whatif_cap.py`).

## 2. 코드 경로 (checkout = 기록을 만든 `snapshot_e3c` 와 같은 파일: `curobo_builder.py` · `pipeline.py` · `config.py` · `curobo_field.py` diff 없음)

| 단계 | 위치 | 하는 일 |
|---|---|---|
| target 공 | `AG3S/ag3s/runtime/pipeline.py:3638-3705` (`_target_ball`) | 중심 = 관측 표면에 맞춘 구 중심, 반지름 = max(target 점 거리 최대, 맞춘 반지름) + pad 1 voxel (`:3684`), `z_min` = 지원면 z + pad (`:3693`) |
| non-target 근거 | `pipeline.py:3280` (`_nontarget_points`, cloud − target 점 − 지원면) → `pipeline.py:3133` 에서 F1 (`nontarget_surface_evidence`, `curobo_builder.py:819`) 이 "맞춘 구 + shell 10 mm 안" 의 target 자기 표면 점만 뺀다 |
| target 없는 미세 층 | `AG3S/ag3s/fields/curobo_builder.py:1758-1812` — 주 미세 TSDF 의 seed 에서 공 후보를 가르고 (`OwnedEvidence` `:1786`) 지운 뒤 다시 전파 (`_build_tiers` `:1799`) |
| 공 후보 | `curobo_builder.py:2839-2858` (`_ball_candidates`): voxel 중심이 공 안 **그리고** `z ≥ z_min` (`:2856`) 인 것만 |
| ownership 분할 | `curobo_builder.py:766-811` (`_ball_owned_split`): unowned = `min(d_t, d_o) > 3 voxel` (`:800`, 지움), ambiguous = `|d_t − d_o| ≤ 1 voxel` (보존), target = `d_t < d_o` (지움), 나머지 non-target (보존). 지우는 것 `:805` |
| trace 기록 | `curobo_builder.py:2871-2960` (`_ball_exclusion`) → 기록 키 `target_free_*_ijk` |
| TrajOpt 가 읽는 값 | `AG3S/ag3s/fields/curobo_field.py:356-433` (`target_free_distance`): 창 안 `min(f, max(d, b))` (`:427`) — 손 근처 (`b` 큼) 는 target 없는 층 값 그대로 |
| ROI carve | `curobo_builder.py:1680-1717` · `_roi_carve :2589` — 관심 영역 안 점유 voxel 을 depth 가 뚫고 본 것만 지운다 (모든 TSDF 공통, target 과 무관) |
| self-filter (로봇을 depth 에서 뺌) | `AG3S/ag3s/stages/robot_filter.py:486` (`depth_robot_mask`) — 구 모델, inflation 기본 0 (`config.py:663`, T26 사용자 판정 "구 반지름 자체로") |

## 3. 측정 결과

### 3.1 사과 몸체 (충돌 geom) 표면 — 손 근처에서는 거의 다 지워져 있다 (A)

| | 값 |
|---|---|
| 관측된 사과 몸체 표면 표본 (D ≤ 5 mm) | 31,629 (59 요청) |
| 그중 target 없는 층에 남은 것 (F ≤ 5 mm) | **3,183 (10.1 %)**, 요청별 0–19.6 % (중앙 9.9 %) |
| 남은 표본의 가장 가까운 seed — `z_min` 아래 (지원면 + 5 mm 띠) | 2,729 (85.7 %): 사과 1,905 · 테이블 824 |
| — ambiguous 보존 | 454 (14.3 %): 사과 407 · 사과 근처 23 · 바나나 23 · 옛 사과 자리 1 |
| — 공 밖 · non-target 보존 · unowned 보존 | 0 · 0 · 0 (`--owned-unowned` 기본 remove → unowned 보존 trace 는 59 요청 모두 0) |
| **권한 구 표면 20 mm 안의 사과 표본** | 5,526 중 **118 (2.1 %)** |
| 공 대 GT 사과 | 공 중심–GT 사과 중심 1.5–4.7 mm (중앙 2.3), **사과 충돌 표면이 공 밖인 비율 0** (59 / 59), 사과 표본 중 `z_min` 아래 12–15 % |
| target 점 | 56 / 59 요청 `visible`. 관측된 사과 표면 중 가장 가까운 target 점이 15 mm (unowned 거리) 넘게 먼 비율 12–25 % (그 seed 는 unowned → 지워진다) |
| 사과 이동 | 처음 요청 대비 중앙 0.2 mm, 최대 6.5 mm — 옛 자리 잔상 아님 |

→ "공 반경 밖 · target 점에 없음 · unowned 보존 · ROI carve 미적용" 은 **사과 몸체** 의 잔여 원인이 아니다. 몸체 잔여는 바닥 띠 (설계상 남김, T13 테이블 구멍 방지) 와 ambiguous 가 전부이고, 쥐는 손 근처에는 거의 없다.

### 3.2 껍질 구의 새 위반 행이 실제로 읽는 표면 (C) — 주 답

1,663 행 · 22 요청 · 12 run (SC 와 같은 수), 서로 다른 최근접 seed 71 개.

| 최근접 seed 의 정체 (GT 8 mm, visual 포함) | 행 | 비율 | 공에 대한 위치 | run |
|---|---:|---:|---|---|
| **사과 꼭지 · 잎 (visual geom)** | **797** | 47.9 % | 공 안 264 (ambiguous 보존) · 공 밖 위쪽 533 (공 표면까지 ≤ +1.0 mm) | ep121 · 250 · 309 · 325 · 1837, case1 L1 s13453 · s33453, L3 s23453 · s33453 |
| 사과 꼭지 근처 (꼭지 · 잎 9 mm, 몸체 7 mm) | 31 | 1.9 % | 공 안 (ambiguous) | ep250 |
| 이웃 과일 표면 (오렌지 · 배) | 475 | 28.6 % | 공 밖 473 · 공 안 2 | ep119 (오렌지 346) · ep1668 (배 129) |
| 이웃 과일 근처 (배 12–14 mm) | 44 | 2.6 % | 공 밖 | ep126 (40) · ep119 (4) |
| **쥐는 손 자신의 표면** (로봇 visual mesh 0.7–5.3 mm) | 316 | 19.0 % | 공 밖 (22–37 mm) | ep1668 (306, 손이 배에 닿은 자리 — 배 5–15 mm) · ep126 (10) |
| 사과 몸체 충돌 표면 (8 mm 안) | **0** | 0 % | — | — |

- 귀속 그림: `RUN/verify/AG3S-PRE-CORE-TO/LTE/lte_attribution_final.png` (그래프 · 표), 원시 `final_rows.json` · `visual_check.json`.
- **처음에는 충돌 GT 로만 갈라서** 꼭지 · 잎 seed 가 "로봇 근처 / 어느 표면에도 없음" 으로 나왔다 (`final_rows.json`: 손 691 · 손 8–20 mm 324 · 사과 8–20 mm 129). seed 가 사과 중심 축에서 수평 4–15 mm, 지원면 위 70–81 mm (사과 몸체 꼭대기 66 mm 위) 에 모여 있는 것을 보고 모델의 사과 geom 을 확인했다: `apple_stem` (z 지원면 + 57…79 mm, 9 × 6 mm) · `apple_leaf` (z + 66…71 mm, 25 × 15 mm), 둘 다 contype 0 · group 2. **이 재분류는 측정 뒤 덧붙인 진단이다** (visual 포함 기준은 결과를 본 뒤 정했다).
- 현재 관측 (depth) 대조: 손 표면 seed 316 행 중 216 행은 그 자리에 **지금 표면이 보인다** (손을 depth 에서 다 빼지 못함 — self-filter 구가 손 visual mesh 를 다 덮지 못함, inflation 0). 꼭지 · 잎 seed 는 손목 camera 가 지금 그 자리를 보는 경우가 많다 (`lte_seed_detail.json` `cams`).
- SC 문서의 25 mm 귀속 "사과 868" 은 이 꼭지 · 잎 seed 가 사과 몸체에서 15–23 mm 라서 생긴 것이다 (충돌 GT 의 사과 몸체가 25 mm 안에서 가장 가까웠다).

### 3.3 왜 꼭지 · 잎이 남는가 (`stem_check.json`, 꼭지 · 잎 행이 있는 13 요청)

| 사실 | 값 |
|---|---|
| 꼭지 · 잎 표면의 공 중심 거리 | 28–52 mm (공 반지름 53.3–55.6 mm) → visual 표면 자체는 **공 안** |
| 그 seed 의 공 표면 거리 | −8.9 … +1.0 mm (중앙 0.0) → TSDF seed 는 표면에서 최대 반 voxel 바깥에 서므로 위쪽 꼭지 seed 가 **공 경계를 1 mm 넘는다** (533 행) |
| 꼭지 · 잎 위의 cloud 점 (6 mm 안, 지원면 제외) | 요청당 4–15 개, 그중 target 점 0–2 개 → **target 점에 꼭지가 거의 없다** |
| F1 "target 자기 표면" 반경 (맞춘 반지름 + 10 mm) | 43.3–44.0 mm. 꼭지 · 잎 표면의 28–37 % 가 그 밖 → 그 점들은 **non-target 근거로 남는다** |
| 결과 | 공 안 꼭지 seed: 가까운 target 점 (사과 꼭대기) 과 가까운 non-target 점 (꼭지 자신) 의 거리가 1 voxel 안 → **ambiguous → 보존** (264 행). 공 밖 꼭지 seed: 후보가 아님 → 보존 (533 행) |

> **정정 (2026-10-27 14:38 KST, lead, `stem_check2.json` · `stem_check2.py`):** 위 표의 "target 점에 꼭지가 거의 없다" 는 **측정 방식이 만든 결론이라 틀렸다.** `stem_check.py` 는 지금 cloud 점이 target 점과 1 mm 안에서 겹치는지로 셌는데, 이 요청들의 target 점은 **기록된 cloud 와 같은 점 집합이 아니라** (같은 frame 관측, 상태 visible, 서버 기록 `points_mismatch`; target 점 → 실제 사과 표면 중앙 1.3 mm, → 기록 cloud 점 중앙 4.7 mm · 90 % 21.6 mm — 회귀 250 요청 9) 지금 cloud 점과 정확히 겹치지 않는다 (14:48 재정정: 처음 적은 '이전 frame 고정 기하' 는 틀렸다 — 13 요청 모두 관측 frame = 요청 frame, `viz_probe2.py`) — 대조로 잰 **사과 몸통** cloud 점도 1 mm 안 일치가 0–12 % 뿐이다. 다시 잰 숫자 (13 요청): **target 점 중 꼭지 · 잎 visual 4 mm 안에 있는 점 10–16 개, 몸통 꼭대기 위 target 점 6–14 개** → ATS (DBSCAN 연결) 는 꼭지를 target 에 **넣고 있다.** 꼭지가 지워지지 않는 원인은 (1) **공 경계** — 지우기는 target 점 자체가 아니라 몸통에 맞춘 공 + voxel 소유 투표로 하고, 꼭지 끝 seed 가 공 밖 ≤ 1 mm (533 행) 라 후보가 아니다, (2) **target 점에서의 5 mm 판정** — 지금 cloud 점은 target 점과 1 fine voxel (5 mm) 안이어야 target 으로 빠진다. 가는 꼭지 위 점 중 1–3 개가 그 밖이고 F1 반경 (43–44 mm) 밖이라 non-target 근거로 남는다 (13 요청 중 9) → 공 안 꼭지 seed 가 ambiguous 로 보존 (264 행). 아래 문장의 "target 점에 없음" 은 "(2) 의 5 mm 판정에서 일부 빠짐" 으로 읽는다.

즉 원인 분류로는 **"target 점에 없음 (꼭지가 target 분할에서 빠짐) + 공 반경 경계 + ambiguous 보존"** 의 결합이다. non-target 보존 · unowned 보존 · ROI carve 는 꼭지 행의 원인이 아니다 (ROI carve 는 지금 보이는 표면을 지우지 않는 것이 맞다 — 꼭지는 실제로 거기 있다).

### 3.4 사과 꼭지 · 잎은 "장애물" 인가

- 시뮬레이터: 충돌 없음 (contype 0). 손이 꼭지를 지나가도 MuJoCo 접촉이 없다 → 금지 접촉 지표에 영향 없음.
- 실물: 꼭지 · 잎은 사과의 일부다 → 권한 손이 닿아도 되는 target 이다.
- 그러므로 **target 없는 층에서 지워야 할 대상**이다. 다만 "사과 위쪽 가는 구조" 를 지우는 규칙은 사과 위에 실제로 있는 다른 물체 (위에서 내려오는 막대, 쌓인 과일, 손 자신의 침입) 와 구별해야 한다 (§5 위험).

## 4. 하지 않을 것 — 이미 기각 · 측정된 선택지 (반복 금지)

| 선택지 | 기록 | 이번 숫자와의 관계 |
|---|---|---|
| 공 반지름 전체 확대 (+5 / +10 / +20 mm) | T45 D-TF (`AG3S_T0T6_LOG.md` §5): 행 575 → 521 / 510 / 446 로 느리게 줄고 이웃 과일 seed 134 / 336 / 815 를 지움 · 17 chunk 에서는 601 → 601 | 꼭지 seed 는 공 위쪽 경계에만 있다 — 전체 확대는 옆의 이웃 과일까지 먹는다 |
| ambiguous 전부 제거 (F2, `--owned-ambiguous remove`) | T46c: 채택 안 함, 대조로만 (이웃 과일 seed 688 손실) | 꼭지 ambiguous 264 행만 노리는 좁은 규칙이 필요 |
| 지원면 띠 일괄 삭제 (ORV_S) | T46e B2 사용자 판정: 기각 (그 띠의 추가 삭제 seed 는 전부 이웃 과일이 최근접) | 사과 바닥 띠 2,729 표본은 손 근처가 아니다 — 건드리지 않는다 |
| self-filter inflation 키우기 | T26 사용자 판정: inflation 0, 새는 곳은 구 모델을 고친다 (50 mm 가 손가락 앞 사과를 지웠다, T14) | 안 3 은 구 모델 보정으로만 설계 |
| 로봇 렌더 mask | 사용자 판정: 보류 (실물 · 실시간) | 안 3 에서 쓰지 않음 |
| 로봇 몸 부피 비우기 (`--robot-occupied-free`, D2) | AG3S-DYNAMIC-FIELD: 기준 미달 | 안 3 에서 전신 대신 손 · target 없는 층에만 거는 변형도 같은 계열 — 우선순위 낮춤 |
| unowned 보존 (`--owned-unowned keep`) | T44: recall 0.548 | 이번 잔여에 unowned 보존 0 — 무관 |

### 4.4 offline what-if (설계 근거, 고정 지표 아님; 규칙 파라미터는 실행 전 고정 2026-10-26 ~21:25 KST)

`whatif_cap.py`: 기록 target 없는 층에서 규칙이 고른 seed 를 빼고, 새 위반 행 값을 "가장 가까운 seed 거리" 로 다시 계산 (새 값 = d_new − (d_old − 기록 값)). 규칙이 뺀 seed 의 GT 범주도 센다 (59 요청 전체).

| 규칙 | 뺀 seed |
|---|---|
| oracle_stem (GT 상한) | 꼭지 · 잎 visual 8 mm 안 |
| cap_in | 공 안 + 맞춘 구 중심 축에서 수평 ≤ 20 mm + 높이 `c_z + 0.5·r_fit … c_z + r_fit + 25 mm` |
| cap_all | 같은 원기둥, 공 제한 없음 |

범위: 계획 20 key · 59 요청 중 19 key · 56 요청 완료 (ep1837 3 요청 · SC 39 행 빠짐; 첫 실행이 오류 없이 중간에 끝나 남은 run만 이어 돌림, 이어 돌린 run은 로봇 visual 표본 간격 2 → 5 mm, 꼭지 · 잎은 2 mm 그대로). 비교 기준 = 덮인 범위의 기록 새 위반 1,624 행 · 21 요청. 원시 `whatif_reg.json` · `whatif_occ.json` · `summary_whatif.json`.

| 규칙 | 남은 행 | 줄어든 행 | 위반 요청 | 함께 지운 seed (56 요청, GT 8 mm) |
|---|---:|---:|---:|---|
| oracle_stem (GT 상한) | 867 | −757 | 21 → 10 | 꼭지 · 잎 404, 사과 몸체 39, 로봇 2 |
| cap_in (안 1 상한) | 1,329 | −295 | 21 → 15 | 꼭지 · 잎 343, 사과 몸체 58, 표면 없음 11 |
| cap_all (안 2 상한) | 835 | −789 | 21 → 9 | 꼭지 · 잎 348, 사과 몸체 58, 표면 없음 11, 로봇 2 |

세 규칙 모두 이웃 과일 · 테이블 · 장애물 seed 0 개. cap_all 뒤 남는 835 행 = 이웃 과일 519 + 손 316.

## 5. 설계안 (구현 전, 사용자 판정 대상)

공통: 모두 **target 없는 미세 층만** 바꾼다 (주 층 · 권한 없는 link 는 한 voxel 도 안 바뀜). 기본 꺼짐 flag, 꺼지면 bit 동일. 기록 키는 켰을 때만.

### 안 1 — 꼭지 근거 정리 (공 안 ambiguous 꼭지만)

- **무엇:** F1 (`nontarget_surface_evidence`) 에 단계 하나: non-target 근거 점 중 (a) 공 안, (b) 맞춘 구의 위 반쪽 (`z > c_z + 0.5·r_fit`) 이고 중심 축에서 수평 `≤ r_cap` (20 mm), (c) target 점에서 시작해 1 voxel (5 mm) 간격 사슬로 이어진 것 (flood fill, 원기둥 안에서만) 을 "target 자기 표면" 으로 보고 근거에서 뺀다. 같은 점을 ownership 의 target 점에 더한다 (그래야 꼭지 끝 seed 가 `d_t < d_o` 가 된다).
- **층:** ATS/LTE 경계 — target extent (F1 의 연장). `pipeline.py:3133` 근처 · `curobo_builder.py:819`.
- **노리는 잔여:** 공 안 꼭지 ambiguous 264 + 31 행 (17.7 %). 공 밖 533 행은 그대로.
- **기대 (what-if cap_in 이 상한):** 새 위반 1,624 → 1,329 행 (−295, −18 %), 위반 요청 21 → 15. 이웃 과일 · 테이블 · 장애물 seed 0 개 지움.
- **위험:** 사과 위에 붙어 있는 다른 물체 (막대가 사과 꼭대기에 걸침, 과일이 쌓임) 의 점이 사슬로 이어지면 target 으로 흡수 → 권한 손에게 그 물체가 사라짐 (진짜 충돌을 숨김). 완화: 원기둥 반지름 20 mm · 높이 제한, 사슬 점 수 상한 (꼭지 · 잎 = 요청당 cloud 점 4–15 개 → 예: ≤ 30 개, 넘으면 적용 안 함), 그 점이 다른 manipulated / 장애물 cluster id 를 가지면 제외.

### 안 2 — 꼭지 덮개 (공 위쪽 원기둥까지 후보 확장) = 안 1 + 후보 확장

- **무엇:** 안 1 에 더해, 공 후보 (`_ball_candidates`, `curobo_builder.py:2839`) 에 **공 위쪽 원기둥** (수평 ≤ 20 mm, 높이 공 꼭대기 + 2 voxel 까지) 을 더한다. 확장 부분의 seed 는 같은 ownership 으로 가르되 **unowned 는 보존** (근거 없는 seed 는 지우지 않음) — 지우는 것은 안 1 로 target 이 된 꼭지 점에 가까운 seed 뿐.
- **층:** LTE (target 없는 층의 후보 영역).
- **노리는 잔여:** 꼭지 · 잎 797 + 31 행 전부 (49.8 %).
- **기대 (what-if cap_all / oracle_stem):** cap_all 1,624 → 835 행 (−789, −49 %), 위반 요청 21 → 9; oracle_stem 867 행 (−757). 이웃 과일 · 테이블 · 장애물 seed 0 개 지움. 남는 835 행은 이웃 과일 519 + 손 316 — LTE로 줄일 몫이 아님.
- **위험:** 안 1 의 위험 + 공 밖으로 후보가 나간다 → 공 밖에 있는 물체 (사과 바로 위를 지나는 막대, 위에서 내려오는 다른 손) 를 지울 수 있다. 확장 부분 unowned 보존 · 원기둥 좁게 · 사슬 조건으로 막는다. 공 위쪽만 넓히므로 옆의 이웃 과일 (T45 의 손실 경로) 은 원기둥에 들지 않는다 — what-if 에서 원기둥이 뺀 seed 의 이웃 과일 · 장애물 수를 반드시 본다.

### 안 3 — 손 자국 막기 (쥐는 손이 거리장에 들어가는 것)

- **무엇:** depth 에서 로봇을 빼는 self-filter 구 모델 (`robot_filter.py:486`) 에 쥐는 손의 **손바닥 껍질 구 (SC, 파지 공간 keep-out 포함) · 손가락 덮개** 를 넣는다 (지금은 constraint 모델에만, `serve_safe.py` palm cover 와 같은 사정 — T45 D-SF). inflation 은 0 그대로 (T26 판정: 새는 곳은 구 모델을 고친다).
- **층:** 관측 (self-filter) — 주 층 · target 없는 층 모두에 영향 (TSDF 에 들어가는 depth 가 바뀜).
- **노리는 잔여:** 손 표면 316 행 (19.0 %, 사실상 ep1668 한 run). 꼭지 · 이웃 과일은 무관.
- **기대:** ep1668 의 손 seed 는 손 visual mesh 0.7–4.2 mm, 그중 216 행은 지금 depth 에 보이는 표면 → 손 구 모델이 덮으면 새로 들어가지 않는다. 이미 들어간 자국은 ROI carve · 다음 관측이 지워야 한다 (offline CPU 로는 TSDF 재적분을 못 해 효과를 잴 수 없다 — GPU replay 필요).
- **위험:** 껍질 overflow 5 mm 안의 실제 물체 pixel (손 옆면에 붙은 이웃 과일 · 막대) 이 TSDF 에 안 들어간다 → 처음 보는 물체면 거리장에서 빠진다 (막대 접촉 자리 = 껍질 −y 모서리, SC §1). 배 (ep1668) 는 손에서 5–15 mm 라 일부가 가려질 수 있다. 1 run 현상이라 효과 대비 위험이 크다 → **안 1 · 2 뒤로 미루기를 권한다.**

### 하지 않는 것 — 이웃 과일 519 행 (31.2 %)

오렌지 (ep119, 손바닥이 오렌지에 1.4–4 mm) · 배 (ep1668 · ep126) 는 실제 장애물이고 target 없는 층이 맞게 남긴 것이다. 이 행을 줄이는 것은 LTE 가 아니라 SCM (껍질 행 여유 축소) 또는 접촉 허용 규칙의 판단이다 (SC §6). 안 1–3 어느 것도 이 행을 건드리지 않도록 시험 기준에 넣는다.

## 6. 시험 방법 (결과 전에 고정할 기준 — 구현 판정 때 확정)

### offline (CPU, 기록 재사용, 서버 없음)

- 입력: 이번과 같은 20 run · 59 요청 (닫기 창) + **같은 run 들의 닫기 전 전 구간** (위험 측정용) + 가림 case3 (막대) 기록 (E3 · E0, SC ⓐ 의 4 접촉 run 포함).
- 방법: 기록 주 미세 층 seed − (기록 removed ∪ 새 규칙이 지울 seed) → EDT + 부호 규칙으로 target 없는 층 재구성 (T45 D-TF 방법, 기록과 > 0.6 mm 차 0 을 먼저 확인) → SC 쥐기 측정 (껍질 on) 과 SC 막대 측정 (ⓐ) 을 다시.
- 기준 (안 1 · 2 공통, 넷 다 만족해야 closed-loop 진행 제안):
  - ⓐ 꼭지 · 잎 귀속 행 (visual GT 8 mm) 이 안 1: 공 안 꼭지 행 −80 % 이상 / 안 2: 전체 꼭지 행 −80 % 이상.
  - ⓑ 새로 지운 seed 중 이웃 과일 · 장애물 · 테이블 (충돌 GT 8 mm) 이 **0** (59 요청 + 닫기 전 전 구간). 1 개라도 있으면 그 자리를 보고하고 멈춘다.
  - ⓒ SC 막대 접촉 4 건의 껍질 계획 여유 (a2) 가 바뀌지 않는다 (±0.5 mm) — 막대를 지우지 않는다는 확인.
  - ⓓ 권한 없는 link 의 행 값 · 주 층은 bit 동일.
- 안 3 offline: GPU replay (`dynamic_field_replay`, 공유 lock) 로 ep1668 · case3 막대 run 의 TSDF 를 다시 적분해 손 자국 seed 수와 막대 · 이웃 과일 seed 손실을 잰다 (CPU 로는 불가).

### closed-loop (offline 통과 뒤, 사용자 판정)

- 구성: 지금 serving (E3) + 채택 안 flag 하나. SCM (`--palm-shell on --palm-shell-margin 5`) 과 **섞지 않는다** — 먼저 E3 + 안 단독으로 회귀 18 · 가림 18, 그 다음 SCM 과 조합.
- 주 지표 (E3 대비, SC/C0F 와 같은 정의): 금지 접촉 없는 성공 가림 ≥ E3 · 회귀 ≥ E3, 금지 접촉 run 수가 늘지 않음, 장면별 성공→실패 수 (1 이상이면 원인 위치).
- 함께 볼 것: 닫기 창 요청의 TrajOpt 수정량 (팔 최대 변화 °) · 껍질을 켰을 때의 새 위반 요청 수 (SC 쥐기 측정 재계산) · 이웃 과일 · 막대 실제 MuJoCo 접촉 (replay) · 꼭지 행 수.
- 10 장면 단위로 빠르게 (규칙), 효과 크기 판정은 하지 않는다.

## 7. 판정을 위한 정리

| | 안 1 꼭지 근거 정리 | 안 2 꼭지 덮개 | 안 3 손 자국 막기 |
|---|---|---|---|
| 노리는 행 (1,663 중) | 295 (17.7 %) | 828 (49.8 %) | 316 (19.0 %, 1 run) |
| what-if 상한 | −295 행 (−18 %), 요청 21 → 15 | −789 행 (−49 %), 요청 21 → 9 | 측정 불가 (GPU 재적분 필요) |
| 진짜 장애물을 지울 위험 | 낮음 (공 안 · 사슬 조건) | 중간 (공 위로 후보 확장) | 중간 (손 옆 overflow 안 물체 미적분) |
| 바꾸는 곳 | target extent (F1 연장) | target 없는 층 후보 | self-filter 구 모델 |

- **사실:** 사과 몸체 잔여는 손 근처에서 2.1 % 이고 껍질 위반 행의 원인이 아니다. 원인은 꼭지 · 잎 (49.8 %), 이웃 과일 (31.2 %, 정당한 장애물), 손 자신 (19.0 %, 1 run).
- **해석 [I]:** 껍질 + 여유 축소 (SCM) 의 쥐기 쪽 위험 중 LTE 로 줄일 수 있는 몫은 꼭지 · 잎 몫 (최대 약 절반) 이고, 이웃 과일 몫은 LTE 로 줄이면 안 된다.
- **사용자에게 물을 것 (권하지 않음):** (1) 안 1 또는 안 2 를 offline 기준 (§6) 으로 구현 · 검증할지, (2) 안 3 을 미룰지, (3) 꼭지 · 잎을 "알려진 한계 (시뮬레이터에서 충돌 없음)" 로 두고 SCM closed-loop 으로 바로 갈지.

## 8. 프로세스 · 산출물

- 산출물 (모두 `RUN/verify/AG3S-PRE-CORE-TO/LTE/`): 스크립트 `lte_measure.py` · `lte_seed_detail.py` · `final_rows.py` · `visual_check.py` · `stem_check.py` · `whatif_cap.py` · `aggregate.py`, 실행 `run_lte.sh` · `run_whatif.sh`, 원시 JSON · log, 그림 `lte_attribution.png` (충돌 GT 만, 첫 집계) · `lte_attribution_final.png` (visual 포함).
- what-if 첫 실행은 두 process 가 중간에 끝났다 (원인 미확인, log 에 오류 없음) → 끝난 run 은 그대로 두고 남은 run 만 이어서 돌렸다 (`whatif_*2.log`; 로봇 visual 표본 간격 2 → 5 mm, 꼭지 · 잎은 2 mm 그대로).
- CPU 만 · 동시 worker ≤ 2 · `nice -n 19`. GPU · 서버 · closed-loop 없음. storage run 안에 바깥을 가리키는 link 를 만들지 않았다. `E1/occ_index/make_links.sh` 미사용, NFS 가 멈춘 옛 폴더 미접근. 내 process 는 모두 끝냈다.
