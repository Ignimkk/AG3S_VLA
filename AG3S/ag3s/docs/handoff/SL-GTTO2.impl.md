# SL-GTTO2 — 구현: GT 인식이 TO까지 전달되지 않는 두 틈 (+ 팔 link 충돌 진단)

> writer: ag3s-implementer (A1) · 2026-11-05 KST · 읽는 쪽: lead, verifier, scribe
> task: [SL-GTTO2.task.md](SL-GTTO2.task.md) (선행 [SL-GTTO.task.md](SL-GTTO.task.md) · [SL-GTTO.impl.md](SL-GTTO.impl.md)).
> 코드 기준: worktree `/mnt/dev/AG3S_VLA/.claude/worktrees/agent-a5af2649e9e869962` (base `89b42459` + 미커밋 SL-GTTO patch = main checkout 의 현재 상태). 커밋하지 않았다.
> **GPU 는 쓰지 않았다.** 모든 진단 · test 는 `CUDA_VISIBLE_DEVICES=''`, 계산 process 2 개 이하. import 확인: `AG3S.__file__` = worktree.
> patch: `/tmp/claude-0/-mnt-dev/576ccf52-3b29-45d1-8286-2a9249abaa31/scratchpad/gtto2.patch` (main checkout 위에 `git apply`, 아래 "patch" 절).

## 무엇을 했나 (평이한 요약 먼저)

- **진단 방법 — 기록 frame offline replay (CPU).** 본 실행의 `run_manifest.json` `server_argv` 그대로 Panda 서버를 CPU 에 만들고 (stub policy), 기록된 `_geom.npz` 의 depth · segmentation · robot_state 로 요청을 다시 만들어 AG3S 를 돌렸다.
  - GT code 영상은 episode 기록의 code 표로 `segmentation_geom` 에서 다시 만들었다 (`oracle_codes.body_code_table`, 기록 표와 일치 확인).
  - 쥐기 상태는 기록 그대로 (`grasp.state`, 기록된 attach call 에서 `AG3S.attach`).
  - ESDF 는 만들 수 없다 (cuRobo TSDF/ESDF 는 CUDA 필요). 그래서 ESDF stage 를 instance 에서 바꿔 **TSDF 가 적분할 depth + mask** 를 받아 기록했다.
  - 재현성: grounding status · fused code-1 점 수가 기록과 같다 — ① 6 episode 312/312 call, ② LIBERO t0e0 43/44 call (seq 15 만 187 vs 188).
- **① (F1) 원인 = 지지면 규칙 (T43 T 3(c), base argv `--supported-max-bottom-mm 60`).** 개도 · destination 이 아니다. GT 입력 버그도 아니다.
  - 그릇이 서랍 안 · 캐비닛 위에 있다. 맞춘 지지면은 둘 다 탁자 높이이고, 그릇 바닥은 그 위 143–234 mm 다 → 모든 call 에서 `not_supported`.
  - → `--oracle-admissible`: GT cluster 는 admissibility · exclusion gate 에서 거절하지 않고 기록만 한다.
- **② (F2) 원인 = 손에 든 GT target 자신의 pixel 이 TSDF 에 적분된 것.** self-filter 가 지우지 못하고 남긴 pixel 이다.
  - 그 표면 뒤 **60 mm truncation band (coarse tier)** 가 attached 구 · 손 행의 "표면" 이 된다.
  - 이후 시선은 손 · 쥔 물체 mask 로 가려져 (masked = 관측 안 함) 그 band 가 지워지지 않는다.
  - mixed 점도 oracle memory 도 원인이 아니다.
  - 쥔 뒤 `no_target_free_layer` 는 설계상 그렇다 — 쥔 물체는 attached (로봇 쪽) 이고 target-free 층은 아무것도 안 쥐었을 때만 만든다.
  - → `--oracle-held-target`: 파지 중 (closing) · 쥔 동안 이 요청의 code 1 pixel 을 TSDF depth mask 에 더한다.
- **③ 진단만.** LIBERO t3e0 link7 행의 표면 = closing 중 (seq 21–22) 손에 든 GT 그릇이 적분된 것 + 그릇 가장자리 static pixel 6–20 개. 이후 로봇이 시선을 가려 지워지지 않았다.
  - 같은 장면 VLA 단독의 link7 은 그 점에서 32–49 mm 떨어져 지나갔다.
  - F2 를 켜면 code 1 근거는 사라진다 (부수 효과). 별도로 고치지 않았다.
- 새 판 **`AG3S-commonto-gt2`** = `AG3S-commonto-gt` + 두 flag. 모두 기본 꺼짐이다. 꺼짐이면 다른 모든 판의 argv · 기록 flag key · identity key · GT 기록이 그대로다 (test).

## ① 진단 — GT target 이 admissibility 에서 거절 (F1)

**어느 조건이 거절했나.** `TargetConfirm._assess_frame` 의 per-cluster 결과 (`last_admissibility`, `last_support`) 를 replay 로 읽었다.

| episode | call | 기록 = replay | 거절 이유 | narrowest / extents (mm) | 개도 78 mm 이하 | destination | 그릇 바닥 높이 (지지면 위, mm) | 지지면 |
|---|---:|---|---|---|---|---|---|---|
| LIBERO t4e0 (서랍 안 그릇) | 44 | 44/44 | `not_supported` 44 | 54.5 / 54.5 × 107.3 × 109.3 | 예 | 없음 | 143.3–152.8 | 2 개, offset −0.011 · 0.014–0.025 m (둘 다 탁자) |
| LIBERO t9e0 (캐비닛 위) | 44 | 44/44 | `not_supported` 44 | 55.0 / 55.0 × 100.7 × 108.8 | 예 | 없음 | 191.9–207.6 | 2 개, 탁자 |
| SafeLIBERO t3e0 (캐비닛 위) | 56 | 56/56 | `not_supported` 56 | 61.2 / 61.2 × 79.5 × 107.7 | 예 | 없음 | 164.3–226.1 | 1–2 개, 탁자 |
| SafeLIBERO t3e2 | 56 | 56/56 | `not_supported` 56 | 49.8 | 예 | 없음 | 176.7–234.3 | 탁자 |
| SafeLIBERO t3e3 | 56 | 56/56 | `not_supported` 56 | 35.9 | 예 | 없음 | 191.6–231.7 | 탁자 |
| SafeLIBERO t3e4 | 56 | 56/56 | `not_supported` 56 | 54.8 | 예 | 없음 | 194.7–220.8 | 탁자 |
| (대조) LIBERO t3e0 (cookie box 위) | 2 | 2/2 | admissible | 72.6 · 69.5 | 예 | 없음 | −0.1 · 0.0 | 탁자 |

**결론.**
- 거절은 T43 T 3(c) 규칙 (`clustering.supported_max_bottom` = 60 mm, `stages/target_grounding.py` `_support_test`) 하나다.
  - 이 규칙은 "cluster 의 가장 낮은 점이 지지면 위 60 mm 이하이거나 손에 있어야 한다" 이다.
  - RB-Y1 에서 hurdle bar 윗면 · crate 테두리를 manipulated 로 잡는 것을 막으려고 만든 인식 추론 규칙이다.
  - 캐비닛 윗면 · 서랍 바닥은 지지면으로 맞춰지지 않는다 (`support_surface.max_planes 2`, 두 면 다 탁자 높이). 그래서 그 위 물체는 늘 "공중" 이다.
- **GT 입력 버그가 아니다.**
  - cluster 는 code 1 점뿐이다. extent 가 그릇 크기 (지름 ~108 mm, 높이 36–61 mm narrowest) 와 맞다.
  - destination 은 등록되지 않았다. 개도 (78 mm) 이하다.
- 기록의 note "graspable at max opening 78.0 mm and off the destination" 은 지지면 규칙을 말하지 않는다 (`pipeline.py` 의 고정 문구). 원인을 숨긴 문구지만 바꾸지 않았다 — 기록 문자열 동일성 때문이다. lead 판정 4.
- **F1 범위.**
  - 실제로 거절한 조건은 지지면 하나다. 그래도 task §1-1 대로 admissibility 전체 (개도 · destination · 지지면) 와 `_exclusion_gate` 를 같은 flag 로 넘겼다.
  - 이유: GT 판에서는 cluster 가 정의상 GT target 이고, 다른 두 규칙도 같은 "잘못된 물체를 잡지 않게" 하는 추론이다.
  - 그리고 GT cluster 를 destination 으로 등록하지 않는다. GT 판에서 cluster 는 모두 target 이라, 큰 target 이 destination 으로 등록되면 버그다.
  - 지지면만 넘기는 좁은 판이 필요하면 lead 판정 1.
- **F1 replay (같은 6 episode, `AG3S-commonto-gt2` 의 두 flag).**
  - 312/312 call 에서 grounding `ok` (전에는 `no_admissible`).
  - `admissibility_would_refuse` = `not_supported` 312 개, `exclusion_gate_would_refuse` 0 개, destination 등록 0 개.
  - 쥐기 이후는 이 replay 에 없다 — 본 실행이 처음부터 HOLD 라 기록된 쥐기가 없다.

## ② 진단 — 쥔 물체의 잔여 점 (F2)

**재는 법.** replay 가 ESDF stage 에 넘기는 `CameraDepth` (depth + TSDF mask) 를 받아 아래 둘을 셌다.
- "TSDF 입력 code-1 pixel" = mask 안 된 code 1 pixel.
- 기록된 충돌 행의 `point_m` (그 행의 구 중심, 계획 step 4) 에 대해, 지난 10 frame 의 각 카메라 투영 5 × 5 pixel 창에서 mask 안 된 pixel 의 관측 표면이 그 점 앞 0–60 mm (= TSDF 의 음수 band 안) 또는 뒤 20 mm 안에 있는 개수 ("band 근거") 를 code 별로 셌다.
- 덧붙여 모든 이전 frame 에서 그 점 30 mm 안의 TSDF 입력 점을 code 별로 셌다.

**LIBERO t0e0 (seq 18 attach).**

| 구간 | code 1 pixel / frame | TSDF 입력 code 1 pixel (base) | fused code 1 점 (replay = 기록) |
|---|---:|---:|---:|
| closing seq 8–17 | 2,601–20,900 | 2,586–20,557 | 105–210 |
| held seq 18–43 | 10,747–11,020 | 38–163 | 7–24 |

- 쥔 동안 code 1 pixel 은 그대로 ~10,800 이다. self-filter 가 그중 대부분을 mask 해서 fused 점이 9–24 로 줄었다. 이것이 task 의 "~170 → 0–20" 이다.
  - grounding 의 `no_cluster` 는 이 잔여 점이 `clustering.min_points` 20 에 못 미쳐서다.
- 잔여가 생기는 이유:
  - 쥔 물체의 self-filter 구 (`filter_spheres_parent_frame`, T32 fit) 와 optimizer 가 묻는 attached 구 (`spheres_parent_frame`) 가 53–77 mm 떨어져 있다. attached 구는 T34 J1 지지면 lift 로 55 mm 올라가 있다.
  - frame 마다 code 1 pixel 3,300–3,600 개가 filter 구 밖, 그중 1,900–3,400 개가 attached 구 안이다 (backproject 기준). 그 일부가 robot 구 mask 도 피해 TSDF 에 들어간다.
- **seq 29 (`attached:panda_hand[1] ↔ esdf −65.6 mm`, coarse, 점 base [0.745, 0.151, 0.103]).**
  - 그 점 30 mm 안에는 **어느 frame 의 어떤 관측 점도 없다.** 그러니 그 표면은 점 자체가 아니다.
  - head 카메라 시선: frame 21–27 · 29 는 mask 1.0 (손 · 쥔 그릇이 가림 → 적분 안 함 → 지우지도 않음).
  - frame 28 은 mask 0.72 이고, 그 시선의 **mask 안 된 pixel 이 code 1 (쥔 그릇)** 이며 점보다 57–60 mm 앞에 있다.
  - 그래서 점은 그 그릇 잔여 표면 뒤 **coarse TSDF truncation 60 mm** (`CuroboFieldBuilder.coarse_truncation`) 의 음수 band 끝에 있다. ESDF ≈ 0 이 그 band 의 끝면이다 (clearance −65.6 ≈ −구 반지름 65.7).
  - 이후 시선이 계속 mask 라 지워지지 않는다. **frame 28 에 관측된 쥔 그릇 잔여 pixel 의 TSDF 기억** 이다.
- seq 32 (`attached[1]` −24.4, fine): frame 28 · 29 의 code 1 잔여 pixel 이 점 앞 55–64 mm · 15–19 mm. 같은 기전이다.
- seq 30 (`attached[0]` −10.9): frame 28 code 1 이 17 mm 뒤, frame 29 wrist 로봇 pixel 이 39 mm 앞. 쥔 그릇 + 로봇 잔여 pixel.
- **mixed 점이 아니다.** `n_mixed` 는 fused voxel 의 통계다. band 근거 pixel 은 순수 code 1 이다.
- **oracle memory 가 아니다.** memory 는 미세 target-free 층만 읽는다. 쥔 뒤에는 그 층이 없다 (`_build_esdf` 의 `nothing_held` 조건 → `no_target_free_layer`). 장애물장은 TSDF 다.

**SafeLIBERO t1e1 (seq 12 attach, 바로 HOLD).**
- seq 12 (`attached[0]` −18.2) · seq 13 (−20.3): 그 점 근처 근거는 **closing frame 9–11** 의 손에 든 그릇이다.
  - seq 12 기준: frame 9 code 1 536 개 · 10.6 mm, frame 11 wrist code 1 이 점 앞 31–37 mm.
  - attach 전이라 held filter 가 없고 그릇 전체가 적분됐다.
  - `_arm_first_free` 는 capture 자세의 구 부피만 비운다. band 와 그 사이 자세의 흔적은 남는다.
- seq 14–29 의 `panda_hand` · `panda_rightfinger` `nearest_is_not_manipulated` 행: 남은 근거 대부분은 **ramekin** pixel (23–28 mm, 그릇이 ramekin 위에 놓인 장면 — D1 이웃 허용을 끈 공용 TO 에서는 실제 이웃 접촉) 과 wrist 카메라의 **로봇 잔여 pixel** (seq 16: 126, seq 21: 226) 이다.
- replay 근사: attach 뒤 fused code 1 점 수가 기록과 몇 개 다르다 (seq 12: 12 vs 7). attach snapshot 이 replay 의 TargetConfirm 을 따르기 때문이다.

**F2 구현 대상.**
- 쥔 물체 자신의 관측 (code 1 pixel) 을 **closing 부터** TSDF 에 넣지 않는다 (self-filter mask 에 더함: 적분 안 함, T32 H1 과 같은 의미).
- closing 부터 하는 이유: t1e1 근거가 closing frame 이다. closing 중 target 은 손가락 사이에 있다.
- fused cloud · grounding 은 바꾸지 않았다 — TargetConfirm · reach · attach 판단이 그대로다.
- cluster 최소 점 수를 낮추는 방법은 쓰지 않았다. 쥔 뒤 잔여 점은 다시 grounding 할 target 이 아니라 self-filter 의 찌꺼기다.

**F2 replay (`--oracle-admissible --oracle-held-target`).**

| episode | closing · held call | TSDF 입력 code 1 pixel base → F2 | `attached_point` 행의 code 1 band 근거 base → F2 | F2 뒤 남은 근거 |
|---|---:|---|---|---|
| LIBERO t0e0 | 36 | 38–20,557 → **0** (모든 call) | seq 29: 14 → 0, seq 30: 1 → 0, seq 32: 25 → 0 | seq 30 로봇 15 px · seq 32 plate 1 · 로봇 1, seq 36/38 탁자 75 (계획이 쥔 구를 탁자 아래 z −0.004/−0.010 로 내림 — 실제 탁자) |
| SafeLIBERO t1e1 (0–29) | 24 | 157–11,158 → **0** | seq 12: 1 → 0, seq 13: 26 → 0 | seq 13 로봇 25 px, finger 행의 ramekin · 로봇 pixel |
| LIBERO t3e0 | 35 | 9,453–10,150 → **0** | (attach 없음) | — |

- 이 표는 **TSDF 입력** 의 변화다. ESDF · clearance · HOLD 의 변화는 GPU 가 있어야 잰다 (task §2 smoke).
- 근거가 창에 안 잡힌 행도 있다 (t1e1 seq 14 · 15 · 17 · 20 · 22–29, t0e0 link 행 일부). 그 행의 표면은 이 검사로 설명되지 않았다.

## ③ 진단 — 팔 link 충돌 (진단만, 고치지 않음)

**LIBERO t3e0** (cookie box 위 그릇, seq 9 부터 closing 이 끝나지 않는다, seq 24 부터 HOLD). link7 `unauthorized_link` 행은 seq 29 · 31–43 이다 (예: seq 43 −45.8 mm, 점 [0.794, 0.198, 0.273]).

| 행 | 30 mm 안의 TSDF 입력 점 (모든 이전 frame) | 그 뒤 head 시선 | F2 를 켜면 |
|---|---|---|---|
| seq 32 link7 −41.0 | frame 22: **code 1 (GT 그릇) 237 px, 4.6 mm** · static 20 px 4.1 mm | frame 23–32 mask 1.0 (로봇이 앞 26–79 mm) | code 1 0, static 20 |
| seq 43 link7 −45.8 | frame 22: code 1 234 px, 8.5 mm · static 19 px | frame 34–43 mask 1.0 (로봇 27 mm 앞) | code 1 0, static 19 |
| seq 29 · 31 · 33 · 35–41 link7 | frame 21–22 code 1 33–244 px, 7.4–29.6 mm · static 2–18 | 로봇 mask | code 1 0, static 남음 |
| seq 34 · 42 link7 | 근거 없음 | — | — |

- **원인 층 판정 (측정 기준).**
  - link7 앞의 "장애물" 은 code 2 정적 fixture 도, code 3+ 물체도 아니다. **closing 중 (seq 21–22) 손에 들려 움직이던 GT target 의 표면** 이다.
  - link7 은 권한 없는 link 라 target-free 층을 읽지 않는다. 그 뒤 로봇 (손목) 이 head 시선을 가려 (mask) 지워지지 않았다.
  - static 6–20 px 는 그릇 가장자리에 붙어 있다 (같은 frame, 같은 4–9 mm). segmentation 이 env.step 반환 시점, depth 가 image 시점이라 (SL-GTTO.impl §3) 움직이는 그릇 가장자리가 배경 code 로 찍힌 것으로 보인다 — **확인하지 않았다** (추정).
- **VLA 단독 (같은 기계 `20261104T233247_…_gtto-vla-libero9-vla_only`, t3e0 성공, 97 step).**
  - 측정된 robot_state 의 link7 구 표면이 위 점들에서 **32.4–48.5 mm** 떨어져 지나간다 (frame 84–88).
  - 즉 VLA 도 그 근처 (3–5 cm) 를 지나지만 그 점을 통과하지는 않는다.
  - GT 판의 link7 은 HOLD 중 그 자리에 머물러 실제로 −9.0 ~ +18.2 mm 까지 갔다 (계획 step 4 점 = 실제 위치).
- **LIBERO t0e0 held seq 33–43** (link6/7): frame 28 의 code 1 잔여 2–22 px 가 18–27 mm. F2 를 켜면 0 이고 로봇 1 px · static 2–5 px 이 남는다.
- **수정하지 않았다.** F2 가 code 1 부분을 부수적으로 없앤다. 남는 static 가장자리 · 로봇 잔여 pixel 은 GT code 0 / segmentation 시점 문제라 F2 범위 밖이다. lead 판정 3.

## 바뀐 파일

줄은 worktree 기준이다. 모두 `SL-GTTO2` 표시가 있다. 비교 기준 = main checkout 의 현재 파일 (= SL-GTTO patch 적용 상태).

### development 소유 core — `pipeline.py` · `target_grounding.py` (이식 요청용, 따로)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/runtime/pipeline.py:852-897` | 새 method `_oracle_held_target_mask(cameras)` — closing · held 이면 각 카메라 TSDF mask \|= (이 요청 code 영상 == 1), 바뀐 카메라는 `device_depth=None` (GPU fast path 가 옛 mask 로 적분하지 못하게), 기록 `held_target_mask {active, phase, n_px, n_px_new, why}` | F2 |
| `…/pipeline.py:2746-2752` | grounding 기록에 `oracle_admissible` · `admissibility_would_refuse` (TargetConfirm 의 이번 call 거절 목록, select 를 안 탄 call 은 None) · `exclusion_gate_would_refuse` | F1 기록 |
| `…/pipeline.py:2908-2911` | ESDF stage: `_depth_cameras_from` (+ render mask) 뒤, `_oracle_update` 가 segmentation 을 소비하기 전에 F2 mask | F2 |
| `…/pipeline.py:3208-3215` | `_exclusion_gate`: `oracle_admissible` 이면 거절 대신 `exclusion_gate_would_refuse` 기록 후 기하 반환 | F1 |
| `AG3S/ag3s/stages/target_grounding.py:104` | `ADMISSIBLE` import | F1 |
| `…/target_grounding.py:801-806` | `TargetConfirm.oracle_admissible` (config 에서), `last_would_refuse` | F1 |
| `…/target_grounding.py:1767-1768` · `:1781-1784` · `:1792-1794` | `_assess_frame`: (on) call 마다 목록 초기화, destination registry 의 `observe` 생략 (주입 destination 은 그대로), 지지면 검사까지 계산한 뒤 거절된 것을 기록하고 `reason=admissible` 로 바꿈 | F1 |

### development 소유 — 그 밖 (`config.py` · `serve_safe.py`)

| 파일:줄 | 무엇이 |
|---|---|
| `AG3S/ag3s/config.py:432-452` | `ClusteringConfig.oracle_admissible: bool = False`, `oracle_held_target: bool = False` (+ 근거 설명) |
| `…/config.py:663-669` | validate: bool 이어야 하고, True 면 `grounding_source == "oracle"` 필요 |
| `AG3S/trajopt/serve_safe.py:1104-1106` | `t43t_identity_section`: 켜졌을 때만 clustering key (꺼짐: key 없음) |
| `…/serve_safe.py:2527-2539` | `--oracle-admissible`, `--oracle-held-target` (store_true) |
| `…/serve_safe.py:3980-3984` | 거절: `--grounding-source oracle` 없이 둘 중 하나 |

### benchmark 소유

| 파일:줄 | 무엇이 |
|---|---|
| `AG3S/adapters/panda/profile.py:503-510` | `GT2_FLAGS`, `GT2_VERSIONS = ("AG3S-commonto-gt2",)`, `AG3S_VERSIONS["AG3S-commonto-gt2"] = COMMONTO_GT_FLAGS + GT2_FLAGS` (파일 끝 SL-GTTO 블록 바로 뒤, 새 블록) |
| `…/profile.py:528` · `:542` | `is_final_version` · `version_server_flags` 에 GT2 (GT 판과 같은 base argv · probe / attention cell 거절) |
| `…/profile.py:604-607` | `version_flag_values`: GT2 dest 는 켜졌을 때만 기록 key 에 (그래서 `AG3S-commonto-gt` 의 기록 key 가 그대로) |
| `AG3S/adapters/panda/server.py:461-463` | `PORTED_FINAL_CONFIG_DESTS` 에 `oracle_admissible`, `oracle_held_target` |
| `…/server.py:372-375` | per-call `gt_perception.grounding` 에 F1 / F2 기록 (key 가 있을 때만) |
| `…/server.py:861-862` | config dump `final_config.gt_perception` 에 켜진 flag 만 True |
| launcher · client | 변경 없음. `version_requires_oracle_segmentation` 이 GT2 에도 참이라 `--oracle-seg --capture-segmentation` 이 자동으로 붙는다 (test) |

### test

| 파일 | 무엇이 |
|---|---|
| `tests/ag3s/test_sl_gtto2_oracle_fixes.py` (새, 10) | config 기본 · 검증 · round trip, CLI (기본 key 없음, flag → clustering key, GT grounding 없이 거절), F1 TargetConfirm (`not_graspable` · `not_supported` 둘 다: 꺼짐 거절 / 켜짐 채택 + 기록, 꺼짐이면 큰 GT cluster 가 destination 으로 등록 · 켜짐이면 안 됨), F1 pipeline (no_admissible → ok, gate 기록, invariant_violation 없음, 꺼짐 새 key 없음), F2 mask 단위 (비활성 = 같은 객체, closing / held = code 1 전부, device_depth 제거), F2 pipeline (esdf backend, ESDF build 를 instance 에서 바꿔 카메라 mask 확인; grasp 중 = mask \| code 1, 아니면 동일; fused cloud · grounding 동일) |
| `tests/benchmark/test_sl_gtto2_versions.py` (새, 8) | GT2 argv = SL-GTTO argv (token 고정) + 두 flag, read-back, 다른 모든 판에 새 flag · 기록 key · identity key 없음, launcher client flag, 서버 ported · CPU build · dump, `AG3S-commonto-gt` 기록에 GT2 key 없음, **기록 frame replay**: F1 (LIBERO t4e0 · SafeLIBERO t3e0 첫 call: base `no_admissible`/`not_supported` → GT2 `ok`), F2 (SafeLIBERO t1e1 call 0–13: closing · held 마다 base TSDF 입력 code 1 > 0 → GT2 0, 그 밖 call 은 같음) |
| `tests/benchmark/libero_gtto2_replay.py` (새) | 위 replay helper (storage 기록 없으면 test 는 skip) |
| `tests/benchmark/test_sl_gtto_gt_perception.py:111-112` · `:144` | "새 판 목록" 비교 두 곳에 `GT2_VERSIONS` 를 더함 (SL-GTTO 가 sl6h 에 한 것과 같은 방식; `GT_VERSIONS == (gt, gt-pass)` 단언은 그대로 통과) |
| `tests/benchmark/test_sl6h_panda_placement_contact.py:61-62` | 같은 목록 비교에 `GT2_VERSIONS` |
| `tests/trajopt/_sl5a_golden.py:414-415` | `PB_DEFAULT_OFF` 에 `oracle_admissible: False`, `oracle_held_target: False` |

## 단위 검증

```bash
# wt.sh = cd <worktree>; PYTHONPATH=<worktree>; CUDA_VISIBLE_DEVICES=''; JAX_PLATFORMS=cpu; MUJOCO_GL=osmesa; AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage;
#         PYTHONDONTWRITEBYTECODE=1; BLAS threads 2; nice 19.  run_gtto_tests.sh = wt.sh + 주 checkout venv · runtime config (SL-GTTO 와 같음)
# worktree 의 빈 submodule 에 test 동안만 자식 symlink (third_party/curobo/curobo, benchmark/LIBERO/libero, SafeLIBERO/{safelibero,main},
# LIBERO-Safety/libero → 주 checkout), 끝나고 지움.
run_gtto_tests.sh <group>      # 아래 group 을 하나씩 순차
```

| group | 파일 | 결과 |
|---|---|---|
| 1 (새) | `tests/ag3s/test_sl_gtto2_oracle_fixes.py` · `tests/benchmark/test_sl_gtto2_versions.py` (기록 frame replay 3 개 포함, skip 0) | **18 passed** |
| 2 (SL-GTTO 그대로) | `tests/ag3s/test_sl_gtto_oracle_grounding.py` · `tests/benchmark/test_sl_gtto_gt_perception.py` (실제 장면 5 개 포함) | **46 passed** |
| 3 | `test_admissibility` · `test_target_confirm` · `test_target_grounding` | 97 passed |
| 4 | `test_t43t_identity` · `test_t43t_held_free_observed` · `test_t32_held_object` · `test_t81_vacancy` · `test_sl8b_identity_rules` | 99 passed |
| 5 | `test_t46e_oracle_extent` · `test_sl8b_identity_versions` · `test_sl7_panda_p0m` | 70 passed, 12 skipped (torch 없음 · CL3/C1a 기록 · HSR6 probe · stored rollout 없음) |
| 6 | `test_sl6a_panda_final` · `test_sl6h_panda_placement_contact` · `test_launcher_record_constraints` · `test_panda_sl4b` · `test_panda_sl5a` · `test_sl7i_surface_record` · `test_sl7f_seed_gate` | 78 passed, 12 skipped, **4 failed — 환경**: worktree 에 `.venv-libero` · `.venv-ag3s` 가 없어 subprocess `FileNotFoundError` (3), `run_parallel_gpu.sh` 가 같은 이유로 job 줄을 거절 (1). SL-GTTO.impl "base 와 같은 실패" 와 같은 부류이고, 바뀐 코드를 지나지 않는다 |
| golden | `python -m tests.trajopt._sl5a_golden --check-serve-safe tests/trajopt/data/sl5a_serve_safe_golden.json` | 24 차이 = SL-GTTO 때와 같음 (새 dest 2 개는 `PB_DEFAULT_OFF` 로 흡수) |

- 전체 suite 는 돌리지 않았다 (지시: pod OOM 이력 → 바뀐 module 을 import 하는 targeted 파일만, 한 번에 process 하나).

- **offline replay (진단, scratchpad 스크립트):** `gtto2_replay.py` (① 6 episode × 전 call, base · GT2), `f2_replay.py` (LIBERO t0e0 · t3e0 전 call, SafeLIBERO t1e1 0–29, base · GT2), `f2_spheres.py` · `f2_backproject.py` (held 구 대 code 1 점), `c3_vla_path.py` (VLA 단독 link 경로). 결과 json: scratchpad `f1/` · `f1on/` · `f2/`.

## verifier 가 알아야 할 것

- **새 서버 flag (정확히):** `--oracle-admissible` · `--oracle-held-target` (둘 다 store_true, 기본 꺼짐, `--grounding-source oracle` 없이 주면 서버가 거절).
  - 새 판 `AG3S-commonto-gt2` = `AG3S-commonto-gt` 의 argv + `--oracle-admissible --oracle-held-target` (그 순서, 끝에).
  - launcher 는 GT2 client 에도 `--oracle-seg --capture-segmentation` 을 자동으로 붙인다.
- **기본값 변경:** 없음.
- **옛 기록과 호환.**
  - 꺼짐이면 다른 모든 판의 argv · `version_flag_values` 기록 key · identity key · `final_config.gt_perception` · per-call `gt_perception` 이 그대로다 (test).
  - 예외 하나: `AG3SConfig.to_dict()` (manifest 의 config dump) 에 `clustering.oracle_admissible: false`, `clustering.oracle_held_target: false` 두 key 가 생긴다. SL-GTTO 의 `grounding_source` 와 같은 방식이다.
- **GT2 의 새 기록 key** (client `*_calls.jsonl` `gt_perception.grounding`):
  - `admissibility_would_refuse` — list of `Admissibility.record()` + `rank`. 빈 list = 거절 없음, None = 이번 call 은 select 를 안 탐.
  - `exclusion_gate_would_refuse` — `{detail, manipulated_id, destination_overlap_now}` 또는 None.
  - `held_target_mask` — `{active, phase: closing|held|null, n_px, n_px_new, why}`.
- **smoke 확인 키 (task §2):**
  - t4e0 · t3e0: 첫 call `gt_perception.grounding.status == "ok"`, `admissibility_would_refuse[0].reason == "not_supported"`.
  - t0e0 · t1e1: attach 뒤 `held_target_mask.phase == "held"` · `n_px_new > 0` 이고 `verdict_reasons` 에 `attached_point` 행이 없을 것.
- **GPU 경로 주의:** F2 로 mask 가 바뀐 카메라는 `device_depth=None` 이라 host depth 를 GPU 로 다시 올린다 (T38 B4 fast path 를 그 카메라만 안 씀). 값은 같고 call 당 수 ms 가 늘 수 있다 (재지 않았다).
- **재생산이 필요한 산출물:** 없음.
- **자원 회수:** 진단 · test process 는 모두 끝났다. 읽은 기록 (`_geom.npz` 17 경로) 의 page cache: `reclaim: ws 57.7->57.6 GiB | cache ?->? GiB, 32500 files, 17 paths` (pod 전체 working set 기준, 내 몫은 작다). test 용 submodule 자식 symlink 5 개는 지웠다.
- **사고 보고:** 진단 중 `ln -s` 두 번째 호출이 이미 있던 symlink 를 따라가 주 checkout 의 `benchmark/SafeLIBERO/safelibero/safelibero` · `…/main/main` 에 symlink 2 개를 만들었다. 몇 초 안에 지웠고, 그 submodule 의 `git status` 는 깨끗하다. 주 checkout 의 다른 파일은 읽기만 했다.

## 판정이 필요한 것 (lead)

1. **F1 범위.**
   - 측정상 거절은 지지면 규칙 (`not_supported`) 하나였다. 그런데 F1 은 task 대로 admissibility 전체 + exclusion gate 를 넘긴다.
   - 좁히려면 (`supported_max_bottom` 만 GT 판에서 끄기 = base argv `--supported-max-bottom-mm 60` 을 GT2 에서 빼기 — core 수정 없이 profile 한 줄) 판정해 달라.
   - 그 경우 개도 · destination 은 계속 거절할 수 있다. 다만 지금 장면에서는 일어나지 않았다.
2. **F2 를 closing 부터 거는 것.**
   - t1e1 의 근거가 closing frame 이라 closing (`set_grasp_active`) 부터 걸었다.
   - closing 중 GT target 의 새 관측은 장애물장에 안 들어간다. 손가락 사이의 그 물체를 권한 없는 link (팔) 도 못 본다. closing 전 위치의 표면은 TSDF 에 남는다 (masked = 지우지 않음).
   - attach 뒤에만 걸려면 한 줄이다.
3. **③ 남는 부분.** F2 뒤에도 남는 것: wrist 카메라의 로봇 잔여 pixel (finger 행, t1e1 seq 16 · 21 에서 126 · 226 px), 그릇 가장자리 static pixel (segmentation 시점 문제로 추정), ramekin 이웃 (t1e1, D1 이웃 허용을 끈 결과).
   - GT 판에서 code 0 (로봇) pixel 도 mask 하는 것은 "로봇 인식" 을 GT 로 바꾸는 일이라 이번 정의 (target identity · extent) 밖이다.
   - 판정해 달라 (같은 helper 로 code 0 을 더하면 된다).
4. **no_admissible note 문구.** `pipeline.py` 의 "(graspable at max opening … and off the destination)" 는 지지면 규칙을 말하지 않아 이번 원인을 숨겼다. 기록 문자열이 바뀌므로 그대로 두었다 — 고칠지 판정해 달라.
5. **ESDF 수준 확인은 GPU smoke 에 남는다.** 이 STEP 의 replay 는 TSDF 입력까지만 보였다.

## patch

- 파일: `/tmp/claude-0/-mnt-dev/576ccf52-3b29-45d1-8286-2a9249abaa31/scratchpad/gtto2.patch` — 이 STEP 의 worktree 변경 전부 (위 표의 파일 9 개 수정 + 새 파일 4 개: test 3 개 · 이 impl.md).
  - 기준 = main checkout 의 현재 파일 (SL-GTTO patch 적용 상태; 작업 전 worktree 와 byte 동일임을 `cmp` 로 확인).
- 확인: main checkout 의 해당 파일을 scratch 사본으로 복사한 뒤 `git apply --check` 통과, 적용 결과가 worktree 파일과 byte 동일 (`cmp`).
- 적용: main checkout 에서 `git apply <patch>` (queue job 경계에서, lead).

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->

- smoke: t4e0 · t3e0 는 첫 call 부터 grounding ok 일 것이다. 그 뒤는 실제 쥐기에서 지지면 · 캐비닛 근처의 다른 층 (팔 link · 이웃) 이 남을 수 있다.
- t0e0 · t1e1: `attached_point` 행의 상당수가 사라질 것이다. 계획이 쥔 구를 탁자 · plate 로 내리는 행과 ramekin · 로봇 잔여 행은 남을 수 있다.
