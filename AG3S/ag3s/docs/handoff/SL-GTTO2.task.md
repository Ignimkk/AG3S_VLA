# SL-GTTO2 — GT 인식이 TO까지 전달되지 않는 두 틈 고치기 (+ 팔 link 충돌 진단)

> writer: lead, 2026-11-05 01:1x KST. 사용자 위임(2026-11-05 ~01:00 KST, "오전 11:30까지 스스로 판정하며 수행")에 따른 lead 자율 판정. **결과를 보기 전에 고정.**
> 선행: [SL-GTTO.task.md](SL-GTTO.task.md) (판 정의 · 장면 · 판정 규칙), [SL-GTTO.impl.md](SL-GTTO.impl.md) (구현).

## 0. 근거 — SL-GTTO 본 실행 (2026-11-04 23:50 – 11-05 01:0x, commit `89b42459` + GTTO patch)

GT + 공용 TO (`AG3S-commonto-gt`) vs 같은 기계 VLA 단독: LIBERO 9 성공 2 vs 9, SafeLIBERO 11 안전 성공 0 vs 2 (공식 성공 0 vs 11). §5 규칙 1 · 2 실패.
실패 episode의 call 기록 분류 (`experiments/segmentation/gtto_aggregate.py`):

| 층 | 증상 (call 기록) | episode |
|---|---|---|
| ① GT target이 admissibility에서 거절 | 모든 call `grounding = no_admissible` → `uncertified` HOLD. note: "none of 1 cluster(s) is admissible (graspable at max opening 78.0 mm and off the destination)". cluster는 code 1 점만으로 만든 것 | LIBERO t4e0 · t9e0, SafeLIBERO t3e0 · e2 · e3 · e4 (44/44 · 56/56 HOLD) |
| ② 쥔 물체의 잔여 점 | attach call에서 cloud의 code 1 점이 ~170 → 0–20 (self-filter가 쥔 물체 구를 지움; code 1 **pixel**은 그대로) → `no_cluster` → 그 frame target 제외 없음 · oracle 미세 층 `no_target_free_layer` → `attached:panda_hand ↔ esdf` (`attached_point`) · `nearest_is_not_manipulated` 충돌 → HOLD | LIBERO t0e0 · t7e0, SafeLIBERO t1e1 · t1e4 · t2e0 · t2e1 |
| ③ 팔 link 충돌 | `panda_link5/6/7 ↔ esdf` `unauthorized_link` (예: LIBERO t3e0 seq 43 link7 −45.8 mm, 점 base [0.794, 0.198, 0.273]) | LIBERO t3e0 · t0e0, SafeLIBERO t2e0 등 |

①과 ②는 "GT가 target identity · 범위를 준다"(SL-GTTO §2)는 정의가 TO 입력까지 도달하지 못한 틈이다 — 인식 기계(admissibility · cluster 최소 점 수)가 GT를 다시 걸러 낸다.
따라서 GT 판 안에서 고친다 (TO 핵심 · 쥐기 처리 flag 표 §1은 바꾸지 않는다). ③은 TO · 장면 층일 수 있어 이번에는 진단만 한다.

## 1. 구현 (모두 기본 꺼짐, 꺼짐이면 `AG3S-commonto-gt` 포함 모든 기존 판의 argv · 기록 동일)

새 판 `AG3S-commonto-gt2` = `AG3S-commonto-gt` + 아래 두 flag. 기존 `AG3S-commonto-gt`는 그대로 둔다(본 실행 재현용).

1. **F1 `--oracle-admissible`** (`grounding_source == oracle` 에서만 허용): GT cluster는 identity가 정답이므로 TargetConfirm admissibility(`_assess_frame`: 최대 개도 대비 narrowest extent, destination 겹침)와
   `_exclusion_gate` 재검사에서 **거절하지 않는다.** 검사 결과는 계산해서 기록만 한다(`gt_perception.grounding.admissibility_would_refuse` 같은 키).
   - **먼저 진단:** 기록된 LIBERO t4e0 · SafeLIBERO t3e0 frame에서 어느 조건(개도 · destination)이 거절했는지, GT cluster의 extent가 실제로 얼마인지 impl.md에 적는다.
     만약 원인이 GT 입력 쪽 버그(예: code 1 cluster에 다른 물체 점이 섞임, destination이 target 자신으로 등록됨)라면 그 버그를 고치는 것이 우선이고 F1의 범위를 그만큼 줄인다.
2. **F2 `--oracle-held-target`** (`grounding_source == oracle` 에서만): GT target의 관측 점은 cluster 성공 여부와 무관하게 target이다.
   - grounding이 `no_cluster` / `no_geometry`라도 이 요청의 code 1 점(1개 이상)과 oracle memory의 target 표본은 장애물장에서 빠져야 한다(미세 target-free 층이 GT target 표본으로 적용됨).
   - 쥔 물체(attach) 동안 쥔 물체 자신의 관측 점(code 1)이 `attached:*` · 손 행의 장애물이 되지 않는다.
   - **먼저 진단:** LIBERO t0e0 seq 29 (`attached:panda_hand[1] ↔ esdf −65.6 mm`, 점 base [0.745, 0.151, 0.103]) · SafeLIBERO t1e1 seq 12에서 그 ESDF 표면을 만든 점이
     code 1 잔여 점인지, mixed 점인지, TSDF/oracle memory 의 이전 frame 표본인지 확인해 impl.md에 적는다. 구현은 그 원인을 겨냥한다.
   - cluster 최소 점 수(`clustering.min_points` 20)를 GT 판에서만 낮추는 방법도 허용하지만, 0점(완전히 self-filter 된) frame도 처리되어야 한다.
3. **③ 진단만:** LIBERO t3e0 seq 43 link7 행의 ESDF 표면이 무엇의 점인지(code 2 정적 fixture · code 3 물체 · 관측되지 않은 영역), VLA 단독이 같은 장면에서 그 근처를 지나는지(같은 기계 VLA 단독 run
   `*_gtto-vla-libero9-*`). 고치지 않는다.
4. profile: 파일 끝 SL-GTTO 블록에 `AG3S-commonto-gt2` 추가. Panda 서버 allow-list에 두 flag.
5. test: 꺼짐 동일성(기존 SL-GTTO test 통과 그대로), F1 · F2 단위 test, **기록된 실패 frame replay**(가능한 범위: `episodes/*_geom.npz` 의 depth · segmentation · robot_state로 pipeline을 CPU에서 다시 돌려
   F1이 t4e0 / t3e0 frame에서 grounding ok, F2가 t0e0 seq 18–43 held frame에서 `attached_point` 충돌 행을 없애는지). replay가 불가능하면 그 이유를 적고 합성 장면 test로 대신한다.

## 2. 판정 (lead, 결과 전 고정)

- 구현 후 GPU smoke(SOLO): LIBERO t4e0 · t0e0, SafeLIBERO t3e0 · t1e1 각 1 episode. 확인 키: t4e0/t3e0 첫 call grounding ok(HOLD 없음), t0e0/t1e1 attach 뒤 `attached_point` 행 없음.
- smoke 키 통과 → `AG3S-commonto-gt2`로 30 장면 전체를 다시 돈다(SL-GTTO §5 규칙 그대로, VLA 단독 기준은 이미 있는 같은 기계 run 재사용).
- 규칙 통과 여부와 별개로 남은 실패 episode를 같은 층 분류로 적는다.

## 3. 범위 · 자원

- 구현은 이 worktree(`.claude/worktrees/agent-a5af2649e9e869962`)에서만. main checkout은 queue job이 import하므로 lead가 job 경계에서 patch로 반영한다.
- CPU만, worker ≤ 2, working set ≤ 20 GiB. GPU 사용 금지. commit 금지.
- core 수정(`pipeline.py` · `target_grounding.py`)은 development 소유 코드다 — 수정 범위를 impl.md에 따로 표로 적는다.
