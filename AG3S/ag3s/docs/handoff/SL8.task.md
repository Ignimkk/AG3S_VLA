# SL8 — ATS target · obstacle 분할 능력 (offline, SafeLIBERO 20 episode) + 손실 단계 찾기

> writer: lead (benchmark checkout), 2026-11-04 14:4x KST. **결과를 보기 전에 고정.**
> 사용자 지시 (2026-11-04 14:3x KST): "ATS를 사용해서 타겟, 장애물 분할 능력을 실험해보자. 그리고 나서 뭐가 문제인지 파악해보는거야."
> 앞 측정: `outputs/2026-11-03_SafeLIBERO_baseline분할_ATS비교/`(1판, 명사 attention L8H6: target cluster 3D recall 0.046,
> 거리장 지움 0.104, obstacle 잘못 지움 0.003), `outputs/2026-11-04_SL7_P0m_SafeLIBERO_오프라인재생/`(P0m 같은 조건).

## 왜 다시 재나

앞 offline 재생은 모든 판에 **명사 attention row(`noun_target:L8H6`)** 를 넣었다. 지금 Panda 후보 판(pc판, `AG3S-final-panda-v3-e0to-pc`)은
closed loop에서 **행동 attention(`action:last:L8H2`)** 을 쓴다(SL7j: `-pc-l8h2explicit`가 pc판과 bit 동일).
그래서 실제로 closed loop에서 도는 ATS의 분할 능력은 아직 재지 않았다. 또 앞 측정은 최종 출력(cluster · 지운 voxel)만 채점했다.
그래서 target이 **어느 단계에서** 빠지는지(attention → seed → cluster 선택 → 거리장 지움)는 나누지 못했다.

## 장면 · 입력

- SafeLIBERO spatial Level I dev, VLA 단독 기록 20 episode(task 0–3 × init 0–4), policy call 796개 × head · wrist camera.
  - 기록: `20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only`
  - GT: image instant 재렌더 class map `20261102T110000_96fda38abd1b_safelibero-seg-dev-dataset-20261102{,-b,-c}`
  - class: target · destination · obstacle · other_object · fixture · support · robot · background
- AG3S 객체: pc판 server flag 그대로(`eval_real_ats_offline.build_ag3s`).
- 쥠 상태: 그리퍼 물리 신호(`eval_real_ats_safelibero.py --grasp physical`와 같은 규칙).
  - **한계:** offline에는 VLA KV feature가 없어 HSR6 probe를 돌리지 않는다.

## 조건 (attention만 바꾼다)

| 조건 | attention | 뜻 |
|---|---|---|
| A | `action:last:L8H2` 행동 attention. 같은 request를 다시 만들어 GPU로 추출(`PandaAttentionReader`, 10 step, noise seed 0) | closed loop의 pc판 ATS |
| B | `noun_target:L8H6` 명사 row(기존 `…-noun-rows-20261102`) | 앞 측정과 같은 attention |
| C | oracle: GT target pixel이 patch를 덮는 비율(privileged) | ATS 기하 · 선택의 상한. attention 문제와 grounding 문제를 가른다 |

## call별 단계 지표

점의 class = 그 점에서 10 mm 안 가장 가까운 GT 점(두 camera back-projection)의 class. 10 mm 안에 없으면 `none`.

| 단계 | 지표 |
|---|---|
| S1 attention | camera별 attention mass가 target · obstacle · 기타 class 위에 놓인 비율(2D patch, GT label 비율 가중), top-1 patch class |
| S2 seed | seed 점(`seed_indices`)의 class 구성. seed 다수 class |
| S3 cluster 선택 | cluster 수, target 다수 cluster가 있는지, 그 cluster의 순위. **선택된 target cluster의 다수 class**, 3D precision · recall(GT target 5 mm voxel) |
| S4 identity | 확정된 조작 물체(`manipulated`) 상태 |
| S5 거리장 지움 | 서버가 실제로 지운 voxel(`exclusion_trace.removed_ijk`)의 GT target recall, **obstacle voxel 중 지워진 비율** |

## 주 지표 (결과 전 고정)

1. **target 선택 정답률:** target이 보이는 call(두 camera GT target pixel 합 ≥ 50) 중 선택된 target cluster의 다수 class가 target인 비율
2. **obstacle 혼동:** 선택된 target cluster의 다수 class가 obstacle인 call 비율, 그리고 obstacle 잘못 지움(voxel 비율)
3. **target 덮음:** 거리장 지움의 target recall(평균, 중앙값)

## 손실 단계 분류 (target이 보이는 call마다 하나)

| 분류 | 규칙 |
|---|---|
| no_output | ATS target 없음(`no_seed` · `no_cluster` · `no_admissible`) |
| attention | 선택 오답이고, seed 다수 class가 target이 아님 |
| grounding | 선택 오답인데 seed 다수 class는 target(cluster 형성 · 점수 · 확정 단계에서 잃음) |
| small_cluster | 선택 정답, cluster recall < 0.3 |
| erase_loss | 선택 정답, cluster recall ≥ 0.3인데 지움 recall < 0.3 |
| ok | 선택 정답, 지움 recall ≥ 0.3 |

조건 A · B · C의 분류 분포를 비교한다.
- C(oracle)에서도 grounding · small_cluster · erase_loss가 크면, 문제는 attention이 아니라 AG3S 기하 · 선택 층에 있다.
- A만 attention 분류가 크면, 문제는 attention 원천에 있다.

판정 규칙은 두지 않는다(진단). 채택 · 코드 변경은 결과 보고 뒤 사용자 판정 사항이다.

## 출력

- 원시 기록: storage `results/benchmark/<날짜_commit>_sl8-ats-seg-{attn-action,A,B,C}`
- 보기: `outputs/2026-11-04_SL8_ATS_target장애물분할/`. 내용:
  - 실제 head 영상 위 GT target · obstacle 윤곽, seed, 선택 cluster
  - 단계별 funnel graph
  - 분류 표
  - README
- GPU: attention 추출 · 재생 모두 shared lock(`scripts/setup/gpu_lock.py shared`)
