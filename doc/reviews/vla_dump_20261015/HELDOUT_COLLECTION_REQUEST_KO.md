# Held-out 수집 요청서 — IFG(Attention + 3D) 검증용 (v2)

작성: 2026-10-15 KST, 분석 session. v2는 같은 날 사용자 답변 3건을 반영했다(§9).

**수신:** VLA dump collector를 맡은 agent. 사용자가 전달한다. 이 요청은 새 rollout과 내부값 dump를 요청하며, 분석 session은 수집을 직접 하지 않는다.

배경 결과는 [SUMMARY_METHODS_RESULTS_KO.md](SUMMARY_METHODS_RESULTS_KO.md)에 있다. 지금까지의 수치는 모두 **dev**(T53 + T54c 32 episode, 이미 결과를 본 episode)에서 나왔다. 방법과 threshold는 `$RUN/LOCK_heldout.json`에 잠갔고, 새 데이터를 본 뒤에는 바꾸지 않는다.

`$RUN = /mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba`

## 1. 왜 새 수집이 필요한가

| 현재 dump의 한계 | 이번 요청 |
|---|---|
| T53·T54c는 이미 결과를 본 episode다 | 분석에 쓴 적 없는 episode |
| Episode당 key moment 3–7 frame만 있다 | episode의 **모든 policy step** |
| 기록에 depth가 없다(replay로 대체) | rollout에서 depth/K/T 직접 기록 |
| 엉뚱한 물체를 쥔 사례가 없다 | instruction swap 재추론(§5.1) |
| Target을 가리는 장애물 사례가 없다 | 가림 0 / 1/3 / 1/2 테스트(§3.2) |

## 2. Episode 선정 규칙 (rollout 전에 확정·기록)

1. **학습 분포 안의 장면을 써도 된다(사용자 답변).** 학습 split 밖일 필요는 없다. 보고서에는 "in-distribution 장면"으로 표기한다.
2. **분석에 이미 쓴 episode는 제외한다.** 그래야 "결과를 보지 않은 데이터로 확인"이 성립한다.
   - 목록: `$RUN/used_episodes_prior.json`
   - 포함: T53 16, T54c 16, T61 16, CL4 9, T58 3, T47 dev 7
3. **성공 episode 중심(사용자 답변):**
   - 주 평가 묶음은 **simulator가 판정한 rollout 성공 episode**로 한다(`atomic_transport` 성공 판정, 기존 client 기록).
   - 이 기준은 분석 session이 결과를 보기 전에 정해지는 것이다.
   - B0와 후보를 **같은 episode에서** 비교하므로 비교 자체는 공정하다.
   - 보고서에 "성공 rollout 한정 평가"라고 명시한다.
4. **실패 episode는 지우지 않는다.** 별도 목록으로 기록·보관하고, 분석 session이 부록으로만 보고한다.
5. **Seed:** `seed = episode_index * 10 + 1`. Rollout 전에 `seeds.json`(episode, seed, port, stratum)을 먼저 쓴다.
6. **Serving 설정:** T53 C1a flag set과 checkpoint `pi05_rby1_randomized_pick_place_16d_lora/.../29999`를 쓴다.

## 3. 시나리오

### 3.1 주 평가 (결론에 쓰는 묶음)

| Stratum | 정의 | 목표 (성공 rollout 기준) |
|---|---|---|
| H-clear | 장애물 없음 | 12 episode |
| H-adjacent | target 옆 장애물(`post_0`/`fence_0`, target 중심에서 5–10 cm), head camera에서 target을 가리지 않음(가림 ≤ 5%) | 12 episode |
| H-place | 바구니 rim 옆 `crate_post_0` | 8 episode |

성공 episode가 목표 수에 도달할 때까지 수집하고, 그 과정의 실패 수를 기록한다.

### 3.2 가림 시나리오 (테스트용 — 별도 보고, 주 결론에 넣지 않음)

**사용자 판정:** 테스트용이다. 결과는 별도 표로만 보고하고 채택 판단에 섞지 않는다. 비중은 사용자가 판단한다.

**가림 수준 3단계(사용자 지정).** 같은 장면에 장애물 위치만 바꿔 세 조건을 만든다(장면 짝 비교).

| 조건 | head camera(`zed_left`)에서 target 가림 비율 |
|---|---|
| O-0 | 0–5% (target 전부 보임, 장애물은 근처에 있음) |
| O-1/3 | 28–38% |
| O-1/2 | 45–55% |

목표: 장면 8개 × 3조건. 각 조건에서 성공 rollout이 나와야 한다.

**장애물 위치 탐색 절차.** T58 `p/search.py`(T42-2)의 검사 틀을 재사용하되 목적을 반대로 바꾼다. T58은 장애물이 경로를 막는 위치를 골랐고, 이번에는 **경로를 막지 않고 시야만 가리는 위치**를 고른다.

1. **기준 궤적:** 먼저 장애물 없이 그 장면을 rollout한다(= H-clear 겸용). 성공한 궤적만 기준으로 쓴다.
2. **후보 위치:** head camera 중심과 target 중심을 잇는 선분 위, target 쪽 구간에 둔다.
   - 기둥(`post_0`): 반지름 0.015–0.02 m, 높이 0.10–0.26 m.
   - 필요하면 가로 막대(`overbar_0`)도 쓴다.
   - 위치를 선분 좌우로 조금씩 옮겨 가림 비율을 조절한다.
3. **가림 비율 측정:** 기준 궤적의 접근 구간(첫 gripper 닫힘 전 8 step)에서 head camera로 두 번 렌더한다. 장애물을 둔 상태와 park(치운) 상태를 비교해 `1 − target_px(장애물) / target_px(park)`를 구하고, 구간 중앙값을 쓴다. 3 camera 값을 모두 기록한다.
4. **Feasibility·reachability 검사(사용자 조건: 장애물 때문에 잡지 못하는 위치는 안 됨).** T42-2 검사를 다음처럼 바꿔 적용한다.
   - (a) reset 상태: 모든 robot geom이 장애물에서 ≥ 0.03 m.
   - (b) **기준 궤적 전체**(접근·grasp·운반·놓기)에서 robot과 target이 장애물에서 ≥ 0.03 m. 같은 동작이 그대로 가능해야 한다.
   - (c) target 주변 grasp 공간: 장애물 표면과 target 표면 사이 수평 간격 ≥ 0.04 m(손가락이 들어갈 자리).
5. **Head camera visibility(사용자 조건):** 모든 조건에서 target이 head camera에 보여야 한다(가림 ≤ 55%). 보이는 target pixel이 50 px 미만인 위치는 쓰지 않는다.
6. **최종 확인:** 장애물을 둔 상태로 실제 rollout한다. 성공한 경우만 테스트 묶음에 넣고, 실패는 사유와 함께 기록한다. 장애물이 보이면 정책 행동 자체가 바뀔 수 있으므로 이 단계가 필요하다.
7. **기록:** 장면마다 장애물 profile JSON(slot, 위치, 크기), 측정한 가림 비율(3 camera), 검사 (a)–(c)의 최소 거리, rollout 결과.

## 4. Rollout 기록 (모든 episode)

- `pi05_infer.py --record-ag3s --record-depth`(인자 없음 = 3 camera)로 모든 inference step의 depth(uint16 mm)·K·T를 기록한다(step당 약 1 MB).
- Serving chunk record를 보존한다(나중에 `hold` 조건 replay에 필요).
- Snapshot XML, qpos/qvel, obstacle profile, prompt, seed, checkpoint, code commit을 기록한다(기존 policy record 형식).

## 5. 내부값 dump — 우리가 쓰는 것만 (사용자 답변)

모든 policy step에 대해 아래만 저장한다. 잠근 방법(`LOCK_heldout.json`)이 실제로 읽는 항목만 골랐다.

| 파일 / key | 내용 | 쓰는 곳 |
|---|---|---|
| `inputs.npz` | 3 image(224), raw_state, prompt, token pieces, query_rows(target/destination/verb), image_order | S0 정합, seed 위치, figure |
| `derived.npz` | `noun_target_L10H7`(768), `serving_L8H2_last_d0`(768), `hsr_p`(3) | seed(noun), B0, 단계 판단 |
| `prefix_L8.npz` | prefix L8 `resid_out`의 **image row만**(768×2048) | 기억·신뢰도 경보 |
| `done.json` | episode, frame, t_step, stratum, prompt_swap 여부, 기존 parity 필드 | S0, 분할 |

- 대략 step당 5 MB, 75 step episode당 약 0.4 GB, 주 평가 32 episode 약 13 GB(실측 보고 요망).
- 이전 collector의 grad, 모든 layer, attention 전체, SigLIP block, MLP 값은 **저장하지 않는다.**
- FP16으로 저장할 때 non-finite가 나오면 해당 key를 FP32로 저장하고 표시한다.

### 5.1 Instruction swap 재추론 (rollout 추가 없음)

주 평가의 성공 episode에서 held 구간(grasp 후 ~ release 전) step만 쓴다. Prompt의 target 명사를 장면에 있는 다른 과일로 바꿔 같은 관측으로 다시 추론하고, 위 4개 항목만 dump한다(`prompt_swap = true`, 원래·바뀐 명사 기록).

## 6. 전달 형식과 수락 조건

- **출력:** 새 run 디렉터리(`results/development/<KST>_<commit>_<uuid>/`). 기존 run은 수정하지 않는다.
- **`manifest.json`:** episode, group, split=heldout, stratum, seed, record 경로, fingerprint, rollout 성공 여부, obstacle profile, 가림 비율, 실패·중단 사유.
- **완료 표시:** frame 디렉터리마다 `done.json`을 마지막에 쓴다.
- **분석 session의 수락 검사(S0와 동일 항목):**
  - state·image와 record 일치
  - depth planar z
  - roundtrip ≤ 0.5 px
  - 입력 finite
  - 실패 frame은 사유와 분모를 보고한다.

## 7. 수집 후 분석 (잠근 설정 그대로)

| 평가 | 내용 |
|---|---|
| Target extent | 후보(`top1-only-ns+rm:N`) vs B0, B0 + render mask, G:N. 3D IoU/precision/recall, 이웃·support·robot FP |
| 시간축 기억 | Mem-first:prefix_L8:δ0.05(centered), 보조 arm |
| 신뢰도 경보 | 잠근 threshold 0.5904에서 잘못된 선택 탐지, FP 제거율·TP 손실률 |
| 단계·파지 | 잠근 LR weight vs HSR argmax·gripper 규칙. 모든 step이 있으므로 **전환 지연**도 측정 |
| P_task | swap dump에서 엉뚱한 물체 지시 상태의 경보·판단 |

- **주 결과:** §3.1 성공 episode.
- **부록:** 실패 episode, §3.2 가림 테스트(O-0 / O-1/3 / O-1/2 짝 비교, 별도 표).
- **분석 쪽 준비:** B0 재실행은 이번 dump의 `derived.npz` map을 읽도록 `ifg_baseline.py`를 고친다(이전에는 T53 추출 npz를 읽었다). 이 수정은 분석 session이 한다.

## 8. 남은 확인 사항

없음. 사용자 답변 3건으로 해소했다(§9).

## 9. 사용자 답변 (2026-10-15)

| 질문 | 답변 | 반영 |
|---|---|---|
| 학습 split 밖 장면이 필요한가 | 필요 없음. 성공 episode 위주로 | §2-1, §2-3. 다만 분석에 쓴 episode 제외, 실패 episode 별도 보관, "성공 rollout 한정" 명시는 유지 |
| 저장 범위 | 우리가 쓰는 데이터 위주로 | §5 최소 4항목 |
| 가림 장애물 위치 | 장애물 때문에 잡지 못하면 안 됨(feasible, reachable). Head camera에서 target은 보여야 함. 가림 0 / 1/3 / 1/2 테스트 | §3.2 탐색 절차와 검사 (a)–(c), visibility 조건 |
