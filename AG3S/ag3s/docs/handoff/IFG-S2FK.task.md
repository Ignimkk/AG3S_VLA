# IFG-S2FK — 렌더 mask 없이(FK 구 mask) 채택안 B의 2단계 요소를 held-out offline으로 다시 잰다 (lead, CPU)

작성: 2026-10-19 03:00 KST, lead. **결과를 보기 전에 고정한 내용이다.**
근거: 사용자 판정 2026-10-19 — 로봇 렌더 mask 보류(실물 custom 기구물 · 실시간성), "1번부터 진행".
run: `/mnt/dev/AG3S_VLA-storage/results/development/20261019T025853+0900_f15e587290f6_93594d3dffc0` (이하 `$RUN`).

## 데이터

- held-out 준비물 `20261016T030741+0900_f15e587290f6_19ea09c7a8b8/heldout_eval`(기록된 rollout depth, 정답 label, `replay/*.npz`의 `robot_mask` = serving FK 구 mask)
- held-out dump `20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout/dump`(VLA 내부값)
- 프레임 집합은 잠근 held-out 평가와 같다(`ifg_heldout_eval.heldout_refs`, s0 무효 29개 제외). 주 결과 = 성공 rollout episode, 실패는 따로
- 잠근 코드(`LOCK_heldout.json`의 `code_sha256`)는 고치지 않는다. 새 wrapper `ifg_stage2_fk.py`만 쓴다

## A. 기억 feature 영역 전환 · 안전 경보 (FK 구 mask)

- 가짜 run `$RUN/s2fk/run`: `rounds/R6/robot_mask` → `$RUN/s2fk/robot_mask_fk`(replay npz의 FK mask를 같은 key로 옮긴 것). 나머지는 잠근 평가와 같다
- 실행: `ifg_memory.episode(center=True)` (잠근 held-out 기억 단계와 같은 호출)
- arm: A-fk = 전환 없음(R10b의 `R6` arm, FK mask) · B-fk = `Mem-first:prefix_L8:d0.05` · 대조로 잠근 평가의 A(렌더) · B(렌더)
- 지표(잠근 `ifg_heldout_summary`와 같은 정의): episode 평균 IoU · precision · recall, B-fk − A-fk paired bootstrap 2000 (seed 20261015), 전환 frame 수
- 경보: 잠근 기준값 0.5904(dev 렌더 조건에서 정한 값, **다시 정하지 않는다**). 잘못 고른 frame(IoU < 0.1) 중 잡은 수, 오경보 수, AUROC, 경보 frame을 버릴 때 FP 제거율과 TP 손실률
- 일관성 검사: A-fk의 IoU가 잠근 평가의 `top1-only-ns:N`(R5, FK mask)과 frame마다 같아야 한다. 다르면 멈추고 보고

## B. 3D 관계로 파지 · 작업 단계 판단

**특징값**은 `ifg_phase.episode_features`와 같은 정의다.
- 영역 = R6 선택(같은 robot mask 조건)
- `hand_d` = 영역 중심 ↔ 두 손 대리점(arm_6 아래 body 평균, base frame) 최소 거리
- `height` = 영역 중심 z − 근처 지지면 node z median
- `lognodes`, `grip` = min(raw_state[7], raw_state[15]), `hsr` = dump `derived.npz` hsr_p
- 손 대리점은 snapshot `model_transport_pick_place_obstacles.xml`(nq 66)에 replay qpos를 넣어 구한다
- 렌더 mask 조건과 FK mask 조건 **둘 다** 계산한다(학습형 weight의 원래 조건 대조)

**정답**: 프레임의 `grasp_state`(시뮬레이터 접촉 기반, pre/post → pick, held → place, placed → home). 평가에만 쓴다.

**학습 없는 규칙 (사전 고정)**
- 상수: 손 대리점 ↔ 손가락 파지점(`left_ee`/`right_ee`) 거리 = **0.0813 m**(모델 `teleop` 자세에서 측정, 양손 같음). 과일 반지름 상한 0.05 m. → **D_hold = 0.131 m**
- 쥠 규칙 `held_rule` = `grip < 0.9` 그리고 (`hand_d ≤ 0.131`, 영역이 없으면 이 조건은 생략 = gripper만)
- 단계 규칙 `phase_rule`(episode 안 시간 순서): `held_rule`이면 place. 아니면 같은 episode의 앞선 frame 중 `held_rule`이 있었고 지금 `grip ≥ 0.9`이면 home. 그 밖에는 pick
- 알려진 한계(미리 적음): 놓친 뒤(떨어뜨림)와 바구니에 놓은 뒤를 구분할 정보가 없다. 바구니 위치를 쓰지 않는다

**비교 대상**
- 기존 serving 규칙: HSR argmax(단계), `grip < 0.9`(쥠)
- 학습형: 잠근 weight `LR[hsr+grip+geo]`(dev 렌더 조건에서 학습)를 그대로 적용 · 참고로 held-out LOEO 재학습(상한 참고, 채택 근거로 쓰지 않음)
- 학습 없는 규칙: `held_rule`, `phase_rule`

**지표**: accuracy, macro-F1, class별 recall, episode 단위 paired bootstrap 2000(seed 20261015) — 규칙 − 기존 serving 규칙, 규칙 − 잠근 학습형. 렌더 · FK 조건 각각.

## 판정 원칙

- 동등성이나 "영향 없음"을 판정하지 않는다. CI가 0을 포함하면 "미확정"이라 쓴다
- 기준값 · 규칙 상수를 결과를 보고 바꾸지 않는다. 바꿀 필요가 보이면 사용자에게 묻는다
- 결과 기록: WORKLOG, figure(`AG3S/ag3s/docs/figures/ifg-s2fk/`, sidecar `.json`)
