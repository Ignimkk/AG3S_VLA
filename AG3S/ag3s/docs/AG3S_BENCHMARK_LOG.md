

**이어서 — 논문 표 episode 의 1/10, part A SafeLIBERO 32 ep (2026-11-12 21:11–22:59 측정, 23:1x 기록, HEAD `7e86ca60`).** 사용자 지시 (21:08): 표에 넣기로 한 episode 의 1/10 을 동결 판으로 돌리고, 표 형식으로 보인 뒤 실패 원인을 분석한다. 용어: '인식된 target' = server 가 조작 대상으로 잡은 점들이 실제로 속한 물체, 'task 물체' = 그 task 가 옮기라고 한 물체.

| 시각 | 한 일 | 결과 |
|---|---|---|
| 21:11–22:59 | `pkgFrs` SafeLIBERO 4 suite × L-I · L-II × task 0–3 × init 19 (`ms-tab10sdm-*-i19`) | L-I TSR · SSR · CAR **8 · 4 · 8 / 16**, L-II **9 · 8 · 14 / 16**. suite 별 안전 성공 Spatial 4 · Object 6 · Goal 1 · Long 1 (각 8) |
| 23:00–23:1x | 안전 성공이 아닌 20 ep: 실패 형태 (`shape.py`) · 정답 물체 이동 (`ep_digest.py`) · 막은 물체 (`surface_owner_task.py`) · 인식된 target (`target_truth.py`) · 영상 | target 오인식 10 · 성공 + 충돌 4 · 옮기기 · 놓기 6. target 이 task 물체인 call 이 절반 미만인 11 ep 중 안전 성공 1, 절반 이상 21 ep 중 11. 거절된 529 call 의 51 % 를 task 물체가 막음 |
| 21:3x | SL-LS1: 공식 `pi05_libero_safety` checkpoint 용 controller 변환 구현 (worktree, commit `3f702a1d`) | 단위 test 7 + 주변 test 통과, launcher dry-run 통과. closed loop 은 아직 |

**[해석]** 표 전체의 분포에서 가장 큰 원인은 target 오인식이다 (Long 의 mug task 4 ep 은 mug 가 target 으로 잡히지 않음; 장애물 · 목적지 · 로봇 손끝이 target 으로 잡힘). 상세 표는 계획서 §9.31.

**실제 장면 · 그림:** `AG3S/ag3s/docs/figures/sl-tab10-safelibero-init19-20261112.png` (+ `.json`) — Long L-II task 3 (쥔 mug 를 장애물로 봄) · Long L-I task 2 · Goal L-I task 1 의 장면, episode 별 target 인식 비율과 결과, 막은 물체, suite 별 안전 성공. script `experiments/segmentation/sota_offline_20261111/fig_tab10_sdm.py`.
