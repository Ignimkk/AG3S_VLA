# IFG-S2SERVE — 안전 경보와 학습 없는 3D 쥠 규칙을 서버에 연결 (implementer)

작성: 2026-10-19 13:30 KST, lead. 사용자 지시: "다음에 돌려야 할 것 권장사항대로 수행".
근거: `IFG-S2FK.task.md` · `IFG-S2FK.verify.json` (held-out offline, FK 구 mask), WORKLOG 10-19 03:50 절.

## 권장안 (사용자 승인된 범위)

1. **안전 경보 (feature 기억 신뢰도 경보, R11)** — 서버에 추가, flag 로 켠다. 기본 off.
2. **학습 없는 3D 쥠 규칙** — grasp latch 판단에 추가, flag 로 켠다. 기본 off.
3. 작업 단계 판단은 **HSR(subtask gate) 그대로** 둔다. 학습형 weight 는 쓰지 않는다.
4. 경보가 울린 frame: 그 frame 의 **target 접촉 예외를 보류**한다(target 을 장애물로 유지). 로봇 렌더 mask 는 쓰지 않는다(보류).
5. 두 flag 를 모두 끄면 현재 serving(C1aN = `--attention-source noun_target:L10H7` + 기존 flag)과 **출력이 비트 단위로 같아야** 한다.

## 1. 안전 경보 — 정의 (잠근 offline 코드와 같게)

- 정의 원본: `AG3S/ag3s/experiments/studies/ifg_memory.py` (R10b/R11 경로, `center=True`), `ifg_confidence.py`, 잠근 설정
  `results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba/LOCK_heldout.json` → `arms.confidence_gate`
  (`score = q_mem_chain:prefix_L8 (centered)`, `abstain_if_below = 0.5904207590672415`).
- 필요한 입력:
  - **영역:** `runtime/attention_surface.select_target_surface`가 고른 영역(잠근 `top1-only-ns:N` 선택). robot mask 는 **FK 구 mask**(serving 기본)
    - 경보에 필요 없는 code image 생성(`surface_codes`)은 경보 전용 경로에서 건너뛸 수 있으면 건너뛴다
    - target extent 자체는 **바꾸지 않는다**(공 영역 `ball_owned` 그대로). 표면 영역은 경보 점수 계산에만 쓴다
  - **VLA feature:** prefix layer 8 의 image token residual(`resid_out`, 오프라인 dump 의 `prefix_L8.npz` `resid_out_image`와 같은 텐서). 서버의 VLA forward 에서 hook 으로 얻는다. dump 를 만든 코드에서 정확한 텐서 위치를 확인하고 같게 맞춘다: held-out collector `results/development/20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout/p/hc_dump.py`(held-out `prefix_L8.npz`), dev dump 는 WORKLOG 10-15 18:17 절의 collector `phase_dump/vla_phase_dump.py`
  - **영역 feature:** `ifg_memory`의 정의 그대로(영역 node 의 patch-pixel 행 합 @ centered unit patch feature, 정규화)
  - **기억:** `chain` = 직전 frame 의 최종 선택 영역 feature. episode 시작(서버 `reset`)에 비운다. 첫 frame 은 기억이 없으므로 경보 없음
- 경보 = 점수 < 0.5904 → 그 frame 은 target 접촉 예외 보류. 보류 방식은 현재 서버에 있는 가장 가까운 경로를 쓴다(예: target permission 거부 / abstain 처리). 어떤 경로를 썼는지 impl.md 에 적는다
- flag 예: `--confidence-gate {off,on}`(기본 off), `--confidence-gate-threshold`(기본 잠근 값). 이름은 기존 관례에 맞춰도 된다
- 기록: `--record-constraints` summary 에 frame 별 `gate_score`, `gate_alert`, `gate_why`(no_memory · no_region · no_feature), 단계별 ms

## 2. 학습 없는 3D 쥠 규칙 — 정의 (IFG-S2FK 에서 결과 전 고정한 것)

- `hand_d` = target 중심과 두 손 대리점 중 가까운 쪽의 거리. **손 대리점** = `link_{left,right}_arm_6` 아래 body 들의 평균 위치(base frame) — offline 정의(`ifg_phase.episode_features`)와 같다. 서버에서는 요청의 관절 상태로 FK 한다
- target 중심 = 서버가 이번 frame 에 고른 target cluster 의 중심(현재 serving 의 grounding 결과)
- 규칙: `grip < 0.9` **그리고** `hand_d ≤ 0.131 m` 이면 쥠. target 이 없으면(이번 frame cluster 없음) gripper 조건만 쓴다
- 연결 지점: 현재 serving 의 쥠 판단(`trajopt/grasp_latch` 의 latch 조건, `pipeline.set_grasp_latched` 경로)을 읽고, **latch 를 거는 조건**에 위 규칙을 AND 로 더한다. 이미 걸린 latch 를 푸는 조건(revoke)은 바꾸지 않는다. 다른 연결이 더 맞다고 판단되면 impl.md 에 근거와 함께 적고 lead 에게 먼저 알린다
- flag 예: `--held-3d-rule {off,on}`(기본 off), 상수 `--held-3d-max-m 0.131`
- 기록: frame 별 `hand_d`, `held_3d`, `held_gripper`

## test (`tests/`, CPU, `.venv-ag3s`)

- flag off: 기존 serving 경로의 출력(summary · refined chunk)이 바뀌지 않음 — 기존 회귀 test 와 regression-baseline skill(legacy · cuRobo)로 확인
- 경보: 고정 입력에서 점수가 잠근 offline 함수와 같은 값(수치 일치)인지, 첫 frame 무기억, 기준값 경계
- 3D 쥠: 경계값(0.131 m 안팎), target 없음 fallback, latch 조건 AND
- held-out frame 몇 개로 서버 경로의 경보 점수 == `IFG-S2FK` offline 점수(`$RUN/s2fk/run/rounds/R11b/conf.jsonl`)인지 확인할 수 있으면 한다(CPU)

## 하지 말 것

로봇 렌더 mask 사용 금지. 잠근 `experiments/studies/ifg_*.py` 수정 금지(import 해서 쓰는 것은 된다). GPU 실행 금지(필요하면 lead 에게 요청). commit 금지. 다른 세션 process 금지.

## 넘길 것

`AG3S/ag3s/docs/handoff/IFG-S2SERVE.impl.md`: 바뀐 파일 · 함수, 새 flag, feature hook 위치와 텐서 일치 근거, 보류 경로, latch 연결 지점, test 출력 마지막 줄, 예상 추가 지연(CPU 측정치가 있으면), 남은 위험.
