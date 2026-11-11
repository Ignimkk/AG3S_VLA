# ST1 — 제약 단계 추종 모듈 (Stage Track v1): 정의 · 위치 · 다시 돌리는 법 · 결과 (2026-11-11, dev-55)

> **다음 세션용 한 줄:** "`doc/active/experiments/STAGE_TRACK_ST1_KO.md` 를 읽고, 미채택된 **ST1 (제약 단계 추종)** 을 지금 구성 위에서 다시 테스트해 주세요."
> 변형은 번호를 올려 부른다 (ST2 = ST1 + 연속 판정, §6). 상태: **미채택** (2026-11-11, §5). 코드는 기본 꺼짐으로 보존한다 (사용자 결정: 채택과 관계없이 commit · push).

자료 표시: **[M]** 측정 · **[D]** 기록에서 계산 · **[I]** 해석. 상세 기록 (git 제외): `outputs/2026-11-10_제약단계_추종/README.md` §1–§12.

---

## 1. 무엇인가

AG3S 의 grasp latch (searching / latched / closing / held / placed) 는 제약 단계를 정한다 — 쥔 물체를 attach 하는 시점, 손 · 쥔 물체의 접촉 허용,
목적지 등록 (`--latch-phase`, `--closing-hand-exempt`, `--held-hand-exempt`, `--destination-after-held`, `--subtask-from-latch` …).
지금 latch (pe123458, `--grasp-opening-rule`) 는 **개도가 멈춘 것을 확인한 뒤** HELD 로 간다.

**ST1** 은 π0.5 의 prefix hidden state 와 gripper feedback 으로 4 단계를 매 요청 추정하고, 그 결과로 latch 를 움직인다.

| 단계 | 뜻 (사용자 결정 2026-11-10) | 정답 (MuJoCo 접촉) |
|---|---|---|
| S1 approach | 대상에 다가감 | 아래 셋이 아님 |
| S2 grasp | 닫는 중, 아직 두 손가락이 대상에 안 닿음 | 닫힘 시작 → 첫 grip |
| S3 carry+place | 두 손가락이 대상에 닿음 | grip (2 frame 틈 메움, < 5 frame 이면서 상자 밖에서 끝나면 제외) |
| S4 released | 목적지 상자 안에서 놓음 | 상자 ±1 cm 안에서 끝난 grip 뒤 |

동작: **S3 → held (attach)**, 그 밖 → not held (detach → SEARCHING) — `--oracle-latch` 와 같은 길 (`safe_policy._oracle_latch_post`);
HELD 중 S3 이면 evidence 의 놓기를 취소; **CLOSING 중 S1 → SEARCHING** (빈 닫힘 풀기).
입력: (a) hidden — attention pass 뒤 prefix pass 1 번 더 (gemma `return_hidden`), **h_10** 의 [agentview 평균 | wrist 평균 | valid-text 평균] 6144 값,
(b) GRIP 10 — 손가락 간격 · 직전 간격 · 차 · 측정 개도 · **그림자 latch** 상태 one-hot (지금 규칙 그대로 도는 별도 latch, feature 전용) · 마지막 실행 gripper 행.
분류기: train PCA 128 + StandardScaler + multinomial logistic (C 0.3), dev 104 ep 로 학습.

## 2. 위치

| 조각 | 위치 |
|---|---|
| branch · commit | `wip/stage-track` (base d452c804): e8ddf8b7 (P5b 구현), 7a140dbc (P6 queue), 이 문서의 commit |
| 모듈 | `AG3S/trajopt/stage_track.py` |
| asset | `AG3S/ag3s/asset/stage_track/h10_grip_v1.npz` (md5 67c2883c38de86ec3d0a542735755c8b) |
| 서버 연결 | `AG3S/trajopt/attention_policy.py` (`stage_hidden=True`), `AG3S/trajopt/safe_policy.py` (`stage_track`, `_stage_track_pre/_post`, 그림자 latch), `AG3S/adapters/panda/server.py` (`load_stage_track`, 응답 chunk 기억), `AG3S/trajopt/serve_safe.py` (`--stage-track`, RB-Y1 거절) |
| 판 | `AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458-st1` (= pe123458 + `--stage-track default`), 짝 `…-pe123458-ol` (= + `--oracle-latch`) — `profile.STAGE_TRACK_VERSIONS` |
| test | `tests/benchmark/test_p5b_stage_track.py` (14 개: off 불변 · 정의 · asset · override · 판) |
| 실험 script | `experiments/segmentation/stage_track/`: `export_contacts.sh` (P1) · `p2_latch_boundary.py` · `p3_extract_hidden.py` (GPU) · `p4_feature_compare.py` · `p5a_estimator.py` · `p5b_build_asset.py` · `p5b_gpu_check.py` (GPU) · `p6_queue.sh` (GPU) · `p6_compare.py` · `finger_bodies_replay.py` |
| 결과 · 그림 (git 제외) | `outputs/2026-11-10_제약단계_추종/` (`p2/` … `p6/pe123458st1/`) |

## 3. 다시 돌리는 법

1. branch 를 받는다: `git worktree add .claude/worktrees/stage-track wip/stage-track` (또는 merge 된 branch). worktree 면 submodule (`benchmark/LIBERO*`, `benchmark/SafeLIBERO`, `third_party/curobo`, `third_party/aloha`) · venv (`.venv-*`, `.python`, `.runtime`) · `.cache/openpi` 를 주 checkout 으로 symlink 하고 `AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage` 를 준다 (2026-11-11 LS run 1 개가 `.venv-libero-safety` 누락으로 실패).
2. CPU test: `.venv-ag3s/bin/python -m pytest -q tests/benchmark/test_p5b_stage_track.py` (14 통과).
3. GPU 확인 (공유 lock, 약 5 분): `python3 scripts/setup/gpu_lock.py shared --est-gib 24 --tag st-check -- .venv-openpi-live/bin/python experiments/segmentation/stage_track/p5b_gpu_check.py outputs/2026-11-10_제약단계_추종` — 서버 경로 h_10 = 학습 feature (상대 오차 < 1e-2).
4. closed-loop (독점, launcher 가 run 마다 lock): `V=<판> TAG=<tag> bash experiments/segmentation/stage_track/p6_queue.sh` (평가 LIBERO 27 + SafeLIBERO 11 + LIBERO-Safety 10, 48 ep, run 9 개).
5. 비교: `.venv-ag3s/bin/python experiments/segmentation/stage_track/p6_compare.py <tag> pe123458` → `outputs/…/p6/<tag>/p6_compare.json` (채택 규칙 자동 판정).
6. **기계 · 구성이 바뀌었으면** 비교 기준 판 (pe123458 또는 그때의 구성) 도 같은 기계에서 다시 돌린다 (XLA 차이로 chunk 가 바뀜).
7. asset 은 pe123458 의 기록으로 학습했다. **구성이 바뀌면 latch 상태 분포 · 궤적이 달라지므로** P1–P5b (접촉 추출 → 정답 → hidden 추출 → asset) 를 그 구성의 dev 기록으로 다시 만드는 것을 권함 [I].

## 4. 오프라인 결과 [M] (dev 104 ep · 3604 요청, leave-one-group-out)

| | S3 놓침 | 잘못된 HELD | 못 잡은 빈 닫힘 (E-fail) | 오류 합 |
|---|---|---|---|---|
| 지금 latch (pe123458) | 197 | **67** | 44 | 308 |
| GRIP + ACT (기준) | 145 | 168 | 3 | 316 |
| V cache 단독 (auto 층) | 139 | 109 | 16 | 264 |
| layer hidden 단독 (auto 층) | 78 | 190 | 14 | 282 |
| **hidden h_10 + GRIP (ST1)** | **69** | 125 | **6** | **200** |
| 같은 + HMM 평활 | 112 | 117 | 10 | 239 |

- 사전 채택 규칙 (오류 합 ≤ 0.7 × GRIP+ACT, 거짓 HELD 안 늘어남) 은 hidden + GRIP 만 통과. 지금 쓰는 kvL4 (V 층 4) 는 340 으로 GRIP 단독과 같음.
- 잘못된 HELD 125 중 86 은 E1a 2 ep 에서 **로봇이 실제로 다른 물체를 쥔 것** (장애물 머그 · 다른 그릇, MuJoCo 재생) → 단계가 아니라 목표 식별 오류. 2 ep 제외 시 latch 7 vs ST1 39.
- 서버 경로 검증: 서버 h_10 = 학습 feature (상대 오차 0.0, 106 요청), 서버 규칙으로 만든 GRIP 의 asset 예측 일치 1.000, 요청당 추가 29 ms (closed-loop 중앙값 46 ms).

## 5. closed-loop 결과 [M] (P6, 2026-11-10 23:20 – 2026-11-11, 같은 기계 · 같은 seed, 비교 기준 = pe123458 기존 기록)

| 세트 | pe123458 | ST1 | 사전 규칙 |
|---|---|---|---|
| 평가 LIBERO 27 성공 | 22 | **21** | ≥ 22 ✗ |
| SafeLIBERO 11 성공 · 안전 성공 · 충돌 · > 50 mm | 8 · 6 · 4 · 1 | 8 · **5** · 5 · **3** | 안전 성공 ≥ 6 ✗, 큰 충돌 안 늘어남 ✗ |
| LIBERO-Safety 10 성공 (위반) | 5 (0) | 5 (0) | ≥ 5 · 위반 0 ✓ |

**판정: 미채택** (사전 규칙 6 중 3 미달: SafeLIBERO 안전 성공 · 큰 충돌, LIBERO 27). P6 은 2026-11-10 23:20 – 2026-11-11 03:41 (LS human_safety 는 첫 시도가 worktree 의 `.venv-libero-safety` link 누락으로 실패해 다시 돌림). 비교 원본 `outputs/2026-11-10_제약단계_추종/p6/pe123458st1/p6_compare.json`.

바뀐 episode [M]:
- 좋아짐: LIBERO s8 t9 (목적지 등록 지연 실패 → 성공), **s9 t1** (첫 held call 에 쥔 그릇이 장애물 → 성공; ST1 이 노린 기제), SafeLIBERO t3 e3 (와인병 296 mm → 안전 성공), t3 e2 (4.8 mm → 안전 성공), t2 e1 (시간 초과 → 성공, 4.7 mm).
- LIBERO-Safety (학습에 안 쓴 세트): human_safety e0 (지금: 쥔 뒤 기하 미인증 HOLD → 시간 초과) · obstacle_avoidance e3 이 성공, human_safety e1 · obstacle_avoidance e1 이 시간 초과 — 2 ↔ 2 맞바꿈.
- 나빠짐: LIBERO s7 t6 · s9 t4 · s9 t7 (시간 초과), SafeLIBERO t1 e4 (새 101.7 mm 충돌), t2 e2 (11.7 → 258.9 mm + 시간 초과), t2 e4 (시간 초과), t3 e4 (4.5 mm).

실패 형태 [D] (call 기록):
1. **깜빡임 detach** — 운반 중 한 요청만 S3 아님 → 즉시 detach → S1 로 CLOSING 해제 → 다시 닫기 반복 (s7 t6, s9 t4, SafeLIBERO t2 e2 는 약 15 번, t2 e4). 다시 잡으려는 움직임이 장애물 충돌과 시간 초과로 이어짐.
2. **이른 attach** — 지금 latch 보다 2–3 call 일찍 HELD (s9 t7: 그 뒤 36 call 내내 HELD 인데 못 끝냄; 실제로 쥐었는지는 정답 미확인).
3. 오프라인 채점은 요청 하나의 오류를 1 로 세지만, closed-loop 에서는 요청 하나의 detach 가 쥔 물체를 장애물로 되돌리고 다시 잡기를 부른다 → **비대칭 비용** [I].

## 6. 다음 변형 후보 (결과 전에 규칙을 고정하고 돌릴 것)

| 이름 | 바꿀 것 | 노리는 실패 |
|---|---|---|
| **ST2** | 상태를 바꾸려면 같은 판정이 **2 요청 연속** (attach 도 detach 도), CLOSING 해제도 S1 2 연속 | 깜빡임 detach · 이른 attach |
| ST3 | ST1 은 **S1 의 CLOSING 해제만** 쓰고 attach / detach 는 지금 latch 그대로 | 빈 닫힘 미감지만 고치기 (가장 보수적) |
| ST4 | attach 는 지금 latch, detach 만 ST1 (2 연속) | 놓기 · 미끄러짐 감지 |
| 정답 latch (`…-ol`) | latch 를 MuJoCo 정답으로 | 단계 추종으로 얻을 수 있는 상한 — 이것도 pe123458 과 차이가 없으면 단계 추종의 여지가 없다 |

오프라인으로 먼저 거를 수 있다: `p5a_proba.npz` 의 확률에 연속 판정 규칙을 씌워 요청 단위 오류와 "상태 전환 횟수" (깜빡임) 를 같이 센다 (CPU 수 분).

## 7. 갱신 기록

| 시각 (KST) | 내용 |
|---|---|
| 2026-11-11 03:5x | 처음 작성 (dev-55). P6 끝, 미채택. |
