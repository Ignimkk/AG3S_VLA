# 다른 서버에서 이어 하기 — 목표 · 현재 상태 · 다음 작업 (작성 2026-11-09 16:1x KST, dev-4a)

> 새 서버에서 `git pull` (branch `feature/panda-benchmark`) 뒤 **이 문서를 가장 먼저 읽는다.** 이 서버의 Claude memory 와
> `outputs/` (gitignore) 는 옮겨지지 않는다 → 필요한 규칙 · 기록은 모두 이 문서와 아래 링크 (git 안) 에 옮겨 두었다.
> 자료 유형: **MEASURED** (기록된 run 의 값) · **DERIVED** (기록에서 계산) · **PLANNED** (아직 안 함).

## 0. 새 세션에 붙여 넣을 프롬프트

```
/mnt/dev/AG3S_VLA (branch feature/panda-benchmark) 의 doc/active/NEXT_SERVER_HANDOFF_20261109_KO.md 를 읽고 이어서 진행해 주세요.
한국어로 답하고 기술 용어는 English 로 씁니다. CLAUDE.md 규칙 A (figure · 표 · graph) · B (날짜 · 시각이 있는 기록) · C 를 지킵니다.
먼저 §3 새 서버 준비를 확인하고 결과를 보고한 뒤, §4 의 다음 작업을 순서대로 진행합니다.
단계가 끝날 때마다 commit · 저장하고 다음 단계로 넘어갑니다 (commit 은 GPU queue 사이에만).
```

## 1. 목표

### 1.1 프로젝트 목적 (CLAUDE.md)
VLA (π0.5) 의 visual attention 으로 **작업 대상 (target) 과 장애물 (obstacle) 을 구분**하고, target 을 충돌 대상에서 뺀 3D 환경을
cuRobo TSDF / ESDF 로 넘겨 **집기 · 놓기는 그대로 하면서 장애물 충돌만 피하는** AG3S 를 Panda (LIBERO 계열) 에서 완성한다.

### 1.2 지금 풀고 있는 문제 — GT 없는 target 인식 (SL-PE)
- 상한 (GT 인식) 구성 **B5** (`AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve-fh-t100`, 사용자 판정 G101) 는
  target · 목적지 · 쥔 물체 · 장애물을 시뮬레이터 GT 로 받는다. 논문 주장은 **GT 없이** (attention + 자기 3D 인식) 해야 한다.
- 사용자 관찰 (2026-11-09 02:06): "장애물은 잘 피하는데 **집으려다 못 집고, 잘 집어도 놓을 때 못 놓는다** — target · 접시를 장애물로 보는가?"
  → 맞았다. SL-PE1–PE11 은 모두 이 문제 (조작 대상 · 목적지 · 쥔 물체가 장애물로 남는 것) 의 대응이다.

### 1.3 수치 목표 (판정 기준)
| 항목 | 목표 | 근거 |
|---|---|---|
| E1a Spatial LI 40 의 SSR (안전 성공) | **> 23** (multi-link CBF, privileged GT) | dev-38 E1a 표 · 원고 Tab. 1 |
| LIBERO 27 성공 | VLA 단독 27 에 가깝게 (B5 GT 27) | 집기 · 놓기 능력 유지 |
| SafeLIBERO 11 | 안전 성공 ↑, **큰 충돌 (> 50 mm) 0** | G163 |
| LIBERO-Safety 10 | 성공 ↑, 위반 0 | |
| 후보 채택 기준 (G163, 결과 전에 고정) | **큰 충돌 0 · SafeLIBERO 11 안전 성공 ≥ 지금 구성** · 평가 규약 전체를 같은 기계에서 | 결과를 보고 기준을 바꾸지 않는다 |

### 1.4 논문 최종 실험과의 연결
기준 문서 `paper/evidence/AG3S-VLA_CVPR2027_FINAL_EXPERIMENT_PLAN_20261108_KO.md` (S0 동결 → E1a → E5 → E2/E3 → E4-F → E6 → E1b → E7).
지금 작업은 **S0 (설정 동결) 직전 단계**: GT 없는 최종 구성을 고르는 중. 미리 모은 자료 (실패 분류 · run time · Fig. 1 후보) 는
`experiments/segmentation/sl_pe_records/2026-11-09_최종실험_자료수집/README.md`.
**분할 규칙 (사용자 판정 G171):** E1a init 5–14 = 개발에 쓴 분할 (후보 비교용). **동결 구성의 headline 은 init 15–49** (아무 arm 도 안 본 분할).

## 2. 현재 상태 (2026-11-09 16:1x)

### 2.1 지금 구성 (최종 후보) — `AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458`
전체 구성 (flag 78 개 실효 값 · 흐름 · 역할별 인식 · 구현 위치) 은 [`doc/active/experiments/AG3S_FINAL_CONFIG_KO.md`](experiments/AG3S_FINAL_CONFIG_KO.md).
구성이 바뀌면 본문 전체를 고치고 맨 아래 '교체 기록' 에 한 줄 (사용자 지시).

| (MEASURED, 이전 서버) | pe123458 (지금) | pe123458-tro (미채택) | gt24att (GT 없는 이전) | B5 (GT 상한) | VLA 단독 |
|---|---|---|---|---|---|
| 보정 LIBERO s10 · s11 (18) | 14 | 13 | — | 17 | — |
| 평가 LIBERO 27 | 22 | 23 | 17 | 27 | 27 |
| SafeLIBERO 11 성공 · 안전 성공 · 충돌 | 8 · 6 · 4 | 7 · 5 · 4 | 4 · 2 · 6 | 9 · 8 · 2 | — |
| LIBERO-Safety 10 | 5 (위반 0) | 5 | 3 | 9 | — |
| E1a 40 (init 5–14) TSR · CAR · SSR | 22 · 23 · **15** | 24 · 20 · 17 | 9 · 16 · 5 | 22 · 21 · 13 (B5-tro 27 · 20 · 20) | 25 · 6 · 6 |

### 2.2 해결한 문제
| 문제 | 대응 (flag, 기본 꺼짐) | 결과 |
|---|---|---|
| 쥐기 전: identity 가 접시 · 다른 그릇 | **PE1** 손 닫는 지점 아래 물체 = identity (+ PE2–PE5: 받침면 · 잠금 · 행 판정 · 목적지 등록) | LIBERO 기제 확인 2 → 8 / 12 |
| 쥔 채 옮기기 · 놓기: 쥔 그릇이 장애물로 남음 | **PE8** 손과 같이 움직인 점을 장애물에서 뺌 (`--scene-obstacle-held-motion`) | 거절 call 28 → 3 → 지금 구성 |
| 장애물이 대상 그릇에 닿으면 한 덩어리로 묶여 장애물에서 빠짐 (E1a 충돌 17 중 7) | **PE11** 대상을 점 단위로 뺌 (`--scene-obstacle-target-cut`) | offline 39 / 39 · dev 127 / 127 유지, closed loop 에서 머그 277 mm · 책 6 mm 충돌 제거 (G173) |

### 2.3 남은 문제 (우선순위)
1. **PE11 단독 판 (`…-pe12345811`) 의 평가 규약 미실행** — 다음 작업 T1.
2. **장애물 행 허용치 τ_obs 10 mm 안 통과로 생기는 작은 접촉** (E1a 충돌 5 ep, tro 에서 1–8 mm 접촉 증가) → τ_obs 5 mm 후보.
3. **목적지 (접시) 등록 지연** — 쥔 뒤 2–16 call 늦게 등록돼 놓기 직전 거절 (s8 t9) → PE10 후보.
4. **PE8 첫 held call 틈** — 손이 움직이기 전 첫 call 에 쥔 그릇이 장애물 (s9 t1 회귀).
5. **서랍 t4** (평가 s7 · s8 실패) — 첫 call 기하 미인증 HOLD + 접근 identity. PE11 'cut' 은 서랍 누출로 t4 를 더 나쁘게 함 (G173).
6. **PE9 (attention 최고점 seed 컷) 조합의 와인병 787 mm 충돌** — PE11 로 안 풀림 (PE7 · PE9 원인), PE9 는 미채택 유지.
7. 무충돌 실패의 내역 (E1a): 놓기 실패 4 · HOLD 시간 초과 2 · 못 쥠 2 (e4f 라벨).

판정 기록 전체: [`doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md`](experiments/E1_GTTO_DECISIONS_20261104_KO.md) G150–G173 (SL-PE = dev-4a), G102–G112 · G169 (SL-QE = dev-38). **다음 판정 번호 G174.**

## 3. 새 서버 준비 (체크리스트)

| # | 할 일 | 확인 |
|---|---|---|
| 1 | `git pull` → `git log --oneline -1` 이 이 문서를 넣은 commit 이상인지 | `git status` 깨끗 |
| 2 | **checkout 경로:** queue script 들이 `/mnt/dev/AG3S_VLA` 를 고정해서 씀 (`pe1k_eval_queue.sh` · `gtto16_queue.sh` · `pe2b_queue.sh` · `e1a_headline_queue.sh`). 다른 경로면 같은 경로에 두거나 symlink | `ls /mnt/dev/AG3S_VLA` |
| 3 | **storage** `AG3S_VLA-storage` (checkout 옆): `checkpoints/panda/pi05_libero`, `pi05_libero_safety` (24 GB), `benchmarks/libero-safety` (30 GB). git · Drive 에 없음 → 원래 출처 · 설치 절차는 [`doc/active/SHARED_STORAGE_AND_PYTHON.md`](SHARED_STORAGE_AND_PYTHON.md) | `ls $AG3S_STORAGE_ROOT/checkpoints/panda` |
| 4 | **venv** `.venv-ag3s` · `.venv-openpi-live` · `.venv-curobo` 독립 설치 (복사 금지): 같은 문서 'Interpreter와 독립 재설치' 절, live 검증 [`migration/LIVE_ENVIRONMENT_IMPLEMENTATION_REPORT.md`](migration/LIVE_ENVIRONMENT_IMPLEMENTATION_REPORT.md) | `source scripts/setup/env.sh` |
| 5 | **LIBERO-Safety client 의 MagickWand** (Ubuntu 24.04): `.runtime/magickwand/lib` (`.so.7`, gitignore) 필요 — queue 가 `LD_LIBRARY_PATH` 에 붙임. Drive `AG3S_VLA_backup/archive/checkout_runtime.tar.gz` (올라갔다면) 를 checkout 에서 풀거나 다시 만든다 | LS 10 smoke |
| 6 | **이전 서버 기록 받기 (선택):** rclone (`apt-get install -y rclone` → `rclone config`, remote 이름 `gdrive`, scope `drive.file`) → 아래 §6.3 으로 `outputs/` · run 기록 (npz 제외) 복원. 비교 script (`tro_compare.py` 등) 가 이전 run 의 `episodes.jsonl` 을 읽을 때 필요 | `rclone lsd gdrive:AG3S_VLA_backup/archive` |
| 7 | **회귀 test** (CPU, 약 4 분): `tests/ag3s/test_sl_pe8_held_motion.py test_sl_pe9_attention_peak.py test_sl_pe11_target_cut.py test_sl_pe_wiring.py test_sl_pe1_close_seed.py tests/trajopt/test_sl_qe2_translation_only.py tests/benchmark/test_sl_gtto2_versions.py test_sl_gtto_gt_perception.py` (2026-11-09 15:4x 이전 서버 79 통과) | 전부 통과 |
| 8 | **GPU 확인:** `nvidia-smi --query-compute-apps=pid,process_name --format=csv` 비어 있음, `/tmp/ag3s-vla-gpu-validation.lock` 이 다른 session 에 안 잡혀 있음 | |

**기계 차이 주의 (중요):** 기계가 바뀌면 XLA kernel 차이로 π0.5 chunk 자체가 달라져 closed-loop 결과가 바뀐다 (2026-11-03 H200 이전 때 확인).
→ **이전 서버 수치와 새 서버 수치를 짝으로 비교하지 않는다.** 새 서버에서 지금 구성 (pe123458) 을 **anchor 로 다시 돌리고**
후보를 그 anchor 와 비교한다 (T0). run-id 는 새 TAG (예: `…n2`) 로 — 비교 script 는 run-id 마다 가장 최근 run 을 쓴다.

## 4.0 서버 역할 · baseline 선행 실행 (사용자 결정 2026-11-09 16:5x–17:xx)

> **갱신 (사용자 결정 2026-11-10 16:2x KST): 새 서버는 없다.** 아래 T0–T6 도 이 H200 (`/mnt/dev/AG3S_VLA`) 에서 baseline queue
> 사이에 끼워 넣어 수행한다 (GPU lock, 바깥 flock 금지, queue 가 도는 동안 HEAD 불변). 따라서 §3 의 '기계 차이' 주의와
> T0 (새 서버 anchor) 는 해당 없음 — 11-09 의 pe123458 run 이 같은 기계 anchor 다. 같은 서버에서 제약 단계 추종
> (hidden state + gripper, `outputs/2026-11-10_제약단계_추종/README.md`) 도 진행 중이며, T3 (목적지 등록 지연) · T4 (첫 held call)
> 와 같은 S2→S3 경계를 다룬다.

- **이전 서버 (H200 1 장, `/mnt/dev/AG3S_VLA`) = baseline 전용.** 2026-11-09 17:xx 부터 `experiments/segmentation/bl_tranche_queue.sh` (tranche 1 = init 15–18,
  SafeLIBERO 7 arm × 128 ep + LIBERO 4 arm × 160 ep, 약 25 h) 실행 중. 이 run 들은 **논문 표에 들어가는 실제 결과**. 계획 · port 근거 · 결과 표:
  [`experiments/BASELINE_TRANCHE_PLAN_PORT_RATIONALE_20261109_KO.md`](experiments/BASELINE_TRANCHE_PLAN_PORT_RATIONALE_20261109_KO.md).
  queue 가 도는 동안 **이전 서버의 HEAD 를 바꾸지 않는다** (git pull · commit 금지). 끝나면 결과를 그 문서 §5 에 채우고 commit · push.
- **새 서버 = AG3S 개발** (아래 T0–T4). 개발 비교는 개발 분할 (init 0–14) 에서만.
- **같은 기계 조건:** 표의 모든 arm 은 같은 기계 → **동결한 AG3S 의 보고 run (T5) 은 이전 서버에서** 돌리거나, 먼저 두 서버의 π0.5 단독 출력이 같은지
  확인한다 (같은 init · seed 의 VLA 단독 몇 ep 의 action 비교).
- 규모: 계획의 **1/10 (tranche 1) 먼저**, 나머지 9/10 은 나중에 `INITS=… T=t2 …` 로 더하고 같은 문서 · 표를 고친다 (사용자 결정).
- AEGIS 는 공식 코드 (GLM-4.5V API) 대신 **port 판** (비용 결정). port 가 원 논문 보고값보다 낮게 나오는 점 (AEGIS CAR, KNOWS 전반) 은 위 문서 §4.3 —
  표에 보고값 (†) 을 같이 두고 주장 범위를 한정한다. KNOWS port 는 9/10 전에 구현 점검 권장.

## 4. 다음 작업 (순서대로, 각 단계 끝에 commit · 저장)

| # | 작업 | 명령 (checkout 에서) | 시간 (이전 서버 실측) | 판정 · 산출 |
|---|---|---|---|---|
| **T0** | 새 서버 anchor: 지금 구성 pe123458 평가 규약 + E1a 개발 분할 | `V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458 TAG=pe123458n2 bash experiments/segmentation/pe2b_queue.sh` | 약 3.5 h | 이전 서버 값 (§2.1) 과 크게 다르면 기계 차이 기록 (판정 G174) |
| **T1** | **PE11 단독 판** 평가 (보정 → LIBERO 27 → SafeLIBERO 11 → LS 10 → E1a 5–14) | `V=AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811 TAG=pe12345811n2 bash experiments/segmentation/pe2b_queue.sh` | 약 3.5 h | G163 기준으로 T0 와 짝 비교 → 채택이면 최종 구성 문서 본문 + 교체 기록. 비교: `sl_pe_records/…/eval_pe123458tro/tro_compare.py` 의 패턴을 TAG 만 바꿔 사용 |
| T2 | τ_obs 5 mm 후보 (E1a 작은 접촉 대응) | profile 에 판 등록: (T1 채택 판) + `("--passthrough-obstacle-tolerance-mm", "5")` (argparse 뒤 값이 이김), 이름 `…-to5`. 회귀 test 뒤 `pe2b_queue.sh` | 3.5 h | 위험 = 장애물 행 거절 증가로 HOLD · 시간 초과 → LIBERO 27 · LS 10 하락 여부 |
| T3 | PE10 목적지 등록 지연 | 설계 메모: 쥔 동안 손 밖 attention 1 위의 3 frame 연속 조건 완화 또는 등록 형상 누적. **offline 채점 먼저** (목적지 점 기록이 chunk 에 없음 → replay 필요). 도구 `sl_pe_records/2026-11-09_집기놓기_장애물취급/dest_latency.py` | 설계 1 일 | 위험 = 장애물이 목적지로 등록 (SafeLIBERO) |
| T4 | PE8 첫 held call 틈 · 서랍 t4 | attach 직전 closing 의 manipulated 형상 + 손 frame 으로 첫 call 보완 (손끝 경계 필수). t4 는 첫 call 기하 미인증 HOLD 3–8 + 접근 identity — PE6 (`--no-target-certify`) 는 미채택 (G162–G163) | | |
| **T5** | **구성 동결 (S0)** → 보고 run: baseline 과 같은 init (tranche 1 = 15–18, 이후 19–49) · 같은 서버 (이전 서버, §4.0) | `V=<동결 판> TAG=<tag> VLA=1 bash experiments/segmentation/e1a_headline_queue.sh` | AG3S 140 ep ≈ 4–5 h + VLA | 원고 Tab. 1 의 AG3S 행. baseline (AEGIS · CBF · KNOWS) 15–49 는 dev-38 (SL-QE) 과 같은 기계에서 |
| T6 | 최종 계획의 자료 다시 뽑기 (동결 판으로) | `sl_pe_records/2026-11-09_최종실험_자료수집/` 의 `e4f_labels.py <tag>` (E4-F 실패 분류) · `timing_summary.py` (E6 · Tab. S11, **같은 날 · 같은 lock 으로 baseline 과 함께 재측정**) · `paired_candidates.py <tag>` (Fig. 1 · Video 후보) | 각 수 분 (CPU) | 원고 반영은 dev-4d (paper/) 담당 |

T0 와 T1 은 하나의 queue 로 이어 돌려도 된다 (두 번째를 첫 번째 queue log 의 'queue pe2b end' 뒤에 시작). 단, **사용자 규칙: 단계 끝마다 commit 후 다음 단계** — 자동으로 이어 붙이기 전에 사용자에게 확인.

## 5. 운영 규칙 (이전 서버 memory 에서 옮김)

- **commit:** 사용자가 요청할 때 / 단계가 끝날 때마다. **GPU queue 가 도는 동안 HEAD 를 바꾸지 않는다** (실행 중 run 의 `record_execution` 이 거절됨 — 2026-11-09 02:07 실제로 run 1 개 무효).
  git identity 는 config 를 바꾸지 말고 env 로: `GIT_AUTHOR_NAME=kmingi98 GIT_AUTHOR_EMAIL=kmingi98@gmail.com GIT_COMMITTER_NAME=kmingi98 GIT_COMMITTER_EMAIL=kmingi98@gmail.com`.
  끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **GPU lock:** `launch_libero_paired.py` · `launch_libero_safety_paired.py` 는 `/tmp/ag3s-vla-gpu-validation.lock` 을 **스스로** 잡는다.
  바깥에서 `flock` 으로 감싸면 deadlock (2026-11-07, 1.5 h 손실) — 감쌀 때는 반드시 `export AG3S_GPU_LOCK_HELD=1`.
- **같은 checkout 을 쓰는 peer session:** `AG3S/` · `benchmark/runners/` 수정 전에 알림, 여러 파일 patch 는 tmp 파일 → `os.replace`
  (반쯤 적용된 `serve_safe.py` 가 peer 서버를 죽인 적 있음). **실행 중인 bash queue script 는 고치지 않는다** (bash 가 이어 읽음) —
  새 queue 가 이전 queue 의 'end' 줄을 기다리게 한다. 멈출 때 `pkill -f` 금지 (자기 shell 까지 죽음) → pid 로.
- **자원:** CPU worker ≤ 2 (pod 전체 ≤ 4), 측정 job 은 nohup · detach 없이. 긴 GPU 작업 전후 peer 에 알림.
- **담당:** `paper/` = 논문 session (dev-4d) — 연구 session 은 고치지 않는다. 연구 자료는 MEASURED / DERIVED / CANDIDATE / PLANNED 표시.
  SL-QE (회전 고정 action QP, baseline port, SOTA 표) = dev-38, 판정 G102–. SL-PE (GT 없는 인식) = dev-4a, 판정 G150–.
- **SafeLIBERO 'collision'** 은 장애물 1 mm 이동 규칙이라 실제 로봇 접촉이 없을 때도 있다 → 실제 접촉은 `gtto_collision_contacts` 로 채점.
- **index 규약:** `calls.jsonl` 의 `obs_index` = wait 20 step 포함 env step; SafeLIBERO `collide_t` = env step; chunk npz `t_step` = call 순번, episode 경계는 `t_step` 가 0 으로 돌아오는 곳.
- 사람이 볼 결과 · 그림은 `outputs/<YYYY-MM-DD>_<설명>/` (git 제외) → 단계 끝에 Drive backup.

## 6. 자료 위치

### 6.1 git 안
| 무엇 | 위치 |
|---|---|
| 전체 구성 · 수치 · 교체 기록 | `doc/active/experiments/AG3S_FINAL_CONFIG_KO.md` |
| 판정 기록 | `doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md` (G150–G173) |
| 판 정의 (모두 기본 꺼짐) | `AG3S/adapters/panda/profile.py` — `PE_VERSIONS` (PE 계열), `QE2_VERSIONS` (`-tro`) |
| PE 구현 | `AG3S/ag3s/runtime/pipeline.py` (`_scene_obstacle_points`), `held_motion.py`, `grasp_latch.py`, `config.py`, `AG3S/trajopt/serve_safe.py` |
| queue | `experiments/segmentation/pe1k_eval_queue.sh` (보정 + `gtto16_queue.sh` 평가 규약), `pe2b_queue.sh` (+ E1a 5–14), `e1a_headline_queue.sh` (15–49) |
| 분석 script · 보고서 사본 (outputs 에서 복사, 2026-11-09 16:1x) | `experiments/segmentation/sl_pe_records/` — `2026-11-09_집기놓기_장애물취급/` (자율작업_보고서.md, e1a_decomp/, eval_pe123458tro/ tro_compare · pe11_mech, pe_rowowner.py 등), `2026-11-09_최종실험_자료수집/`, `2026-11-08_PE_경로끝점_identity/` (py · md) — 경로는 이전 서버 기준 (`/mnt/dev/AG3S_VLA-storage/...`) |
| 최종 실험 계획 | `paper/evidence/AG3S-VLA_CVPR2027_FINAL_EXPERIMENT_PLAN_20261108_KO.md` |
| Drive backup script | `scripts/setup/gdrive_backup.sh` |

### 6.2 git 밖 (이전 서버) → Google Drive `AG3S_VLA_backup/`
- `archive/outputs/<폴더>.tar.gz` (outputs 44 폴더), `archive/results_benchmark/<run>.tar.gz` (run 기록 364 개, **npz 제외** — 사용자 판정 2026-11-09:
  episode · chunk npz 는 backup 안 함 → offline replay 분석 (`e1a_decomp`, `pe11_pointdrop` 등) 은 새 서버에서 새 run 으로만 가능),
  `logs/` (backup log · manifest = 묶음 안 파일 목록).
- 2026-11-09 15:47 시작한 업로드는 rclone 공용 client_id 의 할당량 (403 rateLimitExceeded) 때문에 일부 실패 → 이전 서버에서
  `bash scripts/setup/gdrive_backup.sh` 를 다시 돌리면 실패 · 미완료 폴더만 다시 올라감. **빠르게 하려면 rclone 에 자기 Google client_id 를 만들어 넣는다**
  (rclone 문서 'Making your own client_id', `rclone config` → edit `gdrive` → client_id / client_secret).
- `AG3S_VLA_backup/outputs/` 의 낱개 파일 354 개는 처음 방식 (파일 단위) 의 잔여 — archive 와 중복, 지워도 됨.

### 6.3 복원 (새 서버)
```bash
cd /mnt/dev/AG3S_VLA && mkdir -p outputs && cd outputs
rclone copy gdrive:AG3S_VLA_backup/archive/outputs . && for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done
mkdir -p /mnt/dev/AG3S_VLA-storage/results/old_server && cd /mnt/dev/AG3S_VLA-storage/results/old_server   # 새 run 과 섞지 않음
rclone copy gdrive:AG3S_VLA_backup/archive/results_benchmark . && for f in *.tar.gz; do tar xzf "$f" && rm "$f"; done
```
이전 서버 run 을 `results/benchmark/` 에 바로 풀면 run-id 가 같은 새 run 과 섞여 비교 script 가 기계가 다른 run 을 고를 수 있다 → 별도 폴더에 둔다.

## 7. 이번 세션 기록 (2026-11-09)
| 시각 | 일 | 결과 |
|---|---|---|
| 02:06–11:00 | 자율 작업: PE8 · PE9 · PE11 설계 · 평가 | `sl_pe_records/2026-11-09_집기놓기_장애물취급/자율작업_보고서.md` |
| 12:1x | commit 7ff8bc0d · 827bc5e3, E1a 실패 분해 → PE11 | G170 |
| 13:0x | 분할 판정 (headline = 15–49) | G171 |
| 12:10–15:11 | pe123458-tro 평가 규약 + E1a | **미채택** (G172) |
| 15:11–15:36 | PE11 기제 확인 (pe1234578911) | 노린 충돌 제거, PE9 와인병 남음 (G173) |
| 15:4x–16:1x | commit afe5b285 · 0dbcddf4 · bb4ce8c8 (dev-38 qe*) · 21cf17ba (dev-4d paper), pe2b 시작 → 사용자 지시로 멈춤 (`INVALID-aborted_…`), Drive backup, 이 문서 | |
