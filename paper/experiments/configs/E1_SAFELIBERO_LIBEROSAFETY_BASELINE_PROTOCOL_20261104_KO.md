# E1 baseline R 수집 protocol — SafeLIBERO · LIBERO-Safety (계획, 결과 전 고정 초안)

- 작성: 2026-11-04 19:50 KST, benchmark checkout `feature/panda-benchmark` `89b42459` (dirty: BL-13 기록 · nb1104 스크립트).
- 상위 계획: [AG3S-VLA_CVPR2027_Experimental_Design.md](../../evidence/AG3S-VLA_CVPR2027_Experimental_Design.md) §4 (E1), §7.2 (E4 downstream). 결과 기록장: [AG3S-VLA_EXPERIMENT_RESULTS_KO.md](../../evidence/AG3S-VLA_EXPERIMENT_RESULTS_KO.md) §3.
- 상태: **`PLANNED`**. 실행 승인 아님. 아래 `TBD(사용자)` 항목은 사용자 판정 전 값이다. 이 문서의 명령은 **초안**이다.
- AG3S 수치는 넣지 않는다(계획서 §1). AG3S 행은 최종 구현 동결 뒤 같은 protocol로 채운다.

## 0. 근거로 확인한 사실 (2026-11-04)

| 항목 | 확인 내용 | 근거 |
|---|---|---|
| 공식 SafeLIBERO 규모 | 4 suite × 4 task × Level I/II × 50 episode = 1,600 / 방법. π0.5-LIBERO, Panda OSC_POSE 20 Hz | VLSA PDF p.5, `benchmark/SafeLIBERO/README` |
| harness 지원 | `safelibero_{spatial,object,goal,long}` × I/II, task 0–3, init 50 개 (32 init file 모두 50 state) | `benchmark/configs/suites.json`, `benchmark/runners/suite_registry.py:50-60` |
| 공식 horizon | `main_aegis.py`: wait 20 step 뒤 `t = 0`(l.234) → `while t < max_steps`(l.254), max_steps 300 / Long 550 → **policy 300 / 550 step + wait 20** | `benchmark/SafeLIBERO/main/main_aegis.py:79-85,166-254` |
| harness budget | `--safelibero-step-budget main_aegis` = 위 공식과 같다. 기본값 `total_incl_wait`는 policy 280 / 530 step (nb1104 · BL-7 · SL5c dev가 쓴 값) | `benchmark/runners/libero_safe_client.py:45-69` |
| 충돌 판정 | 작업 공간 안 active obstacle의 변위 > 1 mm (공식 README 규칙). 움직이지 않는 접촉은 세지 않는다 | `libero_safe_client.py` `SAFELIBERO_COLLISION_DISPLACEMENT` |
| CSC (접촉 sim step 평균) | episode 기록에 접촉 step 수가 없다 → **현재 산출 불가** | `episodes.jsonl` field 확인 (nb1104 none-LI) |
| episode 시간 (H200, 2 job 동시) | VLA 단독 ≈ 57 s, filter arm ≈ 64–95 s (spatial, policy 280). 서버 기동 ≈ 40 s / job | nb1104 run `episodes.jsonl` `started_at`/`ended_at` |
| LIBERO-Safety 공식 π0.5 | LIBERO-Safety 데모로 30k step fine-tune, 3 training seed × 10 trial / task. 공개 weight `LIBERO-Safety/pi05_libero_safety`(HF, 2026-03-20, 1개, openpi config 없음) | LIBERO-Safety PDF p.10–11, 35–36 |
| 이 기계의 LIBERO-Safety 상태 | `pi05_libero`만 있음. `pi05_libero_safety` 없음(H1: openpi TrainConfig 없음). fork controller kp 750 · ±2 vs 우리 변환 kp 150 (H2). 공개 `info['cost']`가 robot 접촉을 못 잡는 문제(B1) | `AG3S/ag3s/docs/handoff/BM-SETUP-B.impl.md:17-19,54-70,148-160` |
| 외부 학습 기반 방법 | SafeDojo(OpenVLA-OFT + Wan2.2 14B, 코드 "coming soon"), SafeVLA(AI2-THOR 전용), WMPO(OpenVLA-OFT, SafeLIBERO checkpoint 없음) → π0.5 harness에서 실행 불가 | 각 project page · repo (2026-11-04 확인) |
| AEGIS 공식 코드 | 공개(2026-02-19). GLM-4.5V API로 장애물 이름 → GDINO → ellipsoid CBF-QP 매 env step. 우리 port는 GLM 대신 privileged 이름(`aegis:gt`) 또는 generic prompt, chunk 단위 filter | `Baseline/paper_baselines/methods/aegis/README.md`, `closed_loop/README.md` |
| 같은 TO 통제 baseline | (a) generic geometry: 기존 flag(`--target-field-policy none` 또는 `--no-attention`)로 Panda version 추가만 필요. (b) external segmentation · oracle: AG3S core에 `oracle-visible` code-image 경로가 있으나 Panda 서버가 거절하고(`server.py:364,439-443`) client가 mask를 보내지 않는다 → **구현 필요** | `AG3S/trajopt/serve_safe.py:2388,2489`, `AG3S/ag3s/runtime/pipeline.py:735,761,2634` |

## 1. 기존 기록과 계획서 표의 대응

계획서 §4.2 표는 **동일 evaluator · test split의 R만** 받는다. 아래 기존 기록은 모두 dev(init 0–4)이거나 다른 budget이라 **E1 R 칸에 넣지 않는다**. pilot · 개발 근거로만 인용한다.

| 기록 | 범위 | E1 표에 쓰나 | 이유 |
|---|---|---|---|
| BL-7 (10-24) | spatial LI dev 20 ep × 8 arm, budget 280 | 아니오 | BL-8 이전 구현, 이전 기계 |
| BL-11 (11-03) | spatial LI dev 20 ep, KNOWS · multi-link × det/GT | 아니오 | dev, 이전 기계, raw 없음 |
| **BL-13 (11-04, 이 기계)** | spatial LI · LII dev 20 ep × 7 baseline arm, budget 280 | 아니오 (pilot) | dev split, policy 280 step (공식 300 아님), spatial만 |
| SL7 D13 (다른 서버) | LIBERO-Safety L0 3 suite × task 0–1 × init 0–4, `pi05_libero` + 변환 | 아니오 | `official_comparable: False`. 이 기계의 같은 표본 재실행은 10/30 (D13 14/30, conversion B vs C) |
| nb1104 LIBERO-Safety L0–L2 | VLA 단독 10 ep 씩 | 아니오 | 같은 이유. L2 obstacle_avoidance 공식 위반 6/10 은 후보 조건 선정 근거로만 |

## 2. E1a — SafeLIBERO baseline (지금 실행 가능)

### 2.1 질문
같은 frozen π0.5-LIBERO와 같은 SafeLIBERO evaluator에서, 추론 시 안전 방법(우리 port)이 TSR · SSR · CAR · ETS를 어떻게 바꾸는가? 장애물 입력이 detector일 때와 privileged GT일 때를 분리한다.

### 2.2 arm (모두 `launch_method.py`, 원 방법과 다른 점은 `our reimplementation`으로 표시)

| 우선 | arm | 장애물 입력 | 표기 |
|---|---|---|---|
| 1 | π₀.₅ 단독 (`none`) | — | R |
| 1 | AEGIS port (`aegis`) | GDINO tiny, generic prompt | R, our reimplementation (GLM 단계 대체) |
| 1 | AEGIS port (`aegis --gt`) | privileged 장애물 이름 → GDINO | R, our reimplementation, **privileged** |
| 1 | multi-link CBF (`multilink_cbf`, `--gt`) | detector / privileged GT | R, our reimplementation |
| 2 | KNOWS port (`knows`, `--gt`) | detector / privileged GT | R, our reimplementation |
| 3 | OmniGuide · CFM · NUDGE · Lan-o3dp | — | R, our reimplementation (BL-7 이후 미측정) |
| — | SafeDojo · SafeVLA · WMPO · VLPSA · DetAug | — | **P only** (실행 불가, 사유는 §0) |

### 2.3 고정 조건
- checkpoint `pi05_libero` (`AG3S_VLA-storage/checkpoints/panda/pi05_libero`), config `pi05_libero`, seed 7, `XLA_FLAGS=--xla_gpu_enable_command_buffer=`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **budget `--safelibero-step-budget main_aegis`** (공식 300 / 550 policy step + wait 20). dev 기록(280)과 섞지 않는다.
- 충돌: 공식 1 mm 변위 규칙. 충돌해도 episode는 계속한다(공식과 같음).
- 같은 (suite, level, task, init, seed)를 모든 arm에 쓴다 → paired.
- 실패 · timeout · QP 실패(nominal 반환) · server crash를 버리지 않는다. crash로 끝난 episode는 `UNKNOWN`으로 남기고 같은 seed로 1회만 재실행하며 재실행 사실을 기록한다.
- 동시 실행: baseline arm은 TO가 없으므로 공유 lock에서 2 job까지 허용(기존 규칙). latency 열은 보고하지 않는다(2 job 공유 시간이므로).

### 2.4 표본 — `TBD(사용자)`

| 안 | init | episode / arm | VLA 단독 추정 시간* | 우선 1 arm 5개 추정* | 비고 |
|---|---|---:|---:|---:|---|
| A (권장 1단계) | test 5–14 | 4 suite × 4 task × 2 level × 10 = **320** | ≈ 3 h | ≈ 17–20 h | 계획서 실험 F의 test 범위(5–14)와 같다 |
| B (공식 전체) | 0–49 | **1,600** | ≈ 15 h | ≈ 85–100 h | 원문 P와 같은 규모. dev init 0–4 포함 → test 5–49(1,440)를 별도 집계 |
| C (spatial 우선) | test 5–14, spatial만 | 80 | ≈ 0.8 h | ≈ 4–5 h | 4 suite 평균인 P 표와는 직접 대응 안 됨 |

\* **추정(ESTIMATED)**: nb1104 spatial 실측 57 s/ep(2 job 동시) × policy 300/280 × Long 550/300 가중, filter arm × 1.1–1.7. 실측 후 갱신한다.

### 2.5 지표와 집계
- TSR = official success / N, CAR = 충돌 없음 / N, **SSR = (success ∧ 충돌 없음) / N** (episode 단위로 계산, TSR·CAR에서 유도하지 않음), ETS = 전체 episode 평균 env step(wait 제외, 실패는 budget까지). SBU(성공 ∧ 충돌) · CNF(무충돌 ∧ 미완료)를 함께 보고.
- CSC: **산출 불가(UNKNOWN)** — 접촉 step 기록이 없다. 넣으려면 client에 기록 flag를 추가해야 한다(기본 꺼짐, `TBD(사용자)`).
- 분자/분모를 suite × level × task 셀마다 보존. 4 suite 평균은 suite 동일 가중(원문과 같은 방식인지 실행 전 원문 Table 정의로 재확인).
- 불확실성: 비율은 Wilson 95% CI. arm 간 차이는 같은 episode의 paired 차이에 대해 **scenario(suite × level × task) cluster bootstrap** 2,000회(seed 20261104) 95% CI, 보조로 exact McNemar. 검정 다중성 보정은 하지 않고 모든 비교를 나열한다(탐색적 표시).
- 성능 유지 · 비열등 주장은 하지 않는다(margin 미정).

### 2.6 명령 초안 (실행하지 않음)

```bash
# 공통: source scripts/setup/env.sh; 공용 lock 은 queue runner(experiments/segmentation/nb1104_queue.sh 형식)가 한 번 잡는다
S=safelibero_spatial   # object / goal / long 반복
for L in I II; do
  # registry 정식 label (예: test_SafeLIBERO_Spatial_ep5-49_user_2026-10-14) → episode가 test 범위 밖이면 launcher가 거절한다
  SPLIT=$(.venv-ag3s/bin/python -c "from benchmark.runners import suite_registry as s; print(s.split_label('$S','$L','test'))")
  .venv-ag3s/bin/python Baseline/paper_baselines/closed_loop/launch_method.py --method none \
    --task-suite-name $S --safety-level $L --task-ids 0,1,2,3 --episode-ids 5,6,7,8,9,10,11,12,13,14 \
    --seed 7 --data-split "$SPLIT" --safelibero-step-budget main_aegis --sidecar-device cuda \
    --run-id e1a-none-${S#safelibero_}-L$L
  .venv-ag3s/bin/python Baseline/paper_baselines/closed_loop/launch_method.py --method aegis --gt \
    --task-suite-name $S --safety-level $L --task-ids 0,1,2,3 --episode-ids 5,6,7,8,9,10,11,12,13,14 \
    --seed 7 --data-split "$SPLIT" --safelibero-step-budget main_aegis --sidecar-device cuda \
    --ref-run /mnt/dev/AG3S_VLA-storage/results/benchmark/<e1a-none-...-L$L 의 절대 경로> \
    --run-id e1a-aegis-gt-${S#safelibero_}-L$L
done
```

2026-11-04 19:5x KST `--dry-run` 확인(CPU): spatial · long Level II, `aegis --gt`, `main_aegis` → rc 0, `policy_steps` 300 / 550, loop `t < max_steps + wait` = 320 / 570. 정식 split label로 goal LI `multilink_cbf --gt` dry-run rc 0, init 3을 섞으면 `episodes [3] are not in the test split`로 거절됨을 확인. 실행 전 남은 확인: (2) `--ref-run`은 절대 경로(11-04 rc=1 사고 재발 방지), (3) Long · Goal · Object의 `--sidecar-device cuda` GDINO prompt가 suite별로 정상인지 BL-8 suite_smoke 기록과 대조, (4) `main_aegis` budget에서 1 episode smoke.

## 3. E1b — LIBERO-Safety baseline (전제 작업 필요)

- 원문 Table 3과 비교 가능한 R을 만들려면 **H1**(`pi05_libero_safety` 12.4 GB 다운로드 + openpi TrainConfig 작성), **H2**(fork controller kp 750 · ±2에 맞는 action 처리 확인), **B1**(공식 cost가 robot 접촉을 못 세는 문제를 공식 그대로 쓸지 / 보조 contact 판정을 따로 보고할지) 판정이 필요하다. 셋 다 `TBD(사용자)`.
- 원문 Table 3의 ± 는 3 training seed 분산이다. 공개 weight는 1개라 **seed 분산은 재현할 수 없다**. R은 단일 weight × 평가 seed로 표기한다.
- 표본(전제 해결 후): 물리 4 suite(AAG=affordance, HRI=human_safety, TSA=obstacle_avoidance, FSHOA=obstacle_avoidance_human) × L0–L2 × 5 task × 10 trial = 600 / arm. horizon은 repo evaluator `max_steps: 600`(원문 미기재 → 확인 필요).
- 그 전까지 `pi05_libero` + 변환 run은 `official_comparable: False` 진단으로만 기록한다(원문 zero-shot π0.5 Table A.10 SR 0.7 %와도 조건이 다르다).

## 4. E4 / E1 통제 baseline (같은 TO) — AG3S TO 동결 뒤

- "generic geometry + fixed TO"와 "external segmentation + fixed TO"는 **AG3S와 같은 TO**를 써야 한다. AG3S 최종 구성은 다른 GPU 서버에서 SL9(T96 수정안 ablation)로 아직 바뀌는 중이므로, 지금 측정하면 TO가 바뀔 때 다시 재야 한다 → **AG3S 구성 동결 뒤 실행**을 권장한다.
- 준비는 지금 가능: (a) generic geometry = `AG3S/adapters/panda/profile.py`에 version 1개(base flags − target exclusion · surface · extent 계열 + `--target-field-policy none`). (b) · oracle = client의 GT/외부 code image 전송 flag + Panda 서버의 `oracle-visible` 허용 + `manipulated_geometry` override(ATS 오인 lock이 결과를 흐리지 않게). 모두 기본 꺼짐.
- **충돌 주의:** `profile.py`는 다른 서버의 SL9가 version을 추가할 파일이다. 이쪽에서 수정하기 전에 그쪽 commit을 먼저 받거나 작업 분담을 정한다.
- TO가 들어가는 run은 독점 lock, 한 번에 1 job(시간 예산 공유 방지, p95 latency 열 보고 가능).

## 5. 원자료 · provenance 요건

run마다: storage `results/benchmark/<날짜_commit_run-id>/`의 `manifest.json`(argv · commit · checkpoint path), `episodes.jsonl`(episode 단위 success · collision · 변위 · step), video. 집계 script와 입력 hash를 결과 옆에 저장하고 [결과 기록장](../../evidence/AG3S-VLA_EXPERIMENT_RESULTS_KO.md) §10 양식으로 실행 ID를 등록한다.
**manifest schema 주의:** `paper/experiments/configs/manifest.schema.json`은 RB-Y1 16-D 전용(`action_dimension` const 16)이라 Panda 7-D SafeLIBERO run을 등록할 수 없다. 별도 Panda manifest 형식이 필요하다(`TBD`).

## 6. 문헌 표 점검 (원문 PDF 대조, 2026-11-04)

- SafeDojo Table 1–2, LIBERO-Safety Table 3 L2: 계획서 · SOTA 문서 값 모두 PDF와 일치.
- `SOTA_COMPARISON_KO.md` §1 표에 **Level II WMPO 행(55.75 / 46.12 / 277.17)이 빠져 있다**(계획서와 결과 기록장에는 있음).
- VLSA Table I(p.6)의 full-action OpenVLA-OFT 평균 CAR 5.7 %는 같은 표 suite 행 평균(11.3 %)과 맞지 않는다 → 인용하지 않는다.
- 같은 이름의 AEGIS 수치가 세 조건으로 존재한다(SafeDojo의 OFT 재실행 · VLSA 원문 π0.5 · VLPSA의 VLSA* 재실행). 섞지 않는다.
- 후보 추가 P: **VLPSA** (arXiv 2609.22462, π0.5 SafeLIBERO, 코드 없음) — 가장 가까운 π0.5 비교 대상. 원문 확인 후 P 표에 넣을지 `TBD(사용자)`.

## 7. 미정 항목 (사용자 판정)

1. E1a 표본 안 A / B / C 와 arm 우선순위.
2. CSC 기록 flag 추가 여부.
3. E1b 전제(H1 다운로드 · config, H2, B1 처리).
4. 통제 baseline(§4) 준비를 지금 할지, AG3S 동결 뒤 할지, 그리고 `profile.py` 작업 분담.
5. VLPSA 등 신규 문헌 P 등록 여부.
