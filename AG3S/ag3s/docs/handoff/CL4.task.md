# CL4 — extent 비교 (DT-02) + 접촉 정책 비교 (DT-05A) closed-loop, 최종 형상 · 고정 snapshot

> writer: lead (세션 A) · 2026-10-14 KST · 읽는 쪽: 실행 (A), verifier, scribe.
> 근거: 사용자 판정 (2026-10-14) "CL4 — 90 run · 약 4 시간의 예산을 승인하며, ② global exclusion 을 포함하세요. extent 비교: LTE 를 고정하고 extent 만 변경. 접촉 정책 비교: extent 하나를 고정하고 ② global 과 ③ LTE 를 비교. 전체 factorial 은 필요 없음. 같은 최종 형상 · 초기 상태 · seed · TO 설정. 이 구성과 예산 안이면 재승인 없이 진행."
> develope_PLAN: DX-01 조건 B (고정 identity 의 extent 비교 → **DT-02 extent block**, DF-03B), DX-04 (→ **DT-05A**, DF-04B). §0.10.1-2 · 4.
> 형상 · 정책 정의: [`T46f.task.md`](T46f.task.md) §2 · §3 · 사용자 판정 (C3 · 최종 형상), oracle: [`T46e.task.md`](T46e.task.md).

## 1. 고정 (모든 arm 공통, 결과 전에 고정)

| 항목 | 값 |
|---|---|
| code | **snapshot = `git archive 150d429b`** → `results/development/20261014T231714+0900_150d429b86ff_17b0cec4fb97/snapshot` (서빙 code = `8df34591` 과 같음; SL5b · T57 수정 미포함). 실행 중 메인 tree 변경과 무관 |
| 형상 | `--links gripper_wrist --palm-cover on` (1,349 구, T46f 최종 형상) |
| TO | CL3 와 같은 V11AB TA · TP flag (`--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval --esdf-host-mirror`) + CL3 서버 공통 flag (`--obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both`, ESDF curobo 0.020 / 0.005) |
| W1 | `--placed-no-seed-certify` (모든 arm, T46c 판정) |
| identity | AG3S 서빙 L8H2 그대로 (모든 arm 같음) |
| 장면 · seed | CL3 의 18 쌍: ep 0 · 3 · 11 · 12 · 30 · 54 · 104 · 110 · 123 (train split) × seed {ep·10+1, ep·10+2} (ep0 은 1 · 2). client `--warmup-steps 1000 --max-steps 600`, `XLA_FLAGS=--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0` |
| 평가 접촉 정의 | T46f §2 "평가용 접촉 정의 (세 조건 공통)" 그대로 — 허용 = active hand `EE_BODY_*` · 손가락 geom, 비허용 = 그 밖의 로봇 geom (FT_SENSOR 포함). 조건 · 서버 정책과 무관 |

## 2. arm (5 × 18 = 90 run)

| arm | 블록 | target field policy | extent | 비고 |
|---|---|---|---|---|
| **X1** | extent | ③ LTE = `exclude-authorized --target-permission-strict --nonauthorized-target-margin obstacle` | ball (기본 `--target-exclusion ball`) | V11AB 의 extent |
| **X2** | extent | ③ | ball_owned (`--target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw`) | C1a extent |
| **X3** | extent | ③ | ball_owned + F1 (`--nontarget-evidence surface`) | 채택한 extent 방법 (T46c) |
| **X4** | extent · 정책 | ③ | oracle-visible (X2 bundle + `--target-extent oracle-visible`, client `--safe-oracle-seg`) | 관측 기반 비교 기준선 (상한 아님) |
| **G4** | 정책 | ② global = `exclude-all` (그 밖의 flag 는 X4 와 같음) | oracle-visible (X4 와 같음) | DT-05A 의 ② |

- **extent 비교 = X1 · X2 · X3 · X4** (정책 ③ 고정). **접촉 정책 비교 = X4 대 G4** (extent oracle 고정). 전체 factorial 은 하지 않는다.
- ① none 은 oracle 과 함께 쓸 수 없고 (A14c 거절 규칙) 이번 범위가 아니다.

## 3. 지표 (T60 판정: P 주 · H 필수, T46f §7)

- 주: P (안정 placement) 까지의 **비허용 접촉 없는 성공** (`success_ct`). 함께: P 성공, H 도달, 비허용 접촉 class 별 (hand–fruit_other · robot_other–target · robot_other–fruit_other) run 수 · 지속 row, **비허용 link (FT_SENSOR · arm_5 · arm_6 · 기타) 와 target 의 접촉 · 최소 거리** (T46f-S2 의 checked 거리 계산), HOLD chunk (kind 별), TO 실패, P · H 까지의 시간, 실측 TO 시간.
- 쌍 비교: 같은 (episode, seed) 의 McNemar (실패→성공 / 성공→실패 쌍 수). 비교 쌍: X1–X2, X2–X3, X3–X4 (extent), X4–G4 (정책).
- 보조: LTE ≠ global 판정 chunk 수 (메커니즘 설명용).
- 바닥에 떨어진 target 이 base · 바퀴에 닿는 접촉도 `robot_other–target` 으로 잡힌다 (T46f-S2 에서 관측) — 접촉 body 를 함께 적는다.

## 4. 해석 제한 (사전 고정)

- 정지 (HOLD · TO 실패) 가 늘어 접촉이 줄어든 것은 보호 효과로 해석하지 않는다.
- X4 대 G4 에 차이가 없으면 **"이 평가 범위에서는 link 별 구분의 실행상 차이가 관측되지 않았다"** 로만 쓴다. 일반적인 동등성이나 LTE 무효를 주장하지 않는다 (T46f-S2: 이 장면들에서 비허용 link 의 target 접근이 드물다).
- oracle 은 상한이 아니다 (T46e).
- 18 쌍 · 9 episode (train split) 는 dev 수준이다. 통계적 유의성을 새 진입 조건으로 쓰지 않는다.

## 5. 실행 · 진행 / 중단

- GPU 순서 (사용자 판정): 변형 배치 pilot → **CL4** → T60d. 서버 ≤ 2 (port 8290 · 8291), GPU lock 을 실행 전체 동안 잡는다. 시작 · 끝을 B · benchmark 세션에 알린다.
- 실행 전 arm 마다 STARTUP_ONLY 서버 기동 확인 (flag · 형상 1,349 구 · palm cover · 정책 · oracle · W1 로그 줄).
- **중단:** infra 실패 (server crash · 기록 불완전 · import 경로가 snapshot 밖) 가 run 의 10 % 를 넘으면 멈추고 보고한다. 결과 내용으로는 멈추지 않는다.
- 예산: 90 run, 예상 GPU wall ≈ 3.5 – 4 h (CL3 실측 run 당 ≈ 4.6 min, 서버 2 병렬, 형상 비용 증가분 포함 추정). 초과가 예상되면 추가분만 사용자에게 알린다.
- 산출: `verify/CL4/` (runs · replay · analysis), `handoff/CL4.verify.json`, `figures/cl4/` (실제 장면 · 그래프 · 표).
