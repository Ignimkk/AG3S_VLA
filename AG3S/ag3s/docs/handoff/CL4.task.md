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
| **X1** | extent | ③ LTE = `exclude-authorized --target-permission-strict --nonauthorized-target-margin obstacle` | ball (기본 `--target-exclusion ball`) + grounding split `--attention-split-radius-mm 12 --attention-split-ratio-source raw` (X2 – G4 와 같게) | V11AB 의 extent |
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

## 실행 준비 기록 (lead, 2026-10-14)
- 준비 (verifier): `verify/CL4/p/` 스크립트, CPU dry run 으로 5 arm 의 flag 가 모두 받아들여지고 1,349 구를 짓는 것을 확인, guard 양성 5 · 음성 10 (다른 arm 의 log) 확인, 분석 도구를 CL3 기록으로 시험 (기존 결과 재현).
- **lead 정정 (실행 전):** 처음 준비본은 X1 에 grounding split flag (12 mm · raw) 가 없어 X1 → X2 가 extent 와 split 두 가지를 바꿨다. 사용자 판정 "LTE 고정, extent 만 변경" 에 맞춰 X1 에도 같은 split 을 넣었다 (`env.sh`, X1 guard). 다시 dry run · guard 시험 통과. 이 split 은 attention 후보 재군집 설정이며 extent 가 아니다.
- G4 에서 `--target-permission-strict` · `--nonauthorized-target-margin obstacle` 은 받아들여지지만 exclude-all 에서는 비권한 행이 target 을 보지 않아 효과가 거의 없다 (X4 와 정책 외 flag 를 같게 둔 결과).
- X4 · G4 의 초기 chunk 는 oracle 표본이 쌓이기 전이라 ball_owned 로 돈다 (설계).
- 예상 시간 갱신: 4.0 – 4.5 h (+ cuRobo 기준선 약 3 min). run 수는 90 그대로.
- **STARTUP 확인 (2026-10-14 15:48 – 16:33 UTC):** 첫 시도는 run dir 안의 시험용 symlink (CL3 를 가리킴) 를 출력 경로 검사가 거절해 5 서버 모두 기동 전에 종료 → symlink 제거 (시험 결과는 `prep/test_cl3` 에 남음). 두 번째: X1 · G4 는 모든 guard 통과, X2 · X3 · X4 는 900 s 안에 listening 에 이르지 못함. 최종 형상 서버의 기동이 10 – 15 min 걸렸다 (CL3 는 76 s; 두 서버가 동시에 기동하면 더 늦다). 실패 기록은 `startup_check_failed_symlink/` · `startup_check/` 에 보존.
- **일정 v2 (lead):** 기동 시간 때문에 arm × port 마다 서버를 한 번만 띄운다 — port 8290 = pair 1 – 9 를 X1 X2 X3 X4 G4 순서, port 8291 = pair 10 – 18 을 G4 X4 X3 X2 X1 순서 (같은 pair 는 모든 arm 에서 같은 port), 8291 은 8 min 늦게 시작. `SERVER_UP_TIMEOUT_S` 2400 s. X2 · X3 · X4 의 기동 guard 는 본 실행의 첫 기동에서 같은 검사로 확인한다 (불일치면 그 chain 이 거절).
- **본 실행 시작 16:34 UTC.** 예상 종료 ≈ 21:30 UTC (≈ 4.9 h; 서버 기동 10 회 × 10 – 15 min 포함). 승인 예산 (약 4 h) 보다 약 1 h 길다 — run 수는 90 그대로, 추가분은 서버 기동 시간이다.

## 사용자 판정 (2026-10-15, CL4 결과 뒤)
1. **성공률 하락 원인 진단 최우선.** offline: CL3 와 CL4 의 같은 episode · seed 에서 접근 궤적이 처음 갈리는 구간, 정책 원출력 대 TO 출력, 영향을 준 link · 제약, 손–target 거리 · 그리퍼 닫힘 시점. 대표 실패 전환 · 성공 유지 사례부터. 기록만으로 형상의 영향을 가르지 못하면 **X2 에서 형상만 기존 것으로 바꾼 18 run (X2G)** 을 재승인 없이 진행 (snapshot · margin · W1 · seed 그대로, SL5b · T57 변경 섞지 않음). palm · wrist 각각의 기여는 분리하지 않고, 성공이 회복돼도 누락이 알려진 기존 형상을 최종 채택하지 않는다. GPU 시간은 CL4 의 실제 기동 비용을 반영해 재산정.
2. **DT-02 기록 (고정 문구):** "최종 형상의 18개 dev episode–seed 조건에서 success_ct(P)는 ball 10, oracle-visible 9, ball_owned+F1 7, ball_owned 5였다. 이 조건에서는 ball_owned와 F1의 ball 대비 실행상 우위가 관측되지 않았다." 순서는 관측값의 순서로만 쓰고 방법의 일반적 우열로 쓰지 않는다. 18 = 9 scene × 2 seed. F1 은 X2 대비 5 → 7 의 탐색적 회복, ball 보다 낫다는 근거 없음. 이전 CL3 의 채택 판단은 그 형상 · 설정에 한정.
3. **DT-05A:** "이 평가 범위에서는 link 별 구분의 실행상 차이가 관측되지 않았다." X4 · G4 의 성공 수 차이를 LTE 효과로 귀속하지 않는다. 놓은 뒤의 비허용 접촉은 기존 정의대로 유지하고 FT–target 접촉과 구분해 보고.
4. **T60d 종료:** "TO-off 에서 P 에 도달한 3 run 은 모두 H 에 도달했다. 따라서 TO 가 이전 궤적과 상태를 바꿔 복귀에 간접적으로 영향을 주었을 가능성을 배제할 수 없다." "관측된 복귀 구간에서 정책의 복귀 제안이 없었다" 는 유지, "TO 는 복귀 실패에 기여하지 않았다" 로 확대하지 않는다. 추가 seed 없음.

## X2G 실행 (2026-10-15)
- 기록 진단 (`verify/CL4_diag/`): CL4 X2 · X1 은 파지 전 구간 (t ≤ 160) 에서 손바닥 (`ee_left` / `ee_right`, palm cover 구) 이 TO 의 최악 행 (target-free 층) 이 된 chunk 가 18 쌍 합 109 · 95, CL3 C1a 는 21. 그 손바닥 구의 다수가 AG3S target 점 15 mm 안 (예: X2 ep11_s111 24/33, ep104_s1041 59/69). 같은 형상의 X1 도 손바닥 구속이 많은데 성공은 높아 결과 귀속은 기록만으로 가르지 못함 → X2G 진행.
- X2G = X2 + `--links gripper` (270 구, palm cover off) 외 동일. 18 run, 서버 2, 같은 pair 분할. 재산정 GPU: 기존 형상 서버 기동 (CL3 76 s, 부하 시 수 분) + 9 run × ≈ 6 – 6.5 min ≈ 1 – 1.2 h. GPU lock 03:01 UTC, pod 메모리 guard 대기 후 시작.
