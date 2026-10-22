# SL5i-G — 파지 인식 수정판 · 파지 인식 + closing 면제판 closed-loop 검증 (GPU run 측정)

> writer: lead · 읽는 쪽: verifier · 작성 2026-10-22 KST
> 앞 단계: `SL5i.task.md`(무엇을 왜 바꿨나), `SL5i.impl.md`(구현 · test · offline latch replay)
> 실행 방식: `benchmark/runners/run_parallel_gpu.sh`, 동시 실행 ≤ 2, 역변환 C(`lag_inverse_tracking`) 고정.

## 이름 (사용자 2026-10-22: 기록과 보고는 무엇을 바꿨는지로 부른다)

| 내부 key | 이름 | 비교 기준 |
|---|---|---|
| `V11AB-pb5` | grasp zone 허용판 (직전 최신판) | SL5g run |
| `V11AB-pb6` | **파지 인식 수정판** | grasp zone 허용판 + 같은 닫힘 시도 안에서 reach 유지 + 열린 개도 기준 dropped |
| `V11AB-pb7` | **파지 인식 + closing 면제판** | 파지 인식 수정판 + closing 중 쥔 물체에 대한 손 제약 제외 |

## run (`/mnt/dev/AG3S_VLA-storage/results/benchmark/`, 이름에 `sl5i-` 포함)

pb6-C · pb7-C × {SafeLIBERO LI dev 20, LIBERO 10}, 4 run.

| 비교 기준 | run |
|---|---|
| VLA-only (SL5f) | `20261017T151126_…_sl5f-safelibero-LI-dev-LI-vla_only`, `20261017T170230_…_sl5f-libero10-init0-vla_only` |
| grasp zone 허용판 (SL5g pb5-C) | `20261018T133339_…_V11AB-pb5-C`, `20261018T150153_…_V11AB-pb5-C` |
| 역변환 A · B (SL5h-G) | 참고용 |

VLA-only를 재사용하는 근거는 SL5g의 동일성 점검이다. 이번 변경은 `SafePolicy` 쪽뿐이고 `--no-safe` 경로는 바뀌지 않았다. 확인 방법은 SL5g의 `vla_only_identity.py`와 같은 방식으로 changed files를 확인하는 것이다.

## 무엇을 재나 (→ verifier)

| # | 측정 |
|---|---|
| I0 | **기존 동작 확인.** 각 run manifest의 commit · dirty · server argv(새 flag가 들어갔는지) · conversion을 본다. VLA-only 재사용 근거(changed files 목록, `--no-safe` 경로 무변경)도 확인한다. |
| I1 | **집계.** VLA-only · grasp zone 허용판 · 파지 인식 수정판 · 파지 인식 + closing 면제판의 official / collided / safe / HOLD. 전이표를 만들고 후보 규칙 (a) official F→S, (b) safe F→T를 따로 센다. |
| I2 | **수정 규칙이 실제로 적용됐나.** sticky reach가 쓰인 call, drop-from-open 기준이 쓰인 call, `closing_exempt_rows`. 처음 held가 되는 seq를 MuJoCo 실제 lift 시점과 비교한다. 거짓 held(양손가락이 target에 닿지 않았는데 held)를 센다. offline replay(`outputs/verify/SL5i-pre/`)가 예측한 seq와도 대조한다. |
| I3 | **SL5h 서명.** SL5h와 같은 정의로 잰다: over-lift 수, lifted 실행 포화율, closing·lifted call 수, closing·lifted row 위/아래, AG3S centroid − 실제 bowl 거리(held 이후). SL5h · SL5h-G script를 재사용한다. |
| I4 | **실패 원인 분류.** VLA-only 성공 → 실패 episode에 SL5g 분류(A no-lift / B lost / C over-lift / D other)를 적용한다. held에 도달하지 못한 episode는 실패한 check를 함께 적는다. 특히 LI t1e2(`blocked`)와 t2e0(다른 cluster를 잠금)의 결과를 본다. |
| I5 | **closing 면제의 부작용.** 면제 조건이 섰는데 양손가락이 target에 닿지 않은 call(replay: LI 47, LIBERO 16)에서 충돌이나 target 밀기가 생겼는지 본다. |
| I6 | **사례.** 규칙 (a) · (b) 후보, 그리고 grasp zone 허용판에서 over-lift로 실패했는데 새 판에서 성공한 episode를 묶는다. SL5g 형식을 따르고 `outputs/verify/SL5i-G/cases/`에 둔다. |

- 판정 문장은 쓰지 않는다. 가설마다 지지 · 반박하는 숫자만 둔다.
- figure: `AG3S/ag3s/docs/figures/sl5i-*` + `.json`. 실제 scene(같은 episode의 이전판 vs 새 판, 쥔 순간과 최고점), graph(version별 bar, held 시점 vs lift 시점), table.
- 산출물: raw는 `outputs/verify/SL5i-G/`, verify.json은 `AG3S/ag3s/docs/handoff/SL5i-G.verify.json`.

## 규칙

- code 수정 · commit · GPU 사용 금지.
- CPU worker ≤ 2, RSS ≤ 3 GB. working set이 62 GiB를 넘으면 멈춘다.
