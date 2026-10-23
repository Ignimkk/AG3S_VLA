# 평가 recipe — 새 method를 version으로 등록하고 LIBERO · SafeLIBERO 전체에서 돌리기

> 작성 BM-SETUP-A (2026-10-23, ag3s-implementer). 대상 checkout `feature/panda-benchmark`.
> 모든 명령은 checkout root에서 `source scripts/setup/env.sh` 뒤에 실행한다.
> GPU를 쓰는 단계(3)는 공용 GPU 규칙(`CLAUDE.md`: `/tmp/ag3s-vla-gpu-validation.lock`, 다른 compute process 없음,
> 같은 pod의 다른 세션에 알림)을 따른다. 1 · 2 · 4 단계는 CPU만 쓴다.

## 한눈에

```
(1) profile.py AG3S_VERSIONS 에 key 추가  ──>  (2) eval_matrix.py plan  ──>  jobs.txt + plan.json
                                                      │  (--vla-only reuse: pairing_basis.json)
                                                      v
                                   (3) run_parallel_gpu.sh --jobs jobs.txt   (GPU, 최대 2 job 동시)
                                                      │
                                                      v
                                   (4) eval_matrix.py aggregate --plan plan.json  ──>  aggregate.json / .md
```

| 파일 | 역할 |
|---|---|
| `benchmark/configs/suites.json` | suite별 protocol 표: steps · wait · split · 성공 / 안전 지표 정의 · task 이름 · 공개 기준 수치. 출처 줄 번호 포함 |
| `benchmark/runners/suite_registry.py` | 위 표를 읽는 helper(split label, 선택 검사). launcher가 GPU 작업 전에 task / episode id를 검사한다 |
| `benchmark/runners/launch_libero_paired.py` | 한 job = server + client. VLA-only와 version을 같은 task · init · seed로 돈다 |
| `benchmark/runners/run_parallel_gpu.sh` | jobs 파일을 최대 2개씩 동시에 돈다. GPU lock은 이 script가 잡는다 |
| `benchmark/runners/eval_matrix.py` | jobs 파일 생성(`plan`, `smoke`), VLA-only 재사용 판정, 결과 집계(`aggregate`) |
| `benchmark/evaluators/suite_probe.py` | policy 없는 CPU 점검(env 생성 · 관측 규약 · 충돌 판정). 새 suite / asset을 붙일 때 먼저 돌린다 |

## 1. 새 method를 version으로 등록

1. method가 `serve_safe` flag로 켜지게 구현한다(기본값 off). flag 없이 켜지는 변경은 기존 version의 결과를 바꾼다.
2. `AG3S/adapters/panda/profile.py`의 `AG3S_VERSIONS`에 key를 추가한다. 값은 V11AB argv 뒤에 붙는 flag tuple이다.
   ```python
   "V11AB-pb8": ("--reach-at-capture", ..., "--my-new-flag"),
   ```
   주석에 무엇을 바꿨는지와 근거 STEP을 적는다. 보고서에는 key 대신 바뀐 내용을 이름으로 쓴다(사용자 지시 2026-10-22).
3. CPU 확인:
   ```bash
   .venv-ag3s/bin/python -c "from AG3S.adapters.panda import profile as p; a=p.parse_v11ab_args(argv=p.v11ab_server_argv()+p.version_server_flags('V11AB-pb8')); print(p.match_ag3s_version(a))"
   .venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py --task-ids 0 --episode-ids 0 --run-id x \
       --conversion C --ag3s-version V11AB-pb8 --dry-run
   ```
   `match_ag3s_version`이 새 key를 돌려주고 dry-run server argv 끝에 새 flag가 보이면 된다.

## 2. jobs 만들기 (`plan`)

```bash
.venv-ag3s/bin/python benchmark/runners/eval_matrix.py plan \
    --version V11AB-pb8 --conversion C \
    --suites all \            # all | libero | safelibero | libero_goal,safelibero_long:II ...
    --split dev \             # dev = init/episode 0-4, test = 5-49 (아래 split 표)
    --run-prefix bm-pb8 \
    --out-dir outputs/eval/bm-pb8-dev
```

- 한 job = 한 suite × level의 task들(`--tasks-per-job N`으로 나눌 수 있다) × split의 episode 전부.
  job마다 `--modes vla_only,vla_v11ab`이면 launcher가 `paired_summary.json`을 쓴다.
- `--tasks 0,1` / `--episodes 0,1`로 일부만 고를 수 있다. episode는 split 안에 있어야 한다.
- `--safelibero-step-budget main_aegis`: SafeLIBERO를 공식 `main_aegis.py`대로(아래 "step budget") 돈다. 기본은 SL5c protocol.
- `--launcher-extra "--record-constraints occupancy"`: 모든 job에 launcher 인자를 덧붙인다(그 인자의 제약은 launcher가 검사한다).
- 출력: `jobs.txt`(`run_parallel_gpu.sh` 입력), `plan.json`(job · 예상 run 이름 · 시간 상한).
- 시간 상한은 모든 episode가 step 상한까지 간다고 보고 계산한다(SL5d–SL5i spatial run의 step당 시간 중앙값,
  `outputs/verify/BM-SETUP-A/timing_basis.json`). 성공 episode는 일찍 끝나므로 실제는 더 짧다.

### VLA-only 재사용 (`--vla-only reuse`)

```bash
.venv-ag3s/bin/python benchmark/runners/eval_matrix.py plan ... --vla-only reuse \
    --vla-only-runs $AG3S_STORAGE_ROOT/results/benchmark/<VLA-only run> ... \
    [--accept-diff <path> ...]
```

episode마다 아래를 **모두** 만족하는 VLA-only run이 있을 때만 재사용한다(SL5g pairing basis, 사용자 규칙 2026-10-18).
한 job의 episode가 전부 덮이면 그 job은 `--modes vla_v11ab`만 돈다. 하나라도 빠지면 그 job은 VLA-only를 새로 돈다.

| # | 조건 | 확인 방법 |
|---|---|---|
| 1 | `vla_only` run, 같은 suite · level · seed · step budget | manifest `settings` |
| 2 | protocol · runtime schedule 동일 | 지금 launcher가 그 run의 task / episode로 계산한 값과 비교 |
| 3 | checkpoint · norm stats 동일 | aggregate SHA-256, norm stats SHA-256 |
| 4 | interpreter · package 동일 | `settings.versions.libero_python` / `openpi_python` |
| 5 | VLA-only 경로 code 변경 없음 | `git diff <run commit> HEAD -- VLA_ONLY_CODE_PATHS`와 run 당시 `source.patch`. 바뀐 path는 `--accept-diff`로만 받아들이고, 그 path를 건드린 commit 목록이 `pairing_basis.json`에 남는다 |
| 6 | dependency 동일 | manifest `dependencies` = `doc/migration/source-manifest.json` |
| 7 | episode 있음, `execution_error` 아님, `init_state_index == episode_idx`, `policy_seed = seed*100000 + task*1000 + episode` | `episodes.jsonl` |
| 8 | reset 이력 동일: 그 task의 episode 순서가 해당 episode까지 같다 | run의 `episode_ids` 앞부분 = job의 앞부분. fixture 배치는 task env의 reset 횟수에 따라 달라진다(SL4b). task를 어떻게 묶었는지는 상관없다(task마다 env를 새로 만들고 `env.seed(seed)`로 다시 seed한다) |

`--accept-diff`는 diff가 VLA-only 결과를 바꾸지 않는다는 근거가 있을 때만 쓴다(예: 기본값을 바꾸지 않는 opt-in flag 추가 + identity test).
근거는 그 run의 STEP 기록에 적는다. 확신이 없으면 VLA-only를 다시 돈다(LI dev 20 episode ≈ 23분).

## 3. GPU에서 돌리기

```bash
benchmark/runners/run_parallel_gpu.sh --jobs outputs/eval/bm-pb8-dev/jobs.txt [--max-parallel 2]
```

- lock을 잡고, 다른 compute process가 없고, pod working set + 12 GiB × 동시 job ≤ 64 GiB일 때만 시작한다.
- 같은 checkout 안에서 2 job 동시 실행은 단독 실행과 bit 단위로 같다(SL5h G0). 다른 checkout과는 동시에 돌지 않는다.
- 결과: `$AG3S_STORAGE_ROOT/results/benchmark/<KST>_<commit12>_<run-id>-[L<level>-]<mode tag>/`.
- 실행 중에는 runner code를 고치지 않는다(launcher가 spawn 시점에 code를 읽는다).

## 4. 집계 (`aggregate`)

```bash
.venv-ag3s/bin/python benchmark/runners/eval_matrix.py aggregate --plan outputs/eval/bm-pb8-dev/plan.json
```

`plan.json` 옆에 `aggregate.json`과 `aggregate.md`를 쓴다. run을 못 찾거나 episode가 빠지면 `missing`에 적고 exit 1.

| 항목 | 정의 |
|---|---|
| official success | `env.step()`의 `done` (공식 LIBERO 예제). Wilson 95 % 구간 |
| SafeLIBERO `collided` | README 충돌 규칙: wait 직후 작업공간 안 obstacle이 sum\|Δpos\| > 1 mm 움직임 |
| safe success (SS) | official success이고 충돌 없음 (`main_aegis.py` 420줄 `ss`) |
| **CAR** | collision avoidance rate = 1 − collides / episodes (`main_aegis.py` 405–406줄, Supplementary Table S1–S4) |
| **TSR** | task success rate = successes / episodes (`main_aegis.py` 392줄) |
| **ETS** | execution time steps = episode 끝의 `main_aegis` `t` 평균: 성공이면 policy step − 1(`t += 1` 전에 break), 실패면 budget. step budget이 `main_aegis`일 때만 논문 값과 같은 척도다 |
| paired | 같은 (task, episode)에서 VLA-only 대비 2×2(both / VLA-only만 / version만 / 둘 다 실패), gained · lost episode 목록, 비율 차이와 episode bootstrap 95 % 구간(10 000회, seed 0), McNemar exact p |
| family 합계 | LIBERO 4 suite 합, SafeLIBERO 8 suite × level 합 |

## Suite protocol (요약, 정본은 `benchmark/configs/suites.json`)

| suite | task | init / episode | policy step 상한 | wait | 출처 | 성공 판정 | 공개 기준 (pi05_libero) |
|---|---|---|---|---|---|---|---|
| libero_spatial | 10 | 50 | 220 | 10 | openpi `examples/libero/main.py:60-61, 37` | official | 98.8 % |
| libero_object | 10 | 50 | 280 | 10 | 〃 `:62-63` | official | 98.2 % |
| libero_goal | 10 | 50 | 300 | 10 | 〃 `:64-65` | official | 98.0 % |
| libero_10 | 10 | 50 | 520 | 10 | 〃 `:66-67` | official | 92.4 % |
| libero_90 | 90 | 50 | 400 | 10 | 〃 `:68-69` | — | **범위 밖** (개수만 확인) |
| safelibero_spatial I/II | 4 | 50 | 280 (기본) / 300 (`main_aegis`) | 20 | `main_aegis.py:78-79, 53` | safe success | CAR 14.0 · TSR 59.3 · ETS 199.5 |
| safelibero_object I/II | 4 | 50 | 280 / 300 | 20 | `:80-81` | safe success | 18.0 · 57.5 · 223.5 |
| safelibero_goal I/II | 4 | 50 | 280 / 300 | 20 | `:82-83` | safe success | 20.0 · 60.0 · 199.7 |
| safelibero_long I/II | 4 | 50 | 530 / 550 | 20 | `:84-85` | safe success | 16.5 · 54.3 · 420.5 |

공개 기준: LIBERO는 openpi `examples/libero/README.md:69-71`, SafeLIBERO는 `benchmark/SafeLIBERO/Supplementary_Materials.pdf`
Table S1–S4의 π0.5 full action space 평균(4 task × 2 level). SafeLIBERO goal task 3은 level마다 다르다(I: drawer, II: cream cheese).

### SafeLIBERO step budget

`main_aegis.py`는 wait 20 step 뒤 `t = 0`(234줄)으로 되돌리고 `while t < max_steps`(254줄)를 돈다. 즉 300 / 550은
**wait 다음의 policy step 수**다(실패 episode의 ETS가 300 / 550인 Supplementary 표와 맞다). SL5c부터의 기본 protocol은
"wait 포함 t < 300"(policy step 280 / 530)으로 읽었다. 기본값은 바꾸지 않았다(기존 결과와 비교 가능성 유지).
공식 수치와 비교할 run은 `--safelibero-step-budget main_aegis`로 돈다. 어느 쪽을 쓸지는 사용자 결정이다.

### dev / test split

| family | dev | test | 상태 |
|---|---|---|---|
| LIBERO (spatial · object · goal · 10) | suite마다 init 0–4 | 5–49 | spatial 확정 2026-10-12, 나머지 **확정(사용자 2026-10-23)** |
| SafeLIBERO (spatial · object · goal · long) | suite · level마다 episode 0–4 | 5–49 | spatial 확정 2026-10-14, 나머지 **확정(사용자 2026-10-23)** |

`--data-split` label은 `suite_registry.split_label`이 만든다. LIBERO Spatial dev와 SafeLIBERO Spatial dev label은
SL5 run과 같은 문자열이다. launcher는 registry label이 다른 suite의 것이거나 split 밖 episode를 담으면 시작 전에 거절한다.

## 주의

- 옛 run 이름의 `libero10-init0`은 **LIBERO Spatial task 0–9 × init 0**이다. LIBERO-10(`libero_10`) suite가 아니다.
- 새 suite나 asset을 붙이면 GPU 전에 `suite_probe.py`로 관측 규약 · 충돌 판정을 먼저 확인한다.
- LIBERO-Safety는 별도 client(BM-SETUP-B)다. 이 recipe의 launcher와 client는 LIBERO · SafeLIBERO만 다룬다.
