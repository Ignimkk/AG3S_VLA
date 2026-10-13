# SL4c — ATS grounding 탈락 원인 진단 (t4 · t5 · t9, dev oracle frame) — 진단만

> writer: lead (A0) · 읽는 쪽: verifier · 작성 2026-10-13 KST
> 기준 code: **`85e6d9d1` 고정** — `.runtime/worktrees/sl4c-85e6d9d1` (detached worktree). SL5a의 core 변경이 섞이지 않게 이 worktree의 code로만 실행한다.
> 계획: `doc/develope_PLAN_SUMMARY.md` §0.1(V11AB 유지), §3.1. Dev(init 0–4)만 사용.

## 사용자 판정 (2026-10-13)

1. SL4b 확정: 사전 등록 규칙으로 L8H2 유지(최고 후보 L12H3 +1.48 %p < +10 %p). dev rollout 48/50은 **pi0.5-only** 결과로 구분해 기록.
2. **SL4c는 원인 진단까지.** 측정 항목: 단계별 target 점 생존율, cluster의 물체 구성·크기, **실제 적용된** 개도 한계와 탈락 사유.
   - "cluster 병합"과 "Panda 8 cm 개도 제한"은 **가설**로 남긴다. 측정이 확인하거나 반박하게 한다.
   - code·parameter 변경은 진단 이후 별도로 결정한다. 물리적 robot 사양 매핑 오류는 adapter 보정 후보이고, 성능을 위해 clustering·선택 기준을 바꾸는 것은 **별도 method 변경**이다.
3. SL5a는 병행. 이 진단은 `85e6d9d1` code에 고정한다.

## 무엇을 재나 (→ verifier)

대상 frame: SL4b dev의 **oracle** grounding-only run(`*_sl4b-gonly-oracle`)과 L8H2 run(`*_sl4b-gonly-L8H2`, `8f0a07ec`)의 평가 call(approach+grasp).
비교군으로 t4·t5·t9 전 call, 그리고 정상 task 대조(t2·t7 등 oracle 정답 task)를 포함한다. 입력(관측·attention·GT)은 SL4b와 같은 것을 쓴다.

| # | 측정 | 숫자 | 비고 |
|---|---|---|---|
| 1 | 단계별 target 점 생존율 | GT target(segP) pixel → back-project 점 → self-filter 후 → downsample/fusion 후 → support surface 제외 후 → seed(oracle) → 각 cluster 소속 → admissible 여부 → 최종 선택. 단계마다 남은 target 점 수와 비율 | V11AB 실제 함수를 단계별로 호출하는 측정 script(`outputs/verify/SL4c/scripts/`)로. **code 수정 금지** |
| 2 | cluster 구성 | target 점이 속한 cluster의 물체별 점 구성(GT instance 비율), bbox 크기(mm), 점 수; 선택된 cluster의 같은 정보 | 병합 가설 확인/반박 |
| 3 | 탈락 사유 | grounding 결과의 status·reject 사유(admissibility, score, confirm 등)를 cluster별로; target cluster의 score와 선택 cluster score | V11AB 내부 debug 값 그대로 |
| 4 | 개도 한계 | admissibility가 **실제로 쓴** 최대 개도 값과 그 출처(Panda model의 finger joint limit 등), 측정 폭 정의(어느 축/어떤 폭), RB-Y1에서의 대응 값 | 8 cm 가설 확인/반박; robot 사양 매핑이 맞는지 |
| 5 | support surface | 장면에서 fit된 support plane 목록(높이·법선·inlier 물체), target 점 중 support로 제외된 비율 | cabinet 윗면·서랍 바닥·ramekin 윗면이 support로 잡히는지 |
| 6 | 대조 | 같은 측정을 oracle 정답 task(t2, t7 등)에서 | 무엇이 다른지 |

- 판정 문장을 쓰지 않는다. 가설 각각에 대해 "어느 숫자가 관련되는가"만 정리한다.
- figure 3종(`sl4c-*`): (1) 실제 scene — t4/t5/t9 대표 frame의 3D 점(target 점 단계별 생존, cluster 색, support plane) + 2D overlay, (2) graph — 단계별 생존율 funnel(task별), cluster 크기 vs 개도 한계, (3) table — 탈락 사유 집계, 개도 한계 출처, support plane.
- raw: `outputs/verify/SL4c/` (main checkout 경로). verify.json: `AG3S/ag3s/docs/handoff/SL4c.verify.json`.

## 실행 규칙

- code는 `/mnt/dev/AG3S_VLA-benchmark/.runtime/worktrees/sl4c-85e6d9d1`(`85e6d9d1`, detached)에서 import한다: `PYTHONPATH=<worktree>:...` 를 main checkout의 venv interpreter와 함께 사용(worktree에는 venv가 없다). 실행 전에 `python -c "import AG3S, benchmark; print(AG3S.__file__, benchmark.__file__)"`로 import 출처가 worktree인지 확인하고 기록.
- Storage는 기존 `AG3S_STORAGE_ROOT`. 새 run이 필요하면 `create_run`이 worktree commit(`85e6d9d1`)을 기록하는지 확인.
- GPU가 필요하면 실제로 lock을 잡는 형태(`( flock -w N 9 || exit 1; … ) 9>/tmp/ag3s-vla-gpu-validation.lock`), 시작·종료 main 통보. CPU worker ≤ 2, RSS ≲ 3 GB, foreground.

## 이 STEP 이 답하는 질문

oracle attention이 진짜 bowl을 가리키는데도 t4·t5·t9에서 ATS가 다른 물체를 고르는 것은 V11AB grounding의 어느 단계·어느 기준 때문인가?
