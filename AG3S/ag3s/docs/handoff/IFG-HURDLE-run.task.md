# IFG-HURDLE-run — 허들 closed-loop (1차: 현재 구현 그대로) + 실시간성 (verifier)

작성: 2026-10-17 KST, lead. STEP 번호 없음 (기록은 `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md`).
선행: `IFG-CL1-run.task.md` (held-out 짝 비교, run `20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1`, 그 `p/` 틀과
`p/LEAD_AMENDMENTS.md` 의 수정 사항 전부).

## 사용자 판정 (2026-10-17)

- 허들 테스트를 **현재 구현 그대로 (1차)** 돌리고, 수정안 (하이브리드) 이 준비되면 2차를 따로 돌린다 (이 task 는 1차).
- arm 3 개: **C1a** (collector C1a flag set 그대로 = 현재 serving, action attention L8H2), **C1aN** (C1a + `--attention-source noun_target:L10H7`),
  **SURF** (C1aN + `--target-extent attention-surface --robot-mask render`). IFG-CL1 의 arm 정의 · 서버 log guard 그대로 (C1a 는 collector 것).

## 장면 = T43 H 세트 (허들 14 run)

- 목록: `/mnt/dev/AG3S_VLA-storage/results/workspace/verify/T43/V11/p/lists/hurdle.txt` (7 episode × 2 seed). 목록의 config 경로
  `/mnt/dev/work/...` 는 퇴역 workspace 이므로 쓰지 않는다 — 같은 파일이 `/mnt/dev/AG3S_VLA-storage/results/workspace/verify/T43/P2/configs/` 에 있다
  (내용 동일 여부를 md5 로 확인해 기록). profile `t43p2_overbar`.
- client 인자: T43 V11 의 `p/run_client.sh` · `env.sh` 가 쓴 client argv 를 기준으로 하되, 현재 IFG-CL1 client 인자 (safe-* 옵션, warmup, record
  옵션, `--obstacle-stop-distance 0`) 와 다른 점을 표로 비교하고, **현재 serving 실험 규약 (IFG-CL1 = collector C1a) 을 따른다.** 차이가 판단을
  요하면 (예: `--safe-phase`, hold mode, episode 의 used_arm) 멈추고 lead 에게 묻는다. sides 는 dataset meta used_arm.
- 짝: (episode, seed) 마다 3 arm. 포트 2 개에 arm 을 번갈아 배정 (포트 효과 상쇄). smoke gate: 첫 (episode, seed) × 3 arm 이 완전히 끝나고
  SURF 의 attention_surface 적용 비율 (층이 만들어진 chunk 기준 — IFG-CL1 smoke_check 의 lead 수정판) 을 기록. 적용 비율 < 0.3 이면 멈춘다.

## GPU 순서

benchmark 세션 (ag3s-vla-benchmark-4e) 이 대기열 6 건을 run 마다 lock 을 따로 잡으며 돌리고 있다 (08:45 UTC 쯤 끝 예정). launcher 는
`/tmp/ag3s-vla-gpu-validation.lock` 을 blocking 으로 기다리되, **benchmark process (`/mnt/dev/AG3S_VLA-benchmark`) 가 10 분 연속 없을 때만** 시작한다
(그들 run 사이에 끼어들지 않는다). foreign compute app 없음 · cuRobo baseline 일치는 IFG-CL1 launcher 와 같다. 서버 ≤ 2.

## 결과 (verify.json `AG3S/ag3s/docs/handoff/IFG-HURDLE.verify.json`, 수치만)

run · arm 별: 기존 성공 (S), release 규칙 성공, **접촉 포함 성공 (CT 규칙: release 성공 ∧ place 시점까지 로봇 · 사과–허들 MuJoCo 접촉 0)**,
grasp, failure_mode, 허들 접촉 (로봇 · 쥔 사과, place 전/후), 비허용 link–target 접촉, obstacle stop, HOLD chunk, 완료 시간.
짝 표 + McNemar exact (C1a vs C1aN, C1aN vs SURF, C1a vs SURF). T43 V11AB 결과 (`AG3S/ag3s/docs/handoff/T43V11AB.verify.json`) 는 같은 seed 의 참고 열 (설정이 달라 비교 주의 표시).
실시간성: IFG-CL1 `timing_cl1.py` 와 같은 항목 (단계별 median · p95, 533 ms 초과 비율). SURF 진단: 적용 비율, fallback 이유.
figure: `AG3S/ag3s/docs/figures/ifg-hurdle/` (arm 별 CT 성공 막대 + 짝 표, 대표 장면 1 개의 실제 씬 — 허들 근처 frame, + `.json`).

## 하지 말 것 · 넘길 것

코드 (`AG3S/` · `tests/` 의 `*.py`) 수정 금지, 해석 금지. 다른 세션 process 금지. 새 run 디렉터리 (`python -m common.run_output`), snapshot 은
IFG-CL1 과 같은 방법 (임시 index tree; 현재 working tree 의 `AG3S tests` — 그 사이 implementer 가 수정 중일 수 있으므로 **snapshot 을 만든 시점의
tree SHA 와 `git diff --stat` 를 기록하고, 수정 중인 파일이 섞이면 IFG-CL1 snapshot tree `f0d8046b` 를 그대로 쓴다**).
launcher 를 띄운 뒤 기다리지 말고 보고: run 경로, pid · log, client argv 비교표, 시작 조건, 예상 소요. 집계는 rollout 뒤 자동 (CPU).
