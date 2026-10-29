# IFG-APPLE-TSURF — 장애물 근거에서 target 자신의 표면 점을 빼는 설정의 closed-loop 확인 (사과 10 장면)

> writer: lead · 읽는 쪽: verifier

작성: 2026-10-22 KST, lead. 결과를 보기 전에 고정한다.
근거: WORKLOG_S0S3 2026-10-22 "IFG-SPHERE 1단계" 절, `IFG-SPHERE.impl.md`, 사용자 판정 2026-10-22
("표면 점 빼기 + 놓은 뒤 인증", "새 구성에만 허용").

## 무슨 실험인가

- **질문:** 쥐기 전에 사과 자신이 장애물로 남아 TrajOpt가 손을 밀거나 멈추는 문제를 이미 있는 두 설정으로 줄이면 closed-loop 결과가 어떻게 바뀌는가
- **켜는 설정 2개** (둘 다 코드 수정 없음, 기존 flag)
  1. 장애물 근거에서 target 자신의 표면 점을 뺀다: `--nontarget-evidence surface` (2026-10-12 채택, shell 기본 2.0)
  2. 놓은 뒤 target 덩어리를 못 찾아도 안전 인증을 한다: `--placed-no-seed-certify` (1번 설정이 놓은 뒤 멈춤을 늘린 부작용을 고친 수정)
- **비교 기준 (다시 돌리지 않는다):** `IFG-APPLE-NBC`의 지금 최선 구성
  - 내용: 이웃 과일과 붙은 target 떼어 내기 + 쥐기 직전 손목 관절 경로 가중 0.1 + 손가락–이웃 과일 접촉 20 mm 허용
  - 같은 10 장면 · seed에서 9 / 10 성공, 211 실패
  - 서버 flag는 `IFG-APPLE-NBC/p/env.sh`의 해당 구성 줄 그대로에 위 2개만 더한다

## 실행

- kit: `IFG-APPLE-NBC/p`를 새 run 폴더 `verify/IFG-APPLE-NBC-TARGETSURFACE/p`로 복사한다
  - run root는 `20261019T132719+0900_f15e587290f6_11408fd6d27f`(기존 사과 run들과 같은 root)
  - `LEAD_AMENDMENTS.md` 맨 위에 이 실험 절을 추가한다
- 구성 이름: kit 안에서는 새 이름 하나(예: `NBCTS`)를 쓴다. 보고와 그림 제목에는 위의 설명 이름을 쓴다
  - `env.sh`의 `arm_flags`와 `is_arm`에 추가한다
  - `run_chain.sh` guard: 새 구성은 두 flag가 **있어야** 통과하고, 기존 구성의 "있으면 거부" guard는 그대로 둔다(사용자 판정: 새 구성에만 허용)
  - `t46_stats_check` · `check_run.py`는 새 구성에서 non-target 근거가 surface 방식으로 실제 쓰였는지(chunk stats `nontarget_evidence`)와 놓은 뒤 인증 기록(`subtask.placed_no_seed_certify`)이 남는지를 확인한다
- 장면 · seed: `IFG-APPLE-NBC/seeds.json`과 byte 동일(sha256 확인), `sides.tsv`도 같게
- 코드 snapshot: 현재 HEAD tree. 기존 snapshot_nbc와 runtime 파일 md5를 비교해 차이를 `LEAD_AMENDMENTS.md`에 적는다. 차이가 서빙 동작에 닿으면 실행 전에 멈추고 보고한다
- client: 사용자 규칙(2026-10-21)대로 `--stop-after-place 2.5`를 붙인다(`client_args.py` · `check_run.py` · `run_chain.sh` 세 곳)
  - 비교 기준은 이 규칙 전에 돌아 끝까지 기록됐다. 그래서 비교 지표는 두 쪽 모두 **place 판정 후 2.5 초까지**로 자른 구간에서 계산한다. 놓은 뒤 구간의 멈춤 수는 그 2.5 초 안에서만 비교한다
- smoke: 첫 장면 1개. 새 구성의 첫 VLA chunk가 비교 기준의 첫 chunk와 같고(≤ 1e-5), chunk stats에 surface 근거가 쓰였는지 확인한다
- GPU: `/tmp/ag3s-vla-gpu-validation.lock`을 blocking flock으로 기다린다. `nvidia-smi`에 다른 compute process가 없을 때 시작한다. 서버 ≤ 2. 실행 시작 · 종료를 다른 세션(`ag3s-vla-development-98`)에 알릴 필요가 있으면 lead에게 보고만 한다
- 회귀 기준선(legacy · cuRobo)을 먼저 확인한다

## 보고 (→ `AG3S/ag3s/docs/handoff/IFG-APPLE-NBC-TARGETSURFACE.verify.json`)

| # | 지표 | 비고 |
|---|---|---|
| 1 | 작업 성공 / 10 (잠근 정의), 비교 기준과의 짝 2×2(실패→성공, 성공→실패) | 주 지표 |
| 2 | 실제 MuJoCo 접촉 (이웃 과일 · 바구니 · 탁자 · 미검사 링크), step 수 · 최대 침범 | `replay_contacts.py`, 실제 궤적 기준 |
| 3 | 쥐기 전 HOLD chunk 수, 놓은 뒤 2.5 초 안의 HOLD 수 (인증 실패 여부 포함) | 장면별 |
| 4 | 쥐기 전 손가락 거리장에 남은 GT 사과 voxel 수 (중앙값), 사과 출처 활성 제약 row 수 | `ifg_sphere_attrib.py` · `ifg_precheck_a.py` 방식 재사용 |
| 5 | 211: 요청 10–16의 활성 제약 출처 · HOLD · 손바닥–target 거리, 닫기 시작 위치 | 비교 기준과 나란히 |
| 6 | 결과가 바뀐 장면마다 처음 갈라진 요청과 그 요청의 서버 판단 | |

- figure: 실제 씬 1(211 닫는 순간 두 구성 비교) · 그래프 1 · 표 1 이상, `AG3S/ag3s/docs/figures/ifg-split/`에 `ifg-targetsurface-*` 이름으로, `.json` sidecar 포함
- 해석은 쓰지 않는다. 끝나면 process 종료 + `python3 scripts/setup/reclaim_cache.py "$AG3S_RUN_DIR" <읽은 큰 입력…>` 한 줄

## 미리 정한 해석

- 성공→실패가 0이고 실패→성공이 1이면(211) → 두 설정을 사과 실험의 다음 비교 기준으로 둘지 사용자에게 묻는다
- 성공→실패가 생기면 그 장면을 서버 판단으로 진단한다. 1번 설정 때문이면 채택하지 않고 보고한다
- 211이 그대로 실패하면 → 요청 10–16에서 사과 출처 제약이 실제로 줄었는지를 보고, 남은 원인(옛 기억 · 거리장 잔여)으로 순서 2 설계로 넘어간다
- 10 장면이므로 효과 크기는 판정하지 않는다
