# IFG-APPLE-run — 사과 target 장면에서 VLA 단독 대 기존 방식 paired closed-loop (verifier)

작성: 2026-10-19 13:30 KST, lead. 사용자 지시(10-19): "사과 위주로", "baseline checkpoint 는 성공하는가?", "권장사항대로 수행".
선행 틀: IFG-CL3 run `results/development/20261018T135457+0900_f15e587290f6_2cd4b0b6acad/verify/IFG-CL3/p/` (launcher · chain · client · check · analyze · replay · agg, 모든 `LEAD_AMENDMENTS.md` 포함 — 특히 NFS 사고 기록).

## 목적

1. VLA checkpoint 단독(AG3S 없음)이 사과 장면에서 성공하는가
2. 같은 장면 · 같은 seed 에서 기존 방식(C1aN)이 VLA 단독 대비 성공을 잃거나 얻는가
3. 나중에 새 구성(IFG-S2SERVE: 안전 경보 + 3D 쥠 규칙)을 **같은 seed 로** 추가할 수 있게 seeds · argv · snapshot 을 고정해 둔다

## arm

| arm | 뜻 | 서버 | client |
|---|---|---|---|
| E0 | VLA 단독. AG3S 요청 없음 (T39 · T43 의 E0 규약) | C1aN 서버를 그대로 쓴다 (E0 요청은 seed 만 소비하고 정책만 호출) | CL3 client argv 에서 `--safe-*` flag 만 뺀다. 그 밖(관측 · 기록 · 장애물 · seed)은 같다 |
| C1aN | 기존 방식 = 현재 serving + noun attention target 선택 (CL3 의 C1aN 과 같은 flag) | CL3 의 C1aN | CL3 와 같다 |

- 확인 장치: 같은 seed 에서 E0 run 의 첫 chunk 와 C1aN run 의 첫 `reference_chunk` 가 같아야 한다(둘 다 VLA 원 출력). 다르면 멈추고 보고
- E0 client 에 `--record-ag3s` 등 AG3S 전용 flag 가 남아 오류가 나면 그 flag 만 빼고 argv 차이를 `argv_compare.json` 에 기록

## 장면 선정 (사전 고정, 어느 arm 의 성공도 보지 않음)

- 후보 = dataset meta 의 `task_type == place_one` **이고 `target_fruit == apple`** 인 episode
- 제외 = IFG-CL3 의 제외 집합(`seeds.json` `exclusion` — IFG dev · held-out collector · T43 허들) ∪ IFG-CL3 의 40 episode
- 순서 = `sha256("IFG-APPLE:" + str(episode))` 오름차순. 앞에서부터 **clear 16**, 이어서 **place 8** (CL3 의 `place_geom.py` 규칙 그대로; 기하적으로 불가능하면 건너뛰고 `skipped_geometry` 기록)
- target 옆 기둥(adjacent)은 쓰지 않는다 (사용자: 가림 ≤ 5 % · 궤적 비간섭 배치는 의미가 약하다)
- seed = `episode * 10 + 3`, side = dataset meta `used_arm`
- `seeds.json` 을 rollout 전에 고정하고 sha256 기록. **seed 를 자동으로 늘리지 않는다**

## 실행

- 새 run(`python -m common.run_output`), `verify/IFG-APPLE/`. storage 의 `results/development/RUN_INDEX_IFG.md` 에 한 줄 추가(설명 label)
- snapshot: 지금 작업 트리(HEAD f15e5872 + 미commit 변경)를 CL3 와 같은 방식으로 snapshot 한다. **IFG-S2SERVE 구현 중인 파일이 섞이지 않게**, CL3 snapshot tree(`f0d8046b`)와 서버 · client 코드 md5 가 같은지 확인하고 다르면 CL3 snapshot 을 그대로 복사해 쓴다
- 48 run (24 장면 × 2 arm), 서버 2 개 (8292 / 8293), 짝마다 arm 을 포트에 번갈아 배정
- GPU: CL3 와 같은 `bench_watch` (benchmark 작업이 10 분 없을 때) + `/tmp/ag3s-vla-gpu-validation.lock`. 시작 전과 종료 후 benchmark 세션에 알린다 (`ListAgents` 로 이름 확인)
- **NFS 사고 대비:** client 가 시작 후 10 분 넘게 log 0 byte 이고 D 상태면 lead 에게 알린다 (CL3 `LEAD_AMENDMENTS.md` "lead recovery" 절: run 폴더 전체 `lstat` 이 stuck inode 에서 멈춘 사례). 다른 세션 process 는 건드리지 않는다
- 인프라 오류만 같은 seed 재시도, 재시도 기록. 실패는 모두 남긴다

## 보고 (`AG3S/ag3s/docs/handoff/IFG-APPLE.verify.json`, 수치만)

- 주 지표 = 접촉 없는 작업 성공 (CL3 `agg_cl3.py` 정의). E0 대 C1aN paired 차이와 95 % CI (episode 쌍 bootstrap 10,000, seed 20261019), 2×2, stratum 별
- 잡기 실패 · failure_mode · 접촉(비허용 link–target · 장애물 · 이웃 과일) · HOLD · 완료 시간
- 첫 chunk 일치 검사 결과, 요청 지연(C1aN), 재시도
- figure: `AG3S/ag3s/docs/figures/ifg-apple/` (CL3 의 `ifg_cl3_results.py` 를 arm 이름만 바꿔 쓸 수 있으면 그것으로; arm 설명을 그림 안에 풀어 쓴다) + `.json` sidecar
- 동등성 · "영향 없음" 판정 문구 금지. 메모리 회수(`reclaim_cache.py`) 한 줄

## 하지 말 것

코드(`AG3S/` · `tests/` 의 `*.py`) 수정 금지, commit 금지, 다른 세션 process 금지, lock 없이 GPU 금지, run 밖을 가리키는 symlink 금지(읽기 전용 입력 경로 인자는 된다).
