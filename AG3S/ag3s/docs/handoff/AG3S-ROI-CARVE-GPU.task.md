# AG3S-ROI-CARVE-GPU — 관심 영역 지우기의 비용 원인 측정과 GPU 판정, 그리고 동적 거리장 로그의 미측정 항목 정리

> writer: lead · 실행: lead 직접 (순차 작업 — 원인 측정 → 구현 → 동등성 · 속도 확인 → 기록)

작성: 2026-10-25 04:0x KST (`date -u` 2026-10-24 19:04 UTC + 9 h), lead. **설계 · 지표 · 판정 기준은 결과를 보기 전에 고정한다.**
앞선 STEP: `AG3S-TASK-DYNAMIC-MAP` (E4 채택 · serving 기본값 켬, 로그 `AG3S/ag3s/docs/AG3S_TSDF_DYNAMIC_LOG.md` §8).

## 0. 사용자 판정 (2026-10-25)

- "GPU 변환 전에 원인 측정이 중요하다면 먼저 수행하고, 해결 방법이 (가) 이면 그대로 진행" → 이 세션은 관심 영역 지우기(`curobo_builder.py`) 만
  GPU 로 옮긴다. 나머지 느린 단계 (target 찾기 등) 는 핵심 평가 세션 (`ag3s-vla-development-c4`) 몫이다.
- "로그와 다른 점에서 작성된 것을 해결" → 아래 §1 의 네 항목을 측정하거나 근거가 없는 주장을 거둔다.

## 1. 해결할 항목 (앞 세션 요약과 `AG3S_TSDF_DYNAMIC_LOG.md` 가 다른 점)

| # | 앞 요약의 주장 | 로그의 상태 | 이 STEP 의 처리 |
|---|---|---|---|
| R1 | 거리장 단계 110 → 303 ms 의 주 원인 = CPU 세 부분 (target pixel 판정 · 점유 voxel host 내리기 · 광선 판정) | 단계별 원인 측정 안 함 (§8.4, 재측정 대기 9) | E2–E4 서버 기록의 `profile_ms` · `roi_carve.ms` 집계 + offline replay 단계별 시간 (§2) |
| R2 | target pixel 판정이 원인 중 하나 | — | 코드: `target_layer` 꺼짐에서는 실행되지 않는다 (`update` 가 거부). E4 구성은 꺼짐 → 원인 아님을 기록 |
| R3 | 서버 한 chunk 의 제약 생성 약 380 ms · target 찾기 약 80 ms | — | `constraint_generation` 이 `esdf` 를 안에 포함하는지 (`pipeline.py` stage 중첩) 확인하고 빼서 집계 |
| R4 | 1345 바나나 접촉의 원인 = 궤적 쪽 이웃 접촉 허용 설정 | 원인 미확인 (§10) | 954 진단 (§6.1) 과 같은 항목을 1345 (E4 기록) 에 측정. 측정으로 정해지지 않으면 "미확인" 유지 |
| R5 | target 전용 TSDF 의 사과 pixel 이 정적 TSDF 로 새는 원인 = target 추적 | 가설 [H] | 이 STEP 은 측정하지 않는다 (target 전용 TSDF 는 serving 에서 꺼짐). 앞 요약의 단정만 거둔다 |

## 2. 비용 원인 측정 (R1 · R3)

- **서버 기록 집계 (closed-loop, 이미 있는 기록):** E2 기준 구성 · 관심 영역 지우기, E3, E4 의 모든 chunk 에서 `profile_ms.*` 와
  `esdf_stats.roi_carve.ms`, 관심 영역 점유 voxel 수. `esdf − roi_carve.ms` 가 기준 구성의 `esdf` 와 같은 크기이면 증가분은 관심 영역 지우기다.
- **offline replay 단계별 시간 (신규 도구 `AG3S/ag3s/experiments/studies/roi_carve_profile.py`):** E4 의 18 장면 policy record (요청별 depth
  3 대 · K · T · qpos) 를 E4 서버 argv 의 AG3S builder 에 차례로 넣고, 관심 영역은 서버가 그 요청에 기록한 상자를 쓴다. `_roi_carve` 의
  부분 (점유 voxel 메타 · GPU 선택 · host 복사 · host 계산 · 투영 영역 판정 · 그중 `minimum_filter` · 쓰기) 을 CUDA synchronize 로 잰다.
  replay 의 TSDF 는 서버의 쥔 물체 처리 · target 제외를 빠뜨리므로 (§1 of AG3S-TASK-DYNAMIC-MAP) **점유 voxel 수를 서버 기록과 나란히 적어**
  작업량이 비슷한지 보인다. 시간의 절대값은 서버와 다를 수 있고 비율로 읽는다.

## 3. 구현 — `esdf.roi_carve_device` (`--roi-carve-device {cpu,cuda}`, 코드 기본 `cpu`)

- `cuda`: 점유 voxel 선택 · 중심 · 상자 판정 (`occupied_voxels_gpu`), 투영 · 영역 최솟값 (`seen_through_gpu`: `max_pool2d` 로 같은
  `minimum_filter(mode=constant, cval=0)`), 쓰기를 GPU 에서. host 로는 개수만 온다. depth 는 update 마다 한 번 올린다.
- 판정 규칙은 CPU 와 같다. float64 투영의 연산 순서가 BLAS 와 달라 마지막 bit 에서 pixel 경계 · margin 경계가 갈릴 수는 있다 → §4 에서 잰다.
- `--roi-carve-device` 는 carve flag 가 아니다 (serving 기본값을 끄지 않는다). `--no-roi-carve` 와 `cuda` 는 함께 쓸 수 없다.

## 4. 판정 (결과 전 고정)

| 기준 | 내용 |
|---|---|
| G1 동등성 | E4 18 장면 replay 의 모든 요청 · 두 TSDF 에서, 같은 TSDF 상태에 대한 CPU 판정과 GPU 판정의 **지울 voxel 집합이 같다** (차이 0 개). 차이가 하나라도 있으면 serving 기본값을 바꾸지 않고 차이의 위치를 보고한다 |
| G2 속도 | 같은 replay 에서 `_roi_carve` 시간 (두 TSDF 합, GPU 는 depth 업로드 포함) 중앙값이 CPU 대비 50 % 이상 줄고 p95 도 줄어든다 |
| G3 서버 경로 | `--roi-carve-device cuda` 서버 smoke (closed-loop 1 episode, E4 와 같은 장면 · seed) 가 정상 종료하고 기록의 `roi_carve.device = cuda` · 서버 `profile_ms.esdf` 를 E4 같은 장면과 나란히 적는다 (성공 · 접촉은 판정에 쓰지 않는다 — 1 episode) |

G1 · G2 · G3 모두 통과하면 `serve_safe.ROI_CARVE_SERVING_DEFAULT` 에 `roi_carve_device: cuda` 를 더하고 핵심 평가 세션에 알린다.
`EsdfConfig` 코드 기본값은 `cpu` 유지. 이전 판정으로 되돌리기는 `--roi-carve-device cpu`.

## 5. 1345 바나나 접촉 진단 (R4) — 954 진단과 같은 항목

E4 (가장자리 보존 완화) 기록 1345 에서: 접촉 쌍 · chunk 구간 · 접촉 행 수 · 최대 겹침 (MuJoCo replay), 그 chunk 의 서버 설정
(`--neighbour-contact-mm 20`), 손 근처 (60 mm 안) 바나나 표면이 hand layer 에서 빠진 비율 (기준 구성 · E4 나란히), 그 chunk 의 TrajOpt
궤적 수정 여부 (`refined_chunk` vs `reference_chunk`). 원인 층 (LTE · CCR) 은 측정이 가리키는 만큼만 적는다.

## 6. 산출물

- 원시: `$AG3S_RUN_DIR/verify/AG3S-ROI-CARVE-GPU/` (run `20261025T035831+0900_8da37119afd3_f082f20e9463`)
- `AG3S/ag3s/docs/handoff/AG3S-ROI-CARVE-GPU.verify.json`, 그림 `AG3S/ag3s/docs/figures/roi-carve-gpu/` (+ sidecar)
- 로그: `AG3S/ag3s/docs/AG3S_TSDF_DYNAMIC_LOG.md` 에 이어 쓴다 (이 축의 정본)
- 보기: `outputs/2026-10-25_관심영역지우기_GPU판정/`
