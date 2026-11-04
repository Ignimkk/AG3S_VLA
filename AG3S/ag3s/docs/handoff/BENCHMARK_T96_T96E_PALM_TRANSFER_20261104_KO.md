# benchmark 이식 지시 — 수정안 3 가지 (target 닫힌 물체 제외 · 지운 seed 부호 수정 · 실제 크기 손바닥) 를 LIBERO · SafeLIBERO · LIBERO-safety 에서 ablation 으로 시험

> writer: development lead (세션 ag3s-vla-development-3a) · 작성: 2026-11-04 17:1x KST (`date -u` 08:1x + 9 h).
> 사용자 지시 (2026-11-04 17:0x KST): "이 3 가지를 커밋해서 benchmark checkout 쪽으로 보낼 것입니다. 거기에서 3 가지 벤치마크에서 테스트해 봅니다. 결과에 따라 기존 P0m 으로 갈지 아니면 3 가지 수정안을 반영해 볼지 선택합니다. 테스트할 때는 3 가지를 ablation 적용으로 해서 진행해 주세요."
> 받는 쪽: benchmark checkout (`/mnt/dev/AG3S_VLA-benchmark`, `feature/panda-benchmark`) 세션. 앞선 이식 문서 `P0_FINAL_CONFIG_FOR_BENCHMARK_20261103_KO.md` (P0m) · `BENCHMARK_T90_T93_TRANSFER_20261103_KO.md` 위에 더한다.
> **채택은 아무것도 하지 않았다.** P0m 이 그대로 최종 구성이고, 이 문서의 결과로 사용자가 P0m 유지 / 수정안 반영을 정한다. 판정 · 결과 보고는 benchmark 세션이 사용자에게 직접 한다.

## 0. 코드 전달 경로

- development `feature/ag3s-development` 커밋 (코드 · test) 과 (문서 · figure) → main merge. SHA 는 §5 에 적는다.
- 새 기능은 모두 **기본 꺼짐 flag** 다. 꺼짐 = 이전과 bit 동일 (development 단위 test, §5). merge 만으로 benchmark 동작은 바뀌지 않는다.
- 같은 커밋에 T97 (맞닿은 물체 경계 · 돌출 보존, `--surface-crease-protrusion*`, `--surface-crease-hand-freeze-mm`) 코드도 들어 있다. **이번 ablation 대상이 아니다** (RB-Y1 손 층 재생 기준 미통과, D32). 끈 채로 둔다.

## 1. 수정안 3 가지 — 방법과 RB-Y1 근거

대상은 모두 **손 층** (쥐는 손의 손바닥 · 손가락 구만 보는, target 을 지운 미세 거리장) 이다.

### A. target 을 닫힌 물체로 지우기 (T96 · T96c · T96d)

| | 지금 (P0m) | 닫힌 물체 |
|---|---|---|
| 지우는 범위 | 채택 표면 영역의 표본 + 띠, 영역 convex hull 채우기 (`--surface-hull-fill`), target 표본 10 mm 안 seed 전부 (`--surface-contact-allowance-mm 10`) | 채택 영역 점 R ∪ R 을 받침면 (RANSAC 평면) 으로 수직으로 내린 점의 convex hull S (발자국 · 표면과 탁자 사이) + 카메라가 target 을 본 pixel 뒤 80 mm 안의 안 보인 공간 (그림자) |
| 이웃 | 접촉 여유가 10 mm 안 이웃도 지움 ("쥐는 순간 장애물 없음" 가정) | 이웃 관측 표면 5 mm 안 (τ) 은 어떤 길로도 지우지 않음. 에피소드 중 한 번이라도 target 이었던 기억 칸의 표본은 이웃 증거에서 뺀다 |
| 탁자 | z_floor guard | 같음 (+ 받침면 표본은 이웃 증거에서 뺌) |
| 규칙 수 | heuristic 2 개 (hull 채우기 · 접촉 여유) | 한 원리 (target 소유 부피 − 이웃 보호 띠). hull 채우기 · 접촉 여유와 같이 켜면 시작 거절 |

RB-Y1 고정 입력 재생 (9 run, 쥐기 직전 78 요청, 바닥 띠 제외 남은 사과 voxel 요청 중앙 / 추가로 지운 voxel 중 이웃 · 탁자 · 바구니):
P0m 11.5 % / 22.3 % → 닫힌 물체 11.8 % / 6.4 % (부호 수정과 함께: P0m 7.8 % / 10.0 %, 닫힌 물체 8.2 % / 8.3 %).
**closed-loop 은 돌리지 않았다** (사전 규칙: 재생 잔여 < 5 % 통과 시에만). → 이웃을 덜 지우고 잔여는 같은 수준. 쥐기 성공에 대한 효과는 이 벤치마크 시험이 처음이다.

### B. 지운 seed 의 부호 수정 (T96e, `--surface-removed-sign-free`)

- 문제: target 제외가 지운 seed 바로 옆 (1 · √2 voxel) 에 남은 seed (95–97 % 는 사과 자신) 가 있으면, 거리 변환 뒤 부호 교정의 순도 규칙 (`|d| > 1.5 voxel` 일 때만 양수) 이 건너뛰어 TSDF 안쪽 음수 (−5 ~ −7 mm) 가 남는다. TrajOpt 에게는 지웠는데도 "사과 속".
- 고침: 거리 변환 · 기존 부호 교정 **뒤에**, target 제외가 지운 목록의 voxel 만 `|d|` (남은 seed 까지의 양수 거리) 로. 목록 밖 · 남은 seed · 쥔 물체 경로는 그대로. A · P0m 어느 쪽에도 붙는다.
- RB-Y1: 재생 P0m 11.5 → 7.8 % (모든 run 감소). closed-loop seed 1 — 허들 12 run 9/12 대 P0m 7/12 (금지 접촉 run 2 → 1), 기존 9 run 9/9 대 9/9, 1345 오렌지 · 기둥 두 개 6 run 1/6 대 2/6. 차이 ≤ 2 run 이라 확정 전.
- 한계: 옆의 남은 사과 seed (값 0) 는 그대로 장애물이라 손가락이 얻는 실제 여유는 1 voxel 안팎.
- 위험: 접촉 여유 (P0m) 가 지운 **이웃** seed 도 양수가 된다. RB-Y1 회귀 211 의 맞닿은 바나나 지운 voxel 2,013 → 4,038. benchmark profile (D1) 은 이웃 접촉을 실패로 세므로 **SafeLIBERO · LIBERO-safety 에서 target 옆 장애물 접촉을 따로 본다.**

### C. 실제 크기 손바닥 구 (`--palm-cover on --palm-cover-margin 5`, palm shell 대체)

- RB-Y1: 손바닥 collision mesh 전부를 담는 466 구 (palm shell 606 구 대신). 덮지 못한 mesh 표본 24 / 24,862 → 0 / 35,287, "27 mm 부족" 경고 사라짐, TrajOpt 풀이 중앙 · p95 10–20 % 단축 (offline CPU). `--palm-cover-margin 5` 는 palm shell 과 같은 5 mm 여유를 palm cover 행에 거는 새 flag (D27).
- closed-loop (1345 오렌지 · 기둥 두 개, 6 run): 성공 2/6 대 2/6, 손바닥 몸체 접촉 0 대 0 — **성공 · 접촉 이점은 안 나타났다.** 이점은 모델 정확성 (손바닥 덮개 빈틈 없음) 과 풀이 시간.

근거: development `DECISIONS_20261104_3a_KO.md` D27–D43, `T96.task.md` · `T96c.task.md` · `T96.impl.md` (T96–T96e 절), `T96.verify.json`; outputs `2026-11-04_T96_T97_target닫힌부피_맞닿은경계_재생/`, `2026-11-04_T96e_부호수정_closedloop/`, `2026-11-04_T96_손바닥만비교_1345오렌지_기둥두개_closedloop/`.

## 2. Panda 로 옮길 때 꼭 확인할 것

1. **새 flag 를 Panda 서버 flag 목록에 분류한다.** 지금 `AG3S/adapters/panda/server.py` 의 `UNPORTED_FINAL_CONFIG_DESTS` · `PORTED_FINAL_CONFIG_DESTS` 어디에도 없다. A · B 의 flag 는 `serve_safe.target_exclusion_section` 의 esdf 키로 가므로 (`surface_seed_ats_gate` 와 같은 길) 적용될 것으로 보이지만, **조용히 빠지지 않는지** 기록 키로 확인한다:
   - A: `target_exclusion.attention_surface.solid` (부피 cm³, `R⊥` 점 수, 길별 지운 seed 수, τ 보호 수, `n_removed_shadow`, 카메라별 기여).
   - B: `stats.target_free.oracle_split.removed_sign_free.applied == true`, `n_flipped`.
   - `surface_record` 에 새 키를 더한다.
2. **C 는 Panda 에서 그대로는 안 된다.** `palm_cover` 는 Panda 미이식 (`profile.UNPORTED_BUILD_KEYWORDS`, server 의 UNPORTED 목록) 이고, Panda 손은 franka.yml 구 (`panda_hand` + 손가락 22 구) 를 쓴다. 먼저 **지금 Panda 손 구가 `panda_hand` collision mesh 를 얼마나 못 덮는지** (덮지 못한 표본 수 · 최대 부족 mm) 를 잰다.
   - 부족이 거의 없으면 C 는 Panda 에서 해당 없음 → ablation 은 A × B 2 × 2 로 줄이고 그렇게 기록한다.
   - 부족이 있으면 `panda_hand` mesh 를 담는 덮개 구를 Panda 판으로 만들고 (RB-Y1 palm cover 와 같은 생성 규칙, 기본 꺼짐), 여유는 Panda 의 지금 손바닥 여유와 같게 둔다.
3. **A 의 받침면 · z_floor:** Panda 는 탁자 윗면이 base z −0.012 m, ESDF 하한 −0.10 m. 기록의 `R⊥` 점 수 > 0, 부피가 물체 크기 수준인지 몇 요청에서 먼저 본다 (받침면을 못 찾으면 solid 가 아무것도 안 한다).
4. **A 의 그림자 카메라:** Panda 는 head · left_wrist 만 실제이고 right_wrist 는 dummy. 카메라별 기여에서 dummy 가 0 인지 본다.
5. **flag 조합:** 닫힌 물체 판은 `--surface-hull-fill` · `--surface-contact-allowance-mm` · `--surface-memory-target-sticky` 와 같이 줄 수 없다 (시작 거절). P0m 판에서 이 둘을 **빼고** 넣는다.
6. B 의 효과는 1 voxel 안팎이므로, Panda 손 층 미세 tier 의 voxel 크기를 기록에 같이 적는다.

## 3. Ablation 설계 (사용자 지시, benchmark 세션이 결과 전에 고정)

기준 = benchmark 의 지금 P0m Panda 판 (benchmark 세션이 어느 판 — `AG3S-final-panda-p0m` 또는 SL7b 의 TrajOpt 보정 판 — 을 기준으로 쓸지 먼저 정하고, 8 판 모두 같은 기준 위에서). D1 (이웃 허용 끔) 그대로.

| # | 판 | A 닫힌 물체 | B 부호 수정 | C 실제 크기 손바닥 | 서버 인자 (기준 P0m 에서) |
|---|---|---|---|---|---|
| 1 | P0m (기준) | – | – | – | 그대로 |
| 2 | + B | – | ✓ | – | `+ --surface-removed-sign-free` |
| 3 | A | ✓ | – | – | `− --surface-hull-fill − --surface-contact-allowance-mm 10` `+ --surface-solid-owned --surface-solid-protect-mm 5 --surface-solid-dilate-mm 0 --surface-solid-dn-ever-target --surface-solid-shadow-mm 80` |
| 4 | A + B | ✓ | ✓ | – | 3 + `--surface-removed-sign-free` |
| 5 | + C | – | – | ✓ | 손바닥 판 (§2-2) |
| 6 | + B + C | – | ✓ | ✓ | 2 + 손바닥 판 |
| 7 | A + C | ✓ | – | ✓ | 3 + 손바닥 판 |
| 8 | A + B + C (전부) | ✓ | ✓ | ✓ | 4 + 손바닥 판 |

- 2 × 2 × 2 전체 조합이라 각 수정안의 단독 효과 (1→2, 1→3, 1→5) · 전부에서 하나씩 뺀 효과 (8→7, 8→6, 8→4) · 상호작용 (A 와 B: 1·2·3·4) 을 모두 읽는다.
- `--surface-solid-dn-support exclude` 는 기본값이라 적지 않는다. A 는 RB-Y1 재생에서 가장 나았던 판 (닫힌 부피 δ 0 + 그림자 80 + 옛 target 칸 제외) 하나만 쓴다. δ · 그림자 길이를 결과 보고 바꾸지 않는다.
- 순서 제안: C 는 Panda 이식 (§2-2) 이 먼저라, **1–4 (A × B) 를 먼저** 세 벤치마크에서 돌리고, C 이식 뒤 5–8.
- 장면 · 규모: 각 벤치마크의 benchmark 세션 고정 장면 (VLA 단독 성공 장면, 약 10 episode), 같은 task · episode · seed 를 8 판에 짝지어.
- 주 지표: 성공 (각 벤치마크 정의) + 금지 접촉 (target · 목적지 밖 접촉, D1). HOLD · 정지 증가는 보호로 세지 않는다. 보조: 손 층 기록 (§2-1), 실패마다 ATS · HSR · LTE · CCR 원인 위치.
- 판정 규칙 (benchmark 세션이 결과 전에 확정 · 제안): 한 판이 P0m 을 대신할 후보가 되려면 세 벤치마크 모두에서 성공 ≥ P0m · 금지 접촉 ≤ P0m 이고, 적어도 하나에서 차이 > 2 episode. 아니면 P0m 유지. 최종 선택은 사용자.
- 결과는 benchmark checkout `outputs/<날짜>_<실험>/` (영상 · 표 · README), 숫자는 verify json.

## 4. development 쪽에서 아직 정해지지 않은 것

- RB-Y1 에서 B 의 seed 2 반복 (허들 12 + 9 run, 약 70 분) — 사용자 판정 대기.
- T97 (맞닿은 경계) 를 B 와 함께 다시 잴지, 기둥 두 개 장면의 쥐기 자세 (GIK) 확인, 받침면 guard 높이 (우선순위 7 (b)).

## 5. 커밋

- 코드 · test: `d5a1142e` (feature/ag3s-development)
- 문서 · figure: (아래 커밋 뒤 기입)
- main merge: (기입)
- development 단위 test (2026-11-04 17:0x KST, `.venv-openpi-live`, 한 파일씩, CPU): `test_t96_solid_owned` 31 · `test_t96c_solid_shadow` 15 · `test_t96d_dn_ever_target` 12 · `test_t96e_removed_sign_free` 11 · `test_t97_crease_protrusion` 12 (+1 skip) · `tests/trajopt/test_t96b_palm_cover_margin` 12 · `test_scm_palm_shell_margin` 12 · `test_t78b_reattach_hull_fill` 14 · `test_t78f_contact_allowance` 9 — 전부 passed.
