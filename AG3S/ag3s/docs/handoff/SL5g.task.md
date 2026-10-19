# SL5g — V11AB-pb4 · pb5 수정본(`f4f97337`) 재검증: pairing · 규칙 적용 확인 · 실패 원인 · 사례 묶음

> writer: lead · 읽는 쪽: verifier (다음에 scribe) · 작성 2026-10-18 15:40 KST
> 실행 code: **`a8ad92ef`** (code는 `f4f97337` 수정본과 같고, `a8ad92ef`는 SL5f docs만 더한다). 분석은 CPU만 쓴다. **GPU 금지**
> (development 세션이 ~20:30 KST까지 GPU를 쓴다). 새 rollout이 필요하다고 판단되면 실행하지 말고 이유를 적어 lead에 돌려준다.

## 배경 (log SL5e · SL5f 절)

- pb3 = pb2 + `--links arms`. 충돌은 줄었지만 contact 단계 과잉 제약 때문에 task success가 무너졌다.
- pb4·pb5는 contact 단계 규칙이다: held exemption, latch contact phase, grasp zone. 정확한 flag 구성은 `AG3S/adapters/panda/profile.py`의 `AG3S_VERSIONS`를 직접 읽어 확인한다.
- SL5f 초기 구현(`38722284`)에는 결함 두 개가 있었다:
  - authorized mask가 손가락만 포함했다 (palm 누락).
  - `grasp_contact` 기록이 늘 `{}`였다.
- `f4f97337`에서 고쳤고, smoke(t0·t3)는 통과했다.

## 실행한 run (lead, 2026-10-18 13:33 ~ 15:26 KST, 모두 rc=0)

매 mode 전 guard에서 HEAD = `a8ad92ef`, clean tree를 확인했다. 기록 위치는 각 run의 `logs/sl5g_chain.log`·`logs/sl5g_chain.sh`다.

| 집합 | 조건 | run (`/mnt/dev/AG3S_VLA-storage/results/benchmark/…`) |
|---|---|---|
| SafeLIBERO LI dev (t0–3 × e0–4) | pb5-C | `20261018T133339_a8ad92ef3356_sl5g-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-C` |
| 〃 | pb4-C | `20261018T141823_a8ad92ef3356_sl5g-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb4-C` |
| LIBERO 10 (t0–9 × init 0) | pb5-C | `20261018T150153_a8ad92ef3356_sl5g-libero10-init0-vla_v11ab-V11AB-pb5-C` |
| 〃 | pb4-C | `20261018T151346_a8ad92ef3356_sl5g-libero10-init0-vla_v11ab-V11AB-pb4-C` |

### 비교 기준 (같은 task · init · policy_seed)

- **VLA-only — 재사용, 재실행하지 않음 (사용자 지시 2026-10-18):**
  - LI: `20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only`
  - LIBERO 10: `20261017T170230_38722284e2fa_sl5f-libero10-init0-vla_only`
  - 재사용 근거는 각 새 run의 `sl5g_pairing_basis.json`과 `vla_only_identity.json`에 있다. script는 `logs/vla_only_identity.py`다.
  - 근거 요약: `38722284`→`a8ad92ef`에서 바뀐 것은 `safe_policy.py`(SafePolicy method)·test·docs뿐이다. `--no-safe` server는 `SeededPolicy`만 쓴다. settings, versions, checkpoint, protocol, dependencies의 diff는 0이다. VLA-only가 SL5d·SL5e·SL5f 세 commit에서 npz 전 array까지 bit 단위로 같다.
- **이전 version (재사용):**
  - pb3-C: `20261017T034100_…_sl5e-…-V11AB-pb3-C`, `20261017T042820_…_sl5e-libero10-…-pb3-C`
  - 초기 pb5-C · pb4-C (`38722284`): `20261017T153442_…`, `20261017T161857_…`, `20261017T170859_…`, `20261017T172203_…`

## 무엇을 재나 (→ verifier, Opus)

| # | 측정 | 내용 |
|---|---|---|
| 0 | pairing 근거 재확인 | 위 `vla_only_identity.py`를 다시 돌려 결과가 같은지 확인한다. 결과는 `outputs/verify/SL5g/`에 복사하고 verify.json `numbers.pairing_basis`에 싣는다. 다르면 그 자리에서 멈추고 lead에 보고한다 |
| 1 | pairing 집계 | 집합마다 VLA-only · pb3 · pb4(초기) · pb5(초기) · pb4 · pb5의 official / collided / safe_success / HOLD calls. episode 전이표(VLA-only → 각 version, 초기 → 수정본). 후보 규칙: (a) official fail→success, (b) safety fail→safe success. 두 규칙을 **따로** 센다 |
| 2 | 수정한 규칙이 실제로 적용됐나 | calls log의 `grasp_contact`(held exemption 행 수, grasp zone 행 수)와 `contact_phase`, minimal intervention gate 적용 수. latch phase(free / closing / latched / held …)별·episode별로 센다. 초기 run에서는 `{}`였다는 것도 숫자로 남긴다 |
| 3 | 실패 원인 (VLA-only 성공 → pb4/pb5 실패 episode 전부) | episode마다 다음을 잰다: (i) 실행 궤적이 VLA-only와 처음 갈라지는 call / env step과 그때의 latch phase, verdict reason, 활성 constraint 행. (ii) grasp 성공 여부와 시점, 들어올린 뒤 놓쳤는지. (iii) HOLD 연속 구간. (iv) 실행된 수정량(`max_abs_action_change_executed`)의 phase별 분포. (v) 종료 status. 분류는 숫자로 정의된 기준으로만 한다 (예: "grasp 전 HOLD 연속 ≥ N call"). 기준과 N을 verify.json에 적는다 |
| 4 | contact 단계에 남은 제약 | SL5e의 `outputs/verify/SL5e/scripts/palm_neighbors.py` 방식으로, 수정본 pb5 LIBERO 10과 LI에서 reference가 violated되어 최적화된 closing / latched / held call의 가장 가까운 물체와 link를 다시 잰다. SL5e 표(pb2-C)와 같은 형식으로 낸다 |
| 5 | 충돌 | LI pb5 충돌 11 vs pb4 7 vs VLA-only 18. 충돌 episode마다 충돌 시점의 latch phase, 충돌한 link, 그 직전 call이 HOLD였는지 / gate에 통과됐는지 |
| 6 | pb4 vs pb5 | 같은 episode에서 결과가 갈린 경우 (예: LIBERO t0e0, t1e0: pb5 성공 · pb4 실패). 처음 갈라지는 call과 그때 두 version의 규칙 차이가 낸 행 |

- **판정 문장을 쓰지 않는다.** 각 가설(예: "grasp zone이 너무 좁다", "minimal intervention gate가 contact 단계에서 수정을 막지 못한다", "arm links가 approach를 막는다")마다 관련 숫자만 정리한다.
- 원인 추정이 필요하면 "가설 · 그것을 지지 / 반박하는 숫자" 형태로 둔다.

## 사례 묶음 (→ verifier)

- 규칙 (a)·(b) 후보가 있으면 각 1건을 묶는다. 없으면 "없음"을 숫자로 남긴다.
- 그와 별도로 다음 두 종류를 각 1~2건 묶는다:
  1. **안전 개선 사례:** VLA-only 충돌 → pb 무충돌. official success 여부도 함께 남긴다.
  2. **대표 회귀 사례:** VLA-only 성공 → pb 실패. #3 분류에서 가장 많은 원인을 대표하는 episode로 고른다.
- 각 묶음에 넣을 것:
  - 두 조건의 전체 video 경로
  - 핵심 image: 갈라지는 시점, grasp, 충돌 / 실패 시점
  - 판정·개입 log 발췌 (call별 latch, verdict, HOLD, 수정량)
  - 재현 명령 (launcher 인자 + task / episode)
- 위치는 `outputs/verify/SL5g/cases/<case>/`. 큰 video는 복사하지 않고 경로만 적는다.

## figure (규칙 A, `AG3S/ag3s/docs/figures/sl5g-*` + `.json` sidecar)

1. **실제 scene:** 대표 회귀 사례와 안전 개선 사례의 갈라지는 시점 frame. VLA-only와 pb5를 나란히 두고, 가장 가까운 물체·link를 표시한다.
2. **graph:**
   - version별 official / collided / safe bar (LI, LIBERO 10)
   - 대표 사례의 call별 timeline (latch phase, HOLD, 수정량, grasp_contact 행 수)
3. **table:**
   - episode 전이표 (VLA-only × version)
   - 실패 원인 분류 집계
   - #4 nearest object 표

## 산출물

- raw · script: `outputs/verify/SL5g/` (script는 `outputs/verify/SL5g/scripts/`)
- `AG3S/ag3s/docs/handoff/SL5g.verify.json`
  - 포함: `step`, `date`, `code_commit`, `runs`, `baselines`, `pairing_basis`
  - `numbers`에 위 #0~#6과 사례 목록
  - `figures`

## 실행 규칙

- code 수정 금지. 측정 script만 새로 쓴다.
- 이 checkout의 `.venv-ag3s/bin/python`을 쓴다 (`source scripts/setup/env.sh`).
- CPU worker ≤ 2, foreground. 큰 npz는 chunk로 읽어 RSS를 작게 유지한다 (working set + 추정 ≤ 64 GiB).
- GPU 사용, 다른 세션 process 종료는 금지한다.
- commit하지 않는다 (lead가 한다).

## 이 STEP이 답하는 질문

결함을 고친 pb4·pb5에서 contact 단계 규칙은 실제로 적용됐는가? 그런데도 VLA-only가 성공한 episode가 실패하는 원인은 어느 단계, 어느 제약이나 규칙인가?
