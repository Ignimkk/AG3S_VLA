# CLOSED-LOOP-FIX — pilot 전 3 arm 의 문제 수정 (lan_o3dp action 크기 · knows 제약 강제 · omniguide skip / 재compile)

> writer: lead · 읽는 쪽: implementer K → verifier V5 · 작성 2026-10-24 KST
> 사용자 판정 2026-10-24: "먼저 수정 후 pilot" — 아래 3 문제를 CPU 로 고친 뒤, SafeLIBERO spatial L I dev 20 ep × 8 arm pilot 을 돌린다.
> 근거: `CLOSED-LOOP-METHODS-V3.verify.json` issues, `AG3S_BASELINE_LOG.md` BL-6. commit `7a89ac2b`.

## 고칠 것 (각각 원인을 먼저 확인하고, 원 방법의 정의에 맞춰 고친다)

1. **lan_o3dp_guidance_adapted** — served chunk 의 |a| 가 최대 27.28 이다(nominal 0.96). 실행된 183 행이 |xyz| > 1 이다.
   - 의심 지점: normalized ↔ raw 변환, step_m 누적, 범위 제한 없음.
   - 원 Lan-o3dp 의 guidance 크기 정의와 LIBERO action 범위([-1, 1], `pi05_libero` norm stats)에 맞춘다.
   - test: guidance 뒤 chunk 가 action 범위 안에 있고, 장애물이 없으면 identity 다.
2. **knows** — h_min −0.447, h_neg_steps 419 인데 qp_fail 0 이다.
   - 확인할 것:
     - 제약이 slack 이나 relaxed normal 로 사실상 풀렸는가 (KNOWS `docs/16` 의 알려진 성질)?
     - h 기록 시점과 QP 시점이 맞지 않는가?
     - 시작부터 h<0 인 경우(이미 안쪽)의 처리가 원 논문과 같은가?
   - 원 논문 정의에 맞추고, 의도된 동작이면 그 근거를 impl.md 에 쓴다. 기록 정의(h 를 nominal 기준으로 보나 filtered 기준으로 보나)도 명확히 한다.
3. **omniguide** — 56 call 중 38 call 에서 guidance 를 건너뛰었고, jit 재compile 이 10 회 일어났다.
   - skip 사유(freeze zone / 피할 것 없음)가 원 OmniGuide 의 규칙과 같은지 확인한다.
   - 재compile 은 장면을 traced 인자로 바꾸거나 shape 를 padding 해 없앤다(D 의 지적: closure 가 장면을 상수로 잡는다). lan_o3dp 의 재compile 7 회도 같은 방식으로 줄인다.
   - test: 연속 call 에서 compile 이 1 회다.

추가로, guidance arm 의 diagnostics 시간을 `ms.serve` 에 포함할지 별도 key 로 기록할지 정해서 일관되게 한다. 지금은 client 지연과 0.3–1.4 s 차이가 난다.

## 수용 기준

- 기존 test 가 모두 통과하고, `none` · filter identity 의 bit 동일성이 유지된다.
- 고친 3 arm 을 CPU replay(`replay.py`, V3 의 replay 입력)로 다시 돌려 다음을 보인다:
  - 문제 지표가 사라졌다: |a| ≤ 1, h 강제, compile 수.
  - 다른 arm 의 출력은 바뀌지 않았다.
- 원 방법과 달라지는 점은 README 와 impl.md 에 쓴다.
- 산출물: `CLOSED-LOOP-FIX-K.impl.md`. 내용: 원인, 수정, test, V5 명령.

## 범위 · 제약

- 수정 범위: `Baseline/paper_baselines/closed_loop/**`, `methods/{lan_o3dp_guidance_adapted,knows,omniguide,guidance_common}/**` 와 해당 tests.
- 수정 금지: `benchmark/**`, `AG3S/**`, `third_party/**`, SafeLIBERO submodule.
- GPU 금지. CPU worker 1개. commit 금지. package 설치 금지.

## V5 (K 이후, GPU)

1. 고친 3 arm 의 GPU smoke(t0e0)를 다시 돈다.
2. 이어서 **pilot**: SafeLIBERO spatial L I dev 20 ep(SL5c 와 같은 episode · seed) × 8 arm, `run_parallel_gpu.sh` 방식 2 job.
   - 시작 전에 `ag3s-vla-benchmark-1a`, `ag3s-vla-development-ed` 에 알리고, 공용 파일 수정이 없는지 확인한다.
3. 지표: arm 별 TSR / CAR / ETS(공식), SL5c safe success, KNOWS SR / CR / SSR, multi-link 11 mm, 개입률, latency.
   - SL5c 의 VLA-only · V11AB 기록은 같은 protocol 일 때만 참조 행으로 둔다.
   - paired 비교(arm vs none)를 낸다.
4. figure (`figures/baselines/closed_loop/`, `v5pilot-` 접두어):
   - 실제 장면: arm 별 대표 성공과 실패 rollout strip
   - 핵심 기능: episode 별 개입 · h 와 충돌 시점
   - 표: arm × 지표, paired
5. 산출물: `CLOSED-LOOP-FIX-V5.verify.json`
