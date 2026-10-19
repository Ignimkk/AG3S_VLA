# SL5h-G — GPU counterfactual 결과 측정 (pb5 × 변환 A / B / C, 병렬 bit 재현)

> writer: lead · 읽는 쪽: verifier · 작성 2026-10-19 KST
> run code: `743a1e76`. 실행 경로는 `a8ad92ef`와 같고, 바뀐 것은 launcher의 병렬 option(`e2b5deaa`)과 docs뿐이다.
> 실행 방식: `benchmark/runners/run_parallel_gpu.sh`, 동시 실행 ≤ 2. log는 각 run의 manifest `launcher_result`(`gpu_lock_mode=held_by_caller`)와 lead scratchpad의 `sl5h_g.log`에 있다.

## 배경 (SL5h.verify.json)

- 포화는 TO가 바꾼 row에서만, 역변환 단계에서 잘린다.
- lag-inverse 계열(A · C)은 over-lift가 3~10건이다. B와 pb1-B는 0건이다.
- 다만 비교에 쓴 B run에는 MI gate · arms · pb4/pb5 규칙이 없었다. 그래서 같은 pb5에서 변환만 바꿔 다시 돌렸다.

## run (`/mnt/dev/AG3S_VLA-storage/results/benchmark/`, 이름에 `sl5h-` 포함)

| job | 조건 | 비교 기준 |
|---|---|---|
| repro | pb5-C LIBERO 10, **SL5g run을 병렬로 다시 실행** | `20261018T150153_a8ad92ef3356_sl5g-libero10-init0-vla_v11ab-V11AB-pb5-C` |
| pb5-B | LIBERO 10 · SafeLIBERO LI dev | SL5g pb5-C 두 run, VLA-only (SL5f) |
| pb5-A | LIBERO 10 · SafeLIBERO LI dev | 〃 |

## 무엇을 재나 (→ verifier)

| # | 측정 |
|---|---|
| G0 | **병렬 bit 재현.** repro run과 SL5g pb5-C LIBERO 10을 episode npz, calls npz의 모든 array에서 비교한다. episodes.jsonl의 결과 field도 비교한다. 다르면 처음 달라진 episode · call · array를 찾는다. 같은 시간대에 동시에 돈 job도 기록한다(`sl5h_g.log`). |
| G1 | **집계.** 집합마다 VLA-only · pb5-A · pb5-B · pb5-C의 official / collided / safe / HOLD. 전이표(같은 task · init · seed). 후보 규칙 (a) official F→S, (b) safe F→T는 따로 센다. |
| G2 | **SL5h 서명.** pb5 × A/B/C마다 SL5h와 같은 정의로 잰다: over-lift 수, lifted step의 실행 포화율, D*_z median, realised median, closing·lifted call의 row 위/아래 비율, closing·lifted 중 reach=False 비율. SL5h의 script(`outputs/verify/SL5h/scripts/`)를 재사용한다. |
| G3 | **실패 원인 분류.** VLA-only 성공 → pb5-B 실패 episode에 SL5g #3 분류(A/B/C/D)를 적용하고, pb5-C와 나란히 둔다. |
| G4 | **사례.** 같은 episode에서 pb5-C가 over-lift로 실패하고 pb5-B가 성공했거나, 같은 episode에서 VLA-only가 실패하고 pb5-B가 성공한 경우를 1~2건 묶는다. 묶음은 SL5g 형식(case.json, calls csv, frame)을 따르고 `outputs/verify/SL5h-G/cases/`에 둔다. 없으면 0을 숫자로 남긴다. |

- 판정 문장은 쓰지 않는다. H1 · H2 · H3별로 지지 · 반박하는 숫자만 정리한다.
- figure: `AG3S/ag3s/docs/figures/sl5h-g-*` + `.json`
  - 실제 scene: 같은 episode의 C vs B 최고점 frame
  - graph: 변환별 bar와 서명
  - table: 전이표, G0 결과
- 산출물:
  - raw: `outputs/verify/SL5h-G/`
  - verify.json: `AG3S/ag3s/docs/handoff/SL5h-G.verify.json`

## 규칙

- code 수정 · commit · GPU 사용 금지. CPU worker ≤ 2, RSS ≤ 3 GB.
- 시작 전과 중간에 working set을 확인한다. 62 GiB를 넘으면 멈추고 보고한다.
