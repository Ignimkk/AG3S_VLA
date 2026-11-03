# SL7f — 표면 seed를 ATS target에 맞추기 (`--surface-seed-ats-gate`): 오프라인 SafeLIBERO 재생 + LIBERO 9

> writer: lead (benchmark checkout), 2026-11-04 05:5x KST. **결과를 보기 전에 고정.**
> 판정 주체: lead 자율 판정 D4 (사용자 지시 "11:30까지 스스로 판정") — `doc/active/experiments/SL7_AUTONOMOUS_DECISIONS_20261104_KO.md`.
> 앞 단계: SL7e — pc판 + 표면 추출 3/9, pc판 − HSR6 2/9 → Panda 후보 뼈대 = pc판(HSR6 유지) + 고친 표면 추출. SL7c ② — 표면 seed가 장애물 위(4/6) · 표면은 맞는데 ATS와 어긋나 버려짐(2/6).

## 기능 (새 flag, 기본 꺼짐)

`--surface-seed-ats-gate` → `esdf.surface_seed_ats_gate`. 표면 선택 직전에 카메라마다 attention 지도를 **ATS target 점이 투영되는 patch에만 남기고 나머지는 0**으로 만든다(`attention_surface_post.ats_gate_views`). ATS 점이 없거나 한 점도 그 view에 들어오지 않으면 그 view는 그대로 둔다. `--target-extent attention-surface`와 함께만 쓴다. 기록 `t78.seed_ats_gate`.

## 판

| 판 | 정의 | 용도 |
|---|---|---|
| `AG3S-final-panda-p0m-gate` | P0m + gate | 오프라인 재생 전용(저장된 L8H6 noun row로 재생 가능) |
| `AG3S-final-panda-v3-e0to-pc-surface-gate` | pc판 + 표면 추출(L8H2 seed) + gate | **Panda 최종 후보** closed loop |

## 1. 오프라인 SafeLIBERO 재생 (shared GPU, 20 episode · 796 call, SL7 재생과 같은 입력)

지표: GT 기준 target 지움(평균), obstacle 잘못 지움, 표면 영역 채택 call 수, 지운 voxel 0인 call 비율. 비교: P0m(0.102 · 0.001 · 77 call · 41%), 1판 공(0.104 · 0.003).
규칙: 원인 기록용이다(판정으로 closed loop을 막지 않는다). 단, obstacle 잘못 지움이 1판(0.003)보다 커지면 README에 경고로 적는다.

## 2. LIBERO 9 closed loop — `pc-surface-gate`

장면 · 지표는 SL7e와 같다. 비교: pc판 6/9(포화 10%), pc판 + 표면 추출 3/9(19%).
1. ≥ 5/9 이고 포화 ≤ 15% → 후보로 **SafeLIBERO spatial Level I dev 11 episode** closed loop에 넘긴다(SL7 규약 §2. 비교 pc판 SL6-H: 안전 성공 1 · 공식 성공 1 · 충돌 5, VLA 단독 2 · 11 · 9).
2. ≤ 3/9 → gate가 표면 추출의 손실을 고치지 못한다 → 기록하고 다음 수정(카메라별 정규화 등)을 판정한다.
3. 4/9 → 가르지 못함. SafeLIBERO로 넘기지 않고 원인을 본다.
