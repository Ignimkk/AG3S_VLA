# SL5j-G — 쥔 뒤에도 손을 위로 미는 제약의 출처 (제약 기록 run 측정)

> writer: lead · 읽는 쪽: verifier · 작성 2026-10-23 KST
> 사용자 판정 2026-10-23: A안. 쥔 뒤에 위로 미는 원인을 먼저 진단한다. **진단만 하고, 고치는 것은 결과를 본 뒤 따로 정한다.**
> 이름 규칙(사용자 2026-10-22): 보고와 figure에서는 바꾼 내용으로 부른다. 대상 version은 **파지 인식 + closing 면제판**(내부 key V11AB-pb7)이고, 역변환은 C(`lag_inverse_tracking`)다.

## run

- 파지 인식 + closing 면제판 × {SafeLIBERO LI dev 20, LIBERO 10}
- launcher의 `--record-constraints`(SL5j-launcher)를 켰다. 기록 위치는 각 run의 `server_constraints/`이고, 청크당 npz 하나다.
- 비교 기준: 같은 code와 flag로 기록 없이 돈 SL5i-G run
  - `20261023T024340_1ddb93768ac1_sl5i-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb7-C`
  - `20261023T032837_1ddb93768ac1_sl5i-libero10-init0-vla_v11ab-V11AB-pb7-C`

## 무엇을 재나 (→ verifier, Opus)

| # | 측정 |
|---|---|
| J0 | **기록이 실행을 바꾸지 않았나.** SL5i-G run과 episode npz · calls npz의 모든 array, episodes.jsonl 결과 field를 비교한다. 다르면 처음 달라진 episode · call · array를 찾고 그 자리에서 멈추고 보고한다. 다르면 이후 측정이 SL5i-G 결과를 설명하지 못한다. |
| J1 | **위로 미는 행의 주인.** 대상 call은 MuJoCo 기준 bowl이 들린(lifted) 상태에서 TO가 row를 바꾼 call이다. latch가 held인 call과 closing인 call을 나눠서 본다. 각 call에서 다음을 한다. ① 기록된 clearance(H × S)에서 위반하거나 margin 안에 있는 (step, sphere)를 고른다. ② 각 sphere의 link 이름과 종류를 정한다. 종류는 손가락 · 손바닥 · arm, 그리고 쥔 물체 구(attached / held cover)를 구분한다. ③ 그 sphere 주변 거리장에서 가장 가까운 점유 voxel 위치를 찾는다. 기록된 ESDF 계층에서 찾고, 방법과 반경은 적는다. ④ 그 voxel을 같은 obs의 MuJoCo 상태(sim_state_at_return)에 대어 가장 가까운 geom을 찾는다. ⑤ 다음 class로 센다: **(a) 잔상** — 30 mm 안에 MuJoCo geom이 없고, 쥐기 전 bowl 자리 근처에 있음. 기준 거리는 적는다. **(b) 쥔 bowl 자체** — self-filter가 지우지 못한 경우. **(c) 받침면 / table**. **(d) 목적지**(plate 등). **(e) obstacle**. **(f) 그 밖의 물체**. ⑥ 각 행이 미는 방향(EE z 성분 부호)을 class별 위/아래로 센다. |
| J2 | **over-lift episode별 요약.** 대상: 파지 인식 + closing 면제판의 LI over-lift 7건과 LIBERO에 해당하는 것. 위로 민 행의 class 분포, 시간 순서(처음 위로 밀기 시작한 call의 class), 같은 시점의 AG3S manipulated centroid와 실제 bowl 위치, 쥔 물체 구의 위치. |
| J3 | **대조.** 성공 episode(LI t0e0 · t0e3 · t1e1, LIBERO 성공 6건)에서 같은 측정을 하고, 거기서는 왜 위로 밀지 않았는지 숫자로 낸다. |

- 판정 문장은 쓰지 않는다. 가설마다 지지 · 반박하는 숫자만 둔다. 가설:
  - H-ghost: 쥔 bowl의 옛 자리 잔상
  - H-self: 쥔 bowl 자체가 field에 남음
  - H-support: 받침면 · 목적지
  - H-attached: 쥔 물체 구가 받침에 막힘
- figure (`AG3S/ag3s/docs/figures/sl5j-*` + `.json`, **이미 있는 이름이면 멈추고 새 이름을 정한다**):
  1. 실제 scene: 대표 over-lift call. 손과 쥔 물체 구, 위반 sphere, 가장 가까운 점유 voxel, 실제 bowl, AG3S centroid를 같은 좌표에 겹친다. ESDF 단면이면 어디를 어느 방향에서 잘랐는지 배치도를 함께 둔다.
  2. graph: class별 위/아래 행 수, episode별 timeline.
  3. table: J0 · J1 · J2 · J3.
- 산출물:
  - raw: `outputs/verify/SL5j-G/`
  - verify.json: `AG3S/ag3s/docs/handoff/SL5j-G.verify.json`

## 규칙

- code 수정 · commit · GPU 금지.
- CPU worker ≤ 2, RSS ≤ 3 GB. 기록 npz는 하나씩 열고 바로 닫는다.
- working set이 62 GiB를 넘으면 멈춘다.
