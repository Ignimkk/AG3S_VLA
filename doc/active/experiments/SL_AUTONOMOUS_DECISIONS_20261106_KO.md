# SL15 자율 판정 기록 — 2026-11-06 00:3x ~ 11:00 KST

> writer: lead (benchmark checkout).
> 사용자 지시(2026-11-05 23:3x KST): 오늘 한 일과 결과를 보고하고, 제안한 방법 중 더 권장하는 방법으로 수정하라. 오전 11시까지 이전처럼 스스로 판정하며 수행하고, 작업 내용과 판정 내용을 문서에 적어라.
> 어제 기록 · 하루 보고: `SL_AUTONOMOUS_DECISIONS_20261105_KO.md` 끝 절.

## 출발점 (00:3x KST)

- **원인.** SL14에서 장애물 없는 LIBERO 손실의 주 원인 층은 명령 변환 C(지연 역변환)다.
  - 변환 B: LIBERO 19/27(pc판 10), LIBERO-Safety 8/10(pc판 1, 위반 0)
- **두 가지 제안.**
  - (가) 변환 자체를 고친다.
  - (나) 변환 B + latch 수정 조합에 ATS 분석을 더한다.
- **(가)를 고른 이유.**
  - 원인 층(CCR)에서 고친다는 규칙에 맞는다.
  - 변환 B는 OSC 지연을 무시해 TrajOpt 수정을 정상상태에서 약 25%만 실현한다. 그래서 장애물이 경로를 막는 장면에서 보호가 약해질 수 있다.
  - C의 dead-beat(이득 14 · 32)와 B(이득 1) 사이에, 응답 모형의 정상상태 역이득((1−α)/β ≈ 4)을 쓰는 변환 D가 물리적으로 맞는 중간이다.
- 규칙은 결과 전에 `AG3S/ag3s/docs/handoff/SL15.task.md`에 고정했다.

## 판정 기록

| # | 시간 (KST) | 결정 | 근거 | 대안 · 되돌리기 | 자료 |
|---|---|---|---|---|---|
| F0 | 00:3x | **SL15 고정 · 시작.** 변환 D(정상상태 역이득, 방향 보존 clip)와 E(D + 무시 구간)를 새 mode로 implementer에게 맡겼다(worktree `wip/sl15-convD`). 그동안 새 코드가 필요 없는 변환 B의 SafeLIBERO 11 기준 run(KB-SAFE)을 GPU에서 시작했다. chain은 queue가 빌 때 STOP 파일이 생길 때까지 기다린다 | 변환 B의 장애물 장면 안전 비용이 판정 비교에 필요하다. SafeLIBERO 11은 VLA 단독 충돌 9/11 · pc판 5/11이라 보호 차이가 드러나는 장면이다 | 기존 판 · flag 변경 없음 | `SL15.task.md` |
| F1 | 01:0x | **사용자 질문(HSR · 작업 규칙 수정 계획)에 답하고 SL15 §5로 고정했다.** 변환을 고친 뒤 HSR을 다시 잰다. DS = D + 개도 · 명령 attach/detach(`blocked` 등 네 검사 대체, PLACED 흡수 대신 다시 쥐기, 회수는 빈손 닫힘만). DH = D + HSR6 다섯 규칙 끔. reach 실패가 5%를 넘으면 DR(가까운 기억 물체로 reach)을 맡긴다. HOLD 판정 규칙(미인증 · 충돌)은 안전 규칙이라 바꾸지 않고 남은 HOLD 실패만 센다 | SL14에서 HSR 규칙 변경은 기제는 통과했지만, 변환 C 증폭에 가려 성공 차이가 작았다. 변환을 고친 뒤에야 HSR 규칙의 효과를 가를 수 있다. DS · DH는 새 코드가 없다(기존 판 + `--conversion D`) | 순서: D → DS → DH → 조건부 E · DR | `SL15.task.md` §5 |
| F2 | 01:1x | **변환 B의 SafeLIBERO 11 안전 기준(KB-SAFE): 성공 3/11, 충돌 9/11.** pc판은 성공 1 · 충돌 5, VLA 단독은 성공 11 · 충돌 9다. 고정 조건 (c)(충돌 ≤ 5)를 넘지 못하므로 **변환 B는 후보가 아니다.** 충돌 없이 성공한 episode는 VLA 2 · pc판 1 · B 1이다. 같은 시각 `wip/sl15-convD`(c892d461)를 받았다. 새 test 36개와 SL14 test 74개가 통과했다. 남은 `test_rby1_golden_identity` 실패는 merge 전(2d3a736c)에도 같다. lead 문서를 commit한 뒤(07463672) GPU가 빈 때 merge했다(816f7aa4). D 세 장면과 DH를 queue에 넣었다 | B는 TrajOpt 회피 수정을 정상상태 25%만 실현해 장애물 보호를 잃는다. SL14에서 세운 가설이 맞았다. 지연을 정상상태 이득으로 되돌리는 D가 필요하다는 근거다 | B는 후보 목록에서 뺀다 | `outputs/2026-11-06_SL15_변환D/` |
| F3 | 01:5x | **변환 D LIBERO 27: 15/27(6 · 4 · 5), 고정 기제 미달 → "의도대로 동작하지 않음"(성공은 읽지 않는다).** 모든 call이 D(876)다. 실제 쥔 call(HOLD 제외)의 포화 비율은 0.44로 문턱 0.3을 넘는다(pc판 0.91, B 0.19). **call 표 원인:** 포화는 변환 이득이 아니라 TrajOpt가 경로를 cm 단위로 바꾸는 데서 온다. 실제 쥔 call의 누적 EE 변화는 중앙값 3 mm · 상위 10% 105 mm이고, CLOSING은 중앙값 15 mm · 상위 10% 68 mm다. 수정 방향은 거의 모두 위(+z 17–47 mm)다. 쥐기 전 접근(LATCHED)에서도 +19 mm다. latch가 CLOSING에 머문 채 실제로는 들고 · 옮기고 · 놓는 call이 99개다(attach 누락 → 쥔 bowl이 장애물로 남음). HELD 180 call 중 179개에서 큰 수정이 나고, 부착 구가 탁자에 닿는다(E21). **그래서 E(무시 구간)는 돌리지 않는다.** 무시 구간에 드는 call은 실제 쥔 call의 6%, CLOSING의 38%뿐이라 기제를 바꿀 수 없다. 대신 (1) 고정해 둔 DS(D + 개도 attach · detach)를 앞당긴다. (2) D에 제약 기록을 켠 진단(DREC, LIBERO seed 8, 행동 bit 동일 확인됨 J0)을 돌려 큰 수정을 일으키는 MuJoCo geom(target · 탁자 · 쥔 물체 · 그 밖)을 J1 도구로 가린다 | 변환은 수정량을 얼마나 실행하느냐만 정한다. B가 LIBERO에서 잘 된 것은 잘못된 수정을 25%만 실행해서고, SafeLIBERO에서 보호를 잃은 것도 같은 이유다. 남은 원인은 기하 층(LTE · HSR)의 거짓 장애물이다 | E 생략은 task §2의 문구와 다르다. 근거는 call 표(`data/calls_Dg.jsonl.gz`, `scripts/rowdev.py` · `devdir.py`)다 | `outputs/2026-11-06_SL15_변환D/` |
| F4 | 02:0x | **가설 정정(기록 전).** 위로 미는 원인이 ESDF 기본 여유 50 mm(`--esdf-margin` 기본값)일 거라 봤지만 틀렸다. 실제 서버 인자는 `--esdf-margin 0`, `--obstacle-margin 0.010`(target · 받침면 제외), `--lookahead obstacles --lookahead-gate-mm 20`, `--plan-horizon 16`이다. 어제 pc판 J1(`j1_geom_pc-base`)에서도 쥔 뒤 큰 편차 call 7개의 실행 창 위반 행은 0이다. 그래서 남은 후보는 셋이다. (i) 계획 꼬리(실행 창 뒤 row)의 제약 · lookahead, (ii) 목적 항(연속성 · 꺾임 변화율 · HSR6 목적), (iii) margin 안 행. DREC 기록으로 계획 16 row 전체의 class별 최소 clearance와 band 20 mm 행을 센다 | 기록에 없는 가설을 원인으로 쓰지 않는다 | – | – |
| F5 | 02:1x | **D LIBERO-Safety 10: 4/10, 위반 0 → 후보 조건 (b)(≥ 6) 미달.** pc판 1 · B 8이다. 비권한 link의 target 여유 규칙을 바꾸는 판 DT · DST를 task §6에 결과 전 고정했다. 등록(`wip/sl15-ntm` 1e468a9a)은 implementer가 했다. Panda 서버가 이 flag를 미이식으로 막고 있어 adapter에 이식했다(기존 66판 argv sha256 동일, test 12 + SL14/15 86 통과). 적용 대상은 팔 link1–7의 37행뿐이다(손 · 손가락 · 껍질 135행은 권한 있음). merge는 SafeLIBERO run이 끝나 chain이 빌 때 한다(queue 보류) | 50 mm는 RB-Y1 기본 `geometry.safety_margin`이다. 탁자 위 bowl을 위에서 쥐는 Panda에서는 손목 쪽 link가 bowl 50 mm 안에 들어갈 수 있다 | 되돌리기: 새 판만 추가, 기존 판 불변 | `SL15.task.md` §6, `SL15.ntm.impl.md` |
