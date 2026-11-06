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
| F6 | 02:3x | **D SafeLIBERO 11: 성공 2 · 충돌 7 → (c) 미달. 변환 D는 후보가 아니다**(LIBERO 기제 미달 · LIBERO-Safety 4/10 · SafeLIBERO 충돌 7). 같은 시각 lead 문서를 commit하고(8b25971e) `wip/sl15-ntm`을 merge했다(5bccd363, SL14 · SL15 · panda_sl4 test 100 통과). queue를 풀었다(DREC → DS → DT → DST → DH). **읽기:** SafeLIBERO 충돌은 C 5 · D 7 · B 9이고, LIBERO 성공은 C 10 · D 15 · B 19다. TrajOpt 수정을 많이 실현할수록 장애물 장면은 안전하고 장애물 없는 장면은 실패한다. 수정 안에 거짓 회피(장애물 없는 장면에서도 위로 미는 것)가 섞여 있어서다. 변환만으로는 둘을 함께 얻을 수 없다 | 원인 층을 변환(CCR)에서 기하(LTE · HSR) · 제약 규칙으로 옮긴다. DS(attach 누락) · DT(팔 link의 target 여유 50 mm)가 그 후보를 하나씩 가른다 | 변환 C · B · D 모두 후보 아님 | `outputs/2026-11-06_SL15_변환D/` |
| F7 | 02:5x–03:1x | **DREC(D + 제약 기록, LIBERO seed 8) 분석.** (1) J0: D seed 8과 9/9 episode bit 동일(성공 4 = 4). (2) J1(band 20 mm, 수정 > 0.02 rad인 154 call): 쥐기 전에는 손바닥이 target(bowl) voxel에 가깝다(1,740행). 쥐는 중에는 손바닥이 손 자신(`gripper0_right_gripper`) voxel에 가깝다(2,913행). 쥔 뒤에는 손바닥 · 손가락이 접시 · 탁자 · 손에 가깝다. **팔 link 행은 0–2개다** → DT(팔 link의 target 여유) 근거가 없어 DT · DST를 보류한다. (3) 이 행들의 clearance 중앙값은 약 15 mm로 양수다. 고친 계획은 제약 경계에 붙어 있지 않다. 경계에 붙은 것은 쥔 뒤 손가락 대 탁자(−4.5 · 2.5 mm)뿐이다. **판정:** TrajOpt가 정책 계획이 이미 안전한데도 바꾸는 것이 남은 주 원인 후보다. 이를 막는 기존 규칙 `--minimal-intervention-mm`(pb2)이 최종 구성에서 빠졌다. → MG(pc판 + 문턱 10 mm, 변환 C) · MGD(변환 D)를 task §7에 결과 전 고정했다. 등록은 implementer(`wip/sl15-mig`)가 한다 | 위험할 때만 개입하면 장애물 없는 장면은 정책 bit 그대로 가고, 장애물 장면은 C의 강한 추적으로 지킬 수 있다. C · D · B 사이의 교환(F6)을 피하는 구조다 | DT · DST 보류(queue에서 뺌) | `SL15.task.md` §7, scratchpad `sl15/j1_drec.json` |
| F8 | 03:2x | **MG 정정 · SF 고정.** implementer가 `-pc-mig`가 SL11 `-pc-mi10`과 argv 동일임을 알렸다. MG(C)는 SL11에서 이미 seed 7 6/9, 문턱 통과 0.64였다. 문턱이 닫힌 call은 정책 계획 자체가 음수 clearance였다(접근 −11 mm). → MG(C)는 다시 돌리지 않고 MGD(`-pc-mi10` + D)만 queue에 넣었다. `wip/sl15-mig`(6a7accf2)는 merge하지 않는다. DREC J1이 그 음수 clearance의 주인을 가른다. 쥐는 중에는 손 자신 voxel(자기 필터 누수)이고, 쥐기 전에는 지워지지 않은 target이며, 쥔 뒤에는 접시 · 탁자다. 첫째를 고치는 판 SF(손 link 자기 필터 팽창)를 task §8에 고정했다. 팽창값은 implementer가 DREC 기록의 누수 거리 p95로 결과 전에 정한다(`wip/sl15-sfhand`) | 정상 쥐기가 거짓 기하(자기 손 · 지워지지 않은 target)에 대해 충돌로 계산되는 것이 TrajOpt가 위로 미는 원인이다. 변환 · 문턱은 그 결과를 다룰 뿐이고, 원인은 관측 · 기하 층(LTE · 자기 필터)이다 | 순서: DS → MGD → SFD → DH(시간 남으면) | `SL15.task.md` §7.1 · §8 |
| F9 | 04:1x | **DS(D + 개도 · 명령 attach/detach) 판정: 후보 아님.** LIBERO 27 17/27(7 · 5 · 5), D 대비 +2 → 구분 못 함(§5). HSR 기제 통과(실제 쥔 call 중 HELD 0.96, 손에서 먼 HELD 0.004). 쥔 동안 포화 0.32(문턱 0.3 경계). LIBERO-Safety 10 2/10(D 4), 위반 0 → (b) 미달. SafeLIBERO 11 성공 0 · 충돌 6(D 2 · 7). **읽기:** attach · detach 판정을 정답에 가깝게 고쳐도(SL14 KO 1.00, 여기 0.96) 성공은 오르지 않는다. 남은 손실은 latch 판정이 아니라, 정상 쥐기가 거짓 기하(자기 손 · 지워지지 않은 target · 받침면)와 '충돌'하는 데서 온다(F7). 같은 시각 `wip/sl15-sfhand`(875802cb, 손 자기 필터 4 mm: DREC 깊이 픽셀 누수 p95 3.94 mm, 손가락 제외)를 merge했다(2ad29c18, test 92 통과). queue는 MGD → SFD다 | HSR 규칙 수정은 판정을 바로잡지만 성공의 병목은 아니다. 사용자 질문 F1의 HSR 계획에 대한 측정 답이다 | DH · DT · DST는 보류 중 | `outputs/2026-11-06_SL15_변환D/` |
| F10 | 04:2x | **조합 판 순서를 결과 전에 고정했다**(task §9). SFD 뒤에 SFMD(손 필터 + 문턱 + D), SFMC(같은 판 + C), 시간이 남으면 SFMDS를 돈다. 등록은 implementer(`wip/sl15-combo`)가 한다. MGD가 지금 돈다 | 남은 시간(11:00까지 약 6.5시간, 판 하나 약 80분)에 원인 층별 수정(변환 · 개입 문턱 · 자기 필터 · latch)을 하나씩 본 뒤, 합친 판을 C · D 두 변환으로 본다 | DH · DT · DST 보류 유지 | `SL15.task.md` §9 |
| F11 | 04:4x | **MGD(최소 개입 문턱 10 mm + D) LIBERO 27: 15/27(6 · 2 · 7), D와 같음 → 구분 못 함.** 기제 통과: 문턱이 call의 0.51을 그대로 통과시켰다(찾는 중 118/120, 접근 132/233, 쥐는 중 80/259, 쥠 114/263). 통과 못 한 call의 정책 계획 clearance 중앙값은 접근 −8.5 · 쥐는 중 −5.6 · 쥠 −10.1 mm이고, 72–81%가 음수다. **판정:** '안전한 계획을 괜히 바꾸는 것'은 주 원인이 아니다. 성공을 깎는 call은 정책 계획이 거리장의 거짓 기하와 실제로 겹치는 call이다. LIBERO-Safety · SafeLIBERO는 그대로 돈다 | 원인 층은 관측 · 기하(LTE · 자기 필터 · 받침면)로 좁혀진다. 다음 SFD가 그중 손 자기 필터를 시험한다 | – | `outputs/2026-11-06_SL15_변환D/` |
| F12 | 05:3x | **MGD 판정: 후보 아님.** LIBERO-Safety 10 4/10(D 4) 위반 0, SafeLIBERO 11 성공 3 · 충돌 6(D 2 · 7, pc판 1 · 5). 문턱은 장애물 장면의 충돌을 조금 줄였으나(7 → 6) pc판(5)보다 많다. SFD가 05:33에 시작했다 | 최소 개입은 정책 계획이 안전할 때만 효과가 있다. 성공을 깎는 call은 정책 계획이 거짓 기하와 겹치는 call이라 문턱이 막지 못한다(F11) | – | `outputs/2026-11-06_SL15_변환D/` |
| F13 | 06:0x | **SFD(손 자기 필터 4 mm + D) LIBERO 27: 15/27(6 · 4 · 5), seed별로 D와 같다. 고정 기제 미달 → "의도대로 동작하지 않음".** 쥐는 중 누적 EE 변화 중앙값 13.1 mm(D 15.1, 절반 문턱 7.6), 쥠 8.0 mm. implementer 경고(깊이 픽셀 기준 4 mm는 손 voxel의 22%만 덮음)대로 TrajOpt가 보는 거리장은 거의 바뀌지 않았다. **§9 순서에서 벗어난다:** SF가 효과 없으므로 SF 조합은 이미 잰 판과 사실상 같다(SFMD ≈ MGD). 남은 시간에 새 정보를 얻도록 (1) MGC = `-pc-mi10` + 변환 C 세 장면(SFMC에서 효과 없는 SF를 뺀 판, C+문턱은 SL11에서 LIBERO seed 7만 쟀음), (2) 보류했던 DH(D + HSR6 다섯 규칙 끔, LIBERO 27 · LIBERO-Safety · SafeLIBERO)를 돈다. 조합 등록(`wip/sl15-combo`)은 중단했다(merge 안 함). SFD LIBERO-Safety · SafeLIBERO는 그대로 돈다 | 고정 순서를 바꾼 이유는 SF의 기제 미달이 결과로 확인된 데 있다. 바꾼 판도 결과 전에 정했다. DH는 사용자의 '불필요한 규칙' 질문에 대한 직접 측정이다 | 되돌리기: 순서만 바뀜 | `SL15.task.md` §9 |
| F14 | 07:0x | **SFD 판정: 후보 아님**(LIBERO 15 · 기제 미달, LIBERO-Safety 3/10 위반 0, SafeLIBERO 성공 1 · 충돌 5). MGC가 06:58에 시작했다 | 손 자기 필터 4 mm(깊이 픽셀 누수 p95)는 TrajOpt가 보는 voxel을 바꾸지 못했다. 손 voxel은 TSDF 두께 때문에 손 구에서 중앙값 13.6 mm 떨어져 있다(implementer 측정). 다음 수정은 voxel 단계에서 손 구 둘레를 지우는 것이다(사용자 판정 후보) | – | `outputs/2026-11-06_SL15_변환D/` |
| F15 | 07:3x | **MGC(최소 개입 문턱 10 mm + 변환 C) LIBERO 27: 13/27(6 · 3 · 4)** → pc판 10 대비 +3, 구분 못 함. (a) 미달이다. 문턱 통과 0.58이고, 쥔 동안 포화는 0.42다(pc판 0.91). SL11 seed 7(6/9)과 같은 결과다. LIBERO-Safety · SafeLIBERO는 장애물 보호를 보려고 그대로 돈다 | 문턱은 변환과 무관하게 '정책 계획이 안전한 call'만 통과시킨다. 성공을 깎는 call은 그 밖에 있다(F11) | – | `outputs/2026-11-06_SL15_변환D/` |
| F16 | 08:2x | **MGC 판정: 후보 아님**(LIBERO 13, LIBERO-Safety 3/10 위반 0, SafeLIBERO 성공 0 · 충돌 5). DH가 08:22에 시작했다 | 문턱 + C는 pc판(C)과 거의 같은 자리다 | – | `outputs/2026-11-06_SL15_변환D/` |
| F17 | 09:5x | **DH(D + HSR6 다섯 규칙 끔) 판정: 후보 아님.** LIBERO 27 15/27(6 · 4 · 5) = D → 구분 못 함, 고정 읽기상 'HSR6 규칙은 장애물 없는 장면에서는 불필요할 수 있음'. 그러나 LIBERO-Safety 10 1/10(D 4), SafeLIBERO 성공 1 · 충돌 6이다. 장애물 장면에서는 HSR6 규칙이 성공에 도움이 되는 쪽이다. 기제 통과(HSR6 변경 call 0). 마지막 run이 끝나 chain을 멈췄다(STOP) | 사용자의 '불필요한 규칙' 질문에 대한 측정 답이다. 장애물 없는 장면에서는 효과가 없지만 장애물 장면에서는 필요하다 | – | `outputs/2026-11-06_SL15_변환D/` |

## 아침 요약 (lead, 09:5x KST)

**한 줄.** 후보 조건을 넘은 판은 없다. 변환(CCR)만으로는 장애물 없는 장면의 성공과 장애물 장면의 보호를 함께 얻을 수 없다. 남은 손실은 정책 계획이 거리장의 거짓 기하와 겹치는 데서 온다.
- 쥐는 중: 손 자신 voxel
- 쥐기 전: 지워지지 않은 target
- 쥔 뒤: 받침면

latch(HSR) 규칙 수정 · 최소 개입 문턱 · 손 자기 필터 4 mm는 어느 것도 이 겹침을 없애지 못했다.

| 판 | LIBERO 27 | LIBERO-Safety 10 (위반) | SafeLIBERO 11 성공 / 충돌 | 판정 |
|---|---:|---:|---:|---|
| VLA 단독 | 27 | 10 (0) | 11 / 9 | 기준 |
| pc판 (변환 C) | 10 | 1 (0) | 1 / 5 | 기준 |
| 변환 B | 19 | 8 (0) | 3 / 9 | 아님 — 보호 잃음 (F2) |
| 변환 D | 15 | 4 (0) | 2 / 7 | 아님 — 기제 미달 (F3 · F6) |
| D + 개도 attach · detach | 17 | 2 (0) | 0 / 6 | 아님 (F9) |
| D + 최소 개입 문턱 | 15 | 4 (0) | 3 / 6 | 아님 (F11 · F12) |
| D + 손 자기 필터 4 mm | 15 | 3 (0) | 1 / 5 | 아님 — 기제 미달 (F13 · F14) |
| C + 최소 개입 문턱 | 13 | 3 (0) | 0 / 5 | 아님 (F15 · F16) |
| D + HSR6 끔 | 15 | 1 (0) | 1 / 6 | 아님 (F17) |

**밝힌 것.**
1. 같은 TrajOpt 수정이 장애물 장면에서는 보호하고, 장애물 없는 장면에서는 방해한다(F6).
   - 수정 실현량을 늘리면: SafeLIBERO 충돌 C 5 · D 7 · B 9
   - 줄이면: LIBERO 성공 C 10 · D 15 · B 19
2. 수정이 생기는 것은 정책 계획이 이미 거리장과 겹치기 때문이다(F7 · F11).
   - 정책 계획이 안전하면 문턱이 통과시키지만, 통과 못 한 call의 정책 계획 clearance 중앙값은 −6 ~ −10 mm다.
   - 그 겹침의 주인(DREC J1): 쥐는 중 손 자신 voxel, 쥐기 전 target, 쥔 뒤 접시 · 탁자.
3. HSR(사용자 질문 F1).
   - 개도 · 명령 attach/detach는 판정을 정답에 가깝게 고친다(HELD 0.96). 그러나 성공의 병목은 아니다.
   - HSR6 다섯 규칙은 장애물 없는 장면에는 효과가 없다. 장애물 장면에서는 도움이 된다(LIBERO-Safety 4 → 1).
   - 팔 link의 target 여유 50 mm(DT)는 J1에서 해당 행이 0–2개라 돌리지 않았다.
4. 손 자기 필터 4 mm는 듣지 않았다. 깊이 픽셀 누수는 4 mm지만, TrajOpt가 보는 voxel은 TSDF 두께 때문에 손 구에서 중앙값 13.6 mm다.

**사용자 판정이 필요한 것.**
1. 다음 수정 층. 권장은 (가)다. 쥐는 중 거짓 기하의 최대 주인이고, 안전 쪽 위험이 가장 작다.
   - (가) 거리장 voxel 단계에서 로봇 손 둘레(TSDF 두께 ≈ 15 mm)를 지우기
   - (나) target 지우기의 손바닥 근처 잔여를 정답 target으로 다시 재기(SL12 · SL13의 측정 정정 기준)
   - (다) 쥔 물체 · 받침면 접촉 허용(어제 E21)
2. 변환은 C를 유지할지, D로 바꿀지. D는 LIBERO +5 · LIBERO-Safety +3이지만 SafeLIBERO 충돌이 +2다. 기하 수정 뒤 다시 비교하는 것을 권장한다.
3. HSR6 규칙은 유지(장애물 장면에서 필요)하고, 개도 attach 규칙은 기본 꺼짐으로 둘지.

**운영.**
- commit: 07463672 · 816f7aa4(D) · 8b25971e · 5bccd363(ntm) · fbadde8c · 2ad29c18(sfhand) · 마지막 문서 commit
- `wip/sl15-mig` · `wip/sl15-combo`는 merge하지 않았다.
- chain 거절: 08:49 1회. 회수 후 08:52 통과.
- push 없음.



## SL16 — 사용자 판정 후 (2026-11-06 11:4x KST –)

자율 판정 시간(–11:00)은 끝났다. 아래는 사용자 판정을 받아 진행한 기록이다.

**사용자 판정.**
- "쥔 물체와 받침면의 접촉은 당연히 허용해야 합니다." → (다)를 넣는다.
- "그래 따로 하세요." → (가)와 (다)를 각각 다른 판으로 넣는다. 실제로 쥐었는지는 개도·명령 규칙으로 판단한다(권장안 수락).
- 질문 "손 voxel 지우기는 RB-Y1에서는 잘 됐는데 Panda에서는 왜 안 되나"에는 측정으로 답한다(G3).

| ID | 시각 (KST) | 한 일 · 판정 | 근거 | 바뀐 것 | 자료 |
|---|---|---|---|---|---|
| G0 | 12:0x | **정정.** 앞선 보고의 "TSDF 두께 때문에 손 voxel이 13.6 mm 바깥에 생긴다"는 단정이었다. 측정된 것은 깊이 누수 p95 4 mm와 거리장의 손 voxel 중앙값 13.6 mm뿐이다. 둘의 차이가 번짐인지 잔상인지는 확인되지 않았다 | `SL15.sfhand.impl.md` §4 | 사용자에게 정정해 알림 | – |
| G1 | 12:0x | **SL16 규칙 결과 전 고정.** (다) HSC: 실제로 쥔 동안 모든 단계에서 쥔 물체 구는 아래 면에 막히지 않고, 쥔 손의 손가락·손바닥은 10 mm까지 겹쳐도 된다. (가) HVC: 먼저 손 voxel을 현재 · 잔상 · 번짐으로 나누고, 잔상 비율 ≥ 0.5면 trail, 아니면 current(r = p90, 상한 20 mm)로 정한다. 기준은 DS · D, 후보 조건은 SL15 §2와 같다 | 사용자 판정 | `SL16.task.md`(f16e4b38) | `AG3S/ag3s/docs/handoff/SL16.task.md` |
| G2 | 12:5x | **HSC 구현 merge**(27a226d3 → 4e51aacf). 새 test 40개 통과, 기존 69개 판의 서버 인자는 bit 동일. 손가락·손바닥 행 허용은 사용자가 말한 "쥔 물체와 받침면"보다 넓다. 제약 경계에 실제로 붙은 행이 쥔 뒤 손가락 대 탁자였기 때문이고, 사용자에게 알렸다 | `SL16.hsc.impl.md` | `-pc-openrule-hsc`(기본 꺼짐) | – |
| G3 | 13:3x | **HVC 원인 측정 · 구현**(98f35096). SL15 DREC의 CLOSING · HELD 195 call, 손 voxel 2,095개를 나눴다: 현재 37.3%, 잔상 20.7%, 번짐 42.0%. 규칙대로 mode = current다. 손 voxel의 sd p90은 23.6 mm라 상한에 걸려 r = 20 mm다. 20 mm로 지우면 손 voxel의 82.2%가 지워진다. 보호 적용 전 기준으로 target voxel 12.3%와 그 밖 1.03%도 함께 지워진다. **RB-Y1 비교: 기록 없음.** RB-Y1 제약 기록에는 GT geom 주인과 요청별 관절 상태 이력이 없어 같은 분류를 할 수 없다 | 번짐 + 현재가 79%이므로, 잔상보다 지금 손 둘레가 문제다. RB-Y1이 왜 괜찮았는지는 아직 측정 근거가 없다 | `-pc-openrule-hvc`(기본 꺼짐), merge 준비 commit af9e37e7(HSC 실행이 끝난 뒤 fast-forward) | `SL16.hvc.data.json` · `SL16.hvc.impl.md` |
| G4 | 13:03–13:12 | **HSC 실행 시작이 메모리 검사에 4회 거절.** HVC implementer의 전체 test(RSS 15–18 GiB)가 pod 사용량을 67 GiB까지 올렸다. 그 프로세스는 끄지 않고 관련 test만 돌려 달라고 요청했다. 남은 작업은 잠시 빼 두었다가, 13:12 5회차에 통과한 뒤 다시 넣었다 | 다른 프로세스 종료 금지 규칙 | – | scratchpad `sl16/run_10_HSC_*.log` |
| G5 | 13:4x | **HSC LIBERO 27: 17/27(6 · 6 · 5).** DS(17) 대비 Δ 0이라 구분할 수 없다. 후보 조건 (a) ≥ 17은 통과했다. **기제 시험 1 통과:** 실제로 쥔 call 110개의 누적 EE 경로 변화가 중앙값 2.2 mm, p90 30.3 mm로 문턱 ≤ 52 mm 이하다(D: HELD 실제 쥠 p90 104.8, 실제 쥔 전체 82.5). 새 규칙은 HELD call에만 적용됐다(seed 7 기준 69/69). **기제 시험 2**(쥔 뒤 손가락 대 탁자 binding 행 0)는 제약 기록이 필요하다. 그래서 SL15 DREC와 같은 조건으로 HSCREC(LIBERO seed 8 + `--record-constraints full`)를 HVC 뒤 queue에 넣었다 | 쥔 동안의 큰 수정은 받침면 '충돌'이 원인이었고, HSC가 그것을 없앴다. 그래도 LIBERO 성공은 DS와 같다. 남은 실패가 어느 단계 · 층에 있는지는 아직 세지 않았다 | – | `outputs/2026-11-06_SL16_접촉허용_손voxel/data/mech_HSC.json` · `calls_HSCg.jsonl` |
| G6 | 13:59 | **HSC LIBERO-Safety 10: 2/10, 위반 0.** DS도 2/10이라 같다. 후보 조건 (b) ≥ 6 미달 → **HSC는 후보 아님.** 고정한 대로 SafeLIBERO 11은 그대로 돈다 | §4 규칙 | – | storage `*sl16-HSC-ls10*` |
| G7 | 14:15 | **HSC SafeLIBERO 11: 성공 2 · 충돌 8**(DS 0 · 6, D 2 · 7). (c) 충돌 ≤ 5 미달. **HSC 최종: 17 · 2(위반 0) · 2/8 — 후보 아님.** 충돌이 늘어난 episode(t1e4 · t2e0 · t2e1 · t3e0 · t3e2 · t3e3 · t3e4 · t2e2)의 접촉 상대와 단계는 아직 분석하지 않았다 | §4 규칙 | – | storage `*sl16-HSC-safelibero11*` |
| G8 | 14:18–14:3x | **HVC 시작**(HEAD 1983737c: af9e37e7 fast-forward + 문서). merge 뒤 test 75개 통과. 서버 설정에 `hand_voxel_carve = {mode: current, radius_m: 0.02}`가 들어간 것을 확인했다. 요청별 지운 voxel 수(`n_voxels_freed`)는 제약 기록에만 남는다. 그래서 LIBERO(기록 꺼짐)에서는 볼 수 없고, 기록을 켠 SafeLIBERO에서 > 0인지 확인한다 | 구현 기록 `SL16.hvc.impl.md` 73행 | – | storage `*sl16-HVC-*` |
| G9 | 14:41 | **사용자 요청으로 중단.** queue를 비우고 실행 중이던 HVC seed 9(1/9 episode)를 멈췄다. 이 세션이 띄운 프로세스만 멈췄고, development의 8294 서버는 그대로 두었다. GPU · lock은 비었고 development 세션에 알렸다. **HVC 부분 결과(판정 아님):** seed 7 · 8 성공 10/18(같은 seed에서 D 10, DS 12). CLOSING call 132개의 누적 EE 변화 중앙값 0.5 mm(같은 seed에서 D 15.6, 문턱 ≤ 7.6). 미완 seed 9 run에는 `INCOMPLETE_STOPPED.txt`를 남겼다. `reclaim: ws 23.0->23.0 GiB \| cache 0.0->0.0 GiB, 431 files, 12 paths` | 사용자 지시 | 남은 작업 4개 → `SL16.task.md` §6 | `outputs/2026-11-06_SL16_접촉허용_손voxel/README.md` · `resume/` |
