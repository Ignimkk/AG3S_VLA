# AG3S Baseline — 논문 비교 baseline · 비교 방법 구성과 측정 기록

> **이 파일의 용도.** `feature/panda-benchmark` 에서 논문(`main.pdf`, CVPR 2027 draft)의 **비교 대상**을 구성하고
> 측정한 기록이다. 비교 대상은 두 갈래다.
> (1) **target grounding baseline** — observed target surface 를 external segmenter 나 다른 VLA-internal 방법으로
> 잡는 방법(Grounded-SAM 계열, YOLOE, localization heads, TokenCut, GT oracle). 논문 Tab. 2 의 "External segmenter" 행과
> Sec. 4.1 의 비교가 여기서 나온다.
> (2) **safety method** — 실행 직전에 action 을 고치는 safety filter(KNOWS, VLSA/AEGIS, multi-link)와 denoising 안에서
> action 을 미는 sampling guidance(OmniGuide, constrained flow matching, Lan-o3dp adapted, NUDGE).
> 단계 이름은 `BL-0` 부터 이어 쓴다. 2026-10-23 사용자 지시("`AG3S_BASELINE_LOG.md` 를 만들어서 기록")로 신설했다.
>
> V11AB(paired v2) 평가의 기록은 [`AG3S_BENCHMARK_LOG.md`](AG3S_BENCHMARK_LOG.md)(SL-0~), RB-Y1 16D 모델의 기록은
> [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) 가 맡는다. 세 파일은 대상 · 환경 · 입력이 달라 서로의 숫자를 대신하지 않는다.
>
> **수치 규칙.** 이 파일의 모든 수치는 `AG3S/ag3s/docs/handoff/<STEP>.verify.json` 을 출처로 한다. BL-1 의 수치는
> [`handoff/PAPER-BASELINES-2.verify.json`](handoff/PAPER-BASELINES-2.verify.json) 의 `numbers`(표 · 지표)와
> `runs` · `issues` · `not_measured`(실행 시각 · 실패 · 미측정)에서만 가져왔다. BL-3 의 수치는 [`handoff/BASELINE-EVAL-V1.verify.json`](handoff/BASELINE-EVAL-V1.verify.json), BL-4 의 수치는 [`handoff/BASELINE-EVAL-V2.verify.json`](handoff/BASELINE-EVAL-V2.verify.json) 의 `numbers` · `issues` · `limitations` · `not_measured` 에서 가져왔고, 두 단계의 구현 경위(`BASELINE-EVAL-F.impl.md` · `-G.impl.md`)는 구성 · 방법 사실만 옮겼다(G 의 속도 · 동일성 수치는 옮기지 않았다). BL-5 의 수치는 [`handoff/ATS-CLOUD-DIAG-V4.verify.json`](handoff/ATS-CLOUD-DIAG-V4.verify.json) 의 `numbers` · `runs` · `issues` · `not_measured` 에서 가져왔다. 원인 진단의 근거(`파일:줄` · 경로)는 `ATS-CLOUD-DIAG-H.impl.md`, 구현은 `ATS-CLOUD-DIAG-J.impl.md` 에서 옮겼고 H 의 예비 수치는 옮기지 않았다. BL-6 의 수치는 [`handoff/CLOSED-LOOP-METHODS-V3.verify.json`](handoff/CLOSED-LOOP-METHODS-V3.verify.json) 의 `numbers` · `runs` · `code_state` · `issues` · `not_measured` 에서 가져왔다. 구현 구조 · 설계 결정 · privileged 여부 · 원 방법과의 차이는 `CLOSED-LOOP-METHODS.task.md` · `-I.impl.md` · `Baseline/paper_baselines/closed_loop/README.md` 에서 옮겼고, `-I.impl.md` 의 CPU smoke 수치는 옮기지 않았다(bit 재현 사실만). BL-7 의 수치는 [`handoff/CLOSED-LOOP-FIX-V5.verify.json`](handoff/CLOSED-LOOP-FIX-V5.verify.json) 의 `numbers` · `runs` · `code_state` · `issues` · `not_measured` 에서만 가져왔다. 수정의 원인 · 원 방법의 정의 · 수정 내용은 [`handoff/CLOSED-LOOP-FIX-K.impl.md`](handoff/CLOSED-LOOP-FIX-K.impl.md) 에서 사실만 옮겼고 그 안의 CPU replay 수치(전후 표)는 옮기지 않았다. BL-8 의 수치는 [`handoff/BASELINE-BENCH-READY-V6.verify.json`](handoff/BASELINE-BENCH-READY-V6.verify.json) 의 `numbers` · `runs` · `code_state` · `issues` · `not_measured` 에서만 가져왔다. 구현 사실(원인 · 원 방법 · 수정 · 원 방법과의 차이)은 [`BASELINE-BENCH-READY-L.impl.md`](handoff/BASELINE-BENCH-READY-L.impl.md) · [`-M.impl.md`](handoff/BASELINE-BENCH-READY-M.impl.md) 에서, 결정과 맥락은 [`BASELINE-BENCH-READY.task.md`](handoff/BASELINE-BENCH-READY.task.md) 에서 옮겼고 두 impl 의 CPU replay · CPU episode 수치는 옮기지 않았다(bit 동일성 사실과 test pass / skip 개수만). 출처가 없는 숫자는 쓰지 않았고, 필요하면
> verifier 에게 측정을 요청한다. **implementer 의 smoke 수치(`*.impl.md`)는 이 파일에 옮기지 않는다.** smoke 는 배선
> 확인이지 측정이 아니기 때문이다. BL-0 · BL-2 는 구성과 test 통과 사실(pass / skip 개수)만 `impl.md` 에서 옮겼다.
> 식별자(경로 · SHA · commit · sha256)는 verify.json 또는 impl.md 에 있는 것만 옮긴다.
>
> **시각.** 모든 시각은 KST(UTC+9)다. verify.json 의 UTC 를 +9 h 로 옮겼다. `create_run` 이 만드는 run dir 이름의 시각은
> 원래 KST 다(`AG3S_BENCHMARK_LOG.md` SL0 절).

---

## 진행 현황

| 단계 | 범위 | 상태 | 비고 |
|---|---|---|---|
| **BL-0** PAPER-BASELINES | `Baseline/paper_baselines/` 신설: 공통 interface · lift · 평가 harness · 6 provider · external sources · 환경 | **구현 · test 완료 (2026-10-23, commit `bf1403d5`), 사용자 판정 대기** | [BL-0](#bl-0--paper-baselines-논문-비교-baseline-폴더-2026-10-23) |
| **BL-1** PAPER-BASELINES-2 | SL3 dev 10 episode · 221 frame 에서 grounding baseline 8 spec × prompt GPU 측정 (12 arm + ATS) | **측정 완료 (commit `5752c64c`), 사용자 판정 대기.** ~~`grounded_sam2:track` 은 OOM 으로 **미측정**, SL4b 40 episode 는 **deferred**~~ → track · SL4b 40 episode 는 BL-4 에서 측정(2026-10-24) | [BL-1](#bl-1--paper-baselines-2-grounding-baseline-sl3-dev-gpu-측정-2026-10-23) |
| **BL-2** PAPER-METHODS | `methods/` 신설: safety filter 3 + sampling guidance 4 + 외부 repo 7 fetch. KNOWS 이동 | **구현 · test 완료 (commit `5752c64c`), 사용자 판정 대기.** ~~수치는 BL-3 이후에 채운다~~ → offline(open-loop) 수치는 BL-3 에 채웠다. closed-loop 수치는 ~~BL-5~~ → BL-6 | [BL-2](#bl-2--paper-methods-논문-비교-방법-methods-구성-2026-10-23) |
| **BL-3** BASELINE-EVAL V1 | ATS 와의 공정 비교(cloud labeling) · grounding 고유 지표 · methods offline 지표 · module 별 figure | ~~**진행 중** (implementer F → verifier V1, CPU)~~ → **측정 완료 (2026-10-24 KST, CPU), 사용자 판정 대기.** 지표 registry · ATS 공정 비교(cloud labeling) · grounding 고유 지표 · methods offline · benchmark 대표 장면 | [BL-3](#bl-3--baseline-eval-v1-cloud-labeling-공정-비교-고유-지표-methods-offline-2026-10-24) |
| **BL-4** BASELINE-EVAL V2 | track memory 수정 · 평가 CPU 경로 속도 개선 → SL3 track + SL4b 40 episode 측정 | ~~**진행 중** (implementer G → verifier V2, GPU lock)~~ → **측정 완료 (2026-10-24 KST, GPU), 사용자 판정 대기.** track memory 평탄 · SL3 track · SL4b 40 episode 전체 arm · sm_90 확인 | [BL-4](#bl-4--baseline-eval-v2-track-memory-sl3-track-sl4b-40-episode-2026-10-24) |
| **BL-5** ATS-CLOUD-DIAG | V11AB `cloud_target` 이 비는 원인 진단 · cloud labeling 의 ATS 표현 옵션(`--ats-repr` · `--ats-tol`) · 표현별 공정 비교 재계산 · stale geometry 와 GT 의 거리 | **측정 완료 (2026-10-24 KST, CPU), 사용자 판정 대기.** 코드(J)는 commit 하지 않은 working tree 판 | [BL-5](#bl-5--ats-cloud-diag-v11ab-cloud_target-이-비는-원인-진단과-ats-표현별-공정-비교-재계산-2026-10-24) |
| ~~BL-5 (이름 미정)~~ → **BL-6** CLOSED-LOOP-METHODS | methods closed-loop 연결 → benchmark 고유 지표(TSR · CAR · ETS 등) | ~~미착수~~ ~~**진행 중** (implementer I → verifier V3). 이 기록은 verify.json 이 나온 뒤 쓴다~~ → **구현 · GPU smoke 완료 (2026-10-24 KST, commit `16d0eea1`), 사용자 판정 대기.** 8 arm 1 episode smoke 까지만 했다. ~~**pilot(20 episode × 8 arm)은 아직 돌리지 않았다**~~ → pilot 은 pilot 전 수정(BL-7) 뒤에 BL-7 에서 돌렸다. 이 STEP 의 issue 1 · 3 · 4 · 5 는 BL-7 이 다룬다 | [BL-6](#bl-6--closed-loop-methods-methods-를-safelibero-closed-loop-에-연결--gpu-smoke-2026-10-24). 원본 [`handoff/CLOSED-LOOP-METHODS.task.md`](handoff/CLOSED-LOOP-METHODS.task.md). `benchmark/runners` 는 고치지 않고 `Baseline/paper_baselines/closed_loop/` 의 새 server · launcher 로 붙였다 |
| **BL-7** CLOSED-LOOP-FIX | pilot 전 3 arm(lan_o3dp · knows · omniguide) 수정 → 수정 3 arm GPU smoke 재확인 → SafeLIBERO spatial L I dev 20 episode × 8 arm pilot | **구현 · smoke · pilot 측정 완료 (2026-10-24 KST, commit `b95c1c87`), 사용자 판정 대기.** 8 job 모두 rc 0 · 160 episode · fallback 0. 20 episode pilot 이라 p 값 · CI 는 있는 그대로만 적고 해석하지 않는다 | [BL-7](#bl-7--closed-loop-fix-pilot-전-3-arm-수정--smoke-재확인--pilot-20-episode--8-arm-2026-10-24). 원본 [`handoff/CLOSED-LOOP-FIX.task.md`](handoff/CLOSED-LOOP-FIX.task.md) |
| **BL-8** BASELINE-BENCH-READY | 인식 경로 depth trim · CFM action box · privileged GT 장애물 변형 · benchmark job 생성 · GPU smoke(L_smoke 9 job · gt_smoke 9 job · suite_smoke 36 job) | **구현 · GPU smoke 완료 (2026-10-25 KST, commit `a63f2864` · `98004d0d`), 사용자 판정 대기.** 54 job 모두 rc 0 · fallback 0. **모든 수치는 n = 1(`t0e0`)이다. 성능 비교가 아니라 배선 · 입력 기록 확인이다.** 시뮬레이션의 GT 탁자 높이와의 비교, impl V6 step 1 CPU 재확인은 하지 않았다 | [BL-8](#bl-8--baseline-bench-ready-인식-경로-depth-trim--cfm-action-box--privileged-gt-장애물-변형--benchmark-job-생성--gpu-smoke-2026-10-25). 원본 [`handoff/BASELINE-BENCH-READY.task.md`](handoff/BASELINE-BENCH-READY.task.md). 전체 benchmark 실험은 benchmark 단계에서 한다(사용자 지시) |
| **BL-9** SEG-BASELINE-3D | 외부 segmenter 의 3D target 분할 진단: Spatial 40 episode 873 frame · mask 경계 오류 통제 · Object 281 frame 의 target 단서 × robot mask · init 5·6 552 frame (2026-11-01) | **측정 완료, 사후 기록 (자체 보고, `verify.json` 없음), 사용자 판정 대기.** 개발 자료 | BL-9 (아래) |
| **BL-10** MULTIBENCH-2D | LIBERO · SafeLIBERO · LIBERO-Safety 정적 80 장면의 2D 분할 · 상자 가림 · 정답 위치 단서 · 가림 추적 (2026-11-01) | **측정 완료, 사후 기록 (자체 보고), 사용자 판정 대기.** ATS 는 평가하지 않음 | BL-10 (아래) |
| **BL-11** CBF-QP-SAFELIBERO | SafeLIBERO Level I dev 20 episode closed loop, Multi-link CBF-QP · KNOWS × GT / detector 장애물 (2026-11-03) | **측정 완료 4 arm (README 자체 보고), 사용자 판정 대기.** AEGIS-style 2 arm 은 **미실행** | BL-11 (아래) |
| **BL-12** SAFELIBERO-SEG-ATS | SafeLIBERO 20 episode 796 frame 에서 외부 segmenter 10 arm 대 실제 ATS 4 arm 의 target · obstacle 분할 (2026-11-03) | **측정 완료 (README 자체 보고), 사용자 판정 대기.** 외부 mask 를 서버에 넣은 closed loop 는 하지 않음 | BL-12 (아래) |
| **BL-13** SAME-MACHINE-REMEASURE | 새 GPU container에서 VLA 단독 · AEGIS / KNOWS / multi-link × (detector, GT 장애물) · AG3S 최종(재현 HSR6 probe) · AG3S pb7을 SafeLIBERO spatial LI · LII dev 20 episode로 같은 기계 재측정 + LIBERO-Safety L0 VLA 단독 | **측정 완료 (2026-11-04 04:52–10:46 KST, 야간 자율 수행, commit 안 함), 사용자 판정 대기** | [BL-13](#bl-13--same-machine-remeasure-새-기계에서-baseline-vs-ag3s-closed-loop-재측정-2026-11-04) |

### 사용자 · lead 결정 기록

| 일자 (KST) | 결정 | 출처 |
|---|---|---|
| 2026-10-23 | 논문 baseline 목록을 하나의 폴더로 만든다. 오픈소스가 있으면 가져오고 없으면 일정 부분 구현한다. **Grounded-SAM 은 반드시 포함** | `PAPER-BASELINES.task.md` |
| 2026-10-23 | 논문의 비교 방법도 같은 폴더에 구성한다. **KNOWS 는 해 둔 것을 그대로 가져오고**, 나머지는 구현하거나 git 에 있으면 가져온다. 기능만 잘 작동하면 된다. **SEAM 은 하지 않는다** | `PAPER-METHODS.task.md` |
| 2026-10-23 | grounding baseline 의 GPU 측정을 수행한다 | `PAPER-BASELINES-2.task.md` |
| 2026-10-23 | ~~`grounded_sam2:track` 과 SL4b 40 episode 는 **deferred** (lead)~~ → `BASELINE-EVAL.task.md` 의 순서 2(BL-4)에서 두 측정을 진행(deferred 해제) | `PAPER-BASELINES-2.verify.json` `not_measured` |
| 2026-10-23 | `AG3S_BASELINE_LOG.md` 를 만들어 기록한다. 권장 순서: ① ATS 와의 공정 비교 · 고유 지표 · methods offline 지표 → ② track memory · 속도 → SL4b → ③ methods closed-loop | `BASELINE-EVAL.task.md` |
| 2026-10-23 | 각 baseline 과 benchmark 가 요구하는 지표를 **전부** 기록하고 시각 자료(실제 장면 · segmentation · module 의 핵심 기능)를 남긴다. 그래서 BL-3 에 baseline · benchmark 별 지표 표(원 지표 · 우리 측정값 또는 closed-loop / not-applicable)와 module 별 figure 3 종(scene · core · metrics)을 둔다 | 사용자 지시, `BASELINE-EVAL.task.md` |
| 2026-10-24 | 권장 순서 ①(BL-3) · ②(BL-4)를 진행했다. ③ methods closed-loop 연결은 ~~BL-5~~ → BL-6 으로 남는다(BL-5 번호는 ATS-CLOUD-DIAG 가 쓴다) | `BASELINE-EVAL.task.md` |
| 2026-10-24 | 사용자 지시 "네 진행하세요": (a) methods closed-loop 연결(BL-6)과 (b) ATS 진단(BL-5)을 진행한다 | `CLOSED-LOOP-METHODS.task.md` · `ATS-CLOUD-DIAG.task.md` |
| 2026-10-24 | **BL-6 pilot 은 smoke 뒤 멈췄다.** task 의 규칙("시간이 4 h 를 넘을 것으로 보이면 smoke 뒤 lead 에게 추정 시간을 알리고 기다린다")에 따라, verifier 추정이 순차 6.63 h · 2 job 병렬 3.35 h 로 나온 뒤 pilot 을 실행하지 않았다. pilot 실행 여부와 job 구성은 사용자 결정 대기 | `CLOSED-LOOP-METHODS.task.md` · `CLOSED-LOOP-METHODS-V3.verify.json` `not_measured` |
| 2026-10-24 | **lead 결정(사용자 판정 전)**: 공정 비교의 ATS 주 표현은 **exclusion geometry 를 같은 cloud 에 5 mm 로 대응시킨 것**, `cloud_target` 은 fresh-only 진단 행, exact · 10 mm 는 sensitivity 행. 기본값을 지금과 같게 두는 옵션 추가라 기존 수치는 바뀌지 않는다 | `ATS-CLOUD-DIAG.task.md` "후속" 절 |
| 2026-10-24 | **사용자 판정 "먼저 수정 후 pilot"**: BL-6 의 3 문제(lan_o3dp action 크기 · knows 의 h<0 와 qp_fail 0 · omniguide skip 과 재compile)를 CPU 에서 먼저 고친 뒤 pilot(SafeLIBERO spatial L I dev 20 episode × 8 arm)을 돌린다. BL-6 의 "pilot 실행 여부 · 구성은 사용자 결정 대기" 에 대한 답이다 | `CLOSED-LOOP-FIX.task.md` |
| 2026-10-24 | **lead go (pilot 시작 전)**: 수정 3 arm 의 GPU smoke 에서 knows · omniguide 의 \|a\| 가 1 을 넘는 것(knows served 최대 1.434, omniguide 1.0084)은 **원 방법의 성질로 보고 known issue 로 기록한 뒤 pilot 을 시작한다.** robosuite OSC 가 입력을 [-1, 1] 로 자르기 때문이다(`base_controller.py:120`, `osc_pose.json:3`). | `CLOSED-LOOP-FIX-V5.verify.json` `issues` · `numbers.robosuite_osc_clip` |
| 2026-10-24 | `Baseline/paper_baselines/methods/guidance_common/core.py` 는 `.gitignore` 의 `core.*`(core dump pattern)에 걸려 한 번도 git 에 올라간 적이 없었다. commit `b95c1c87`(19:05 KST)이 `.gitignore` 에 2 줄을 더하고 이 파일을 추가했다 | `CLOSED-LOOP-FIX-K.impl.md` "git 주의", `git show --stat b95c1c87` |
| 2026-10-25 | **사용자 지시**: "권장 순서로 진행. baseline 방법들은 benchmark 에서 실험할 예정." 그래서 전체 suite × level 실험은 benchmark 단계(사용자)에서 하고, 이 STEP 은 모든 조합의 job 을 만들 수 있게 하고 조합마다 1 episode smoke 까지만 한다 | `BASELINE-BENCH-READY.task.md` |
| 2026-10-25 | **lead 권장 순서**: (a) 인식 경로 수정 + CFM action 범위 진단 → (c) privileged 변형 → (b) benchmark job 생성. 근거: (a) 는 pilot 수치를 흐리는 구현 문제라 전체 실험 전에 고쳐야 한다(검출 상자에 depth trim 이 없어 ellipsoid 가 최대 0.77 m, CFM 의 executed \|xyz\| 가 7.28 까지). (c) 는 원 논문 조건(AEGIS 의 VLM 이름, CFM 의 GT sphere 등)과 같은 행이 있어야 Tab. 4 "reported" 행과 비교할 수 있다. (b) 는 실제 실험이 benchmark 단계이므로 job 생성과 조합별 1 episode smoke 까지만 | `BASELINE-BENCH-READY.task.md` |
| 2026-10-25 | **lead 결정 `gt_obstacle_set`**: `gt_seg` 의 기본은 `all_non_target`(지금 동작), `obstacle_names`(`privileged/obstacle_names` 에 있는 물체만)는 sensitivity 변형이다 | `BASELINE-BENCH-READY-L.impl.md` |
| 2026-10-25 | **같은 checkout 의 benchmark session 과의 합의**: client 수정은 flag `--emit-privileged-obstacles` 하나(기본 off, off 일 때 byte 동일)이고, 그쪽 test 한 줄(`tests/benchmark/test_bm_setup_a_identity.py` 에서 새 parser key `emit_privileged_obstacles` 를 비교에서 제외)을 같은 commit 에 묶는다 | commit `a63f2864` message, `BASELINE-BENCH-READY-M.impl.md` |
| 2026-10-25 | **lead 추가 요청**: 모든 arm 의 `extras["suite"]` 를 server 가 넣는다(`--suite`). V5 동작은 `--no-suite-extras` 로 되돌리고, 이것과 L 의 `crop_mode="box"` · `action_box=false` 를 `<arm>@v5` preset 으로 묶는다 | `BASELINE-BENCH-READY-M.impl.md` |
| 2026-10-25 | **GPU lock 양보 2 회**: 사용자 지시를 받은 development session 의 요청으로 verifier 가 공용 lock 을 두 번 넘겼다(19:23:46 KST gt_smoke 종료 뒤, 20:00:11 KST suite_smoke 일시 정지). 일시 정지 · 재개 시각과 jobs 분할은 BL-8 "GPU lock 양보 경위" | `BASELINE-BENCH-READY-V6.verify.json` `numbers.gpu` |

---

## 용어

기록에 쓰는 말을 여기 모은다. **새 용어를 쓸 때는 여기 추가한다.** (V11AB · ATS · LTE · oracle · phase · dev / test split 등 앞선 로그의 용어는 `AG3S_BENCHMARK_LOG.md` 의 용어 절을 따른다.)

### grounding baseline (BL-0 · BL-1)

| 용어 | 뜻 |
|---|---|
| **paper_baselines** | `Baseline/paper_baselines/`. 논문 비교 대상을 한 폴더에 모은 것. 기존 `Baseline/seam_vla`, `benchmark/SafeLIBERO`(VLSA)는 옮기지 않고 catalog 에서 링크만 한다(옮기면 기존 run · import 가 깨진다) |
| **provider (`TargetMaskProvider`)** | RGB(+ instruction)를 받아 camera 별 **target mask** 를 내는 baseline 의 공통 interface. 검출 실패면 `found=False` 와 빈 mask 를 돌려주고 **지어내지 않는다**(ATS 원칙과 같다). 후보가 여럿이면 score 최고 1 개를 target 으로 고르고 GT 는 선택에 쓰지 않는다 |
| **lift** | provider 가 낸 2D mask 를 depth · K · `T_base_cam` 으로 base frame 3D 점 · 5 mm voxel 로 올리는 단계. 모든 provider 가 **같은 depth · 같은 back-projection(V11AB `backproject`, range 2 m) · 같은 robot 제거 옵션**을 쓰게 하려고 `lift.py` 한 곳에 모았다. pixel frame 변환(policy 224 mirror ↔ OpenCV)도 여기서만 한다 |
| **soft voxel · soft IoU** | development branch(`ifg_graph.py` 의 `build_geometry` · `evaluate`)의 정의를 읽기만 하고 같은 식으로 구현했다. eval voxel = 모든 camera 의 유효 depth pixel 을 `floor(p / 5 mm)` 격자에 넣은 합집합. voxel `v` 마다 `t(v)` = 그 voxel 안 pixel 중 target class 의 비율. 예측 voxel 집합 `P` 에 대해 `TP = Σ_{v∈P} t`, `FP = Σ_{v∈P} (1−t)`, `FN = Σ_{v∉P} t`, `IoU = TP / (TP + FP + FN)`. 경계 voxel 은 target · 비-target 이 섞여 있으므로 0/1 로 자르지 않고 비율로 센다. `t ≥ 0.5` 로 자른 판은 **hard IoU** 로 함께 낸다 |
| **neighbour leak (`neighbour_leak_voxel_mass`)** | 예측 voxel 위에서 **other_object + destination class 의 비율을 합한 값**(= GT 의 비-target 물체 표면에 들어간 예측의 양, voxel mass). 개수가 아니라 비율의 합이라 소수다. frame → episode 평균 → episode 간 평균으로 집계한다. LIBERO 의 fixture(cabinet · stove)와 support(table)는 neighbour 에 넣지 않는다 |
| **neighbour fraction (pooled)** | `Σ 3d_neighbor_fp / Σ (3d_TP + 3d_FP)` (valid frame 전체를 합쳐서). leak 이 예측 크기에 얼마나 비례하는가를 본다 |
| **found rate (`found_any`)** | 두 camera 중 하나라도 `found=True` 인 frame 의 비율. 2D 지표의 per-camera `found_rate` 는 그 camera 한 대에 대한 값이다. 2D 지표에서 not-found 인 camera 는 GT 가 보이면 IoU 0 으로, GT 가 안 보이면 값 없음으로 들어간다 |
| **lift within 10 mm** | lift 된 점 중 scene GT 의 target visual surface 에서 10 mm 이내인 비율. provider 와 무관하게 lift 가 맞는지 보는 독립 검사다 |
| **wrong-object rate (3D)** | valid 3D frame 중 `found_any` 이고 `3d_neighbor_fp > 3d_TP` 인 비율. 예측이 target 보다 비-target 물체 위에 더 많이 놓인 frame. 변형: `3d_neighbor_fp / 예측 voxel 수 > 0.5` |
| **wrong-object rate (2D instance)** | mask 가 비어 있지 않은 frame 중, 두 camera 의 mask pixel 이 **같은 종류의 다른 instance**(`akita_black_bowl_2`)에 더 많이 놓인 frame 의 비율 |
| **referring expression (이 평가의)** | SL3 dev 의 10 task 는 모두 "pick up the black bowl <위치 관계> …" 이고 scene 에 `akita_black_bowl` 이 둘 있다(target `_1`, distractor `_2`). 어느 bowl 인지는 위치 관계 문구로만 구분된다 |
| **phrase prompt · instruction prompt** | detector 에 주는 text 의 두 모드. phrase = target 이름 명사구(이 평가에서는 10 episode 모두 `akita black bowl`, BDDL 에서 유도), instruction = task 문장 전체 |
| **text-free arm (`\|no_text`)** | query text 를 읽지 않는 provider(`localization_heads`, `tokencut`)의 arm. prompt mode 마다 같은 결과를 두 번 내지 않고 한 번만 돈다 |
| **`gt_oracle`** | simulator segmentation(`segmentation_geom`)과 BDDL target geom id 로 만든 mask. 상한선이자 lift 검증용. privileged 다 |
| **`per_frame` · `track` (Grounded-SAM-2)** | `per_frame` = 매 frame 검출 + SAM 2 image predictor. `track` = 처음 found 된 frame 에서 prompt 한 뒤 SAM 2 video predictor(streaming)로 앞으로 전파한다. **causal** 이어야 하므로 frame t 의 결과가 t 이후 frame 에 의존하지 않는다 |
| **localization heads** | Kang et al.(CVPR 2025)의 방법: LVLM 의 많은 attention head 중 image 의 물체 위치를 잘 가리키는 소수의 head 를 attention 비율 · spatial entropy · 선택 빈도로 고른다. 여기서는 그 기준을 pi0.5 의 기록된 attention 에 적용한다. mask 모드 `threshold`(선택 head 합 → upsample → 임계), `sam_point`(peak 한 점으로 SAM prompt) |
| **frozen selection** | dev episode 에서만 고른 head 목록을 json 으로 고정한 것. 편집 여부는 `selection_sha256` 으로 검사한다 |
| **in-sample** | 측정 집합이 그 방법의 선택 · calibration 에 쓴 집합과 같은 것. 이 경우 수치는 held-out 결과가 아니다. localization heads 의 head 는 SL3 + SL4b dev(init 0–4)에서 골랐고 BL-1 의 측정 집합(SL3 dev, init 0)이 그 안에 있다. BL-3 · BL-4 의 SL3 · SL4b 측정도 모두 in-sample 이고, NUDGE 의 κ 도 SL3 dev 로 calibrate 한 값이라 in-sample 이다(verify `in_sample`) |
| **TokenCut (`ncut`)** | Wang et al.(CVPR 2022). DINO ViT-S/16 의 마지막 block key feature 로 patch affinity graph 를 만들고 normalized cut 의 2 번째 eigenvector 를 평균으로 이분해 전경을 얻는다. language 를 쓰지 않는다. `foreground` = seed 가 든 connected component(원 방법), `attention_overlap` = noun attention 과 가장 많이 겹치는 component(**원 방법이 아니다**) |
| **대표점 표현 (ATS)** | V11AB ATS 의 `target_points` 는 voxel-downsample 된 대표점이다. 5 mm voxel 로 모든 pixel 을 세는 이 harness 의 정답 표현과 해상도가 달라 recall 이 구조적으로 낮게 나올 수 있다(A impl.md). BL-1 에서 ATS 행은 이 이유로 **비교에 쓸 수 없다**(아래) → BL-3 에서 같은 cloud 점 위로 다시 비교했다. 표현 차이 외에 `cloud_target` 이 `has_target` call 의 대부분에서 비어 있다는 문제가 드러났다(BL-3 결과 C) |
| **`pred_offgrid`** | eval voxel 에 없는 예측 voxel. FP 1 로 센다. ATS 점에서만 생긴다 |
| **paired bootstrap (vs ATS)** | 같은 episode 의 `arm − ATS` episode 평균 차이에 대해 2,000 회 resample, seed 20261015, percentile 95 % 구간. verify.json 의 `better` / `worse` 는 episode 별 차이의 부호 개수다~~(leak 열에서는 차이가 양수인 episode 수가 `better` 로 세어진다 — 지표 방향이 아니라 부호 기준으로 보인다. `_definitions` 에 정의가 없다)~~ → V1 · V2 verify 로 확인: `better` / `worse` 는 diff 가 `> 0` / `< 0`(V2 정의: `1e-9` 기준)인 episode 수다(`metrics.py:183`). 지표 방향이 아니다 |

### methods (BL-2)

| 용어 | 뜻 |
|---|---|
| **`SafetyFilter`** | 실행 직전에 action chunk 를 고쳐서 내는 방법의 interface(`methods/interface.py`). `filter(obs, actions) → FilterResult(actions, intervened, info)`. KNOWS · AEGIS · multi-link 가 구현한다 |
| **`SamplingGuidance`** | flow-matching denoising **안에서** action 을 미는 방법의 interface. `make_guidance_fn(obs, norm)` 이 openpi `pi0.sample_actions` 의 `guidance_fn(x_next, t_next)` hook 에 들어가는 pure JAX 함수를 만든다. `guidance_fn=None`(기본)이면 hook 이 없는 baseline 과 bit 단위로 같다. OmniGuide · constrained flow matching · Lan-o3dp adapted · NUDGE 가 구현한다 |
| **`SceneObservation`** | 한 policy call 의 관측(RGB · depth · K · `T_base_cam` · q · EE pose · gripper · instruction). GT segmentation · 물체 이름 같은 privileged 입력은 `extras` 에 `privileged/` 접두사 key 로만 넣어 grep 할 수 있게 했다 |
| **privileged** | simulator GT 에서 온 입력(GT segmentation, BDDL 물체 이름, SafeLIBERO obstacle metadata). 실제 perception 이 아니므로 어디에 썼는지 표시한다 |
| **CBF · QP** | Control Barrier Function(`h(x) ≥ 0` 이 안전 집합이고 `ḣ ≥ −αh` 를 제약으로 둔다)과, 그 제약 아래서 nominal action 에 가장 가까운 action 을 구하는 quadratic program. KNOWS · AEGIS · multi-link 가 쓴다 |
| **recorded-chunk 대체 velocity (surrogate sampler)** | pi0.5 network 를 붙이지 않고 guidance 를 CPU 에서 시험하기 위해, 기록된 action chunk 를 목표로 하는 가짜 velocity 로 pi0 와 같은 Euler 적분을 돌리는 것. 양 끝 경우로 `tracking`(velocity 가 목표를 따라감)과 `constant`(수정이 누적됨)가 있다. 실제 network 의 효과가 아니다 |
| **No exemption · All-link exemption** | 논문의 exemption ablation. No exemption = observed target surface 가 모든 link 에게 obstacle. All-link exemption = accepted target 을 **모든** constraint link 에서 뺀다. V11AB 는 target 을 **허가된 link(gripper)에서만** 뺀다(LTE) |
| **deferred** | lead 가 이번 측정 범위에서 뺀 것. 실패나 판정이 아니라 일정상 미룬 것이다 |

### 지표 · cloud labeling · methods offline · track memory (BL-3 · BL-4)

| 용어 | 뜻 |
|---|---|
| **cloud labeling** | ATS 와 segmenter 를 V11AB 가 그 call 에 만든 **같은 fused / filtered cloud 점** 위에서 비교하는 방법. 점마다 GT · ATS · segmenter label 을 매겨 point IoU · P · R · neighbour 비율을 낸다. harness 의 5 mm voxel 비교와 달리 ATS 의 대표점 표현이 평가 단위가 되지 않는다 |
| **`cloud_target` · `target_points`** | V11AB perception run 이 기록한 ATS 출력 두 가지. `cloud_target` = fused cloud 점 중 target 으로 flag 한 점, `target_points` = voxel-downsample 된 대표점. SL4 run 에서 `cloud_target` 은 `has_target` 179 call 중 134 call 에서 비어 있다 |
| **`ats_v11ab_tp_exact` · `ats_v11ab_tp_5mm`** | verifier 보충 ATS 행. cloud 점이 기록된 `target_points` 와 좌표 일치(1 μm 이내) / 5 mm 이내이면 ATS positive 로 센다 |
| **stand-in (V11AB)** | 이번 frame cloud 에서 새로 찾은 cluster 대신 **이전 frame 에 관측한 geometry** 로 V11AB 가 target 을 보고하는 경우. anchor(이번 cluster 가 anchor 의 부분집합이라 anchor geometry 를 유지) 와 occluded(target 이 가려져 마지막 관측 geometry 를 유지)가 있다. 이번 cloud 의 점이 아니므로 `point_indices` 가 비어 있다 |
| **geometry frame** | stand-in 이 가진 geometry 를 마지막으로 관측한 call. frames.jsonl `manipulated.anchor.geometry_frame` |
| **exclusion geometry** | V11AB 가 ESDF 에서 target 으로 제외하는 점 집합(`target-free` layer 를 만들 때 쓴다). has_target call 에서는 `target_points`, has_target=False 이지만 `manipulated_state` 가 있는 call 에서는 episode 안의 직전 has_target call 의 `target_points` 다 |
| **ATS 표현 (`--ats-repr`)** | cloud labeling 에서 ATS positive 점을 세우는 방법. `cloud_target` = V11AB 가 flag 한 cloud 점(현행, 기본), `target_points` = 기록된 대표점에서 tol 이내인 cloud 점(has_target call 만), `exclusion` = exclusion geometry 에서 tol 이내인 cloud 점. `--ats-tol` 기본 0.005 m |
| **call 범주 (fresh · anchor · occluded · excluded_no_target · no_admissible)** | npz 만으로 정한다. `fresh` = has_target 이고 `target_indices` 길이 > 0. `anchor` / `occluded` = has_target 이고 길이 0 이며 `manipulated_state` 가 visible / occluded. `excluded_no_target` = has_target=False, `manipulated_state` ≠ "", episode 안에 이전 has_target call 이 있음. `no_admissible` = 나머지 has_target=False(SL4 에서 4 call 모두 grounding_status `no_admissible`) |
| **`d_surfc_mm` · `pt_med_mm` · `frac_10` · `shift_body_mm`** | stale geometry 와 GT 의 거리 지표(V4). `d_surfc_mm` = \|mean(exclusion geometry) − mean(그 시점 GT target visual surface 표본)\|. `pt_med_mm` = call 마다 exclusion 점 각각에서 GT target visual surface 까지 정확한 거리의 median. `frac_10` = exclusion 점 중 GT surface 10 mm 이내인 비율. `shift_body_mm` = GT target body 위치의 (그 call − geometry frame) 이동 거리 |
| **wrong-object (V4 정의)** | geometry frame 에서 exclusion 점 중 BDDL target 으로 라벨된(가장 가까운 scene instance 가 10 mm 이내) 비율이 50 % 미만인 call. SL4c 의 오선택과 같은 현상을 call 단위로 센 것이다. 이 call 의 `shift_body_mm` 는 BDDL target 의 이동이지 ATS 가 잡은 물체의 이동이 아니다 |
| **native metrics (NM)** | `eval/native_metrics.py`. 각 논문 고유의 지표를 저장 mask 에서 사후 계산한다 |
| **Acc@0.5** | 예측 box 와 GT box 의 IoU ≥ 0.5 이면 정답, 정답 sample 의 비율(REC, localization heads 논문). 여기서는 mask 의 tight box 로 계산하고 detector box 도 같은 값이다 |
| **CorLoc (any-object / target)** | TokenCut 의 object discovery 지표. 예측 box 가 GT box 중 하나와 IoU ≥ 0.5(code 는 `>= 0.5`)이면 정답. any-object = 장면의 모든 BDDL 물체 instance 의 box 를 GT 로(class-agnostic, 원 정의), target = target box 만(= Acc@0.5 mask) |
| **mIoU · oIoU (= cIoU)** | mIoU = sample 별 IoU 의 평균. oIoU = Σ교집합 / Σ합집합(cumulative IoU, RES 의 지표) |
| **pointing game** | 예측 attention 의 최대점이 GT 위(+ tolerance 15 px)이면 hit, hit 비율. tol 0 / tol 15 두 값을 낸다. detector 의 box 중심 hit 은 원 정의의 변형이다 |
| **Fβ · maxFβ** | Fβ = (1+β²)·P·R / (β²·P + R), code 기준 β² = 0.3. maxFβ = 255 개 threshold 에서 sample 평균 Fβ 의 최댓값. binary mask 는 모든 threshold 에서 같아 평균 Fβ 와 같다 |
| **J · F · J&F (DAVIS)** | J = region IoU, F = 경계(contour)의 precision / recall 로 만든 F-measure(허용 반경 ceil(0.008·‖(H, W)‖) pixel), J&F = 둘의 평균. J recall = mean(J > 0.5), J decay = 첫 1/4 구간 평균 − 마지막 1/4 구간 평균. 여기서 sequence = (episode, camera), 5 step 간격의 정책 call frame |
| **temporal IoU** | 연속한 평가 frame 사이 mask IoU. 논문 지표가 아니다 |
| **`conf` sweep · any-candidate recall** | top-1 box 의 detector score threshold 별 P / R. any-candidate recall = 후보 중 하나라도 GT 를 맞힌 sample 의 비율 |
| **open-loop replay** | 기록된 chunk 를 method 에 넣어 수정 chunk 를 얻되 실행하지 않는 평가. 수정이 이후 관측에 미치는 영향은 반영되지 않는다 |
| **개입률 · ‖Δa‖** | 개입률(행) = 수정된 행 / 전체 행(filter: `max\|Δa[:6]\| > 1e-6`, guidance: `\|Δa_xyz\| > 1e-4`), 개입률(call) = 개입 행이 하나라도 있는 call 의 비율. ‖Δa_xyz‖ = 행마다 xyz 의 L2(LIBERO raw action 단위) |
| **`h` · `h_min` · `h < 0` step** | method 자신의 barrier 값. `h_min` = 최솟값, `h < 0` step = 예측 상태가 제약을 어긴 step 수. method 마다 정의가 달라 서로 비교하지 않는다 |
| **scene-GT clearance** | 모든 method 에 같은 decode(OSC response, 자세 고정) → DLS IK → cuRobo `franka.yml` sphere 61 개 → scene GT 표면까지 정확한 거리(안쪽 음수) − 반지름, 0.05 m 에서 자른다(cap). EE sphere(hand + finger)와 arm sphere(link 1–7)로 나누고 obstacle 군은 target · destination · 받침대를 뺀 instance 다 |
| **`scene` · `injected` variant** | guidance 평가의 장애물 두 종류. `scene` = depth lift 에서 privileged target · robot · support · destination mask 를 뺀 점. `injected` = 기록 경로 옆 1 cm 에 반경 3 cm 의 privileged 구를 row 6 에 놓은 합성 시험 |
| **frozen call (OmniGuide)** | OmniGuide 의 규칙으로 guidance 를 건너뛴 call(`skipped_calls` = `frozen_calls`) |
| **bounded memory · read window** | `grounded_sam2:track` 의 streaming state 를 다음 frame 이 읽는 범위(read window)만 남기고 버리는 방식(G). window = mask memory 6 frame, object pointer 15 frame(모델 속성에서 계산) |
| **`track_lost` · `redetect_on_loss`** | track provider 가 target 을 잃었다고 기록한 `not_found_reason`. `redetect_on_loss=false` 이면 잃은 뒤 다시 검출하지 않는다 |
| **`provider_error` 행** | provider 예외가 난 arm 의 그 episode 남은 평가 frame 을 기록하는 행(`missing_reason="provider_error"`). run 은 다음 arm 으로 계속 간다. `--fail-fast` 가 예전 동작이다 |

### closed-loop methods (BL-6)

| 용어 | 뜻 |
|---|---|
| **`MethodServer` (`serve_method.py`)** | V11AB / VLA-only Panda server 와 같은 wire protocol 을 받는 새 server. 기존 client(`libero_safe_client.py`)를 수정 없이 쓴다. arm 에 따라 `none`(VLA-only 와 같은 객체 사슬) · filter(nominal chunk 를 sidecar 에서 고친다) · guidance(`pi0.sample_actions` 의 `guidance_fn` hook) 로 갈린다 |
| **sidecar (`sidecar.py`)** | filter 의 의존성(cvxpy · torch · Grounding DINO)이 `.venv-baselines` 에만 있어서 따로 띄운 process. server(`.venv-openpi-live`)와 pipe 로 통신한다(JSON + `.npy`, pickle 없음) |
| **`GuidedPolicy` · callback / inline** | guidance 를 sampler 안에서 돌리는 방식. `callback`(기본) = 고정된 `jax.pure_callback` hook 으로 JAX CPU backend 에서 guidance 를 실행해 network 는 한 번만 compile 한다. `inline` = closure 상수를 jit 안으로 넘기고 구조가 바뀌면 sampler 를 재compile 한다. 이 STEP 의 GPU smoke 는 `callback` 만 돌았다 |
| **non-privileged arm** | GT segmentation · obstacle 이름 · SafeLIBERO GT sphere 를 쓰지 않고 wire 의 depth · K · `T_base_cam` · q · EE · RGB · instruction 만으로 장애물을 만드는 arm. 8 arm 모두 해당한다. 장애물은 filter 3 arm 이 Grounding DINO(`"object"` box − target box), guidance 4 arm 이 depth lift(− robot sphere + 2 cm, − table plane, − target box)에서 얻는다 |
| **TSR · CAR · ETS (SafeLIBERO 공식)** | VLSA 논문의 지표. TSR = budget 안에 task 를 끝낸 episode 비율, CAR = collision 이 없는 episode 비율(active obstacle 의 `Σ\|Δpos\| > 1 mm`, 공식 code 의 판정), ETS = episode 종료까지 step 수의 평균(time out 포함, 성공이면 steps − 1, 아니면 budget). SL5c 기본 budget 은 280 policy step 이다 |
| **SL5c safe success** | SL5c(V11AB 의 SafeLIBERO closed-loop 평가) 정의: `official_success` 이고 `collided` 가 아닌 episode. client 가 기록한 값을 그대로 쓴다. 공식 TSR / CAR / ETS 와 섞지 않고 따로 낸다 |
| **KNOWS SR · CR · SSR** | KNOWS 논문의 지표. SR = task done, CR = non-target 장애물과 contact 한 episode(여기서는 SafeLIBERO 의 1 mm collision 규칙을 contact 판정으로 썼다), SSR = done 이고 contact 가 없는 episode |
| **multi-link 11 mm** | `max_displacement_m > 0.011` 을 collision 으로 보는 기준(multi-link 논문 규칙). 공식 1 mm 규칙과 따로 낸다 |
| **`h_neg_steps` · `qp_fail`** | filter 의 자기 barrier `h` 가 음수였던 step 수와 QP 가 풀리지 않은 step 수. `h` 는 method 마다 정의가 달라 서로 비교하지 않는다(위 `h` 항목) |
| **guided call · skip call · jit compile** | guidance arm 의 call 기록. guided = `guidance_fn` 이 실제로 chunk 를 바꾼 call, skip = method 규칙으로 identity 가 된 call(OmniGuide 의 freeze zone 등), jit compile = 한 episode 안에서 guidance 구조 변화로 JAX 가 새로 compile 한 횟수 |
| **`ms.serve` · client round-trip** | server 가 응답을 만든 시간(`diagnostics` 제외)과 client 가 request 를 보내고 응답을 받은 시간. 둘의 차이는 응답 전후의 전송 · 진단 기록 시간이다 |
| **ref run (`--ref-run`)** | 같은 episode 를 paired 로 비교하려고 지정하는 기존 run. 여기서는 SL5c LI VLA-only(`20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only`) |

### closed-loop fix · pilot (BL-7)

| 용어 | 뜻 |
|---|---|
| **pilot (BL-7)** | SafeLIBERO spatial Level I dev 의 20 episode(task 0–3 × episode 0–4, seed 7, 280 policy step + wait 20 step) × 8 arm. SL5c 의 VLA-only · V11AB 기록과 같은 episode 집합이다. 20 episode 라 통계 검정력이 작다(아래 "paired") |
| **served · executed (chunk)** | served = method 가 server 에서 내보낸 action chunk 전체(행 10 개). executed = 그중 client 가 실제로 env 에 넣은 행(chunk 의 0–4 행, wait 제외). 이 기록의 `executed max \|xyz\|` 는 robosuite OSC 가 자르기 전의 값이다 |
| **OSC clip** | robosuite 의 operational space controller 가 입력 action 을 `[-1, 1]`(`input_min` / `input_max`)로 자르는 것(`base_controller.py:120`, `osc_pose.json:3`). \|a\| 가 1 을 넘는 행도 실제 제어에서는 1 로 실행된다 |
| **`clip_sample` (lan_o3dp)** | Lan-o3dp 의 DDIM scheduler 가 예측 `pred_original_sample`(x0)을 `[-1, 1]` 로 자르는 옵션(`scheduling_ddim.py:296-297`). 원 repo 의 모든 workspace config 가 `True` 다. pi0.5 sampler 에는 이것이 없어서 BL-6 의 guidance push 가 쌓였다. BL-7 은 guidance 가 더한 부분만 `pi05_libero` quantile stats 로 normalized 한 action box 안에 남도록 제한한다 |
| **relaxed normal (KNOWS Eq. 12)** | KNOWS 의 CBF 행에서 separating normal 을 ε box(ε = 0.05) 안에서 기울여 `δc` 를 건드리지 않고 행을 만족시키는 자유도. `h < 0` 인 행도 normal 을 기울이는 것만으로 만족될 수 있어 filter 가 사실상 비활성이 된다. KNOWS `docs/11 §4` 에 이미 기록된 논문의 성질이다 |
| **`h(n_stored)` · `h*` (KNOWS)** | `h(n_stored)`(= `min_h` · `h_min`) = QP 가 쓰는 행 값, 저장된 normal 에서의 Eq. 6 `h`. `h*` = `maxₙ h(n)` = 진짜 분리 거리. `h(n_stored) ≤ h*`. 그래서 `h(n_stored) < 0` 은 겹침이 아니라 normal 이 최적값을 따라가지 못한 것일 수 있다. `h_star_neg_steps` = `h* < 0`(실제로 겹친) step 수 |
| **`cbf_violated_steps` · `relaxation_covers_need_steps`** | `cbf_violated_steps` = docs/16 §1 조건(모든 obstacle 에서 `h*_next ≥ (1−γ)h*`)을 어긴 step 수. `relaxation_covers_need_steps` = `relax_m ≥ need_m` 인 step 수(`relax_m = ε‖∇ₙh‖₁`, `need_m = γ_h·\|h\|`), 즉 normal 기울이기만으로 충족되는 행 |
| **track association (KNOWS adapter)** | 첫 call 의 ellipsoid 를 물체별 track 으로 유지하는 방식. BL-7 은 Hungarian 으로 track 의 마지막 centroid 와 현재 cloud centroid 를 잇는다(gate 0.10 m). 안 맞은 track 은 마지막 centroid 를 유지(freeze)한다. detector 가 call 마다 이름(`object#j`)을 다시 매기므로 이름으로 잇던 BL-6 판과 다르다 |
| **freeze zone · target cache (OmniGuide)** | freeze zone = EE 가 target 점에 가까이 가면(들어갈 때 점을 저장하고) guidance 를 끄는 구간. 나갈 때는 저장한 점과의 거리로 판정한다. target cache = target 점을 episode 에 한 번 검출해 저장하는 것(`target_cache="episode"`). BL-6 판은 매 call 검출한 점을 썼다(`"call"`) |
| **`ms.total` · `ms.method` · `ms.diagnostics` · `ms.serve`** | `ms.total` = `MethodServer.infer` 의 wall time(client 가 기다리는 시간, 전송 제외). `ms.diagnostics` = chunk 확정 뒤의 기록용 계산. `ms.method` = `total − diagnostics`. **`ms.serve` 는 BL-7 에서 `total` 과 같게 바뀌었다**(BL-6 V3 기록에서는 `total − diagnostics`). 비교는 `ms.total` · `ms.method` 로 한다 |
| **paired (arm vs none)** | 같은 (task, episode) 20 쌍에서 arm 과 `none` 의 결과를 맞춘다. `+a / −b` = arm 에만 사건이 있는 episode 수 a / `none` 에만 있는 episode 수 b(불일치 쌍). **exact sign test** = 불일치 쌍만으로 한 양측 이항 검정(McNemar exact, `p`). 불일치가 0 이면 `p` 가 정의되지 않아 `-` 로 쓴다. **bootstrap CI** = episode 평균 차이(arm − none)의 percentile 95 % 구간, 10,000 회 resample, `numpy default_rng(0)` |
| **working set (pod)** | cgroup `memory.current − inactive_file`. 공용 pod 규칙은 working set + 추정 ≤ 64 GiB(= 65,536 MiB) 다 |

### 인식 경로 · CFM action box · privileged · job 생성 (BL-8)

| 용어 | 뜻 |
|---|---|
| **`t0e0` · n = 1** | task 0, episode 0 의 1 episode. BL-8 의 모든 수치가 이 한 episode 의 값이다. 성능 비교가 아니다 |
| **`crop_mode` (`box` · `depth_trim`)** | filter arm(aegis · knows · multilink)이 Grounding DINO 검출 상자에서 장애물 cloud 를 만드는 방식. `box` = 상자 안의 모든 depth 점에서 robot sphere 만 뺀다(BL-7 까지의 동작). `depth_trim`(새 기본) = 아래 단계로 상자 중심의 물체 하나만 남긴다 |
| **support plane · `support_z`** | 외부 카메라(head)의 전체 lift 에서 가장 밀한 수평면의 높이(base frame). `depth_trim` 이 그 위 1.5 cm 이하와 0.58 m 초과의 점을 지울 때 쓴다. AEGIS workspace box 의 z 범위를 suite 이름 없이 대신한다. non-privileged 추정이다 |
| **center component** | 1 cm voxel 26-연결 성분 중 검출 상자의 중앙 40 % 창에 든 점을 가장 많이 가진 성분. 창이 비면 상자 중심에 가장 가까운 성분 |
| **3D association** | 상자 · 카메라마다 나온 cloud 를 3D 위치(centroid 거리 ≤ 5 cm, 또는 작은 cloud 의 50 % 이상이 1 cm 안에 겹침)로 같은 물체인지 판단해 합치는 것. 옛 방식은 이름 `object#j` 의 j 를 카메라마다 다시 세어 합쳤다 |
| **Panda 받침 envelope** | `depth_trim` 이 지우는 robot 받침 영역(base frame 에서 r_xy 0.18 m, z < 0.16 m). robot sphere model 의 가장 낮은 구가 z 0.085 m 라 받침을 덮지 못해서 따로 둔다 |
| **MVEE (KNOWS 의 ellipsoid)** | minimum volume enclosing ellipsoid. KNOWS 가 장애물 cloud 에 맞추는 타원체. BL-8 은 반축(semi-axes)을 비교한다 |
| **AEGIS workspace box** | AEGIS 원 code 가 suite 이름으로 고르는 world 좌표의 작업 영역(탁자 scene 은 z 0.92–1.5 m, \|x\|, \|y\| < 0.3 m)으로, 탁자면과 배경을 자른다. 이 기록의 `workspace` 열은 server 가 받은 suite 에서 고른 값(`table` · `living_room` · `floor`)이다 |
| **`extras["suite"]` · `--suite` · `--no-suite-extras`** | server 가 `SceneObservation.extras` 에 suite 이름을 넣는 것(모든 arm). BL-7 까지는 없었다. `--no-suite-extras` 가 BL-7 동작이다 |
| **`<arm>@v5` preset** | BL-7(V5)의 동작을 재현하는 조합: `crop_mode="box"`(CFM 은 `action_box=false`) + `--no-suite-extras` + run tag `v5` |
| **`action_box` (CFM)** | CFM guidance 가 served chunk 를 LIBERO OSC 입력 범위 raw [-1, 1](`pi05_libero` quantile stats 로 normalized)의 box 안에 두는 option. 원 CFM 에는 없다. lan 의 `clip_sample` 과 같은 규칙으로 구현했다 |
| **`gt_obstacle_set`** | `gt_seg` 가 어떤 물체를 장애물로 볼지: `all_non_target`(기본, target 이 아닌 모든 instance), `obstacle_names`(`privileged/obstacle_names` 에 있는 물체만. 원 논문 조건과 비교하는 sensitivity) |
| **privileged 변형 (`<arm>:gt[-source]`)** | GT 장애물을 쓰는 변형. source: `names`(GT 이름을 GDINO prompt 로, aegis) · `seg`(GT segmentation, knows · multilink) · `spheres`(GT collision geom 을 감싸는 구, CFM) · `repo`(CFM repo 규칙의 geom, CFM) · `geoms`(collision geom 을 primitive 로, nudge · omniguide · lan). manifest 에 `privileged: true` 가 남는다 |
| **`--emit-privileged-obstacles`** | client flag(기본 off). 켜면 request 에 SafeLIBERO 장애물 GT 가 실린다. 꺼져 있으면 request · 기록 파일이 수정 전 client 와 byte 단위로 같다. `--capture-segmentation` 이 함께 켜지면 GT segmentation 도 실린다 |
| **`active == monitor_obstacle`** | privileged 요약에서 그 call 의 active obstacle(SafeLIBERO env 에서 장면 안에 있는 obstacle)이 client 의 `CollisionMonitor` 가 보는 obstacle 과 같은지. 같은 call 수 / privileged 기록이 있는 call 수로 쓴다 |
| **`jobs.py` (`plan` · `smoke` · `aggregate`)** | `closed_loop/jobs.py`. arm × suite × level × task × split 의 job 목록(`jobs.txt` + `plan.json`)을 `run_parallel_gpu.sh` 형식으로 만들고, 끝난 run 을 suite × level × arm 별 공식 지표와 `none` 대비 paired 로 집계한다. `smoke` = 12 조합 × (`none`, `aegis`, `constrained_flow_matching`) `t0e0` |
| **L_smoke · gt_smoke · suite_smoke** | V6 의 세 GPU phase. L_smoke = 고친 4 arm 의 새 기본 대 `@v5`(9 job), gt_smoke = privileged 8 변형 + `none`(9 job), suite_smoke = 12 suite × level × 3 arm(36 job) |
| **S / F · `coll` · `no coll`** | suite_smoke 표의 칸. S / F = 그 1 episode 의 official success 여부, `coll` / `no coll` = SafeLIBERO 의 충돌 판정(LIBERO 4 suite 는 충돌 지표가 없다) |
| **`T_world_base`** | world frame 에서 robot base 의 pose. `support_z`(base frame)에 base 의 z 를 더하면 world z 다(표의 값으로 확인) |
| **LIBERO problem class (`Tabletop` · `LivingRoom` · `Floor`)** | LIBERO suite 의 장면 종류. 탁자 · 바닥 높이 상수(`table_offset`, `floor_offset`)가 problem class 마다 정해져 있다 |
| **이웃 FP (BL-9)** | 예측 영역 안에서 다른 물체 · destination 으로 잘못 들어간 5 mm voxel mass. 비율이나 mm³ 가 아니며 예측 크기와 장면 표면 구성에 따라 달라진다 |
| **abstain (BL-9)** | attention surface selector 가 최종 후보(점수 ≥ 0.5 인 region)를 못 내서 그 frame 을 빈 target 예측으로 점수화한 것. `no_region` 이 그 사유 |
| **oracle-assisted target-name 조건 (BL-9)** | 외부 segmenter 의 `phrase` 입력이 BDDL goal 의 정답 객체 이름을 문구로 바꾼 것. GT mask 는 아니지만 정답 이름을 참조한다. Spatial 40/40 episode 에서 지시문만으로 뽑은 문구와 달랐다 |
| **FK sphere mask (BL-9)** | Panda 관절값 · camera calibration · self-filter 기하 모델로 만든 robot 구(球) mask. GT segmentation 은 쓰지 않는다. `GT robot` 은 simulator 가 준 특권 mask |
| **가려짐 비율 (BL-10)** | 1 − visible target pixel / target-only in-frame target pixel. 화면 밖 잘림 · 물체 뒷면은 포함하지 않는다 |
| **target 지움 · 남김 · obstacle 잘못 지움 (BL-12)** | 지움 = 그 arm 의 voxel 이 덮는 GT target 비율(recall), 남김 = 1 − 지움(target 이 장애물로 남는 위험), obstacle 잘못 지움 = 장애물 voxel 이 거리장에서 사라지는 비율(충돌 방향의 위험) |

---

## 공통 고정 사항 (BL-0 ~ BL-1)

| 항목 | 값 |
|---|---|
| branch | `feature/panda-benchmark` |
| 폴더 | `Baseline/paper_baselines/` (`external/` 는 pinned SHA 로 fetch 하고 clone 본체는 추적하지 않는다. `git submodule add` 는 같은 checkout 의 다른 session 의 index 를 건드려 쓰지 않았다. submodule 전환은 사용자 결정) |
| venv | 새 `.venv-baselines`(py3.11, checkout 의 `.python`, uv). `.venv-ag3s` · `.venv-openpi-live` 를 오염시키지 않는다 |
| cache · weights | `HF_HOME` · `TORCH_HOME` · ultralytics weights → checkout `.cache/paper_baselines/` |
| 평가 입력 | `--capture-segmentation` 이 켜진 rollout run 의 `_geom.npz` + scene GT. segmentation 이 있는 recorded run 은 SL3 dev `20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0`(10 ep)과 SL4b dev `20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4`(40 ep) 둘뿐이다. SL1/SL2 replay-seg 는 t0e0 1 ep 씩이고 SL5* closed-loop run 에는 segmentation 이 없다 |
| 결과 위치 | `AG3S_VLA-storage/results/benchmark/<날짜_commit_run-id>/` (`common/run_artifacts.create_run`). implementer smoke 는 `outputs/paper_baselines_smoke/` |
| 손대지 않은 것 | `AG3S/**`, `benchmark/runners/**`, `integration/**`, `third_party/**`, `.gitmodules`, 루트 `.gitignore`. V11AB 평가 대상이 바뀌지 않고, 진행 중인 run 을 건드리지 않기 위해서다 |

---

## BL-0 — PAPER-BASELINES: 논문 비교 baseline 폴더 (2026-10-23)

> 2026-10-23 KST. implementer A 작업 11:58–13:00(`.venv-baselines` 준비 12:20), implementer B 같은 날, commit `bf1403d5`(13:21).
> 원본: [`handoff/PAPER-BASELINES.task.md`](handoff/PAPER-BASELINES.task.md) ·
> [`-A.impl.md`](handoff/PAPER-BASELINES-A.impl.md) · [`-B.impl.md`](handoff/PAPER-BASELINES-B.impl.md).
> **이 STEP 은 측정하지 않았다.** 구성과 test 통과 사실만 기록한다. 측정은 BL-1 이다.

### 이 STEP 이 답하는 질문

논문의 "external segmenter" 비교 행(Grounded-SAM 포함)을 **같은 depth · 같은 lift · 같은 지표로** 채울 수 있는 실행 가능한
baseline 이 이 checkout 에 있는가?

### 왜 지금 필요했나

논문의 핵심 주장 하나는 "observed target surface 를 external segmenter 없이 frozen VLA 의 attention 으로 grounding 한다"이다.
그런데 비교 숫자가 비어 있었다: Tab. 2 의 "External segmenter" 행, Sec. 4.1 의 TBD("Grounded-SAM / YOLOE masks lifted with the
same depth"), Fig. 2 의 "Prior: external segmenter" 열. 이 STEP 은 그 비교를 돌릴 수 있는 baseline 과 offline 평가 harness 를 만든다.

```
RGB(+instruction) ──[provider]──> camera 별 target mask
                                      │
depth + K + T_base_cam ──[lift.py]────┴──> base frame 3D 점 · 5 mm voxel ──> eval/metrics.py vs GT
                                           (같은 점을 LTE / CCR 에 넣는 closed-loop 연결은 다음 STEP)
```

### 무엇을 만들었나

기존 tracked 파일은 하나도 고치지 않았다(전부 새 파일).

**provider 6 개 + ATS 비교 행**

| baseline | 논문 인용 | 역할 | 출처 repo @ SHA | license | 구현 형태 |
|---|---|---|---|---|---|
| `gt_oracle` | — | Tab. 2 의 상한선 · lift 검증 (privileged) | 이 repo(`task_target` + `segmentation_geom`) | — | re-impl |
| `grounded_sam` | bib 에 아직 없음(본문 Sec. 4.1 이 인용) | Tab. 2 "External segmenter", Fig. 2 prior 열 | IDEA-Research/Grounded-Segment-Anything @ `126abe6` (recipe · threshold) | Apache-2.0 | wrapper: HF Grounding DINO + HF SAM v1 |
| `grounded_sam2:per_frame` | bib 에 없음 | 같은 행의 현행 recipe | IDEA-Research/Grounded-SAM-2 @ `b7a9c29` | Apache-2.0 | vendored-run (pinned clone 을 `sam2` 로 설치) + HF Grounding DINO |
| `grounded_sam2:track` | bib 에 없음 | 같은 행의 시간 변형: 한 번 검출 후 SAM 2 streaming tracker | 같음 | Apache-2.0 | vendored-run. `CausalTrack` |
| `yoloe` | [25] | 실시간 open-vocabulary segmentation | THU-MIG/yoloe @ `40cd606` (recipe), runtime `ultralytics==8.4.184` | **AGPL-3.0** | wrapper |
| `localization_heads` | [14] | VLA-internal grounding 선행 연구(Sec. 2) | seilk/LocalizationHeads @ `9ffe219` | pinned SHA 에 LICENSE 파일 없음(README 는 MIT) → code 를 가져오지 않고 다시 썼다 | partial re-impl (B) |
| `tokencut` | [26] | feature graph 기반 object discovery 선행 연구(Sec. 2) | YangtaoWANG95/TokenCut @ `fed52cd` | MIT | re-impl + parity test (B) |
| `ats_v11ab` (Coarse) | — | Tab. 2 "Coarse" 행, 같은 frame | 이 repo 의 `perception_replay` run | — | existing: perception run 의 `target_points` 를 읽는다(`--ats-run`) |

- 그 밖에 catalog 에 올린 기존 항목: π0.5-only[18](`launch_libero_pi05_only.py`), KNOWS[17](BL-2 에서 이동), VLSA/AEGIS[10](`benchmark/SafeLIBERO`, 이동하지 않음), SEAM[31](관련 연구, 비교 arm 이 아님 — 사용자 지시로 쓰지 않는다).
- 공통 interface(`interface.py`)는 provider 가 입력 RGB · depth 와 **같은 pixel frame** 의 bool mask 를 내도록 고정한다. 좌우가 뒤집힌 policy 224 image 와 OpenCV image 의 변환은 `lift.py` 한 곳에서만 하고 test 로 고정했다.
- `eval/offline_grounding.py`(harness)는 recorded `_geom.npz` 위에서 provider → lift → 지표(2D IoU · P · R, 3D soft IoU · P · R, neighbour leak, found rate, latency)를 내고 episode 단위 paired bootstrap 을 한다. `create_run` 으로 `frames.jsonl` · `episodes.jsonl` · `summary.json` · `provenance.json`(provider 와 source run 의 SHA-256)을 쓴다.
- Grounded-SAM 의 threshold 는 task 예시(box 0.35)가 아니라 **공식 demo 의 기본값**(`grounded_sam_demo.py:165-166` 의 box 0.3 / text 0.25)을 썼다. Grounded-SAM-2 는 0.4 / 0.3, YOLOE 는 conf 0.25 / IoU 0.7. 어느 값도 held-out 결과로 고르지 않았다.
- weights 는 preset 두 개다: `default`(큰 모델: grounding-dino-base · sam-vit-huge · sam2.1-hiera-large · yoloe-11l-seg)와 `smoke`(작은 모델). id · revision · sha256 을 provenance 에 남긴다. default weights 도 미리 받아 두어 측정 때 다운로드가 필요 없다.

**환경 · external sources**

- `.venv-baselines`: torch 2.7.1+cu126, torchvision 0.22.1, transformers 4.53.2, sam-2 1.0(editable, Grounded-SAM-2 @ `b7a9c29`), ultralytics 8.4.184, numpy 2.2.6 등 lock 으로 고정. 첫 설치에서 NFS copy 오류가 나 lock 전체를 `--reinstall` 로 다시 깔았다. `opencv-python-headless` 대신 `opencv-python` 하나만 pin 했다(ultralytics 가 요구하고 두 package 가 같은 `cv2` 를 덮어쓴다).
- `external/fetch_sources.py` 가 5 repo 를 pinned SHA 로 shallow fetch 한다: Grounded-SAM-2 `b7a9c29f…`, Grounded-Segment-Anything `126abe63…`, yoloe `40cd606c…`(AGPL-3.0), TokenCut `fed52cd5…`(MIT), LocalizationHeads `9ffe219d…`(LICENSE 파일 없음).

**B 의 문서와 frozen selection**

| 산출물 | 내용 |
|---|---|
| `docs/survey.md` | 구현하지 않는 8 편([1] multi-link, [5] constrained flow matching, [9] NUDGE, [11] RoboGround, [13] GuidedVLA, [15] Lan-o3dp, [19] OmniGuide, [24] OBEYED-VLA)의 공개 code · license · checkpoint · 보고 수치 · 통합 비용. primary source 만 썼다. **표의 수치는 WebFetch 로 arXiv HTML 을 요약 추출한 것이어서 논문에 옮기기 전에 PDF 표와 대조해야 한다.** [5] 와 [9] 는 원문 자체에 표 사이 불일치가 있다 |
| `docs/exemption_ablations.md` | No exemption · All-link exemption 을 현재 V11AB flag 로 표현할 수 있는지 조사(code 수정 없음). 아래 "exemption ablation" 참조 |
| `docs/catalog_vla_internal.md` | localization heads · TokenCut 의 catalog 행과 **원 논문과 다른 점**(아래) |
| `providers/frozen/localization_heads_pi05_libero_spatial_dev_init0-4_official_code.json` · `…_paper.json` | dev 50 episode(SL3 audit + SL4b audit)의 policy call 에서 고른 head 목록. 공식 code 방식(기본)과 논문 방식 두 가지. 같은 입력의 run 순서를 바꿔도 `selection_sha256` 이 같았다 |

### 원 방법과 다른 점 (B, 요약)

| 방법 | 차이 | 결과 해석에 미치는 영향 |
|---|---|---|
| localization heads | 모델이 LLaVA 계열이 아니라 pi0.5(PaliGemma prefix, 18 layer × 8 head). **query 가 "마지막 text token" 이 아니라 "마지막 action token, denoise step 0"** 이다. 한 장이 아니라 camera 2 대(head · left_wrist)를 각각 16 × 16 으로 보고 camera 별로 head 를 고른다. 표본은 RefCOCO 1,000 장이 아니라 LIBERO Spatial dev 의 policy call 이다. 논문 Sec. 4.2 의 표본 간 빈도 집계는 공식 code 에 없어서 논문 규칙(상위 k = 3)을 썼다. `threshold` mask 는 component 가 여럿이면 모두 포함한다(convex hull 단계는 code 에도 없다). `sam_point` 는 원 방법이 아니다(원은 box → SAM) | "action query → image" 이므로 원 방법의 "text query → image" 결과와 같은 것이 아니다. attention 의 image 몫 합이 1 보다 작아 기준 1 의 절대값을 LLaVA 와 비교할 수 없다 |
| TokenCut | ncut 은 공식 `object_discovery.ncut` 를 numpy 로 다시 썼고 같은 입력에서 mask · seed · label · bbox 가 같다(parity 7 case). 전경 = **seed 가 든 connected component**(task 문서의 "largest connected component" 가 아니라 공식 code 를 따랐다). LIBERO 256 × 256 → 16 × 16 patch 라 물체가 몇 patch 밖에 안 된다 | 경계가 16 px 단위다. language 를 쓰지 않으므로 여러 물체 장면에서 target 이 아닌 물체를 잡을 수 있다(원 방법의 성질) |

### exemption ablation 설계 (B, code 수정 없음)

| ablation | 현재 V11AB 로 표현? | 방법 | 남는 문제 |
|---|---|---|---|
| No exemption | 예 | V11AB argv 뒤에 `--target-field-policy relax` | launcher 가 `AG3S_VERSIONS` 의 이름만 받는다. 수동으로 덧붙이면 server 기록의 version 이름이 `V11AB` 로 **잘못 남는다** |
| All-link exemption | 예(flag 는 있다) | `--target-field-policy exclude-all` | **V11AB 는 `--links gripper` 라 constraint link 가 전부 contact link 이므로, V11AB 위에서는 target 을 빼는 sphere 집합이 V11AB 와 같다.** 의미 있는 비교는 `--links arms` 판(pb3 이상)에서만 된다. V11AB 위의 `exclude-all` run 을 All-link exemption 의 증거로 쓰면 안 된다 |

최소 code 변경안(실행하지 않음): `AG3S/adapters/panda/profile.py` 의 `AG3S_VERSIONS` 에 항목 추가(기본 off, 기존 version bit-identical). 실제 수정은 run 이 없을 때 다음 STEP 에서 한다. runtime 동작(`relax` 에서 target-free layer 가 없을 때 기록 경로가 빈 값을 견디는지)은 server 를 띄우지 않아 확인하지 못했다. 이 결론은 Panda 단일 팔에만 해당한다(RB-Y1 은 `GRIPPER_LINKS` 에 양손이 들어 있어 sphere 집합이 다를 수 있다).

### 단위 검증

`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python -m pytest -q Baseline/paper_baselines/tests`

| 범위 | 결과 (impl.md 의 실제 출력) |
|---|---|
| A: interface · lift · metrics · track causality · harness | 29 passed |
| B: localization heads · TokenCut | 28 passed (`.venv-baselines`). `.venv-ag3s`(torch 없음)에서는 18 passed, 10 skipped (torch parity test) |
| 전체 `Baseline/paper_baselines/tests` | 57 passed (A 29 + B 28) |

test 가 고정하는 것: not-found 경로, 지어낸 mask 거부, top-1 동점 규칙, 알려진 K · T round-trip, 무효 depth, robot 제거, policy mirror 규약(`camera.policy_to_opencv_pixel` 과 대조), 손으로 계산한 2-voxel soft IoU, bootstrap 재현성, `CausalTrack` 의 causality(fake predictor 로 t 이후 frame 이 달라도 t 까지의 결과가 같은가), TokenCut parity, head 선택의 결정성, localization heads 의 mirror 규칙이 `lift.policy_mask_to_opencv` 와 일치하는가.

CPU smoke 는 provider 별로 recorded frame 1 장에 smallest weights 로 돌려 **배선만** 확인했다(숫자 판정 없음, 이 파일에 옮기지 않는다). 질적 사실 하나: `yoloe-11s-seg` 는 이 256 px LIBERO frame 에서 기본 conf 0.25 를 넘는 instance 가 없었고, 같은 model 이 다른 사진(`bus.jpg`)에서는 검출했다. 큰 모델(11l)은 smoke 에서 돌리지 않았으며 BL-1 에서 측정했다.

### 이 STEP 에서 알게 된 것 (평이한 설명)

- 비교 baseline 을 "같은 depth · 같은 lift · 같은 지표" 로 돌리려면 provider 가 어떤 pixel frame 의 mask 를 내는가부터 고정해야 했다. policy image 는 OpenCV image 의 좌우 반전이라, 변환을 한 곳(`lift.py`)에만 두고 localization heads 의 grid un-mirror 와 일치하는지까지 test 로 묶었다. B 의 mask 에 `lift.policy_mask_to_opencv` 를 **다시 적용하면 두 번 뒤집힌다.**
- soft IoU 는 정의를 새로 만들지 않고 development branch 의 식을 읽어 같은 식으로 구현했다. 거기에 세 가지를 더했다: (a) eval voxel 밖의 예측 voxel 을 FP 1 로 세는 규칙(`pred_offgrid`), (b) hard 지표 병기, (c) scene GT 로 lift 를 독립 검사하는 `lift within 10 mm`.
- 논문 bib 에 Grounded-SAM · Grounded-SAM-2 항목이 없다(YOLOE [25] 는 있음). 논문에 비교 행을 쓰기 전에 bib 에 추가해야 한다.

### 측정하지 않은 것 / 열린 항목

| 항목 | 상태 |
|---|---|
| GPU 실행 · default weights · latency | 이 STEP 금지 범위. BL-1 에서 측정 |
| HF Grounding DINO / SAM port 와 원 repo(GroundingDINO CUDA op) 출력의 수치 동등성 | **미확인.** 같은 weights · 같은 threshold 규칙이라는 근거만 있다(`docs/providers.md`) |
| SAM 2 CUDA extension | 이 commit 시점에는 build 하지 않았다(`SAM2_BUILD_CUDA=0`). video predictor 의 `fill_hole_area=8` 후처리가 경고와 함께 생략된다. BL-1 에서 build 했으나 sm_90 이 빠진 사실이 드러났다 → BL-4 V2 가 재빌드한 `_C.so` 에 sm_90 cubin · PTX 가 들어 있음을 `cuobjdump` 로 확인했다 |
| DINO ViT-S/16 공식 weight 로 TokenCut | B 단계에서는 weight 가 없어 random-init 로 배선만 봤다. A 가 weight 를 받은 뒤 smoke 에서 `All keys matched successfully` 를 확인했다. 정확도는 BL-1 이 처음 측정 |
| VLA patch feature 생산자 | 없다(attention audit 은 512-d pooled `kv_feature` 만 남긴다). TokenCut `feature_source="aux"` 는 입력이 없어 쓰지 못한다 |
| localization heads 의 적용 범위 | `pooled_f16` 이 있는 frame(SL3 · SL4b dev 의 policy call)에서만 돈다. 다른 frame(test split 등)에 쓰려면 같은 runner 로 audit 을 다시 떠야 한다(GPU) |
| exemption ablation 의 runtime 동작 | 미확인(위) |
| `survey.md` 표 수치 | PDF 대조 전 |

### 판정

**사용자 판정 대기.** 구현 · test 는 끝났고 측정은 BL-1 이다.

---

## BL-1 — PAPER-BASELINES-2: grounding baseline SL3 dev GPU 측정 (2026-10-23)

> 2026-10-23 KST. implementer E 13:34, verifier 측정 13:37–16:03, commit `5752c64c`(16:09).
> 원본: [`handoff/PAPER-BASELINES-2.task.md`](handoff/PAPER-BASELINES-2.task.md) · [`-E.impl.md`](handoff/PAPER-BASELINES-2-E.impl.md) ·
> [`handoff/PAPER-BASELINES-2.verify.json`](handoff/PAPER-BASELINES-2.verify.json). raw 출력은 `outputs/verify/PAPER-BASELINES-2/`.
> figure: [`figures/paper-baselines-2/`](figures/paper-baselines-2/).

### 이 STEP 이 답하는 질문

같은 depth · 같은 lift · 같은 지표에서 external segmenter(Grounded-SAM 계열 · YOLOE)와 VLA-internal baseline 이
LIBERO dev frame 의 target 표면을 얼마나 맞히고, V11AB ATS(Coarse)와 비교해 어떤가?

### 타임라인 (KST)

| 시각 | 사건 | 출처 |
|---|---|---|
| 13:25 | lead 가 SAM 2 CUDA extension 을 build (`_C.so` mtime 04:25:02 UTC) | verify issues |
| 13:34 | implementer E: `--attention-run` 등 harness 보강 완료(CPU) | E impl.md |
| 13:37 | 회귀 기준선 확인(CPU, legacy 16D) — 일치 | verify `baseline_check` |
| 13:39:01 – 13:47:28 | **run 1** (track 포함, `--run-id paper-baselines-2-sl3dev`) — `grounded_sam2:track` 에서 GPU OOM 으로 종료(exit 1, 507 s, 부분 `frames.jsonl` 139 행) | verify `runs[1]` |
| 13:48:24 – 16:03:04 | **run 2** (track 제외 8 spec, `--run-id paper-baselines-2-sl3dev-notrack`) — GPU wall 8,080 s. 이 run 이 아래 모든 수치의 출처다 | verify `runs[0]` |
| 16:09 | commit `5752c64c` | `git log` |

### E 가 harness 에 더한 것 (구현)

- `eval/offline_grounding.py` 에 `--attention-run <attention_audit run>` 을 추가했다. 평가 frame 마다 그 frame 의 policy call 에서 기록한 pi0.5 attention 을 `aux` 로 넘긴다. 짝은 `obs_index` 로 맞추고, audit 의 `source_run` · `npz_sha256` 이 평가 episode 와 다르면 멈춘다.
- `--providers` 에 `localization_heads:threshold|sam_point|sam_box`, `tokencut:foreground|attention_overlap` 을 쓸 수 있다. `localization_heads` · `tokencut` 은 query text 를 읽지 않으므로 arm 이 `<spec>|no_text` 한 개씩이다.
- `provenance.json["attention"]["selection_in_sample"]` 이 frozen selection 의 입력에 이 audit run 이 들어 있으면 `true` 를 자동으로 기록한다(아래 in-sample 의 근거).
- 부수 수정: 모든 provider 에 `size=` 를 넘겨 `localization_heads` · `tokencut` 생성이 TypeError 로 실패하던 것을 provider 별 생성자 인자로 고쳤다. `setup_venv.sh` 의 SAM 2 CUDA build 기본값을 1 로 바꿨다. `localization_heads.py` docstring 의 SAM wrapper 위치를 `sam_backend` 로 고쳤다.
- dry-run: SL3 dev 와 SL4b dev 모두 평가 frame 전부에 audit call 이 있고(`missing_reason` 행 0 이어야 한다), SL3 dev 는 ATS 기록도 전부 있다.
- test: `.venv-baselines` 62 passed(전체 `Baseline/paper_baselines/tests`), `.venv-ag3s` 20 passed 2 skipped. E 의 새 test 5 개가 `:mode` → 생성자 인자 매핑, `aux` 값 일치, source run 불일치 시 ValueError, 두 VLA-internal provider 에 같은 `aux` 를 넣었을 때 `validate_results` 통과, 실제 SL3 frame 의 audit join 을 고정한다.

### 실행

```bash
source scripts/setup/env.sh; source Baseline/paper_baselines/setup/paper_baselines_env.sh
flock -w 3600 /tmp/ag3s-vla-gpu-validation.lock env PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.eval.offline_grounding \
  --source-run 20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0 --scene-gt outputs/verify/SL4/raw/scene \
  --providers gt_oracle grounded_sam grounded_sam2:per_frame yoloe localization_heads:threshold localization_heads:sam_point \
              tokencut:foreground tokencut:attention_overlap \
  --prompt-modes phrase instruction --size default \
  --attention-run 20261012T202459_309e4c6da477_sl3-attention-audit-dev \
  --ats-run 20261012T225820_46a08d6ff644_sl4-perception-dev --save-masks --run-id paper-baselines-2-sl3dev-notrack
```

(verify.json `runs[0].command` 에서는 `--providers` 가 `gt_oracle grounded_sam grounded_sam2:per_frame yoloe localization_heads:threshold …` 순서로 기록돼 있다. 위는 가독성을 위해 줄만 나눈 것이다.)

| 항목 | 값 |
|---|---|
| 측정 집합 | SL3 dev, `libero_spatial` t0–9 e0 — 10 episode · **221 evaluated frame**, camera head + left_wrist |
| weights | `--size default` (큰 모델), device cuda (H200) |
| run dir | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack` |
| mask 파일 | 2,652 개(`--save-masks`) |
| 산출물 sha256 | `frames.jsonl` `05bee04e…078e7`, `episodes.jsonl` `399aeab9…c61e052`, `summary.json` `b558c921…e7425`, `provenance.json` `34a080c3…ca987`, `run_manifest.json` `9b00df25…6101d` |
| code 상태 | HEAD `bf1403d5` + **dirty 7 파일**(`eval/offline_grounding.py`, `providers/localization_heads.py`, `setup/setup_venv.sh`, `setup/requirements-baselines.in` · `.txt`, `tests/test_offline_grounding.py`, `external/sources.json`). 이 dirty 변경(E · C · D 의 것)은 이후 commit `5752c64c` 에 들어갔다. 파일별 sha256 은 `outputs/verify/PAPER-BASELINES-2/code_state.txt` |
| packages | torch 2.7.1+cu126, torchvision 0.22.1, transformers 4.53.2, sam-2 1.0(editable `b7a9c29`), ultralytics 8.4.184, numpy 2.2.6 |
| 분석 · figure script | `outputs/verify/PAPER-BASELINES-2/analyze_run.py`, `make_figures.py` |

**회귀 기준선(legacy 16D, CPU, 13:37)** — 일치. 15 frame 중 위반으로 시작 14, `has_target` 15, frame 0 의 `clearance_before` −29.031 mm. 측정 환경이 성한지 보는 용도다.

### 결과 — Figure

| figure | 종류 | 무엇을 보여 주나 |
|---|---|---|
| [`pb2-sl3dev-scene-overlay.png`](figures/paper-baselines-2/pb2-sl3dev-scene-overlay.png) · [`.json`](figures/paper-baselines-2/pb2-sl3dev-scene-overlay.json) | 실제 scene | `libero_spatial_t9_e0` obs 120: arm 간 head 2D IoU 의 퍼짐이 가장 큰 frame 한 장에서, 두 camera 에 12 arm 의 mask 를 겹쳤다(채움 = 예측 mask, 초록 = GT target 윤곽, 주황 = 같은 종류 distractor 윤곽). 패널마다 mask pixel 수와 2D IoU 가 적혀 있고 값은 sidecar 에 있다 |
| [`pb2-sl3dev-referring-failure.png`](figures/paper-baselines-2/pb2-sl3dev-referring-failure.png) · [`.json`](figures/paper-baselines-2/pb2-sl3dev-referring-failure.json) | 실제 scene | `libero_spatial_t5_e0` obs 105: referring-expression 실패 예시. Grounded-SAM 의 phrase · instruction 두 prompt 모두에서 mask pixel 이 target 이 아니라 same-category distractor(`akita_black_bowl_2`) 위에 놓인 두 camera 의 장면 |
| [`pb2-sl3dev-iou-leak.png`](figures/paper-baselines-2/pb2-sl3dev-iou-leak.png) · [`.json`](figures/paper-baselines-2/pb2-sl3dev-iou-leak.json) | graph | arm 별 3D soft IoU(왼쪽)와 neighbour leak(오른쪽). 점 = episode 평균 10 개, 막대 = episode 간 평균 |
| [`pb2-sl3dev-table.png`](figures/paper-baselines-2/pb2-sl3dev-table.png) · [`.json`](figures/paper-baselines-2/pb2-sl3dev-table.json) | 표 | arm × 지표 전체(아래 표들의 그림판) |

![scene overlay](figures/paper-baselines-2/pb2-sl3dev-scene-overlay.png)

![referring-expression failure](figures/paper-baselines-2/pb2-sl3dev-referring-failure.png)

![3D soft IoU and neighbour leak](figures/paper-baselines-2/pb2-sl3dev-iou-leak.png)

![metric table](figures/paper-baselines-2/pb2-sl3dev-table.png)

### 결과 — #1 3D soft IoU · precision · recall (5 mm voxel, episode 평균 → episode 간 평균)

> ~~**ATS 행은 비교에 쓸 수 없다.**~~ → harness(5 mm voxel) 행 자체는 그대로 쓰지 않는다. ATS 의 비교는 같은 cloud 점 위의 cloud labeling(BL-3)으로 옮겼고, 거기서 쓸 ATS 표현은 BL-5 에서 lead 가 exclusion 5 mm 로 정했다(사용자 판정 대기). 아래는 BL-1 측정 당시의 기록이다. ATS 의 `target_points` 는 voxel-downsample 된 **대표점 표현**이라 이 harness 의 5 mm soft voxel 정답과
> 해상도가 맞지 않는다. 3D soft recall 이 0.068 이고, eval voxel 에 없는 예측 voxel(`pred_offgrid`)이 합계 10,524 개이며
> `found_any` 가 179 / 221 이다. 표현이 원인이라는 설명은 A 의 impl.md(그리고 lead)의 것이고 이 측정이 원인을 분리하지는 않았다.
> 그래서 아래 "ATS 대비 paired gain"(#4)도 **표현 차이가 섞인 수치**다. 같은 점 집합 · 같은 해상도 위의 공정 비교는 BL-3 에서 한다. → BL-3 에서 했다(cloud labeling). 이때 `cloud_target` 이 `has_target` call 의 대부분에서 비어 있다는 별도 문제도 확인됐다(BL-3 결과 C).

| arm | 3D soft IoU | soft P | soft R | hard IoU | found any (call 수 / 221) | lift within 10 mm |
|---|---|---|---|---|---|---|
| GT oracle (phrase) | 0.968 | 0.968 | 1.000 | 0.976 | 221 / 221 (1.000) | 0.994 |
| GT oracle (instruction) | 0.968 | 0.968 | 1.000 | 0.976 | 221 / 221 (1.000) | 0.994 |
| Grounded-SAM (phrase) | 0.515 | 0.577 | 0.717 | 0.516 | 221 / 221 (1.000) | 0.799 |
| Grounded-SAM (instruction) | 0.546 | 0.601 | 0.739 | 0.548 | 221 / 221 (1.000) | 0.802 |
| Grounded-SAM-2 per_frame (phrase) | 0.511 | 0.597 | 0.698 | 0.513 | 220 / 221 (0.995) | 0.798 |
| Grounded-SAM-2 per_frame (instruction) | 0.545 | 0.643 | 0.709 | 0.547 | 219 / 221 (0.991) | 0.822 |
| YOLOE-11L (phrase) | 0.313 | 0.511 | 0.424 | 0.314 | 188 / 221 (0.851) | 0.630 |
| YOLOE-11L (instruction) | 0.110 | 0.377 | 0.134 | 0.110 | 98 / 221 (0.443) | 0.407 |
| localization_heads:threshold (in-sample) | 0.075 | 0.076 | 0.832 | 0.076 | 221 / 221 (1.000) | 0.197 |
| localization_heads:sam_point (in-sample) | 0.197 | 0.255 | 0.378 | 0.198 | 221 / 221 (1.000) | 0.265 |
| TokenCut foreground | 0.080 | 0.081 | 0.737 | 0.081 | 221 / 221 (1.000) | 0.321 |
| TokenCut attention_overlap | 0.053 | 0.056 | 0.539 | 0.054 | 221 / 221 (1.000) | 0.119 |
| ATS V11AB (Coarse) | 0.064 | 0.393 | 0.068 | 0.065 | 179 / 221 (0.810) | 0.535 |

- `gt_oracle` 의 3D soft IoU 는 0.968 이고 다른 모든 arm 의 최댓값은 0.546 이다(verify.json `gt_oracle_3d_iou` · `max_3d_iou_other_arms`). task 의 통과 기준("gt_oracle 3D soft IoU 가 다른 모든 행보다 높아야 lift 가 성하다")의 입력이다. 판정은 사용자.
- `gt_oracle` 의 2D IoU 는 1.000 인데 3D soft IoU 는 0.968, lift within 10 mm 는 0.994 다. A 의 impl.md 는 segmentation 이 RGB · depth 보다 한 physics substep 늦게 render 되기 때문이라고 설명한다(`conventions_json`). 이 측정이 그 원인을 따로 확인하지는 않았다.
- `gt_oracle` 의 phrase · instruction 행이 같은 값인 것은 oracle 이 text 를 읽지 않기 때문이다.

### 결과 — #2 2D 지표 (camera 별)

| arm | head IoU / P / R (found) | left_wrist IoU / P / R (found) |
|---|---|---|
| GT oracle (phrase) | 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 (0.968) |
| GT oracle (instruction) | 1.000 / 1.000 / 1.000 (1.000) | 1.000 / 1.000 / 1.000 (0.968) |
| Grounded-SAM (phrase) | 0.281 / 0.308 / 0.304 (0.977) | 0.842 / 0.841 / 0.875 (1.000) |
| Grounded-SAM (instruction) | 0.353 / 0.369 / 0.378 (0.991) | 0.825 / 0.824 / 0.854 (1.000) |
| Grounded-SAM-2 per_frame (phrase) | 0.242 / 0.334 / 0.256 (0.860) | 0.842 / 0.845 / 0.876 (0.968) |
| Grounded-SAM-2 per_frame (instruction) | 0.286 / 0.442 / 0.302 (0.805) | 0.824 / 0.856 / 0.853 (0.937) |
| YOLOE-11L (phrase) | 0.093 / 0.160 / 0.104 (0.638) | 0.490 / 0.741 / 0.520 (0.579) |
| YOLOE-11L (instruction) | 0.002 / 0.007 / 0.017 (0.249) | 0.155 / 0.755 / 0.166 (0.199) |
| localization_heads:threshold (in-sample) | 0.042 / 0.042 / 0.766 (1.000) | 0.232 / 0.263 / 0.724 (1.000) |
| localization_heads:sam_point (in-sample) | 0.219 / 0.265 / 0.272 (1.000) | 0.287 / 0.286 / 0.326 (1.000) |
| TokenCut foreground | 0.033 / 0.033 / 0.598 (1.000) | 0.547 / 0.554 / 0.764 (1.000) |
| TokenCut attention_overlap | 0.040 / 0.042 / 0.406 (1.000) | 0.240 / 0.242 / 0.525 (1.000) |
| ATS V11AB (Coarse) | - | - |

- 2D 에서 not-found 인 camera 는 GT 가 보이면 IoU 0 으로 들어간다(위 용어). `found` 는 그 camera 의 found rate 다.
- `gt_oracle` 의 left_wrist found 0.968 은 GT target 이 그 camera 에서 보이지 않는 frame 이 있다는 뜻으로 읽힌다(oracle 의 mask 가 비면 not-found 로 센다).

### 결과 — #3 neighbour leak · wrong-object rate

| arm | neighbour leak (voxel mass, frame 평균 → episode 평균) | neighbour fraction (pooled) | wrong-object 3D (frame 수 / 221, 비율) | wrong-object 3D, neighbour 비율 > 0.5 (frame 수, 비율) | wrong-object 2D instance (frame 수 / 비어 있지 않은 frame, 비율) |
|---|---|---|---|---|---|
| GT oracle (phrase) | 1.3 | 0.002 | 0 / 221 (0.000) | 0 (0.000) | 0 / 221 (0.000) |
| GT oracle (instruction) | 1.3 | 0.002 | 0 / 221 (0.000) | 0 (0.000) | 0 / 221 (0.000) |
| Grounded-SAM (phrase) | 370.1 | 0.380 | 62 / 221 (0.281) | 49 (0.222) | 22 / 221 (0.100) |
| Grounded-SAM (instruction) | 341.0 | 0.360 | 55 / 221 (0.249) | 46 (0.208) | 20 / 221 (0.090) |
| Grounded-SAM-2 per_frame (phrase) | 341.2 | 0.375 | 56 / 221 (0.253) | 46 (0.208) | 21 / 220 (0.095) |
| Grounded-SAM-2 per_frame (instruction) | 292.3 | 0.339 | 51 / 221 (0.231) | 41 (0.186) | 20 / 219 (0.091) |
| YOLOE-11L (phrase) | 268.1 | 0.416 | 66 / 221 (0.299) | 63 (0.285) | 45 / 188 (0.239) |
| YOLOE-11L (instruction) | 84.3 | 0.091 | 39 / 221 (0.176) | 26 (0.118) | 33 / 98 (0.337) |
| localization_heads:threshold (in-sample) | 657.3 | 0.079 | 123 / 221 (0.557) | 0 (0.000) | 10 / 221 (0.045) |
| localization_heads:sam_point (in-sample) | 191.7 | 0.051 | 95 / 221 (0.430) | 36 (0.163) | 5 / 221 (0.023) |
| TokenCut foreground | 474.5 | 0.068 | 90 / 221 (0.407) | 0 (0.000) | 15 / 221 (0.068) |
| TokenCut attention_overlap | 919.1 | 0.072 | 189 / 221 (0.855) | 5 (0.023) | 15 / 221 (0.068) |
| ATS V11AB (Coarse) | 33.2 | 0.242 | 55 / 221 (0.249) | 46 (0.208) | - |

### 결과 — #4 ATS 대비 paired gain (episode 단위, 10 episode)

`arm − ATS` 의 episode 평균 차이, paired bootstrap 2,000 회(seed 20261015), 95 % percentile 구간. **ATS 행이 대표점 표현이므로 해석에 주의한다**(#1 의 박스).

| arm | Δ 3D soft IoU [95 % CI] (better / worse) | Δ soft P [95 % CI] (better / worse) | Δ soft R [95 % CI] (better / worse) | Δ neighbour leak [95 % CI] (better / worse) |
|---|---|---|---|---|
| GT oracle (phrase) | +0.904 [+0.876, +0.932] (10 / 0) | +0.575 [+0.423, +0.744] (10 / 0) | +0.932 [+0.905, +0.960] (10 / 0) | -31.9 [-61.7, -4.5] (4 / 6) |
| GT oracle (instruction) | +0.904 [+0.876, +0.932] (10 / 0) | +0.575 [+0.423, +0.744] (10 / 0) | +0.932 [+0.905, +0.960] (10 / 0) | -31.9 [-61.7, -4.5] (4 / 6) |
| Grounded-SAM (phrase) | +0.450 [+0.340, +0.567] (10 / 0) | +0.185 [+0.039, +0.361] (8 / 2) | +0.649 [+0.539, +0.741] (10 / 0) | +336.9 [+222.2, +447.1] (10 / 0) |
| Grounded-SAM (instruction) | +0.481 [+0.367, +0.586] (10 / 0) | +0.208 [+0.067, +0.363] (8 / 2) | +0.670 [+0.567, +0.752] (10 / 0) | +307.8 [+196.9, +425.4] (10 / 0) |
| Grounded-SAM-2 per_frame (phrase) | +0.447 [+0.339, +0.553] (10 / 0) | +0.204 [+0.059, +0.382] (8 / 2) | +0.629 [+0.529, +0.708] (10 / 0) | +308.0 [+181.4, +430.3] (9 / 1) |
| Grounded-SAM-2 per_frame (instruction) | +0.481 [+0.365, +0.590] (10 / 0) | +0.250 [+0.092, +0.430] (8 / 2) | +0.641 [+0.543, +0.722] (10 / 0) | +259.2 [+128.2, +396.0] (10 / 0) |
| YOLOE-11L (phrase) | +0.249 [+0.177, +0.307] (9 / 1) | +0.118 [-0.051, +0.299] (7 / 3) | +0.355 [+0.264, +0.425] (9 / 1) | +234.9 [+145.6, +328.1] (10 / 0) |
| YOLOE-11L (instruction) | +0.045 [-0.023, +0.123] (6 / 4) | -0.016 [-0.198, +0.157] (6 / 4) | +0.066 [-0.008, +0.145] (6 / 4) | +51.1 [-26.4, +156.7] (7 / 3) |
| localization_heads:threshold (in-sample) | +0.011 [-0.017, +0.041] (6 / 4) | -0.316 [-0.469, -0.146] (3 / 7) | +0.763 [+0.733, +0.802] (10 / 0) | +624.2 [+522.6, +732.7] (10 / 0) |
| localization_heads:sam_point (in-sample) | +0.133 [+0.100, +0.165] (10 / 0) | -0.137 [-0.280, +0.022] (3 / 7) | +0.310 [+0.264, +0.352] (10 / 0) | +158.6 [+137.2, +179.1] (10 / 0) |
| TokenCut foreground | +0.016 [-0.020, +0.051] (6 / 4) | -0.311 [-0.468, -0.139] (3 / 7) | +0.669 [+0.578, +0.754] (10 / 0) | +441.4 [+308.3, +591.0] (10 / 0) |
| TokenCut attention_overlap | -0.011 [-0.035, +0.011] (5 / 5) | -0.337 [-0.483, -0.176] (2 / 8) | +0.470 [+0.380, +0.561] (10 / 0) | +886.0 [+797.9, +983.9] (10 / 0) |

- 3D soft IoU 차이의 구간이 0 을 포함하는 arm: YOLOE-11L(instruction), localization_heads:threshold, TokenCut foreground, TokenCut attention_overlap. 나머지는 구간이 0 위다(`gt_oracle`, Grounded-SAM 계열, YOLOE-11L(phrase), localization_heads:sam_point).
- soft precision 차이의 구간이 0 아래인 arm: localization_heads:threshold, TokenCut foreground, TokenCut attention_overlap. 0 을 포함하는 arm: YOLOE-11L 둘, localization_heads:sam_point. ATS 의 soft precision 은 0.393, soft recall 은 0.068 이다.
- phrase 대 instruction 의 차이에는 paired bootstrap 을 따로 하지 않았다(ATS 와의 차이만 구했다). #1 · #3 의 두 prompt 값 차이는 보고만 하고 유의성은 주장하지 않는다.

### 결과 — #5 latency (GPU, 첫 frame 제외)

| arm | frame 당 median / p95 (ms, 두 camera 합, 첫 frame 제외) | camera call 당 median / p95 (ms) |
|---|---|---|
| GT oracle (phrase) | 0.3 / 0.5 | 0.2 / 0.3 |
| GT oracle (instruction) | 0.4 / 0.4 | 0.2 / 0.2 |
| Grounded-SAM (phrase) | 514.4 / 730.4 | 243.1 / 323.9 |
| Grounded-SAM (instruction) | 504.8 / 542.6 | 232.7 / 310.4 |
| Grounded-SAM-2 per_frame (phrase) | 381.0 / 433.0 | 195.7 / 221.7 |
| Grounded-SAM-2 per_frame (instruction) | 376.9 / 443.3 | 194.8 / 223.5 |
| YOLOE-11L (phrase) | 112.1 / 266.6 | 49.2 / 191.6 |
| YOLOE-11L (instruction) | 160.7 / 277.7 | 73.4 / 194.3 |
| localization_heads:threshold (in-sample) | 3.2 / 3.8 | 1.7 / 2.1 |
| localization_heads:sam_point (in-sample) | 325.5 / 357.7 | 148.9 / 220.0 |
| TokenCut foreground | 391.8 / 509.8 | 204.1 / 315.2 |
| TokenCut attention_overlap | 406.4 / 597.5 | 205.9 / 397.9 |
| ATS V11AB (Coarse) | 94.7 / 121.0 | - |

- latency 는 provider 가 잰 벽시계 ms 를 그대로 썼다. CUDA sync 여부는 provider `Timer` 를 따른다(GPU 전용 시간은 `not_measured`).
- ATS 는 V11AB 의 `t_process_s`(**perception 전체 처리**, grounding 단독이 아니다). `localization_heads:threshold` 는 기록된 attention 에서 mask 를 만드는 시간뿐이다(pi0.5 forward 가 들어 있지 않다). 두 값은 다른 provider 의 latency 와 같은 척도가 아니다.
- 첫 call(warm-up) 제외는 summary 가 아니라 verifier 가 `frames.jsonl` 에서 했다(arm 마다 첫 frame 행 제외, n = 220 frame · 440 camera call).

### 결과 — #6 referring-expression: phrase 대 instruction

SL3 dev 의 10 task 는 모두 two black bowls 이다. phrase prompt 는 `akita black bowl` 이라 두 bowl 을 다 가리킨다. 위치 관계는 instruction 문장에만 있다. 표는 #3 의 wrong-object 열이다(frame 수 / 221).

| provider | wrong-object 3D: phrase → instruction | wrong-object 2D instance: phrase → instruction | 3D soft IoU: phrase → instruction |
|---|---|---|---|
| Grounded-SAM | 62 → 55 (0.281 → 0.249) | 22 / 221 → 20 / 221 (0.100 → 0.090) | 0.515 → 0.546 |
| Grounded-SAM-2 per_frame | 56 → 51 (0.253 → 0.231) | 21 / 220 → 20 / 219 (0.095 → 0.091) | 0.511 → 0.545 |
| YOLOE-11L | 66 → 39 (0.299 → 0.176) | 45 / 188 → 33 / 98 (0.239 → 0.337) | 0.313 → 0.110 |

- Grounded-SAM 계열은 instruction 에 위치 관계가 들어 있어도 3D wrong-object 가 0.23–0.25 로 남는다(Grounded-SAM 0.281 → 0.249, per_frame 0.253 → 0.231). 원인은 이 측정이 분리하지 않았다(box 후보별 score 분해 없음, score 최고 1 개를 고르는 선택 규칙의 영향 미확인).
- YOLOE-11L 은 instruction prompt 에서 found rate 가 0.443 으로 떨어져(phrase 0.851) wrong-object 의 분모(frame 수)가 달라진다. 3D wrong-object 가 줄어든 것은 found 가 줄어든 효과가 섞여 있다.
- ATS 의 wrong-object 3D 는 55 / 221, neighbour 비율 > 0.5 는 46 frame 으로 **Grounded-SAM(instruction)과 두 값이 모두 같다.** ~~같은 frame 인지 우연인지는 확인하지 않았다(verifier 에게 확인을 요청할 항목).~~ → BL-4 V2 가 확인: 두 frame 집합은 같지 않다(55 중 교집합 22, 46 중 교집합 14). 우연의 일치다.

### 결과 — #7 in-sample 표시

`localization_heads:threshold` 와 `localization_heads:sam_point` 는 **`in_sample: true`** 다. head 선택에 쓴 audit run 이 `20261012T202459_309e4c6da477_sl3-attention-audit-dev`(이 측정의 SL3 dev)와 `20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4` 이기 때문이다. 이 두 행의 수치는 held-out 이 아니다. `tokencut:attention_overlap` 은 V11AB 의 고정 cell(L8H2, last action token, denoise step 0)의 attention 을 쓰며 head 를 새로 고르지 않는다.

### 실패 · 한계 (verify.json `issues` · `not_measured`)

| # | 사건 | 상세 | 영향 |
|---|---|---|---|
| 1 | **`grounded_sam2:track` GPU OOM** | run 1 의 `libero_spatial_t0_e0` phrase arm 에서 obs 70(track_frames 71)까지 기록된 뒤 obs 75 이전 frame 의 `add_new_frame`(`sam2_video_predictor.py:791` 의 `torch.cat`)에서 OOM(147,696,150,272 / 146,800,640,000 byte), 이어 `CUDA error: unrecognized error code` 로 process 전체가 종료(rc 1, 13:47:28). camera 2 대 × 약 71 frame 에 약 147 GB, ~~**frame · camera 당 약 0.9–1.0 GB 가 누적**된다~~ → BL-4 V2 의 직접 측정(unbounded `memory_allocated` 기울기)은 frame 당 137.45 MiB 다. 두 값의 차이 원인은 분리하지 않았다(BL-4 결과 B). 평가 frame 이 obs 120 이상인 episode 가 있어 현재 code 로는 끝까지 못 돈다. 이 실행은 잠시 공용 GPU 한도 130 GB 를 넘었다 | ~~track 은 SL3 · SL4b 모두 **미측정**.~~ → BL-4 에서 SL3 · SL4b 모두 측정했다. 누적 원인은 측정에서 분해하지 않았다. A 의 impl.md 는 streaming state 의 image tensor 가 episode 안에서 계속 쌓인다고(frame 당 1024² float) 미리 적었고, lead 의 BL-4 task 는 memory bank 로 본다. BL-4 의 G 가 memory 상한을 둔다 → 해소(BL-4 결과 B: bounded 는 143 frame 까지 평탄, SL3 · SL4b 모두 OOM 없이 완주) |
| 2 | **SAM 2 `_C.so` 에 sm_90 이 없다** | `_C.so`(build 13:25 KST)에 sm_70 / 75 / 80 / 86 cubin 만 있고 **sm_90 cubin · PTX 가 없다**(`cuobjdump --list-elf` · `--list-ptx`). 원인(lead 확인): Grounded-SAM-2 `@b7a9c29` 의 `setup.py:101-104` 가 `-gencode` 를 sm_70/75/80/86 으로 고정해 `TORCH_CUDA_ARCH_LIST=9.0` 이 **무시**된다. 영향: video predictor 의 `fill_holes_in_mask_scores` 가 `no kernel image is available` 경고와 함께 생략된다(track 경로). `SAM2ImagePredictor` 는 `max_hole_area=0` 이라 `_C` 를 부르지 않아 **per_frame 은 영향이 없다.** `provenance` 의 `sam2_cuda_ext` 는 import 성공 기준이라 `built` 로 기록된다 | 이번 측정의 `per_frame` 수치에는 영향 없음. track 에만 해당 → 재빌드 후 sm_90 확인(BL-4 결과 A) |
| 3 | provider 예외 하나가 run 전체를 멈춘다 | `offline_grounding` 은 한 process 에서 모든 provider 를 돌려 한 provider 의 예외가 run 전체를 멈춘다. track 을 뺀 8 spec 으로 다시 돌렸다(run 2) | BL-4 의 G 가 arm 단위로 격리하고 실패 행을 기록하도록 고친다 → 고쳤고 V2 의 두 run 에서 `provider_errors` 가 `[]` 였다 |
| 4 | 평가 CPU 경로가 느리다 | GPU wall 8,080 s 중 provider latency 합은 episode 당 약 100 s, 나머지 약 600 s / episode 는 CPU 쪽(lift, eval grid, scene-GT `lift_check` 추정; **profile 하지 않았다**). process CPU 약 2.6 core | SL4b 40 episode 는 episode 당 약 700 s, ~~약 8 h 의 lock 점유로 추정되어 **deferred**(lead)~~ → SL4b 40 episode 전체 arm run 이 wall 3,660 s 에 끝났다(BL-4. thread 2 · arm 구성이 달라 속도 개선의 기여는 분리하지 않았다). BL-4 의 G 가 profile 후 vectorize / cache 했다 |
| 5 | `gt_oracle` 2D 1.000 대 3D 0.968 | 위(#1) | 설명은 A impl.md 의 것 |

**SAM 2 sm_90 재빌드 (기록 시점의 상태).** 이 측정 뒤에 `setup_venv.sh` 가 고쳐졌다. `NVCC_APPEND_FLAGS="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90"` 를 export 해 nvcc 의 환경 hook 으로 sm_90 cubin 과 PTX 를 덧붙이고, 주석에 `cuobjdump --list-elf external/Grounded-SAM-2/sam2/_C.so` 로 확인하라고 적었다(lead, commit `5752c64c`). `_C.so` 의 mtime 은 측정이 끝난 뒤(16:08 KST)로 갱신되어 있다. ~~**재빌드한 `_C.so` 에 sm_90 이 실제로 들어갔는지는 verify.json 에 없다**(`cuobjdump` 결과 미기록). BL-4 의 V2 가 확인해야 한다.~~ → BL-4 V2 가 확인했다: `cuobjdump --list-elf` 가 sm_70 / 75 / 80 / 86 / **90**, `--list-ptx` 가 sm_90 을 보인다(BL-4 결과 A). 한편 이번 `numbers` 의 어떤 값도 이 재빌드에 의존하지 않는다(per_frame 은 `_C` 를 쓰지 않고, track 은 측정하지 못했다).

### 측정하지 않은 것

| 항목 | 이유 |
|---|---|
| `grounded_sam2:track` (SL3, SL4b) | **deferred**(lead 2026-10-23). 위 #1 OOM → BL-4 에서 측정 |
| SL4b dev 40 episode 전체(`#2` 의 요청) | **deferred**(lead 2026-10-23). episode 당 약 700 s, 약 8 h 의 lock 점유 추정. 평가 CPU 경로 개선(BL-4) 뒤에 한다 → BL-4 에서 측정 |
| latency 의 GPU 전용 시간 | provider 가 잰 벽시계 ms 를 그대로 썼다 |
| 판정 기준이 있는 통과 / 실패 | 이 측정은 비교용이다(gt_oracle 만 "다른 모든 행보다 높아야" 기준이 있다) |
| 공정 비교(같은 점 집합 위의 ATS) | BL-3 → BL-3 에서 측정 |
| 고유 지표(box Acc@0.5, CorLoc, pointing game, maxFβ, J / F / J&F 등) | BL-3 → BL-3 에서 측정 |

### 열린 항목

1. ~~ATS 와의 공정 비교(V11AB perception run 이 기록한 같은 fused / filtered cloud 위에서 점마다 label) — BL-3.~~ → BL-3 에서 했다.
2. ~~`grounded_sam2:track` memory 상한 · SL3 track 측정 · SL4b 40 episode — BL-4.~~ → BL-4 에서 했다.
3. ~~sm_90 재빌드의 `cuobjdump` 확인 — BL-4(V2).~~ → BL-4 에서 확인했다.
4. ~~ATS 와 Grounded-SAM(instruction)의 wrong-object 값이 일치하는 이유 확인(위 #6).~~ → 우연의 일치였다(BL-4).
5. localization_heads 를 held-out 위에서 돌리려면 test split 의 attention audit 이 필요하다(GPU).
6. 논문 bib 에 Grounded-SAM · Grounded-SAM-2 항목 추가(BL-0).

### 판정

**사용자 판정 대기.**

---

## BL-2 — PAPER-METHODS: 논문 비교 방법 `methods/` 구성 (2026-10-23)

> 2026-10-23 KST. implementer C(safety filter 계열) 작성 15:50 경, implementer D(sampling guidance 계열) 같은 날, commit `5752c64c`(16:09).
> 원본: [`handoff/PAPER-METHODS.task.md`](handoff/PAPER-METHODS.task.md) · [`-C.impl.md`](handoff/PAPER-METHODS-C.impl.md) · [`-D.impl.md`](handoff/PAPER-METHODS-D.impl.md).
> **이 STEP 은 측정하지 않았다.** 구성 · 원 방법과의 차이 · test 통과 사실만 기록한다. smoke 수치는 `impl.md` 에 있지만 verify.json 이 없으므로 옮기지 않았다. ~~수치는 BL-3 이후에 채운다.~~ → offline 수치는 BL-3 에 채웠다.

### 이 STEP 이 답하는 질문

논문의 비교 · 관련 방법(safety filter · sampling guidance)을 `paper_baselines` 에 구성해, recorded Panda / LIBERO frame 에서
**끝까지 실행되어 수정된 action chunk 또는 guidance 결과를 내는가**? ("기능이 작동한다"의 기준. closed-loop 연결은 다음 STEP.)

### 무엇을 만들었나

공통 interface 는 `methods/interface.py`(C): `SceneObservation` · `FilterResult` · `SafetyFilter` · `SamplingGuidance`(용어 절)와
registry `get_method(name)`. contract 검사(`validate_filter_result`)와 openpi `Normalize` 와 같은 식의 `unnormalize_actions` / `normalize_actions` 를 함께 둔다.

| method | 인용 | 종류 | 출처 | license | 구현 형태 | 원 방법과의 주요 차이 |
|---|---|---|---|---|---|---|
| **KNOWS** (`knows`) | [17] | `SafetyFilter` (ellipsoid EE CBF-QP) | 기존 `Baseline/knows_vla` 를 `git mv` 로 `methods/knows` 로 이동(history 보존) + `adapter.py` | — | existing + adapter | 공식 code 가 없어 이전 P0–P3 에서 우리가 재구현한 것이다. target 제외는 attention(논문 Eq. 4)이 아니라 privileged BDDL target + destination. 물체 분할은 GT seg. chunk 뒤쪽 action 은 실측 gain 으로 예측한 EE 자세에서 filter |
| **VLSA / AEGIS** (`aegis`) | [10] | `SafetyFilter` (CBF-QP) | 공식 `benchmark/SafeLIBERO/main/{utils,main_aegis}.py`(submodule, **수정 금지**). helper 는 import, script 안의 per-step QP 는 원 줄 번호를 주석으로 달아 이식 | SafeLIBERO LICENSE | wrapper | **VLM(GLM API) 단계를 쓰지 않았다**(외부 API 금지). 대체: (a) privileged 장애물 이름 + Grounding DINO, (b) Grounding DINO 일반 prompt − target box, (GT) segmentation mask. 장애물마다 constraint row 를 둔다. box crop 에서 Panda sphere(+2 cm) 안의 점을 제거(원본에 없음). open3d DBSCAN → scipy. 원본은 agentview + backview 1024 px, 여기는 head 256 px. QP 실패 시 원본은 `NameError` 로 episode 종료, 여기는 nominal 을 내고 log. 공식 code 의 회전 규약(OSC world-frame 회전을 body-frame ω 로 사용)은 그대로 두었고 검증하지 않았다 |
| **Multi-link safety filter** (`multilink_cbf`) | [1] | `SafetyFilter` (link 별 CBF-QP) | 공개 code 없음(`survey.md`) → 논문(arXiv:2609.40007) Eq. 2 로 구현 | — | re-impl (partial) | SO-101 5 ellipsoid → Panda gripper / wrist / forearm × 3(adapter sphere model 의 MVEE). **moving-hazard optical-flow 이동 생략**(기본 off = 논문의 "frozen at reset"). VLM 대신 AEGIS 와 같은 대체. scale factor(논문 미기재)는 AEGIS 의 값을 썼다 |
| **OmniGuide** (`omniguide`) | [19] | `SamplingGuidance` (attractor / repeller energy) | `daniilidis-group/OmniGuide`(code 는 submodule `omniguide-openpi` `663f1cb2…`) | openpi 부분 Apache-2.0, 상위 repo 는 LICENSE 없음 | port | DROID joint velocity + FK → LIBERO EE-delta + OSC response model. network 를 거치는 backprop 을 하지 않는다. ESDF Chebyshev 근사 → 정확한 거리. best-of-N noise 선택과 subtask monitor 생략. LIBERO 에서는 action → 변위 gain 이 작아 attractor 가 약하다(D 의 설명) |
| **Constrained flow matching** (`constrained_flow_matching`) | [5] | `SamplingGuidance` (CBF 제약 + min-norm 수정) | `Dubascudes/VLA-Safety-ConstrainedFlowMatching` @ `b91153ad…` | **없음 → code 를 복사하지 않았다** | re-impl | 논문 Eq. 7–11 과 repo 기본값을 다시 적었다. SciPy SLSQP → jit SQP(Hildreth). greedy fallback · ellipsoid primitive 없음. obstacle 을 주지 않으면 depth 점 구름 전체가 제약이 된다(원은 SafeLIBERO GT sphere 만) |
| **Lan-o3dp guidance** (`lan_o3dp_guidance_adapted`) | [15] | `SamplingGuidance` (obstacle cost) | `qqianfeng/lan-o3dp` @ `b89c2898…` | MIT | adapted | 원 policy 는 RLBench 용이고 checkpoint 가 없어 **guidance 만** pi0.5 `guidance_fn` 으로 옮겼다. DDIM → flow, 절대 위치 → EE-delta(pseudo-inverse). **원 방법의 결과가 아니다.** 이름의 `adapted` 가 그 표시 |
| **NUDGE** (`nudge`) | [9] | `SamplingGuidance` | 공개 code 없음 → 논문(arXiv:2609.35231)으로 구현 | — | re-impl (partial) | hand + finger sphere 22 개와 OSC path 만 본다(팔 link 와 IK 없음). `d_safe` 0.02 는 논문에 값이 없어 우리가 정했다. **κ 는 SL3 dev call 로 calibrate 한 값(`KAPPA_SL3_DEV`)이다** → 이후 SL3 dev 위의 평가는 in-sample 이다 |
| RoboGround · GuidedVLA · OBEYED-VLA | [11] [13] [24] | 학습이 필요한 grounding policy | `ZzZZCHS/RoboGround` @ `25a86263…`(MIT), `GuidedVLA/GuidedVLA` @ `04be059e…`(Apache-2.0), `UARK-AICV/DRAGON_VLA` @ `bdf25556…`(license 없음) | | fetch + README | 실행하지 않았다. 입력 · checkpoint · 우리 protocol 에 넣으려면 필요한 것만 README 에 적었다 |

- **guidance 의 hook:** openpi 의 `pi0.sample_actions` 에 이미 있는 `guidance_fn(x_next, t_next)`(`third_party/openpi/src/openpi/models/pi0.py:226-283`)에 들어간다. openpi 는 수정하지 않았다. action 은 normalized 공간이라 `norm` 으로 unnormalize / normalize 한다. 네 방법 모두 `enabled=False` 이면 입력 객체를 그대로 돌려준다.
- **외부 repo 7 개를 full SHA 로 pin 하고 fetch 했다**(요청은 6 개). OmniGuide 의 실제 code 가 submodule `omniguide-openpi` 에 있어 하나를 더 넣었다. **license 가 없는 repo([5] · [24])의 code 는 tracked 파일로 복사하지 않았다**(참조만). `fetch_sources.py --check --used-by D` 결과 7 개 모두 `ok`.
- **KNOWS 이동의 호환:** 수정 금지 파일이 옛 이름 `Baseline.knows_vla` 를 import 한다(`AG3S/ag3s/stages/geometry.py:188` 이 `mvee` 를 재사용하고, 없으면 조용히 covariance fit 으로 바뀐다. 그 밖에 `benchmark/__init__.py:12`, `scripts/setup/check_workspace.py:19`). 그래서 `Baseline/knows_vla/__init__.py` 를 **호환 alias package**(MetaPathFinder, 새 module 과 같은 module 객체)로 새로 두었다. `doc/migration/validation/*.json` 의 옛 경로는 provenance 기록이라 고치지 않았다.
- **`.venv-baselines` 설치(C 만):** `cvxpy 1.7.3`, `osqp 1.1.3` 과 의존 6 개(신규 8 개). 기존 package 의 변경 · 재설치 없음. `requirements-baselines.in` / `.txt` 를 갱신했다.

### KNOWS rename 이 commit `88f0c735` 에 섞인 경위

`git mv Baseline/knows_vla Baseline/paper_baselines/methods/knows` 는 implementer C 가 PAPER-METHODS 에서 한 이동이다. `git mv` 는 rename 을 index 에 바로 올리므로, 같은 checkout 의 다른 session 이 BM-SETUP-B(LIBERO-Safety 평가 환경 셋업, 15:46 KST)를 commit 할 때 index 에 올라 있던 이 rename 이 함께 담긴 것으로 보인다(경위는 `git show 88f0c735` 의 파일 목록에서 추정한 것이고, 두 session 이 직접 확인한 기록은 없다).

| commit | 시각 (KST) | 내용 | 상태 |
|---|---|---|---|
| `88f0c735` | 15:46 | BM-SETUP-B(LIBERO-Safety)의 파일 + `Baseline/knows_vla/**` → `methods/knows/**` 의 **순수 rename(R100, 내용 무수정)** | 이 commit 의 tree 에는 `Baseline/knows_vla` 가 **없다.** 그런데 `AG3S/ag3s/stages/geometry.py` · `benchmark/__init__.py` · `scripts/setup/check_workspace.py` 는 여전히 `Baseline.knows_vla` 를 참조한다 → **이 commit 단독 checkout 은 `Baseline.knows_vla` import 가 깨진 중간 상태다**(`git ls-tree` · `git grep` 으로 확인) |
| `5752c64c` | 16:09 | import 이름 · 경로 문자열 갱신(`M`), `adapter.py`, 호환 alias `Baseline/knows_vla/__init__.py` 등 PAPER-METHODS 의 나머지 | 여기서 복구된다. **현재 HEAD 는 정상** |

의미: KNOWS 이동의 history(`git log --follow`)는 `88f0c735` 의 rename 으로 이어지지만, 그 commit message 와 `BM-SETUP-B.impl.md` 는 이 이동을 언급하지 않는다. `git bisect` 에서 `88f0c735` 가 나오면 LIBERO-Safety 변경이 아니라 이 rename 때문에 깨진 것일 수 있다. 이 기록이 그 설명이다. 이후 commit 을 다시 쓸지(squash · 순서 조정)는 사용자 결정이며 이 파일은 사실만 남긴다.

### 단위 검증

| 범위 | 환경 | 결과 (impl.md 의 실제 출력) |
|---|---|---|
| KNOWS 이동 전 `Baseline/knows_vla/tests` | `.venv-ag3s`(MuJoCo) | 49 passed (`test_cbf` 42 + `test_mujoco_bridge` 7) |
| KNOWS 이동 후 `methods/knows/tests`(원 2 파일) | `.venv-ag3s` | 49 passed — test id 집합이 경로 치환 뒤 동일 |
| C 신규 + KNOWS 원 test | `.venv-baselines`(MuJoCo 없음) | 74 passed, 1 skipped (skip = `test_mujoco_bridge` module) |
| alias 확인: `tests/ag3s/test_geometry.py` | `.venv-ag3s` | 47 passed (`fit_ellipsoid_mvee` 가 alias 를 통해 KNOWS `mvee` 를 실제로 import 함을 확인) |
| D: `tests/test_methods_guidance.py` | `.venv-openpi-live`, `JAX_PLATFORMS=cpu` | 35 passed, 1 skipped (skip = OmniGuide × `points` repeller. OmniGuide 가 EE 5 cm 안의 점을 일부러 지우는 원 설계라 이 합성 장면에서는 앞쪽 반구가 사라진다. prim 경우로 대신 확인) |

`.venv-ag3s` · `.venv-baselines` 에는 JAX 가 없어 D 의 test 파일은 skip 된다 — **반드시 `.venv-openpi-live` 로 돌려야 한다.**

test 가 고정하는 것(요약): registry 의 Protocol 만족 · contract 위반 검출 · normalize 왕복 · 공식 AEGIS 함수가 submodule 파일에서 온다는 것 · 공식 gradient 와 유한차분의 일치 · **장애물 1 개 + 직진 action 에서 위반 chunk 는 수정되고 안전한 chunk 는 bit 동일** · multi-link 의 constraint row 가 h 변화량을 예측한다 · **forearm 위 hazard 에서 multi-link 는 개입하고 EE-only AEGIS 는 개입하지 않는다**(multi-link 를 따로 두는 이유) · guidance off 이면 bit 동일 · 장애물이 없으면 identity · repeller 가 waypoint 를 밀어낸다 · CFM 수정 뒤 제약 위반이 해소된다 · NUDGE 가 논문의 flow 식을 따른다.

recorded SL3 frame 의 CPU smoke 는 세 filter 와 네 guidance 모두 **끝까지 실행되어 contract 를 만족하는 수정 chunk 를 냈다**(impl.md). 수치는 `impl.md` 에만 있어 여기에 옮기지 않는다. smoke 가 보인 비수치 사실만 옮긴다.
- GDINO-tiny 256 px 의 box 는 크고 부정확했다(C). AEGIS 원 해상도(1024 px, backview)와 같은 조건은 재현하지 않았다.
- NUDGE 는 처음 call 마다 ℓ∞ 로 정규화했더니 scene 의 hand SDF 가 오히려 나빠졌다. 작은 gradient 에도 큰 normalized step 이 걸려 overshoot 하기 때문이었다. 그래서 논문 정의대로 calibration set 의 중앙값(`kappa`)을 기본값으로 바꿨다(D).
- 모든 guidance 출력이 유한했고 같은 입력을 두 번 돌리면 같은 값이 나왔다(`deterministic`).

### 측정하지 않은 것 / 한계

| 항목 | 상태 |
|---|---|
| closed-loop(SafeLIBERO Level I / II, LIBERO)에서의 성공률 · 충돌률 | 다음 STEP(~~BL-5~~ → BL-6). `benchmark/runners` 연결이 필요하다. per-step(T = 1) 호출이 원본과 같은 timing 이다 |
| pi0.5 network 위에서의 guidance 효과 | **미실행.** CPU 에서는 recorded-chunk 대체 velocity 의 양 끝 경우(`tracking` / `constant`)만 봤다. GPU smoke 명령은 `methods/guidance_common/gpu_smoke.py` docstring · README 에 준비했다(verifier 몫). guidance closure 가 scene 을 상수로 잡아 call 마다 다시 trace 되므로 closed-loop 에서는 scene 을 traced 인자로 바꾸는 편이 낫다(D) → BL-3 에서 같은 대체 velocity 로 SL3 dev 221 call 전체를 CPU 에서 측정했다(pi0.5 network 위 효과는 여전히 미실행) |
| non-privileged 입력 | smoke 는 GT mask(privileged)만 썼다. `extras["target_mask"]` 로 grounding provider 와 연결하는 경로는 code 에만 있다 |
| SafeLIBERO recorded frame | segmentation 을 기록한 run 이 SL3 LIBERO 뿐이라 smoke 하지 않았다. SafeLIBERO 에서는 `privileged/obstacle_names` 를 env 의 `*obstacle*` joint 이름으로 채워야 한다(`main_aegis.py:238-248`) |
| 공식 AEGIS 의 ω frame 규약 | 원본 그대로 두고 검증하지 않았다 |
| CFM 의 ellipsoid primitive · greedy fallback, NUDGE 의 팔 link sphere(IK) | 구현하지 않았다 |
| KNOWS relaxed normal | 논문 기본 relaxed normal 이 실제 barrier 를 `(1−γ)h` 보다 빨리 줄이는 경우가 있다(KNOWS docs/16 의 기존 측정과 같은 현상, 합성 test 에서도 재현) |
| `survey.md` 표 수치 | PDF 대조 전 |
| `methods/README.md`(method 목록) | C / D 공유라 만들지 않았다. 목록은 `Baseline/paper_baselines/README.md` 의 Methods 절에 있다 |

**in-sample 주의(BL-3 이후 측정에서).** NUDGE 의 κ 와 localization heads 의 head 목록은 SL3 dev(와 SL4b dev)로 정한 값이다. 같은 집합 위에서 재면 held-out 이 아니다. 평가 집합을 정할 때 표시해야 한다. → BL-3 · BL-4 의 NUDGE · localization heads 수치는 in-sample 로 표시했다.

### 판정

**사용자 판정 대기.** ~~수치는 BL-3 이후에 채운다.~~ → offline(open-loop) 수치는 BL-3 에 채웠다(결과 E). closed-loop 수치는 ~~BL-5~~ → BL-6.

---

## BL-3 — BASELINE-EVAL V1: cloud labeling 공정 비교, 고유 지표, methods offline (2026-10-24)

> ~~(진행 중)~~ → 2026-10-24 KST 측정 완료. 아래 "계획"은 착수 시 기록이고, 결과는 그 뒤의 "결과 보고"에 이어 썼다.

> 2026-10-23 KST 시작 지시. 원본: [`handoff/BASELINE-EVAL.task.md`](handoff/BASELINE-EVAL.task.md)(순서 1). ~~**verify.json 이 아직 없어 수치를 쓰지 않는다.**~~ → 수치는 아래 "결과 보고"에 적었다(출처 `BASELINE-EVAL-V1.verify.json`).
> implementer F(새 파일만, CPU) → verifier V1(CPU)이 `BASELINE-EVAL-V1.verify.json` 과 figure 를 만든다.

**무엇을 하나.** 논문 핵심 주장에 바로 필요한 것을 먼저 한다. BL-1 의 ATS 행은 대표점 표현 때문에 recall 0.068 이라 비교로 쓸 수 없었다. SL3 run 이 저장한 mask 를 다시 쓰므로 CPU 만 든다.

| 하위 항목 | 내용 |
|---|---|
| 지표 registry `docs/metrics.md` | 각 원 논문 · benchmark 문서(primary source)에서 지표 정의를 확인한다: 정의(식), 출처(논문 절 · 표 번호 또는 공식 code 줄), 우리가 지금 잴 수 있는가(`offline-now` / `closed-loop` / `not-applicable` 과 이유). 대상: AG3S(ours), Grounded-SAM, Grounded-SAM-2(track), YOLOE, localization heads, TokenCut, GT oracle, VLSA / AEGIS · SafeLIBERO, KNOWS, multi-link, CFM, NUDGE, OmniGuide, Lan-o3dp, LIBERO, SafeLIBERO, LIBERO-Safety |
| grounding 고유 지표 `eval/native_metrics.py` | SL3 run 의 `--save-masks` 산출물을 사후 계산: box Acc@0.5, CorLoc, pointing game(예측 peak 이 GT 안에 있는가), mIoU / oIoU, maxFβ, J / F / J&F(track 결과가 생기면 같은 code) |
| **ATS 와의 공정 비교** `eval/cloud_labeling.py` | V11AB perception run 이 기록한 **같은 fused / filtered cloud 점 집합** 위에서 점마다 label 을 매긴다: GT = scene GT 의 가장 가까운 물체(거리 임계 포함), ATS = target flag, 각 segmenter = 그 점이 어느 camera 에서든 mask 안에 투영되는가. 그 위에서 point-level IoU · P · R · neighbour 비율과 paired bootstrap. perception run 이 cloud 를 기록하지 않았다면(`--no-clouds`) 그 사실을 보고하고 대안(ATS target 점을 같은 voxel 로 dilate 하는 등)을 제안만 한다 |
| methods offline 평가 `methods/evaluate_offline.py` | SL3 dev 10 episode 전체 call 에서 7 method(KNOWS, AEGIS 3 변형, multi-link, OmniGuide, CFM, Lan-o3dp adapted, NUDGE)를 CPU 로 돌린다. 지표: intervention rate, ‖Δa‖, barrier h_min, 제약 위반, QP 실패, solve time, scene GT 기준 예측 EE / link clearance. guidance 는 recorded-chunk 대체 velocity 를 쓰고 그 한계를 산출물에 기록한다. figure 용 dump(ellipsoid · h trace · energy slice · path before / after · attention map · eigenvector)를 남긴다 |
| figure(규칙 A) | `figures/baselines/<module>/` 에 module 마다 실제 장면 · 핵심 기능 · graph 또는 표 세 종류와 `.json` sidecar. 지표 표는 `docs/metrics.md` 의 모든 행에 "측정값 또는 `closed-loop 에서 측정`" 이 채워진 표 하나 |

수정 금지: `eval/offline_grounding.py`, `providers/grounded_sam2.py`, `lift.py`(G 소유), `AG3S/**`, `benchmark/**`, `third_party/**`.

~~**상태: 진행 중. verify 가 생기면 이어 쓴다. 판정 없음.**~~ → **측정 완료, 사용자 판정 대기.**

### 결과 보고 (2026-10-24 KST)

> 2026-10-24 KST. verifier V1 의 CPU 측정은 03:58 KST(UTC 2026-10-23 18:58)에 시작했다(verify `code_state.note`). run dir 이름의 시각은 KST 다: native `035852`, cloud `035953`, methods `040141`.
> 원본: [`handoff/BASELINE-EVAL-F.impl.md`](handoff/BASELINE-EVAL-F.impl.md) · [`handoff/BASELINE-EVAL-V1.verify.json`](handoff/BASELINE-EVAL-V1.verify.json). raw 출력은 `outputs/verify/BASELINE-EVAL-V1/`, run dir 은 `AG3S_VLA-storage/results/benchmark/20261024T035852_5752c64c7006_baseline-eval-native-sl3dev` · `…035953_…_baseline-eval-cloud-sl3dev` · `…040141_…_baseline-eval-methods-sl3dev`.
> 지표 registry: [`Baseline/paper_baselines/docs/metrics.md`](../../../Baseline/paper_baselines/docs/metrics.md). figure: [`figures/baselines/`](figures/baselines/) 아래 `v1-*` 51 개(각각 같은 이름의 `.json` sidecar 가 있다).
> **판정은 사용자 판정 대기.** 아래는 수치와 관찰 가능한 사실만 적는다.

### 이 STEP 이 답하는 질문 (BL-3)

1. BL-1 에서 "비교로 쓸 수 없다"고 한 ATS 행을, ATS 와 segmenter 가 **같은 점 집합 위에** 놓이게 하면 어떻게 되는가? (cloud labeling)
2. 각 baseline · benchmark 논문이 요구하는 지표는 무엇이고, 지금 있는 기록(SL3 dev 의 저장 mask · V11AB perception cloud · 기록된 action chunk)으로 얼마나 잴 수 있는가? (registry + 고유 지표)
3. methods 9 arm 을 recorded π0.5 chunk 에 걸면 offline 지표는 어떻게 나오는가? (open loop)

### 구현 경위 — implementer F (새 파일만, GPU 없음)

기존 tracked 파일은 고치지 않았다(`offline_grounding.py` · `metrics.py` · `lift.py` 는 G 소유라 읽기만 했다). F impl.md 에서 옮긴 구성 사실이다.

| 산출물 | 무엇을 하나 |
|---|---|
| [`docs/metrics.md`](../../../Baseline/paper_baselines/docs/metrics.md) | 지표 registry. 대상 21 행(AG3S · grounding 6 · ATS · method 7 · benchmark 3 · pointing game). 항목마다 정의 · 출처(논문 절 · 표 번호 또는 공식 code 줄) · status(`offline-now` / `closed-loop` / `not-applicable`) · 우리 대응 |
| `eval/native_metrics.py` | SL3 run 의 `--save-masks` 산출물에서 각 논문 고유의 2D 지표를 사후 계산: Acc@0.5, CorLoc, pointing game, mIoU / oIoU(cIoU), maxFβ, J / F / J&F, detector `conf` sweep. J / F 는 pinned Grounded-SAM-2 의 `sav_benchmark.py` 와, saliency 지표는 pinned TokenCut 의 `metric.py` 와 같은 값이 나오는지 test 로 확인 |
| `eval/cloud_labeling.py` | V11AB perception run 이 기록한 **같은 fused / filtered cloud 점** 위에서 GT · ATS · segmenter 가 점마다 label 을 매기고 point IoU · P · R · neighbour 비율과 ATS 대비 paired bootstrap 을 낸다. SL4 perception run 이 `cloud_points` 를 기록해 두었으므로 대안(`--no-clouds`) 경로는 쓰지 않았다 |
| `methods/evaluate_offline.py` | `filters` · `guidance` · `summarize` 하위 명령. 모든 method 에 같은 decode(OSC response) · IK · 61 sphere 로 scene GT clearance 를 잰다. guidance 는 recorded-chunk 대체 velocity(BL-2 용어)를 쓴다 |
| `tests/test_native_metrics.py` · `test_cloud_labeling.py` · `test_evaluate_offline.py` | DAVIS J / F · `seg2bmap` 이 pinned `sav_benchmark.Evaluator` 와, TokenCut maxFβ · Acc · IoU 가 pinned `metric.metrics` 와 같은 값인지, 실제 SL3 frame 의 IoU 가 harness `frames.jsonl` 의 `2d_iou` 와 같은지, 거리 pruning 이 `exact_distance_near` 와 bit 단위로 같은지, trace 판 sampler 가 D 의 `surrogate_sample` 과 bit 단위로 같은지 등을 고정한다 |

F 의 smoke 수치는 옮기지 않았다(수치 규칙). V1 이 test 를 돌렸다: `.venv-baselines` 32 passed 1 skipped, `.venv-openpi-live` 11 passed(verify `unit_tests_F`).

### 실행 — verifier V1 (CPU, worker 1, `CUDA_VISIBLE_DEVICES=''`)

모든 사용 code 의 mtime 이 실행 시작 이전이다(`code_state.note`). 회귀 기준선(legacy 16D, CPU)은 일치했다: 위반으로 시작 14 / 15, `has_target` 15 / 15, frame 0 의 `clearance_before` −29.031 mm.

| run | venv | wall (s) | max RSS (GB) | exit |
|---|---|---|---|---|
| native_metrics (`--tokencut-dino`) | `.venv-baselines` | 61.5 | 0.93 | 0 |
| cloud_labeling | `.venv-baselines` | 107.7 | 0.21 | 0 |
| methods filters (5 arm) | `.venv-baselines` | 870.4 | 2.51 | 0 |
| methods guidance (4 method × 2 variant × 2 surrogate) | `.venv-openpi-live`, `JAX_PLATFORMS=cpu` | 3,357.9 | 2.69 | 0 |
| methods summarize | `.venv-baselines` | 0.3 | 0.03 | 0 |
| ATS `target_points` 보충 계산 (verifier) | `.venv-baselines` | 93.2 | 0.22 | 0 |

측정 집합은 BL-1 과 같다: SL3 dev, `libero_spatial` t0–9 e0, 10 episode · 221 policy-call frame, camera head + left_wrist. grounding mask 는 BL-1 의 run `20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack`, V11AB cloud 는 SL4 perception run `20261012T225820_46a08d6ff644_sl4-perception-dev`, methods 의 chunk 는 SL3 dev run `20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0` 에서 읽었다.

### 결과 A — 지표 registry

각 논문의 원 지표를 primary source(논문 절 · 표 번호, 공식 code 줄)에서 확인해 `metrics.md` 에 정리했다. status 의 뜻은 다음과 같다.

| status | 뜻 |
|---|---|
| `offline-now` | 지금 있는 기록만으로 CPU 에서 잰다 |
| `closed-loop` | rollout(성공 · 충돌 · episode 길이)이 있어야 잰다. ~~BL-5~~ → BL-6 에서 잰다 |
| `not-applicable` | 우리 setting 에서 정의가 성립하지 않는다(dataset · 학습 방식 · task 가 다르다). 이유를 적었다 |

lead 초안과 달라진 점 6 개(registry 가 정정한 것):
1. Localization heads 의 RES 지표는 mIoU / oIoU 가 아니라 **cIoU**(누적 교집합 / 누적 합집합)다. 우리 `oIoU` 가 그것이다.
2. **pointing game 은 [14] 에 없다.** 원 정의는 Zhang et al. 2016 이고, 우리가 추가한 지표로 표시했다.
3. Grounded-SAM 의 SegInW AP 가 box AP 인지 mask AP 인지 논문이 밝히지 않는다.
4. Multi-link [1] 은 link 별 충돌 수를 보고하지 않는다(EE-only 대 multi-link coverage 비교뿐).
5. CFM [5] 은 제약 위반률을 따로 정의하지 않는다. 자기 제약(B ≥ 0)은 우리가 offline 으로 센다.
6. KNOWS [17] 는 CAR / TSR 대신 SR / CR / SSR 을 쓴다.

registry 에서 확인하지 못한 것(`metrics.md` §5): 논문 HTML 은 WebFetch 요약으로 읽었고 표의 숫자는 옮기지 않았다. davis2017-evaluation 의 `db_statistics` decay bin 은 pinned clone 이 없어 기억과 요약을 맞춰 구현했다(test 가 bin 경계를 고정). TokenCut 논문 HTML 의 "β = 0.3" 과 code 의 `betta_sq = 0.3`(β² = 0.3)이 달라 code 를 따랐다. Multi-link [1] 의 11 mm 와 공식 SafeLIBERO code 의 1 mm 는 서로 다른 collision 기준이라 closed-loop 비교 때 어느 쪽인지 밝혀야 한다.

![metrics registry overview](figures/baselines/metrics_overview/v1-metrics-registry-overview.png)

### 결과 B — baseline · benchmark 별 지표 표 (사용자 지시 2026-10-23)

원 지표 · status · 우리 측정값(SL3 dev 10 episode, 두 camera 합, `numbers` 의 값)을 한 표에 둔다. "closed-loop 에서 측정"은 ~~BL-5~~ → BL-6 의 몫이다. (3D soft IoU 등 harness 값은 `BASELINE-EVAL-V2.verify.json` `numbers.sl3` · `numbers.sl4b` 와 BL-1 표에서 같은 값이다.)

| 대상 | 원 지표 (출처) | status | 우리 측정값 · 대응 |
|---|---|---|---|
| AG3S (ours) — grounding | observed-surface soft IoU · Prec. · neighbour leak · paired gain (main.pdf Sec. 4.1, Tab. 2) | `offline-now` (benchmark branch 에서는 V11AB ATS 행) | harness(5 mm voxel): ATS 3D soft IoU 0.064, P 0.393, R 0.068, neighbour pooled 0.242. **같은 cloud 점 위(결과 C)**: `cloud_target` IoU 0.082 · P 0.693 (45 call) · R 0.084 · neighbour 0.301, `target_points` 좌표 일치 IoU 0.190, 5 mm IoU 0.258 |
| AG3S (ours) — task | success · safe success · hold · contact class (Tab. 1), latency (Tab. 3), SafeLIBERO TSR · CAR (Tab. 4) | `closed-loop` | closed-loop 에서 측정 |
| Grounded-SAM | SegInW zero-shot mean AP (arXiv:2401.14159 Tab. 1) | AP: `not-applicable` (SegInW 없음, referring target 하나) | phrase / instruction: mIoU 0.559 / 0.586, oIoU 0.764 / 0.769, Acc@0.5(detector box) 0.600 / 0.628, J&F 0.547 / 0.576 |
| Grounded-SAM-2 | J&F (SAM 2 Sec. 6, `sav_benchmark.py:215-303`) | `offline-now` | per_frame J&F 0.527 / 0.540 (SL3, 20 sequence). track J&F: BL-4 — SL3 0.381 / 0.474, SL4b 0.333 / 0.436 |
| YOLOE [25] | LVIS / COCO AP, AP_r/c/f, FPS (arXiv:2503.07465 Tab. 1–4) | AP: `not-applicable`. 속도: `offline-now` | phrase / instruction: any-candidate recall 0.356 / 0.092, Acc@0.5 0.310 / 0.087. latency median ms / frame: 112.1 / 160.7 (BL-1, H200, SL3); 24.7 / 23.8 (BL-4 SL4b, thread 2 — 두 값은 thread 설정이 달라 비교하지 않는다) |
| Localization heads [14] | REC Acc@0.5, RES cIoU (arXiv:2503.06287 Tab. 1–2) | `offline-now` (in-sample) | threshold / sam_point: Acc@0.5 0.048 / 0.292, cIoU(oIoU) 0.162 / 0.078, mIoU 0.137 / 0.250 |
| (추가) pointing game | 최대점이 GT 안(tol 15 px, Zhang et al. 2016) | `offline-now` | tol 0 / 15 px: LH peak 0.283 / 0.543, TokenCut seed 0.359 / 0.361, V11AB attention L8H2 peak 0.198 / 0.411 |
| TokenCut [26] | CorLoc, maxFβ · IoU · Acc (arXiv:2202.11539 Sec. 4.1, 4.3) | `offline-now` | foreground / attention_overlap: CorLoc any-object 0.386 / 0.303, CorLoc target 0.326 / 0.145, Fβ(binary) 0.319 / 0.157, mIoU 0.289 / 0.137, pixel Acc 0.874 / 0.541. 연속 map maxFβ: LH 결합 map 0.201, V11AB L8H2 0.138 |
| GT oracle | — (상한) | `offline-now` | native 지표 전부 1.000. harness `lift_within_10mm` 0.994. V1 lift check: 27,224 점 중 10 mm 이내 0.993, median 1.91 mm, p95 3.54 mm. cloud 위 IoU 0.850 · P 0.893 · R 0.948 |
| ATS V11AB | (AG3S 이전판) | `offline-now` | 결과 C |
| VLSA / AEGIS [10] | SafeLIBERO CAR · TSR · ETS (arXiv:2512.11891 Sec. V-A; 충돌 판정 `main_aegis.py:370`) | `closed-loop` | closed-loop 에서 측정. offline 대리(결과 E): `aegis` / `detector_names` / `detector_all` 개입률(행) 0.179 / 0.144 / 0.118 |
| KNOWS [17] | SR · CR · SSR (arXiv:2606.09749 Tab. 1), latency (Tab. 2) | `closed-loop` | closed-loop 에서 측정. offline 개입률(행) 0.252 |
| Multi-link [1] | collision(hazard 변위 > 11 mm) · contact · safe-success · latency · energy (arXiv:2609.40007 Tab. I–II) | `closed-loop` | closed-loop 에서 측정. offline 개입률(행) 0.160, EE-only AEGIS 가 개입하지 않은 call 에서 개입한 것 2 개(비-gripper binding 0) |
| CFM [5] | CAR · TSR · ETS (arXiv:2607.01378 Tab. 1–2) | `closed-loop`. 자기 제약 `offline-now` | 위반 행(call 평균) 9.670 → 3.213 (tracking) / 1.204 (constant), scene variant |
| NUDGE [9] | LIBERO SR · CR, sphere task Coll. Rate · Coll. Ep. (arXiv:2609.35231 Tab. II–III) | `closed-loop`. 자기 cost `offline-now` | `nudge_c_sdf` 0.0176 → 0.0090 (tracking) / 0.0043 (constant), scene variant. **κ 가 in-sample** |
| OmniGuide [19] | Success Rate · Safety Rate (arXiv:2603.10052 Sec. IV-A) | `closed-loop` | `omniguide_cost` −3.860 → −3.850 (tracking) / −3.788 (constant), scene variant |
| Lan-o3dp [15] | RLBench 성공률 (arXiv:2407.00451 Sec. IV-A, Tab. II) | 원 지표 `not-applicable`. adapted 는 `closed-loop` | 원 안 waypoint 8.520 → 7.679 (tracking) / 1.489 (constant), scene variant |
| LIBERO | success rate; FWT · NBT · AUC (arXiv:2306.03310 Sec. 5.1) | SR: `closed-loop`. FWT / NBT / AUC: `not-applicable` (frozen policy) | closed-loop 에서 측정 |
| SafeLIBERO | CAR · TSR · ETS, Level I / II | `closed-loop` | closed-loop 에서 측정 |
| LIBERO-Safety | SR · CR · Execution Time · LDLJ (arXiv:2606.23686 Sec. 4.1). Refusal Rate (SSR suite) | SR · CR · time · LDLJ: `closed-loop`. Refusal Rate: `not-applicable` | closed-loop 에서 측정 |

### 결과 C — ATS 와의 cloud labeling 공정 비교

**방법.** V11AB 가 그 call 에 만든 fused / filtered cloud(`cloud_points`)의 같은 점마다 label 을 매긴다.
- GT: scene GT instance 의 visual surface 까지 정확한 거리 ≤ 10 mm 인 최근접 instance(없으면 robot sphere 10 mm 이내면 robot, 아니면 none). SL4 · SL4b `ats_calibration.label_points` 와 같은 규칙이다.
- ATS: `cloud_target`(V11AB 가 target 으로 flag 한 cloud 점).
- segmenter: 그 점을 관측한 camera 의 mask 안에 투영되면 positive(visibility `observed`).
- 집계: call → episode 평균 → 평균(그리고 pooled). `arm − ATS` paired bootstrap 은 episode 단위 2,000 회, seed 20261015, 95 % percentile. `better` / `worse` 는 **diff 의 부호가 양 / 음인 episode 수**다(지표 방향이 아니다. neighbour 비율에서는 diff > 0 이 "ATS 보다 neighbour 비율이 높다"는 뜻).

![cloud labeling: 같은 점 위의 GT / ATS / segmenter label](figures/baselines/ats_v11ab/v1-ats-cloud-labeling-same-points.png)

**ATS 행의 새 문제 (verify `issues`).** SL4 perception run 의 `cloud_target` 이 **`has_target` 179 call 중 134 call 에서 비어 있다**(`target_points` 는 기록되어 있다). 그래서 `cloud_target` 으로 만든 ATS 행은 positive 가 있는 call 이 **45 call**(found 0.204)이고, **precision · neighbour 비율의 episode 평균은 그 45 call 만으로 계산**되며(표의 `n call`), recall · IoU 는 221 call 전체로 계산된다. 따라서 ATS `cloud_target` 행의 P 0.693 은 45 call 선택 효과가 섞여 있어 다른 arm 의 P(221 call 기준)와 같은 척도가 아니다. 이 원인(왜 비는가)은 측정하지 않았다. 그래서 verifier 가 보충 행 두 개를 같은 점 · 같은 GT class 위에 더했다.

| 보충 행 (verifier, F 의 도구가 아니다) | 정의 | positive 가 있는 call |
|---|---|---|
| `ats_v11ab_tp_exact` | `has_target` call 에서 cloud 점이 기록된 `target_points` 와 1 μm 이내(좌표 일치) | 151 / 221 |
| `ats_v11ab_tp_5mm` | 같음, 5 mm 이내 | 162 / 221 |

F 의 ATS 행과 같은 점 · 같은 class 라는 것은 221 / 221 call 의 `n_points` · `n_pred` · `n_gt` · `tp` 일치로 확인했다(`identity_check`, `all_equal` true). `ats_v11ab_seeds`(attention seed 점 기반)는 F 가 둔 참고 행이다.

![cloud labeling table](figures/baselines/ats_v11ab/v1-ats-cloud-labeling-table.png)

**표 C-1. 같은 cloud 점 위의 지표** (GT 점 수 n_gt 는 pooled 45,269)

| arm | positive 가 있는 call / 221 | IoU | P (n call) | R | neighbour 비율 (n call) | support 비율 | pooled IoU | pooled P | pooled R | pooled neighbour |
|---|---|---|---|---|---|---|---|---|---|---|
| ATS V11AB (`cloud_target`) | 45 / 221 | 0.082 | 0.693 (45) | 0.084 | 0.301 (45) | 0.000 | 0.082 | 0.534 | 0.088 | 0.462 |
| ATS `target_points` (좌표 일치) | 151 / 221 | 0.190 | 0.636 (151) | 0.192 | 0.352 (151) | 0.000 | 0.174 | 0.546 | 0.204 | 0.435 |
| ATS `target_points` (5 mm) | 162 / 221 | 0.258 | 0.649 (162) | 0.261 | 0.338 (162) | 0.000 | 0.230 | 0.594 | 0.272 | 0.388 |
| ATS attention seed (참고 행) | 210 / 221 | 0.207 | 0.569 (210) | 0.232 | 0.379 (210) | 0.000 | 0.205 | 0.567 | 0.243 | 0.305 |
| Grounded-SAM (phrase) | 221 / 221 | 0.459 | 0.528 (221) | 0.660 | 0.413 (221) | 0.046 | 0.404 | 0.503 | 0.673 | 0.439 |
| Grounded-SAM (instruction) | 221 / 221 | 0.493 | 0.554 (221) | 0.684 | 0.384 (221) | 0.045 | 0.423 | 0.519 | 0.694 | 0.404 |
| Grounded-SAM-2 per_frame (phrase) | 220 / 221 | 0.451 | 0.545 (220) | 0.642 | 0.382 (220) | 0.061 | 0.403 | 0.511 | 0.657 | 0.419 |
| Grounded-SAM-2 per_frame (instruction) | 219 / 221 | 0.486 | 0.593 (219) | 0.655 | 0.331 (219) | 0.062 | 0.432 | 0.550 | 0.668 | 0.378 |
| YOLOE-11L (phrase) | 188 / 221 | 0.285 | 0.481 (188) | 0.406 | 0.444 (188) | 0.053 | 0.283 | 0.450 | 0.432 | 0.470 |
| YOLOE-11L (instruction) | 98 / 221 | 0.101 | 0.370 (98) | 0.128 | 0.342 (98) | 0.054 | 0.070 | 0.120 | 0.143 | 0.095 |
| localization_heads:threshold (in-sample) | 221 / 221 | 0.104 | 0.105 (221) | 0.865 | 0.118 (221) | 0.630 | 0.099 | 0.101 | 0.865 | 0.118 |
| localization_heads:sam_point (in-sample) | 221 / 221 | 0.266 | 0.386 (221) | 0.400 | 0.208 (221) | 0.332 | 0.081 | 0.092 | 0.405 | 0.053 |
| TokenCut foreground | 221 / 221 | 0.212 | 0.220 (221) | 0.737 | 0.195 (221) | 0.322 | 0.189 | 0.203 | 0.733 | 0.200 |
| TokenCut attention_overlap | 221 / 221 | 0.075 | 0.078 (221) | 0.543 | 0.166 (221) | 0.594 | 0.034 | 0.035 | 0.560 | 0.082 |
| GT oracle (두 prompt 같음) | 221 / 221 | 0.850 | 0.893 (221) | 0.948 | 0.018 (221) | 0.073 | 0.850 | 0.891 | 0.948 | 0.017 |

- `support 비율` = 예측 positive 점 중 GT class 가 support(table 면)인 비율의 episode 평균. `robot` · `none` 비율은 ATS · segmenter 모두 0.000–0.001 이었다(`tokencut:attention_overlap` 의 `none` 0.001).

**표 C-2. arm − ATS 의 Δ IoU (ATS 정의 3 가지)** — 95 % CI, (better / worse)

| arm − ATS | Δ IoU, ATS = `cloud_target` [95 % CI] (better / worse) | Δ IoU, ATS = `target_points` 좌표 일치 | Δ IoU, ATS = `target_points` 5 mm |
|---|---|---|---|
| ATS attention seed (참고 행) | +0.125 [0.083, 0.163] (9 / 1) | +0.017 [-0.050, 0.093] (4 / 6) | -0.051 [-0.145, 0.055] (3 / 7) |
| Grounded-SAM (phrase) | +0.377 [0.247, 0.504] (10 / 0) | +0.269 [0.144, 0.411] (9 / 1) | +0.201 [0.068, 0.362] (8 / 2) |
| Grounded-SAM (instruction) | +0.411 [0.282, 0.533] (10 / 0) | +0.303 [0.174, 0.427] (9 / 1) | +0.236 [0.098, 0.371] (8 / 2) |
| Grounded-SAM-2 per_frame (phrase) | +0.369 [0.253, 0.486] (10 / 0) | +0.261 [0.143, 0.393] (9 / 1) | +0.193 [0.066, 0.341] (7 / 3) |
| Grounded-SAM-2 per_frame (instruction) | +0.404 [0.281, 0.519] (10 / 0) | +0.296 [0.168, 0.421] (9 / 1) | +0.228 [0.091, 0.369] (7 / 3) |
| YOLOE-11L (phrase) | +0.203 [0.121, 0.276] (9 / 1) | +0.095 [0.002, 0.191] (7 / 3) | +0.027 [-0.092, 0.150] (5 / 5) |
| YOLOE-11L (instruction) | +0.019 [-0.065, 0.110] (6 / 4) | -0.089 [-0.181, 0.001] (4 / 6) | -0.157 [-0.262, -0.051] (3 / 7) |
| localization_heads:threshold (in-sample) | +0.022 [-0.027, 0.066] (7 / 3) | -0.086 [-0.169, -0.002] (3 / 7) | -0.154 [-0.262, -0.042] (3 / 7) |
| localization_heads:sam_point (in-sample) | +0.183 [0.150, 0.222] (10 / 0) | +0.076 [0.004, 0.150] (6 / 4) | +0.008 [-0.094, 0.115] (5 / 5) |
| TokenCut foreground | +0.130 [0.058, 0.193] (8 / 2) | +0.022 [-0.086, 0.122] (6 / 4) | -0.046 [-0.178, 0.085] (4 / 6) |
| TokenCut attention_overlap | -0.007 [-0.046, 0.028] (6 / 4) | -0.115 [-0.187, -0.043] (3 / 7) | -0.183 [-0.280, -0.082] (2 / 8) |
| GT oracle (두 prompt 같음) | +0.768 [0.719, 0.813] (10 / 0) | +0.660 [0.576, 0.744] (10 / 0) | +0.593 [0.485, 0.704] (10 / 0) |

**표 C-3. arm − ATS(`cloud_target`) 의 Δ precision · recall · neighbour 비율**

| arm − ATS (`cloud_target`) | Δ precision | Δ recall | Δ neighbour 비율 |
|---|---|---|---|
| Grounded-SAM (phrase) | -0.165 [-0.392, 0.109] (2 / 8) | +0.577 [0.451, 0.690] (10 / 0) | +0.111 [-0.157, 0.336] (8 / 2) |
| Grounded-SAM (instruction) | -0.139 [-0.363, 0.109] (2 / 8) | +0.600 [0.483, 0.698] (10 / 0) | +0.083 [-0.158, 0.299] (8 / 2) |
| Grounded-SAM-2 per_frame (phrase) | -0.149 [-0.379, 0.124] (2 / 8) | +0.559 [0.442, 0.658] (10 / 0) | +0.080 [-0.190, 0.307] (8 / 2) |
| Grounded-SAM-2 per_frame (instruction) | -0.100 [-0.344, 0.164] (2 / 8) | +0.571 [0.460, 0.667] (10 / 0) | +0.030 [-0.235, 0.270] (7 / 3) |
| YOLOE-11L (phrase) | -0.212 [-0.471, 0.078] (3 / 7) | +0.322 [0.228, 0.393] (9 / 1) | +0.143 [-0.145, 0.406] (7 / 3) |
| YOLOE-11L (instruction) | -0.323 [-0.547, -0.108] (3 / 7) | +0.044 [-0.047, 0.143] (6 / 4) | +0.041 [-0.350, 0.401] (7 / 3) |
| localization_heads:threshold (in-sample) | -0.588 [-0.831, -0.320] (2 / 8) | +0.782 [0.731, 0.833] (10 / 0) | -0.184 [-0.451, 0.057] (5 / 5) |
| localization_heads:sam_point (in-sample) | -0.308 [-0.526, -0.065] (3 / 7) | +0.316 [0.284, 0.356] (10 / 0) | -0.094 [-0.312, 0.090] (6 / 4) |
| TokenCut foreground | -0.473 [-0.726, -0.192] (2 / 8) | +0.653 [0.536, 0.753] (10 / 0) | -0.107 [-0.376, 0.124] (7 / 3) |
| TokenCut attention_overlap | -0.615 [-0.841, -0.360] (2 / 8) | +0.460 [0.364, 0.553] (10 / 0) | -0.135 [-0.424, 0.115] (6 / 4) |
| GT oracle (두 prompt 같음) | +0.200 [-0.038, 0.465] (5 / 5) | +0.864 [0.818, 0.906] (10 / 0) | -0.283 [-0.541, -0.046] (5 / 5) |

![paired bootstrap](figures/baselines/ats_v11ab/v1-ats-cloud-labeling-paired-bootstrap.png)

![target_points 보충](figures/baselines/ats_v11ab/v1-ats-cloud-labeling-target-points-supplement.png)

관찰(수치에서 읽히는 것만):
- Grounded-SAM · Grounded-SAM-2 per_frame 4 arm 은 ATS 를 어느 정의(`cloud_target` · `target_points` 좌표 일치 · 5 mm)로 잡아도 Δ IoU 의 95 % CI 가 0 위다. 같은 점 · 같은 GT 위에서도 이 방향이 유지된다.
- ATS 정의에 따라 ATS 자체의 IoU 가 0.082 → 0.190 → 0.258 로 움직인다. `cloud_target` 이 비는 call 이 많다는 것이 ATS 행의 수치에 영향을 준다는 뜻이다. BL-1 harness 의 recall 0.068(5 mm voxel)과 이번 recall 0.084(`cloud_target`) · 0.192 · 0.261 은 평가 단위가 달라(voxel 대 cloud 점) 같은 값이 아니다. 표현 차이(대표점 대 cloud 점)와 `cloud_target` 이 비는 것 중 어느 쪽이 얼마나 기여했는지는 이 측정이 분리하지 않았다.
- precision 은 Δ P 의 CI 가 0 을 포함하는 arm 이 다수이고, better / worse 는 ATS 쪽이 높은 episode 가 더 많다(예: Grounded-SAM phrase 2 / 8). 단 ATS 의 P 는 45 call 기준이다.
- neighbour 비율의 Δ 는 `gt_oracle` 을 빼고 모든 arm 에서 CI 가 0 을 포함한다.
- `localization_heads:threshold` · `tokencut:attention_overlap` 은 recall 이 높지만(0.865 / 0.543) precision 이 0.105 / 0.078 이고 예측 positive 중 support 비율이 0.630 / 0.594 다.
- `gt_oracle` 은 같은 점 위에서 IoU 0.850 · P 0.893 · R 0.948(두 prompt 같음). projection · 경계 때문에 1 이 아니다.

### 결과 D — grounding 고유 지표 (`native_metrics`)

**sample 정의.** (frame, camera) 중 GT target mask 가 비지 않은 것. 못 찾은 arm(빈 mask)은 miss / IoU 0 으로 남는다. GT box 는 GT segmentation 에서 보이는 부분의 tight box 다(amodal box 는 기록에 없다). 집계는 sample 전체 pooled 비율 + episode resample 2,000 회 95 % CI(seed 20261015). 두 camera 합 435 sample(head 221, left_wrist 214). DAVIS 블록(J / F)만 sequence = (episode, camera) 20 개의 평균이고 GT 가 빈 frame 도 포함한다. sequence 의 frame 은 5 step 간격의 정책 call frame 이라 원 benchmark 의 연속 frame 이 아니다.

**표 D-1. box · mask 수준** (Acc@0.5 는 detector box 와 mask box 가 같은 값이다. CorLoc target = Acc@0.5 mask)

| arm | sample 수 | Acc@0.5 (mask box) [95 % CI] | CorLoc any-object [95 % CI] | mIoU | oIoU (= cIoU) | pixel Acc | Fβ (binary mask) |
|---|---|---|---|---|---|---|---|
| GT oracle (두 prompt 같음) | 435 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
| Grounded-SAM (phrase) | 435 | 0.600 [0.478, 0.724] | 0.979 [0.950, 0.998] | 0.559 | 0.764 | 0.982 | 0.576 |
| Grounded-SAM (instruction) | 435 | 0.628 [0.505, 0.743] | 0.986 [0.971, 0.998] | 0.586 | 0.769 | 0.982 | 0.603 |
| Grounded-SAM-2 per_frame (phrase) | 435 | 0.568 [0.460, 0.673] | 0.910 [0.855, 0.958] | 0.538 | 0.776 | 0.983 | 0.548 |
| Grounded-SAM-2 per_frame (instruction) | 435 | 0.582 [0.474, 0.679] | 0.869 [0.790, 0.939] | 0.552 | 0.781 | 0.983 | 0.561 |
| YOLOE-11L (phrase) | 435 | 0.310 [0.240, 0.365] | 0.602 [0.507, 0.711] | 0.287 | 0.356 | 0.952 | 0.293 |
| YOLOE-11L (instruction) | 435 | 0.087 [0.022, 0.155] | 0.149 [0.058, 0.255] | 0.082 | 0.112 | 0.926 | 0.084 |
| localization_heads:threshold (in-sample) | 435 | 0.048 [0.014, 0.086] | 0.062 [0.026, 0.102] | 0.137 | 0.162 | 0.778 | 0.179 |
| localization_heads:sam_point (in-sample) | 435 | 0.292 [0.252, 0.336] | 0.476 [0.438, 0.521] | 0.250 | 0.078 | 0.832 | 0.272 |
| TokenCut foreground | 435 | 0.326 [0.258, 0.390] | 0.386 [0.342, 0.426] | 0.289 | 0.316 | 0.874 | 0.319 |
| TokenCut attention_overlap | 435 | 0.145 [0.115, 0.176] | 0.303 [0.230, 0.384] | 0.137 | 0.051 | 0.541 | 0.157 |

**표 D-2. camera 별**

| arm | Acc@0.5 head | Acc@0.5 left_wrist | CorLoc any head | CorLoc any left_wrist | mIoU head | mIoU left_wrist |
|---|---|---|---|---|---|---|
| GT oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Grounded-SAM (phrase) | 0.308 | 0.902 | 0.964 | 0.995 | 0.282 | 0.844 |
| Grounded-SAM (instruction) | 0.385 | 0.879 | 0.982 | 0.991 | 0.352 | 0.827 |
| Grounded-SAM-2 per_frame (phrase) | 0.267 | 0.879 | 0.860 | 0.963 | 0.243 | 0.843 |
| Grounded-SAM-2 per_frame (instruction) | 0.312 | 0.860 | 0.805 | 0.935 | 0.288 | 0.824 |
| YOLOE-11L (phrase) | 0.109 | 0.519 | 0.638 | 0.565 | 0.096 | 0.485 |
| YOLOE-11L (instruction) | 0.000 | 0.178 | 0.095 | 0.206 | 0.002 | 0.165 |
| localization_heads:threshold (in-sample) | 0.000 | 0.098 | 0.000 | 0.126 | 0.042 | 0.235 |
| localization_heads:sam_point (in-sample) | 0.262 | 0.322 | 0.271 | 0.687 | 0.218 | 0.283 |
| TokenCut foreground | 0.000 | 0.664 | 0.005 | 0.780 | 0.033 | 0.553 |
| TokenCut attention_overlap | 0.018 | 0.276 | 0.258 | 0.350 | 0.038 | 0.239 |

**표 D-3. pointing game · DAVIS 형 J / F** (pointing: LH = 결합 map peak, TokenCut = seed patch 중심, detector = box 중심(원 정의의 변형이라 표시), J&F 의 `temporal IoU` 는 연속한 평가 frame 사이 mask IoU 로 논문 지표가 아니다)

| arm | pointing game tol 0 / 15 px | J mean | F mean | J&F | J recall | J decay | temporal IoU (예측) | temporal IoU 차이 (예측 대 GT, 절댓값) |
|---|---|---|---|---|---|---|---|---|
| GT oracle | - | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.662 | 0.000 |
| Grounded-SAM (phrase) | 0.595 / 0.605 (box 중심, 변형) | 0.548 | 0.547 | 0.547 | 0.582 | -0.098 | 0.699 | 0.201 |
| Grounded-SAM (instruction) | 0.618 / 0.632 (box 중심, 변형) | 0.576 | 0.576 | 0.576 | 0.613 | -0.014 | 0.694 | 0.213 |
| Grounded-SAM-2 per_frame (phrase) | 0.563 / 0.570 (box 중심, 변형) | 0.528 | 0.527 | 0.527 | 0.554 | -0.073 | 0.712 | 0.210 |
| Grounded-SAM-2 per_frame (instruction) | 0.572 / 0.582 (box 중심, 변형) | 0.541 | 0.540 | 0.540 | 0.568 | 0.000 | 0.716 | 0.214 |
| YOLOE-11L (phrase) | 0.308 / 0.322 (box 중심, 변형) | 0.287 | 0.287 | 0.287 | 0.310 | 0.066 | 0.740 | 0.302 |
| YOLOE-11L (instruction) | 0.113 / 0.120 (box 중심, 변형) | 0.092 | 0.110 | 0.101 | 0.097 | -0.027 | 0.889 | 0.332 |
| localization_heads:threshold (in-sample) | 0.283 / 0.543 | 0.133 | 0.029 | 0.081 | 0.002 | -0.039 | 0.742 | 0.294 |
| localization_heads:sam_point (in-sample) | 0.283 / 0.543 | 0.250 | 0.354 | 0.302 | 0.275 | 0.041 | 0.302 | 0.421 |
| TokenCut foreground | 0.359 / 0.361 | 0.280 | 0.151 | 0.215 | 0.317 | -0.201 | 0.690 | 0.247 |
| TokenCut attention_overlap | 0.359 / 0.361 | 0.136 | 0.157 | 0.147 | 0.140 | 0.020 | 0.589 | 0.373 |
| LH 결합 attention map (연속) | 0.283 / 0.543 | - | - | - | - | - | - | - |
| V11AB attention L8H2 (연속) | 0.198 / 0.411 | - | - | - | - | - | - | - |

**표 D-4. 연속 attention map** (provider 의 mask 가 아니라 attention audit 기록에서 다시 만든 map. min-max 정규화 후 maxFβ)

| 연속 map | maxFβ (all) | maxFβ 의 threshold | maxFβ head | maxFβ left_wrist | mIoU (0.5 binarize) | oIoU | pixel Acc |
|---|---|---|---|---|---|---|---|
| LH 결합 attention map (연속) | 0.201 | 0.741 | 0.272 | 0.298 | 0.119 | 0.085 | 0.900 |
| V11AB attention L8H2 (연속) | 0.138 | 0.118 | 0.156 | 0.185 | 0.053 | 0.016 | 0.926 |

**표 D-5. detector `conf` sweep** (저장된 후보는 provider 자신의 `conf` threshold(0.25 등) 이상뿐이라 곡선이 그 threshold 에서 시작한다. `conf` 0 부터의 곡선은 없다)

| arm | any-candidate recall | conf 0.25: P / R (n pred) | conf 0.5 | conf 0.7 |
|---|---|---|---|---|
| Grounded-SAM (phrase) | 0.671 | 0.607 / 0.600 (430) | 0.631 / 0.503 (347) | 0.675 / 0.315 (203) |
| Grounded-SAM (instruction) | 0.676 | 0.630 / 0.628 (433) | 0.708 / 0.384 (236) | 1.000 / 0.016 (7) |
| Grounded-SAM-2 per_frame (phrase) | 0.579 | 0.622 / 0.568 (397) | 0.631 / 0.503 (347) | 0.675 / 0.315 (203) |
| Grounded-SAM-2 per_frame (instruction) | 0.582 | 0.669 / 0.582 (378) | 0.708 / 0.384 (236) | 1.000 / 0.016 (7) |
| YOLOE-11L (phrase) | 0.356 | 0.515 / 0.310 (262) | 0.618 / 0.145 (102) | 0.595 / 0.057 (42) |
| YOLOE-11L (instruction) | 0.092 | 0.384 / 0.087 (99) | 0.321 / 0.021 (28) | 0.000 / 0.000 (1) |

Grounded-SAM 의 phrase 와 instruction 이 Acc@0.5(mask box)를 맞힌 sample 의 짝 집계(435 sample): 둘 다 246, phrase 만 15, instruction 만 27, 둘 다 틀림 147.

![Grounded-SAM prompt 별 box 선택](figures/baselines/grounded_sam/v1-gsam-prompt-selection.png)

![Localization heads: 선택된 head 의 attention map 과 선택 기준](figures/baselines/localization_heads/v1-lh-heads-selection.png)

![TokenCut: affinity graph 의 2nd eigenvector · bipartition · seed](figures/baselines/tokencut/v1-tokencut-ncut.png)

관찰:
- harness `frames.jsonl` 의 `2d_iou` 와 `native_metrics` 의 IoU 는 5,220 sample 에서 최대 절대 차이 0.0 이다(`check_vs_harness_2d_iou`).
- Grounded-SAM · Grounded-SAM-2 per_frame 은 camera 에 따라 크게 갈린다: Acc@0.5 가 left_wrist 에서 0.86–0.90, head 에서 0.27–0.39. CorLoc any-object 는 head 에서도 0.80–0.98 이다(다른 물체의 box 는 맞힌다).
- TokenCut 의 pointing(seed)은 head 에서 0.000 / 0.005, left_wrist 에서 0.729 다(두 TokenCut arm 이 같은 seed 를 쓴다).
- localization heads 의 pointing 은 head 0.606 · left_wrist 0.477 (tol 15 px, 두 mask 모드가 같은 peak 를 쓴다)이다. **in-sample** 이다.
- TokenCut 재계산 parity: DINO ViT-S/16 을 CPU 로 다시 돌려 저장 mask 와 비교한 40 sample 중 39 가 같고 **1 sample 이 달랐다**(`libero_spatial_t3_e0` obs 65 head, `same_as_saved` false). verify.json `issues` 는 이 불일치를 언급하지 않았다. 원인은 측정하지 않았다(verifier 에게 확인을 요청할 항목).

### 결과 E — methods offline 지표 (open loop, SL3 dev 221 call · 2,210 행)

**방법.** 기록된 π0.5 chunk 를 method 마다 call 단위로 돌려 수정 chunk 를 얻는다(상태가 있는 method 는 episode 마다 reset). 수정 chunk 는 **실행하지 않는다**(open loop). filter 는 recorded chunk 를 고치되, chunk 안의 뒤 action 은 method 자기 EE 예측(측정된 gain 0.0115 m / unit) 위에서 filter 한다(공식 per-step 호출이 아니다). guidance 는 π0.5 network 가 아니라 D 의 대체 velocity(기록 chunk 로 가는 직선 flow)를 쓴다. `tracking` 은 이후 denoising step 이 수정을 되돌리는 쪽, `constant` 는 수정이 끝까지 남는 쪽이다. 장애물 variant 는 `scene`(depth lift 에서 privileged target · robot · support · destination mask 를 뺀 점)과 `injected`(기록 경로 옆 1 cm, 반경 3 cm 의 privileged 구를 row 6 에 놓은 D 의 합성 repeller 시험)다. filter 는 5 arm(`aegis` = GT segmentation 장애물, `aegis:detector_names` = Grounding DINO tiny + privileged LIBERO 장애물 이름(VLM 대체), `aegis:detector_all` = Grounding DINO 일반 prompt − target box, `knows`, `multilink_cbf`).

**scene-GT clearance (모든 method 공통 기준).** chunk → OSC response 모델 EE 경로(자세 고정, action ∈ [−1, 1]) → 기록 q 에서 DLS IK → cuRobo `franka.yml` sphere 61 개 → scene GT 표면까지의 정확한 거리(안쪽은 음수) − 반지름, **0.05 m 에서 자른다(cap)**. "obstacle" 군은 target · destination · 받침대(`main_table` 등)를 뺀 instance 다. SL3 dev 는 장애물을 일부러 놓은 장면이 아니므로 obstacle 군은 같은 장면의 다른 물체다. `h` 는 method 마다 정의가 달라(KNOWS = EE ellipsoid–물체 MVEE, AEGIS = 장애물별 최소, multi-link = (body, 장애물) 최소, CFM = 이산 CBF 행) **method 사이에서 비교하지 않는다**.

![methods offline overview](figures/baselines/methods_overview/v1-methods-offline-overview.png)

**표 E-1. filter 의 자기 지표**

| arm | 개입률 (행) | 개입률 (call) | ‖Δa_xyz‖ 평균 | ‖Δa_xyz‖ 최대 | h_min (자기 barrier) | h < 0 step | QP 실패 step | median ms / call (CPU 1 thread) |
|---|---|---|---|---|---|---|---|---|
| aegis | 0.179 | 0.262 | 0.066 | 1.604 | -0.1223 | 237 | 0 | 4.7 |
| aegis:detector_all | 0.118 | 0.163 | 0.055 | 1.701 | -0.1401 | 193 | 0 | 1.5 |
| aegis:detector_names | 0.144 | 0.199 | 0.070 | 1.709 | -0.1411 | 232 | 0 | 3.1 |
| knows | 0.252 | 0.398 | 0.109 | 1.470 | -0.1745 | 1246 | 0 | 13.3 |
| multilink_cbf | 0.160 | 0.258 | 0.053 | 1.558 | -0.1207 | 178 | 0 | 116.4 |

**표 E-2. filter 의 scene-GT clearance** (mm, nominal → 수정 chunk. EE = hand + finger sphere, arm = link 1–7 sphere. IK 위치 오차 최대 `ik_pos_err_max_m` 는 nominal · 수정 모두 0.0658 m, 5 filter 같은 값)

| arm | EE sphere – obstacle 최소 clearance (mm), nominal → 수정 | EE 음수 행 (2,210 행 중) | arm sphere – obstacle 최소 clearance (mm) | arm 음수 행 |
|---|---|---|---|---|
| aegis | -24.0 → -15.7 | 82 → 24 | 7.9 → 46.1 | 0 → 0 |
| aegis:detector_all | -24.0 → -14.4 | 82 → 45 | 7.9 → 48.1 | 0 → 0 |
| aegis:detector_names | -24.0 → -15.0 | 82 → 33 | 7.9 → 49.8 | 0 → 0 |
| knows | -24.0 → -23.2 | 82 → 59 | 7.9 → 5.2 | 0 → 0 |
| multilink_cbf | -24.0 → -17.6 | 82 → 32 | 7.9 → 46.7 | 0 → 0 |

**표 E-3. guidance (`scene` variant)** — 자기 cost / 제약은 call 평균의 전 → 후

| method | surrogate | 개입률 (행) | ‖Δa_xyz‖ 평균 | ‖Δa_xyz‖ 최대 | 자기 cost / 제약, 전 → 후 (call 평균) | GT EE 음수 행 | GT arm 음수 행 | 수정 chunk 의 IK 위치 오차 최대 (m, nominal 은 0.066) | median ms (jit 포함) |
|---|---|---|---|---|---|---|---|---|---|
| omniguide | tracking | 0.240 | 0.006 | 0.13 | omniguide_cost: -3.8598 → -3.8497 | 82 → 82 | 0 → 0 (최소 7.9 → 7.9 mm) | 0.066 | 65.0 |
| omniguide | constant | 0.240 | 0.050 | 0.90 | omniguide_cost: -3.8598 → -3.7877 | 82 → 82 | 0 → 0 (최소 7.9 → 7.9 mm) | 0.066 | 65.0 |
| constrained_flow_matching | tracking | 0.595 | 1.083 | 14.86 | cfm_min_g: -0.0958 → -0.0215 ; cfm_violated_rows: 9.6697 → 3.2127 | 82 → 43 | 0 → 0 (최소 7.9 → 24.1 mm) | 0.286 | 4151.0 |
| constrained_flow_matching | constant | 0.814 | 1.434 | 18.82 | cfm_min_g: -0.0958 → -0.0030 ; cfm_violated_rows: 9.6697 → 1.2036 | 82 → 58 | 0 → 3 (최소 7.9 → -7.1 mm) | 1.281 | 4151.0 |
| lan_o3dp_guidance_adapted | tracking | 0.975 | 0.427 | 2.74 | lan_rows_inside_circle: 8.5204 → 7.6787 | 82 → 80 | 0 → 0 (최소 7.9 → 8.8 mm) | 0.033 | 150.9 |
| lan_o3dp_guidance_adapted | constant | 0.977 | 6.142 | 46.45 | lan_rows_inside_circle: 8.5204 → 1.4887 | 82 → 71 | 0 → 0 (최소 7.9 → 13.3 mm) | 0.088 | 150.9 |
| nudge | tracking | 0.879 | 0.804 | 2.64 | nudge_c_sdf: 0.0176 → 0.0090 | 82 → 27 | 0 → 4 (최소 7.9 → -37.7 mm) | 1.579 | 673.7 |
| nudge | constant | 0.881 | 0.442 | 2.56 | nudge_c_sdf: 0.0176 → 0.0043 | 82 → 34 | 0 → 0 (최소 7.9 → 25.2 mm) | 0.561 | 673.7 |

**표 E-4. guidance (`injected` variant)**

| method | surrogate | 개입률 (행) | ‖Δa_xyz‖ 평균 | ‖Δa_xyz‖ 최대 | 자기 cost / 제약, 전 → 후 (call 평균) | GT EE 음수 행 | GT arm 음수 행 | 수정 chunk 의 IK 위치 오차 최대 (m, nominal 은 0.066) | median ms (jit 포함) |
|---|---|---|---|---|---|---|---|---|---|
| omniguide | tracking | 0.240 | 0.019 | 0.36 | omniguide_cost: -10.0017 → -9.9399 | 82 → 82 | 0 → 0 (최소 7.9 → 7.9 mm) | 0.066 | 61.8 |
| omniguide | constant | 0.240 | 0.137 | 1.60 | omniguide_cost: -10.0017 → -8.8884 | 82 → 82 | 0 → 0 (최소 7.9 → 7.9 mm) | 0.066 | 61.8 |
| constrained_flow_matching | tracking | 0.845 | 1.100 | 11.54 | cfm_min_g: -0.0963 → 0.0003 ; cfm_violated_rows: 10.0000 → 0.0181 | 82 → 60 | 0 → 1 (최소 7.9 → -0.6 mm) | 0.647 | 3798.8 |
| constrained_flow_matching | constant | 0.891 | 1.008 | 10.72 | cfm_min_g: -0.0963 → 0.0002 ; cfm_violated_rows: 10.0000 → 1.0226 | 82 → 52 | 0 → 0 (최소 7.9 → 29.0 mm) | 1.367 | 3798.8 |
| lan_o3dp_guidance_adapted | tracking | 1.000 | 0.575 | 1.99 | lan_rows_inside_circle: 9.0995 → 8.4253 | 82 → 78 | 0 → 0 (최소 7.9 → 7.4 mm) | 0.026 | 142.1 |
| lan_o3dp_guidance_adapted | constant | 1.000 | 5.747 | 19.86 | lan_rows_inside_circle: 9.0995 → 0.0000 | 82 → 84 | 0 → 2 (최소 7.9 → -6.7 mm) | 0.094 | 142.1 |
| nudge | tracking | 1.000 | 1.369 | 2.50 | nudge_c_sdf: 0.0316 → 0.0055 | 82 → 19 | 0 → 2 (최소 7.9 → -48.3 mm) | 1.686 | 502.9 |
| nudge | constant | 1.000 | 0.667 | 2.22 | nudge_c_sdf: 0.0316 → 0.0044 | 82 → 32 | 0 → 0 (최소 7.9 → 3.0 mm) | 1.212 | 502.9 |

관찰:
- 모든 filter 에서 QP 실패 step 0, gripper action 불변, 비상 정지 0, 모든 guidance 출력이 유한(`finite_all`)이다. trace 판 sampler 는 D 의 `surrogate_sample` 과 16 / 16(arm × variant × surrogate)에서 bit 동일하다(`max_abs` 0.0).
- nominal(기록 chunk)의 EE sphere 가 obstacle 군 안에 들어간 행이 2,210 행 중 82 행(13 call)이다. 이 값은 filter · guidance 전 기준선이다(SL3 dev 는 장애물 장면이 아니다).
- AEGIS 세 변형 사이에서 개입률(행)이 0.179 → 0.144 → 0.118 로 다르다(GT segmentation, 이름 + detector, 일반 prompt + detector 순). EE 음수 행은 82 → 24 / 33 / 45 이다.
- KNOWS 는 EE 음수 행이 82 → 59 이고 arm sphere 의 최소 clearance 가 7.9 → 5.2 mm 로 **줄었다**(AEGIS 변형 · multi-link 는 46–50 mm 로 늘었다). arm 음수 행은 모든 filter 에서 0 이다. `h < 0` step 은 KNOWS 1,246 으로 가장 많지만 h 정의가 달라 비교하지 않는다.
- multi-link 가 개입했고 EE-only AEGIS(같은 GT segmentation 장애물)는 개입하지 않은 call 이 **2 개**이고 두 call 모두 binding body 가 gripper 다(비-gripper 0, `libero_spatial_t1_e0` obs 65, `libero_spatial_t6_e0` obs 35). 즉 multi-link 를 따로 두는 이유인 forearm 사례는 SL3 dev 에 없다.
- guidance 가 **arm sphere clearance 를 음수로 만든 경우**가 있다: NUDGE tracking(scene 4 행, 최소 −37.7 mm / injected 2 행, 최소 −48.3 mm), CFM constant(scene 3 행, −7.1 mm), CFM tracking(injected 1 행, −0.6 mm), Lan-o3dp constant(injected 2 행, −6.7 mm). 같은 기록 chunk 의 nominal 은 arm 음수 행 0 이다. 인과(왜 그런지)는 측정하지 않았다. **단 신뢰도 주의:** 수정 chunk 를 decode 한 경로에서 IK 위치 오차 최대(`ik_pos_err_max_m`)가 CFM 0.286–1.367 m, NUDGE 0.561–1.686 m 로 nominal(0.0658 m)보다 훨씬 크다(표 E-3 · E-4 마지막 열 앞 열). IK 가 경로를 따르지 못한 경로 위의 clearance 일 수 있다. verify.json `issues` 는 이것을 언급하지 않았다. OmniGuide 는 nominal 과 같고(0.0658 m) Lan-o3dp 는 0.026–0.095 m 다.
- OmniGuide 는 2,210 행 중 개입 행이 0.240 이다(53 call 에서만 개입, 168 call 은 `frozen`). EE 음수 행 82 → 82 로 변하지 않았다.
- Lan-o3dp 의 `constant` surrogate 에서 ‖Δa_xyz‖ 평균이 5.7–6.1, 최대 19.9–46.4 로 action 범위(−1~1, decode 에서 clip)를 크게 넘는다.
- `ms` 열은 call 마다 closure 를 새로 만드는 jit compile 을 포함한다(실제 sampling 시간이 아니다). `tracking` · `constant` 두 surrogate 는 한 jit 호출에서 같이 계산되어 같은 값이다. detector 두 arm 의 첫 call 시간에는 Grounding DINO tiny CPU 탐지가 포함된다.
- 이 CPU 측정(03:58 KST 이후)과 V2 의 GPU 측정(SL3 track 03:50–04:00, SL4b 04:00–05:01 KST)이 시간상 겹쳤다. 같은 pod 의 CPU 부하가 위 `ms` 에 영향을 주었는지는 확인하지 않았다.

figure(module 별 3 종은 아래 목록): [KNOWS h(t)](figures/baselines/knows/v1-knows-h-path.png) · [multi-link link 별 h(t)](figures/baselines/multilink_cbf/v1-multilink-link-h.png) · [AEGIS 변형 간 차이](figures/baselines/aegis/v1-aegis-variant-differences.png) · [OmniGuide field slice](figures/baselines/omniguide/v1-omniguide-field-slices-steps.png)

![multi-link: link 별 h(t), 개입 call](figures/baselines/multilink_cbf/v1-multilink-link-h.png)

### 결과 F — benchmark 대표 장면과 지표 정의

LIBERO 4 suite, SafeLIBERO 4 suite × Level I / II, LIBERO-Safety(L0 의 4 task 유형)의 대표 장면을 기존 run 기록 · env render 로 만들고 geom-id segmentation 으로 target(초록)과 obstacle / hazard(빨강)를 표시했다. LIBERO 에는 obstacle 이 없다. SafeLIBERO 의 obstacle 은 active obstacle(README 규칙: 작업공간 안), LIBERO-Safety 의 obstacle 은 BDDL constraint 의 target 이외 인자다(sidecar `note`). suite 별 task 수 · init state 수 · constraint predicate 는 [`v1-benchmarks-protocol.png`](figures/baselines/benchmarks/v1-benchmarks-protocol.png)(sidecar 값이며 verify.json `numbers` 에는 없어 이 파일에 옮기지 않았다)에, 지표 정의 표는 [`v1-benchmarks-metric-definitions.png`](figures/baselines/benchmarks/v1-benchmarks-metric-definitions.png) 에 있다. 지표는 모두 `closed-loop`(또는 `not-applicable`)이라 이번 STEP 에서 측정값이 없다(결과 B).

![benchmark 대표 장면](figures/baselines/benchmarks/v1-benchmarks-scenes.png)

### 검증 장치가 확인한 것

| 장치 | 결과 |
|---|---|
| `check_vs_harness_2d_iou` (native IoU 대 harness `2d_iou`) | 5,220 sample, 최대 절대 차이 0.0 |
| `tokencut_dino_parity` (DINO CPU 재계산 mask 대 저장 mask) | 40 sample 중 39 일치, 1 불일치 (결과 D) |
| trace 판 guidance sampler 대 `surrogate_sample` | 16 / 16 bit 동일 |
| 보충 ATS 행의 점 · class 가 F 의 ATS 행과 같은가 | 221 / 221 call 일치 |
| 회귀 기준선 (legacy 16D) | 일치 |
| F 의 unit test | `.venv-baselines` 32 passed 1 skipped, `.venv-openpi-live` 11 passed |

### module 별 figure (규칙 A: 실제 장면 · 핵심 기능 · 지표)

경로는 모두 `figures/baselines/<module>/` 이고 같은 이름의 `.json` sidecar 가 있다. `v1-*` 는 BL-3, `v2-*` 는 BL-4.

| module | 실제 장면 · segmentation | 핵심 기능 | 지표 (graph · 표) |
|---|---|---|---|
| gt_oracle | [scene](figures/baselines/gt_oracle/v1-gt-oracle-scene.png) | [lift check](figures/baselines/gt_oracle/v1-gt-oracle-lift-check.png) | [metrics](figures/baselines/gt_oracle/v1-gt-oracle-metrics.png) |
| grounded_sam | [scene](figures/baselines/grounded_sam/v1-gsam-scene.png) | [prompt 별 box 선택](figures/baselines/grounded_sam/v1-gsam-prompt-selection.png) | [native metrics](figures/baselines/grounded_sam/v1-gsam-native-metrics.png) |
| grounded_sam2 (per_frame) | [scene](figures/baselines/grounded_sam2/v1-gsam2-perframe-scene.png) | [strip + J/F](figures/baselines/grounded_sam2/v1-gsam2-perframe-strip-jf.png) | [native metrics](figures/baselines/grounded_sam2/v1-gsam2-perframe-native-metrics.png) · [DAVIS](figures/baselines/grounded_sam2/v1-gsam2-perframe-davis.png) |
| grounded_sam2 (track, BL-4) | [strip](figures/baselines/grounded_sam2_track/v2-sl3-track-strip.png) · [max-diff strip](figures/baselines/grounded_sam2_track/v2-sl3-track-strip-maxdiff.png) | [GPU memory](figures/baselines/grounded_sam2_track/v2-sl3-track-gpu-memory.png) | [IoU series](figures/baselines/grounded_sam2_track/v2-sl3-track-iou-series.png) · [표](figures/baselines/grounded_sam2_track/v2-sl3-track-table.png) |
| yoloe | [scene](figures/baselines/yoloe/v1-yoloe-scene.png) | [conf P/R](figures/baselines/yoloe/v1-yoloe-conf-pr.png) | [native metrics](figures/baselines/yoloe/v1-yoloe-native-metrics.png) |
| localization_heads | [scene](figures/baselines/localization_heads/v1-lh-scene.png) | [head 선택](figures/baselines/localization_heads/v1-lh-heads-selection.png) | [native metrics](figures/baselines/localization_heads/v1-lh-native-metrics.png) |
| tokencut | [scene](figures/baselines/tokencut/v1-tokencut-scene.png) | [ncut](figures/baselines/tokencut/v1-tokencut-ncut.png) | [native metrics](figures/baselines/tokencut/v1-tokencut-native-metrics.png) |
| ats_v11ab | [seed · cluster](figures/baselines/ats_v11ab/v1-ats-scene-seed-cluster.png) | [같은 점 위 label](figures/baselines/ats_v11ab/v1-ats-cloud-labeling-same-points.png) | [표](figures/baselines/ats_v11ab/v1-ats-cloud-labeling-table.png) · [paired](figures/baselines/ats_v11ab/v1-ats-cloud-labeling-paired-bootstrap.png) · [보충](figures/baselines/ats_v11ab/v1-ats-cloud-labeling-target-points-supplement.png) |
| knows | [ellipsoid](figures/baselines/knows/v1-knows-scene-ellipsoids.png) | [h(t) · path](figures/baselines/knows/v1-knows-h-path.png) | [offline metrics](figures/baselines/knows/v1-knows-offline-metrics.png) |
| aegis (3 변형) | [detection · ellipsoid](figures/baselines/aegis/v1-aegis-scene-detection-ellipsoid.png) | [h · action · 변형](figures/baselines/aegis/v1-aegis-h-actions-variants.png) | [offline metrics](figures/baselines/aegis/v1-aegis-offline-metrics.png) · [변형 간 차이](figures/baselines/aegis/v1-aegis-variant-differences.png) |
| multilink_cbf | [link ellipsoid](figures/baselines/multilink_cbf/v1-multilink-scene-link-ellipsoids.png) | [link 별 h(t)](figures/baselines/multilink_cbf/v1-multilink-link-h.png) | [offline metrics](figures/baselines/multilink_cbf/v1-multilink-offline-metrics.png) |
| omniguide | [경로](figures/baselines/omniguide/v1-omniguide-scene-paths.png) | [field slice · step](figures/baselines/omniguide/v1-omniguide-field-slices-steps.png) | [offline metrics](figures/baselines/omniguide/v1-omniguide-offline-metrics.png) |
| constrained_flow_matching | [경로](figures/baselines/constrained_flow_matching/v1-constrained-scene-paths.png) | [field slice · step](figures/baselines/constrained_flow_matching/v1-constrained-field-slices-steps.png) | [offline metrics](figures/baselines/constrained_flow_matching/v1-constrained-offline-metrics.png) |
| lan_o3dp_guidance_adapted | [경로](figures/baselines/lan_o3dp_guidance_adapted/v1-lan-scene-paths.png) | [field slice · step](figures/baselines/lan_o3dp_guidance_adapted/v1-lan-field-slices-steps.png) | [offline metrics](figures/baselines/lan_o3dp_guidance_adapted/v1-lan-offline-metrics.png) |
| nudge | [경로](figures/baselines/nudge/v1-nudge-scene-paths.png) | [field slice · step](figures/baselines/nudge/v1-nudge-field-slices-steps.png) | [offline metrics](figures/baselines/nudge/v1-nudge-offline-metrics.png) |
| benchmarks | [대표 장면](figures/baselines/benchmarks/v1-benchmarks-scenes.png) | — | [지표 정의](figures/baselines/benchmarks/v1-benchmarks-metric-definitions.png) · [protocol](figures/baselines/benchmarks/v1-benchmarks-protocol.png) |
| 총괄 | — | — | [지표 registry](figures/baselines/metrics_overview/v1-metrics-registry-overview.png) · [methods offline](figures/baselines/methods_overview/v1-methods-offline-overview.png) |
| sl4b (BL-4) | [overlay](figures/baselines/sl4b/v2-sl4b-scene-overlay.png) | — | [IoU · leak](figures/baselines/sl4b/v2-sl4b-iou-leak.png) · [표](figures/baselines/sl4b/v2-sl4b-table.png) |

- scene figure 에 쓴 frame 은 sidecar 에 있다(예: KNOWS = `libero_spatial_t4_e0` call 12 obs 70, multi-link = `libero_spatial_t4_e0` call 8 obs 50, OmniGuide = `libero_spatial_t1_e0` call 6 obs 40 injected).
- `omniguide` 의 field slice figure 는 첫 패널(배치도)의 두 제목 글자가 겹쳐 있다(외관 결함, 값 영향 없음).

### 실패 · 한계 (verify.json `limitations` · `issues`)

| # | 내용 | 영향 |
|---|---|---|
| 1 | **ATS `cloud_target` 이 `has_target` 179 call 중 134 call 에서 비어 있다.** ATS 행이 45 call 로 계산됐다. `target_points` 는 기록되어 있다. verifier 보충 행(좌표 일치 · 5 mm)을 함께 적었다 | ATS 행 해석(결과 C). ~~왜 비는지는 측정하지 않았다~~ → BL-5 에서 진단 |
| 2 | `better` / `worse` 는 diff 의 부호 개수(`metrics.py:183`)다. 지표 방향이 아니다 | BL-1 용어 절의 같은 항목을 고쳤다(용어 절) |
| 3 | **guidance 는 π0.5 가 아닌 대체 velocity** 를 쓰고 **open loop** 다. 수정 chunk 를 실행하지 않는다. filter 는 chunk 뒤 행을 method 자기 EE 예측으로 filter 한다 | guidance · filter 수치는 실제 closed-loop 효과가 아니다 |
| 4 | `ms_*` 에 jit compile 이 들어 있다(closure 가 call 마다 재생성) | latency 로 읽지 않는다 |
| 5 | **`h` 정의가 method 마다 달라 서로 비교할 수 없다** | `h_min` · `h < 0` step 을 method 사이에서 비교하지 않는다 |
| 6 | **`localization_heads` 의 head 선택과 NUDGE 의 κ 는 in-sample** 이다 | 해당 행은 held-out 이 아니다 |
| 7 | scene-GT clearance 는 0.05 m 에서 잘린다. 모든 method 에 같은 decode(OSC response, 자세 고정) · DLS IK · 61 sphere 를 쓴다 | method 내부 예측과 다를 수 있다 |
| 8 | native J / F 의 sequence 는 5 step 간격 정책 call frame 이다 | 원 benchmark(연속 frame)의 J / F 와 값이 같은 의미가 아니다 |
| 9 | YOLOE `conf` sweep 은 provider `conf` threshold 이상 후보만 저장돼 그 threshold 에서 시작한다 | `conf` 0 부터의 곡선 없음 |
| 10 | `tracking` · `constant` 두 surrogate 는 한 jit 호출에서 같이 계산되어 `ms_jit_and_sample` 이 같다 | |
| 11 | detector 두 arm 의 첫 call 시간은 Grounding DINO tiny CPU 탐지를 포함한다 | |
| 12 | OSQP 의 "Polishing not needed" stdout 이 길게 나온다(F 기록). V1 명령에서 `grep -v` 로 걸렀다 | 로그만 |
| 13 | TokenCut DINO 재계산 parity 40 중 1 sample 불일치(결과 D). verify `issues` 에는 없다 | 원인 미확인 |
| 14 | V1 의 CPU 측정과 V2 의 GPU 측정이 시간상 겹쳤다 | `ms` 에 대한 영향 미확인 |
| 15 | **guidance 의 수정 chunk decode 에서 IK 위치 오차 최대가 CFM · NUDGE 에서 0.29–1.69 m**(nominal 0.0658 m). verify.json `numbers` 에 있고 `issues` 에는 없다 | guidance arm 의 scene-GT clearance(특히 arm sphere 음수 행)를 IK 가 따르지 못한 경로 위의 값으로 읽을 수 있다. 결론에 쓰기 전에 확인이 필요하다 |

### 측정하지 않은 것

| 항목 | 이유 |
|---|---|
| `grounded_sam2:track` 의 J / F 시계열 · SL4b | BL-4 (V2) |
| closed-loop 지표(LIBERO SR, SafeLIBERO CAR / TSR / ETS, LIBERO-Safety SR / CR / time / LDLJ, methods 의 원 지표) | closed-loop 연결 전(~~BL-5~~ → BL-6) |
| YOLOE `conf` 0 부터의 P / R 곡선 | GPU 재실행 필요 |
| multi-link 의 비-gripper(forearm) binding 사례 | SL3 dev 에 없다(binding body 가 gripper 인 call 2 개뿐) |
| SegInW AP, LVIS / COCO AP, LIBERO FWT / NBT / AUC, LIBERO-Safety Refusal Rate, Lan-o3dp 원 RLBench SR | `not-applicable` (`metrics.md`) |
| π0.5 network 위의 guidance 효과 | 대체 velocity 만 사용 |
| ~~보충 ATS 행에서 `target_points` 와 cloud 점이 일치하지 않는 call(`has_target` 179 중 `tp_exact` 151, `tp_5mm` 162)의 이유~~ → | BL-5 에서 진단했다. 그 call 은 stand-in(이전 frame geometry)이다(H 진단). 28 / 17 call 의 범주 분해는 V4 가 재측정하지 않아 이 기록에 옮기지 않았다 |
| verifier 가 figure 를 사용자에게 직접 보내는 것 | verifier session 에 파일 전송 tool 이 없어 경로만 보고했다 |

### 열린 항목

1. ~~`cloud_target` 이 `has_target` call 의 대부분(134 / 179)에서 비는 이유 — V11AB perception 쪽 확인. 이유에 따라 ATS 행의 정의(`cloud_target` 대 `target_points`)를 정해야 한다.~~ → BL-5 에서 확인했다. V11AB 가 stand-in 을 target 으로 쓰고 그 `point_indices` 가 비기 때문이며 기록 버그가 아니다. ATS 행의 정의는 lead 가 exclusion 5 mm 로 정했다(사용자 판정 대기).
2. TokenCut DINO 재계산 parity 1 sample 불일치의 원인(`libero_spatial_t3_e0` obs 65 head).
3. guidance 를 π0.5 network 위에서(GPU smoke), filter 를 closed-loop 에서 측정 — ~~BL-5~~ → BL-6.
4. NUDGE κ · localization heads 를 held-out 위에서 재측정.
5. guidance 수정 chunk 의 IK 위치 오차(CFM · NUDGE 최대 1.69 m)가 scene-GT clearance 해석에 미치는 영향 — verifier 에게 확인을 요청할 항목.

### 판정

**사용자 판정 대기.**


---

## BL-4 — BASELINE-EVAL V2: track memory, SL3 track, SL4b 40 episode (2026-10-24)

> ~~(진행 중)~~ → 2026-10-24 KST 측정 완료. 아래 "계획"은 착수 시 기록이고, 결과는 그 뒤의 "결과 보고"에 이어 썼다.

> 2026-10-23 KST 시작 지시. 원본: [`handoff/BASELINE-EVAL.task.md`](handoff/BASELINE-EVAL.task.md)(순서 2). ~~**verify.json 이 아직 없어 수치를 쓰지 않는다.**~~ → 수치는 아래 "결과 보고"에 적었다(출처 `BASELINE-EVAL-V2.verify.json`).
> implementer G(CPU) → verifier V2(GPU, 공용 lock)가 `BASELINE-EVAL-V2.verify.json` 을 만든다.

**무엇을 하나.** BL-1 에서 남은 GPU 측정 두 가지(track, SL4b 40 episode)와 그 선행 수정이다. 속도 개선이 lock 점유 시간을 줄인다.

| 하위 항목 | 내용 |
|---|---|
| `grounded_sam2:track` memory 상한 | streaming state 의 memory bank 를 최근 N frame(SAM 2 기본 `num_maskmem` = 7)과 conditioning frame 만 남기도록 정리하고 처리된 이전 frame 의 feature · output 을 버린다. causal 성질을 유지하고 기존 causality test 를 통과해야 한다. CPU 에서 긴 sequence 로 memory 가 평평해지는 것을 보이고, GPU 확인은 V2 가 한다(BL-1 실패 #1 의 해소) |
| 평가 CPU 경로 속도 개선 | `offline_grounding.py` / `lift.py` / `eval/metrics.py` 를 cProfile 로 profile 해 병목(frame 마다 반복되는 GT voxel 계산, soft voxel 집계 등)을 vectorize · cache 한다. **숫자는 bit 단위로 같아야 한다**(SL3 run 의 `frames.jsonl` 중 1 episode 를 새 code 로 다시 계산해 비교)(BL-1 실패 #4 의 해소) |
| provider 예외 격리 | arm 단위로 격리하고 실패 행을 기록한 뒤 계속 진행(BL-1 실패 #3) |
| V2 측정 | SL3 track + SL4b 40 episode 전체 arm(track 포함). 시작 전 `nvidia-smi --query-compute-apps` 가 비어 있음을 확인하고 `flock -w <초> /tmp/ag3s-vla-gpu-validation.lock CMD` 로 돌린다. 시작 전과 후에 같은 pod 의 다른 session 에 알린다. sm_90 재빌드의 `cuobjdump` 확인(BL-1 열린 항목 3)도 포함한다 |

수정 금지: F 의 새 파일, `methods/**`, `AG3S/**`, `benchmark/**`.

~~**상태: 진행 중. verify 가 생기면 이어 쓴다. 판정 없음.**~~ → **측정 완료, 사용자 판정 대기.**

### 결과 보고 (2026-10-24 KST)

> 2026-10-24 KST. verifier V2 의 시각은 UTC 2026-10-23 18:47–20:10(KST +9 h = 03:47–05:10)이다. verify.json 의 `date`(2026-10-23)는 UTC 기준이고 run dir 이름의 시각(`20261024T035117` · `…T040140`)은 KST 다(`date_note`).
> 원본: [`handoff/BASELINE-EVAL-G.impl.md`](handoff/BASELINE-EVAL-G.impl.md) · [`handoff/BASELINE-EVAL-V2.verify.json`](handoff/BASELINE-EVAL-V2.verify.json). raw 출력은 `outputs/verify/BASELINE-EVAL-V2/`.
> figure: [`figures/baselines/grounded_sam2_track/`](figures/baselines/grounded_sam2_track/)(`v2-sl3-track-*` 5 개), [`figures/baselines/sl4b/`](figures/baselines/sl4b/)(`v2-sl4b-*` 3 개), 각각 `.json` sidecar.
> **판정은 사용자 판정 대기.**

### 이 STEP 이 답하는 질문 (BL-4)

1. `grounded_sam2:track` 의 GPU memory 가 episode 길이에 따라 계속 커지던 것(BL-1 실패 #1, OOM)이 해소되었는가? 이제 SL3 의 가장 긴 episode 까지 끝나는가?
2. SL3 에서 track 은 per_frame 대비 어떤가? (BL-1 에서 미측정)
3. SL4b dev 40 episode 에서 grounding baseline 전체 arm 은 어떤가? (BL-1 에서 deferred)
4. BL-1 에서 확인하지 못한 SAM 2 `_C.so` 의 sm_90 이 실제로 들어갔는가?

### 구현 경위 — implementer G (CPU 만)

G impl.md 의 구성 사실만 옮긴다. **G 의 속도 · 동일성 수치(초 · 배율 · 행 수)는 옮기지 않았다**(수치 규칙: impl.md 의 수치는 이 파일에 옮기지 않는다).

| 항목 | 원인 · 고친 것 · 검증 |
|---|---|
| `grounded_sam2:track` memory 상한 | **원인:** pinned code 의 `add_new_frame` 이 모든 frame 을 `torch.cat` 으로 `state["images"]` 에 쌓고 모든 frame 의 backbone feature 도 `cached_features` 에 남긴다(`sam2_video_predictor.py:779-802`). **고친 것(provider 쪽, pinned clone 은 수정하지 않음):** 새 frame 을 1-frame batch 로 넘기고(`state["images"]=None` 으로 두면 `torch.cat` 이 일어나지 않는다), infer 뒤에 `cached_features` 를 비우고 창 밖의 non-cond output 을 `output_dict` · `output_dict_per_obj` 에서 지운다. 남기는 것은 prompt(cond) frame, 최근 6 frame 의 mask memory, 최근 15 frame 의 object pointer 다. `CausalTrack(bounded_memory=True)` 가 기본이다. **검증(CPU, hiera-tiny):** bounded 와 unbounded 의 mask 가 bit 단위로 같고(pointer 창을 넘긴 뒤의 frame 포함) state 크기가 창 이후 평평하다. fake predictor 로 causality(t 이후 frame 을 바꿔도 t 까지의 결과가 같다)도 확인 |
| 평가 CPU 경로 속도 | cProfile 결과 병목은 거의 전부 `lift_check` 였다(arm 마다 target mesh 까지의 거리를 처음부터 다시 계산). (a) frame 마다 모든 유효 depth pixel 의 거리를 한 번만 계산하고 arm 은 자기 mask 로 index 한다. (b) 거리 계산을 reference(`libero_scene_gt.py:212`)와 **결과가 bit 단위로 같은** exact pruning 으로 바꿨다. **검증:** SL3 run 의 `frames.jsonl` 을 새 code 로 다시 계산해 저장 행과 byte 단위로 같았다(GPU provider 는 저장 mask 를 replay, 예외는 `gt_oracle` 의 wall-clock latency 칸). pruning 거리는 synthetic 과 기록 pixel 에서 reference 와 bit 동일. 주의: stored run 은 numpy 2.2.6(`.venv-baselines`)에서 나왔고 numpy 2.4.6(`.venv-ag3s`)에서는 원 code 도 마지막 bit 가 다르다(float sum 구현 차이). 그래서 byte test 는 2.2.6 에서만 돈다 |
| provider 예외 격리 | provider 하나가 build · reset · predict 에서 예외를 내거나 mask 계약을 어기면 그 arm 의 그 episode 남은 평가 frame 만 `missing_reason="provider_error"` 행으로 기록하고 다음 arm 으로 계속 간다. `--fail-fast` 는 예전 동작(첫 예외에서 중단)이다. `summary.json` 에 `provider_errors`, `provenance.json` 에 `provider_build_errors` |

**track 의 pointer 창을 15 frame 으로 한 이유.** 창은 모델 속성에서 계산한다(`read_window(vp)`, 하드코딩하지 않음). SAM 2 가 다음 frame 에서 실제로 읽는 범위는 두 가지다. mask memory 는 `num_maskmem` 7 에서 6 frame(`memory_temporal_stride_for_eval` 1), object pointer 는 `max_obj_ptrs_in_encoder` 16 에서 15 frame(`sam2_base.py:534-568` mask memory, `:611-620` object pointer; 7 → 6, 16 → 15 의 뺄셈은 현재 frame 몫으로 읽히며 impl.md 는 결과값만 적었다). 지운 frame 을 다음 frame 이 읽으면 결과가 바뀌므로 읽는 범위의 최댓값(pointer 15 frame)을 trim 창으로 잡았다. mask memory 창(6) 밖이고 pointer 창(15) 안인 frame 은 `obj_ptr` · `object_score_logits` 만 남긴다. 이 창을 줄이면 unbounded 와 mask 가 달라질 수 있어, fake predictor test 가 "창 안에서 지워진 것을 읽으면 실패"하도록 읽기 규칙을 `sam2_base.py` 대로 구현해 고정했다. V2 가 기록한 값: `read_window` = {`num_maskmem` 7, `memory_temporal_stride` 1, `mask_memory_frames` 6, `obj_ptr_frames` 15, `frames` 15}.

G 의 impl.md "내가 기대하는 결과" 절은 V2 verifier 가 측정 전에 읽었다(verify `issues`: 한 번의 `cat` 에 그 절이 포함됐다). 측정 명령과 분석 정의는 task · impl 의 명령 절과 BL-1 정의를 따랐다.

### 실행 — verifier V2 (GPU, 공용 lock)

`flock -w 3600 /tmp/ag3s-vla-gpu-validation.lock` 아래 실행, 시작 전 `nvidia-smi` 가 비어 있음을 확인했다. 알림 대상으로 지정된 `ag3s-vla-benchmark-67` 은 `SendMessage` 에서 "No agent named … is reachable" 였다. 제안된 `ag3s-vla-benchmark-1a` 와 `ag3s-vla-development-ed` 에 각 GPU 단계의 시작 · 끝을 보냈다(각 4 회). GPU 4 run 모두 rc 0, `guard_hit` 0(guard 는 `nvidia-smi memory.used` 125,000 MiB). 회귀 기준선(legacy 16D, CPU)은 일치했다(위반으로 시작 14 / 15, `has_target` 15 / 15, frame 0 `clearance_before` −29.031 mm). 첫 시도는 `/usr/bin/time` 이 이 pod 에 없어 rc 127 로 실패했고 `time` 없이 다시 돌렸다.

| run | KST (2026-10-24) | wall (s) | `nvidia-smi` 최대 `memory.used` (MiB, 5 s 표본) | 비고 |
|---|---|---|---|---|
| `check_track_memory` t0_e0 head (`sam2.1-hiera-large`, `--frames 150 --compare-frames 40`) | 03:49:09–03:49:30 | 21 | 2,466 | t0_e0 의 기록이 114 frame 뿐이라 bounded 는 114 frame 만 돌았다 |
| `check_track_memory` t2_e0 head · left_wrist | 03:49:48–03:50:28 | 40 | 22,710 | 가장 긴 SL3 episode(143 frame, 마지막 평가 obs 140). verifier 가 추가 |
| SL3 dev track (`gt_oracle` · `grounded_sam2:per_frame` · `track` × phrase / instruction + ATS) | 03:50:50–04:00:20 | 570 | 6,702 | run `20261024T035117_5752c64c7006_baseline-eval-v2-sl3dev-track`, mask 파일 1,326 개 |
| SL4b dev 전체 arm(9 spec: `gt_oracle` · `grounded_sam` · `per_frame` · `track` · `yoloe` · `localization_heads` 2 · `tokencut` 2, × prompt) | 04:00:40–05:01:40 | 3,660 | 15,602 | run `20261024T040140_5752c64c7006_baseline-eval-v2-sl4bdev`, mask 파일 12,222 개 |

- 두 GPU run 은 `OMP/MKL/OPENBLAS_NUM_THREADS=2` 로 돌았다. **BL-1 run 의 thread 설정과 같은지 확인하지 않았다.** latency 를 BL-1 과 비교할 때 이 차이가 섞인다(예: SL4b `tokencut:foreground` median 27.6 ms / frame, BL-1 SL3 391.8 ms / frame).
- `nvidia-smi memory.used` 는 5 s 간격 표본이라 짧은 최고값을 놓칠 수 있다.
- SL4b run 의 `source.patch` 에는 SL3 run 에 없던 `Baseline/paper_baselines/README.md` diff 가 있다(다른 session 이 run 사이에 수정). harness 파일(`offline_grounding.py` · `metrics.py` · `lift.py` · `providers/grounded_sam2.py` · `native_metrics.py`)의 sha256 은 실행 전후 같다.
- SL3 · SL4b 두 run 의 `provider_errors` 는 둘 다 `[]`, `provider_build_errors` 는 둘 다 `{}` 다. 모든 arm 의 `n_missing` · `n_provider_error` 는 0 이다.
- SL3 · SL4b log 에 SAM 2 `no kernel image is available` 경고는 없다(`no_kernel_image_warning_in_run_logs` 0). torch SDPA 의 Flash / Memory-efficient / CuDNN attention 미사용 UserWarning(`sam2/modeling/sam/transformer.py:270`, float32 입력)이 1 회씩 있다.
- V1 의 CPU 측정(03:58 KST 시작)과 시간상 겹친다.

### 결과 A — SAM 2 `_C.so` 의 sm_90 (BL-1 열린 항목 3 해소)

| 항목 | 값 |
|---|---|
| `cuobjdump --list-elf external/Grounded-SAM-2/sam2/_C.so` | sm_70, sm_75, sm_80, sm_86, **sm_90** |
| `cuobjdump --list-ptx` | **sm_90** |
| `_C.so` mtime | 2026-10-23T07:08:47 UTC = 16:08:47 KST (BL-1 측정이 끝난 뒤) |
| sha256 | `7827c7e1d0b6a065b13a8a8c5b34bc723f15b7d945dbd2be702924307ff00093` |
| `from sam2 import _C` (`.venv-baselines`, CUDA 가림) | ok (rc 0) |
| SL3 · SL4b run log 의 `no kernel image is available` 경고 | 0 |

`setup_venv.sh` 가 export 하는 `NVCC_APPEND_FLAGS="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90"`(BL-1 기록)가 sm_90 cubin 과 PTX 를 덧붙였음이 확인됐다. BL-1 의 track 은 측정하지 못했고 per_frame 은 `_C` 를 쓰지 않으므로 BL-1 수치는 이 재빌드에 의존하지 않는다.

### 결과 B — track GPU memory 평탄성 (BL-1 실패 #1 해소)

`check_track_memory.py` 가 GT mask 로 prompt 를 준 뒤 매 frame `CausalTrack.step` 을 부르고, `torch.cuda.memory_allocated` · `memory_reserved` 와 state footprint 를 기록한다. bounded(G 의 수정) 대 unbounded(원 동작)를 같은 입력으로 비교하고(비교 frame 40), unbounded 는 40 frame 뒤에 멈춘다. 모델은 `sam2.1-hiera-large`, H200.

**표 B-1. bounded (G 의 수정)** — 세 기록의 값이 같아 곡선이 겹친다

| 기록 (camera) | 입력 frame 수 | bounded: `memory_allocated` MiB, 첫 → 끝 | `allocated` 가 일정해진 시점 | bounded `reserved` 최대 MiB | bounded state bytes, 첫 → 끝 | state tensor / non-cond output / `cached_features` / `images` frame (끝) | bounded = unbounded mask 일치 frame |
|---|---|---|---|---|---|---|---|
| libero_spatial_t0_e0 head | 114 | 1010.3 → 1047.3 (최대 1047.3) | frame 7 | 1778 | 31,196,164 → 35,930,176 (frame 16 부터 일정) | 109 / 15 / 0 / 1 | 40 / 40 |
| libero_spatial_t2_e0 head | 143 | 1010.3 → 1047.3 (최대 1047.3) | frame 7 | 1778 | 31,196,164 → 35,930,176 (frame 16 부터 일정) | 109 / 15 / 0 / 1 | 40 / 40 |
| libero_spatial_t2_e0 left_wrist | 143 | 1042.8 → 1047.3 (최대 1047.3) | frame 7 | 22022 | 31,196,164 → 35,930,176 (frame 16 부터 일정) | 109 / 15 / 0 / 1 | 40 / 40 |

**표 B-2. unbounded (원 동작, 40 frame 까지)** — `allocated` 기울기 = (frame 40 의 값 − frame 2 의 값) / 38

| 기록 (camera) | unbounded: `memory_allocated` MiB, frame 2 → 40 | `allocated` 기울기 MiB / frame | state bytes (40 frame) | state 기울기 MiB / frame | `reserved` MiB (40 frame) |
|---|---|---|---|---|---|
| libero_spatial_t0_e0 head | 1154.8 → 6514.6 | 137.45 | 5,740,994,720 | 136.75 | 22022 |
| libero_spatial_t2_e0 head | 1154.8 → 6514.6 | 137.45 | 5,740,994,720 | 136.75 | 22022 |
| libero_spatial_t2_e0 left_wrist | 1154.8 → 6506.6 | 137.24 | 5,740,994,720 | 136.75 | 22022 |

![track GPU memory: bounded 대 unbounded](figures/baselines/grounded_sam2_track/v2-sl3-track-gpu-memory.png)

관찰:
- bounded 는 `memory_allocated` 가 track frame 7 부터 1,047.3 MiB 로 일정하고, state 는 frame 16 부터 일정하다. 143 frame(SL3 의 가장 긴 episode, 마지막 평가 obs 140)까지 늘지 않았다. 끝 상태에서 non-cond output 15 개, `images` 1 frame, `cached_features` 0 이다.
- unbounded 는 frame 마다 `allocated` 137.45 MiB(head) · 137.24 MiB(left_wrist)씩 늘고 40 frame 에서 6,514.6 MiB 다.
- bounded 의 mask 는 비교한 40 frame 에서 모두 unbounded 와 같다(세 기록 모두 40 / 40). GPU(CUDA)에서 G 의 CPU 검증 결과가 재현됐다.
- t2_e0 left_wrist 의 bounded `reserved` 최대 22,022 MiB 는 같은 process 에서 head unbounded 실행 뒤에 돌아 caching allocator 가 앞 실행의 reserved 를 유지한 것이다(`allocated` 는 1,047.3 MiB 로 head 와 같다)(verify `issues`).
- **BL-1 의 추정 정정.** BL-1 은 OOM 시점의 byte 수(147,696,150,272)에서 "frame · camera 당 약 0.9–1.0 GB 가 누적"이라고 적었다. V2 의 직접 측정(unbounded, `memory_allocated` 기울기)은 137.45 MiB / frame 이다. 두 값은 크게 다르다. BL-1 의 값은 OOM 메시지에서 나눈 추정이었고, 차이의 원인은 이 측정이 분리하지 않았다(BL-1 run 1 은 provider 여러 개와 두 prompt mode 가 한 process 에서 돌았다).

### 결과 C — SL3 dev: track 대 per_frame (10 episode · 221 frame)

BL-1 에서 OOM 으로 미측정이던 `grounded_sam2:track` 을 같은 SL3 dev 집합에서 측정했다. 같은 run 에 `gt_oracle` · `per_frame` · ATS 를 넣어 paired 비교를 가능하게 했다. **재현성:** 이 run 의 `per_frame` 2 arm · `gt_oracle` 2 arm · ATS 는 BL-1 의 notrack run(`20261023T134909_…`)과 221 / 221 행이 latency 칸을 빼고 JSON 단위로 같다(지표 최대 절대 차이 0.0). G 의 수정(속도 개선)이 GPU run 에서도 값을 바꾸지 않았다는 증거로 읽힌다.

| arm | PAPER-BASELINES-2 notrack run 과 JSON 이 같은 행 (latency 제외) / 공통 행 | 지표 최대 절대 차이 |
|---|---|---|
| GT oracle (phrase) | 221 / 221 | 0.0 |
| GT oracle (instruction) | 221 / 221 | 0.0 |
| Grounded-SAM-2 per_frame (phrase) | 221 / 221 | 0.0 |
| Grounded-SAM-2 per_frame (instruction) | 221 / 221 | 0.0 |
| ATS V11AB (harness) | 221 / 221 | 0.0 |

**표 C-1. harness 지표** (5 mm voxel, episode 평균 → 평균, 정의는 BL-1 과 같다)

| arm | 3D soft IoU | soft P | soft R | hard IoU | found any (call) | lift within 10 mm | head 2D IoU (found) | left_wrist 2D IoU (found) | neighbour leak (voxel mass) | neighbour pooled | wrong-object 3D (frame, 비율) | wrong-object 3D, 비율 > 0.5 (frame, 비율) | wrong-object 2D instance (frame / 비어 있지 않은 frame) | latency median / p95 ms / frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GT oracle (phrase) | 0.968 | 0.968 | 1.000 | 0.976 | 221 / 221 | 0.994 | 1.000 (1.000) | 1.000 (0.968) | 1.3 | 0.002 | 0 (0.000) | 0 (0.000) | 0 / 221 (0.000) | 0.2 / 0.4 |
| Grounded-SAM-2 per_frame (phrase) | 0.511 | 0.597 | 0.698 | 0.513 | 220 / 221 | 0.798 | 0.242 (0.860) | 0.842 (0.968) | 341.2 | 0.375 | 56 (0.253) | 46 (0.208) | 21 / 220 (0.095) | 246.5 / 265.8 |
| Grounded-SAM-2 per_frame (instruction) | 0.545 | 0.643 | 0.709 | 0.547 | 219 / 221 | 0.822 | 0.286 (0.805) | 0.824 (0.937) | 292.3 | 0.339 | 51 (0.231) | 41 (0.186) | 20 / 219 (0.091) | 241.7 / 253.5 |
| Grounded-SAM-2 track (phrase) | 0.391 | 0.557 | 0.532 | 0.394 | 221 / 221 | 0.669 | 0.455 (1.000) | 0.288 (0.493) | 294.2 | 0.393 | 71 (0.321) | 54 (0.244) | 43 / 221 (0.195) | 144.2 / 148.3 |
| Grounded-SAM-2 track (instruction) | 0.490 | 0.671 | 0.627 | 0.493 | 221 / 221 | 0.765 | 0.636 (1.000) | 0.289 (0.493) | 219.2 | 0.305 | 48 (0.217) | 31 (0.140) | 25 / 221 (0.113) | 143.6 / 147.6 |
| ATS V11AB (harness) | 0.064 | 0.393 | 0.068 | 0.065 | 179 / 221 | 0.535 | - (-) | - (-) | 33.2 | 0.242 | 55 (0.249) | 46 (0.208) | - | 94.7 / 121.0 |

- latency: track 의 값은 **평가 frame 자신의 step 시간만**이다. 평가 frame 사이의 frame(obs 0 부터 마지막 평가 frame 까지 매 frame)도 provider 에 넣지만 행에 시간이 남지 않는다(`not_measured`). per_frame 과 같은 척도가 아니다. per_frame 의 값(246.5 ms)은 BL-1(381.0 ms)과 다르다. 같은 mask 이므로 thread 설정(위) 등 환경 차이일 수 있으나 원인은 확인하지 않았다. ATS 는 V11AB 의 `t_process_s`(perception 전체)다.

**표 C-2. track − per_frame (paired, episode 단위 2,000 회 bootstrap)** — 95 % CI, (better / worse = diff 부호 개수)

| a − b (SL3, 10 episode) | Δ 3D soft IoU | Δ soft P | Δ soft R | Δ neighbour leak | Δ head 2D IoU | Δ left_wrist 2D IoU |
|---|---|---|---|---|---|---|
| track (phrase) − per_frame (phrase) | -0.120 [-0.221, -0.025] (2 / 8) | -0.039 [-0.177, 0.093] (3 / 7) | -0.166 [-0.310, -0.050] (2 / 8) | -47.018 [-155.719, 37.768] (6 / 4) | +0.213 [0.000, 0.445] (5 / 2) | -0.553 [-0.807, -0.273] (3 / 7) |
| track (instruction) − per_frame (instruction) | -0.055 [-0.138, 0.028] (4 / 6) | +0.028 [-0.057, 0.120] (6 / 4) | -0.082 [-0.169, 0.009] (3 / 7) | -73.150 [-175.051, 8.949] (4 / 6) | +0.350 [0.163, 0.550] (7 / 2) | -0.535 [-0.792, -0.258] (3 / 7) |
| track (instruction) − track (phrase) | +0.098 [-0.001, 0.245] (4 / 2) | +0.114 [-0.002, 0.274] (3 / 3) | +0.095 [0.000, 0.224] (3 / 0) | -74.976 [-183.861, 2.463] (3 / 2) | +0.181 [-0.000, 0.451] (2 / 1) | +0.000 [-0.000, 0.000] (2 / 1) |

**표 C-3. arm − ATS (5 mm voxel harness, 대표점 표현이 섞인 수치 — BL-1 #1 박스 참조)**

| arm − ATS (SL3, 5 mm voxel harness) | Δ 3D soft IoU | Δ soft P | Δ soft R | Δ neighbour leak |
|---|---|---|---|---|
| GT oracle (두 prompt 같음) | +0.904 [0.876, 0.932] (10 / 0) | +0.575 [0.423, 0.744] (10 / 0) | +0.932 [0.905, 0.960] (10 / 0) | -31.872 [-61.709, -4.516] (4 / 6) |
| Grounded-SAM-2 per_frame (phrase) | +0.447 [0.339, 0.553] (10 / 0) | +0.204 [0.059, 0.382] (8 / 2) | +0.629 [0.529, 0.708] (10 / 0) | +308.016 [181.360, 430.294] (9 / 1) |
| Grounded-SAM-2 per_frame (instruction) | +0.481 [0.365, 0.590] (10 / 0) | +0.250 [0.092, 0.430] (8 / 2) | +0.641 [0.543, 0.722] (10 / 0) | +259.171 [128.176, 395.969] (10 / 0) |
| Grounded-SAM-2 track (phrase) | +0.327 [0.183, 0.442] (8 / 2) | +0.164 [-0.076, 0.414] (6 / 4) | +0.464 [0.268, 0.611] (8 / 2) | +260.997 [135.520, 382.317] (8 / 2) |
| Grounded-SAM-2 track (instruction) | +0.425 [0.294, 0.536] (9 / 1) | +0.278 [0.110, 0.467] (8 / 2) | +0.559 [0.404, 0.679] (9 / 1) | +186.021 [71.180, 308.791] (8 / 2) |

**표 C-4. track 의 camera · source 상태**

| 집합 | arm | head `found` | head `track_lost` | head detect 실패 (`no_box_above_threshold`) | left_wrist `found` | left_wrist `track_lost` | left_wrist `track_lost` 가 있는 episode | head `track_lost` 가 있는 episode |
|---|---|---|---|---|---|---|---|---|
| SL3 (221 frame · 10 episode) | Grounded-SAM-2 track (phrase) | 221 | 0 | 0 | 109 | 112 | 7 / 10 | 0 |
| SL3 (221 frame · 10 episode) | Grounded-SAM-2 track (instruction) | 221 | 0 | 0 | 109 | 112 | 7 / 10 | 0 |
| SL4b (873 frame · 40 episode) | Grounded-SAM-2 track (phrase) | 854 | 19 | 0 | 372 | 501 | 28 / 40 | 4 |
| SL4b (873 frame · 40 episode) | Grounded-SAM-2 track (instruction) | 866 | 1 | 6 | 333 | 540 | 30 / 40 | 1 |

**표 C-5. 고유 지표 (Acc@0.5 · CorLoc · mIoU · oIoU · J / F / J&F)** — SL3 와 SL4b, per_frame 과 track

| 집합 | arm | camera | sample 수 | Acc@0.5 | CorLoc any | mIoU | oIoU | J | F | J&F | J recall | J decay | temporal IoU 예측 | temporal IoU GT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SL3 | Grounded-SAM-2 per_frame (phrase) | all | 435 | 0.568 | 0.910 | 0.538 | 0.776 | 0.528 | 0.527 | 0.527 | 0.554 | -0.073 | 0.712 | 0.662 |
| SL3 | Grounded-SAM-2 per_frame (phrase) | head | 221 | 0.267 | 0.860 | 0.243 | 0.160 | 0.242 | 0.257 | 0.249 | 0.262 | 0.063 | 0.799 | 0.629 |
| SL3 | Grounded-SAM-2 per_frame (phrase) | left_wrist | 214 | 0.879 | 0.963 | 0.843 | 0.871 | 0.813 | 0.797 | 0.805 | 0.847 | -0.209 | 0.625 | 0.695 |
| SL3 | Grounded-SAM-2 per_frame (instruction) | all | 435 | 0.582 | 0.869 | 0.552 | 0.781 | 0.541 | 0.540 | 0.540 | 0.568 | 0.000 | 0.716 | 0.662 |
| SL3 | Grounded-SAM-2 per_frame (instruction) | head | 221 | 0.312 | 0.805 | 0.288 | 0.206 | 0.286 | 0.300 | 0.293 | 0.306 | 0.221 | 0.813 | 0.629 |
| SL3 | Grounded-SAM-2 per_frame (instruction) | left_wrist | 214 | 0.860 | 0.935 | 0.824 | 0.865 | 0.796 | 0.779 | 0.787 | 0.830 | -0.220 | 0.619 | 0.695 |
| SL3 | Grounded-SAM-2 track (phrase) | all | 435 | 0.418 | 0.731 | 0.391 | 0.386 | 0.372 | 0.390 | 0.381 | 0.400 | 0.016 | 0.780 | 0.662 |
| SL3 | Grounded-SAM-2 track (phrase) | head | 221 | 0.493 | 0.982 | 0.461 | 0.356 | 0.455 | 0.479 | 0.467 | 0.500 | 0.044 | 0.784 | 0.629 |
| SL3 | Grounded-SAM-2 track (phrase) | left_wrist | 214 | 0.341 | 0.472 | 0.318 | 0.390 | 0.288 | 0.300 | 0.294 | 0.300 | -0.012 | 0.776 | 0.695 |
| SL3 | Grounded-SAM-2 track (instruction) | all | 435 | 0.501 | 0.724 | 0.472 | 0.404 | 0.462 | 0.485 | 0.474 | 0.500 | 0.024 | 0.764 | 0.662 |
| SL3 | Grounded-SAM-2 track (instruction) | head | 221 | 0.656 | 0.968 | 0.621 | 0.507 | 0.636 | 0.670 | 0.653 | 0.700 | 0.060 | 0.751 | 0.629 |
| SL3 | Grounded-SAM-2 track (instruction) | left_wrist | 214 | 0.341 | 0.472 | 0.319 | 0.390 | 0.289 | 0.300 | 0.294 | 0.300 | -0.012 | 0.777 | 0.695 |
| SL4b | Grounded-SAM-2 per_frame (phrase) | all | 1715 | 0.581 | 0.913 | 0.553 | 0.803 | 0.533 | 0.530 | 0.531 | 0.559 | -0.042 | 0.724 | 0.653 |
| SL4b | Grounded-SAM-2 per_frame (phrase) | head | 873 | 0.262 | 0.847 | 0.244 | 0.169 | 0.227 | 0.237 | 0.232 | 0.245 | 0.132 | 0.808 | 0.608 |
| SL4b | Grounded-SAM-2 per_frame (phrase) | left_wrist | 842 | 0.912 | 0.982 | 0.874 | 0.899 | 0.839 | 0.822 | 0.830 | 0.874 | -0.216 | 0.640 | 0.698 |
| SL4b | Grounded-SAM-2 per_frame (instruction) | all | 1715 | 0.550 | 0.836 | 0.525 | 0.794 | 0.512 | 0.509 | 0.510 | 0.536 | -0.046 | 0.721 | 0.653 |
| SL4b | Grounded-SAM-2 per_frame (instruction) | head | 873 | 0.227 | 0.727 | 0.211 | 0.149 | 0.207 | 0.216 | 0.211 | 0.222 | 0.128 | 0.817 | 0.608 |
| SL4b | Grounded-SAM-2 per_frame (instruction) | left_wrist | 842 | 0.886 | 0.948 | 0.849 | 0.887 | 0.817 | 0.801 | 0.809 | 0.850 | -0.221 | 0.624 | 0.698 |
| SL4b | Grounded-SAM-2 track (phrase) | all | 1715 | 0.373 | 0.689 | 0.348 | 0.284 | 0.326 | 0.341 | 0.333 | 0.350 | 0.016 | 0.801 | 0.653 |
| SL4b | Grounded-SAM-2 track (phrase) | head | 873 | 0.457 | 0.964 | 0.421 | 0.329 | 0.364 | 0.387 | 0.375 | 0.400 | 0.037 | 0.816 | 0.608 |
| SL4b | Grounded-SAM-2 track (phrase) | left_wrist | 842 | 0.286 | 0.403 | 0.273 | 0.277 | 0.289 | 0.294 | 0.291 | 0.300 | -0.005 | 0.787 | 0.698 |
| SL4b | Grounded-SAM-2 track (instruction) | all | 1715 | 0.457 | 0.668 | 0.428 | 0.231 | 0.426 | 0.447 | 0.436 | 0.461 | 0.034 | 0.771 | 0.653 |
| SL4b | Grounded-SAM-2 track (instruction) | head | 873 | 0.677 | 0.969 | 0.632 | 0.538 | 0.612 | 0.647 | 0.630 | 0.671 | 0.073 | 0.755 | 0.608 |
| SL4b | Grounded-SAM-2 track (instruction) | left_wrist | 842 | 0.229 | 0.355 | 0.218 | 0.191 | 0.240 | 0.246 | 0.243 | 0.250 | -0.004 | 0.787 | 0.698 |

![track vs per-frame: 대표 episode strip](figures/baselines/grounded_sam2_track/v2-sl3-track-strip.png)

![track vs per-frame: |Δ| 가 가장 큰 episode strip](figures/baselines/grounded_sam2_track/v2-sl3-track-strip-maxdiff.png)

![frame 별 3D soft IoU: per_frame 대 track](figures/baselines/grounded_sam2_track/v2-sl3-track-iou-series.png)

관찰:
- SL3 에서 track(phrase)의 3D soft IoU 는 0.391, per_frame(phrase)은 0.511 이다. 차이 −0.120(CI −0.221 ~ −0.025, 2 / 8). instruction 은 track 0.490, per_frame 0.545 로 차이 −0.055 (CI −0.138 ~ +0.028, 4 / 6)이다.
- **camera 별로 방향이 반대다.** head 2D IoU 는 track 이 높다(phrase +0.213, instruction +0.350; CI 하한이 0 이상 / 0 위). left_wrist 2D IoU 는 track 이 크게 낮다(phrase −0.553, instruction −0.535; CI 모두 0 아래).
- 원인 중 하나로 보이는 사실: track 의 left_wrist 는 **221 frame 중 112 frame 이 `track_lost`**(7 / 10 episode), `found` 는 109 frame(0.493)이다. provider 가 `redetect_on_loss=false` 라 잃은 뒤 다시 검출하지 않는다. head 는 221 frame 모두 `found` 다. SL3 의 phrase · instruction 두 track arm 이 같은 left_wrist 상태 count 를 보인다.
- track(instruction)의 wrong-object 3D 는 48 / 221(0.217), per_frame(instruction)은 51 / 221(0.231)이다. 2D instance 기준 track(phrase)은 43 / 221(0.195), per_frame(phrase)은 21 / 220(0.095)이다.
- 같은 mask 로 만든 SL3 의 J&F: track 0.381(phrase) · 0.474(instruction), per_frame 0.527 · 0.540.
- ATS 대비 Δ 3D soft IoU(harness)는 track 도 CI 가 0 위(phrase +0.327, instruction +0.425)다. 단 ATS 행의 표현 문제(BL-1, BL-3)가 섞인 수치다.
- 그림에서 확인되는 것(수치는 아님): 대표 strip 의 left_wrist 한 episode 에서 track mask 가 GT target 이 아니라 같은 종류의 distractor 위에 놓인 frame 이 있다(`track-strip-maxdiff`, `libero_spatial_t8_e0`).

### 결과 D — SL4b dev 40 episode 전체 arm (873 frame)

SL4b dev: `libero_spatial` t0–9 e1–4, 40 episode · 873 evaluated frame, camera head + left_wrist, `--size default`, H200. **SL4b 에는 V11AB perception run 이 없어 ATS 행이 없다**(`not_measured`). paired 기준은 `grounded_sam2:per_frame|instruction` 이다. `localization_heads` 의 head 는 SL3 + SL4b dev 의 attention audit 에서 골랐으므로 **이 행은 in-sample** 이다(verify `in_sample` true).

**표 D-1. harness 지표 (13 arm + GT oracle)**

| arm | 3D soft IoU | soft P | soft R | hard IoU | found any (call) | lift within 10 mm | head 2D IoU (found) | left_wrist 2D IoU (found) | neighbour leak (voxel mass) | neighbour pooled | wrong-object 3D (frame, 비율) | wrong-object 3D, 비율 > 0.5 (frame, 비율) | wrong-object 2D instance (frame / 비어 있지 않은 frame) | latency median / p95 ms / frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GT oracle (phrase) | 0.968 | 0.968 | 1.000 | 0.976 | 873 / 873 | 0.995 | 1.000 (1.000) | 1.000 (0.964) | 1.4 | 0.002 | 0 (0.000) | 0 (0.000) | 0 / 873 (0.000) | 0.2 / 0.3 |
| Grounded-SAM (phrase) | 0.511 | 0.575 | 0.732 | 0.513 | 873 / 873 | 0.808 | 0.268 (0.968) | 0.860 (0.999) | 373.0 | 0.359 | 314 (0.360) | 273 (0.313) | 68 / 873 (0.078) | 449.2 / 543.3 |
| Grounded-SAM (instruction) | 0.534 | 0.588 | 0.743 | 0.537 | 873 / 873 | 0.815 | 0.320 (0.990) | 0.854 (0.999) | 360.5 | 0.374 | 312 (0.357) | 266 (0.305) | 67 / 873 (0.077) | 445.6 / 519.4 |
| Grounded-SAM-2 per_frame (phrase) | 0.519 | 0.598 | 0.721 | 0.521 | 873 / 873 | 0.814 | 0.227 (0.848) | 0.869 (0.986) | 351.4 | 0.367 | 299 (0.342) | 244 (0.279) | 72 / 873 (0.082) | 281.2 / 317.9 |
| Grounded-SAM-2 per_frame (instruction) | 0.528 | 0.632 | 0.698 | 0.530 | 865 / 873 | 0.823 | 0.207 (0.727) | 0.844 (0.948) | 303.4 | 0.344 | 278 (0.318) | 226 (0.259) | 72 / 865 (0.083) | 277.8 / 312.2 |
| Grounded-SAM-2 track (phrase) | 0.333 | 0.477 | 0.462 | 0.335 | 855 / 873 | 0.596 | 0.364 (0.978) | 0.289 (0.426) | 332.2 | 0.439 | 347 (0.397) | 302 (0.346) | 228 / 855 (0.267) | 147.4 / 153.1 |
| Grounded-SAM-2 track (instruction) | 0.450 | 0.641 | 0.568 | 0.453 | 873 / 873 | 0.725 | 0.612 (0.992) | 0.240 (0.381) | 220.2 | 0.320 | 255 (0.292) | 211 (0.242) | 142 / 873 (0.163) | 147.2 / 154.4 |
| YOLOE-11L (phrase) | 0.355 | 0.545 | 0.457 | 0.355 | 746 / 873 | 0.664 | 0.104 (0.614) | 0.539 (0.630) | 255.4 | 0.389 | 276 (0.316) | 253 (0.290) | 175 / 746 (0.235) | 24.7 / 108.7 |
| YOLOE-11L (instruction) | 0.117 | 0.432 | 0.150 | 0.117 | 405 / 873 | 0.470 | 0.002 (0.297) | 0.172 (0.179) | 89.8 | 0.072 | 151 (0.173) | 76 (0.087) | 103 / 405 (0.254) | 23.8 / 44.4 |
| localization_heads:threshold (in-sample) | 0.073 | 0.074 | 0.815 | 0.073 | 873 / 873 | 0.190 | 0.039 (1.000) | 0.229 (1.000) | 679.9 | 0.083 | 491 (0.562) | 0 (0.000) | 40 / 873 (0.046) | 2.4 / 3.2 |
| localization_heads:sam_point (in-sample) | 0.215 | 0.277 | 0.402 | 0.217 | 873 / 873 | 0.285 | 0.240 (1.000) | 0.308 (1.000) | 195.4 | 0.061 | 342 (0.392) | 148 (0.170) | 21 / 873 (0.024) | 271.7 / 340.8 |
| TokenCut foreground | 0.078 | 0.079 | 0.746 | 0.078 | 873 / 873 | 0.325 | 0.034 (1.000) | 0.577 (1.000) | 466.5 | 0.066 | 355 (0.407) | 4 (0.005) | 60 / 873 (0.069) | 27.6 / 32.5 |
| TokenCut attention_overlap | 0.072 | 0.075 | 0.557 | 0.073 | 873 / 873 | 0.155 | 0.047 (1.000) | 0.287 (1.000) | 863.7 | 0.074 | 708 (0.811) | 42 (0.048) | 61 / 873 (0.070) | 29.6 / 35.3 |

**표 D-2. arm − `grounded_sam2:per_frame|instruction` (paired, 40 episode)** — 95 % CI, (better / worse)

| arm − Grounded-SAM-2 per_frame (instruction), SL4b 40 episode | Δ 3D soft IoU | Δ soft P | Δ soft R | Δ neighbour leak | Δ head 2D IoU | Δ left_wrist 2D IoU |
|---|---|---|---|---|---|---|
| GT oracle (두 prompt 같음) | +0.440 [0.380, 0.499] (40 / 0) | +0.337 [0.264, 0.407] (40 / 0) | +0.302 [0.257, 0.348] (40 / 0) | -302.020 [-373.342, -228.086] (1 / 39) | +0.793 [0.717, 0.862] (40 / 0) | +0.156 [0.117, 0.202] (40 / 0) |
| Grounded-SAM (phrase) | -0.017 [-0.035, -0.000] (19 / 21) | -0.057 [-0.086, -0.032] (14 / 26) | +0.033 [0.016, 0.052] (33 / 7) | +69.611 [45.718, 97.936] (30 / 10) | +0.061 [0.022, 0.106] (19 / 8) | +0.016 [-0.008, 0.037] (22 / 18) |
| Grounded-SAM (instruction) | +0.006 [-0.011, 0.019] (28 / 12) | -0.043 [-0.066, -0.023] (16 / 24) | +0.045 [0.028, 0.061] (34 / 6) | +57.082 [39.594, 75.237] (33 / 7) | +0.113 [0.078, 0.152] (27 / 0) | +0.010 [-0.011, 0.028] (23 / 17) |
| Grounded-SAM-2 per_frame (phrase) | -0.009 [-0.025, 0.006] (18 / 22) | -0.034 [-0.056, -0.014] (18 / 22) | +0.022 [0.003, 0.041] (29 / 10) | +47.962 [27.522, 72.237] (28 / 12) | +0.021 [-0.018, 0.062] (12 / 12) | +0.025 [0.012, 0.041] (35 / 5) |
| Grounded-SAM-2 track (phrase) | -0.195 [-0.269, -0.121] (8 / 32) | -0.155 [-0.256, -0.061] (13 / 27) | -0.237 [-0.334, -0.142] (11 / 29) | +28.776 [-30.890, 89.249] (24 / 16) | +0.157 [0.038, 0.286] (16 / 11) | -0.555 [-0.697, -0.404] (10 / 30) |
| Grounded-SAM-2 track (instruction) | -0.078 [-0.136, -0.017] (13 / 27) | +0.009 [-0.053, 0.075] (18 / 22) | -0.130 [-0.198, -0.056] (12 / 28) | -83.223 [-139.749, -34.483] (18 / 22) | +0.405 [0.296, 0.521] (27 / 4) | -0.604 [-0.734, -0.457] (8 / 32) |
| YOLOE-11L (phrase) | -0.174 [-0.225, -0.122] (6 / 34) | -0.086 [-0.159, -0.010] (12 / 28) | -0.241 [-0.299, -0.185] (4 / 36) | -48.062 [-120.954, 20.278] (23 / 17) | -0.103 [-0.195, -0.016] (10 / 22) | -0.304 [-0.374, -0.241] (0 / 40) |
| YOLOE-11L (instruction) | -0.411 [-0.475, -0.350] (1 / 39) | -0.206 [-0.341, -0.070] (12 / 27) | -0.548 [-0.597, -0.499] (0 / 40) | -213.667 [-290.970, -139.725] (6 / 34) | -0.204 [-0.281, -0.136] (1 / 26) | -0.672 [-0.743, -0.602] (0 / 40) |
| localization_heads:threshold (in-sample) | -0.455 [-0.516, -0.396] (0 / 40) | -0.558 [-0.631, -0.487] (0 / 40) | +0.117 [0.052, 0.181] (29 / 11) | +376.470 [277.881, 477.330] (33 / 7) | -0.167 [-0.245, -0.098] (16 / 24) | -0.615 [-0.662, -0.565] (0 / 40) |
| localization_heads:sam_point (in-sample) | -0.313 [-0.380, -0.246] (4 / 36) | -0.355 [-0.434, -0.272] (4 / 36) | -0.296 [-0.354, -0.234] (4 / 36) | -108.030 [-181.563, -34.115] (17 / 23) | +0.033 [-0.049, 0.110] (24 / 16) | -0.536 [-0.589, -0.476] (0 / 40) |
| TokenCut foreground | -0.450 [-0.511, -0.391] (0 / 40) | -0.553 [-0.625, -0.481] (0 / 40) | +0.048 [-0.014, 0.108] (24 / 16) | +163.019 [61.391, 276.757] (21 / 19) | -0.173 [-0.251, -0.104] (16 / 24) | -0.266 [-0.313, -0.222] (2 / 38) |
| TokenCut attention_overlap | -0.456 [-0.521, -0.394] (0 / 40) | -0.556 [-0.632, -0.482] (0 / 40) | -0.141 [-0.203, -0.080] (9 / 31) | +560.271 [455.482, 664.964] (37 / 3) | -0.159 [-0.232, -0.094] (15 / 25) | -0.557 [-0.611, -0.495] (1 / 39) |

**표 D-3. track − per_frame (SL4b)**

| a − b (SL4b, 40 episode) | Δ 3D soft IoU | Δ soft P | Δ soft R | Δ neighbour leak | Δ head 2D IoU | Δ left_wrist 2D IoU |
|---|---|---|---|---|---|---|
| track (phrase) − per_frame (phrase) | -0.186 [-0.258, -0.115] (7 / 33) | -0.121 [-0.223, -0.029] (14 / 26) | -0.259 [-0.353, -0.167] (7 / 33) | -19.186 [-81.267, 40.581] (20 / 20) | +0.137 [0.021, 0.262] (16 / 10) | -0.580 [-0.722, -0.432] (9 / 31) |
| track (instruction) − per_frame (instruction) | -0.078 [-0.136, -0.017] (13 / 27) | +0.009 [-0.053, 0.075] (18 / 22) | -0.130 [-0.198, -0.056] (12 / 28) | -83.223 [-139.749, -34.483] (18 / 22) | +0.405 [0.296, 0.521] (27 / 4) | -0.604 [-0.734, -0.457] (8 / 32) |

![SL4b: 3D soft IoU 와 neighbour leak](figures/baselines/sl4b/v2-sl4b-iou-leak.png)

관찰:
- 40 episode · 873 frame 에서 `gt_oracle` 의 3D soft IoU 는 0.968 이고 다른 arm 의 최댓값은 Grounded-SAM(instruction) 0.534 다. lift 가 성하다는 task 의 통과 기준 입력이다(판정은 사용자).
- Grounded-SAM · per_frame 의 3D soft IoU 는 0.511–0.534 로 SL3 dev(0.511–0.546)와 같은 범위다. per_frame(instruction) 대비 Grounded-SAM(phrase)은 Δ −0.017(CI −0.035 ~ −0.000), (instruction)은 +0.006 이다.
- track 은 phrase 0.333 · instruction 0.450 으로 per_frame 보다 낮다(Δ −0.186 [−0.258, −0.115] · −0.078 [−0.136, −0.017]). SL3 와 같은 방향이다. camera 별로는 head 2D IoU 가 높고(Δ +0.137 · +0.405) left_wrist 2D IoU 가 크게 낮다(Δ −0.580 · −0.604).
- track 의 left_wrist `track_lost` 는 phrase 501 / 873 frame(28 / 40 episode), instruction 540 / 873 frame(30 / 40 episode)이다. head 는 phrase 19 frame(4 episode), instruction 1 frame(1 episode)이고, instruction 의 head 6 frame 은 track 시작 전 detect 단계의 `no_box_above_threshold` 다.
- YOLOE-11L 은 phrase 0.355, instruction 0.117 이고 found any 가 0.855 / 0.464 다. `localization_heads` 는 threshold 0.073 / sam_point 0.215(in-sample), TokenCut 은 foreground 0.078 / attention_overlap 0.072 다.
- neighbour leak 은 TokenCut attention_overlap(863.7)과 `localization_heads:threshold`(679.9)가 가장 크고, 같은 두 arm 의 neighbour pooled 비율(0.074 · 0.083)은 낮고 soft P 가 0.075 · 0.074 이다(예측 voxel 대부분이 target 이 아니다).
- latency(thread 2, BL-1 과 비교 불가): median ms / frame — `grounded_sam` 449.2 · 445.6, per_frame 281.2 · 277.8, track 147.4 · 147.2(평가 frame 의 step 만), YOLOE 24.7 · 23.8, `localization_heads:threshold` 2.4, `:sam_point` 271.7, TokenCut 27.6 · 29.6.

![SL4b 장면 overlay](figures/baselines/sl4b/v2-sl4b-scene-overlay.png)

### BL-1 열린 항목 4 해소 — ATS 와 Grounded-SAM(instruction)의 wrong-object 일치

SL3 notrack run 에서 ATS 와 Grounded-SAM(instruction)의 wrong-object 3D 가 둘 다 55 / 221(기준 `3d_neighbor_fp > 3d_TP`)이고 변형 기준(neighbour 비율 > 0.5)도 둘 다 46 이다. verifier 가 frame 집합을 대조했다(`scribe_check`).

| 규칙 | ATS | Grounded-SAM(instruction) | 교집합 | ATS 만 | Grounded-SAM 만 | 같은 집합? |
|---|---|---|---|---|---|---|
| `3d_neighbor_fp > 3d_TP` | 55 | 55 | 22 | 33 | 33 | 아니오 |
| neighbour 비율 > 0.5 | 46 | 46 | 14 | 32 | 32 | 아니오 |

두 값이 같은 것은 **우연의 일치**였다(frame 집합이 다르다).

### 실행 시간과 lock 점유 (BL-1 실패 #3 · #4 의 결과)

- SL4b 40 episode 전체 arm run 이 wall 3,660 s(61 분)에 끝났다. BL-1 은 episode 당 약 700 s, 약 8 h 로 추정해 deferred 했다. BL-4 의 측정은 arm 구성(track 포함)이 다르고 thread 설정이 달라(2) G 의 속도 개선의 기여를 분리하지 않았다.
- SL3 track run 은 wall 570 s 다(BL-1 의 notrack run 은 8,080 s, arm 구성이 다름).
- 두 run 에서 provider 예외 격리가 작동할 일이 없었다(`provider_errors` `[]`).

### 실패 · 한계 (verify.json `issues` · `not_measured`)

| # | 내용 | 영향 |
|---|---|---|
| 1 | **track left_wrist 의 `track_lost`**: SL3 221 중 112 frame(7 / 10 episode), SL4b phrase 501 / 873(28 / 40), instruction 540 / 873(30 / 40). provider 는 `redetect_on_loss=false` | track 의 left_wrist 수치는 "잃은 뒤 다시 찾지 않는" 설정의 결과다 |
| 2 | **V2 GPU run 은 thread 2 로 돌았다.** BL-1 run 의 thread 설정과 같은지 확인하지 않았다 | latency 를 BL-1(PAPER-BASELINES-2)과 비교할 때 thread 차이가 섞인다 |
| 3 | **V2 verifier 가 측정 전에 G 의 impl.md "내가 기대하는 결과" 절을 읽었다** | 측정 명령 · 분석 정의는 task · impl 명령 절과 BL-1 정의를 따랐다. 기대가 분석 선택에 영향을 주었을 가능성은 배제하지 못한다 |
| 4 | `check_track_memory --frames 150` 에서 t0_e0 는 기록이 114 frame 뿐이라 bounded 114 frame 만 돌았다 | 가장 긴 SL3 episode(t2_e0, 143 frame)를 head · left_wrist 로 한 번 더 돌렸다 |
| 5 | t2_e0 left_wrist 의 bounded `reserved` 최대 22,022 MiB 는 앞 실행(head unbounded)의 reserved 를 caching allocator 가 유지한 것 | `allocated`(1,047.3 MiB)로 읽는다 |
| 6 | `nvidia-smi memory.used` 는 5 s 표본이라 짧은 최고값을 놓칠 수 있다. 125,000 MiB guard 는 걸리지 않았다 | |
| 7 | track latency 는 평가 frame 의 step 시간만이다(사이 frame 은 행에 시간이 없다) | per_frame 과 같은 척도가 아니다 |
| 8 | SL4b 에 ATS 행이 없다. SL4b `source.patch` 에 다른 session 의 `README.md` diff 가 있다 | harness 파일의 sha256 은 실행 전후 같다 |
| 9 | 알림 대상 `ag3s-vla-benchmark-67` 이 도달 불가였다 | 제안된 두 session 에 알렸다 |
| 10 | `localization_heads` 는 SL3 · SL4b 모두 in-sample 이다 | 이 행은 held-out 이 아니다 |
| 11 | V2 의 `native_metrics` 는 `--no-maps` 로 돌렸다 | 연속 map arm · TokenCut DINO 재계산은 V1 범위 |

### 측정하지 않은 것

| 항목 | 이유 |
|---|---|
| SL4b ATS(`ats_v11ab`) 행 | SL4b 에는 V11AB perception run 이 없다 |
| track 의 평가 frame 사이 frame 시간 | 행에 기록되지 않는다 |
| closed-loop 지표(TSR · CAR · ETS 등) | ~~BL-5~~ → BL-6 |
| `redetect_on_loss=true`(잃은 뒤 재검출) 변형의 track | 측정하지 않았다 |
| BL-1 과 같은 thread 로 latency 재측정 | thread 설정 확인 필요 |

### 열린 항목

1. track 의 left_wrist `track_lost` 가 왜 이렇게 많은가(가려짐 · 시야 이탈로 보이나 이 측정이 분리하지 않았다), `redetect_on_loss` 의 효과.
2. latency 를 BL-1 과 같은 thread 로 다시 재거나 thread 영향을 분리한다.
3. SL4b held-out(test split)에서 localization heads 재측정(attention audit 필요, GPU).
4. 논문 bib 에 Grounded-SAM · Grounded-SAM-2 항목 추가(BL-0 부터 열려 있음).

### 판정

**사용자 판정 대기.**

---

## BL-5 — ATS-CLOUD-DIAG: V11AB `cloud_target` 이 비는 원인 진단과 ATS 표현별 공정 비교 재계산 (2026-10-24)

> 2026-10-24 KST. 진단 H 14:55, 구현 J 15:12, 측정 V4 15:16 이후(run dir 이름의 시각이 KST 다: `20261024T151602` ~ `…T152242`, 마지막 run 의 wall 244 s).
> 원본: [`handoff/ATS-CLOUD-DIAG.task.md`](handoff/ATS-CLOUD-DIAG.task.md) · [`-H.impl.md`](handoff/ATS-CLOUD-DIAG-H.impl.md)(진단) · [`-J.impl.md`](handoff/ATS-CLOUD-DIAG-J.impl.md)(구현) · [`handoff/ATS-CLOUD-DIAG-V4.verify.json`](handoff/ATS-CLOUD-DIAG-V4.verify.json)(측정). raw 출력은 `outputs/verify/ATS-CLOUD-DIAG-V4/`.
> figure: [`figures/baselines/ats_v11ab/`](figures/baselines/ats_v11ab/) 의 `v4-*` 4 개(각각 같은 이름의 `.json` sidecar). 이 기록은 링크만 한다.
> 대상: SL3 dev `libero_spatial` t0–9 e0, 10 episode · 221 policy call(BL-3 과 같은 집합). V11AB cloud = SL4 perception run `20261012T225820_46a08d6ff644_sl4-perception-dev`, segmenter mask = `20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack`, scene GT = `outputs/verify/SL4/raw/scene`.
> **판정은 사용자 판정 대기.** 아래는 수치와 관찰 가능한 사실만 적는다. `cloud_target` 이 비는 원인(H 진단, lead 확인)을 제외하고 측정 결과의 인과 해석은 적지 않는다.

### 이 STEP 이 답하는 질문 (BL-5)

BL-3 결과 C 에서 V11AB perception run 의 `cloud_target` 이 `has_target` 179 call 중 134 call 에서 비었고, 그래서 ATS 행이 45 call 로만 계산됐다.

1. 왜 `cloud_target` 이 비는가? 기록 경로(`perception_replay`)의 문제인가, V11AB 산출물 구조의 문제인가?
2. 공정 비교에 쓸 ATS 표현은 무엇이고, 표현을 바꾸면 ATS 행과 segmenter 와의 paired 비교는 어떻게 달라지는가?
3. V11AB 가 ESDF 에서 target 으로 빼는 영역(exclusion geometry)은 그 시점의 GT target 위치와 얼마나 어긋나는가?

### 경위

| 역할 | 한 일 | 산출물 |
|---|---|---|
| implementer H (진단, code 수정 없음, CPU) | 기록된 npz 와 code 를 읽어 원인을 진단하고 표현 대안을 비교했다. 분석 script 와 진단 그림은 `outputs/ats_cloud_diag/` 에만 두었다 | `ATS-CLOUD-DIAG-H.impl.md` |
| lead (2026-10-24, 사용자 \"진행하세요\" 의 범위) | H 의 진단을 받아 표현 결정(아래)과 J · V4 task 를 냈다 | `ATS-CLOUD-DIAG.task.md` \"후속\" 절 |
| implementer J | `eval/cloud_labeling.py` 에 `--ats-repr {cloud_target,target_points,exclusion}`(기본 `cloud_target`)와 `--ats-tol`(기본 0.005 m)을 추가했다 | `ATS-CLOUD-DIAG-J.impl.md` |
| verifier V4 (CPU, worker 1, `CUDA_VISIBLE_DEVICES=''`) | exclusion exact / 5 mm / 10 mm 와 `cloud_target` 을 돌리고 segmenter arm 과의 paired bootstrap · 범주별 표 · stale geometry 와 GT 의 거리 분포 · figure 4 개를 만들었다 | `ATS-CLOUD-DIAG-V4.verify.json` |

**결정(lead, 사용자 판정 전).** 공정 비교의 ATS 주 표현은 **(b+) exclusion geometry 를 같은 cloud 에 5 mm 로 대응시킨 것**이다. (a) `cloud_target` 은 fresh-only 진단 행으로 남기고, exact · 10 mm 는 sensitivity 행이다. 5 mm 는 V11AB fusion voxel 10 mm(`fusion.fusion_voxel_size_m = 0.01`, H 가 221 / 221 call 에서 확인)의 반 폭이다(H 의 근거). 기본값은 지금과 같아서 기존 수치(BL-3)는 바뀌지 않는다.

### 구현 (J)

| 파일 | 내용 |
|---|---|
| `Baseline/paper_baselines/eval/cloud_labeling.py` | 옵션 `--ats-repr` · `--ats-tol`. `exclusion` 은 has_target call 에서 `target_points`, has_target=False 이고 `manipulated_state` 가 있는 call 에서 episode 안의 직전 has_target call 의 `target_points` 를 이어 붙인 것에서 tol 이내인 cloud 점을 ATS positive 로 센다. carry 한 call 에서 centroid 오차가 1e-5 m 이상이거나 없으면 `ValueError` 로 멈춘다. call 범주를 새 파일 `ats_calls.jsonl` 에 매 run 마다 기록한다. 기본이 아닌 run 은 `calls.jsonl` 의 모든 행에 `ats_call_cat` 를 넣고, summary 에 `ats_by_category` 를 넣는다. 기본 run 은 `calls.jsonl` 이 V1 과 byte 동일해야 하므로 `ats_call_cat` 를 넣지 않는다 |
| `Baseline/paper_baselines/tests/test_cloud_labeling.py` | test 5 개(parametrize 포함 8 case) 추가. J 의 보고: `.venv-baselines` 에서 **15 passed**(기존 7 + 새 8) |

위 두 파일은 **commit 하지 않은 working tree 판**이다(HEAD `3c677553`). V4 가 이 판으로 측정했다(`code_state`, sha256 은 `outputs/verify/ATS-CLOUD-DIAG-V4/code_state_sha256.txt`).

### 원인 — `cloud_target` 이 비는 이유 (H 진단, code 근거)

**결론(H, lead 확인).** 기록 버그가 아니다. V11AB 는 일부 frame 에서 target 을 이번 frame cloud 에서 새로 찾은 cluster 가 아니라 **이전 frame 에 관측한 geometry(stand-in: anchor 또는 occluded)** 로 보고한다. 그 geometry 가 ESDF 의 target 제외에 들어간다. stand-in 의 `point_indices` 는 비어 있다(그 점들은 이번 cloud 의 점이 아니다). `perception_replay` 는 `cloud_target` 을 `point_indices` 로만 세우므로 stand-in call 에서 flag 가 0 개가 되어 `cloud_target` 이 0 이다.

| 점검 | H 가 확인한 것 | 근거(commit `46a08d6f` 기준) |
|---|---|---|
| `--no-clouds` 였나 | 아니다. `save_clouds: true`, npz 에 `cloud_points` · `cloud_target` 이 있다 | `run_manifest.json` |
| 다른 cloud 에 flag 를 달았나 | 아니다. grounding 과 replay 가 같은 `debug[\"filtered_cloud\"]` 를 쓴다 | `pipeline.py:1475` · `:1513` · `:1852`, `perception_replay.py:385` |
| flag 를 세우는 규칙 | `target_flag[target.point_indices] = True` 뿐이다. `target_indices` 도 같은 배열이다 | `perception_replay.py:420-422`, `:404` (HEAD 에서 `:629-631`, `:610`) |
| stand-in 의 `point_indices` | 설계상 비어 있다. docstring: \"point_indices is empty (none of *this* cloud's points are it)\" | `target_grounding.py:1959` (`_anchor_result`), `:1995` (`_unobserved_result`), `:1966-1971` |
| stand-in 으로 가는 분기 | `decision.cluster is None` 이면 `_unobserved_result`(occluded 또는 lost), `decision.reason` 이 `SUBSET_KEPT_ANCHOR` · `UNSPLIT_KEPT_ANCHOR` · `CLOSING_KEPT_GEOMETRY` 이면 `_anchor_result` | `target_grounding.py:1764-1777` |
| ESDF 가 받는 geometry | `manipulated_geometry` 를 받아 그 `target.points` 로 `TARGET_LABEL` 과 target-free layer 를 만든다 | `pipeline.py:1542` · `:1679` · `:2511` · `:2547` · `:2566` |
| has_target=False 인데 제외가 계속되는 경우 | no-target 분기 note: \"contact permission and target exclusion keep its last observed geometry\" | `pipeline.py:1597-1602` |

`파일:줄` 은 H 가 읽은 것이고 V4 가 줄 번호를 다시 확인하지는 않았다. 아래 V4 측정은 이 진단과 맞는 데이터 성질을 확인한다.

**V4 가 확인한 데이터 성질.**

| 확인 | V4 결과 |
|---|---|
| stand-in call 의 geometry frame 이 그 call 자신이 아닌가 | stand-in 134 call 전부 `geometry_frame ≠ self`. fresh 45 call 은 전부 `geometry_frame = self` |
| exclusion geometry 가 geometry frame 의 `target_points` 와 byte 동일한가 | 217 / 217 call(exclusion geometry 가 있는 call = 221 − no_admissible 4) |
| `excluded_no_target` 에서 이어 붙인 직전 has_target call(`carried_from`)이 geometry frame 과 같은가 | 38 call 중 37 call 이 **다르다** |
| exclusion geometry 의 centroid 오차 최대(`checks.centroid_err_max_mm`) | 5e-06 mm |
| GT target 위치 기록(`gt_target_position`)과 scene GT 의 차이 최대 | 0.0 mm |
| V4 범주 대 H `calls_diag` 범주 | 221 call 모두 대응(fresh 45 · anchor 62 · occluded 72 · excluded_no_target 38 · no_admissible 4). `shift_body_mm` 은 H 의 `gt_target_shift_since_geom_mm` 와 217 call 에서 최대 차이 0 mm, geometry frame 불일치 0 |
| `cloud_target` run 의 `calls.jsonl` 이 V1(`…035953_…_baseline-eval-cloud-sl3dev`)과 같은가 | sha256 동일 → 기본 옵션이 BL-3 수치를 바꾸지 않았다 |
| exclusion run 3 개의 mask arm 행 · 범주 수 · re-aggregation | 모든 run 에서 일치(`consistency`) |
| 회귀 기준선(legacy 16D, CPU) | 일치: 위반으로 시작 14 / 15, `has_target` 15 / 15, frame 0 의 `clearance_before` −29.031 mm |

### 실행 (V4, CPU, worker 1)

`.venv-baselines`, `HF_HUB_OFFLINE=1`, `MUJOCO_GL=osmesa`. 각 run 의 명령은 V1 과 같고 `--ats-repr` · `--ats-tol` 만 다르다(`--dump-figure-data --dump-per-episode 2`).

| run (`create_run` 의 run-id) | 옵션 | wall (s) | max RSS (GB) | exit |
|---|---|---|---|---|
| `ats-cloud-exclusion-5mm-sl3dev` (`…T151602_3c6775531841_…`) | `--ats-repr exclusion --ats-tol 0.005` | 145.0 | 0.197 | 0 |
| `ats-cloud-exclusion-exact-sl3dev` (`…T151826_…`) | `--ats-repr exclusion --ats-tol 1e-6` | 134.4 | 0.207 | 0 |
| `ats-cloud-exclusion-10mm-sl3dev` (`…T152040_…`) | `--ats-repr exclusion --ats-tol 0.010` | 121.4 | 0.211 | 0 |
| `ats-cloud-cloudtarget-sl3dev` (`…T152242_…`) | `--ats-repr cloud_target` | 244.2 | 0.336 | 0 |

run dir 은 `AG3S_VLA-storage/results/benchmark/` 아래다. 같은 pod 에서 다른 session 의 GPU server · rollout · pytest 가 돌고 있었으나 이 단계는 CPU 결정적 계산이고 기준선은 일치했다(`issues`).

### 결과 A — call 범주와 ATS 표현별 positive 수

V4 `category_counts`: **fresh 45 · anchor 62 · occluded 72 · excluded_no_target 38 · no_admissible 4**(합 221). 범주 정의는 용어 절의 \"call 범주\" 를 본다. `cloud_target` 의 positive 는 fresh 45 call 에만 있고 나머지 4 범주에는 0 이다. exclusion 5 mm 는 fresh 45 · anchor 62 · occluded 55 · excluded_no_target 15 call 에 positive 가 있다(아래 표 B-1).

![call 범주 시계열과 거리](figures/baselines/ats_v11ab/v4-ats-call-category-series.png)

(위 figure: episode 별 call 범주(색 띠), exclusion geometry 와 GT target 의 거리 · GT target 이동(선), wrong-object call 표시(▼), 하단은 범주별 거리 분포.)

**표 B-1. 범주별 ATS 지표 — exclusion 5 mm(주 표현)와 `cloud_target`** (episode 평균, P · neighbour 는 positive 가 있는 call 만, pooled 는 call 전체를 합친 값)

| 범주 | call (episode) | ATS found (exclusion 5 mm) | IoU | P (n call) | R | neighbour | pooled P | pooled R | pooled neighbour | `cloud_target` found |
|---|---|---|---|---|---|---|---|---|---|---|
| fresh | 45 (10) | 45 / 45 | 0.504 | 0.692 (45) | 0.515 | 0.302 | 0.536 | 0.415 | 0.460 | 45 / 45 |
| anchor | 62 (10) | 62 / 62 | 0.357 | 0.676 (62) | 0.359 | 0.324 | 0.864 | 0.525 | 0.136 | 0 / 62 |
| occluded | 72 (9) | 55 / 72 | 0.131 | 0.609 (55) | 0.132 | 0.374 | 0.297 | 0.113 | 0.641 | 0 / 72 |
| excluded_no_target | 38 (9) | 15 / 38 | 0.005 | 0.400 (15) | 0.005 | 0.566 | 0.027 | 0.006 | 0.826 | 0 / 38 |
| no_admissible | 4 (1) | 0 / 4 | 0.000 | — (0) | 0.000 | — | — | 0.000 | — | 0 / 4 |

**표 B-2. 범주별 IoU 의 ATS 표현 간 변화** (found / call · IoU)

| 범주 | exact found / IoU | 5 mm found / IoU | 10 mm found / IoU | `cloud_target` found / IoU |
|---|---|---|---|---|
| fresh | 45 / 45 · 0.495 | 45 / 45 · 0.504 | 45 / 45 · 0.551 | 45 / 45 · 0.495 |
| anchor | 56 / 62 · 0.225 | 62 / 62 · 0.357 | 62 / 62 · 0.484 | 0 / 62 · 0.000 |
| occluded | 50 / 72 · 0.085 | 55 / 72 · 0.131 | 64 / 72 · 0.181 | 0 / 72 · 0.000 |
| excluded_no_target | 14 / 38 · 0.000 | 15 / 38 · 0.005 | 31 / 38 · 0.019 | 0 / 38 · 0.000 |
| no_admissible | 0 / 4 · 0.000 | 0 / 4 · 0.000 | 0 / 4 · 0.000 | 0 / 4 · 0.000 |

**표 B-3. 범주별 segmenter 와의 paired 비교** (같은 cloud 점 · 같은 GT. Δ IoU = arm − ATS(exclusion 5 mm), 95 % CI, (better / worse) = diff 가 양 / 음인 episode 수. 범주별 paired 는 그 범주 call 이 있는 episode 만 쓴다: occluded 9 · excluded_no_target 9 · no_admissible 1 episode)

| 범주 | call (ep) | ATS IoU | GSAM (instruction) IoU · Δ IoU [CI] (b / w) | GSAM-2 per_frame (instruction) IoU · Δ IoU [CI] (b / w) | GT oracle IoU · Δ IoU [CI] (b / w) | YOLOE (phrase) IoU · Δ IoU [CI] (b / w) |
|---|---|---|---|---|---|---|
| fresh | 45 (10) | 0.504 | 0.507 · +0.003 [-0.222, +0.220] (5 / 5) | 0.516 · +0.012 [-0.244, +0.251] (5 / 5) | 0.868 · +0.364 [+0.188, +0.561] (8 / 2) | 0.335 · -0.168 [-0.377, +0.023] (3 / 7) |
| anchor | 62 (10) | 0.357 | 0.458 · +0.101 [-0.068, +0.262] (6 / 4) | 0.449 · +0.092 [-0.082, +0.257] (7 / 3) | 0.846 · +0.489 [+0.319, +0.666] (10 / 0) | 0.251 · -0.106 [-0.284, +0.107] (2 / 7) |
| occluded | 72 (9) | 0.131 | 0.544 · +0.413 [+0.229, +0.574] (8 / 1) | 0.534 · +0.404 [+0.223, +0.568] (8 / 1) | 0.862 · +0.731 [+0.640, +0.810] (9 / 0) | 0.283 · +0.152 [-0.025, +0.308] (7 / 2) |
| excluded_no_target | 38 (9) | 0.005 | 0.523 · +0.519 [+0.303, +0.730] (8 / 0) | 0.476 · +0.471 [+0.287, +0.646] (8 / 0) | 0.825 · +0.821 [+0.798, +0.848] (9 / 0) | 0.279 · +0.274 [+0.168, +0.382] (8 / 0) |
| no_admissible | 4 (1) | 0.000 | 0.723 · +0.723 [+0.723, +0.723] (1 / 0) | 0.719 · +0.719 [+0.719, +0.719] (1 / 0) | 0.943 · +0.943 [+0.943, +0.943] (1 / 0) | 0.578 · +0.578 [+0.578, +0.578] (1 / 0) |

![범주별 표](figures/baselines/ats_v11ab/v4-ats-category-table.png)

관찰:
- `cloud_target` 은 anchor 62 · occluded 72 · excluded_no_target 38 · no_admissible 4 call 에서 positive 가 0 개다.
- exclusion 5 mm 의 IoU(episode 평균)는 fresh 0.504 · anchor 0.357 · occluded 0.131 · excluded_no_target 0.005 다. pooled precision 은 anchor 0.864 · occluded 0.297 · excluded_no_target 0.027, pooled neighbour 비율은 anchor 0.136 · occluded 0.641 · excluded_no_target 0.826 다.
- fresh 범주에서 Grounded-SAM · Grounded-SAM-2(instruction)의 Δ IoU 는 +0.003 · +0.012 이고 CI 가 0 을 포함한다(5 / 5). occluded · excluded_no_target 범주에서는 CI 가 0 위다(Grounded-SAM-2 instruction +0.404 · +0.471).
- Grounded-SAM · Grounded-SAM-2(instruction) · GT oracle 의 Δ IoU 는 fresh < anchor < occluded < excluded_no_target 순으로 커진다(예: Grounded-SAM instruction +0.003 · +0.101 · +0.413 · +0.519).

### 결과 B — ATS 표현 × 지표

**표 C. 같은 cloud 점 위의 ATS 표현별 지표** (221 call, 10 episode, episode 평균. 아래 3 행은 같은 점의 segmenter 행이다)

| ATS 표현 | positive 가 있는 call / 221 | IoU | P (n call) | R | neighbour 비율 | pooled IoU / P / R | pooled neighbour |
|---|---|---|---|---|---|---|---|
| exclusion 5 mm (**주 표현**) | 177 / 221 | 0.259 | 0.633 (177) | 0.262 | 0.353 | 0.224 / 0.551 / 0.273 | 0.421 |
| exclusion exact (1e-6 m, sensitivity) | 165 / 221 | 0.190 | 0.620 (165) | 0.192 | 0.367 | 0.169 / 0.499 / 0.204 | 0.472 |
| exclusion 10 mm (sensitivity) | 202 / 221 | 0.324 | 0.524 (202) | 0.333 | 0.327 | 0.273 / 0.572 / 0.343 | 0.375 |
| `cloud_target` (fresh-only 진단) | 45 / 221 | 0.082 | 0.693 (45) | 0.084 | 0.301 | 0.082 / 0.534 / 0.088 | 0.462 |
| (참고) ATS attention seed | 210 / 221 | 0.207 | 0.569 (210) | 0.232 | 0.379 | 0.205 / 0.567 / 0.243 | 0.305 |
| Grounded-SAM-2 per_frame (instruction) (같은 점) | 219 / 221 | 0.486 | 0.593 (219) | 0.655 | 0.331 | 0.432 / 0.550 / 0.668 | 0.378 |
| Grounded-SAM (instruction) (같은 점) | 221 / 221 | 0.493 | 0.554 (221) | 0.684 | 0.384 | 0.423 / 0.519 / 0.694 | 0.404 |
| GT oracle (같은 점) | 221 / 221 | 0.850 | 0.893 (221) | 0.948 | 0.018 | 0.850 / 0.891 / 0.948 | 0.017 |

![ATS 표현 × 지표 표](figures/baselines/ats_v11ab/v4-ats-repr-table.png)

관찰:
- ATS 의 IoU(episode 평균)는 표현에 따라 0.082(`cloud_target`) → 0.190(exclusion exact) → 0.259(5 mm) → 0.324(10 mm) 로 움직인다. positive 가 있는 call 은 45 → 165 → 177 → 202 / 221 이다.
- `cloud_target` run 의 ATS 행은 BL-3 표 C-1 의 ATS `cloud_target` 행(45 / 221, IoU 0.082, P 0.693, R 0.084)과 같다.
- ATS 의 P 는 `cloud_target` 에서 45 call, 5 mm 에서 177 call 로 계산된 값이다. positive 가 있는 call 집합이 달라 두 값을 같은 척도로 읽을 수 없다(BL-3 결과 C 의 지적과 같은 성질).
- `(b)` `target_points` 표현의 run(`--ats-repr target_points`)은 돌리지 않았다. BL-3 의 verifier 보충 행(좌표 일치 151 / 221 · 5 mm 162 / 221)이 그 표현의 이전 측정이다.

### 결과 C — segmenter 와의 paired bootstrap (12 arm)

방법은 BL-3 과 같다: episode 단위 2,000 회 resample, seed 20261015, 95 % percentile. 기준(ATS) 표현만 바꿨다. `arm − ATS` 이므로 Δ > 0 은 arm 이 높다는 뜻이다. `cloud_target` 열은 BL-3 표 C-2 의 첫 열과 같은 값이다(V4 `consistency`: `cloud_target` run 의 `calls.jsonl` 이 V1 과 byte 동일).

**표 D-1. Δ IoU — ATS 표현 4 가지** (95 % CI, (better / worse))

| arm − ATS | Δ IoU, ATS = exclusion 5 mm (**주**) | exclusion exact | exclusion 10 mm | `cloud_target` (fresh-only) |
|---|---|---|---|---|
| Grounded-SAM-2 per_frame (instruction) | +0.227 [+0.090, +0.368] (7 / 3) | +0.296 [+0.168, +0.421] (9 / 1) | +0.161 [+0.012, +0.318] (7 / 3) | +0.404 [+0.281, +0.519] (10 / 0) |
| Grounded-SAM-2 per_frame (phrase) | +0.192 [+0.064, +0.340] (7 / 3) | +0.261 [+0.143, +0.393] (9 / 1) | +0.127 [-0.016, +0.291] (6 / 4) | +0.369 [+0.253, +0.486] (10 / 0) |
| Grounded-SAM (instruction) | +0.235 [+0.097, +0.370] (8 / 2) | +0.303 [+0.174, +0.427] (9 / 1) | +0.169 [+0.023, +0.318] (7 / 3) | +0.411 [+0.282, +0.533] (10 / 0) |
| Grounded-SAM (phrase) | +0.200 [+0.067, +0.361] (8 / 2) | +0.269 [+0.144, +0.411] (9 / 1) | +0.135 [-0.015, +0.311] (6 / 4) | +0.377 [+0.247, +0.504] (10 / 0) |
| GT oracle (instruction) | +0.592 [+0.484, +0.704] (10 / 0) | +0.660 [+0.576, +0.744] (10 / 0) | +0.526 [+0.399, +0.661] (10 / 0) | +0.768 [+0.719, +0.813] (10 / 0) |
| GT oracle (phrase) | +0.592 [+0.484, +0.704] (10 / 0) | +0.660 [+0.576, +0.744] (10 / 0) | +0.526 [+0.399, +0.661] (10 / 0) | +0.768 [+0.719, +0.813] (10 / 0) |
| localization_heads:sam_point (in-sample) | +0.007 [-0.095, +0.114] (5 / 5) | +0.076 [+0.004, +0.150] (6 / 4) | -0.059 [-0.187, +0.074] (4 / 6) | +0.183 [+0.150, +0.222] (10 / 0) |
| localization_heads:threshold (in-sample) | -0.155 [-0.264, -0.042] (3 / 7) | -0.086 [-0.169, -0.002] (3 / 7) | -0.220 [-0.350, -0.082] (3 / 7) | +0.022 [-0.027, +0.066] (7 / 3) |
| TokenCut attention_overlap | -0.184 [-0.282, -0.082] (2 / 8) | -0.115 [-0.187, -0.044] (3 / 7) | -0.250 [-0.370, -0.124] (2 / 8) | -0.007 [-0.046, +0.028] (6 / 4) |
| TokenCut foreground | -0.047 [-0.180, +0.084] (4 / 6) | +0.022 [-0.086, +0.122] (6 / 4) | -0.113 [-0.260, +0.044] (3 / 7) | +0.130 [+0.058, +0.193] (8 / 2) |
| YOLOE-11L (instruction) | -0.158 [-0.263, -0.052] (3 / 7) | -0.089 [-0.181, +0.001] (4 / 6) | -0.223 [-0.340, -0.105] (3 / 7) | +0.019 [-0.065, +0.110] (6 / 4) |
| YOLOE-11L (phrase) | +0.026 [-0.093, +0.149] (5 / 5) | +0.095 [+0.002, +0.191] (7 / 3) | -0.040 [-0.175, +0.110] (5 / 5) | +0.203 [+0.121, +0.276] (9 / 1) |
| (참고) ATS `cloud_target` 행 | -0.176 [-0.262, -0.091] (0 / 8) | -0.108 [-0.159, -0.055] (0 / 8) | -0.242 [-0.352, -0.132] (0 / 8) | — |
| (참고) ATS attention seed 행 | -0.052 [-0.147, +0.054] (3 / 7) | +0.017 [-0.050, +0.093] (4 / 6) | -0.117 [-0.235, +0.011] (3 / 7) | +0.125 [+0.083, +0.163] (9 / 1) |

**표 D-2. 주 표현(exclusion 5 mm) 기준 Δ precision · recall · neighbour 비율**

| arm − ATS (exclusion 5 mm) | Δ precision | Δ recall | Δ neighbour 비율 |
|---|---|---|---|
| Grounded-SAM-2 per_frame (instruction) | -0.040 [-0.264, +0.212] (3 / 7) | +0.393 [+0.278, +0.520] (10 / 0) | -0.022 [-0.254, +0.195] (5 / 5) |
| Grounded-SAM-2 per_frame (phrase) | -0.088 [-0.308, +0.169] (3 / 7) | +0.381 [+0.267, +0.514] (10 / 0) | +0.029 [-0.212, +0.240] (7 / 3) |
| Grounded-SAM (instruction) | -0.079 [-0.288, +0.154] (3 / 7) | +0.422 [+0.308, +0.544] (10 / 0) | +0.032 [-0.177, +0.230] (7 / 3) |
| Grounded-SAM (phrase) | -0.105 [-0.324, +0.155] (3 / 7) | +0.399 [+0.275, +0.545] (10 / 0) | +0.060 [-0.183, +0.263] (7 / 3) |
| GT oracle (instruction) | +0.260 [+0.029, +0.522] (6 / 4) | +0.686 [+0.575, +0.801] (10 / 0) | -0.335 [-0.585, -0.119] (4 / 6) |
| GT oracle (phrase) | +0.260 [+0.029, +0.522] (6 / 4) | +0.686 [+0.575, +0.801] (10 / 0) | -0.335 [-0.585, -0.119] (4 / 6) |
| localization_heads:sam_point (in-sample) | -0.247 [-0.472, -0.000] (3 / 7) | +0.138 [+0.036, +0.243] (8 / 2) | -0.145 [-0.354, +0.042] (4 / 6) |
| localization_heads:threshold (in-sample) | -0.528 [-0.756, -0.263] (3 / 7) | +0.604 [+0.510, +0.704] (10 / 0) | -0.235 [-0.492, -0.015] (4 / 6) |
| TokenCut attention_overlap | -0.555 [-0.775, -0.304] (2 / 8) | +0.282 [+0.179, +0.382] (9 / 1) | -0.186 [-0.475, +0.059] (4 / 6) |
| TokenCut foreground | -0.413 [-0.649, -0.141] (3 / 7) | +0.475 [+0.321, +0.621] (10 / 0) | -0.158 [-0.415, +0.064] (6 / 4) |
| YOLOE-11L (instruction) | -0.263 [-0.464, -0.079] (3 / 7) | -0.133 [-0.248, -0.014] (3 / 7) | -0.011 [-0.355, +0.316] (6 / 4) |
| YOLOE-11L (phrase) | -0.152 [-0.410, +0.138] (3 / 7) | +0.144 [+0.024, +0.260] (7 / 3) | +0.091 [-0.187, +0.344] (7 / 3) |
| (참고) ATS `cloud_target` 행 | +0.060 [+0.005, +0.138] (4 / 1) | -0.178 [-0.263, -0.092] (0 / 8) | -0.051 [-0.135, +0.009] (2 / 4) |
| (참고) ATS attention seed 행 | -0.064 [-0.253, +0.134] (4 / 6) | -0.029 [-0.136, +0.088] (4 / 6) | +0.026 [-0.186, +0.214] (6 / 4) |

관찰:
- 주 표현(5 mm)에서 Grounded-SAM · Grounded-SAM-2 4 arm 과 GT oracle 2 arm 의 Δ IoU 는 CI 가 0 위다. Grounded-SAM(instruction)의 Δ IoU 는 `cloud_target` +0.411 → exact +0.303 → 5 mm +0.235 → 10 mm +0.169 로 줄고, 10 mm 에서 Grounded-SAM(phrase) · Grounded-SAM-2(phrase)의 CI 는 0 을 포함한다.
- 주 표현(5 mm)에서 CI 가 0 아래인 arm 은 `localization_heads:threshold` · `TokenCut attention_overlap` · `YOLOE-11L (instruction)` 이다. CI 가 0 을 포함하는 arm 은 `localization_heads:sam_point` · `TokenCut foreground` · `YOLOE-11L (phrase)` 다. `cloud_target` 기준에서는 같은 6 arm 중 `localization_heads:threshold`(+0.022) · `TokenCut attention_overlap`(−0.007) · `YOLOE-11L (instruction)`(+0.019)의 CI 가 0 을 포함하고, 나머지 3 arm(+0.183 · +0.130 · +0.203)은 CI 가 0 위였다.
- 주 표현(5 mm)에서 Grounded-SAM 계열 4 arm 의 Δ recall 은 +0.381 ~ +0.422 로 CI 가 0 위다. Δ precision · Δ neighbour 비율은 CI 가 0 을 포함한다. GT oracle 은 Δ precision +0.260, Δ neighbour −0.335 로 CI 가 0 을 포함하지 않는다.
- 주 표현 기준 ATS `cloud_target` 행은 ATS exclusion 5 mm 행보다 IoU 가 −0.176 [−0.262, −0.091] 낮다(0 / 8). attention seed 참고 행은 −0.052 [−0.147, +0.054] 다.
- BL-3 표 C-2 의 셋째 열(`target_points` 5 mm 기준)과 이 표의 5 mm 열은 ATS 정의가 다르다(has_target call 만 대 이어 붙인 geometry 포함). 예: Grounded-SAM(instruction) +0.236 대 +0.235.

### 결과 D — stale geometry 와 GT 의 거리

V11AB 가 ESDF 에서 target 으로 빼는 영역(exclusion geometry)을 그 call 시점의 GT target 과 비교했다. 거리 지표의 정의는 용어 절의 `d_surfc_mm` 항목을 본다. 점 geometry 만 쟀고 ESDF voxel footprint 는 재지 않았다(`not_measured`). `shift_body_mm` 는 BDDL target 의 이동이다. fresh 는 geometry frame 이 그 call 자신이므로 `shift_body_mm` = 0 이다.

![stale geometry 와 GT target 의 장면](figures/baselines/ats_v11ab/v4-ats-stale-geometry-scene.png)

(위 figure: 위부터 anchor `t2_e0` call 15 — geometry frame call 0 · GT 이동 0 mm · `d_surfc` 5 mm, occluded `t7_e0` call 22 — geometry frame call 3 · GT 이동 471 mm · `d_surfc` 464 mm, excluded_no_target `t9_e0` call 19 — geometry frame call 0 · wrong-object · GT 이동 420 mm · `d_surfc` 458 mm. 열: head camera 투영, 위에서 본 base frame, 옆에서 본 base frame. 주황 점은 exclusion geometry, 노랑 = geometry frame 시점의 GT target, 파랑 = 이번 call 의 GT target.)

**표 E-1. 범주별 거리 분포** (단위 mm. 전 call)

| 범주 | call | `d_surfc_mm` median / p95 / max | `pt_med_mm` median / p95 | `frac_10` median / mean | `shift_body_mm` median / p95 / max | geometry 이후 call 수 median / max |
|---|---|---|---|---|---|---|
| fresh | 45 | 51.4 / 404.9 / 411.2 | 0.9 / 373.0 | 0.82 / 0.52 | 0.0 / 0.0 / 0.0 | 0 / 0 |
| anchor | 62 | 11.6 / 91.9 / 301.3 | 0.6 / 73.3 | 1.00 / 0.82 | 0.0 / 61.0 / 141.2 | 4 / 17 |
| occluded | 72 | 127.3 / 372.6 / 464.1 | 92.6 / 339.5 | 0.00 / 0.31 | 15.5 / 319.3 / 471.3 | 5 / 26 |
| excluded_no_target | 38 | 211.3 / 440.4 / 458.0 | 180.2 / 394.4 | 0.00 / 0.04 | 166.8 / 415.7 / 445.4 | 12 / 23 |
| no_admissible | 4 | (exclusion geometry 없음) | | | | |

**표 E-2. right-object 대 wrong-object 로 나눈 분포** (wrong-object = geometry frame 에서 exclusion 점 중 BDDL target 으로 라벨된 비율 < 50 %. V4 정의)

| 범주 | call | `d_surfc_mm` median / p95 / max | `pt_med_mm` median / p95 | `frac_10` median / mean | `shift_body_mm` median / p95 / max | geometry 이후 call 수 median / max |
|---|---|---|---|---|---|---|
| fresh · right-object | 24 | 11.4 / 50.4 / 67.1 | 0.5 / 0.9 | 1.00 / 0.97 | 0.0 / 0.0 / 0.0 | 0 / 0 |
| anchor · right-object | 55 | 11.5 / 19.6 / 35.3 | 0.5 / 9.4 | 1.00 / 0.93 | 0.0 / 16.1 / 44.6 | 5 / 17 |
| occluded · right-object | 53 | 67.1 / 320.8 / 464.1 | 17.0 / 275.3 | 0.29 / 0.42 | 35.6 / 327.5 / 471.3 | 6 / 26 |
| excluded_no_target · right-object | 29 | 159.6 / 393.5 / 439.4 | 130.1 / 353.3 | 0.00 / 0.05 | 165.4 / 398.7 / 445.4 | 12 / 23 |
| fresh · wrong-object | 21 | 381.9 / 409.7 / 411.2 | 345.6 / 382.1 | 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | 0 / 0 |
| anchor · wrong-object | 7 | 92.4 / 255.8 / 301.3 | 73.8 / 200.4 | 0.00 / 0.00 | 61.5 / 132.6 / 141.2 | 1 / 3 |
| occluded · wrong-object | 19 | 356.6 / 379.2 / 439.1 | 319.4 / 345.7 | 0.00 / 0.00 | 0.0 / 126.4 / 313.5 | 4 / 17 |
| excluded_no_target · wrong-object | 9 | 402.8 / 453.2 / 458.0 | 371.7 / 403.8 | 0.00 / 0.00 | 168.3 / 398.3 / 420.5 | 14 / 19 |

**표 E-3. geometry frame 이후 GT target 의 이동 구간별 call 수**

| 범주 | 0–1 mm | 1–5 mm | 5–10 mm | 10–50 mm | ≥ 50 mm | 합 |
|---|---|---|---|---|---|---|
| fresh | 45 | 0 | 0 | 0 | 0 | 45 |
| anchor | 44 | 2 | 4 | 7 | 5 | 62 |
| occluded | 31 | 1 | 2 | 10 | 28 | 72 |
| excluded_no_target | 0 | 0 | 0 | 3 | 35 | 38 |

**표 E-4. wrong-object call 에서 ATS 가 잡은 instance 기준 거리** (V4: majority instance 1 개 기준)

| 범주 (wrong-object call) | call | ATS 가 쥔 instance 의 이동 `ats_object_shift_surfc_mm` median / max | 그 instance 표면까지 `d_surfc_mm` median / max | `frac_10` median / mean |
|---|---|---|---|---|
| fresh | 21 | 0.0000 / 0.0000 | 5.2 / 8.8 | 1.00 / 0.99 |
| anchor | 7 | 0.0000 / 0.0003 | 5.3 / 8.8 | 1.00 / 0.99 |
| occluded | 19 | 0.0010 / 0.0010 | 6.3 / 8.8 | 0.91 / 0.95 |
| excluded_no_target | 9 | 0.0010 / 0.0010 | 6.3 / 6.3 | 0.91 / 0.92 |

**표 E-5. wrong-object call 의 episode 분포** (합 56 call. instance 는 t4 · t6 의 `plate_1` 23 call, t9 의 `akita_black_bowl_2` 20 + `plate_1` 3 call, t5 의 `akita_black_bowl_2` 10 call)

| episode | wrong-object call: fresh | anchor | occluded | excluded_no_target | 합 |
|---|---|---|---|---|---|
| libero_spatial_t4_e0 | 16 | 3 | 0 | 1 | 20 |
| libero_spatial_t5_e0 | 2 | 2 | 6 | 0 | 10 |
| libero_spatial_t6_e0 | 1 | 1 | 1 | 0 | 3 |
| libero_spatial_t9_e0 | 2 | 1 | 12 | 8 | 23 |

관찰(수치에서 읽히는 것만):
- **occluded 72 call**: GT target 이 geometry frame 이후 10 mm 넘게 움직인 call 이 38 개(10–50 mm 10 + ≥ 50 mm 28)이고, 이동 median 15.5 mm · p95 319.3 mm · max 471.3 mm 다. exclusion geometry 와 GT surface 거리의 median 은 127.3 mm(`d_surfc_mm`)이고 `frac_10` median 은 0.00 이다.
- **excluded_no_target 38 call**: 이동이 10 mm 넘는 call 이 38 / 38(10–50 mm 3 + ≥ 50 mm 35)이다. 이동 median 166.8 mm · p95 415.7 mm · max 445.4 mm, `d_surfc_mm` median 211.3 mm, `frac_10` median 0.00(mean 0.04)이다. geometry frame 이후 call 수의 median 은 12(max 23)다.
- **anchor 62 call**: `d_surfc_mm` median 11.6 mm, `frac_10` median 1.00, 이동 median 0.0 mm · p95 61.0 mm · max 141.2 mm. 이동이 10 mm 넘는 call 은 12 개(10–50 mm 7 + ≥ 50 mm 5)다.
- **right-object 만 보면**: occluded 53 call 의 이동 median 35.6 mm · p95 327.5 mm · max 471.3 mm, `d_surfc_mm` median 67.1 mm. excluded_no_target 29 call 의 이동 median 165.4 mm · p95 398.7 mm · max 445.4 mm, `d_surfc_mm` median 159.6 mm.
- **wrong-object 56 call**(fresh 21 · anchor 7 · occluded 19 · excluded_no_target 9; episode t4 20 · t5 10 · t6 3 · t9 23): exclusion 점이 BDDL target surface 10 mm 안에 든 비율(`frac_10`)이 전 범주에서 median 0.00 이다. ATS 가 잡은 instance 기준으로는 `d_surfc_mm` median 5.2 ~ 6.3 mm, 그 instance 의 이동은 최대 0.0010 mm 다(표 E-4).
- fresh 45 call 에서 `d_surfc_mm` 의 max 411.2 mm 는 wrong-object 21 call 의 max 와 같다(wrong-object median 381.9 mm). fresh right-object 24 call 은 median 11.4 mm · max 67.1 mm 다.
- geometry frame 이후 call 수: occluded 는 median 5 · max 26, excluded_no_target 은 median 12 · max 23 이다(call 간격은 obs 5 개: `obs_since` median 25 · 60).

### 이 STEP 에서 알게 된 것 (평이한 설명)

1. `cloud_target` 이 비는 call 은 V11AB 가 target 을 **이번 cloud 에서 새로 찾지 못하고 이전 geometry 를 유지한** call 이다. 그 geometry 는 ESDF 에는 target 으로 들어간다(exclusion geometry 가 geometry frame 의 `target_points` 와 217 / 217 call 에서 byte 동일). 그래서 \"비어 있음\" 은 기록 누락이 아니라 ATS 의 출력 구조다.
2. 같은 점 위에서 ATS 를 어떻게 세우느냐에 따라 ATS 의 IoU 가 0.082 ~ 0.324 로 움직인다. segmenter 와의 paired Δ IoU 의 부호는 Grounded-SAM 계열 · GT oracle 에서 유지되고 크기는 줄었다. 다른 arm 에서는 CI 의 위치가 표현에 따라 바뀐다(결과 C).
3. occluded · excluded_no_target call 에서 exclusion geometry 와 그 시점 GT target 은 크게 벗어나 있다(결과 D). wrong-object call 에서는 geometry 가 BDDL target 이 아닌 다른 instance 위에 있다.

### 실패 · 한계 (verify.json `issues` · `not_measured`)

| # | 내용 | 영향 |
|---|---|---|
| 1 | **V4 가 J impl.md 의 \"내가 기대하는 결과\" 절을 측정 전에 읽었다**(파일 전체를 한 번에 읽음). 수치는 run 산출물에서 직접 집계했다 | 기대가 분석 선택에 영향을 주었을 가능성은 배제하지 못한다 |
| 2 | **측정 code 는 commit 되지 않은 working tree 판이다**(`eval/cloud_labeling.py`, J 변경, mtime 15:02 KST, V4 run 보다 앞섬) | 재현에는 같은 working tree 가 필요하다. sha256 은 `code_state_sha256.txt` |
| 3 | excluded_no_target 38 call 중 37 call 에서 `carried_from ≠ geometry_frame`. 거리 계산의 geometry frame 은 frames.jsonl `manipulated.anchor.geometry_frame`(217 / 217 존재)을 썼다 | `carried_from`(직전 has_target call)과 geometry frame 은 다른 개념이다 |
| 4 | fresh 45 call 중 21 call 이 wrong-object 이다. 그래서 fresh 거리 분포의 p95 · max 는 이 call 들이다. right / wrong 으로 나눈 분포를 같이 냈다 | 범주 전체의 거리 요약만 읽으면 wrong-object 가 섞여 있다 |
| 5 | wrong-object 정의는 V4 의 것이다(geometry frame 시점 exclusion 점의 nearest-instance label, 10 mm, 50 %) | call 수 fresh 21 · anchor 7 · occluded 19 · excluded_no_target 9 는 이 정의의 값이다 |
| 6 | 범주별 paired bootstrap 은 그 범주 call 이 있는 episode 만 쓴다(occluded 9 · excluded_no_target 9 · no_admissible 1 episode) | no_admissible 의 CI 는 episode 1 개라 폭이 없다 |
| 7 | `gt_oracle|instruction` 과 `gt_oracle|phrase` 는 모든 지표가 같다 | 표에 두 행을 모두 적었다 |
| 8 | 같은 pod 에 다른 session 의 GPU server · rollout · pytest 가 돌고 있었다 | 이 단계는 CPU 결정적 계산이고 기준선 수치는 일치했다 |

### 측정하지 않은 것 (`not_measured`)

| 항목 | 상태 |
|---|---|
| (b) `target_points` 표현 run(`--ats-repr target_points`, J 의 선택 명령) | 돌리지 않았다 |
| ESDF / target-free layer 의 voxel footprint 와 GT 의 차이 | 점 geometry 만 쟀다 |
| stale geometry 가 closed-loop 결과(충돌 · 성공)에 주는 영향 | 재지 않았다 |
| wrong-object call 에서 ATS 물체 기준 거리 | majority instance 1 개로만 계산했다 |
| no_admissible 4 call 의 거리 | exclusion geometry 가 없어 거리 항목이 없다 |
| H 의 예비 수치 중 V4 가 다시 재지 않은 것(`target_points` 대응 실패 28 / 17 call 의 범주 · 이동 구간, camera mask 별 exact 일치 점 수, 범주별 `target_points` 대응률 등) | 이 기록에 옮기지 않았다 |

### 열린 항목

1. ATS 주 표현(exclusion 5 mm)의 사용자 확정. 현재는 lead 결정이다.
2. 이 결과를 논문의 ATS 행(Tab. 2)에 어느 표현으로 쓸지 — 사용자 판정.
3. J 의 변경(`eval/cloud_labeling.py` · test)을 commit 할지 — 사용자 판정 후.
4. exclusion geometry 와 GT 의 어긋남이 ESDF voxel footprint 와 closed-loop 결과에 미치는 영향 — 측정하려면 verifier 에게 요청한다.
5. `target_points` 표현 run 의 필요 여부.
6. H 가 제안한 다음 perception run 용 기록 옵션(`record_ats_geometry`, 기본 off, `perception_replay.py`)의 도입 여부 — 현재는 필요하지 않다(H).

### 판정

**사용자 판정 대기.**

---

## BL-6 — CLOSED-LOOP-METHODS: methods 를 SafeLIBERO closed loop 에 연결 + GPU smoke (2026-10-24)

> 2026-10-24 KST. 구현 implementer I(commit `16d0eea1`), 검증 verifier V3 15:57–16:38 KST(GPU lock 점유 15:59:17–16:38:08, 아래 "실행"). 이 STEP 의 범위는 **구현과 GPU smoke(arm 당 1 episode)** 이고, **pilot(20 episode × 8 arm)은 돌리지 않았다.**
> 원본: [`handoff/CLOSED-LOOP-METHODS.task.md`](handoff/CLOSED-LOOP-METHODS.task.md) · [`-I.impl.md`](handoff/CLOSED-LOOP-METHODS-I.impl.md)(구현) · [`handoff/CLOSED-LOOP-METHODS-V3.verify.json`](handoff/CLOSED-LOOP-METHODS-V3.verify.json)(측정) · `Baseline/paper_baselines/closed_loop/README.md`. raw 출력은 `outputs/verify/CLOSED-LOOP-METHODS-V3/` 와 `AG3S_VLA-storage/results/benchmark/20261024T16*_16d0eea18e44_clm-v3smoke-LI-*/`.
> figure: [`figures/baselines/closed_loop/`](figures/baselines/closed_loop/) 의 `v3smoke-*`(각각 같은 이름의 `.json` sidecar). 이 기록은 링크만 한다.
> 대상: SafeLIBERO spatial Level I dev, task 0 init 0(`t0e0`), seed 7, `policy_seed` 700000, arm 당 1 episode, budget 280 policy step(wait 20 step 을 합치면 300 env step). 기준 비교 run = SL5c LI VLA-only(`20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only`).
> **판정은 사용자 판정 대기.** 인과 해석은 적지 않는다. **1 episode 라서 이 STEP 의 어떤 수치도 method 간 성능 비교가 아니다.**

### 이 STEP 이 답하는 질문 (BL-6)

BL-3 까지 methods 9 arm 은 open-loop 대리 지표(개입률 · ‖Δa‖ · scene-GT clearance)만 있었다. 논문이 요구하는 SafeLIBERO 지표(TSR · CAR · ETS)와 KNOWS 지표(SR · CR · SSR)는 rollout 이 있어야 잰다.

1. 기존 client · runner · openpi 를 고치지 않고 methods 8 arm(`none` 1 + safety filter 3 + sampling guidance 4)을 SafeLIBERO closed loop 에 붙일 수 있는가?
2. `none` arm 이 기존 VLA-only server 와 같은 chunk 를 내는가(붙인 경로가 VLA-only 결과를 바꾸지 않는가)?
3. 8 arm 이 실물 pi0.5 위에서 1 episode 를 끝까지 도는가, 지표 · 지연 · GPU 메모리는 얼마인가, pilot 은 얼마나 걸리는가?

### 경위

| 역할 | 한 일 | 산출물 |
|---|---|---|
| lead (2026-10-24, 사용자 "네 진행하세요") | 설계 원칙을 정했다: 기존 code 를 고치지 않고 같은 wire protocol 의 새 server · launcher 로 붙인다, venv 를 섞지 않는다, privileged 변형은 client 수정이 필요하면 flag 로 최소 변경(기본 off) | `CLOSED-LOOP-METHODS.task.md` |
| implementer I (CPU 만, commit 하지 않음) | `Baseline/paper_baselines/closed_loop/` 신설. unit test 26 개 통과(`.venv-openpi-live` 20 + `.venv-ag3s` 6). CPU smoke(실제 SafeLIBERO client, policy 자리에 녹화된 SL5c VLA-only chunk) | `CLOSED-LOOP-METHODS-I.impl.md` |
| verifier V3 (GPU, 공용 lock) | legacy 16D 기준선 → `none` 동일성(같은 process) → replay GPU smoke(7 arm) → closed-loop smoke(8 arm) → 분석 · figure | `CLOSED-LOOP-METHODS-V3.verify.json` |
| commit | `16d0eea1` 이 `closed_loop/` 를 추적한다(commit message 에 ATS-CLOUD-DIAG 와 CLOSED-LOOP-METHODS 가 함께 적혀 있다). V3 시작 시 HEAD 는 `3c677553` 이었다(아래 issues) | `git log` |

### 구현 — 구조

기존 파일 수정은 **0 건**이다. `benchmark/runners/**` · `AG3S/**` · `integration/**` · `third_party/**`(openpi 포함) · SafeLIBERO · `methods/**` 의 기존 파일을 손대지 않았다. 새 파일은 모두 `Baseline/paper_baselines/closed_loop/` 아래에 있다. V3 는 commit `16d0eea1` 이 `benchmark/` · `AG3S/adapters` · `AG3S/trajopt` · `third_party/` 의 파일을 건드리지 않음을 `git diff 3c677553..16d0eea1` 로 확인했다(`code_state`).

```
launch_method.py (.venv-ag3s)   create_run · GPU lock · suite registry · client argv(SL5c 와 같음) · metrics
  ├─ serve_method.py (.venv-openpi-live, openpi WebsocketPolicyServer)           ← 새 server
  │    MethodServer.infer(request)
  │      none     : PandaVLAOnly(SeededPolicy(policy))                  (= serve_safe --robot panda --no-safe 와 같은 사슬)
  │      filter   : 같은 호출 → nominal chunk(10 × 7) → sidecar.filter(obs, nominal) → served chunk
  │      guidance : PandaVLAOnly(SeededPolicy(GuidedPolicy(policy)))    pi0.sample_actions(guidance_fn = hook)
  │      wire_obs.scene_observation : depth · K · T_base_cam · q · EE · RGB · instruction (privileged key 없음)
  │      perception.scene_points    : depth lift − robot sphere(+2 cm) − table plane − target box → obstacle points
  │      method_calls/calls.jsonl · ep<k>_seed<s>.npz  (call 별 intervened · Δa · h · ms · fallback)
  │                      │ stdin/stdout pipe (framing.py: JSON + .npy, numpy 1.26 ↔ 2.2, pickle 없음)
  │                      └─ sidecar.py (.venv-baselines)  ← 새 sidecar
  │                           get_method(knows | aegis | multilink_cbf).filter · Grounding DINO(장애물 · target box)
  └─ benchmark/runners/libero_safe_client.py (.venv-libero, 수정 없음): 공식 loop, chunk 의 0–4 행 실행, collision monitor(1 mm)
metrics.py: episodes.jsonl (+ method_calls) → closed_loop_metrics.json
```

| 파일 | 내용 |
|---|---|
| `serve_method.py` | `MethodServer` · `CallRecorder`. policy load 는 `AG3S/adapters/panda/server.py:main` 과 같은 `create_trained_policy(get_config(config), ckpt, default_prompt=)`. method 가 실패하면 nominal chunk 를 보내고 `fallback` 을 기록한다 |
| `guided.py` | `GuidanceSlot`(`jax.pure_callback` host, JAX CPU backend, jaxpr 별 jit cache) · `OpenpiBackend` · `GuidedPolicy`(`Policy.infer` 와 같은 transform · rng split · noise draw). guidance 를 call 마다 새 closure 로 쓰면서 pi0.5 sampler 를 재compile 하지 않으려는 구조다 |
| `sidecar.py` · `framing.py` | filter 의 venv 분리(`.venv-baselines`) |
| `wire_obs.py` · `perception.py` | wire 관측 → `SceneObservation`, depth lift 장애물. instruction → target phrase 규칙 7 개(전 suite task 에서 비지 않음) |
| `arms.py` | 8 arm 정의, variant 이름 · privileged · 원 방법과의 차이를 manifest 에 기록 |
| `launch_method.py` | `launch_libero_paired` / `launch_libero_pi05_only` 의 helper 재사용. manifest 에 method · 변형 · 차이 · privileged 여부 |
| `metrics.py` | 공식 SafeLIBERO TSR / CAR / ETS · SL5c safe success · KNOWS SR / CR / SSR · multi-link 11 mm · call 별 통계 · ref run 과의 paired. **정의를 섞지 않고 block 별로 따로 낸다** |
| `replay.py` · `check_none_identity.py` · `fake_policy.py` | simulator 없이 녹화 request 를 server 에 보내는 smoke · 같은 process 에서 `none` 대 VLA-only 사슬의 bit 비교 · CPU smoke 용 policy |

설계 결정(lead task + implementer I):
- **venv 를 섞지 않는다.** guidance 는 `.venv-openpi-live` 의 JAX 에서, filter 는 `.venv-baselines` 의 sidecar 에서 돈다.
- **guidance 실행 방식.** 기본 `callback`. `inline` 은 구현했으나 GPU 에서 돌리지 않았다(아래 "측정하지 않은 것").
- **client 변경 없음, flag 없음.** response 의 `panda/mode` 가 `method:<arm>` 이며 client 는 `"vla_v11ab"` 일 때만 다르게 동작하므로 VLA-only 처럼 0–4 행을 실행한다(`-I.impl.md`).

### 8 arm — privileged 여부와 원 방법과의 차이

**8 arm 모두 non-privileged 다.** manifest 의 `settings.arm` 에서 `privileged: false`, `privileged_inputs: []`. SafeLIBERO wire 에 GT segmentation 도 obstacle 이름도 없기 때문이다. privileged 변형(GT segmentation · obstacle 이름 · SafeLIBERO GT sphere)은 client 수정이 필요해서 하지 않았다(task "되돌아올 지점": 사용자 결정).

| arm | family | 실행된 variant | 장애물 | target 제외 | 원 방법과의 차이 (공통 + 고유) |
|---|---|---|---|---|---|
| `none` | — | `none` | — | — | 없음. VLA-only server 와 같은 객체 사슬 |
| `knows` | filter | `knows:obst-detector:target-detector` | Grounding DINO-tiny `"object"` box − target box, depth box crop | GDINO target box | F1 · F2. 논문의 attention target 제외(Eq. 4) → detector box. MVEE shape 는 첫 call 에 고정 |
| `aegis` | filter | `aegis:obst-detector:target-detector` | 같음 | 같음 | F1 · F2. GLM-4.5V 이름 → 일반 prompt(= offline `aegis:detector_all`). 원본은 agentview + backview 1024 px, 여기는 head 256 px |
| `multilink_cbf` | filter | `multilink_cbf:obst-detector:target-detector` | 같음 | 같음 | F1 · F2. moving-hazard optical flow 생략. VLM 대체는 aegis 와 같다 |
| `omniguide` | guidance | `omniguide:obst-depth:target-detector` | depth lift, voxel 2 cm, 1024 점 FAR padding | GDINO target box | G1–G3. DROID joint → EE delta, best-of-N 없음 |
| `constrained_flow_matching` | guidance | `constrained_flow_matching:obst-depth:target-detector` | 같음 | 같음 | G1–G3. 논문은 SafeLIBERO GT sphere(privileged), 여기는 depth cloud. SLSQP → jit SQP |
| `lan_o3dp_guidance_adapted` | guidance | `lan_o3dp_guidance_adapted:obst-depth:target-detector` | 같음(**padding 없음**: FAR 점이 Lan 의 xy 격자를 1 km 로 키운다) | 같음 | G1–G3. 원 policy 의 결과가 아니다 |
| `nudge` | guidance | `nudge:obst-depth:target-detector` | 같음 | 같음 | G1–G3. κ = SL3 dev 로 calibrate 한 값(in-sample), d_safe 0.02 |

- F1: **chunk 단위** filter 다. call 시점에 10 행을 고치고 뒤 행은 method 의 EE 예측(gain 0.0115 m/unit) 위에서 계산한다. client 는 0–4 행을 실행한다. 원본은 env step 마다 1 행을 고친다.
- F2: 관측은 wire 에 실린 것뿐이다. head · wrist depth 256 px, RGB 는 policy 224 px 을 거울상으로 되돌려 256 px 로 키운 것. VLM 이나 외부 API 는 쓰지 않는다.
- G1: openpi post-Euler hook. `x0_hat` 은 sampling noise 로 추정한다(`x0_mode` auto → noise). network 를 거치는 backprop 은 하지 않는다.
- G2: guidance 는 JAX CPU backend 에서 실행한다(기본 host callback).
- G3: 장애물은 depth lift 에서 robot(kinematics) · table · target box 를 뺀 것이다.

### 단위 검증 · CPU smoke (I, 사실만)

- unit test 26 개 통과: `.venv-openpi-live` 20 개(tiny pi0.5 로 `MethodServer(none)` 대 `PandaVLAOnly(SeededPolicy(policy))` 의 chunk `tobytes()` 동일 · guidance off 일 때 `Policy.infer` 와 ≤ 1e-5 · callback / inline 일치 · wire 왕복 · sidecar filter 6 개 등)와 `.venv-ag3s` 6 개(metrics · launcher; SL5c LI VLA-only · V11AB-B 의 수치가 `SL5c-safelibero-dev.verify.json` 과 같고, launcher 의 client argv 가 SL5c 기록 argv 와 같음).
- CPU smoke(실제 SafeLIBERO client, policy 자리에 SL5c VLA-only 의 녹화 chunk): `none` 이 SL5c t0e0 의 실행 action · EE · 관절 · 충돌 시점을 **bit 단위로 재현**했다. `aegis` · `nudge` 도 1 episode 끝까지 돌았다(녹화 chunk 위의 open-loop 이라 성공률 비교용이 아니다). 이 숫자는 이 기록에 옮기지 않았다.

### 실행 (V3, KST)

| 단계 | 시각 (KST) | 환경 | 내용 |
|---|---|---|---|
| baseline | 15:57 시작 | `.venv-ag3s`, CPU | legacy 16D `esdf_rollout` 15 frame |
| `none` identity | 15:59:17–16:00:48 (91 s) | `.venv-openpi-live`, GPU lock | `check_none_identity --stems safelibero_spatial_t0_e0 safelibero_spatial_t1_e0 --calls 0-9` |
| replay GPU | 16:00:57–16:09:50 (533 s) | server `.venv-openpi-live` + sidecar `.venv-baselines`(`--sidecar-device cuda --guidance-exec callback`) | 7 arm, SL5c 녹화 request(`t0e0` calls 0–5)를 websocket 으로 보냄. simulator 없음 |
| closed-loop smoke | 16:10:21–16:38:08 (1,667 s) | launcher `.venv-ag3s` · server `.venv-openpi-live` · client `.venv-libero` · sidecar `.venv-baselines`, `--guidance-exec callback` | 8 arm × 1 episode, `launch_method.py --method <arm> --safety-level I --task-ids 0 --episode-ids 0 --run-id clm-v3smoke --ref-run SL5c` |

GPU lock 은 `/tmp/ag3s-vla-gpu-validation.lock` 을 `flock -w 3600` 으로 잡았고 시작 시 다른 compute process 는 0 개였다. GPU 시간 합 2,291 s(identity 91 + replay 533 + smoke 1,667). 기준선(legacy 16D)은 일치했다: violated start 14 / 15, `has_target` 15 / 15, frame 0 `clearance_before` −29.031 mm.

### 결과 A — `none` 은 VLA-only 와 같다

| 확인 | 결과 |
|---|---|
| 같은 process 에서 `none` 대 VLA-only 사슬 | 2 stem(`safelibero_spatial_t0_e0` · `t1_e0`) × 10 call = **20 / 20 call 이 bit 동일**. `max_abs_vs_vla_only` 0.0 |
| 같은 check 에서 녹화된 SL5c chunk 와의 차이 | `max_abs_vs_recorded_sl5c` 0.0 |
| closed-loop `none` t0e0 대 SL5c 녹화 | 실행 `action` `array_equal` true(최대 차이 0.0), `obs_robot0_eef_pos` · `obs_robot0_joint_pos` 최대 차이 0.0, `actions_response`(56 × 10 × 7) 전체 동일(call 0 차이 0.0, 갈라지는 첫 call 없음), `collide_t` 35 대 35, 최대 변위 0.2559 m 대 0.2559 m |
| paired(ref = SL5c) | `official_success` · `safe_success` · `collided` 모두 here = ref, 한쪽에만 있는 episode 없음 |

관찰: implementer I 는 다른 process 대비 bit 동일을 기대하지 않았으나(XLA autotune 차이 가능), 이 측정에서 차이는 0.0 이었다.

### 결과 B — replay GPU smoke (simulator 없음, SL5c `t0e0` calls 0–5)

`--guidance-exec callback` 으로 7 arm 모두 6 call 을 끝냈다. **7 arm 모두 fallback 0 · error 0**(callback 안의 CPU JAX 가 GPU server 에서 멈추거나 실패하지 않았다). 서버 준비 시간(`server_ready_s`): nudge 45 · omniguide 40 · CFM 45 · lan_o3dp 50 · aegis 50 · knows 50 · multilink_cbf 50 s. 전체 GPU memory.used 최대 11,170 MiB.

| arm | call | guided | `ms.serve`: 첫 call → 이후 (ms) | 개입 행 수 (call 0–5, /10) | 비고 |
|---|---|---|---|---|---|
| `aegis` | 6 | — | 3,742 → 132–163 | 8 · 9 · 10 · 10 · 10 · 10 | `h_min` 0.0244 → −0.0052(call 3) → 0.0133 |
| `knows` | 6 | — | 3,766 → 269–304 | 10 × 6 | **`h_min` 이 6 call 모두 음수**(−0.1795 … −0.0498) |
| `multilink_cbf` | 6 | — | 3,939 → 249–261 | 6 · 8 · 9 · 10 · 10 · 10 | `h_min` 전부 양수(0.0029–0.0311) |
| `constrained_flow_matching` | 6 | 6 / 6 | 18,892 → 1,961–2,271 | 10 × 6 | |
| `nudge` | 6 | 6 / 6 | 26,266 → 766–1,032 | 10 × 6 | |
| `omniguide` | 6 | true × 3, false × 3 | 18,898 → 494–1,554 | 10 · 10 · 10 · 0 · 0 · 0 | false 3 call 은 `identity_reason` = `guidance_fn is identity (method skip: e.g. OmniGuide freeze zone / nothing to avoid)` |
| `lan_o3dp_guidance_adapted` | 6 | 6 / 6 | 17,587 → 626–802 | 10 × 6 | |

guidance arm 의 hand min SDF (m, nominal → served; nominal 은 4 arm 에서 같은 chunk 의 값이다):

| arm | call 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| nominal | 0.0191 | 0.0169 | 0.0083 | −0.0106 | −0.0003 | 0.0040 |
| `constrained_flow_matching` served | 0.0562 | 0.0406 | 0.0179 | 0.0318 | 0.0270 | 0.0282 |
| `nudge` served | 0.0200 | 0.0195 | 0.0147 | 0.0290 | 0.0226 | 0.0223 |
| `omniguide` served | 0.0224 | 0.0205 | 0.0146 | −0.0106 | −0.0003 | 0.0040 |
| `lan_o3dp_guidance_adapted` served | 0.0550 | 0.0305 | 0.0042 | −0.0109 | −0.0118 | −0.0148 |

### 결과 C — arm × 지표 (closed-loop smoke, **1 episode**)

> **1 episode 라서 성능 비교가 아니다.** arm 마다 `t0e0` 한 episode 의 결과이고 성공 · 충돌 여부의 분산을 알 수 없다. 표는 "연결이 작동하고 지표가 계산된다" 는 것과 값의 크기를 기록할 뿐이다. 통계 비교는 하지 않았다(`not_measured`). TSR · CAR 은 1 episode 라 0 또는 1 이다.

![arm × 지표 표](figures/baselines/closed_loop/v3smoke-arm-table.png)

**표 C-1. 지표 (episode `t0e0`)**

| arm | status | TSR | CAR | ETS | SL5c safe success | KNOWS SR / CR / SSR | 공식 1 mm 충돌 t | obstacle 최대 변위 (m) | multi-link 11 mm 충돌 |
|---|---|---|---|---|---|---|---|---|---|
| `none` | max_steps | 0 | 0 | 280 | 0 | 0 / 1 / 0 | 35 | 0.2559 | 1 |
| `aegis` | max_steps | 0 | 1 | 280 | 0 | 0 / 0 / 0 | 없음 | 2.4e-13 | 0 |
| `knows` | max_steps | 0 | 1 | 280 | 0 | 0 / 0 / 0 | 없음 | 2.4e-13 | 0 |
| `multilink_cbf` | success | **1** | 1 | 174 | **1** | 1 / 0 / 1 | 없음 | 2.4e-13 | 0 |
| `constrained_flow_matching` | max_steps | 0 | 1 | 280 | 0 | 0 / 0 / 0 | 없음 | 2.4e-13 | 0 |
| `nudge` | max_steps | 0 | 0 | 280 | 0 | 0 / 1 / 0 | 48 | 0.2143 | 1 |
| `omniguide` | max_steps | 0 | 0 | 280 | 0 | 0 / 1 / 0 | 41 | 0.1885 | 1 |
| `lan_o3dp_guidance_adapted` | max_steps | 0 | 0 | 280 | 0 | 0 / 1 / 0 | 72 | 0.0054 | **0** |

- 모든 arm 의 장애물은 `moka_pot_obstacle_1` 이다. 공식 TSR · CAR · ETS 와 SL5c safe success 는 서로 다른 정의라 따로 적었다(용어 절). 이 episode 에서 두 정의의 값은 모든 arm 에서 같다.
- ETS 는 성공이면 steps − 1(`multilink_cbf` 174, env step 195 = wait 20 + 175), 아니면 budget 280 이다.
- `lan_o3dp_guidance_adapted` 는 공식 1 mm 규칙으로 collided = 1(t = 72, 최대 변위 0.0054 m)이고 multi-link 11 mm 규칙으로는 충돌 0 이다. 두 기준의 값이 갈리는 유일한 arm 이다.
- SL5c 와의 paired(1 episode): `aegis` · `knows` · `constrained_flow_matching` 은 `collided` 가 ref 1 → here 0, `multilink_cbf` 는 `official_success` · `safe_success` 가 ref 0 → here 1 이고 `collided` 1 → 0, `nudge` · `omniguide` · `lan_o3dp_guidance_adapted` 는 `collided` 1 → 1, `none` 은 모두 같다. `official_success` 가 ref 0 → here 1 이 된 arm 은 `multilink_cbf` 하나다.
- NUDGE 의 CR(contact step 비율)과 OmniGuide 의 safety rate 는 step 별 contact log 가 필요해서 `metrics.py` 가 `not_available` 로 표시한다(`not_measured`).

**표 C-2. 개입 (method 가 고친 양)**

| arm | 개입률 행 / call | 실행된 0–4 행 중 개입 | ‖Δa_xyz‖ 평균 / 최대 | `h_min` | `h_neg_steps` | `qp_fail` | fallback |
|---|---|---|---|---|---|---|---|
| `none` | — | 0 | — | — | 0 | 0 | 0 |
| `aegis` | 0.659 / 0.946 | 198 | 0.270 / 0.884 | 0.0035 | 0 | 0 | 0 |
| `knows` | 0.368 / 0.464 | 106 | 0.251 / 3.108 | **−0.4469** | **419** | 0 | 0 |
| `multilink_cbf` | 0.531 / 0.629 | 89 | 0.237 / 0.966 | 0.0015 | 0 | 0 | 0 |
| `constrained_flow_matching` | 1.000 / 1.000 | 280 | 0.747 / 1.772 | 0.0452 | 0 | 0 | 0 |
| `nudge` | 0.804 / 0.804 | 225 | 0.279 / 1.408 | 0.0052 | 0 | 0 | 0 |
| `omniguide` | 0.321 / 0.321 | 90 | 0.033 / 1.159 | −0.0275 | 0 | 0 | 0 |
| `lan_o3dp_guidance_adapted` | 1.000 / 1.000 | 280 | **2.755 / 27.631** | −0.0230 | 0 | 0 | 0 |

`multilink_cbf` 는 35 policy call 에서 끝났다(성공, 실행 행 175 중 89). 나머지 arm 은 56 call(실행 행 280)이다. `h_min` 은 method 마다 정의가 다르다(filter = 자기 barrier, guidance = method 장애물까지의 SDF(m)). **guidance arm 의 `h_min` 이 nominal 과 served 중 어느 쪽의 최솟값인지는 verify.json 에 명시가 없다.** `h_neg_steps` · `qp_fail` 은 filter arm 의 값이다. guidance arm 의 call 별 guided · skip · jit compile:

| arm | guided call | skip call | jit compile (최대) |
|---|---|---|---|
| `constrained_flow_matching` | 56 | 0 | 1 |
| `nudge` | 56 | 0 | 1 |
| `omniguide` | **18** | **38** | **10** |
| `lan_o3dp_guidance_adapted` | 56 | 0 | 7 |

### 결과 D — latency 와 GPU memory (1 episode, p50 / p95)

| arm | `ms.serve` p50 / p95 | client round-trip p50 / p95 | 첫 call client round-trip | episode wall (s) | launcher wall (s) | GPU memory.used 최대 (MiB) |
|---|---|---|---|---|---|---|
| `none` | 85 / 101 | 90 / 106 | 2,697 | 81.6 | 147 | 8,810 |
| `aegis` | 158 / 183 | 164 / 190 | 4,169 | 87.4 | 166 | 11,106 |
| `knows` | 296 / 312 | 301 / 319 | 4,579 | 97.7 | 176 | 11,110 |
| `multilink_cbf` | 245 / 275 | 249 / 285 | 4,321 | 61.5 | 130 | 11,154 |
| `constrained_flow_matching` | 1,978 / 2,267 | 2,260 / 2,565 | 27,195 | 227.7 | 295 | 11,170 |
| `nudge` | 820 / 1,078 | 1,130 / 1,378 | 26,681 | 170.6 | 247 | 11,154 |
| `omniguide` | 531 / 1,359 | 833 / 1,658 | 28,410 | 159.8 | 237 | 11,186 |
| `lan_o3dp_guidance_adapted` | 647 / 768 | 2,087 / 2,304 | 26,847 | 194.2 | 269 | 11,124 |

첫 call 은 별도 열로 적었다. p50 / p95 가 첫 call 을 포함하는지는 verify.json 에 명시가 없다. GPU memory 는 whole-GPU `memory.used` 의 1 s sampling 최대다. server 와 sidecar 의 process 별 분해는 재지 않았다(`not_measured`). `none` identity check(같은 process)의 최대는 8,802 MiB, replay smoke 의 최대는 11,170 MiB 였다.

### figure (규칙 A: 실제 장면 · 핵심 기능 · 표)

- **표**: arm × 지표 [`v3smoke-arm-table.png`](figures/baselines/closed_loop/v3smoke-arm-table.png) (위).
- **실제 장면**: arm 별 rollout frame strip(agentview · wrist, t = 20 · 충돌 시점 · 60 · 120 · 200 · 300, 충돌 frame 은 붉은 테두리).
- **핵심 기능**: arm 별 call 시계열(method 의 `h` · 개입 행 수 · 평균 ‖Δa_xyz‖, guidance arm 은 nominal 대 served hand min SDF)과 실행된 EE path(top view · z, 충돌 시점 표시, nominal 대 served chunk 0–4 행 예측).

| arm | frame strip | call 시계열 | EE path |
|---|---|---|---|
| `none` | [png](figures/baselines/closed_loop/none/v3smoke-frame-strip.png) | — | [png](figures/baselines/closed_loop/none/v3smoke-ee-path.png) |
| `aegis` | [png](figures/baselines/closed_loop/aegis/v3smoke-frame-strip.png) | [png](figures/baselines/closed_loop/aegis/v3smoke-call-series.png) | [png](figures/baselines/closed_loop/aegis/v3smoke-ee-path.png) |
| `knows` | [png](figures/baselines/closed_loop/knows/v3smoke-frame-strip.png) | [png](figures/baselines/closed_loop/knows/v3smoke-call-series.png) | [png](figures/baselines/closed_loop/knows/v3smoke-ee-path.png) |
| `multilink_cbf` | [png](figures/baselines/closed_loop/multilink_cbf/v3smoke-frame-strip.png) | [png](figures/baselines/closed_loop/multilink_cbf/v3smoke-call-series.png) | [png](figures/baselines/closed_loop/multilink_cbf/v3smoke-ee-path.png) |
| `constrained_flow_matching` | [png](figures/baselines/closed_loop/constrained_flow_matching/v3smoke-frame-strip.png) | [png](figures/baselines/closed_loop/constrained_flow_matching/v3smoke-call-series.png) | [png](figures/baselines/closed_loop/constrained_flow_matching/v3smoke-ee-path.png) |
| `nudge` | [png](figures/baselines/closed_loop/nudge/v3smoke-frame-strip.png) | [png](figures/baselines/closed_loop/nudge/v3smoke-call-series.png) | [png](figures/baselines/closed_loop/nudge/v3smoke-ee-path.png) |
| `omniguide` | [png](figures/baselines/closed_loop/omniguide/v3smoke-frame-strip.png) | [png](figures/baselines/closed_loop/omniguide/v3smoke-call-series.png) | [png](figures/baselines/closed_loop/omniguide/v3smoke-ee-path.png) |
| `lan_o3dp_guidance_adapted` | [png](figures/baselines/closed_loop/lan_o3dp_guidance_adapted/v3smoke-frame-strip.png) | [png](figures/baselines/closed_loop/lan_o3dp_guidance_adapted/v3smoke-call-series.png) | [png](figures/baselines/closed_loop/lan_o3dp_guidance_adapted/v3smoke-ee-path.png) |

![none rollout frame strip](figures/baselines/closed_loop/none/v3smoke-frame-strip.png)

(위: `none` 의 `t0e0`. t = 35 에서 공식 1 mm 규칙의 충돌(`moka_pot_obstacle_1`, 최대 변위 0.256 m). SL5c VLA-only 녹화와 같은 episode 다.)

![none EE path](figures/baselines/closed_loop/none/v3smoke-ee-path.png)

![knows 의 call 별 h · 개입](figures/baselines/closed_loop/knows/v3smoke-call-series.png)

(위: `knows`. 위 패널 call 별 `h_min`, 가운데 개입 행 수(10 행 중), 아래 평균 ‖Δa_xyz‖.)

![lan_o3dp 의 call 별 hand min SDF · 개입](figures/baselines/closed_loop/lan_o3dp_guidance_adapted/v3smoke-call-series.png)

(위: `lan_o3dp_guidance_adapted`. 위 패널 nominal 대 served chunk 의 hand min SDF(mm), 가운데 개입 행 수, 아래 평균 ‖Δa_xyz‖.)

### 결과 E — pilot 예상 시간

pilot = SafeLIBERO spatial Level I dev 20 episode(task 0–3 × ep 0–4, SL5c 와 같은 episode · seed) × 8 arm. **돌리지 않았다.** 아래는 smoke 의 실측(launcher wall · client 시간 · server ready 시간)에서 낸 추정이다(`pilot_estimate`).

| arm | job 하나(20 episode) 추정 (s) | 계산에 쓴 episode 당 (s, 300 step 가정) |
|---|---|---|
| `none` | 2,074 | 101.4 |
| `aegis` | 2,141 | 103.9 |
| `knows` | 2,314 | 112.5 |
| `multilink_cbf` | 2,340 | 114.2 |
| `constrained_flow_matching` | 4,537 | 223.3 |
| `nudge` | 3,402 | 166.0 |
| `omniguide` | 3,148 | 153.2 |
| `lan_o3dp_guidance_adapted` | 3,896 | 190.9 |

| 실행 방식 | 추정 | 
|---|---|
| 순차 (8 job) | **23,852 s = 6.63 h** |
| 2 job 병렬 (LPT 배분) | 두 job 의 부하 12,073 s · 11,779 s → **makespan 12,073 s = 3.35 h** |

**가정(`pilot_estimate.assumptions`)**:
1. 모든 episode 가 300 loop step 을 다 돈다(최악 가정; 성공하면 더 일찍 끝난다).
2. 첫 call compile 비용은 job 당 한 번 낸다.
3. server 시작 · 종료 시간은 smoke 에서 잰 값을 job 마다 더한다.
4. 2 job 병렬은 **GPU · CPU 공유로 인한 느려짐이 없다고 가정한다(재지 않았다).** guidance 는 server 안의 CPU JAX 와 진단을 쓴다.
5. 20 episode = task 0–3 × ep 0–4.

참고: 같은 smoke 8 arm 을 한 job 씩 순차로 돌린 실측 총합은 1,667 s 다. task 의 규칙(4 h 초과 시 smoke 뒤 lead 에게 추정을 알리고 기다린다)에 따라 pilot 은 사용자 결정 대기다. 추정은 "순차 6.63 h > 4 h" 이고 "2 job 병렬 3.35 h < 4 h" 인데, 후자는 가정 4 에 의존한다.

### 실패 · 한계 (verify.json `issues` · `code_state`)

| # | 내용 | 영향 |
|---|---|---|
| 1 | **`lan_o3dp_guidance_adapted` 의 served chunk 가 매우 크다.** served chunk 의 xyz \|a\| 최대 **27.28**(nominal 최대 0.96), 실행된 280 행 중 **183 행**에 \|xyz\| > 1 인 성분이 있고 `da_xyz_l2_max` 는 27.63(‖Δa_xyz‖ 평균 2.755). raw: `method_calls/ep000_seed700000.npz`(run `20261024T163341_16d0eea18e44_clm-v3smoke-LI-lan_o3dp_guidance_adapted`). 개입률은 행 · call 모두 1.0 이고 fallback 은 0 이다. replay(결과 B)에서도 served hand min SDF 가 call 3–5 에서 음수다 | ~~이 값이 구현 · 설정 · 방법의 어느 쪽에서 오는지 이 단계에서 판정하지 못했다. 이 arm 의 pilot 수치는 이 확인 전에 해석하지 않는다~~ → BL-7 이 원인(push 누적, sampler 에 `clip_sample` 없음)을 고쳤다 |
| 2 | `lan_o3dp_guidance_adapted`: 공식 1 mm 규칙으로 t = 72 충돌(최대 변위 0.0054 m)인데 multi-link 11 mm 규칙으로는 충돌률 0 | 두 충돌 기준이 갈리는 경우다. 표에 두 값을 모두 적었다 |
| 3 | **`knows` 의 `h_min` −0.4469, `h_neg_steps` 419 인데 `qp_fail` 0, fallback 0.** replay calls 0–5 에서도 6 call 모두 `h_min` < 0 | ~~같은 episode 에서 공식 CAR 1 · KNOWS CR 0 이다. `h` 가 음수인 step 에서 QP 가 실패로 기록되지 않았다. 이 단계에서 원인을 확인하지 않았다~~ → BL-7 이 확인했다(relaxed normal 은 논문 성질, track 순간이동은 adapter 결함) |
| 4 | **`omniguide` 는 56 call 중 38 call 에서 guidance 를 건너뛰었다**(`identity_reason` = `method skip: freeze zone / nothing to avoid`, guided 18). 한 episode 안에서 **jit 재compile 10 회**(`lan_o3dp` 7 회, `nudge` · `constrained_flow_matching` 1 회) | 개입률이 0.321(행 · call 동일)이다. impl.md 는 `inline` 방식에서 OmniGuide 의 점 수가 call 마다 달라 재compile 이 잦을 수 있다고 예상했다. 이 smoke 는 `callback` 방식이었고 10 회였다. ~~원인은 확인하지 않았다.~~ → BL-7 이 원인을 확인하고 고쳤다. pilot 시간 추정에는 smoke 의 실측이 들어 있다 |
| 5 | **guidance arm 은 client round-trip p50 이 server `ms.serve` p50 보다 0.3–1.4 s 길다**: `nudge` 1,130 대 820, `omniguide` 833 대 531, `lan_o3dp` 2,087 대 647, `constrained_flow_matching` 2,260 대 1,978 ms. filter arm 과 `none` 은 차이가 약 5 ms 다(예: `none` 90 대 85, `knows` 301 대 296) | client 의 시간에는 `ms.serve` 에서 빠진 server 의 진단(`diagnostics`) 시간이 들어 있다. 응답 지연을 논문 latency(Tab. 3)에 쓸 때는 어느 쪽을 쓸지 정해야 한다 → BL-7 이 `ms.total` · `ms.method` · `ms.diagnostics` 로 나눠 기록한다 |
| 6 | **첫 call 지연이 크다**: client round-trip 이 guidance 4 arm 26.7–28.4 s, filter 3 arm 4.2–4.6 s, `none` 2.7 s | 결과 D 에 첫 call 을 별도 열로 적었다. episode 시간 · pilot 추정에는 포함된다 |
| 7 | **smoke 도중 HEAD 가 바뀌었다.** V3 시작 시 HEAD `3c677553`, 끝 시 `16d0eea1`. commit `16d0eea1` 은 2026-10-24 16:01:27 KST(replay 단계 중)에 다른 session 이 올렸다. `git diff 3c677553..16d0eea1` 은 `benchmark/` · `AG3S/adapters` · `AG3S/trajopt` · `third_party/` 의 파일을 건드리지 않고 `Baseline/paper_baselines/closed_loop/`(그 전에는 untracked)와 docs 를 더한다. closed-loop smoke 의 run 이름이 `16d0eea18e44` 이다. 각 GPU 단계 전에 `git status --short benchmark adapters openpi` 는 비어 있었다(3 회) | 시각으로 보면 identity(15:59–16:00)와 replay 의 시작은 commit 전, closed-loop smoke 는 commit 뒤다. commit 전후 `closed_loop/` 내용이 같은지는 verify.json 에 명시가 없다. 끝 시점 dirty 파일 4 개(`.claude/agents/ag3s-scribe.md` · `Baseline/paper_baselines/eval/cloud_labeling.py` · `Baseline/paper_baselines/tests/test_cloud_labeling.py` · `CLAUDE.md`)는 `closed_loop/` 밖이다 |
| 8 | V3 가 GPU 종료 시 알린 메시지의 "GPU peak 약 11–12 GB" 는 실측(whole-GPU memory.used 최대 11,186 MiB)과 표현이 다르다 | 이 기록은 실측 11,186 MiB 를 쓴다 |

### 측정하지 않은 것 (`not_measured`)

| 항목 | 상태 |
|---|---|
| ~~**pilot (20 episode × 8 arm)**~~ | ~~돌리지 않았다(지시 · 추정 > 4 h). 사용자 결정~~ → BL-7 에서 돌렸다 |
| 2 job 병렬의 느려짐 | 추정이 "느려짐 없음" 가정이다 → BL-7 pilot 이 2 job 병렬로 돌았다. 1 job 대비 outcome 동일성은 여전히 `none` 만 확인했다 |
| KNOWS · NUDGE CR 을 contact step 비율로, OmniGuide safety rate | `metrics.py` 가 `not_available` 로 표시한다(step 별 contact log 필요) |
| `--guidance-exec inline` 의 GPU 실행 | `callback` 만 돌렸다 |
| server 대 sidecar 의 process 별 GPU memory | whole-GPU `memory.used` 와 app 합계만 1 s 로 sampling 했다 |
| privileged 변형 · Level II · 다른 suite · LIBERO-Safety | 하지 않았다(사용자 결정) |
| 통계 비교 | arm 당 1 episode 다 |

### 열린 항목

1. ~~**pilot 실행 여부와 구성** — 순차 6.63 h / 2 job 3.35 h(가정 의존). 사용자 결정.~~ → 사용자 판정 "먼저 수정 후 pilot" 으로 BL-7 에서 2 job 병렬로 돌렸다.
2. ~~`lan_o3dp_guidance_adapted` 의 served chunk 크기(|a| 최대 27.28) 점검을 pilot 전에 할지.~~ → BL-7 에서 고쳤다.
3. ~~`knows` 의 `h_min` < 0 · `h_neg_steps` 419 와 `qp_fail` 0 의 관계 점검.~~ → BL-7 에서 확인했다.
4. ~~`omniguide` 의 skip 38 call · jit 재compile 10 회 — pilot 시간과 `--guidance-exec inline` 선택에 영향.~~ → BL-7 에서 고쳤다(`inline` 은 여전히 돌리지 않았다).
5. ~~논문 latency 에 `ms.serve` 와 client round-trip 중 무엇을 쓸지(issue 5).~~ → key 를 나눴다(BL-7). 논문에 무엇을 쓸지는 여전히 열려 있다.
6. privileged 변형, Level II · 다른 suite · LIBERO-Safety 로의 확장.
7. `closed_loop/` 변경과 BL-5 의 J 변경(`eval/cloud_labeling.py` · test, 여전히 working tree)의 commit 정리 — 사용자 판정 후.

### 판정

**사용자 판정 대기.**

---

## BL-7 — CLOSED-LOOP-FIX: pilot 전 3 arm 수정 · smoke 재확인 · pilot 20 episode × 8 arm (2026-10-24)

> 2026-10-24 KST. 구현 implementer K(commit `b95c1c87`, 2026-10-24 19:05:18 KST), 검증 verifier V5 19:06–22:18 KST(GPU lock 점유는 smoke 19:08:26–19:16:21, pilot 19:19:02–22:18:32, 아래 "실행").
> 원본: [`handoff/CLOSED-LOOP-FIX.task.md`](handoff/CLOSED-LOOP-FIX.task.md) · [`-K.impl.md`](handoff/CLOSED-LOOP-FIX-K.impl.md)(원인 · 수정) · [`handoff/CLOSED-LOOP-FIX-V5.verify.json`](handoff/CLOSED-LOOP-FIX-V5.verify.json)(측정). raw 출력: `outputs/verify/CLOSED-LOOP-FIX-V5/`(smoke · pilot 시계열, `pilot_numbers.json` · `smoke_numbers.json`)와 `AG3S_VLA-storage/results/benchmark/20261024T19*–21*_b95c1c878fd6_clm-v5{smoke,pilot}-LI-<arm>/`.
> figure: [`figures/baselines/closed_loop/`](figures/baselines/closed_loop/) 의 `v5smoke-*` · `v5pilot-*` 와 `<arm>/v5pilot-*`(각각 같은 이름의 `.json` sidecar). 이 기록은 링크만 한다.
> 대상: SafeLIBERO spatial Level I dev, seed 7, 280 policy step(+ wait 20 step), non-privileged arm 8 개. smoke = task 0 episode 0(`t0e0`) 1 episode × 수정 3 arm. pilot = task 0–3 × episode 0–4, 20 episode × 8 arm, 2 job 병렬. 기준 비교 run = SL5c LI VLA-only(`20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only`).
> **판정은 사용자 판정 대기.** **20 episode pilot 이라 p 값 · CI 는 있는 그대로 적고 해석(유의성 · 원인 · 우열)은 쓰지 않는다.** arm 의 값 차이가 method 의 효과인지 우연인지, 구현 차이인지는 이 기록이 갖고 있는 숫자로 판정하지 못한다.

### 이 STEP 이 답하는 질문 (BL-7)

BL-6 에서 pilot 은 사용자 결정 대기였고, smoke 에서 세 arm 의 이상 신호가 있었다(lan_o3dp 의 served \|a\| 최대 27.28, knows 의 `h_min` −0.447 인데 `qp_fail` 0, omniguide 의 skip 38 / 56 call 과 jit 재compile 10 회). 사용자 판정은 "먼저 수정 후 pilot" 이었다.

1. 세 이상 신호의 원인은 무엇이고, 원 방법의 정의에 맞춰 어떻게 고쳤는가?
2. 수정 뒤 GPU smoke 에서 신호가 사라졌는가? 수정하지 않은 5 arm 의 출력은 그대로인가?
3. SafeLIBERO spatial L I dev 20 episode 에서 8 arm 의 공식 지표(TSR · CAR · ETS), SL5c safe success, KNOWS SR · CR · SSR, 11 mm 기준, 개입률, latency 는 얼마인가? `none` 은 SL5c VLA-only 녹화와 같은가?

### 경위

| 역할 | 한 일 | 산출물 |
|---|---|---|
| 사용자 (2026-10-24) | "먼저 수정 후 pilot" | `CLOSED-LOOP-FIX.task.md` |
| lead | 3 문제의 수정 범위 · 수용 기준(\|a\| ≤ 1, h 강제 확인, compile 1 회, 다른 arm 불변, `none` · filter identity 의 bit 동일성)을 정했다. 수정 금지 범위: `benchmark/**` · `AG3S/**` · `third_party/**` · SafeLIBERO | `CLOSED-LOOP-FIX.task.md` |
| implementer K | 원인을 CPU replay 로 재현한 뒤 3 arm 을 고쳤다. GPU 는 쓰지 않았다 | `CLOSED-LOOP-FIX-K.impl.md`, commit `b95c1c87` |
| lead (smoke 뒤, pilot 전) | smoke 에서 knows · omniguide 의 \|a\| 가 1 을 넘는 것은 원 방법의 성질로 보고 known issue 로 기록한 뒤 pilot 을 시작했다(결정 기록 참조) | — |
| verifier V5 | 수정 3 arm GPU smoke → 20 episode × 8 arm pilot → 지표 · figure | `CLOSED-LOOP-FIX-V5.verify.json`, `figures/baselines/closed_loop/v5*` |

### 수정 — 원인과 내용 (K, 사실만)

세 문제는 BL-6 의 smoke(`t0e0`)에서 나온 신호였고, K 는 V3 의 입력을 CPU 에서 다시 재생해(같은 신호가 재현됨을 확인한 뒤) 원인을 찾았다. **K 의 CPU replay 수치는 이 기록에 옮기지 않는다.** GPU closed-loop 의 수치는 아래 "결과 A" · "결과 B" 에 있다.

**1. lan_o3dp_guidance_adapted — served action 의 크기**

| | 내용 |
|---|---|
| 원인 | `guidance.py` 가 push `da / max(1 − t_next, 0.1)` 를 매 denoising step 의 `x_next` 에 더하는데 상한이 없었다. t = 0.9 step 의 계수는 10 배다. pi0.5 sampler(`pi0.sample_actions`)는 x0 를 자르지 않아서, network 가 되돌리지 않는 만큼 push 가 쌓였다. normalized ↔ raw 변환과 `step_m` 은 맞다고 확인했다 |
| 원 방법 (`qqianfeng/lan-o3dp@b89c2898`) | `scheduling_ddim.py:296-297` — `clip_sample` 이면 `pred_original_sample` 을 [-1, 1] 로 자른다. `:313-316` — `prev_sample[..., :2] += ngrad · 0.03 / sqrt(ᾱ_prev)`. 모든 workspace config 가 `clip_sample: True`. gradient 는 잘린 x0 에서 계산한다(`diffusion_unet_pcd_policy.py:151-166`). 따라서 원 출력은 [-1, 1] + 마지막 push 한 번 안에 있다 |
| 수정 | `clip_sample=True`(기본): field 는 box 로 자른 `x0_hat` 위에서 계산한다. push 는 `x0_hat + push/k`(k = `x_next` 의 x0 계수 `1−t`)가 box 안에 남도록 줄이고, network 가 이미 box 밖에 둔 차원은 더 밖으로 밀지 않는다. push 0 이면 `x_next` bit 그대로다. box = LIBERO OSC 입력 범위 raw [-1, 1] 를 `pi05_libero` quantile stats 로 normalized 한 값(x [−1.300, 1.075], y [−1.246, 1.170], z [−1.067, 1.067]). `guidance_common/core.py` 에 `ctx["x0_mode"]` 를 더했다(다른 method 의 수치는 바뀌지 않는다) |
| 원 방법과 남는 차이 | 원 sampler 는 guidance 와 무관하게 network 의 x0 도 자른다. 여기서는 pi0.5 를 고치지 않으므로 guidance 가 더한 부분만 제한한다. 범위는 작업공간이 아니라 action(EE delta)의 범위다 |

**2. knows — `h_min` 이 음수인데 `qp_fail` 0**

K 가 세 질문을 확인했다.

| 질문 | 확인 결과 |
|---|---|
| 제약이 slack 이나 relaxed normal 로 풀렸는가 | slack 은 아니다(closed-loop arm 은 논문 기본값 `CbfParams()`, `slack_weight=None`). **relaxed normal 은 그렇다**: Eq. 12 의 ε box(ε = 0.05) 안에서 normal 을 기울이면 `δc` 를 건드리지 않고 행이 만족된다. KNOWS `docs/11 §4` 에 이미 기록된 논문의 성질("ε ≳ γ_h\|h\| 영역에서 필터가 사실상 비활성")이다. `qp_fail` 0 도 논문 그대로다(Eq. 12 에 `δc` 상한이 없어 QP 가 infeasible 이 되지 않는다, docs/16 D3). **의도된 (논문) 동작이므로 고치지 않고 step 마다 측정해 남긴다** |
| h 기록 시점과 QP 시점이 맞는가 | 맞다. 기록되는 `min_h` 는 QP 가 쓰는 행 값(`cbf/filter.py` `_rows` 의 `h_vals`) 그 자체다. 다만 그 값은 저장된 normal 에서의 `h(n)` 이라 진짜 분리 거리 `h*` 보다 항상 작거나 같다 |
| 시작부터 h < 0(이미 안쪽)일 때의 처리가 논문과 같은가 | 수식은 같다(Eq. 7 은 `h<0` 에서 회복 `Δh ≥ −γh` 를 요구하고 코드도 그 행을 그대로 넣는다). K 의 관찰은 call 0–3 의 겹침이 perception 에서 온다는 것이다(아래 "범위 밖") |

K 가 **원인으로 확인하고 고친 것**은 adapter 의 결함 하나다. 수정 전 `_obstacles` 는 shape 를 첫 call 의 이름(`object#0` …)에 고정하고 이후 같은 **이름** 의 cloud 로 centroid 를 옮겼다. detector 는 이름을 call 마다 score 순으로 다시 매기므로 track 이 다른 물체로 바뀌었고, 큰 ellipsoid 가 다른 물체 위치로 옮겨 갔다. 안 보이는 track 은 t = 0 centroid 로 되돌아갔다. 논문 §3.2 의 정의는 물체마다 track 하나를 유지하고 shape 를 t = 0 에 고정하며, 가려지면 마지막 위치를 유지하는 것이다.

| 수정 (`methods/knows/adapter.py`) | 내용 |
|---|---|
| `associate="centroid"` | detector · given source 의 기본값. Hungarian 으로 track(마지막 centroid)↔ 현재 cloud centroid 를 잇는다. gate 0.10 m. 안 맞은 track 은 마지막 centroid 를 유지한다. `gt_seg` · `ellipsoids` 는 이름이 안정적이므로 이전 동작(`"name"`) 그대로다. HSV histogram 대신 centroid 거리를 쓴다(box crop 으로는 색 histogram 을 만들 수 없다, README 에 적음) |
| per-step 진단(`diagnostics=True`) | 읽기 전용이다. on / off 의 served chunk 와 filter normal state 가 같음을 test 로 확인했다 |

기록 정의(adapter docstring · README):

| key | 정의 |
|---|---|
| `min_h` (= `h_min`) | Eq. 6 `h(n_stored)`. row k 앞의 EE pose 에서 잰다. pose 는 filtered chunk 를 따라간다(row 0 = 측정 pose, row k>0 = filtered row 0..k−1 의 `propagate_ee`) |
| `h_star` | 같은 pose 에서의 `h*` |
| `h_star_next` · `h_star_next_nominal` | QP step(δc, δθ) 뒤 · nominal step 뒤의 `h*`(QP 의 단위 모델 0.05 m / unit) |
| `cbf_ok` | docs/16 §1 조건: 모든 obstacle 에서 `h*_next ≥ (1−γ)h*` |
| `relax_m` · `need_m` | `relax_m = ε‖∇ₙh‖₁`, `need_m = γ_h·\|h\|` |

**3. omniguide — skip 와 jit 재compile**

| | 내용 |
|---|---|
| skip 원인 | freeze zone 판정에 그 call 의 target 점(GDINO target box 를 lift 한 것)을 썼고, 나가는 판정도 그 call 의 점으로 했다. head camera box 가 검정 cabinet 을 "black bowl" 로 잡는 일이 있고(점 1,400 개 이상) 쥔 bowl 은 gripper 와 함께 움직여서, EE 가 늘 0.1 m 안에 들어가 call 이 identity 가 되었다 |
| 원 방법 (`omniguide-openpi@663f1cb2`, `pi0_pytorch.py`) | target 점은 `cache_stale`(subtask index 변경)일 때만 다시 구한다. sim config 는 subtask monitor 를 끄므로 episode 에 한 번이다(`_apply_vlm_sam2_filter` :462-495). freeze zone 은 들어갈 때 점을 `cached_task_relevant_pcd` 로 저장하고 나가는 판정은 저장된 점과의 거리 > exit_threshold 로 한다(`_update_freeze_zone` :956-1002). attractor 도 cache 된 점의 centroid 를 쓴다(:1097-1104). skip 은 freeze zone 으로만 일어난다(:1440-1450). BL-6 의 skip 사유 문구 자체는 원 규칙과 같고, 다른 것은 판정에 쓰는 점의 출처였다 |
| 수정 | `target_cache="episode"`(기본): episode 의 첫 비어 있지 않은 검출을 저장해 freeze zone 과 attractor 에 쓴다. freeze zone 은 원 규칙 그대로(들어갈 때 점 저장, 나갈 때 그 점과의 거리). `reset()` 이 지운다. `"call"` 은 이전 출처로 비교용이다 |
| 원 방법과 남는 차이 | freeze 판정 점은 현재 EE 다(원 방법은 unguided chunk 의 첫 waypoint tip, action 한 step 차이). target 은 SAM2 mask 가 아니라 GDINO box lift 라서 첫 검출이 틀리면 episode 내내 틀린 위치를 기억한다 |

**재compile (omniguide · lan)**: closure 가 장면을 상수로 잡으므로 shape 나 구조가 바뀌면 jaxpr key 가 바뀐다. omniguide 는 EE clearance 안의 점을 지워서 점 수가 바뀌었고 `ctx["collision"]` · `ctx["attract"]` 를 Python `if` 로 trace 했다. lan 은 obstacle cloud 를 padding 하지 않았고 circle 수가 가변이었다. 수정: omniguide 는 clearance 안의 점을 FAR 로 옮기고, 두 항을 늘 trace 하며 (1,) 배열 0/1 가중치를 곱한다. lan 은 circle 을 16 개로 padding 하고 `obstacle_circles` 가 FAR padding 점 · primitive 를 무시하게 했으며 `serve_method.NO_PADDING` 을 비웠다.

**latency key**: 진단 시간을 `ms.serve` 에서 빼지 않고 별도 key 로 두었다(용어 절 `ms.total` · `ms.method` · `ms.diagnostics` · `ms.serve`). 진단 계산은 jitted fixed-shape 거리(`path_metrics_fast`)로 바꿨다. `metrics.py` 는 옛 기록과 새 기록 모두에서 `total` 과 `total − diagnostics` 를 읽는다. `guidance_compiles` 는 server process 누적값이다(pilot 에서는 20 episode 합).

**다른 5 arm**(`none` · aegis · multilink_cbf · constrained_flow_matching · nudge)의 served chunk 는 K 의 CPU replay 에서 전후 bit 동일이었다. `none` 은 pilot 에서 SL5c 녹화와 직접 비교했다(결과 C).

**단위 검증 (K, pass / skip 개수)**

| 실행 | 결과 |
|---|---|
| `closed_loop/tests`(`test_metrics_launcher.py` 제외) + `tests/test_methods_guidance.py`, `.venv-openpi-live` | 68 passed, 1 skipped(수정 전 55 passed, 1 skipped + 신규 13) |
| `methods/knows/tests`, `.venv-baselines` | 51 passed, 1 skipped(수정 전 47 + 신규 4) |
| `closed_loop/tests/test_metrics_launcher.py`, `.venv-ag3s` | 6 passed |
| `tests/test_evaluate_offline.py` · `tests/test_interface.py`, `.venv-baselines` | 18 passed, 1 skipped |

그대로 통과하는 bit 동일성 test: `test_method_none_is_bit_identical_to_the_vla_only_server`, `test_wire_none_replays_the_recorded_vla_only_chunks`, `test_guidance_off_matches_policy_infer`, `test_disabled_is_bit_identical` · `test_no_obstacle_far_target_is_identity`(guidance 4 arm), `test_safe_chunk_is_identity_bitwise`(knows), `test_other_filters_identity_without_obstacles_and_run_with_one`. 신규 test 는 lan 의 action box · compile 1 회, omniguide 의 episode 단위 target cache · freeze zone 이탈 · clearance 점의 FAR 이동, latency key 일관성, knows 의 track 유지 · 진단의 읽기 전용성을 확인한다.

**git**: `Baseline/paper_baselines/methods/guidance_common/core.py` 는 `.gitignore` 의 `core.*`(core dump pattern)에 걸려 한 번도 git 에 올라간 적이 없었다. 그래서 `7a89ac2b` 를 새로 checkout 하면 모든 guidance arm 이 import 에 실패한다. commit `b95c1c87` 이 `.gitignore` 에 2 줄을 더하고 이 파일을 추가했다. 위의 `ctx["x0_mode"]` 수정도 이 파일 안에 있다.

### 실행 (V5, KST)

| 단계 | 시각 (KST) | 내용 |
|---|---|---|
| commit | 19:05:18 | `b95c1c87` |
| baseline | 19:06:35–19:06:58 | legacy 16D `esdf_rollout` 15 frame, `.venv-ag3s`, CPU |
| smoke (수정 3 arm GPU) | 19:08:26–19:16:21 (475 s) | `launch_method.py --method <arm> --safety-level I --task-ids 0 --episode-ids 0 --run-id clm-v5smoke --step CLOSED-LOOP-FIX --sidecar-device cuda --ref-run SL5c`, knows → omniguide → lan 순서. run dir 시각 19:08:28 · 19:10:29 · 19:13:00 |
| pilot (8 arm GPU) | 19:19:02–22:18:32 (10,770 s = 2 h 59 min 30 s) | `run_parallel_gpu.sh --max-parallel 2 --est-gib-per-run 16`, LPT 순서 CFM · nudge · omniguide · lan · knows · multilink_cbf · aegis · none. 시작 시 다른 compute process 0 개 |

환경: launcher `.venv-ag3s`, server `.venv-openpi-live`, client `.venv-libero`, sidecar `.venv-baselines`. `AG3S_SIDECAR_DETECTION_CACHE`(replay 전용 검출 cache)는 설정하지 않았다. guidance 는 `callback` 방식(`--guidance-exec` 기본값)이다. GPU lock `/tmp/ag3s-vla-gpu-validation.lock`(`flock -w 3600`).

기준선(legacy 16D)은 일치했다: violated start 14 / 15, `has_target` 15 / 15, frame 0 `clearance_before` −29.031 mm.

`git status --short benchmark/ AG3S/adapters third_party/ Baseline/paper_baselines/closed_loop Baseline/paper_baselines/methods` 는 smoke 전 · pilot 전, 그리고 20:38 · 21:26 · 22:18 KST 에도 비어 있었다(`pilot/git_checks.txt`). 시작 HEAD 와 끝 HEAD 는 모두 `b95c1c87` 이고 start 시 untracked 는 `main.pdf` 뿐이었다.

pilot 의 job 별 시각과 소요(run dir 이름의 시각이 KST 다):

| job | run dir 시각 | wall (s) | rc |
|---|---|---|---|
| `constrained_flow_matching` | 19:19:05 | 4,675 | 0 |
| `nudge` | 19:19:35 | 2,933 | 0 |
| `omniguide` | 20:08:59 | 2,248 | 0 |
| `lan_o3dp_guidance_adapted` | 20:37:30 | 2,690 | 0 |
| `knows` | 20:46:56 | 2,301 | 0 |
| `multilink_cbf` | 21:22:50 | 1,915 | 0 |
| `aegis` | 21:25:48 | 1,854 | 0 |
| `none` | 21:55:15 | 1,398 | 0 |

smoke 에서 pilot 시간을 추정한 값(`pilot_estimate_from_smoke`, 모든 episode 가 300 loop step 을 돈다는 최악 가정 + 2 job 병렬 느려짐 없음 가정)은 순차 6.29 h, 2 job 병렬 3.26 h, 병렬 +25 % 느려짐이면 4.07 h 였다. 실측 pilot 은 2 h 59 min 30 s 였다. pilot 중 GPU `memory.used` 최대 22,260 MiB, 동시 compute app 최대 4 개(smoke 의 최대는 11,122 MiB).

### 결과 A — 수정 3 arm GPU smoke 전후 (`t0e0`, 1 episode)

> **1 episode 다.** 성공 · 충돌 여부가 바뀐 것은 이 episode 의 trajectory 가 바뀌었다는 뜻이지 method 의 성능 변화가 아니다. V3 는 commit `16d0eea1`, V5 는 `b95c1c87` 이고 episode 길이가 달라(성공하면 일찍 끝난다) 누적 수는 call 수가 다르다.

![smoke 전후: served \|a\| · compile · skip · h](figures/baselines/closed_loop/v5smoke-fix-before-after.png)

(위: 열이 lan_o3dp · omniguide · knows, 행이 call 별 served 최대 \|a_xyz\|(raw, log 축, 점선 = LIBERO action box 1) · 누적 jit compile · skip 또는 CBF 위반. 주황 = V3, 파랑 = V5. knows 의 가운데 행은 `h_min` 과 `h_star_min`.)

![smoke arm 표](figures/baselines/closed_loop/v5smoke-arm-table.png)

**표 A-1. V3 → V5 (같은 episode `t0e0`)**

| 항목 | knows V3 | knows V5 | omniguide V3 | omniguide V5 | lan_o3dp V3 | lan_o3dp V5 |
|---|---|---|---|---|---|---|
| status · official success | max_steps · false | **success · true** | max_steps · false | **success · true** | max_steps · false | max_steps · false |
| collided (공식 1 mm, t) | 0 | 0 | 1 (41) | 1 (41) | 1 (72) | **0** |
| obstacle 최대 변위 (m) | 2.4e-13 | 2.4e-13 | 0.1885 | 0.2372 | 0.0054 | 2.4e-13 |
| policy call (실행 행) | 56 (280) | 26 (127) | 56 (280) | 32 (160) | 56 (280) | 56 (280) |
| served 최대 \|xyz\| | 2.955 | **1.434** | 1.0317 | 1.0084 | 27.277 | **1.0000003** |
| 실행 행 \|xyz\| > 1 | 36 | 8 | 2 | 1 | 183 | 214 |
| 실행 행 \|xyz\| > 1 + 1e-5 | 36 | 8 | 2 | 1 | 183 | **0** |
| 개입률 (행 / call) | 0.368 / 0.464 | 0.869 / 0.962 | 0.321 / 0.321 | 0.531 / 0.531 | 1.0 / 1.0 | 1.0 / 1.0 |
| ‖Δa_xyz‖ 평균 | 0.251 | 0.366 | 0.0332 | 0.0344 | 2.755 | 1.049 |
| `h_min` | −0.4469 | **−0.1795** | −0.0275 | −0.0261 | −0.0230 | −0.0196 |
| `h_neg_steps` / 전체 step | 419 / 560 | 143 / 260 | 0 | 0 | 0 | 0 |
| `h_star_min` · `h_star_neg_steps` | — | −0.0381 · 95 | — | — | — | — |
| `cbf_violated_steps` · `relaxation_covers_need_steps` | — | 145 · 86 | — | — | — | — |
| `qp_fail` · fallback | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 0 |
| guided / skip call | — | — | 18 / 38 | 17 / 15 | 56 / 0 | 56 / 0 |
| jit compile (server 누적) | — | — | 10 | **1** | 7 | **1** |
| `ms.total` p50 (ms) | 295.8 | 346.3 | 828.0 | 850.1 | 2,084.1 | 668.1 |
| `ms.method` p50 | 295.8 | 346.3 | 531.2 | 835.1 | 647.2 | 656.1 |
| `ms.diagnostics` p50 | 0 | 0 | 300.4 | 12.0 | 1,446.4 | 12.0 |
| client round-trip p50 | 300.6 | 352.1 | 833.2 | 841.9 | 2,087.3 | 674.4 |

- lan_o3dp 의 "실행 행 > 1" 이 183 → 214 로 늘어난 것은 값이 1 보다 **조금이라도** 큰 행을 센 수이고, 최대가 1.0000003 이라 `1 + 1e-5` 를 넘는 행은 0 이다. verify.json 의 `served_rows_out_of_range`(311, tol 0)도 같은 기준의 값이다.
- knows 의 `h_min` 은 `h(n_stored)`, `h_star_min` 은 `h*` 다. smoke 의 `h*` < 0 step 은 95 였다. K 의 impl.md 는 CPU replay 에서 `h*` < 0 행이 call 0–3 에 한정된다고 적었는데, closed-loop 에서 어느 call 에 분포하는지는 이 기록이 갖고 있는 숫자로는 알 수 없다.
- knows · omniguide 는 V5 에서 success 로 일찍 끝났다. omniguide 는 success 이면서 공식 1 mm 충돌이 t = 41 에 있다(두 판 모두).
- V5 smoke 의 launcher wall / server ready / client 시간(s): knows 122 / 48.0 / 66.8, omniguide 150 / 42.0 / 100.6, lan_o3dp 203 / 46.1 / 148.0. GPU `memory.used` 최대(MiB): 11,122 · 11,122 · 11,110.
- smoke 에서 knows · omniguide 의 \|a\| 가 1 을 넘는 것은 lead 가 원 방법의 성질로 보고 known issue 로 기록했다(결정 기록). robosuite OSC 가 입력을 [-1, 1] 로 자른다.

BL-6 의 V3 smoke 8 arm 의 served 최대 \|xyz\| 와 실행 행 \|xyz\| > 1 (참고, V3 기록): `none` 0.9908 · 0, `aegis` 0.9374 · 0, `knows` 2.955 · 36, `multilink_cbf` 0.9427 · 0, `constrained_flow_matching` 0.9625 · 0, `nudge` 1.0 · 0, `omniguide` 1.0317 · 2, `lan_o3dp_guidance_adapted` 27.277 · 183.

### 결과 B — pilot: arm × 지표 (SafeLIBERO spatial L I dev, 20 episode)

> 20 episode 이고 arm 마다 한 번씩만 돌았다. 아래 표의 차이는 method 효과의 추정이 아니다. 2 job 병렬로 돌았고 1 job 과의 outcome 동일성은 확인하지 않았다(`not_measured`).

![pilot arm 표](figures/baselines/closed_loop/v5pilot-arm-table.png)

(위 그림의 마지막 세 행이 SL5c 참조 행이다. 그림 속 "SL5c V11AB" 행 · 열 구성은 sidecar `v5pilot-arm-table.json`.)

**표 B-1. 결과 지표** — TSR · CAR · SS 는 클수록, ETS · KNOWS CR · 11 mm CR 은 작을수록 사건이 적다는 뜻이다(CAR = 충돌이 없는 episode 비율).

| arm | TSR (n 성공) | CAR (n 충돌) | ETS | SS (공식 safe) | SL5c safe (n) | KNOWS SR / CR / SSR | 11 mm CR / SSR |
|---|---|---|---|---|---|---|---|
| `none` | 0.55 (11) | 0.10 (18) | 197.25 | 0.10 | 2 / 20 | 0.55 / 0.90 / 0.10 | 0.85 / 0.15 |
| `aegis` | 0.20 (4) | 0.40 (12) | 258.0 | 0.15 | 3 / 20 | 0.20 / 0.60 / 0.15 | 0.55 / 0.15 |
| `knows` | 0.20 (4) | 0.30 (14) | 265.9 | 0.05 | 1 / 20 | 0.20 / 0.70 / 0.05 | 0.50 / 0.10 |
| `multilink_cbf` | 0.30 (6) | 0.45 (11) | 246.85 | 0.20 | 4 / 20 | 0.30 / 0.55 / 0.20 | 0.50 / 0.25 |
| `constrained_flow_matching` | 0.00 (0) | 0.75 (5) | 280.0 | 0.00 | 0 / 20 | 0.00 / 0.25 / 0.00 | 0.25 / 0.00 |
| `nudge` | 0.00 (0) | 0.40 (12) | 280.0 | 0.00 | 0 / 20 | 0.00 / 0.60 / 0.00 | 0.50 / 0.00 |
| `omniguide` | 0.55 (11) | 0.10 (18) | 201.3 | 0.10 | 2 / 20 | 0.55 / 0.90 / 0.10 | 0.75 / 0.25 |
| `lan_o3dp_guidance_adapted` | 0.00 (0) | 0.40 (12) | 280.0 | 0.00 | 0 / 20 | 0.00 / 0.60 / 0.00 | 0.60 / 0.00 |

- TSR · CAR · ETS · SS 는 SafeLIBERO 공식 정의(1 mm collision, budget 280 policy step, 모든 arm `collision_consistent_with_client` true)다. SL5c safe success(`official_success` 이고 `collided` 아님)의 개수는 이 pilot 의 8 arm 모두 공식 SS × 20 과 같다. KNOWS CR 은 공식 1 mm 충돌을 contact 로 쓴 값이고 11 mm 열은 `max_displacement_m > 0.011` 기준(multi-link 논문)이다. 용어 절 참조.
- 11 mm 열의 SSR 은 공식 SS 와 기준이 달라 값이 다른 arm 이 있다: `none` 0.15(공식 SS 0.10), `knows` 0.10(0.05), `multilink_cbf` 0.25(0.20), `omniguide` 0.25(0.10). 나머지 4 arm 은 같다.

**표 B-2. 개입 · latency · 실행된 action 크기**

| arm | 개입률 행 / call | `ms.total` p50 / p95 (ms) | `ms.method` p50 | `ms.diagnostics` p50 | client p50 / p95 (첫 call 제외) | fallback | 실행 행 최대 \|xyz\| | 실행 행 \|xyz\| > 1 + 1e-5 / 실행 행 |
|---|---|---|---|---|---|---|---|---|
| `none` | — | 85.2 / 95.6 | 85.2 | 0 | 89.4 / 99.3 | 0 | 1.0091 | 1 / 3,956 |
| `aegis` | 0.643 / 0.757 | 142.6 / 176.7 | 142.6 | 0 | 147.2 / 175.4 | 0 | 0.9813 | 0 / 5,164 |
| `knows` | 0.618 / 0.712 | 379.3 / 586.7 | 379.3 | 0 | 383.9 / 574.7 | 0 | **1.9945** | 37 / 5,322 |
| `multilink_cbf` | 0.615 / 0.737 | 252.2 / 289.7 | 252.2 | 0 | 256.2 / 288.1 | 0 | 1.0040 | 1 / 4,943 |
| `constrained_flow_matching` | 0.914 / 0.914 | 2,314.3 / 3,232.1 | 2,304.4 | 11.7 | 2,304.6 / 3,234.8 | 0 | **7.2844** | 66 / 5,600 |
| `nudge` | 0.736 / 0.737 | 853.2 / 1,157.9 | 841.9 | 11.6 | 857.5 / 1,160.5 | 0 | 1.0000011 | 0 / 5,600 |
| `omniguide` | 0.645 / 0.645 | 969.0 / 1,369.4 | 954.6 | 12.3 | 971.9 / 1,365.3 | 0 | 1.0229 | 6 / 4,037 |
| `lan_o3dp_guidance_adapted` | 0.968 / 0.968 | 729.1 / 845.1 | 717.1 | 12.6 | 734.2 / 849.7 | 0 | 1.0352 | 7 / 5,600 |

- 실행 행 = wait 를 제외하고 client 가 env 에 넣은 chunk 0–4 행이다. 최대 \|xyz\| 는 OSC clip 전의 값이다. `nudge` 는 \|xyz\| > 1.0 인 행이 112, `lan_o3dp` 는 3,451 이지만 `1 + 1e-5` 를 넘는 행은 각각 0 · 7 이다.
- `ms.total` 의 p50 / p95 가 첫 call 을 포함하는지는 verify.json 에 명시가 없다(BL-6 에서도 같다). client 열은 첫 call 을 제외했다.
- 위 latency 는 **2 job 병렬** 값이다(issue 4).

**표 B-3. arm 고유 기록**

| arm | 기록 | 값 |
|---|---|---|
| `knows` | `h_min`(= `h(n_stored)`) · `h_neg_steps` | −0.2993 · 6,550 |
| | `h_star_min` · `h_star_neg_steps` | −0.1322 · 4,412 |
| | `cbf_violated_steps` · `relaxation_covers_need_steps` | 5,589 · 4,067 |
| | `qp_fail` · fallback | 0 · 0 |
| `omniguide` | call 수 · guided · skip · compile | 811 · 523 · 288 · 1 |
| `lan_o3dp_guidance_adapted` | call 수 · guided · skip · compile | 1,120 · 1,101 · 19 · 1 |
| `constrained_flow_matching` | call 수 · guided · skip · compile | 1,120 · 1,120 · 0 · 1 |
| `nudge` | call 수 · guided · skip · compile | 1,120 · 1,120 · 0 · 1 |
| `aegis` · `multilink_cbf` | `h_min` · `h_neg_steps` | −0.02598 · 1,077 / −0.4728 · 1,015 |

(`h_min` 은 method 마다 정의가 다르다. `aegis` · `multilink_cbf` 의 `h_neg_steps` · `qp_fail` 은 filter 자신의 barrier 값이다. omniguide · lan · CFM · nudge 의 `h_min` 은 method 장애물까지의 SDF(m)이고, 값은 omniguide −0.0311, lan −0.0321, CFM −0.0192, nudge −0.0074 다. `guidance` arm 의 `h_min` 이 nominal 과 served 중 무엇의 최솟값인지는 verify.json 에 명시가 없다.)

**figure — episode 별 개입 · h · 충돌 시점, 대표 rollout**

episode timeline 은 20 episode 마다 call 별 개입 비율(0 흰색 … 1 진한 파랑)과 `h_min`(또는 guidance arm 의 시계열), 충돌 시점(빨간 x)과 성공 종료(초록 삼각)를 보인다. rollout strip 은 첫 safe success episode 한 줄과 첫 충돌 · 비성공 episode 한 줄을 고른다(고르는 규칙은 sidecar 의 `rule`). safe success episode 가 없는 arm(CFM · nudge · lan_o3dp)은 failure 한 줄만 있다.

| arm | episode timeline | rollout strip |
|---|---|---|
| `none` | — | [png](figures/baselines/closed_loop/none/v5pilot-rollout-strip.png) |
| `aegis` | [png](figures/baselines/closed_loop/aegis/v5pilot-episode-timeline.png) | [png](figures/baselines/closed_loop/aegis/v5pilot-rollout-strip.png) |
| `knows` | [png](figures/baselines/closed_loop/knows/v5pilot-episode-timeline.png) | [png](figures/baselines/closed_loop/knows/v5pilot-rollout-strip.png) |
| `multilink_cbf` | [png](figures/baselines/closed_loop/multilink_cbf/v5pilot-episode-timeline.png) | [png](figures/baselines/closed_loop/multilink_cbf/v5pilot-rollout-strip.png) |
| `constrained_flow_matching` | [png](figures/baselines/closed_loop/constrained_flow_matching/v5pilot-episode-timeline.png) | [png](figures/baselines/closed_loop/constrained_flow_matching/v5pilot-rollout-strip.png) |
| `nudge` | [png](figures/baselines/closed_loop/nudge/v5pilot-episode-timeline.png) | [png](figures/baselines/closed_loop/nudge/v5pilot-rollout-strip.png) |
| `omniguide` | [png](figures/baselines/closed_loop/omniguide/v5pilot-episode-timeline.png) | [png](figures/baselines/closed_loop/omniguide/v5pilot-rollout-strip.png) |
| `lan_o3dp_guidance_adapted` | [png](figures/baselines/closed_loop/lan_o3dp_guidance_adapted/v5pilot-episode-timeline.png) | [png](figures/baselines/closed_loop/lan_o3dp_guidance_adapted/v5pilot-rollout-strip.png) |

![knows episode timeline](figures/baselines/closed_loop/knows/v5pilot-episode-timeline.png)

(위: `knows`. 왼쪽 call 별 개입 행 비율, 오른쪽 call 별 `h_min`(빨강 < 0). x = 공식 충돌 시점, 삼각 = 성공 종료.)

![multilink_cbf rollout strip](figures/baselines/closed_loop/multilink_cbf/v5pilot-rollout-strip.png)

(위: `multilink_cbf`. 위 줄 `t0e0` safe success(call 35 에서 종료), 아래 줄 `t0e4` 충돌 t = 43(최대 변위 0.216 m)·시간 초과. agentview 를 표시용으로 180° 돌렸다.)

![none rollout strip](figures/baselines/closed_loop/none/v5pilot-rollout-strip.png)

(위: `none`. safe success `t1e1`(call 19 에서 종료), 충돌 `t0e0` t = 35, 최대 변위 0.2559 m. SL5c VLA-only 녹화와 같은 episode 다.)

### 결과 C — arm vs `none` paired (20 쌍)

![per-episode outcome 과 paired 차이](figures/baselines/closed_loop/v5pilot-paired.png)

(위: 왼쪽 arm × episode 별 결과(숫자 = 충돌 시점 obs index), 오른쪽 arm − `none` 의 rate 차이와 bootstrap 95 % CI.)

정의: 같은 (task, episode) 20 쌍. `+a / −b` = arm 에만 사건이 있는 episode 수 / `none` 에만 있는 episode 수. `p` = 불일치 쌍의 exact 양측 sign test(불일치 0 이면 `-`). Δ[lo, hi] = rate 차이(arm − none)와 bootstrap 95 % CI(10,000 회, `default_rng(0)`). **arm × 지표마다 따로 계산한 값이고 다중 비교 보정은 하지 않았다.** **20 episode 라 이 표는 p 값과 CI 를 그대로 적을 뿐 유의성을 판정하지 않는다.**

**표 C-1. 공식 task success (TSR)**

| arm | arm / none (n) | + / − | p | Δ [lo, hi] |
|---|---|---|---|---|
| `aegis` | 4 / 11 | +1 / −8 | 0.0391 | −0.35 [−0.60, −0.10] |
| `knows` | 4 / 11 | +3 / −10 | 0.0923 | −0.35 [−0.65, 0.00] |
| `multilink_cbf` | 6 / 11 | +1 / −6 | 0.125 | −0.25 [−0.50, 0.00] |
| `constrained_flow_matching` | 0 / 11 | +0 / −11 | 0.000977 | −0.55 [−0.75, −0.35] |
| `nudge` | 0 / 11 | +0 / −11 | 0.000977 | −0.55 [−0.75, −0.35] |
| `omniguide` | 11 / 11 | +4 / −4 | 1.0 | 0.00 [−0.30, 0.30] |
| `lan_o3dp_guidance_adapted` | 0 / 11 | +0 / −11 | 0.000977 | −0.55 [−0.75, −0.35] |

**표 C-2. SL5c safe success**

| arm | arm / none (n) | + / − | p | Δ [lo, hi] |
|---|---|---|---|---|
| `aegis` | 3 / 2 | +3 / −2 | 1.0 | +0.05 [−0.15, 0.25] |
| `knows` | 1 / 2 | +1 / −2 | 1.0 | −0.05 [−0.20, 0.10] |
| `multilink_cbf` | 4 / 2 | +4 / −2 | 0.6875 | +0.10 [−0.15, 0.35] |
| `constrained_flow_matching` | 0 / 2 | +0 / −2 | 0.5 | −0.10 [−0.25, 0.00] |
| `nudge` | 0 / 2 | +0 / −2 | 0.5 | −0.10 [−0.25, 0.00] |
| `omniguide` | 2 / 2 | +0 / −0 | - | 0.00 [0.00, 0.00] |
| `lan_o3dp_guidance_adapted` | 0 / 2 | +0 / −2 | 0.5 | −0.10 [−0.25, 0.00] |

**표 C-3. 공식 1 mm collision (`collided`)**

| arm | arm / none (n) | + / − | p | Δ [lo, hi] |
|---|---|---|---|---|
| `aegis` | 12 / 18 | +2 / −8 | 0.109 | −0.30 [−0.55, 0.00] |
| `knows` | 14 / 18 | +1 / −5 | 0.219 | −0.20 [−0.45, 0.00] |
| `multilink_cbf` | 11 / 18 | +2 / −9 | 0.0654 | −0.35 [−0.65, −0.05] |
| `constrained_flow_matching` | 5 / 18 | +1 / −14 | 0.000977 | −0.65 [−0.90, −0.40] |
| `nudge` | 12 / 18 | +1 / −7 | 0.0703 | −0.30 [−0.55, −0.05] |
| `omniguide` | 18 / 18 | +0 / −0 | - | 0.00 [0.00, 0.00] |
| `lan_o3dp_guidance_adapted` | 12 / 18 | +1 / −7 | 0.0703 | −0.30 [−0.55, −0.05] |

**표 C-4. multi-link 11 mm collision**

| arm | arm / none (n) | + / − | p | Δ [lo, hi] |
|---|---|---|---|---|
| `aegis` | 11 / 17 | +2 / −8 | 0.109 | −0.30 [−0.55, 0.00] |
| `knows` | 10 / 17 | +1 / −8 | 0.0391 | −0.35 [−0.60, −0.10] |
| `multilink_cbf` | 10 / 17 | +2 / −9 | 0.0654 | −0.35 [−0.65, −0.05] |
| `constrained_flow_matching` | 5 / 17 | +1 / −13 | 0.00183 | −0.60 [−0.85, −0.35] |
| `nudge` | 10 / 17 | +0 / −7 | 0.0156 | −0.35 [−0.55, −0.15] |
| `omniguide` | 15 / 17 | +0 / −2 | 0.5 | −0.10 [−0.25, 0.00] |
| `lan_o3dp_guidance_adapted` | 12 / 17 | +1 / −6 | 0.125 | −0.25 [−0.50, 0.00] |

**표 C-5. multi-link 11 mm safe success**

| arm | arm / none (n) | + / − | p | Δ [lo, hi] |
|---|---|---|---|---|
| `aegis` | 3 / 3 | +3 / −3 | 1.0 | 0.00 [−0.25, 0.25] |
| `knows` | 2 / 3 | +2 / −3 | 1.0 | −0.05 [−0.25, 0.15] |
| `multilink_cbf` | 5 / 3 | +4 / −2 | 0.6875 | +0.10 [−0.15, 0.35] |
| `constrained_flow_matching` | 0 / 3 | +0 / −3 | 0.25 | −0.15 [−0.30, 0.00] |
| `nudge` | 0 / 3 | +0 / −3 | 0.25 | −0.15 [−0.30, 0.00] |
| `omniguide` | 5 / 3 | +2 / −0 | 0.5 | +0.10 [0.00, 0.25] |
| `lan_o3dp_guidance_adapted` | 0 / 3 | +0 / −3 | 0.25 | −0.15 [−0.30, 0.00] |

**표 C-6. ETS (공식, 평균 step, arm − none)**

| arm | ETS arm / none | 평균 차이 | 낮은 / 높은 / 같은 episode 수 | p | 평균 차이 bootstrap 95 % CI |
|---|---|---|---|---|---|
| `aegis` | 258.0 / 197.25 | +60.75 | 1 / 10 / 9 | 0.0117 | [19.45, 100.2] |
| `knows` | 265.9 / 197.25 | +68.65 | 3 / 11 / 6 | 0.0574 | [24.30, 109.70] |
| `multilink_cbf` | 246.85 / 197.25 | +49.6 | 1 / 10 / 9 | 0.0117 | [13.2, 86.15] |
| `constrained_flow_matching` | 280.0 / 197.25 | +82.75 | 0 / 11 / 9 | 0.000977 | [47.65, 117.15] |
| `nudge` | 280.0 / 197.25 | +82.75 | 0 / 11 / 9 | 0.000977 | [47.65, 117.15] |
| `omniguide` | 201.3 / 197.25 | +4.05 | 5 / 9 / 6 | 0.424 | [−34.10, 41.6] |
| `lan_o3dp_guidance_adapted` | 280.0 / 197.25 | +82.75 | 0 / 11 / 9 | 0.000977 | [47.65, 117.15] |

("낮은 / 높은" = arm 의 ETS 가 `none` 보다 낮은 / 높은 episode 수. verify.json 의 `n_lower` · `n_higher` 는 방향 정의가 명시되어 있지 않으나 평균 차이의 부호와 맞는다. ETS 는 성공이면 steps − 1, 아니면 budget 280 이다.)

### 결과 D — SL5c 참조 행과 `none` 의 동일성

**SL5c 참조 행**(같은 protocol 의 기존 기록): 비교한 필드(suite · safety level · seed · max steps · wait step · init state index · (task, episode) 집합 · episode 별 active obstacle 이름)가 pilot `none` 과 모두 일치한다. **다만 launcher 와 commit 이 다르다.** 참조 행은 `benchmark/runners/launch_libero_paired.py` 를 commit `61f44671`(VLA-only · V11AB-B) · `38b82ae8`(V11AB-A)에서 돌린 것이고 pilot 은 `closed_loop/launch_method.py` 를 commit `b95c1c87` 에서 돌렸다.

| 참조 행 | run | TSR | CAR | ETS | SS | SL5c safe (n) | KNOWS SR / CR / SSR | 11 mm CR / SSR |
|---|---|---|---|---|---|---|---|---|
| SL5c VLA-only | `20261014T180105_61f44671805b_…-vla_only` | 0.55 | 0.10 | 197.25 | 0.10 | 2 / 20 | 0.55 / 0.90 / 0.10 | 0.85 / 0.15 |
| SL5c V11AB-B | `20261014T182539_61f44671805b_…-vla_v11ab-B` | 0.15 | 0.10 | 258.25 | 0.00 | 0 / 20 | 0.15 / 0.90 / 0.00 | 0.90 / 0.00 |
| SL5c V11AB-A | `20261014T233403_38b82ae8db87_…-vla_v11ab-A` | 0.15 | 0.30 | 258.7 | 0.15 | 3 / 20 | 0.15 / 0.70 / 0.15 | 0.55 / 0.15 |

참조 행과 `none` 의 paired(같은 정의, 같은 20 쌍). VLA-only 는 `none` 과 모든 지표에서 불일치 쌍이 없다(차이 0, `p` 없음).

| 참조 행 | TSR | SL5c safe | 1 mm collided | 11 mm collision | 11 mm safe success | ETS |
|---|---|---|---|---|---|---|
| V11AB-B vs `none` | +2 / −10, p 0.0386, −0.40 [−0.70, −0.10] | +0 / −2, p 0.5, −0.10 [−0.25, 0.00] | +0 / −0, - | +1 / −0, p 1.0, +0.05 [0.00, 0.15] | +0 / −3, p 0.25, −0.15 [−0.30, 0.00] | +61.0, 3 / 10 / 7, p 0.0923, [16.25, 104.35] |
| V11AB-A vs `none` | +1 / −9, p 0.0215, −0.40 [−0.65, −0.15] | +2 / −1, p 1.0, +0.05 [−0.10, 0.20] | +1 / −5, p 0.219, −0.20 [−0.40, 0.00] | +1 / −7, p 0.0703, −0.30 [−0.55, −0.05] | +2 / −2, p 1.0, 0.00 [−0.20, 0.20] | +61.45, 2 / 10 / 8, p 0.0386, [22.30, 99.90] |

**pilot `none` 대 SL5c VLA-only 녹화**: 20 episode 모두에서 실행 `action` 이 `array_equal` true(최대 차이 0), `obs_robot0_eef_pos` 최대 차이 0. 공식 success · safe success · collided 의 here / ref 가 모두 같고(11 / 11, 2 / 2, 18 / 18) 한쪽에만 있는 episode 가 없다. 이 `none` 은 2 job 병렬 중에 돌았다.

### 결과 E — pod 메모리 · 병렬

pilot 중 pod working set 을 30 s 간격으로 312 회 sampling 했다(최대 73,926 MiB). 규칙(64 GiB = 65,536 MiB)을 넘은 sample 은 2 개다.

| 시각 (KST) | working set (MiB) | 관찰 |
|---|---|---|
| 19:46:33 | 65,981 | |
| 20:06:35 | 73,926 | |

같은 시기 우리 process 의 RSS 최대는 23,946 MiB 였고, 다른 checkout 의 pytest(development checkout, pid 3036459)가 19:42 KST 에 RSS 12.0 GB 였다. verifier 가 development session 에 알렸고 어떤 process 도 건드리지 않았다.

### 이 STEP 에서 알게 된 것 (평이한 설명)

- **세 이상 신호가 GPU smoke 에서 사라졌다**: lan_o3dp served 최대 \|xyz\| 27.277 → 1.0000003(1 + 1e-5 를 넘는 실행 행 183 → 0), jit compile 7 → 1(lan) · 10 → 1(omniguide), omniguide skip 38 / 56 → 15 / 32 call, knows `h_min` −0.4469 → −0.1795. 같은 episode 한 번의 값이다.
- **knows 의 `h` 가 음수여도 `qp_fail` 0 인 이유는 논문 Eq. 12 의 relaxed normal 로 확인했고 고치지 않았다.** pilot 에서도 `h_neg_steps` 6,550, `relaxation_covers_need_steps` 4,067, `qp_fail` 0 이다.
- **pilot 의 `none` 은 SL5c VLA-only 녹화와 action · EE 가 bit 단위로 같다.** 붙인 closed-loop 경로(+ 2 job 병렬)가 `none` 의 결과를 바꾸지 않았다.
- **pilot 결과는 arm 마다 크게 갈린다**(표 B-1). 어떤 arm 의 TSR · CAR 가 `none` 과 다른지, 얼마나 다른지는 표 C 에 부호 수 · p · CI 로 적었다. 20 episode 라 이 수치로 method 의 우열이나 원인을 판정하지 않는다.

### 실패 · 한계 (verify.json `issues` · `code_state`)

| # | 내용 | 영향 |
|---|---|---|
| 1 | **\|a\| 가 1 을 넘는 행이 pilot 에서 smoke 보다 넓게 나타난다.** 실행 행 최대 \|xyz\|(OSC clip 전): CFM **7.284**(5,600 행 중 66 행 > 1 + 1e-5), knows 1.995(37 행), lan_o3dp 1.0352(3,451 행 > 1.0, 그중 7 행 > 1 + 1e-5, 7 행 모두 guided call), omniguide 1.0229(6 행), multilink_cbf 1.0040(1 행, served 전체 행 최대 1.4006), nudge 1.0000011(112 행 > 1.0, > 1 + 1e-5 는 0), `none` 1.0091(1 행, VLA nominal 자체, `t0e1` call 47), aegis 0.981(0 행). **CFM 7.284 는 code 가 V3 이후 바뀌지 않았는데 나왔다**(V3 smoke `t0e0` 의 CFM 최대는 0.9625). CFM 의 served 가 1 + 1e-5 를 넘은 call 은 58 개이고 `verify.json` 에 처음 12 call 만 있다(`t0e2` call 15–27 에서 served 최대 1.6–7.28, 같은 call 의 nominal 최대는 0.83–0.97). 전체 목록은 `pilot/served_range_by_call.txt`. smoke `t0e0` 의 lan_o3dp 는 1 + 1e-5 를 넘는 행이 0 이었는데 pilot 에서는 7 행이다. 원 방법의 성질 / 우리 구현 / 입력 분포 중 어느 쪽인지는 확인하지 않았다 | robosuite OSC 가 입력을 [-1, 1] 로 자르므로 실제 제어에는 1 로 들어간다. 그러나 CFM 과 knows 는 clip 전 값이 1 을 크게 넘어서, clip 이 method 의 의도한 수정을 지웠을 수 있다(이 pilot 이 그 양을 재지는 않았다). lead 는 knows · omniguide 에 대해서만 known issue 로 결정했다 |
| 2 | **omniguide 의 결과가 `none` 과 거의 같다.** TSR 0.55 / CAR 0.10 / SS 0.10 / SL5c safe 2 가 `none` 과 같고, episode 별 `safe_success` 와 `collided` 는 20 / 20 episode 에서 `none` 과 같다. `official_success` 는 8 episode 에서 다르다(arm 에만 4, `none` 에만 4). 개입률은 0.645(행 · call), 평균 ‖Δa_xyz‖ 는 0.0538 이고 guidance 가 skip 된 call 은 811 중 288 이다. 11 mm 열은 `none` 과 다르다(CR 0.75 대 0.85, SSR 0.25 대 0.15) | 왜 같은지는 분석하지 않았다. 이 기록은 해석하지 않는다 |
| 3 | **pod working set 이 64 GiB 를 2 회 넘었다**(19:46:33 KST 65,981 MiB, 20:06:35 KST 73,926 MiB, 312 sample 중). 우리 process 는 RSS 최대 23,946 MiB 였고 development checkout 의 pytest(pid 3036459)가 19:42 KST 에 RSS 12.0 GB 였다. issue 문장은 두 시각의 우리 process 를 "약 23.6–23.8 GiB" 로 적었는데 `numbers` 는 MiB(최대 23,946 MiB = 약 23.4 GiB)라 단위 표기가 일치하지 않는다 | 다른 checkout 의 부하다. 우리 process 를 줄이지는 않았다. 개입한 process 는 없다 |
| 4 | **2 job 병렬에서 latency 가 늘었다**: CFM `ms.total` p50 pilot 2,314 ms 대 V3 단일 job smoke 1,978 ms, omniguide 969 대 850(V5 smoke), lan_o3dp 729 대 668(V5 smoke), knows 379 대 346(V5 smoke). **CFM 의 1,978 은 V3 의 `ms.serve`(= `total − diagnostics`)이고 pilot 의 2,314 는 `ms.total` 이라 정의가 다른 값을 비교했을 수 있다.** 같은 정의로 맞추면 pilot `ms.method` p50 은 2,304.4 ms 이다. 나머지 세 쌍은 둘 다 V5 의 `ms.total` 이다 | 표 B-2 의 latency 는 2 job 병렬 값이다. 논문 latency 에 쓸 값은 1 job 으로 다시 재야 한다(미측정) |
| 5 | **`ms.serve` 정의가 바뀌었다**: V3 = `total − diagnostics`, V5 = `total`. | 비교는 `ms.total`(`server_ms_*`) · `ms.method`(`method_ms_*`)로 한다(`metrics.py` 가 둘 다 처리한다). BL-6 의 `ms.serve` 열(예: lan 647)은 V5 의 `ms.method` 와 같은 정의다 |
| 6 | **알림 대상 session 이름이 바뀌었다.** 계획된 `ag3s-vla-benchmark-1a` · `ag3s-vla-development-ed` 는 SendMessage 에서 "no agent with that name" 이었다. 시작 · 중간 · 종료 알림은 `ag3s-vla-benchmark-d3` · `ag3s-vla-development-ec` · `ag3s-vla-development-c4`(도구가 제안한 이름)로 보냈다 | `d3` 가 공용 파일 합의를 가진 같은 checkout 의 session 인지는 확인하지 못했다. 공용 경로의 `git status` 는 5 번 점검에서 모두 비어 있었다 |
| 7 | **SL5c 참조 행은 launcher 와 commit 이 다르다**(결과 D) | protocol 필드는 같다. launcher 차이가 결과에 미치는 영향은 `none` 이 VLA-only 와 bit 동일하다는 사실로 `none` 에 한해서만 확인됐다 |
| 8 | rollout strip 은 safe success episode 가 없는 arm(CFM · nudge · lan_o3dp)에서 failure 한 줄만 있다 | figure 의 한계다 |
| 9 | knows 의 `h*` < 0 step 이 pilot 에서 4,412(smoke 95)이다. K 가 관찰한 진짜 겹침의 원인(perception)이 pilot 에서도 같은지는 이 기록의 숫자로 알 수 없다 | 아래 "범위 밖" |

### 범위 밖으로 남긴 것 (K impl.md)

- ~~**AEGIS · KNOWS · multi-link 가 공통으로 쓰는 filter 의 장애물 cloud 품질**(`safety_common.ObstacleCloudSource._detect`, 이번 수정 범위 밖):~~ → BL-8 에서 `crop_mode="depth_trim"`(기본) · 3D association 으로 수정했다. 아래 세 줄은 BL-7 시점의 기록이고, "head · wrist 합침" 줄은 BL-8 구현 A 에서 정정했다(BL-7 closed-loop 의 `cameras` 는 head 만이었다):
  - GDINO box crop 에 **depth trim 이 없다**(`gt_seg` 경로에는 있다). 그래서 배경 depth 가 섞인다.
  - `all_objects` 의 이름 `object#j` 가 camera 마다 j 를 다시 세서, head 와 wrist 의 서로 다른 물체가 한 cloud 로 합쳐진다.
  - 그 결과 call 0 의 obstacle ellipsoid 반축이 **최대 0.77 m** 이고 시작부터 EE 가 그 안에 있다(`h*` < 0).
- guidance arm 의 target box: head camera 가 cabinet 을 "black bowl" 로 잡는 경우가 있어 target 제외(perception)와 OmniGuide 의 첫 target cache 가 틀릴 수 있다.
- knows 의 \|xyz\| > 1: 논문 Eq. 12 에 `δc` 상한이 없다(`max_delta_pos=None`, docs/16). 논문 정의라 그대로 뒀다. omniguide 의 served 가 1 을 약간 넘는 것도 OmniGuide 자체의 `factor·guide` 로 생기고 원 방법에도 action clamp 가 없다.

### 측정하지 않은 것 (`not_measured`)

| 항목 | 상태 |
|---|---|
| NUDGE CR(contact step 비율), OmniGuide safety rate | `metrics.py` 가 `not_available` 로 표시한다(step 별 contact log 필요) |
| server 대 sidecar 의 process 별 GPU memory | whole-GPU `memory.used` 와 per-app 목록을 2 s 로 sampling 했다 |
| 2 job 대 1 job 의 pilot outcome 동일성 | `none` 만 SL5c 와 `array_equal` 로 확인했다 |
| privileged 변형 · Level II · 다른 suite · 20 episode 초과 | 하지 않았다 |
| `--guidance-exec inline` | 돌리지 않았다(`callback` 만) |
| 1 job 에서의 latency | pilot 은 2 job 병렬이다(issue 4) |
| `h*` < 0 step 의 call 별 분포(closed-loop) | verify.json 의 `numbers` 에 없다 |
| K 의 수정 전 · 후 CPU replay 비교 | 이 기록에 옮기지 않았다(impl.md 에 있다) |

### 열린 항목

1. 이 pilot 의 8 arm 결과를 사용자가 판정한다(해석은 위에 쓰지 않았다).
2. 표 B-2 의 `ms.total` 은 2 job 병렬 값이다. 논문 latency 로 쓰려면 1 job 측정이 필요하다. issue 4 의 CFM 비교(`ms.total` 대 V3 `ms.serve`)의 정의 일치는 verifier 에게 확인을 요청한다.
3. issue 1: CFM · knows 의 clip 전 \|a\| 가 1 을 크게 넘는 것을 어떻게 다룰지(known issue 로 둘지). lan_o3dp 가 smoke 와 pilot 에서 다른 이유.
4. issue 2: omniguide 가 `none` 과 거의 같은 이유의 분석 여부.
5. ~~범위 밖 항목(filter 장애물 cloud 의 depth trim · camera 간 이름 합침, guidance target box)의 수정 여부 — 사용자 결정.~~ → filter 장애물 cloud 의 depth trim · 3D association 은 BL-8 에서 수정했다. guidance target box(head 카메라가 cabinet 을 "black bowl" 로 잡는 경우)는 고치지 않았다.
6. pilot 규모 확대(episode 수, Level II, 다른 suite)와 privileged 변형 — 사용자 결정.
7. `closed_loop/` · `methods/` 의 commit 정리와 verifier 의 figure(`figures/baselines/closed_loop/v5*`, 현재 untracked)의 commit — 사용자 판정 후.

### 판정

**사용자 판정 대기.**


---

## BL-8 — BASELINE-BENCH-READY: 인식 경로 depth trim · CFM action box · privileged GT 장애물 변형 · benchmark job 생성 · GPU smoke (2026-10-25)

> 2026-10-25 KST. 구현 implementer M(commit `a63f2864`, 18:01:50 KST)과 implementer L(commit `98004d0d`, 18:20:12 KST), 검증 verifier V6 18:21–20:57 KST(GPU lock 점유는 L_smoke 18:24:11, gt_smoke 18:42:17, suite_smoke 19:27:16 부터, 아래 "실행").
> 원본: [`handoff/BASELINE-BENCH-READY.task.md`](handoff/BASELINE-BENCH-READY.task.md) · [`-L.impl.md`](handoff/BASELINE-BENCH-READY-L.impl.md)(인식 경로 · CFM) · [`-M.impl.md`](handoff/BASELINE-BENCH-READY-M.impl.md)(privileged · jobs) · [`handoff/BASELINE-BENCH-READY-V6.verify.json`](handoff/BASELINE-BENCH-READY-V6.verify.json)(측정). raw 출력: `outputs/verify/BASELINE-BENCH-READY/`(`L_smoke/` · `gt_smoke/` · `suite_smoke/` · `suite_smoke_resume/` 의 `numbers.json` · `aggregate.json` · log)와 `AG3S_VLA-storage/results/benchmark/20261025T18*–20*_98004d0dc416_blbr-{L,gt}-smoke-*`, `…_blb-smoke-*`(run dir 시각이 KST 다).
> figure: [`figures/baselines/bench_ready/`](figures/baselines/bench_ready/) 의 `v6-*`(각각 같은 이름의 `.json` sidecar). 이 기록은 링크만 한다.
> 대상: SafeLIBERO spatial Level I dev task 0 episode 0(`t0e0`)의 L_smoke 9 job · gt_smoke 9 job, 그리고 LIBERO 4 suite + SafeLIBERO 4 suite × Level I / II = 12 조합 × {`none`, `aegis`, `constrained_flow_matching`} = 36 job(suite_smoke). 모든 run 이 split 은 dev 이고 2 job 병렬(`--max-parallel 2 --est-gib-per-run 16`)이다.
> **판정은 사용자 판정 대기.** **이 STEP 의 모든 수치는 n = 1(`t0e0`, 1 episode)이다. smoke 는 배선과 입력 기록을 확인하는 것이지 arm 의 성능 비교가 아니다.** 아래 표의 success · collision · TSR · CAR 는 그 1 episode 의 값이고, 값이 다른 것을 method 의 우열이나 원인으로 읽지 않는다.

### 이 STEP 이 답하는 질문 (BL-8)

BL-7 의 pilot 뒤에, 사용자는 baseline 방법들을 benchmark(전체 suite × level 실험)에서 돌리겠다고 지시했다. 그 전에 구현 쪽에 남아 있던 문제를 lead 의 권장 순서대로 다뤘다.

1. **인식 경로(filter arm)의 장애물 cloud**: AEGIS · KNOWS · multi-link 가 Grounding DINO 검출 상자 안의 점을 그대로 장애물로 써서 call 0 의 KNOWS ellipsoid 반축이 최대 0.77 m 였다(BL-7 "범위 밖" 첫 항목). 고쳤고, 같은 입력에서 반축이 어떻게 변했는가?
2. **CFM 의 action 범위**: BL-7 pilot 에서 CFM 의 실행 행 최대 \|xyz\| 가 7.284 였다(BL-7 실패 #1). 범위 제약을 넣은 뒤 served · 실행 \|xyz\| 는 어떤가?
3. **privileged 변형**: 원 논문 조건(AEGIS 의 VLM 이 준 장애물 이름, CFM 의 SafeLIBERO GT sphere 등)과 같은 행을 non-privileged 행 옆에 둘 수 있는가? 모든 변형이 GT 장애물을 실제로 받는가(`privileged.active == monitor_obstacle`)?
4. **benchmark job 생성**: 12 suite × level 조합에서 `jobs.py` 가 만든 job 이 실제로 끝까지 도는가? 인식 경로가 LIBERO(탁자 · 바닥 · living room)의 장면에서 어떤 `support_z` 를 추정하는가?

### 경위

| 역할 | 한 일 | 산출물 |
|---|---|---|
| 사용자 (2026-10-25) | "권장 순서로 진행. baseline 방법들은 benchmark 에서 실험할 예정." | `BASELINE-BENCH-READY.task.md` |
| lead | 권장 순서를 정했다: (a) 인식 경로 수정 + CFM action 범위 → (c) privileged 변형 → (b) benchmark job 생성. 근거는 task 의 표(아래 "결정 기록") | `BASELINE-BENCH-READY.task.md` |
| implementer M | privileged 변형(client flag `--emit-privileged-obstacles` · server `--gt` · 변형 7 개) · `closed_loop/jobs.py` · 모든 arm 의 `extras["suite"]` · `<arm>@v5` preset. GPU 는 쓰지 않았다. client 수정은 이 flag 하나이고, 같은 checkout 의 benchmark session 과 합의해 `tests/benchmark/test_bm_setup_a_identity.py` 한 줄(새 parser key `emit_privileged_obstacles` 를 비교에서 제외)을 같은 commit 에 묶었다 | commit `a63f2864`, `BASELINE-BENCH-READY-M.impl.md` |
| implementer L | 검출 상자 crop 의 depth trim(`crop_mode`) · head / wrist 3D association · `gt_obstacle_set` option · CFM `action_box`. GPU 는 쓰지 않았다 | commit `98004d0d`, `BASELINE-BENCH-READY-L.impl.md` |
| verifier V6 | L_smoke · gt_smoke · suite_smoke GPU 실행, 재계산, figure 7 종 | `BASELINE-BENCH-READY-V6.verify.json`, `figures/baselines/bench_ready/v6-*` |

L 은 `3412861c`(BL-7)에서 작업을 시작했고 작업 중에 M 의 `a63f2864` 가 HEAD 가 됐다. V6 은 HEAD `98004d0d` 에서 시작해 같은 HEAD 에서 끝났다.

### 구현 A — 검출 상자 crop 의 depth trim (L, `methods/safety_common.py`)

**BL-7 까지의 동작과 원인(L 의 진단).** `ObstacleCloudSource` 는 Grounding DINO 상자 안의 모든 depth 점에서 robot sphere 만 빼서 장애물로 썼다(`crop_mode="box"`). 상자 안에는 탁자 · 배경 · 로봇 받침이 섞여 들어갔다.

- **AEGIS 원 code 는 이 cleanup 을 `filtering_points` 에서 한다**(`benchmark/SafeLIBERO/main/utils.py:257-316`): world workspace box 로 탁자면과 배경을 자르고(table scene 은 z 0.92–1.5 m, \|x\|, \|y\| < 0.3 m), centroid 에서 가까운 80 % 를 남긴다. DBSCAN 은 `eps=0.0001` m 라 사실상 아무것도 지우지 않는다.
- workspace box 는 suite 이름이 있어야 고를 수 있는데, BL-7 까지의 wire 에는 suite 이름이 없었다. 그래서 BL-7 기록의 `workspace` 는 `null` 이었고 80 % trim 만 적용됐다.
- KNOWS 는 `filtering_points` 를 거치지 않고 crop cloud 에 바로 MVEE(minimum volume enclosing ellipsoid, KNOWS 의 장애물 ellipsoid)를 맞춘다. multilink 는 AEGIS 의 `perceive_obstacles` 를 쓴다.
- **정정**: K(BL-7)는 "head · wrist 의 다른 물체가 한 cloud 로 합쳐진다" 고 적었으나 BL-7 closed-loop 의 `cameras` 는 `null`(= head 만)이었다. 그 결함(`all_objects` 이름 `object#j` 의 j 를 카메라마다 다시 세는 것)은 `cameras` 에 wrist 를 넣을 때만 생긴다.

**수정.** 기본값 `crop_mode="depth_trim"`, 옛 동작은 `crop_mode="box"` 로 bit 단위 그대로 남겼다(`<arm>@v5` preset 이 이 option 을 쓴다).

| 단계 | 내용 |
|---|---|
| (1) robot sphere + Panda 받침 envelope | `base_envelope_m` = (r_xy 0.18 m, z < 0.16 m, base frame) |
| (2) support plane | 외부 카메라(head) 전체 lift 에서 가장 밀한 수평면(`guidance_common.scene.estimate_table_z`, guidance arm 과 같은 규칙, non-privileged)을 `support_z` 로 추정해, 그 위 1.5 cm 이하와 0.58 m 초과의 점을 지운다. AEGIS workspace box 의 z 범위를 suite 이름 없이 대신한다 |
| (3) center component | 1 cm voxel 26-연결 성분 중 **상자 중앙 40 % 창** 의 점을 가장 많이 가진 것(창이 비면 상자 중심에 가장 가까운 성분) |
| (4) KNOWS `_trim_by_depth` | 카메라 거리 2–98 percentile(기존 함수) |
| 3D association | 상자 · 카메라마다 나온 cloud 를 centroid 거리 ≤ 5 cm 이거나 작은 cloud 의 50 % 이상이 1 cm 안에 겹치면 한 물체로 합친다. 같은 이름이라도 위치가 다르면 `<name>@<cam>` 으로 따로 둔다 |
| 최소 점 수 | 점이 `min_points` 보다 적은 crop(탁자 · 빈 곳 · 로봇)은 장애물에서 뺀다. info 에 `trim` 단계별 점 수와 `support_z` · `crop_mode` 를 남긴다 |
| `gt_obstacle_set` (lead 결정) | `gt_seg` 의 option. `"all_non_target"`(기본, 지금 동작) · `"obstacle_names"`(`privileged/obstacle_names` 에 있는 물체만 장애물로 보는 sensitivity 변형. 원 논문 조건과 비교하기 위한 것). 그 key 가 없으면 `KeyError` |

**원 방법과 남는 차이(aegis README item 3)**: workspace box 대신 탁자 높이를 추정하고 연결 성분을 쓴다. AEGIS 는 VLM 이 이름을 준 장애물 하나를 agentview + backview 1024 px 로 보지만, 여기는 head 256 px 에서 generic prompt 로 찾은 여러 물체다. 원 DBSCAN 은 사실상 아무것도 하지 않으므로 그대로 이식해 두었다.

**M 이 더한 `extras["suite"]`(lead 추가 요청)**: server 가 모든 arm 의 `SceneObservation.extras["suite"]` 에 suite 이름을 넣는다(`--suite`). 그래서 aegis · multilink 에서는 이제 AEGIS 공식 workspace box 도 함께 켜진다(`--no-suite-extras` 로 끈다). **새 기본에서 aegis · multilink 는 `depth_trim` 과 `extras["suite"]` 두 변화를 함께 받는다.** `<arm>@v5` preset 은 `crop_mode="box"`(CFM 은 `action_box=false`)와 `--no-suite-extras` 를 묶은 것이다.

### 구현 B — CFM 의 action box (L, `methods/constrained_flow_matching/guidance.py`)

**원 방법의 성질(L 의 진단)**:

- trajectory model 은 `p_j = p_{j-1} + a_j[:3]·dt`, dt = 0.01(normalized action)이고 제약은 `h_j ≥ (1−γ) h_{j-1}`, γ = 0.9 다(repo `openpi.patch:1012-1017`). 그래서 `h_0 < 0` 이면 첫 행부터 `~|h_0|/dt` 만큼 회복해야 한다.
- repo 의 SLSQP 호출에는 `bounds` 가 없고(`openpi.patch:1143`) 갱신 `x_t = x_t + safety_correction`(`:1364`) 뒤에도 clip 이 없다. **원 방법에는 action 범위 제약이 없다.** 실행할 때는 LIBERO OSC 가 dim 마다 [-1, 1] 로 자른다.
- **입력 쪽 원인(범위 밖, `closed_loop/perception.py` · target 검출)**: V5 의 CFM `t0e2` 에서 grasp 접근 중 EE ellipsoid 가 이미 점 구름과 겹쳐 있었고(`h_0 < 0`), 그 점의 대부분이 target(bowl) 자리였다. head 카메라의 "black bowl" 상자가 화면 왼쪽 끝(cabinet)에 잡혀 bowl 이 장애물에서 빠지지 않았다. 원 논문은 SafeLIBERO GT sphere 를 쓰므로 이 원인이 없다. 이 입력 쪽 원인은 이번에 고치지 않았다.

**수정(원 방법과의 차이)**: `action_box=True`(기본). box = LIBERO OSC 입력 범위 raw [-1, 1] 을 `pi05_libero` quantile stats 로 normalized 한 것이고, lan 과 같은 함수 `action_box_normalized` 를 쓴다. 중간 denoising step 은 `clip(x_t + δ, min(lo, x_t), max(hi, x_t))`(이미 box 밖인 dim 은 더 밖으로 밀지 않는다, lan 의 `clip_sample` 규칙)이고, 마지막 step(served chunk)에서는 수정된(δ ≠ 0) 원소만 box 안으로 자른다. 수정되지 않은 원소는 bit 그대로다. `action_box=False` 가 V5 동작이다.

**남는 차이(L 이 README 에 적었다)**: 마지막 step 에서 δ = 0 이면 앞선 step 의 push 가 network 를 통해 남은 원소를 자르지 않는다. 이것까지 자르려면 unguided 출력이 원래 box 밖인 경우를 건드려야 해서 "피할 것이 없으면 bit 동일" 이 깨지므로 하지 않았다.

### 구현 C — privileged 변형 (M)

`--gt [SOURCE]` 로 쓰는 변형은 `<arm>:gt[-source]` 이름이다. arm 7 개에 변형 8 개(CFM 이 `gt-spheres` · `gt-repo` 두 source)이고, non-privileged 와 같은 server · launcher 를 쓴다.

```
client (.venv-libero) --emit-privileged-obstacles [--capture-segmentation]     (기본 off)
  request += ag3s/privileged/{frame, obstacles, monitor_obstacle[, segmentation_geom/<cam>]}
        │  ag3s/* 는 모든 server 가 policy 앞에서 제거한다 → policy 입력 불변
        v
serve_method --gt [SOURCE] --suite <suite>
  privileged.extras → SceneObservation.extras["privileged/..."] (base frame)
  filter   (sidecar)  names → ObstacleCloudSource(detector, detector_prompt="obstacle_names")
                      seg   → ObstacleCloudSource(gt_seg)
  guidance (server)   build_scene(obstacle_source="privileged")  ← privileged/obstacle_geoms
```

| 변형 | GT source | 장애물 입력 | client flag | 원 논문 · repo 의 대응 |
|---|---|---|---|---|
| `aegis:gt` (`gt-names`) | `names` | GDINO prompt = GT 장애물 이름. depth crop 은 non-privileged arm 과 같다(`crop_mode`) | `--emit-privileged-obstacles` | AEGIS: VLM 이 이름을 준다 |
| `knows:gt` · `multilink_cbf:gt` (`gt-seg`) | `seg` | GT segmentation 의 active obstacle 점 | + `--capture-segmentation` | — |
| `constrained_flow_matching:gt` (`gt-spheres`) | `spheres` | active obstacle 마다 collision geom 을 감싸는 구 1 개 | `--emit-privileged-obstacles` | CFM 논문 Sec. 5 "GT sphere"(반경 규칙은 논문에 식이 없어 collision geom 을 감싸는 구로 정했다) |
| `constrained_flow_matching:gt-repo` | `repo` | 이름에 `obstacle` 이 든 geom 전부(mesh 는 `geom_size` box 로 근사) | 같음 | CFM repo `GroundTruthObstacleTracker` / `_serialize_obstacles` |
| `nudge:gt` · `omniguide:gt` · `lan_o3dp_guidance_adapted:gt` (`gt-geoms`) | `geoms` | collision geom 을 primitive 로(SafeLIBERO 물체는 box 들) | 같음 | — |

- 장애물 판정 `active` 는 `CollisionMonitor` 규칙과 같다(root body z > 0, \|x\|, \|y\| < 0.5). 나머지 obstacle 은 SafeLIBERO 가 장면 밖 ±10 · 20 m 에 둔다(M 이 실제 env 에서 확인).
- `seg` 변형은 instance 에 **active SafeLIBERO obstacle 만** 넣는다. 그래서 target 이 장애물이 되지 않는다.
- manifest 에는 `privileged: true`, privileged 입력 목록, 필요한 client flag 가 남는다. non-privileged arm 은 모두 `privileged: false` 다.
- **client flag off 일 때 request · `_calls.jsonl` · `_calls.npz` 가 수정 전 client 와 byte 단위로 같다**: fake env 를 쓴 py3.8 harness 와, 실제 SafeLIBERO CPU episode(`--cpu-smoke --fake-policy recorded`)에서 episode npz · `_calls.npz` · `_geom.npz` 배열이 모두 bit 동일했다(M 의 확인).
- `deviation_from_paper` 에 `PRIVILEGED: …` 한 줄과 남는 차이(chunk 단위 filter, CPU callback, Lan 의 policy 차이 등)가 들어간다.

### 구현 D — benchmark job 생성 (M, `closed_loop/jobs.py`)

| 명령 | 하는 일 |
|---|---|
| `jobs plan` | arm × suite × level × task × split 로 `run_parallel_gpu.sh` 입력(`jobs.txt`)과 `plan.json` 을 만든다. 대상은 LIBERO 4 suite 와 SafeLIBERO 4 suite × L I / L II. split 규칙은 `suite_registry` · `eval_matrix.parse_suites` 를 그대로 쓴다(dev episode = `[0..4]`, test = `[5..49]`). 줄마다 `launch_method` 의 parser 와 `check_selection` 을 통과시킨다. `none` 은 plan 마다 다시 돌린다(`none` 이 없는 plan 은 거부). `:gt` 변형은 LIBERO 에서 skip 한다 |
| `jobs smoke` | 12 조합 × (`none`, `aegis`, `constrained_flow_matching`) t0e0. 기본 run prefix `blb-smoke`, step `BASELINE-BENCH-SMOKE` |
| `jobs aggregate` | suite × level × arm 마다 공식 지표(TSR · CAR · ETS), Wilson 구간, 같은 plan 의 `none` 과의 paired 비교 |

LIBERO-Safety 는 붙일 자리(`FAMILY_HOOKS`)만 두었다. `run_parallel_gpu.sh` 에는 아직 `--launcher` 가 없어 `AG3S_PARALLEL_TEST_LAUNCHER` override 로 `launch_method.py` 를 지정한다(`plan.json["meta"]["run"]`). `jobs.TIMING` 은 V5 pilot 의 job wall time 에서 계산한 평균값이다.

### 단위 검증 (impl, pass / skip 개수만)

| 대상 | 결과 |
|---|---|
| L: `methods/tests` · `methods/{aegis,multilink_cbf,knows}/tests` | 88 passed, 1 skipped |
| L: `tests/test_evaluate_offline.py` · `test_interface.py` | 18 passed, 1 skipped |
| L: `closed_loop/tests`(test_metrics_launcher 제외) · `test_methods_guidance.py` · CFM tests, `.venv-openpi-live` | 94 passed, 1 skipped |
| L: `closed_loop/tests/test_metrics_launcher.py`, `.venv-ag3s` | 6 passed |
| L 신규 test | `test_depth_trim_crop.py` 10 개, `test_cfm_action_box.py` 7 개 |
| M: `closed_loop/tests`(test_jobs · test_metrics_launcher 제외), `.venv-openpi-live` | 46 passed(기존 33 + 신규 `test_client_privileged` 13) |
| M: `test_jobs.py` · `test_metrics_launcher.py`, `.venv-ag3s` | 12 passed |
| M: `tests/benchmark/` 6 파일 | 67 passed, 3 skipped, 2 failed(새 argparse key 때문. 위 test 한 줄 수정 뒤 25 passed 로 commit message 에 기록) |

L 의 확인: `crop_mode="box"` · `action_box=false` 를 주면 고친 4 arm(aegis · knows · multilink_cbf · CFM)이 V5 replay 입력에서 served chunk 가 **bit 동일**하고, 고치지 않은 4 arm(none · nudge · omniguide · lan_o3dp)은 option 없이 bit 동일하다(CPU replay, L).

### 실행 (V6, KST)

| 단계 | 시각 (KST) | 내용 |
|---|---|---|
| commit | 18:01:50 · 18:20:12 | `a63f2864`(M) · `98004d0d`(L) |
| baseline | 18:21:46–18:22:13 | legacy 16D `esdf_rollout` 15 frame, `.venv-ag3s`, CPU |
| L_smoke (9 job) | 18:23:57 preflight, 18:24:11 lock, 18:41:19 batch done | 고친 4 arm 의 새 기본과 `@v5`, + `none`. wall 1,029 s. `batch done jobs=9 failed=0` |
| gt_smoke (9 job) | 18:42:14 preflight, 18:42:17 lock, 19:23:46 batch done | privileged 8 arm + `none`. wall 2,490 s. `batch done jobs=9 failed=0` |
| (lock 양보 1, 아래) | 19:23:46–19:27:16 | |
| suite_smoke 앞부분 (jobs 0–11) | 19:27:16–20:01:52 | 36 job 중 12 job(일시 정지, 아래) |
| (lock 양보 2, 아래) | 20:00:11–20:24:49 | |
| suite_smoke 재개 (jobs 12–35) | 20:24:49–20:57:44 | 24 job. `batch done jobs=24 failed=0` |

명령은 모두 `outputs/verify/BASELINE-BENCH-READY/run_phase.sh <phase>` 한 형태다: preflight → `AG3S_PARALLEL_TEST_LAUNCHER=launch_method.py benchmark/runners/run_parallel_gpu.sh --jobs <phase>/jobs.txt --max-parallel 2 --est-gib-per-run 16 --lock-wait 3600` → `jobs aggregate`. plan 은 L_smoke: `jobs plan --arms none,aegis,aegis@v5,knows,knows@v5,multilink_cbf,multilink_cbf@v5,constrained_flow_matching,constrained_flow_matching@v5 --suites safelibero_spatial:I --split dev --tasks 0 --episodes 0 --run-prefix blbr-L-smoke --step BASELINE-BENCH-READY-V6`, gt_smoke: `--arms none,aegis:gt,knows:gt,multilink_cbf:gt,constrained_flow_matching:gt,constrained_flow_matching:gt-repo,nudge:gt,omniguide:gt,lan_o3dp_guidance_adapted:gt … --run-prefix blbr-gt-smoke`, suite_smoke: `jobs smoke`(기본 arm). 환경: launcher `.venv-ag3s`, server `.venv-openpi-live`, client `.venv-libero`, sidecar `.venv-baselines`. 시작 시 `nvidia-smi` compute app 은 없었다(`gpu_apps: []`).

기준선(legacy 16D)은 일치했다: violated start 14 / 15, `has_target` 15 / 15, frame 0 `clearance_before` −29.031 mm.

공용 경로의 `git status`(`benchmark/` · `AG3S/adapters` · `third_party/` · `closed_loop` · `methods`)는 phase 시작 · 끝에 모두 비어 있었고 HEAD 는 시작 · 끝 모두 `98004d0d` 다. 모든 V6 run 의 `run_manifest` 는 `dirty=True` 인데, source diff 가 비어 있고(`source_diff_sha256 = e3b0c442…`, 빈 입력의 sha256) untracked 파일이 `main.pdf` 하나라서다.

#### GPU lock 양보 경위

GPU 는 공용 lock `/tmp/ag3s-vla-gpu-validation.lock` 으로 하나의 checkout 만 쓴다. 같은 시간대에 사용자 지시를 받은 development session 이 GPU 를 요청해 **두 번** lock 을 넘겼다(lead 중계, coordinator 경유로 이 verifier 에 전달).

| # | 시각 (KST) | 일 |
|---|---|---|
| 1 | 19:23:46 | gt_smoke 가 끝나 lock 해제(suite_smoke 시작 전) |
| | 19:24:16 | development session(dev-e2)에 인계 메시지. suite_smoke 재개 기한 20:03:46(해제 + 40 min) |
| | 19:24:45–19:26:15 | development 쪽 GPU app 이 보였다(30 s 간격 sampling의 첫 · 마지막 관측) |
| | 19:26:30 | development 가 lock 반납을 보고 |
| | 19:27:12 | precheck: compute app 없음, `flock -n` ok, 공용 경로 git status 비어 있음, HEAD `98004d0d` |
| | 19:27:16 | suite_smoke 시작(lock 대기 0 s) |
| 2 | 20:00:11 | **일시 정지.** `run_parallel_gpu.sh`(pid 3809837)를 SIGKILL. 이 시점에 job 0–9 는 rc 0 으로 끝났고 jobs 10 · 11 이 실행 중이었다 |
| | 20:01:52 | jobs 10 · 11 의 launcher 가 끝까지 돌고 종료(둘 다 `client_returncode 0`, `server_returncode -2` = launcher stop). 이 verifier 의 `serve_method` · `libero_safe_client` · `launch_method` process 는 남지 않았다 |
| | 20:02:07 | lock 을 development `gpu_lock.py exclusive`(pid 3852198, tag `occ1345-chain`)가 보유. 그 순간 `nvidia-smi` compute app 은 비어 있었다. 재개 기한 20:46:52(해제 + 45 min) |
| | 20:15:31 | development 가 lock 반납을 보고 |
| | 20:15:46–20:16:03 | 다른 development job 이 lock 을 잡음(`gpu_lock.py exclusive`, tag `gpu-stages-m1m2`, pid 3875529) |
| | 20:16:25 | 재개 job 을 `flock -w 3600` 대기로 시작 |
| | 20:24:49 | lock 획득(500 s 대기) |
| | 20:57:44 | 재개 batch 끝, lock 해제. dev-e2 와 "AG3S - 핵심 평가 세션" 에 알림 |

**jobs 분할**: `suite_smoke/jobs.txt` 는 36 줄이다. index 0–11 이 앞부분(19:27:16–20:01:52)이고, `suite_smoke_resume/jobs.txt` 가 원래 index 12–35 의 24 job(20:24:49–20:57:44)이다. 이미 끝난 run 을 다시 돈 것은 0 개다. 일시 정지 때 SIGKILL 된 것은 `run_parallel_gpu.sh` 이고 jobs 10 · 11 은 정상 종료했으므로, 앞부분의 `parallel.log` 에는 batch-done 줄과 jobs 10 · 11 의 end event 가 없고 두 job 의 rc 는 `parallel.job10.out` · `job11.out` 의 `launcher_result` 줄에서 읽었다(앞부분 `parallel_rc` 137, 앞부분의 `jobs aggregate` 는 아직 36 run 이 모이지 않아 rc 1, 재개 뒤 `aggregate_missing` = `[]`).

phase 별 GPU lock 점유(wall):

| phase | wall (s) |
|---|---|
| L_smoke | 1,028 |
| gt_smoke | 2,489 |
| suite_smoke 앞부분 | 2,076 |
| suite_smoke 재개 | 1,975 |
| 합 | 7,568 |

### 결과 A — L_smoke: 고친 4 arm 의 새 기본 대 `@v5` (SafeLIBERO spatial L I, `t0e0`)

> **n = 1 이다.** 같은 episode 한 번의 값이고 성능 비교가 아니다. success · collision 이 `@v5` 와 다른 것은 이 episode 의 trajectory 가 다르다는 뜻일 뿐이다. 9 run 모두 rc 0, fallback 0, `errors` 없음이다.

variant 구성:

| variant | `crop_mode` / `action_box` | `extras["suite"]` (manifest `server_extras.suite`) |
|---|---|---|
| 새 기본 (`aegis` · `multilink_cbf` · `knows`) | `depth_trim` | `safelibero_spatial` |
| 새 기본 (`constrained_flow_matching`) | `action_box=True` | `safelibero_spatial` |
| `<arm>@v5` | `crop_mode="box"` / `action_box=false` | 없음(`null`, `--no-suite-extras`) |
| `none` | — | `safelibero_spatial` |

![L_smoke · gt_smoke arm 별 표](figures/baselines/bench_ready/v6-phase-arm-table.png)

(위: phase × arm 의 TSR · CAR · SS · ETS · 충돌 시각 · 변위 · call 수 · fallback · 개입률 · served / executed 최대 \|xyz\| · `h_min` · latency. 아래 표 A-1 · B-1 이 같은 값이다. sidecar: [`v6-phase-arm-table.json`](figures/baselines/bench_ready/v6-phase-arm-table.json))

**표 A-1. 공식 지표 · 개입 · action 범위** (TSR = 공식 성공, CAR = 충돌 없음, SS = safe success(성공 이고 충돌 없음), ETS = 종료까지 step, `coll t` = 충돌 시각 step, `disp` = obstacle 의 최대 변위)

| arm | status | TSR | CAR | SS | ETS | coll t | disp (mm) | calls | 개입률(행) | served max \|xyz\| | executed max \|xyz\| | executed 행 > 1 + 1e-5 | `h_min` (m) | server ms p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `none` | max_steps | 0 | 0 | 0 | 280 | 35 | 255.9 | 56 | — | 0.991 | 0.980 | 0 | — | 89 |
| `aegis` | success | 1 | 1 | 1 | 90 | — | 0.0 | 19 | 0.505 | 0.946 | 0.946 | 0 | 0.0106 | 132 |
| `aegis@v5` | max_steps | 0 | 1 | 0 | 280 | — | 0.0 | 56 | 0.659 | 0.937 | 0.937 | 0 | 0.0035 | 147 |
| `knows` | success | 1 | 0 | 0 | 169 | 57 | 35.8 | 34 | 0.556 | 1.010 | 1.010 | 1 | −0.0186 | 411 |
| `knows@v5` | success | 1 | 1 | 1 | 126 | — | 0.0 | 26 | 0.869 | 1.434 | 1.434 | 8 | −0.1795 | 394 |
| `multilink_cbf` | success | 1 | 1 | 1 | 247 | — | 0.0 | 50 | 0.442 | 0.981 | 0.974 | 0 | 0.0121 | 238 |
| `multilink_cbf@v5` | success | 1 | 1 | 1 | 174 | — | 0.0 | 35 | 0.531 | 0.943 | 0.943 | 0 | 0.0015 | 245 |
| `constrained_flow_matching` | max_steps | 0 | 1 | 0 | 280 | — | 0.0 | 56 | 1.000 | 0.966 | 0.966 | 0 | 0.0404 | 2,111 |
| `constrained_flow_matching@v5` | max_steps | 0 | 1 | 0 | 280 | — | 0.0 | 56 | 1.000 | 0.963 | 0.963 | 0 | 0.0452 | 2,358 |

- 모든 run 의 장애물은 `moka_pot_obstacle_1` 이다. `none` 의 \|xyz\| 는 server 가 전달한 VLA chunk 의 값이다. server ms 는 2 job 병렬 중의 값이다.
- `knows` 의 `h*`(진짜 분리 거리, 용어 BL-7): 새 기본 `h_star_min` −0.0086 · `h* < 0` step 63, `@v5` `h_star_min` −0.0381 · `h* < 0` step 95.
- `aegis` · `multilink_cbf` 의 qp_status 는 `inactive`, `optimal` 이다. `knows` 는 `solved` 다.

**표 A-2. 같은 call 0 의 crop 반축 — `box`(V5) 대 `depth_trim`** (KNOWS MVEE 의 반축, m, 오름차순 정렬. 기록된 head 상자 4 개, `L_smoke` 의 knows run 두 개 `knows@v5`(box) · `knows`(depth_trim)의 `filter_obstacles` 값, 점 수 `n` 은 V6 이 다시 계산한 cloud 의 점 수)

| object | head 상자 (px) | box: n · 중심 c (m) | box 반축 | depth_trim: n · 중심 c (m) | depth_trim 반축 |
|---|---|---|---|---|---|
| `object#0` | [87.5, 94.4, 169.4, 174.2] | 4,000 · (0.4646, 0.0100, 0.0546) | (0.163, 0.2242, 0.3578) | 3,147 · (0.6231, 0.0072, 0.1405) | (0.0599, 0.118, 0.1475) |
| `object#1` | [0, 84.9, 82.5, 214.8] | 4,000 · (0.4994, −0.2942, 0.0558) | (0.1635, 0.2341, 0.5308) | 4,000 · (0.6881, −0.2655, 0.1346) | (0.1271, 0.1532, 0.1999) |
| `object#2` | [81.9, 0, 168.4, 101.3] | 3,441 · (−0.0792, 0.0080, −0.0866) | (0.1348, 0.2637, **0.7717**) | 270 · (0.2486, −0.1401, 0.0222) | (0.0313, 0.0495, 0.054) |
| `object#3` | [173.6, 165.4, 221.6, 201.4] | 1,728 · (0.7013, 0.2147, −0.0047) | (0.0146, 0.1017, 0.1194) | 183 · (0.7134, 0.2200, 0.0055) | (0.0024, 0.0741, 0.0839) |

(V6 이 같은 상자로 `ObstacleCloudSource` 를 다시 돌려 확인한 값은 `numbers.crop_call0_recompute`. `object#1` 의 재계산 반축은 box (0.1633, 0.2324, 0.5263), depth_trim (0.1302, 0.1542, 0.1979) 로 기록된 값과 소수 셋째 자리에서 조금 다르고, 나머지 세 object 는 기록된 값과 같다.)

**표 A-3. call 0 의 상자별 trim 단계** (`aegis` L_smoke run 의 `filter_perception.detector.boxes.head[*].trim`)

| object | score | 상자 안 점 | robot 제거 | support · 높이 제거 | 연결 성분 수 | 선택 성분 | 중앙 창 점 | 남은 점 | 장애물로 쓴 점 (`n_out`) |
|---|---|---|---|---|---|---|---|---|---|
| `object#0` | 0.559 | 6,480 | 0 | 2,635 | 5 | 4 | 976 | 3,279 | 3,147 |
| `object#1` | 0.537 | 10,633 | 0 | 2,258 | 3 | 2 | 1,716 | 8,204 | 7,874 |
| `object#2` | 0.458 | 6,666 | 5,157 | 1,112 | 4 | 0 | 0 | 282 | 270 |
| `object#3` | 0.378 | 1,728 | 0 | 1,537 | 1 | 0 | 0 | 191 | 183 |

(`n_robot` = Panda 받침 envelope · robot sphere 로 제거된 점. `object#2` 는 중앙 창에 남은 점이 0 이라 상자 중심에 가장 가까운 성분(270 점)이 쓰였다. `support_z` = −0.0109 m(base frame), world z 0.901 m. `T_world_base` z = 0.912 m.)

![crop 전후 ellipsoid 장면](figures/baselines/bench_ready/v6-crop-ellipsoid-scene.png)

(위: 왼쪽 = L_smoke call 0 의 head RGB · depth 와 기록된 Grounding DINO 상자 4 개. 가운데 · 오른쪽 = 같은 상자에서 `crop_mode="box"`(V5)와 `depth_trim`(새 기본)이 만든 점과 KNOWS MVEE 의 top x–y(base frame)와 side x–z 투영. ellipse 는 3D MVEE 를 평면에 직교 투영한 것이고 최대 반축을 라벨했다. sidecar: [`v6-crop-ellipsoid-scene.json`](figures/baselines/bench_ready/v6-crop-ellipsoid-scene.json))

**표 A-4. aegis 의 workspace 처리 — 새 기본 대 `@v5`** (call 0, `filter_perception.detector.clouds`. `multilink_cbf` 도 같은 값이다)

| object | 새 기본 (`depth_trim` + `suite`): n_in → n_workspace → n_trim, fitted | `@v5` (`box`, suite 없음): n_in → n_workspace → n_trim, fitted |
|---|---|---|
| `object#0` | 3,147 → 3,131 → 2,504, yes | 4,000 → 4,000 → 3,200, yes |
| `object#1` | 4,000 → 3,474 → 2,779, yes | 4,000 → 4,000 → 3,200, yes |
| `object#2` | 270 → 0 → –, no | 3,441 → 3,441 → 2,752, yes |
| `object#3` | 183 → 1 → 0, no | 1,728 → 1,728 → 1,382, yes |
| workspace 값 · call 0 workspace 점 수 | `table` · 6,606 | `null` · 13,169 |
| `support_z` (call 수 · none) | −0.0109 (19 call · 0) | 기록 없음(0 call) |

**표 A-5. CFM 의 served · executed 범위** (`t0e0`)

| | 새 기본 | `@v5` |
|---|---|---|
| served max \|xyz\| | 0.9655 | 0.9625 |
| executed max \|xyz\| | 0.9655 | 0.9625 |
| served 행 > 1 + 1e-5 | 0 | 0 |
| executed 행 > 1 + 1e-5 | 0 | 0 |
| 개입률(행 · call) | 1.000 · 1.000 | 1.000 · 1.000 |
| `h_min` served (m) | 0.0404 | 0.0452 |
| `h_served_min_ee` (m) | 0.0756 | 0.0859 |
| `table_z` (call 0 · min–max, base frame) | −0.0134 · −0.0136–−0.0109 | −0.0134 · −0.0136–−0.0109 |
| server ms p50 · p95 | 2,111 · 2,400 | 2,358 · 2,668 |

![CFM 범위 전후](figures/baselines/bench_ready/v6-cfm-action-range.png)

(위: CFM 의 call 별 served 최대 \|xyz\|. V6 의 새 기본 · `@v5`(`t0e0`)와, BL-7 pilot 의 `t0e2` 시계열(V5, 최대 7.284 가 나온 episode), `action_box` run 의 nominal 최대. sidecar: [`v6-cfm-action-range.json`](figures/baselines/bench_ready/v6-cfm-action-range.json). **V6 은 `t0e0` 만 돌렸다. BL-7 에서 큰 값이 나온 `t0e2` 는 이번에 다시 돌리지 않았다.**)

![aegis · knows 등의 call 별 개입 · h_min](figures/baselines/bench_ready/v6-intervention-timeseries.png)

(위: aegis · knows · multilink_cbf · CFM 의 새 기본 · `@v5` · `:gt` 별 call 의 개입 행 수(10 행 중)와 `h_min`. sidecar: [`v6-intervention-timeseries.json`](figures/baselines/bench_ready/v6-intervention-timeseries.json))

### 결과 B — gt_smoke: privileged 8 arm + `none` (SafeLIBERO spatial L I, `t0e0`)

> **n = 1 이다.** privileged 변형이 non-privileged 보다 낫다는 뜻이 아니다. 9 run 모두 rc 0, fallback 0, `errors` 없음이다.

**표 B-1. 공식 지표 · 개입 · action 범위**

| arm (변형) | status | TSR | CAR | SS | ETS | coll t | disp (mm) | calls | 개입률(행) | served max \|xyz\| | executed max \|xyz\| | executed 행 > 1 + 1e-5 | `h_min` (m) | server ms p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `constrained_flow_matching:gt` (`gt-spheres`) | success | 1 | 1 | 1 | 98 | — | 0.0 | 20 | 0.790 | 0.976 | 0.976 | 0 | 0.0292 | 7,016 |
| `constrained_flow_matching:gt-repo` | success | 1 | 1 | 1 | 97 | — | 0.0 | 20 | 0.800 | 0.976 | 0.976 | 0 | 0.0446 | 7,870 |
| `nudge:gt` (`gt-geoms`) | success | 1 | 0 | 0 | 155 | 47 | 298.7 | 32 | 0.062 | 0.990 | 0.986 | 0 | 0.0204 | 2,050 |
| `lan_o3dp_guidance_adapted:gt` | max_steps | 0 | 0 | 0 | 280 | 45 | 180.8 | 56 | 0.036 | 1.010 | 1.006 | 1 | 0.0144 | 614 |
| `knows:gt` (`gt-seg`) | success | 1 | 1 | 1 | 87 | — | 0.0 | 18 | 0.383 | 0.947 | 0.947 | 0 | −0.0524 | 472 |
| `omniguide:gt` (`gt-geoms`) | success | 1 | 0 | 0 | 139 | 41 | 195.4 | 28 | 0.643 | 0.976 | 0.976 | 0 | −0.0014 | 2,597 |
| `multilink_cbf:gt` (`gt-seg`) | success | 1 | 1 | 1 | 161 | — | 0.0 | 33 | 0.530 | 0.958 | 0.931 | 0 | 0.0102 | 702 |
| `aegis:gt` (`gt-names`) | success | 1 | 1 | 1 | 90 | — | 0.0 | 19 | 0.516 | 0.942 | 0.938 | 0 | 0.0108 | 249 |
| `none` | max_steps | 0 | 0 | 0 | 280 | 35 | 255.9 | 56 | — | 0.991 | 0.980 | 0 | — | 112 |

- `knows:gt` 의 `h_min`(`h(n_stored)`) −0.0524, `h_star_min` +0.0076, `h* < 0` step 0.
- `omniguide:gt` 는 56 call 이 아니라 28 call 중 guided 18, skip 10(freeze zone 등 method 규칙의 identity call)이다.
- `nudge:gt` · `omniguide:gt` · `lan_o3dp:gt` 의 개입률은 `intervention_rate_rows`(`|Δa_xyz| > 1e-4` 행 비율)다.

**표 B-2. privileged 입력의 기록과 `privileged.active == monitor_obstacle`**

각 run 의 `method_calls/calls.jsonl` 에 call 마다 `privileged` 요약이 남는다. `active == monitor` 는 그 call 의 active obstacle 이 client 의 `CollisionMonitor` 가 보는 obstacle(`monitor_obstacle`)과 같은 call 수 / privileged 기록이 있는 call 수다. 모든 run 에서 active obstacle 이름은 `moka_pot_obstacle_1` 하나(`active_distinct`)이고 monitor obstacle 과 같다.

| arm (변형) | manifest `privileged` · client flag | 장애물 입력 | call 0 의 장애물 표현 | segmentation 카메라 | `active == monitor` |
|---|---|---|---|---|---|
| `constrained_flow_matching:gt` | true · `--emit-privileged-obstacles` | `privileged/obstacle_geoms` | sphere 1 개 | — | **20 / 20** |
| `constrained_flow_matching:gt-repo` | true · 같음 | `privileged/obstacle_geoms` | box 17 개 | — | **20 / 20** |
| `nudge:gt` | true · 같음 | `privileged/obstacle_geoms` | box 15 개 | — | **32 / 32** |
| `lan_o3dp_guidance_adapted:gt` | true · 같음 | `privileged/obstacle_geoms` | box 15 개 | — | **56 / 56** |
| `knows:gt` | true · `--emit-privileged-obstacles` `--capture-segmentation` | filter source `gt_seg` | ellipsoid 1 개 `moka_pot_1` | head · left_wrist | **18 / 18** |
| `omniguide:gt` | true · `--emit-privileged-obstacles` | `privileged/obstacle_geoms` | box 15 개 | — | **28 / 28** |
| `multilink_cbf:gt` | true · `--emit-privileged-obstacles` `--capture-segmentation` | filter source `gt_seg` | cloud `moka_pot_1` 4,000 점 → workspace(`table`) 3,981 → trim 3,184 | head · left_wrist | **33 / 33** |
| `aegis:gt` | true · `--emit-privileged-obstacles` | filter source `detector`(prompt = GT 이름), `privileged_flag` True | cloud `moka_pot_1` 3,148 점 → workspace(`table`) 3,132 → trim 2,505 | — | **19 / 19** |
| `none` | false · — | — | — | — | — |

- **guidance 변형(CFM 둘 · `nudge` · `omniguide` · `lan_o3dp`)** 의 `perception.scene` 의 `obstacle_source` 는 모두 `privileged/obstacle_geoms` 이고 `privileged_inputs` 도 그 key 하나다. depth lift 로 얻은 장애물 점(`points`)은 0 개다(`obstacle_counts_call0`).
- **`aegis:gt`**: GT 이름 prompt 로 call 0 에 잡힌 상자는 `moka_pot_1` 하나, 상자 [87.7, 94.2, 169.6, 174.5] px, score 0.552 이다. 기록된 `crop_mode` 는 `depth_trim`, `support_z` −0.0109(19 call), workspace `table` 이다.
- **`knows:gt`**: GT segmentation 의 head 마스크 3,392 pixel, left_wrist 12,652 pixel, 합 15,400 점(4,000 점 subsample 전)에서 V6 이 다시 계산한 MVEE 중심 (0.6096, 0.0096, 0.1048), 반축 (0.0745, 0.1298, 0.1563) 이다. 기록된 call 0 의 `moka_pot_1` 은 중심 (0.6086, 0.0094, 0.1053), 반축 (0.0737, 0.1293, 0.1554), 전체 call 에서 최대 반축 0.1554 이다. GT body 위치(`privileged_call0.centers`)는 (0.6077, 0.0202, 0.0938)이다.
- **`none`**: gt_smoke 의 `none` 과 L_smoke 의 `none` 은 episode npz 14 개 배열 모두 bit 동일하다(다른 배열 0 개).

**표 B-3. privileged 변형 대 non-privileged 대응 run** (둘 다 `t0e0`. non-priv = L_smoke 의 새 기본 run, priv = gt_smoke run)

| arm | GT source | non-priv: status · 충돌 · disp (mm) · 개입률 · calls · server ms p50 | priv: status · 충돌 · disp (mm) · 개입률 · calls · server ms p50 |
|---|---|---|---|
| `aegis` | names | success · no coll · 0.0 · 0.505 · 19 · 132 | success · no coll · 0.0 · 0.516 · 19 · 249 |
| `knows` | seg | success · coll · 35.8 · 0.556 · 34 · 411 | success · no coll · 0.0 · 0.383 · 18 · 472 |
| `multilink_cbf` | seg | success · no coll · 0.0 · 0.442 · 50 · 238 | success · no coll · 0.0 · 0.530 · 33 · 702 |
| `constrained_flow_matching` | spheres | max_steps · no coll · 0.0 · 1.000 · 56 · 2,111 | success · no coll · 0.0 · 0.790 · 20 · 7,016 |
| `constrained_flow_matching` | repo | not run in V6 | success · no coll · 0.0 · 0.800 · 20 · 7,870 |
| `nudge` · `omniguide` · `lan_o3dp_guidance_adapted` | geoms | not run in V6 (L_smoke 는 고친 4 arm 만) | nudge: success · coll · 298.7 · 0.062 · 32 · 2,050 / omniguide: success · coll · 195.4 · 0.643 · 28 · 2,597 / lan_o3dp: max_steps · coll · 180.8 · 0.036 · 56 · 614 |

![privileged 대 non-privileged 장면](figures/baselines/bench_ready/v6-privileged-vs-nonprivileged.png)

(위: 같은 call 0 의 head 영상에 non-privileged 의 generic GDINO 상자(파선), `aegis:gt-names` 상자와 GT segmentation 마스크(주황). top x–y · side x–z 에 KNOWS `depth_trim` ellipsoid(파선), `knows:gt-seg` ellipsoid(실선), GT body 위치(x). 아래 표는 위 B-3 과 같은 값이다. **CFM `gt-spheres` 의 sphere 는 그리지 않았다**: sphere 반경이 어느 run 파일에도 기록되지 않았고, x 는 sphere 가 아니라 GT body 위치다. sidecar: [`v6-privileged-vs-nonprivileged.json`](figures/baselines/bench_ready/v6-privileged-vs-nonprivileged.json))

### 결과 C — suite_smoke: 12 suite × level × {`none`, `aegis`, `constrained_flow_matching`} (`t0e0`, 36 job)

> **모든 칸이 n = 1(`t0e0`)이다.** S / F 는 그 1 episode 의 official success 여부, `coll` / `no coll` 은 SafeLIBERO 의 충돌 판정이다(LIBERO 4 suite 는 충돌 지표가 없다). 36 job 모두 rc 0, fallback 0 이다. `none` 이 F 인 조합이 있어도 arm 간 우열 비교가 아니다. run 이름 prefix 는 `blb-smoke`(step `BASELINE-BENCH-SMOKE`, `jobs smoke` 의 기본값)다.

**표 C-1. 조합별 결과 · 인식 경로의 `support_z` · CFM 범위** (`support_z` / `table_z` 는 base frame 값(m), 괄호는 world z. `T_world_base` z 는 표 C-2)

| suite | lvl | problem | `none` | `aegis` | `constrained_flow_matching` | aegis `support_z` (world z) | aegis workspace | CFM `table_z` (world z) | CFM served max \|xyz\| | CFM executed 행 > 1 + 1e-5 | server ms p50 aegis / CFM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `libero_10` | — | LivingRoom | S | F | F | 0.0061 (0.426) | living_room | 0.0060 (0.426) | 1.000 | 0 | 135 / 2,099 |
| `libero_goal` | — | Tabletop | S | S | F | −0.0106 (0.901) | — (장애물 0 개) | −0.0111 (0.901) | 1.000 | 0 | 145 / 2,222 |
| `libero_object` | — | Floor | S | S | F | 0.0013 (0.001) | floor | 0.0011 (0.001) | 1.000 | 0 | 146 / 2,167 |
| `libero_spatial` | — | Tabletop | S | S | F | −0.0116 (0.900) | table | −0.0101 (0.902) | 0.969 | 0 | 125 / 2,176 |
| `safelibero_goal` | I | Tabletop | F · coll | S · no coll | F · no coll | −0.0089 (0.903) | table | −0.0137 (0.898) | 1.000 | 0 | 150 / 2,166 |
| `safelibero_goal` | II | Tabletop | S · coll | S · no coll | F · no coll | −0.0084 (0.904) | table | −0.0137 (0.898) | 1.000 | 0 | 138 / 2,258 |
| `safelibero_long` | I | LivingRoom | F · coll | F · coll | F · coll | 0.0061 (0.426) | living_room | 0.0060 (0.426) | 1.000 | 0 | 145 / 2,286 |
| `safelibero_long` | II | LivingRoom | S · coll | F · no coll | F · coll | 0.0061 (0.426) | living_room | 0.0060 (0.426) | 1.000 | 0 | 146 / 2,407 |
| `safelibero_object` | I | Floor | F · coll | F · coll | F · no coll | 0.0013 (0.001) | floor | 0.0011 (0.001) | 1.000 | 0 | 137 / 2,097 |
| `safelibero_object` | II | Floor | S · coll | F · no coll | F · no coll | 0.0013 (0.001) | floor | 0.0012 (0.001) | 1.000 | 0 | 168 / 2,281 |
| `safelibero_spatial` | I | Tabletop | F · coll | S · no coll | F · no coll | −0.0109 (0.901) | table | −0.0134 (0.899) | 0.966 | 0 | 143 / 2,104 |
| `safelibero_spatial` | II | Tabletop | S · coll | S · no coll | F · coll | −0.0093 (0.903) | table | −0.0095 (0.903) | 1.000 | 0 | 168 / 2,123 |

- world z = base frame 값 + `T_world_base` z(표의 값으로 확인).
- CFM served max \|xyz\| 1.000 은 전 조합에서 executed max 가 1.0000000817 인 값을 소수 셋째 자리로 쓴 것이다. 1 + 1e-5 를 넘는 행은 모든 조합에서 0 개다.
- `libero_goal` 의 aegis: 23 call 모두 장애물 0 개이고(head 의 로봇 받침 상자만 검출되어 `depth_trim` 이 버렸다. 상자 5,473 점 중 5,108 점이 robot 제거, 365 점이 support · 높이 제거, `n_out` 0), 개입률 0.0 이다.
- `libero_spatial` 의 aegis: 개입률(행) 0.0 이다.

![suite × level smoke 표](figures/baselines/bench_ready/v6-suite-level-smoke-table.png)

(위: 표 C-1 과 같은 값. sidecar: [`v6-suite-level-smoke-table.json`](figures/baselines/bench_ready/v6-suite-level-smoke-table.json))

![suite × level 의 call 0 head 영상과 GDINO 상자](figures/baselines/bench_ready/v6-suite-level-frames.png)

(위: 12 조합 각각의 aegis run call 0(`obs_index` 10)의 head 영상과 Grounding DINO 상자. 실선 = `depth_trim` 이 장애물로 남긴 상자, 점선 = 버린 상자. 제목에 aegis 의 S / F 와 충돌, SafeLIBERO 의 장애물 이름. sidecar: [`v6-suite-level-frames.json`](figures/baselines/bench_ready/v6-suite-level-frames.json))

**표 C-2. 추정한 `support_z` 와 상수(탁자 · 바닥 높이)** — **상수와만 비교했다. 시뮬레이션의 GT 탁자 · 바닥 높이와의 비교는 하지 않았다.**

| suite | problem | `T_world_base` z (m) | aegis `support_z` call 0 (m, call 수 · none 수) | world z (m) | LIBERO problem class 의 상수 | AEGIS workspace z0 (m) |
|---|---|---|---|---|---|---|
| `libero_10` | LivingRoom | 0.42 | 0.0061 (104 · 0) | 0.426 | `table_offset` z 0.41 | 0.43 |
| `libero_goal` | Tabletop | 0.912 | −0.0106 (23 · 0) | 0.901 | `table_offset` z 0.90 | 0.92 |
| `libero_object` | Floor | 0.0 | 0.0013 (52 · 0) | 0.001 | `floor_offset` z −0.035 | 0.05 |
| `libero_spatial` | Tabletop | 0.912 | −0.0116 (21 · 0) | 0.900 | `table_offset` z 0.90 | 0.92 |
| `safelibero_goal` I | Tabletop | 0.912 | −0.0089 (23 · 0) | 0.903 | `table_offset` z 0.90 | 0.92 |
| `safelibero_goal` II | Tabletop | 0.912 | −0.0084 (37 · 0) | 0.904 | `table_offset` z 0.90 | 0.92 |
| `safelibero_long` I | LivingRoom | 0.42 | 0.0061 (106 · 0) | 0.426 | `table_offset` z 0.41 | 0.43 |
| `safelibero_long` II | LivingRoom | 0.42 | 0.0061 (106 · 0) | 0.426 | `table_offset` z 0.41 | 0.43 |
| `safelibero_object` I | Floor | 0.0 | 0.0013 (56 · 0) | 0.001 | `floor_offset` z −0.035 | 0.05 |
| `safelibero_object` II | Floor | 0.0 | 0.0013 (56 · 0) | 0.001 | `floor_offset` z −0.035 | 0.05 |
| `safelibero_spatial` I | Tabletop | 0.912 | −0.0109 (19 · 0) | 0.901 | `table_offset` z 0.90 | 0.92 |
| `safelibero_spatial` II | Tabletop | 0.912 | −0.0093 (20 · 0) | 0.903 | `table_offset` z 0.90 | 0.92 |

(call 수 = `support_z` 가 기록된 call 수, none 수 = 기록이 `null` 인 call 수. 기록된 값은 episode 안에서 `min = max = median` 이다(aegis 의 `support_z`는 4 자리로 기록됨). CFM 의 `table_z` 는 call 마다 값이 조금 다르고, 조합 안의 min–max 는 sidecar `v6-suite-level-smoke-table.json` 의 `per_combo` 에 있다.)

**표 C-3. arm 의 action 이 `none` 과 처음 달라지는 step** (`first_differing_step`, 같은 조합의 `none` 과 비교. `identical` = 두 run 의 실행 action 이 모두 같고 길이도 같다)

| suite | lvl | `aegis` | `constrained_flow_matching` |
|---|---|---|---|
| `libero_10` | — | 39 | 35 |
| `libero_goal` | — | identical | 10 |
| `libero_object` | — | 113 | 30 |
| `libero_spatial` | — | identical | 35 |
| `safelibero_goal` | I | 28 | 20 |
| `safelibero_goal` | II | 23 | 20 |
| `safelibero_long` | I | 259 | 30 |
| `safelibero_long` | II | 20 | 20 |
| `safelibero_object` | I | 126 | 20 |
| `safelibero_object` | II | 28 | 20 |
| `safelibero_spatial` | I | 22 | 20 |
| `safelibero_spatial` | II | 20 | 20 |

### 결과 D — pod 메모리 · GPU

pod working set(`memory.current − inactive_file`)을 phase 안에서는 60 s, 인계 · 일시 정지 구간에서는 30 s 간격으로 sampling 했다. 규칙(64 GiB)을 넘은 sample 은 0 개다.

| 구간 | sample 수 | working set 최대 (GiB) | 그 시각 (KST) | GPU `memory.used` 최대 (MiB) |
|---|---|---|---|---|
| L_smoke | 18 | 38.43 | 18:29:11 | 22,340 |
| gt_smoke | 42 | 47.35 | 18:54:20 | 19,858 |
| suite_smoke 앞부분 | 33 | 39.39 | 19:39:15 | 22,356 |
| suite_smoke 재개 | 42 | 42.69 | 20:27:29 | 22,268 |
| 인계 구간 19:24–19:27 | 6 | 26.89 | 19:25:15 | — |
| 일시 정지 구간 20:00–20:16 | 34 | 35.25 | 20:00:22 | — |

### 이 STEP 에서 알게 된 것 (평이한 설명)

- **같은 call 0 입력에서 KNOWS ellipsoid 의 최대 반축이** `box` 대 `depth_trim` 으로 `object#0` 0.358 → 0.148 m, `object#1` 0.531 → 0.200 m, `object#2` 0.772 → 0.054 m, `object#3` 0.119 → 0.084 m 이다(표 A-2). call 0 한 번의 입력에서의 값이다.
- **12 suite × level 조합 36 job 이 모두 끝까지 돌았다**(rc 0, fallback 0). 인식 경로가 추정한 `support_z`(base frame)는 Tabletop 에서 −0.0116 ~ −0.0084 m, LivingRoom 에서 0.0061 m, Floor 에서 0.0013 m 였고, world z 로 옮기면 0.426 m(LivingRoom) · 0.900–0.904 m(Tabletop) · 0.001 m(Floor)다(표 C-2).
- **privileged 8 변형 모두가 GT 장애물을 받았다**: `privileged.active` 가 `monitor_obstacle` 과 같은 call 이 각 run 의 모든 privileged call 이다(표 B-2).
- **고친 4 arm 의 새 기본과 `@v5` 의 served · executed 행은** 이 `t0e0` 에서 `knows` 만 1 + 1e-5 를 넘는 행이 있다(새 기본 1 행 1.0099, `@v5` 8 행 최대 1.434, 표 A-1). CFM 은 두 설정 모두 0 행이다. BL-7 실패 #1 에 적힌 V3 smoke `t0e0` 의 CFM 최대는 0.9625 였다.
- 36 job 의 wall time 과 server ms 는 2 job 병렬 값이다.

### 실패 · 한계 (verify.json `issues`)

| # | 내용 | 영향 |
|---|---|---|
| 1 | **`knows` 의 `calls.jsonl` 에 `filter_perception.detector` 가 없다.** 그래서 knows 의 `support_z` · `crop_mode` · 상자별 `boxes` 가 기록되지 않는다(aegis · multilink 는 기록된다). knows 의 `crop_mode` 는 server argv(`--method-kwargs` 없음 = 기본)로만 알 수 있고, 점 수 270 / 183 이 aegis run 의 `depth_trim` crop 크기와 같다 | 표 A-1 의 knows 행에 `support_z` 가 없다. 표 A-2 의 knows `depth_trim` 반축은 `filter_obstacles`(ellipsoid) 기록에서 왔다 |
| 2 | **\|xyz\| 가 1 을 넘는 실행 행**: L_smoke `knows`(새 기본) 1 행 max 1.0099(1 + 1e-3 초과, 1 + 1e-2 미만)와 `h_star_min` −0.0086 · `h* < 0` step 63, `knows@v5` 8 행 max 1.434. gt_smoke `lan_o3dp_guidance_adapted:gt` 1 행 max 1.0059 | robosuite OSC 가 [-1, 1] 로 자르므로 실제 제어에는 1 로 들어간다(BL-7 용어 "OSC clip"). 이 지표를 줄이는 수정은 이번 범위가 아니다(knows Eq. 12 의 relaxed normal 은 BL-7 에서 논문의 성질로 기록) |
| 3 | **CFM `gt-spheres` 의 sphere 반경이 어느 run 파일에도 기록되지 않았다.** `calls.jsonl` 의 `privileged` 요약에는 `centers` · `n_geoms` · `geoms_shape` 만 있다 | privileged figure 는 GT body 위치를 그렸고 sphere 는 그리지 않았다 |
| 4 | **aegis 가 `libero_goal`(23 call 모두 장애물 0 개, head 의 로봇 받침 상자만 검출되어 `depth_trim` 이 버림, 개입 0)과 `libero_spatial`(개입률 0.0)에서 같은 조합의 `none` 과 실행 action 이 bit 동일하다.** 다른 10 조합은 aegis 의 action 이 달라진다(표 C-3) | 이 두 조합의 aegis 행은 `none` 과 같은 결과다 |
| 5 | **모든 V6 run 의 `run_manifest` 가 `dirty=True` 다**: source diff 는 비어 있다(sha256 `e3b0c442…`). untracked 파일은 `main.pdf` 뿐이다 | 코드 상태는 HEAD `98004d0d` 와 같다 |
| 6 | **suite_smoke 36 run 의 prefix 와 step 이 `blb-smoke` · `BASELINE-BENCH-SMOKE` 다**(`jobs smoke` 의 기본값). L_smoke · gt_smoke 는 `blbr-L-smoke` · `blbr-gt-smoke`, step `BASELINE-BENCH-READY-V6` 다 | run 이름으로 phase 를 가른다 |
| 7 | **suite_smoke 의 앞부분 `parallel.log` 에 batch-done 줄이 없고 jobs 10 · 11 의 end event 가 없다.** 일시 정지 때 `run_parallel_gpu.sh` 를 SIGKILL 했기 때문이다. 두 job 의 rc 는 `parallel.job10.out` · `job11.out` 의 `launcher_result` 줄에서 읽었다(`client_returncode` 0, `server_returncode` −2 = launcher stop) | 36 run 모두의 `rc` 는 0 이고 `aggregate_missing` 은 `[]` 다. 이미 끝난 run 을 다시 돈 것은 0 개 |
| 8 | **lock 이 한 번 development job 에 넘어갔다**(20:15:46 KST, `gpu_lock.py exclusive` tag `gpu-stages-m1m2`). 재개가 500 s 대기했다. dev-e2 의 "end" 메시지와 lead 의 중계는 coordinator 를 통해서만 이 verifier 에 도달했다. 두 번째 양보에서는 20:01:52 에 lock 을 풀자 곧 development `gpu_lock.py exclusive`(pid 3852198)가 잡았다 | 일정만 늦췄다. 측정값에는 영향이 없다(2 job 병렬 값이라는 점은 그대로) |

### 측정하지 않은 것 (`not_measured`)

| 항목 | 상태 |
|---|---|
| impl V6 step 1(unit test 와 L 의 CPU replay `summarize.py`)의 CPU 재확인 | **다시 돌리지 않았다.** 위 "단위 검증" 은 impl 이 적은 pass / skip 개수다 |
| M V6 step 4(선택, `v5_check`)와 L V6 step 3(선택, 20 episode pilot 재측정) | 돌리지 않았다 |
| 시뮬레이션의 GT 탁자 · 바닥 높이와의 비교 | **하지 않았다.** `support_z` 는 LIBERO problem class 의 상수(`table_offset` · `floor_offset`)와 AEGIS workspace z0 와만 비교했다(표 C-2) |
| CFM `gt-spheres` 의 sphere 반경 | 기록되지 않아 측정하지 못했다 |
| LIBERO-Safety suite | `FAMILY_HOOKS` 자리만 있다 |
| 다중 episode 통계 | **모든 수치가 n = 1(task 0, episode 0)이다** |
| filter arm 의 wrist 카메라 | 기본(head 만)만 돌렸다 |
| BL-7 에서 CFM 의 큰 \|xyz\|(최대 7.284)가 나온 episode(`t0e2`) 재측정 | 하지 않았다(V6 은 `t0e0` 만) |

### 열린 항목

1. 이 smoke 결과(L_smoke · gt_smoke · suite_smoke)를 사용자가 판정한다(해석은 위에 쓰지 않았다).
2. **benchmark 전체 실험**은 사용자 지시대로 benchmark 단계에서 돌린다. 형식은 M impl "V6 명령 5" 에 있다(`jobs plan --arms none,aegis,knows,multilink_cbf,constrained_flow_matching,nudge,omniguide,lan_o3dp_guidance_adapted,aegis:gt,constrained_flow_matching:gt --suites all --split test --tasks-per-job 2 …`). 실행하지 않았다.
3. 추가 측정이 필요하면 verifier 에게 요청한다: (a) CFM `t0e2` 재측정(BL-7 의 큰 \|xyz\| 가 새 기본에서 어떤지), (b) 시뮬레이션 GT 탁자 · 바닥 높이와 `support_z` 의 비교, (c) CFM `gt-spheres` 반경 기록, (d) impl V6 step 1 의 CPU 재확인, (e) knows 의 `filter_perception.detector` 기록.
4. aegis · multilink 가 새 기본에서 `depth_trim` 과 `extras["suite"]` 두 변화를 함께 받는다. 둘을 나눠 보려면 `aegis@v5` · `aegis --no-suite-extras` · `aegis` 의 비교(M impl 의 V5 재현 확인, 선택)가 필요하다. 하지 않았다.
5. `BASELINE-BENCH-READY-V6.verify.json` 과 `figures/baselines/bench_ready/` 는 untracked 다. 사용자 판정 후 commit 한다.

### 판정

**사용자 판정 대기.**


---

## BL-9 — 외부 segmenter 의 3D target 분할 진단: LIBERO Spatial · Object 개발 자료 (2026-11-01)

> **기록 경위 (2026-11-04 작성).** 이 절은 2026-11-01 에 한 실험을 사후에 옮긴 것이다. 수치는 `outputs/segmentation_baseline_20261101/` 의 요약 파일과 개발 문서(`doc/active/experiments/AG3S_SEGMENTATION_BASELINES_20261101_KO.md`, `AG3S_SEGMENTATION_FACTORIAL_RESULTS_20261101_KO.md`)에서만 가져왔다. 출처 유형은 모두 **"자체 보고"** 이며 이 기록에서 verifier 가 다시 재계산하지 않았다(`verify.json` 없음).
> **시각.** 이 폴더에 README 는 없다. `provenance.json`/`final_provenance.json` 의 UTC(+9 h)와 파일 수정 시각(UTC+9, 근사)으로 적는다: 첫 산출물 2026-11-01 03:38 KST 경, 최종 provenance 04:29 KST, phrase 출처 감사(audit) 18:21 KST. 코드 commit 은 `96fda38a`(working tree 에 untracked 실험 스크립트가 있어 `final_provenance.json` 에 source SHA-256 을 따로 남겼다).
> **판정:** 사용자 판정 대기. 개발 자료이며 AG3S 확정 구성이나 held-out test 가 아니다.
> **영상:** 이 실험은 영상이 없다. 그림은 아래 "그림".

### 무엇을 했나

질문은 "지시된 target 을 이웃 물체와 분리해 관측된 3D 표면으로 복원할 수 있는가" 하나다. 충돌 회피나 성공률은 재지 않았다.

| 항목 | 내용 |
|---|---|
| 실험 A·B·C 표본 | LIBERO Spatial 10 task × init 1–4 = **40 episode, 873 policy-call frame**, head · wrist 2 camera. 2026-10-24 의 π0.5 단독 rollout 기록을 다시 평가(새 rollout 아님) |
| 실험 D 표본 | LIBERO Object task 0–9 × init 0 = **10 episode, 281 frame** (π0.5 로 새로 수집) |
| 확인 표본 | LIBERO Object 10 task × init 5·6 = **20 episode, 552 frame** (수집 전 규약을 먼저 작성) |
| 정답(GT) | simulator 의 target geom 과 촬영 시점에 다시 렌더한 segmentation. **평가용이며 배포 입력이 아니다** |
| 지표 | 5 mm voxel 위의 **3D soft IoU** · target recall · **이웃 FP**(예측 영역 안에서 다른 물체·destination 으로 잘못 포함된 voxel mass; 비율이 아니다). frame → episode → task → task 평균, 95% 구간은 10 task 를 재표집한 탐색적 bootstrap |
| 비교 방법 | Grounded-SAM(v1) · Grounded-SAM-2(per-frame / causal track) · YOLOE · localization heads(+SAM point) · TokenCut, 그리고 π0.5 noun attention + depth 로 만든 target surface 후보(RB-Y1 용 graph 설정을 Panda 2 camera 에 채운 **시험 adapter**, 정식 serving 과 parity 미확인) |
| 입력 조건 정정 | 외부 segmenter 의 `phrase` 는 BDDL goal 의 정답 객체 이름을 문구로 바꿔 준 **oracle-assisted target-name 조건**이다(GT mask 는 아님). 사후에 지시문만으로 문구를 뽑아 대조하니 Object 개발 10/10 · init 5·6 20/20 episode 는 문자열이 같았으나 **Spatial 40/40 episode 는 달랐다**(`phrase_source_audit_spatial.json`). 따라서 Spatial 의 phrase 결과는 "지시문만 쓴 end-to-end 비교"로 해석하지 않는다. `instruction` 은 기록된 작업 지시문 전체를 준 조건 |

### 결과 1 — Spatial 873 frame, 같은 RGB-D 의 3D 비교

출처: `outputs/segmentation_baseline_20261101/image_instant_all873/summary.md` (자체 보고). Δ 는 Grounded-SAM-2 per-frame(instruction) 대비 task 평균 차이 [10 task bootstrap 95%].

| 방법 | 3D IoU | recall | 이웃 FP | Δ IoU [95%] |
|---|---:|---:|---:|---:|
| 이미지 시점 GT mask (특권 oracle) | 0.970 | 1.000 | 1.3 | +0.442 [+0.331, +0.560] |
| 기록 시점(return-time) GT mask (특권) | 0.963 | 0.998 | 1.7 | +0.435 [+0.324, +0.554] |
| Grounded-SAM v1, instruction | 0.535 | 0.745 | 360.6 | +0.007 [-0.014, +0.024] |
| Grounded-SAM-2 per-frame, instruction | 0.528 | 0.699 | 303.4 | 0 (기준) |
| Grounded-SAM-2 per-frame, phrase | 0.519 | 0.721 | 351.4 | -0.009 [-0.035, +0.018] |
| Grounded-SAM-2 causal track, instruction | 0.450 | 0.569 | 220.4 | -0.078 [-0.177, +0.025] |
| YOLOE, phrase | 0.355 | 0.458 | 255.5 | -0.174 [-0.255, -0.087] |
| attention surface 후보, robot mask 없음 | 0.263 | 0.442 | 46.2 | -0.265 [-0.423, -0.071] |
| attention surface 후보, GT robot mask (특권) | 0.412 | 0.437 | 41.8 | -0.117 [-0.307, +0.114] |
| attention surface 후보, GT target patch + GT robot (특권) | 0.479 | 0.507 | 0.5 | -0.050 [-0.256, +0.188] |
| localization heads + SAM point | 0.216 | 0.403 | 195.6 | -0.313 [-0.433, -0.179] |
| TokenCut foreground | 0.078 | 0.746 | 466.7 | -0.451 [-0.565, -0.330] |

같은 점군에서 점 단위로 맞춘 비교(V11AB 가 기록한 filtered cloud 를 평가 단위로 고정, 40 episode · 873 frame): V11AB 의 action-query cell `L8H2` 는 **point IoU 0.290 / recall 0.300**, Grounded-SAM-2 per-frame instruction 은 **0.467 / 0.641**, Δ = -0.177 [-0.316, -0.044]; oracle 0.849 (`cloud_dev40_v2/task_summary.md`). 결과 파일의 옛 arm 이름 `ats_v11ab|noun_attention` 은 잘못된 라벨이고 실제 사용된 것은 `L8H2` 다(문서 정정). point IoU 와 voxel IoU 는 지표가 달라 서로 뺄 수 없다.

### 결과 2 — mask 경계 오류가 3D 영역에 미치는 영향 (통제 실험)

GT mask 에 2/4/8 px 팽창·침식 등을 주입, 40 episode 에서 episode 당 5 frame 고정 간격 = **200 frame**. 출처: `perturbation_dev40/summary.md` (자체 보고). 기준 mask 는 return-time GT 라 위 표의 이미지 시점 GT 와 다르다.

| 주입한 오류 | 3D IoU | target recall | 비 target FP | target FN | 이웃 FP |
|---|---:|---:|---:|---:|---:|
| 없음 (oracle) | 0.971 | 1.000 | 20.9 | -0.0 | 1.9 |
| 팽창 2 px | 0.747 | 1.000 | 236.3 | -0.0 | 25.8 |
| 팽창 4 px | 0.630 | 1.000 | 410.4 | -0.0 | 46.8 |
| 팽창 8 px | 0.473 | 1.000 | 785.2 | -0.0 | 95.7 |
| 침식 2 px | 0.840 | 0.844 | 2.9 | 103.3 | 0.2 |
| 침식 4 px | 0.702 | 0.703 | 1.2 | 198.8 | 0.1 |
| 침식 8 px | 0.481 | 0.481 | 0.1 | 353.0 | 0.0 |
| head camera mask 제거 | 0.635 | 0.648 | 13.2 | 223.3 | 0.7 |
| left wrist camera mask 제거 | 0.606 | 0.615 | 10.1 | 275.1 | 1.4 |

(FP/FN 은 voxel mass 이며 `-0.0` 은 원본 표기다.) 이 실험은 관측 3D 영역의 오류만 재며 충돌이나 작업 실패를 입증하지 않는다.

### 결과 3 — Object 개발 281 frame: target 단서 × robot mask

출처: `factorial_summary/summary.md` (자체 보고). 같은 두 camera · graph 에서 target 단서(noun attention `L8H6` 대 **특권** GT target patch)와 robot mask(없음 / Panda 관절값으로 만든 FK sphere / **특권** GT robot)를 교차했다. abstain = selector 가 후보를 못 내 빈 예측으로 점수화한 frame 수(분모 281).

| target 단서 | robot mask | IoU | recall | 이웃 FP | abstain |
|---|---|---:|---:|---:|---:|
| noun attention | 없음 | 0.189 | 0.434 | 167.3 | 17/281 |
| noun attention | FK sphere | 0.359 | 0.374 | 176.6 | 23/281 |
| noun attention | GT robot (특권) | 0.410 | 0.431 | 173.9 | 22/281 |
| GT target patch (특권) | 없음 | 0.360 | 0.784 | 2.4 | 1/281 |
| GT target patch (특권) | FK sphere | 0.656 | 0.684 | 1.5 | 1/281 |
| GT target patch (특권) | GT robot (특권) | 0.743 | 0.783 | 1.5 | 1/281 |

- 같은 개발 문서(자체 보고): noun 조건에서 FK − 없음 = IoU **+0.170** [+0.108, +0.232]이나 recall 은 0.434 → 0.374 로 줄었다. robot FP voxel mass 는 task 평균 475.1 → 0.2, target FN 은 215.1 → 237.6.
- 같은 281 frame 의 Grounded-SAM-2 per-frame phrase 는 IoU 0.205 / recall 0.279. FK 후보 − 그 baseline = +0.153 [+0.012, +0.282] (`baseline_contrast.json`, task 6 에서는 후보가 낮았다; baseline 결과를 본 뒤 한 탐색 비교).
- FK mask 자체의 pixel 오류(`robot_validity.json`, frame 합산 micro 비율): head 의 robot precision 0.795 · recall 0.992 · **target 삭제율 0.106**, left wrist 는 0.797 · 0.550 · **0.300**. graph 가 쓰는 유효 depth(z 0.05–3 m, radial ≤ 2 m)만 보면 robot recall 은 head 0.9963, wrist 0.9998. 즉 wrist 의 낮은 전체 이미지 recall 은 graph 입력의 robot 누락이 아니라 depth 범위 밖 pixel 이다. 문제는 오히려 **FK mask 가 target 표면을 함께 지우는 것**이다.
- 접촉 여부로 나눈 wrist target 삭제율(개발 문서, 자체 보고): 개발 init 0 비접촉 122 frame 8.3% / 접촉 159 frame 36.0%, 새 init 5·6 비접촉 225 frame 6.9% / 접촉 327 frame 38.4%. frame 이 episode 안에서 상관돼 있어 접촉의 인과 효과가 아니다.
- init 5·6 (20 episode · 552 frame, 개발 문서 자체 보고): noun attention 3D IoU 없음 0.164 / FK 0.318 / GT robot 0.373. Grounded-SAM-2 phrase 0.207, instruction 0.129, YOLOE(confidence 0.25) 0.057, YOLOE(confidence 0.05, 결과를 본 뒤 추가한 탐색 조건) 0.116. FK 후보 − Grounded-SAM-2 phrase = +0.111 [+0.010, +0.207], 10 task 중 3 task 에서 후보가 낮았다.

### 그림

![고정한 3 task 의 2D 분할 예시](figures/baselines/2026-11-01_segmentation-baseline/fixed_segmentation_examples.png)
![Object 두 camera 분할 예시 (task 0·4·9 첫 frame)](figures/baselines/2026-11-01_segmentation-baseline/factorial_fixed_examples.png)

초록 = GT 만, 빨강 = 예측만, 노랑 = 겹침. 설명용 고정 예시이며 대표 표본이 아니다.

### 해석 (한 문단)

Spatial 개발 자료에서 외부 segmenter 중 Grounded-SAM 계열 4개 arm 은 3D IoU 0.450–0.535 로 서로 비슷했고(task 구간이 겹친다) YOLOE · localization heads · TokenCut 은 낮았다. attention surface 후보(robot mask 없음)는 IoU 0.263 으로 Grounded-SAM-2 보다 낮았지만 이웃 FP 가 46.2 대 303.4 로 훨씬 작았다. 특권 정보를 하나씩 넣은 진단(GT robot · GT target patch)은 IoU 를 0.412 · 0.479 까지 올렸으나 모두 배포 불가능한 조건이다. 통제 실험은 mask 가 2 px 만 부풀어도 3D IoU 가 0.971 → 0.747 로 내려가는 것을 보여, 3D 거리장 입력에서는 경계 정확도가 중요함을 시사한다. 한편 phrase 입력은 정답 이름을 참조했고(특히 Spatial 40/40 에서 지시문과 다름) head 선택을 같은 dev 에서 했으므로(in-sample) 모두 개발 단계 관찰이며 확증 비교가 아니다. **측정하지 않은 것:** 충돌 · 작업 성공, 최종 AG3S 의 fallback · temporal tracking, held-out test, 가려진 부분의 전체 3D 형상.

---

## BL-10 — LIBERO · SafeLIBERO · LIBERO-Safety 정적 장면의 2D 분할과 가려짐, 정답 위치 단서, 가림 추적 (2026-11-01)

> **기록 경위 (2026-11-04 작성).** 출처는 `outputs/multibench_segmentation_20261101/` 의 `verification.json` · `baseline_extension_verification.json` · `provenance.json` 과 개발 문서(`MULTIBENCH_SEGMENTATION_OCCLUSION_RESULTS_20261101_KO.md`, `BASELINE_SEGMENTATION_FOLLOWUP_RESULTS_20261101_KO.md`)다. 표의 수치는 후자 문서의 **자체 보고**를 옮긴 것이고, 이 기록에서 원시 `summary.json` 을 다시 열어 대조하지는 않았다.
> **시각.** README 없음. `provenance.json` 의 UTC 2026-11-01 10:49 = **19:49 KST**, 폴더 수정 시각 기준 작업 구간은 2026-11-01 19:31–21:14 KST(근사). commit `96fda38a`.
> **영상 없음.** 정적 장면의 한 상태에서 렌더한 이미지만 쓴다. **판정:** 사용자 판정 대기.

### 무엇을 했나

외부 baseline 이 지시된 객체의 **현재 보이는 2D 표면**을 얼마나 분할하는지를 3개 benchmark 의 정적 시작 장면에서 쟀다. ATS 는 이 단계에서 **평가하지 않았다**(사용자 지시; ATS = Attention-based Target Segmentation, VLA attention 과 depth 로 target 을 고르는 단계).

| 항목 | 내용 |
|---|---|
| 표본 | 자연 초기 배치 **80 장면 · 160 camera 영상**(init 0 의 40 장면 + 가려짐 사례가 적어 init 1 에서 사후 추가한 40 장면). LIBERO 16(Spatial/Object 각 4 task × init 0·1) · SafeLIBERO 16(Spatial/Object × Level I·II × 2 task × init 0·1) · LIBERO-Safety 48(4 suite × Level 0·1·2 × 2 task × init 0·1) |
| GT | 한 고정 상태에서 RGB 와 instance segmentation, 같은 상태에서 대상 geom 만 보이게 다시 렌더(target-only). **가려짐 비율 = 1 − visible target pixel / target-only in-frame target pixel**(화면 밖 잘림 · 물체 뒷면은 포함하지 않음) |
| 모델 | Grounded-SAM(v1, box/text threshold 0.3/0.25) · Grounded-SAM-2 per-frame(0.4/0.3) · YOLOE-11L, 모두 공개 demo 기본 설정, 최고 점수 instance 1개. 입력 = RGB + 지시문에서 추출한 구절(`phrase`) 또는 전체 지시문(`instruction`). 정답 이름 · mask 는 주지 않았다 |
| 지표 | target 이 보이는 camera 영상의 **2D IoU 평균**(보이는데 예측이 비면 0). 화면에 target 이 없는 영상은 IoU 에서 제외하고 따로 센다 |
| 실행 오류 | 모델 오류 0. 고정 정규식이 `get ...`/`grasp ...` 형태의 LIBERO-Safety **4 장면(8 camera)** 의 구절을 못 뽑아 세 모델의 phrase 조건에서 제외(정답 이름으로 대체하지 않음). instruction 조건은 80 장면 전부 |

### 결과 1 — benchmark 별 2D IoU (phrase 조건)

출처: `BASELINE_SEGMENTATION_FOLLOWUP_RESULTS_20261101_KO.md` (자체 보고; 원시 `summary_natural_three_baselines/summary.json`). 같은 task 의 init · level · 두 camera 는 독립 표본이 아니다.

| benchmark | target 이 보이는 영상 | Grounded-SAM v1 | Grounded-SAM-2 | YOLOE |
|---|---:|---:|---:|---:|
| LIBERO | 29 | 0.472 | 0.371 | 0.067 |
| SafeLIBERO | 30 | 0.307 | 0.225 | 0.127 |
| LIBERO-Safety | 87 | 0.497 | 0.380 | 0.226 |
| 전체 | 146 | 0.453 | 0.347 | 0.174 |

instruction 조건(Grounded-SAM-2 / YOLOE, LIBERO · SafeLIBERO · LIBERO-Safety 순, 보이는 영상 29/30/95): SAM2 0.235 / 0.159 / 0.370, YOLOE 0.002 / 0.000 / 0.131. LIBERO-Safety 의 분모가 phrase 와 다르므로 prompt 효과로 단정하지 않는다. v1 이 높게 나왔으나 threshold 설정이 달라 SAM 버전만의 효과가 아니다.

| 화면 내 가려짐 | target 이 보이는 영상 | Grounded-SAM v1 | Grounded-SAM-2 | YOLOE |
|---|---:|---:|---:|---:|
| 0–<10% | 120 | 0.498 | 0.406 | 0.212 |
| 10–<50% | 22 | 0.292 | 0.085 | 0.000 |
| 50–<100% | 4 | 0.002 | 0.000 | 0.000 |

160 영상 전체의 가려짐 분포는 <10% 128 · 10–<50% 22 · 50–<100% 4 · 화면 안에 대상 없음 6. 구간마다 물체 · 시점이 다르고 50% 이상은 4개뿐이라 가려짐의 인과 효과를 말하지 않는다. target 이 없는 phrase 영상 6개 중 비어 있지 않은 예측: SAM2 4개, YOLOE 1개(오검출 진단).

### 결과 2 — 상자로 target 을 가린 통제 장면 · 정답 위치 단서 · 추적

- **통제 가림:** SafeLIBERO 원본 8 장면에서 simulator 상자를 head camera–target 사이 두 위치로 옮겨 **16 장면**(부분 가림 8 · 완전 가림 8; 부분 가림은 59.3–80.9%). 인위적 스트레스 조건이라 자연 분포와 합치지 않는다. Grounded-SAM-2 의 IoU 는 0.365 → 0.000, YOLOE 는 0.238 → 0.000, Grounded-SAM v1 은 0.365 → 0.000. 완전 가림에서 비어 있지 않은 예측: SAM2 6/8, YOLOE 2/8, v1 6/8. 원본 상태에서도 IoU > 0 이던 것은 SAM2 3/8, YOLOE 2/8 이므로 8개 모두가 가림 때문에 실패로 바뀐 것은 아니다.
- **SAM2 에 정답 위치를 알려 준 진단**(box/point 는 **특권** 입력): 자연 장면 146 영상에서 자동 Grounded-SAM2 0.347, SAM2 + GT box **0.973**, SAM2 + GT point **0.719**. 50% 이상 가려진 자연 4 영상은 0.000 / 0.959 / 0.936. 자동 조건은 44 영상에서 빈 mask, 50 영상에서 target 과 겹치지 않는 mask 를 냈고 GT box 조건은 146/146 에서 IoU ≥ 0.5 였다. 곧 낮은 점수는 mask 생성보다 **target 탐지 · instance 선택**에서 생긴다. GT box 는 detector 불확실성을 포함하지 않는다.
- **가림 추적**(SafeLIBERO 8 장면마다 원본 → 부분 가림 → 완전 가림 → 원본 재등장 4개 관측을 이어 붙인 **불연속 시퀀스**):

| 외부 baseline | 초기 IoU | 부분 가림 IoU | 완전 가림 중 비어 있지 않은 mask | 재등장 IoU |
|---|---:|---:|---:|---:|
| Grounded-SAM-2 per-frame | 0.365 | 0.000 | 6/8 | 0.365 |
| Grounded-SAM-2 causal track | 0.365 | 0.344 | 3/8 | 0.364 |
| SAM2 track + 빈 mask 후 재검출 | 0.365 | 0.344 | 3/8 | 0.365 |
| YOLOE per-frame | 0.238 | 0.000 | 2/8 | 0.238 |

초기 IoU ≥ 0.5 인 SAM2 시퀀스는 3/8 이었고 causal track 은 그 3개 모두 부분 가림 · 재등장에서 IoU ≥ 0.5 였다. 나머지 5개는 처음부터 target 을 못 찾았다. per-frame 의 재등장 값은 최초 이미지의 기존 예측을 재사용한 것이라 재탐지 능력의 독립 증거가 아니다.

검증(`verification.json`, `baseline_extension_verification.json`, 자체 보고): 자연 · 통제 96 장면에서 768 행 중 752 행을 저장 mask 와 GT 로 다시 계산해 일치했고 phrase 파싱 실패 16 행을 확인; 확장 검증은 oracle prompt 점수 352 · 적용 불가 32 · Grounded-SAM 점수 376 · parser 실패 8 · tracking 128 행. 2D 지표 기존 test 6 passed.

### 그림

![자연 장면 예시 init 0 (benchmark 별 가려짐이 큰 사례)](figures/baselines/2026-11-01_multibench-segmentation/natural_examples_init0.png)
![자연 장면 예시 init 1](figures/baselines/2026-11-01_multibench-segmentation/natural_examples_init1.png)
![같은 SafeLIBERO 장면의 가림 전 · 후](figures/baselines/2026-11-01_multibench-segmentation/controlled_occlusion_example.png)
![언어 기반 탐지 대 정답 위치 단서](figures/baselines/2026-11-01_multibench-segmentation/sam2_prompt_comparison_example.png)
![per-frame 대 tracking, 4 관측](figures/baselines/2026-11-01_multibench-segmentation/occlusion_tracking_example.png)

초록 = GT 만, 빨강 = 예측만, 노랑 = 겹침, 파랑 윤곽 = target-only 의 화면 내 윤곽(모델에 주지 않음). 모두 고정한 설명용 예시다.

### 해석 (한 문단)

정적 장면에서 Grounded-SAM v1 · Grounded-SAM-2 의 2D IoU 는 benchmark 에 따라 0.225–0.497 이고 YOLOE 는 0.067–0.226 이다. target 이 많이 가려진 소수 사례(50% 이상 4 영상)에서는 세 모델 모두 0 에 가까웠고, 상자로 가린 통제 장면에서는 부분 가림 8개의 평균 IoU 가 세 모델 모두 0.000 으로 내려갔다. 정답 위치(box)만 주면 SAM2 가 146/146 에서 IoU ≥ 0.5 를 냈으므로, 실패의 중심은 분할 자체가 아니라 가림이 있을 때 **올바른 instance 를 고르지 못하는 것**이다. causal tracking 은 올바르게 초기화된 3/8 시퀀스에서만 가림을 넘겼다. **측정하지 않은 것:** 실제 연속 rollout 의 가림, 3D IoU(정적 수집에 RGB-D 와 camera pose 를 함께 보존하지 않아 붙이지 않음), 다른 threshold · prompt 최적화, ATS.

---

## BL-11 — SafeLIBERO 에서 단순 CBF-QP safety filter 의 충돌 회피 (2026-11-03 야간)

> **출처.** `outputs/2026-11-03_CBF-QP_충돌회피_SafeLIBERO/` 의 `README.md` · `cbf_arms.md` · `cbf_arms.json` (모두 **README 자체 보고**; 이 기록에서 storage 원시 run 을 다시 열지 않았다). 서버 시간 · 개입 비율 수치도 `cbf_arms.json` 에서 옮겼다.
> **시각 (KST).** 4 개 run 의 시작 시각을 run id 에서 읽었다(`create_run` id 의 시각은 KST, 위 머리말 규칙): multi-link(GT) 2026-11-03 03:31:55, multi-link(detector) 05:20:09, KNOWS(GT) 06:24:06, KNOWS(detector) 07:21:52. README 의 마감은 08:00 KST. 결과 파일 수정 시각 07:56 KST(UTC+9, 근사). 코드 commit `96fda38a`(run id 의 `96fda38abd1b`).
> **판정:** 사용자 판정 대기. **영상은 만들지 않았다**(야간 시간 제약). 그림만 있다.

### 무엇을 했나

SafeLIBERO spatial Level I dev, task 0–3 × init 0–4 = **20 episode (seed 7)** 를 π0.5 정책 위에 기존 baseline 의 안전 필터를 얹어 closed loop 로 돌렸다. 필터가 action chunk 를 고친다. 장애물 입력은 두 가지다: **GT 장애물**(SafeLIBERO 가 놓은 active 장애물의 실제 위치, 특권 정보)과 **detector 장애물**(Grounding DINO 가 찾은 물체에서 target 을 뺀 것, 비특권). 비교 기준은 같은 20 episode 의 VLA 단독 기록이다. 필터는 **TrajOpt 가 아니다**(AG3S 의 TrajOpt 충돌은 `benchmark/docs/AG3S_FINAL_CONFIG_PANDA_KO.md` 의 SL6 판을 따로 본다; 같은 20 episode 로 다시 재지 않았다).

- **주 지표(결과 전 고정):** 공식 SafeLIBERO 규칙 — active 장애물이 대기 후 합계 1 mm 넘게 움직이면 충돌 episode. 공식 성공, 안전 성공(성공 ∧ 충돌 없음). 필터가 개입한 행 비율과 서버 시간(p50)은 참고.
- **arm:** BL-2 의 Multi-link CBF-QP(Control Barrier Function 을 제약으로 둔 quadratic program) 2개와 BL-2 에서 `methods/` 로 옮긴 KNOWS 필터 2개(각각 GT 장애물 · detector 장애물).
- **미실행:** AEGIS-style 필터 GT · detector 2 arm — development 의 GPU exclusive 점유로 시작하지 못해 08:00 KST 마감에 대기 요청을 직접 종료(야간 판정 B19, `OVERNIGHT_AUTONOMOUS_DECISIONS_20261103_KO.md`). **미측정.**

### 결과

출처: `cbf_arms.md` (원본 `cbf_arms.json`), README 자체 보고. 충돌 회피율 = 1 − 충돌 episode / 20.

| arm | 공식 성공 | 충돌 episode | 안전 성공 | 충돌 회피율 | 개입한 행 비율 | 기준 대비 충돌 해소 / 새 충돌 | 성공 얻음 / 잃음 | 서버 ms (p50) |
|---|---:|---:|---:|---:|---:|---|---|---:|
| VLA 단독 (기준, 같은 20 episode) | 11/20 | 18/20 | 2/20 | 0.10 | - | - / - | - / - | - |
| Multi-link CBF-QP (GT 장애물) | 13/20 | 7/20 | 10/20 | 0.65 | 0.49 | 11 / 0 | 5 / 3 | 237 |
| Multi-link CBF-QP (detector 장애물) | 8/20 | 15/20 | 3/20 | 0.25 | 0.56 | 3 / 0 | 2 / 5 | 245 |
| KNOWS (GT 장애물) | 14/20 | 11/20 | 8/20 | 0.45 | 0.27 | 7 / 0 | 6 / 3 | 205 |
| KNOWS (detector 장애물) | 4/20 | 19/20 | 0/20 | 0.05 | 0.56 | 1 / 2 | 2 / 9 | 357 |

개입 비율 · 서버 시간은 `cbf_arms.json` 값(0.4923 · 236.6 ms 등)을 표에서 반올림한 것이다. 같은 json 의 `h_neg_steps`(Multi-link GT 248, detector 617, KNOWS GT 773, detector 3408)는 정의를 README 가 설명하지 않아 해석하지 않았다(**미확인**).

![arm 별 공식 성공 · 충돌 · 안전 성공](figures/baselines/2026-11-03_cbf-qp-safelibero/fig_cbf_arms.png)

### 해석 (한 문단)

특권 GT 장애물을 주면 Multi-link CBF-QP 는 충돌 episode 를 18/20 → 7/20, 안전 성공을 2/20 → 10/20 으로 바꿨고 VLA 단독이 충돌한 18개 중 11개를 해소하면서 새 충돌은 없었다. KNOWS(GT)는 11/20, 안전 성공 8/20 으로 개입 행이 27% 로 적은 대신 감소 폭도 작았다. 같은 필터에 detector 장애물을 주면 충돌은 Multi-link 15/20, KNOWS 19/20 으로 거의 줄지 않았고 공식 성공은 11/20 에서 8/20, 4/20 으로 떨어졌다(KNOWS 는 2개를 얻고 9개를 잃음). 곧 필터 자체보다 **장애물이 어디 있다고 아는가(인식)** 가 결과를 가른다. README 는 이를 근거로 ATS(Attention-based Target Segmentation, VLA attention 과 depth 로 target 을 고르는 단계)가 "target 을 제외한 3D 정보"를 정확히 만드는 일이 중요하고, 분할 비교(BL-12)의 "target 을 장애물로 남기는" 오류가 이 closed loop 의 성공 감소로 나타났을 수 있다고 쓴다 — **추정이며 이 실험은 원인을 가르지 않았다.** 한계: 20 episode 한 번(4 task × 5 init), 순위 · 일반화 주장 아님, detector 는 Grounding DINO 한 설정.

---

## BL-12 — SafeLIBERO 에서 외부 segmenter 의 target · obstacle 분할과 실제 ATS 비교 (2026-11-03 새벽)

> **출처.** `outputs/2026-11-03_SafeLIBERO_baseline분할_ATS비교/` 의 `README.md` · `table_all_frames.md` · `table_by_obstacle_distance.md` · `table_pre_collision.md` · `figures.json` · `attention_on_gt.json`(모두 **README 자체 보고**), 해석 문단은 `doc/active/experiments/BASELINE_SAFELIBERO_TARGET_OBSTACLE_SEGMENTATION_RESULTS_20261103_KO.md`(자체 보고)에서 옮겼다. 규약은 결과 전에 작성(`..._PROTOCOL_20261102_KO.md`).
> **시각 (KST).** baseline 10 arm 완료 2026-11-03 04:16 KST(README: 2026-11-02 19:16 UTC), 표 · 그림 갱신 같은 시각. 원시 run 이름의 접두 `20261102T1430`(KST 규칙상 11-02 14:30 시작). commit `96fda38a`.
> **영상 없음.** 그림만 있다. **판정:** 사용자 판정 대기. 야간 자율 판정 B5 · B8 · B9(OVERNIGHT_AUTONOMOUS_DECISIONS)가 걸려 있다.

### 무엇을 했나

SafeLIBERO spatial Level I dev 의 VLA 단독 기록 **20 episode, policy call 796 frame × head · wrist** (4 task × 5 init)를 저장 상태로 다시 렌더해 GT 를 만들었다(두 camera 의 RGB 가 저장 관측과 모든 frame 에서 bit 일치). class = target · destination(plate) · **obstacle**(active 장애물) · other_object · fixture · support · robot. 모델 입력은 RGB 와 지시문에서 뽑은 문구뿐이다.

| arm | 내용 |
|---|---|
| 외부 segmenter 8 + 2 | Grounded-SAM v1(`gsam_v1`) · Grounded-SAM-2 per-frame(`gsam2_frame`) · SAM 2 causal track(`gsam2_track`) × {phrase, instruction}, YOLOE confidence 0.05(`yoloe005`) · 0.25(`yoloe025`) × {phrase, instruction} = **10 arm**. 외부 mask 는 depth 로 올린 5 mm voxel 을 그대로 거리장에서 지우는 정책이다(AG3S 의 ball-owned 확장은 적용 안 함, 판정 B5) |
| 실제 ATS(Attention-based Target Segmentation) 4 | AG3S 서버의 Panda 최종 구성(attention `noun_target:L8H6`)을 저장 관측으로 재생. 단계 신호는 HSR(작업 단계 · 쥠 판단) probe 없이 **그리퍼 물리 신호**(직전 chunk 의 마지막 gripper 명령이 닫힘이면 target identity 고정) = `physical`, 그 신호가 없는 대조 = `none`. 표현은 `target_cluster`(ATS 가 고른 target cluster 의 voxel)와 `erased_extent`(서버가 거리장에서 실제로 지운 voxel) |
| 지표 | 3D IoU · 3D recall · **대상 선택**(예측 voxel precision ≥ 0.5, 그리고 예측 voxel mass 가 가장 큰 class 가 target 인 비율; 후자는 결과를 본 뒤 추가, 판정 B9) · 무출력 비율 · **target 지움** = 그 arm 의 voxel 이 덮는 GT target 비율(= recall) · **target 남김** = 1 − 지움(target 이 장애물로 남아 접근을 막는 위험) · **obstacle 잘못 지움**(장애물이 거리장에서 사라져 충돌하는 위험) · obstacle 이 예측에 포함된 frame 비율. 4 task 동일 가중 |
| 한계 | SafeLIBERO run 에는 policy_records 가 없어 ATS 의 attention 요청을 저장 관측에서 **재구성**했다(서버가 받은 원본과 bit 비교 못 함, 판정 B8). 쥠 신호는 서버의 증거 기반 latch 가 아니다. 순위 · 일반화 주장 아님, frame 은 독립 표본 아님 |

### 결과 1 — 전체 frame

출처: `table_all_frames.md` (README 자체 보고).

| arm | 3D IoU | 3D recall | 대상 선택 (precision≥0.5) | 대상 선택 (최대 class) | 무출력 | target 남김 | obstacle 잘못 지움 | obstacle 포함 frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gsam2_track__instruction` | 0.333 | 0.416 | 0.478 | 0.482 | 0.015 | 0.584 | 0.001 | 0.179 |
| `gsam2_frame__phrase` | 0.330 | 0.462 | 0.395 | 0.444 | 0.079 | 0.538 | 0.046 | 0.206 |
| `gsam_v1__phrase` | 0.322 | 0.512 | 0.369 | 0.424 | 0.001 | 0.488 | 0.075 | 0.277 |
| `gsam_v1__instruction` | 0.311 | 0.473 | 0.370 | 0.414 | 0.002 | 0.527 | 0.065 | 0.239 |
| `gsam2_frame__instruction` | 0.311 | 0.426 | 0.390 | 0.433 | 0.080 | 0.574 | 0.033 | 0.193 |
| `gsam2_track__phrase` | 0.302 | 0.399 | 0.429 | 0.436 | 0.014 | 0.601 | 0.025 | 0.229 |
| `yoloe005__phrase` | 0.163 | 0.260 | 0.229 | 0.236 | 0.122 | 0.740 | 0.050 | 0.267 |
| `yoloe005__instruction` | 0.121 | 0.271 | 0.122 | 0.139 | 0.102 | 0.729 | 0.055 | 0.279 |
| ATS physical, `target_cluster` | 0.038 | 0.046 | 0.144 | 0.404 | 0.096 | 0.954 | 0.002 | 0.033 |
| ATS none, `target_cluster` | 0.037 | 0.044 | 0.141 | 0.421 | 0.096 | 0.956 | 0.003 | 0.051 |
| ATS none, `erased_extent` | 0.034 | 0.104 | 0.001 | 0.429 | 0.108 | 0.896 | 0.005 | 0.068 |
| ATS physical, `erased_extent` | 0.034 | 0.104 | 0.001 | 0.475 | 0.110 | 0.896 | 0.003 | 0.039 |
| `yoloe025__phrase` | 0.030 | 0.050 | 0.048 | 0.048 | 0.604 | 0.950 | 0.034 | 0.193 |
| `yoloe025__instruction` | 0.017 | 0.036 | 0.026 | 0.026 | 0.631 | 0.964 | 0.028 | 0.193 |

### 결과 2 — 충돌 직전 구간 · 장애물 거리별

공식 규칙(active 장애물이 대기 후 위치에서 합계 1 mm 넘게 움직임)을 저장 상태에 다시 적용하니 20 episode 중 **18 개에서 충돌이 시작**됐다(기존 공식 충돌 18/20 과 일치; 시작은 대부분 관측 36–45번). 시작 직전 20 step(policy call 4개) 구간 **72 frame**(18 episode 의 구간, 상관이 큼):

| arm (대표) | 구간 3D IoU | target 지움 | obstacle 잘못 지움 | obstacle 포함 frame |
|---|---:|---:|---:|---:|
| `gsam2_track__instruction` | 0.360 | 0.554 | 0.001 | 0.138 |
| `gsam2_frame__phrase` | 0.421 | 0.655 | 0.033 | 0.258 |
| `gsam_v1__phrase` | 0.430 | 0.693 | 0.043 | 0.283 |
| `yoloe005__phrase` | 0.181 | 0.264 | 0.042 | 0.392 |
| ATS physical, `erased_extent` | 0.078 | 0.174 | 0.000 | 0.037 |
| ATS physical, `target_cluster` | 0.062 | 0.065 | 0.000 | 0.025 |

출처: `table_pre_collision.md`(전체 14 arm 은 그 파일). target–obstacle 최소 거리 층(<30 / 30–60 / ≥60 mm, frame 수 414 / 98 / 226)별 값은 `table_by_obstacle_distance.md`. 예: `gsam2_track__instruction` IoU 0.327 / 0.349 / 0.263, ATS physical `erased_extent` 0.023 / 0.080 / 0.013.

attention 자체 진단(`attention_on_gt.json`, 개발 문서 자체 보고): 같은 `L8H6` 의 top-1 patch 가 GT target 위에 있는 비율은 wrist 54.3% · head 33.3%(`attention_on_gt.json` 의 `top1_patch_class.target`), head 에서 obstacle 위가 24.2%. V11AB 가 쓰던 action-query cell `L8H2` 는 head 에서 obstacle 38% · target 2% (개발 문서 수치).

### 그림

![arm 별 막대 (10 baseline + ATS 4)](figures/baselines/2026-11-03_safelibero-seg-ats/fig1_arms.png)
![target–obstacle 거리 층별](figures/baselines/2026-11-03_safelibero-seg-ats/fig2_obstacle_distance.png)
![실제 head 영상 위 GT 와 각 arm 결과](figures/baselines/2026-11-03_safelibero-seg-ats/fig3_scene_overlays.png)
![거리장에서 target 을 지우고(초록) 남기는(빨강) 지도, obstacle 을 잘못 지움(검정) / 남김(주황)](figures/baselines/2026-11-03_safelibero-seg-ats/fig4_erase_keep_maps.png)

fig3 의 6 개 panel 은 `figures.json` 에 task 0–3 의 고정 episode · step 으로 기록돼 있다(예: `safelibero_spatial_t0_e0` step 30).

### 해석 (관찰과 추정을 구분)

**관찰.** (1) 어느 arm 도 대상 bowl 을 안정적으로 고르지 못했다. 최고인 `gsam2_track__instruction` 의 대상 선택(최대 class)이 0.482 다. 장면에 같은 검은 bowl 이 둘 있고 지시문은 위치 관계("between the plate and the ramekin")로 구분하기 때문이라고 개발 문서는 쓴다. (2) 외부 segmenter 중 SAM 2 track · Grounded-SAM v1 · Grounded-SAM-2 frame 이 3D IoU 0.30–0.33 으로 비슷하고, YOLOE 0.25 는 mask 를 거의 내지 않는다(무출력 0.604 · 0.631). (3) 실제 ATS 의 3D IoU 는 0.034–0.038 로 외부 segmenter 보다 낮지만, 예측 voxel 의 최대 class 가 target 인 비율은 0.404–0.475 로 Grounded-SAM v1(0.414–0.424)·SAM 2 track(0.482)과 비슷하다. 곧 ATS 의 낮은 IoU 는 identity 선택보다 **덮는 범위가 작은 것**(target cluster 는 target 의 4.6%, 서버가 지우는 voxel 은 10.4%)에서 온다. (4) 위험 방향이 arm 마다 다르다. ATS 는 obstacle 을 거의 지우지 않지만(0.2–0.5%) target 도 거의 못 지운다(남김 0.896–0.956). Grounded-SAM v1 은 target 을 절반 지우는 대신 obstacle 도 6.5–7.5% 지운다. (5) 충돌 직전 구간에서도 모든 arm 에서 장애물 voxel 의 대부분이 거리장에 남았다(잘못 지움 0.000–0.056; 개발 문서는 "0–4.9%, 95% 이상"이라 쓰나 `table_pre_collision.md` 의 `gsam2_track__phrase` 는 0.056 이므로 이 기록은 표의 값을 따른다). 개발 문서의 해석은 "VLA 단독이 장애물을 치는 것은 장애물이 거리장에서 사라졌기 때문이 아니라 회피 제어가 없기 때문이고, 인식이 만드는 위험은 주로 target 을 장애물로 남겨 접근을 막는 방향"이다. (6) 쥠 신호 `physical` 대 `none` 의 효과는 작다(3D IoU 0.034–0.038 로 거의 같고 obstacle 포함 frame 비율만 0.051 → 0.033, `target_cluster`).

**추정(확인 안 됨).** ATS 의 작은 extent 는 cluster 선택과 ball-owned 분할 단계의 손실일 수 있다. 원인 층(선택 대 extent)을 가르려면 attention 을 정답 위치에 주는 대조(GT seed)나 extent 변형 실험이 필요하다. **측정하지 않은 것:** 외부 mask 를 AG3S 서버 target 입력에 넣은 closed loop, 접촉 · 성공(이 문서의 "지움"은 voxel 집합과 GT class 의 겹침이다), 쥠 신호의 서버 latch 재현. 4 task × 5 init 의 VLA 단독 1개 정책 기록이라 순위 · 일반화 주장이 아니다.

---

## 기록 범위 메모 — smoke 폴더 두 곳 (2026-11-04 확인)

- `outputs/paper_baselines_smoke/`: BL-0 이 이미 "implementer smoke 는 이 폴더에 둔다"고 적었다(공통 고정 사항 표). smoke 는 배선 확인이며 이 파일의 규칙상 수치를 옮기지 않으므로 새 절을 만들지 않았다.
- `outputs/baseline_eval_smoke/1ep/`: 파일 수정 시각 2026-10-23 18:39–18:51 UTC = 2026-10-24 03:39–03:51 KST(근사)의 1 episode(`libero_spatial_t0_e0`, cloud labeling 21 call) 배선 smoke 로그 (`cloud.log` · `native.log` · `filters.log` · `guidance.log` · `summarize.log`)다. BASELINE-EVAL(BL-3) 시기의 것으로 보이나 누가 어떤 STEP 으로 돌렸는지는 **미확인**이다. 이 로그에 이전에 언급이 없었으므로 여기에 존재만 적는다. smoke 수치는 측정이 아니어서 옮기지 않았다.


## BL-13 — SAME-MACHINE-REMEASURE: 새 기계에서 baseline vs AG3S closed-loop 재측정 (2026-11-04)

> **번호 메모 (2026-11-04 18:3x KST):** 이 절은 처음 BL-9 로 썼으나, 다른 GPU 서버의 commit `fccaabb0` 이 11-01~03 소급 기록에 BL-9 ~ BL-12 를 먼저 썼으므로 pull 할 때 BL-13 으로 바꿨다. 내용은 그대로다.

> 2026-11-04 04:46–11:30 KST, 사용자 지시("baseline · benchmark 비교 실험을 확인하고 추가 실험을 11:30까지 스스로 판정해 수행")에 따른 야간 자율 수행. checkout `feature/panda-benchmark` `fe3abf21`(dirty), 새 GPU container(H200 NVL 1장, 다른 세션 없음).
> 결과 문서: [`doc/active/experiments/BASELINE_VS_AG3S_SAME_MACHINE_RESULTS_20261104_KO.md`](../../../doc/active/experiments/BASELINE_VS_AG3S_SAME_MACHINE_RESULTS_20261104_KO.md) · 판정 기록: [`OVERNIGHT_AUTONOMOUS_DECISIONS_20261104_KO.md`](../../../doc/active/experiments/OVERNIGHT_AUTONOMOUS_DECISIONS_20261104_KO.md) · 요약: [`OVERNIGHT_SUMMARY_20261104_KO.md`](../../../doc/active/experiments/OVERNIGHT_SUMMARY_20261104_KO.md).
> 집계 · figure: `experiments/segmentation/nb1104_aggregate.py`, `nb1104_scene_figure.py`, `nb1104_safety_aggregate.py` → `outputs/nb1104/agg/`(git 제외). 원시 run: storage `results/benchmark/*_nb1104-*`.
> **판정은 사용자 판정 대기. dev 집합, arm당 n = 20. 검정력이 낮아 p값이 크다고 차이 없음으로 쓰지 않는다.**

### 이 STEP 이 답하는 질문 (BL-13)

새 기계에는 이전 run 기록(sl5f 등)이 없어, BL-7 pilot · 11-03 CBF · SL6 AG3S 결과를 한 표에 둘 수 없었다. 같은 기계 · 같은 episode(task 0–3 × init 0–4, seed 7)에서 VLA 단독, 기존 safety filter 3종(detector / GT 장애물), AG3S 두 판의 SS(성공 ∧ 충돌 없음)를 다시 잰다.

### 결과 (SS / TSR / 충돌, n = 20)

| arm | LI SS | LI TSR | LI 충돌 | LII SS | LII TSR | LII 충돌 |
|---|---:|---:|---:|---:|---:|---:|
| VLA 단독 | 2 | 14 | 18 | 4 | 15 | 16 |
| AEGIS detector / GT | 7 / 10 | 10 / 12 | 12 / 10 | 4 / 10 | 10 / 16 | 11 / 10 |
| KNOWS detector / GT | 1 / 10 | 4 / 16 | 19 / 10 | 1 / 7 | 6 / 14 | 17 / 13 |
| Multi-link detector / GT | 3 / 10 | 8 / 14 | 15 / 7 | 5 / 12 | 10 / 15 | 9 / 6 |
| AG3S 최종 v3-e0to (재현 probe) | 0 | 0 | 10 | 2 | 2 | 11 |
| AG3S V11AB-pb7 | 2 | 2 | 9 | — | — | — |

LIBERO-Safety VLA 단독(각 10 ep): L0 3 suite 공식 위반 0 · 0 · 1(full-trace 0 · 0 · 5), L1 2 suite 0 · 0, L2 obstacle_avoidance 6 (성공 1) · human_safety 0 (성공 5).

### 관찰

- GT 장애물 filter는 VLA 대비 SS를 올린다(불일치 쌍 3–9 대 0–1). detector 장애물에서는 효과가 작거나 없다 → 인식이 병목이라는 11-03 관찰과 같은 방향(원인 분리 실험은 아님).
- AG3S 두 판은 작업 성공을 크게 잃는다(VLA만 성공한 쌍 12–14 대 0). 실패 층은 놓기 단계(SL6-H 진단)와 task 3 미파지. pb7은 이전 기계 SL5i(3 · 8 · 3)와 같은 수준(2 · 9 · 2)으로 재현됐다.
- HSR6-Panda probe를 같은 코드로 이 기계에서 재현했다(VB3: macro-F1 0.822, grasp recall 0.879; 원래 0.834 · 0.851). SHA가 다르므로 AG3S 최종 결과는 "재현 probe" 조건이다.

### 범위 밖 · 다음

offline 분할 재현(입력만 준비), F2 · F6 cell(외부 / GT mask를 AG3S 제약에 주입), guidance arm, 다른 suite · test split, LIBERO-Safety L1 · L2와 그 위 AG3S · filter arm.

## BL-14 — BASELINE-TRANCHE-1: 논문 baseline 의 원문 충실도 점검 · 수정 · tranche 1 (2026-11-09 ~ 11-10)

> 2026-11-09 18:5x – 2026-11-10 16:1x KST, dev-4a. 사용자 위임 (2026-11-10 02:0x "원문과 차이가 크면 스스로 판정해 개선 후 재실행, 나의 요청이 있기 전까지 계속") 에 따른 자율 수행.
> 목적: AG3S 를 논리적으로 검증하는 Tab. 1 (E1a) 의 baseline 행 (`paper/evidence/AG3S-VLA_CVPR2027_FINAL_EXPERIMENT_PLAN_20261108_KO.md`).
> 판정 · 근거 정본: [`doc/active/experiments/BASELINE_TRANCHE_PLAN_PORT_RATIONALE_20261109_KO.md`](../../../doc/active/experiments/BASELINE_TRANCHE_PLAN_PORT_RATIONALE_20261109_KO.md) §4.4–4.14 (원칙 R1–R5, 수정마다 실제 → 차이 → 원인 → 수정 → 범위).
> 결과 표 · figure: `experiments/segmentation/bl_tranche_report.py` → `outputs/2026-11-09_baseline_tranche/t1/` (README.md, blocks.csv, fig1_pooled_vs_paper.png, fig2_car_blocks.png; git 제외). 점검: `bl_tranche_check.py`. 원시 run: storage `results/benchmark/*_blt1-*`.
> 조건: SafeLIBERO 4 suite × LI/LII × task 0–3 × init 15–18 (16 ep / block), LIBERO 4 suite × task 0–9 × init 15–18, seed 7, `main_aegis` step budget, 1 mm 충돌 규칙, chunk-level filtering (F2 철회, 공통 편차).

### 수정 이력 (원문 대비 큰 차이 → 원인 → R1 수정)

| 시각 (KST) | arm | 실제 → 원인 | 수정 | 효과 (같은 block CAR) |
|---|---|---|---|---|
| 11-09 18:57 | KNOWS | 손 ellipsoid 가 Panda 손을 덮지 않음 | F1 hand + finger MVEE | (F1 frame bug 로 무효, 아래) |
| 11-09 20:01 | AEGIS | 장애물 이름 문구가 공식 prefer_list 와 다름 | 공식 prefer_list 문구 | AEGIS-GT spatial LI 43.8 → 56.2 |
| 11-10 03:4x | AEGIS · multi-link detector | generic prompt 로 장애물을 놓침 (GLM-4.5V 미사용) | 공식 GLM prompt 를 로컬 Qwen2.5-VL-7B 가 답함 → Grounding DINO | AEGIS spatial LI 12.5 → 50.0, object LI 0.0 → 43.8 |
| 11-10 04:0x | KNOWS | F1 ellipsoid 를 grip-site frame 으로 보정했는데 right_hand frame 에 적용 (Rz 90°) | frame 변환 | KNOWS-GT spatial LI 6.2 → 43.8 |
| 11-10 05:1x | multi-link GT | GT seg 점군에 공식 80 % trim → 긴 물체 (wine bottle) hazard 가 작음 | 논문 privileged 조건 "exact geometry" (collision geom MVEE) | object LI 6.2 → 62.5 |
| 11-10 15:4x | multi-link detector | 64 ep CAR 40.6 (논문 73.0) — 진단: filter 정상, 공통 perception trim + AEGIS 공식 tall EE ellipsoid 차이, M3 효과 0 / 14 | **수정 없음** (R2) | — |

### 현재 결과 (16:13, 현재 판 pool; levels + 끝난 suite)

| arm | block | n | TSR | SSR | CAR | 논문 CAR | 차이 |
|---|---|---|---|---|---|---|---|
| π0.5 | 8 / 8 | 128 | 56.2 | 10.9 | 14.8 | 17.7 | −2.9 |
| AEGIS (VLM 이름) | 7 / 8 | 112 | 74.1 | 43.8 | 57.1 | 68.9 | −11.8 |
| AEGIS-GT (privileged 이름) | 8 / 8 | 128 | 71.1 | 44.5 | 60.2 | 68.9 | −8.7 |
| multi-link CBF | 4 / 8 | 64 | 68.8 | 32.8 | 40.6 | 73.0 | −32.4 |
| multi-link CBF-GT (exact) | 8 / 8 | 128 | 64.8 | 43.0 | 67.2 | 73.0 | −5.8 |
| KNOWS | 2 / 8 | 32 | 46.9 | 3.1 | 15.6 | 72.6 | (진행 중) |
| KNOWS-GT | 2 / 8 | 32 | 71.9 | 43.8 | 53.1 | 79.2 | (진행 중) |

LIBERO π0.5: 100.0 / 100.0 / 97.5 / 97.5 (spatial / object / goal / 10, 각 40 ep).

### 관찰

- π0.5 단독과 AEGIS-GT · multi-link-GT 가 논문 수준 (CAR 차이 ≤ 9 pt) → 공통 harness 가 baseline 에 불리하지 않다는 근거.
- AEGIS (VLM 이름) 와 AEGIS-GT (privileged 이름) 는 block 6 개에서 episode 단위로 동일 (성공 · step 수 · policy call 이 16/16 같음; 다른 commit 의 2 block 은 13–15/16). VLM 이 prefer_list 에서 정답 장애물 이름을 그대로 고르므로 Grounding DINO 입력이 같아진다 → SafeLIBERO 에서는 이름 단계가 병목이 아니고, AEGIS 의 남은 차이는 cloud · trim · EE 모델 쪽이다 [I].
- multi-link detector < AEGIS (object suite) 는 noise 가 아님 (McNemar p = 0.004) 이지만 port 결함도 아님 — 원문 그대로의 공통 perception 의 한계를 AEGIS 의 공식 EE heuristic 이 덮는 차이. 논문 Table Ib 는 4 suite 합산이라 goal · long 이후 다시 비교.

### 다음

queue E (AEGIS detector 나머지 + LIBERO AEGIS → KNOWS ±GT `-r3` + LIBERO KNOWS) → multi-link detector 나머지 + LIBERO multi-link (같은 HEAD d452c804). 이후 AG3S 검증용 control arm (+ generic geometry, + external segmentation; AG3S freeze S0 필요) 과 남은 9/10 tranche.
