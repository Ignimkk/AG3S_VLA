# AG3S 논문 실험 계획서 — benchmark · baseline · ablation

작성: 2026-10-30 00:40 KST · **개정 2판: 2026-10-30 (사용자 임시 novelty · contribution과 `novelty_contribution_후보.pdf` 반영)** · writer: lead (benchmark checkout) · 기준: `feature/panda-benchmark` HEAD `96fda38a`.
개정 2판의 변경: (1) 문제정의 · contribution · method를 따로 검증하는 구조 (§0) (2) 임시 contribution과 최종 구성의 간극 (§2.1) (3) 문제정의 실험 P · perception × constraint 실험 F 신설 (§5) (4) baseline 재선정 (§6.1) (5) 판정 항목 (§10).
**개정 3판 (2026-10-30 오후, 사용자 판정 4개 반영):** (1) WBAG를 참고 문헌 · baseline에서 제거 (2) 임시 contribution 문장 유지 + 최종 구성은 사용자가 수정 (§2.1) (3) 평가 지표 · 비교 방식을 선행 논문 관행에 맞춰 점검 (§4.6) (4) 양팔 · 휴머노이드 benchmark 조사 (§5 실험 3b).
**contribution 문장은 바뀔 예정이므로 이 문서는 문장이 아니라 "주장 → 증거" 표로 쓴다. 문장이 바뀌어도 §3 표의 행을 옮기면 된다.**
상태: **계획 (결과 전). 아래 §10의 사용자 판정을 받은 뒤 각 실험의 task 문서에 기준을 고정한다.**
이 문서는 새 수치를 만들지 않는다. 인용한 숫자는 모두 기존 기록에 있는 값이며 출처를 함께 적었다.

> 읽는 순서: §0 검증 구조 → §2 위험 (특히 §2.1) → §3 주장과 증거의 연결 → §5 실험 (P · F가 새로 생김) → §6.1 baseline → §10 판정 요청.

---

## 0. 검증 구조 — 문제정의 · contribution · method (개정 2판)

사용자 제안(문제정의 · contribution · method를 각각 검증)에 **동의한다.** 한 가지를 더한다. 세 층은 필요한 증거의 종류가 다르다. 그래서 한 층의 실험으로 다른 층을 대신할 수 없다. 특히 **문제정의 검증은 우리 방법 없이도 보여야 한다.** 이것이 빠지면 reviewer는 "문제가 실재하는가"와 "해법이 그 문제를 푸는가"를 분리하지 못한다.

| 층 | 반증 가능한 질문 | 증거의 종류 | 실험 |
|---|---|---|---|
| **문제정의** | P1. 외부 segmenter의 오류가 action 오류(접촉 · 실패)로 이어지는가? <br> P2. **정확한 target mask를 줘도** 불필요한 접촉이 남는가? (target을 찾는 것 ≠ 안전하게 만지는 것) <br> P3. 같은 물체의 접촉 허용이 단계 · link마다 달라서, 고정 규칙(항상 제외 / 항상 장애물)은 서로 반대 방향으로 실패하는가? | AG3S 없이 baseline · 구성 변형만으로 보이는 존재 증명 + 실패 분류 | **P** · F · 4-B · 4-C |
| **contribution** | C-A. VLA 내부 신호(attention · depth)만으로, 외부 segmenter 없이 target 3D 영역을 얻는가? <br> C-B. 단계 · link별 접촉 제약이 같은 인식에서 접촉을 줄이는가? | 인식을 고정하고 제약을 바꾸는 비교, 제약을 고정하고 인식을 바꾸는 비교 (**2 × 3 factorial**) + noun 교체 · offline 3D 지표 | **F** · 5-A · 5-B · 4-A · 4-B · 4-C |
| **method** | 모듈이 설계대로 동작하는가? 전체가 작업을 유지하며 접촉을 줄이는가? 비용은? | 모듈 offline 지표 · benchmark closed-loop · 실행 시간 | 5-C · 5-D · 1 · 2 · 3 · 6 |

**문제정의 실험은 contribution의 종료 조건(kill criterion)이다.** P1이 성립하지 않으면(외부 segmenter 오류가 접촉으로 이어지지 않으면) C-A는 필요성이 없다. P2가 성립하지 않으면(정확한 mask + 일반 필터가 이미 안전하면) C-B는 필요성이 없다. 그 경우 그 contribution은 쓰지 않거나 "동등한 비용으로 segmenter를 제거"로 좁힌다.

PDF의 핵심 구분도 반영한다. "Attention으로 SAM만큼 정확한 2D mask를 만들 수 있는가"가 아니라 **"정밀한 2D mask 없이도 VLA의 행동 의도에 맞는 안전한 3D target 영역을 얻는가"**가 질문이다. 그래서 mask IoU는 보조 지표이고, 주 지표는 3D 거리장의 오류(잘못 지운 장애물 voxel · 남은 target voxel)와 실제 접촉 결과다.

---

## 1. 평가 대상 — AG3S 최종 구성

정본 spec: [ADOPTED_KO.md](../reviews/AG3S_decision/ADOPTED_KO.md) §A8 "A8 최종 구성 확정" 6. 서버 인자: [AG3S-CORE-EVAL-A6.impl.md](../../../AG3S/ag3s/docs/handoff/AG3S-CORE-EVAL-A6.impl.md) §3. 코드: commit `cc0d527f`.
Panda 이식: [SL6.task.md](../../../AG3S/ag3s/docs/handoff/SL6.task.md) (`AG3S-final-panda`, 진행 중).

| 모듈 | 최종 구성에서 실제로 하는 일 | 핵심 flag 묶음 |
|---|---|---|
| **ATS** (target 찾기) | prompt의 target noun token에서 image로 가는 attention(L10H7)을 depth 점에 올려 target cluster를 고른다. ball 모양 제외 + 소유 분할(가까운 이웃 표면과 나눔). 쥔 뒤에는 identity를 고정한다 | `--attention-source noun_target:L10H7` `--target-exclusion ball-owned` `--attention-split-partition` `--latched-identity-hold` |
| **HSR** (단계 판단) | VLA KV feature(layer 4)에 학습한 6 단계 linear probe(approach · grasp · lift · transport · placement · retreat) + 물리 근거(두 손가락 접촉)에 의한 grasp latch. 단계별로 허용 · 제약을 바꾼다 | `--hsr6-*` 4 flag, `--held-slip-detach` |
| **LTE** (link별 허용) | target을 뺀 거리장은 권한 link(손가락 · 손바닥)만 읽는다. 손바닥 껍질 구 +5 mm. 쥐는 순간에 한해 이웃 접촉 여유를 줄인다(손가락 20 mm · 손바닥 10 mm). 쥔 물체는 로봇에 붙은 기하가 된다. 보이는 빈 공간은 거리장에서 지운다(ROI carve) | `--target-field-policy exclude-authorized` `--palm-shell on --palm-shell-margin 5` `--neighbour-contact-mm 20` `--neighbour-palm-contact-mm 10` `--held-body-cover` `--roi-carve …` |
| **CCR** (chunk 정제) | VLA chunk를 따라가는 SQP/OSQP. 위반이 남으면 cuRobo GPU 초기값 16개로 시작점을 고른다. 앞부분 검사에 실패하면 HOLD | `--to-multi-seed 16 --to-gpu-seeds curobo` `--lookahead obstacles` `--blas-threads 1` |

> **2026-10-30:** 아래 표는 2026-10-29에 확정된 구성이다. 사용자가 최종 구성을 수정할 예정이므로, 확정(§7 단계 4) 전에는 AG3S arm의 argv · 코드를 고정하지 않는다. 이 문서의 실험은 "수정된 최종 구성"에 그대로 적용한다.

**이 구성으로 주장하지 않는 것 (논문에서도 빼거나 한계로만 적는다):**

| 항목 | 이유 | 출처 |
|---|---|---|
| 기하 grasp 보정 (G1) · 쥐기 목표 항 (A4) | flag는 켜져 있지만 HSR ④ 때문에 0 요청에서 작동했다 | ADOPTED §A8 확정 5 |
| 표면 graph extent (`attention-surface`) · feature 재연결 · 경보 | 최종 argv에 없다. 최종 extent는 ball-owned다 | A6.impl §3 argv |
| 렌더 robot mask | 끔 (구 mask만) | ADOPTED A7 |
| 꼭지 덮개 · 팔 충돌 구 | 검증 전 · 판정 대기 | OPEN_ISSUES O1 · O7 |
| "학습 없는 방법" | HSR은 학습한 probe다. "VLA는 update하지 않는다"까지만 쓴다 | 연구 범위 결정문 |

---

## 2. 다시 생각한 점 — 기존 계획 · 초안과 달라져야 하는 것

기존 계획([SAFELIBERO_PLAN](../../../paper/notes/SAFELIBERO_PLAN.md) · [EXPERIMENTS](../../../paper/evidence/EXPERIMENTS.md) P001–P006 · [연구 범위](AG3S_RESEARCH_SCOPE_20261023_KO.md))은 방향이 맞다. 다만 2026-10-29 최종 구성이 정해지면서 아래 일곱 가지가 계획에 반영되어야 한다.

### 2.1 임시 contribution과 최종 구성 사이의 간극 (가장 큼)

사용자의 임시 contribution 두 문장과 최종 구성(2026-10-29)을 줄 단위로 맞춰 본다.

| 임시 문장 | 최종 구성에서의 상태 | 지금 있는 증거 | 증거를 만들려면 |
|---|---|---|---|
| **C-A** "attention으로 target seed를 얻고 **depth 연속성으로 경계**를 정해 관측 3D 표면을 복원, 이웃 장애물과 분리" | **최종 argv에 depth 연속성 graph(`attention-surface`)가 없다.** extent는 `--target-exclusion ball-owned` + `--attention-split-partition`이고, 표면 관련 요소는 `--nontarget-evidence surface`(공 표면 구 맞춤 근거, [serve_safe.py:2180](../../../AG3S/trajopt/serve_safe.py#L2180))뿐이다 | offline: 표면 graph IoU 0.773 → 0.891 (held-out 34 episode, `main.pdf` 초안). closed-loop 허들: **Coarse(ball) 11/14 대 surface 8/14** (`main.pdf` Tab. 1). 곧 closed-loop 근거는 surface를 지지하지 않는다 | (a) surface를 켠 arm을 정식 비교에 넣거나 (b) 문장을 최종 구성에 맞게 고친다 (아래 선택지) |
| **C-B** "6 단계로 나누고 **단계마다 접촉을 허용할 물체와 link**를 정한다" | 6 단계 probe + grasp latch는 있다 (HSR6). **link 구분은 Panda `--links gripper`에서 성립하지 않는다** (§2.3). placement 단계 접촉 허용은 SL6-H에서 시험 중이다 (`--hsr6-placement-contact`, commit `96fda38a`) | RB-Y1 개발 장면의 결과 (§2.4) | 비권한 link가 있는 구성에서의 4-B · 4-C, placement 허용 시험 결과 |
| Novelty "이 예외를 지정한 link와 해당 단계**에만** 허용" | "에만"은 `exclude-authorized` + 단계 조건으로 구현돼 있다. 반증 가능한 형태는 "모든 link 면제"(`exclude-all`)와의 비교다 | 없음 (비교 자체가 없다) | 4-C |
| (PDF) "불확실한 영역은 **보수적**으로 처리: 경계가 불확실하면 target으로 단정해 장애물을 지우지 않고 Unknown에 보수적 제약" | **최종 구성에는 없다.** confidence 경보는 끔 (연구 범위 결정문). `--target-permission-strict`는 *권한 link*에만 예외를 주는 장치이고 불확실도를 쓰지 않는다 ([serve_safe.py:1753](../../../AG3S/trajopt/serve_safe.py#L1753)) | 없음 | 새 기능 + 검증이 필요하다. 최종 구성 고정 전에 구현하지 않으면 contribution에서 뺀다 (§10 판정 12) |

**사용자 판정 (2026-10-30): 임시 contribution 문장(C-A · C-B)을 유지하고, 최종 구성은 사용자가 수정한다.** 곧 위의 선택지 (a)로 확정됐다. 결과는 다음과 같다.

1. depth 연속성 surface extent가 수정된 최종 구성에 들어간다는 전제로 계획을 쓴다. 최종 구성이 확정되기 전에는 AG3S arm을 고정하지 않는다 (§1 note).
2. **extent ablation 4-G(surface 대 ball-owned 대 외부 mask extent)를 필수로 올린다.** 부록이 아니다. C-A의 "depth 연속성으로 경계를 정한다"는 이 비교가 직접 증거다.
3. 허들에서 surface가 졌던 기록(8/14 대 11/14, 이전 구성)은 **수정된 구성으로 다시 측정한다.** 다시 져도 숨기지 않고, 그 경우 C-A 문장을 "identity + 3D 영역"으로 좁힌다는 기준을 결과 전에 고정한다.
4. PDF의 "불확실 영역 보수 처리"는 임시 contribution 두 문장에 없으므로 이 계획에서 제외한다.

PDF 자신도 이 위험을 적고 있다: "Frozen VLA Attention으로 타깃 선택은 KNOWS와 겹쳐 신규성이 낮고, Visual Feature 확장은 TokenCut · CutLER와 겹친다" · "외부 Segmenter 제거와 정밀한 3D 분리가 정말 도움이 되는지는 KNOWS 대비 실험으로 증명해야 한다". 그래서 **C-A의 신규성은 "attention을 쓴다"가 아니라 "외부 segmenter를 바꿔도(또는 없어도) 같은 제약 아래 안전성이 같거나 높다"는 비교(실험 F)에 걸려 있다.**

초안의 나머지 불일치(feature 재연결 · 경보 · 표면 graph)는 최종 구성에 없으므로 본문 주장에서 뺀다. 이 방향은 판정 1이다.

### 2.2 Panda에서 AG3S는 지금까지 작업 성공을 크게 잃었다

- SafeLIBERO spatial Level I dev 20 episode (SL5i-G):
  - VLA 단독: 공식 성공 11, 충돌 18, safe success 2.
  - 가장 나은 V11AB 개선판(파지 인식 + closing 면제): 공식 성공 3, 충돌 8, safe success 3.
- LIBERO task 0–9 init 0: VLA 단독 9/10 → V11AB-B 3/10 (SL5b-libero10).
- 이 결과는 최종 구성 이전의 것이다. 하지만 **Panda 최종 구성이 이 저하를 해결했는지 모르는 채로 test를 돌리면 안 된다.** dev gate를 두고 (§7 단계 2), 통과하지 못하면 개발 한도 안에서 원인 층을 고친 뒤 고정한다.

### 2.3 Panda의 `--links gripper`에서는 "link별 허용"이 시험되지 않는다

- Panda의 constraint link는 손 + 두 손가락뿐이다. 이 셋이 모두 권한 link라서 `exclude-authorized` 와 `exclude-all` 이 같은 sphere 집합이 된다 ([exemption_ablations.md](../../../Baseline/paper_baselines/docs/exemption_ablations.md) §2.2).
- RB-Y1 최종 구성도 `--links gripper`다. 이때 비권한 link는 반대 손뿐이다. 팔과 손목은 제약에 없다.
- 논문이 "the wrist must not touch"를 주장하려면 **비권한 link가 constraint model에 있는 구성에서** ablation을 해야 한다. 그런 구성은 Panda의 `--links arms`나 RB-Y1의 팔 구(O7)다. 아니면 주장을 "권한 link와 반대 손"으로 좁힌다 (§10 판정 4).
- 같은 이유로 Panda 결과 표에는 **보호 link 열**을 둔다. 팔 link가 장애물과 부딪힌 충돌은 AG3S가 막지 않는 범위이므로 link 종류별로 나눠 보고한다.

### 2.4 RB-Y1 근거는 모두 개발 장면이다

- 최종 구성의 근거는 9 장면 × 2 (금지 접촉 없는 성공 14/18), 바구니 위 허들 7 장면 × 2 (VLA 0/14 → AG3S 6/14)다. 둘 다 구성을 고르는 데 쓴 장면이다.
- 허들 장면은 "VLA 단독 궤적이 부딪히도록" 배치했다. 그래서 이 결과는 **보호가 필요한 조건에서의 효과(stress)**를 보여 준다. 평균적인 성능(population)을 보여 주지 않는다.
- 논문에는 두 가지가 모두 필요하다. 하나는 metadata로만 고른 **held-out 대표 집합**, 다른 하나는 미리 정한 **stress 집합**이다 (실험 3).

### 2.5 baseline 구현의 충실도를 먼저 확인해야 한다

- BL-7 pilot (SafeLIBERO spatial L I dev 20 episode)에서 constrained flow matching · NUDGE · Lan-o3dp는 공식 성공이 0/20이었다. AEGIS와 KNOWS는 4/20이었다 (VLA 단독 11/20).
- 이 수치가 방법 자체의 한계인지 재구현 차이인지는 아직 가리지 못했다. 이 상태로 "AG3S가 baseline보다 낫다"를 쓰면 reviewer가 곧바로 구현을 문제 삼는다.
- 그래서 test 전에 **원 논문 조건 재현 점검**을 둔다 (§6.2). 같은 조건의 보고 수치가 있는 방법은 privileged 변형으로 맞춰 보고 차이를 기록한다.

### 2.6 이웃 접촉 여유 축소는 평가 정의와 분리해야 한다

- 최종 구성은 쥐는 순간에 이웃과의 여유를 줄인다 (손가락 20 mm · 손바닥 10 mm). 이것은 controller의 여유 설정이다.
- 평가에서는 실제 이웃 접촉을 그대로 금지 접촉으로 센다 (정의 불변). 이 원칙을 모든 표에 적는다.
- 실험 4에서 이 여유를 끈 ablation을 돌려, 이 기능이 성공과 접촉에 각각 무엇을 바꾸는지 보인다.

### 2.7 실시간성은 한계로 보고한다

- RB-Y1 요청 왕복 732 ms는 실행 창 533 ms보다 길다 (ADOPTED §A8 알려진 한계). Panda는 요청 간격이 0.25 s다.
- runtime은 주장하지 않는다. 같은 측정 경계(`ms.total`, client 왕복)로 baseline과 나란히 보고한다 (실험 6).

---

## 3. 주장 → 실험 지도

contribution 문장은 판정 1 뒤에 확정한다. **문장이 바뀌면 이 표의 "주장" 열만 바꾼다.** 증거 열과 실험은 그대로 쓴다.

| # | 층 | 주장 (제안) | 그것을 지지하는 결과 | 실험 | 무엇이 나오면 주장을 줄이나 |
|---|---|---|---|---|---|
| P1 | 문제정의 | 외부 segmenter의 오류는 거리장 오류를 거쳐 접촉으로 이어진다 | 실제 segmenter(Grounded-SAM-2 · YOLOE)의 자연 오류가 가림 · 인접 구간에 몰리고, 그 구간에서 접촉이 늘어난다. 통제된 mask 오류를 주입하면 거리장 오류 → 접촉이 단조 증가한다 | P-1 | segmenter 오류가 드물거나 접촉과 무관하면 C-A의 필요성을 "segmenter 의존성 제거(비용 · 배포)"로 좁힌다 |
| P2 | 문제정의 | **정확한 target mask(GT)를 줘도** 일반 obstacle filter는 불필요한 접촉을 남긴다 | (GT, 일반 filter) 접촉 · 실패가 (GT, AG3S 제약)보다 많다 (실험 F의 F5 대 F6). 남는 접촉 유형의 분류 (비권한 link–target, 쥔 물체–장애물, 이웃 접촉) | F · P-2 | 차이가 없으면 C-B를 쓰지 않는다 |
| P3 | 문제정의 | 접촉 허용은 단계 · link마다 다르므로 고정 규칙은 양쪽 방향으로 실패한다 | 면제 없음 → 쥐지 못함 · HOLD, 모든 link 면제 → 비권한 접촉, 단계 없음 → 운반 · 놓기에서 오류 | 4-B · 4-C | 비권한 link가 없는 구성에서만 보이면 범위를 좁힌다 (§2.3) |
| C-A | contribution | VLA 내부 신호만으로 target identity와 3D 영역을 얻는다 (외부 segmenter 불필요) | (a) 같은 제약 아래 attention 인식 ≥ 외부 segmenter 인식 (F의 AG3S 대 F2). (b) noun만 바꾸면 선택이 따라 바뀐다. (c) 3D 지표(잘못 지운 장애물 · 남은 target voxel)가 segmenter와 비슷하거나 낫다 | F · 5-A · 5-B · 4-A | (a) 외부 segmenter가 크게 낫다면 "segmenter 없이도 동작"으로만 쓴다. (b) noun을 따라가지 않으면 saliency일 수 있다. identity 주장을 뺀다 |
| C-B | contribution | 단계 · link별 접촉 제약이 같은 인식 아래 접촉을 줄이고 작업을 유지한다 | F5 대 F6 (GT 인식 고정), F1 대 F2 (segmenter 인식 고정). 4-B · 4-C · 4-D · 4-E | F · 4-B–E · 5-C | 작업 성공을 크게 잃으면 "접촉–성공 교환"으로 쓰고 크기를 보고한다 |
| S1 | method | 같은 frozen policy에서 접촉을 줄이고 작업을 유지하며 기존 filter · guidance보다 safe success가 높다 | SafeLIBERO safe success · CAR · TSR, LIBERO 비저하, RB-Y1 held-out | 1 · 2 · 3 | TSR 비열등을 통과하지 못하면 "success–safety trade-off" |
| S2 | method | 두 embodiment(RB-Y1 양팔, Panda)에 적용 | 두 로봇에서 같은 방향의 결과 | 1 · 3 | 한쪽에서만 나오면 그렇게 쓴다 |
| (한계) | — | runtime · 팔 보호 범위 · 움직이는 장애물 · **파지력 · 미끄러짐 같은 접촉 물리 실패** | 측정 그대로. PDF의 지적대로 "실패가 충돌 제약 때문"이라고 단정하지 않는다. 실패 원인 층에 "VLA 고유" · "접촉 물리"를 별도로 둔다 (§4.4) | 6 · 1 | — |

---

## 4. 모든 실험에 공통인 protocol (결과 전 고정)

### 4.1 비교 조건

| 항목 | 고정값 |
|---|---|
| Policy | Panda: `pi05_libero` (frozen, SHA 고정). RB-Y1: 16-D 30k LoRA checkpoint. LIBERO-Safety는 `pi05_libero_safety` (실험 7) |
| Paired | 같은 task · init state · policy seed (`seed*100000 + task*1000 + episode`, `suites.json`) · step 한도 · camera · controller. 조건만 바꾼다 |
| Split | LIBERO 계열: init 0–4 = dev, 5–49 = test (`suites.json` splits, 사용자 확정). dev는 head 선택 · HSR6 학습 · 개발 · pilot에만 쓴다. **test로 수정 방향을 고르지 않는다** |
| Episode 종료 | 각 benchmark의 공식 규칙 (LIBERO: `done` 또는 max_steps. SafeLIBERO: 280 policy step + wait). RB-Y1: 기존 client 규칙 (`--stop-after-place 2.5`) |
| Version | 실행 전에 commit · server argv · config dump hash를 run manifest에 기록한다. 기본 꺼짐 flag로만 구성 차이를 만든다 |

### 4.2 지표

| 층 | 지표 | 정의 · 비고 |
|---|---|---|
| **주 지표** | **safe success (SS)** | 공식 성공 ∧ 금지 접촉 없음. SafeLIBERO는 공식 판정기 (장애물 1 mm). RB-Y1는 금지 = 장애물 · 이웃 물체 · 비권한 link–target |
| 공식 지표 | TSR · CAR · ETS (SafeLIBERO), SR · CR · SSR (KNOWS 정의), official success (LIBERO) | 정의가 다른 지표를 같은 이름으로 섞지 않는다. 원자료(접촉 event · 성공)를 남겨 모든 정의를 다시 계산할 수 있게 한다 |
| 접촉 분해 | 로봇(손 · 손가락 · 팔 link별)–장애물, 쥔 물체–장애물, 손–이웃, 비권한 link–target, 탁자 · 용기 | MuJoCo 기록 재생으로 센다. controller의 판정을 쓰지 않는다 |
| 개입 비용 | HOLD 요청 비율, VLA 대비 수정량(EE mm · 관절 rad), 개입 요청 비율, 완료 시간 | **HOLD는 성공이 아니다.** 개입이 많다는 것이 보호를 뜻하지 않는다 |
| 시간 | 요청 처리 p50 · p95, client 왕복, 실행 창 초과 비율 | 첫 JIT 호출은 따로 보고한다 |
| 모듈 | ATS: identity 정답률 · 표면 precision · 보이는 표면 덮은 비율 · 다른 물체 선택. HSR: 단계 macro-F1 · 전환 지연. LTE: 잘못 지운 장애물 voxel · 남은 target voxel. CCR: 수정량 · 풀이 시간 | core evaluation §0.1과 같은 정의 |

### 4.3 통계 · 판정

- **사전 등록:** 실험마다 task 문서에 episode 목록 hash · arm · 주 지표 · 판정 규칙을 먼저 commit하고 test를 시작한다.
- **paired 비교:** 이진 결과는 exact McNemar. 효과 크기는 task(또는 장면) 단위 cluster bootstrap 95 % CI (2,000회).
- **다중 비교 순서 (gatekeeping):** ① AG3S 대 VLA 단독 SS 개선 → ② TSR 비열등 → ③ 각 baseline 대비 SS (Holm 보정). 앞 단계를 통과해야 다음 주장을 쓴다.
- **판정 기준 (제안, §10 판정 7):** SS 개선 = 점추정 +5 %p 이상 ∧ CI 하한 > 0. TSR 비열등 = ΔTSR의 CI 하한 > −5 %p (연구 범위 결정문의 5 %p를 따름).
- **검정력 (참고 계산):** paired 이진에서 불일치 비율 0.2, 효과 +10 %p를 검정력 0.8로 보려면 약 160 쌍이 필요하다. 효과 +5 %p(불일치 0.1)는 약 310 쌍이다. 그래서 suite 단위 표는 기술 통계로만 보고하고, 판정은 suite를 합친 전체로 한다.
- 실패 · HOLD · timeout · execution_error를 분모에서 빼지 않는다. 판정할 수 없는 값은 unknown으로 남기고 coverage를 보고한다.

### 4.4 실패 원인 기록

AG3S arm의 실패마다 아래 중 하나를 적는다. 같은 episode에서 VLA 단독이 어떻게 끝났는지도 함께 적는다.

| 원인 층 | 판단 근거 |
|---|---|
| VLA 고유 | VLA 단독도 같은 방식으로 실패하고 AG3S가 궤적을 거의 바꾸지 않음 |
| ATS | 실패 시점에 target이 틀렸거나 없음 (요청 기록 + GT) |
| HSR | 단계 · latch가 틀려 허용이 잘못 켜지거나 꺼짐 |
| LTE | 거리장에 target이 남았거나 장애물 · 이웃이 지워짐 |
| CCR | HOLD 반복, 해 없음, 수정 궤적 때문에 쥐기를 놓침 |
| 접촉 물리 | 충돌 제약과 무관한 파지력 부족 · 미끄러짐 · 물체 미끄러짐으로 실패 (PDF 지적). **AG3S가 해결한다고 주장하지 않고** 별도 행으로 센다 |
| adapter | 역변환 · 관측 변환 문제 |

baseline은 인식 · 필터 · 변환으로 나눈다. 원인별 표를 각 실험 결과에 넣는다.

### 4.5 결과 보기와 기록

- 실험 단계가 끝날 때마다 `outputs/<날짜>_<실험 설명>/`에 영상(`python -m common.experiment_view`) · 실제 씬 · 그래프 · 표 · README를 만든다.
- 원시 기록은 `AG3S_VLA-storage/results/benchmark/<날짜_commit_run-id>/`에 둔다.
- 기록 위치: benchmark(SL · P 단계)는 `AG3S_BENCHMARK_LOG.md`, baseline(BL 단계)은 `AG3S_BASELINE_LOG.md`. scribe가 `verify.json`의 수치만 옮긴다.

### 4.6 평가 지표 · 비교 방식 — 선행 논문 관행과 우리의 선택 (개정 3판)

2026-10-30 웹 검색으로 확인했다. **검색 결과 요약과 초록만 읽었고 본문은 읽지 않은 항목이 많다.** 본문 인용 전에 원문을 확인한다 (표의 "확인 수준").

| 논문 | benchmark · 규모 | 지표 | 확인 수준 |
|---|---|---|---|
| VLSA / AEGIS ([2512.11891](https://arxiv.org/abs/2512.11891)) | SafeLIBERO, 32 scenario · 1,600 episode | CAR · TSR · ETS. 남는 충돌 22.15 %를 인식 · 위치 오류로 설명 | 검색 요약 |
| KNOWS ([2606.09749](https://arxiv.org/pdf/2606.09749)) | SafeLIBERO + 움직이는 장애물 | SR · CR · SSR (repo 노트). 충돌률 43 % 이상 감소 | 검색 요약 + repo 노트 |
| SafeVLA-Bench ([2606.00773](https://arxiv.org/abs/2606.00773)) | LIBERO · RoboCasa-365, 정책 27 조합 | native success · safety rate · **success-but-unsafe(SBU)** · **Violation Severity Index(VSI, 최악 위반 깊이)**. 안전 조건은 STL: 과도한 접촉 · 주변 물체 교란 · 쥔 물체 불안정 · self-contact. RoboCasa-365 성공 rollout의 36–56 %가 위반 | 초록 |
| LIBERO-Safety ([2606.23686](https://arxiv.org/abs/2606.23686)) | 물리 · 의미 안전, 8 VLA | 충돌 없는 미완료(보수적 정지로 성공을 포기)가 생김을 지적 | 초록 + 검색 요약 |
| SafeBimanual ([CoRL 2025](https://proceedings.mlr.press/v305/deng25c.html)) | RoboTwin 8 task + 실물 4 task, diffusion policy | success rate · unsafe interaction rate | 검색 요약 |
| CoFreeVLA ([2601.21712](https://arxiv.org/abs/2601.21712)) | 양팔 5 task, VLA 6종, variant당 30 trial | success 0.45 → 0.61, collision 0.54 → 0.23 | 초록 |
| RoboTwin 2.0 ([2506.18088](https://arxiv.org/abs/2506.18088)) | 50 양팔 task | task당 100 episode, Easy · Hard 분리 보고 | 검색 요약 |
| 통계 관행 ([PhAIL 2605.29710](https://arxiv.org/pdf/2605.29710), [N-SCORE, RSS 2025](https://www.roboticsproceedings.org/rss22/p076.pdf) 등) | — | 많은 연구가 조건당 20–40 trial, CI 없음, paired 검정 없음. paired 차이 5 %p는 이진 결과로 수백 쌍 이상 필요. Wilson CI 권장 | 검색 요약 (원문 미확인) |

**이 관행에서 우리 계획에 가져오는 것**

| 관행 | 우리의 선택 | 반영 위치 |
|---|---|---|
| 성공과 안전을 **따로 + 함께** 보고 (SafeVLA-Bench의 SBU, LIBERO-Safety의 "충돌 없는 미완료" 지적) | 주 지표 SS 외에 **SBU(성공했지만 금지 접촉이 있음)** 와 **충돌 없는 미완료**를 모든 표에 낸다. CAR만 높은 방법이 정지로 얻은 결과인지 가른다 | §4.2 |
| 위반의 **심각도** (VSI) | 보조 지표로 **금지 접촉의 최대 침투 깊이(mm) · 지속 시간**을 MuJoCo 기록에서 계산한다. 사전 정의를 task 문서에 고정한다 | §4.2 |
| 접촉 유형 분류 (SafeVLA-Bench의 STL 조건) | 우리 접촉 분해(로봇–장애물 · 쥔 물체–장애물 · 이웃 · 비권한 link–target)의 이름을 이 네 유형과 대응시켜 표에 병기한다 | §4.2 |
| 각 benchmark의 **공식 지표를 공식 정의대로** (VLSA · KNOWS가 서로 다름) | 이미 §4.2에 있다. re-run과 reported를 한 열에 섞지 않는다 | §4.2 · 실험 1 |
| 적은 trial · CI 없음은 약점 | 우리는 조건당 640 / 160 episode, **paired exact McNemar + task 단위 cluster bootstrap + 셀별 Wilson 95 % CI**를 쓴다. suite별 표는 기술 통계로만 | §4.3 |
| 외부 평가자 | SafeVLA-Bench 등 **제3자 안전 정의로 같은 rollout을 다시 판정**하면 "우리가 정의한 안전" 비판을 줄인다. rollout 형식 호환 · 코드 공개는 **미확인** → 판정 14 | §10 |

---

## 5. 실험

### 실험 P — 문제정의 검증 (AG3S 없이 보이는 존재 증명)

| ID | 질문 | 방법 | 사전 고정할 기준 (제안) |
|---|---|---|---|
| P-1 | 외부 segmenter의 오류가 접촉으로 이어지나 (P1) | **offline:** 기록 frame에서 target mask에 통제된 오류를 넣는다 — 경계 팽창 · 침식(2 · 4 · 8 px), 이웃 물체와 병합, 한 camera 누락, 이웃 identity 교체. 같은 depth · lift · robot 제거(`paper_baselines/lift.py`)로 3D 거리장 오류(잘못 지운 장애물 voxel · 남은 target voxel)를 잰다. 대조로 Grounded-SAM-2 · YOLOE의 **실제** 오류율을 같은 frame에서 잰다. **closed-loop:** 실험 F의 F1 · F2 실패를 인식 층 대 제약 층으로 분류한다 | 자연 오류가 가림 · 인접 frame에 집중되는지(층별 오류율 표), 오류 크기 대 거리장 오류의 단조성. 접촉과의 연결은 F의 실패 분류로 본다 |
| P-2 | 정확한 mask에서도 접촉이 남나 (P2) | 실험 F의 (GT, 일반 filter) arm F5의 접촉을 분류한다: 비권한 link–target · 쥔 물체–장애물 · 이웃 접촉 · 경로 접촉 | F5 대 F6의 SS 차이 (paired McNemar). 분류표 |
| P-3 | 허용이 단계 · link마다 다른가 (P3) | 4-B · 4-C를 그대로 쓴다 | 4-B · 4-C와 같다 |

- 외부 segmenter 오류가 "충분히 흔한가"는 숫자로 먼저 본다. 이 값이 작으면 P1의 필요성이 약하므로 §0의 종료 조건을 적용한다.
- RoboGround의 "GT mask를 써도 Easy task 성공률이 64 % → 68 %뿐"이라는 보고는 PDF에서 가져온 것이며 원문은 확인하지 않았다. P2의 외부 선행 근거로 쓰려면 먼저 원문에서 확인한다.

### 실험 F — perception × constraint factorial (contribution의 핵심 비교)

PDF의 B1–B3를 확장한 **인식(행) × 제약(열) 표**다. 한 번에 한 축만 바꾼다. 그래서 "VLA 내부 grounding이 도움이 되는가"와 "정확한 인식에서도 접촉 제약이 필요한가"를 분리한다.

| 인식 ↓ / 제약 → | **일반 obstacle filter** (타원체 CBF, target은 mask로 제외. AEGIS · KNOWS 방식) | **AG3S 접촉 제약** (거리장 + 단계 + link별 허용 + 붙은 물체 + CCR) |
|---|---|---|
| **외부 segmenter** (Grounded-SAM-2 per-frame · causal track) | **F1** = AEGIS · KNOWS (실험 1 주 표 재사용) · PDF의 B1 | **F2** = PDF의 **B2** (같은 제약, 외부 인식. 강한 외부 baseline) |
| **VLA attention + depth** | (선택) F3 = attention target + 일반 filter | **AG3S 최종 구성** |
| **GT mask** (privileged) | **F5** = privileged AEGIS · KNOWS (GT) | **F6** = PDF의 **B3** (인식 상한) |

| 비교 | 답하는 질문 |
|---|---|
| **AG3S 대 F2** | 같은 제약에서 VLA 내부 grounding이 외부 segmenter보다 나은가, 최소한 같은가? **(C-A의 직접 증거)** |
| **F6 대 AG3S** | AG3S의 실패가 인식 때문인가, 제약 · 경로 때문인가? (PDF: B3와의 비교로 구분) |
| **F5 대 F6** | **정확한 인식에서도 접촉 제약이 필요한가?** (P2 · C-B의 직접 증거) |
| **F1 대 F2** | 같은 외부 인식에서 접촉 제약의 효과 (C-B의 재현) |

| 항목 | 내용 |
|---|---|
| 집합 | SafeLIBERO **Level I** (target 인접 장애물, 인식 · 구분이 핵심) 4 suite × 4 task × **test init 5–14 = 160 episode/cell**. 실험 1의 test 목록(init 5–24)에 포함되므로 F1 · AG3S · VLA 단독은 재사용한다. 새로 도는 cell은 F2 · F5 · F6 (480 episode) |
| 주 지표 | SS. 판정은 위 네 비교 각각의 paired 차이와 CI. 비교 순서는 AG3S 대 F2 → F5 대 F6 → 나머지 (사전 등록) |
| 해석 규칙 (결과 전 고정) | AG3S ≥ F2 (CI 하한 > −5 %p): C-A 성립. AG3S < F2: "외부 segmenter가 더 안전"이라고 쓰고 C-A를 "segmenter 없는 대안"으로 좁힌다. F6 ≈ AG3S: 병목은 인식이 아니다 |
| 전제 작업 | **closed-loop에서 외부 mask를 target 인식으로 넣는 경로가 아직 없다.** `paper_baselines` provider는 offline(frame → mask → lift → 3D 지표)에서만 쓰이고 `serve_safe` · Panda server에 provider 입력 flag가 없다 (2026-10-30 grep 확인). 기본 꺼짐 flag로 구현해야 한다 (implementer, task 문서). 구현은 SL6 smoke가 runner를 쓰는 동안 시작하지 않는다 |
| B2 강화 | PDF는 B2를 "Grounded SAM + 다중 뷰 · 추적 개선"으로 정의한다. 기본은 `grounded_sam2:track`(이미 있음). PDF 표의 DAM4SAM · MaskClustering · AutoSeg3D · MVGGT는 원문 · 공개 코드를 확인한 뒤 dev smoke에서 쓸지 정한다. **외부 baseline을 일부러 약하게 두지 않는다** — 약하면 C-A가 무의미해진다 |
| 비용 | 480 episode × t_AG3S. t_AG3S = 140 s(가정, SL6 G3에서 측정)이면 약 19 h, 독점 lock, 동시 1 job |

---

### 실험 1 — SafeLIBERO 주 비교 (논문 주 결과 표)

| 항목 | 내용 |
|---|---|
| 질문 | 같은 `pi05_libero`에서 AG3S가 VLA 단독과 기존 safety filter · guidance보다 safe success가 높은가? 작업 성공은 얼마나 잃는가? |
| 왜 이 benchmark | 가장 가까운 선행연구(VLSA/AEGIS, KNOWS, multi-link CBF, constrained flow matching)가 같은 checkpoint로 보고한 공개 benchmark다. Level I(target 근처 장애물)은 target · 장애물 구분(ATS · LTE)을, Level II(경로 위 장애물)는 경로 회피 · 붙은 물체(CCR)를 시험한다. 그래서 level별 결과가 모듈 주장과 바로 이어진다 |
| 범위 | 4 suite (spatial · object · goal · long) × Level I · II × 4 task. **test init 5–24 (task당 20) = arm당 640 episode** (제안, §10 판정 2). 전체 45개(1,440)는 결과를 보기 전에 정할 때만 쓴다 |
| arm — 주 표 | VLA 단독 · **AG3S 최종 구성 (Panda)** · AEGIS · KNOWS · multi-link CBF · OmniGuide (모두 non-privileged) |
| arm — 보조 표 (spatial만) | constrained flow matching · NUDGE · Lan-o3dp guidance (non-privileged), 그리고 privileged 변형 (GT 장애물: AEGIS · KNOWS · CFM · multi-link) |
| 보고 블록 | 원 논문 보고 수치는 따로 둔다 ("reported", 조건이 같을 때만 같은 열). 각 셀에 re-run / reported를 표시한다 |
| 주 지표 · 판정 | SS. §4.3의 순서를 따른다 |
| 함께 보고 | TSR · CAR · ETS · KNOWS 지표, link 종류별 접촉, HOLD · 개입률, 시간, level별 · suite별 표, 실패 원인 층 |
| 재사용 | SL5c VLA 단독 · BL-7 pilot은 dev 기록이다. 주 표에 넣지 않고 개발 근거(부록)로만 쓴다 |
| 선행 조건 | 실험 1을 시작하기 전에 §7 단계 1–3(Panda 이식 · dev gate · baseline 충실도)을 마친다 |

### 실험 2 — LIBERO 비저하 ("장애물이 없으면 해가 없다")

| 항목 | 내용 |
|---|---|
| 질문 | 장애물이 없는 표준 과제에서 AG3S가 작업 성공을 떨어뜨리지 않는가? (safety layer의 거짓 개입 비용) |
| 범위 | libero_spatial · object · goal · 10, 각 10 task × test init 5–14 = **arm당 400 episode** (제안) |
| arm | VLA 단독 · AG3S. baseline은 spatial만 (같은 400의 1/4)에서 주 표 4개 방법 |
| 판정 | official success 비열등 (ΔTSR CI 하한 > −5 %p) |
| 함께 보고 | 개입 요청 비율 · HOLD · 수정량 (장애물이 없는데 얼마나 개입하나), 실패 원인 층 |
| 기준 | 공개 수치 98.8 / 98.2 / 98.0 / 92.4 % (`suites.json` reference)는 VLA 단독 재현 확인용이다 |

### 실험 3 — RB-Y1 MuJoCo: held-out 대표 집합 + stress 집합

| 항목 | 내용 |
|---|---|
| 질문 | 방법을 개발한 양팔 16-D 환경에서, 개발에 쓰지 않은 장면으로 같은 결론이 나오는가? 보호가 필요한 조건에서 얼마나 보호하는가? |
| 대표 집합 | metadata로만 고른 새 장면 40개 (장애물 없음 · 이웃 과일 인접 · 놓는 곳 기둥 · 가림 기둥 층별로 나눔) × 흔들림 2 = arm당 80 run. **개발에 쓴 장면 ID(119 · 250 · 320 · 850 · 954 · 211 · 1345 등)는 뺀다.** 목록 hash를 먼저 commit한다 |
| stress 집합 | 미리 정한 배치 규칙(T43 P2 · core evaluation §0.2)으로 만든 장애물: 운반 경로 막대, 바구니 위 허들, 가림 기둥, 이웃 과일 10 mm 이내. 각 10 장면 × 2. 배치는 VLA 궤적과 겹치게 고르므로 "보호가 필요한 조건"이라고 명시한다 |
| arm | VLA 단독 · AG3S 최종 구성. stress 집합에서는 실험 4의 ablation arm을 함께 돌린다 |
| 주 지표 | 금지 접촉 없는 성공 (이웃 과일 · 장애물 · 비권한 link–target 접촉 = 금지). 탁자 · 바구니 접촉은 따로 센다 |
| 함께 보고 | 장애물 있음 / 없음 층, 접촉 분해, 쥐기 성공, 실패 원인 층, 시간 |
| 재사용 | 9 장면 × 2 · 허들 7 × 2 결과는 "개발 결과"로 부록에 둔다 |

### 실험 3b — 공개 양팔 · 휴머노이드 benchmark (선택, feasibility 먼저)

사용자 질문: "RB-Y1처럼 양팔 · 휴머노이드에 특화된 benchmark가 있는가?" 조사 결과 (2026-10-30, 검색 기반):

| benchmark | 로봇 · simulator | 규모 · 특징 | 장애물 · 접촉 안전 지표 | VLA 지원 | AG3S 적용 시 필요한 일 |
|---|---|---|---|---|---|
| **RoboTwin 2.0** ([2506.18088](https://arxiv.org/abs/2506.18088)) | Aloha-AgileX 등 5 embodiment · SAPIEN | 양팔 50 task, Hard = clutter · 배경 · 조명 · 책상 높이 randomization. 공식 task당 100 episode | 공식 안전 지표 **없음**. SafeBimanual이 unsafe interaction을 자체 정의 | π0 · π0.5 결과가 여러 논문에 있고 LeRobot 문서가 있다. 공개 π0.5 checkpoint의 품질은 **미확인** | 로봇 모델(cuRobo용) · camera · 14-D joint action 역변환 · SAPIEN 접촉 읽기 · 금지 접촉 정의. **큼** |
| **BiGym** (CoRL 2024) | Unitree H1 휴머노이드 · **MuJoCo** | 40 task, RGB-D 3대, whole-body / bimanual 모드, human demo | 없음 | 모방 · RL 중심. VLA checkpoint는 **없는 것으로 보임**(미확인) | 우리 MuJoCo 접촉 평가 도구는 재사용 가능. 단 VLA를 학습해야 함. **큼** |
| **RoboCasa GR1 tabletop** (GR00T N1) | GR-1 휴머노이드 · 손가락 hand | 탁자 rearrangement | 없음 | GR00T · OmniGuide가 사용 | 손가락 hand의 link 모델 필요. 우리 gripper 모델과 다름 |
| **BEHAVIOR-1K challenge** | OmniGibson (Isaac Sim) | 가정 장기 50 task, 우승 q-score 26 % | 없음 | π0.5 기반 우승 솔루션 | 성공률이 낮아 safe-success 비교가 어렵다. **부적합** |
| HumanoidBench · SIMPLE | MuJoCo 휴머노이드 · whole-body | RL 중심 / 60 whole-body task (SIMPLE은 미확인) | 없음 | VLA 평가 용도 아님 | 부적합 |

**결론:**

- **양팔 · 휴머노이드에서 장애물 접촉 안전을 공식 지표로 평가하는 공개 benchmark는 이 조사 범위에서 찾지 못했다.** 한 survey는 표준 양팔 benchmark 자체가 없다고 적었다 (검색 요약). SafeBimanual · CoFreeVLA는 자체 정의 지표로 보고한다.
- 따라서 **RB-Y1 MuJoCo 평가는 우리가 정의하는 benchmark다.** reviewer가 재현할 수 있도록 장면 생성 규칙 · 목록 hash · 접촉 정의를 공개하고, 본문에서 "공개 양팔 safety benchmark가 없어 직접 만들었다"고 쓴다. 이것은 약점이면서 설명 가능한 사실이다.
- 공개 benchmark 한 개를 더하려면 **RoboTwin 2.0 Hard의 clutter가 장애물이 되는 task 8–10개**가 가장 현실적이다. 같은 `π0.5 + AG3S 대 π0.5` 비교를 우리 접촉 평가자로 센다. SafeBimanual의 보고 수치는 base policy(diffusion)가 달라서 같은 열에 두지 않는다. **로봇 이식 비용이 크므로 판정 13에서 feasibility만 먼저 확인한다.**
- BiGym은 MuJoCo라 평가 도구는 맞지만 VLA를 새로 학습해야 해서 이 논문의 범위를 넘는다.

### 실험 4 — 모듈 대체 ablation (closed-loop, "왜 각 모듈이 필요한가")

원칙: **최종 구성에서 한 가지만 바꾼다** (leave-one-out). 바꾼 자리에는 가능하면 선행연구의 방식을 넣는다. 그래서 각 행이 "그 모듈을 기존 방식으로 하면 무엇을 잃는가"에 답한다.
장소: 효과가 나타나는 집합에서만 돌린다. RB-Y1 stress 집합(실험 3)과 SafeLIBERO의 해당 level dev가 그런 집합이다. dev 결과는 ablation의 방향만 보여 주며 test 주 표와 섞지 않는다.

| ID | 바꾸는 것 | 대체 (선행 방식) | 기대 · 관찰 | 집합 |
|---|---|---|---|---|
| 4-A | ATS 입력 (Panda SafeLIBERO는 **실험 F로 승격**. 여기는 RB-Y1 stress용) | ① 외부 segmenter (Grounded-SAM-2 instruction prompt, KNOWS · VLSA 방식) ② action token attention L8H2 (이전 AG3S) ③ GT target mask (oracle, privileged) | C1(c). ③은 상한이다 | RB-Y1 stress + SafeLIBERO L I dev |
| 4-B | HSR 단계 근거 | ① gripper 상태 규칙 ② GT 단계 (oracle) ③ 단계 없음 (허용 고정) | C2 | RB-Y1 stress (허들 · 놓는 곳) |
| 4-C | target 면제 | ① 면제 없음 = target도 장애물 (일반 geometric safety layer, `--target-field-policy relax`) ② 모든 link 면제 (`exclude-all`, 부피 면제 방식) | ①은 쥐지 못함 · HOLD, ②는 비권한 link–target 접촉 | **비권한 link가 있는 구성에서만** (§2.3): RB-Y1 + 팔 구 또는 Panda `--links arms` |
| 4-D | 붙은 물체 기하 | 끔 (쥔 물체를 충돌 계산에서 뺌) | 운반 중 물체–장애물 접촉 증가 | RB-Y1 운반 막대 · SafeLIBERO L II dev |
| 4-E | 손바닥 껍질 구 · 쥐는 순간 이웃 여유 | 각각 끔 | 손바닥–장애물 접촉 대 쥐기 성공의 교환 | RB-Y1 인접 장면 |
| **4-G** | **target extent** | ball-owned (2026-10-29 구성) 대 depth 연속성 surface (수정 구성) 대 외부 segmenter mask로 extent를 정함 (Grounded-SAM-2) | **C-A의 "depth 연속성 경계" 직접 증거.** 잘못 지운 장애물 · 남은 target voxel + 접촉 | SafeLIBERO L I dev · test 일부 + RB-Y1 인접 장면 |
| 4-F | CCR 초기값 | GPU 여러 초기값 끔 (CPU 단일) · HOLD만 하는 정지 필터 | 성공 대 시간 | RB-Y1 stress |

- 표본: 각 집합 20 장면 × 2 (arm당 40 run)가 상한이다. 효과 크기를 판정하지 않고 방향 · 접촉 유형의 변화를 보고한다. 큰 효과(예: 4-C ①의 쥐기 실패)만 주장한다.
- 4-A ③ · 4-B ②의 oracle 행은 privileged로 표시한다. deployable 방법과 같은 열에 두지 않는다.

### 실험 5 — offline 모듈 분석 (같은 입력, closed-loop 없이)

| ID | 질문 | 방법 | 데이터 | 비교 |
|---|---|---|---|---|
| 5-A | 지시 물체를 얼마나 정확히 찾나 | 기록 frame에 attention source만 바꿔 다시 재생 (T65 고정 입력 재생 도구) | Panda: 실험 1 · 2의 AG3S test 기록. RB-Y1: 실험 3 대표 집합 기록 | Grounded-SAM / SAM-2 (per frame · track), YOLOE, localization heads, TokenCut, KNOWS head L12H3, GT oracle. 같은 depth · lift · robot 제거 (`paper_baselines/lift.py`) |
| 5-B | attention이 정말 지시문을 따르는가 (saliency가 아닌가) | **noun 교체 counterfactual**: 같은 RGB-D에서 prompt의 target noun을 장면에 있는 다른 물체로 바꾼다. 선택된 물체가 바뀐 noun을 따라가는 비율을 잰다. action은 실행하지 않는다 | libero_object · libero_goal (한 장면에 여러 물체), RB-Y1 과일 장면 (사과 · 배 · 오렌지 · 바나나) | 같은 교체 prompt를 받은 외부 segmenter, 무작위 head, uniform |
| 5-C | 단계 판단이 맞나 | HSR6 probe 예측 대 MuJoCo 물리 단계 label (SL6-B B3 규칙) | Panda: test 기록 (학습은 spatial dev만 → 다른 suite는 분포 밖 일반화로 따로 표시). RB-Y1: 대표 집합 | gripper 상태 규칙, GT |
| 5-D | 거리장이 맞게 만들어지나 | 쥐기 직전 요청에서 잘못 지운 장애물 · 이웃 voxel, 남은 target voxel, class별 | 실험 1 · 3 기록 | 4-A arm들의 같은 요청 |

- 5-A · 5-B는 GPU 공유 lock으로 돌릴 수 있다 (offline 추출, 시간을 보고하지 않음).
- extent(표면 graph 대 ball-owned) 분석은 사용자 판정 1에 따라 부록에 넣는다.

### 실험 6 — 실행 비용

- 측정 경계: 서버 요청 처리(`ms.total`) · 단계별(인식 · 거리장 · TrajOpt) · client 왕복 · GPU memory.
- arm: AG3S와 실험 1의 모든 baseline (같은 경계). VLA 단독은 기준이다.
- 보고: p50 · p95, 실행 창 초과 비율 (Panda 0.25 s, RB-Y1 533 ms). 첫 JIT 호출은 따로 적는다.
- **주의:** TrajOpt는 시간 마감으로 반복 수가 정해진다. GPU를 나눠 쓰면 결과가 바뀐다. 그래서 AG3S closed-loop은 독점 lock, 동시 1 job으로 돌린다 (SL6 §5 G3과 같다).

### 실험 7 — LIBERO-Safety (선택, 외부 일반화)

| 항목 | 내용 |
|---|---|
| 왜 선택인가 | (1) controller가 LIBERO와 다르다 (OSC ±2, kp 750 — action 1 단위가 약 1 m). 그래서 Panda 역변환을 다시 맞춰야 한다. (2) checkpoint가 다르다 (`pi05_libero_safety`). head 선택과 HSR6 학습을 다시 해야 한다. (3) 공식 cost 신호가 robot 접촉을 잡지 못하고, rollout script와 step 한도가 공개되지 않았다 ([BM-SETUP-B.impl.md](../../../AG3S/ag3s/docs/handoff/BM-SETUP-B.impl.md)) |
| 제안 범위 | 실험 1–3이 끝난 뒤. 물리 suite 중 tabletop spatial avoidance · free-space hand–object avoidance의 L0 (정적), 5 task × test 10 init. VLA 단독 대 AG3S. 접촉은 우리 MuJoCo evaluator로 센다 |
| 판정 | §10 판정 6 |

### 실험 8 — 실물 RB-Y1 (선택, 시연)

- 개선을 주장하지 않는다. 실행 가능성만 보인다 (기존 계획 P006).
- 렌더 mask를 끈 결정(A7)의 근거 중 하나가 실물 calibration이다. 그래서 실물 시연은 같은 구성으로 한다.

---

## 6. Baseline 선정과 공정성

### 6.1 선정 이유 (개정 2판: 사용자 PDF의 선행연구 비교를 반영)

**WBAG는 사용자 판정(2026-10-30)으로 참고 문헌 · baseline에서 제외했다.** 제거하면 C-B(붙은 물체 · whole-body 보호)에서 선행연구 비교가 multi-link filter [1] 하나로 줄고, 이 주제를 아는 reviewer가 누락을 지적할 수 있다. 이 위험은 알고 간다. C-B의 "붙은 물체" 부분은 비교 대상 없이 ablation 4-D로만 증거를 만든다.

선정 기준 세 가지: (1) 우리 contribution의 가장 가까운 선행연구인가 (2) 같은 benchmark · 같은 `pi05_libero`로 비교할 수 있는가 (3) 실행 가능한가 (code · checkpoint). 어느 하나라도 빠지면 "보고 수치 인용" 또는 "관련 연구"로 내린다.

| 등급 | 방법 | 가장 가까운 contribution | 이 논문에서 답하는 질문 | 실행 |
|---|---|---|---|---|
| **0 하한** | π0.5 단독 | — | 장애물을 모르는 policy | 실험 1 · 2 · 3 |
| **1 주 표 (실행)** | **KNOWS** [17] | C-A | **가장 가깝다.** VLA attention으로 target을 고르고 YOLOE 타원체 CBF를 쓴다. 우리와 차이: target 3D 영역, link · 단계별 허용, 거리장. PDF: attention으로 target을 고르는 것만으로는 신규성이 낮다 → 이 비교가 C-A의 본체다 | 실험 1 |
| | **VLSA / AEGIS** [10] | C-A · P1 | SafeLIBERO 저자의 방법. 외부 detector + 타원체 CBF (VLM 단계는 대체, `deviations`) | 실험 1 |
| | **multi-link safety filter** [1] | C-B (link) | link별 보호 (CBF). 우리 LTE와 대비. 재구현(partial) | 실험 1 |
| | **OmniGuide** [19] | S1 | sampling 단계 guidance. EE delta로 이식 | 실험 1 |
| **1b 주 표 (구성한 baseline, 같은 시스템)** | **B2: Grounded-SAM-2 + AG3S 제약** (F2) | C-A | 가장 강한 외부 인식 baseline. 같은 제약에서 인식만 바꾼다 | 실험 F |
| | **B3: GT mask + AG3S 제약** (F6) | — | 인식 상한. 실패가 인식 때문인지 구분 | 실험 F |
| | **GT mask + 일반 filter** (F5) | C-B · P2 | 정확한 인식에서도 접촉 제약이 필요한가 | 실험 F |
| **2 보조 (spatial만)** | constrained flow matching [5] · NUDGE [9] · Lan-o3dp guidance [15] | S1 | sampling guidance 계열. 0/20 pilot이 있어 충실도 점검과 함께 | 실험 1 보조 |
| **offline 비교** | Grounded-SAM / SAM-2 · YOLOE · localization heads · TokenCut · KNOWS head · GT | C-A | 3D 영역의 정확도 · 다른 물체 선택. PDF가 겹친다고 적은 TokenCut · CutLER는 이 offline 비교에서 다룬다 (CutLER는 카탈로그에 없다) | 실험 5-A · P-1 |
| **모듈 대체 (같은 CCR)** | 면제 없음 · 모든 link 면제 · 단계 없음 · 붙은 물체 끔 | C-B · P3 | RB-Y1 + 비권한 link 구성 | 실험 4 |
| **관련 연구만 (실행 안 함)** | RoboGround [11] · GuidedVLA [13] · OBEYED-VLA [24] | — | 학습된 grounding policy라 다른 policy가 된다. 본문에서는 "segmentation 오류가 행동 오류로 전파된다"(OBEYED-VLA)는 P1의 외부 근거로만 인용한다 (원문 확인 후) | — |
| | AFI (CVPR 2026) · EXAONE-VLA | — | AFI는 Grounded-SAM + depth로 affordance field를 만들어 policy에 개입한다. 안전 필터가 아니다. EXAONE-VLA는 mobile manipulation. 둘 다 카탈로그에 없고 원문을 확인하지 않았다 | — |

**PDF의 "참고할 만한 논문"(MaskClustering · DAM4SAM · AutoSeg3D · MVGGT · ReconVLA · AtVLA)** 은 baseline이 아니라 (1) B2를 강화할 후보와 (2) 관련 연구 서술의 후보다. 이 문서는 PDF의 한 줄 설명만 읽었다. 코드 공개 · 적용 가능성은 확인하지 않았다.

### 6.2 공정성 · 충실도 점검 (실험 1 전에)

1. **같은 입력:** 모든 방법은 같은 camera · depth · checkpoint · seed · step 한도를 쓴다. 장애물 인식이 필요한 방법은 non-privileged(검출)와 privileged(GT)를 따로 표시한다.
2. **보호 범위 열:** 각 방법이 보호하는 link(EE만 · 손 · 전체 link)를 표에 적는다.
3. **충실도 점검:** 원 논문이 같은 benchmark · checkpoint로 보고한 방법(AEGIS · multi-link · CFM)은 원 조건과 맞춘 privileged 변형으로 dev에서 돌려 보고 수치와 비교한다. 차이가 크면 원인(구현 · 조건 차이)을 기록하고, 주 표에 "re-implementation, not reproduced"로 표시한다. **이 점검 결과로 AG3S 쪽 설정을 바꾸지 않는다.**
4. **원 방법과의 차이:** `paper_baselines/README.md`의 deviations 열을 논문 부록 표로 옮긴다 (예: AEGIS의 VLM 단계 대체, OmniGuide의 EE delta).
5. **조정 예산:** baseline의 hyperparameter는 원 논문 값을 쓴다. 우리가 정해야 하는 값(예: NUDGE d_safe)은 dev에서 정하고, AG3S의 개발 예산과 같은 수준으로 제한한다.

---

## 7. 진행 순서와 gate

| 단계 | 내용 | 통과 조건 (결과 전 고정) | GPU |
|---|---|---|---|
| **0** | 이 계획 판정 · 실험별 task 문서 · 사전 등록 (episode 목록 hash) | 사용자 판정 | — |
| **1** | Panda 최종 구성 이식 마무리 (SL6: HSR6 학습 · head 재선택 · G1/A4 smoke) | SL6 §2.1 · §3.1 · §6 기준 | 독점 / 공유 |
| **2** | **Panda dev gate:** SafeLIBERO dev (4 suite × 2 level × 4 task × init 0–4 = 160 episode) + LIBERO dev (4 suite × 10 task × init 0 = 40), VLA 단독 대 AG3S | 제안: SS가 VLA보다 높고 ∧ TSR 하락 ≤ 10 %p (dev라 느슨하게). 실패하면 원인 층을 진단하고 **수정 묶음은 최대 2 번** (각각 새 version 이름). 그 뒤에는 결과와 상관없이 고정한다 | 독점 |
| **3** | baseline 충실도 점검 (§6.2) + 주 표 방법 dev smoke. **외부 mask provider flag 구현 + F2 · F5 · F6 dev smoke** (판정 11) | 실행 오류 · fallback 0, 원 수치 대비 차이 기록 | 공유 가능 (TrajOpt 없음) |
| **4** | 구성 고정: AG3S commit · argv hash, baseline commit | — | — |
| **5** | test 실행: 실험 1 → 실험 2 → 실험 3 | 사전 등록 그대로. 중간 결과로 arm · 표본을 바꾸지 않는다 | 독점 (AG3S), baseline 2 job |
| **6** | 실험 4 ablation · 실험 5 offline · 실험 6 시간 | — | 혼합 |
| **7** | 실험 7 · 8 (선택), 논문 표 · 그림 | — | — |

- 긴 GPU 작업 전후에는 같은 pod의 다른 세션에 알린다. 시작 전에 `nvidia-smi --query-compute-apps`를 확인한다.
- 단계 2의 dev episode와 단계 5의 test episode는 겹치지 않는다 (init 0–4 대 5–).

---

## 8. 계산 예산 (추정)

측정값: BL-7 pilot (SafeLIBERO spatial L I dev, 20 episode, 2 job 병렬)의 arm별 wall time.
VLA 단독 1,398 s · AEGIS 1,854 s · multi-link 1,915 s · OmniGuide 2,248 s · KNOWS 2,301 s · Lan-o3dp 2,690 s · NUDGE 2,933 s · CFM 4,675 s.
AG3S 최종 구성(Panda)의 episode당 시간은 아직 없다. SL6 G3 smoke에서 잰다.

| 묶음 | episode | 추정 job 시간 | 비고 |
|---|---|---|---|
| 실험 1 주 표 baseline 5 arm (VLA 단독 · AEGIS · KNOWS · multi-link · OmniGuide) | 5 × 640 | 20 episode당 합 9,716 s × 32 ≈ 86 h → 2 job 병렬 ≈ 43 h | spatial 기준이다. long suite는 episode가 약 2배 길다 (공식 ETS 420 대 200). 그래서 +25 % 정도를 더 잡는다 |
| 실험 1 보조 3 arm (spatial만) | 3 × 160 | 10,298 s × 8 ≈ 23 h → ≈ 11.5 h | |
| 실험 1 AG3S | 640 | episode당 t_AG3S × 640, 동시 1 job | t_AG3S = 140 s이면 약 25 h |
| **실험 F 새 cell 3개 (F2 · F5 · F6)** | 3 × 160 = 480 | t_AG3S = 140 s(가정)이면 약 19 h, 독점 lock | F5(GT + 일반 filter)는 baseline 구현 시간. 외부 segmenter 추론이 더해지므로 F2는 더 길다 |
| 실험 2 | 400 (+ baseline spatial 4 × 100) | VLA 단독 · AG3S 중심 | |
| 실험 3 · 4 (RB-Y1) | 대표 80 + stress 80 per arm, ablation 6행 × 40 | 기존 core evaluation run 시간으로 task 문서에서 계산한다 | |

총량이 크므로 §10 판정 2 · 3에서 범위를 줄일 수 있다. 줄이는 순서는 다음과 같다: 보조 arm → test episode 수 (20 → 10) → suite 수 (spatial · long 우선).

---

## 9. 논문 표 · 그림 대응 (제안)

| 위치 | 내용 | 실험 |
|---|---|---|
| Tab. 0 (또는 Fig.) | **인식 × 제약 factorial** (F1 · F2 · F5 · F6 · AG3S): SS · 접촉 분해 · 실패 원인 층 | F |
| Fig. | 외부 segmenter 오류 → 거리장 오류 → 접촉 (통제 오류 주입 곡선 + 자연 오류 분포) | P-1 |
| Tab. 1 | SafeLIBERO 주 비교: 방법 × (SS · TSR · CAR · ETS), 보호 link 열, privileged 열, re-run / reported 표시. level별 열 | 1 |
| Tab. 2 | LIBERO 비저하 + 개입률 | 2 |
| Tab. 3 | RB-Y1 대표 · stress: 금지 접촉 없는 성공, 접촉 분해 | 3 |
| Tab. 4 | 모듈 대체 ablation (행 = 바꾼 모듈 · 대체 방식) | 4 |
| Tab. 5 | grounding: ATS 대 segmenter · head (identity · precision · 다른 물체), noun 교체 결과 | 5-A · 5-B |
| Fig. | 같은 episode의 VLA 단독 실패 → AG3S 성공 영상 frame (SafeLIBERO L I · L II, RB-Y1 허들) | 1 · 3 |
| Fig. | 실패 원인 층 분포 (arm별) · 시간 분포 | 4.4 · 6 |
| 부록 | 개발 결과 (9 장면, 허들 7, BL-7 pilot), extent 분석, baseline deviation 표 | — |

---

## 10. 사용자 판정이 필요한 것

lead 추천을 먼저 적었다. 판정 뒤 각 실험 task 문서에 고정한다. **새로 생긴 항목은 10–12이고, 1은 개정 2판에서 선택지가 구체화됐다.**

| # | 판정할 것 | lead 추천 | 이유 |
|---|---|---|---|
| 1 | ~~C-A 문장 선택~~ **확정 (2026-10-30, 사용자):** 임시 contribution 유지, 최종 구성은 사용자가 수정 (§2.1). 4-G 필수 | — | — |
| 2 | SafeLIBERO test 규모 | **4 suite × 2 level, task당 20 (init 5–24) = arm당 640** | suite 합산 판정에 검정력이 충분하고, 전체 45개보다 계산량이 절반 이하다 |
| 3 | baseline 범위 (§6.1) | **1등급 4개 + 1b 구성 baseline 3개(F2 · F5 · F6), 보조 3은 spatial만** | 계산량. 가장 가까운 선행연구 KNOWS를 주 표에 둔다. (WBAG 제외는 사용자 판정 2026-10-30) |
| 4 | link별 허용 주장의 범위 | **ablation 4-C를 비권한 link가 있는 구성에서 돌린다** (RB-Y1 팔 구 또는 Panda `--links arms`). 못 하면 "권한 link · 반대 손"으로 주장을 좁힌다 | 최종 구성의 `--links gripper`에서는 비교가 성립하지 않는다 (§2.3) |
| 5 | RB-Y1 held-out 장면 | **개발 장면 ID를 뺀 새 40 장면**을 metadata로 고른다 | IFG-CL3의 40 장면은 이전 구성으로 결과를 이미 봤다 |
| 6 | LIBERO-Safety | **실험 1–3 뒤 선택 실험** (정적 L0 두 suite) | controller · checkpoint · 평가 script 차이로 adapter 작업이 크다 |
| 7 | 주 지표 · 판정 기준 | **SS 주 지표, 개선 +5 %p ∧ CI 하한 > 0, TSR 비열등 −5 %p** | 연구 범위 결정문(2026-10-23)과 core evaluation 판정 2(2026-10-25)를 그대로 잇는다 |
| 8 | RB-Y1에서 KNOWS · AEGIS를 통째로 옮길지 | **옮기지 않는다.** 모듈 대체(실험 4)로 비교 | EE 하나를 위한 CBF를 양팔 joint 공간으로 옮기면 그것 자체가 새 방법이 된다 |
| 9 | Panda dev gate 기준 (§7 단계 2) | SS > VLA ∧ TSR 하락 ≤ 10 %p, 수정 묶음 최대 2 번 | Panda에서 작업 성공을 잃은 기록이 있다 (§2.2). 진행 중인 SL6 smoke(v2 · v3 · TrajOpt E0)가 이 기준에 입력이 된다 |
| **10** | **실험 P · F를 추가하고 문제정의 실험을 contribution의 종료 조건으로 쓰는가** (§0) | **예.** P1 · P2가 성립하지 않으면 해당 contribution을 쓰지 않거나 좁힌다 | 문제가 실재함을 baseline만으로 보여야 한다. 이것이 없으면 reviewer가 "필요성"을 의심한다 |
| **11** | **외부 mask provider를 closed-loop에 넣는 기본 꺼짐 flag를 구현하는가** (F2 · F5 · F6의 전제, implementer 작업) | **예.** SL6 smoke가 끝난 뒤 task 문서로 시작한다 | 없으면 C-A의 직접 비교가 불가능하다. 외부 segmenter를 약하게 두지 않는다 |
| **12** | PDF의 "불확실 영역 보수 처리" | 임시 contribution에 없어 **제외로 두었다** (이의가 있으면 알려 주세요) | 구현이 없고 검증되지 않았다 |
| **13** | **양팔 · 휴머노이드 공개 benchmark를 한 개 추가하는가** (§5 실험 3b). 후보: RoboTwin 2.0 Hard 일부 task | **실험 1–3이 끝난 뒤 선택 실험으로 두고, 먼저 feasibility만 확인한다** (checkpoint · 로봇 모델 · 접촉 읽기). 안 되면 RB-Y1 자체 benchmark + 장면 생성 규칙 공개로 둔다 | 공개 양팔 safety benchmark는 조사 범위에서 찾지 못했다. 이식 비용이 크다 |
| **14** | **외부 평가자(SafeVLA-Bench 등)로 같은 rollout을 다시 판정해 "우리가 정의한 안전"이라는 비판을 줄이는가** (§4.6) | **feasibility 확인 뒤 가능하면 예** | 안전 정의를 제3자 기준으로 교차 검증할 수 있다. rollout 형식 호환은 미확인 |

---

## 11. 근거 문서

- 최종 구성: [ADOPTED_KO.md](../reviews/AG3S_decision/ADOPTED_KO.md) · [NOT_ADOPTED_KO.md](../reviews/AG3S_decision/NOT_ADOPTED_KO.md) · [OPEN_ISSUES_KO.md](../reviews/AG3S_decision/OPEN_ISSUES_KO.md) · [AG3S-CORE-EVALUATION.task.md](../../../AG3S/ag3s/docs/handoff/AG3S-CORE-EVALUATION.task.md) §0.1 · §0.2
- 연구 범위: [AG3S_RESEARCH_SCOPE_20261023_KO.md](AG3S_RESEARCH_SCOPE_20261023_KO.md) · [AG3S_SCOPE_DECISION_20261023_KO.md](../reviews/AG3S_SCOPE_DECISION_20261023_KO.md)
- Panda 이식: [SL6.task.md](../../../AG3S/ag3s/docs/handoff/SL6.task.md) · [AG3S_FINAL_MERGE_REPORT.md](../migration/AG3S_FINAL_MERGE_REPORT.md)
- benchmark 기록: [AG3S_BENCHMARK_LOG.md](../../../AG3S/ag3s/docs/AG3S_BENCHMARK_LOG.md) (SL5b · SL5c · SL5i) · [EVAL_RECIPE.md](../../../benchmark/docs/EVAL_RECIPE.md) · [suites.json](../../../benchmark/configs/suites.json)
- baseline 기록: [AG3S_BASELINE_LOG.md](../../../AG3S/ag3s/docs/AG3S_BASELINE_LOG.md) (BL-0 – BL-8) · [paper_baselines/README.md](../../../Baseline/paper_baselines/README.md) · [exemption_ablations.md](../../../Baseline/paper_baselines/docs/exemption_ablations.md)
- 논문: `main.pdf` (2026-10-23 초안) · [SAFELIBERO_PLAN.md](../../../paper/notes/SAFELIBERO_PLAN.md) · [EXPERIMENTS.md](../../../paper/evidence/EXPERIMENTS.md) · [BASELINES.md](../../../paper/evidence/BASELINES.md) · [REVIEWER_QUESTIONS.md](../../../paper/notes/REVIEWER_QUESTIONS.md)
- 선행 논문 지표 · 양팔 benchmark 조사 (2026-10-30 웹 검색): §4.6 · 실험 3b 표의 링크. 검색 요약 · 초록 수준이며 본문 인용 전 원문 확인이 필요하다.
- LIBERO-Safety: [BM-SETUP-B.impl.md](../../../AG3S/ag3s/docs/handoff/BM-SETUP-B.impl.md)
