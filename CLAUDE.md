# AG3S / trajopt 검토 — 작업 규칙

> 이 파일은 세션 시작 때 자동으로 읽힌다. 같은 규칙이 매 프롬프트마다
> `.claude/hooks/ag3s-rules.sh` (UserPromptSubmit hook) 로도 주입된다 — **이중이다.**
> 규칙 본문을 고칠 때는 **두 곳을 같이** 고친다 (`.claude/ag3s-rules.md` 와 이 파일).

## 읽는 순서

1. `benchmark/ag3s/docs/AG3S_T0T6_LOG.md` — **정본.** 16D 모델 위의 T0→T6 실행 기록.
   맨 위의 **용어 · 진행 현황 · 물려받은 결정 · 재측정 대기** 네 표를 먼저 읽으면 위치를 안다.
2. `benchmark/ag3s/docs/AG3S_CUROBO_LIVE_TEST_PLAN.md` — T0~T6 실행 계획. 게이트별 판정 항목.
3. `benchmark/ag3s/docs/AG3S_REVIEW_PLAN.md` — 무엇을 왜 하는가. "방향 결정" 절이 핵심.
4. `benchmark/ag3s/docs/AG3S_REVIEW_LOG.md` — **닫힌 국면의 archive.** 이어 쓰지 않는다.
   모듈 검토(Step 0~11) · 통합 · cuRobo 이관 · I1~I4. **발견 번호의 원 측정을 찾을 때 여기로 온다.**
5. `benchmark/ag3s/docs/VLA_ACTION_CONDITIONED_LOCAL_TSDF_ESDF_PLAN.md` — 미세 ESDF 의
   중심을 action chunk 의 swept volume 으로 잡는다는 결정.

**`CONTAINER_SETUP.md` 는 폐기됐고 (2026-09-12) 2026-09-21 에 삭제됐다. 찾지 않는다.**

## 매 턴 지킬 것

### 규칙 A — 시각화

구현이든 검증이든 **모든 스텝은 시각화 자료를 남긴다.** 가장 좋은 것은 셋을 **다** 만드는 것이다.

1. **실제 씬** — MuJoCo 장면, depth, ESDF 단면, 로봇 구를 실제 좌표에 겹쳐 그린 것.
   *무엇이 어디서 일어나는가.* 단면을 그릴 때는 **어디를 어느 방향에서 잘랐는지 배치도로 먼저 보인다.**
2. **그래프** — 프레임별 추이, 분포, 계층 간 차이. *얼마나, 어떤 추세로.*
3. **표·수식·도식** — 숫자를 나란히 놓은 표, 제약 식, 파이프라인 지도. *무엇과 무엇을 비교하나.*

**숫자를 문장 안에 흩어 놓는 것은 시각화가 아니다.** 그림은 `benchmark/ag3s/docs/figures/` 에
저장하고 로그에서 링크하며, **`SendUserFile` 로 사용자에게 보낸다.**

> *왜* — 이 검토에서 판정을 두 번 뒤집었다. C2("33 mm 대 2 mm")는 옛 측정법이 대역 폭을 잰
> 것이었고, D2("미세 계층은 값어치를 안 한다")는 미세 창이 로봇을 안 덮은 한 프레임만 본
> 것이었다. **둘 다 그림 한 장이면 즉시 보였을 오류다.**

주의: matplotlib 은 `**굵게**` 를 렌더하지 않는다 — 제목/라벨에 별표를 쓰지 말 것.
한글은 `figstyle.use_korean()` 을 부르기 전에 Noto CJK 를 `addfont` 해야 한다.

### 규칙 B — 전체 틀에서 지금 어디인가

스텝을 끝낼 때마다 (1) 무엇이 닫혔는지, (2) 다음 무엇으로 이어지는지 말한다. 비유·예시 환영.

```
카메라 depth → 로봇 마스크 → [attention lifting → target grounding] → TSDF/ESDF
            → 거리장 어댑터 → SQP 선형화 → QP 해
```

### 규칙 C — 용어를 설명한다

새 용어를 쓰면 **그 답변 안에서 풀어 쓰고**, `AG3S_REVIEW_LOG.md` 의 **"용어"** 절에 추가한다.
설명 없이 쓰지 말 것: `violated`/`feasible`, 거친 계층 · 미세 계층 · 2계층, 여유거리(clearance),
ESDF/TSDF, 복셀, 청크, 지평, 로봇 구, eikonal, swept volume, carving …

### 규칙 G — 발견 번호를 그냥 쓰지 않는다

`E3`, `F11`, `G2`, `C5`, `X1` 같은 발견 ID 를 말할 때는 **그 자리에서 무슨 내용인지
한 줄로 풀어 쓴다.** 번호만 던지면 읽는 쪽이 로그를 뒤져야 한다.
예: "E3(쥔 물체가 optimizer 에 도달하지 않는다)". 스텝 번호도 같다.

### 규칙 H — 로그에 무엇을 쓰는가

`AG3S_REVIEW_LOG.md` 는 **구현과 검증의 기록이다.** 발표 자료·슬라이드·대본처럼 산출물을
만든 과정은 쓰지 않는다. 그런 작업은 해당 문서 안에서 끝내고, 로그에는 **코드가 무엇이
바뀌었고 무엇을 수치로 확인했는지**만 남긴다.

### 규칙 D — 기록

**`AG3S_T0T6_LOG.md` 에 이어 쓴다.** 새 기록 파일을 만들지 않는다. 기록이 갈라지면 어느 쪽이
최신인지 알 수 없게 된다.

> **2026-09-24 에 한 번 나눴다** (사용자 판정). 옛 `AG3S_REVIEW_LOG.md` 는 모듈 검토 국면의
> archive 이고 **이어 쓰지 않는다** — 그 파일 머리에 그렇게 박혀 있다. 나누면서 이 규칙이
> 막으려던 실패("어느 쪽이 최신인가")를 피하기 위해 셋을 같이 했다: 옛 파일에 표지, 용어
> 표를 새 파일로 이동, 그리고 이 규칙과 위 "읽는 순서" 갱신. **다음에 또 나누지 않는다.**

## 병렬 세션 — 누가 무엇을 쓰나

창을 여러 개 띄울 때는 **파일 소유권으로 자른다.** 같은 파일을 둘이 쓰면 한쪽이 덮는다.

| 역할 | 세션 시작 | **쓴다** | 안 쓴다 |
|---|---|---|---|
| **A0 lead** | `claude` | `docs/handoff/*.task.md` · `AG3S_REVIEW_PLAN.md` | — |
| **A1 구현** | `claude --agent ag3s-implementer --settings .claude/roles/implementer.settings.json` | `benchmark/**/*.py` · `tests/**` · `*.impl.md` | docs · figures · papers |
| **A2 검증** | `claude --agent ag3s-verifier --settings .claude/roles/verifier.settings.json` | `docs/figures/**` · `*.verify.json` · `outputs/verify/**` | 코드(읽기만) · docs · papers |
| **A3 기록** | `claude --agent ag3s-scribe --settings .claude/roles/scribe.settings.json` | `docs/*.md` · `papers/**` | 코드 · figures · handoff 산출물 |

**`AG3S_T0T6_LOG.md` 의 writer 는 A3 하나다** (규칙 D). 나머지는 `docs/handoff/` 로 넘긴다.

**A1 과 A2 를 분리하는 이유**: 고친 사람이 스스로 재면 기대한 것이 보인다. C2(*"33 mm 대
2 mm"* — 옛 측정법이 대역폭을 잰 것)와 D2(*"미세 계층은 값어치를 안 한다"* — 미세 창이
로봇을 안 덮은 한 프레임만 본 것)가 그렇게 뒤집혔다.

**A3 의 철칙**: `<STEP>.verify.json` 의 `numbers` 에 없는 숫자는 쓰지 않는다. 필요하면
A2 에게 측정을 요청한다. 이것이 *"추측을 기록에 남기지 않는다"* 를 구조로 만든다.

세션끼리는 `ListAgents` → `SendMessage` 로 연락한다. 자세한 것은
`benchmark/ag3s/docs/handoff/README.md` 와 `.claude/roles/README.md`.

절차는 skill 로 빠져 있다 (필요할 때만 로드된다): `regression-baseline`(회귀 기준선),
`make-figure`(규칙 A 의 figure 3종), `curobo-esdf`(함정 5개 · eikonal 자가진단).
경로별 규칙은 `.claude/rules/` — 해당 파일을 만질 때만 들어온다.

## 진행 방식

각 스텝은 `설명 → 사용자 질문(여기서 멈춤) → 공동 판정 → 확정된 것만 수정+검증 → 기록`.

- **추측을 기록에 남기지 않는다.** 가능하면 그 자리에서 실행해 수치로 확정한다.
- **설명은 평이한 요약을 먼저, 코드 근거(`파일:줄`)는 그 다음.**
- 체크포인트/MuJoCo 실행이 필요하면 **먼저 사용자에게 요청**한다.
- 갈림길에서 고르지 않은 선택지는 로그의 **"되돌아올 지점"** 절에 전환 신호와 함께 남긴다.

## 환경

작업 루트 `/mnt/dev/work`. venv 두 개이고 **numpy 버전 때문에 합칠 수 없다.**

| venv | 무엇이 있나 | 무엇에 쓰나 |
|---|---|---|
| `.venv-ag3s` (py3.11, numpy 2.4.6) | mujoco · casadi · osqp · scipy | AG3S 파이프라인, trajopt, 그림 |
| `.venv-curobo` (py3.10, numpy 1.26.4) | torch · curobo · warp | TSDF/ESDF 생산만 |

`.venv-curobo` 에서 `benchmark` 임포트는 되지만 mujoco/casadi/osqp 가 없다. 그래서
**생산(curobo) → npz → 소비(ag3s)** 로 나눈다. 거리장 어댑터는 순수 numpy 라 양쪽에서 돈다.

명령은 항상 `MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work` 를 앞에 붙인다.

**MuJoCo 는 이 서버에서 돈다** — GPU 없이 OSMesa 소프트웨어 렌더링으로. 느릴 뿐 동작한다
(3 대 캡처 × 15 프레임이 약 5 분).

### 자산 경로 — 심볼릭 링크는 필요 없다 (2026-09-14 부터)

기록(`run_0004/meta.json`)의 `model_xml` 은 **녹화한 PC 의 절대경로**이고, 코드 기본값
(`TRANSPORT_MODEL`, `RBY1_URDF`)도 `src/rby1_description/...` 상대경로다. 이 컨테이너에서
자산의 실제 자리는 `pi05_TO_hybrid/rby1_description/` 이라 예전에는 링크 둘로 맞췄는데,
컨테이너 재시작에 링크가 날아가 재생이 전부 죽었다.

**지금은 `ag3s/runtime/asset_path.py` 가 찾는다** — 준 경로가 있으면 그것, 없으면 `AG3S_ASSET_ROOT`,
그다음 `rby1_description/` 부터의 꼬리를 `src` → `pi05_TO_hybrid` → `.` 에 붙여 본다.
MJCF(`TransportScene`)와 URDF(`parse_urdf`) 양쪽이 이걸 거친다.

자산을 다른 곳에 두려면 링크를 만들지 말고 `AG3S_ASSET_ROOT` 를 쓴다.

### 회귀 기준선 (이 숫자가 다르면 코드를 고치기 전에 원인부터 찾는다)

```bash
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \
  benchmark.trajopt.experiments.esdf_rollout --records run_16d_ep1800 \
  --attention benchmark/ag3s/asset/data/attention_16d_ep1800.npz \
  --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all \
  --support-surfaces field --constraint-links arms --out-json /tmp/base.json
# -> 위반으로 시작 15/15 · 해소 13 / 개선 15 / feasible 13, violated 2
```

**개수만 인용한다 — 프레임별 여유거리와 SQP 반복 수는 기준선이 아니다.** `sqp.time_budget_ms`
는 벽시계 **마감**이라(`trajopt/sqp.py:159`) 기계 부하에 따라 반복이 하나씩 달라진다. 4 회 실행에서
위 개수는 전부 같았고, 프레임 13·14 의 `clearance_after` 가 4.19 ↔ 4.97 mm 로 흔들렸다.

**14D 기준선(`run_0004` · 해소 14 / feasible 8, violated 7)은 은퇴했다 (2026-09-24).**
`wire.ACTION_WIDTH` 가 16 이 되어 14 열 기록으로는 아예 돌지 않는다 (*"chunk has 14 columns
but the layout needs at least 16"*) — 사용자 판정 "16D 로 전환, 14D 는 버린다" 의 결과다.
`run_0004` 와 `attention_step1_run0004.npz` 는 **다른 용도로는 그대로 쓴다** (14D 기록을 읽는
과거 실험 스크립트들).

**`--cameras all` 이 새로 들어갔다.** 옛 기준선은 head 한 대였는데, 에피소드 기반 씬
(`--episode-index`)에서는 head 만으로 **시작 위반이 0/15** 다 — TO 가 고칠 것이 없어 기준선
구실을 못 한다. 세 대는 서빙 경로와 같은 진입점이기도 하다.

**이 기준선은 attention → target grounding 을 재지 않는다.** 두 카메라 구성 모두
`has_target` 이 15/15 프레임에서 false 다 (live 경로는 같은 에피소드에서 성공한다 — 차이는
오프라인이 고정된 attention 셀 선택을 쓰는 것이고, 그 셀은 14D 모델에서 정해졌다). 재는
범위는 depth → 자기 필터 → 필드 → SQP → QP 다. 자세한 것은 로그의 **"회귀 기준선을 16D 로
갈았다"** 절.

### cuRobo 함정 (유래는 archive `AG3S_REVIEW_LOG.md`)

1. `esdf_origin` 은 격자 코너가 아니라 **중심**이다.
2. 인덱싱을 손으로 쓰지 말고 `create_xyzr_tensor(transform_to_origin=True)` 를 쓴다.
3. 첫 `compute_esdf` 호출 시간(약 1500 ms)은 JIT 다 — 인용하지 말 것.
4. **`compute_esdf()` 결과는 다음 호출 전에 복사한다** (`feature_tensor.detach().clone()`).
5. `VoxelGrid.get_occupied_voxels()` 는 반대 부호 규약 — 쓰지 말 것.

**1·3·4 는 자기 진단 하나로 잡힌다**: 자유공간에서 `|∇d| ≈ 1` 인가 (eikonal).
