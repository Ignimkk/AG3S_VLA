# AG3S / trajopt 검토 — 작업 규칙

> **규칙 본문의 정본은 `.claude/ag3s-rules.md` 다.** UserPromptSubmit hook 이 매 턴 주입하므로
> 규칙은 대화 내내 살아 있다. 규칙을 고칠 때는 **그 파일만** 고친다.
> 이 파일은 세션 시작에 한 번만 필요한 것 — 어디를 읽나, 규칙이 왜 그렇게 생겼나, 환경 — 을 담는다.

## 읽는 순서

1. `benchmark/ag3s/docs/AG3S_T0T6_LOG.md` — **정본.** 16D 모델 위의 T0→T6 실행 기록.
   맨 위의 **용어 · 진행 현황 · 물려받은 결정 · 재측정 대기** 네 표를 먼저 읽으면 위치를 안다.
2. `benchmark/ag3s/docs/AG3S_CUROBO_LIVE_TEST_PLAN.md` — T0~T6 실행 계획. 게이트별 판정 항목.
3. `benchmark/ag3s/docs/AG3S_REVIEW_PLAN.md` — 무엇을 왜 하는가. "방향 결정" 절이 핵심.
4. `benchmark/ag3s/docs/AG3S_REVIEW_LOG.md` — **닫힌 국면의 archive.** 이어 쓰지 않는다.
   모듈 검토(Step 0~11) · 통합 · cuRobo 이관 · I1~I4. **발견 번호의 원 측정을 찾을 때 여기로 온다.**
5. `benchmark/ag3s/docs/VLA_ACTION_CONDITIONED_LOCAL_TSDF_ESDF_PLAN.md` — 미세 ESDF 의
   중심을 action chunk 의 swept volume 으로 잡는다는 결정.

## 규칙이 왜 그렇게 생겼나

본문은 hook 이 준다. 여기 있는 것은 **이유**다 — 이유를 모르면 규칙이 형식으로 굳는다.

- **규칙 A(시각화)** — 이 검토에서 판정을 두 번 뒤집었다. C2(*"33 mm 대 2 mm"* — 옛 측정법이
  실제로는 대역폭을 잰 것)와 D2(*"미세 계층은 값어치를 안 한다"* — 미세 창이 로봇을 안 덮은
  한 프레임만 본 것). **둘 다 그림 한 장이면 즉시 보였을 오류다.**
- **규칙 D(단일 로그)** — 기록이 갈라지면 어느 쪽이 최신인지 알 수 없게 된다.
  2026-09-24 에 사용자 판정으로 **한 번** 나눴고(옛 파일에 표지 + 용어 표 이동 + 이 문서 갱신을
  같이 했다), **다시 나누지 않는다.**
- **규칙 B·H 는 지금 꺼져 있다** (hook 사본에서 주석 처리, 사용자 의도).
  H 의 취지 — 로그는 구현과 검증의 기록이고 산출물을 만든 과정은 쓰지 않는다 — 는
  그대로 따르되, 규칙으로 강제하지는 않는다. 다시 켜려면 `.claude/ag3s-rules.md` 의 주석을 푼다.
- **규칙 C·G(용어와 발견 번호)** — 읽는 쪽이 로그를 뒤지게 만들지 않기 위해서다.
  용어 표의 자리는 `AG3S_T0T6_LOG.md` 의 **"용어"** 절이다 (2026-09-24 에 archive 에서 옮겨 왔다).

### 파이프라인 지도 (규칙 B 는 꺼져 있다 — 지도는 참조용으로 남긴다)

```
카메라 depth → 로봇 마스크 → [attention lifting → target grounding] → TSDF/ESDF
            → 거리장 어댑터 → SQP 선형화 → QP 해
```

## 병렬 세션 — 누가 무엇을 쓰나

창을 여러 개 띄울 때는 **파일 소유권으로 자른다.** 같은 파일을 둘이 쓰면 한쪽이 덮는다.
`claude` 는 PATH 에 없다 — 아래 스크립트가 VSCode 확장 번들을 찾아 준다.

| 역할 | 띄우기 | **쓴다** | 안 쓴다 |
|---|---|---|---|
| **A0 lead** | `.claude/roles/launch.sh lead` | `docs/handoff/*.task.md` · PLAN | — |
| **A1 구현** | `.claude/roles/launch.sh implementer` | `benchmark/**/*.py` · `tests/**` · `*.impl.md` | docs · figures · papers |
| **A2 검증** | `.claude/roles/launch.sh verifier` | `docs/figures/**` · `*.verify.json` · `outputs/verify/**` | 코드(읽기만) · docs · papers |
| **A3 기록** | `.claude/roles/launch.sh scribe` | `docs/*.md` · `papers/**` | 코드 · figures · handoff 산출물 |

- **`AG3S_T0T6_LOG.md` 의 writer 는 A3 하나다.** 나머지는 `docs/handoff/` 로 넘긴다.
- **A1 과 A2 를 나누는 이유**: 고친 사람이 스스로 재면 기대한 것이 보인다. C2·D2 가 그렇게 뒤집혔다.
- **A3 의 철칙**: `<STEP>.verify.json` 의 `numbers` 에 없는 숫자는 쓰지 않는다. 필요하면
  A2 에게 측정을 요청한다. 이것이 *"추측을 기록에 남기지 않는다"* 를 구조로 만든다.
- 세션끼리는 `ListAgents` → `SendMessage`. 계약은 `benchmark/ag3s/docs/handoff/README.md`.

### 모델을 일의 모양에 맞춘다 (2026-09-25 실측으로 확정)

| 역할 | 모델 | 왜 |
|---|---|---|
| A1 구현 | `opus` | 왜 깨지는지 코드에서 되짚는 일 (X1, `attached_self_mask` 가 그랬다) |
| A2 검증 | `sonnet` | 명령을 돌리고 결과를 json 으로 내는 절차. 판단이 적다 |
| A3 기록 | `haiku` | 주어진 수치를 정해진 형식으로 옮긴다. 2 회 연속 숫자 오염 **0** 이었고 4 배 싸다 |

**A3 가 haiku 로 되는 것은 철칙이 구조를 단순하게 만들어 둔 덕이다** — *"verify.json 의 numbers 에
없는 숫자는 안 쓴다"* 가 판단할 여지를 없앴다. **논문 초고처럼 문장을 스스로 구성해야 하면
sonnet 으로 올린다.**

### sub agent 를 쓸 때 (호출당 100k 를 넘으면 일이 너무 큰 것이다)

- **긴 실행을 agent 안에 넣지 않는다.** lead 가 `nohup ... python -u` 로 띄우고 agent 는 그
  산출 파일만 소비한다. agent 는 세션과 함께 죽지만 프로세스는 산다 — 2026-09-25 에 A1·A2 가
  세션 종료로 중단되며 작업 700k 를 잃었다.
- **관찰할 수 없는 실행은 돌지 않는 실행과 구별이 안 된다.** 긴 실행은 반드시 `python -u`.
  버퍼링된 채로 띄웠다가 7 분간 진행 여부를 몰라 멀쩡한 실행을 죽인 적이 있다.
- **`pkill -f` 는 자기 셸을 죽인다.** 패턴이 명령줄에 그대로 들어가기 때문이다. `[p]i05_infer`
  처럼 대괄호로 끊는다.
- **한 agent = 한 산출 파일.** 이식 8 건을 한 agent 에 몰아 419k 를 썼다. 셋으로 쪼갰으면
  중단돼도 한 조각만 잃는다.
- **브리핑에 숫자를 박고 "이 밖은 쓰지 마라"를 명시한다.** 열어 두면 모델이 클수록 그럴듯하게 채운다.
- **값을 하는 것은 속도가 아니라 불일치다.** 병렬이 잡아낸 셋은 전부 두 agent 가 다른 답을 냈을
  때 나왔다. 한쪽에 몰아 시켰으면 그대로 기록에 들어갔을 것이다.

### 세션 경계 — 한 세션 = 한 STEP

`task.md → impl.md / verify.json → 로그 기록 → 커밋` 한 바퀴가 끝나면 끊는다. **그때 모든 것이
파일에 있어서** 새 세션이 셋만 읽으면 이어받는다: 로그의 **진행 현황** 표 · `handoff/` 의 최신
`task.md` · `git log --oneline -3`.

- **`.claude/agents/` 를 고쳤으면 반드시 새 세션으로 간다.** 커스텀 역할은 **세션 시작 때만
  등록**된다. 고친 그 세션에서는 `Agent type not found` 가 난다.
- **세션 후반에 새 agent 를 띄우지 않는다.** 중단 위험이 크다.
- **큰 문서를 통째로 읽지 않는다.** 로그는 1200 줄이 넘는다 — 필요한 줄만 `sed`/`grep` 으로 집는다.
  A3 에게 문서 소유권을 준 이유이기도 하다.
- 끝내기 전: 돌던 agent 를 거두고 · 로그와 계획서를 갱신하고 · 커밋하고 · 긴 실행은 `nohup` 으로
  띄워 두고 간다.

## 진행 방식

각 스텝은 `설명 → 사용자 질문(여기서 멈춤) → 공동 판정 → 확정된 것만 수정+검증 → 기록`.

- **추측을 기록에 남기지 않는다.** 가능하면 그 자리에서 실행해 수치로 확정한다.
- **설명은 평이한 요약을 먼저, 코드 근거(`파일:줄`)는 그 다음.**
- 체크포인트/MuJoCo 실행이 필요하면 **먼저 사용자에게 요청**한다.
- 갈림길에서 고르지 않은 선택지는 로그의 **"되돌아올 지점"** 절에 전환 신호와 함께 남긴다.
- **커밋하기 전에 `git-commit` skill 을 읽는다.** 출처 표기 금지·메시지 형식·어느 repo 로 가나·병렬 세션에서 무엇만 스테이지하나가 거기 있다. push 는 사용자가 한다.

## 환경

작업 루트 `/mnt/dev/work`. 명령은 항상 `MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work` 를 앞에 붙인다.

| venv | 무엇이 있나 | 무엇에 쓰나 |
|---|---|---|
| `.venv-ag3s` (py3.11, numpy 2.4.6) | mujoco · casadi · osqp · scipy | AG3S 파이프라인, trajopt, 그림 |
| `.venv-curobo` (py3.10, numpy 1.26.4) | torch · curobo · warp | TSDF/ESDF 생산만 |
| **`.venv-openpi-live`** (py3.11, numpy 1.26.4) | **mujoco 3.11 · casadi · osqp · torch 2.7.1 · curobo 0.8.0 · warp** | **둘이 같이 필요한 것 전부** |

**앞의 둘은 numpy 버전 때문에 합칠 수 없어** 그때는 **생산(curobo) → npz → 소비(ag3s)** 로 나눴다.
거리장 어댑터는 순수 numpy 라 양쪽에서 돈다.

**세 번째가 그 분리를 푼다** (2026-09-25 실측 확인). openpi venv 복제본에 `warp-lang`·`curobo` 를
더한 것이고, **mujoco 와 curobo 가 한 프로세스에서 같이 돈다.** 정책·MuJoCo·AG3S·cuRobo·SQP 를
한 번에 돌려야 하는 것(I1 이 연 경로, live 검증 스크립트 대부분)은 **여기서 돌린다.**
`.venv-ag3s` 에서 `import torch` 하면 즉사하고, `.venv-curobo` 에는 mujoco 가 없다 —
**`broken` 의 상당수가 실은 venv 선택이다.**

**MuJoCo 는 이 서버에서 돈다** — OSMesa 소프트웨어 렌더링. 느릴 뿐 동작한다
(3 대 캡처 × 15 프레임이 약 5 분). GPU 는 H200 한 장이고 `.venv-curobo` 작업이 쓴다.

**자산 경로 — 심볼릭 링크를 만들지 않는다.** 기록의 `model_xml` 은 녹화한 PC 의 절대경로이고
코드 기본값도 `src/rby1_description/...` 상대경로인데, 이 컨테이너의 실제 자리는
`pi05_TO_hybrid/rby1_description/` 이다. 예전에는 링크로 맞췄다가 컨테이너 재시작에 날아가
재생이 전부 죽었다. 지금은 `ag3s/runtime/asset_path.py` 가 찾는다 — 준 경로 → `AG3S_ASSET_ROOT`
→ `rby1_description/` 꼬리를 `src` → `pi05_TO_hybrid` → `.` 에 붙여 보기. 자산을 옮기려면
링크 대신 `AG3S_ASSET_ROOT` 를 쓴다.

### GPU 작업 — 재현할 때는 인자가 아니라 **기동 스크립트**부터 찾는다

`meta.json` 의 `argv` 에는 **클라이언트 인자만** 있다. 서버를 어떻게 띄웠는지도, 어느 config 를
썼는지도 없다. 그것을 추측으로 재구성하면 그 순간 조건이 갈라진다 (2026-09-25 에 그렇게 40 분을
태웠다). **`pi05_TO_hybrid/logs/serve_rby1_16d.sh` 가 정답이고, 그 환경변수를 그대로 쓴다:**

```bash
export XLA_FLAGS="--xla_gpu_enable_command_buffer="   # 이게 없으면 CUDA graph 캡처가 실패한다
export XLA_PYTHON_CLIENT_PREALLOCATE=false            # 없으면 JAX 가 GPU 105 GB 를 선점한다
export OPENPI_DATA_HOME=/mnt/dev/work/pi05_TO_hybrid/cache/openpi
PORT=8123 bash pi05_TO_hybrid/logs/serve_rby1_16d.sh  # bare policy (safety layer 없음)
```

- **attention 을 뽑을 때 `--config` 를 반드시 준다.** 기본값이 14D 용 `pi05_rby1_lora` 라
  norm stats 를 못 찾는다. 16D 는 `pi05_rby1_randomized_pick_place_16d_lora`.
- **포트는 소켓 연결로만 확인한다.** `ss` 는 이 컨테이너에서 리스너를 아예 못 본다 (8000 조차).
  서버 로그의 `listening` 은 그 순간의 사실일 뿐이다.

### 긴 실행을 관찰하는 법 (ptrace 가 막혀 있다)

`ptrace_scope=1` 이고 `CAP_SYS_PTRACE` 가 없어 **py-spy 가 못 붙는다.** 밖에서 볼 수 없으면
안에서 말하게 한다 — 래퍼에서 `faulthandler.dump_traceback_later(45, repeat=True)` 를 걸면
45 초마다 스스로 스택을 뱉는다. 40 분 헤매던 것이 이것 한 줄로 45 초에 끝났다.

- 긴 실행은 반드시 **`python -u`**. 버퍼링된 출력은 "멈춤" 과 구별되지 않는다.
- **`pkill -f` 를 쓰지 않는다.** 패턴이 내 명령줄에 들어 있어 자기 셸을 죽인다 (하루에 세 번 했다).
  PID 를 뽑아 `kill <PID>` 로만 죽인다.
- **"되던 게 안 된다" 면 플래그를 더하기 전에 마지막으로 성공한 조건과의 차이부터 센다.**

**회귀 기준선**: `위반으로 시작 15/15 · 해소 13 · 개선 15 · feasible 13 · violated 2` · **`has_target` 9/15**.
**이 숫자가 다르면 코드를 고치기 전에 원인부터 찾는다.** 명령과 판정 요령(개수만 인용하는
이유, `--cameras all` 이 필요한 이유, 이 기준선이 재지 **않는** 것)은 `regression-baseline` skill 에.

## 어디에 무엇이 있나

| | 언제 들어오나 | 무엇 |
|---|---|---|
| `.claude/ag3s-rules.md` | **매 턴** (hook) | 규칙 본문 — 정본 |
| `CLAUDE.md` (이 파일) | 세션 시작 | 읽는 순서 · 규칙의 이유 · 소유권 · 환경 |
| `.claude/rules/*.md` | 해당 파일을 만질 때 | `logging`(docs/*.md) · `figures`(figures·experiments) · `paper`(papers) |
| `.claude/skills/*/SKILL.md` | 그 작업을 할 때 | `regression-baseline` · `make-figure` · `curobo-esdf`(함정 5개 · eikonal 자가진단) · **`git-commit`** |
| `.claude/roles/` | 세션 띄울 때 | 역할별 settings(소유권 차단) · `launch.sh` |
