# SL3 — V11AB attention read-out audit on pi05_libero (+ SL2 후속 2건)

> writer: lead (A0) · 읽는 쪽: implementer, verifier · 작성 2026-10-12 KST
> branch `feature/panda-benchmark`, 시작 HEAD `9b51a3c4`. 계획: `doc/develope_PLAN_SUMMARY.md` §2.1 SL-3, §3.1(dev/test 분리).
> 사용자 판정 2026-10-12: SL2 확정. Panda self-filter 기본값 = confined hand+mount cover. constraint model = cuRobo `franka.yml` 그대로.

## 왜 지금 이것인가

V11AB의 ATS(Attention-based Target Segmentation — VLA attention + depth로 target을 고르는 단계)는 VLA attention map을
depth pixel에 대응시켜 target geometry를 얻는다. V11AB는 그 attention을 **고정된 한 cell**에서 읽는다:
`AG3S/trajopt/attention_policy.py:38-41` — denoise step 0, aggregation `last`, layer 8, head 2. 이 cell은 RB-Y1 checkpoint에서
`step-01-attention.json`의 점수로 고른 것이다. pi05_libero는 다른 fine-tuned model이므로, **같은 cell이 Panda/LIBERO에서도 target에
attention을 주는지**를 먼저 측정해야 SL-4(ATS → ESDF 연결)로 넘어갈 수 있다. SL1에서 policy image가 OpenCV image의
**좌우 반전**임을 측정했으므로 attention grid → depth pixel 대응에서 반전을 되돌려야 한다.

```
policy image 224×224 (= OpenCV 256 image 좌우 반전 + resize)
  → pi05 prefix: base_0_rgb 256 tok | left_wrist_0_rgb 256 tok | right_wrist_0_rgb 256 tok(LIBERO는 padding, mask False)
  → AttentionSampler (denoise 0, agg last, L8, H2) → [768] → 16×16 × camera
  → [SL3] un-mirror + 224→256 scale → OpenCV pixel (head / left_wrist) → SL-4 attention lifting
```

## 확인된 사실 (audit)

| 항목 | 근거 |
|---|---|
| 고정 cell (step 0, last, L8, H2), 두 번째 model 사본(`return_attn_probs=True`) | `AG3S/trajopt/attention_policy.py:38-50` |
| token slicing·camera binding (`cam_high` 0–256, `cam_left_wrist` 256–512, `cam_right_wrist` 512–768) | `AG3S/ag3s/experiments/sources/policy_record.py:43-50` |
| 16×16 grid → depth (H,W) bilinear, pixel-centre, padding/flip 없음 가정 | `AG3S/ag3s/stages/attention_lifting.py:39-75`, `policy_record.py:603` |
| server alias (`cam_high`→`zed_left` 등) | `AG3S/trajopt/serve_safe.py:1037-1058` |
| subtask probe `kv_L4_v1`(pick/place/home)는 RB-Y1 feature로 학습됨 | `attention_policy.py:16-21`, `AG3S/ag3s/asset/subtask_probe/` → **Panda 적용은 SL-6 audit 대상** |
| policy image = OpenCV 좌우 반전 (228/228) | `SL1.verify.json` |

## Dev/test 분리 (lead 제안 — 사용자 확인 대기, 확인 전까지는 이 STEP 안에서만 사용)

이 STEP의 측정은 **dev**다. LIBERO Spatial 각 task의 init state index **0–4를 dev 예약**으로 두고, P2 이후 test episode 목록에서 제외한다.
이 STEP은 그중 index 0만 쓴다. Layer/head **선택은 이 STEP에서 하지 않는다**(V11AB 고정 cell을 그대로 측정). 전체 layer×head 표는 진단용이며,
만약 V11AB cell이 Panda에서 target을 못 잡으면 calibration은 dev-only **별도 variant**로 선언하고 사용자 판정을 받는다(KNOWS L12·H3 자동 채택 금지).

## 구현에 요청하는 것 (→ implementer)

### 0. SL2 후속 (사용자 확정)
1. confined hand+mount cover를 Panda self-filter **기본값**으로: `robots/panda/description/`에 생성 JSON(provenance·digest 포함)을 두고 builder 기본값을 바꾼다.
   기존 default JSON은 이름을 바꿔 비교용으로 보존(덮어쓰지 않음). manifest의 robot_model 항목에 cover 이름·sphere 수 기록. test 갱신.
2. `benchmark/adapters/libero_panda/robot_geometry.py` 실행 시 pod `LD_LIBRARY_PATH`의 외부 torch lib 때문에 import 실패 → launcher와 같은 sanitize를
   CLI 진입점(또는 얇은 shell wrapper)에 적용.

### A. Panda attention binding (`AG3S/adapters/panda/attention.py`, 새 파일; AG3S core 수정 금지)
3. pi05_libero token block ↔ camera 표: `base_0_rgb → head`, `left_wrist_0_rgb → left_wrist`, `right_wrist_0_rgb → (padding, 사용 안 함)`.
   실제 block 순서와 image mask를 pi05_libero의 model input에서 **읽어서** 확인하는 함수(가정으로 두지 않음).
4. 16×16 attention grid → OpenCV pixel 대응: policy image 좌우 반전 해제 + 224→256 scale(+ resize_with_pad가 square라 padding 없음을 확인).
   기존 `patch_to_uv`/`attention_lifting`이 쓰는 규약(pixel-centre)과 같은 결과를 내도록 하고, 반전을 **하지 않은** mapping도 비교용으로 제공.
5. V11AB의 `AttentionSampler`·`AGGREGATIONS`를 그대로 사용해 recorded policy input에서 V11AB cell attention을 계산하는 함수. 진단용으로 같은 pass의
   전체 [layer, head] 배열(denoise 0, agg `last`)도 저장 가능하게(float16).

### B. Dev 데이터 수집과 offline 계산
6. Dev rollout: 기존 `launch_libero_pi05_only.py`로 libero_spatial **task 0–9 × init state 0**, `--capture-geometry --capture-segmentation`(policy_records 포함).
   (실행은 verifier. implementer는 dry-run까지.) 각 task의 target object를 LIBERO BDDL `obj_of_interest`(또는 동등 정보)에서 읽어
   segmentation instance와 연결하는 함수를 제공.
7. Offline audit runner `benchmark/runners/attention_audit.py`(`.venv-openpi-live`, GPU): source run(s)의 policy_records + geom npz를 읽어
   per-call attention(V11AB cell + 진단 전체 grid)을 **새 run**(`create_run`, source run id·manifest SHA 기록)에 저장. source run은 읽기만.
8. Test (CPU): mapping 함수(반전·scale·pixel-centre) 단위 test, camera 표, BDDL target 파싱, LD_LIBRARY_PATH sanitize.

**손대지 않을 것:** `AG3S/ag3s/**`·`AG3S/trajopt/**` 기존 파일(V11AB cell 값 포함), OpenPI model 코드, submodule, storage input.
Core에 주입 지점이 필요하면 구현하지 말고 impl.md에 적는다.

## 측정에 요청하는 것 (→ verifier)

| # | 무엇을 재나 | 어떤 숫자로 | 통과 기준 |
|---|---|---|---|
| 1 | SL2 후속 | confined cover 기본값의 robot 제거율·target/stove/table 오제거가 SL2 verify의 confined hand+mount 값과 일치; robot_geometry CLI가 기본 shell에서 동작 | 일치 / 동작 |
| 2 | Dev rollout | task 0–9 × init 0: official_success, status, step, policy 호출 수 | 기록만 (성공률 주장 아님) |
| 3 | token layout | pi05_libero 실제 image token block 순서·길이, right_wrist mask, padding block의 attention mass | 표 |
| 4 | 방향 대응 | V11AB cell attention의 target mask 내 mass 비율을 **un-mirror mapping vs mirror 미해제 mapping**으로 비교 | 수치 + 어느 쪽이 맞는지의 근거 |
| 5 | V11AB cell의 target grounding (head, left_wrist) | call별: target mask mass 비율, target area 비율(uniform 기준), argmax patch가 target에 있는 비율, target vs 다른 물체 mass 순위; task별·phase(접근/grasp/이동)별 | 숫자만. 판정은 lead·사용자 |
| 6 | 진단 grid (dev only) | 전체 layer×head에서 #5 지표의 분포와 V11AB cell의 순위 | 숫자만 — 선택하지 않음 |
| 7 | 비용 | attention pass latency(첫 JIT 별도) p50/p95, GPU memory | 측정만 (E10 자료) |
| 8 | 회귀 | `scripts/testing/run_cpu_regression.sh`, legacy 16D baseline, SL0 경로 보존 | 기존과 동일 |

- figure 3종(`sl3-*`): (1) 실제 scene — task별 대표 frame의 head/wrist OpenCV RGB 위 V11AB-cell attention overlay + target contour(반전 해제 전/후 대비 1장 포함),
  (2) graph — call별 target mass 추이(phase 표시), task별 분포, 진단 grid heatmap(V11AB cell 표시), (3) table — token layout, mapping 규약, 지표 요약.
- raw: `outputs/verify/SL3/`. GPU는 server 1개, pod budget 확인(다른 namespace process 존재).

## 이 STEP 이 답하는 질문

V11AB가 RB-Y1에서 고른 attention cell(step 0, last, L8, H2)을 pi05_libero에 그대로 적용하고 좌우 반전을 되돌렸을 때,
그 attention이 LIBERO Spatial task의 target object에 모이는가? (모이지 않으면 어떤 근거로 calibration variant가 필요한가)

## 되돌아올 지점

- #4에서 반전 해제가 더 나쁘면 → SL1 orientation 측정과 policy input 경로(transform)를 다시 대조. 원인 확정 전 SL-4 금지.
- #5에서 V11AB cell이 target을 잡지 못하면 → 사용자 판정: (a) V11AB cell 유지(그대로 Panda main, 결과로 보고), (b) dev-only calibration variant(별도 이름,
  init 0–4 dev에서만 선택, test 분리). main에 섞지 않는다.
- Subtask probe가 Panda에 쓸 수 없으면 → SL-6에서 strong-rule/GT phase 등 계획의 대안과 함께 blocker로 기록.
