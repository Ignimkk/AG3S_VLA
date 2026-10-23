# SL5j-recorder — Panda server에 제약 기록기 연결 (adapter, 기록 경로만)

> writer: lead · 읽는 쪽: implementer · 작성 2026-10-23 KST
> 앞 단계: `SL5j-launcher.impl.md`. launcher에서 flag를 넘길 수는 있게 됐지만, Panda server가 기록기를 만들지 않는다.

## 왜

- `serve_safe.main`은 `--robot panda`이면 `AG3S.adapters.panda.server.main`으로 바로 넘긴다(`serve_safe.py:2244`).
- 그런데 `ConstraintRecordWriter`를 만드는 code(`serve_safe.py:2287` 부근)는 RB-Y1 분기에만 있다.
- `AG3S/adapters/panda/server.py:305`의 `build_safe_server`는 `SafePolicy(..., recorder=None)`을 고정으로 넘긴다. `profile.py:52`, `:523`에도 "recording flags are parsed but not used"라고 적혀 있다.

## 무엇을 바꾸나

1. **기록기 생성:** `build_safe_server`(또는 Panda server main)에서 `args.record_constraints`가 있으면 `ConstraintRecordWriter`를 만들어 `SafePolicy(recorder=...)`에 넘긴다.
   - 생성 방식은 RB-Y1 분기(`serve_safe.py` 2287–2360행)와 같게 한다: `background=True`, 같은 `esdf_mode`, 같은 meta 항목. meta에는 robot=panda, version flags, conversion을 추가한다.
   - backend 검사도 RB-Y1과 같게 한다. cuRobo + occupancy 경고 / 거부, `SUPPORTED_ESDF_BACKENDS`.
2. **종료 시 `close()`:** server 종료 시 반드시 `close()`를 부른다. 정상 종료, SIGINT, SIGTERM 모두 해당한다. launcher는 `stop_own_process`로 SIGINT를 보낸다. `close()`가 없으면 `meta.json`이 남지 않고 큐에 남은 청크도 잃는다. RB-Y1 쪽에 같은 누락이 있으면 기록만 해 두고 고치지 않는다(범위 밖).
3. **기록 경로만 추가한다.** flag가 없으면 `recorder=None`, 지금과 같다. 정책 · 제약 · 응답 계산은 바꾸지 않는다. `SafePolicy`가 recorder에 넘기는 값은 이미 정해져 있다.
4. `profile.py`의 "parsed but not used" 주석을 고친다.
5. **test (GPU 없이):**
   - flag가 없으면 recorder가 None이고 server build 인자가 그대로인지
   - flag가 있으면 writer가 run_dir 아래에 만들어지고, SafePolicy에 전달되고, 종료 hook에서 close가 불리는지(mock)
   - SL5i의 RB-Y1 golden · 회귀 기준선이 그대로인지
6. 가능하면 **CPU에서 Panda SafePolicy 한 call을 기록기와 함께 돌려** 청크 npz에 `sphere_centres` · `sphere_link_names` · `clearance` · ESDF 계층이 실제로 들어가는지 확인한다. 기존 Panda offline replay fixture를 쓸 수 있으면 쓴다. 안 되면 그 이유를 적고 GPU smoke로 넘긴다.

## 넘겨줄 것

- `AG3S/ag3s/docs/handoff/SL5j-recorder.impl.md`
- commit하지 않는다.

## 규칙

GPU 금지. CPU worker ≤ 2. 진행 중인 run은 없다. launcher 변경(SL5j-launcher, 아직 commit 전)은 그대로 둔다.
