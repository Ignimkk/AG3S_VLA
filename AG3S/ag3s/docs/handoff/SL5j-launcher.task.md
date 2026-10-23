# SL5j-launcher — benchmark launcher에 제약 기록 option 추가 (orchestration만)

> writer: lead · 읽는 쪽: implementer · 작성 2026-10-23 KST
> 사용자 판정 2026-10-23: "A안으로 갑시다" — 쥔 뒤에도 손을 위로 미는 제약이 어느 물체에서 나오는지 진단한다.

## 왜

SL5i-G에서 파지 인식 + closing 면제판은 bowl을 쥔 상태(held)로 들고 있는 동안에도 TO 행을 위로 1317, 아래로 307 밀었다. calls log에는 행의 **개수**만 남아서 어느 obstacle이 미는지 알 수 없다.

`serve_safe.py`에는 이미 `--record-constraints DIR`(+ `--record-constraints-esdf none|occupancy|full`)가 있다. 이 option은 call마다 sphere 중심 · 반지름 · link 이름, 행별 clearance, ESDF, 점구름을 npz로 남긴다(`AG3S/ag3s/experiments/sources/constraint_record.py`). 그런데 benchmark launcher(`benchmark/runners/launch_libero_paired.py`)에는 server argv에 이 option을 넣는 통로가 없다.

## 무엇을 바꾸나

1. **`launch_libero_paired.py`에 `--record-constraints {none,occupancy,full}` 인자를 추가한다.** 기본값은 none이다.
   - none이 아니면 `vla_v11ab` mode의 server argv 끝에 다음을 붙인다:
     - `--record-constraints <run_dir>/server_constraints`
     - `--record-constraints-esdf <값>`
   - `vla_only` mode에서 이 인자를 주면 거부한다(`--no-safe`와 함께 쓸 수 없다).
   - run manifest의 settings와 `server_argv`에 그대로 기록된다. 이것은 기존 경로이므로 따로 할 일이 없다.
   - 기본값(none)일 때 server argv는 지금과 byte 단위로 같아야 한다.
2. `run_parallel_gpu.sh` jobs 파일에서 이 인자를 쓸 수 있으면 된다. 지금도 인자를 그대로 넘기므로 따로 바꿀 것은 없을 것이다. 확인만 한다.
3. **단위 test**(GPU 없이, dry-run 또는 mock):
   - 기본값일 때 argv가 바뀌지 않는다.
   - none이 아닐 때 두 flag가 run_dir 아래 경로로 붙는다.
   - `vla_only`와 함께 주면 거부한다.
4. **바꾸지 않는 것:** `serve_safe.py`, `SafePolicy`, client, adapter, profile.

## 넘겨줄 것

- `AG3S/ag3s/docs/handoff/SL5j-launcher.impl.md`: 바뀐 file, 사용법, test 결과, 기본값일 때 그대로라는 근거를 적는다.
- **기록 용량 추정:** 기존 run이 있으면 그 기록으로, 없으면 CPU에서 `ConstraintRecordWriter` dummy 기록으로 청크당 크기를 잰다. 이 수치로 `full`과 `occupancy` 중 어느 것이 맞는지 정한다(SafeLIBERO LI 20 episode ≈ 1,000 call).
- commit하지 않는다.

## 규칙

GPU 금지. CPU worker ≤ 1. 진행 중인 run은 없다.
