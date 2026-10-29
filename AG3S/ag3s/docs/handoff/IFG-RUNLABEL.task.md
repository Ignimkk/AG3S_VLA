# IFG-RUNLABEL — run 폴더 이름에 선택적 label (implementer)

작성: 2026-10-18 KST, lead. 근거: `doc/active/reviews/vla_dump_20261015/FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md` §1 (사용자 지시: storage 폴더 이름만 보고 무슨 실험인지 알 수 있어야 한다).
**상태: 초안. 사용자 확인 뒤 띄운다 (IFG-CL3 결과 보고 다음 순서).**

## 목적

`common/run_output.py` 가 만드는 run 폴더 `<KST>_<commit>_<uuid12>` 뒤에 **선택적 label** 을 붙인다.
label 을 주지 않으면 지금과 **바이트 단위로 같다** (하위 호환). 다른 세션(benchmark checkout)도 쓰는 공용 모듈이다.

## 요구 사항

1. **이름 형식:** `<KST %Y%m%dT%H%M%S%z>_<commit 12>_<uuid 12>_<label>`. 예 `20261019T093000+0900_f15e587290f6_1a2b3c4d5e6f_ifg-chain-search`.
2. **label 규칙:** 영문 소문자 · 숫자 · `-` 만, 1–40 자, 첫 글자는 소문자 또는 숫자, 끝 글자는 `-` 가 아니다 (`^[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$`). 어긋나면 `ValueError` 로 **쓰기 전에** 거부한다 (디렉터리 · manifest 생성 없음).
3. **입력 경로 (우선순위):** `allocate_run_dir(..., label=None)` keyword 인자 → 환경 변수 `AG3S_RUN_LABEL` → 없음. CLI: `python -m common.run_output --label <label>` (`add_run_dir_argument` 에 `--run-label` 를 추가하고 `prepare_run_outputs` 가 넘긴다; 기존 호출자의 argparse namespace 에 `run_label` 이 없으면 `getattr(args, "run_label", None)` 로 무시).
4. **상속된 run (`AG3S_RUN_DIR` 또는 `run_dir` 인자가 있을 때):** label 은 폴더 이름을 바꾸지 **않는다**. 주어진 label 이 상속 run 의 manifest `label` 과 다르면 `logging.warning` 한 줄만 남기고 상속 run 을 그대로 쓴다.
5. **manifest:** `run_manifest.json` 에 `"label": <label 또는 null>` 을 추가한다. 기존 키는 그대로.
6. **검사 유지:** `_contained` · `_check_existing_links` · "direct child of results/development" 검사는 바뀌지 않는다. label 이 들어가도 `resolve_output_path` 의 `root.parent != results` 검사가 그대로 통과함을 test 로 보인다.
7. **run 이름을 parsing 하는 다른 코드가 있는지 grep** 한다 (`scripts/setup/reclaim_cache.py`, `scripts/setup/*.py`, `common/*.py`, `tests/`, `AG3S/ag3s/experiments/studies/*.py` 의 `\d{8}T\d{6}` · `split("_")` 류). 세 토큰을 가정하는 곳이 있으면 label 토큰을 허용하도록 고치고 impl.md 에 목록을 적는다. 없으면 "없음" 이라고 적는다.

## test (`tests/test_run_output.py` 에 추가; CPU, `.venv-ag3s`)

- 기존 `test_fresh_runs_are_unique_and_manifest_records_own_checkout` 의 정규식은 label 없는 경우 그대로 통과해야 한다 (바꾸지 않는다).
- 새 test: (a) `label="ifg-chain-search"` → 이름이 `..._ifg-chain-search` 로 끝나고 manifest `label` 이 같다; (b) `AG3S_RUN_LABEL` 환경 변수만 준 경우 (a) 와 같다; keyword 인자가 환경 변수보다 우선한다; (c) 잘못된 label (`"IFG"`, `"a_b"`, `"-x"`, `"x-"`, 41 자, `""`) 은 `ValueError` 이고 results/development 아래에 새 디렉터리가 생기지 않는다; (d) 상속 run 에 다른 label 을 주면 폴더 · manifest 가 그대로이고 warning 이 기록된다 (`caplog`); (e) label 이 있는 run 에서 `resolve_output_path("videos/x.mp4", ...)` 가 run 안으로 풀린다; (f) `python -m common.run_output --label demo` subprocess 출력 경로가 `_demo` 로 끝난다.
- 전체: `.venv-ag3s/bin/python -m pytest tests/test_run_output.py tests/test_policy_record_output.py -q` 통과.

## 하지 말 것

기존 run 폴더 이름 변경 · 이동 금지 (로그 안 절대 경로). `AG3S_RUN_DIR` 의 의미 변경 금지. 문서(`doc/`, `AG3S/ag3s/docs/*.md`) 수정 금지 — 문서 반영은 lead/scribe. commit 금지.

## 넘길 것

`AG3S/ag3s/docs/handoff/IFG-RUNLABEL.impl.md`: 바뀐 함수 · 시그니처, grep 결과 (7번), test 출력 마지막 줄, 하위 호환 확인 방법 (label 없이 만든 이름이 기존 정규식에 맞음).
