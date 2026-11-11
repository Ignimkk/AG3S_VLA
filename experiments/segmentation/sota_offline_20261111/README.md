# SL-SOTA offline 분석 도구 (dev-57, 2026-11-11 – 11-12, AMRC)

AMRC checkout 의 `outputs/2026-11-11_SOTA_개선/` (git 제외) 와 세션 scratchpad 에 있던 분석 script 의 사본이다. SDM 에서 이어 쓰려고 repo 에 넣었다 (계획 문서 `doc/active/experiments/AG3S_PLAN_SCHEDULE_20261110_KO.md` §9.5–§9.23). 결과 JSON · queue log · README 원본은 SDM 의 `/mnt/dev/AG3S_VLA_amrc/outputs/2026-11-11_SOTA_개선/` 에 있다 (참고용 사본).

- 입력은 `--record-constraints full` 로 남긴 `vla_v11ab` run (`AG3S_VLA-storage/results/benchmark/<run>/` 의 `episodes.jsonl`, `episodes/*_calls.jsonl · *_calls.npz · *_geom.npz`, `server_constraints/run_*/chunk_*.npz`). script 안의 경로는 `/mnt/dev/AG3S_VLA` · `/mnt/dev/AG3S_VLA-storage` 로 박혀 있다.
- GT (sim 상태 · 물체 이름) 는 채점에만 쓴다.
- CPU job 은 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, 세션당 worker 2 개 이하.

| script | 환경 | 무엇을 답하나 | 쓰는 법 |
|---|---|---|---|
| `tri.py` | `.venv-ag3s` | E1a 판끼리 episode 별 결과 · 얻음 / 잃음 | `tri.py qe-pe123458 sub-pkgFqs sub-pkgFin` |
| `guard.py` | `.venv-ag3s` | LIBERO 27 · SafeLIBERO 11 · LS 10 요약 | `guard.py pkgFin` |
| `scr.py` | `.venv-ag3s` | 거르기 run 의 episode 별 latch 상태열 · attach call · sticky 사용 · zone 풀린 행 | `scr.py sub-pkgSrs 0` |
| `ep.py` · `tl.py` | `.venv-ag3s` | 한 episode 의 call 별 통과 / QP / HOLD · 최악 행 · 인내 한도 | `ep.py sub-pkgFin 0 13 3 14` |
| `shape.py` | `.venv-ag3s` | 안전 성공이 아닌 episode 의 형태 분류 (쥐기 전 교착 · 쥔 채 교착 · 놓은 뒤 미완 …) | `shape.py --out=F.json "2*_sub-pkgFin-e1a-spatial-LI-t*"` |
| `stall.py` | `.venv-ag3s` | 막힌 call 의 link · 여유 · 인내 한도 통계 | `stall.py sub-pkgFin "1:6 1:8" F.json` |
| `osqp_stat.py` | `.venv-ag3s` | action QP 의 OSQP 반복 · 상한 도달 | `osqp_stat.py sub-pkgEhr150 sub-pkgFqs` |
| `surface_owner.py` · `surface_owner_task.py` | `.venv-libero` | 최악 장애물 행의 표면이 실제로 어느 물체인가 (`_task` = 여러 task 가 한 run 에 든 경우 · plain LIBERO) | `surface_owner.py --run-dir RUN --episode 13 --out F.json` |
| `target_truth.py` | `.venv-libero` | 인식된 target 점이 실제로 어느 물체인가 (call 별) | `target_truth.py --run-dir RUN --out F.json` |
| `zone_rows.py` | `.venv-libero` | grasp zone 에서 풀린 손 · 손가락 행 중 침투가 예측된 행의 실제 물체 | `zone_rows.py --run-dir RUN --out F.json` |
| `zp_kept.py` | `.venv-ag3s` | `-zp` run 에서 손바닥 행이 남은 call 수 (latch 상태별) · 최대 행 수 | `zp_kept.py sub-pkgSzp F.json` |
| `mscmp.py` | `.venv-ag3s` | 다른 suite (init 15 · 16) cell 을 판끼리 episode 별로 비교 | `mscmp.py pe123458 pkgFqs pkgFrs` |
| `first_contact.py` | `.venv-libero` | 충돌 episode 에서 로봇의 어느 geom 이 언제 장애물에 처음 닿았나 | script 머리말 |
| `latch_rs_replay.py` | `.venv-ag3s` (`PYTHONPATH=<checkout>`) | 기록된 latch 입력을 `-rs` 켜고 / 끄고 다시 돌려 판정이 바뀌는 call 을 찾는다 | `latch_rs_replay.py --out F.json "2*_sub-pkgFin-e1a-*"` |
| `fig_qs.py` · `fig_ms.py` · `fig_fin.py` · `fig_rs_zr.py` · `fig_zp.py` · `fig_e1a40.py` | `.venv-ag3s` | `AG3S/ag3s/docs/figures/sl-sota-*.png` 를 만든 script | `fig_fin.py OUT.png` |
| 나머지 (`row_owner.py`, `peak_owner.py`, `obst_class_diag.py`, `obs_margin_scan.py`, `exec_clearance_scan.py`, `held_motion_*.py`, `fig_first_contact.py`, `sota_compare.py`) | script 머리말 | 11-11 낮의 개별 진단 (계획 문서 §9.5–§9.11) | script 머리말 |

재생 도구 (`.venv-libero`) 는 `LIBERO_CONFIG_PATH=$PWD/.runtime/libero MUJOCO_GL=egl` 을 준다. env 를 다시 만들 때 `RandomizationError: Cannot place all objects` 가 가끔 난다 — 다시 돌리면 된다.

SDM 에서는 run-id 가 `-sdm` 으로 끝나므로 tag 인자도 그렇게 준다 (예: `tri.py sub-pkgFinsdm sub-pkgFrssdm`). 이전 후 첫 일 (§9.23 A-0) 은 `zone_rows.py` · `surface_owner.py` 를 `sub-pkgSzp` run 의 잃은 6 ep 에 돌리는 것이다.
