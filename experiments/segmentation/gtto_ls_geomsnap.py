"""LIBERO-Safety 판 geom snapshot (원격 SL10 J1 `panda_geom_snapshot.py` 와 같은 `_geomsnap.npz` 형식).

원격 도구는 LIBERO (`env_history` · `task_target`) 전용이라 LIBERO-Safety run 에는 쓸 수 없다. 여기서는 그 run 의 client 와 같은 방식
(`activate_libero_safety` → `LevelSuite` → `base.make_task_env(task, seed, capture_geometry=True)`) 으로 env 를 만들고,
run 의 episode 순서대로 `reset + set_init_state` 를 반복해 fixture 배치 이력을 맞춘 뒤, 기록 `_geom.npz` 의 `sim_state_image` 를
call 프레임마다 넣어 geom pose 를 쓴다 (`panda_geom_snapshot.snapshot_arrays` · `write_snapshot` 그대로).

검사: 모델 표 (body · geom 이름 · geom→body) 가 기록과 같다 (`check_tables_against_record`), sim_state 길이 = 1 + nq + nv,
그리고 되돌린 상태에서 다시 그린 RGB 가 기록 프레임과 같은지 (agentview 만, 같은 비율을 provenance 에 적음 — 다르면 실패하지 않고 기록).

실행 (`.venv-libero-safety`, `client_env` 로 재실행)::

    .venv-libero-safety/bin/python experiments/segmentation/gtto_ls_geomsnap.py --run-dir <run> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-bodies", default="", help="쉼표 목록 (기본: gt_perception grounding target 이름으로 찾음)")
    a = ap.parse_args(argv)
    for p in (str(ROOT), str(ROOT / "benchmark" / "runners")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    import libero_safety_client as lsc

    info = lsc.activate_libero_safety()
    import libero_pi05_client as base
    from AG3S.trajopt.experiments import panda_geom_snapshot as pgs

    run = pathlib.Path(a.run_dir).resolve()
    man = json.loads((run / "run_manifest.json").read_text())
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("episode_idx") is not None or r.get("episode_id") is not None]
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from libero.libero import benchmark as lb

    results, t0 = [], time.time()
    by_task = {}
    for r in rows:
        by_task.setdefault((r["suite"], int(r["level"]), int(r["task_id"])), []).append(r)
    for (suite, level, task_id), rs in by_task.items():
        bench = lb.get_benchmark_dict()[suite]() if suite in lb.get_benchmark_dict() else None
        if bench is None:
            raise SystemExit("unknown suite {}".format(suite))
        ts = lsc.LevelSuite(bench, level)
        task = ts.get_task(task_id)
        init = ts.get_task_init_states(task_id)
        seed = int(rs[0]["seed"])
        env, desc = base.make_task_env(task, seed, True)
        for r in sorted(rs, key=lambda x: int(x.get("episode_idx", x.get("episode_id")))):
            ep = int(r.get("episode_idx", r.get("episode_id")))
            base.start_episode(env, init, ep)
            stem = lsc.episode_stem(suite, level, task_id, ep)
            inner = lsc._inner(env)
            m = inner.sim.model._model
            g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
            pgs.check_tables_against_record(pgs.model_tables(m), g)
            rec_idx = np.asarray(g["obs_index"]).astype(np.int64).reshape(-1)
            want = pgs.call_frames_of(run, stem, rec_idx)
            pos = {int(f): i for i, f in enumerate(rec_idx)}
            states = np.stack([np.asarray(g["sim_state_image"][pos[f]], np.float64) for f in want])
            arrays = pgs.snapshot_arrays(m, states, want)
            names = [str(n) for n in arrays["body_names"]]
            tb = [b for b in a.target_bodies.split(",") if b]
            if not tb:
                calls = [json.loads(l) for l in (run / "episodes" / (stem + "_calls.jsonl")).read_text().splitlines() if l.strip()]
                tname = next((((c.get("gt_perception") or {}).get("grounding") or {}).get("target") for c in calls
                              if ((c.get("gt_perception") or {}).get("grounding") or {}).get("target")), None)
                tb = [n for n in names if tname and n.startswith(str(tname))]
            prov = {"run": str(run), "stem": stem, "frames": "calls", "n_frames": len(want), "task": desc,
                    "history": "make_task_env(seed={}) + start_episode per episode in run order".format(seed),
                    "tables_checked_against_record": True, "libero_safety": info, "tool": "experiments/segmentation/gtto_ls_geomsnap.py"}
            path = pgs.write_snapshot(pgs.snapshot_path(out, stem), arrays, target_bodies=tb, destination_bodies=[],
                                      provenance=prov, exclusive=False)
            res = {"file": path.name, "frames": len(want), "target_bodies": tb, "seconds": round(time.time() - t0, 1)}
            results.append(res)
            print(json.dumps(res), flush=True)
    (out / "geomsnap_export.json").write_text(json.dumps({"run": str(run), "episodes": results}, indent=1) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.path.append(str(ROOT / "benchmark" / "adapters" / "libero_panda"))
    import client_env  # noqa: E402

    client_env.reexec_with_client_environment()
    raise SystemExit(main())
