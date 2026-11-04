"""SL10 J1 — 기록된 Panda run 에서 MuJoCo geom snapshot 을 만든다 (`panda_fixed_input_ablation --geom-snapshot-dir` 의 입력).

기록 `episodes/<stem>_geom.npz` 에는 상태(`sim_state_image`)와 body · geom 이름 표만 있고 `MjModel` 이 없다. 그래서 그 episode 의
LIBERO env 를 rollout 과 **같은 이력**(task 당 env 하나 + 앞선 episode 마다 `reset` + `set_init_state`, `env_history.episode_envs`)으로
다시 만들어 모델을 얻고, call 프레임마다 `sim_state_image` 를 되돌려 geom 의 world pose 를 쓴다.

검사 (하나라도 어긋나면 그 episode 는 쓰지 않고 실패한다):

* 모델의 body 이름 · geom 이름 · geom→body 표가 기록의 `model_body_names` · `model_geom_names` · `model_geom_bodyid` 와 같다.
* `sim_state_image` 의 길이가 `1 + nq + nv` 이다.
* 되돌린 상태에서 다시 그린 RGB 가 기록 프레임과 비트 단위로 같다 (`env_history.rgb_rerender_equal`, 두 카메라 모두; `--no-rgb-check` 로 끈다).

산출: `<out-dir>/<stem>_geomsnap.npz` — `obs_index`, `geom_xpos`, `geom_xmat`(프레임 × geom), 모델 표(`geom_type` · `geom_size` · `geom_bodyid` ·
`geom_contype` · `geom_conaffinity` · `body_names` · `geom_names`), 그리고 `target_bodies` · `destination_bodies`(BDDL 의 target / 목적지 instance 의 body 이름) ·
`provenance_json`. `panda_fixed_input_ablation.load_geom_snapshot` 이 읽는다.

실행 (`.venv-libero`, Python 3.8; 스스로 `client_env` 환경으로 재실행한다)::

    LIBERO_CONFIG_PATH=<checkout>/.runtime/libero .venv-libero/bin/python AG3S/trajopt/experiments/panda_geom_snapshot.py \\
        --run-dir <run> --out-dir <dir> [--episodes libero_spatial_t1_e0,...] [--frames calls|all]

이 파일의 순수 함수(검사 · 쓰기)는 numpy 와 `mujoco` 뿐이라 Python 3.11 에서도 시험할 수 있다. CPU 단일 process, GPU 불필요.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Optional, Sequence

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "sl10-geomsnap-1"
SUFFIX = "_geomsnap.npz"


def snapshot_path(directory, stem: str) -> pathlib.Path:
    return pathlib.Path(directory) / (stem + SUFFIX)


# --------------------------------------------------------------------------------------
# 순수 함수 (모델 표 검사 · pose 계산 · 쓰기)
# --------------------------------------------------------------------------------------

def model_tables(model) -> dict:
    """`MjModel` 의 body 이름 · geom 이름 · geom→body 표."""
    import mujoco
    import numpy as np

    return {
        "body_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)],
        "geom_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)],
        "geom_bodyid": np.asarray(model.geom_bodyid).copy(),
    }


def check_tables_against_record(tables: dict, record) -> None:
    """모델 표가 기록(`_geom.npz` 의 `model_*`)과 다르면 `ValueError`. 기록에 표가 없는 키는 검사하지 않고 알린다."""
    import numpy as np

    keys = set(getattr(record, "files", record.keys()))
    if "model_body_names" in keys:
        rec = [str(x) for x in record["model_body_names"]]
        if rec != list(tables["body_names"]):
            raise ValueError("body table differs from the record: model {} bodies vs record {} (first difference at {})".format(
                len(tables["body_names"]), len(rec), next((i for i, (a, b) in enumerate(zip(tables["body_names"], rec)) if a != b),
                                                           min(len(rec), len(tables["body_names"])))))
    else:
        raise ValueError("the record has no model_body_names: cannot verify the rebuilt model")
    if "model_geom_names" in keys:
        rec = [str(x) for x in record["model_geom_names"]]
        if rec != list(tables["geom_names"]):
            raise ValueError("geom name table differs from the record: model {} geoms vs record {}".format(
                len(tables["geom_names"]), len(rec)))
    if "model_geom_bodyid" in keys:
        if not np.array_equal(np.asarray(record["model_geom_bodyid"]), np.asarray(tables["geom_bodyid"])):
            raise ValueError("geom->body table differs from the record")


def snapshot_arrays(model, sim_states, obs_indices: Sequence[int]) -> dict:
    """`sim_states` (robosuite `sim.get_state().flatten()` = `[time, qpos, qvel]`) 프레임들의 geom pose 와 모델 표."""
    import mujoco
    import numpy as np

    states = np.asarray(sim_states, np.float64).reshape(-1, np.asarray(sim_states).shape[-1])
    if states.shape[1] != 1 + model.nq + model.nv:
        raise ValueError("sim_state length {} != 1 + nq + nv = {}".format(states.shape[1], 1 + model.nq + model.nv))
    data = mujoco.MjData(model)
    xpos, xmat = [], []
    for s in states:
        data.time = float(s[0])
        data.qpos[:] = s[1:1 + model.nq]
        data.qvel[:] = s[1 + model.nq:1 + model.nq + model.nv]
        mujoco.mj_kinematics(model, data)
        xpos.append(data.geom_xpos.copy())
        xmat.append(data.geom_xmat.copy())
    tables = model_tables(model)
    return {"obs_index": np.asarray(list(obs_indices), np.int64), "geom_xpos": np.asarray(xpos), "geom_xmat": np.asarray(xmat),
            "geom_type": np.asarray(model.geom_type).copy(), "geom_size": np.asarray(model.geom_size).copy(),
            "geom_bodyid": tables["geom_bodyid"], "geom_contype": np.asarray(model.geom_contype).copy(),
            "geom_conaffinity": np.asarray(model.geom_conaffinity).copy(),
            "body_names": np.asarray(tables["body_names"]), "geom_names": np.asarray(tables["geom_names"])}


def write_snapshot(path, arrays: dict, *, target_bodies: Optional[Sequence[str]] = None,
                   destination_bodies: Optional[Sequence[str]] = None, provenance: Optional[dict] = None,
                   exclusive: bool = False) -> pathlib.Path:
    """`snapshot_arrays` 의 결과를 npz 로 쓴다. `exclusive=True` 면 이미 있는 파일을 덮지 않는다."""
    import numpy as np

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(arrays)
    if target_bodies is not None:
        out["target_bodies"] = np.asarray([str(b) for b in target_bodies])
    if destination_bodies is not None:
        out["destination_bodies"] = np.asarray([str(b) for b in destination_bodies])
    prov = dict(provenance or {})
    prov["schema"] = SCHEMA
    out["provenance_json"] = np.asarray(json.dumps(prov, sort_keys=True))
    with open(str(path), "xb" if exclusive else "wb") as stream:
        np.savez_compressed(stream, **out)
    return path


# --------------------------------------------------------------------------------------
# episode / CLI (LIBERO env 가 필요; .venv-libero)
# --------------------------------------------------------------------------------------

def call_frames_of(run, stem: str, geom_obs_index) -> list:
    """`<stem>_calls.jsonl` 의 call 프레임(`obs_index`, call 순서). 기록된 프레임에 없으면 `ValueError`."""
    rows = [json.loads(l) for l in (pathlib.Path(run) / "episodes" / (stem + "_calls.jsonl")).read_text().splitlines() if l.strip()]
    frames = [int(r["obs_index"]) for r in rows]
    have = set(int(x) for x in geom_obs_index)
    missing = sorted(set(frames) - have)
    if missing:
        raise ValueError("{}: call frames {} are not recorded frames".format(stem, missing[:5]))
    return frames


def export_episode(run, stem: str, row: dict, env, out_dir, *, frames: str = "calls", rgb_check: bool = True,
                   exclusive: bool = True, history: Optional[dict] = None) -> dict:
    """한 episode 의 snapshot 을 쓴다. `env` = `env_history.episode_envs` 가 준 그 episode 시작 직후의 env."""
    import numpy as np
    from benchmark.adapters.libero_panda import env_history, task_target as tt

    run = pathlib.Path(run)
    sim = env.sim
    m, d = sim.model._model, sim.data._data
    g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
    check_tables_against_record(model_tables(m), g)
    e = np.load(str(run / "episodes" / (stem + ".npz"))) if rgb_check else None
    rec_idx = np.asarray(g["obs_index"]).astype(np.int64).reshape(-1)
    want = call_frames_of(run, stem, rec_idx) if frames == "calls" else [int(x) for x in rec_idx]
    pos = {int(f): i for i, f in enumerate(rec_idx)}
    states, rgb_equal = [], {}
    for f in want:
        sim.set_state_from_flattened(np.asarray(g["sim_state_image"][pos[f]]).copy())
        sim.forward()
        states.append(np.asarray(g["sim_state_image"][pos[f]], np.float64))
        if rgb_check:
            for cam, equal in env_history.rgb_rerender_equal(sim, e, f).items():
                rgb_equal[cam] = rgb_equal.get(cam, 0) + int(equal)
    if rgb_check:
        env_history.require_rgb_equal(stem, rgb_equal, len(want))
    arrays = snapshot_arrays(m, np.stack(states), want)
    info = tt.task_target(row["suite"], int(row["task_id"]))
    names = [str(n) for n in arrays["body_names"]]
    target_bodies = [names[i] for i in tt.instance_body_ids(info["target"], names)]
    dest_bodies = [names[i] for i in tt.instance_body_ids(info["destination"], names)] if info.get("destination") else []
    import mujoco
    prov = {"run": str(run), "run_id": run.name, "stem": stem, "frames": frames, "n_frames": len(want),
            "target": info["target"], "destination": info.get("destination"), "mujoco": mujoco.__version__,
            "python": sys.version.split()[0], "state": "sim_state_image (image instant) restored, mj_kinematics",
            "rgb_rerender_equal_call_frames": rgb_equal if rgb_check else "not checked",
            "env_history": history, "tables_checked_against_record": True}
    path = write_snapshot(snapshot_path(out_dir, stem), arrays, target_bodies=target_bodies, destination_bodies=dest_bodies,
                          provenance=prov, exclusive=exclusive)
    return {"file": path.name, "frames": len(want), "target_bodies": target_bodies, "destination_bodies": dest_bodies,
            "rgb_rerender_equal": rgb_equal if rgb_check else None}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, help="geometry-capture run (episodes/ · episodes.jsonl · run_manifest.json); read only")
    ap.add_argument("--out-dir", required=True, help="snapshot 을 쓸 디렉터리 (<stem>_geomsnap.npz, 덮어쓰지 않는다)")
    ap.add_argument("--episodes", default="all", help="`all` 또는 episode stem 목록 (쉼표)")
    ap.add_argument("--frames", choices=("calls", "all"), default="calls")
    ap.add_argument("--no-rgb-check", dest="rgb_check", action="store_false",
                    help="되돌린 상태의 RGB 가 기록과 같은지 검사하지 않는다 (env 이력이 틀려도 모른다)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from benchmark.adapters.libero_panda import env_history

    run, out = pathlib.Path(args.run_dir).resolve(), pathlib.Path(args.out_dir)
    stems_all = [env_history.stem_of(r) for r in env_history.read_rows(run)]
    wanted = set(stems_all) if args.episodes == "all" else set(args.episodes.split(","))
    unknown = wanted - set(stems_all)
    if unknown:
        raise SystemExit("unknown episodes {}".format(sorted(unknown)))
    env_history.run_history(run)  # 알 수 없는 순서는 아무것도 만들기 전에 거부
    out.mkdir(parents=True, exist_ok=True)
    t0, results = time.time(), []
    for row, env, note in env_history.episode_envs(run, wanted):
        stem = env_history.stem_of(row)
        res = export_episode(run, stem, row, env, out, frames=args.frames, rgb_check=args.rgb_check, history=note)
        res["seconds"] = round(time.time() - t0, 1)
        results.append(res)
        print(json.dumps({k: res[k] for k in ("file", "frames", "rgb_rerender_equal", "seconds")}), flush=True)
    with open(str(out / "geomsnap_export.json"), "w") as stream:
        stream.write(json.dumps({"schema": SCHEMA, "run": str(run), "argv": sys.argv, "episodes": results,
                                 "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
    sys.stdout.flush()
    os._exit(0)  # LIBERO / OSMesa teardown can hang (phase_export does the same)


if __name__ == "__main__":
    # 스크립트로 실행될 때만: client_env (LD_LIBRARY_PATH 정리 · PYTHONPATH 제거) 로 재실행. numpy / LIBERO 를 import 하기 전에.
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "adapters" / "libero_panda"))
    import client_env  # noqa: E402

    client_env.reexec_with_client_environment()
    raise SystemExit(main())
