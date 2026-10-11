"""T42 — re-render a recorded ``--record-frames`` run with the avoidance arrow (no policy server).

Input: one run directory with ``manifest.json`` + ``frames.jsonl``, as written by
``pi05_infer.py --record-frames``. The run must have used ``--safe-remote``, because its planning
records carry ``actions`` (refined) and ``actions_reference`` (policy).

What is replayed, step by step (``t`` = control step):

* Scene: the model and obstacle profile of the run (``manifest.extra.argv``), with
  ``qpos = control[t + 1].qpos``. That is the state after action ``t`` ran, which is what the live
  ``--record`` frame ``t`` shows: the live loop renders after the ``mj_step`` calls of step
  ``t``, and ``control[t + 1].qpos`` is taken before any ``mj_step`` of ``t + 1``. The last step
  has no successor, so it uses its own ``qpos``.
* Arrow: the same ``avoidance_arrow.AvoidanceArrows.step`` that the live path calls, with
  ``executed`` from the control record, ``refined`` / ``reference`` from row ``step_in_chunk`` of
  the planning record ``chunk_seq``, and on HOLD the applied arm target (``applied_ctrl.arm``,
  which is ``q_hold``).

Outputs in ``--out``: ``<name>.mp4`` (arrows + caption), ``stills/*.png``, ``vectors.json``
(per-step vector, magnitude, state, arm), ``deflection.png`` (|v| over t) and ``replay_check.json``.
With ``--compare-original``, ``replay_check.json`` also holds the per-frame pixel difference
between the replay **without** the overlay and the run's own mp4 (a fidelity check of the replay
itself).

Run (CPU, osmesa)::

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-openpi-live/bin/python \
        integration/rby1/bringup/replay_avoidance_arrow.py RUN_DIR --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for p in (str(HERE), str(REPO_ROOT / "integration" / "rby1" / "manipulation" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import avoidance_arrow as aa  # noqa: E402
from rby1_manipulation.simulation.pick_place_obstacles import (  # noqa: E402
    PickPlaceObstacleManager, load_pick_place_obstacle_config)

MJCF = REPO_ROOT / "robots" / "rby1" / "description" / "models" / "rby1a" / "mujoco"
TRANSPORT_XML = MJCF / "model_transport.xml"
TRANSPORT_OBSTACLE_XML = MJCF / "model_transport_pick_place_obstacles.xml"


def _argv_value(argv: list, flag: str, default=None):
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def _argv_values(argv: list, flag: str) -> list:
    if flag not in argv:
        return []
    out = []
    for v in argv[argv.index(flag) + 1:]:
        if v.startswith("--"):
            break
        out.append(v)
    return out


def configure_front_camera(cam: mujoco.MjvCamera) -> None:
    """Same preset as ``pi05_infer.configure_view_camera(cam, "front")``."""
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray([0.45, 0.0, 0.85], dtype=np.float64)
    cam.distance = 1.7
    cam.azimuth = 180.0
    cam.elevation = -18.0


def load_run(run_dir: pathlib.Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    planning, control = {}, []
    with (run_dir / "frames.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["kind"] == "planning":
                planning[int(rec["seq"])] = rec
            elif rec["kind"] == "control":
                control.append(rec)
    control.sort(key=lambda r: int(r["t_step"]))
    return {"manifest": manifest, "planning": planning, "control": control}


def copy_visual_rgba(m: mujoco.MjModel, rgba_xml: str) -> list:
    """Copy ``geom_rgba`` by geom name from ``rgba_xml`` (visual only; returns the changed names).

    T42-2 recoloured the obstacle visual geoms black and changed nothing else in the obstacle XML,
    so the run's own XML gives back the colours the run's mp4 showed.
    """
    src = mujoco.MjModel.from_xml_path(rgba_xml)
    changed = []
    for g in range(src.ngeom):
        name = mujoco.mj_id2name(src, mujoco.mjtObj.mjOBJ_GEOM, g)
        if not name:
            continue
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0 and not np.array_equal(m.geom_rgba[gid], src.geom_rgba[g]):
            m.geom_rgba[gid] = src.geom_rgba[g]
            changed.append(name)
    return changed


def build_scene(manifest: dict, model_xml: str | None = None, rgba_xml: str | None = None):
    argv = list(manifest.get("extra", {}).get("argv") or [])
    profile = _argv_value(argv, "--obstacle-profile", "clear")
    cfg_path = _argv_value(argv, "--obstacle-config")
    if model_xml is None:
        model_xml = str(TRANSPORT_OBSTACLE_XML if profile != "clear" else TRANSPORT_XML)
    m = mujoco.MjModel.from_xml_path(model_xml)
    recoloured = copy_visual_rgba(m, rgba_xml) if rgba_xml else []
    d = mujoco.MjData(m)
    manager = None
    if profile != "clear":
        config = (load_pick_place_obstacle_config(cfg_path) if cfg_path
                  else load_pick_place_obstacle_config())
        manager = PickPlaceObstacleManager(m, d, config)
        manager.activate(profile)
    return m, d, manager, {"model_xml": model_xml, "obstacle_profile": profile,
                           "obstacle_config": cfg_path, "argv": argv,
                           "rgba_from": rgba_xml, "recoloured_geoms": recoloured}


def _applied_arm(rec: dict) -> dict | None:
    arm = (rec.get("applied_ctrl") or {}).get("arm")
    if arm is None or len(arm) < 14:
        return None
    arm = np.asarray(arm, dtype=np.float64)
    return {"left": arm[0:7], "right": arm[7:14]}


def _row(chunk, i):
    if chunk is None or len(chunk) <= i:
        return None
    return np.asarray(chunk[i], dtype=np.float64)


def replay(run_dir, out_dir, *, threshold_m=aa.DEFAULT_THRESHOLD_M, scale=aa.DEFAULT_SCALE,
           obstacle_normal=False, start=0, stop=None, stills=None, n_auto_stills=4,
           compare_original=False, model_xml=None, rgba_xml=None, width=640, height=480, name=None,
           caption=True, inset=False) -> dict:
    run_dir, out_dir = pathlib.Path(run_dir), pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = load_run(run_dir)
    manifest, planning, control = run["manifest"], run["planning"], run["control"]
    if not planning or all(p.get("actions_reference") is None for p in planning.values()):
        raise SystemExit(f"{run_dir}: no planning record carries actions_reference "
                         "(E0 / --no-safe run?) - there is no refined/reference pair to draw")
    m, d, manager, scene_info = build_scene(manifest, model_xml, rgba_xml)
    scene = manifest.get("scene", {})
    action_format = scene.get("action_format", "rby1_16d")
    used_arm = scene.get("episode_used_arm")
    arms = ((used_arm,) if used_arm in ("left", "right") else
            tuple(a for a in _argv_values(scene_info["argv"], "--safe-manipulators")
                  if a in ("left", "right")) or ("left", "right"))
    viz = aa.AvoidanceArrows(
        m, action_format=action_format, arms=arms, threshold_m=threshold_m, scale=scale,
        obstacle_geom_ids=(sorted(manager.collision_geom_ids) if manager else ()),
        obstacle_normal=obstacle_normal)
    if len(control[0]["qpos"]) != m.nq:
        raise SystemExit(f"recorded qpos has {len(control[0]['qpos'])} values, model nq={m.nq}: "
                         "pass --model-xml of the run's asset tree")

    renderer = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    view = _argv_value(scene_info["argv"], "--view", "free")
    if view == "front":
        configure_front_camera(cam)
    else:
        cam = -1

    inset_renderer, inset_cam = None, None
    if inset:
        inset_renderer = mujoco.Renderer(m, height=height * 2 // 5, width=width * 2 // 5)
        inset_cam = mujoco.MjvCamera()
        configure_front_camera(inset_cam)
        inset_cam.distance = 0.55

    original = None
    if compare_original:
        mp4 = manifest.get("render", {}).get("third_person_video")
        if mp4 and pathlib.Path(mp4).exists():
            import imageio.v2 as iio

            original = iio.get_reader(mp4)

    stop = len(control) if stop is None else min(stop, len(control))
    frames, per_step, diffs, plain_kept = [], [], [], []
    for t in range(start, stop):
        rec = control[t]
        nxt = control[t + 1] if t + 1 < len(control) else rec
        d.qpos[:] = np.asarray(nxt["qpos"], dtype=np.float64)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        plan = planning.get(int(rec["chunk_seq"]), {})
        k = int(rec["step_in_chunk"])
        ov = viz.step(d, executed=bool(rec.get("executed")),
                      refined_row=_row(plan.get("actions"), k),
                      reference_row=_row(plan.get("actions_reference"), k),
                      applied_arm=_applied_arm(rec))
        renderer.update_scene(d, camera=cam)
        if original is not None:
            plain = renderer.render().copy()
            try:
                ref_img = np.asarray(original.get_data(t))
                if ref_img.shape == plain.shape:
                    if len(plain_kept) < 30:
                        plain_kept.append(plain)
                    diffs.append(float(np.abs(ref_img.astype(np.int16)
                                              - plain.astype(np.int16)).mean()))
            except IndexError:
                pass
            renderer.update_scene(d, camera=cam)
        aa.draw(renderer.scene, ov.items)
        img = renderer.render().copy()
        if inset_renderer is not None and ov.side is not None:
            inset_cam.lookat[:] = viz.anchor(d, ov.side)
            inset_renderer.update_scene(d, camera=inset_cam)
            aa.draw(inset_renderer.scene, ov.items)
            small = inset_renderer.render()
            h, w = small.shape[:2]
            img[-h - 4:-4, -w - 4:-4] = small
            img[-h - 6:-h - 4, -w - 6:-2] = 255
            img[-h - 6:-2, -w - 6:-w - 4] = 255
        if caption:
            lines = ov.caption(viz.scale, viz.threshold_m) or ["(no refined/reference pair)"]
            lines = lines + [f"replay t={t} chunk={rec['chunk_seq']} row={k}"]
            img = aa.caption(img, lines, aa.overlay_colour(ov))
        frames.append(img)
        per_step.append({
            "t": t, "chunk_seq": int(rec["chunk_seq"]), "step_in_chunk": k,
            "state": ov.state, "arm": ov.side,
            "vector_m": None if ov.vector_m is None else [float(x) for x in ov.vector_m],
            "magnitude_mm": None if ov.magnitude_m is None else ov.magnitude_m * 1000.0,
            "drawn": ov.drawn, "n_items": len(ov.items),
            "normal_mm": None if ov.normal is None else ov.normal["distance_m"] * 1000.0,
            "gate_action": (plan.get("gate") or {}).get("action")})

    import imageio.v2 as iio

    name = name or f"{run_dir.name}_arrow"
    mp4_out = out_dir / f"{name}.mp4"
    iio.mimsave(mp4_out, frames, fps=int(manifest.get("timing", {}).get("ctrl_hz", 15)))

    # stills: given steps, else the largest execute deflections + the largest HOLD one
    if stills is None:
        exe = sorted((s for s in per_step if s["state"] == "execute" and s["drawn"]),
                     key=lambda s: -s["magnitude_mm"])
        chosen = []
        for s in exe:
            if all(abs(s["t"] - c) >= 24 for c in chosen):
                chosen.append(s["t"])
            if len(chosen) >= n_auto_stills:
                break
        hold = [s for s in per_step if s["state"] == "hold"]
        if hold:
            chosen.append(max(hold, key=lambda s: s["magnitude_mm"] or 0.0)["t"])
        stills = sorted(chosen)
    still_dir = out_dir / "stills"
    still_dir.mkdir(exist_ok=True)
    still_paths = []
    from PIL import Image

    for t in stills:
        if start <= t < stop:
            p = still_dir / f"{name}_t{t:04d}.png"
            Image.fromarray(frames[t - start]).save(p)
            still_paths.append(str(p))

    mags = [s["magnitude_mm"] for s in per_step if s["magnitude_mm"] is not None]
    summary = {
        "run": str(run_dir), "out_mp4": str(mp4_out), "stills": still_paths,
        "scene": scene_info | {"arms": list(arms), "action_format": action_format},
        "threshold_mm": threshold_m * 1000.0, "scale": scale, "obstacle_normal": obstacle_normal,
        "steps": [start, stop], "n_frames": len(frames),
        "n_execute": sum(s["state"] == "execute" for s in per_step),
        "n_hold": sum(s["state"] == "hold" for s in per_step),
        "n_drawn": sum(s["drawn"] for s in per_step),
        "magnitude_mm": {"max": max(mags) if mags else None,
                         "median": float(np.median(mags)) if mags else None,
                         "n_above_threshold": sum(x > threshold_m * 1000.0 for x in mags)},
        "render_rule": "frame t = qpos of control[t+1] (state after action t), arrow of action t",
    }
    if original is not None:
        floor = None
        if plain_kept:
            # the same frames through the same mp4 writer and back: what compression alone costs
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                f = pathlib.Path(tmp) / "floor.mp4"
                iio.mimsave(f, plain_kept, fps=15)
                back = iio.get_reader(f)
                floor = float(np.mean([np.abs(np.asarray(back.get_data(i)).astype(np.int16)
                                              - plain_kept[i].astype(np.int16)).mean()
                                       for i in range(len(plain_kept))]))
                back.close()
        summary["fidelity_vs_original_mp4"] = {
            "compression_floor_mean_abs_pixel_diff": floor,
            "n_compared": len(diffs),
            "mean_abs_pixel_diff": {"mean": float(np.mean(diffs)) if diffs else None,
                                    "max": float(np.max(diffs)) if diffs else None},
            "note": "replay rendered WITHOUT the overlay vs the run's own --record mp4 "
                    "(both mp4-compressed: the original once, the replay not)"}
    (out_dir / "vectors.json").write_text(json.dumps(per_step, indent=1))
    (out_dir / "replay_check.json").write_text(json.dumps(summary, indent=1))
    _plot(per_step, out_dir / "deflection.png", threshold_m, run_dir.name)
    return summary


def _plot(per_step, path, threshold_m, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)
    for state, colour, label in (("execute", "#1a9e3a", "execute: |TCP(refined) - TCP(reference)|"),
                                 ("hold", "#d62020", "HOLD: |TCP(q_hold) - TCP(reference)|")):
        pts = [(s["t"], s["magnitude_mm"]) for s in per_step
               if s["state"] == state and s["magnitude_mm"] is not None]
        if pts:
            t, v = zip(*pts)
            ax.scatter(t, v, s=6, color=colour, label=label)
    marks = [s["t"] for s in per_step if s["state"] == "hold" and s["magnitude_mm"] is None]
    if marks:
        ax.scatter(marks, [0] * len(marks), marker="x", s=12, color="#d62020",
                   label="HOLD without reference (marker only)")
    ax.axhline(threshold_m * 1000.0, color="#666", lw=0.8, ls="--",
               label=f"threshold {threshold_m * 1000:.0f} mm")
    ax.set_xlabel("control step t (15 Hz)")
    ax.set_ylabel("TO deflection at TCP [mm]")
    ax.set_title(f"{title}: avoidance vector magnitude per step")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--threshold-mm", type=float, default=aa.DEFAULT_THRESHOLD_M * 1000.0)
    ap.add_argument("--scale", type=float, default=aa.DEFAULT_SCALE)
    ap.add_argument("--obstacle-normal", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--stills", type=int, nargs="*", default=None,
                    help="control steps to save as PNG (default: auto)")
    ap.add_argument("--compare-original", action="store_true")
    ap.add_argument("--model-xml", default=None)
    ap.add_argument("--rgba-from", default=None, metavar="XML",
                    help="copy visual geom_rgba by name from this XML (e.g. the run's own asset "
                         "tree), so the replay shows the colours the run showed")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--inset", action="store_true",
                    help="paste a close-up (same view, 0.55 m from the TCP) bottom-right")
    args = ap.parse_args(argv)
    summary = replay(args.run_dir, args.out, threshold_m=args.threshold_mm / 1000.0,
                     scale=args.scale, obstacle_normal=args.obstacle_normal, start=args.start,
                     stop=args.stop, stills=args.stills, compare_original=args.compare_original,
                     model_xml=args.model_xml, rgba_xml=args.rgba_from, name=args.name, caption=not args.no_caption, inset=args.inset)
    print(json.dumps({k: v for k, v in summary.items() if k != "scene"}, indent=1))


if __name__ == "__main__":
    main()
