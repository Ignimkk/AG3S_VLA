"""T63 step 0 — export the **baseline-side** inputs that the cuRobo v2 side consumes.

This script runs against the **baseline snapshot tree** (`f0d8046`, the C1aN serving code of the
IFG-APPLE run), not against the working tree, because the working-tree `AG3S/trajopt` carries
other sessions' uncommitted flags. It is executed as a *script* with the snapshot first on
`PYTHONPATH`, so every `import AG3S...` below resolves into the snapshot:

    SNAP=$AG3S_STORAGE_ROOT/results/development/20261019T132719+0900_f15e587290f6_11408fd6d27f/snapshot
    cd "$SNAP" && MUJOCO_GL=osmesa PYTHONPATH="$SNAP" \\
        $AG3S_PROJECT_ROOT/.venv-ag3s/bin/python \\
        $AG3S_PROJECT_ROOT/AG3S/trajopt/curobo_v2/export_inputs.py \\
        --snapshot "$SNAP" --ifg "$SNAP/../verify/IFG-APPLE" \\
        --episodes 211 320 572 850 901 954 250 1345 126 309 \\
        --out "$AG3S_RUN_DIR/impl/T63/inputs"

(`cd "$SNAP"` matters: the snapshot resolves its URDF and the client gripper convention relative to
the working directory.)

What it writes (`--out`):

| file | content |
|---|---|
| `robot_spec.json` | the constraint sphere model **as the server built it** (same builder, same flags parsed from the server argv): URDF, every sphere's link / link-frame centre / radius in the server's order, finger joints (prismatic axis, limits), torso joints, the 14 optimized arm joints in chunk-column order and the TO's effective joint limits (`build_limits`, `limits.source = model_xml`, margin and 0.9 scales applied). Also the checks: 270 spheres, radii == record `sphere_radii` |
| `episode_<E>.npz` | per chunk of that episode: record file, `q_now` (20, `DEFAULT_RBY1_JOINTS`), `T_base_torso5` (4x4, baseline FK at `q_now`), finger joint path over the plan window and over the full chunk (`FingerJointMap.path`, the server's envelope), baseline FK sphere centres of the **recorded refined chunk** over the plan window (the reference for the cuRobo FK parity), pairing residuals, attached flag, TCP displacement refined vs reference, recorded TO-view / recorder clearances, target-field-exclude mask |
| `export_manifest.json` | inputs, episode -> server / chunk range, the server argv, the snapshot path actually imported |

Nothing in the input records is modified.

T63 step 2 (B4) adds `--contact-policy-from <export dir> --out <new dir>`: the contact policy of
every exported chunk by the snapshot's `fixed_input_ablation.contact_policy` (`export_contact_policy`).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import pathlib
import re
import shlex
import sys
from typing import Any, Optional

import numpy as np

TOOL = "AG3S/trajopt/curobo_v2/export_inputs.py"

#: The 16D chunk columns (`ChunkLayout.rby1`, arm_joint_dim = 7): left arm 0-6, left gripper,
#: right arm 0-6, right gripper. The cuRobo c-space uses the 14 arm joints in this order.
ARM_JOINTS_CHUNK_ORDER: tuple[str, ...] = (
    *(f"left_arm_{i}" for i in range(7)), *(f"right_arm_{i}" for i in range(7)))
GRIPPER_COLUMNS = {"left": 7, "right": 15}
#: TCP = mean of that hand's finger spheres (`fixed_input_ablation.FINGER_LINKS`, URDF names).
FINGER_LINKS = {"left": ("ee_finger_l1", "ee_finger_l2"), "right": ("ee_finger_r1", "ee_finger_r2")}
CUROBO_BASE_LINK = "link_torso_5"

# server flags that change the constraint sphere model; the ones this exporter rebuilds are parsed,
# the rest must be absent (otherwise the rebuilt model is not the served one).
_UNSUPPORTED_MODEL_FLAGS = ("--gripper-cover", "--palm-cover", "--wrist-cover", "--exclude-links",
                            "--capsule-radius-scale-link", "--capsule-extent-link",
                            "--finger-cover-overflow", "--finger-cover-gap",
                            "--finger-cover-max-spheres", "--finger-cover-solver")


# =================================================================================================
# server argv -> model options
# =================================================================================================

def read_server_argv(server_dir: pathlib.Path) -> list[str]:
    """The server command line recorded in `server_proc.txt` (the `-m AG3S.trajopt.serve_safe` line)."""
    text = (server_dir / "server_proc.txt").read_text(errors="replace")
    for line in text.splitlines():
        if "AG3S.trajopt.serve_safe" in line:
            argv = shlex.split(line)
            return argv[argv.index("AG3S.trajopt.serve_safe") + 1:]
    raise SystemExit(f"{server_dir}/server_proc.txt has no serve_safe command line")


def model_options_from_argv(argv: list[str]) -> dict[str, Any]:
    """The subset of serve_safe flags that build the constraint sphere model and the TO limits."""
    bad = [f for f in _UNSUPPORTED_MODEL_FLAGS if f in argv]
    if bad:
        raise SystemExit(f"server argv uses {bad}; this exporter does not rebuild that model")
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--links", default="gripper")
    ap.add_argument("--sphere-spacing", type=float)
    ap.add_argument("--max-spheres-per-capsule", type=int)
    ap.add_argument("--capsule-radius-scale", type=float)
    ap.add_argument("--max-sphere-radius", type=float)
    ap.add_argument("--finger-cover", nargs="?", const="both", default=None)
    ap.add_argument("--model-xml", required=True)
    ap.add_argument("--esdf-margin", type=float, default=None)
    ap.add_argument("--plan-horizon", type=int, default=None)
    ap.add_argument("--target-field-policy", default=None)
    ap.add_argument("--limits-source", default="model_xml")
    ns, _ = ap.parse_known_args(argv)
    if ns.limits_source != "model_xml":
        raise SystemExit(f"--limits-source {ns.limits_source} is not handled by this exporter")
    sphere = {k: v for k, v in {
        "sphere_spacing": ns.sphere_spacing,
        "max_spheres_per_capsule": ns.max_spheres_per_capsule,
        "capsule_radius_scale": ns.capsule_radius_scale,
        "max_sphere_radius": ns.max_sphere_radius}.items() if v is not None}
    return {"links": ns.links, "sphere_options": sphere,
            "finger_cover": ({"hands": ns.finger_cover} if ns.finger_cover else None),
            "model_xml": ns.model_xml, "esdf_margin": ns.esdf_margin,
            "plan_horizon": ns.plan_horizon, "target_field_policy": ns.target_field_policy,
            "limits_source": ns.limits_source}


# =================================================================================================
# episode <-> server record mapping
# =================================================================================================

def episode_servers(runs: pathlib.Path, arm: str = "C1aN") -> dict[int, dict[str, Any]]:
    """`{episode: {server_dir, order}}` from the chain logs.

    `chain_<port>.log` lists, per server, `server up arm=<arm> ... rec=<dir>` followed by the
    `start <arm>/<arm>_ep<E>_s<S> attempt=<n>` lines of the episodes served by it, in order. A
    server keeps one `run_0000` for all of them, so the k-th episode of a server owns the k-th block
    of chunk files (the block size is checked against the episode's planning rows, and every chunk
    is checked against the client's `actions` afterwards — this mapping is only the hypothesis).
    """
    out: dict[int, dict[str, Any]] = {}
    for log in sorted(runs.glob("chain_*.log")):
        current: Optional[str] = None
        order = 0
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"server up arm=(\S+) pid=\d+ rec=(\S+)", line)
            if m:
                current = m.group(2) if m.group(1) == arm else None
                order = 0
                continue
            m = re.match(rf"start {arm}/{arm}_ep(\d+)_s(\d+) attempt=(\d+)", line)
            if m and current is not None:
                if int(m.group(3)) != 1:
                    raise SystemExit(f"{log}: retried episode ({line[:80]}) — block mapping unsafe")
                out[int(m.group(1))] = {"server_dir": current, "order": order,
                                        "seed": int(m.group(2))}
                order += 1
    return out


def episode_dir(runs: pathlib.Path, arm: str, episode: int) -> pathlib.Path:
    hits = [pathlib.Path(p) for p in glob.glob(str(runs / arm / f"{arm}_ep{episode}_s*"))
            if pathlib.Path(p).is_dir()]
    if len(hits) != 1:
        raise SystemExit(f"episode {episode}: expected one client dir, found {hits}")
    return hits[0]


# =================================================================================================
# export
# =================================================================================================

def build_model(opts: dict[str, Any]):
    import mujoco

    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, GRIPPER_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    link_filter = {"all": None, "arms": ARM_LINKS, "gripper": GRIPPER_LINKS}[opts["links"]]
    m = mujoco.MjModel.from_xml_path(str(pathlib.Path(opts["model_xml"]).resolve()))
    scene = TransportScene.attach(m, mujoco.MjData(m))
    cover = {"finger_cover": opts["finger_cover"]} if opts["finger_cover"] else {}
    robot = build_constraint_robot_model(scene, link_filter=link_filter,
                                         sphere_options=dict(opts["sphere_options"]), **cover)
    qadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i])
            for i in range(m.njnt)}
    return robot, qadr


def robot_spec(robot, opts: dict[str, Any], dt: float) -> dict[str, Any]:
    """JSON-able description of the served constraint model + the TO's effective joint limits."""
    from AG3S.trajopt.config import LimitsConfig
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout

    layout = ChunkLayout.rby1(robot.joint_names)
    limits = build_limits(robot, layout, dt=dt,
                          config=LimitsConfig(source="model_xml", model_xml=opts["model_xml"]))
    opt_names = [str(robot.joint_names[i]) for i in layout.q_indices]
    if opt_names != list(ARM_JOINTS_CHUNK_ORDER):
        raise SystemExit(f"layout order {opt_names} != chunk order {ARM_JOINTS_CHUNK_ORDER}")
    spheres = [{"index": i, "link": str(link), "center": [float(v) for v in centre],
                "radius": float(radius)}
               for i, (link, centre, radius) in enumerate(robot._local_spheres)]
    fingers = {}
    for name in robot.param_joint_names:
        j = robot.model.joint_by_name(name)
        fingers[name] = {"type": j.type, "axis": [float(v) for v in j.axis], "parent": j.parent,
                         "child": j.child, "lower": j.lower, "upper": j.upper}
    lin_cfg = LimitsConfig()
    from AG3S.ag3s.config import AG3SConfig

    ag3s_cfg = AG3SConfig.from_dict({"collision_backend": "esdf"})
    return {
        # out-of-grid answer of our field (`fixed_input_ablation.main`: unknown_policy free ->
        # +max_distance) — the evaluator side needs it
        "field": {"max_distance": float(ag3s_cfg.esdf.max_distance),
                  "unknown_policy": str(ag3s_cfg.esdf.unknown_policy),
                  "esdf_margin": opts["esdf_margin"]},
        "urdf_path": str(pathlib.Path(robot.model.path if hasattr(robot.model, "path") else
                                      "robots/rby1/description/models/rby1a/urdf/model.urdf")
                         .resolve()),
        "fk_root_link": "base",
        "curobo_base_link": CUROBO_BASE_LINK,
        "model_joint_names": [str(n) for n in robot.joint_names],
        "torso_joint_names": [n for n in robot.joint_names if str(n).startswith("torso_")],
        "arm_joint_names": list(ARM_JOINTS_CHUNK_ORDER),
        "chunk_arm_columns": [int(c) for c, _ in layout.pairs] if hasattr(layout, "pairs")
        else [*range(0, 7), *range(8, 15)],
        "gripper_columns": GRIPPER_COLUMNS,
        "finger_joints": fingers,
        "n_spheres": int(robot.n_spheres),
        "sphere_link_names": [str(n) for n in robot.sphere_link_names],
        "spheres": spheres,
        "limits": {
            "joint_names": opt_names,
            "lower": [float(v) for v in limits.lower],
            "upper": [float(v) for v in limits.upper],
            "velocity": [float(v) for v in np.asarray(limits.max_step) / dt],
            "acceleration": [float(v) for v in np.asarray(limits.max_step_change) / dt / dt],
            "dt": float(dt),
            "definition": ("TO effective limits (`build_limits`): position from model_xml with "
                           f"position_margin {lin_cfg.position_margin} rad; velocity / acceleration "
                           f"= URDF x {lin_cfg.velocity_scale} / x {lin_cfg.acceleration_scale}"),
        },
        "model_options": opts,
    }


def export_episode(robot, qadr, spec, episode: int, info: dict[str, Any], runs: pathlib.Path,
                   plan_horizon: int, out: pathlib.Path) -> dict[str, Any]:
    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap

    fmap = FingerJointMap(robot)
    edir = episode_dir(runs, "C1aN", episode)
    rows = [json.loads(line) for line in (edir / "frames.jsonl").read_text().splitlines()
            if line.strip()]
    plan = sorted((r for r in rows if r.get("kind") == "planning"), key=lambda r: int(r["seq"]))
    manifest = json.loads((edir / "manifest.json").read_text())
    manipulators = list((manifest.get("policy") or {}).get("safe_manipulators") or [])
    hand = manipulators[0] if manipulators else "left"
    rec_dir = pathlib.Path(info["server_dir"]) / "server_constraints" / "run_0000"
    files = sorted(rec_dir.glob("chunk_*.npz"))
    n = len(plan)
    block = files[info["order"] * n:(info["order"] + 1) * n]
    if len(block) != n:
        raise SystemExit(f"episode {episode}: {len(block)} chunk files for {n} planning rows")

    names = list(robot.sphere_link_names)
    fidx = {h: np.asarray([i for i, nm in enumerate(names) if nm in FINGER_LINKS[h]])
            for h in FINGER_LINKS}
    arm_cols = [*range(0, 7), *range(8, 15)]
    q_index = {nm: i for i, nm in enumerate(robot.joint_names)}
    arm_q = [q_index[nm] for nm in ARM_JOINTS_CHUNK_ORDER]
    default_params = robot.param_vector()

    acc: dict[str, list] = {k: [] for k in (
        "chunk_file", "chunk_index", "t_step", "seq", "q_now", "T_base_torso5", "finger_measured",
        "finger_path_plan", "finger_path_full", "centres_refined_plan", "record_centres_err_mm",
        "pairing_refined", "pairing_reference", "attached", "n_attached_points",
        "tcp_disp_plan_mm", "tcp_disp_full_mm", "to_view_clearance_m", "record_clearance_min_mm",
        "target_field_exclude", "status", "has_target", "to_status", "grasp_state",
        "grasp_active")}
    for row, f in zip(plan, block):
        with np.load(f, allow_pickle=False) as z:
            summary = json.loads(str(z["summary_json"]))
            reference = np.asarray(z["reference_chunk"], np.float64)
            refined = np.asarray(z["refined_chunk"], np.float64)
            rec_centres = np.asarray(z["sphere_centres"], np.float64)
            chunk_index = int(z["chunk_index"])
            t_step = int(z["t_step"])
        if t_step != int(row["seq"]):
            raise SystemExit(f"ep{episode} {f.name}: record t_step {t_step} != seq {row['seq']}")
        pr = float(np.abs(np.asarray(row["actions"], np.float64) - refined).max())
        pf = (float(np.abs(np.asarray(row["actions_reference"], np.float64) - reference).max())
              if "actions_reference" in row else float("nan"))
        q_now = np.asarray([row["qpos"][qadr[j]] for j in DEFAULT_RBY1_JOINTS], np.float64)

        fj = summary.get("finger_joints") or {}
        measured = dict(fj.get("opening_norm") or {})
        commanded = {h: reference[:, c] for h, c in GRIPPER_COLUMNS.items()}
        P_plan, _ = fmap.path(measured, commanded, plan_horizon, default=default_params)
        P_full, _ = fmap.path(measured, commanded, reference.shape[0], default=default_params)

        def centres(chunk, steps, P):
            out = np.empty((steps, robot.n_spheres, 3))
            for k in range(steps):
                q = q_now.copy()
                q[arm_q] = chunk[k, arm_cols]
                out[k] = robot.sphere_centers_numeric(q, params=P[:, k])[0]
            return out

        c_ref = centres(reference, plan_horizon, P_plan)
        c_fin = centres(refined, plan_horizon, P_plan)
        tcp_d = np.linalg.norm(c_fin[:, fidx[hand]].mean(1) - c_ref[:, fidx[hand]].mean(1), axis=1)
        c_ref_full = centres(reference, reference.shape[0], P_full)
        c_fin_full = centres(refined, refined.shape[0], P_full)
        tcp_full = np.linalg.norm(c_fin_full[:, fidx[hand]].mean(1)
                                  - c_ref_full[:, fidx[hand]].mean(1), axis=1)
        stats = summary.get("esdf_stats") or {}
        n_att = int(stats.get("n_attached_points", 0) or 0)
        to = summary.get("to") or {}
        mask = (summary.get("target_permission") or {}).get("target_field_exclude")

        acc["chunk_file"].append(str(f))
        acc["chunk_index"].append(chunk_index)
        acc["t_step"].append(t_step)
        acc["seq"].append(int(row["seq"]))
        acc["q_now"].append(q_now)
        acc["T_base_torso5"].append(robot.link_pose(q_now, CUROBO_BASE_LINK))
        acc["finger_measured"].append([float(measured.get("left", np.nan)),
                                       float(measured.get("right", np.nan))])
        acc["finger_path_plan"].append(P_plan)
        acc["finger_path_full"].append(P_full)
        acc["centres_refined_plan"].append(c_fin.astype(np.float32))
        acc["record_centres_err_mm"].append(float(np.abs(c_fin[0] - rec_centres).max() * 1e3))
        acc["pairing_refined"].append(pr)
        acc["pairing_reference"].append(pf)
        acc["attached"].append(n_att > 0)
        acc["n_attached_points"].append(n_att)
        acc["tcp_disp_plan_mm"].append(tcp_d * 1e3)
        acc["tcp_disp_full_mm"].append(tcp_full * 1e3)
        acc["to_view_clearance_m"].append(float((to.get("metrics") or {}).get("clearance_m",
                                                                               np.nan)))
        acc["record_clearance_min_mm"].append(float(summary.get("clearance_min_mm", np.nan)))
        acc["target_field_exclude"].append(np.asarray(mask if mask is not None
                                                      else [False] * robot.n_spheres, bool))
        acc["status"].append(str(summary.get("status")))
        acc["has_target"].append(bool(summary.get("has_target")))
        acc["to_status"].append(str(to.get("status")))
        acc["grasp_state"].append(str((summary.get("grasp") or {}).get("state")))
        acc["grasp_active"].append(bool((summary.get("exclusion") or {}).get("grasp_active")))

    arrays = {k: np.asarray(v) for k, v in acc.items()}
    arrays["hand"] = np.asarray(hand)
    arrays["episode"] = np.asarray(episode)
    arrays["client_dir"] = np.asarray(str(edir))
    path = out / f"episode_{episode}.npz"
    np.savez_compressed(path, **arrays)
    return {"episode": episode, "file": str(path), "client_dir": str(edir), "hand": hand,
            "server_dir": info["server_dir"], "order": info["order"],
            "chunk_index_range": [int(arrays["chunk_index"].min()),
                                  int(arrays["chunk_index"].max())],
            "n_chunks": int(n), "n_attached": int(arrays["attached"].sum()),
            "n_grasp_active": int(arrays["grasp_active"].sum()),
            "pairing_refined_max": float(arrays["pairing_refined"].max()),
            "pairing_reference_max": float(np.nanmax(arrays["pairing_reference"])),
            "record_centres_err_mm_max": float(arrays["record_centres_err_mm"].max()),
            # the cuRobo side holds one finger value per chunk (step 0 = measured); how far the
            # server's per-step envelope moved away from it inside the plan window:
            "finger_path_plan_drift_mm_max": float(max(
                np.abs(P - P[:, :1]).max() for P in arrays["finger_path_plan"]) * 1e3),
            "finger_path_plan_chunks_drift_over_0p5mm": int(sum(
                np.abs(P - P[:, :1]).max() > 5e-4 for P in arrays["finger_path_plan"])),
            "torso_ptp_rad": [float(v) for v in np.ptp(arrays["q_now"][:, :6], axis=0)]}


def export_contact_policy(inputs: pathlib.Path, out: pathlib.Path) -> dict[str, Any]:
    """T63 step 2 (B4) — the contact policy of every exported chunk, **by the snapshot's own
    `fixed_input_ablation.contact_policy`** (the function that rebuilds `to_adapter`'s
    `manipulated_link_margin` · `target_field_exclude` from `(phase, manipulators)`).

    Writes `contact_<E>.npz` (per chunk: `margin (n, S)` with NaN where the policy gives no target
    margin, `mask (n, S)` target-field-exclude rows, `has_target`, `has_free`, `attached`,
    `mask_equals_record`) and `contact_policy.json`. Reads the records; writes nothing next to them.
    """
    from types import SimpleNamespace

    from AG3S.trajopt.experiments.fixed_input_ablation import contact_policy

    manifest = json.loads((inputs / "export_manifest.json").read_text())
    spec = json.loads((inputs / "robot_spec.json").read_text())
    names = list(spec["sphere_link_names"])
    argv0 = next(iter(manifest["server_argv"].values()))
    opts = model_options_from_argv(argv0)
    policy_name = str(opts["target_field_policy"] or "relax").replace("-", "_")
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "tool": TOOL + " --contact-policy-out", "inputs": str(inputs),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "function": "snapshot AG3S/trajopt/experiments/fixed_input_ablation.py:contact_policy",
        "target_field_policy": policy_name,
        "server_flags_not_mirrored_by_fixed_input_ablation": {
            k: (k in argv0) for k in ("--obstacle-margin", "--target-permission-strict",
                                      "--lookahead")},
        "episodes": []}
    for ep in manifest["episodes"]:
        e = int(ep["episode"])
        z = np.load(ep["file"], allow_pickle=False)
        pol = json.loads((pathlib.Path(ep["client_dir"]) / "manifest.json").read_text())["policy"]
        settings = SimpleNamespace(phase=str(pol["safe_phase"]),
                                   manipulators=[str(m) for m in pol["safe_manipulators"]],
                                   target_field_policy=policy_name)
        margins, masks, has_t, has_f, att, eq = [], [], [], [], [], []
        authorized = None
        for i, f in enumerate(z["chunk_file"]):
            with np.load(str(f), allow_pickle=False) as rz:
                summary_json = json.loads(str(rz["summary_json"]))
                has_free = "esdf_free_distance" in rz.files
                has_points = "target_points" in rz.files and len(rz["target_points"]) > 0
            has_target = bool(summary_json.get("has_target")) and has_points
            attached = bool(z["attached"][i])
            margin, mask, authorized, _, _ = contact_policy(
                settings, names, has_target=has_target, has_free=has_free, attached=attached)
            margins.append(np.full(len(names), np.nan) if margin is None
                           else np.asarray(margin, np.float64))
            masks.append(np.zeros(len(names), bool) if mask is None else np.asarray(mask, bool))
            has_t.append(has_target)
            has_f.append(has_free)
            att.append(attached)
            eq.append(bool(np.array_equal(masks[-1], np.asarray(z["target_field_exclude"][i]))))
        arrays = {"margin": np.asarray(margins), "mask": np.asarray(masks),
                  "has_target": np.asarray(has_t), "has_free": np.asarray(has_f),
                  "attached": np.asarray(att), "mask_equals_record": np.asarray(eq),
                  "chunk_index": np.asarray(z["chunk_index"])}
        np.savez_compressed(out / f"contact_{e}.npz", **arrays)
        summary["episodes"].append({
            "episode": e, "phase": settings.phase, "manipulators": settings.manipulators,
            "authorized_links": authorized, "n_chunks": len(eq),
            "n_with_target_margin": int(np.isfinite(arrays["margin"]).any(1).sum()),
            "n_with_mask": int(arrays["mask"].any(1).sum()),
            "n_mask_differs_from_record": int((~arrays["mask_equals_record"]).sum()),
            "margin_by_link_mm": ({n: float(np.nanmax(arrays["margin"][:, names.index(n)]) * 1e3)
                                   for n in sorted(set(names))}
                                  if np.isfinite(arrays["margin"]).any() else None)})
        print(f"[t63-contact] ep{e}: {summary['episodes'][-1]}", flush=True)
    (out / "contact_policy.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="baseline snapshot tree (f0d8046)")
    ap.add_argument("--ifg", default=None, help="verify/IFG-APPLE dir of the baseline run")
    ap.add_argument("--episodes", type=int, nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=1.0 / 15.0, help="control period (s)")
    ap.add_argument("--contact-policy-from", default=None,
                    help="T63 step 2: an existing export dir; write only the contact policy "
                         "(`export_contact_policy`) to --out")
    args = ap.parse_args(argv)

    import AG3S

    snap = pathlib.Path(args.snapshot).resolve()
    imported = pathlib.Path(AG3S.__file__).resolve()
    if snap not in imported.parents:
        raise SystemExit(f"AG3S imported from {imported}, not from the snapshot {snap} — "
                         "put the snapshot first on PYTHONPATH")
    if args.contact_policy_from:
        export_contact_policy(pathlib.Path(args.contact_policy_from).resolve(),
                              pathlib.Path(args.out))
        return 0
    if not args.ifg or not args.episodes:
        raise SystemExit("--ifg and --episodes are required for the full export")
    runs = pathlib.Path(args.ifg).resolve() / "runs"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    servers = episode_servers(runs)
    missing = [e for e in args.episodes if e not in servers]
    if missing:
        raise SystemExit(f"episodes {missing} not found in the C1aN chain logs")
    argvs = {}
    for e in args.episodes:
        sd = pathlib.Path(servers[e]["server_dir"])
        argvs[str(sd)] = read_server_argv(sd)
    opts_by_server = {sd: model_options_from_argv(a) for sd, a in argvs.items()}
    keyset = {json.dumps(o, sort_keys=True) for o in opts_by_server.values()}
    if len(keyset) != 1:
        raise SystemExit(f"servers disagree on the model options: {opts_by_server}")
    opts = next(iter(opts_by_server.values()))
    plan_horizon = int(opts["plan_horizon"] or 16)

    robot, qadr = build_model(opts)
    spec = robot_spec(robot, opts, args.dt)
    # check: the rebuilt model is the recorded one (count, order of links, radii)
    first = servers[args.episodes[0]]
    rec0 = sorted((pathlib.Path(first["server_dir"]) / "server_constraints" / "run_0000")
                  .glob("chunk_*.npz"))[0]
    with np.load(rec0) as z:
        radii_rec = np.asarray(z["sphere_radii"], np.float64)
        names_rec = json.loads(str(z["summary_json"])).get("sphere_link_names")
    spec["checks"] = {
        "record_file": str(rec0),
        "n_spheres": int(robot.n_spheres), "n_spheres_record": int(radii_rec.size),
        "radii_max_abs_diff_m": float(np.abs(robot.radii - radii_rec).max()),
        "link_names_equal_record": names_rec == list(robot.sphere_link_names),
        "sphere_fit_used": False,
        "note": "spheres are the server's own UrdfSphereChain spheres (link frame), not a cuRobo fit",
    }
    if robot.n_spheres != radii_rec.size or spec["checks"]["radii_max_abs_diff_m"] > 1e-6 \
            or not spec["checks"]["link_names_equal_record"]:
        raise SystemExit(f"rebuilt model differs from the record: {spec['checks']}")
    (out / "robot_spec.json").write_text(json.dumps(spec, indent=1))

    episodes = []
    for e in args.episodes:
        rec = export_episode(robot, qadr, spec, e, servers[e], runs, plan_horizon, out)
        episodes.append(rec)
        print(f"[t63-export] ep{e}: {rec['n_chunks']} chunks, attached {rec['n_attached']}, "
              f"pairing {rec['pairing_refined_max']:.2e}, baseline FK vs record "
              f"{rec['record_centres_err_mm_max']:.4f} mm", flush=True)
    manifest = {
        "tool": TOOL, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv, "snapshot": str(snap), "ag3s_imported_from": str(imported),
        "ifg": str(pathlib.Path(args.ifg).resolve()), "plan_horizon": plan_horizon,
        "dt": args.dt, "server_argv": argvs, "model_options": opts, "episodes": episodes,
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[t63-export] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
