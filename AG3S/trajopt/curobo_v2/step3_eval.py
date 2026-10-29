"""T63 step 3 — the common evaluator, and our TO re-solved on the same inputs.

Runs against the **baseline snapshot tree** (`f0d8046`, the C1aN serving code) exactly like
`export_inputs.py`: as a *script*, with the snapshot first on `PYTHONPATH`, so every `import AG3S...`
resolves into the snapshot. The two pure-numpy helpers of this directory (`step3_metrics.py`,
`record_context.py`) and the model builder of `export_inputs.py` are loaded **by file path**.

    SNAP=$AG3S_STORAGE_ROOT/results/development/20261019T132719+0900_f15e587290f6_11408fd6d27f/snapshot
    cd "$SNAP" && MUJOCO_GL=osmesa PYTHONPATH="$SNAP" $AG3S_PROJECT_ROOT/.venv-ag3s/bin/python \\
        $AG3S_PROJECT_ROOT/AG3S/trajopt/curobo_v2/step3_eval.py --snapshot "$SNAP" \\
        --inputs $AG3S_RUN_DIR/impl/T63/inputs --curobo <t63_curobo_compare --stage step3 --out> \\
        --out <dir> [--episodes 211 126] [--chunks 0,20,40,60 | all] [--resolve fresh|chained|off]
    # after per-episode runs in parallel (each with its own --episodes): merge
    # (parallel: one process per episode set with --rows-only, then)
    ... step3_eval.py --snapshot "$SNAP" --inputs ... --out <dir> --aggregate-only

## One evaluator for every candidate (`evaluate_trajectory`)

| quantity | how | code |
|---|---|---|
| clearance `(50, S)` | `CollisionLinearizer.esdf_clearance` of **our TO** (horizon 50, no look-ahead drop) on the scene our TO certified against: `scene_from_constraint_set` of the snapshot with the server's `TrajOptConfig` (`serve_safe.trajopt_config_from_args` on the recorded argv: `esdf_margin 0`, `obstacle_margin 0.010`, `target_permission_strict`), the record's field layers, the **record's** `target_field_exclude` mask, the contact margin of `fixed_input_ablation.contact_policy`, the target points, and the support planes re-fitted by AG3S's own seeded `fit_support_surfaces` on the recorded cloud (the inlier mask is checked against the record's `support_mask`) | `build_scene` |
| finger opening | per step: the server's envelope (`export_inputs` `finger_path_full`, `FingerJointMap.path`) | `set_joint_parameter_path` |
| TCP `(50, 3)` | mean of the manipulating hand's finger spheres | `evaluate_trajectory` |
| metrics | `step3_metrics.candidate_metrics` (one function, every candidate) | |
| **physical clearance** `(50, S)` (T63b user decision 2) | the same `esdf_clearance` call on `physical_scene(scene)`: every margin 0 (`obstacle_margin`, the contact margin `manipulated_link_margin`, `esdf_margin`, destination margin), the contact policy unchanged — rows allowed to touch the target still ask the target-free layer (T30 F1). = `d − r` per row. The certified clearance above includes the 10 mm obstacle margin, so its `< 0` is a **certified-clearance violation**, not a physical penetration | `physical_scene`, `evaluate_trajectory` |

## Re-evaluating saved trajectories (CPU, no cuRobo, no TO re-solve)

`--curobo` takes one or more `t63_curobo_compare --stage step3` output dirs (candidate names must be
unique). `--resolve saved:<earlier eval dir>` reads `to_resolved` from that dir
(`trajectories/ep<E>.npz` `to_resolved_chunk`, solve info from `rows_ep<E>.json`) instead of
re-solving, so an earlier run's 16 candidates are re-measured with the metrics added since
(e.g. the physical clearance) from what was saved.

## B0m rule export (`--export-b0m-rules DIR`, T63b)

Per episode `b0m_rules_<E>.npz`: the inputs of our TO's obstacle-margin classification
(`b0m.rules_from_scene` on the scene `build_scene` rebuilds — contact rows, support planes, margin,
tolerances, has-target) for every chunk, plus, for the parity check, our TO's view of the recorded
`refined_chunk`'s plan window: per row clearance, class (`_last_obstacle_class`), the distance `d`
the row reads (target-free layer on mask rows) and the margin `d − r − clearance`.

**Fidelity check (per chunk, before anything is compared):** the evaluator's min over the 8 executed
steps of the recorded `refined_chunk` must equal the recorded `summary.to.metrics.clearance_m` (the
TO's certified window clearance). Reported as `fidelity`; a chunk over `--fidelity-tol-mm` is
flagged (`fidelity.within_tol = false`), not dropped.

## Candidates

| name | trajectory |
|---|---|
| `reference` | `reference_chunk` (VLA original) |
| `to_recorded` | `refined_chunk` (our TO, served) — solve time = the record's `to.solve_ms` (server CPU, live) |
| `to_resolved` | our TO re-solved here (`TrajOptChunkRefiner` of the snapshot, server config with the wall-clock budget off — the record shows no budget hit — server context rebuilt from the record: continuity (`exec_continuity`), deflection history (`record_context`), finger path, `geometry_certified`). `--resolve fresh` (default) = a fresh solver per chunk (the `fixed_input_ablation` replay); `chained` = one solver per episode without reset (the server carries its OSQP warm start across chunks; needs `--chunks all`). Its error against `to_recorded` is reported |
| cuRobo `B0`, `B1_lam*`…`B4_lam*` | `t63_curobo_compare --stage step3` output (`--curobo`) |
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import pathlib
import sys
import time
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

TOOL = "AG3S/trajopt/curobo_v2/step3_eval.py"
HERE = pathlib.Path(__file__).resolve().parent
FIDELITY_TOL_MM = 0.01


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"t63_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("step3_metrics")
RC = _load("record_context")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None if np.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


def _parse_dict(v) -> dict:
    if isinstance(v, dict):
        return v
    if v is None:
        return {}
    import ast

    try:
        out = ast.literal_eval(str(v))
        return out if isinstance(out, dict) else {}
    except (ValueError, SyntaxError):
        return {}


# =================================================================================================
# shared setup (snapshot side)
# =================================================================================================

class Shared:
    """Model, configs, linearizers and refiner — built once per process."""

    def __init__(self, inputs: pathlib.Path, resolve: str):
        ei = _load("export_inputs")
        import AG3S.trajopt.serve_safe as ss
        from AG3S.ag3s.config import AG3SConfig
        from AG3S.trajopt.linearize import CollisionLinearizer
        from AG3S.trajopt.types import ChunkLayout

        self.manifest = json.loads((inputs / "export_manifest.json").read_text())
        self.spec = json.loads((inputs / "robot_spec.json").read_text())
        argvs = list(self.manifest["server_argv"].values())
        configs = []
        for argv in argvs:
            a = ss.build_parser().parse_args(argv)
            configs.append(ss.trajopt_config_from_args(a))
        keys = {json.dumps(c.to_dict(), sort_keys=True, default=str) for c in configs}
        if len(keys) != 1:
            raise SystemExit("the baseline servers disagree on the TO config")
        self.server_cfg = configs[0]
        # the wall-clock budget is this machine's, not the server's; no record hit it
        self.resolve_cfg = self.server_cfg.with_overrides({"sqp": {"time_budget_ms": 1e9}})
        self.opts = ei.model_options_from_argv(argvs[0])
        self.target_field_policy = str(self.opts["target_field_policy"] or "relax").replace("-", "_")
        t0 = time.perf_counter()
        self.robot, _ = ei.build_model(self.opts)
        self.layout = ChunkLayout.rby1(self.robot.joint_names)
        self.lin = CollisionLinearizer(self.robot, self.layout, 50)
        self.build_s = time.perf_counter() - t0
        self.names = [str(n) for n in self.robot.sphere_link_names]
        if self.names != list(self.spec["sphere_link_names"]):
            raise SystemExit("rebuilt sphere model differs from robot_spec.json")
        self.ag3s_cfg = AG3SConfig.from_dict({"collision_backend": "esdf"})
        self.outside = (float(self.ag3s_cfg.esdf.max_distance)
                        if self.ag3s_cfg.esdf.unknown_policy == "free"
                        else -float(self.ag3s_cfg.esdf.max_distance))
        self.lim = M.step_limits(self.spec["limits"])
        self.resolve = resolve
        self.refiner = None
        self.saved = None
        self._scene_holder: dict[str, Any] = {}
        if resolve.startswith("saved:"):
            self.saved = SavedResolve(pathlib.Path(resolve.split(":", 1)[1]))
        elif resolve != "off":
            from AG3S.trajopt.refiner import TrajOptChunkRefiner

            self.refiner = TrajOptChunkRefiner(self.robot, self.layout,
                                               lambda _ctx: self._scene_holder["value"],
                                               self.resolve_cfg)


def build_scene(sh: Shared, z, summary: dict, mask: np.ndarray, settings) -> tuple[Any, dict]:
    """The SceneSnapshot our TO certified against, rebuilt by the snapshot's own
    `scene_from_constraint_set` (see module doc)."""
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.types import PointCloud
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field, contact_policy
    from AG3S.trajopt.linearize import scene_from_constraint_set

    field = build_field(z, outside_distance=sh.outside)
    method = (summary.get("target_exclusion") or {}).get("exclusion_method")
    field.stats = {"target_free": {"exclusion_method": method}}
    tp = np.asarray(z["target_points"], np.float64) if "target_points" in z.files else None
    has_target = bool(summary.get("has_target")) and tp is not None and len(tp) > 0
    attached = int((summary.get("esdf_stats") or {}).get("n_attached_points", 0) or 0) > 0
    if attached:
        raise ValueError("attached query points are not recorded (fixed_input_ablation skips)")
    margin, _, authorized, _, _ = contact_policy(
        settings, sh.names, has_target=has_target, has_free="esdf_free_distance" in z.files,
        attached=False)
    cloud = PointCloud(points=np.asarray(z["filtered_cloud"], np.float64))
    surfaces, smask = fit_support_surfaces(cloud, sh.ag3s_cfg.support_surface)
    rec_smask = np.asarray(z["support_mask"], bool) if "support_mask" in z.files else None
    cs = SimpleNamespace(
        constraints=None, esdf=field,
        manipulated=SimpleNamespace(points=tp, sphere_centers=None, sphere_radii=None)
        if has_target else None,
        manipulated_link_margin=(None if margin is None or not has_target
                                 else np.asarray(margin, np.float64)),
        attached=None, destination_label=None, destination_margin=None,
        target_field_exclude=mask if mask.any() else None,
        target_field_policy=sh.target_field_policy, support_surfaces=surfaces,
        nonauthorized_target_rows=None, metrics={})
    radii = np.asarray(z["sphere_radii"], np.float64)
    scene = scene_from_constraint_set(cs, radii, sh.server_cfg)
    info = {"n_support_planes": len(surfaces),
            "support_mask_equals_record": (None if rec_smask is None
                                           else bool(np.array_equal(rec_smask, smask))),
            "has_target": has_target, "n_mask_rows": int(mask.sum()),
            "authorized_links": authorized, "exclusion_method": method,
            "contact_margin_links_mm": (None if margin is None else
                                        {n: float(np.asarray(margin)[sh.names.index(n)] * 1e3)
                                         for n in sorted(set(sh.names))})}
    return scene, info


def physical_scene(scene):
    """T63b user decision 2 — `scene` with every margin 0 and the contact policy unchanged.

    `obstacle_margin` 0 (no class rule runs), `manipulated_link_margin` None (no contact margin on
    target rows), `esdf_margin` 0 (already 0 on the server), destination margin 0 (held rows only;
    no held rows in the evaluator). Untouched: `target_free_mask` / `target_free_exempt` (rows allowed
    to touch the target read the target-free layer), `target_contact_rows`, the field. So the
    clearance is `d − r` with `d` exactly the row's field answer of the certified scene."""
    import dataclasses

    return dataclasses.replace(scene, esdf_margin=0.0, obstacle_margin=0.0,
                               manipulated_link_margin=None, destination_margin=0.0)


def evaluate_trajectory(sh: Shared, traj14: np.ndarray, q_now20: np.ndarray, scene, fidx,
                        scene_phys=None) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """**The** evaluator: `(50, 14)` arm joints -> `(clearance (50, S) m, tcp (50, 3) m,
    physical clearance (50, S) m or None)`. The finger path of the chunk must already be set on
    `sh.lin`. The physical clearance is the same call on `scene_phys` (`physical_scene`)."""
    traj = np.asarray(traj14, np.float64).T
    states = sh.lin.sphere_states(traj, q_now20)
    clearance = np.asarray(sh.lin.esdf_clearance(traj, q_now20, scene, states)[..., 0], np.float64)
    phys = (None if scene_phys is None else
            np.asarray(sh.lin.esdf_clearance(traj, q_now20, scene_phys, states)[..., 0], np.float64))
    tcp = np.asarray(states[0], np.float64)[:, fidx].mean(axis=1)
    return clearance, tcp, phys


def row_distance(sh: Shared, traj14: np.ndarray, q_now20: np.ndarray, scene) -> np.ndarray:
    """`(T, S)` the distance `d` each robot row reads in `_esdf_clearance_core` (main field; the
    target-free layer on `target_free_mask` rows) at the sphere centres."""
    traj = np.asarray(traj14, np.float64).T
    centres = np.asarray(sh.lin.sphere_states(traj, q_now20)[0], np.float64)
    S = sh.lin.n_spheres
    flat = centres[:, :S].reshape(-1, 3)
    d = np.asarray(scene.esdf.distance(flat), np.float64).reshape(centres.shape[0], S)
    mask = getattr(scene, "target_free_mask", None)
    if mask is not None and np.asarray(mask, bool).any():
        df = np.asarray(scene.esdf.target_free_distance(flat), np.float64).reshape(d.shape)
        d = np.where(np.asarray(mask, bool)[None, :S], df, d)
    return d


def resolve_context(path: pathlib.Path, summary: dict, k: int) -> tuple[dict, dict]:
    """The `refine` context the server built for this chunk (`SafePolicy.infer`), from the record."""
    ec = summary.get("exec_continuity") or {}
    ctx: dict[str, Any] = {"t_step": int(summary.get("t_step", 0))}
    note = {"continuity": None, "deflection": None}
    prev = None
    prev_path = RC.previous_record_path(path, k)
    if ec.get("off_by"):
        ctx["continuity_off"] = str(ec["off_by"])
        note["continuity"] = f"off_by {ec['off_by']}"
    elif ec.get("mode") == "feedback" and ec.get("previous_chunk") in ("refined", "reference") \
            and prev_path.exists():
        with np.load(prev_path, allow_pickle=False) as pz:
            if int(pz["t_step"]) == int(ec.get("plan_seq")):
                prev = np.asarray(pz[f"{ec['previous_chunk']}_chunk"], np.float64)
        if prev is not None:
            ctx["previous_physical_chunk"] = prev
            ctx["previous_executed_steps"] = int(ec.get("n_exec"))
            note["continuity"] = f"{ec['previous_chunk']} of {prev_path.name}, n_exec {ec.get('n_exec')}"
    if prev is None and "continuity_off" not in ctx:
        ctx["previous_physical_chunk"] = None
        note["continuity"] = note["continuity"] or f"none ({ec.get('source')})"
    rc = RC.chunk_record_context(path)
    if rc["deflection_history"] is not None:
        ctx["deflection_history"] = np.asarray(rc["deflection_history"], np.float64)
    note["deflection"] = rc["notes"].get("deflection")
    note["continuity_used_by_server"] = bool(ec.get("used"))
    return ctx, note


class SavedResolve:
    """`--resolve saved:<eval dir>`: `to_resolved` as an earlier evaluator run saved it (trajectory
    `trajectories/ep<E>.npz` `to_resolved_chunk`; solve info and `to_resolve` record from
    `rows_ep<E>.json`) — re-measured, not re-solved."""

    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root).resolve()
        if not (self.root / "trajectories").is_dir():
            raise SystemExit(f"--resolve saved: {self.root}/trajectories does not exist")
        self._ep: dict[int, Any] = {}

    def get(self, e: int, k: int) -> tuple[Optional[np.ndarray], Optional[dict], Optional[dict]]:
        if e not in self._ep:
            t = np.load(self.root / "trajectories" / f"ep{e}.npz")
            rows = json.loads((self.root / f"rows_ep{e}.json").read_text())["rows"]
            self._ep[e] = ({int(c): i for i, c in enumerate(np.asarray(t["chunk_index"]))},
                           np.asarray(t["to_resolved_chunk"], np.float64),
                           {int(r["chunk_index"]): r for r in rows})
        idx, chunks, rows = self._ep[e]
        row = rows.get(int(k))
        i = idx.get(int(k))
        q = None if i is None else chunks[i]
        if q is not None and not np.all(np.isfinite(q)):
            q = None
        solve = None if row is None else (row["candidates"].get("to_resolved") or {}).get("solve")
        info = None if row is None else row.get("to_resolve")
        return q, solve, info


def run_resolve(sh: Shared, chunk: np.ndarray, ctx: dict, scene, q_now20, certified: bool,
                P_plan: np.ndarray, reset: bool) -> dict[str, Any]:
    ref = sh.refiner
    if reset:
        ref.reset()
    sh._scene_holder["value"] = (scene, q_now20, certified)
    ref.optimizer.linearizer.set_joint_parameter_path(P_plan)
    t0 = time.perf_counter()
    out = ref.refine(np.asarray(chunk, np.float64), dict(ctx))
    wall = (time.perf_counter() - t0) * 1e3
    res = ref.last_result
    if res is None:
        return {"chunk": None, "status": "no_result", "wall_ms": wall,
                "failure": ref.last_failure}
    m = res.metrics
    return {"chunk": np.asarray(out, np.float64), "status": res.status.value, "wall_ms": wall,
            "solve_ms": float(res.solve_time_ms), "sqp_iterations": m.get("sqp_iterations"),
            "returned": m.get("returned"), "clearance_m": m.get("clearance_m"),
            "lookahead_gate": (m.get("lookahead") or {}).get("gate"),
            "continuity_active": m.get("continuity_active")}


# =================================================================================================
# cuRobo candidates
# =================================================================================================

class CuroboOutputs:
    """`t63_curobo_compare --stage step3` outputs: one dir, or several (T63b: e.g. the T63 13
    candidates + a B0m run) — candidate names must be unique across them."""

    def __init__(self, root):
        roots = ([] if root is None else list(root) if isinstance(root, (list, tuple))
                 else [root])
        self.root = roots[0] if len(roots) == 1 else (None if not roots else roots)
        self.roots = [pathlib.Path(r) for r in roots]
        self.rows: dict[tuple, dict] = {}
        self.names: list[str] = []
        self.meta: dict[str, Any] = {}
        self._traj: dict[tuple, Any] = {}
        self._root_of: dict[str, pathlib.Path] = {}
        metas = []
        for r_dir in self.roots:
            mine: list[str] = []
            rows_path = r_dir / "curobo_rows.jsonl"
            for line in rows_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[(r["candidate"], int(r["episode"]), int(r["chunk_index"]))] = r
                    if r["candidate"] not in mine:
                        mine.append(r["candidate"])
            for c in mine:
                if c in self._root_of:
                    raise SystemExit(f"candidate {c} is in both {self._root_of[c]} and {r_dir}")
                self._root_of[c] = r_dir
                self.names.append(c)
            meta_path = r_dir / "curobo_step3.json"
            if meta_path.exists():
                m = json.loads(meta_path.read_text())
                metas.append({"candidates": {k: {kk: v.get(kk) for kk in (
                    "variant", "lambda", "world", "terms", "weights_used", "cold_solve", "init_s")}
                    for k, v in m.get("candidates", {}).items()},
                    "elapsed_per_chunk_s": m.get("elapsed_per_chunk_s"),
                    "n_chunks": m.get("n_chunks"), "b4_mask": m.get("b4_mask"),
                    "lambdas": m.get("lambdas"), "file": str(meta_path)})
        if len(metas) == 1:
            self.meta = metas[0]
        elif metas:
            self.meta = {"candidates": {k: v for m in metas for k, v in m["candidates"].items()},
                         "parts": metas}

    def trajectory(self, cand: str, ep: int, chunk_index: int) -> Optional[np.ndarray]:
        key = (cand, ep)
        if key not in self._traj:
            root = self._root_of.get(cand)
            f = (pathlib.Path("/nonexistent") if root is None
                 else root / "trajectories" / cand / f"ep{ep}.npz")
            self._traj[key] = dict(np.load(f)) if f.exists() else None
        t = self._traj[key]
        if t is None:
            return None
        hit = np.flatnonzero(np.asarray(t["chunk_index"]) == chunk_index)
        return None if not hit.size else np.asarray(t["cspace"][hit[0]], np.float64)


# =================================================================================================
# per episode
# =================================================================================================

def arm_cols(spec) -> list[int]:
    return [int(c) for c in spec["chunk_arm_columns"]]


def run_episode(sh: Shared, inputs: pathlib.Path, e: int, positions, cu: CuroboOutputs,
                tol_mm: float) -> tuple[list[dict], dict[str, np.ndarray]]:
    E = dict(np.load(inputs / f"episode_{e}.npz", allow_pickle=False))
    n = len(E["chunk_file"])
    idx = list(range(n)) if positions == "all" else [p for p in positions if p < n]
    if sh.resolve == "chained" and idx != list(range(n)):
        raise SystemExit("--resolve chained needs --chunks all (the server's solver chain)")
    pol = json.loads((pathlib.Path(str(E["client_dir"])) / "manifest.json").read_text())["policy"]
    settings = SimpleNamespace(phase=str(pol["safe_phase"]),
                               manipulators=[str(m) for m in pol["safe_manipulators"]],
                               target_field_policy=sh.target_field_policy)
    hand = str(E["hand"])
    fl = {"left": ("ee_finger_l1", "ee_finger_l2"), "right": ("ee_finger_r1", "ee_finger_r2")}
    fidx = np.asarray([k for k, nm in enumerate(sh.names) if nm in fl[hand]])
    cols = arm_cols(sh.spec)
    rows: list[dict] = []
    traj_out: dict[str, list] = {"chunk_index": [], "position": [], "to_resolved_chunk": []}
    for pos_n, i in enumerate(idx):
        path = pathlib.Path(str(E["chunk_file"][i]))
        z = np.load(path, allow_pickle=False)
        summary = json.loads(str(z["summary_json"]))
        k = int(z["chunk_index"])
        q_now20 = np.asarray(E["q_now"][i], np.float64)
        q_now14 = q_now20[[sh.robot.joint_names.index(nm) for nm in sh.spec["arm_joint_names"]]]
        mask = np.asarray(E["target_field_exclude"][i], bool)
        scene, scene_info = build_scene(sh, z, summary, mask, settings)
        scene_phys = physical_scene(scene)
        sh.lin.set_joint_parameter_path(np.asarray(E["finger_path_full"][i], np.float64))
        reference = np.asarray(z["reference_chunk"], np.float64)
        refined = np.asarray(z["refined_chunk"], np.float64)
        ref_q = reference[:, cols]
        ref_cl, ref_tcp, ref_ph = evaluate_trajectory(sh, ref_q, q_now20, scene, fidx, scene_phys)
        to = summary.get("to") or {}
        tom = to.get("metrics") or {}
        cands: dict[str, dict[str, Any]] = {}

        def add(name, q, solve, cl_tcp=None):
            if q is None or not np.all(np.isfinite(q)):
                cl, tcp, ph = None, None, None
            else:
                cl, tcp, ph = cl_tcp if cl_tcp is not None else evaluate_trajectory(
                    sh, q, q_now20, scene, fidx, scene_phys)
            met = M.candidate_metrics(q, q_now14, cl, tcp, ref_q, ref_tcp, sh.lim,
                                      clearance_phys_m=ph)
            cands[name] = {"metrics": met, "solve": solve}
            traj_out.setdefault(f"minclr_{name}", []).append(
                np.full(50, np.nan) if cl is None else cl.min(axis=1))
            traj_out.setdefault(f"minclr_phys_{name}", []).append(
                np.full(50, np.nan) if ph is None else ph.min(axis=1))
            traj_out.setdefault(f"tcp_{name}", []).append(
                np.full((50, 3), np.nan) if tcp is None else tcp)
            return cl, tcp

        add("reference", ref_q, None, (ref_cl, ref_tcp, ref_ph))
        rec_status = str(to.get("status"))
        rec_cl, rec_tcp = add("to_recorded", refined[:, cols], {
            "status": rec_status, "success": rec_status not in ("violated", "solver_failed"),
            "wall_ms": to.get("solve_ms"), "cold": False,
            "device": "cpu (server, live; record to.solve_ms)",
            "sqp_iterations": to.get("sqp_iterations")})
        # fidelity: the evaluator reproduces what our TO certified (8 executed steps)
        mine = float(rec_cl[:8].min())
        recd = tom.get("clearance_m")
        fid = {"evaluator_min_exec8_mm": mine * 1e3,
               "record_to_clearance_mm": None if recd is None else float(recd) * 1e3,
               "abs_diff_mm": None if recd is None else abs(mine - float(recd)) * 1e3}
        fid["within_tol"] = fid["abs_diff_mm"] is not None and fid["abs_diff_mm"] <= tol_mm
        tcp_rec = np.linalg.norm(rec_tcp - ref_tcp, axis=1) * 1e3
        to_active_mm = float(tcp_rec[:16].max())

        # our TO re-solved
        res_info = None
        if sh.refiner is not None:
            ctx, ctx_note = resolve_context(path, summary, k)
            certified = bool(tom.get("geometry_certified", summary.get("status") == "ok"))
            reset = sh.resolve == "fresh" or pos_n == 0
            r = run_resolve(sh, reference, ctx, scene, q_now20, certified,
                            np.asarray(E["finger_path_plan"][i], np.float64), reset)
            # cold = the first solve of this process (CasADi / OSQP set-up)
            first = not getattr(sh, "_resolved_once", False)
            sh._resolved_once = True
            rq = None if r["chunk"] is None else r["chunk"][:, cols]
            st = r["status"]
            add("to_resolved", rq, {"status": st, "success": st not in ("violated", "solver_failed",
                                                                        "no_result"),
                                    "wall_ms": r["wall_ms"], "internal_solve_ms": r.get("solve_ms"),
                                    "cold": first, "device": "cpu (offline, this machine)",
                                    "sqp_iterations": r.get("sqp_iterations")})
            traj_out["to_resolved_chunk"].append(np.full((50, 16), np.nan) if r["chunk"] is None
                                                 else r["chunk"])
            res_info = {"status": st, "recorded_status": rec_status,
                        "same_status": st == rec_status,
                        "sqp_iterations": r.get("sqp_iterations"),
                        "recorded_sqp_iterations": to.get("sqp_iterations"),
                        "returned": r.get("returned"), "recorded_returned": tom.get("returned"),
                        "lookahead_gate": r.get("lookahead_gate"),
                        "recorded_lookahead_gate": _parse_dict(tom.get("lookahead")).get("gate"),
                        "context": ctx_note}
            if rq is not None:
                d = np.abs(rq - refined[:, cols])
                tr = cands["to_resolved"]
                res_info.update({
                    "max_joint_diff_vs_recorded_rad_plan16": float(d[:16].max()),
                    "max_joint_diff_vs_recorded_rad_all50": float(d.max()),
                    "tcp_diff_vs_recorded_mm_max_plan16": float(np.nanmax(np.linalg.norm(
                        traj_out["tcp_to_resolved"][-1][:16] - rec_tcp[:16], axis=1)) * 1e3),
                    "min_clearance_exec8_diff_mm": (tr["metrics"]["min_clearance_mm_exec8"]
                                                    - cands["to_recorded"]["metrics"]
                                                    ["min_clearance_mm_exec8"]),
                    "resolved_reported_clearance_mm": (None if r.get("clearance_m") is None
                                                       else float(r["clearance_m"]) * 1e3)})
        elif sh.saved is not None:
            sq, ssolve, sinfo = sh.saved.get(e, k)
            add("to_resolved", None if sq is None else sq[:, cols], ssolve)
            traj_out["to_resolved_chunk"].append(np.full((50, 16), np.nan) if sq is None else sq)
            res_info = sinfo
        else:
            traj_out["to_resolved_chunk"].append(np.full((50, 16), np.nan))

        # cuRobo candidates
        for name in cu.names:
            q = cu.trajectory(name, e, k)
            crow = cu.rows.get((name, e, k))
            solve = None
            if crow is not None:
                solve = {"status": "success" if crow["success"] else
                         ("fail:" + ",".join(sorted(crow.get("failing_constraints") or {}))),
                         "success": bool(crow["success"]), "wall_ms": crow["wall_ms"],
                         "gpu_solve_ms": crow.get("gpu_solve_ms"), "cold": False,
                         "device": "gpu", "world": crow.get("world"),
                         "failing_constraints": crow.get("failing_constraints")}
            add(name, None if crow is None else q, solve)

        rows.append({
            "episode": e, "chunk_index": k, "seq": int(E["seq"][i]), "position": int(i),
            "file": str(path), "box_layer": "free" if "esdf_free_distance" in z.files else "main_fine",
            "grasp_active": bool(E["grasp_active"][i]), "grasp_state": str(E["grasp_state"][i]),
            "to_active": bool(to_active_mm > M.TO_ACTIVE_TCP_MM),
            "to_active_tcp_max_mm_plan16": to_active_mm,
            "export_tcp_disp_plan_max_mm": float(np.max(E["tcp_disp_plan_mm"][i])),
            "record_to_status": rec_status, "scene": scene_info, "fidelity": fid,
            "to_resolve": res_info, "candidates": cands})
        traj_out["chunk_index"].append(k)
        traj_out["position"].append(int(i))
        z.close()
        print(f"[t63-step3-eval] ep{e} chunk {k}: fidelity {fid['abs_diff_mm']:.4f} mm, "
              f"to_active {rows[-1]['to_active']}, resolve "
              f"{None if res_info is None else res_info['status']} "
              f"(rec {rec_status}, dq {None if res_info is None else res_info.get('max_joint_diff_vs_recorded_rad_plan16')})",
              flush=True)
    return rows, {k2: np.asarray(v) for k2, v in traj_out.items()}


# =================================================================================================
# aggregate
# =================================================================================================

def summarize_all(rows: list[dict], candidates: list[str], extra_meta: dict) -> dict[str, Any]:
    fid = [r["fidelity"]["abs_diff_mm"] for r in rows if r["fidelity"]["abs_diff_mm"] is not None]
    res = [r["to_resolve"] for r in rows if r.get("to_resolve")]
    out = {
        **extra_meta,
        "candidates": candidates,
        "n_chunks": len(rows),
        "split_sizes": {k: sum(1 for r in rows if f(r)) for k, f in M.SPLITS.items()},
        "fidelity": {
            "definition": "evaluator min over the 8 executed steps of the recorded refined_chunk "
                          "vs the record's summary.to.metrics.clearance_m (what our TO certified)",
            "abs_diff_mm": M.summarize(fid),
            "n_over_tol": int(sum(not r["fidelity"]["within_tol"] for r in rows)),
            "support_mask_equals_record": int(sum(bool(r["scene"]["support_mask_equals_record"])
                                                  for r in rows))},
        "to_resolved_vs_recorded": None if not res else {
            "n": len(res),
            "n_same_status": int(sum(x["same_status"] for x in res)),
            "max_joint_diff_rad_plan16": M.summarize(
                x.get("max_joint_diff_vs_recorded_rad_plan16") for x in res),
            "tcp_diff_mm_max_plan16": M.summarize(
                x.get("tcp_diff_vs_recorded_mm_max_plan16") for x in res),
            "n_tcp_diff_over_1mm": int(sum((x.get("tcp_diff_vs_recorded_mm_max_plan16") or 0) > 1.0
                                           for x in res)),
            "n_same_sqp_iterations": int(sum(x["sqp_iterations"] == x["recorded_sqp_iterations"]
                                             for x in res)),
            "n_same_lookahead_gate": int(sum(x["lookahead_gate"] == x["recorded_lookahead_gate"]
                                             for x in res))},
        "aggregate": M.aggregate(rows, candidates),
    }
    return out


def export_b0m_rules(sh: Shared, inputs: pathlib.Path, e: int, positions, out: pathlib.Path
                     ) -> dict[str, Any]:
    """`--export-b0m-rules`: `b0m_rules_<E>.npz` (module doc). Our sphere order throughout."""
    b0m = _load("b0m")
    E = dict(np.load(inputs / f"episode_{e}.npz", allow_pickle=False))
    n = len(E["chunk_file"])
    idx = list(range(n)) if positions == "all" else [p for p in positions if p < n]
    pol = json.loads((pathlib.Path(str(E["client_dir"])) / "manifest.json").read_text())["policy"]
    settings = SimpleNamespace(phase=str(pol["safe_phase"]),
                               manipulators=[str(m) for m in pol["safe_manipulators"]],
                               target_field_policy=sh.target_field_policy)
    cols = arm_cols(sh.spec)
    S = len(sh.names)
    keys = ("chunk_index", "position", "contact", "planes", "n_planes", "obstacle_margin",
            "target_tol", "support_tol", "support_exempt", "strict", "has_target",
            "n_target_points", "to_clear", "to_cls", "to_d", "to_margin", "radii")
    acc: dict[str, list] = {k: [] for k in keys}
    for i in idx:
        path = pathlib.Path(str(E["chunk_file"][i]))
        with np.load(path, allow_pickle=False) as z:
            summary = json.loads(str(z["summary_json"]))
            mask = np.asarray(E["target_field_exclude"][i], bool)
            scene, _ = build_scene(sh, z, summary, mask, settings)
            refined = np.asarray(z["refined_chunk"], np.float64)
        rules = b0m.rules_from_scene(scene, S)
        q_now20 = np.asarray(E["q_now"][i], np.float64)
        sh.lin.set_joint_parameter_path(np.asarray(E["finger_path_full"][i], np.float64))
        q = refined[:, cols]                     # the linearizer's horizon (50); keep plan16
        cl, _, _ = evaluate_trajectory(sh, q, q_now20, scene, np.arange(1))
        cls = getattr(sh.lin, "_last_obstacle_class", None)
        cls = (np.full(cl.shape, -1, np.int8) if cls is None
               else np.asarray(cls, np.int8)[:, :S])
        d = row_distance(sh, q, q_now20, scene)
        cl, cls, d = cl[:16], cls[:16], d[:16]
        r = np.asarray(sh.lin.robot_radii, np.float64)
        P = np.zeros((b0m.MAX_PLANES, 4))
        P[:rules["planes"].shape[0]] = rules["planes"]
        tgt = rules["target"]
        vals = {"chunk_index": int(E["chunk_index"][i]), "position": int(i),
                "contact": rules["contact"], "planes": P, "n_planes": rules["planes"].shape[0],
                "obstacle_margin": rules["obstacle_margin"], "target_tol": rules["target_tol"],
                "support_tol": rules["support_tol"], "support_exempt": rules["support_exempt"],
                "strict": rules["strict"], "has_target": tgt["kind"] is not None,
                "n_target_points": 0 if tgt["kind"] != "points" else len(tgt["points"]),
                "to_clear": cl[:, :S], "to_cls": cls, "to_d": d, "to_margin": d - r[None] - cl[:, :S],
                "radii": r}
        if tgt["kind"] == "spheres":
            raise SystemExit("held target spheres in an offline record (not exported)")
        for k in keys:
            acc[k].append(vals[k])
    f = out / f"b0m_rules_{e}.npz"
    np.savez_compressed(f, **{k: np.asarray(v) for k, v in acc.items()})
    return {"episode": e, "file": str(f), "n_chunks": len(idx),
            "n_chunks_with_target": int(np.sum(acc["has_target"])),
            "n_planes": {str(k): int(np.sum(np.asarray(acc["n_planes"]) == k))
                         for k in sorted(set(acc["n_planes"]))}}


def candidate_order(cu_names: list[str], resolve: bool) -> list[str]:
    return ["reference", "to_recorded"] + (["to_resolved"] if resolve else []) + list(cu_names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--inputs", required=True, help="export_inputs.py output dir")
    ap.add_argument("--curobo", nargs="*", default=None,
                    help="t63_curobo_compare --stage step3 --out dir(s); several are merged "
                         "(candidate names must be unique)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--chunks", default="all", help="`all` or chunk positions `0,20,40,60`")
    ap.add_argument("--resolve", default="fresh",
                    help="fresh | chained | off | saved:<earlier eval dir> (to_resolved read from "
                         "that run's trajectories, not re-solved)")
    ap.add_argument("--export-b0m-rules", default=None, metavar="DIR",
                    help="T63b: write b0m_rules_<E>.npz (B0m margin-rule inputs + our TO's view of "
                         "the recorded refined plan window) for --episodes / --chunks and exit")
    ap.add_argument("--fidelity-tol-mm", type=float, default=FIDELITY_TOL_MM)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="merge every rows_ep*.json in --out into step3_eval.json")
    ap.add_argument("--rows-only", action="store_true",
                    help="write rows_ep<E>.json + trajectories only (parallel per-episode runs; "
                         "merge afterwards with --aggregate-only)")
    args = ap.parse_args(argv)
    if not (args.resolve in ("fresh", "chained", "off") or args.resolve.startswith("saved:")):
        ap.error("--resolve must be fresh | chained | off | saved:<dir>")

    import AG3S

    snap = pathlib.Path(args.snapshot).resolve()
    imported = pathlib.Path(AG3S.__file__).resolve()
    if snap not in imported.parents:
        raise SystemExit(f"AG3S imported from {imported}, not from the snapshot {snap} — "
                         "put the snapshot first on PYTHONPATH")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = pathlib.Path(args.inputs).resolve()
    meta = {"tool": TOOL, "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "argv": sys.argv, "snapshot": str(snap), "ag3s_imported_from": str(imported),
            "inputs": str(inputs), "metric_definitions": "step3_metrics.py module doc",
            "windows": M.WINDOWS, "limit_tol_rad": M.LIMIT_TOL_RAD,
            "to_active_tcp_mm": M.TO_ACTIVE_TCP_MM, "control_period_ms": M.CONTROL_PERIOD_MS}

    if args.aggregate_only:
        files = sorted(out.glob("rows_ep*.json"))
        if not files:
            raise SystemExit(f"no rows_ep*.json in {out}")
        rows, cands, parts = [], None, []
        for f in files:
            d = json.loads(f.read_text())
            rows.extend(d["rows"])
            parts.append({"file": str(f), **{k: d.get(k) for k in ("resolve", "curobo")}})
            if cands is None:
                cands = d["candidates"]
            elif cands != d["candidates"]:
                raise SystemExit(f"{f} has different candidates")
        res = summarize_all(rows, cands, {**meta, "parts": parts,
                                          "curobo": parts[0].get("curobo")})
        (out / "step3_eval.json").write_text(json.dumps(_jsonable({**res, "rows": rows}), indent=1))
        print(f"[t63-step3-eval] merged {len(files)} files, {len(rows)} chunks -> "
              f"{out / 'step3_eval.json'}")
        return 0

    episodes = args.episodes or [int(x["episode"]) for x in json.loads(
        (inputs / "export_manifest.json").read_text())["episodes"]]
    positions = "all" if args.chunks == "all" else [int(c) for c in args.chunks.split(",")]
    if args.export_b0m_rules:
        sh = Shared(inputs, "off")
        rdir = pathlib.Path(args.export_b0m_rules)
        rdir.mkdir(parents=True, exist_ok=True)
        parts = [export_b0m_rules(sh, inputs, e, positions, rdir) for e in episodes]
        info = {**meta, "episodes": parts, "server_to_config": sh.server_cfg.to_dict()}
        tag = "_".join(str(e) for e in episodes)
        (rdir / f"b0m_rules_meta_{tag}.json").write_text(json.dumps(_jsonable(info), indent=1))
        print(f"[t63-step3-eval] b0m rules: {parts}", flush=True)
        return 0
    cu = CuroboOutputs([pathlib.Path(c).resolve() for c in args.curobo] if args.curobo else None)
    t0 = time.perf_counter()
    sh = Shared(inputs, args.resolve)
    meta.update(build_s=sh.build_s, resolve=args.resolve,
                resolve_config_changes={"sqp.time_budget_ms": [sh.server_cfg.sqp.time_budget_ms,
                                                               1e9]},
                server_to_config=sh.server_cfg.to_dict(), curobo=cu.meta or None)
    cands = candidate_order(cu.names, args.resolve != "off")
    all_rows = []
    for e in episodes:
        te = time.perf_counter()
        rows, traj = run_episode(sh, inputs, e, positions, cu, args.fidelity_tol_mm)
        (out / "trajectories").mkdir(exist_ok=True)
        np.savez_compressed(out / "trajectories" / f"ep{e}.npz", **traj)
        ep_meta = {**meta, "episode": e, "candidates": cands, "n_chunks": len(rows),
                   "elapsed_s": time.perf_counter() - te,
                   "trajectories": str(out / "trajectories" / f"ep{e}.npz")}
        (out / f"rows_ep{e}.json").write_text(json.dumps(_jsonable({**ep_meta, "rows": rows}),
                                                          indent=1))
        all_rows.extend(rows)
    elapsed = time.perf_counter() - t0
    if args.rows_only:
        print(f"[t63-step3-eval] {len(all_rows)} chunks x {len(cands)} candidates in {elapsed:.0f} s "
              f"(rows only)", flush=True)
        return 0
    res = summarize_all(all_rows, cands, {**meta, "elapsed_s": elapsed,
                                          "elapsed_per_chunk_s": elapsed / max(len(all_rows), 1),
                                          "episodes": episodes})
    (out / "step3_eval.json").write_text(json.dumps(_jsonable({**res, "rows": all_rows}), indent=1))
    print(f"[t63-step3-eval] {len(all_rows)} chunks x {len(cands)} candidates in {elapsed:.0f} s "
          f"-> {out / 'step3_eval.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
