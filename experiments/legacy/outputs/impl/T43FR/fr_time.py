"""T43 FR — TO wall time per finger_cover option, interleaved A/B on the same recorded chunks as T43 FC's to_time.py
(T40 G E3b C1807 119–130, B1968 140–151; iterations = the record's, clock off). CPU only, self-measured.

Interleaving: for every pass and chunk, every model (bare + each option) runs once, in a fresh random order. Pass 0 is a
warm-up and is dropped. Per (chunk, model): min over passes (as FC: min of reps). Per model: median / p90 over chunks,
and the paired ratio to fc25 (same chunk, min over passes) median.
usage: fr_time.py OUT_JSON N_PASSES SEED [MODEL ...]   (MODEL default = all of fr_options.OPTIONS)
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, logging, sys, time, pathlib
import numpy as np, mujoco
logging.disable(logging.WARNING)
sys.path.insert(0, _ag3s_workspace_path('outputs/impl/T43FR'))
from fr_options import OPTIONS  # noqa: E402
import AG3S.trajopt.experiments.fixed_input_ablation as fia  # noqa: E402
from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model, GRIPPER_LINKS  # noqa: E402
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene  # noqa: E402
from AG3S.trajopt.types import ChunkLayout  # noqa: E402
from AG3S.trajopt.refiner import TrajOptChunkRefiner  # noqa: E402

out_path, n_pass, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
names = sys.argv[4:] or list(OPTIONS)
S = _ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313')
RUNS = {"C1807": (S + '/server_constraints/run_0000', _ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/C/E3b_ep1807_s18071'), range(119, 131)),
        "B1968": (S + '/server_constraints/run_0000', _ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/B/E3b_ep1968_s19681'), range(140, 152))}
M = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m = mujoco.MjModel.from_xml_path(M); scene = TransportScene.attach(m, mujoco.MjData(m))
opts = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
models = {"bare": build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts)}
for n in names:
    models[n] = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts, finger_cover=dict(OPTIONS[n]))
    print(n, models[n].n_spheres, flush=True)
qadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}

jobs = []   # (run, k, model name, refiner, robot, reference, ctx, finger joints)
for run, (cdir, rdir, chunks) in RUNS.items():
    args = fia.build_parser().parse_args(["--constraints", cdir, "--frames", rdir + "/frames.jsonl", "--out", "/dev/null", "--server-log", S + "/serve_safe.log"])
    manifest = json.loads(pathlib.Path(rdir + "/manifest.json").read_text())
    st = fia.resolve_settings(args, fia.parse_server_log(S + "/serve_safe.log"), manifest)
    rows = [json.loads(l) for l in open(rdir + "/frames.jsonl") if l.strip()]
    plan = {int(r["seq"]): r for r in rows if r.get("kind") == "planning"}
    for k in chunks:
        z = np.load(f"{cdir}/chunk_{k:05d}.npz"); s = json.loads(str(z["summary_json"]))
        row = plan.get(int(s["t_step"]))
        if row is None or np.abs(np.asarray(row["actions"]) - z["refined_chunk"]).max() > 1e-5:
            continue
        q_now = fia.q_now_from_qpos(row["qpos"], qadr)
        field = fia.build_field(z, outside_distance=0.4)
        tp = z["target_points"] if "target_points" in z.files else None
        iters = int((s.get("to") or {}).get("sqp_iterations") or 1)
        fj = (s.get("finger_joints") or {}).get("joints")
        ref_chunk = np.asarray(z["reference_chunk"], np.float64)
        for mname, robot in models.items():
            layout = ChunkLayout.rby1(robot.joint_names)
            margin, mask, auth, _, _ = fia.contact_policy(st, robot.sphere_link_names, has_target=bool(s.get("has_target")) and tp is not None,
                                                          has_free=bool(field.target_free_layers), attached=False)
            sc = fia.make_scene(field, np.asarray(robot.radii), st.esdf_margin, target_points=tp, margin=margin, mask=mask)
            cfg = fia.to_config(st, collision=st.collision_enabled, cost=st.cost, sqp={"time_budget_ms": fia.NO_CLOCK_MS, "max_iterations": iters})
            ref = TrajOptChunkRefiner(robot, layout, (lambda sc_, qn_: (lambda _c: (sc_, qn_, True)))(sc, q_now), cfg)
            jobs.append((run, k, mname, ref, robot, ref_chunk, {"t_step": int(s["t_step"])}, fj))
        z.close()

rng = np.random.default_rng(seed)
keys = sorted({(j[0], j[1]) for j in jobs})
by = {(j[0], j[1], j[2]): j for j in jobs}
wall = {}      # (run, k, model) -> [ms per pass]
comp = {}
for p in range(n_pass + 1):
    t_pass = time.time()
    for key in keys:
        order = list(models)
        rng.shuffle(order)
        for mname in order:
            run, k, _, ref, robot, rc, ctx, fj = by[(key[0], key[1], mname)]
            if fj:
                robot.set_joint_parameters(fj)
            ref.reset()
            t = time.perf_counter()
            ref.refine(rc, {**ctx, "previous_physical_chunk": None})
            w = (time.perf_counter() - t) * 1e3
            robot.set_joint_parameters(None)
            if p == 0:
                continue
            wall.setdefault(f"{run}|{k}|{mname}", []).append(w)
            tm = ref.last_result.metrics.get("timing_ms")
            tm = eval(tm) if isinstance(tm, str) else tm
            if tm:
                comp.setdefault(f"{run}|{k}|{mname}", []).append({c: float(tm.get(c, 0)) for c in ("linearize", "assemble", "qp", "check")})
    print(f"pass {p} {time.time() - t_pass:.1f}s", flush=True)

summary = {}
for run in RUNS:
    ks = sorted({k for (r, k) in keys if r == run})
    base = {k: min(wall[f"{run}|{k}|fc25"]) for k in ks} if "fc25" in models else None
    for mname, robot in models.items():
        w = np.array([min(wall[f"{run}|{k}|{mname}"]) for k in ks])
        ratio = np.array([min(wall[f"{run}|{k}|{mname}"]) / base[k] for k in ks]) if base else None
        summary.setdefault(mname, {"n_spheres": robot.n_spheres})[run] = {
            "median_ms": float(np.median(w)), "p90_ms": float(np.percentile(w, 90)), "max_ms": float(w.max()),
            "ratio_to_fc25_median": None if ratio is None else float(np.median(ratio)),
            "components_median": {c: float(np.median([min(x[c] for x in comp[f"{run}|{k}|{mname}"]) for k in ks])) for c in ("linearize", "assemble", "qp", "check")}}
for mname, v in summary.items():
    print(mname, v["n_spheres"], " | ".join(f"{r} med {v[r]['median_ms']:.1f} p90 {v[r]['p90_ms']:.1f} ×fc25 {v[r]['ratio_to_fc25_median']:.3f}" for r in RUNS), flush=True)
json.dump({"definitions": __doc__, "n_pass": n_pass, "seed": seed, "summary": summary, "wall_ms": wall}, open(out_path, "w"), indent=1)
