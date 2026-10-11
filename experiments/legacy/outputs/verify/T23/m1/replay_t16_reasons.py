"""T23 측정 #1 — T16 의 unsafe chunk 를 T24 도구로 재구성하고 SafePolicy._verdict (T23) 로 다시 판정한다.

가정 (옛 기록에 manipulated 기록이 없다 — lead 지시):
  manipulated = 사과 참값 `object_poses.apple.pos` (그 planning 행, MuJoCo world → base 변환),
  반지름 0.045 m, state "visible", 권한 link = contact.contact_links["left"].
변형:
  main      : scene = T24 재구성 씬(target_free_mask 기록대로) + manipulated_link_margin(권한 표),
              조작 대상 기하 없음 → classify 의 근거는 `centroid` (공 반지름 0.045)
  distance  : main 과 같되 scene.manipulated_spheres = [사과 참값] r=0.045 → 근거 `distance`
  nomask    : main 과 같되 target_free_mask=None (모든 행이 coarse/fine 합성 필드에 묻는다)
각 변형마다 allow_unresolved_tier False(기본)/True 둘 다.
궤적 = 기록 refined (T16 은 --no-collision 이라 refined ≡ reference).
max_violation = 그 변형 씬에서 같은 궤적의 최악 행 (T16 은 충돌 OFF 라 씬이 궤적을 바꾸지 않는다).
status = 그 값이 tolerance 를 넘으면 violated, 아니면 optimal.
attached chunk: 쥔 물체의 질의점은 기록에 없다 → 로봇 구 행만 재분류하고, 기록의 최악 행이
attached 점이면 classify 규칙(q ≥ n_spheres → collision "attached_point")대로 collision 행 하나를 더한다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json, pathlib, sys, types, time
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.trajopt.experiments import fixed_input_ablation as fia  # noqa: E402

RUN = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t16'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T23/m1'))
APPLE_R = 0.045


def main():
    t0 = time.time()
    args = fia.build_parser().parse_args([
        "--constraints", str(RUN / "server_constraints/run_0000"),
        "--frames", str(RUN / "execute_ep1807/frames.jsonl"),
        "--server-log", str(RUN / "serve_safe.log"),
        "--out", str(OUT / "unused.json"), "--variants", "replay", "--no-chain"])
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout, TrajOptStatus
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.safe_policy import SafePolicy
    from AG3S.trajopt import wire
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.ag3s.types import ContactPolicyContext
    import mujoco

    manifest = json.loads((RUN / "execute_ep1807/manifest.json").read_text())
    log = fia.parse_server_log(args.server_log)
    settings = fia.resolve_settings(args, log, manifest)
    robot, qadr, xml_path = fia.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, max_distance, unknown_policy = fia.contact_policy(
        settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    outside = max_distance if unknown_policy == "free" else -max_distance
    rows = [json.loads(l) for l in (RUN / "execute_ep1807/frames.jsonl").read_text().splitlines() if l.strip()]
    plan_rows = {int(r["seq"]): r for r in rows if r.get("kind") == "planning"}
    planning = fia.to_config(settings, collision=True, cost=settings.cost,
                             sqp={"time_budget_ms": 50.0, "max_iterations": 3})
    limits = build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits)
    meter = fia.Meter(robot, layout, planning.horizon.planned, settings.hand, limits)
    sh = fia.Shared(args=args, settings=settings, robot=robot, qadr=qadr, layout=layout,
                    meter=meter, plan_rows=plan_rows, variant_configs={}, active_rows_config=None,
                    outside_distance=outside, recorded_iters={})

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    apple_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "apple")

    config = AG3SConfig.from_dict({"collision_backend": "esdf"})
    cpolicy = ClearancePolicy.from_config(config.contact, config.geometry, config.support_surface)
    ctx = ContactPolicyContext.make(settings.phase, settings.manipulators)
    authorized_contact_links = list(config.contact.contact_links["left"])
    names = [str(n) for n in robot.sphere_link_names]
    to_cfg = TrajOptConfig()
    tol = float(to_cfg.safety.violation_tolerance)

    files = sorted((RUN / "server_constraints/run_0000").glob("chunk_*.npz"), key=fia._chunk_index)
    out_chunks = []
    for f in files:
        k = fia._chunk_index(f)
        row = plan_rows.get(k + 1)
        if row is None or row.get("safe", True):
            continue  # safe chunk 는 T16 에서 이미 실행됐다 — 대상은 unsafe 25
        rec, inp = fia.prepare_chunk(sh, f)
        z = np.load(f)
        summary = json.loads(str(z["summary_json"]))
        to = summary.get("to") or {}
        rmetrics = to.get("metrics") or {}
        attached = bool(rec.get("attached"))
        field = fia.build_field(z, outside_distance=outside)
        q_now = fia.q_now_from_qpos(row["qpos"], qadr)
        radii = meter.lin.robot_radii
        target_points = z["target_points"] if "target_points" in z.files else None
        has_target = bool(summary.get("has_target")) and target_points is not None
        _, mask_rec, _, _, _ = fia.contact_policy(settings, names, has_target=has_target,
                                                  has_free=bool(field.target_free_layers), attached=attached)
        margin_hyp, _, authorized_policy, _, _ = fia.contact_policy(
            settings, names, has_target=True, has_free=bool(field.target_free_layers), attached=False)
        planned = meter.lin.horizon
        refined_traj = layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], np.float64))[:, :planned]

        # 사과 참값 → base
        qpos = np.asarray(row["qpos"], np.float64)
        mj_data.qpos[:] = qpos
        mujoco.mj_kinematics(mj_model, mj_data)
        R, p = mj_data.xmat[base_id].reshape(3, 3), mj_data.xpos[base_id].copy()
        apple_w = np.asarray(row["object_poses"]["apple"]["pos"], np.float64)
        apple_b = R.T @ (apple_w - p)
        apple_from_qpos_w = mj_data.xpos[apple_id].copy()

        man_rec = {"id": "apple_truth(assumed)", "state": "visible", "centroid": apple_b.tolist(),
                   "radius": APPLE_R, "age_frames": 0}
        recorded_pair = fia._parse_pair(rmetrics.get("max_violation_pair"))
        ch = {"chunk_index": k, "seq": k + 1, "t_step": row.get("t_step"),
              "recorded": {"safe": row.get("safe"), "hold_reason": row.get("hold_reason"),
                           "ag3s_status": summary.get("status"), "validity": summary.get("validity"),
                           "grounding_status": summary.get("grounding_status"),
                           "has_target": summary.get("has_target"),
                           "target_centroid": np.asarray(z["target_centroid"]).tolist() if "target_centroid" in z.files else None,
                           "trajopt_status": to.get("status"), "clearance_m": rmetrics.get("clearance_m"),
                           "max_violation_pair": recorded_pair,
                           "n_attached_points": (summary.get("esdf_stats") or {}).get("n_attached_points")},
              "fidelity": rec.get("fidelity"), "skipped_reason_t24": rec.get("skipped_reason"),
              "attached": attached,
              "apple_world": apple_w.tolist(), "apple_base": apple_b.tolist(),
              "apple_world_from_qpos": apple_from_qpos_w.tolist(),
              "base_world_pos": p.tolist(),
              "target_free_mask_recorded_links": sorted({names[i] for i in np.flatnonzero(mask_rec)}) if mask_rec is not None else [],
              "variants": {}}
        for vname in ("main", "distance", "nomask"):
            scene = fia.make_scene(field, radii, settings.esdf_margin, target_points=None, margin=None,
                                   mask=(None if vname == "nomask" else mask_rec))
            import dataclasses as _dc
            upd = {"manipulated_link_margin": np.asarray(margin_hyp, np.float64)}
            if vname == "distance":
                upd["manipulated_spheres"] = apple_b[None, :].copy()
                upd["manipulated_sphere_radii"] = np.asarray([APPLE_R])
            scene = _dc.replace(scene, **upd)
            worst_c, worst_pair = meter.lin.worst_row(refined_traj, q_now, scene)
            hyp_violation = float(max(0.0, -worst_c))
            extra_attached = None
            if attached and recorded_pair and str(recorded_pair.get("link", "")).startswith("attached"):
                extra_attached = {"block": "esdf", "step": recorded_pair.get("step"), "query": recorded_pair.get("query"),
                                  "link": recorded_pair.get("link"), "clearance_m": recorded_pair.get("clearance_m"),
                                  "class": "collision", "why": "attached_point", "source": "recorded max_violation_pair"}
                hyp_violation = max(hyp_violation, -float(recorded_pair["clearance_m"]))
            for allow in (False, True):
                status = TrajOptStatus.VIOLATED if hyp_violation > tol else TrajOptStatus.OPTIMAL
                result = types.SimpleNamespace(
                    status=status, max_violation=hyp_violation, trajectory=refined_traj, notes=[],
                    metrics={"max_violation_pair": worst_pair,
                             "time_budget_hit": bool(rmetrics.get("time_budget_hit")),
                             "max_iterations_hit": bool(rmetrics.get("max_iterations_hit")),
                             "sqp_iterations": rmetrics.get("sqp_iterations")})
                cs = types.SimpleNamespace(
                    status=types.SimpleNamespace(value=summary.get("status")),
                    grounding_status=types.SimpleNamespace(value=summary.get("grounding_status")),
                    validity=types.SimpleNamespace(value=summary.get("validity")),
                    geometry_certified=summary.get("status") == "ok",
                    manipulated=object(), contact_context=ctx, candidates=[],
                    metrics={"manipulated": man_rec})
                sp = SafePolicy.__new__(SafePolicy)
                sp.to_config = to_cfg
                sp.verdict_policy = "reasons"
                sp.allowed_contact_pad_m = 0.01
                sp.allow_unresolved_contact_tier = allow
                sp.ag3s = types.SimpleNamespace(builder=types.SimpleNamespace(clearance_policy=cpolicy))
                sp.refiner = types.SimpleNamespace(last_result=result,
                                                   optimizer=types.SimpleNamespace(linearizer=meter.lin))
                sp._last_constraint_set = cs
                sp._last_snapshot = scene
                sp._last_q_now = q_now
                verdict = SafePolicy._verdict(sp, None)
                reasons = list(verdict.verdict_reasons or [])
                cls = dict(sp._last_classification or {})
                if extra_attached is not None:
                    # 쥔 물체의 점: classify 규칙상 언제나 collision — 기록 최악 행 하나로 대신한다.
                    kinds = [r["kind"] for r in reasons]
                    if "collision" not in kinds:
                        reasons.append(wire.make_reason("collision", "recorded attached-point row (not reconstructable)",
                                                        row=extra_attached))
                    action, _ = wire.gate_decision(reasons)
                    safe = action == "execute"
                else:
                    safe = bool(verdict.safe)
                # 행 전부 (keep_rows) — 같은 입력으로 한 번 더
                from AG3S.trajopt.safe_policy import classify_violations
                full = classify_violations(meter.lin, refined_traj, q_now, scene,
                                           authorized_links=authorized_contact_links, tolerance=tol,
                                           manipulated=man_rec, pad_m=0.01, allow_unresolved_tier=allow,
                                           keep_rows=True) if hyp_violation > tol else {"rows": []}
                rws = full.get("rows", [])
                def summ(rs):
                    out = {}
                    for r in rs:
                        key = f"{r['class']}|{r['why']}|{r['link']}|{r.get('tier')}"
                        o = out.setdefault(key, {"n": 0, "worst_clearance_mm": 0.0, "min_object_distance_mm": None})
                        o["n"] += 1
                        o["worst_clearance_mm"] = min(o["worst_clearance_mm"], r["clearance_m"] * 1000)
                        if r.get("object_distance_m") is not None:
                            dmm = r["object_distance_m"] * 1000
                            o["min_object_distance_mm"] = dmm if o["min_object_distance_mm"] is None else min(o["min_object_distance_mm"], dmm)
                    return out
                # 질의점–사과 중심 거리 (행별)
                cen_d = [float(np.linalg.norm(np.asarray(r["point_m"]) - apple_b)) for r in rws]
                entry = {
                    "safe": safe, "kinds": [r["kind"] for r in reasons],
                    "reasons": [{k2: v for k2, v in r.items() if k2 in ("kind", "action", "detail")} for r in reasons],
                    "hyp_max_violation_mm": hyp_violation * 1000.0, "hyp_status": status.value,
                    "worst_pair": fia._jsonable(worst_pair),
                    "classification": {k2: fia._jsonable(v) for k2, v in cls.items() if k2 != "rows"},
                    "row_groups": summ(rws),
                    "n_rows_total": len(rws),
                    "authorized_links_policy": sorted(authorized_policy),
                    "row_query_to_apple_centre_mm": {"min": (min(cen_d) * 1000 if cen_d else None),
                                                      "max": (max(cen_d) * 1000 if cen_d else None)},
                    "attached_recorded_row": extra_attached,
                }
                ch["variants"][f"{vname}|allow_unresolved={allow}"] = entry
                if vname == "main" and not allow:
                    ch["rows_main"] = [{kk: fia._jsonable(r.get(kk)) for kk in ("block", "step", "query", "link", "clearance_m", "tier", "class", "why", "evidence", "object_distance_m", "point_m", "label")} for r in rws]
        z.close()
        m = ch["variants"]["main|allow_unresolved=False"]
        print(f"chunk {k:2d} t={ch['t_step']:3d} att={int(attached)} kinds={m['kinds']} safe={m['safe']} "
              f"viol={m['hyp_max_violation_mm']:.1f}mm groups={list(m['row_groups'])[:4]}", flush=True)
        out_chunks.append(ch)

    doc = {"tool": "outputs/verify/T23/m1/replay_t16_reasons.py",
           "reconstruction": "AG3S.trajopt.experiments.fixed_input_ablation (T24) prepare_chunk/build_field/contact_policy",
           "assumption": {"manipulated_centroid": "object_poses.apple.pos (T16 planning row, MuJoCo world → base via body 'base')",
                          "manipulated_radius_m": APPLE_R, "manipulated_state": "visible",
                          "authorized_links": authorized_contact_links},
           "settings": settings.to_json(), "tolerance_m": tol, "elapsed_s": time.time() - t0,
           "chunks": out_chunks}
    (OUT / "t16_reasons.json").write_text(json.dumps(doc, indent=1, default=fia._jsonable))
    print("wrote", OUT / "t16_reasons.json")


if __name__ == "__main__":
    main()
