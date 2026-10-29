"""T43 S (5) — at the first apple-bar contact row of each contact run: where on the apple is the contact, is that point inside
the TO's held query spheres, and how far is the hand (EE_BODY_L + fingers) from the bar. Records only, CPU.
Held spheres = the last held planning chunk at or before the contact row (summary spheres_parent_frame), placed with URDF FK of
the parent link at the contact row's measured qpos (the same rigid attachment the TO uses).
usage (cwd ${AG3S_PROJECT_ROOT}-t41c): contact_detail.py"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, mujoco
import measure_held as MH

R = json.loads((MH.OUT / "held_runs_all.json").read_text())["runs"]


def main():
    out = {}
    for key, r in R.items():
        t_c = r["apple_bar_contact_t"]
        if t_c is None: continue
        ix = MH.IDX[key]; rd = pathlib.Path(key)
        man = json.loads((rd / "manifest.json").read_text()); argv = man["extra"]["argv"]
        G = MH.setup(pathlib.Path(ix["server"]) / "serve_safe.log", man)
        m, d = MH.mj()
        mg = MH.PPO.PickPlaceObstacleManager(m, d, MH.PPO.load_pick_place_obstacle_config(argv[argv.index("--obstacle-config") + 1]))
        mg.activate(argv[argv.index("--obstacle-profile") + 1])
        bar = [g for g in mg.collision_geom_ids if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("collision_bar")][0]
        rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
        ctrl = {x["t_step"]: x for x in rows if x["kind"] == "control"}
        c = [h for h in r["held_chunks"] if h["t_step"] <= t_c][-1]
        local = np.asarray(c["held_local"], float); rad = np.asarray(c["held_radii_mm"], float) / 1000.0
        q = np.asarray(ctrl[t_c]["qpos"], float); MH.set_q(q)
        ft = np.zeros(6); dist = mg.pair_distance(MH._G["apple_g"], bar, 0.30, ft)
        p_apple, p_bar = ft[:3].copy(), ft[3:].copy()
        ac = d.xpos[MH._G["B"]("apple")].copy()
        T = MH._link_pose_numeric(G["robot"], MH.FIA.q_now_from_qpos(q, G["qadr"]), c["held"]["parent_link"])
        cb = local @ T[:3, :3].T + T[:3, 3]; cw = MH.to_world(cb, q)
        gap = float((np.linalg.norm(cw - p_apple, axis=1) - rad).min())
        hand = []
        for bn in ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2"):
            bid = MH._G["B"](bn)
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) == bid and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    hand.append(mg.pair_distance(g, bar, 0.30, ft))
        V = MH.geom_vertices_world(MH._G["apple_g"])
        out[key] = {"contact_t": t_c, "held_chunk_seq": c["seq"], "apple_bar_mm": 1e3 * dist,
                    "contact_point_minus_apple_centre_mm": (1e3 * (p_apple - ac)).tolist(),
                    "contact_point_height_above_apple_centre_mm": float(1e3 * (p_apple[2] - ac[2])),
                    "apple_half_height_mm": float(1e3 * (V[:, 2].max() - V[:, 2].min()) / 2),
                    "contact_point_gap_to_held_spheres_mm": 1e3 * gap,
                    "contact_point_inside_held_sphere": bool(gap <= 0),
                    "hand_to_bar_min_mm": 1e3 * float(min(hand)),
                    "held_radii_mm": (1e3 * rad).tolist(), "held_centre_world": cw.tolist(), "apple_centre_world": ac.tolist(),
                    "contact_point_world": p_apple.tolist(), "bar_point_world": p_bar.tolist()}
        print(key.split("verify/")[1], {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out[key].items() if not k.endswith("world")}, flush=True)
    (MH.OUT / "contact_detail.json").write_text(json.dumps({"definitions": __doc__, "runs": out}, indent=1))


if __name__ == "__main__":
    main()
