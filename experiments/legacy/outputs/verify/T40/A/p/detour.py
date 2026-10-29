"""T40 A detour room for the best placement of each selected pair / kind (verifier, CPU).
over the top : tcp_z_max = workspace.max_tcp_z at the obstacle xy (starts: carry rows of E0 and E3b, max of the two);
               apple_bottom_max_est = tcp_z_max - median(tcp z - apple bottom z) over the carry rows (E0+E3b);
               free_height_over_top = apple_bottom_max_est - obstacle top (bollard 287 mm, divider 80 mm above table).
               recorded: max apple-bottom height above table in [lift_t, place_t] of each run.
lateral      : at the t = 0 scene, GJK separation from the tall part only (bollard post cylinder r27 z[10,287] mm;
               divider panel box 208x36x80 mm) to the crate (all crate geoms) and to each fruit; plus the top heights
               above the table of the crate (incl. handles) and each fruit at t = 0 (AABB of collision geoms).
usage: detour.py TAG [TAG ...]   (reads raw/replay_<TAG>.json, writes raw/detour_<TAG>.json)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
import t40a_lib as L, workspace as W, search as S

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A/raw'))
TOP = {"bollard_0": 0.287, "divider_0": 0.080}
TALL = {"bollard_0": "pick_obstacle_bollard_0_collision", "divider_0": "pick_obstacle_divider_0_collision"}


def top_z(geoms):
    z = -9.0
    for g in geoms:
        sup = L.gjk.make_support(L.m, L.d, int(g)); z = max(z, sup(np.array([0, 0, 1.0]))[2])
    return z - L.TABLE_Z


def main(tag):
    rp = json.load(open(RAW / f"replay_{tag}.json"))
    allp = {f"{p['episode']}_{p['seed']}": p for p in json.load(open(RAW / "carry_paths_22pairs.json"))["pairs"]}
    ent = allp[tag]
    runs = {c: L.load_run(ent["runs"][c]["run_dir"]) for c in ("E0", "E3b")}
    out = {"tag": tag, "kinds": {}}
    for slot, k in rp["kinds"].items():
        b = k["best"]
        if b is None:
            out["kinds"][slot] = None; continue
        xy = np.array(b["xy"]); yaw = b["yaw_rad"]
        Qc, rec_bottom, offs = [], {}, []
        for c, R in runs.items():
            w = rp["windows"][c]; mk = (R["t"] >= w["lift_t"]) & (R["t"] <= w["place_t"])
            Q = R["q"][mk]; Qc.append(Q); offs.append(W.tcp_minus_apple_bottom(Q))
            bz = []
            for q in Q:
                L.set_q(q); bz.append(L.gjk.make_support(L.m, L.d, int(L.B_APPLE[0]))(np.array([0, 0, -1.0]))[2] - L.TABLE_Z)
            rec_bottom[c] = float(max(bz))
        ws = {c: W.max_tcp_z(xy, Q) for c, Q in zip(runs, Qc)}
        tz = max(v["tcp_z_max_above_table_m"] for v in ws.values() if v["tcp_z_max_above_table_m"] is not None)
        off = float(np.median(offs))
        # lateral, t = 0 (E3b q0; identical to E0 q0)
        L.set_q(runs["E3b"]["q0"]); L.place(slot, xy, yaw); mujoco.mj_kinematics(L.m, L.d)
        tall = [L.GEOM(TALL[slot])]
        lat = {"crate": L.min_dist(tall, L.CR_CRATE), "apple": L.min_dist(tall, L.B_APPLE)}
        for n, g in L.F_FRUITS.items():
            lat[n] = L.min_dist(tall, g)
        heights = {"crate": top_z(L.CR_CRATE), "apple": top_z(L.B_APPLE)}
        for n, g in L.F_FRUITS.items():
            heights[n] = top_z(g)
        out["kinds"][slot] = {
            "xy": b["xy"], "yaw_rad": yaw, "obstacle_top_above_table_m": TOP[slot],
            "tcp_z_max_above_table_m": tz, "tcp_minus_apple_bottom_median_m": off,
            "apple_bottom_max_est_m": tz - off, "free_height_over_top_m": tz - off - TOP[slot],
            "recorded_apple_bottom_max_m": rec_bottom,
            "workspace": {c: {"tcp_z_max_above_table_m": v["tcp_z_max_above_table_m"],
                              "best": v["best"] and {kk: v["best"][kk] for kk in ("start_row", "xy_err_m", "axis_cos")}} for c, v in ws.items()},
            "lateral_t0_tall_part_gap_m": lat, "t0_top_height_above_table_m": heights}
        print(tag, slot, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in out["kinds"][slot].items() if kk not in ("workspace",)}, flush=True)
    (RAW / f"detour_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
