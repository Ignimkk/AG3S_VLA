"""T81 — per-request vacancy evidence of recorded runs (offline), for fixing the §1 thresholds before results.

For every recorded request j of a run (server chunk j ↔ client step j, `module_eval` convention):

* **old place** = the manipulated object's standing geometry *before* request j: the chunk j−1 `target_points`
  when chunk j−1 has a manipulated object (what `TargetConfirm._geometry` held when request j started —
  the last observed geometry, or the kept anchor / occluded stand-in), at most `--samples` points
  (`target_vacancy.sample_points`);
* **cameras** = the client step j depth (``uint16`` mm → m), K and camera → base of the head and the two
  wrist cameras — what the server received for request j (1 mm quantised copy);
* **robot pixels** (approximate): a pixel whose back-projected depth point lies inside one of chunk j's
  recorded constraint spheres (`sphere_centres` / `sphere_radii`, at q_now) + `--robot-pad-mm`. The server's
  own mask is the self-filter model's (587 spheres, inflation 0); it is not recorded;
* the judgement = `stages.target_vacancy.judge` — the same function the server rule calls — for every δ of
  `--deltas-mm`;
* **applies** = the rule's condition at request j: the latch state (`grasp.state`) is searching · latched ·
  closing, nothing attached, PLACED not seen yet, and an old place exists;
* **GT** (MuJoCo, the run's own model, client step qpos): the target fruit's body centre (base frame),
  its displacement since the request the old place was last seen in (`last_seen_frame` of chunk j−1) and
  since the first request, and the share of old-place samples within `--gt-surface-mm` of the fruit's
  current GT surface (`ref_on_apple`). `apple_there` = `ref_on_apple` ≥ `--gt-there` (a release then is wrong, §3);
  `vacated` = `ref_on_apple` < `--gt-gone` (the old place does not hold the fruit — after a push / move, or a
  stale stand-in). `moved` = the fruit moved > `--gt-moved-mm` since it was last seen (reported only).

Then the §1.4 rule (2 consecutive requests, empty ≥ p_empty ∧ present ≤ p_present, ≥ M judged points) is
simulated on the recorded evidence over a grid of thresholds: per setting, the requests that would release
in each run, wrong releases (GT apple there), and the first release after the move / push. The recorded
reference does not change after a simulated release (the run was recorded with the rule off), so only the
**first** release per run is meaningful.

  MUJOCO_GL=osmesa PYTHONPATH=<checkout> python -m AG3S.ag3s.experiments.studies.t81_vacancy --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import pathlib

import numpy as np

from AG3S.ag3s.stages.target_vacancy import VacancyView, judge, sample_points, vacant

STORE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development")
T71 = STORE / "20261029T232703+0900_8ab3c7ef7757_7199b750125b/verify/T71/S1"
T80 = STORE / "20261030T004052+0900_8ab3c7ef7757_1c61eaabd79c/verify/T80/CL/ag3s/r1"
#: The runs of T81.task §1 (`kind`: `push` / `move` = the old place empties; `success` = it must never release).
DEFAULT_ITEMS = [
    {"name": "reg211_T71R0", "kind": "push", "client": str(T71 / "REG_R0_k0/runs/B_ROI_FPR/B_ROI_FPR_ep211_s2113"),
     "server": str(T71 / "REG_R0_k0/runs/server_8292_20261029T154310"), "segment": 0},
    {"name": "T80C_forward", "kind": "move", "client": str(T80 / "runs/C_ep56_forward"),
     "server": str(T80 / "server_ag3s_8298_20261031T133627"), "segment": "C_ep56_forward"},
    {"name": "T80C_back", "kind": "move", "client": str(T80 / "runs/C_ep56_back"),
     "server": str(T80 / "server_ag3s_8298_20261031T133627"), "segment": "C_ep56_back"},
    {"name": "reg119_T71R0k1", "kind": "success",
     "client": str(T71 / "REG_R0_k1/runs/B_ROI_FPR/B_ROI_FPR_ep119_s1193"),
     "server": str(T71 / "REG_R0_k1/runs/server_8292_20261029T163157"), "segment": 0},
    {"name": "occ1345_case1_L2_s13453", "kind": "success", "client": str(T71 / "OCC_R0/k0/runs/AG3S_case1_L2_s13453"),
     "server": str(T71 / "OCC_R0/k0/server_ag3s_8296_20261029T154140"), "segment": "AG3S_case1_L2_s13453"},
]
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
APPLIES_STATES = ("searching", "latched", "closing")


def episode_chunks(server: pathlib.Path, segment) -> list[str]:
    """Chunk files of one episode: episodes start where `policy_seed.applied` is set (t80_eval convention);
    `segment` = index, or a client name looked up in `order.txt`."""
    files = sorted(glob.glob(str(server / "server_constraints" / "run_*" / "chunk_*.npz")))
    segs: list[list[str]] = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"]))
        if (s.get("policy_seed") or {}).get("applied") is not None or not segs:
            segs.append([])
        segs[-1].append(f)
    if isinstance(segment, str):
        order = [x.strip() for x in (server / "order.txt").read_text().splitlines() if x.strip()]
        if len(order) != len(segs):
            raise ValueError(f"{server}: {len(segs)} episodes in the chunks, {len(order)} in order.txt")
        segment = order.index(segment)
    return segs[int(segment)]


def robot_mask_near(points, depth_m, K, T, centres, radii, pad, margin_px=6):
    """(H, W) robot mask decided only in the pixel box the points project into (+ margin): a pixel is the robot
    when its back-projected depth point is inside a recorded sphere + `pad`."""
    from scipy.spatial import cKDTree

    h, w = depth_m.shape
    mask = np.zeros((h, w), bool)
    pc = (np.asarray(points) - T[:3, 3]) @ T[:3, :3]
    ok = pc[:, 2] > 1e-6
    if not ok.any() or not len(centres):
        return mask
    u = K[0, 0] * pc[ok, 0] / pc[ok, 2] + K[0, 2]
    v = K[1, 1] * pc[ok, 1] / pc[ok, 2] + K[1, 2]
    u0, u1 = int(max(0, np.floor(u.min()) - margin_px)), int(min(w, np.ceil(u.max()) + margin_px + 1))
    v0, v1 = int(max(0, np.floor(v.min()) - margin_px)), int(min(h, np.ceil(v.max()) + margin_px + 1))
    if u0 >= u1 or v0 >= v1:
        return mask
    vv, uu = np.mgrid[v0:v1, u0:u1]
    d = depth_m[v0:v1, u0:u1]
    good = d > 0
    if not good.any():
        return mask
    x = (uu[good] - K[0, 2]) / K[0, 0] * d[good]
    y = (vv[good] - K[1, 2]) / K[1, 1] * d[good]
    p = np.stack([x, y, d[good]], -1) @ T[:3, :3].T + T[:3, 3]
    tree = cKDTree(centres)
    rmax = float(radii.max()) + pad
    inside = np.zeros(len(p), bool)
    for i, nb in enumerate(tree.query_ball_point(p, rmax)):
        if nb:
            inside[i] = bool((np.linalg.norm(centres[nb] - p[i], axis=1) <= radii[nb] + pad).any())
    sub = np.zeros_like(good)
    sub[good] = inside
    mask[v0:v1, u0:u1] = sub
    return mask


class FruitTruth:
    """The target fruit's body centre and surface samples (base frame) at a qpos — `t80_eval.T80Truth` sampler."""

    def __init__(self, model_xml: str, body: str):
        from AG3S.ag3s.experiments.studies.t80_eval import build_truth

        self.gt = build_truth(model_xml, extra=False)
        self.idx = [i for i, (k, n, b) in enumerate(self.gt.kind) if n == body]
        if not self.idx:
            raise ValueError(f"{model_xml}: no collision geom of body {body!r}")
        self.body = self.gt.kind[self.idx[0]][2]

    def at(self, qpos):
        samples, _ = self.gt.at(np.asarray(qpos, float), only=self.idx)
        R, o = self.gt.base_pose
        centre = (self.gt.d.xpos[self.body] - o) @ R
        return centre, np.vstack([samples[i] for i in self.idx])


def measure_run(item: dict, *, deltas_mm, samples: int, robot_pad: float, gt_surface: float, gt_moved: float,
                gt_there: float = 0.5, gt_gone: float = 0.2, log=print) -> list[dict]:
    from scipy.spatial import cKDTree

    client = pathlib.Path(item["client"])
    chunks = episode_chunks(pathlib.Path(item["server"]), item["segment"])
    steps = sorted(glob.glob(str(client / "ag3s" / "run_0000" / "step_*.npz")),
                   key=lambda p: int(np.load(p)["t_step"]))
    man = json.loads((client / "manifest.json").read_text())
    fruit = man["scene"].get("episode_target_fruit") or "apple"
    truth = FruitTruth(man["scene"]["model_xml"], fruit)
    n = min(len(steps), len(chunks))
    log(f"[t81] {item['name']}: {n} requests ({len(chunks)} chunks, {len(steps)} steps), fruit {fruit}")
    centres_gt = []
    rows, prev, placed = [], None, False
    for j in range(n):
        st = np.load(steps[j])
        with np.load(chunks[j], allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"]))
            sph_c = np.asarray(z["sphere_centres"], np.float64).reshape(-1, 3) if "sphere_centres" in z.files \
                else np.zeros((0, 3))
            sph_r = np.asarray(z["sphere_radii"], np.float64).reshape(-1) if "sphere_radii" in z.files else np.zeros(0)
            tp = (np.asarray(z["target_points"], np.float64).reshape(-1, 3) if "target_points" in z.files
                  else np.zeros((0, 3)))
        g = s.get("grasp") or {}
        state = g.get("state")
        attached = bool(g.get("attached_in_constraints")) or g.get("attached") not in (None, False, {})
        placed = placed or state == "placed"
        m = s.get("manipulated") or {}
        centre, surf = truth.at(st["qpos"])
        centres_gt.append(centre)
        row = {"request": j, "seq": j + 1, "t_step": int(st["t_step"]), "grasp_state": state, "attached": attached,
               "subtask": (s.get("subtask") or {}).get("label"),
               "manipulated": {"id": m.get("id"), "state": m.get("state"),
                               "occluded_by_hand": (m.get("hand") or {}).get("occluded_by_hand"),
                               "frozen": ((s.get("admissibility") or {}).get("decision") or {}).get("frozen")},
               "gt": {"fruit_centre_base": [round(float(x), 4) for x in centre],
                      "disp_from_start_mm": round(float(np.linalg.norm(centre - centres_gt[0])) * 1e3, 1)}}
        ref = prev
        prev = ({"points": tp, "id": m.get("id"), "last_seen": m.get("last_seen_frame")}
                if m.get("id") is not None and len(tp) else None)
        if ref is None:
            row["applies"] = False
            row["why"] = "no_old_place"
            rows.append(row)
            continue
        row["applies"] = bool(state in APPLIES_STATES and not attached and not placed)
        if not row["applies"]:
            row["why"] = "held" if state == "held" or attached else "placed" if placed else f"state_{state}"
        pts = sample_points(ref["points"], samples)
        ls = ref["last_seen"]
        seen_at = centres_gt[ls] if isinstance(ls, int) and 0 <= ls < len(centres_gt) else centres_gt[0]
        row["gt"].update({
            "shift_since_seen_mm": round(float(np.linalg.norm(centre - seen_at)) * 1e3, 1),
            "ref_on_apple": round(float((cKDTree(surf).query(pts)[0] <= gt_surface).mean()), 3),
            "ref_centroid_to_centre_mm": round(float(np.linalg.norm(pts.mean(0) - centre)) * 1e3, 1)})
        row["gt"]["moved"] = bool(row["gt"]["shift_since_seen_mm"] > gt_moved)
        row["gt"]["apple_there"] = bool(row["gt"]["ref_on_apple"] >= gt_there)
        row["gt"]["vacated"] = bool(row["gt"]["ref_on_apple"] < gt_gone)
        views = []
        for cam in CAMS:
            if f"depth_{cam}" not in st.files:
                continue
            d = np.asarray(st[f"depth_{cam}"], np.float64) / 1000.0
            K = np.asarray(st[f"K_{cam}"], np.float64)
            T = np.asarray(st[f"T_base_cam_{cam}"], np.float64)
            views.append(VacancyView(cam, d, K, T, robot_mask_near(pts, d, K, T, sph_c, sph_r, robot_pad)))
        row["ref_n_points"], row["n_samples"] = int(len(ref["points"])), int(len(pts))
        row["evidence"] = {}
        for dm in deltas_mm:
            ev = judge(pts, views, delta=dm / 1000.0, min_points=1)
            row["evidence"][str(dm)] = {k: ev[k] for k in ("camera", "n_judged", "n_valid", "n_robot", "empty",
                                                            "occluded", "present", "per_camera")}
        rows.append(row)
    return rows


def simulate(rows: list[dict], *, delta_mm, p_empty, p_present, min_points, frames=2) -> list[int]:
    """Requests where the §1.4 rule would release on the recorded evidence (streak reset after a release)."""
    out, streak = [], 0
    for r in rows:
        ev = (r.get("evidence") or {}).get(str(delta_mm))
        if not r.get("applies") or ev is None:
            streak = 0
            continue
        e = dict(ev, decidable=ev["n_judged"] >= min_points and ev["n_valid"] > 0 and ev["empty"] is not None)
        streak = streak + 1 if vacant(e, p_empty=p_empty, p_present=p_present) else 0
        if streak >= frames:
            out.append(r["request"])
            streak = 0
    return out


def sweep(runs: dict, items: list[dict], grid: dict) -> list[dict]:
    kinds = {it["name"]: it["kind"] for it in items}
    out = []
    for dm, pe, pp, mp in itertools.product(grid["delta_mm"], grid["p_empty"], grid["p_present"],
                                            grid["min_points"]):
        res = {"delta_mm": dm, "p_empty": pe, "p_present": pp, "min_points": mp, "runs": {}}
        bad_success = wrong = 0
        hits = 0
        for name, rows in runs.items():
            rel = simulate(rows, delta_mm=dm, p_empty=pe, p_present=pp, min_points=mp)
            by = {r["request"]: r for r in rows}
            first = rel[0] if rel else None
            first_vac = next((r["request"] for r in rows if r.get("applies") and (r.get("gt") or {}).get("vacated")),
                             None)
            wrong_first = first is not None and by[first]["gt"].get("apple_there", False)
            wrong_all = [q for q in rel if by[q]["gt"].get("apple_there", False)]
            res["runs"][name] = {"releases": rel, "first": first, "first_gt_vacated_applies": first_vac,
                                 "first_is_wrong": wrong_first, "wrong_releases": wrong_all,
                                 "first_after_vacated": None if first is None or first_vac is None
                                 else first - first_vac}
            if kinds[name] == "success":
                bad_success += len(rel)
            else:
                hits += int(first is not None and not wrong_first)
            wrong += int(wrong_first)
        res.update({"success_releases": bad_success, "moved_runs_released": hits, "wrong_first_releases": wrong})
        out.append(res)
    return out


def _table(rows, delta_mm) -> str:
    head = ("| seq | latch | applies | manip state | cam | n judged | robot px | empty | occluded | present | "
            "GT ref on apple | GT shift since seen (mm) |")
    lines = [head, "|" + "---|" * 12]
    f = (lambda x: "-" if x is None else f"{x:.2f}")
    for r in rows:
        ev = (r.get("evidence") or {}).get(str(delta_mm))
        gt = r.get("gt") or {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["seq"], r.get("grasp_state"), "Y" if r.get("applies") else "-",
            (r.get("manipulated") or {}).get("state"),
            ev["camera"] if ev else "-", ev["n_judged"] if ev else "-", ev["n_robot"] if ev else "-",
            f(ev["empty"]) if ev else "-", f(ev["occluded"]) if ev else "-", f(ev["present"]) if ev else "-",
            gt.get("ref_on_apple", "-"), gt.get("shift_since_seen_mm", "-")))
    return "\n".join(lines)


def margins(runs: dict, items: list[dict], delta_mm, min_points) -> dict:
    """Per run at one δ: the largest empty / smallest present share over applying requests with the fruit there
    (success runs: all applying requests), and the 2-request windows (min empty, max present) once vacated."""
    kinds = {it["name"]: it["kind"] for it in items}
    out = {}
    for name, rows in runs.items():
        ok = []
        for r in rows:
            ev = (r.get("evidence") or {}).get(str(delta_mm))
            if r.get("applies") and ev is not None and ev["empty"] is not None and ev["n_judged"] >= min_points:
                ok.append((r, ev))
        there = [(r, ev) for r, ev in ok if kinds[name] == "success" or r["gt"].get("apple_there")]
        gone = [(r, ev) for r, ev in ok if r["gt"].get("vacated")]
        win = [(min(a[1]["empty"], b[1]["empty"]), max(a[1]["present"], b[1]["present"]), a[0]["seq"])
               for a, b in zip(gone, gone[1:]) if b[0]["request"] == a[0]["request"] + 1]
        out[name] = {"kind": kinds[name],
                     "there_max_empty": max((ev["empty"] for _, ev in there), default=None),
                     "there_max_empty_seq": max(there, key=lambda x: x[1]["empty"])[0]["seq"] if there else None,
                     "there_n": len(there),
                     "vacated_first_window": win[0] if win else None,
                     "vacated_best_window": max(win) if win else None,
                     "vacated_n": len(gone)}
    return out


def plot(runs: dict, delta_mm, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.3 * len(runs)), sharex=True)
    for ax, (name, rows) in zip(np.atleast_1d(axes), runs.items()):
        seq = [r["seq"] for r in rows]
        get = lambda k: [((r.get("evidence") or {}).get(str(delta_mm)) or {}).get(k) for r in rows]   # noqa: E731
        e, pr, o = get("empty"), get("present"), get("occluded")
        nan = lambda xs: [np.nan if x is None else x for x in xs]   # noqa: E731
        ax.plot(seq, nan(e), "-o", ms=3, label="empty", color="#2a6fdb")
        ax.plot(seq, nan(pr), "-s", ms=3, label="present", color="#e07b00")
        ax.plot(seq, nan(o), "-", lw=1, label="occluded", color="#888888")
        on = [r["gt"].get("ref_on_apple", np.nan) if r.get("gt") else np.nan for r in rows]
        ax.plot(seq, on, ":", lw=1.5, label="GT ref on apple", color="#1b9e77")
        for r in rows:
            if not r.get("applies"):
                ax.axvspan(r["seq"] - 0.5, r["seq"] + 0.5, color="#000000", alpha=0.06, lw=0)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("share")
        ax.set_title(f"{name}  (delta {delta_mm:g} mm; grey = rule does not apply: held / placed / no old place)",
                     fontsize=9)
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=7, ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel("server request (seq)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--items", type=pathlib.Path, default=None, help="JSON list like DEFAULT_ITEMS (default: §1 runs)")
    ap.add_argument("--deltas-mm", type=float, nargs="+", default=[10.0, 15.0, 20.0, 30.0])
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--robot-pad-mm", type=float, default=5.0)
    ap.add_argument("--gt-surface-mm", type=float, default=10.0)
    ap.add_argument("--gt-moved-mm", type=float, default=30.0)
    ap.add_argument("--gt-there", type=float, default=0.5)
    ap.add_argument("--gt-gone", type=float, default=0.2)
    a = ap.parse_args(argv)
    items = json.loads(a.items.read_text()) if a.items else DEFAULT_ITEMS
    a.out.mkdir(parents=True, exist_ok=True)
    deltas = [float(x) for x in a.deltas_mm]
    runs = {}
    for it in items:
        runs[it["name"]] = measure_run(it, deltas_mm=deltas, samples=a.samples, robot_pad=a.robot_pad_mm / 1000.0,
                                       gt_surface=a.gt_surface_mm / 1000.0, gt_moved=a.gt_moved_mm / 1000.0,
                                       gt_there=a.gt_there, gt_gone=a.gt_gone)
        (a.out / f"{it['name']}.rows.json").write_text(json.dumps(runs[it["name"]], indent=1))
    grid = {"delta_mm": deltas, "p_empty": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "p_present": [0.05, 0.1, 0.2, 0.3],
            "min_points": [5, 10, 20, 40]}
    sw = sweep(runs, items, grid)
    meta = {"items": items, "args": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(a).items()},
            "grid": grid, "frames": 2}
    (a.out / "sweep.json").write_text(json.dumps({"meta": meta, "settings": sw}, indent=1))
    good = [x for x in sw if x["success_releases"] == 0 and x["wrong_first_releases"] == 0
            and not any(v["wrong_releases"] for v in x["runs"].values())
            and x["moved_runs_released"] == sum(1 for it in items if it["kind"] != "success")]
    mg = {str(dm): {str(mp): margins(runs, items, dm, mp) for mp in grid["min_points"]} for dm in deltas}
    (a.out / "summary.json").write_text(json.dumps({"n_settings": len(sw), "n_admissible_settings": len(good),
                                                    "admissible_settings": good, "margins": mg}, indent=1))
    for dm in deltas:
        plot(runs, dm, a.out / f"evidence_delta{dm:g}mm.png")
    md = [f"# T81 vacancy evidence per request (offline)\n", "Fractions: of the chosen camera's valid points "
          "(robot pixels = occluded). Old place = chunk j-1 target_points. GT = MuJoCo, client qpos.\n"]
    for name, rows in runs.items():
        for dm in deltas:
            md += [f"\n## {name} — delta {dm:g} mm\n", _table(rows, dm), ""]
    (a.out / "tables.md").write_text("\n".join(md))
    print(json.dumps({"out": str(a.out), "runs": {k: len(v) for k, v in runs.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
