"""T43 U — run list (frozen at build time). Writes raw/runs_u.json.

Every run that has server records (E3b / V arms) in T40 G/J, T41 a/c, T43 V (V0-V3, H and B), T42-2 (place, invis, clear42c),
plus the T42-1 no-obstacle runs whose (episode, seed) equal a T42-2 run. E0 runs (no server) are listed for outcomes only.

Fields: key, campaign, cond, stage, obstacle (config, profile) or None, run_dir, chunks (seq -> npz), server dir, arm, fruit,
episode, seed, obstacle_margin_mm (server log 'TO obstacle_margin'), path_check (server log 'TO path check'),
gripper_cover (server log 'constraint gripper cover'), held_body_cover (server log)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, re

V = pathlib.Path(_ag3s_workspace_path('outputs/verify'))
OUT = V / "T43/U/raw/runs_u.json"
IX = {}
for f in (V / "T43/S/raw/chunk_index_t40t41.json", V / "T43/S/raw/chunk_index_t42.json", V / "T43/U/raw/chunk_index_t43v_t42_2.json"):
    IX.update(json.loads(f.read_text()))


def obstacle_of(start):
    if not start.exists():
        return None
    for l in start.read_text(errors="replace").splitlines():
        if l.startswith("obstacle="):
            v = l.split("=", 1)[1]
            if v and v != ":":
                c, p = v.split(":", 1)
                return [c, p]
    return None


def flags(srv):
    txt = (pathlib.Path(srv) / "serve_safe.log").read_text(errors="replace")
    m = re.search(r"TO obstacle_margin: ([\d.]+) mm", txt)
    return {"obstacle_margin_mm": float(m.group(1)) if m else 0.0,
            "path_check": "TO path check (T43 Q)" in txt,
            "gripper_cover": "constraint gripper cover (T43 R)" in txt,
            "held_body_cover": bool(re.search(r"held body cover|held-body-cover|T43 R2", txt))}


def campaign_of(rd):
    s = str(rd)
    for c in ("T40/G", "T40/J", "T41/a", "T41/c", "T43/V", "T42/2", "T42/1"):
        if f"/{c}/" in s:
            return c
    return "?"


def main():
    runs = {}
    for rd, ix in IX.items():
        if "error" in ix:
            continue
        camp = campaign_of(rd)
        rdp = pathlib.Path(rd)
        name = rdp.name
        m = re.match(r"(\w+)_ep(\d+)_s(\d+)", name)
        ep, seed = int(m.group(2)), int(m.group(3))
        stage = rdp.parent.name
        if camp == "T42/1":
            continue  # added below only when paired with T42-2
        man = json.loads((rdp / "manifest.json").read_text())
        ob = obstacle_of(rdp.parent / f"{name}.start")
        runs[f"{camp}/{stage}/{name}"] = {"campaign": camp, "cond": m.group(1), "stage": stage, "episode": ep, "seed": seed,
                                         "run_dir": rd, "server": ix["server"], "chunks": ix["chunks"], "obstacle": ob,
                                         "arm": man["scene"]["episode_used_arm"], "fruit": man["scene"]["episode_target_fruit"],
                                         **flags(ix["server"])}
    t422 = {(r["episode"], r["seed"]) for r in runs.values() if r["campaign"] == "T42/2"}
    for rd, ix in IX.items():
        if campaign_of(rd) != "T42/1" or "error" in ix:
            continue
        rdp = pathlib.Path(rd); m = re.match(r"(\w+)_ep(\d+)_s(\d+)", rdp.name)
        ep, seed = int(m.group(2)), int(m.group(3))
        if (ep, seed) not in t422:
            continue
        man = json.loads((rdp / "manifest.json").read_text())
        runs[f"T42/1/noobs/{rdp.name}"] = {"campaign": "T42/1", "cond": m.group(1), "stage": "noobs", "episode": ep, "seed": seed,
                                          "run_dir": rd, "server": ix["server"], "chunks": ix["chunks"], "obstacle": None,
                                          "arm": man["scene"]["episode_used_arm"], "fruit": man["scene"]["episode_target_fruit"],
                                          **flags(ix["server"])}
    # E0 (outcome only)
    e0 = {}
    for root in (V / "T40/G/runs/E0", V / "T43/V/runs/E0", V / "T42/2/runs/E0", V / "T40/J/runs/E0"):
        for done in sorted(root.glob("*/*.done")):
            rd = done.with_suffix("")
            if not (rd / "frames.jsonl").exists():
                continue
            m = re.match(r"(\w+)_ep(\d+)_s(\d+)", rd.name)
            man = json.loads((rd / "manifest.json").read_text())
            e0[f"{campaign_of(rd)}/{rd.parent.name}/{rd.name}"] = {
                "campaign": campaign_of(rd), "cond": "E0", "stage": rd.parent.name, "episode": int(m.group(2)), "seed": int(m.group(3)),
                "run_dir": str(rd), "obstacle": obstacle_of(rd.parent / f"{rd.name}.start"),
                "arm": man["scene"]["episode_used_arm"], "fruit": man["scene"]["episode_target_fruit"]}
    OUT.write_text(json.dumps({"doc": __doc__, "server_runs": runs, "e0_runs": e0}, indent=0))
    from collections import Counter
    print(len(runs), Counter((r["campaign"], r["stage"], r["obstacle"] is not None) for r in runs.values()))
    print(len(e0), Counter((r["campaign"], r["stage"], r["obstacle"] is not None) for r in e0.values()))
    print(Counter((r["campaign"], r["obstacle_margin_mm"], r["path_check"], r["gripper_cover"], r["held_body_cover"]) for r in runs.values()))


if __name__ == "__main__":
    main()
