"""IFG-APPLE-FAIL helper: initial fruit / crate positions of IFG-APPLE scenes, rebuilt with the replay reset (hold_w.reset on the run snapshot client,
1000 warm-up mj_steps, the same as replay_contacts.py). Read only. Run with .venv-openpi-live, cwd = run snapshot, AG3S_STORAGE_ROOT set, CL_SNAP = snapshot,
PYTHONPATH = snapshot (+ integration/rby1/manipulation/src, third_party/openpi/src, openpi-client/src):
  python ifg_apple_scene_bodies.py <run>/verify/IFG-APPLE <ep,ep,...> <out.json>
  python ifg_apple_scene_bodies.py <kit dir with p/hold_w.py> --seeds <seeds.json> <out.json>   (IFG-CL3-SPLIT: episode / seed /
      obstacle config from seeds.json, no run directory read — two IFG-CL3 run directories are held by NFS-stuck processes)
"""
import json, sys, pathlib, os
import numpy as np, mujoco
V = pathlib.Path(sys.argv[1]); P = V / "p"; sys.path.insert(0, str(P))
SEEDS = json.load(open(sys.argv[3]))["items"] if sys.argv[2] == "--seeds" else None
EPS = [x["episode"] for x in SEEDS] if SEEDS else [int(x) for x in sys.argv[2].split(",")]
OUT = sys.argv[4] if SEEDS else sys.argv[3]; sys.argv = [sys.argv[0]]
import hold_w as HW
cl = HW.load_client()
out = {}
for ep in EPS:
    if SEEDS:
        it = next(x for x in SEEDS if x["episode"] == ep)
        seed, prof, cfg = int(it["seed"]), it.get("obstacle_profile"), (it.get("obstacle_config") or None)
    else:
        rd = [p for p in (V / "runs" / "C1aN").glob(f"C1aN_ep{ep}_s*") if p.is_dir()][0]
        man = json.load(open(rd / "manifest.json")); sc = man["scene"]; prof = sc.get("obstacle_profile")
        seed = int(np.load(rd / "traj.npz")["policy_seed"])
        st = rd.parent / f"{rd.name}.start"
        cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in st.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None) if st.exists() else None
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]
    for _ in range(1000):
        mujoco.mj_step(m, d)
    pos = {}
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if any(s in n for s in ("apple", "banana", "orange", "pear", "crate", "obstacle", "post", "pillar", "hurdle")):  # noqa
            pos[n] = [round(float(x), 4) for x in d.xpos[b]]
    out[ep] = pos; print(ep, json.dumps(pos), flush=True)
json.dump(out, open(OUT, "w"), indent=1)
