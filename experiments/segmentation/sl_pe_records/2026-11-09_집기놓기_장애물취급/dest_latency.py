"""(offline, 2026-11-09 05:1x dev-4a) per held episode: first held call, call at which perception registered the destination
(summary 'destination'), and calls refused by an obstacle row whose GT owner is the destination (rowowner_<tag>.json) before /
after the registration. usage: python dest_latency.py <tag> <run glob> [...]"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS
HERE = pathlib.Path(__file__).resolve().parent
R = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
for tag, glob in zip(sys.argv[1::2], sys.argv[2::2]):
    ro = json.loads((HERE / f"rowowner_{tag}.json").read_text())
    for run in [p for p in sorted(R.glob(glob)) if (p / "episodes.jsonl").exists() and not p.name.startswith("INVALID")]:
        eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
        order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
        ok = {o: e["official_success"] for o, e in zip(order, eps)}
        ch = PS.episode_chunks(run, order)
        for stem in order:
            first_held = reg = None
            for ts, cp in ch.get(stem, []):
                s = json.loads(str(np.load(cp)["summary_json"]))
                if s["grasp"]["state"] == "held" and first_held is None: first_held = ts
                if first_held is not None and reg is None and isinstance(s.get("destination"), dict): reg = ts
            if first_held is None: continue
            rows = next(v["rows"] for k, v in ro.items() if k.startswith(run.name[:60]) and k.endswith(stem))
            ref = [r["seq"] for r in rows if (r["changed"] or r["hold"]) and r.get("owner") == "destination" and r["reason"] == "obstacle"]
            print(f"{tag:12s} {stem[-6:]} {'ok ' if ok[stem] else 'FAIL'} held@{first_held:3d} dest@{reg} | plate refusals before {[x for x in ref if reg is None or x < reg]} after {[x for x in ref if reg is not None and x >= reg]} | calls {len(rows)}")
