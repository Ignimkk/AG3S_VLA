# copy of T43/S/p/scan_t42.py (unchanged logic) for T43 U: T43 V and T42-2 roots
"""T43 S — chunk index for T42-1 E3b runs (read-only). Server npz (record-constraints) <-> client run, by chain-log order and
content check (refined_chunk == planning 'actions', reference_chunk == planning 'actions_reference').
Out: outputs/verify/T43/S/raw/chunk_index_t42.json   {run_name: {server, chunks: [npz paths by seq], pair_max_abs, ...}}
Generic: also used for T40/T41 E3b dirs (argv: OUT ROOT [ROOT ...]; ROOT = .../runs/E3b)."""
import json, re, sys, pathlib
import numpy as np

def runs_of(root):
    root = pathlib.Path(root); order = []; srv = None
    for log in sorted(root.glob("chain_*.log")):
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"server up .* rec=(\S+)", line)
            if m: srv = pathlib.Path(m.group(1)); continue
            m = re.match(r"start (\S+) attempt=(\d+)", line)
            if m: order.append((srv, m.group(1), int(m.group(2))))
    return order

def main():
    out = pathlib.Path(sys.argv[1]); res = {}
    for root in sys.argv[2:]:
        root = pathlib.Path(root)
        order = runs_of(root)
        by_srv = {}
        for srv, name, att in order: by_srv.setdefault(srv, []).append((name, att))
        for srv, names in by_srv.items():
            files = sorted((srv / "server_constraints").glob("run_*/chunk_*.npz"))
            meta = []
            for f in files:
                with np.load(f) as z:
                    meta.append((f, int(z["t_step"]), np.asarray(z["refined_chunk"], float), np.asarray(z["reference_chunk"], float)))
            cur = 0
            for name, att in names:
                rd = root / name
                fr = rd / "frames.jsonl"
                if not fr.exists():
                    res[f"{root}/{name}"] = {"error": "no frames.jsonl (moved to failed/?)", "attempt": att}; continue
                plan = [json.loads(l) for l in fr.read_text().splitlines() if l.strip() and '"planning"' in l[:40]]
                plan = [p for p in plan if p.get("kind") == "planning"]
                if not plan: continue
                a0 = np.asarray(plan[0]["actions"], float)
                # find the first chunk from cur whose refined == planning seq 1 actions
                k = cur
                while k < len(meta) and not (meta[k][2].shape == a0.shape and np.abs(meta[k][2] - a0).max() < 1e-5):
                    k += 1
                if k >= len(meta):
                    res[f"{root}/{name}"] = {"error": "no matching first chunk", "server": str(srv)}; continue
                chunks = {}; pmax = 0.0; rmax = 0.0; ok = True
                for i, p in enumerate(plan):
                    if k + i >= len(meta): ok = False; break
                    f, ts, ref_c, ref_r = meta[k + i]
                    d = float(np.abs(ref_c - np.asarray(p["actions"], float)).max())
                    dr = float(np.abs(ref_r - np.asarray(p["actions_reference"], float)).max()) if p.get("actions_reference") is not None else None
                    pmax = max(pmax, d); rmax = max(rmax, dr or 0.0)
                    chunks[int(p["seq"])] = {"npz": str(f), "npz_t_step": ts, "plan_t_step": int(p["t_step"])}
                cur = k + len(plan)
                res[f"{root}/{name}"] = {"server": str(srv), "first_chunk_file": str(meta[k][0]), "n_plan": len(plan),
                                         "complete": ok, "refined_vs_actions_max_abs": pmax,
                                         "reference_vs_actions_reference_max_abs": rmax, "chunks": chunks}
                print(name, srv.name, meta[k][0].name, len(plan), f"{pmax:.2e} {rmax:.2e}", flush=True)
    out.write_text(json.dumps(res, indent=0))

if __name__ == "__main__":
    main()
