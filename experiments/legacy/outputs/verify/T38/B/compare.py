"""T38 Phase B verifier comparison (verifier's own code). usage: compare.py <runA dir> <runB dir> <out.json>

Per chunk of pass T (seq 1..N, after warm-up):
  actions (raw bytes, dtype, shape) · verdict fields · worst row (max_violation_pair) · max_violation_m · grasp record ·
  has_target · AG3S status · manipulated (id, state, centroid, geometry) · target points · ESDF tiers (distance / label
  array sha256, origin, voxel, shape, max_distance, outside, label names) · fused cloud arrays · per-camera cloud ·
  per-camera depth robot masks (pixel XOR count) · full response leaves.
Record: every chunk_*.npz of rec/run_0000 — every array byte-compared; summary_json leaf-compared.
Leaves whose key looks like a clock or a duration are listed separately (TIME_KEY), never silently dropped.
"""
import ast, json, pathlib, re, sys
import numpy as np

TIME_KEY = re.compile(r"(_ms$|^ms$|_s$|_sec$|_at$|^t$|time|stamp|clock|elapsed|duration|wall|^infer$|^ag3s$|^trajopt$|^total$)")


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}")
    elif isinstance(o, list):
        if not o:
            yield p, "[]"
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}[{i}]")
    elif isinstance(o, str) and o.startswith("{'"):
        try:  # a dict stored as its Python repr (e.g. esdf_stats.target_free) -- compare its leaves, not the string
            v = ast.literal_eval(o)
        except (ValueError, SyntaxError):
            yield p, o
        else:
            yield from leaves(v, f"{p}<repr>")
    else:
        yield p, o


def is_time(path):
    keys = [k for k in re.split(r"[.\[\]]|<repr>", path) if k and not k.isdigit()]
    return any(TIME_KEY.search(k) for k in keys) or path.endswith(".f") and any(TIME_KEY.search(k) for k in keys[:-1])


def diff_tree(a, b):
    la, lb = dict(leaves(a)), dict(leaves(b))
    out = {"n_leaves_a": len(la), "n_leaves_b": len(lb), "non_time": [], "time": []}
    for k in sorted(set(la) | set(lb)):
        if la.get(k, "<missing>") != lb.get(k, "<missing>"):
            (out["time"] if is_time(k) else out["non_time"]).append(k)
    return out


def load(d):
    d = pathlib.Path(d)
    rows = {r["seq"]: r for r in map(json.loads, open(d / "pass_T.jsonl"))}
    acts = np.load(d / "actions_T.npz")
    masks = np.load(d / "masks_T.npz") if (d / "masks_T.npz").exists() else None
    return rows, acts, masks


def main():
    A, B, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    ra, aa, ma = load(A); rb, ab, mb = load(B)
    seqs = sorted(set(ra) & set(rb))
    items = ["actions_bytes", "verdict", "worst_row", "max_violation_m", "grasp", "has_target", "ag3s_status",
             "manipulated", "target_points", "esdf_tiers", "fused_cloud", "per_camera_cloud", "depth_masks", "response_non_time"]
    per = {k: 0 for k in items}
    bad = {k: [] for k in items}
    mask_px = {"pairs": 0, "xor_px_total": 0, "pairs_nonzero": 0}
    time_keys = set()
    n_tiers = []
    for s in seqs:
        x, y = ra[s], rb[s]
        k = f"seq{s:03d}"
        p, q = aa[k], ab[k]
        same = p.dtype == q.dtype and p.shape == q.shape and p.tobytes() == q.tobytes()
        res = {"actions_bytes": same}
        vx = {kk: x["resp"].get(kk) for kk in ("safe", "trajopt_status", "ag3s_status", "geometry_certified", "verdict_reasons", "verdict", "gate")}
        vy = {kk: y["resp"].get(kk) for kk in vx}
        res["verdict"] = vx == vy
        res["worst_row"] = x["resp"].get("max_violation_pair") == y["resp"].get("max_violation_pair")
        res["max_violation_m"] = x["resp"].get("max_violation_m") == y["resp"].get("max_violation_m")
        dg = diff_tree(x["grasp"], y["grasp"]); time_keys.update("grasp" + t for t in dg["time"])
        res["grasp"] = not dg["non_time"]
        res["has_target"] = x["has_target"] == y["has_target"]
        res["ag3s_status"] = x["ag3s_status"] == y["ag3s_status"]
        res["manipulated"] = x.get("manipulated") == y.get("manipulated")
        res["target_points"] = x.get("target_points") == y.get("target_points")
        fx, fy = x.get("field"), y.get("field")
        strip = lambda f: None if f is None else {"outside_distance": f["outside_distance"],
                                                  "tiers": [{kk: vv for kk, vv in t.items() if kk != "type"} for t in f["tiers"]]}
        res["esdf_tiers"] = strip(fx) == strip(fy)
        n_tiers.append(None if fx is None else len(fx["tiers"]))
        res["fused_cloud"] = x.get("fused") == y.get("fused")
        res["per_camera_cloud"] = x.get("per_camera") == y.get("per_camera")
        mok = x.get("masks") == y.get("masks")
        if ma is not None and mb is not None:
            for cam in (x.get("masks") or {}):
                kk = f"{k}/{cam}"
                if kk in ma.files and kk in mb.files:
                    xor = int(np.unpackbits(np.bitwise_xor(ma[kk], mb[kk])).sum())
                    mask_px["pairs"] += 1; mask_px["xor_px_total"] += xor; mask_px["pairs_nonzero"] += int(xor > 0)
                    mok = mok and xor == 0
        res["depth_masks"] = mok
        dr = diff_tree(x["resp"], y["resp"]); time_keys.update("resp" + t for t in dr["time"])
        res["response_non_time"] = not dr["non_time"]
        for it in items:
            if res[it]:
                per[it] += 1
            else:
                bad[it].append({"seq": s, **({"paths": (dr if it == "response_non_time" else dg)["non_time"][:20]}
                                            if it in ("response_non_time", "grasp") else {})})
    # record npz
    rec_a = sorted((pathlib.Path(A) / "rec" / "run_0000").glob("chunk_*.npz"))
    rec_b = sorted((pathlib.Path(B) / "rec" / "run_0000").glob("chunk_*.npz"))
    rec = {"n_a": len(rec_a), "n_b": len(rec_b), "names_equal": [p.name for p in rec_a] == [p.name for p in rec_b],
           "chunks_arrays_identical": 0, "n_arrays_compared": 0, "arrays_differing": [], "keysets_equal": 0,
           "summary_non_time_identical": 0, "summary_non_time_diffs": [], "summary_time_keys_differing": set()}
    for pa, pb in zip(rec_a, rec_b):
        za, zb = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
        rec["keysets_equal"] += int(set(za.files) == set(zb.files))
        ok = True
        for name in sorted(set(za.files) & set(zb.files)):
            if name == "summary_json":
                continue
            u, v = za[name], zb[name]
            rec["n_arrays_compared"] += 1
            if not (u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes()):
                ok = False; rec["arrays_differing"].append(f"{pa.name}:{name}")
        rec["chunks_arrays_identical"] += int(ok and set(za.files) == set(zb.files))
        if "summary_json" in za.files and "summary_json" in zb.files:
            d = diff_tree(json.loads(str(za["summary_json"])), json.loads(str(zb["summary_json"])))
            rec["summary_time_keys_differing"].update(d["time"])
            if d["non_time"]:
                rec["summary_non_time_diffs"].append({"chunk": pa.name, "paths": d["non_time"][:30]})
            else:
                rec["summary_non_time_identical"] += 1
    rec["summary_time_keys_differing"] = sorted(rec["summary_time_keys_differing"])
    out = {"a": A, "b": B, "n_chunks": len(seqs), "seqs": [seqs[0], seqs[-1]] if seqs else None,
           "same_count": per, "differing_seqs": {k: v for k, v in bad.items() if v},
           "depth_mask_pixels": mask_px, "n_esdf_tiers_per_chunk": sorted(set(t for t in n_tiers if t is not None)),
           "response_time_like_keys_differing": sorted(time_keys), "record_npz": rec}
    pathlib.Path(outp).write_text(json.dumps(out, indent=1))
    print(json.dumps({"n": len(seqs), "same": per, "masks": mask_px,
                      "rec": {k: (v if not isinstance(v, list) else len(v)) for k, v in rec.items()}}, indent=None))


if __name__ == "__main__":
    main()
