from common.workspace import workspace_path as _ag3s_workspace_path
import json, hashlib, re
O = _ag3s_workspace_path('outputs/verify/MERGE-o4')
T38 = _ag3s_workspace_path('outputs/verify/T38/B/baseline')
EXP = {"legacy": (14, 15, -29.031048280806342, "40798fb0a4d2"),
       "curobo": (10, 15, -9.171877401271193, "fe73bd7a6ba6")}
BASE_PIDS = {"2793673", "2805050", "2827656", "2846628"}
res = {}
for k, e in EXP.items():
    d = json.load(open(f"{O}/{k}.json")); fr = d["frames"]
    cb = [f["clearance_before_mm"] for f in fr]
    t38 = [f["clearance_before_mm"] for f in json.load(open(f"{T38}/{k}.json"))["frames"]]
    m = {"frames": len(fr),
         "violated_start": sum(c < 0 for c in cb),
         "has_target": sum(bool(f["target"]) for f in fr),
         "frame0_clearance_before_mm": cb[0],
         "clearance_before_15_sha1_12": hashlib.sha1(json.dumps(cb).encode()).hexdigest()[:12],
         "per_frame_clearance_before_equal_to_T38_V3": sum(a == b for a, b in zip(cb, t38)),
         "esdf_backend_frames": sorted({f["esdf_backend"] for f in fr})}
    log = open(f"{O}/{k}.log").read()
    m["log_line_all_spheres"] = re.search(r"전체 구\s*:.*", log).group(0).strip()
    m["log_status_line"] = re.search(r"상태:.*", log).group(0).strip()
    got = (m["violated_start"], m["has_target"], m["frame0_clearance_before_mm"], m["clearance_before_15_sha1_12"])
    m["expected"] = {"violated_start": e[0], "has_target": e[1], "frame0_clearance_before_mm": e[2], "clearance_before_15_sha1_12": e[3]}
    m["equal_to_expected"] = {n: g == x for n, g, x in zip(m["expected"], got, e)}
    # GPU
    tot, own, util = [], [], []
    for line in open(f"{O}/{k}.gpu_samples.txt"):
        p = line.split()
        if len(p) < 2: continue
        mu, ut = p[1].split(","); tot.append(int(mu)); util.append(int(ut))
        if len(p) >= 4:
            for kv in p[3].split(";"):
                if kv and kv.split(",")[0] not in BASE_PIDS:
                    own.append(int(kv.split(",")[1]))
    m["gpu"] = {"samples": len(tot), "total_used_MiB_max": max(tot), "total_used_MiB_min": min(tot),
                "new_process_MiB_max": max(own) if own else 0, "util_pct_max": max(util)}
    m["imports"] = {kk: vv for kk, vv in json.load(open(f"{O}/{k}.imports.json")).items() if kk != "modules"}
    res[k] = m
json.dump(res, open(f"{O}/summary.json", "w"), indent=1)
print(json.dumps(res, indent=1, ensure_ascii=False))
