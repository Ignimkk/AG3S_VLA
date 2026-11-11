"""(2026-11-11 dev-57, offline, CPU) What model clearance did the EXECUTED chunk have?
The pass-through test and the action QP both end in `_finish`; the verdict executes a returned chunk whose violation is
<= tau (100 mm in ...-t100). This scan reads, for every recorded call, the returned chunk's window clearance
(`to.metrics.clearance_m`, floor included: < 0 = inside the margin), its worst row (link / tier / step), the path
(pass / qp / hold), the grasp state, and for SafeLIBERO whether the obstacle moved during the steps this call executed.
Question: would a rule 'do not execute a chunk whose window clearance < -X mm' separate the collision calls from the
calls of successful episodes?  usage: python exec_clearance_scan.py <tag>
out: exec_clearance_scan_<tag>.json (one row per call)"""
import ast, collections, glob, json, sys
import numpy as np

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
OUT = "/mnt/dev/AG3S_VLA/outputs/2026-11-11_SOTA_개선/offline/"


def latest(pat):
    by = {}
    for d in sorted(glob.glob(B + pat)):
        if "INVALID" in d:
            continue
        by[d.split("/")[-1].split("_", 2)[2]] = d
    return sorted(by.values())


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return x or {}


def per_episode_chunks(d):
    per, cur, last = [], [], None
    for c in sorted(glob.glob(d + "/server_constraints/run_*/chunk_*.npz")):
        z = np.load(c)
        ts = int(z["t_step"])
        if last is not None and ts <= last:
            per.append(cur)
            cur = []
        cur.append((ts, json.loads(str(z["summary_json"]))))
        last = ts
    per.append(cur)
    return per


def rows_of(d, bench):
    eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
    per = per_episode_chunks(d)
    if len(per) != len(eps):
        print("  !! episode/chunk mismatch", d.split("/")[-1], len(eps), len(per))
        return []
    out = []
    for e, ch in zip(eps, per):
        s = e.get("safelibero") or {}
        ct = s.get("collide_t") if s.get("collided") else None
        cf = glob.glob(f"{d}/episodes/*_t{e['task_id']}_e{e['episode_idx']}_calls.jsonl")
        cs = {}
        if cf:
            for x in open(cf[0]):
                x = json.loads(x)
                cs[x["seq"]] = x
        seqs = sorted(cs)
        hit = None
        if ct is not None and seqs:
            hit = max((q for q in seqs if cs[q]["obs_index"] <= ct), default=None)
        for ts, sm in ch:
            to = sm.get("to") or {}
            m = to.get("metrics") or {}
            pc = lit(m.get("passthrough_clear"))
            ob = pc.get("obstacle_rows") or {}
            mv = lit(m.get("max_violation_pair"))
            qp = lit(m.get("ee_action_qp"))
            fb = sm.get("exec_feedback") or {}
            c = cs.get(ts) or {}
            out.append(dict(
                bench=bench, run=d.split("/")[-1].split("_", 2)[2], task=int(e["task_id"]), ep=int(e["episode_idx"]),
                init=e.get("init_state_index"), success=bool(e["official_success"]), collided=bool(s.get("collided")),
                seq=ts, hit=(ts == hit), before_hit=(hit is None or ts <= hit),
                path=("hold" if c.get("hold") else "pass" if pc.get("applied") else "qp" if qp else "other"),
                reason=pc.get("reason"), status=to.get("status"),
                clr=None if m.get("clearance_m") is None else round(1000 * float(m["clearance_m"]), 2),
                link=mv.get("link"), tier=mv.get("tier"), step=mv.get("step"), block=mv.get("block"),
                ref_viol=None if pc.get("reference_violation_m") is None else round(1000 * float(pc["reference_violation_m"]), 2),
                obs_viol=None if ob.get("violation_m") is None else round(1000 * float(ob["violation_m"]), 2),
                slack=None if not qp else round(1000 * float(qp.get("slack_max_m") or 0.0), 2),
                sat=None if not qp else qp.get("saturated_components"),
                grasp=(sm.get("grasp") or {}).get("state"), label=(sm.get("subtask") or {}).get("label")))
    return out


def main(tag):
    benches = {"LIBERO27": f"*_gtto-{tag}-libero9s[789]-*", "SafeLIBERO11": f"*_gtto-{tag}-safelibero11-t*",
               "LS10": f"*_gtto-{tag}-ls10-*", "E1a40": f"*_qe-{tag}-e1a-spatial-LI-t*"}
    allrows = []
    for b, pat in benches.items():
        for d in latest(pat):
            allrows += rows_of(d, b)
    json.dump(allrows, open(OUT + f"exec_clearance_scan_{tag}.json", "w"))
    X = (0, 5, 10, 15, 20, 30, 50)
    for b in benches:
        r = [x for x in allrows if x["bench"] == b and x["clr"] is not None and x["path"] != "hold"]
        print(f"== {b}: calls {len(r)}  path {dict(collections.Counter(x['path'] for x in r))}")
        groups = {"success, no collision": [x for x in r if x["success"] and not x["collided"]],
                  "collision call (hit)": [x for x in r if x["hit"]],
                  "collided ep, calls before hit": [x for x in r if x["collided"] and x["before_hit"] and not x["hit"]]}
        for g, rr in groups.items():
            if not rr:
                continue
            a = np.array([x["clr"] for x in rr])
            print(f"   {g:32s} n {len(rr):4d}  clr<-X: " + "  ".join(f"{t}:{int((a < -t).sum())}" for t in X))
            for p in ("pass", "qp"):
                a = np.array([x["clr"] for x in rr if x["path"] == p])
                if a.size:
                    print(f"      {p:5s} n {a.size:4d}  clr<-X: " + "  ".join(f"{t}:{int((a < -t).sum())}" for t in X))
        eps_ok = collections.defaultdict(list)
        for x in groups["success, no collision"]:
            eps_ok[(x["run"], x["task"], x["ep"])].append(x["clr"])
        print("   successful episodes whose worst executed call has clr < -X: " + "  ".join(
            f"{t}:{sum(min(v) < -t for v in eps_ok.values())}/{len(eps_ok)}" for t in X))


if __name__ == "__main__":
    main(sys.argv[1])
