"""Aggregate raw/{main,arms}_*.json (+npz) of replay_i2b.py with the V-I2 raw (head_*, wt_*)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, os, sys, collections
import numpy as np

R = _ag3s_workspace_path('outputs/verify/T44/I2b/raw')
R2 = _ag3s_workspace_path('outputs/verify/T44/I2/raw')
OUT = _ag3s_workspace_path('outputs/verify/T44/I2b')
CONTACT = {"left": ("ee_left", "ee_finger_l1", "ee_finger_l2"), "right": ("ee_right", "ee_finger_r1", "ee_finger_r2")}
deg = lambda a, b: float(np.degrees(np.abs(a - b).max())) if a.shape == b.shape else float("nan")


def link_class(name, rec):
    """class of a robot link relative to the holding hand record"""
    side = rec["side"]
    other = "right" if side == "left" else "left"
    if name in CONTACT[side]:
        return "held_gripper"
    if name in (rec["parent_links"] or [])[:1]:
        return "held_parent1"          # first parent (N=1)
    if name in (rec["parent_links"] or [])[1:]:
        return "held_parent2plus"
    if name in CONTACT[other]:
        return "other_gripper"
    if name.startswith(f"link_{other}_arm") or name.startswith(f"link_{side}_arm"):
        return "arm_rest_" + ("held_side" if f"_{side}_" in name else "other_side")
    return "rest"


def load(model):
    runs = sorted(os.path.basename(p)[len(model) + 1:-5] for p in glob.glob(f"{R}/{model}_*.json"))
    return runs


def agg(model):
    runs = load(model)
    P = []   # per chunk records
    cmp_ = collections.defaultdict(lambda: dict(n_chunks=0, own_differs=0, traj_differs=0, status_differs=0,
                                               hold_differs=0, max_joint_deg=0.0, n_collision_rows_differs=0))
    for tag in runs:
        nw = json.load(open(f"{R}/{model}_{tag}.json")); nz = np.load(f"{R}/{model}_{tag}.npz")
        have = os.path.exists(f"{R2}/head_{tag}.json")
        if model == "main" and have:
            hd = json.load(open(f"{R2}/head_{tag}.json")); hz = np.load(f"{R2}/head_{tag}.npz")
            ow = json.load(open(f"{R2}/wt_{tag}.json")); oz = np.load(f"{R2}/wt_{tag}.npz")
            hmap = {c["seq"]: c for c in hd["chunks"] if "arms" in c}
            omap = {c["seq"]: c for c in ow["chunks"] if "arms" in c}
        for c in nw["chunks"]:
            if "skipped" in c:
                continue
            s = c["seq"]; A = c["arms"]
            if model == "main" and have:
                pairs = [("head_off_vs_new_off", hmap[s]["arms"]["off"], hz[f"s{s}_off_own"], hz[f"s{s}_off_traj"], "off"),
                         ("head_off_vs_new_offR", hmap[s]["arms"]["off"], hz[f"s{s}_off_own"], hz[f"s{s}_off_traj"], "offR"),
                         ("head_sw_vs_new_sw", hmap[s]["arms"]["sw"], hz[f"s{s}_sw_own"], hz[f"s{s}_sw_traj"], "sw"),
                         ("head_sw_vs_new_swR", hmap[s]["arms"]["sw"], hz[f"s{s}_sw_own"], hz[f"s{s}_sw_traj"], "swR"),
                         ("viI2wt_off_vs_new_off", omap[s]["arms"]["off"], oz[f"s{s}_off_own"], oz[f"s{s}_off_traj"], "off"),
                         ("viI2wt_off_vs_new_offR", omap[s]["arms"]["off"], oz[f"s{s}_off_own"], oz[f"s{s}_off_traj"], "offR"),
                         ("viI2wt_sw_vs_new_sw", omap[s]["arms"]["sw"], oz[f"s{s}_sw_own"], oz[f"s{s}_sw_traj"], "sw"),
                         ("viI2wt_sw_vs_new_swR", omap[s]["arms"]["sw"], oz[f"s{s}_sw_own"], oz[f"s{s}_sw_traj"], "swR"),
                         # control: the I2 strict arms re-run in the new tree vs the V-I2 raw
                         ("viI2wt_strict_vs_new_strict_i2", omap[s]["arms"]["strict"], oz[f"s{s}_strict_own"], oz[f"s{s}_strict_traj"], "strict_i2"),
                         ("viI2wt_swS_vs_new_swS_i2", omap[s]["arms"]["swS"], oz[f"s{s}_swS_own"], oz[f"s{s}_swS_traj"], "swS_i2")]
                for name, ra, oa, ta, arm in pairs:
                    d = cmp_[name]; d["n_chunks"] += 1
                    rb = A[arm]; ob = nz[f"s{s}_{arm}_own"]; tb = nz[f"s{s}_{arm}_traj"]
                    d["own_differs"] += int(not np.array_equal(oa, ob))
                    d["traj_differs"] += int(not np.array_equal(ta, tb))
                    d["status_differs"] += int(ra["status"] != rb["status"])
                    d["hold_differs"] += int(ra["hold_candidate"] != rb["hold_candidate"])
                    d["n_collision_rows_differs"] += int(ra["n_collision_rows"] != rb["n_collision_rows"])
                    if oa.shape == ob.shape:
                        d["max_joint_deg"] = max(d["max_joint_deg"], deg(oa, ob))
            rec = dict(run=tag, seq=s, held=c["held"], has_target=c["has_target"], hold_link=c["hold_link"],
                       manifest_hand=c["manifest_hand"], held_record=c["held_record"], rows_links=c["rows_links"])
            hr = c["held_record"]
            rec["side_matches_manifest"] = None if hr is None else (hr["side"] == c["manifest_hand"])
            ref = "off" if "off" in A else "offR"
            def own(a): return nz[f"s{s}_{a}_own"]
            def st(a): return A[a]["status"]
            rec["strict"] = {}
            for v in ("i2", "i2s", "b1", "b0"):
                a = f"strict_{v}"
                if a not in A:
                    continue
                rec["strict"][v] = dict(own_differs=not np.array_equal(own(ref), own(a)), deg=deg(own(ref), own(a)),
                                        status=(st(ref), st(a)), hold=(A[ref]["hold_candidate"], A[a]["hold_candidate"]),
                                        n_tail_target_unauthorized=(A[a]["lookahead"] or {}).get("n_tail_target_unauthorized"),
                                        n_tail_dropped=((A[ref]["lookahead"] or {}).get("n_tail_dropped"), (A[a]["lookahead"] or {}).get("n_tail_dropped")))
            if "strict_i2" in A and "strict_b1" in A:
                rec["b1_vs_i2"] = dict(own_differs=not np.array_equal(own("strict_i2"), own("strict_b1")), deg=deg(own("strict_i2"), own("strict_b1")),
                                       status=(st("strict_i2"), st("strict_b1")))
            if "strict_i2s" in A and "strict_b1" in A:
                rec["b1_vs_i2s"] = dict(own_differs=not np.array_equal(own("strict_i2s"), own("strict_b1")), deg=deg(own("strict_i2s"), own("strict_b1")),
                                        status=(st("strict_i2s"), st("strict_b1")))
            if "strict_b0" in A and "strict_b1" in A:
                rec["b0_vs_b1"] = dict(own_differs=not np.array_equal(own("strict_b0"), own("strict_b1")), deg=deg(own("strict_b0"), own("strict_b1")),
                                       status=(st("strict_b0"), st("strict_b1")))
            pr = c.get("probes") or {}
            if not c["held"]:
                # non-held chunk: the I2 / I2b / parent-0 scenes are the same object, the i2 probe is the I2b probe
                for key in list(pr):
                    if key.endswith("_i2"):
                        pr[key[:-3] + "_b1"] = pr[key]; pr[key[:-3] + "_b0"] = pr[key]
            rec["probes"] = pr
            # sweep
            sw = {}
            for v in ("i2", "i2s", "b1"):
                a = f"swS_{v}"
                if a not in A: continue
                ro, rs = A["swR"], A[a]
                sw[v] = dict(off_excluded=ro["path_check"]["target_rows_excluded"], strict_excluded=rs["path_check"]["target_rows_excluded"],
                             strict_unauth_kept=rs["path_check"]["target_rows_unauthorized_kept"],
                             own_differs=not np.array_equal(own("swR"), own(a)), deg=deg(own("swR"), own(a)),
                             status=(st("swR"), st(a)), hold=(ro["hold_candidate"], rs["hold_candidate"]),
                             path_clear_mm=(ro["path_check"]["clearance_m"], rs["path_check"]["clearance_m"]),
                             off_links=ro["path_target_links"], strict_links=rs["path_target_links"],
                             window_violation_mm=(ro["window_violation_mm"], rs["window_violation_mm"]))
            rec["sw"] = sw
            P.append(rec)
    return runs, P, dict(cmp_)


def split_auth(P, variant):
    """sweep: excluded rows split by the permission of `variant` (rows named for the holding hand)."""
    tot = collections.Counter(); nonauth_links = collections.Counter(); chunks_nonauth = collections.Counter()
    for r in P:
        allowed = set(r["rows_links"]["b1"].keys())   # the true holding-hand permission for every variant
        sw = r["sw"].get(variant)
        if sw is None: continue
        for tag, key in (("off", "off_links"), ("strict", "strict_links")):
            L = sw[key] or {}
            a = sum(v for k, v in L.items() if k in allowed); n = sum(v for k, v in L.items() if k not in allowed)
            tot[tag + "_auth"] += a; tot[tag + "_nonauth"] += n
            if n: chunks_nonauth[tag] += 1
            if tag == "off":
                for k, v in L.items():
                    if k not in allowed: nonauth_links[k] += v
        pr = r["probes"]
        if "path_off" in pr and ("path_strict_" + variant) in pr:
            for tag, key in (("off", "path_off"), ("strict", "path_strict_" + variant)):
                L = pr[key]["excluded_links"]
                tot["p_" + tag + "_auth"] += sum(v for k, v in L.items() if k in allowed)
                tot["p_" + tag + "_nonauth"] += sum(v for k, v in L.items() if k not in allowed)
    return dict(tot), dict(nonauth_links), dict(chunks_nonauth)


def summarize(model):
    runs, P, cmp_ = agg(model)
    out = dict(model=model, n_runs=len(runs), n_chunks=len(P), n_held_chunks=sum(r["held"] for r in P), compare=cmp_)
    held = [r for r in P if r["held"]]
    out["held_side_vs_manifest_hand_mismatch_chunks"] = sum(1 for r in held if r["side_matches_manifest"] is False)
    out["hold_links"] = dict(collections.Counter(r["hold_link"] for r in held))
    out["parent_links_seen"] = {"|".join(k): v for k, v in collections.Counter(tuple(r["held_record"]["parent_links"]) for r in held).items()}
    # static rows
    rows = {}
    for v in ("i2", "i2s", "b1", "b0"):
        c = collections.Counter(); 
        for r in held:
            for k, n in r["rows_links"][v].items(): c[k] += n
        rows[v] = dict(c)
    out["rows_permitted_held_chunks_sum_by_link"] = rows
    out["n_chunks_rows_differ_b0_b1"] = sum(1 for r in held if r["rows_links"]["b0"] != r["rows_links"]["b1"])
    out["n_rows_differ_b0_b1_total"] = int(sum(sum(r["rows_links"]["b1"].values()) - sum(r["rows_links"]["b0"].values()) for r in held))
    out["n_chunks_rows_differ_i2_b1"] = sum(1 for r in held if r["rows_links"]["i2"] != r["rows_links"]["b1"])
    out["n_rows_differ_i2_b1_total_abs"] = int(sum(abs(sum(r["rows_links"]["b1"].values()) - sum(r["rows_links"]["i2"].values())) for r in held))
    # lookahead probes
    la = {}
    for ref in ("reference", "off_refined"):
        for v in ("i2", "i2s", "b1", "b0"):
            k = f"lookahead_{ref}_{v}"
            rr = [r["probes"][k] for r in P if k in r["probes"]]
            cls = collections.Counter(); drop_cls = collections.Counter(); chunks_cls = collections.defaultdict(set)
            bylink = collections.Counter(); rev = 0
            for r in P:
                if k not in r["probes"]: continue
                x = r["probes"][k]; rev += x["reverse_changed"]
                for nm, (doff, don, ntail) in x["dropped_off_dropped_on_by_link"].items():
                    bylink[nm] += doff - don          # rows kept only by strict (reverse_changed == 0 checked below)
                    if not r["held"]: continue
                    nm2 = "held_query" if nm.startswith("attached:") else nm
                    cl = "held_query" if nm2 == "held_query" else link_class(nm, r["held_record"])
                    drop_cls[cl + "|dropped_off"] += doff; drop_cls[cl + "|dropped_strict"] += don
                    if doff - don:
                        cls[cl] += doff - don; chunks_cls[cl].add((r["run"], r["seq"]))
            la[k] = dict(n_chunks=len(rr), rows_changed_total=int(sum(x["n_rows_changed"] for x in rr)),
                         chunks_changed=sum(1 for x in rr if x["n_rows_changed"] > 0),
                         reverse_changed_total=int(sum(x["reverse_changed"] for x in rr)),
                         tail_target_unauthorized_total=int(sum(x["n_tail_target_unauthorized"] or 0 for x in rr)),
                         chunks_tail_unauthorized_gt0=sum(1 for x in rr if (x["n_tail_target_unauthorized"] or 0) > 0),
                         changed_by_class_rows=dict(cls), changed_by_class_chunks={a: len(b) for a, b in chunks_cls.items()},
                         tail_rows_dropped_by_class=dict(drop_cls),
                         changed_by_link={a: b for a, b in bylink.items() if b}, reverse_changed_check=rev,
                         value_diff_max_mm=max([x["value_diff_max_mm_on_common"] for x in rr] or [0.0]))
    out["lookahead_probes"] = la
    # solve (V11AB config, 8-step window with look-ahead)
    sv = {}
    for v in ("i2", "i2s", "b1", "b0"):
        rr = [r["strict"][v] for r in P if v in r["strict"]]
        sv[v] = dict(n_chunks=len(rr), refined_chunks_changed=sum(1 for x in rr if x["own_differs"]),
                     refined_joint_deg_max=max([x["deg"] for x in rr] or [0.0]),
                     status_changes=sum(1 for x in rr if x["status"][0] != x["status"][1]),
                     hold_changes=sum(1 for x in rr if x["hold"][0] != x["hold"][1]),
                     n_tail_target_unauthorized_total=int(sum(x["n_tail_target_unauthorized"] or 0 for x in rr)),
                     chunks_tail_unauthorized_gt0=sum(1 for x in rr if (x["n_tail_target_unauthorized"] or 0) > 0),
                     hold_candidate_off=sum(1 for x in rr if x["hold"][0]), hold_candidate_strict=sum(1 for x in rr if x["hold"][1]))
    rr = [r["b1_vs_i2"] for r in P if "b1_vs_i2" in r]
    sv["b1_vs_i2"] = dict(refined_chunks_differ=sum(1 for x in rr if x["own_differs"]), refined_joint_deg_max=max([x["deg"] for x in rr] or [0.0]),
                          status_differs=sum(1 for x in rr if x["status"][0] != x["status"][1]))
    rr = [r["b1_vs_i2s"] for r in P if "b1_vs_i2s" in r]
    sv["b1_vs_i2s"] = dict(n_chunks=len(rr), refined_chunks_differ=sum(1 for x in rr if x["own_differs"]), refined_joint_deg_max=max([x["deg"] for x in rr] or [0.0]),
                           status_differs=sum(1 for x in rr if x["status"][0] != x["status"][1]))
    rr = [r["b0_vs_b1"] for r in P if "b0_vs_b1" in r]
    sv["b0_vs_b1"] = dict(n_chunks=len(rr), refined_chunks_differ=sum(1 for x in rr if x["own_differs"]), refined_joint_deg_max=max([x["deg"] for x in rr] or [0.0]),
                          status_differs=sum(1 for x in rr if x["status"][0] != x["status"][1]))
    out["solve"] = sv
    # sweep
    swo = {}
    for v in ("i2", "i2s", "b1"):
        rr = [r["sw"][v] for r in P if v in r["sw"]]
        same = [r for r in P if "path_off" in r["probes"] and f"path_strict_{v}" in r["probes"]]
        unk = [x["strict_unauth_kept"] or 0 for x in rr]
        a, nl, cn = split_auth(P, v)
        swo[v] = dict(n_chunks=len(rr),
                      off_excluded_total=int(sum(x["off_excluded"] for x in rr)), off_chunks_excluded=sum(1 for x in rr if x["off_excluded"] > 0),
                      strict_excluded_total=int(sum(x["strict_excluded"] for x in rr)), strict_chunks_excluded=sum(1 for x in rr if x["strict_excluded"] > 0),
                      strict_unauthorized_kept_total=int(sum(unk)), strict_chunks_unauthorized_kept_gt0=sum(1 for u in unk if u > 0),
                      refined_chunks_changed=sum(1 for x in rr if x["own_differs"]), refined_joint_deg_max=max([x["deg"] for x in rr] or [0.0]),
                      status_changes=sum(1 for x in rr if x["status"][0] != x["status"][1]),
                      hold_changes=sum(1 for x in rr if x["hold"][0] != x["hold"][1]),
                      split_authorization=a, nonauth_links_excluded_off=nl, chunks_with_nonauth_excluded=cn,
                      same_traj=dict(n_chunks=len(same),
                                     off_excluded_total=int(sum(r["probes"]["path_off"]["target_rows_excluded"] for r in same)),
                                     strict_excluded_total=int(sum(r["probes"][f"path_strict_{v}"]["target_rows_excluded"] for r in same)),
                                     strict_unauth_kept_total=int(sum(r["probes"][f"path_strict_{v}"]["target_rows_unauthorized_kept"] or 0 for r in same)),
                                     new_path_violations=sum(1 for r in same if r["probes"][f"path_strict_{v}"]["violation"] and not r["probes"]["path_off"]["violation"]),
                                     path_violations_off=sum(1 for r in same if r["probes"]["path_off"]["violation"]),
                                     path_violations_strict=sum(1 for r in same if r["probes"][f"path_strict_{v}"]["violation"])))
        flips = [dict(run=r["run"], seq=r["seq"], status=r["sw"][v]["status"], path_clear_mm=r["sw"][v]["path_clear_mm"],
                      window_violation_mm=r["sw"][v]["window_violation_mm"], strict_unauth_kept=r["sw"][v]["strict_unauth_kept"],
                      off_links=r["sw"][v]["off_links"], strict_links=r["sw"][v]["strict_links"])
                 for r in P if v in r["sw"] and r["sw"][v]["status"][0] != r["sw"][v]["status"][1]]
        swo[v]["status_change_chunks"] = flips
        tgt = [r for r in P if r["run"].endswith("ep1834_s18341") and r["seq"] == 34]
        if tgt and v in tgt[0]["sw"]:
            x = tgt[0]["sw"][v]
            swo[v]["ep1834_s18341_seq34"] = dict(status=x["status"], hold=x["hold"], path_clear_mm=x["path_clear_mm"], window_violation_mm=x["window_violation_mm"],
                                                 held=tgt[0]["held"], strict_unauth_kept=x["strict_unauth_kept"], off_links=x["off_links"], strict_links=x["strict_links"],
                                                 off_excluded=x["off_excluded"], strict_excluded=x["strict_excluded"],
                                                 rows_links=tgt[0]["rows_links"][v], held_record=tgt[0]["held_record"])
    out["sweep"] = swo
    return out, P


if __name__ == "__main__":
    res = {}
    for m in sys.argv[1:] or ["main", "arms"]:
        if not glob.glob(f"{R}/{m}_*.json"): continue
        o, P = summarize(m)
        res[m] = o
        json.dump([{k: v for k, v in r.items()} for r in P], open(f"{OUT}/per_chunk_{m}.json", "w"), default=str)
    json.dump(res, open(f"{OUT}/agg.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:6000])
