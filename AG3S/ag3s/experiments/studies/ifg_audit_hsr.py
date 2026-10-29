"""IFG-AUDIT-HSR: how accurately and how fast the AG3S stage trackers (grasp latch + sub-task probe) follow the true task
stage on recorded closed-loop runs. CPU only, read only (inputs are never written).

  cd /mnt/dev/AG3S_VLA-development; PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python \
      -m AG3S.ag3s.experiments.studies.ifg_audit_hsr --run <RUN> --fig AG3S/ag3s/docs/figures/ifg-audit

Inputs (24 apple scenes, arm SPL = current serving): <RUN>/verify/IFG-APPLE-SPLIT
  replay/runs_SPL_SPL_ep*.npz   per control step (600): apple_pos, crate_pos/quat, grip_cmd (used arm, 1 = open),
                                cnt_apple-hand, finger_open_norm, plan_index
  replay/runs_SPL_SPL_ep*.json  server_chunk_alignment.block -> request k is chunk_dir/chunk_{k+offset:05d}.npz
  server chunk npz summary_json  grasp (latch), subtask (probe), manipulated
  analysis/SPL_ep*.json         success / failure_mode (cross-check only)
  p/analyze_t42.py              in_crate rule (imported, not copied)

GT definitions (per control step t, 15 Hz; request k is evaluated at its planning step t = 8k):
  close cycle   [s, e): s = a step with grip_cmd < 0.5 (closing command), e = first later step with grip_cmd >= 0.9 (open command).
                The carry command is ~0.58, so 0.5 cannot mark the end of a close.
  lift onset    first step t where, for LIFT_L = 5 consecutive steps, apple z >= z0 + 10 mm AND cnt_apple-hand > 0
                (z0 = apple z at step 0).
  release       first step s > lift onset with cnt_apple-hand == 0 for REL_N = 8 consecutive steps (contact lost and not
                regained within one chunk). HELD = [lift onset, release).
  place         if the apple is in the crate (kit in_crate rule) at step 599: start of that final in-crate streak, but not
                earlier than the last release. home onset = first step >= place with grip_cmd >= 0.9.
  stage(t)      APPROACH      t < first closing command, or between close cycles before any lift (re-approach)
                CLOSING       inside a close cycle that contains a lift onset, before the lift
                FAILED_GRASP  inside a close cycle that contains no lift onset, before any lift
                HELD          [lift onset, release)
                RELEASED      [last release, place) when the apple ends in the crate (falling into the crate)
                DROPPED       after a release, apple not (yet) placed and not RELEASED
                PLACED        [place, home onset);  HOME  [home onset, end)
  task phase    pick = APPROACH, CLOSING, FAILED_GRASP, DROPPED; place = HELD, RELEASED; home = PLACED, HOME.
  latch phase   searching / latched / closing -> pick; held -> place; placed -> home.
Latency: GT event at step t_e; first request able to observe it k_GT = ceil(t_e / 8); latency in requests = k_latch - k_GT,
in seconds = (8 k_latch - t_e) / 15.
"""
import argparse, ast, datetime, glob, importlib.util, json, math, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
HZ = 15.0; SPR = 8                                   # control rate, steps per request
CLOSE_TH, OPEN_TH = 0.5, 0.9
LIFT_DZ, LIFT_L, REL_N = 0.010, 5, 8
STAGES = ["APPROACH", "CLOSING", "HELD", "RELEASED", "PLACED", "HOME", "FAILED_GRASP", "DROPPED"]
LATCH = ["searching", "latched", "closing", "held", "placed"]
PHASES = ["pick", "place", "home"]
STAGE_PHASE = {"APPROACH": "pick", "CLOSING": "pick", "FAILED_GRASP": "pick", "DROPPED": "pick",
               "HELD": "place", "RELEASED": "place", "PLACED": "home", "HOME": "home"}
LATCH_PHASE = {"searching": "pick", "latched": "pick", "closing": "pick", "held": "place", "placed": "home"}
# latch state(s) counted correct per GT stage (RELEASED is transitional: not scored)
LATCH_OK = {"APPROACH": {"searching", "latched"}, "CLOSING": {"closing"}, "HELD": {"held"}, "PLACED": {"placed"},
            "HOME": {"placed"}, "FAILED_GRASP": {"latched", "closing"}, "DROPPED": {"searching", "latched", "closing"}}
MAIN = ["APPROACH", "CLOSING", "HELD", "PLACED", "HOME"]   # success-path stages used for the main accuracy
# precision / recall classes: latch class <- GT class
PR = {"pre-grasp (searching+latched)": ({"searching", "latched"}, {"APPROACH"}), "closing": ({"closing"}, {"CLOSING"}),
      "held": ({"held"}, {"HELD"}), "placed": ({"placed"}, {"PLACED", "HOME"})}
STAGE_COL = {"APPROACH": FS.CATEGORICAL[0], "CLOSING": FS.CATEGORICAL[3], "HELD": FS.CATEGORICAL[2], "RELEASED": FS.CATEGORICAL[4],
             "PLACED": FS.CATEGORICAL[6], "HOME": FS.SEQ_BLUE[2], "FAILED_GRASP": FS.CATEGORICAL[7], "DROPPED": FS.CATEGORICAL[1]}
LATCH_COL = {"searching": FS.GRID_INK, "latched": FS.CATEGORICAL[0], "closing": FS.CATEGORICAL[3], "held": FS.CATEGORICAL[2],
             "placed": FS.CATEGORICAL[6]}
PHASE_COL = {"pick": FS.CATEGORICAL[0], "place": FS.CATEGORICAL[2], "home": FS.CATEGORICAL[6], None: FS.GRID_INK}
MAN_COL = {"visible": FS.SEQ_BLUE[3], "occluded": FS.CATEGORICAL[1], "kept": FS.CATEGORICAL[4], None: FS.GRID_INK}


def P(v):
    """Nested summary values may be Python reprs."""
    if isinstance(v, str):
        for f in (ast.literal_eval, json.loads):
            try:
                return f(v)
            except Exception:
                pass
    return v


def load_in_crate(kit):
    spec = importlib.util.spec_from_file_location("ifg_kit_analyze_t42", kit / "p" / "analyze_t42.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.in_crate


def gt_stages(z, in_crate):
    ap = z["apple_pos"][:600]; dz = ap[:, 2] - ap[0, 2]; cnt = z["cnt_apple-hand"] > 0; g = z["grip_cmd"]; n = len(g)
    inc = np.array([in_crate(ap[t], z["crate_pos"][t], z["crate_quat"][t])[0] for t in range(n)])
    cycles, t = [], 0
    while t < n:
        if g[t] < CLOSE_TH:
            e = next((u for u in range(t + 1, n) if g[u] >= OPEN_TH), n); cycles.append([t, e]); t = e
        else:
            t += 1
    ok = (dz >= LIFT_DZ) & cnt; lifts, t = [], 0
    while t <= n - LIFT_L:
        if ok[t:t + LIFT_L].all():
            r = next((s for s in range(t + 1, n) if not cnt[s:min(n, s + REL_N)].any()), n); lifts.append([t, r]); t = r
        else:
            t += 1
    place = home = None
    if inc[-1]:
        k = n - 1
        while k > 0 and inc[k - 1]:
            k -= 1
        place = max(k, lifts[-1][1]) if lifts else k
        home = next((u for u in range(place, n) if g[u] >= OPEN_TH), n)
    first_close = cycles[0][0] if cycles else n
    st = []
    for t in range(n):
        if place is not None and t >= place:
            st.append("PLACED" if t < home else "HOME"); continue
        inl = [l for l in lifts if l[0] <= t < l[1]]
        if inl:
            st.append("HELD"); continue
        ended = [l for l in lifts if l[1] <= t]
        if ended:
            st.append("RELEASED" if (place is not None and ended[-1] is lifts[-1]) else "DROPPED"); continue
        if t < first_close:
            st.append("APPROACH"); continue
        cyc = [c for c in cycles if c[0] <= t < c[1]]
        if cyc:
            c = cyc[0]; st.append("CLOSING" if any(c[0] <= l[0] < c[1] for l in lifts) else "FAILED_GRASP")
        else:
            st.append("APPROACH")
    ev = {"first_close_t": int(first_close) if cycles else None, "close_cycles": cycles,
          "lift_close_t": next((c[0] for c in cycles if lifts and c[0] <= lifts[0][0] < c[1]), None),
          "lift_t": lifts[0][0] if lifts else None, "lifts": lifts, "release_t": lifts[-1][1] if lifts else None,
          "place_t": place, "home_t": home, "dz_max_contact_mm": float(1000 * dz[cnt].max()) if cnt.any() else None}
    ev = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in ev.items()}
    return st, ev, dz, cnt


def server_rows(jf, z):
    j = json.load(open(jf)); al = j["server_chunk_alignment"]["block"]; cd, off = pathlib.Path(al["chunk_dir"]), int(al["offset"])
    rows = []
    for k in range(int(z["plan_index"].max()) + 1):
        with np.load(cd / f"chunk_{k + off:05d}.npz", allow_pickle=True) as c:
            s = json.loads(str(c["summary_json"]))
        g = P(s.get("grasp")) or {}; sb = P(s.get("subtask")) or {}; m = P(s.get("manipulated")) or {}; e = P(g.get("evidence")) or {}
        rows.append({"k": k, "server_t_step": s.get("t_step"), "latch": g.get("state"), "latch_before": g.get("state_before"),
                     "transition": bool(g.get("transition")), "attach": bool(g.get("attach")), "revoked": g.get("attach_revoked"),
                     "detach": bool(g.get("detach")), "note": g.get("note") or "", "checks": P(e.get("checks")) or {},
                     "latch_target": e.get("manipulated_id"), "argmax": sb.get("argmax"), "label": sb.get("label"),
                     "p": sb.get("p"), "released": sb.get("released"), "placed_seen": sb.get("placed_seen"),
                     "man_state": m.get("state"), "man_id": m.get("id"), "man_switched_from": m.get("switched_from")})
    return rows, {"chunk_dir": str(cd), "offset": off}


def first(rows, pred):
    return next((r["k"] for r in rows if pred(r)), None)


def lat(k_l, t_e):
    if k_l is None or t_e is None:
        return None
    return {"req": int(k_l - math.ceil(t_e / SPR)), "s": round((SPR * k_l - t_e) / HZ, 4), "k_latch": int(k_l), "t_event": int(t_e)}


def dist(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v, float)
    return {"n": len(v), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max()), "min": float(a.min())}


def guard(paths, overwrite):
    ex = [str(p) for p in paths if p.exists()]
    if ex and not overwrite:
        raise SystemExit(f"refusing to overwrite existing files: {ex}")


def heat(ax, M, rows, cols, title, ok=None):
    M = np.asarray(M, float); Rn = M / np.maximum(M.sum(1, keepdims=True), 1)
    ax.imshow(Rn, cmap=FS.sequential_cmap(), vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=7.5,
                        color=FS.SURFACE if Rn[i, j] > 0.55 else FS.INK, fontweight="bold" if ok and ok(rows[i], cols[j]) else None)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([str(c) for c in cols], rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r} (n={int(M[i].sum())})" for i, r in enumerate(rows)], fontsize=7.5)
    ax.set_title(title, fontsize=9, loc="left", color=FS.INK)
    for s in ax.spines.values():
        s.set_visible(False)


def notes(N):
    fe = N["false_events"]; an = fe["attach_not_held"]; b = an["breakdown"]; lt = N["latency"]
    return [
        "GT HELD needs a lift (dz >= 10 mm with hand contact for 5 steps); the latch attaches when the closed fingers settle "
        "on the object, before the lift. Negative attach latency and 'attach while GT not HELD' in CLOSING follow from this "
        f"definition: {b['gt_CLOSING_contact_lift_within_2req']} of {an['count']} such attaches had hand contact at the request "
        "and a GT lift within 2 requests.",
        "GT close = grip_cmd < 0.5. The latch uses its own command threshold; in SPL_ep1313 the command ramped 0.88 -> 0.56 and the "
        "latch counted 7 closing steps (last command 0.75) one request before grip_cmd crossed 0.5 (latency -1 request).",
        "Latency floor: the latch decides from execution feedback of the previous chunk, so a GT event inside chunk k is first "
        "visible at request k+1; latency in requests counts from ceil(t_event / 8).",
        "latch_attach_vs_lift and probe_*_place_vs_lift take the first event in the run; *_same_cycle take the first event at or "
        f"after the close command of the cycle that produced the lift (SPL_ep211 first attach was in a failed cycle: "
        f"min {lt['latch_attach_vs_lift']['requests'].get('min')} requests).",
        "RELEASED has 0 requests and PLACED only " + str(N["gt_stage_counts_requests"]["PLACED"]) + ": the apple is put into the crate "
        "while held, and the reopen command (PLACED -> HOME) follows within about one request.",
        "manipulated.state takes only visible / occluded / None in these 1800 requests; 'kept' never occurs.",
        "Probe label is None for the first 2 requests of every run (debounce warm-up); accuracy_labeled excludes them, accuracy_all counts them wrong.",
        "GT 'lift' with L=1 would add a 2-step 14.6 mm rise in SPL_ep320 (slipped); with L=5 it is FAILED_GRASP.",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    ap.add_argument("--verify", type=pathlib.Path, default=pathlib.Path("AG3S/ag3s/docs/handoff/IFG-AUDIT-HSR.verify.json"))
    ap.add_argument("--overwrite", action="store_true", help="allow replacing this study's own outputs")
    a = ap.parse_args()
    kit = a.run / "verify" / "IFG-APPLE-SPLIT"
    figs = {n: a.fig / f"ifg_audit_hsr_{n}.png" for n in ("timelines", "confusion", "latency")}
    guard([*figs.values(), *[p.with_suffix(".json") for p in figs.values()], a.verify], a.overwrite)
    a.fig.mkdir(parents=True, exist_ok=True)
    in_crate = load_in_crate(kit)

    runs = []
    for jf in sorted(glob.glob(str(kit / "replay" / "runs_SPL_SPL_ep*.json")), key=lambda f: int(f.split("_ep")[1].split("_")[0])):
        name = pathlib.Path(jf).stem.replace("runs_SPL_", "")
        an = json.load(open(kit / "analysis" / f"{name}.json"))
        man = json.load(open(kit / "runs" / "SPL" / name / "manifest.json"))["scene"]
        with np.load(jf[:-5] + ".npz") as zz:
            z = {k: zz[k] for k in ("apple_pos", "crate_pos", "crate_quat", "grip_cmd", "cnt_apple-hand", "plan_index", "finger_open_norm")}
        assert (z["plan_index"] == np.arange(len(z["plan_index"])) // SPR).all(), name
        st, ev, dz, cnt = gt_stages(z, in_crate)
        rows, al = server_rows(jf, z)
        for r in rows:
            t = SPR * r["k"]; r["t"] = t; r["gt"] = st[t]; r["phase"] = STAGE_PHASE[st[t]]; r["dz_mm"] = round(1000 * float(dz[t]), 1)
            r["gt_window"] = sorted(set(st[max(0, t - SPR):min(len(st), t + SPR + 1)]))
            r["contact_t"] = bool(cnt[t]); r["contact_prev_chunk"] = bool(cnt[max(0, t - SPR):t + 1].any())
            r["finger_open_t"] = round(float(z["finger_open_norm"][t]), 3)
        runs.append({"name": name, "episode": man["episode_index"], "used_arm": man["episode_used_arm"], "target": man["episode_target_fruit"],
                     "kit_success": an["success"], "kit_failure_mode": an.get("failure_mode"), "kit_grasp_t": an.get("grasp_t"),
                     "kit_place_t": an.get("place_t"), "ev": ev, "rows": rows, "align": al})
    allr = [(R["name"], r) for R in runs for r in R["rows"]]
    N = {"n_runs": len(runs), "n_requests": len(allr), "n_kit_success": sum(R["kit_success"] for R in runs),
         "gt_stage_counts_requests": {s: sum(r["gt"] == s for _, r in allr) for s in STAGES}}
    N["gt_vs_kit"] = {"runs_with_gt_place": sum(R["ev"]["place_t"] is not None for R in runs),
                      "runs_with_gt_lift": sum(R["ev"]["lift_t"] is not None for R in runs),
                      "place_mismatch_vs_kit_success": [R["name"] for R in runs if (R["ev"]["place_t"] is not None) != bool(R["kit_success"])]}

    # 1. latch confusion
    CM = [[sum(r["gt"] == s and r["latch"] == l for _, r in allr) for l in LATCH] for s in STAGES]
    other = sum(r["latch"] not in LATCH for _, r in allr)
    main_rows = [r for _, r in allr if r["gt"] in MAIN]
    N["latch"] = {"confusion": {"rows": STAGES, "cols": LATCH, "M": CM, "latch_state_outside_cols": other},
                  "accuracy_main": {"def": "GT in APPROACH/CLOSING/HELD/PLACED/HOME, correct per LATCH_OK", "n": len(main_rows),
                                    "correct": sum(r["latch"] in LATCH_OK[r["gt"]] for r in main_rows)},
                  "accuracy_all_scored": {"def": "all GT stages except RELEASED", "n": sum(r["gt"] != "RELEASED" for _, r in allr),
                                          "correct": sum(r["gt"] != "RELEASED" and r["latch"] in LATCH_OK[r["gt"]] for _, r in allr)},
                  "phase_accuracy": {"n": len(allr), "correct": sum(LATCH_PHASE.get(r["latch"]) == r["phase"] for _, r in allr)},
                  "precision_recall": {}}
    for k in ("accuracy_main", "accuracy_all_scored", "phase_accuracy"):
        d = N["latch"][k]; d["acc"] = round(d["correct"] / d["n"], 4)
    for cls, (ls, gs) in PR.items():
        tp = sum(r["latch"] in ls and r["gt"] in gs for _, r in allr)
        pred_main = sum(r["latch"] in ls for r in main_rows); pred_all = sum(r["latch"] in ls for _, r in allr)
        true = sum(r["gt"] in gs for _, r in allr)
        N["latch"]["precision_recall"][cls] = {"gt": sorted(gs), "tp": tp, "n_pred_main": pred_main, "n_pred_all": pred_all, "n_true": true,
                                               "precision_main": round(tp / pred_main, 4) if pred_main else None,
                                               "precision_all": round(tp / pred_all, 4) if pred_all else None,
                                               "recall": round(tp / true, 4) if true else None}

    # 2. latency
    L = {"latch_closing_vs_close_cmd": [], "latch_attach_vs_lift": [], "latch_placed_vs_place": [],
         "latch_closing_vs_lifting_close_cmd": [], "probe_label_place_vs_lift": [], "probe_label_home_vs_place": [],
         "probe_argmax_place_vs_lift": [], "probe_argmax_home_vs_place": [],
         "latch_attach_vs_lift_same_cycle": [], "probe_label_place_vs_lift_same_cycle": [], "probe_argmax_place_vs_lift_same_cycle": []}
    miss = {k: [] for k in L}
    for R in runs:
        rw, ev = R["rows"], R["ev"]
        kc = first(rw, lambda r: r["latch"] == "closing"); ka = first(rw, lambda r: r["attach"]); kp = first(rw, lambda r: r["latch"] == "placed")
        lift_k = math.ceil(ev["lift_t"] / SPR) if ev["lift_t"] is not None else None
        kc_l = first(rw, lambda r: r["latch"] == "closing" and ev["lift_close_t"] is not None and SPR * r["k"] >= ev["lift_close_t"])
        kpl = first(rw, lambda r: r["label"] == "place"); kph = first(rw, lambda r: r["label"] == "home")
        kal = first(rw, lambda r: r["argmax"] == "place"); kah = first(rw, lambda r: r["argmax"] == "home")
        lc = ev["lift_close_t"]
        sc = (lambda key, val: first(rw, lambda r: r[key] == val and lc is not None and SPR * r["k"] >= lc))
        ka_s = first(rw, lambda r: r["attach"] and lc is not None and SPR * r["k"] >= lc)
        for key, kl, te in (("latch_closing_vs_close_cmd", kc, ev["first_close_t"]), ("latch_attach_vs_lift", ka, ev["lift_t"]),
                            ("latch_placed_vs_place", kp, ev["place_t"]), ("latch_closing_vs_lifting_close_cmd", kc_l, ev["lift_close_t"]),
                            ("probe_label_place_vs_lift", kpl, ev["lift_t"]), ("probe_label_home_vs_place", kph, ev["place_t"]),
                            ("probe_argmax_place_vs_lift", kal, ev["lift_t"]), ("probe_argmax_home_vs_place", kah, ev["place_t"]),
                            ("latch_attach_vs_lift_same_cycle", ka_s, ev["lift_t"]), ("probe_label_place_vs_lift_same_cycle", sc("label", "place"), ev["lift_t"]),
                            ("probe_argmax_place_vs_lift_same_cycle", sc("argmax", "place"), ev["lift_t"])):
            x = lat(kl, te)
            if x:
                x["run"] = R["name"]; L[key].append(x)
            elif (kl is None) != (te is None):
                miss[key].append({"run": R["name"], "k_module": kl, "t_event": te})
        R["lat_keys"] = {"lift_k": lift_k}
    N["latency"] = {k: {"requests": dist([x["req"] for x in v]), "seconds": dist([x["s"] for x in v]), "per_run": v,
                        "one_side_only": miss[k]} for k, v in L.items()}

    # 3. false events
    FE = {"attach_not_held": [], "revoke_while_held": [], "held_to_latched_or_closing": [], "relatch_other_target": [],
          "manipulated_switch": [], "held_not_attached_requests": []}
    for R in runs:
        prev_t = None
        for r in R["rows"]:
            base = {"run": R["name"], "k": r["k"], "t": r["t"], "gt": r["gt"], "gt_window_pm1req": r["gt_window"], "dz_mm": r["dz_mm"],
                    "contact_t": r["contact_t"], "contact_prev_chunk": r["contact_prev_chunk"], "finger_open_t": r["finger_open_t"],
                    "lift_t": R["ev"]["lift_t"], "steps_to_lift": (R["ev"]["lift_t"] - r["t"]) if R["ev"]["lift_t"] is not None else None}
            if r["attach"] and r["gt"] != "HELD":
                FE["attach_not_held"].append({**base, "held_within_pm1req": "HELD" in r["gt_window"]})
            if r["revoked"] and r["gt"] == "HELD":
                FE["revoke_while_held"].append({**base, "revoked": str(r["revoked"])[:200]})
            if r["latch_before"] == "held" and r["latch"] in ("latched", "closing"):
                FE["held_to_latched_or_closing"].append({**base, "to": r["latch"], "note": r["note"][:160]})
            if r["latch"] in ("latched", "closing", "held") and r["latch_target"] is not None:
                if prev_t is not None and r["latch_target"] != prev_t:
                    FE["relatch_other_target"].append({**base, "from": prev_t, "to": r["latch_target"], "latch": r["latch"]})
                prev_t = r["latch_target"]
            if r["man_switched_from"] is not None:
                FE["manipulated_switch"].append({**base, "from": r["man_switched_from"], "to": r["man_id"], "latch": r["latch"]})
    N["false_events"] = {k: {"count": len(v), "runs": sorted({x["run"] for x in v}), "events": v} for k, v in FE.items() if k != "held_not_attached_requests"}
    N["false_events"]["attach_not_held"]["count_tolerant_pm1req"] = sum(not x["held_within_pm1req"] for x in FE["attach_not_held"])
    an_ = FE["attach_not_held"]
    N["false_events"]["attach_not_held"]["breakdown"] = {
        "gt_CLOSING_contact_lift_within_2req": sum(x["gt"] == "CLOSING" and x["contact_t"] and x["steps_to_lift"] is not None and 0 < x["steps_to_lift"] <= 2 * SPR for x in an_),
        "gt_CLOSING_other": sum(x["gt"] == "CLOSING" and not (x["contact_t"] and x["steps_to_lift"] is not None and 0 < x["steps_to_lift"] <= 2 * SPR) for x in an_),
        "gt_FAILED_GRASP": sum(x["gt"] == "FAILED_GRASP" for x in an_), "gt_DROPPED": sum(x["gt"] == "DROPPED" for x in an_),
        "gt_other": sum(x["gt"] not in ("CLOSING", "FAILED_GRASP", "DROPPED") for x in an_),
        "no_hand_contact_at_t": sum(not x["contact_t"] for x in an_)}
    N["false_events"]["counts_total"] = {"attach": sum(r["attach"] for _, r in allr), "revoked": sum(bool(r["revoked"]) for _, r in allr),
                                         "transitions": sum(r["transition"] for _, r in allr)}
    tr = {}
    for _, r in allr:
        if r["transition"]:
            kk = f"{r['latch_before']}->{r['latch']}"; tr[kk] = tr.get(kk, 0) + 1
    N["false_events"]["transition_counts"] = tr

    # 4. probe
    for key in ("argmax", "label"):
        cols = PHASES + [None]
        M = [[sum(r["phase"] == p and r[key] == c for _, r in allr) for c in cols] for p in PHASES]
        lab = [r for _, r in allr if r[key] is not None]
        N[f"probe_{key}"] = {"confusion": {"rows": PHASES, "cols": [str(c) for c in cols], "M": M},
                             "accuracy_labeled": {"n": len(lab), "correct": sum(r[key] == r["phase"] for r in lab),
                                                  "acc": round(sum(r[key] == r["phase"] for r in lab) / len(lab), 4)},
                             "accuracy_all": {"n": len(allr), "correct": sum(r[key] == r["phase"] for _, r in allr),
                                              "acc": round(sum(r[key] == r["phase"] for _, r in allr) / len(allr), 4)},
                             "n_none": len(allr) - len(lab),
                             "per_stage_correct": {s: [sum(r["gt"] == s and r[key] == STAGE_PHASE[s] for _, r in allr), sum(r["gt"] == s for _, r in allr)] for s in STAGES}}
    for R in runs:
        R["label_changes"] = sum(1 for a_, b_ in zip(R["rows"], R["rows"][1:]) if a_["label"] is not None and b_["label"] != a_["label"])
    N["probe_label"]["label_changes_per_run"] = {R["name"]: R["label_changes"] for R in runs}
    N["probe_label"]["place_label_never_but_gt_lift"] = [R["name"] for R in runs if R["ev"]["lift_t"] is not None and not any(r["label"] == "place" for r in R["rows"])]
    N["probe_label"]["place_label_without_gt_lift"] = [R["name"] for R in runs if R["ev"]["lift_t"] is None and any(r["label"] == "place" for r in R["rows"])]
    # agreement probe (debounced label) vs latch phase
    D = {"both_right": 0, "latch_right_probe_wrong": 0, "probe_right_latch_wrong": 0, "both_wrong": 0, "probe_none": 0}
    Dh = dict(D); pl_nh = []
    for nm, r in allr:
        lp = LATCH_PHASE.get(r["latch"]); pr = r["label"]
        for DD, cond in ((D, True), (Dh, r["gt"] == "HELD")):
            if not cond:
                continue
            if pr is None:
                DD["probe_none"] += 1
            else:
                DD[{(True, True): "both_right", (True, False): "latch_right_probe_wrong", (False, True): "probe_right_latch_wrong",
                    (False, False): "both_wrong"}[(lp == r["phase"], pr == r["phase"])]] += 1
        if pr == "place" and r["latch"] != "held":
            pl_nh.append({"run": nm, "k": r["k"], "latch": r["latch"], "gt": r["gt"], "dz_mm": r["dz_mm"]})
    N["probe_vs_latch"] = {"all_requests": D, "gt_held_requests": Dh,
                           "disagree_requests": sum(1 for _, r in allr if r["label"] is not None and LATCH_PHASE.get(r["latch"]) != r["label"]),
                           "probe_place_latch_not_held": {"count": len(pl_nh), "by_gt": {s: sum(x["gt"] == s for x in pl_nh) for s in STAGES},
                                                          "by_latch": {l: sum(x["latch"] == l for x in pl_nh) for l in LATCH},
                                                          "runs": {nm: sum(x["run"] == nm for x in pl_nh) for nm in sorted({x["run"] for x in pl_nh})},
                                                          "events": pl_nh}}

    # 5. memory during GT HELD
    held = [(nm, r) for nm, r in allr if r["gt"] == "HELD"]
    seg_flip = []
    for R in runs:
        rw = R["rows"]
        for i in range(1, len(rw)):
            if rw[i]["gt"] == "HELD" and rw[i - 1]["gt"] == "HELD" and rw[i]["latch"] != rw[i - 1]["latch"]:
                seg_flip.append({"run": R["name"], "k": rw[i]["k"], "from": rw[i - 1]["latch"], "to": rw[i]["latch"]})
    N["memory_held"] = {"n_gt_held_requests": len(held), "latch_counts": {l: sum(r["latch"] == l for _, r in held) for l in LATCH},
                        "manipulated_state_counts": {str(s): sum(r["man_state"] == s for _, r in held) for s in ("visible", "occluded", "kept", None)},
                        "manipulated_state_when_latch_held": {str(s): sum(r["man_state"] == s and r["latch"] == "held" for _, r in held) for s in ("visible", "occluded", "kept", None)},
                        "latch_changes_inside_gt_held": seg_flip,
                        "latch_held_after_first_attach_in_run": None,
                        "manipulated_state_all_requests": {str(s): sum(r["man_state"] == s for _, r in allr) for s in ("visible", "occluded", "kept", None)},
                        "manipulated_state_by_gt": {s: {str(m): sum(r["gt"] == s and r["man_state"] == m for _, r in allr) for m in ("visible", "occluded", "kept", None)} for s in STAGES},
                        "manipulated_id_switch_during_held": sum(1 for x in FE["manipulated_switch"] if x["gt"] == "HELD")}
    kept = 0; after = 0
    for R in runs:
        ka = first(R["rows"], lambda r: r["attach"])
        for r in R["rows"]:
            if r["gt"] == "HELD" and ka is not None and r["k"] >= ka:
                after += 1; kept += r["latch"] == "held"
    N["memory_held"]["latch_held_after_first_attach_in_run"] = {"n": after, "held": kept}

    # 6. per-run summary (failures in full)
    PRS = []
    for R in runs:
        rw = R["rows"]
        PRS.append({"run": R["name"], "kit_success": R["kit_success"], "kit_failure_mode": R["kit_failure_mode"], "gt_events": R["ev"],
                    "gt_stage_counts": {s: sum(r["gt"] == s for r in rw) for s in STAGES if any(r["gt"] == s for r in rw)},
                    "latch_counts": {l: sum(r["latch"] == l for r in rw) for l in LATCH if any(r["latch"] == l for r in rw)},
                    "label_counts": {str(p): sum(r["label"] == p for r in rw) for p in PHASES + [None] if any(r["label"] == p for r in rw)},
                    "attach_k": [r["k"] for r in rw if r["attach"]], "revoke_k": [r["k"] for r in rw if r["revoked"]],
                    "label_changes": R["label_changes"],
                    "latch_seq": "".join({"searching": "s", "latched": "l", "closing": "c", "held": "H", "placed": "P"}.get(r["latch"], "?") for r in rw),
                    "gt_seq": "".join({"APPROACH": "a", "CLOSING": "c", "HELD": "H", "RELEASED": "r", "PLACED": "P", "HOME": "h", "FAILED_GRASP": "F", "DROPPED": "D"}[r["gt"]] for r in rw),
                    "label_seq": "".join({"pick": "k", "place": "L", "home": "h", None: "-"}[r["label"]] for r in rw),
                    "man_seq": "".join({"visible": "v", "occluded": "o", "kept": "K", None: "-"}.get(r["man_state"], "?") for r in rw)})
    N["per_run"] = PRS
    N["failure_notes"] = {x["run"]: [r["note"][:200] for r in R["rows"] if r["note"]][:6]
                          for x, R in zip(PRS, runs) if not x["kit_success"]}

    # ---------- figure a: timelines ----------
    pick = ["SPL_ep119_s1193", "SPL_ep1837_s18373", "SPL_ep753_s7533", "SPL_ep211_s2113", "SPL_ep320_s3203", "SPL_ep1313_s13133"]
    pick = [p for p in pick if any(R["name"] == p for R in runs)]
    fig, axs = plt.subplots(len(pick), 1, figsize=(11, 1.55 * len(pick) + 1.0), sharex=True)
    FS.style_axes(fig, axs); side_a = {"definitions": __doc__, "scenes": {}}
    lanes = [T("GT 단계", "GT stage"), T("latch", "latch"), T("probe label", "probe label"), T("manipulated", "manipulated")]
    for ax, nm in zip(np.atleast_1d(axs), pick):
        R = next(R for R in runs if R["name"] == nm); rw = R["rows"]
        for k, r in enumerate(rw):
            for y, c in ((3, STAGE_COL[r["gt"]]), (2, LATCH_COL.get(r["latch"], FS.INK)), (1, PHASE_COL.get(r["label"], FS.INK)),
                         (0, MAN_COL.get(r["man_state"], FS.INK))):
                ax.add_patch(plt.Rectangle((k, y - 0.42), 1, 0.84, color=c, lw=0))
            if r["attach"]:
                ax.scatter(k + 0.5, 2.55, marker="v", s=22, color=FS.INK, zorder=4)
            if r["revoked"]:
                ax.scatter(k + 0.5, 2.55, marker="x", s=22, color=FS.CATEGORICAL[7], zorder=4, lw=1.4)
        ev = R["ev"]
        for t_e, ls in ((ev["first_close_t"], ":"), (ev["lift_t"], "-"), (ev["place_t"], "--")):
            if t_e is not None:
                ax.axvline(t_e / SPR, color=FS.INK, ls=ls, lw=0.8)
        ax.set_xlim(0, len(rw)); ax.set_ylim(-0.6, 3.6); ax.set_yticks(range(4)); ax.set_yticklabels(lanes[::-1], fontsize=7)
        tag = "✓" if R["kit_success"] else "×"
        ax.set_title(f"{nm.replace('SPL_', '')}  {tag} {R['kit_failure_mode']}  ·  attach k={[r['k'] for r in rw if r['attach']]}"
                     f"  revoke k={[r['k'] for r in rw if r['revoked']]}", fontsize=8.2, loc="left", color=FS.INK)
        side_a["scenes"][nm] = {"gt": [r["gt"] for r in rw], "latch": [r["latch"] for r in rw], "label": [r["label"] for r in rw],
                                "manipulated_state": [r["man_state"] for r in rw], "attach_k": [r["k"] for r in rw if r["attach"]],
                                "revoke_k": [r["k"] for r in rw if r["revoked"]], "gt_events": ev, "kit_success": R["kit_success"],
                                "kit_failure_mode": R["kit_failure_mode"]}
    ax = np.atleast_1d(axs)[-1]
    ax.set_xlabel(T("request k (1 request = 8 control step = 0.533 s)  ·  세로선: 점선=첫 닫힘 명령, 실선=GT lift, 파선=GT place",
                    "request k (8 control steps = 0.533 s)  ·  vlines: dotted=first close cmd, solid=GT lift, dashed=GT place"), fontsize=8)
    hs = [Patch(color=STAGE_COL[s], label=s) for s in STAGES] + [Patch(color=LATCH_COL[l], label=f"latch {l}") for l in LATCH] + \
         [Patch(color=PHASE_COL[p], label=f"probe {p}") for p in PHASES] + [Patch(color=FS.GRID_INK, label="probe None / manip. None")] + \
         [Patch(color=MAN_COL[m], label=f"manip. {m}") for m in ("visible", "occluded")] + \
         [plt.Line2D([], [], marker="v", ls="", color=FS.INK, label="attach"), plt.Line2D([], [], marker="x", ls="", color=FS.CATEGORICAL[7], label="attach_revoked")]
    fig.legend(handles=hs, ncol=7, fontsize=6.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(T("a  요청별 단계 추적: GT 단계 vs grasp latch vs sub-task probe (debounced label) vs manipulated.state — SPL 6 장면",
                   "a  per-request stage tracking: GT vs latch vs probe label vs manipulated.state — 6 SPL scenes"), fontsize=10, x=0.01, ha="left", color=FS.INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97)); fig.savefig(figs["timelines"], dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    figs["timelines"].with_suffix(".json").write_text(json.dumps(side_a, indent=1, ensure_ascii=False, default=str))

    # ---------- figure b: confusion + memory ----------
    fig = plt.figure(figsize=(13.5, 8.4)); gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 0.82])
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[0, 2]); FS.style_axes(fig, [ax0, ax1, ax2])
    la = N["latch"]
    heat(ax0, CM, STAGES, LATCH, T(f"a  latch state vs GT 단계 (요청 {len(allr)}; 굵게=정답)\n주 정확도 {la['accuracy_main']['acc']:.3f}"
                                   f" ({la['accuracy_main']['correct']}/{la['accuracy_main']['n']})",
                                   f"a  latch vs GT stage (bold = correct)\nmain acc {la['accuracy_main']['acc']:.3f}"), ok=lambda s, l: l in LATCH_OK.get(s, ()))
    for axx, key, tag in ((ax1, "probe_argmax", "b  probe argmax"), (ax2, "probe_label", "c  probe label (debounced)")):
        d = N[key]
        heat(axx, d["confusion"]["M"], PHASES, d["confusion"]["cols"],
             f"{tag} vs GT phase\nacc {d['accuracy_labeled']['acc']:.3f} ({d['accuracy_labeled']['correct']}/{d['accuracy_labeled']['n']}), None {d['n_none']}",
             ok=lambda p, c: p == c)
    bx = fig.add_subplot(gs[1, 0]); cx = fig.add_subplot(gs[1, 1]); dx = fig.add_subplot(gs[1, 2]); FS.style_axes(fig, [bx, cx, dx])
    pr = N["latch"]["precision_recall"]; names = list(pr); y = np.arange(len(names))
    bx.barh(y - 0.2, [pr[n]["precision_all"] or 0 for n in names], 0.38, color=FS.CATEGORICAL[0], label=T("precision (전체 요청)", "precision (all)"))
    bx.barh(y + 0.2, [pr[n]["recall"] or 0 for n in names], 0.38, color=FS.CATEGORICAL[2], label="recall")
    for i, n in enumerate(names):
        bx.text((pr[n]["precision_all"] or 0) + 0.01, i - 0.2, f"{pr[n]['precision_all']:.3f} ({pr[n]['tp']}/{pr[n]['n_pred_all']})", va="center", fontsize=6.8)
        bx.text((pr[n]["recall"] or 0) + 0.01, i + 0.2, f"{pr[n]['recall']:.3f} ({pr[n]['tp']}/{pr[n]['n_true']})", va="center", fontsize=6.8)
    bx.set_yticks(y); bx.set_yticklabels(names, fontsize=7.5); bx.set_xlim(0, 1.35); bx.invert_yaxis(); bx.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    bx.set_title(T("d  latch class별 precision / recall", "d  latch precision / recall"), fontsize=9, loc="left", color=FS.INK)
    mh = N["memory_held"]
    lc = mh["latch_counts"]; mc = mh["manipulated_state_counts"]
    cx.bar(range(len(LATCH)), [lc[l] for l in LATCH], color=[LATCH_COL[l] for l in LATCH])
    for i, l in enumerate(LATCH):
        cx.text(i, lc[l] + 1, str(lc[l]), ha="center", fontsize=7.5)
    cx.set_xticks(range(len(LATCH))); cx.set_xticklabels(LATCH, fontsize=7.5, rotation=20)
    cx.set_title(T(f"e  GT HELD 요청 {mh['n_gt_held_requests']}개의 latch 상태", f"e  latch during GT HELD (n={mh['n_gt_held_requests']})"), fontsize=9, loc="left", color=FS.INK)
    ms = ["visible", "occluded", "kept", "None"]
    dx.bar(range(4), [mc[m] for m in ms], color=[MAN_COL["visible"], MAN_COL["occluded"], MAN_COL["kept"], FS.GRID_INK])
    for i, m in enumerate(ms):
        dx.text(i, mc[m] + 1, str(mc[m]), ha="center", fontsize=7.5)
    dx.set_xticks(range(4)); dx.set_xticklabels(ms, fontsize=7.5)
    dx.set_title(T("f  GT HELD 동안 manipulated.state (손에 가려짐)", "f  manipulated.state during GT HELD"), fontsize=9, loc="left", color=FS.INK)
    fig.tight_layout(); fig.savefig(figs["confusion"], dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    figs["confusion"].with_suffix(".json").write_text(json.dumps({"latch": N["latch"], "probe_argmax": N["probe_argmax"], "probe_label": N["probe_label"],
                                                                  "memory_held": {k: v for k, v in mh.items()}, "definitions": __doc__}, indent=1, ensure_ascii=False, default=str))

    # ---------- figure c: latency ----------
    keys = [("latch_closing_vs_close_cmd", T("latch closing\nvs 첫 닫힘 명령", "latch closing\nvs first close cmd")),
            ("latch_closing_vs_lifting_close_cmd", T("latch closing\nvs 성공 닫힘 명령", "latch closing\nvs lifting close cmd")),
            ("latch_attach_vs_lift_same_cycle", T("latch attach(held)\nvs GT lift*", "latch attach\nvs GT lift*")),
            ("latch_placed_vs_place", T("latch placed\nvs GT place", "latch placed\nvs GT place")),
            ("probe_label_place_vs_lift_same_cycle", T("probe label place\nvs GT lift*", "probe label place\nvs GT lift*")),
            ("probe_label_home_vs_place", T("probe label home\nvs GT place", "probe label home\nvs GT place"))]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]}); FS.style_axes(fig, [ax, ax2])
    rng = np.random.default_rng(0); side_c = {}
    for i, (k, lab) in enumerate(keys):
        v = [x["s"] for x in N["latency"][k]["per_run"]]; d = N["latency"][k]["seconds"]; dr = N["latency"][k]["requests"]
        col = FS.CATEGORICAL[0] if k.startswith("latch") else FS.CATEGORICAL[2]
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.17, 0.17, len(v)), v, s=16, color=col, alpha=0.85, zorder=3)
        if d["n"]:
            ax.plot([i - 0.3, i + 0.3], [d["median"]] * 2, color=FS.INK, lw=1.6); ax.plot([i - 0.22, i + 0.22], [d["p90"]] * 2, color=FS.INK, lw=0.9, ls="--")
            ax.text(i, d["max"] + 0.25, f"n={d['n']}\nmed {d['median']:.2f}s / {dr['median']:.0f}req\np90 {d['p90']:.2f}s\nmax {d['max']:.2f}s / {dr['max']:.0f}req",
                    ha="center", fontsize=6.3, color=FS.INK_2)
        side_c[k] = {"seconds": d, "requests": dr, "per_run": N["latency"][k]["per_run"], "one_side_only": N["latency"][k]["one_side_only"]}
    ax.axhline(0, color=FS.GRID_INK, lw=0.8)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys], fontsize=7.2)
    ax.set_ylabel(T("지연 (s) = (8·k_module − t_GT) / 15", "latency (s)")); ax.grid(axis="y", color=FS.GRID_INK, lw=0.4); ax.set_axisbelow(True)
    ymax = max([x["s"] for kk, _ in keys for x in N["latency"][kk]["per_run"]] + [1]); ax.set_ylim(min(-0.5, ax.get_ylim()[0]), ymax + 2.2)
    ax.set_title(T("a  전이 지연 분포 (실선 median, 파선 p90; 파랑=latch, 초록=probe; *=lift 를 낸 닫힘 cycle 이후 첫 사건; 음수=GT 보다 먼저)",
                   "a  transition latency (solid median, dashed p90; *=first event in the lifting close cycle; <0 = before GT)"), fontsize=9, loc="left", color=FS.INK)
    # b: per-run lift-related latency in requests (bar: attach vs probe place label)
    lat_a = {x["run"]: x["req"] for x in N["latency"]["latch_attach_vs_lift_same_cycle"]["per_run"]}
    lat_p = {x["run"]: x["req"] for x in N["latency"]["probe_label_place_vs_lift_same_cycle"]["per_run"]}
    rr = [R["name"] for R in runs if R["ev"]["lift_t"] is not None]; yy = np.arange(len(rr))
    ax2.barh(yy - 0.2, [lat_a.get(n, np.nan) for n in rr], 0.38, color=FS.CATEGORICAL[0], label="latch attach")
    ax2.barh(yy + 0.2, [lat_p.get(n, np.nan) for n in rr], 0.38, color=FS.CATEGORICAL[2], label="probe label place")
    for i, n in enumerate(rr):
        for off, dd in ((-0.2, lat_a), (0.2, lat_p)):
            ax2.text((dd[n] if n in dd else 0) + 0.1, i + off, str(dd[n]) if n in dd else T("없음", "none"), va="center", fontsize=6)
    ax2.set_yticks(yy); ax2.set_yticklabels([n.replace("SPL_", "") for n in rr], fontsize=6.5); ax2.invert_yaxis()
    ax2.set_xlabel(T("GT lift 이후 지연 (request)", "latency after GT lift (requests)")); ax2.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax2.set_title(T("b  GT lift 가 있는 run 별 지연 (request, lift cycle 기준)", "b  per-run latency after GT lift (requests)"), fontsize=9, loc="left", color=FS.INK)
    side_c["panel_b"] = {"runs": rr, "latch_attach_req": lat_a, "probe_label_place_req": lat_p}
    fig.tight_layout(); fig.savefig(figs["latency"], dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    figs["latency"].with_suffix(".json").write_text(json.dumps({"numbers": side_c, "definitions": __doc__}, indent=1, ensure_ascii=False, default=str))

    # ---------- verify.json ----------
    kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    V = {"step": "IFG-AUDIT-HSR", "written": kst,
         "data": {"run": str(a.run), "kit": str(kit), "arm": "SPL", "n_runs": len(runs), "runs": [R["name"] for R in runs],
                  "server_chunk_dirs": sorted({R["align"]["chunk_dir"] for R in runs}),
                  "in_crate_rule": str(kit / "p" / "analyze_t42.py")},
         "definitions": {"doc": __doc__, "constants": {"CLOSE_TH": CLOSE_TH, "OPEN_TH": OPEN_TH, "LIFT_DZ_m": LIFT_DZ, "LIFT_L_steps": LIFT_L,
                                                       "REL_N_steps": REL_N, "steps_per_request": SPR, "hz": HZ},
                         "latch_ok": {k: sorted(v) for k, v in LATCH_OK.items()}, "stage_phase": STAGE_PHASE, "latch_phase": LATCH_PHASE,
                         "precision_recall_classes": {k: {"latch": sorted(v[0]), "gt": sorted(v[1])} for k, v in PR.items()}},
         "scripts": ["AG3S/ag3s/experiments/studies/ifg_audit_hsr.py"],
         "figures": {k: str(v) for k, v in figs.items()},
         "numbers": {k: v for k, v in N.items() if k not in ("per_run",)},
         "per_run": N["per_run"],
         "notes": notes(N)}
    a.verify.parent.mkdir(parents=True, exist_ok=True)
    a.verify.write_text(json.dumps(V, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"latch_acc_main": la["accuracy_main"], "latch_pr": {k: (v["precision_all"], v["recall"]) for k, v in la["precision_recall"].items()},
                      "probe_label_acc": N["probe_label"]["accuracy_labeled"], "probe_argmax_acc": N["probe_argmax"]["accuracy_labeled"],
                      "lat_s": {k: v["seconds"] for k, v in N["latency"].items()}, "lat_req": {k: v["requests"] for k, v in N["latency"].items()},
                      "fe": {k: (v["count"], v["runs"]) for k, v in N["false_events"].items() if isinstance(v, dict) and "count" in v},
                      "trans": tr, "pvl": N["probe_vs_latch"]["all_requests"], "pvl_held": N["probe_vs_latch"]["gt_held_requests"],
                      "pl_nh": {k: v for k, v in N["probe_vs_latch"]["probe_place_latch_not_held"].items() if k != "events"},
                      "mem": {k: v for k, v in mh.items() if k not in ("manipulated_state_by_gt",)}, "gt": N["gt_stage_counts_requests"], "gt_vs_kit": N["gt_vs_kit"]},
                     indent=None, default=str))


if __name__ == "__main__":
    main()
