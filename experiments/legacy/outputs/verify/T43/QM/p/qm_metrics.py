"""T43 QM — V8 vs V11AB trajectory / performance metrics, paired by (stage, episode, seed) (verifier, CPU only, recorded data + CPU replays; numbers only).
usage: qm_metrics.py OUT_JSON
Runs: V8 = outputs/verify/T43/V8/runs/V8/{H,B}, V11AB = outputs/verify/T43/V11/runs/V11AB/{H,B} (failed/ attempts excluded: analysis_V11AB.json reads only the
  run directories with .done at the stage root). 14 H (hurdle) + 5 B (no obstacle) pairs. Hurdle-dependent metrics on H only.
Inputs (read only):
  analysis   V8/analysis/analysis.json, V11/analysis/analysis_V11AB.json (analyze_t43w / analyze_t39 rules; server segment paired by the analyze_t41 rule;
             old success / grasp / place, grasp_t, place_t, server_chunks, hurdle series).
  CT / Z4    H: V11/ct_V11AB.json (both arms; z4_release score() + CT contact rule).  B: z4_base.score() (CT/p, unchanged) on the B replays
             QM/replay (CT/p/replay_ct.py unchanged, run by QM/p/run_replays_b.sh); B has no obstacle, so success_ct = success_old and success_rel.
  replays    H: V8/replay, V11/replay (V11AB_*); B: QM/replay.  Post state of control row i = replay state index i+1.
  rows       QM/raw/rows/*.npz (qm_rows.py: tcp C / M, palm M, apple M, chunk deflection at the planning-row state).
  cross      XM/raw/cross_V8.json, V11/raw/cross_V11AB.json (xm_cross.py: s, z, apple bottom C / M per row, crossing rows, in_window, TO-row reconstruction).
  latency    analysis server_chunks solve_ms (first chunk excluded) + load samples (V8/raw/load_samples.txt, V11/AB/load_samples_V11AB.txt) during each run
             (compare_ab.py rule); controlled: outputs/impl/T43TP/timing/{cuda,cpu}/*.json (variants V8, V11AB on the same V8 chunks; first chunk excluded).
  sharpness  V11/sharp_V11AB.json per_run (both arms) acc_boundary_rad (joint space, executed command; T43TA boundary rule).
Statistics: continuous -> paired difference d = V11AB - V8 per pair where both values exist; scipy.stats.wilcoxon(d) two-sided (zero_method 'wilcox':
  zero differences dropped; method 'auto'). binary -> exact McNemar two-sided on discordant pairs b = V8 only, c = V11AB only (p = min(1, 2 P(X <= min(b,c)))).
Definitions: see DEFS below (also written to OUT_JSON)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib, math, datetime, collections, glob
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/CT/p'))
import z4_base as Z  # noqa: E402

T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); Q = T / "QM"
ARMS = ("V8", "V11AB")
HZ = 15.0
WIN_MM = 100.0
MARGIN_MM = 10.0   # TO obstacle margin (xm_cross OBST_MARGIN, server log obstacle_margin_mm 10)
DEFL_MM = 5.0
DEFS = {
    "pairing": "key = STAGE:EPISODE_sSEED; 14 H + 5 B pairs; d = V11AB - V8",
    "crossing_row": "carry crossing row of the measured apple centre: first row t >= first in-hand row with s(t-1) < 0 <= s(t), s = signed horizontal distance of the apple centre across the bar's vertical plane (+ toward the crate), post state of row t (xm_cross crossing_rows apple_M carry). C variant: same on the commanded state (apple_C).",
    "crossing_window": "rows with min over groups (fingers, palm, apple in hand) of min(geo C, geo M) to the hurdle <= 100 mm (xm_cross in_window, HX rule)",
    "deflection_onset_rows": "crossing_row - t_first, t_first = first EXECUTED control row t with first_inhand_t <= t < crossing_row whose chunk-step palm |refined - reference| (EE_BODY origin, FK at the planning-row state, row k of its chunk) > 5 mm. None if no such row. Variants: _from_t0 (search from row 0), _contiguous (start of the run of consecutive executed deflected rows that contains crossing_row - 1; 0 if crossing_row - 1 not deflected).",
    "lift_hold_rows": "length of the maximal run of consecutive control rows, each EXECUTED and with commanded apple-bottom height above the policy reference by > 5 mm (apple_bottom(refined row k) - apple_bottom(reference row k), apple rigid with EE_BODY at the planning-row state, in hand at that state), that contains the crossing row; 0 if the crossing row is not such a row. Also rows of that run before / after the crossing row, and lifted executed rows inside the crossing window.",
    "acc_jerk": "TCP = site <side>_ee. commanded C(t) = executed_actions[t] (both arms) on the post state of row t; measured M(t) = post state of row t (recorded qpos of row t+1). a(t) = |p(t+1) - 2p(t) + p(t-1)| * 15^2 (m/s^2); j(t) = |p(t+1) - 3p(t) + 3p(t-1) - p(t-2)| * 15^3 (m/s^3). A stencil is used only when all its rows are executed. boundary = stencil rows belong to >= 2 chunks (chunk_seq); inner = otherwise. window_in / window_out = stencil centre row t in / not in the crossing window (H); window_in_carry = window_in and carry (V8 runs that stopped on the return have no return-window rows). carry = centre row in [grasp_t, release_t]. Per run: median, p90, max of each class; paired tests on the per-run value.",
    "success": "old = analysis success (analyze_e0: grasp = planning-row apple z >= z0 + 50 mm; place = apple centre in the crate box at the last planning row). Z4 = success_rel (release_t exists, separation kept, apple in crate at the end). CT = old and Z4 and no robot/apple-hurdle contact row at or before release_t (B: no obstacle -> CT = old and Z4).",
    "collision": "H replay obstacle_contact_rows (MuJoCo contacts, any sub-step, incl. margin contacts as in CT) with the hurdle collision geoms; robot = hand (palm + both fingers of the active arm) or robot_other; apple = apple geom. Phases: before_grasp t < grasp_t; carry grasp_t <= t <= release_t (Z4 release_t; none -> to the end); return t > release_t. run flag = >= 1 contact row in the phase; row share = contact rows / rows of the phase.",
    "min_clearance_M": "analysis hurdle series (mj_geomDistance / client pair_distance on the recorded qpos, capped 300 mm, signed; per control row): fingers = Af (both fingers of the active arm), palm = Ae (EE_BODY), robot_other = C; min over all rows; apple = B over carry rows grasp_t..release_t (Z4; none -> to the end).",
    "safety_violation": "plan C: executed rows of crossing-window chunks whose TO-row reconstruction of the refined row (xm_cross to_ref[k].min_mm = min over all field queries of d - r - margin, 10 mm obstacle margin, held rows margin 0) < 0, share of executed rows of window chunks. measured M: rows whose post-state measured hurdle distance (analysis series at index t+1: min(Af, Ae, C), and B while t in carry) < 10 mm; share of all rows and of crossing-window rows. verdict violated share = server to_status 'violated' chunks / server chunks; HOLD share = HOLD chunks / planning rows.",
    "grasp_retention": "grasp (old) and Z4 release_t exists and apple centre over the crate xy box at release (in_crate_xy_at_release); no grasp -> false. held_to_end = grasp and no release_t (apple still in hand when the run ended).",
    "slip": "replay, state indices from attach_t (first server chunk with attach_now; else grasp_t) to the post state of row release_t - 1 (Y2 window, only with release_t). finger-midpoint: |R_tcp^T(apple - finger_mid)(i) - (same)(attach)|; Y2 TCP: |R_tcp^T(apple - tcp)(i) - (same)(attach)| (y_metrics). max over the window and the value at its last state (just before release). pre_open variant (finger-midpoint): window end = state index of min(release_t, applied_open_t) (post state of the row before the gripper-open command row, or Y2 end if earlier) — the Y2 last state already contains the opening on open_cmd releases.",
    "path_length": "measured, post states, rows grasp_t .. release_t (Z4; both must exist): apple centre path, TCP path (and commanded TCP path), sum of |p(t+1) - p(t)| (mm); window part = steps whose two rows are both in the crossing window (H).",
    "execution_time": "rows (and s = rows / 15): grasp_t (old), crossing_row (H), release_t (Z4), place_t (old); durations release - grasp, crossing - grasp.",
    "latency_server": "server-recorded TO solve_ms per chunk, first chunk of each run excluded; per run median / p90; pod load1 mean during the run (load samples between .start and .end).",
    "latency_controlled": "T43 TP tp_timing (server mode, same V8 chunks, configs rotated in one process): wall_ms per chunk for variants V8 and V11AB, first chunk excluded; per run median / p90; field cuda (= server path) and cpu.",
}


def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def st(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(a), "median": float(np.median(a)), "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())}


def ok(x):
    return x is not None and not (isinstance(x, float) and not np.isfinite(x))


def wil(per, field, stages=("H", "B")):
    """per[key][arm][field] -> paired Wilcoxon d = V11AB - V8."""
    ks = sorted(k for k in per if k.split(":")[0] in stages)
    a = {arm: [per[k][arm].get(field) for k in ks] for arm in ARMS}
    pk = [k for i, k in enumerate(ks) if ok(a["V8"][i]) and ok(a["V11AB"][i])]
    d = [per[k]["V11AB"][field] - per[k]["V8"][field] for k in pk]
    o = {"n_runs": {arm: sum(ok(x) for x in a[arm]) for arm in ARMS}, "V8": st(a["V8"]), "V11AB": st(a["V11AB"]), "n_pairs": len(d),
         "V8_paired_median": float(np.median([per[k]["V8"][field] for k in pk])) if pk else None,
         "V11AB_paired_median": float(np.median([per[k]["V11AB"][field] for k in pk])) if pk else None,
         "diff_V11AB_minus_V8": st(d), "n_V11AB_higher": int(sum(x > 0 for x in d)), "n_V11AB_lower": int(sum(x < 0 for x in d)), "n_tied": int(sum(x == 0 for x in d)),
         "test": "wilcoxon signed-rank two-sided (zero_method wilcox, method auto)"}
    nz = [x for x in d if x != 0]
    if len(nz) >= 1:
        try:
            r = wilcoxon(d)
            o["W"] = float(r.statistic); o["p"] = float(r.pvalue)
        except Exception as ex:
            o["W"] = None; o["p"] = None; o["test_error"] = repr(ex)
    else:
        o["W"] = None; o["p"] = None
    o["n_nonzero"] = len(nz)
    return o


def mcb(per, field, stages=("H", "B")):
    ks = sorted(k for k in per if k.split(":")[0] in stages and per[k]["V8"].get(field) is not None and per[k]["V11AB"].get(field) is not None)
    fa = {k: bool(per[k]["V8"][field]) for k in ks}; fb = {k: bool(per[k]["V11AB"][field]) for k in ks}
    b = sum(fa[k] and not fb[k] for k in ks); c = sum(fb[k] and not fa[k] for k in ks); both = sum(fa[k] and fb[k] for k in ks)
    return {"n_pairs": len(ks), "V8_true": both + b, "V11AB_true": both + c, "both": both, "V8_only": b, "V11AB_only": c, "neither": len(ks) - both - b - c,
            "test": "exact McNemar two-sided", "p": mcn(b, c), "V8_only_keys": [k for k in ks if fa[k] and not fb[k]], "V11AB_only_keys": [k for k in ks if fb[k] and not fa[k]]}


# ------------------------------------------------------------------ inputs
AN = {"V8": json.load(open(T / "V8/analysis/analysis.json"))["runs"], "V11AB": json.load(open(T / "V11/analysis/analysis_V11AB.json"))["runs"]}
CTJ = json.load(open(T / "V11/ct_V11AB.json"))["runs"]
CR = {"V8": json.load(open(T / "XM/raw/cross_V8.json"))["runs"], "V11AB": json.load(open(T / "V11/raw/cross_V11AB.json"))["runs"]}
REPH = {"V8": T / "V8/replay", "V11AB": T / "V11/replay"}
SHARP = json.load(open(T / "V11/sharp_V11AB.json"))
Z.REPLAY = collections.defaultdict(lambda: Q / "replay")


def runs_of(arm):
    out = {}
    for k, r in AN[arm].items():
        if not k.startswith(arm + "/"):
            continue
        st_, nm = k.split("/")[1], k.split("/")[2]
        out[f"{st_}:{nm.split('_ep')[1]}"] = (k, r)
    return out


RUNS = {a: runs_of(a) for a in ARMS}
KEYS = sorted(set(RUNS["V8"]) & set(RUNS["V11AB"]))
assert len(KEYS) == 19, KEYS


def first(it):
    return next(iter(it), None)


def load_file(p):
    L = [l.split() for l in open(p) if len(l.split()) >= 6 and l.split()[-1].startswith("wset=")]
    return np.array([[float(x[0]), float(x[1])] for x in L])


LD = {"V8": load_file(T / "V8/raw/load_samples.txt"), "V11AB": load_file(T / "V11/AB/load_samples_V11AB.txt")}
RUNDIR = {"V8": T / "V8/runs/V8", "V11AB": T / "V11/runs/V11AB"}


def iso(p):
    return datetime.datetime.fromisoformat(open(p).read().splitlines()[0].strip()).timestamp()


def tp_runs(field):
    o = {}
    for f in sorted(glob.glob(f"{_ag3s_workspace_path()}/outputs/impl/T43TP/timing/{field}/V8_V8_*.json")):
        j = json.load(open(f)); st_, nm = j["key"].split("/")[1], j["key"].split("/")[2]
        ch = [c for c in j["chunks"] if c.get("variants")][1:]
        o[f"{st_}:{nm.split('_ep')[1]}"] = {v: [float(np.median(c["variants"][v]["wall_ms"])) for c in ch] for v in ARMS}
        o[f"{st_}:{nm.split('_ep')[1]}"]["load_before_after"] = [j["load_before"]["loadavg"], j["load_after"]["loadavg"]]
    return o


TP = {"cuda": tp_runs("cuda"), "cpu": tp_runs("cpu")}
TPAGG = {f: json.load(open(f"{_ag3s_workspace_path()}/outputs/impl/T43TP/agg/timing_{f}.json")) for f in ("cuda", "cpu")}


def runs_true(mask):
    """maximal runs of True: list of (start, end_inclusive)."""
    out = []; s = None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def qmat(q):
    import mujoco
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


# ------------------------------------------------------------------ per run
per = {k: {} for k in KEYS}
series = {k: {} for k in KEYS}
for key in KEYS:
    stg = key.split(":")[0]
    for arm in ARMS:
        akey, r = RUNS[arm][key]; name = akey.split("/")[2]
        o = {"run": akey, "n_control": r["n_control"]}
        # ---- success (old / Z4 / CT)
        if stg == "H":
            c = CTJ[akey]
            z4 = c
            o.update(success_old=bool(c["success"]), success_z4=bool(c["success_rel"]), success_ct=bool(c["success_ct"]), pre_release_contact=bool(c["pre_release_contact"]),
                     obstacle_stop=bool(c["obstacle_stop"]))
            rep = REPH[arm] / f"{arm}_H_{name}"
        else:
            z4 = Z.score(akey, r)
            o.update(success_old=bool(r["success"]), success_z4=bool(z4["success_rel"]), success_ct=bool(r["success"] and z4["success_rel"]), pre_release_contact=False,
                     obstacle_stop=False)
            rep = Q / "replay" / f"{arm}_B_{name}"
        o["z4_B"] = {k_: z4.get(k_) for k_ in ("release_t", "attach_t", "released", "in_crate_end", "in_crate_xy_at_release", "sep_kept", "release_mode")} if stg == "B" else None
        grasp, grasp_t, place_t = bool(r["grasp"]), r["grasp_t"], r["place_t"]
        rel_t, att_t = z4.get("release_t"), z4.get("attach_t")
        o.update(grasp=grasp, place=bool(r["place"]), grasp_t=grasp_t, place_t=place_t, release_t=rel_t, attach_t=att_t)
        zr = np.load(str(rep) + ".npz"); tt = zr["t"]; N = len(tt)
        rw = np.load(Q / "raw/rows" / f"{arm}_{stg}_{name}.npz")
        assert len(rw["t"]) == N, (akey, N, len(rw["t"]))
        exe = rw["executed"].astype(bool)
        # ---- retention
        o["retained"] = bool(grasp and rel_t is not None and bool(z4.get("in_crate_xy_at_release")))
        o["held_to_end"] = bool(grasp and rel_t is None)
        # ---- slip (Y2 window)
        o["slip_fm_max_mm"] = o["slip_fm_at_release_mm"] = o["slip_tcp_max_mm"] = o["slip_tcp_at_release_mm"] = None
        if grasp and rel_t is not None:
            at = att_t if att_t is not None else (grasp_t or 0)
            ia = int(np.searchsorted(tt, at)) + 1; j = int(np.searchsorted(tt, rel_t))
            Rt = zr["tcp_R"]
            a_fm = np.array([Rt[i].T @ (zr["apple_pos"][i] - zr["finger_mid"][i]) for i in range(N + 1)])
            a_tc = np.array([Rt[i].T @ (zr["apple_pos"][i] - zr["tcp"][i]) for i in range(N + 1)])
            if j >= ia:
                s1 = np.linalg.norm(a_fm[ia:j + 1] - a_fm[ia], axis=1); s2 = np.linalg.norm(a_tc[ia:j + 1] - a_tc[ia], axis=1)
                o.update(slip_fm_max_mm=1e3 * float(s1.max()), slip_fm_at_release_mm=1e3 * float(s1[-1]), slip_tcp_max_mm=1e3 * float(s2.max()),
                         slip_tcp_at_release_mm=1e3 * float(s2[-1]), slip_window_states=[ia, j])
            op_t = z4.get("applied_open_t") if stg == "H" else z4.get("applied_open_t")
            jo = int(np.searchsorted(tt, min(rel_t, op_t))) if op_t is not None else j
            if jo >= ia:
                s3 = np.linalg.norm(a_fm[ia:jo + 1] - a_fm[ia], axis=1)
                o.update(slip_fm_pre_open_max_mm=1e3 * float(s3.max()), slip_fm_pre_open_end_mm=1e3 * float(s3[-1]), slip_pre_open_window_states=[ia, jo])
        # ---- window / crossing (H)
        win = np.zeros(N, bool); xr = xrC = None; cr = None
        if stg == "H":
            cr = CR[arm][akey]; win = np.array([bool(x["in_window"]) for x in cr["rows"]])
            xr = cr["crossing_rows"]["apple_M"]["carry"]; xrC = cr["crossing_rows"]["apple_C"]["carry"]
            o["crossing_t"] = xr; o["crossing_t_C"] = xrC; o["first_inhand_t"] = cr["first_inhand_t"]; o["drop_t"] = cr["drop_t"]; o["n_window_rows"] = int(win.sum())
            cx = cr["crossing"]
            for s_ in ("C", "M"):
                a_ = cx.get(f"carry_apple_{s_}")
                o[f"apple_bottom_minus_bar_top_at_crossing_{s_}_mm"] = a_["apple_bottom_minus_bar_top_mm"] if a_ and a_["inhand"] else None
                o[f"apple_bar_geo_at_crossing_{s_}_mm"] = a_["geo_bar_mm"]["apple"] if a_ and a_["inhand"] else None
            # deflection onset
            dm = rw["dpalm_mag"]; fi = cr["first_inhand_t"]
            if xr is not None:
                cand = [t for t in range((fi or 0), xr) if exe[t] and np.isfinite(dm[t]) and dm[t] > DEFL_MM]
                o["deflection_onset_rows"] = (xr - cand[0]) if cand else None
                o["deflection_onset_t"] = cand[0] if cand else None
                c0 = [t for t in range(0, xr) if exe[t] and np.isfinite(dm[t]) and dm[t] > DEFL_MM]
                o["deflection_onset_rows_from_t0"] = (xr - c0[0]) if c0 else None
                t_ = xr - 1; s0 = None
                while t_ >= 0 and exe[t_] and np.isfinite(dm[t_]) and dm[t_] > DEFL_MM:
                    s0 = t_; t_ -= 1
                o["deflection_onset_rows_contiguous"] = (xr - s0) if s0 is not None else 0
                o["palm_deflection_at_crossing_mm"] = float(dm[xr]) if np.isfinite(dm[xr]) else None
                # lift hold
                lz = rw["dapple_bottom_dz"]
                lifted = np.array([bool(exe[t] and np.isfinite(lz[t]) and lz[t] > DEFL_MM) for t in range(N)])
                run_ = next(((s, e) for s, e in runs_true(lifted) if s <= xr <= e), None)
                o["lift_hold_rows"] = (run_[1] - run_[0] + 1) if run_ else 0
                o["lift_hold_rows_before_crossing"] = (xr - run_[0]) if run_ else 0
                o["lift_hold_rows_after_crossing"] = (run_[1] - xr) if run_ else 0
                o["lift_hold_span"] = list(run_) if run_ else None
                o["lifted_rows_in_window"] = int((lifted & win).sum())
                o["lift_dz_at_crossing_mm"] = float(lz[xr]) if np.isfinite(lz[xr]) else None
                o["lift_dz_max_in_window_mm"] = float(np.nanmax(np.where(win & exe, lz, np.nan))) if np.any(win & exe & np.isfinite(lz)) else None
            else:
                for f_ in ("deflection_onset_rows", "deflection_onset_rows_from_t0", "deflection_onset_rows_contiguous", "lift_hold_rows", "lift_hold_rows_before_crossing",
                           "lift_hold_rows_after_crossing", "lifted_rows_in_window", "palm_deflection_at_crossing_mm", "lift_dz_at_crossing_mm", "lift_dz_max_in_window_mm"):
                    o[f_] = None
            # series for figure 1 (in-hand rows)
            ih = [x for x in cr["rows"] if x["inhand"]]
            series[key][arm] = {"t": [x["t"] for x in ih], "s_M": [x["s_M"]["apple"] for x in ih], "h_M": [x["apple_bottom_minus_bar_top_M"] for x in ih],
                                "s_C": [x["s_C"]["apple"] for x in ih], "h_C": [x["apple_bottom_minus_bar_top_C"] for x in ih], "executed": [x["executed"] for x in ih],
                                "crossing_t": xr, "success_ct": o["success_ct"], "lift_dz_mm": [None if not np.isfinite(rw["dapple_bottom_dz"][x["t"]]) else float(rw["dapple_bottom_dz"][x["t"]]) for x in ih]}
        # ---- collision (H)
        if stg == "H":
            rj = json.load(open(str(rep) + ".json"))
            def phase(t):
                if not grasp or grasp_t is None or t < grasp_t:
                    return "before_grasp"
                return "carry" if (rel_t is None or t <= rel_t) else "return"
            nph = collections.Counter(phase(int(t)) for t in tt)
            crow = {ph: {"robot": set(), "apple": set()} for ph in ("before_grasp", "carry", "return")}
            for x in rj["obstacle_contact_rows"]:
                ph = phase(int(x["t"]))
                for g, v in x["geoms"].items():
                    kind = v[2]; crow[ph]["apple" if kind == "apple" else "robot"].add(int(x["t"]))
            for ph in ("before_grasp", "carry", "return"):
                o[f"rows_{ph}"] = int(nph[ph])
                for who in ("robot", "apple"):
                    o[f"contact_{who}_{ph}"] = bool(crow[ph][who]) if nph[ph] else (False if ph != "return" else None)
                    o[f"contact_{who}_{ph}_rows"] = len(crow[ph][who])
                    o[f"contact_{who}_{ph}_row_share"] = (len(crow[ph][who]) / nph[ph]) if nph[ph] else None
                ra = crow[ph]["robot"] | crow[ph]["apple"]
                o[f"contact_any_{ph}"] = bool(ra) if nph[ph] else None
                o[f"contact_any_{ph}_row_share"] = (len(ra) / nph[ph]) if nph[ph] else None
            o["contact_any_run"] = any(bool(crow[ph]["robot"] | crow[ph]["apple"]) for ph in crow)
            o["contact_any_run_row_share"] = len(set().union(*[crow[ph]["robot"] | crow[ph]["apple"] for ph in crow])) / N
            # ---- min clearance (measured)
            hs = r["hurdle"]["series"]
            Af, Ae, Cc, Bp = (np.asarray(hs[s_], float) for s_ in ("Af", "Ae", "C", "B"))
            o["minclr_fingers_mm"] = 1e3 * float(Af.min()); o["minclr_palm_mm"] = 1e3 * float(Ae.min()); o["minclr_robot_other_mm"] = 1e3 * float(Cc.min())
            o["minclr_hand_mm"] = min(o["minclr_fingers_mm"], o["minclr_palm_mm"])
            if grasp and grasp_t is not None:
                e_ = (rel_t if rel_t is not None else N - 1)
                o["minclr_apple_carry_mm"] = 1e3 * float(Bp[grasp_t:e_ + 1].min())
                # in carry, hand separately
                o["minclr_hand_carry_mm"] = 1e3 * float(np.minimum(Af, Ae)[grasp_t:e_ + 1].min())
                o["minclr_hand_return_mm"] = 1e3 * float(np.minimum(Af, Ae)[e_ + 1:].min()) if e_ + 1 < N else None
            else:
                o["minclr_apple_carry_mm"] = o["minclr_hand_carry_mm"] = o["minclr_hand_return_mm"] = None
            # ---- safety violation
            # measured: post state of row t = series index t+1 (last row: index t)
            idx = np.minimum(np.arange(N) + 1, N - 1)
            rob = np.minimum(np.minimum(Af, Ae), Cc)[idx]
            carry = np.array([phase(int(t)) == "carry" for t in tt])
            dmin = np.where(carry, np.minimum(rob, Bp[idx]), rob)
            vM = dmin * 1e3 < MARGIN_MM
            o["viol_M_rows"] = int(vM.sum()); o["viol_M_row_share"] = float(vM.mean())
            o["viol_M_window_row_share"] = float(vM[win].mean()) if win.any() else None
            # plan C (TO-row reconstruction, window chunks)
            nexe = nv = 0
            for c in cr["chunks"]:
                if not c.get("in_window") or not c.get("to_ref"):
                    continue
                for k_, ex_ in enumerate(c["executed"]):
                    if ex_ and k_ < len(c["to_ref"]) and c["to_ref"][k_] is not None:
                        nexe += 1; nv += c["to_ref"][k_]["min_mm"] < 0
            o["viol_C_window_exec_rows"] = nexe; o["viol_C_rows"] = int(nv); o["viol_C_row_share"] = (nv / nexe) if nexe else None
        # ---- verdict violated / HOLD shares (all)
        sc = r["server_chunks"]
        o["n_server_chunks"] = len(sc); o["violated_chunk_share"] = sum(c.get("to_status") == "violated" for c in sc) / len(sc)
        o["hold_chunk_share"] = r["n_hold_chunks"] / r["n_planning"]; o["n_hold_chunks"] = r["n_hold_chunks"]; o["n_planning"] = r["n_planning"]
        # ---- path length (measured)
        if grasp and grasp_t is not None and rel_t is not None:
            i0, i1 = int(np.searchsorted(tt, grasp_t)), int(np.searchsorted(tt, rel_t))
            ap = zr["apple_pos"][1:]
            seg = lambda P: np.linalg.norm(np.diff(P[i0:i1 + 1], axis=0), axis=1)
            o["path_apple_mm"] = 1e3 * float(seg(ap).sum()); o["path_tcp_M_mm"] = 1e3 * float(seg(rw["tcp_M"]).sum()); o["path_tcp_C_mm"] = 1e3 * float(seg(rw["tcp_C"]).sum())
            if stg == "H":
                wm = (win[i0:i1] & win[i0 + 1:i1 + 1])
                o["path_apple_window_mm"] = 1e3 * float(seg(ap)[wm].sum()); o["path_tcp_M_window_mm"] = 1e3 * float(seg(rw["tcp_M"])[wm].sum())
        # ---- execution time
        o["t_grasp_s"] = None if grasp_t is None else grasp_t / HZ
        o["t_release_s"] = None if rel_t is None else rel_t / HZ
        o["t_place_s"] = None if place_t is None else place_t / HZ
        o["t_crossing_s"] = None if (stg != "H" or xr is None) else xr / HZ
        o["dur_grasp_to_release_s"] = (rel_t - grasp_t) / HZ if (rel_t is not None and grasp_t is not None and grasp) else None
        o["dur_grasp_to_crossing_s"] = (xr - grasp_t) / HZ if (stg == "H" and xr is not None and grasp_t is not None) else None
        o["dur_crossing_to_release_s"] = (rel_t - xr) / HZ if (stg == "H" and xr is not None and rel_t is not None) else None
        # ---- acceleration / jerk
        seqs = rw["seq"]
        for src in ("C", "M"):
            P = rw[f"tcp_{src}"]
            A = {c_: [] for c_ in ("all", "boundary", "inner", "window_in", "window_out", "carry", "window_in_boundary", "window_in_carry")}
            Jk = {c_: [] for c_ in A}
            for t in range(1, N - 1):
                if not (exe[t - 1] and exe[t] and exe[t + 1]):
                    continue
                a = float(np.linalg.norm(P[t + 1] - 2 * P[t] + P[t - 1])) * HZ ** 2
                bnd = len({int(seqs[t - 1]), int(seqs[t]), int(seqs[t + 1])}) > 1
                incarry = grasp and grasp_t is not None and t >= grasp_t and (rel_t is None or t <= rel_t)
                for c_, f_ in (("all", True), ("boundary", bnd), ("inner", not bnd), ("window_in", stg == "H" and win[t]), ("window_out", stg == "H" and not win[t]),
                               ("carry", incarry), ("window_in_boundary", stg == "H" and win[t] and bnd), ("window_in_carry", stg == "H" and win[t] and incarry)):
                    if f_:
                        A[c_].append(a)
            for t in range(2, N - 1):
                if not (exe[t - 2] and exe[t - 1] and exe[t] and exe[t + 1]):
                    continue
                jv = float(np.linalg.norm(P[t + 1] - 3 * P[t] + 3 * P[t - 1] - P[t - 2])) * HZ ** 3
                bnd = len({int(seqs[t - 2]), int(seqs[t - 1]), int(seqs[t]), int(seqs[t + 1])}) > 1
                incarry = grasp and grasp_t is not None and t >= grasp_t and (rel_t is None or t <= rel_t)
                for c_, f_ in (("all", True), ("boundary", bnd), ("inner", not bnd), ("window_in", stg == "H" and win[t]), ("window_out", stg == "H" and not win[t]),
                               ("carry", incarry), ("window_in_boundary", stg == "H" and win[t] and bnd), ("window_in_carry", stg == "H" and win[t] and incarry)):
                    if f_:
                        Jk[c_].append(jv)
            for c_ in A:
                for nm_, arr in (("acc", A[c_]), ("jerk", Jk[c_])):
                    if arr:
                        o[f"{nm_}_{src}_{c_}_median"] = float(np.median(arr)); o[f"{nm_}_{src}_{c_}_p90"] = float(np.percentile(arr, 90)); o[f"{nm_}_{src}_{c_}_max"] = float(np.max(arr))
                        o[f"{nm_}_{src}_{c_}_n"] = len(arr)
                    else:
                        o[f"{nm_}_{src}_{c_}_median"] = o[f"{nm_}_{src}_{c_}_p90"] = o[f"{nm_}_{src}_{c_}_max"] = None; o[f"{nm_}_{src}_{c_}_n"] = 0
        # ---- joint-space boundary acceleration (sharp_V11AB, T43TA rule)
        sp = SHARP["arms"][arm]["per_run"].get(akey) or {}
        o["joint_acc_boundary_rad_median"] = float(np.median(sp["acc_boundary_rad"])) if sp.get("acc_boundary_rad") else None
        o["joint_acc_boundary_rad_max"] = float(np.max(sp["acc_boundary_rad"])) if sp.get("acc_boundary_rad") else None
        # ---- latency (server record)
        sv = [c["solve_ms"] for c in sc[1:] if c.get("solve_ms") is not None]
        o["to_server_median_ms"] = float(np.median(sv)) if sv else None; o["to_server_p90_ms"] = float(np.percentile(sv, 90)) if sv else None
        o["to_server_n_chunks"] = len(sv)
        try:
            t0, t1 = iso(RUNDIR[arm] / stg / f"{name}.start"), iso(RUNDIR[arm] / stg / f"{name}.end")
            msk = (LD[arm][:, 0] >= t0) & (LD[arm][:, 0] <= t1)
            o["load1_mean_during_run"] = float(LD[arm][msk, 1].mean()) if msk.any() else None; o["load1_n_samples"] = int(msk.sum())
        except Exception as ex:
            o["load1_mean_during_run"] = None; o["load1_error"] = repr(ex)
        # ---- latency (controlled T43 TP, same V8 chunks)
        for fld in ("cuda", "cpu"):
            x = TP[fld].get(key)
            o[f"to_tp_{fld}_median_ms"] = float(np.median(x[arm])) if x else None
            o[f"to_tp_{fld}_p90_ms"] = float(np.percentile(x[arm], 90)) if x else None
        per[key][arm] = o

# ------------------------------------------------------------------ paired statistics
out = {"definitions": DEFS, "doc": __doc__, "keys": KEYS, "per_run": per}
M = {}
for f in ("success_old", "success_z4", "success_ct", "retained", "grasp", "held_to_end"):
    M[f] = {"ALL": mcb(per, f), "H": mcb(per, f, ("H",)), "B": mcb(per, f, ("B",))}
M["retained_given_grasp"] = {a: {"n_grasp": sum(per[k][a]["grasp"] for k in KEYS), "n_retained": sum(per[k][a]["retained"] for k in KEYS)} for a in ARMS}
for ph in ("before_grasp", "carry", "return"):
    for who in ("robot", "apple", "any"):
        f = f"contact_{who}_{ph}"
        M[f] = mcb(per, f, ("H",))
        M[f + "_row_share"] = wil(per, f + "_row_share", ("H",))
M["contact_any_run"] = mcb(per, "contact_any_run", ("H",)); M["contact_any_run_row_share"] = wil(per, "contact_any_run_row_share", ("H",))
for f in ("minclr_fingers_mm", "minclr_palm_mm", "minclr_hand_mm", "minclr_robot_other_mm", "minclr_apple_carry_mm", "minclr_hand_carry_mm", "minclr_hand_return_mm",
          "viol_M_row_share", "viol_M_window_row_share", "viol_C_row_share",
          "apple_bottom_minus_bar_top_at_crossing_C_mm", "apple_bottom_minus_bar_top_at_crossing_M_mm", "apple_bar_geo_at_crossing_C_mm", "apple_bar_geo_at_crossing_M_mm",
          "deflection_onset_rows", "deflection_onset_rows_from_t0", "deflection_onset_rows_contiguous", "palm_deflection_at_crossing_mm",
          "lift_hold_rows", "lift_hold_rows_before_crossing", "lift_hold_rows_after_crossing", "lifted_rows_in_window", "lift_dz_at_crossing_mm", "lift_dz_max_in_window_mm",
          "t_crossing_s", "dur_grasp_to_crossing_s", "dur_crossing_to_release_s", "path_apple_window_mm", "path_tcp_M_window_mm", "n_window_rows"):
    M[f] = wil(per, f, ("H",))
for f in ("violated_chunk_share", "hold_chunk_share", "slip_fm_max_mm", "slip_fm_at_release_mm", "slip_fm_pre_open_max_mm", "slip_fm_pre_open_end_mm", "slip_tcp_max_mm", "slip_tcp_at_release_mm",
          "path_apple_mm", "path_tcp_M_mm", "path_tcp_C_mm", "t_grasp_s", "t_release_s", "t_place_s", "dur_grasp_to_release_s",
          "joint_acc_boundary_rad_median", "joint_acc_boundary_rad_max", "to_server_median_ms", "to_server_p90_ms", "load1_mean_during_run",
          "to_tp_cuda_median_ms", "to_tp_cuda_p90_ms", "to_tp_cpu_median_ms", "to_tp_cpu_p90_ms"):
    M[f] = wil(per, f)
for nm_ in ("acc", "jerk"):
    for src in ("C", "M"):
        for c_ in ("all", "boundary", "inner", "carry", "window_in", "window_out", "window_in_boundary", "window_in_carry"):
            for s_ in ("median", "p90", "max"):
                f = f"{nm_}_{src}_{c_}_{s_}"
                M[f] = wil(per, f, ("H",) if c_.startswith("window") else ("H", "B"))
out["paired"] = M
# pooled server latency + controlled TP pooled tables (not mixed)
out["latency_server_pooled"] = {a: st([c["solve_ms"] for k in KEYS for c in RUNS[a][k][1]["server_chunks"][1:] if c.get("solve_ms") is not None]) for a in ARMS}
out["load1_during_runs"] = {a: st([per[k][a]["load1_mean_during_run"] for k in KEYS]) for a in ARMS}
out["latency_tp_controlled_pooled"] = {f: {"load": TPAGG[f]["load"], "n_runs": TPAGG[f]["n_runs"],
                                           "V8": {k_: TPAGG[f]["table"]["V8"][k_] for k_ in ("n_chunks", "wall_ms", "stage_median_ms")},
                                           "V11AB": {k_: TPAGG[f]["table"]["V11AB"][k_] for k_ in ("n_chunks", "wall_ms", "stage_median_ms")}} for f in ("cuda", "cpu")}
out["fig1_series"] = series
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
print("written", sys.argv[1])
for f, v in M.items():
    if "p" in v and "V8_true" in v:
        print(f"{f:42s} V8 {v['V8_true']:2d} V11AB {v['V11AB_true']:2d} /{v['n_pairs']}  b {v['V8_only']} c {v['V11AB_only']}  p {v['p']:.3f}")
    elif "ALL" in v:
        for s_ in ("ALL", "H", "B"):
            x = v[s_]; print(f"{f + ' ' + s_:42s} V8 {x['V8_true']:2d} V11AB {x['V11AB_true']:2d} /{x['n_pairs']}  b {x['V8_only']} c {x['V11AB_only']}  p {x['p']:.3f}")
    elif "n_pairs" in v and "diff_V11AB_minus_V8" in v:
        p = v.get("p"); print(f"{f:42s} n {v['n_pairs']:2d}  V8 {v['V8_paired_median']}  V11AB {v['V11AB_paired_median']}  dmed {v['diff_V11AB_minus_V8'].get('median')}  W {v.get('W')}  p {p}")
