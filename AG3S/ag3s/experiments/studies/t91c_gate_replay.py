"""T91c — fixed-input replay: does the T91b phase machine give the 3-stage gate a different label than the grasp latch (T92)?

Reads the T92 closed-loop records of the P0m configuration (``--subtask-from-latch`` servers; read only) and, per run and per
request, builds what the server would have handed `PhaseGate` (`AG3S/trajopt/phase_gate.py`, the code `SafePolicy` runs with
``--subtask-from-phase-machine``):

  measured opening          client step ``state[7 | 15]`` (the policy observation's gripper column)
  hand site                 MuJoCo forward kinematics of the client ``qpos`` (the MJCF ``<side>_ee`` site, base frame; equals the
                            server's ``hold_site_position``)
  ATS target / destination  the **previous** request's ``manipulated`` / ``destination`` of the server chunk record (the server
                            hands the label to AG3S before this request's grounding); none at request 0
  VLA intent                server chunk record ``reference_chunk[:, gripper column]`` (the policy's own chunk)
  wrist depth               client step ``depth_wrist_cam_<l|r>`` (uint16 mm, what the request carried), stride 4

and compares per request

  (a) the latch label   recorded ``summary_json.subtask_latch.label`` (the one-hot the T92 server gave AG3S)
  (b) the phase-machine label (`PhaseGate`)
  (c) the MuJoCo ground truth (T77 definitions: ``t77_labels.phase5_sequence`` -> coarse pick / place / home), if the model loads

Each label stream goes through AG3S's own `SubtaskDebounce` (3 consecutive identical requests) and the gate's rule (the confirmed
label in {place, home} blocks a new manipulated object). Reported (per run and in total):

  n_label_differs            requests where (a) != (b)
    .pm_right / .latch_right / .both_wrong   which side the ground truth supports
  n_gated_differs            requests where the debounced gate state (place/home = blocking) differs
  gate_decision_changes      of those, the ones where a recorded event decides it:
    pm_would_block           pm gates, latch does not, and the recorded run adopted a different manipulated object at that
                             request (``manipulated.id`` changed) -> the phase machine would have blocked that adoption
    latch_blocked_pm_open    latch gates and its record says ``subtask.blocked`` (first | switch), pm does not gate -> the
                             phase machine would have let that adoption through
  gated_before_place         requests where the source gates while the ground truth is still pick (approach / grasp)
  danger                     ground truth approach / grasp read as carry / place (5-stage) — pm; latch (held->place only)

Pairing of records: ``server_*/order.txt`` lists the runs in server order; the i-th chunk segment (split where ``t_step``
restarts) is the i-th run, kept only when its length equals the client's step count (a server restarted mid-run leaves a stray
short segment; the run is then taken from the server that holds a full segment). ``HURDLE_ep211_s2113`` and any scene in
`t91_phase_eval.EXCLUDED_SCENES` are skipped.

usage (CPU, .venv-ag3s; MuJoCo for FK and the ground truth)
  PYTHONPATH=$PWD MUJOCO_GL=osmesa .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t91c_gate_replay \\
      --records <.../verify/T92/CL/p0m/k0> --out <dir>/t91c_gate_replay.json [--only REG_ep954_s9543 ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.stages.subtask_probe import SubtaskDebounce
from AG3S.trajopt.phase_gate import PhaseGate

GRIP_COL = {"left": 7, "right": 15}
WRIST_KEY = {"left": "depth_wrist_cam_l", "right": "depth_wrist_cam_r"}
EXCLUDED_RUNS = ("HURDLE_ep211_s2113",)
COARSE = {"approach": "pick", "grasp": "pick", "carry": "place", "place": "place", "retreat": "home"}
GATING = ("place", "home")
DEBOUNCE_FRAMES = 3                        # `clustering.subtask_confirm_frames` None -> `target_confirm_frames` (3), as recorded


# ----------------------------------------------------------------------------------------------------------- records
def pair_records(records: pathlib.Path) -> list[dict]:
    """``[{name, group, server, chunks}]``: every run of the records with its full chunk segment (see module docstring)."""
    from AG3S.ag3s.experiments.studies import t77_labels as L

    cands: list[dict] = []
    for sd in sorted(records.glob("server_*")):
        order = (sd / "order.txt").read_text().split() if (sd / "order.txt").exists() else []
        segs = L.server_segments(sd)
        for name, seg in zip(order, segs):
            cands.append({"name": name, "server": sd.name, "chunks": seg["chunks"], "seed": seg["seed"]})
    out = []
    first_server = sorted(p.name for p in records.glob("server_*"))[0]
    for rd in sorted(p for p in (records / "runs").iterdir() if p.is_dir()):
        n_steps = len(list((rd / "ag3s" / "run_0000").glob("step_*.npz")))
        full = [c for c in cands if c["name"] == rd.name and len(c["chunks"]) == n_steps]
        if not full:
            out.append({"name": rd.name, "skipped": f"no server segment with {n_steps} chunks "
                                                    f"(have {[len(c['chunks']) for c in cands if c['name'] == rd.name]})"})
            continue
        c = full[-1]
        out.append({"name": rd.name, "client_dir": str(rd), "server": c["server"], "chunks": c["chunks"],
                    "group": "core9" if c["server"] == first_server else "hurdle_d1", "n": n_steps})
    return out


def client_info(client_dir: str) -> dict:
    from AG3S.ag3s.experiments.studies import t77_labels as L

    return L._client_info(pathlib.Path(client_dir))


def load_request_inputs(entry: dict, fk, probe) -> list[dict]:
    """Per request the machine's inputs plus the recorded values, in order. `probe` may be None (no ground truth)."""
    from AG3S.ag3s.experiments.studies import t77_labels as L

    info = client_info(entry["client_dir"])
    side = info["side"]
    steps = L._step_tsteps(pathlib.Path(entry["client_dir"], "ag3s", "run_0000"))
    col = GRIP_COL[side]
    rows, z0 = [], None
    held, palm, basket = [], [], []
    prev = None
    for j, ((t_step, path), cf) in enumerate(zip(steps, entry["chunks"])):
        with np.load(path) as st:
            qpos = np.asarray(st["qpos"], float)
            state = np.asarray(st["state"], float)
            depth = np.asarray(st[WRIST_KEY[side]]) if WRIST_KEY[side] in st.files else None
        with np.load(cf, allow_pickle=True) as z:
            summ = json.loads(str(z["summary_json"]))
            ref = np.asarray(z["reference_chunk"], float)[:, col]
        ee, _apple = fk.at(qpos, side)
        man_prev = (prev or {}).get("manipulated") or {}
        dst_prev = (prev or {}).get("destination") or {}
        row = {
            "request": j, "t_step": int(t_step), "side": side,
            "gripper_raw": float(state[col]), "ee": ee,
            "target": man_prev.get("centroid"), "target_fresh": man_prev.get("state") == "visible",
            "destination": dst_prev.get("centroid"), "chunk": ref, "wrist_depth_mm": depth,
            "latch_label": (summ.get("subtask_latch") or {}).get("label"),
            "latch_phase": (summ.get("subtask_latch") or {}).get("phase"),
            "recorded_confirmed": (summ.get("subtask") or {}).get("label"),
            "recorded_blocked": (summ.get("subtask") or {}).get("blocked"),
            "recorded_would_block": (summ.get("subtask") or {}).get("would_block"),
            "manipulated_id": (summ.get("manipulated") or {}).get("id"),
            "prev_manipulated_id": man_prev.get("id"),
        }
        if probe is not None:
            mes = probe.measure(qpos, side)
            if "error" not in mes:
                if z0 is None:
                    z0 = mes["apple_z_base"]
                held.append(L.gt_held(mes["fingers_touch"], (mes["apple_z_base"] - z0) * 1e3))
                palm.append(mes["palm_dist_m"])
                basket.append(mes["in_basket"])
                row["_gt_i"] = len(held) - 1
        rows.append(row)
        prev = summ
    if held:
        seq = L.phase5_sequence(held, palm, basket)
        for r in rows:
            r["gt5"] = seq[r["_gt_i"]] if "_gt_i" in r else None
    for r in rows:
        r.pop("_gt_i", None)
        r.setdefault("gt5", None)
        r["gt3"] = COARSE.get(r["gt5"]) if r["gt5"] else None
    return rows


# --------------------------------------------------------------------------------------------------------- replay
def run_gate(rows: Sequence[dict]) -> list[dict]:
    """`PhaseGate` over the requests of one run, one step per request in order (a fresh gate = the server's reset())."""
    gate = PhaseGate((GRIP_COL["left"], GRIP_COL["right"]))
    out = []
    for r in rows:
        p, rec = gate.step(hand=r["side"], gripper_raw=r["gripper_raw"], ee_pos=r["ee"], target_pos=r["target"],
                           target_fresh=r["target_fresh"], destination_pos=r["destination"], chunk_gripper=r["chunk"],
                           wrist_depth_mm=r["wrist_depth_mm"])
        out.append({"p": p, "record": rec})
    return out


def debounce(labels: Sequence[Optional[str]]) -> list[dict]:
    """AG3S's `SubtaskDebounce` over a label stream (None = no label): confirmed label and the gate state after each request."""
    deb = SubtaskDebounce(DEBOUNCE_FRAMES)
    out = []
    for lab in labels:
        deb.update(None if lab is None else {c: (1.0 if c == lab else 0.0) for c in ("pick", "place", "home")})
        out.append({"confirmed": deb.label, "gated": bool(deb.gated)})
    return out


def analyse_run(rows: Sequence[dict], gated: Sequence[dict]) -> dict:
    """Per-request comparison and the run's counts. `gated` = `run_gate` output."""
    n = len(rows)
    latch = [r["latch_label"] for r in rows]
    pm = [None if g["p"] is None else max(g["p"], key=g["p"].get) for g in gated]
    stage = [g["record"].get("stage") for g in gated]
    d_latch, d_pm = debounce(latch), debounce(pm)
    gt3 = [r["gt3"] for r in rows]
    gt5 = [r["gt5"] for r in rows]
    has_gt = any(g is not None for g in gt3)
    differs = [i for i in range(n) if latch[i] != pm[i]]
    gated_differs = [i for i in range(n) if d_latch[i]["gated"] != d_pm[i]["gated"]]
    pm_would_block, latch_blocked_pm_open = [], []
    for i in gated_differs:
        r = rows[i]
        if d_pm[i]["gated"] and not d_latch[i]["gated"]:
            if r["manipulated_id"] is not None and r["manipulated_id"] != r["prev_manipulated_id"]:
                pm_would_block.append(i)
        elif d_latch[i]["gated"] and not d_pm[i]["gated"] and r["recorded_blocked"]:
            latch_blocked_pm_open.append(i)
    sides = {"pm_right": 0, "latch_right": 0, "both_wrong": 0, "both_right": 0, "no_gt": 0}
    for i in differs:
        if gt3[i] is None:
            sides["no_gt"] += 1
        elif pm[i] == gt3[i] and latch[i] != gt3[i]:
            sides["pm_right"] += 1
        elif latch[i] == gt3[i] and pm[i] != gt3[i]:
            sides["latch_right"] += 1
        else:
            sides["both_wrong"] += 1
    sides.pop("both_right")

    def gated_before_place(d):
        return sum(1 for i in range(n) if d[i]["gated"] and gt3[i] == "pick") if has_gt else None

    def acc(lab):
        ok = [(lab[i] == gt3[i]) for i in range(n) if gt3[i] is not None]
        return None if not ok else round(float(np.mean(ok)), 4)

    pm_danger = sum(1 for i in range(n) if gt5[i] in ("approach", "grasp") and stage[i] in ("carry", "place")) if has_gt else None
    latch_danger = sum(1 for i in range(n) if gt5[i] in ("approach", "grasp") and latch[i] == "place") if has_gt else None
    rec_match = [d_latch[i]["confirmed"] == rows[i]["recorded_confirmed"] for i in range(n)]
    return {
        "n_requests": n, "has_ground_truth": has_gt,
        "n_label_differs": len(differs), "label_differs_requests": differs, "label_differs_vs_gt": sides,
        "n_gated_differs": len(gated_differs), "gated_differs_requests": gated_differs,
        "gate_decision_changes": {"pm_would_block": pm_would_block, "latch_blocked_pm_open": latch_blocked_pm_open,
                                  "n": len(pm_would_block) + len(latch_blocked_pm_open)},
        "gated_before_place": {"latch": gated_before_place(d_latch), "phase_machine": gated_before_place(d_pm)},
        "recorded_blocked_requests": [i for i, r in enumerate(rows) if r["recorded_blocked"]],
        "accuracy_3stage": {"latch": acc(latch), "phase_machine": acc(pm)},
        "danger_5stage": {"phase_machine": pm_danger, "latch": latch_danger},
        "latch_debounce_matches_record": round(float(np.mean(rec_match)), 4),
        "pm_no_label_requests": sum(1 for x in pm if x is None),
        "pm_first_label_per_stage": {s: next((i for i in range(n) if stage[i] == s), None)
                                     for s in ("approach", "grasp", "carry", "place", "retreat")},
        "latch_first_label": {s: next((i for i in range(n) if latch[i] == s), None) for s in ("pick", "place", "home")},
        "per_request": {
            "request": [r["request"] for r in rows], "t_step": [r["t_step"] for r in rows],
            "gt5": gt5, "gt3": gt3, "latch_phase": [r["latch_phase"] for r in rows], "latch": latch, "pm_stage": stage,
            "pm": pm, "latch_confirmed": [d["confirmed"] for d in d_latch], "pm_confirmed": [d["confirmed"] for d in d_pm],
            "latch_gated": [d["gated"] for d in d_latch], "pm_gated": [d["gated"] for d in d_pm],
            "recorded_blocked": [r["recorded_blocked"] for r in rows],
            "gap_depth": [(g["record"].get("evidence") or {}).get("gap_depth") for g in gated],
            "gap_hit": [(g["record"].get("evidence") or {}).get("gap_hit") for g in gated],
            "open_norm": [(g["record"].get("evidence") or {}).get("open_norm") for g in gated],
            "closed": [(g["record"].get("evidence") or {}).get("closed") for g in gated],
            "wrist_depth_available": [g["record"]["inputs"]["wrist_depth"] for g in gated],
        },
    }


def total(results: Sequence[dict]) -> dict:
    keys = ("n_requests", "n_label_differs", "n_gated_differs")
    t: dict[str, Any] = {k: int(sum(r["analysis"][k] for r in results)) for k in keys}
    t["n_runs"] = len(results)
    t["gate_decision_changes"] = {
        k: int(sum(len(r["analysis"]["gate_decision_changes"][k]) for r in results))
        for k in ("pm_would_block", "latch_blocked_pm_open")}
    t["gate_decision_changes"]["n"] = sum(t["gate_decision_changes"].values())
    t["runs_with_decision_change"] = [r["name"] for r in results if r["analysis"]["gate_decision_changes"]["n"] > 0]
    t["runs_with_label_difference"] = [r["name"] for r in results if r["analysis"]["n_label_differs"] > 0]
    vs = collections.Counter()
    for r in results:
        vs.update(r["analysis"]["label_differs_vs_gt"])
    t["label_differs_vs_gt"] = dict(vs)
    for k, sub in (("gated_before_place", ("latch", "phase_machine")), ("danger_5stage", ("phase_machine", "latch"))):
        t[k] = {s: (None if any(r["analysis"][k][s] is None for r in results)
                    else int(sum(r["analysis"][k][s] for r in results))) for s in sub}
    return t


# ------------------------------------------------------------------------------------------------------------- main
def build_probe(xml: str, entry: dict):
    from AG3S.ag3s.experiments.studies import t77_labels as L

    info = client_info(entry["client_dir"])
    return L.SceneProbe(xml, tuple(info["obstacle"]))


def main(argv: Optional[Sequence[str]] = None) -> None:
    from AG3S.ag3s.experiments.studies import t77_labels as L
    from AG3S.ag3s.experiments.studies.t91_phase_eval import FK

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True, help="the T92 P0m records dir (contains runs/ and server_*/)")
    ap.add_argument("--out", required=True, help="output json")
    ap.add_argument("--model-xml", default=None, help="MuJoCo XML for FK and ground truth (default: the T77/T91 scene XML)")
    ap.add_argument("--only", nargs="+", default=None, help="run names (e.g. REG_ep954_s9543)")
    ap.add_argument("--no-gt", action="store_true", help="skip the MuJoCo ground truth")
    args = ap.parse_args(argv)

    records = pathlib.Path(args.records)
    xml = args.model_xml or L.default_xml()
    fk = FK(xml)
    results, skipped = [], []
    for e in pair_records(records):
        name = e["name"]
        if name in EXCLUDED_RUNS or (args.only and name not in args.only):
            continue
        if "skipped" in e:
            skipped.append({"name": name, "reason": e["skipped"]})
            continue
        probe = None
        if not args.no_gt:
            try:
                probe = build_probe(xml, e)
            except Exception as exc:  # noqa: BLE001 — ground truth is optional, say so
                print(f"[warn] {name}: no ground truth ({type(exc).__name__}: {exc})", file=sys.stderr)
        rows = load_request_inputs(e, fk, probe)
        an = analyse_run(rows, run_gate(rows))
        results.append({"name": name, "group": e["group"], "server": e["server"], "analysis": an})
        print(f"{name}: n={an['n_requests']} label_differs={an['n_label_differs']} gated_differs={an['n_gated_differs']} "
              f"decision_changes={an['gate_decision_changes']['n']} gated_before_place={an['gated_before_place']} "
              f"danger={an['danger_5stage']} latch_debounce_ok={an['latch_debounce_matches_record']}", flush=True)
    out = {"step": "T91c", "records": str(records), "model_xml": xml, "excluded": list(EXCLUDED_RUNS), "skipped": skipped,
           "total": total(results) if results else None,
           "total_by_group": {g: total([r for r in results if r["group"] == g])
                              for g in sorted({r["group"] for r in results})},
           "runs": results, "doc": __doc__}
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out["total"], ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
