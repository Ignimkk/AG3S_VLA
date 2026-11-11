#!/usr/bin/env python3
"""P2 (제약 단계 추종, 2026-11-10 KST): how well does the current stage signal of pe123458 (the opening-rule grasp latch)
follow the constraint stages S1 approach / S2 grasp / S3 carry+place / S4 released, against simulator contact truth?

Definitions are fixed in `outputs/2026-11-10_제약단계_추종/README.md` §3–§4 (written before this script ran).

truth (frame f, image instant):
  grip(f)  both finger groups touch the BDDL target (contact export), gaps <= 2 frames filled
  close(f) the env action that produced frame f closes (action[f-1, -1] > 0)
  S2       a close run from its start to the frame before its first grip frame (whole run if never grip)
  S3       grip frames
  S4       after a grip segment that ends with the target xy inside the destination xy box (+-1 cm), until the next close run
  S1       everything else
  E-fail   close run without grip; time = first frame with finger gap <= 3 mm in the run, else the run end
  E-slip   grip segment ending outside the destination box; time = first frame after the segment
reader (request k at frame obs_index): latch `grasp.state` searching/latched -> S1, closing -> S2, held -> S3, placed -> S4.

usage: .venv-ag3s/bin/python experiments/segmentation/stage_track/p2_latch_boundary.py OUT_DIR
  (reads OUT_DIR/contacts/{libero,safelibero}/<tag>/phase_*.npz, writes OUT_DIR/p2/)
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import pathlib
import sys

import numpy as np

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
VSUFFIX = "-vla_v11ab-AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458-C"
STAGES = ("S1", "S2", "S3", "S4")
LATCH_STAGE = {"searching": 0, "latched": 0, "closing": 1, "held": 2, "placed": 3}
FILL, EMPTY_GAP, BOX_PAD, MIN_GRIP = 2, 0.003, 0.01, 5


def group_of(tag: str) -> str:
    if "libero9" in tag or "libero4" in tag:
        return "LIBERO"
    if "safelibero11" in tag:
        return "SafeLIBERO11"
    return "E1a-" + tag.split("-LI-")[1][:2]          # qe-pe123458-e1a-spatial-LI-t0-LI -> E1a-t0


def run_dir(tag: str) -> pathlib.Path:
    cands = sorted(RESULTS.glob(f"*_{tag}{VSUFFIX}"))
    if not cands:
        raise FileNotFoundError(tag)
    return cands[-1]


def fill_gaps(x: np.ndarray, n: int) -> np.ndarray:
    x = x.copy()
    idx = np.nonzero(x)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= n + 1:
            x[a:b] = True
    return x


def segments(mask: np.ndarray):
    out, f, N = [], 0, len(mask)
    while f < N:
        if mask[f]:
            s = f
            while f < N and mask[f]:
                f += 1
            out.append((s, f))                       # [s, f)
        else:
            f += 1
    return out


def truth(C, E):
    N = len(C["target_xpos"])
    grip = fill_gaps(np.asarray(C["contact_left_finger_target"]) & np.asarray(C["contact_right_finger_target"]), FILL)
    # amendment A1 (README §3, 2026-11-10 16:4x, from the 6-episode smoke test, before the full run): a grip segment shorter
    # than one request interval (5 frames) that ends outside the destination box is a touch during a failing close, not S3
    box0, xy0 = C["destination_xy_box"], C["target_xpos"][:, :2]
    for s, e in segments(grip):
        b, p = box0[e - 1], xy0[e - 1]
        inside = (b[0] - BOX_PAD <= p[0] <= b[2] + BOX_PAD) and (b[1] - BOX_PAD <= p[1] <= b[3] + BOX_PAD)
        if e - s < MIN_GRIP and not inside:
            grip[s:e] = False
    q = E["obs_robot0_gripper_qpos"]
    gap = q[:, 0] - q[:, 1]
    act = E["action"]
    close = np.zeros(N, bool)
    close[1:] = act[:N - 1, -1] > 0
    stage = np.zeros(N, int)
    events = []
    runs = segments(close)
    for s, e in runs:
        g = np.nonzero(grip[s:e])[0]
        stage[s:s + (g[0] if len(g) else e - s)] = 1
        if not len(g):
            emp = np.nonzero(gap[s:e] <= EMPTY_GAP)[0]
            events.append({"type": "fail", "t": int(s + emp[0]) if len(emp) else int(e), "run": [int(s), int(e)],
                           "outcome": "EMPTY" if len(emp) else "PARTIAL", "min_gap_mm": float(gap[s:e].min() * 1e3)})
    gsegs = segments(grip)
    stage[grip] = 2
    box = C["destination_xy_box"]
    xy = C["target_xpos"][:, :2]
    for s, e in gsegs:
        last = e - 1
        b = box[last]
        inside = bool((xy[last, 0] >= b[0] - BOX_PAD) and (xy[last, 0] <= b[2] + BOX_PAD)
                      and (xy[last, 1] >= b[1] - BOX_PAD) and (xy[last, 1] <= b[3] + BOX_PAD))
        if e >= N:
            continue                                  # episode ended while holding
        if inside:
            nxt = [rs for rs, _ in runs if rs >= e]
            end = nxt[0] if nxt else N
            m = np.arange(e, end)
            m = m[stage[m] == 0]
            stage[m] = 3
        else:
            events.append({"type": "slip", "t": int(e), "seg": [int(s), int(e)]})
    return stage, events, gsegs, gap


def consecutive(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def evaluate(out_root: pathlib.Path):
    rows, eps = [], []
    for kind in ("libero", "safelibero"):
        for tagdir in sorted((out_root / "contacts" / kind).glob("*")):
            if not tagdir.is_dir():
                continue
            tag = tagdir.name
            rd = run_dir(tag)
            meta = {}
            for l in (rd / "episodes.jsonl").read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    meta[f"{r['suite']}_t{r['task_id']}_e{r['episode_idx']}"] = r
            for cf in sorted(tagdir.glob("phase_*.npz")):
                stem = cf.stem[len("phase_"):]
                C = dict(np.load(cf, allow_pickle=False))
                E = dict(np.load(rd / "episodes" / f"{stem}.npz", allow_pickle=False))
                calls = [json.loads(l) for l in (rd / "episodes" / f"{stem}_calls.jsonl").read_text().splitlines() if l.strip()]
                obs = [int(c["obs_index"]) for c in calls]
                if obs != [int(x) for x in C["call_frames"]]:
                    raise ValueError(f"{tag}/{stem}: call frames differ")
                stage, events, gsegs, gap = truth(C, E)
                lat = [LATCH_STAGE.get((c.get("grasp") or {}).get("state"), -1) for c in calls]
                tr = [int(stage[o]) for o in obs]
                m = meta[stem]
                ok = bool(m.get("official_success"))
                ep = {"tag": tag, "group": group_of(tag), "stem": stem, "success": ok, "n_req": len(obs),
                      "status": m.get("status"), "events": events, "grip_segments": [list(map(int, g)) for g in gsegs]}
                # S3 entry latency per grip segment
                ent = []
                for s, e in gsegs:
                    hit = [o for o, l in zip(obs, lat) if s <= o < e and l == 2]
                    inreq = [o for o in obs if s <= o < e]
                    ent.append({"seg": [int(s), int(e)], "n_req": len(inreq),
                                "latency": (hit[0] - s) if hit else None, "missed": not hit and len(inreq) > 0})
                ep["s3_entry"] = ent
                # event detection
                det = []
                for ev in events:
                    t = ev["t"]
                    nxt = [rs for rs, _ in segments(stage == 1) if rs > t] + [len(stage)]
                    horizon = nxt[0]
                    win = [(o, l) for o, l in zip(obs, lat) if t <= o < horizon]
                    good = [o for o, l in win if (l == 0 if ev["type"] == "fail" else l != 2)]
                    fa = any(l == 2 for o, l in zip(obs, lat) if ev.get("run", [0, 0])[0] <= o <= t) if ev["type"] == "fail" else None
                    det.append({**ev, "horizon": int(horizon), "n_req_window": len(win), "latency": (good[0] - t) if good else None,
                                "undetected": bool(win) and not good, "false_attach_in_run": fa})
                ep["event_detection"] = det
                # per-request error types
                after_event = np.zeros(len(stage), bool)
                for ev in events:
                    nxt = [rs for rs, _ in segments(stage == 1) if rs > ev["t"]] + [len(stage)]
                    after_event[ev["t"]:nxt[0]] = True
                miss = [t == 2 and l != 2 for t, l in zip(tr, lat)]
                fh = [t in (0, 1) and l == 2 for t, l in zip(tr, lat)]
                stale = [bool(after_event[o]) and t == 0 and l in (1, 2) for o, t, l in zip(obs, tr, lat)]
                ep["max_run"] = {"miss_S3": consecutive(miss), "false_held": consecutive(fh), "stale_after_event": consecutive(stale)}
                ep["boundary_error_2plus"] = any(v >= 2 for v in ep["max_run"].values())
                eps.append(ep)
                for k, (o, t, l) in enumerate(zip(obs, tr, lat)):
                    rows.append({"group": ep["group"], "tag": tag, "stem": stem, "success": ok, "k": k, "obs": o,
                                 "truth": t, "latch": l, "latch_state": (calls[k].get("grasp") or {}).get("state"),
                                 "gap_mm": float(gap[o] * 1e3), "miss_S3": miss[k], "false_held": fh[k], "stale": stale[k]})
    return rows, eps


def summarize(rows, eps):
    def block(rs, es):
        t = np.array([r["truth"] for r in rs])
        l = np.array([r["latch"] for r in rs])
        cm = np.zeros((4, 5), int)                    # truth x latch (+ unknown)
        for a, b in zip(t, l):
            cm[a, b if b >= 0 else 4] += 1
        acc = float((t == l).mean()) if len(t) else None
        f1 = []
        for s in range(4):
            tp = int(((t == s) & (l == s)).sum()); fp = int(((t != s) & (l == s)).sum()); fn = int(((t == s) & (l != s)).sum())
            if tp + fp + fn:
                f1.append(2 * tp / (2 * tp + fp + fn))
        n3 = int((t == 2).sum()); n2 = int((t == 1).sum()); n12 = int(np.isin(t, (0, 1)).sum())
        ents = [x for e in es for x in e["s3_entry"] if x["n_req"] > 0]
        lats = [x["latency"] for x in ents if x["latency"] is not None]
        evs = [x for e in es for x in e["event_detection"]]
        fails = [x for x in evs if x["type"] == "fail" and x["n_req_window"] > 0]
        slips = [x for x in evs if x["type"] == "slip" and x["n_req_window"] > 0]
        fe = [e for e in es if not e["success"]]
        se = [e for e in es if e["success"]]
        return {
            "n_episodes": len(es), "n_success": len(se), "n_requests": len(rs),
            "truth_counts": {s: int((t == i).sum()) for i, s in enumerate(STAGES)},
            "confusion_truth_x_latch": cm.tolist(), "accuracy4": acc, "macro_f1_4": float(np.mean(f1)) if f1 else None,
            "S3_miss": {"n": int(sum(r["miss_S3"] for r in rs)), "of_truth_S3": n3,
                        "rate": (sum(r["miss_S3"] for r in rs) / n3) if n3 else None},
            "false_held": {"n": int(sum(r["false_held"] for r in rs)), "of_truth_S2": n2, "of_truth_S1S2": n12,
                           "n_in_S2": int(sum(r["false_held"] and r["truth"] == 1 for r in rs)),
                           "rate_vs_S2": (sum(r["false_held"] and r["truth"] == 1 for r in rs) / n2) if n2 else None},
            "S3_entry": {"segments": len(ents), "missed": sum(x["missed"] for x in ents),
                         "latency_median_steps": float(np.median(lats)) if lats else None,
                         "latency_p90_steps": float(np.percentile(lats, 90)) if lats else None},
            "E_fail": {"n": len(fails), "undetected": sum(x["undetected"] for x in fails),
                       "false_attach_in_run": sum(bool(x["false_attach_in_run"]) for x in fails),
                       "latency_median_steps": float(np.median([x["latency"] for x in fails if x["latency"] is not None]))
                       if any(x["latency"] is not None for x in fails) else None},
            "E_slip": {"n": len(slips), "undetected": sum(x["undetected"] for x in slips)},
            "episodes_boundary_error_2plus": {"failed": sum(e["boundary_error_2plus"] for e in fe), "of_failed": len(fe),
                                              "success": sum(e["boundary_error_2plus"] for e in se), "of_success": len(se)},
        }

    out = {"all": block(rows, eps)}
    for g in sorted({e["group"] for e in eps}):
        out[g] = block([r for r in rows if r["group"] == g], [e for e in eps if e["group"] == g])
    a = out["all"]
    fe = a["episodes_boundary_error_2plus"]
    rate_f = fe["failed"] / fe["of_failed"] if fe["of_failed"] else 0.0
    rate_s = fe["success"] / fe["of_success"] if fe["of_success"] else 0.0
    gate = {"a_S3_miss_ge_10pct": (a["S3_miss"]["rate"] or 0) >= 0.10,
            "b_false_held_ge_5pct_of_S2": (a["false_held"]["rate_vs_S2"] or 0) >= 0.05,
            "c_failed_eps_ge_20pct_and_gt_success": rate_f >= 0.20 and rate_f > rate_s,
            "failed_rate": rate_f, "success_rate": rate_s}
    gate["go_P3"] = gate["a_S3_miss_ge_10pct"] or gate["b_false_held_ge_5pct_of_S2"] or gate["c_failed_eps_ge_20pct_and_gt_success"]
    out["gate_P3"] = gate
    return out


def figures(rows, eps, summ, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, "/mnt/dev/AG3S_VLA")
    from AG3S.ag3s.experiments.common import figstyle
    figstyle.use_korean()                              # False on this pod -> English labels
    CAT = figstyle.CATEGORICAL
    # 1. confusion + error rates by group
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    cm = np.array(summ["all"]["confusion_truth_x_latch"])
    im = ax[0].imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap=figstyle.sequential_cmap(), vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    ax[0].set_xticks(range(5), ["S1\nsearch/latched", "S2\nclosing", "S3\nheld", "S4\nplaced", "?"])
    ax[0].set_yticks(range(4), ["S1 approach", "S2 grasp", "S3 carry/place", "S4 released"])
    ax[0].set_xlabel("latch (pe123458 opening rule)"); ax[0].set_ylabel("truth (sim contact)")
    ax[0].set_title(f"{summ['all']['n_requests']} requests - acc {summ['all']['accuracy4']:.3f}")
    fig.colorbar(im, ax=ax[0], fraction=0.04)
    groups = [g for g in summ if g not in ("all", "gate_P3")]
    x = np.arange(len(groups)); w = 0.27
    s3 = [summ[g]["S3_miss"]["rate"] or 0 for g in groups]
    fh = [summ[g]["false_held"]["rate_vs_S2"] or 0 for g in groups]
    ef = [(summ[g]["E_fail"]["undetected"] / summ[g]["E_fail"]["n"]) if summ[g]["E_fail"]["n"] else 0 for g in groups]
    ax[1].bar(x - w, s3, w, label="S3 miss / truth S3", color=CAT[0])
    ax[1].bar(x, fh, w, label="false HELD / truth S2", color=CAT[1])
    ax[1].bar(x + w, ef, w, label="undetected E-fail / E-fail", color=CAT[2])
    ax[1].axhline(0.10, ls="--", c="C0", lw=0.8); ax[1].axhline(0.05, ls="--", c="C1", lw=0.8)
    ax[1].set_xticks(x, groups, rotation=20); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8)
    ax[1].set_title("boundary error rates (dashed: P3 gate a 10 %, b 5 %)")
    figstyle.style_axes(fig, ax); fig.tight_layout(); fig.savefig(out / "그래프_P2_요약.png", dpi=130); plt.close(fig)

    # 2. timelines of failed episodes with boundary errors (up to 8)
    bad, seen = [], collections.Counter()
    for e in eps:                                     # up to 2 failed episodes with boundary errors per group
        if not e["success"] and e["boundary_error_2plus"] and seen[e["group"]] < 2:
            bad.append(e); seen[e["group"]] += 1
    if bad:
        fig, axs = plt.subplots(len(bad), 1, figsize=(12, 1.7 * len(bad)), squeeze=False)
        for a, e in zip(axs[:, 0], bad):
            rs = [r for r in rows if r["tag"] == e["tag"] and r["stem"] == e["stem"]]
            o = [r["obs"] for r in rs]
            a.step(o, [r["truth"] for r in rs], where="post", lw=2.2, label="truth", color=CAT[0])
            a.step(o, [r["latch"] + 0.08 for r in rs], where="post", lw=1.3, label="latch", color=CAT[1])
            for r in rs:
                if r["miss_S3"] or r["false_held"] or r["stale"]:
                    a.axvspan(r["obs"] - 2.5, r["obs"] + 2.5, color="red", alpha=0.15, lw=0)
            for ev in e["events"]:
                a.axvline(ev["t"], c="k", ls=":", lw=1)
                a.text(ev["t"], 3.3, ev["type"], fontsize=7)
            a2 = a.twinx(); a2.plot(o, [r["gap_mm"] for r in rs], c="gray", lw=0.8); a2.set_ylabel("gap mm", fontsize=7)
            a.set_yticks(range(4), STAGES, fontsize=7); a.set_ylim(-0.4, 3.7)
            a.set_title(f"{e['group']} {e['tag'].split('pe123458-')[-1]} {e['stem']} ({e['status']}) - red = boundary error", fontsize=8)
        axs[0, 0].legend(fontsize=7, loc="upper left")
        axs[-1, 0].set_xlabel("env step (obs_index)")
        fig.tight_layout(); fig.savefig(out / "그래프_P2_실패_타임라인.png", dpi=120); plt.close(fig)


def main():
    root = pathlib.Path(sys.argv[1])
    out = root / "p2"
    out.mkdir(exist_ok=True)
    rows, eps = evaluate(root)
    summ = summarize(rows, eps)
    (out / "p2_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    (out / "p2_episodes.json").write_text(json.dumps(eps, indent=1, ensure_ascii=False))
    with open(out / "p2_requests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    figures(rows, eps, summ, out)
    a = summ["all"]
    print(json.dumps({k: a[k] for k in ("n_episodes", "n_success", "n_requests", "truth_counts", "accuracy4", "macro_f1_4",
                                         "S3_miss", "false_held", "S3_entry", "E_fail", "E_slip",
                                         "episodes_boundary_error_2plus")}, ensure_ascii=False, indent=1))
    print(json.dumps(summ["gate_P3"], indent=1))


if __name__ == "__main__":
    main()
