"""T66 — why the arm swerves right after the grasp in the HSR6 combined run (diagnosis, read-only on session A's records).

For each request: VLA reference vs executed (refined) chunk, max arm-joint deviation over the 8 executed rows; latch
state; HSR6 held-gate detail (`lift_withdrawn` = HSR6 not in {lift, transport, placement} while the held body is near its
support, so the "lifted" held-body spheres are replaced by the unlifted ones); multi-seed trigger and standard-solve
violation. Compared with the same scene without HSR6 (geometry-correction run).

usage: t66_held_gate_jump.py OUTPUTS_DIR      (.venv-openpi-live)
"""

from __future__ import annotations

import ast
import collections
import glob
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import figstyle  # noqa: E402

V = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958"
                 "/verify/AG3S-CORE-EVAL")
RUNS = {"geometry correction (no HSR6)": "OCC_CB_G1", "geometry correction + HSR6": "OCC_CB_HSR"}
SCENES = {"AG3S_case1_L2_s13453": "occlusion 121", "AG3S_case1_L2_s23453": "occlusion 122",
          "AG3S_case1_L2_s33453": "occlusion 123"}


def segments(run_dir: pathlib.Path) -> dict:
    sd = sorted(run_dir.glob("server_*"))[0]
    order = (sd / "order.txt").read_text().split()
    fs = sorted((sd / "server_constraints/run_0000").glob("chunk_*.npz"))
    ts = [int(np.load(f)["t_step"]) for f in fs]
    segs = [[fs[0]]]
    for f, a, b in zip(fs[1:], ts, ts[1:]):
        if b <= a:
            segs.append([])
        segs[-1].append(f)
    return dict(zip(order, segs))


def row(f, side=0):
    z = np.load(f, allow_pickle=True)
    s = json.loads(str(z["summary_json"]))
    ref, rf = np.asarray(z["reference_chunk"], float), np.asarray(z["refined_chunk"], float)
    dq = np.degrees(np.abs(rf[:8, side:side + 7] - ref[:8, side:side + 7]))
    m = (s.get("to") or {}).get("metrics") or {}
    ms = m.get("multi_seed")
    if isinstance(ms, str):
        try:
            ms = ast.literal_eval(ms)
        except (ValueError, SyntaxError):
            ms = {}
    hg = (((s.get("hsr6") or {}).get("detail") or {}).get("held_gate")) or {}
    return {"t_step": int(z["t_step"]), "latch": (s.get("grasp") or {}).get("state"),
            "hsr6": (s.get("hsr6") or {}).get("phase"), "dq_max_deg": float(dq.max()),
            "dq_joint_deg": dq.max(0).round(2).tolist(), "withdrawn": bool(hg.get("lift_withdrawn")),
            "gate_allowed": hg.get("allowed"), "to_status": (s.get("to") or {}).get("status"),
            "multi_seed": bool((ms or {}).get("triggered")), "standard_violation_mm": (ms or {}).get("standard_violation_mm"),
            "solve_ms": (s.get("to") or {}).get("solve_ms"),
            "max_violation_link": str(m.get("max_violation_pair", ""))[:120]}


def main(o: pathlib.Path) -> None:
    o.mkdir(parents=True, exist_ok=True)
    data = {}
    for lab, r in RUNS.items():
        for k in ("k0", "k1"):
            for name, seg in segments(V / r / k).items():
                data[(lab, k, name)] = [row(f) for f in seg]
    # graph 1: per-request deviation, scene 121 draw 0 (the user's video) and scene 123 draw 1
    cols = figstyle.CATEGORICAL
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    nums = {}
    for ax, (k, name) in zip(axes, (("k0", "AG3S_case1_L2_s13453"), ("k1", "AG3S_case1_L2_s33453"))):
        for i, lab in enumerate(RUNS):
            rows = data[(lab, k, name)]
            t = [x["t_step"] for x in rows]
            ax.plot(t, [x["dq_max_deg"] for x in rows], "-o", ms=3, color=cols[i], label=lab)
            nums.setdefault(f"{SCENES[name]} {k}", {})[lab] = [(x["t_step"], round(x["dq_max_deg"], 2), x["latch"], x["hsr6"],
                                                                 x["withdrawn"]) for x in rows]
            if i == 1:
                for x in rows:
                    if x["withdrawn"]:
                        ax.axvspan(x["t_step"] - 0.5, x["t_step"] + 0.5, color=cols[3], alpha=0.25, lw=0)
            h = next((x["t_step"] for x in rows if x["latch"] == "held"), None)
            if h is not None:
                ax.axvline(h - 0.5, color=cols[i], ls=":", lw=1)
        ax.set_ylabel("max joint change by TrajOpt (deg)")
        ax.set_title(f"{SCENES[name]}, draw {k[-1]}: dotted = first HELD request; shaded = HSR6 withdrew the held-body lift",
                     fontsize=9)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("request (t_step)")
    figstyle.style_axes(fig, list(axes)); fig.tight_layout()
    fig.savefig(o / "graph_deviation_per_request.png", dpi=150); plt.close(fig)
    (o / "graph_deviation_per_request.json").write_text(json.dumps(nums, ensure_ascii=False, indent=1))

    # graph 2 + table: all HELD requests of the HSR6 runs (occlusion + regression), by gate state
    agg = collections.defaultdict(list)
    for lab, k, name in data:
        if lab.endswith("HSR6"):
            agg_key = None
            for x in data[(lab, k, name)]:
                if x["latch"] == "held":
                    agg_key = "withdrawn" if x["withdrawn"] else "allowed"
                    agg[agg_key].append(x)
    for f in glob.glob(str(V / "CB_HSR_k*/**/chunk_*.npz"), recursive=True):
        z = np.load(f, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        if (s.get("grasp") or {}).get("state") != "held":
            continue
        side = 8 if np.abs(z["refined_chunk"][:8, 8:15] - z["reference_chunk"][:8, 8:15]).max() > \
            np.abs(z["refined_chunk"][:8, :7] - z["reference_chunk"][:8, :7]).max() else 0
        x = row(f, side)
        agg["withdrawn" if x["withdrawn"] else "allowed"].append(x)
    summ = {}
    for k, v in agg.items():
        d = np.array([x["dq_max_deg"] for x in v])
        summ[k] = {"n_requests": len(v), "median_deg": float(np.median(d)), "p90_deg": float(np.percentile(d, 90)),
                   "max_deg": float(d.max()), "n_over_5deg": int((d > 5).sum()),
                   "multi_seed_triggered": int(sum(x["multi_seed"] for x in v))}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ks = [k for k in ("allowed", "withdrawn") if k in agg]
    ax.boxplot([[x["dq_max_deg"] for x in agg[k]] for k in ks], tick_labels=[f"{k}\n(n={len(agg[k])})" for k in ks])
    ax.set_ylabel("max joint change by TrajOpt (deg)")
    ax.set_title("HELD requests of the HSR6 runs, by held-gate state", fontsize=9)
    figstyle.style_axes(fig, [ax]); fig.tight_layout()
    fig.savefig(o / "graph_gate_state_box.png", dpi=150); plt.close(fig)
    (o / "graph_gate_state_box.json").write_text(json.dumps(summ, indent=1))

    # table: the user's video run around the grasp
    lines = ["| 요청 | latch | HSR6 | 쥔 물체 lift 허용 | 표준 solve 위반 (mm) | multi-seed | TrajOpt 관절 변화 max (°) | HSR 없는 run (°) |",
             "|---|---|---|---|---|---|---|---|"]
    g1 = {x["t_step"]: x for x in data[("geometry correction (no HSR6)", "k0", "AG3S_case1_L2_s13453")]}
    for x in data[("geometry correction + HSR6", "k0", "AG3S_case1_L2_s13453")]:
        if 14 <= x["t_step"] <= 22:
            y = g1.get(x["t_step"], {})
            lines.append(f"| {x['t_step']} | {x['latch']} | {x['hsr6']} | {'회수' if x['withdrawn'] else '유지'} | "
                         f"{x['standard_violation_mm'] if x['multi_seed'] else '-'} | {'예' if x['multi_seed'] else '-'} | "
                         f"{x['dq_max_deg']:.1f} | {y.get('dq_max_deg', float('nan')):.1f} |")
    lines += ["", "| HSR6 run 의 HELD 요청 | 요청 수 | 중앙값 (°) | 90 % (°) | 최대 (°) | 5° 초과 | multi-seed |", "|---|---|---|---|---|---|---|"]
    for k in ks:
        s = summ[k]
        lines.append(f"| lift 허용 {'회수' if k == 'withdrawn' else '유지'} | {s['n_requests']} | {s['median_deg']:.2f} | "
                     f"{s['p90_deg']:.2f} | {s['max_deg']:.1f} | {s['n_over_5deg']} | {s['multi_seed_triggered']} |")
    (o / "table_t66.md").write_text("\n".join(lines) + "\n")
    (o / "rows.json").write_text(json.dumps({" | ".join(k): v for k, v in data.items()}, ensure_ascii=False, indent=0))
    # scene strip: head camera of both runs, requests 15-20 of occlusion 121 draw 0 (the user's video)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.6))
    strip = {}
    for i, (lab, r) in enumerate(RUNS.items()):
        steps = sorted((V / r / "k0/runs/AG3S_case1_L2_s13453/ag3s/run_0000").glob("step_*.npz"),
                       key=lambda p: int(np.load(p)["t_step"]))
        rows = {x["t_step"]: x for x in data[(lab, "k0", "AG3S_case1_L2_s13453")]}
        for j, t in enumerate(range(15, 21)):
            st = np.load(steps[t - 1]); x = rows.get(t, {})
            ax = axes[i, j]; ax.imshow(st["image_cam_high"]); ax.axis("off")
            ax.set_title(f"t{t} {x.get('latch')} / HSR6 {x.get('hsr6') or '-'}\nTrajOpt change {x.get('dq_max_deg', float('nan')):.1f} deg"
                         + (" (lift withdrawn)" if x.get("withdrawn") else ""), fontsize=7,
                         color=cols[1] if x.get("withdrawn") else figstyle.INK)
            strip.setdefault(lab, {})[t] = {k: x.get(k) for k in ("latch", "hsr6", "dq_max_deg", "withdrawn")}
        axes[i, 0].text(-10, 112, lab, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("occlusion 121, draw 0: head camera around the grasp (top: no HSR6, bottom: HSR6)", fontsize=10)
    fig.tight_layout(); fig.savefig(o / "scene_occlusion121_k0_grasp.png", dpi=140); plt.close(fig)
    (o / "scene_occlusion121_k0_grasp.json").write_text(json.dumps(strip, ensure_ascii=False, indent=1, default=float))
    print("->", o, summ)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
