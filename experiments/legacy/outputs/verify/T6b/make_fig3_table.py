"""T6b figure 3 (table / schematic) — the three runs side by side, and the verdict rule.

Numbers come only from outputs/verify/T6b/aggregate_T6b.json.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T6b/make_fig3_table.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6b'))
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t6b'))
PNG = OUTDIR / "t6b-three-runs-table-and-rule.png"
SIDECAR = OUTDIR / "t6b-three-runs-table-and-rule.json"

ORDER = ["shadow_threshold", "shadow_rank1", "execute"]
HEAD = ["shadow · threshold 0.25", "shadow · rank 1 + confirm 3", "execute · closed loop"]


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def ranges_str(ranges):
    out = []
    for a, b in ranges:
        out.append(str(a) if a == b else "%d-%d" % (a, b))
    return ", ".join(out)


def main() -> None:
    figstyle.use_korean()
    agg = json.loads((VER / "aggregate_T6b.json").read_text())
    per = agg["per_run"]
    trend = agg["violation_trend_execute"]
    overlay = agg["task_phase_overlay"]

    rows = []

    def add(label, fn):
        rows.append([label] + [fn(per[n]) for n in ORDER])

    add("record", lambda p: p["record_dir"].split("/")[-2])
    add("mode", lambda p: "shadow (override)" if p["argv_has_safe_shadow"] else "execute (hold)")
    add("chunks sent", lambda p: p["n_chunks"])
    add("safe", lambda p: p["n_safe"])
    add("hold", lambda p: p["n_hold"])
    add("timeout / stale / error", lambda p: "%d / %d / %d" % (
        p["client_ipc_stats"]["timeout"], p["client_ipc_stats"]["stale"],
        p["client_ipc_stats"]["error"]))
    add("completeness pass", lambda p: str(p["completeness_pass"]))
    rows.append(["", "", "", ""])
    add("hold (ag3s ok, violated)", lambda p: p["hold_classes"].get("(ok, violated)", 0))
    add("hold (degraded, violated)", lambda p: p["hold_classes"].get("(degraded, violated)", 0))
    add("hold (no_target, violated)", lambda p: p["hold_classes"].get("(no_target, violated)", 0))
    add("hold with max_violation == 0", lambda p: p["max_violation_zero_hold_chunks"])
    rows.append(["", "", "", ""])
    add("hold seq", lambda p: ranges_str(p["hold_seq_ranges"]))
    add("safe seq range", lambda p: "%s..%s" % (p["safe_seq_min"], p["safe_seq_max"]))
    add("first unbroken hold seq", lambda p: fmt(p["first_unbroken_hold_seq"]))
    rows.append(["", "", "", ""])
    add("max_violation, holds > 0  n",
        lambda p: p["max_violation_hold_chunks_nonzero"].get("n", 0))
    add("  min / p50 / p95 / max  [mm]", lambda p: " / ".join(
        fmt(p["max_violation_hold_chunks_nonzero"].get(k), 2)
        for k in ("min_mm", "p50_mm", "p95_mm", "max_mm")))
    add("max_violation, all holds p50 [mm]",
        lambda p: fmt(p["max_violation_hold_chunks"].get("p50_mm"), 2))
    rows.append(["", "", "", ""])
    add("field tiers 2 / 1 per chunk", lambda p: "%s / %s" % (
        p["n_tiers_counts"].get("2", 0), p["n_tiers_counts"].get("1", 0)))
    add("ag3s_reason_codes  skew / stale [chunks]", lambda p: "%s / %s" % (
        p["ag3s_reason_code_counts"].get("camera_skew", 0),
        p["ag3s_reason_code_counts"].get("camera_transform_stale", 0)))
    rows.append(["", "", "", ""])
    for stage in ("infer", "ag3s", "trajopt", "total"):
        add("timing %s p50/p95/max [ms]" % stage, lambda p, s=stage: " / ".join(
            fmt(p["timing_ms"][s][k], 1) for k in ("p50_ms", "p95_ms", "max_ms")))
    add("timing total, seq 1 excluded", lambda p: " / ".join(
        fmt(p["timing_ms"]["total"][k], 1)
        for k in ("p50_ms_excl_seq1", "p95_ms_excl_seq1", "max_ms_excl_seq1")))
    add("seq 1 total [ms]", lambda p: fmt(p["timing_ms"]["total"]["seq1_ms"], 1))
    rows.append(["", "", "", ""])
    add("camera skew min/p50/p95/max [ms]", lambda p: " / ".join(
        fmt(p["camera_skew_ms"][k], 1) for k in ("min_ms", "p50_ms", "p95_ms", "max_ms")))
    add("frames with skew > 100 ms", lambda p: p["camera_skew_ms"]["over_100ms_frames"])
    add("control steps executed / not", lambda p: "%d / %d" % (
        p["control_executed_true"], p["control_executed_false"]))

    fig = plt.figure(figsize=(18.2, 11.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.03)
    ax_t = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    for ax in (ax_t, ax_r):
        ax.set_axis_off()
        # Everything below is placed in [0, 1] data coordinates; pin the limits so that the
        # autoscaler does not crop the rows (an earlier draft mixed transAxes and data coords
        # and matplotlib then clipped most of the table away).
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    fig.patch.set_facecolor(figstyle.SURFACE)

    # ---- table -------------------------------------------------------------------
    n = len(rows)
    y0, dy = 0.965, 0.965 / (n + 2)
    xs = [0.004, 0.375, 0.585, 0.795]
    for j, h in enumerate(HEAD):
        ax_t.text(xs[j + 1], y0, h, fontsize=9.0, color=figstyle.INK,
                  ha="left", va="top", fontweight="normal")
    ax_t.plot([0.0, 1.0], [y0 - 0.018, y0 - 0.018], color=figstyle.GRID_INK, lw=1.0)
    for i, row in enumerate(rows):
        y = y0 - 0.030 - (i + 1) * dy
        if not any(str(c) for c in row):
            ax_t.plot([0.0, 1.0], [y + dy * 0.45, y + dy * 0.45],
                      color=figstyle.GRID_INK, lw=0.6)
            continue
        ax_t.text(xs[0], y, str(row[0]), fontsize=8.6, color=figstyle.INK_2,
                  ha="left", va="center")
        for j in range(3):
            ax_t.text(xs[j + 1], y, str(row[j + 1]), fontsize=8.6,
                      color=figstyle.INK if ORDER[j] == "execute" else figstyle.INK_2,
                      ha="left", va="center")
    ax_t.set_title("T6b · three live runs of episode 1807, per-chunk verdicts aggregated",
                   fontsize=11.5, color=figstyle.INK, loc="left", pad=16)

    # ---- rule schematic ----------------------------------------------------------
    boxes = [
        (0.06, 0.86, 0.88, 0.075,
         "server: trajopt status  (AG3S/trajopt/sqp.py:275-293)\n"
         "violation > tolerance  ->  VIOLATED"),
        (0.06, 0.745, 0.88, 0.075,
         "server: status.safe and require_certified_geometry and not geometry_certified\n"
         "->  VIOLATED   (sqp.py:286-294)"),
        (0.06, 0.630, 0.88, 0.075,
         "client: safe = result['safe'];  if not safe and not shadow  ->  _hold(...)\n"
         "(AG3S/trajopt/client.py:178-180, 306-337)"),
        (0.06, 0.515, 0.88, 0.075,
         "hold chunk = current state tiled over 50 rows\n"
         "(client.py:334-337)"),
    ]
    for (x, y, w, h, text) in boxes:
        ax_r.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                      linewidth=1.0, edgecolor=figstyle.GRID_INK,
                                      facecolor="#f2f4f7"))
        ax_r.text(x + 0.018, y + h / 2, text, fontsize=8.4, color=figstyle.INK,
                  ha="left", va="center", linespacing=1.5)
    for y in (0.845, 0.730, 0.615):
        ax_r.add_patch(FancyArrowPatch((0.5, y), (0.5, y - 0.023),
                                       arrowstyle="-|>", mutation_scale=9,
                                       color=figstyle.INK_2))
    ax_r.text(0.06, 0.955, "what makes a chunk a hold", fontsize=11.0,
              color=figstyle.INK, ha="left", va="top")

    lines = [
        "hold onset and the reference-episode phase markers",
        "  first_closed_seq %d · first_reopen_seq %d  (T5d.verify.json,"
        % (overlay["first_closed_seq"], overlay["first_reopen_seq"]),
        "  measured on the offline replay of the reference episode)",
        "  execute: first hold seq %d · unbroken hold from seq %d"
        % (per["execute"]["hold_seq_min"], overlay["execute_hold_onset_seq"]),
        "  execute: safe %d of %d chunks in seq 1-37 (%.1f %%)"
        % (overlay["execute_n_safe_in_seq_1_37"], overlay["execute_n_chunks_seq_1_37"],
           100.0 * overlay["execute_n_safe_in_seq_1_37"] / overlay["execute_n_chunks_seq_1_37"]),
        "  execute: max_violation p50 over seq 1-37 = %.2f mm, from seq %d = %.2f mm"
        % (trend["before_onset"]["p50_mm"], trend["hold_onset_seq"],
           trend["from_onset"]["p50_mm"]),
        "",
        "not measured in this record (no qpos, no object pose, no actions)",
        "  when the apple entered the basket in the execute run",
        "  whether the arm actually stopped moving after seq 38",
        "  ground-truth clearance and the closest pair at the held pose",
        "  which constraint the `violated` verdict refers to",
        "  the execute run's own task phase",
        "",
        "offline replay cannot stand in for it: the execute trajectory left",
        "the reference at its first hold (seq %d), so replaying the recorded"
        % per["execute"]["hold_seq_min"],
        "episode does not visit the poses the execute run visited.",
    ]
    yy = 0.455
    for ln in lines:
        head = not ln.startswith(" ") and ln
        ax_r.text(0.06, yy, ln, fontsize=8.6 if not head else 9.4,
                  color=figstyle.INK if head else figstyle.INK_2,
                  ha="left", va="top")
        yy -= 0.0265 if not head else 0.030

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=180, facecolor=figstyle.SURFACE)
    side = {"source": "outputs/verify/T6b/aggregate_T6b.json",
            "columns": ORDER,
            "rows": rows,
            "violation_trend_execute": trend,
            "task_phase_overlay": overlay}
    SIDECAR.write_text(json.dumps(side, ensure_ascii=False, indent=1))
    print("wrote", PNG)
    print("wrote", SIDECAR)


if __name__ == "__main__":
    main()
