"""T6b figure 2 (graph) — three runs' per-chunk safe/hold bands stacked, plus the violation curve.

Reads only outputs/verify/T6b/chunks_T6b.json and aggregate_T6b.json (written by aggregate_T6b.py).

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T6b/make_fig2_bands.py
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6b'))
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t6b'))
PNG = OUTDIR / "t6b-chunk-bands-and-violation.png"
SIDECAR = OUTDIR / "t6b-chunk-bands-and-violation.json"

RUN_LABEL = {
    "shadow_threshold": ("shadow · threshold 0.25\n20260925_t5shadow", "shadow (threshold 0.25)"),
    "shadow_rank1": ("shadow · rank 1 + confirm 3\n20260925_t5e_shadow", "shadow (rank 1 + confirm 3)"),
    "execute": ("execute (closed loop)\n20260926_t6", "execute (closed loop)"),
}
ORDER = ["shadow_threshold", "shadow_rank1", "execute"]

SAFE_C = figstyle.CATEGORICAL[2]      # green slot
HOLD_C = figstyle.CATEGORICAL[1]      # orange slot
DEGR_C = figstyle.CATEGORICAL[6]      # violet slot
NOTGT_C = figstyle.CATEGORICAL[7]     # red slot


def main() -> None:
    korean = figstyle.use_korean()
    chunks = json.loads((VER / "chunks_T6b.json").read_text())
    agg = json.loads((VER / "aggregate_T6b.json").read_text())
    overlay = agg["task_phase_overlay"]
    trend = agg["violation_trend_execute"]

    fig = plt.figure(figsize=(13.0, 9.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 3.1], hspace=0.55)
    band_axes = [fig.add_subplot(gs[i]) for i in range(3)]
    ax_v = fig.add_subplot(gs[3])
    figstyle.style_axes(fig, band_axes + [ax_v])

    first_closed = overlay["first_closed_seq"]
    first_reopen = overlay["first_reopen_seq"]
    onset = overlay["execute_hold_onset_seq"]

    side = {
        "source": {
            "chunks": "outputs/verify/T6b/chunks_T6b.json",
            "aggregate": "outputs/verify/T6b/aggregate_T6b.json",
        },
        "band_encoding": {
            "safe": "planning record safe == true",
            "hold (ag3s ok)": "safe == false and verdict.ag3s_status == 'ok'",
            "hold (degraded)": "safe == false and verdict.ag3s_status == 'degraded'",
            "hold (no_target)": "safe == false and verdict.ag3s_status == 'no_target'",
        },
        "markers": {
            "first_closed_seq": first_closed,
            "first_reopen_seq": first_reopen,
            "first_closed/first_reopen_measured_on": overlay["measured_on"],
            "execute_unbroken_hold_onset_seq": onset,
        },
        "runs": {},
    }

    for ax, name in zip(band_axes, ORDER):
        rows = chunks[name]
        for r in rows:
            if r["safe"]:
                colour = SAFE_C
            elif r["ag3s_status"] == "degraded":
                colour = DEGR_C
            elif r["ag3s_status"] == "no_target":
                colour = NOTGT_C
            else:
                colour = HOLD_C
            ax.broken_barh([(r["seq"] - 0.5, 1.0)], (0.0, 1.0),
                           facecolors=colour, edgecolors="none")
        p = agg["per_run"][name]
        ax.set_xlim(0.5, 75.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_title("%s   —   safe %d / hold %d" % (RUN_LABEL[name][0].replace("\n", "  ·  "),
                                                     p["n_safe"], p["n_hold"]),
                     fontsize=9.5, color=figstyle.INK, loc="left")
        for x in (first_closed, first_reopen):
            ax.axvline(x, color=figstyle.INK_2, lw=0.8, ls=":")
        if name == "execute":
            ax.axvline(onset - 0.5, color=figstyle.INK, lw=1.2)
        side["runs"][name] = {
            "n_safe": p["n_safe"], "n_hold": p["n_hold"],
            "safe_seq_ranges": p["safe_seq_ranges"],
            "hold_seq_ranges": p["hold_seq_ranges"],
            "hold_classes": p["hold_classes"],
        }

    band_axes[-1].set_xlabel("chunk seq (t_step = (seq - 1) * 8)" if not korean
                             else "chunk seq (t_step = (seq - 1) × 8)")

    # --- violation curve -------------------------------------------------------------
    for i, name in enumerate(ORDER):
        rows = chunks[name]
        xs = [r["seq"] for r in rows]
        ys = [(r["max_violation_m"] or 0.0) * 1e3 for r in rows]
        ax_v.plot(xs, ys, lw=1.4 if name != "execute" else 2.0,
                  color=figstyle.CATEGORICAL[i], marker="o", ms=2.6,
                  label=RUN_LABEL[name][1])
        side["runs"][name]["max_violation_mm_by_seq"] = dict(zip(map(str, xs), ys))

    ax_v.axvline(onset - 0.5, color=figstyle.INK, lw=1.2,
                 label="execute: unbroken hold from seq %d" % onset)
    for x, lab in ((first_closed, "first_closed_seq %d" % first_closed),
                   (first_reopen, "first_reopen_seq %d" % first_reopen)):
        ax_v.axvline(x, color=figstyle.INK_2, lw=0.8, ls=":")
        ax_v.text(x - 0.5, 1.0, lab, rotation=90, fontsize=7.6,
                  color=figstyle.INK_2, ha="right", va="bottom")
    ax_v.set_xlim(0.5, 75.5)
    ax_v.set_ylim(-1.5, 42.0)
    ax_v.set_xlabel("chunk seq")
    ax_v.set_ylabel("verdict.max_violation_m  [mm]")
    ax_v.legend(frameon=False, fontsize=8.5, loc="upper left", ncols=2)
    ax_v.set_title(
        "max_violation_m per chunk   ·   execute: slope from seq %d = %+.3f mm/chunk"
        "   ·   first 10 from onset p50 %.2f mm  ->  last 10 p50 %.2f mm  ·  argmax seq %d"
        % (onset, trend["slope_mm_per_chunk_from_onset"],
           trend["first10_from_onset_p50_mm"], trend["last10_p50_mm"], trend["argmax_seq"]),
        fontsize=9.0, color=figstyle.INK, loc="left")

    handles = [Patch(facecolor=SAFE_C, label="safe"),
               Patch(facecolor=HOLD_C, label="hold · ag3s ok"),
               Patch(facecolor=DEGR_C, label="hold · ag3s degraded"),
               Patch(facecolor=NOTGT_C, label="hold · ag3s no_target"),
               Line2D([0], [0], color=figstyle.INK_2, ls=":",
                      label="first_closed / first_reopen (reference episode, T5d)")]
    band_axes[0].legend(handles=handles, frameon=False, fontsize=8.2,
                        loc="lower left", bbox_to_anchor=(0.0, 1.35), ncols=5)

    fig.suptitle("T6b · per-chunk verdicts of three live runs, episode 1807"
                 if not korean else
                 "T6b · 세 실행의 chunk 별 판정 (episode 1807)",
                 fontsize=12.5, color=figstyle.INK, x=0.012, ha="left", y=0.995)
    fig.text(0.012, 0.005,
             "shadow runs executed the reference chunk on every chunk; the execute run held. "
             "The execute trajectory diverged from the shadow runs at its first hold (seq %d), "
             "so the seq axis is a common clock, not a common pose."
             % min(s for rng in agg["per_run"]["execute"]["hold_seq_ranges"] for s in rng[:1]),
             fontsize=8.0, color=figstyle.INK_2)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=190, facecolor=figstyle.SURFACE, bbox_inches="tight")
    SIDECAR.write_text(json.dumps(side, ensure_ascii=False, indent=1))
    print("wrote", PNG)
    print("wrote", SIDECAR)


if __name__ == "__main__":
    main()
