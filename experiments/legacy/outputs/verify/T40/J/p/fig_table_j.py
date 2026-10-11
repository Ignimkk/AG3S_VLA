"""T40 J table: the T40 G/D outcome table (t40fg-table) with the 5 J reruns (fixed client distance, snapshot ${AG3S_PROJECT_ROOT}-t40h)
inserted after the run they replace (stage 'C-J' / 'D-J'). Columns as t40fg-table; J rows use hurdle_metrics_j distances; G rows
keep the values of the original table (G/analysis_all.json). Extra column 'prefix=G' = J run bit-identical to its G record up to
the old stop step. -> AG3S/ag3s/docs/figures/t40/t40fg-table-j.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle
figstyle.use_korean()
G = json.loads(open(_ag3s_workspace_path('outputs/verify/T40/G/analysis_all.json')).read())["runs"]
J = json.loads(open(_ag3s_workspace_path('outputs/verify/T40/J/analysis_J.json')).read())["runs"]
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t40'))
cols = ["ep", "stage", "cond", "success", "grasp", "place", "HOLD", "HOLD kinds", "min grip mm", "min apple mm", "min robot mm",
        "contact / stop", "stop t", "AG3S tgt on hurdle", "bit-id", "prefix=G"]


def row(r, st, jr=None):
    h = r.get("hurdle") or {}; ph = (h.get("phases") or {}).get("all") or {}; cs = h.get("client_obstacle_safety") or {}
    f = lambda k: "-" if not ph.get(k) else f"{1e3 * ph[k]['min_m']:.1f}"
    stop = bool(h.get("client_obstacle_stop_lines"))
    ct = "-" if st == "B" else (("robot" if cs.get("robot_collision") else "") + ("+apple" if cs.get("payload_collision") else "")).strip("+") or "none"
    if st != "B" and stop and not cs.get("robot_collision"):
        ct += " (clearance 0.0)"
    ag = "-" if r["cond"] == "E0" or st == "B" else f"{h.get('ag3s_n_chunks_with_target_points_on_hurdle', 0)}/{h.get('ag3s_n_chunks', 0)}"
    bi = r.get("bit_identity_vs_T39")
    pre = "-" if jr is None else ("Y" if jr["all_bit_identical_prefix"] else "N")
    return [r["episode"], st, r["cond"], "Y" if r["success"] else "N", "Y" if r["grasp"] else "N", "Y" if r["place"] else "N", r["n_hold_chunks"],
            ",".join(f"{k}:{v}" for k, v in r["hold_kind_counts"].items()) or "-", f("A"), f("B"), f("C"), ct,
            str(r["n_control"] - 1) if stop else "-", ag, "-" if bi is None else ("Y" if bi["all_bit_identical"] else "N"), pre]


order = [1807, 1968, 1982, 1828, 1995, 1967]
rows = []
for ep in order:
    for st in ("B", "C", "D"):
        for cond in ("E0", "E3b"):
            k = next((k for k, r in G.items() if r["episode"] == ep and r["stage"] == st and r["cond"] == cond), None)
            if k is None:
                continue
            rows.append(row(G[k], st))
            if k in J:
                rows.append(row(J[k], f"{st}-J", J[k]["prefix_identity_vs_G"]))
fig, ax = plt.subplots(figsize=(19, 0.27 * len(rows) + 0.9)); ax.set_axis_off(); fig.patch.set_facecolor(figstyle.SURFACE)
tb = ax.table(cellText=[[str(x) for x in r] for r in rows], colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1])
tb.auto_set_font_size(False); tb.set_fontsize(7.5)
for (i, j), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        cell.set_text_props(color=figstyle.INK, weight="bold")
    elif rows[i - 1][1].endswith("-J"):
        cell.set_facecolor(figstyle.SEQ_BLUE[0] if hasattr(figstyle, "SEQ_BLUE") else "#eef")
ax.set_title("T40 outcomes with the J reruns (C-J / D-J rows: client distance fixed, pi05_TO_hybrid bd6a002, same seed and config). "
             "Min distances: whole run, control rows. prefix=G: J run bit-identical to its G record up to the old stop step", color=figstyle.INK, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "t40fg-table-j.png", dpi=150, facecolor=figstyle.SURFACE)
(OUT / "t40fg-table-j.json").write_text(json.dumps({"definitions": __doc__, "columns": cols, "rows": [dict(zip(cols, r)) for r in rows]}, indent=1, default=str))
print(OUT / "t40fg-table-j.png")
