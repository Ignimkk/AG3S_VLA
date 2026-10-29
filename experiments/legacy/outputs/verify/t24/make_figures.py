"""T24 verifier figures: graph / table / scene. Numbers from outputs/verify/t24/*.json (budget clock off unless noted)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle

KO = figstyle.use_korean()
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/t24'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t24'))
FIG.mkdir(parents=True, exist_ok=True)
C = figstyle.CATEGORICAL
VARS = ["recorded_server_refined", "replay", "track_only", "smooth_only", "continuity_only", "collision", "collision_budget_x2", "collision_budget_x4"]
LBL = {"recorded_server_refined": "recorded refined (server)", "replay": "replay (fresh solver)", "track_only": "track_only",
       "smooth_only": "smooth_only", "continuity_only": "continuity_only", "collision": "collision",
       "collision_budget_x2": "collision x2 budget", "collision_budget_x4": "collision x4 budget"}

m3 = json.load(open(V / 'm3_t17_t72_120.json'))['t17_clockoff']
ts = m3['t_step']

# ---------------- 1. graph ----------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
figstyle.style_axes(fig, axes)
keys = [("per_chunk_median_mm", "fingertip disp. median over 8 exec steps [mm]" if not KO else "손끝 변위 중앙값 (실행 창 8 step) [mm]"),
        ("per_chunk_max_mm", "fingertip disp. max over 8 exec steps [mm]" if not KO else "손끝 변위 최대 (실행 창 8 step) [mm]"),
        ("per_chunk_joint_max_deg", "joint max deviation [deg]" if not KO else "관절 최대 편차 [deg]")]
side = {}
for ax, (k, yl) in zip(axes, keys):
    for i, v in enumerate(VARS):
        y = m3['rows'][v][k]
        ls = '--' if v == 'recorded_server_refined' else '-'
        ax.plot(ts, y, ls, marker='o', ms=4, lw=1.6, color=C[i], label=LBL[v])
        side.setdefault(v, {})[k] = y
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_xlabel('t_step (T17 chunk 9..15)')
    ax.set_ylabel(yl)
    ax.set_xticks(ts)
    ax.grid(True, color=figstyle.GRID_INK, lw=0.6)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=8, fontsize=8.5, frameon=False)
fig.subplots_adjust(bottom=0.2)
fig.suptitle("T24 #3 — T17 approach t=72..120, fixed input, per variant vs reference (budget clock off; symlog, linthresh 1)", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG / 't24-fingertip-graph.png', dpi=150)
json.dump({"source": "outputs/verify/t24/m3_t17_t72_120.json [t17_clockoff]", "t_step": ts, "chunks": m3['chunks'], "series": side,
           "returned": {v: m3['rows'][v].get('returned') for v in VARS[1:]}}, open(FIG / 't24-fingertip-graph.json', 'w'), indent=1)
plt.close(fig)

# ---------------- 2. table ----------------
S14 = json.load(open(V / 'summary_t14_clockoff.json'))
S17 = json.load(open(V / 'summary_t17_clockoff.json'))
S17w = json.load(open(V / 'summary_t17_wall.json'))
PC = {n: json.load(open(V / f'{n}_prefix_chain.json')) for n in ('t14', 't17')}
rows17 = {r['chunk']: r for r in S17['rows']}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(figstyle.SURFACE)

def table(ax, title, col, cells, widths=None):
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=11, color=figstyle.INK)
    t = ax.table(cellText=cells, colLabels=col, loc='upper left', cellLoc='center', colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.35)
    cd = t.get_celld()
    nl = {}
    for (r, c), cell in cd.items():
        nl[r] = max(nl.get(r, 1), cell.get_text().get_text().count('\n') + 1)
    for (r, c), cell in cd.items():
        cell.set_height(cell.get_height() * nl[r])
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor('#eeeeea'); cell.set_text_props(weight='bold')
    return t

# A: active rows at reference, T17 chunk 9..15
ax = fig.add_axes([0.02, 0.60, 0.96, 0.38])
cellsA = []
for k in range(9, 16):
    r = rows17[k]
    ar = r['active_rows']
    comp = "\n".join(f"{kk.split('|')[0]}/{kk.split('|')[1]}/{kk.split('|')[2]}: {n}" for kk, n in sorted(ar['by_link_tier_tf'].items()))
    col = r['variants']['collision']
    cellsA.append([k, r['t_step'], ar['n_selected'], comp, ar['n_negative'], f"{ar['min_clearance_mm']:.2f}",
                   col['returned'], col['sqp_iter'], f"{col['max_violation_mm']:.2f}", f"{col['min_clearance_returned_mm']:.2f}", col['worst_row_link']])
table(ax, "A. T17 chunk 9..15 — rows the QP receives at iteration 1 (linearize(reference)), and the collision variant result (budget clock off)",
      ["chunk", "t", "rows", "link / tier / target_free : count", "rows < 0", "min clr\nat ref [mm]", "collision\nreturned", "SQP it", "final viol\n[mm]", "min clr\nreturned [mm]", "worst link\n(returned)"],
      cellsA, [0.04, 0.03, 0.04, 0.36, 0.05, 0.08, 0.08, 0.05, 0.07, 0.09, 0.1])

# B: returned distribution
ax = fig.add_axes([0.02, 0.22, 0.96, 0.34])
cellsB = []
def fmt(d):
    return " / ".join(f"{k} {d.get(k, 0)}" for k in ("initial", "accepted", "best_unaccepted"))
for v in ["replay", "track_only", "smooth_only", "continuity_only", "collision", "collision_budget_x2", "collision_budget_x4"]:
    cellsB.append([v, fmt(S14['summary']['returned_counts'][v]), fmt(S14['summary']['legacy_counts_recount'][v]),
                   fmt(S17['summary']['returned_counts'][v]), fmt(S17['summary']['legacy_counts_recount'][v])])
import collections
for n, S in (("t14", S14), ("t17", S17)):
    pass
def pcc(n, intact_only):
    ch = [c.get('replay_chained') for c in PC[n]['chunks']]
    ch = [c for c in ch if c and c.get('ran') and (c['chain_intact'] or not intact_only)]
    return dict(collections.Counter(c['legacy_returned'] for c in ch))
cellsB.append(["chained replay, new code (all / intact)", fmt(S14['summary']['chained']['returned_new_intact']) + "  (all 25 intact)", "-",
               fmt(S17['summary']['chained']['returned_new_intact']) + "  (intact 16)", "-"])
cellsB.append(["chained replay, tool legacy (all)", "-", fmt(S14['summary']['chained']['legacy_returned_all']), "-", fmt(S17['summary']['chained']['legacy_returned_all'])])
cellsB.append(["chained replay, tool legacy (intact)", "-", fmt(S14['summary']['chained']['legacy_returned_intact']), "-", fmt(S17['summary']['chained']['legacy_returned_intact'])])
cellsB.append(["chained replay, HEAD sqp.py run (all)", "-", fmt(pcc('t14', False)), "-", fmt(pcc('t17', False))])
cellsB.append(["chained replay, HEAD sqp.py run (intact)", "-", fmt(pcc('t14', True)), "-", fmt(pcc('t17', True))])
table(ax, f"B. returned = initial / accepted / best_unaccepted — T14 ({S14['summary']['n_ran']} chunks run) and T17 ({S17['summary']['n_ran']} chunks run); fresh solver per chunk unless 'chained'",
      ["variant", "T14 new rule", "T14 pre-fix rule", "T17 new rule", "T17 pre-fix rule"], cellsB, [0.2, 0.2, 0.2, 0.2, 0.2])

# C: fidelity + chained reproduction
ax = fig.add_axes([0.02, 0.01, 0.96, 0.18])
def pch(n, key):
    ch = [c['replay_chained'] for c in PC[n]['chunks'] if c.get('replay_chained') and c['replay_chained'].get('ran') and c['replay_chained']['chain_intact']]
    return max(c['new_vs_recorded_refined']['joint_dev_max_deg'] for c in ch)
cellsC = []
for n, S in (("T14", S14['summary']), ("T17", S17['summary'])):
    rows = (S14 if n == 'T14' else S17)['rows']
    ran = [r for r in rows if r.get('variants')]
    cellsC.append([n, f"{S['n_fid_measured']}", f"{S['fid_record_clearance_max_mm']:.2e}",
                   f"{max(r['fid_to_view_mm'] for r in ran):.2e}  (same row {sum(r['fid_to_view_same_row'] for r in ran)}/{len(ran)})",
                   f"{S['n_ran']} / skipped {S['n_chunks'] - S['n_ran']}",
                   f"{S['track_only']['n_joint_dev_exact_zero']}/{S['track_only']['n']}  (max {S['track_only']['joint_dev_max_deg_over_chunks']:.1f} deg)",
                   f"{S['chained']['legacy_vs_rec_joint_max_deg_intact']:.4f}  ({S['chained']['n_intact']} intact)",
                   f"{pch(n.lower(), 0):.4f}",
                   f"{S['replay_fresh_vs_recorded_joint_max_deg']['min']:.3f} .. {S['replay_fresh_vs_recorded_joint_max_deg']['max']:.3f}"])
table(ax, "C. #1 reconstruction fidelity, #2 track_only, chained replay reproduction of the recorded refined (exec window joint max, deg)",
      ["run", "chunks\nmeasured", "#1 record clearance\nmax |diff| [mm]", "TO view worst-row\n|diff| [mm] (chunks run)", "chunks run", "#2 track_only\njoint dev == 0",
       "chained, tool legacy\nvs recorded [deg]", "chained, HEAD sqp.py\nvs recorded [deg]", "replay fresh solver\nvs recorded [deg] min..max"],
      cellsC, [0.04, 0.07, 0.12, 0.16, 0.08, 0.12, 0.13, 0.12, 0.13])
fig.savefig(FIG / 't24-table.png', dpi=150)
json.dump({"A_rows_T17": cellsA, "B_returned": cellsB, "C_fidelity": cellsC,
           "sources": ["outputs/verify/t24/summary_t14_clockoff.json", "outputs/verify/t24/summary_t17_clockoff.json",
                       "outputs/verify/t24/t14_prefix_chain.json", "outputs/verify/t24/t17_prefix_chain.json"]},
          open(FIG / 't24-table.json', 'w'), indent=1, default=str)
plt.close(fig)

# ---------------- 3. scene ----------------
d17 = json.load(open(V / 't17_clockoff.json'))
chunks = {c['chunk_index']: c for c in d17['chunks']}
SC = [10, 15]
fig, axes = plt.subplots(len(SC), 2, figsize=(15, 6.6 * len(SC)))
figstyle.style_axes(fig, axes.ravel())
side = {}
for row, k in enumerate(SC):
    c = chunks[k]
    z = np.load(f"{_ag3s_workspace_path()}/outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_{k:05d}.npz", allow_pickle=True)
    cloud = np.asarray(z['filtered_cloud'], float); tgt = np.asarray(z['target_points'], float)
    paths = {"reference": np.asarray(c['reference']['fingertip_path_m']),
             "recorded refined (server)": np.asarray(c['recorded_refined_vs_reference']['fingertip_path_m']),
             "track_only": np.asarray(c['variants']['track_only']['exec_window']['fingertip_path_m']),
             "collision": np.asarray(c['variants']['collision']['exec_window']['fingertip_path_m'])}
    sty = {"reference": (figstyle.INK, '-', 'o', 7, 2.4), "track_only": (C[2], '--', 'x', 9, 1.4),
           "collision": (C[1], '-', 's', 5, 2.0), "recorded refined (server)": (C[6], ':', '^', 5, 1.6)}
    ap = np.asarray(c['object_poses']['apple']['pos']); cr = np.asarray(c['object_poses']['crate']['pos'])
    allp = np.concatenate(list(paths.values()) + [ap[None], cr[None]])
    lo = allp.min(0) - 0.18; hi = allp.max(0) + 0.18
    for col, (i, j, xl, yl) in enumerate([(0, 1, 'x [m] (base)', 'y [m] (base)'), (0, 2, 'x [m] (base)', 'z [m] (base)')]):
        ax = axes[row, col]
        m = np.all((cloud >= lo) & (cloud <= hi), axis=1)
        other = 2 if col == 0 else 1
        sc = ax.scatter(cloud[m, i], cloud[m, j], c=cloud[m, other], s=1.5, cmap=figstyle.sequential_cmap(), alpha=0.6, linewidths=0, rasterized=True)
        ax.scatter(tgt[:, i], tgt[:, j], s=2, color=C[3], alpha=0.35, linewidths=0, label=f'target_points ({len(tgt)})')
        ax.scatter([ap[i]], [ap[j]], marker='*', s=260, color=C[7], edgecolor=figstyle.INK, zorder=6, label='apple (MuJoCo pose)')
        ax.scatter([cr[i]], [cr[j]], marker='P', s=200, color=C[4], edgecolor=figstyle.INK, zorder=6, label='crate (MuJoCo pose)')
        for name, p in paths.items():
            colr, ls, mk, ms, lw = sty[name]
            ax.plot(p[:, i], p[:, j], ls, marker=mk, ms=ms, lw=lw, color=colr, label=name, zorder=7)
            ax.scatter([p[0, i]], [p[0, j]], s=30, color=colr, zorder=8)
        ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[j], hi[j]); ax.set_aspect('equal')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"T17 chunk {k} (t={c['t_step']}) — {'top view' if col == 0 else 'side view'}; cloud colour = {'z' if col == 0 else 'y'}", fontsize=10)
        ax.grid(True, color=figstyle.GRID_INK, lw=0.5)
        if row == 0 and col == 0:
            ax.legend(fontsize=7.5, loc='upper left', frameon=True)
    side[k] = {n: p.tolist() for n, p in paths.items()} | {"apple": ap.tolist(), "crate": cr.tolist(), "target_n": len(tgt),
               "collision_fingertip_disp_mm_per_step": c['variants']['collision']['exec_window']['fingertip_disp_mm_per_step'],
               "track_only_fingertip_disp_mm_per_step": c['variants']['track_only']['exec_window']['fingertip_disp_mm_per_step'],
               "recorded_fingertip_disp_mm_per_step": c['recorded_refined_vs_reference']['fingertip_disp_mm_per_step']}
fig.suptitle("T24 scene — left-hand fingertip (mean of ee_finger_l1/l2 spheres, FK) over the 8-step exec window; filtered cloud from the server record (base frame); budget clock off", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / 't24-scene.png', dpi=140)
json.dump({"frame": "base (object_poses are MuJoCo world; apple world pos vs target_points centroid at T17 chunk 0/5 differ by <=2 mm in x,y)",
           "chunks": side}, open(FIG / 't24-scene.json', 'w'), indent=1)
print("ok", KO)
