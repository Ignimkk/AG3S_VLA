"""T44 I2 figures: (a) real scene of the largest strict effect, (b) per-chunk graph, (c) summary table."""
from common.workspace import workspace_path as _ag3s_workspace_path
import dataclasses
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OUT = _ag3s_workspace_path('AG3S/ag3s/docs/figures/t44')
os.makedirs(OUT, exist_ok=True)
V = _ag3s_workspace_path('outputs/verify/T44/I2')
agg = json.load(open(f"{V}/agg.json"))
P = json.load(open(f"{V}/per_chunk.json"))
SA = json.load(open(f"{V}/split_auth.json"))
L = (lambda ko, en: ko if KO else en)
SPLIT = L("test split (ep 1807-1995)", "test split (ep 1807-1995)")


def save(fig, name, side):
    for e in ("png",):
        fig.savefig(f"{OUT}/{name}.png", dpi=150, facecolor=figstyle.SURFACE, bbox_inches="tight")
    side = dict(side, figure=f"{name}.png", split="test (episodes 1807-1995, recorded V11AB H runs)",
                source="outputs/verify/T44/I2/agg.json, per_chunk.json, split_auth.json")
    json.dump(side, open(f"{OUT}/{name}.json", "w"), indent=1)
    plt.close(fig)


# ------------------------------------------------------------------ (a) real scene
def scene():
    from AG3S.trajopt.experiments import ta_offline as TA
    from AG3S.trajopt import linearize as Lz

    best = max(P, key=lambda r: r.get("lookahead_reference", {}).get("n_rows_changed", 0))
    tag, seq = best["run"], best["seq"]
    idx = json.load(open(_ag3s_workspace_path('outputs/verify/T43/V11/analysis/analysis_V11AB.json')))["runs"]
    cli = type("C", (), dict(hm=False, no_plane_rows=False))()
    run = TA.Run("V11AB/H/" + tag.replace("V11AB_H_", ""), idx["V11AB/H/" + tag.replace("V11AB_H_", "")], cli)
    inp, why = run.inputs(seq)
    scene0, q_now = inp["scene"], inp["q_now"]
    rows = Lz.target_contact_rows_of(scene0.target_free_mask, None, scene0.manipulated_link_margin,
                                     "exclude_authorized", manipulated=True)
    sc_off = dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=False)
    sc_on = dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=True)
    lin = run.meter(16)
    lin.lookahead_from = 8
    lin.set_joint_parameter_path(run.finger_path(inp["summary"], inp["ref_chunk"], 16))
    lin.set_attached(scene0.attached_points, scene0.attached_parent_link)
    trj = run.layout.chunk_to_trajectory(inp["ref_chunk"])[:, :16]
    cen = lin.sphere_states(trj, q_now)[0]                       # (16, Q, 3)
    c_off = np.array(lin._esdf_clearance(cen, sc_off)[..., 0])
    c_on = np.array(lin._esdf_clearance(cen, sc_on)[..., 0])
    n = lin.n_spheres
    changed = np.zeros(c_off.shape, bool)
    changed[8:] = np.isfinite(c_on[8:]) & ~np.isfinite(c_off[8:])
    changed[:, n:] = False
    names = [lin._query_name(i) for i in range(cen.shape[1])]
    rad = np.asarray(lin.query_radii)
    held_c = np.asarray(scene0.manipulated_spheres)
    held_r = np.asarray(scene0.manipulated_sphere_radii)
    data = dict(run=tag, seq=seq, n_changed=int(changed.sum()), n_contact_rows=int(rows.sum()),
                changed_by_link={k: int(changed[:, [i for i, nm in enumerate(names) if nm == k]].sum())
                                 for k in sorted(set(nm for nm, ch in zip(names, changed.any(0)) if ch))},
                authorized_links=sorted(set(names[i] for i in np.flatnonzero(rows))),
                held_apple_spheres=int(len(held_c)), held_apple_radius_mm=float(held_r.max() * 1e3))
    ctr = held_c.mean(0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    views = [(0, 1, "x [m]", "y [m]", L("위에서 (xy)", "top (xy)")),
             (0, 2, "x [m]", "z [m]", L("옆에서 (xz)", "side (xz)")),
             (1, 2, "y [m]", "z [m]", L("앞에서 (yz)", "front (yz)"))]
    figstyle.style_axes(fig, axs)
    R = 0.28
    for ax, (a, b, xl, yl, ttl) in zip(axs, views):
        ax.set_aspect("equal")
        ax.set_xlim(ctr[a] - R, ctr[a] + R)
        ax.set_ylim(ctr[b] - R, ctr[b] + R)
        # tail-0 positions of all robot spheres near the apple (context)
        for i in range(n):
            p = cen[0, i]
            if np.linalg.norm(p - ctr) > R * 1.6:
                continue
            col = C[0] if rows[i] else "#bbbbbb"
            ax.add_patch(plt.Circle((p[a], p[b]), rad[i], fc="none", ec=col, lw=0.6, alpha=0.7))
        # tail rows that strict keeps (changed), robot sphere at the tail step
        for k, i in zip(*np.nonzero(changed)):
            p = cen[k, i]
            ax.add_patch(plt.Circle((p[a], p[b]), rad[i], fc=C[7], ec="none", alpha=0.10))
        for hc, hr in zip(held_c, held_r):
            ax.add_patch(plt.Circle((hc[a], hc[b]), hr, fc=C[1], ec=C[1], alpha=0.55, lw=1))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ttl)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axs[2].legend(handles=[
        Line2D([0], [0], color=C[0], lw=1.5, label=L("권한 행 (손/손가락), 실행 창 첫 행", "authorized rows (hand/fingers), row 0")),
        Line2D([0], [0], color="#bbbbbb", lw=1.5, label=L("비권한 행, 첫 행", "non-authorized rows, row 0")),
        Patch(fc=C[7], alpha=0.35, label=L("strict 가 look-ahead 에 남긴 비권한 tail 행 (구 위치)", "non-authorized tail rows strict keeps (sphere at tail step)")),
        Patch(fc=C[1], alpha=0.6, label=L("target = 쥔 사과 (held 구)", "target = held apple (held spheres)"))],
        loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.suptitle(L(f"T44 I2 실제 장면 — {tag} chunk {seq} (test split, 반대 손이 쥔 사과 곁을 지나는 tail 행 {data['n_changed']} 개)",
                   f"T44 I2 real scene - {tag} chunk {seq} (test split; {data['n_changed']} tail rows of the other hand passing the held apple)"),
                 fontsize=10)
    save(fig, "t44i2-scene-strict-tail", dict(kind="scene", **data))
    return data


# ------------------------------------------------------------------ (b) graph
def graph():
    fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    figstyle.style_axes(fig, axs)
    x = np.arange(len(P))
    lk = np.array([r.get("lookahead_reference", {}).get("n_rows_changed", 0) for r in P])
    tu = np.array([r["n_tail_target_unauthorized"] or 0 for r in P])
    sp = {(r["run"], r["seq"]): r for r in SA["per_chunk"]}
    off_n = np.array([sp[(r["run"], r["seq"])]["off_nonauth"] for r in P])
    st_n = np.array([sp[(r["run"], r["seq"])]["strict_nonauth"] for r in P])
    uk = np.array([r["sw_strict_unauth_kept"] or 0 for r in P])
    dj = np.array([r["strict_joint_deg_max"] for r in P])
    ds = np.array([r["sw_joint_deg_max"] for r in P])
    axs[0].bar(x, lk, width=1.0, color=C[0], label=L("바뀐 look-ahead tail 행 (같은 기준 궤적, strict 가 더 남김)", "tail rows changed (same reference trajectory)"))
    axs[0].set_ylabel(L("tail 행 수", "tail rows"))
    axs[0].legend(fontsize=8, loc="upper left")
    axs[1].bar(x, off_n, width=1.0, color=C[1], label=L("off: 비권한 행이 path 검사에서 제외됨", "off: non-authorized rows excluded from path check"))
    axs[1].bar(x, st_n, width=1.0, color=C[2], label=L("strict: 비권한 제외 행 (모두 0)", "strict: non-authorized excluded (all 0)"))
    axs[1].set_ylabel(L("path 행 수", "path rows"))
    axs[1].legend(fontsize=8, loc="upper left")
    axs[2].bar(x, dj, width=1.0, color=C[0], label=L("V11AB 설정: refined chunk 최대 joint 차이 (deg)", "V11AB config: max refined-chunk joint diff (deg)"))
    axs[2].bar(x, ds, width=1.0, color=C[3], alpha=0.8, label=L("sweep 설정 (lookahead off)", "sweep config (lookahead off)"))
    axs[2].set_ylabel("deg")
    axs[2].set_xlabel(L("chunk (14 run 이어 붙임, test split)", "chunk index (14 runs concatenated, test split)"))
    axs[2].legend(fontsize=8, loc="upper left")
    b = 0
    runs = []
    for i, r in enumerate(P):
        if i == 0 or r["run"] != P[i - 1]["run"]:
            runs.append(i)
    for ax in axs:
        for s in runs[1:]:
            ax.axvline(s - 0.5, color=figstyle.GRID_INK, lw=0.6)
    fig.suptitle(L("T44 I2 — chunk 별 strict 의 효과: 전부 쥔 물체 (carry) chunk 의 비권한 link 행", "T44 I2 - per-chunk effect of strict: all in carry chunks, non-authorized links"), fontsize=10)
    save(fig, "t44i2-per-chunk-rows", dict(kind="graph", x_chunk_index=x.tolist(),
                                           tail_rows_changed=lk.tolist(), tail_unauthorized=tu.tolist(),
                                           path_nonauth_excluded_off=off_n.tolist(), path_nonauth_excluded_strict=st_n.tolist(),
                                           sweep_unauth_kept=uk.tolist(), joint_deg_v11ab=dj.tolist(), joint_deg_sweep=ds.tolist(),
                                           chunk_ids=[[r["run"], r["seq"]] for r in P]))


# ------------------------------------------------------------------ (c) table
def table():
    v3, v4, v2 = agg["v3"], agg["v4"], agg["v2"]
    pt = open(f"{V}/pytest_full.out").read().strip().splitlines()[-1] if os.path.exists(f"{V}/pytest_full.out") else "n/a"
    cmpd = v2["compared"]
    rows = [
        ["V-I2-1", L("전체 pytest (tests/trajopt tests/ag3s)", "full pytest"), pt, L("실패 0", "0 failed")],
        ["V-I2-2", L("HEAD 498f713 대 working tree(flag off), V11AB 설정 replay", "HEAD vs working tree (flag off), V11AB replay"),
         f"{v2['n_chunks']} chunks / {v2['n_runs']} runs; refined differs {cmpd['head_off_vs_wt_off']['own_differs']} / {cmpd['head_off_vs_wt_offR']['own_differs']}; status differs {cmpd['head_off_vs_wt_off']['status_differs']}",
         L("차이 0", "0 diffs")],
        ["V-I2-2b", L("같은 비교, sweep 설정 (lookahead off + sweep_check)", "same, sweep config"),
         f"refined differs {cmpd['head_sw_vs_wt_sw']['own_differs']} / {cmpd['head_sw_vs_wt_swR']['own_differs']}; status differs {cmpd['head_sw_vs_wt_sw']['status_differs']}", L("참고", "info")],
        ["V-I2-3", L("strict on: look-ahead tail 행", "strict on: look-ahead tail rows"),
         f"changed rows {v3['lookahead_rows_changed_reference_total']} in {v3['lookahead_rows_changed_reference_chunks']} chunks; n_tail_target_unauthorized sum {v3['tail_unauthorized_rows_total']} in {v3['n_chunks_tail_unauthorized_gt0']} chunks", L("보고", "report")],
        ["V-I2-3", L("strict on: refined chunk / verdict", "strict on: refined chunk / verdict"),
         f"refined changed {v3['refined_chunks_changed']} chunks, max {v3['refined_joint_deg_max']:.3f} deg; status changes {v3['status_changes']}; HOLD changes {v3['hold_changes']}", L("보고", "report")],
        ["V-I2-4", L("sweep: 비권한 행 제외 (off -> strict)", "sweep: non-authorized rows excluded (off -> strict)"),
         f"{SA['totals']['off_nonauth']} -> {SA['totals']['strict_nonauth']} (same traj {SA['totals']['p_off_nonauth']} -> {SA['totals']['p_strict_nonauth']})", L("on 에서 0", "0 on strict")],
        ["V-I2-4", L("sweep: 권한 행 제외 (off -> strict)", "sweep: authorized rows excluded (off -> strict)"),
         f"{SA['totals']['off_auth']} -> {SA['totals']['strict_auth']} (same traj {SA['totals']['p_off_auth']} -> {SA['totals']['p_strict_auth']})", L("유지", "kept")],
        ["V-I2-4", L("sweep: 새 path violation / strict 가 남긴 비권한 행", "sweep: new path violations / unauthorized kept"),
         f"{v4['same_traj']['new_path_violations']} (same traj; {v4['same_traj']['path_violations_off']} -> {v4['same_traj']['path_violations_strict']}); kept rows {v4['strict_unauthorized_kept_total']} in {v4['strict_chunks_unauthorized_kept_gt0']} chunks; solve status changes {v4['status_changes']}", L("보고", "report")],
    ]
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.axis("off")
    fig.patch.set_facecolor(figstyle.SURFACE)
    t = ax.table(cellText=rows, colLabels=[L("항목", "item"), L("무엇", "what"), L("측정값", "measured"), L("기준", "criterion")],
                 loc="center", cellLoc="left", colWidths=[0.07, 0.30, 0.50, 0.10])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 2.0)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor(figstyle.SEQ_BLUE[0])
            cell.set_text_props(color=figstyle.INK)
        else:
            cell.set_facecolor(figstyle.SURFACE)
    ax.set_title(L("T44 I2 요약 — test split V11AB 14 run, CPU offline replay (working tree 에 A1 변경 혼재)", "T44 I2 summary - test split V11AB, 14 runs, CPU offline replay (A1 edits present in working tree)"), fontsize=10)
    save(fig, "t44i2-summary-table", dict(kind="table", rows=rows))


if __name__ == "__main__":
    which = sys.argv[1:] or ["scene", "graph", "table"]
    if "scene" in which:
        print(scene())
    if "graph" in which:
        graph()
    if "table" in which:
        table()
