"""T44b I2b figures: (a) real scene of a carry chunk, (b) per-chunk graph I2 vs I2b, (c) summary table."""
from common.workspace import workspace_path as _ag3s_workspace_path
import dataclasses, json, os, sys
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OUT = _ag3s_workspace_path('AG3S/ag3s/docs/figures/t44')
V = _ag3s_workspace_path('outputs/verify/T44/I2b')
os.makedirs(OUT, exist_ok=True)
agg = json.load(open(f"{V}/agg.json"))
PM = json.load(open(f"{V}/per_chunk_main.json"))
PA = json.load(open(f"{V}/per_chunk_arms.json")) if os.path.exists(f"{V}/per_chunk_arms.json") else []
Lg = (lambda ko, en: ko if KO else en)
IDX = _ag3s_workspace_path('outputs/verify/T43/V11/analysis/analysis_V11AB.json')


def save(fig, name, side):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, facecolor=figstyle.SURFACE, bbox_inches="tight")
    side = dict(side, figure=f"{name}.png", split="test (episodes 1807-1995, recorded V11AB H runs; regression check only)",
                source="outputs/verify/T44/I2b/agg.json, per_chunk_main.json, per_chunk_arms.json")
    json.dump(side, open(f"{OUT}/{name}.json", "w"), indent=1, default=str)
    plt.close(fig)


def changed_rows(r, v, ref="reference"):
    k = f"lookahead_{ref}_{v}"
    x = (r.get("probes") or {}).get(k)
    if not x:
        return 0
    return int(sum(max(a - b, 0) for a, b, _ in x["dropped_off_dropped_on_by_link"].values()))


# ------------------------------------------------------------------ (a) scene
def scene():
    from AG3S.trajopt.experiments import ta_offline as TA
    from AG3S.trajopt import linearize as Lz
    from AG3S.ag3s.constraints import to_adapter as TOA
    from AG3S.ag3s.config import AG3SConfig
    pool, model = (PA, "arms") if PA else (PM, "main")
    held = [r for r in pool if r["held"]]
    best = max(held, key=lambda r: changed_rows(r, "b1"))
    tag, seq = best["run"], best["seq"]
    key = "V11AB/H/" + tag.replace("V11AB_H_", "")
    idx = json.load(open(IDX))["runs"]
    cli = SimpleNamespace(hm=False, no_plane_rows=False, variants=["base"])
    run = TA.Run(key, idx[key], cli, extra_server_args=("--links", "arms") if model == "arms" else ())
    inp, why = run.inputs(seq)
    scene0, q_now = inp["scene"], inp["q_now"]
    contact = AG3SConfig.from_dict({"collision_backend": "esdf"}).contact
    bld = SimpleNamespace(sphere_link_names=run.names, robot_model=run.robot,
                          clearance_policy=SimpleNamespace(contact=contact))
    hrec = TOA._held_permission_record(bld, SimpleNamespace(parent_link=str(inp["held"][0])))
    rows = Lz.held_contact_rows_of(hrec, 1, len(run.names))
    depth = np.asarray(hrec["depth"])
    sc_off = dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=False)
    sc_on = dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=True)
    lin = run.meter(16)
    lin.lookahead_from = 8
    lin.set_joint_parameter_path(run.finger_path(inp["summary"], inp["ref_chunk"], 16))
    lin.set_attached(scene0.attached_points, scene0.attached_parent_link)
    trj = run.layout.chunk_to_trajectory(inp["ref_chunk"])[:, :16]
    cen = lin.sphere_states(trj, q_now)[0]
    c_off = np.array(lin._esdf_clearance(cen, sc_off)[..., 0])
    c_on = np.array(lin._esdf_clearance(cen, sc_on)[..., 0])
    n = lin.n_spheres
    changed = np.zeros(c_off.shape, bool)
    changed[8:] = np.isfinite(c_on[8:]) & ~np.isfinite(c_off[8:])
    changed[:, n:] = False
    names = [lin._query_name(i) for i in range(cen.shape[1])]
    rad = np.asarray(lin.query_radii)
    held_c = np.asarray(scene0.manipulated_spheres); held_r = np.asarray(scene0.manipulated_sphere_radii)
    ctr = held_c.mean(0)
    cls = np.full(n, "checked", object)
    cls[depth == 0] = "gripper"
    cls[(depth >= 1) & (depth <= 1)] = "parent"
    cb = {}
    for i in np.flatnonzero(changed.any(0)):
        cb[names[i]] = cb.get(names[i], 0) + int(changed[:, i].sum())
    data = dict(model=model, run=tag, seq=seq, hold=hrec["hold_link"], side=hrec["side"], parent_links=hrec["parent_links"],
                n_tail_rows_kept_by_strict=int(changed.sum()), changed_by_link=cb,
                permitted_rows_by_link={k: int(sum(1 for i in np.flatnonzero(rows) if names[i] == k)) for k in sorted({names[i] for i in np.flatnonzero(rows)})},
                checked_rows_total=int(n - rows.sum()), held_apple_spheres=int(len(held_c)))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
    views = [(0, 1, "x [m]", "y [m]", Lg("위에서 (xy)", "top (xy)")), (0, 2, "x [m]", "z [m]", Lg("옆에서 (xz)", "side (xz)")),
             (1, 2, "y [m]", "z [m]", Lg("앞에서 (yz)", "front (yz)"))]
    figstyle.style_axes(fig, axs)
    R = 0.30
    col = {"gripper": C[0], "parent": C[2], "checked": "#9aa0a6"}
    for ax, (a, b, xl, yl, ttl) in zip(axs, views):
        ax.set_aspect("equal"); ax.set_xlim(ctr[a] - R, ctr[a] + R); ax.set_ylim(ctr[b] - R, ctr[b] + R)
        for i in range(n):
            p = cen[0, i]
            if np.linalg.norm(p - ctr) > R * 1.7: continue
            ax.add_patch(plt.Circle((p[a], p[b]), rad[i], fc="none", ec=col[cls[i]], lw=0.9 if cls[i] != "checked" else 0.6, alpha=0.85))
        for k, i in zip(*np.nonzero(changed)):
            p = cen[k, i]
            ax.add_patch(plt.Circle((p[a], p[b]), rad[i], fc=C[7], ec="none", alpha=0.10))
        for hc, hr in zip(held_c, held_r):
            ax.add_patch(plt.Circle((hc[a], hc[b]), hr, fc=C[1], ec=C[1], alpha=0.55, lw=1))
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axs[2].legend(handles=[
        Line2D([0], [0], color=C[0], lw=1.6, label=Lg(f"권한: 쥔 손 ({hrec['side']}) gripper (ee + 손가락)", f"permitted: holding hand ({hrec['side']}) gripper")),
        Line2D([0], [0], color=C[2], lw=1.6, label=Lg(f"권한: gripper 위 1 link ({', '.join(hrec['parent_links'][:1]) or '없음: 이 제약 모델에 구 없음'})", f"permitted: parent ({', '.join(hrec['parent_links'][:1]) or 'none: no spheres in this model'})")),
        Line2D([0], [0], color="#9aa0a6", lw=1.6, label=Lg("비권한 (반대 손 · forearm · 나머지): strict 가 검사", "not permitted (other hand, forearm, rest): checked")),
        Patch(fc=C[7], alpha=0.35, label=Lg("strict 가 look-ahead 에 남긴 tail 행 (구 위치)", "tail rows strict keeps (sphere at tail step)")),
        Patch(fc=C[1], alpha=0.6, label=Lg("target = 쥔 사과 (held 구)", "target = held apple"))], loc="lower right", fontsize=7.2, framealpha=0.9)
    fig.suptitle(Lg(f"T44b I2b 실제 장면 - {tag} chunk {seq} (test split, 제약 모델 {'--links arms' if model == 'arms' else '--links gripper (V11AB 기록)'}; 쥔 손 {hrec['side']}, strict 가 남긴 tail 행 {data['n_tail_rows_kept_by_strict']})",
                    f"T44b I2b real scene - {tag} chunk {seq} (test split, constraint model {'--links arms' if model == 'arms' else '--links gripper (V11AB record)'}; holding {hrec['side']}, {data['n_tail_rows_kept_by_strict']} tail rows kept by strict)"), fontsize=9.5)
    save(fig, "t44i2b-scene-carry-permission", dict(kind="scene", **data))
    return data


# ------------------------------------------------------------------ (b) graph
def graph():
    heldM = [r for r in PM if r["held"]]
    fig, axs = plt.subplots(2 if PA else 1, 1, figsize=(13, 7 if PA else 4), sharex=False)
    axs = np.atleast_1d(axs)
    figstyle.style_axes(fig, axs)
    side = {}
    for ax, pool, ttl in ([(axs[0], heldM, Lg("V11AB 기록 모델 (--links gripper): carry chunk", "V11AB recorded model (--links gripper): carry chunks"))] +
                          ([(axs[1], [r for r in PA if r["held"]], Lg("보충: --links arms 모델 (arm_6 · forearm 구 있음): carry chunk", "supplement: --links arms model: carry chunks"))] if PA else [])):
        x = np.arange(len(pool))
        vals = {v: np.array([changed_rows(r, v) for r in pool]) for v in ("i2", "i2s", "b1", "b0")}
        w = 0.27
        ax.bar(x - w, vals["i2s"], width=w, color=C[1], label=Lg("I2 (서버 carry: 권한 행 0)", "I2 (server carry: no permitted row)"))
        ax.bar(x, vals["i2"], width=w, color=C[3], label=Lg("I2 (V-I2 replay: 접근 단계 margin 행)", "I2 (V-I2 replay rule)"))
        ax.bar(x + w, vals["b1"], width=w, color=C[0], label=Lg("I2b parent=1 (쥔 손 gripper + arm_6)", "I2b parent=1"))
        ax.set_ylabel(Lg("바뀐 tail 행 수\n(strict 가 off 보다 더 남김)", "tail rows changed\n(kept by strict, not by off)"))
        ax.set_xlabel(Lg("carry chunk (run 이어 붙임)", "carry chunk (runs concatenated)"))
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        side[ttl] = dict(chunk_ids=[[r["run"], r["seq"]] for r in pool], **{f"rows_changed_{v}": vals[v].tolist() for v in vals})
    fig.subplots_adjust(hspace=0.45)
    fig.suptitle(Lg("T44b I2b - chunk 별 바뀐 look-ahead tail 행 (기준 궤적, 같은 trajectory)", "T44b I2b - per-chunk look-ahead tail rows changed (same reference trajectory)"), fontsize=10)
    save(fig, "t44i2b-per-chunk-rows", dict(kind="graph", series=side))


# ------------------------------------------------------------------ (c) table
def table():
    M = agg["main"]; A = agg.get("arms")
    pt = open(f"{V}/pytest_full.out").read().strip().splitlines()[-1]
    rr = open(f"{V}/pytest_rerun4.out").read().strip().splitlines()[-1]
    c = M["compare"]
    def cc(n): d = c[n]; return f"{d['own_differs']}/{d['status_differs']}/{d['hold_differs']}"
    lk = M["lookahead_probes"]; sv = M["solve"]; sw = M["sweep"]
    def lkrow(m):
        return f"i2s {m['lookahead_reference_i2s']['rows_changed_total']} | i2 {m['lookahead_reference_i2']['rows_changed_total']} | i2b {m['lookahead_reference_b1']['rows_changed_total']}"
    rows = [
        ["1", Lg("전체 pytest", "full pytest"), "4 failed 2582 passed 3 skipped; the 4 = concurrent serve_safe.py edit; rerun 4 passed", ""],
        ["2", Lg("strict off: refined/status/HOLD 차이 (HEAD 498f713 대 새 tree off / offR)", "strict off diffs HEAD vs new tree off / offR"),
         f"{cc('head_off_vs_new_off')} / {cc('head_off_vs_new_offR')} (of {c['head_off_vs_new_off']['n_chunks']})", ""],
        ["2", Lg("strict off: sweep 설정; V-I2 working tree 대비", "strict off: sweep config; vs V-I2 working tree"),
         f"sw {cc('head_sw_vs_new_sw')} / swR {cc('head_sw_vs_new_swR')}; vs V-I2 wt off {cc('viI2wt_off_vs_new_off')}", ""],
        ["3", Lg("carry 에서 바뀐 tail 행 (기준 궤적)", "tail rows changed in carry"), lkrow(lk | {}) if False else lkrow(lk), Lg("행", "rows")],
        ["3", "n_tail_target_unauthorized (solve)", f"I2s {sv['i2s']['n_tail_target_unauthorized_total']} | I2 {sv['i2']['n_tail_target_unauthorized_total']} | I2b {sv['b1']['n_tail_target_unauthorized_total']}", Lg("행", "rows")],
        ["3", Lg("refined chunk 변화 (최대 joint deg) / status / HOLD", "refined changes (max deg) / status / HOLD"),
         " | ".join(f"{k} {sv[k]['refined_chunks_changed']} ({sv[k]['refined_joint_deg_max']:.3f}) / {sv[k]['status_changes']} / {sv[k]['hold_changes']}" for k in ("i2s", "i2", "b1")), ""],
        ["3", Lg("I2b 행 분류 (main 모델, tail 행: off 가 버림 -> strict 가 버림)", "I2b row classes (main model, tail rows dropped off -> strict)"),
         "held gripper {a} -> {b} (exempt) | other gripper {c} -> {d} (checked)".format(
             a=lk['lookahead_reference_b1']['tail_rows_dropped_by_class']['held_gripper|dropped_off'], b=lk['lookahead_reference_b1']['tail_rows_dropped_by_class']['held_gripper|dropped_strict'],
             c=lk['lookahead_reference_b1']['tail_rows_dropped_by_class']['other_gripper|dropped_off'], d=lk['lookahead_reference_b1']['tail_rows_dropped_by_class']['other_gripper|dropped_strict']),
         Lg("행", "rows")],
        ["3", Lg("I2b 행 분류 (arms 모델): 권한 쪽", "I2b classes (arms model): permitted side"),
         (lambda t: "held gripper {} -> {} | arm_6 {} -> {} (exempt, tail rows dropped off -> strict)".format(
             t['held_gripper|dropped_off'], t['held_gripper|dropped_strict'], t['held_parent1|dropped_off'], t['held_parent1|dropped_strict']))(A['lookahead_probes']['lookahead_reference_b1']['tail_rows_dropped_by_class']) if A else "n/a", Lg("행", "rows")],
        ["3", Lg("I2b 행 분류 (arms 모델): 검사 쪽", "I2b classes (arms model): checked side"),
         (lambda t: "holding-arm forearm+ {} -> {} | other arm {} -> {} | other gripper {} -> {}".format(
             t['held_parent2plus|dropped_off'], t['held_parent2plus|dropped_strict'], t['arm_rest_other_side|dropped_off'], t['arm_rest_other_side|dropped_strict'],
             t['other_gripper|dropped_off'], t['other_gripper|dropped_strict']))(A['lookahead_probes']['lookahead_reference_b1']['tail_rows_dropped_by_class']) if A else "n/a", Lg("행", "rows")],
        ["4", Lg("parent=0 대 1: 달라진 행 / chunk", "parent 0 vs 1: rows / chunks differing"),
         f"main {M['n_rows_differ_b0_b1_total']} / {M['n_chunks_rows_differ_b0_b1']}" + (f"; arms {A['n_rows_differ_b0_b1_total']} / {A['n_chunks_rows_differ_b0_b1']}" if A else ""), ""],
        ["5", Lg("sweep I2b: 비권한 제외 / 권한 제외 (off -> strict)", "sweep I2b: non-permitted / permitted excluded"),
         f"{sw['b1']['split_authorization']['off_nonauth']} -> {sw['b1']['split_authorization']['strict_nonauth']} / {sw['b1']['split_authorization']['off_auth']} -> {sw['b1']['split_authorization']['strict_auth']}", ""],
        ["5", Lg("sweep I2b: 새 path violation / status 변화", "sweep I2b: new path violations / status changes"),
         f"{sw['b1']['same_traj']['new_path_violations']} / {sw['b1']['status_changes']}" + (f"; arms model {A['sweep']['b1']['same_traj']['new_path_violations']} / {A['sweep']['b1']['status_changes']}" if A else ""), ""],
        ["5", "ep1834_s18341 seq 34 (hold right)", f"I2 {sw['i2']['ep1834_s18341_seq34']['status'][0]} -> {sw['i2']['ep1834_s18341_seq34']['status'][1]} ; I2b {sw['b1']['ep1834_s18341_seq34']['status'][0]} -> {sw['b1']['ep1834_s18341_seq34']['status'][1]} (window viol {sw['b1']['ep1834_s18341_seq34']['window_violation_mm'][1]:.2f} mm; checked opposite gripper left)", ""],
    ]
    fig, ax = plt.subplots(figsize=(15, 6.4)); ax.axis("off"); fig.patch.set_facecolor(figstyle.SURFACE)
    t = ax.table(cellText=rows, colLabels=[Lg("항목", "item"), Lg("무엇", "what"), Lg("측정값", "measured"), Lg("단위", "unit")], loc="center", cellLoc="left", colWidths=[0.04, 0.34, 0.54, 0.06])
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 2.0)
    for (r, cidx), cell in t.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        cell.set_facecolor(figstyle.SEQ_BLUE[0] if r == 0 else figstyle.SURFACE)
        if r == 0: cell.set_text_props(color=figstyle.INK)
    ax.set_title(Lg("T44b I2b 요약 - test split V11AB 14 run, CPU offline replay", "T44b I2b summary - test split V11AB, 14 runs, CPU offline replay"), fontsize=10)
    save(fig, "t44i2b-summary-table", dict(kind="table", rows=rows))


if __name__ == "__main__":
    which = sys.argv[1:] or ["scene", "graph", "table"]
    if "scene" in which: print(scene())
    if "graph" in which: graph()
    if "table" in which: table()
