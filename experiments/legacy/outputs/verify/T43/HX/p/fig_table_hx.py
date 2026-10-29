"""T43 HX table figure: cause counts (i)-(v), servo-model accuracy, 22-contact detection, code facts.  usage: fig_table_hx.py OUT_PNG
All numbers from raw/hx_summary.json (hx_agg.py definitions)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out_png = pathlib.Path(sys.argv[1]); assert not out_png.exists(), out_png
S = json.load(open(_ag3s_workspace_path('outputs/verify/T43/HX/raw/hx_summary.json')))
figstyle.use_korean()
CAUSES = ["i", "ii", "iii", "iv_held", "iv_robot_class_target", "v_coverage", "v_no_to_row"]
hdr1 = ["arm", "phase", "group", "window rows exec/hold", "min P", "min M", "min TO", "M<10 rows (episodes)", "M<0 rows",
        "(i) lag", "(ii) TO saw", "(iii) ESDF", "(iv) margin 0", "(v) other", "HOLD rows M<10", "P<10 rows: cause", "S flags geo/ESDF"]
t1 = []
for arm, A in S["arms"].items():
    for ph, P in A["phase"].items():
        for g, G in P["groups"].items():
            me = G["moments_exec"]
            iv = sum(v for k, v in me.items() if k.startswith("iv")); v_ = sum(v for k, v in me.items() if k.startswith("v"))
            pm = ", ".join(f"{k}:{v}" for k, v in G["P_moments"].items()) or "0"
            f = lambda d: "-" if d.get("n", 0) == 0 else f"{d['min']:.1f}"
            t1.append([arm, ph, g, f"{P['n_window_rows_exec']}/{P['n_window_rows_hold']}", f(G["min_P_geo_mm"]), f(G["min_M_geo_mm"]),
                       f(G["min_TO_clear_mm"]), f"{G['n_moments_exec']} ({G['episodes_exec']})", str(G["n_moments_exec_contact"]),
                       str(me.get("i", 0)), str(me.get("ii", 0)), str(me.get("iii", 0)), str(iv), str(v_), str(G["moments_hold"]), pm,
                       f"{G['servo_flag_geo']}/{G['servo_flag_esdf']} of {G['n_with_S']}"])
hdr2 = ["arm", "rows", "servo |EE| err med / p90 / max (mm)", "command |EE| err med / p90 / max (mm)", "servo joint err p90 (deg)",
        "stop rows servo err med / max (mm)", "window chunks: status", "sqp iter / max_iter_hit / budget_hit", "TO<0 executed rows (class)"]
t2 = []
for arm, A in S["arms"].items():
    sv = A["servo"]["all_exec"]; st_ = A["servo"]["stop"]; w = A["window_chunks"]
    neg = A["to_neg_exec_rows"]
    t2.append([arm, str(sv["servo_err_mm"]["n"]),
               f"{sv['servo_err_mm']['median']:.2f} / {sv['servo_err_mm']['p90']:.2f} / {sv['servo_err_mm']['max']:.2f}",
               f"{sv['cmd_err_mm']['median']:.1f} / {sv['cmd_err_mm']['p90']:.1f} / {sv['cmd_err_mm']['max']:.1f}",
               f"{sv['servo_err_joint_deg_max']['p90']:.3f}",
               f"{st_['servo_err_mm']['median']:.1f} / {st_['servo_err_mm']['max']:.1f} (n={st_['servo_err_mm']['n']})",
               ", ".join(f"{k} {v}" for k, v in w["status"].items()),
               f"{'/'.join(w['sqp_iterations'])}x{list(w['sqp_iterations'].values())[0]} / {w['max_iterations_hit']} / {w['time_budget_hit']}",
               f"{len(neg)} ({', '.join(sorted({e['cls'] for e in neg})) or '-'})"])
c = S["contacts22_summary"]
hdr3 = ["contact set", "n", "S geo < 0", "S geo < 10 mm", "TO-sphere ESDF at S < 0", "TO at P < 0"]
t3 = [["apple, carry", str(c["apple|carry"]["n"]), str(c["apple|carry"]["flag_geo_contact"]), str(c["apple|carry"]["flag_geo_margin"]),
       str(c["apple|carry"]["flag_esdf"]), "0"],
      ["fingers, after release", str(c["fingers|after_release"]["n"]), str(c["fingers|after_release"]["flag_geo_contact"]),
       str(c["fingers|after_release"]["flag_geo_margin"]), str(c["fingers|after_release"]["flag_esdf"]), "0"],
      ["all 22", str(c["n"]), str(c["flag_geo_contact"]), str(c["flag_geo_margin"]), str(c["flag_esdf"]), str(c["to_clear_lt0"])]]
facts = [
    "linearize.py:758,784-785 (work-t43v8): margin = max(margin, obstacle_margin 10 mm) only where class == obstacle; linearize.py:848 class held for query >= n_spheres -> held rows keep esdf_margin (0)",
    "linearize.py:764-778: E1 per-link margin on rows with |d - d_object| <= voxel (class target): 20 mm authorized links / 50 mm others (reconstructed, matches records)",
    "safe_policy.py:1273-1276: record clearance (8,S) = main-layer d - r - esdf_margin at refined rows; sphere_centres = refined row 0",
    "problem.py:256-257: anchor |Q0 - q_now| <= max_step; config.py:261 v_max*dt = 10.8 deg; server log: plan window 8 of 50, SQP min = max 3 iterations before the 50 ms budget (sqp.py:275-277)",
    "servo.py ServoObserver/ServoModel; serve_safe.py:1107 --servo-model off in V7g/V8/V5 (server_proc.txt); linearize.py:441-460 + gripper_state.py:147 finger path per row",
]
figstyle.use_korean()
fig = plt.figure(figsize=(22, 10.5)); fig.patch.set_facecolor(figstyle.SURFACE)
for i, (hdr, rows, y, h, title) in enumerate(((hdr1, t1, 0.52, 0.40, "Below-margin moments in the crossing window (executed rows, M geometric < 10 mm) and cause (first rule that holds)"),
                                               (hdr2, t2, 0.33, 0.15, "Servo model offline (rows from grasp, executed) and TO window chunks"),
                                               (hdr3, t3, 0.17, 0.12, "22 contacts (T43TK list: apple carry 8 + fingers after release 14), servo-predicted state S"))):
    ax = fig.add_axes([0.01, y, 0.98, h]); ax.axis("off"); ax.set_title(title, fontsize=11, loc="left", color=figstyle.INK)
    tb = ax.table(cellText=rows, colLabels=hdr, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8 if i == 0 else 9); tb.scale(1, 1.25 if i == 0 else 1.5)
    for (r, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor(figstyle.SEQ_BLUE[0]); cell.set_text_props(fontweight="bold")
ax = fig.add_axes([0.01, 0.0, 0.98, 0.14]); ax.axis("off")
ax.set_title("code facts (file:line)", fontsize=11, loc="left", color=figstyle.INK)
for j, f in enumerate(facts):
    ax.text(0.0, 0.85 - j * 0.18, "- " + f, fontsize=9, color=figstyle.INK_2, transform=ax.transAxes)
fig.suptitle("T43 HX - why the TO did not fix the crossing: cause table (i) lag  (ii) TO saw, not solved  (iii) ESDF error  (iv) margin-0 row  (v) other", fontsize=13)
fig.savefig(out_png, dpi=110)
json.dump({"script": __file__, "table_causes": {"header": hdr1, "rows": t1}, "table_servo": {"header": hdr2, "rows": t2},
           "table_contacts": {"header": hdr3, "rows": t3}, "code_facts": facts}, open(out_png.with_suffix(".json"), "w"), indent=1)
print("wrote", out_png)
