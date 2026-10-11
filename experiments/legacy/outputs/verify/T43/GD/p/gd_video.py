"""T43GD comparison video: V7 (gate OFF) mp4 on the left, V7g (gate ON) mp4 on the right at the same control step, below them a time line per arm and a moving cursor.
usage: gd_video.py AGG_JSON OUT_DIR STAGE:EP:SEED [STAGE:EP:SEED ...]   -> OUT_DIR/gd_<ep>_s<seed>.mp4 (hurdle pairs; a no-obstacle pair would be gd_<ep>_s<seed>_B.mp4)
Video frame k = control step k (third_person_front.mp4: one frame per control row, 15 fps; checked: n_frames == n_control).  A run that ended earlier (obstacle stop) freezes on its last frame, text 'run ended'.
Time line rows per arm (x axis = control step; one chunk = 8 steps; cell of a chunk = [t_step, t_step + 8) ):
  label     confirmed subtask label of the server summary (pick / place / home / none before confirmation)
  latch     grasp latch state (SEARCHING / LATCHED / CLOSING / HELD / PLACED)
  verdict   client chunk outcome: HOLD (not every control row of the chunk executed) / execute
  follow_to server verdict.follow_to decision: follow_to (hold-follow executed) / hold / not_needed
  grounding AG3S grounding status of the chunk: ok / subtask_gated / no_admissible / ... (colour per status)
  snt       tick where the client reason kind subtask_no_target is present"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); ROOTS = {"V7": T / "GD/runs", "V7g": T / "V7g/runs"}
FF = "/usr/bin/ffmpeg"
C = figstyle.CATEGORICAL
COL = {"label": {"pick": C[0], "place": C[1], "home": C[2], "None": "#d9d9d9"},
       "latch": {"searching": "#d9d9d9", "latched": C[0], "closing": C[3], "held": C[1], "placed": C[2]},
       "verdict": {"hold": "#c0392b", "execute": "#7fb77e"},
       "follow_to": {"follow_to": C[4], "hold": "#c0392b", "not_needed": "#e8e8e8", "None": "#e8e8e8"}}
G_COL = {"ok": "#e8e8e8", "subtask_gated": C[5], "no_admissible": C[6], "occluded": C[3]}
ROWS = ("label", "latch", "verdict", "follow_to", "grounding", "snt")


def rowvals(o):
    return {"label": [str(x) for x in o["label_seq"]], "latch": o["latch_seq"], "verdict": [str(x) for x in o["verdict_seq"]],
            "follow_to": [str(x) for x in o["follow_to_seq"]], "grounding": [str(x) for x in o["grounding_seq"]], "snt": o["snt_seq"]}


def panel(oa, ob, nmax, w=1280, h=430):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100); fig.patch.set_facecolor("white")
    axes = {}
    x0, x1 = 0.105, 0.985
    ytop = 0.985; hrow = 0.0605; gap = 0.02
    y = ytop
    labels_seen = {}
    for arm, o, title in (("V7", oa, "V7 gate OFF"), ("V7g", ob, "V7g gate ON")):
        rv = rowvals(o)
        fig.text(0.003, y - 0.006, title, fontsize=8, fontweight="bold", va="top")
        y -= 0.03
        for r in ROWS:
            ax = fig.add_axes([x0, y - hrow * 0.8, x1 - x0, hrow * 0.8]); ax.set_xlim(0, nmax); ax.set_ylim(0, 1); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_ylabel(r, rotation=0, ha="right", va="center", fontsize=7, labelpad=4)
            ax.set_xticks([] if not (arm == "V7g" and r == "snt") else list(range(0, nmax + 1, 100))); ax.tick_params(labelsize=6)
            for t, v in zip(o["t_seq"], rv[r]):
                if r == "snt":
                    if v:
                        ax.add_patch(plt.Rectangle((t, 0.15), 8, 0.7, color="#555555", lw=0))
                    continue
                col = COL[r].get(v) if r != "grounding" else G_COL.get(v, C[7 % len(C)])
                ax.add_patch(plt.Rectangle((t, 0), 8, 1, color=col or "#bbbbbb", lw=0))
                if r in ("label", "latch", "grounding") and False:
                    pass
            t_pl = o["t_placed"]
            if t_pl is not None and r == "latch":
                ax.text(t_pl, 0.5, " PLACED", fontsize=6, va="center")
            axes[(arm, r)] = ax
            y -= hrow
        y -= gap
    # legend strip
    leg = [("pick", COL["label"]["pick"]), ("place", COL["label"]["place"]), ("home", COL["label"]["home"]), ("HOLD", COL["verdict"]["hold"]), ("execute", COL["verdict"]["execute"]),
           ("follow_to", COL["follow_to"]["follow_to"]), ("subtask_gated", G_COL["subtask_gated"]), ("no_admissible", G_COL["no_admissible"]), ("occluded", G_COL["occluded"]),
           ("latch: searching", COL["latch"]["searching"]), ("latched", COL["latch"]["latched"]), ("closing", COL["latch"]["closing"]), ("held", COL["latch"]["held"]), ("placed", COL["latch"]["placed"])]
    xx = 0.003
    for name, col in leg:
        fig.patches.append(plt.Rectangle((xx, 0.012), 0.008, 0.02, transform=fig.transFigure, color=col, lw=0)); fig.text(xx + 0.010, 0.013, name, fontsize=6, va="bottom")
        xx += 0.010 + 0.0065 * len(name) + 0.012
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    bb = axes[("V7", "label")].get_window_extent(); bb2 = axes[("V7g", "snt")].get_window_extent()
    px0, px1 = bb.x0, bb.x1; ytop_px = h - bb.y1; ybot_px = h - bb2.y0
    plt.close(fig)
    return img, px0, px1, int(ytop_px), int(ybot_px)


def frames(path):
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release(); return out


def make(agg, out_dir, spec):
    stage, ep, seed = spec.split(":"); ep = int(ep); seed = int(seed)
    ka, kb = (f"V7/{stage}/V7_ep{ep}_s{seed}", f"V7g/{stage}/V7g_ep{ep}_s{seed}")
    oa, ob = agg["runs"][ka], agg["runs"][kb]
    fa = frames(ROOTS["V7"] / f"V7/{stage}/V7_ep{ep}_s{seed}/third_person_front.mp4"); fb = frames(ROOTS["V7g"] / f"V7g/{stage}/V7g_ep{ep}_s{seed}/third_person_front.mp4")
    n = max(len(fa), len(fb)); nmax = max(n, max(max(o["t_seq"]) + 8 for o in (oa, ob)))
    pimg, px0, px1, ytop, ybot = panel(oa, ob, nmax)
    name = f"gd_{ep}_s{seed}" + ("" if stage == "H" else "_B")
    out = pathlib.Path(out_dir) / f"{name}.mp4"
    H = 480 + pimg.shape[0]; W = 1280
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "15", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
    pb = cv2.cvtColor(pimg, cv2.COLOR_RGB2BGR)
    for k in range(n):
        canvas = np.full((H, W, 3), 255, np.uint8)
        for j, (fs, o, ttl) in enumerate(((fa, oa, "V7  gate OFF"), (fb, ob, "V7g  gate ON"))):
            f = fs[min(k, len(fs) - 1)].copy()
            cv2.rectangle(f, (0, 0), (360, 26), (255, 255, 255), -1)
            cv2.putText(f, f"{ttl}  step {min(k, len(fs) - 1)}" + ("  run ended" if k >= len(fs) else ""), (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            canvas[0:480, 640 * j:640 * (j + 1)] = f
        canvas[480:, :, :] = pb
        x = int(px0 + (px1 - px0) * k / nmax)
        cv2.line(canvas, (x, 480 + ytop), (x, 480 + ybot), (0, 0, 0), 2)
        p.stdin.write(canvas.tobytes())
    p.stdin.close(); p.wait()
    # still frame at a chosen step is cut by the caller (ffmpeg -ss)
    return {"file": str(out), "pair": spec, "n_frames": n, "n_frames_V7": len(fa), "n_frames_V7g": len(fb), "size": [W, H], "nmax_steps": nmax}


if __name__ == "__main__":
    agg = json.load(open(sys.argv[1])); od = pathlib.Path(sys.argv[2]); od.mkdir(parents=True, exist_ok=True)
    res = [make(agg, od, s) for s in sys.argv[3:]]
    print(json.dumps(res, indent=1))
