"""T60 step 1 figures — real scene at the P / H endpoints, ready-pose gap distribution, removal table.

usage: t60_figures.py STEP1_DIR FIG_DIR      (.venv-openpi-live: cv2 for the recorded video)
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402
from AG3S.ag3s.experiments.studies.t60_timestamps import SETS  # noqa: E402

SCENE_RUNS = ("CL3/C1a_ep11_s111", "CL3/C1a_ep110_s1101")  # one returned, one placed but not returned


def frame(video: pathlib.Path, k: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, img = cap.read()
    cap.release()
    assert ok, (video, k)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def scene(ts: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    p = T.Params()
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2 * len(SCENE_RUNS), 4, height_ratios=[1.25, 0.75] * len(SCENE_RUNS))
    side = {}
    for i, key in enumerate(SCENE_RUNS):
        r = ts["runs"][key]
        base = SETS[r["set"]]
        rd = base / "runs" / r["arm"] / "R" / r["name"]
        s = T.series_from_record(base / "replay" / f"{r['arm']}_R_{r['name']}.npz", rd / "traj.npz", r["used_arm"])
        n = len(s["t"])
        marks = [("recorded place_t", r["recorded"]["place_t"]), ("t_P", r["t_P"]),
                 ("t_H" if r["t_H"] is not None else ("H 미도달 · 마지막 row" if ko else "no H · last row"),
                  r["t_H"] if r["t_H"] is not None else n - 1),
                 ("horizon end", n - 1)]
        marks = marks[:3] + ([marks[3]] if r["t_H"] is not None else [])
        for j, (lab, k) in enumerate(marks):
            ax = fig.add_subplot(gs[2 * i, j])
            ax.imshow(frame(rd / "third_person_front.mp4", int(k)))
            ax.set_title(f"{lab} = row {k}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[2 * i + 1, :])
        t = np.arange(n)
        ax.plot(t, s["q_err"], color=figstyle.CATEGORICAL[0], lw=1.4,
                label="ready-pose joint error max|q−q0| (rad)")
        ax.plot(t, s["tcp_err"] * 5, color=figstyle.CATEGORICAL[1], lw=1.4, label="TCP error ×5 (m)")
        ax.plot(t, np.minimum(s["speed"], 0.5) * 2, color=figstyle.CATEGORICAL[2], lw=1.0,
                label="target speed ×2 (m/s, clip 0.5)")
        ax.fill_between(t, 0, 1.6, where=s["in_crate"][:n], color=figstyle.SEQ_BLUE[0], alpha=0.6, lw=0,
                        label="target in crate")
        ax.fill_between(t, 1.6, 1.75, where=s["hand_target"], color=figstyle.CATEGORICAL[3], lw=0,
                        label="hand–target contact")
        ax.axhline(p.q_tol, color=figstyle.CATEGORICAL[0], ls=":", lw=1)
        for lab, k, c in (("t_P", r["t_P"], figstyle.CATEGORICAL[4]), ("t_H", r["t_H"], figstyle.CATEGORICAL[5])):
            if k is not None:
                ax.axvline(k, color=c, lw=1.6); ax.text(k + 3, 1.45, lab, color=c, fontsize=9)
                ax.axvspan(k, n, color=c, alpha=0.06)
        ax.set_xlim(0, n); ax.set_ylim(0, 1.8); ax.set_xlabel("control row (15 Hz)")
        ax.set_title(f"{key} ({r['used_arm']} arm) — t_P={r['t_P']}, t_H={r['t_H']}", fontsize=10, loc="left")
        if i == 0:
            ax.legend(fontsize=7.5, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.28), frameon=False)
        side[key] = {"t_P": r["t_P"], "t_H": r["t_H"], "place_t": r["recorded"]["place_t"], "frames": dict(marks),
                     "q_err_final": r["q_err_final"], "tcp_err_final": r["tcp_err_final"]}
    figstyle.style_axes(fig, fig.axes)
    fig.suptitle("T60 — P (placement 안정) · H (준비자세 복귀) 종료 시점의 실제 장면" if ko else
                 "T60 — real scene at P (stable placement) and H (ready-pose return)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "t60-endpoint-scene.png", dpi=130); plt.close(fig)
    out = {"runs": side, "params": p.__dict__, "video_frame_rule": "frame k of third_person_front.mp4 (600 frames, 15 fps) = control row k"}
    T.write_json(fig_dir / "t60-endpoint-scene.json", out)
    return out


def gap(ts: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    p = T.Params()
    rows = [r for r in ts["runs"].values() if r["t_P"] is not None and r["q_err_min_after_P"] is not None]
    # minimum over rows after P of the dwell_h rolling max (= what H tests), recomputed from the grid-free series
    qs, cs, labs = [], [], []
    for r in rows:
        base = SETS[r["set"]]
        s = T.series_from_record(base / "replay" / f"{r['arm']}_R_{r['name']}.npz",
                                 base / "runs" / r["arm"] / "R" / r["name"] / "traj.npz", r["used_arm"])
        k0 = r["t_P"] + 1; D = p.dwell_h
        both = [max(s["q_err"][k - D + 1:k + 1].max() / p.q_tol, s["tcp_err"][k - D + 1:k + 1].max() / p.tcp_tol)
                for k in range(k0 + D - 1, len(s["q_err"]))]
        j = int(np.argmin(both)) + k0 + D - 1
        qs.append(float(s["q_err"][j - D + 1:j + 1].max())); cs.append(float(s["tcp_err"][j - D + 1:j + 1].max()))
        labs.append(f"{r['set']}/{r['name']}")
    qs = np.array(qs); cs = np.array(cs)
    hit = np.array([ts["runs"][k]["t_H"] is not None for k in labs])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(qs[hit], cs[hit], s=22, color=figstyle.CATEGORICAL[0], label=f"H reached (n={hit.sum()})")
    ax.scatter(qs[~hit], cs[~hit], s=22, color=figstyle.CATEGORICAL[1], marker="x", label=f"H not reached (n={(~hit).sum()})")
    ax.axvline(p.q_tol, color=figstyle.INK_2, ls="--", lw=1); ax.axhline(p.tcp_tol, color=figstyle.INK_2, ls="--", lw=1)
    ax.axvspan(0.25, 0.50, color=figstyle.SEQ_BLUE[0], alpha=0.5, lw=0, label="q_tol grid 0.25–0.50")
    ax.axhspan(0.08, 0.16, color=figstyle.SEQ_BLUE[2], alpha=0.25, lw=0, label="tcp_tol grid 0.08–0.16")
    ax.set_xlabel("best 8-row window after t_P: max joint error (rad)")
    ax.set_ylabel("same window: max TCP error (m)")
    ax.set_title("준비자세 오차 — P 이후 가장 가까웠던 8 row" if ko else "ready-pose error, closest 8-row window after P", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    order = np.argsort(qs)
    ax.bar(np.arange(len(qs)), qs[order], color=[figstyle.CATEGORICAL[0] if h else figstyle.CATEGORICAL[1] for h in hit[order]])
    ax.axhline(p.q_tol, color=figstyle.INK_2, ls="--", lw=1)
    hi = qs[hit].max(); lo = qs[~hit].min()
    ax.annotate(f"gap {hi:.3f} → {lo:.3f} rad", xy=(hit.sum(), (hi + lo) / 2), fontsize=9)
    ax.set_xlabel("placed runs (sorted)"); ax.set_ylabel("max joint error (rad)")
    ax.set_title("89 placed run 정렬 — 두 무리 사이 간격" if ko else "89 placed runs sorted — the gap", loc="left")
    figstyle.style_axes(fig, axes)
    fig.tight_layout(); fig.savefig(fig_dir / "t60-ready-gap.png", dpi=130); plt.close(fig)
    out = {"params": p.__dict__, "n": len(qs), "n_H": int(hit.sum()),
           "q_err_window_max_H_reached": float(hi), "q_err_window_min_H_not_reached": float(lo),
           "tcp_err_window_max_H_reached": float(cs[hit].max()), "tcp_err_window_min_H_not_reached": float(cs[~hit].min()),
           "per_run": {k: {"q": q, "tcp": c, "H": bool(h)} for k, q, c, h in zip(labs, qs, cs, hit)}}
    T.write_json(fig_dir / "t60-ready-gap.json", out)
    return out


def table(ts: dict, fig_dir: pathlib.Path, ko: bool, proxy_t53_frames: int) -> dict:
    R = list(ts["runs"].values())
    cols = ["set", "runs", "recorded success", "P", "P = success", "H", "rows removed P", "rows removed H",
            "frames removed P", "frames removed H"]
    body, num = [], {}
    for st in ("CL2", "CL3", "T53", "all"):
        rs = [r for r in R if st == "all" or r["set"] == st]
        n = len(rs)
        v = {"runs": n, "success": sum(r["recorded"]["success"] for r in rs),
             "P": sum(r["t_P"] is not None for r in rs),
             "P_agree": sum((r["t_P"] is not None) == r["recorded"]["success"] for r in rs),
             "H": sum(r["t_H"] is not None for r in rs),
             "rows": 600 * n, "frames": 75 * n,
             "rows_removed_P": sum(r["rows_removed_P"] for r in rs), "rows_removed_H": sum(r["rows_removed_H"] for r in rs),
             "frames_removed_P": sum(r["frames_removed_P"] for r in rs), "frames_removed_H": sum(r["frames_removed_H"] for r in rs)}
        num[st] = v
        pct = lambda a, b: f"{a} ({100 * a / b:.1f}%)"
        body.append([st, n, v["success"], v["P"], f"{v['P_agree']}/{n}", v["H"],
                     pct(v["rows_removed_P"], v["rows"]), pct(v["rows_removed_H"], v["rows"]),
                     pct(v["frames_removed_P"], v["frames"]), pct(v["frames_removed_H"], v["frames"])])
    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.axis("off")
    tb = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.6)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(figstyle.GRID_INK)
        if i == 0:
            c.set_facecolor(figstyle.SEQ_BLUE[0])
    note = (f"T53 의 잠정 proxy (첫 physical placed) 는 401 frame 을 제거했다 → 정식 P 는 {num['T53']['frames_removed_P']} frame. "
            "arm 별 수치는 파라미터 확정 전이라 열지 않았다." if ko else
            f"T53 provisional proxy (first physical placed) removed 401 frames; formal P removes {num['T53']['frames_removed_P']}. "
            "Arm-level numbers not opened before parameter confirmation.")
    ax.set_title("T60 — 종료 이후 제거되는 control row · inference frame (기록 수정 없이 mask 로)" if ko else
                 "T60 — control rows / inference frames after completion (mask only)", fontsize=12)
    fig.text(0.01, 0.02, note, fontsize=9)
    figstyle.style_axes(fig, [ax])
    fig.savefig(fig_dir / "t60-removed-table.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    num["T53_proxy_frames_removed"] = proxy_t53_frames
    T.write_json(fig_dir / "t60-removed-table.json", num)
    return num


def main(step1: pathlib.Path, fig_dir: pathlib.Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    ko = figstyle.use_korean()
    ts = json.loads((step1 / "timestamps.json").read_text())
    proxy = json.loads((ROOT / "doc/reviews/TERMINATION_SCOPE_AUDIT_20261013.json").read_text())["frames_removed"]
    scene(ts, fig_dir, ko)
    g = gap(ts, fig_dir, ko)
    t = table(ts, fig_dir, ko, proxy)
    print(json.dumps({k: v for k, v in g.items() if k != "per_run"}, indent=1)); print(json.dumps(t, indent=1))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
