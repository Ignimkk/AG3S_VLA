"""T60b figures — real scene + per-chunk proposal of a not-returned vs a returned run, run-level graph, table.

usage: t60b_figures.py RETURN_JSON FIG_DIR      (.venv-openpi-live: cv2 for the recorded video)
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

PAIR = ("CL3/C1a_ep110_s1101", "CL3/C1a_ep11_s111")


def frame(video: pathlib.Path, k: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, img = cap.read(); cap.release()
    assert ok
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def chunk_series(key: str, r: dict) -> dict:
    s, name = key.split("/")
    base = SETS[s]; arm = r["arm"]
    rd = base / "runs" / arm / "R" / name
    tj = np.load(rd / "traj.npz", allow_pickle=True)
    sl = T.ARM_SLICE[r["used_arm"]]
    q = tj["measured_qpos"]; q0 = q[0, sl]; cs = np.asarray(tj["chunk_start_steps"]); L = int(tj["execution_length"])
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {int(x["t_step"]): x for x in rows if x["kind"] == "planning" and x.get("actions_reference") is not None}
    err = lambda a: np.abs(np.asarray(a)[..., sl] - q0).max(axis=-1)
    out = {"t": [], "ref8": [], "ref50": [], "to8": []}
    for t in cs:
        x = plan.get(int(t))
        if x is None:
            continue
        ref = np.asarray(x["actions_reference"], float); act = np.asarray(x["actions"], float)
        out["t"].append(int(t)); out["ref8"].append(float(err(ref[:L]).min())); out["ref50"].append(float(err(ref).min()))
        out["to8"].append(float(err(act[:L]).min()))
    out["meas"] = np.abs(q[:, sl] - q0).max(axis=1).tolist()
    out["video"] = str(rd / "third_person_front.mp4")
    return out


def scene(d: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.2, 0.8, 1.2, 0.8])
    side = {}
    for i, key in enumerate(PAIR):
        r = d["runs"][key]; cs = chunk_series(key, r)
        n = len(cs["meas"]); tp = r["t_P"]
        picks = [("t_P", tp), ("t_P + 60", min(n - 1, tp + 60)), ("t_P + 150", min(n - 1, tp + 150)), ("last row", n - 1)]
        for j, (lab, k) in enumerate(picks):
            ax = fig.add_subplot(gs[2 * i, j]); ax.imshow(frame(pathlib.Path(cs["video"]), int(k)))
            ax.set_title(f"{lab} = row {k}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[2 * i + 1, :])
        ax.plot(np.arange(n), cs["meas"], color=figstyle.CATEGORICAL[0], lw=1.4, label="measured max|q−q0|")
        ax.plot(cs["t"], cs["ref8"], "o", ms=3, color=figstyle.CATEGORICAL[1], label="policy proposal, executed part (first 8): min err")
        ax.plot(cs["t"], cs["ref50"], ".", ms=3, color=figstyle.CATEGORICAL[2], label="policy proposal, full 50 steps: min err")
        ax.plot(cs["t"], cs["to8"], "x", ms=3, color=figstyle.CATEGORICAL[4], label="after TO, first 8: min err")
        ax.axhline(0.35, color=figstyle.INK_2, ls=":", lw=1); ax.axvline(tp, color=figstyle.CATEGORICAL[5], lw=1.4)
        ax.text(tp + 3, 1.3, "t_P", color=figstyle.CATEGORICAL[5], fontsize=9)
        if r["t_H"] is not None:
            ax.axvline(r["t_H"], color=figstyle.CATEGORICAL[3], lw=1.4); ax.text(r["t_H"] + 3, 1.3, "t_H", color=figstyle.CATEGORICAL[3], fontsize=9)
        ax.set_xlim(0, n); ax.set_ylim(0, 1.6); ax.set_xlabel("control row")
        ax.set_title(f"{key} — {(r['category'] if not r['returned'] else 'returned (H reached)')}  (episode {r['episode']}, {r['used_arm']} arm)", loc="left", fontsize=10)
        if i == 0:
            ax.legend(fontsize=7.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.28), frameon=False)
        side[key] = {"t_P": tp, "t_H": r["t_H"], "category": r["category"], "picked_rows": dict(picks),
                     "chunks": {k: cs[k] for k in ("t", "ref8", "ref50", "to8")}}
    figstyle.style_axes(fig, fig.axes)
    fig.suptitle("T60b — 놓은 뒤 정책이 제안한 동작: 복귀 미도달 run (위) 대 복귀 run (아래)" if ko else
                 "T60b — what the policy proposes after placement: not returned (top) vs returned (bottom)", fontsize=13)
    fig.tight_layout(); fig.savefig(fig_dir / "t60b-proposal-scene.png", dpi=120); plt.close(fig)
    T.write_json(fig_dir / "t60b-proposal-scene.json", side)
    return side


def graph(d: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    runs = list(d["runs"].values())
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    ax = axes[0]
    for ret, c, m, lab in ((True, figstyle.CATEGORICAL[0], "o", "H reached"), (False, figstyle.CATEGORICAL[1], "x", "H not reached")):
        xs = [v["t_P"] for v in runs if v["returned"] == ret]; ys = [v["pred8_ref_min"] for v in runs if v["returned"] == ret]
        ax.scatter(xs, ys, s=24, color=c, marker=m, label=f"{lab} (n={len(xs)})")
    ax.axhline(0.35, color=figstyle.INK_2, ls=":", lw=1)
    ax.set_xlabel("t_P (control row)"); ax.set_ylabel("closest proposed pose, executed part (rad)")
    ax.set_title("정책 제안 (실행 구간) 이 준비자세에 가장 가까웠던 값 — run 별" if ko else "closest executed-part proposal per run", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    eps = d["by_episode"]; keys = sorted(eps, key=lambda k: (-eps[k]["not_returned"], int(k)))
    nr = [eps[k]["not_returned"] for k in keys]; rr = [eps[k]["returned"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, rr, color=figstyle.CATEGORICAL[0], label="returned")
    ax.bar(x, nr, bottom=rr, color=figstyle.CATEGORICAL[1], label="not returned")
    ax.set_xticks(x); ax.set_xticklabels([f"ep{k}\n{eps[k]['used_arm'][0].upper()}" for k in keys], fontsize=7)
    ax.set_ylabel("placed runs (all arms, all seeds)")
    ax.set_title("episode 별 — 복귀 미도달은 소수 scene 에 몰린다" if ko else "per episode", loc="left"); ax.legend(fontsize=8)
    figstyle.style_axes(fig, axes)
    fig.tight_layout(); fig.savefig(fig_dir / "t60b-runs.png", dpi=130); plt.close(fig)
    side = {"per_run": {k: {f: v[f] for f in ("t_P", "pred8_ref_min", "returned", "category")} for k, v in d["runs"].items()},
            "by_episode": eps}
    T.write_json(fig_dir / "t60b-runs.json", side)
    return side


def table(d: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    nr, rr = d["not_returned"], d["returned"]
    f = lambda x: f"{x['min']:.3f} / {x['median']:.3f} / {x['max']:.3f}" if x else "—"
    rows = [
        ["runs", nr["n"], rr["n"]],
        ["(b) blocked: HOLD after t_P (runs)", nr["hold_after_P_runs"], rr["hold_after_P_runs"]],
        ["policy proposal reaches ready, executed part (runs)", nr["pred8_ref_within_qtol_runs"], rr["pred8_ref_within_qtol_runs"]],
        ["policy proposal reaches ready, full 50 steps (runs)", nr["pred50_within_qtol_runs"], rr["pred50_within_qtol_runs"]],
        ["closest proposal, executed part (rad) min / med / max", f(nr["pred8_ref_min"]), f(rr["pred8_ref_min"])],
        ["TO change of the executed part (rad) min / med / max", f(nr["to_dev_max"]), f(rr["to_dev_max"])],
        ["TCP move after t_P (m) min / med / max", f(nr["tcp_move_m"]), f(rr["tcp_move_m"])],
        ["rows after t_P min / med / max", f(nr["rows_after_P"]), f(rr["rows_after_P"])],
        ["category (pre-registered)", ", ".join(f"{k} {v}" for k, v in nr["category"].items()), "— (returned)"],
    ]
    fig, ax = plt.subplots(figsize=(15, 3.6)); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["", "H not reached", "H reached"], loc="center", cellLoc="center",
                  colWidths=[0.44, 0.28, 0.28])
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(figstyle.GRID_INK)
        if i == 0:
            c.set_facecolor(figstyle.SEQ_BLUE[0])
    ax.set_title("T60b — 복귀 미도달의 기록 기반 분류" if ko else "T60b — record-based classification", fontsize=12)
    figstyle.style_axes(fig, [ax])
    fig.savefig(fig_dir / "t60b-table.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    T.write_json(fig_dir / "t60b-table.json", {"rows": rows})
    return {"rows": rows}


def main(ret_json: pathlib.Path, fig_dir: pathlib.Path) -> None:
    ko = figstyle.use_korean()
    d = json.loads(ret_json.read_text())
    fig_dir.mkdir(parents=True, exist_ok=True)
    scene(d, fig_dir, ko); graph(d, fig_dir, ko); table(d, fig_dir, ko)
    print("ok")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
