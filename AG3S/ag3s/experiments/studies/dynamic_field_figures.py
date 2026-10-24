"""AG3S-DYNAMIC-FIELD figures — scene (one replayed request, two configurations), per-request time series, summary table.

  MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.dynamic_field_figures \
      --verify <run>/verify/AG3S-DYNAMIC-FIELD --kit <run>/verify/IFG-APPLE-NBC-TARGETSURFACE --configuration NBCTS \
      --scene-episode 126 --scene-request 25 --fig AG3S/ag3s/docs/figures/dynamic-field

Reads `summary_all.json` and `full_*/rows_<configuration>.json`; the scene panel replays one episode (GPU) for
`off` and `D1_D2` and draws the main TSDF's surface-band voxels at that request (stale memory / other stale / ok)
over the GT surfaces, from above (x–y) and from the side (x–z). Every figure has a `.json` sidecar with its numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402
from AG3S.ag3s.experiments.studies import dynamic_field_replay as R  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
NAMES = {"off": T("끔 (지금 서버)", "off (current server)"),
         "D1_carve": T("보이는 빈 공간 비우기", "free seen-through space"),
         "D2_robot_free": T("로봇 몸 공간 비우기", "free robot-occupied space"),
         "D1_D2": T("둘 다", "both"),
         "D0_curobo_decay": T("cuRobo 감쇠 (비교)", "cuRobo decay (comparison)")}
ORDER = ["off", "D1_carve", "D2_robot_free", "D1_D2", "D0_curobo_decay"]


def load_rows(verify: pathlib.Path):
    rows = {}
    for f in glob.glob(str(verify / "full_*" / "rows_*.json")):
        v = pathlib.Path(f).stem[len("rows_"):]
        rows[v] = json.loads(pathlib.Path(f).read_text())
    return rows


def fig_timeseries(rows, out: pathlib.Path, episodes=("211", "126", "954")):
    fig, axes = plt.subplots(2, len(episodes), figsize=(4.6 * len(episodes), 6.2), sharex="col")
    side = {}
    for c, ep in enumerate(episodes):
        for k, v in enumerate(ORDER):
            if v not in rows or ep not in rows[v]:
                continue
            rr = rows[v][ep]
            x = [r["request"] for r in rr]
            st = [r["main"]["stale"] for r in rr]
            mem = [r["main"]["stale_memory"] for r in rr]
            side.setdefault(ep, {})[v] = {"request": x, "stale": st, "stale_memory": mem}
            kw = dict(color=FS.CATEGORICAL[k], lw=1.6, label=NAMES[v])
            axes[0, c].plot(x, st, **kw)
            axes[1, c].plot(x, mem, **kw)
        axes[0, c].set_title(T(f"장면 {ep}", f"scene {ep}"))
        axes[1, c].set_xlabel(T("요청 번호", "request"))
    axes[0, 0].set_ylabel(T("잔상 voxel 수", "stale voxels"))
    axes[1, 0].set_ylabel(T("그중 옛 기억", "of which: memory of moved objects"))
    axes[0, -1].legend(fontsize=8, frameon=False)
    FS.style_axes(fig, axes.ravel())
    fig.suptitle(T("요청별 잔상 voxel: 같은 기록을 다섯 구성으로 다시 쌓은 거리장 (main 5 mm TSDF)",
                   "Stale voxels per request: the same recording replayed under five configurations"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    out.with_suffix(".json").write_text(json.dumps(side))
    plt.close(fig)


def fig_table(summary, out: pathlib.Path):
    base = summary["off"]
    cols = [T("구성", "configuration"), T("잔상", "stale"), T("옛 기억", "memory"),
            T("잊음: 보임", "missed: visible"), T("잊음: 가려짐·봤던", "missed: occluded, seen"),
            T("시간 중앙값 ms", "ms median"), T("판정", "verdict")]
    cells, side = [], {}
    for v in ORDER:
        d = summary.get(v)
        if d is None:
            continue
        ds = 100.0 * (d["stale"] - base["stale"]) / base["stale"]
        dm = 100.0 * (d["stale_memory"] - base["stale_memory"]) / base["stale_memory"]
        pv = 100.0 * (d["missed"]["visible"]["frac"] - base["missed"]["visible"]["frac"])
        po = 100.0 * (d["missed"]["occluded_seen_before"]["frac"] - base["missed"]["occluded_seen_before"]["frac"])
        ok = v != "off" and ds <= -50.0 and pv <= 1.0 and po <= 1.0
        verdict = "—" if v == "off" else (T("기준 충족", "meets") if ok else T("기준 미달", "below"))
        cells.append([NAMES[v], f"{d['stale']:,} ({ds:+.0f}%)", f"{d['stale_memory']:,} ({dm:+.0f}%)",
                      f"{100*d['missed']['visible']['frac']:.2f}% ({pv:+.2f}pp)",
                      f"{100*d['missed']['occluded_seen_before']['frac']:.2f}% ({po:+.2f}pp)",
                      f"{d['ms_median']:.0f}", verdict])
        side[v] = {"stale": d["stale"], "stale_change_pct": round(ds, 2), "memory": d["stale_memory"],
                   "memory_change_pct": round(dm, 2), "missed_visible_pp": round(pv, 3),
                   "missed_occluded_seen_pp": round(po, 3), "ms_median": d["ms_median"], "ms_p95": d["ms_p95"],
                   "meets_criterion": ok}
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    ax.set_title(T("사과 10 장면 거리장 replay 요약 — 판정 기준(결과 전 고정): 잔상 50% 이상 감소 · 잊은 실제 장애물 +1 pp 이하",
                   "Field replay, 10 apple scenes — criterion (fixed before results): stale −50% or more, missed obstacles +1 pp or less"),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    out.with_suffix(".json").write_text(json.dumps(side, indent=1))
    plt.close(fig)


def fig_scene(args, out: pathlib.Path):
    kit = pathlib.Path(args.kit)
    argv = shlex.split((kit / "pre" / f"argv_{args.configuration}.txt").read_text())
    argv = [t for i, t in enumerate(argv) if t not in ("--run-dir", "--record-constraints")
            and (i == 0 or argv[i - 1] not in ("--run-dir", "--record-constraints"))]
    gt = R.GroundTruth(argv[argv.index("--model-xml") + 1])
    pdir, chunks = R.episodes_of(kit, args.configuration)[args.scene_episode]
    chunks = [c for c in chunks if c[0] <= args.scene_request]
    caps = {}
    for v in ("off", "D1_D2"):
        cap = {args.scene_request: None}
        R.run_episode(argv, R.VARIANTS[v], pdir, chunks, gt, capture=cap)
        caps[v] = cap[args.scene_request]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    side = {}
    for c, v in enumerate(("off", "D1_D2")):
        cp = caps[v]
        band, stale, mem = cp["band"], cp["stale"], cp["memory"]
        other = stale & ~mem
        ok = ~stale
        lo = np.percentile(np.vstack([p for kind, _, p in cp["gt"] if len(p) and kind == "fruit"]), 0, axis=0) - 0.35
        hi = lo + 0.85
        sel = np.all((band >= lo) & (band <= hi), axis=1)
        side[v] = {"request": args.scene_request, "state": cp["state"], "n_band_in_view": int(sel.sum()),
                   "n_stale_memory_in_view": int((mem & sel).sum()), "n_stale_other_in_view": int((other & sel).sum())}
        for r, (ia, ib, lab) in enumerate(((0, 1, ("x", "y")), (0, 2, ("x", "z")))):
            ax = axes[r, c]
            for kind, name, p in cp["gt"]:
                if not len(p):
                    continue
                q = p[np.all((p >= lo) & (p <= hi), axis=1)]
                if not len(q):
                    continue
                col = FS.CATEGORICAL[2] if kind == "fruit" and name == "apple" else (FS.CATEGORICAL[4] if kind == "fruit" else "#c9c8c3")
                ax.scatter(q[:, ia], q[:, ib], s=0.3, c=col, alpha=0.4, linewidths=0)
            ax.scatter(band[sel & ok, ia], band[sel & ok, ib], s=0.5, c="#8a8a85", alpha=0.25, linewidths=0,
                       label=T("TSDF 표면 voxel", "TSDF surface voxels"))
            ax.scatter(band[sel & other, ia], band[sel & other, ib], s=2.0, c=FS.CATEGORICAL[3], linewidths=0,
                       label=T("잔상 (그 밖)", "stale (other)"))
            ax.scatter(band[sel & mem, ia], band[sel & mem, ib], s=3.0, c=FS.CATEGORICAL[7], linewidths=0,
                       label=T("잔상: 움직인 물체의 옛 자리", "stale: old place of a moved object"))
            cr, rr = cp["robot"]
            ins = np.all((cr >= lo) & (cr <= hi), axis=1)
            for cc, r0 in zip(cr[ins], rr[ins]):
                ax.add_patch(plt.Circle((cc[ia], cc[ib]), r0, fill=False, lw=0.4, color=FS.CATEGORICAL[0], alpha=0.5))
            ax.set_xlim(lo[ia], hi[ia])
            ax.set_ylim(lo[ib], hi[ib])
            ax.set_aspect("equal")
            ax.set_xlabel(f"{lab[0]} (m)")
            ax.set_ylabel(f"{lab[1]} (m)")
            ax.set_title(f"{NAMES[v]} — {T('위에서', 'top view') if r == 0 else T('옆에서', 'side view')}", fontsize=10)
        axes[0, c].legend(fontsize=7, markerscale=4, loc="upper right", frameon=False)
    FS.style_axes(fig, axes.ravel())
    fig.suptitle(T(f"장면 {args.scene_episode} 요청 {args.scene_request} ({caps['off']['state']}): 초록 = 사과 GT, 분홍 = 다른 과일 GT, 파랑 원 = 로봇 구",
                   f"scene {args.scene_episode} request {args.scene_request}: green = apple GT, pink = other fruit GT, blue = robot spheres"),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    out.with_suffix(".json").write_text(json.dumps(side, indent=1))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--configuration", required=True)
    ap.add_argument("--scene-episode", type=int, default=126)
    ap.add_argument("--scene-request", type=int, default=25)
    ap.add_argument("--fig", required=True)
    ap.add_argument("--skip-scene", action="store_true")
    a = ap.parse_args()
    verify, fig = pathlib.Path(a.verify), pathlib.Path(a.fig)
    fig.mkdir(parents=True, exist_ok=True)
    summary = json.loads((verify / "summary_all.json").read_text())["summary"]
    rows = load_rows(verify)
    fig_timeseries(rows, fig / "dynamic-field-timeseries")
    fig_table(summary, fig / "dynamic-field-table")
    if not a.skip_scene:
        fig_scene(a, fig / f"dynamic-field-scene-ep{a.scene_episode}-r{a.scene_request}")
    print("figures ->", fig)


if __name__ == "__main__":
    main()
