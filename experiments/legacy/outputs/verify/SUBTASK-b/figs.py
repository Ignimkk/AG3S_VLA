"""Rule-A figures for SUBTASK-b (scene / graph / table + json sidecars). Run in .venv-openpi-live."""
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import av  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

SP = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-b'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/subtask-b'))
OUT.mkdir(parents=True, exist_ok=True)
DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1'))
KO = figstyle.use_korean()
R = json.load(open(SP / "probe_results.json"))
P = pd.read_csv(SP / "predictions.csv")
BEST = R["best_feature"]
FEATS = list(R["features"])
CLASSES = ["pick", "place", "home"]
CC = dict(zip(CLASSES, figstyle.CATEGORICAL[:3]))
MODES = ["normal", "mask_a", "mask_b"]
MLABEL = {"normal": "normal", "mask_a": "mask (a) train normal", "mask_b": "mask (b) train masked"}
MCOLOR = {"normal": figstyle.INK, "mask_a": figstyle.CATEGORICAL[3], "mask_b": figstyle.CATEGORICAL[6]}
BG = {0: "#eef3fb", 1: "#fdf0e6", 2: "#e9f6ef"}
RUNS = ["ep1807_apple", "ep0_orange", "ep500_banana", "ep1200_pear"]


def save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=150, facecolor=figstyle.SURFACE, bbox_inches="tight")
    json.dump(data, open(OUT / f"{name}.json", "w"), indent=1)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def probs(set_, ep, t, mode, feat=BEST):
    r = P[(P.set == set_) & (P.episode == ep) & (P.t == t) & (P.feature == feat) & (P["mode"] == mode)]
    assert len(r) == 1, (set_, ep, t, mode, len(r))
    r = r.iloc[0]
    return [float(r.p_pick), float(r.p_place), float(r.p_home)], int(r.label)


def bars(ax, set_, ep, t):
    out = {}
    w = 0.26
    for j, m in enumerate(MODES):
        p, lab = probs(set_, ep, t, m)
        out[m] = p
        ax.bar(np.arange(3) + (j - 1) * w, p, w, color=[CC[c] for c in CLASSES],
               edgecolor=MCOLOR[m], linewidth=1.2, alpha=[1.0, 0.65, 0.4][j], label=MLABEL[m])
    ax.set_xticks(range(3), CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("p")
    return out, lab


def fig_scene():
    sets = []
    # closed-loop ep1807: middle frame of each GT segment
    cl = P[(P.set == "closed414") & (P.episode == "ep1807_apple") & (P.feature == BEST) & (P["mode"] == "normal")]
    for c in range(3):
        ts = sorted(cl[cl.label == c].t)
        sets.append(("closed414", "ep1807_apple", ts[len(ts) // 2], c))
    va = P[(P.set == "val") & (P.feature == BEST) & (P["mode"] == "normal")]
    ep = sorted(va.episode.unique())[0]
    for c in range(3):
        ts = sorted(va[(va.episode == ep) & (va.label == c)].t)
        sets.append(("val", ep, ts[len(ts) // 2], c))

    fig, axes = plt.subplots(4, 3, figsize=(12, 12), gridspec_kw=dict(height_ratios=[1.1, 0.7, 1.1, 0.7]))
    data = dict(feature=BEST, frames=[])
    for k, (s, ep, t, c) in enumerate(sets):
        row, col = (k // 3) * 2, k % 3
        ax = axes[row, col]
        if s.startswith("closed"):
            import glob
            files = sorted(glob.glob(_ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000/step_*.npz')))
            z = next(np.load(f) for f in files if int(np.load(f)["t_step"]) == t)
            img = np.concatenate([z["image_cam_high"], z["image_cam_left_wrist"]], 1)
            grip = float(min(z["state"][7], z["state"][15]))
            title = f"closed-loop ep1807  t_step={t}  GT={CLASSES[c]}"
        else:
            e = int(ep)
            ch = e // 1000
            ims = []
            for cam in ("cam_high", "cam_left_wrist"):
                with av.open(str(DS / f"videos/chunk-{ch:03d}/observation.images.{cam}/episode_{e:06d}.mp4")) as cnt:
                    for i, fr in enumerate(cnt.decode(video=0)):
                        if i == t:
                            ims.append(fr.to_ndarray(format="rgb24"))
                            break
            img = np.concatenate(ims, 1)
            df = pd.read_parquet(DS / f"data/chunk-{ch:03d}/episode_{e:06d}.parquet")
            row_ = df[df.frame_index == t].iloc[0]
            st = np.asarray(row_["observation.state"])
            grip = float(min(st[7], st[15]))
            title = f"val ep{e}  frame={t}  phase={int(row_.phase_index)}  GT={CLASSES[c]}"
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title + f"\ncam_high | cam_left_wrist   min grip={grip:.3f}", fontsize=8.5)
        bx = axes[row + 1, col]
        pr, lab = bars(bx, s, ep, t)
        data["frames"].append(dict(set=s, episode=str(ep), t=int(t), gt=CLASSES[c], min_grip=grip, probs=pr))
    axes[1, 0].legend(fontsize=7, loc="upper right")
    figstyle.style_axes(fig, [a for a in axes[[1, 3]].ravel()])
    fig.suptitle(f"SUBTASK-b probe ({BEST}) on real frames: p(pick/place/home), normal vs gripper-masked State",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "subtask-b-scene", data)


def fig_graph():
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.1])
    data = dict(feature=BEST, timelines={}, val_acc={})
    for i, rk in enumerate(RUNS):
        for j, mode in enumerate(MODES):
            ax = fig.add_subplot(gs[j, i])
            d = P[(P.set == "closed414") & (P.episode == rk) & (P.feature == BEST) & (P["mode"] == mode)].sort_values("t")
            t = d.t.to_numpy()
            lab = d.label.to_numpy()
            for k in range(len(t)):
                lo = t[k] - (t[k] - t[k - 1]) / 2 if k else t[k] - 0.5
                hi = t[k] + (t[k + 1] - t[k]) / 2 if k < len(t) - 1 else t[k] + 0.5
                ax.axvspan(lo, hi, color=BG[lab[k]], lw=0)
            for c, col in zip(CLASSES, ["p_pick", "p_place", "p_home"]):
                ax.plot(t, d[col], color=CC[c], lw=1.4, label=f"p({c})")
            acc = float((d.pred == d.label).mean())
            ax.set_title(f"{rk} [{mode}]  acc={acc:.3f}  n={len(d)}", fontsize=9)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("t_step")
            data["timelines"][f"{rk}__{mode}"] = dict(acc=acc, n=int(len(d)))
            figstyle.style_axes(fig, ax)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="center left")
    ax = fig.add_subplot(gs[3, :])
    x = np.arange(len(FEATS))
    w = 0.26
    for j, m in enumerate(MODES):
        v = [R["features"][f][f"val__{m}"]["accuracy"] for f in FEATS]
        ax.bar(x + (j - 1) * w, v, w, color=[figstyle.CATEGORICAL[0], figstyle.CATEGORICAL[3], figstyle.CATEGORICAL[6]][j],
               label=MLABEL[m])
        data["val_acc"][m] = dict(zip(FEATS, v))
    rv = R["rule"]["val"]["accuracy"]
    ax.axhline(rv, color=figstyle.INK, ls="--", lw=1)
    ax.text(-0.45, rv + 0.002, f"rule {rv:.3f}", ha="left", va="bottom", fontsize=8)
    ax.set_xticks(x, FEATS)
    lo = min(min(d.values()) for d in data["val_acc"].values())
    ax.set_ylim(max(0, lo - 0.05), 1.0)
    ax.set_ylabel("val 3-class accuracy")
    ax.set_title(f"val accuracy per feature (n={R['counts']['val']['frames_used']} frames, "
                 f"{R['counts']['val']['episodes']} ep)  -- background bands = closed-loop GT (blue reach, orange held, green released)",
                 fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    figstyle.style_axes(fig, ax)
    data["rule_val_acc"] = rv
    fig.suptitle(f"SUBTASK-b  closed-loop timelines ({BEST}; rows: normal / mask a / mask b; all closed-loop frames) and val accuracy per feature",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "subtask-b-graph", data)


def fig_table():
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    cols = ["feature"] + [f"{e}\n{m}" for e in ("val", "closed246") for m in ("normal", "mask a", "mask b")]
    rows, table = [], {}
    CT = json.load(open(SP / "controls_results.json"))
    for f in FEATS + ["rule", "ctrl: frame_index only", "ctrl: arm joints (14, no grip)"]:
        cells = [f]
        for e in ("val", "closed246"):
            for m in MODES:
                if f == "rule":
                    r = R["rule"][e] if m == "normal" else None
                elif f.startswith("ctrl: frame"):
                    r = CT["C1_frame_index_val"] if (m == "normal" and e == "val") else None
                elif f.startswith("ctrl: arm"):
                    r = CT["C2_arm_joint_probe"][e] if m == "normal" else None
                else:
                    r = R["features"][f][f"{e}__{m}"]
                if r is None:
                    cells.append("-")
                    continue
                cells.append(f"{r['accuracy']:.3f} / {r['auroc_macro']:.3f}")
                table[f"{f}|{e}|{m}"] = dict(acc=r["accuracy"], auroc_macro=r["auroc_macro"], n=r["n"])
        rows.append(cells)
    t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 2.0)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor("#efeee9")
        if r > 0 and rows[r - 1][0] == BEST:
            cell.set_facecolor("#eef3fb")
    ax.set_title("3-class accuracy / macro OvR AUROC   (train: "
                 f"{R['counts']['train']['episodes']} ep, {R['counts']['train']['frames_used']} frames; "
                 "mask a = train normal, eval masked; mask b = train & eval masked;\n"
                 "rule: min(s7,s15)<0.9 -> place, else ever-closed -> home, else pick; rule / frame_index AUROC on one-hot;\n"
                 "ctrl rows added by verifier (not in task.md): frame_index-only majority label, logistic probe on raw arm joints)",
                 fontsize=9)
    ax2 = fig.add_subplot(gs[1])
    f14 = R["fine14"]
    cm = np.asarray(f14["val__normal"]["confusion"], float)
    rs = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    cmap = figstyle.sequential_cmap()
    cmap.set_bad(figstyle.SURFACE)
    ax2.imshow(np.ma.masked_where(cm == 0, cmn), cmap=cmap, vmin=0, vmax=1)
    for i in range(14):
        for j in range(14):
            if cm[i, j]:
                ax2.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                         color="white" if cmn[i, j] > 0.6 else figstyle.INK)
    ax2.set_xticks(range(14))
    ax2.set_yticks(range(14))
    ax2.set_xlabel("predicted phase_index")
    ax2.set_ylabel("true phase_index")
    ax2.set_title(f"14-way probe ({f14['feature']}, normal) val confusion\nacc={f14['val__normal']['accuracy']:.3f}",
                  fontsize=9)
    figstyle.style_axes(fig, ax2)
    fig.tight_layout()
    save(fig, "subtask-b-table", dict(cells=table, fine14=f14, counts=R["counts"], controls=CT))


if __name__ == "__main__":
    fig_scene()
    fig_graph()
    fig_table()
