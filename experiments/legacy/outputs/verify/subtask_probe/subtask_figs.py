"""Rule-A figures for the pi0.5 subtask-decoding feasibility probe (scene / graph / table + json sidecars)."""
from common.workspace import workspace_path as _ag3s_workspace_path

import glob
import json
import pathlib
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

SP = pathlib.Path(__file__).parent
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/subtask'))
OUT.mkdir(parents=True, exist_ok=True)
KO = figstyle.use_korean()
R = json.load(open(SP / "episode.json"))
DISC = json.load(open(SP / "discover.json"))

VARIANTS = ["base", "ft16d", "ft_nolora"]
VLABEL = {"base": "pi05_base (released)", "ft16d": "16D ft (LoRA on, served)", "ft_nolora": "16D ft, LoRA off"}
VCOLOR = dict(zip(VARIANTS, figstyle.CATEGORICAL[:3]))
EXPECT = {"reach": "pick", "held": "place", "released": "home"}
PHASE_BG = {"reach": "#eef3fb", "held": "#fdf0e6", "released": "#e9f6ef"}
CLS_COLOR = {"pick": figstyle.CATEGORICAL[0], "place": figstyle.CATEGORICAL[1], "home": figstyle.CATEGORICAL[2],
             "move": figstyle.CATEGORICAL[3], "other": "#9a9a95", "empty": "#d8d7d2", "action_tokens": "#3a3a38"}
GEN_T = list(R["gen_templates"])


def metrics():
    rows = []
    for rk, run in R["runs"].items():
        fr = run["frames"]
        for mk in VARIANTS:
            m = dict(run=rk, variant=mk, n=len(fr))
            exp = [EXPECT[f["gt"]["phase"]] for f in fr]
            arg = [max(f["score"][mk], key=f["score"][mk].get) for f in fr]
            m["score_argmax_acc"] = float(np.mean([a == e for a, e in zip(arg, exp)]))
            pp = [(f["score"][mk]["place"] - f["score"][mk]["pick"], f["gt"]["phase"]) for f in fr
                  if f["gt"]["phase"] in ("reach", "held")]
            m["pick_vs_place_acc"] = float(np.mean([(d > 0) == (ph == "held") for d, ph in pp])) if pp else None
            for tk in GEN_T:
                cls = [f["gen"][mk][tk]["cls"] for f in fr]
                m[f"gen_acc[{tk}]"] = float(np.mean([c == e for c, e in zip(cls, exp)]))
                m[f"gen_action_tokens[{tk}]"] = float(np.mean([c == "action_tokens" for c in cls]))
                txt = [f["gen"][mk][tk]["text"].lower() for f in fr if f["gen"][mk][tk]["cls"] in ("pick", "place")]
                m[f"gen_object_ok[{tk}]"] = (float(np.mean([run["fruit"] in t for t in txt])) if txt else None)
                m[f"gen_n_pickplace[{tk}]"] = len(txt)
            rows.append(m)
    return rows


def pooled(rows, key):
    out = {}
    for mk in VARIANTS:
        num = den = 0.0
        for r in rows:
            if r["variant"] == mk and r[key] is not None:
                w = r["n"] if not key.startswith("gen_object_ok") else r[key.replace("object_ok", "n_pickplace")]
                num += r[key] * w
                den += w
        out[mk] = num / den if den else None
    return out


def fig_scene():
    run = R["runs"]["ep1807_apple"]
    files = sorted(glob.glob(run["run"] + "/step_*.npz"))
    by_t = {int(np.load(f)["t_step"]): f for f in files}
    ts = [0, 152, 200, 240, 320]
    fr = {f["t_step"]: f for f in run["frames"]}
    fig, axes = plt.subplots(3, len(ts), figsize=(3.3 * len(ts), 9.2),
                             gridspec_kw=dict(height_ratios=[1, 1, 1.25]))
    for j, t in enumerate(ts):
        d = np.load(by_t[t])
        f = fr[t]
        axes[0, j].imshow(d["image_cam_high"])
        axes[1, j].imshow(d["image_cam_left_wrist"])
        ph = f["gt"]["phase"]
        axes[0, j].set_title(f"t={t}  GT: {ph}", fontsize=11, color=figstyle.INK,
                             backgroundcolor=PHASE_BG[ph])
        for a in axes[:2, j]:
            a.set_xticks([]), a.set_yticks([])
        ax = axes[2, j]
        ax.axis("off")
        y = 0.97
        for mk in VARIANTS:
            g = f["gen"][mk]["T;nl"]
            txt = g["text"].replace("Subtask:", "").strip()
            if g["cls"] == "action_tokens":
                txt = "<action tokens>"
            ax.text(0.0, y, VLABEL[mk], color=VCOLOR[mk], fontsize=8.5, fontweight="bold", va="top",
                    transform=ax.transAxes)
            ax.text(0.0, y - 0.075, "\n".join(textwrap.wrap(txt or "<empty>", 30)[:2]), fontsize=9,
                    va="top", transform=ax.transAxes, color=figstyle.INK)
            s = f["score"][mk]
            ax.text(0.0, y - 0.215, f"logp pick {s['pick']:.1f} / place {s['place']:.1f} / home {s['home']:.1f}",
                    fontsize=7.5, va="top", transform=ax.transAxes, color=figstyle.INK_2)
            y -= 0.33
    axes[0, 0].set_ylabel("cam_high", fontsize=10)
    axes[1, 0].set_ylabel("cam_left_wrist", fontsize=10)
    title = ("ep1807 (apple): pi0.5 가 prefix 에서 말한 subtask — template 'Task: {p};\\n'" if KO else
             "ep1807 (apple): subtask text decoded from the pi0.5 prefix — template 'Task: {p};\\n'")
    fig.suptitle(title, fontsize=12, color=figstyle.INK)
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.tight_layout()
    p = OUT / "subtask-scene-ep1807.png"
    fig.savefig(p, dpi=130, facecolor=figstyle.SURFACE)
    json.dump({str(t): dict(gt=fr[t]["gt"], gen={mk: fr[t]["gen"][mk] for mk in VARIANTS},
                            score=fr[t]["score"]) for t in ts}, open(p.with_suffix(".json"), "w"), indent=1)
    return p


def shade(ax, fr):
    t = [f["t_step"] for f in fr]
    ph = [f["gt"]["phase"] for f in fr]
    start = 0
    for i in range(1, len(t) + 1):
        if i == len(t) or ph[i] != ph[start]:
            hi = t[i] if i < len(t) else t[-1] + (t[-1] - t[-2])
            ax.axvspan(t[start], hi, color=PHASE_BG[ph[start]], zorder=0, lw=0)
            start = i


def fig_timeline():
    runs = list(R["runs"])
    fig, axes = plt.subplots(len(runs), 2, figsize=(15, 3.1 * len(runs)),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    side = {}
    for i, rk in enumerate(runs):
        run = R["runs"][rk]
        fr = run["frames"]
        t = np.array([f["t_step"] for f in fr])
        ax = axes[i, 0]
        shade(ax, fr)
        for mk in VARIANTS:
            dlt = np.array([f["score"][mk]["place"] - f["score"][mk]["pick"] for f in fr])
            ax.plot(t, dlt, color=VCOLOR[mk], lw=1.8, label=VLABEL[mk])
            side.setdefault(rk, {})[mk] = dict(t=t.tolist(), delta_place_minus_pick=dlt.round(3).tolist())
        ax.axhline(0, color=figstyle.INK_2, lw=0.8, ls="--")
        ax.set_ylabel("Δ = log p(place) − log p(pick)", fontsize=9)
        ax.set_title(f"{rk}  —  prompt: {run['prompt']!r}", fontsize=10, loc="left")
        if i == 0:
            ax.legend(fontsize=8, loc="upper left", ncol=3)
        # class strip: GT row + 3 variants x template T;nl
        ax2 = axes[i, 1]
        rows = [("GT → expected", [EXPECT[f["gt"]["phase"]] for f in fr])]
        for mk in VARIANTS:
            for tk in ("T;nl", "T.Sub_causal"):
                rows.append((f"{mk} · {tk}", [f["gen"][mk][tk]["cls"] for f in fr]))
        dt = np.median(np.diff(t))
        for r, (lab, cls) in enumerate(rows):
            for tt, c in zip(t, cls):
                ax2.add_patch(plt.Rectangle((tt, len(rows) - 1 - r - 0.4), dt, 0.8, color=CLS_COLOR[c], lw=0))
        ax2.set_xlim(t[0], t[-1] + dt)
        ax2.set_ylim(-0.6, len(rows) - 0.4)
        ax2.set_yticks(range(len(rows)))
        ax2.set_yticklabels([lab for lab, _ in rows][::-1], fontsize=8)
        ax2.set_title("generated subtask — first-verb class", fontsize=10, loc="left")
        side[rk]["class_rows"] = {lab: cls for lab, cls in rows}
    for ax in axes[-1]:
        ax.set_xlabel("t_step (sim step)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLS_COLOR.values()]
    fig.legend(handles, list(CLS_COLOR), loc="lower right", ncol=7, fontsize=8, frameon=False,
               bbox_to_anchor=(0.99, 0.0))
    hb = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_BG.values()]
    fig.legend(hb, [f"GT {k}" for k in PHASE_BG], loc="lower left", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.01, 0.0))
    figstyle.style_axes(fig, axes.ravel().tolist())
    fig.suptitle("Subtask 신호가 참값 phase 를 따라가는가 — 4 episodes" if KO else
                 "Does the decoded subtask track the true phase? — 4 episodes", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    p = OUT / "subtask-timeline.png"
    fig.savefig(p, dpi=120, facecolor=figstyle.SURFACE)
    json.dump(side, open(p.with_suffix(".json"), "w"))
    return p


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def fig_table(rows):
    runs = list(R["runs"])
    au = {}
    for rk in runs:
        fr = R["runs"][rk]["frames"]
        ph = [f["gt"]["phase"] for f in fr]
        for mk in VARIANTS:
            s = [f["score"][mk] for f in fr]
            d = [x["place"] - x["pick"] for x in s]
            h = [x["home"] - max(x["pick"], x["place"]) for x in s]
            au.setdefault(mk, {})[rk] = dict(
                held_vs_reach=auroc([x for x, p in zip(d, ph) if p == "held"], [x for x, p in zip(d, ph) if p == "reach"]),
                released_vs_rest=auroc([x for x, p in zip(h, ph) if p == "released"], [x for x, p in zip(h, ph) if p != "released"]),
                argmax_home_frac=float(np.mean([max(x, key=x.get) == "home" for x in s])))
    gen_keys = [("gen_acc[p_nl]", "gen acc\n'{p}\\n'"), ("gen_acc[T.Sub_causal]", "gen acc\n'Task: {p}. Subtask: '"),
                ("gen_action_tokens[T;nl]", "action-token rate\n'Task: {p};\\n'"),
                ("gen_object_ok[T;nl]", "object name ok\n'Task: {p};\\n'")]
    pool = {k: pooled(rows, k) for k, _ in gen_keys}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 6.4), gridspec_kw=dict(height_ratios=[1.1, 1]))
    for a in (a1, a2):
        a.axis("off")
    hdr = ["variant"] + [f"{rk}\nΔ held vs reach" for rk in runs] + [f"{rk}\nreleased vs rest" for rk in runs]
    cell = [[VLABEL[mk]] + [f"{au[mk][rk]['held_vs_reach']:.2f}" for rk in runs]
            + [f"{au[mk][rk]['released_vs_rest']:.2f}" for rk in runs] for mk in VARIANTS]
    t1 = a1.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
    a1.set_title("(a) score 기반 AUROC — 0.5 = 무작위, 1.0 = 완전 분리 (threshold 없이 잰 분리도)" if KO else
                 "(a) score-based AUROC (0.5 = chance, 1.0 = perfect separation)", fontsize=10, loc="left")
    cell2 = [[VLABEL[mk]] + [("—" if pool[k][mk] is None else f"{100 * pool[k][mk]:.0f}%") for k, _ in gen_keys]
             + [f"{100 * np.mean([au[mk][rk]['argmax_home_frac'] for rk in runs]):.0f}%"] for mk in VARIANTS]
    t2 = a2.table(cellText=cell2, colLabels=["variant"] + [lab for _, lab in gen_keys] + ["3-way argmax\n= 'home' 비율" if KO else "3-way argmax\n= 'home'"],
                  loc="center", cellLoc="center")
    tm = R["timing"]
    n = sum(len(v["frames"]) for v in R["runs"].values())
    a2.set_title((f"(b) 자유 생성 ({n} frames, 기대: reach→pick, held→place, released→home). 3-way argmax 는 항상 'home' → majority baseline 64% 와 같다.  "
                  f"latency: {tm['median_ms']:.1f} ms/token (naive, no KV cache, H200)") if KO else
                 f"(b) free generation ({n} frames). latency {tm['median_ms']:.1f} ms/token", fontsize=9, loc="left")
    for tb in (t1, t2):
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.5)
        tb.scale(1, 2.3)
        for (r, c), cl in tb.get_celld().items():
            cl.set_edgecolor(figstyle.GRID_INK)
            if r == 0:
                cl.set_facecolor("#efeeea")
    fig.patch.set_facecolor(figstyle.SURFACE)
    p = OUT / "subtask-accuracy-table.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=figstyle.SURFACE)
    json.dump(dict(auroc=au, pooled_generation=pool, per_run=rows, timing=tm,
                   phase_counts={rk: {ph: sum(f["gt"]["phase"] == ph for f in R["runs"][rk]["frames"]) for ph in PHASE_BG} for rk in runs}),
              open(p.with_suffix(".json"), "w"), indent=1)
    return p


def fig_discover():
    """Template discovery on 3 frames: which prefix makes each variant speak English at all."""
    pref = list(dict.fromkeys(r["prefix"] + (" (causal)" if r["causal"] else "") for r in DISC))
    frames = sorted({(r["t_step"], r["phase"]) for r in DISC})
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), sharey=True)
    side = {}
    for ax, mk in zip(axes, VARIANTS):
        grid = np.zeros((len(pref), len(frames)))
        texts = {}
        for r in DISC:
            if r["model"] != mk:
                continue
            i = pref.index(r["prefix"] + (" (causal)" if r["causal"] else ""))
            j = frames.index((r["t_step"], r["phase"]))
            t = r["text"]
            eng = t.strip() != "" and sum(ord(c) > 127 for c in t) / max(1, len(t)) < 0.2
            grid[i, j] = 1.0 if eng else (0.0 if t.strip() == "" else -1.0)
            texts[(i, j)] = t
        ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#3a3a38", "#f2f1ed", "#9ec5f4"]), vmin=-1, vmax=1,
                  aspect="auto")
        for (i, j), t in texts.items():
            s = t.replace("Subtask:", "S:").strip()
            s = "<action tokens>" if grid[i, j] < 0 else (s[:26] + ("…" if len(s) > 26 else ""))
            ax.text(j, i, s, ha="center", va="center", fontsize=6.6,
                    color="white" if grid[i, j] < 0 else figstyle.INK)
        ax.set_xticks(range(len(frames)))
        ax.set_xticklabels([f"t={t}\n{p}" for t, p in frames], fontsize=8)
        ax.set_title(VLABEL[mk], fontsize=10, color=VCOLOR[mk])
        side[mk] = {f"{pref[i]} | t={frames[j][0]}": t for (i, j), t in texts.items()}
    axes[0].set_yticks(range(len(pref)))
    axes[0].set_yticklabels([p.replace("\n", "\\n") for p in pref], fontsize=8)
    fig.suptitle("Template discovery (ep1807, 3 frames): 파랑 = 영어 문장, 검정 = FAST action token, 흰색 = 빈 출력" if KO else
                 "Template discovery (ep1807): blue = English, black = FAST action tokens, white = empty", fontsize=11)
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.tight_layout()
    p = OUT / "subtask-template-discovery.png"
    fig.savefig(p, dpi=130, facecolor=figstyle.SURFACE)
    json.dump(dict(prefixes={p: p for p in pref}, outputs=side), open(p.with_suffix(".json"), "w"), indent=1)
    return p


if __name__ == "__main__":
    rows = metrics()
    for p in (fig_discover(), fig_scene(), fig_timeline(), fig_table(rows)):
        print("wrote", p)
    for k in ("score_argmax_acc", "pick_vs_place_acc", "gen_acc[T;nl]", "gen_acc[p_nl]", "gen_acc[T.Sub_causal]",
              "gen_action_tokens[T;nl]", "gen_object_ok[T;nl]", "gen_object_ok[T.Sub_causal]"):
        print(k, {m: (None if v is None else round(v, 3)) for m, v in pooled(rows, k).items()})
