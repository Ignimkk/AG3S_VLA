from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json
sys.path.insert(0, _ag3s_workspace_path(''))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from AG3S.ag3s.experiments.common import figstyle as fs

KO = fs.use_korean()
OUT = Path(_ag3s_workspace_path('outputs/verify/T20'))
FIG = Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t20'))
FIG.mkdir(parents=True, exist_ok=True)
REC = Path(_ag3s_workspace_path('outputs/live_test/20260928_t14/server_constraints/run_0000'))
APPLE = np.array([0.564, 0.320, 0.850]); CRATE = np.array([0.517, -0.016, 0.880])
MS = [0.0, 0.05, 0.1, 0.3]; LOST = [None, 10, 20]
rep = {(m, l): json.load(open(OUT / f"t14_replay_min{m}_lost{l}.json")) for m in MS for l in LOST}
fid = json.load(open(OUT / "t14_fidelity.json"))
summ = json.load(open(OUT / "t14_sweep_summary.json"))
syn = json.load(open(OUT / "synth_summary.json"))
seqs = [r["seq"] for r in rep[(0.0, None)]]
L = (lambda ko, en: ko if KO else en)
C_DEF, C_MIN, C_LIVE = fs.CATEGORICAL[0], fs.CATEGORICAL[2], fs.INK_2
ST_COL = {"visible": fs.SEQ_BLUE[3], "occluded": fs.CATEGORICAL[1], "lost": fs.CATEGORICAL[7], None: "#ffffff"}

# ------------------------------------------------------------------ 1. graph
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8.2), gridspec_kw=dict(height_ratios=[1.0, 1.25]), sharex=True)
live = [r["live_target_to_apple_mm"] for r in fid]
d0 = [r["manip_to_apple_mm"] for r in rep[(0.0, None)]]
d1 = [r["manip_to_apple_mm"] for r in rep[(0.05, None)]]
ax.axvspan(11.5, 21.5, color=fs.GRID_INK, alpha=0.35, lw=0)
ax.plot(seqs, [np.nan if v is None else v for v in live], ls="--", color=C_LIVE, marker=".", lw=1.2,
        label=L("T14 live 기록 target (옛 코드)", "T14 live target (old code)"))
ax.plot(seqs, d0, color=C_DEF, marker="o", ms=4, lw=1.6, label=L("재생 manipulated · min_score 0.0 (기본)", "replay manipulated, min_score 0.0 (default)"))
ax.plot(seqs, d1, color=C_MIN, marker="s", ms=4, lw=1.6, label=L("재생 manipulated · min_score 0.05 / 0.1 / 0.3 (세 값 동일)", "replay manipulated, min_score 0.05/0.1/0.3 (identical)"))
ax.set_yscale("log"); ax.set_yticks([10, 20, 50, 100, 200, 500]); ax.set_yticklabels(["10", "20", "50", "100", "200", "500"]); ax.minorticks_off(); ax.set_ylabel(L("centroid → 사과 [mm]", "centroid to apple [mm]"))
ax.axhline(60, color=fs.INK_2, lw=0.8, ls=":"); ax.text(31.4, 62, "60 mm tolerance", fontsize=7, color=fs.INK_2, ha="right", va="bottom")
for s, v in zip(seqs, d1):
    pass
ax.legend(fontsize=8, frameon=False, loc="center left")
ax.set_title(L("T14 seq 1–31 오프라인 재생 (seq = chunk+1) — 음영 seq 12–21", "T14 seq 1-31 offline replay (seq = chunk+1), shaded seq 12-21"), fontsize=10, color=fs.INK, loc="left")
# state strip
rows = [("live", None)] + [(m, l) for l in LOST for m in MS]
for i, key in enumerate(rows):
    y = len(rows) - 1 - i
    for j, s in enumerate(seqs):
        if key[0] == "live":
            f = fid[j]; nm = (f["live_target_nearest"] or [None])[0]
            col = "#ffffff"; txt = {"apple": L("사과", "app"), "crate": "crate", "pear": "pear", "orange": "orng", None: "—"}.get(nm, nm)
            ax2.add_patch(Rectangle((s - 0.5, y - 0.45), 1, 0.9, fc=fs.SURFACE, ec=fs.GRID_INK, lw=0.5))
            ax2.text(s, y, txt, ha="center", va="center", fontsize=5.5, color=fs.INK)
            continue
        r = rep[key][j]; m = r["manipulated"] or {}
        st = m.get("state")
        ax2.add_patch(Rectangle((s - 0.5, y - 0.45), 1, 0.9, fc=ST_COL[st], ec=(fs.INK if r["decision_mode"] == "switch" else fs.SURFACE), lw=(1.4 if r["decision_mode"] == "switch" else 0.5)))
        t = f"{m.get('id')}" + (f"·{m.get('age_frames')}" if st in ("occluded", "lost") else "")
        ax2.text(s, y, t, ha="center", va="center", fontsize=5.5, color=fs.INK)
    lab = L("live 기록 (최근접 물체)", "live record (nearest obj)") if key[0] == "live" else f"min {key[0]:g} · lost {key[1]}"
    ax2.text(0.3, y, lab, ha="right", va="center", fontsize=7.5, color=fs.INK_2)
ax2.set_xlim(-3.2, 31.6); ax2.set_ylim(-0.7, len(rows) - 0.3); ax2.set_yticks([])
ax2.set_xticks(seqs); ax2.set_xlabel("seq")
for st, lab in [("visible", "visible"), ("occluded", "occluded"), ("lost", "lost")]:
    ax2.add_patch(Rectangle((0, 0), 0, 0, fc=ST_COL[st], label=lab))
ax2.add_patch(Rectangle((0, 0), 0, 0, fc="none", ec=fs.INK, lw=1.4, label="switch"))
ax2.legend(fontsize=7.5, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.1, -0.12))
ax2.set_title(L("칸 글자 = manipulated id · age(프레임) — 굵은 테두리 = switch 프레임", "cell = manipulated id · age (frames); bold border = switch frame"), fontsize=9, color=fs.INK, loc="left")
fs.style_axes(fig, [ax, ax2]); ax2.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "t20-t14-replay-state.png", dpi=150)
json.dump(dict(seqs=seqs, live_target_to_apple_mm=live, live_nearest=[f["live_target_nearest"] for f in fid],
               manip_to_apple_mm={f"min{m}_lost{l}": [r["manip_to_apple_mm"] for r in rep[(m, l)]] for m in MS for l in LOST},
               states={f"min{m}_lost{l}": [(r["manipulated"] or {}).get("state") for r in rep[(m, l)]] for m in MS for l in LOST},
               ids={f"min{m}_lost{l}": [(r["manipulated"] or {}).get("id") for r in rep[(m, l)]] for m in MS for l in LOST},
               ages={f"min{m}_lost{l}": [(r["manipulated"] or {}).get("age_frames") for r in rep[(m, l)]] for m in MS for l in LOST},
               modes={f"min{m}_lost{l}": [r["decision_mode"] for r in rep[(m, l)]] for m in MS for l in LOST},
               source="outputs/verify/T20/t14_replay_*.json, t14_fidelity.json"),
          open(FIG / "t20-t14-replay-state.json", "w"), indent=1)
plt.close(fig)

# ------------------------------------------------------------------ 2. scene (top view)
objs = None
for line in open(_ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl')):
    r = json.loads(line)
    if r.get("kind") == "planning" and "object_poses" in r:
        objs = {k: np.asarray(v["pos"] if isinstance(v, dict) else v, float) for k, v in r["object_poses"].items()}; break
fig = plt.figure(figsize=(16, 7.8)); gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1]); axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
def cloud(ax, k, title):
    z = np.load(REC / f"chunk_{k:05d}.npz")
    p = z["attention_cloud"]; sup = z["support_mask"]
    keep = (~sup) & (p[:, 2] > 0.78) & (p[:, 2] < 1.2) & (p[:, 0] > 0.2) & (p[:, 0] < 0.9) & (np.abs(p[:, 1]) < 0.6)
    ax.scatter(p[keep, 1], p[keep, 0], s=0.3, c=fs.GRID_INK, rasterized=True)
    ax.set_title(title, fontsize=9.5, color=fs.INK, loc="left")
    return int(keep.sum())
def traj(ax, key, col, lab, mk):
    rs = [r for r in rep[key] if r["manipulated"]]
    c = np.array([r["manipulated"]["centroid"] for r in rs])
    ax.plot(c[:, 1], c[:, 0], color=col, lw=1.3, marker=mk, ms=4, label=lab, alpha=0.9)
def objects(ax):
    for n, v in objs.items():
        ax.plot(v[1], v[0], marker="+", color=fs.INK, ms=9, mew=1.2)
        ax.text(v[1] + 0.015, v[0] + 0.012, n, fontsize=8, color=fs.INK)
nk = {}
for ax, k, t in [(axes[0], 10, L("seq 11 cloud (사과 보임)", "seq 11 cloud (apple visible)")),
                 (axes[1], 15, L("seq 16 cloud (자기 필터가 사과를 지움)", "seq 16 cloud (apple erased by self-filter)"))]:
    nk[k] = cloud(ax, k, t)
    lc = np.array([f["live_target_centroid"] for f in fid if f["live_target_centroid"]])
    ax.plot(lc[:, 1], lc[:, 0], color=C_LIVE, ls="--", lw=1.0, marker=".", label=L("live 기록 target (옛 코드)", "live target (old code)"))
    traj(ax, (0.0, None), C_DEF, "min_score 0.0", "o")
    traj(ax, (0.05, None), C_MIN, "min_score 0.05 / 0.1 / 0.3", "s")
    objects(ax)
    ax.set_xlim(0.55, -0.55); ax.set_ylim(0.38, 0.66); ax.set_aspect("equal"); ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]")
axes[0].legend(fontsize=7.5, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
# zoom
ax = axes[2]
r5 = rep[(0.05, None)]
cz = np.array([r["manipulated"]["centroid"] for r in r5])
st = [r["manipulated"]["state"] for r in r5]
for i, (c, s_) in enumerate(zip(cz, st)):
    ax.plot(c[1], c[0], marker=("o" if s_ == "visible" else "s"), color=ST_COL[s_], ms=6, mec=fs.INK, mew=0.4)
for i in [0, 10, 25]:
    ax.annotate(f"seq {seqs[i]}" + (L(" (마지막 관측, seq 12–25 occluded 가 이 점)", " (last seen; seq 12-25 occluded here)") if i == 10 else L(" visible · seq 27 occluded 가 이 점", " visible; seq 27 occluded here") if i == 25 else ""), (cz[i, 1], cz[i, 0]), fontsize=7, color=fs.INK_2, xytext=(4, 4), textcoords="offset points")
ax.plot(APPLE[1], APPLE[0], marker="+", color=fs.INK, ms=12, mew=1.5)
ax.text(APPLE[1] + 0.012, APPLE[0] + 0.004, L("사과 참값", "apple truth"), fontsize=8)
ax.add_patch(Circle((APPLE[1], APPLE[0]), 0.04, fill=False, ec=fs.INK_2, ls=":", lw=0.8))
lastc = r5[11]["manipulated"]["centroid"]
ax.add_patch(Circle((lastc[1], lastc[0]), 0.06, fill=False, ec=C_MIN, ls="--", lw=0.9))
ax.text(lastc[1] + 0.058, lastc[0] + 0.05, L("마지막 관측 centroid 주변 60 mm\n(identity tolerance)", "60 mm identity tolerance\naround last seen centroid"), fontsize=7, color=C_MIN)
ax.set_xlim(APPLE[1] + 0.08, APPLE[1] - 0.08); ax.set_ylim(APPLE[0] - 0.08, APPLE[0] + 0.08); ax.set_aspect("equal")
ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]")
ax.set_title(L("사과 주변 확대 · min_score 0.05 (원=visible, 사각=occluded)", "apple zoom, min_score 0.05 (o visible, square occluded)"), fontsize=9.5, color=fs.INK, loc="left")
ax.text(APPLE[1] + 0.075, APPLE[0] - 0.075, L("점선 원 = 사과 반지름 40 mm", "dotted = apple radius 40 mm"), fontsize=7, color=fs.INK_2)
fs.style_axes(fig, np.array(axes))
fig.suptitle(L("T20 · T14 서버 기록 top view — 설정별 manipulated centroid 궤적 (seq 1–31)", "T20, T14 server record top view: manipulated centroid per setting (seq 1-31)"), fontsize=11, color=fs.INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "t20-t14-topview.png", dpi=150)
json.dump(dict(object_poses={k: v.tolist() for k, v in objs.items()}, apple_truth=APPLE.tolist(), crate_truth=CRATE.tolist(),
               live_target_centroid=[f["live_target_centroid"] for f in fid],
               manipulated_centroid={f"min{m}_lostNone": [(r["manipulated"] or {}).get("centroid") for r in rep[(m, None)]] for m in MS},
               cloud_points_drawn=nk, cloud_filter="not support_mask, 0.78<z<1.2, 0.2<x<0.9, |y|<0.6",
               source="chunk_00010.npz, chunk_00015.npz attention_cloud; outputs/verify/T20/t14_replay_*.json"),
          open(FIG / "t20-t14-topview.json", "w"), indent=1)
plt.close(fig)

# ------------------------------------------------------------------ 3. table
fig = plt.figure(figsize=(14, 13.5))
def table(ax, cells, cols, title, widths=None, fs_=7.5):
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="center", colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(fs_); t.scale(1, 1.25)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(fs.GRID_INK)
        c.set_facecolor(fs.SEQ_BLUE[0] if i == 0 else fs.SURFACE)
    ax.set_title(title, fontsize=10, color=fs.INK, loc="left")
ax = fig.add_axes([0.02, 0.80, 0.96, 0.17])
rule = [
    ["first", L("잡은 것이 없음", "nothing held"), L("1등을 새 id 로 채택", "adopt leader, new id"), "visible"],
    ["keep", L("잡은 물체가 1등 (tolerance 60 mm 안)", "held object is the leader (within 60 mm)"), L("centroid 갱신, 도전자 초기화", "track centroid, reset challenger"), "visible"],
    ["hold", L("다른 물체가 1등, 잡은 물체도 cluster 있음", "other leads, held object has a cluster"), L("잡은 물체 유지, 도전자 streak 계산", "keep held, count challenger streak"), "visible"],
    ["occluded", L("잡은 물체의 cluster 없음, age ≤ lost_frames", "held object has no cluster, age <= lost_frames"), L("마지막 관측 기하를 target 으로 (leader 아님)", "last observed geometry as target (not leader)"), "occluded"],
    ["lost", L("age > target_lost_frames", "age > target_lost_frames"), L("target 없음, GroundingStatus.LOST", "no target, GroundingStatus.LOST"), "lost"],
    ["switch", L("도전자가 target_confirm_frames(3) 연속 1등이고 매 프레임 score ≥ target_switch_min_score", "challenger leads 3 frames running, each score >= min_score"), L("도전자를 새 id 로, switched_from = 이전 id", "adopt challenger, new id, switched_from"), "visible"],
    ["(unseen)", L("cluster 가 하나도 없음 (NO_SEED 등)", "no cluster at all (NO_SEED etc.)"), L("age +1, streak 불변, status 그대로", "age +1, streak unchanged, status unchanged"), "occluded / lost"],
]
table(ax, rule, ["mode", L("조건", "condition"), L("동작", "action"), "state"],
      L("전환 규칙 (stages/target_grounding.py TargetConfirm.select / unseen, 코드에서 읽음)", "switch rule (TargetConfirm.select / unseen, read from code)"), [0.08, 0.42, 0.38, 0.12])
ax = fig.add_axes([0.02, 0.49, 0.96, 0.28])
cells = []
for l in LOST:
    for m in MS:
        s = summ[f"min{m}_lost{l}"]
        cells.append([f"{m:g}", str(l), "yes" if s["seq12_21_all_id0"] else "no", f"{s['seq12_21_n_occluded']}/10",
                      ", ".join(f"{a}({o})" for a, _, _, o in s["switches_all_seq"]) or "—",
                      str(s["max_occluded_age_all"]), ", ".join(map(str, s["lost_seqs"])) or "—",
                      f"{s['seq22']['status']} / {s['seq22']['state']} / {s['seq22']['age']}",
                      f"{s['final_id']} ({s['final_to_apple_mm']:.1f} mm)"])
table(ax, cells, ["min_score", "lost_frames", L("seq12–21 id 0", "seq12-21 id 0"), L("seq12–21 occluded", "seq12-21 occluded"),
                  L("switch seq(대상)", "switch seq (to)"), L("occluded 최대 age", "max occluded age"), L("lost seq", "lost seq"),
                  L("seq22 status/state/age", "seq22 status/state/age"), L("seq31 id (→사과)", "seq31 id (to apple)")],
      L("#1 T14 재생 sweep (seq 1–31 을 순서대로; 입력 fidelity: seed 31/31 일치, n_clusters 31/31, best score |Δ| ≤ 3.0e-8)", "#1 T14 replay sweep (seed 31/31 equal, n_clusters 31/31, |d best| <= 3e-8)"),
      [0.07, 0.08, 0.09, 0.1, 0.2, 0.09, 0.11, 0.16, 0.1])
ax = fig.add_axes([0.02, 0.24, 0.96, 0.22])
cs = sorted({round(d["crate_score"], 4) for d in syn["A"]})
cells = []
for seqname in "AB":
    for m in MS:
        row = [L("A 사과 보임" if seqname == "A" else "B 사과 없음", "A apple visible" if seqname == "A" else "B apple absent"), f"{m:g}"]
        for c in cs:
            d = [d for d in syn[seqname] if d["min_score"] == m and d["lost_frames"] is None and round(d["crate_score"], 4) == c][0]
            row.append("—" if d["switch_after_n_attention_frames"] is None else f"{d['switch_after_n_attention_frames']}")
        cells.append(row)
table(ax, cells, [L("시퀀스", "sequence"), "min_score"] + [f"crate {c:.4f}" for c in cs],
      L("#2·#3 합성: 사과 3 프레임 뒤 attention 이 crate 로 10 프레임 — switch 까지 attention 프레임 수 (— = 10 프레임 내 switch 없음; lost 10/20 도 같은 값)", "#2/#3 synthetic: frames of crate attention until switch (— none in 10)"),
      [0.16, 0.08] + [0.12] * len(cs))
ax = fig.add_axes([0.02, 0.02, 0.96, 0.19])
cells = []
for occl in ["crate_low", "no_cluster"]:
    for grp, ms_ in [("0.0", [0.0]), ("0.05/0.1/0.3", [0.05, 0.1, 0.3])]:
        for l in LOST:
            row = [occl, grp, str(l)]
            for K in [5, 15, 25]:
                parts = []
                for off in [0, 30, 80]:
                    ds = [d for d in syn["C"] if d["occluder"] == occl and d["min_score"] in ms_ and d["lost_frames"] == l and d["K"] == K and d["reobserve_offset_mm"] == off]
                    assert len({(d["id_on_return"], d["id_final"], d["state_on_return"]) for d in ds}) == 1
                    d = ds[0]
                    parts.append(f"{d['id_before']}→{d['id_on_return']}→{d['id_final']}")
                row.append("  ".join(parts))
            cells.append(row)
table(ax, cells, [L("가림 중 프레임", "occluded frames"), "min_score", "lost_frames", "K=5 (0/30/80 mm)", "K=15 (0/30/80 mm)", "K=25 (0/30/80 mm)"],
      L("#2(b) 합성: K 프레임 가림 뒤 사과 재관측 (오프셋 0/30/80 mm) — id: 가림 전 → 재관측 첫 프레임 → 3 프레임 뒤", "#2(b) synthetic: id before -> on return -> 3 frames later"),
      [0.12, 0.12, 0.09, 0.22, 0.22, 0.22], fs_=7)
fig.patch.set_facecolor(fs.SURFACE)
fig.savefig(FIG / "t20-switch-rule-table.png", dpi=150)
json.dump(dict(rule=rule, t14_sweep=summ, synth_AB=[{k: v for k, v in d.items() if k in ("min_score", "lost_frames", "crate_level", "crate_score", "switch_frames", "switch_after_n_attention_frames", "final_id", "final_state")} for s in "AB" for d in syn[s]],
               synth_C=[{k: v for k, v in d.items() if k not in ("modes", "ids", "states", "ages")} for d in syn["C"]]),
          open(FIG / "t20-switch-rule-table.json", "w"), indent=1)
print("ok", KO)
