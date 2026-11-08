"""사용자 주의 (1) 진단 — 쥔 후 HOLD 를 만든 거리장 voxel 은 무엇인가 (2026-11-07, G52–G53).

입력: `--record-constraints full` 로 돈 run (server_constraints/run_0000/chunk_*.npz 에 미세 · 거친 ESDF 격자) 과
geom snapshot (`gtto_ls_geomsnap.py` 또는 원격 `panda_geom_snapshot.py`, 같은 `_geomsnap.npz` 형식).

call 마다:
  1. 손 주변 점유 voxel — 미세 계층 `d < 0.5 voxel` 이고 손 · 손가락 구 (step 0 = 지금 자세) 중심에서 `--hand-radius` 안.
     world 로 옮겨 MuJoCo collision geom 중 가장 가까운 것의 body 로 class 를 매긴다 (원격 J1 `classify_points` 그대로:
     target · support · destination · other · robot · ghost(가장 가까운 geom 이 30 mm 밖)).
     robot 은 다시 hand (gripper0_*) / arm 으로 나눈다.
  2. 서버가 hold 로 판정한 대표 충돌 행 (`verdict_reasons[kind=collision].evidence.row.point_m`) 근처 점유 voxel 의 class
     (원격 `_classify_near`, 반경 60 mm + 2 voxel) — 이 HOLD 의 주인.
단계는 sim 진실 (`gtto_contact_phase.truth`): approach · closing · carry · place · after.

산출: `<out>/voxel_origin.json`, `<out>/voxel_origin_표.md`, `<out>/그래프_voxel출처.png`, 그리고 첫 HOLD call 의
`<out>/장면_<stem>_seq<k>.png` (agentview · wrist 실제 프레임 + 손 주변 voxel 3D 산점도, class 색).
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "segmentation"))

from AG3S.trajopt.experiments import panda_fixed_input_ablation as fa  # noqa: E402
import gtto_contact_phase as cp  # noqa: E402

HAND_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
CLASSES = ("target", "hand", "arm", "support", "other", "ghost")
COLORS = {"target": "#d95f02", "hand": "#1b9e77", "arm": "#7570b3", "support": "#999999", "other": "#e7298a",
          "ghost": "#000000"}


def split_robot(label, body):
    if label != "robot":
        return label if label in CLASSES else "other"
    return "hand" if body.startswith("gripper0_") else "arm"


def occupied_near(dist, origin, voxel, centres, radius):
    """미세 계층의 점유 voxel (d < 0.5 voxel) 중 어떤 centre 에서든 `radius` 안 (base 좌표)."""
    lo = np.floor((centres.min(0) - radius - origin) / voxel).astype(int)
    hi = np.ceil((centres.max(0) + radius - origin) / voxel).astype(int) + 1
    lo, hi = np.maximum(lo, 0), np.minimum(hi, np.asarray(dist.shape))
    if np.any(hi <= lo):
        return np.zeros((0, 3))
    sub = np.asarray(dist[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32)
    ijk = np.argwhere(sub < 0.5 * voxel) + lo
    if not len(ijk):
        return np.zeros((0, 3))
    pts = origin + ijk * voxel
    dmin = np.min(np.linalg.norm(pts[:, None, :] - centres[None], axis=2), axis=1)
    return pts[dmin <= radius]


def classify(world, snap, target_bodies):
    if not len(world):
        return []
    res = fa.classify_points(world, snap, target_body=target_bodies, destination_body=None)
    return [split_robot(c, b) if c != "other" else "other:" + b for c, b in res["geom_of_point"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=pathlib.Path)
    ap.add_argument("--snap-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--hand-radius", type=float, default=0.10)
    ap.add_argument("--max-voxels", type=int, default=4000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    run = a.run_dir
    stems = sorted(p.name[:-len("_calls.jsonl")] for p in (run / "episodes").glob("*_calls.jsonl"))
    near = collections.Counter()      # (phase, hold?, class) voxel counts near the hand
    near_calls = collections.Counter()
    verdict = collections.Counter()   # (phase, why, link kind, nearest voxel class)
    verdict_ep = collections.Counter()
    per_call = []
    scenes = []
    for stem in stems:
        calls = fa.load_episode_calls(run / "episodes", stem)
        info = fa.attach_chunks(calls, run / "server_constraints" / "run_0000")
        snap_path = a.snap_dir / (stem + "_geomsnap.npz")
        if not snap_path.is_file():   # 2026-11-07 G70: only the episodes snapshotted
            continue
        tname = cp.target_name([c.row for c in calls])
        tr = cp.truth(str(run / "episodes" / (stem + ".npz")), tname) if tname else None
        first_hold_scene, first_hold_ph = None, None
        for c in calls:
            if c.chunk_path is None:
                continue
            snap = fa.load_geom_snapshot(snap_path, c.obs_index)
            if snap is None:
                continue
            tb = [str(x) for x in snap.get("target_bodies", [])] if snap.get("target_bodies") is not None else []
            z = np.load(c.chunk_path, allow_pickle=False)
            summ = json.loads(str(z["summary_json"]))
            names = summ.get("sphere_link_names") or []
            centres = np.asarray(z["sphere_centres"], np.float64)
            hand = np.array([n in HAND_LINKS for n in names[:len(centres)]])
            if not hand.any():
                continue
            dist, origin, voxel = z["esdf_distance"], np.asarray(z["esdf_origin"]), float(z["esdf_voxel_size"])
            vox = occupied_near(dist, origin, voxel, centres[hand], a.hand_radius)
            if len(vox) > a.max_voxels:
                vox = vox[np.random.default_rng(0).choice(len(vox), a.max_voxels, replace=False)]
            T = np.asarray(c.T_world_base, np.float64)
            world = (T[:3, :3] @ vox.T).T + T[:3, 3]
            full = classify(world, snap, tb)
            labels = [x.split(":")[0] for x in full]
            ph = str(tr["phase"][min(c.obs_index, len(tr["phase"]) - 1)]) if tr else c.phase
            hold = bool(c.row.get("hold"))
            cnt = collections.Counter(labels)
            for k, v in cnt.items():
                near[(ph, hold, k)] += v
            near_calls[(ph, hold)] += 1
            vrows = fa.verdict_collision_rows(c.row)
            vout = []
            for vr in vrows:
                pt = vr.get("point_m")
                if pt is None:
                    continue
                vv = occupied_near(dist, origin, voxel, np.asarray(pt, np.float64)[None], fa.VERDICT_POINT_RADIUS_M + 2 * voxel)
                if len(vv):
                    dn = np.linalg.norm(vv - np.asarray(pt), axis=1)
                    order = np.argsort(dn)[:20]
                    lab = classify((T[:3, :3] @ vv[order].T).T + T[:3, 3], snap, tb)
                    nearest = lab[0]
                    share = collections.Counter(lab)
                    if nearest.startswith("other:"):
                        nearest = nearest[len("other:"):]
                else:
                    nearest, share = "none", collections.Counter()
                kind = "attached" if vr.get("is_attached") else ("hand" if (vr.get("link") or "") in HAND_LINKS else "arm")
                verdict[(ph, vr.get("why"), kind, nearest)] += 1
                verdict_ep[(stem, ph, vr.get("why"), kind, nearest)] += 1
                vout.append({"why": vr.get("why"), "link": vr.get("link"), "step": vr.get("step"),
                             "clearance_mm": vr.get("clearance_mm"), "nearest_voxel_class": nearest,
                             "voxel_class_share_20": dict(share)})
            per_call.append({"stem": stem, "seq": c.seq, "obs_index": c.obs_index, "phase": ph, "latch": c.phase,
                             "hold": hold, "near_hand_voxels": dict(cnt), "verdict": vout})
            if hold and ph in ("carry", "closing", "place") and (
                    first_hold_scene is None or (ph == "carry" and first_hold_ph != "carry")):
                first_hold_scene, first_hold_ph = (c, world, labels, T, centres, hand, z), ph
        if first_hold_scene is not None:
            scenes.append(_scene(a.out, run, stem, *first_hold_scene))
        print(json.dumps({"stem": stem, "chunks": info["matched"], "calls_without_chunk": info["calls_without_chunk"]}), flush=True)
    js = {"run": str(run), "snap_dir": str(a.snap_dir), "hand_radius_m": a.hand_radius, "classes": CLASSES,
          "near_hand": {"|".join(map(str, k)): v for k, v in sorted(near.items())},
          "near_hand_calls": {"|".join(map(str, k)): v for k, v in sorted(near_calls.items())},
          "verdict_rows": {"|".join(map(str, k)): v for k, v in sorted(verdict.items())},
          "verdict_rows_by_episode": {"|".join(map(str, k)): v for k, v in sorted(verdict_ep.items())},
          "scenes": scenes, "calls": per_call}
    (a.out / "voxel_origin.json").write_text(json.dumps(js, indent=1, default=str) + "\n")
    _table(a.out, near, near_calls, verdict)
    _figure(a.out, near, near_calls, verdict)


def _table(out, near, near_calls, verdict):
    phases = [p for p in ("approach", "closing", "carry", "place", "after") if any(k[0] == p for k in near_calls)]
    L = ["# 손 주변 점유 voxel 의 출처 (call 당 평균 voxel 수, MuJoCo geom class)", "",
         "| 단계 | HOLD | call | " + " | ".join(CLASSES) + " |", "|---|---|---|" + "---|" * len(CLASSES)]
    for p in phases:
        for h in (False, True):
            n = near_calls.get((p, h), 0)
            if not n:
                continue
            L.append(f"| {p} | {'예' if h else '아니오'} | {n} | " + " | ".join(f"{near.get((p, h, k), 0) / n:.0f}" for k in CLASSES) + " |")
    L += ["", "# HOLD 대표 충돌 행 → 그 점에서 가장 가까운 점유 voxel 의 class", "",
          "| 단계 | why | 행 | 가장 가까운 voxel | 행 수 |", "|---|---|---|---|---|"]
    for k, v in sorted(verdict.items(), key=lambda kv: -kv[1]):
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {v} |")
    (out / "voxel_origin_표.md").write_text("\n".join(L) + "\n")


def _figure(out, near, near_calls, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phases = [p for p in ("approach", "closing", "carry", "place", "after") if any(k[0] == p for k in near_calls)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    xs, labels = [], []
    for p in phases:
        for h in (False, True):
            if near_calls.get((p, h)):
                xs.append((p, h))
                labels.append(f"{p}\n{'HOLD' if h else 'run'} (n={near_calls[(p, h)]})")
    bottom = np.zeros(len(xs))
    for k in CLASSES:
        vals = np.array([near.get((p, h, k), 0) / near_calls[(p, h)] for p, h in xs])
        ax.bar(range(len(xs)), vals, bottom=bottom, color=COLORS[k], label=k, width=0.7)
        bottom += vals
    ax.set_xticks(range(len(xs)), labels, fontsize=8)
    ax.set_ylabel("occupied voxels <10 cm of hand (mean/call)")
    ax.set_title("Owner of field voxels around the hand", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    keys = sorted(verdict.items(), key=lambda kv: -kv[1])[:10]
    ax.barh(range(len(keys)), [v for _, v in keys], color=[COLORS.get(k[3], COLORS["other"]) for k, _ in keys])
    ax.set_yticks(range(len(keys)), [f"{k[0]} · {k[1]} · {k[2]} → {k[3]}" for k, _ in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("HOLD rows")
    ax.set_title("HOLD collision row -> owner of the nearest voxel")
    fig.tight_layout()
    fig.savefig(out / "그래프_voxel출처.png", dpi=130)
    plt.close(fig)


def _scene(out, run, stem, c, world, labels, T, centres, hand, z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.load(run / "episodes" / (stem + ".npz"))
    imgs = [k for k in ep.files if k.startswith("obs_") and k.endswith("image") or k in ("obs_agentview_image", "obs_robot0_eye_in_hand_image")]
    fig = plt.figure(figsize=(15, 5))
    for i, k in enumerate(sorted(imgs)[:2]):
        ax = fig.add_subplot(1, 3, i + 1)
        im = np.asarray(ep[k][c.obs_index])
        ax.imshow(im[::-1, ::-1] if "agentview" in k or "eye" in k else im)
        ax.set_title(f"{k} · obs {c.obs_index}", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    lab = np.array(labels)
    for k in CLASSES:
        m = lab == k
        if m.any():
            ax.scatter(*world[m].T, s=3, c=COLORS[k], label=f"{k} ({int(m.sum())})")
    hc = (T[:3, :3] @ centres[hand].T).T + T[:3, 3]
    ax.scatter(*hc.T, s=40, marker="x", c="red", label="hand/finger sphere centres")
    if "attached_query_centres" in z.files:
        ac = np.asarray(z["attached_query_centres"])[0]
        ac = (T[:3, :3] @ ac.T).T + T[:3, 3]
        ax.scatter(*ac.T, s=60, marker="o", facecolors="none", edgecolors="blue", label="held-object query spheres")
    ax.set_title(f"{stem} seq {c.seq}: occupied voxels within 10 cm of the hand", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    name = f"장면_{stem}_seq{c.seq}.png"
    fig.savefig(out / name, dpi=120)
    plt.close(fig)
    return name


if __name__ == "__main__":
    main()
