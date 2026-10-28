"""T64 Part B — 6-phase HSR vs the same 6 phases by a serving-signal rule, against a physics truth, on the T55 test
re-simulations of the T53 rollouts (original prompt, 16 episodes) — `handoff/T64.task.md`.

Truth (MuJoCo, target fruit; thresholds 2 cm / 5 cm fixed before results):
    approach   before the first finger contact on the target; also after an empty grasp (contact ended, never held)
    grasp      finger contact (or `grasping`) until held ∧ rise ≥ 2 cm
    lift       held ∧ rise ≥ 2 cm ∧ horizontal travel < 5 cm
    transport  held ∧ horizontal travel ≥ 5 cm ∧ target not over the crate footprint
    placement  target over the crate footprint (held or released) until released ∧ hand–target > 5 cm
    retreat    after that
    failure    no longer held after having been held, target neither over the crate nor in it (excluded from F1)
Rule (serving signals only): latch state of the re-run grounding (`t55_perception` ``latch_reground__phase``),
hand position (grasp parent link), crate footprint; same 2 cm / 5 cm thresholds on the hand.
HSR: `t64_phase6` 6-phase probe (Part A, ``probe_6.npz``) on the T55 ``kv_l4`` feature, raw argmax per frame.

usage: t64_partB.py PARTA_DIR OUT_DIR [--video]      (.venv-ag3s; --video needs cv2)
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

T55 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c/verify/T55")
PHASES = ("approach", "grasp", "lift", "transport", "placement", "retreat")
FAIL = 6
RISE, TRAVEL, AWAY = 0.02, 0.05, 0.05
CRATE_Q = 31
B, SEED = 10000, 6401


def crate_local_xy(p, q):
    from AG3S.ag3s.experiments.studies.t60_termination import quat_to_R

    c, R = np.asarray(q[CRATE_Q:CRATE_Q + 3], float), quat_to_R(q[CRATE_Q + 3:CRATE_Q + 7])
    return ((np.asarray(p, float) - c) @ R)[..., :2]


def over_crate(p, q) -> bool:
    from AG3S.ag3s.experiments.studies.t60_termination import CRATE_HALF_XY

    xy = crate_local_xy(p, q)
    return bool(abs(xy[0]) < CRATE_HALF_XY[0] and abs(xy[1]) < CRATE_HALF_XY[1])


def truth_phases(z, target: str) -> np.ndarray:
    fruits = [str(f) for f in z["fruits"]]
    j = fruits.index(target)
    st = np.asarray(z["phys_state"])[:, j]
    contact = np.asarray(z["finger_contact"])[:, j].sum(1) > 0
    fp, hand, qpos = np.asarray(z["fruit_pos"], float)[:, j], np.asarray(z["parent_pos"], float), np.asarray(z["qpos"], float)
    z0, xy0 = fp[0, 2], fp[0, :2]
    out = np.zeros(len(st), int)
    ever_contact = ever_held = False
    released_far = False
    for t in range(len(st)):
        held = str(st[t]) == "held"
        ever_contact |= bool(contact[t]) or str(st[t]) == "grasping"
        ever_held |= held
        rise, travel = fp[t, 2] - z0, float(np.linalg.norm(fp[t, :2] - xy0))
        oc = over_crate(fp[t], qpos[t]) or str(st[t]) == "in_crate"
        if not ever_contact:
            out[t] = 0
        elif oc and ever_held:
            released_far |= (not held) and float(np.linalg.norm(hand[t] - fp[t])) > AWAY
            out[t] = 5 if released_far else 4
        elif held:
            out[t] = 1 if rise < RISE else (2 if travel < TRAVEL else 3)
        elif ever_held:
            out[t] = FAIL
        else:  # touched, never held: grasping while in contact, back to approach after it ends
            out[t] = 1 if (contact[t] or str(st[t]) == "grasping") else 0
    return out


def rule_phases(latch: np.ndarray, hand: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    out = np.zeros(len(latch), int)
    a_xyz = r_xyz = None
    for t, L in enumerate(latch):
        L = str(L)
        if L in ("searching", "latched", ""):
            out[t], a_xyz, r_xyz = 0, None, None
        elif L == "closing":
            out[t] = 1
        elif L == "held":
            if a_xyz is None:
                a_xyz = hand[t].copy()
            rise, travel = hand[t, 2] - a_xyz[2], float(np.linalg.norm(hand[t, :2] - a_xyz[:2]))
            out[t] = 4 if over_crate(hand[t], qpos[t]) else (1 if rise < RISE else (2 if travel < TRAVEL else 3))
        elif L == "placed":
            if r_xyz is None:
                r_xyz = hand[t].copy()
            out[t] = 5 if float(np.linalg.norm(hand[t] - r_xyz)) > AWAY else 4
        else:
            out[t] = 0
    return out


def macro_f1(y, p, labels):
    f = []
    for c in labels:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


def transitions(y, p, a, b):
    """Lead (> 0: predictor showed b before the truth did, continuously) or lag (< 0) in frames, per a→b."""
    out = []
    for i in range(1, len(y)):
        if not (y[i - 1] == a and y[i] == b):
            continue
        j = i
        while j > 0 and p[j - 1] == b and y[j - 1] == a:
            j -= 1
        if j < i:
            out.append(i - j)
        else:
            k = next((k for k in range(i, len(y)) if p[k] == b), None)
            out.append(None if k is None else -(k - i))
    return out


def main(partA: pathlib.Path, out: pathlib.Path, video: bool) -> None:
    from AG3S.ag3s.experiments.studies import t55_study as S
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    out.mkdir(parents=True, exist_ok=True)
    pz = np.load(partA / "probe_6.npz", allow_pickle=True)
    probe = LogReg(mean=pz["mean"], scale=pz["scale"], coef=pz["coef"], intercept=pz["intercept"], C=float(pz["C"]),
                   n_iter=0, converged=True)
    traces = [t for t in S.load_resim_traces(T55 / "resim/test", T55 / "feat/test", "test",
                                             perception_dir=T55 / "perception/test/t55-legacy")
              if t.condition == "natural"]
    cache = S.FeatureCache()
    eps = {}
    for tr in traces:
        z = np.load(tr.meta["resim"], allow_pickle=False)
        q = np.load(tr.meta["perception"], allow_pickle=False)
        y = truth_phases(z, tr.noun)
        h = np.argmax(probe.proba(np.asarray(cache.get(tr)["kv_l4"], np.float64)), 1)
        r = rule_phases(np.asarray(q["latch_reground__phase"]), np.asarray(z["parent_pos"], float),
                        np.asarray(z["qpos"], float))
        keep = y != FAIL
        labels = sorted(set(y[keep].tolist()))
        eps[tr.key] = {"truth": y.tolist(), "hsr": h.tolist(), "rule": r.tolist(), "labels": labels,
                       "f1_hsr": macro_f1(y[keep], h[keep], labels), "f1_rule": macro_f1(y[keep], r[keep], labels),
                       "acc_hsr": float(np.mean(h[keep] == y[keep])), "acc_rule": float(np.mean(r[keep] == y[keep])),
                       "n_failure": int((~keep).sum()),
                       "lead": {f"{a}->{b}": {"hsr": transitions(y, h, a, b), "rule": transitions(y, r, a, b)}
                                for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))}}
    keys = sorted(eps)
    d = np.array([eps[k]["f1_hsr"] - eps[k]["f1_rule"] for k in keys])
    rng = np.random.default_rng(SEED)
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    lead = {}
    for bnd in ("0->1", "1->2", "2->3", "3->4", "4->5"):
        for m in ("hsr", "rule"):
            v = [x for k in keys for x in eps[k]["lead"][bnd][m] if x is not None]
            lead.setdefault(bnd, {})[m] = {"n": len(v), "median_lead_frames": float(np.median(v)) if v else None,
                                           "missed": sum(x is None for k in keys for x in eps[k]["lead"][bnd][m])}
    # boundary F1 (the two classes of a boundary, pooled frames)
    res = {"definitions": __doc__, "episodes": eps, "n_episodes": len(keys),
           "macro_f1": {"hsr_mean": float(np.mean([eps[k]["f1_hsr"] for k in keys])),
                        "rule_mean": float(np.mean([eps[k]["f1_rule"] for k in keys])),
                        "diff_mean": float(d.mean()), "diff_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "n_up_down": [int((d > 0).sum()), int((d < 0).sum())]},
           "lead": lead}
    yy = np.concatenate([np.array(eps[k]["truth"]) for k in keys]); hh = np.concatenate([np.array(eps[k]["hsr"]) for k in keys])
    rr = np.concatenate([np.array(eps[k]["rule"]) for k in keys])
    res["per_class_f1"] = {PHASES[c]: {"hsr": macro_f1(yy[yy != FAIL], hh[yy != FAIL], [c]),
                                       "rule": macro_f1(yy[yy != FAIL], rr[yy != FAIL], [c]),
                                       "n": int((yy == c).sum())} for c in range(6)}
    res["n_failure_frames"] = int((yy == FAIL).sum())
    (out / "partB.json").write_text(json.dumps(res, indent=1, default=float))
    print({k: res[k] for k in ("macro_f1", "n_failure_frames")})
    print({b: v for b, v in lead.items()})
    if video:
        write_videos(traces, eps, out / "videos")


def write_videos(traces, eps, vd: pathlib.Path) -> None:
    import cv2

    vd.mkdir(parents=True, exist_ok=True)
    names = PHASES + ("failure",)
    for tr in traces:
        e = eps[tr.key]
        z = np.load(tr.meta["resim"], allow_pickle=False)
        hi, wl, wr = (np.asarray(z[f"image_{c}"]) for c in ("cam_high", "cam_left_wrist", "cam_right_wrist"))
        path = vd / f"{tr.key}_phases.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 3, (3 * 336, 336 + 96))
        for t in range(len(e["truth"])):
            row = np.concatenate([cv2.resize(x[t], (336, 336)) for x in (hi, wl, wr)], 1)
            canvas = np.full((336 + 96, 3 * 336, 3), 255, np.uint8)
            canvas[:336] = row
            img = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            for i, (lab, key) in enumerate((("truth (physics)", "truth"), ("HSR 6-phase", "hsr"), ("rule 6-phase", "rule"))):
                ph = names[e[key][t]]
                ok = key == "truth" or e[key][t] == e["truth"][t]
                cv2.putText(img, f"{lab:16s}: {ph}", (10, 336 + 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 0) if ok else (0, 0, 200), 2)
            cv2.putText(img, f"{tr.key} frame {t}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vw.write(img)
        vw.release()
    print("videos ->", vd)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), "--video" in sys.argv)
