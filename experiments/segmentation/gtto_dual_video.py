#!/usr/bin/env python3
"""Rule A (user 2026-11-08): real-scene video of every experiment, with the WRIST camera. For each episode of a run, an mp4 with
agentview | wrist side by side (both from the episode npz, all env steps incl. the 10 LIBERO wait steps), and a caption per
frame from the call record that covers it: obs index, grasp latch state, what the TO did (pass-through / changed / HOLD +
reason). 10 fps (the client's video rate).
usage: gtto_dual_video.py --run <glob under results/benchmark> --out DIR [--stems a,b] [--prefix NAME]"""
import argparse
import json
import pathlib

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")


def call_caption(r):
    if r is None:
        return "no call"
    pc = (r.get("to") or {}).get("passthrough_clear") or {}
    ob = r.get("observe") or {}
    verdict = str(r.get("verdict") or r.get("status") or "")
    latch = (r.get("grasp") or {}).get("state") or "-"
    if ob:
        what = ("would HOLD" if ob.get("would_be_hold") else
                f"would change {float(ob.get('would_be_max_abs_action_change') or 0):.2f}" if float(
                    ob.get("would_be_max_abs_action_change") or 0) > 0 else "pass")
        what = "[observe] " + what
    else:
        held = any((v.get("action") == "hold") for v in (r.get("verdict_reasons") or []))
        what = "HOLD" if held else ("pass-through" if pc.get("applied") else "TO changed")
    why = pc.get("reason") or "-"
    if why == "obstacle":
        w = ((pc.get("obstacle_rows") or {}).get("worst") or {})
        why += f" {w.get('link')} {1000 * float(w.get('clearance_m') or 0):.1f}mm"
    return f"obs {r['obs_index']} | {latch} | {what} | {why} {verdict[:20]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--stems", default="")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    runs = [p for p in sorted(RESULTS.glob(a.run)) if (p / "episodes.jsonl").is_file()]
    if not runs:
        raise SystemExit(f"no run {a.run}")
    run = runs[-1]
    a.out.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    want = {s for s in a.stems.split(",") if s}
    index = []
    for ep in eps:
        npz = sorted((run / "episodes").glob(f"*_t{ep['task_id']}_e{ep['episode_idx']}.npz"))
        if not npz:
            continue
        stem = npz[0].stem
        if want and stem not in want:
            continue
        z = np.load(npz[0], allow_pickle=True)
        av, wr = z["obs_agentview_image"], z["obs_robot0_eye_in_hand_image"]
        cf = run / "episodes" / f"{stem}_calls.jsonl"
        calls = [json.loads(l) for l in cf.read_text().splitlines() if l.strip()] if cf.is_file() else []
        calls.sort(key=lambda r: r["obs_index"])
        ok = bool(ep.get("official_success"))
        sl = ep.get("safelibero") or {}
        coll = sl.get("collide_t") if isinstance(sl, dict) else None
        tag = "success" if ok else "failure"
        if coll is not None:
            tag += f"_collide{coll}"
        name = f"{a.prefix}{stem}_{tag}_agent+wrist.mp4"
        frames, j = [], -1
        for t in range(len(av)):
            while j + 1 < len(calls) and calls[j + 1]["obs_index"] <= t:
                j += 1
            img = np.concatenate([av[t][::-1, ::-1], wr[t][::-1, ::-1]], axis=1)
            im = Image.fromarray(np.ascontiguousarray(img)).resize((1024, 512), Image.NEAREST)
            canvas = Image.new("RGB", (1024, 548), (0, 0, 0))
            canvas.paste(im, (0, 36))
            d = ImageDraw.Draw(canvas)
            d.text((6, 2), f"{run.name[16:70]}  {stem}  {'SUCCESS' if ok else 'FAIL'}"
                   + (f"  collide@{coll}" if coll is not None else ""), fill=(255, 255, 255))
            d.text((6, 18), f"t {t}  " + call_caption(calls[j] if j >= 0 else None), fill=(255, 220, 0))
            d.text((6, 40), "agentview", fill=(255, 255, 255))
            d.text((518, 40), "wrist", fill=(255, 255, 255))
            frames.append(np.asarray(canvas))
        imageio.mimwrite(str(a.out / name), frames, fps=10, macro_block_size=1)
        index.append({"run": run.name, "stem": stem, "video": name, "success": ok, "collide_t": coll, "frames": len(frames)})
        print(a.out / name)
    (a.out / f"{a.prefix}index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
