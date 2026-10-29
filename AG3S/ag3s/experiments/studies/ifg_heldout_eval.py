"""IFG held-out — run the LOCKED dev code on the held-out dump (no locked file is modified).

Adapter only:
  * `ifg_reader.load_frame` is replaced in this process by `load_heldout_frame`, which reads the collector's held-out dump
    (`inputs.npz`, `derived.npz` = noun_target_L10H7 / serving_L8H2_last_d0 / hsr_p, `prefix_L8.npz` = resid_out_image).
  * `ifg_memory.FEATS` is narrowed to ("prefix_L8",): the held-out dump has no projector tokens by design (request §5 —
    only what the locked arms read; the adopted arm and the gate use prefix_L8).
  * a pseudo run dir `<out>/run` links `s0/replay`, `s0/labels`, `rounds/R6/robot_mask` to `ifg_heldout` outputs, so the locked
    per-frame / per-episode functions read the same file names as on dev.

Stages:
  ns      `ifg_round_ns.frame` -> rounds/R5 (FK sphere mask) and rounds/R6 (render mask) rows: G:N, top1-only-ns(+rm):N, ...
  memory  `ifg_memory.episode(center=True)` -> rounds/R10b rows (R6 = arm A, Mem-first:prefix_L8:d0.05 = arm B) and
          rounds/R11b/conf.jsonl (q_mem_chain:prefix_L8 = locked gate score)

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_heldout_eval --dump <collector>/heldout/dump \
      --prep <analysis run>/heldout_eval --out <analysis run>/heldout_locked --stage ns|memory --workers 2
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_reader as R

SUB = {"clear": "clear", "adjacent": "obst_adjacent", "place": "obst_place"}


def load_heldout_frame(ref, *, features=R.FEATURES):
    p = ref.path
    with np.load(p / "inputs.npz") as z:
        order = [str(c) for c in z["image_order"]]
        if tuple(order) != R.DUMP_CAMERAS:
            raise ValueError(f"{ref.frame_id}: image_order {order}")
        # held-out inputs.npz stores the recorded uint8 images under the record camera names, in image_order
        imgs = np.stack([np.asarray(z[f"image_{c}"]) for c in R.RECORD_CAMERAS]).astype(np.float32)
        if imgs.max() > 1.5:  # uint8 as recorded -> [-1, 1]
            imgs = imgs / 127.5 - 1.0
        raw_state = np.asarray(z["raw_state"], np.float64)
        prompt = str(z["prompt"])
        qrows = dict(zip([str(s) for s in z["query_row_names"]], [int(r) for r in z["query_rows"]])) if "query_rows" in z.files else {}
    feats = {}
    if "prefix_L8" in features:
        with np.load(p / "prefix_L8.npz") as z:
            feats["prefix_L8"] = np.asarray(z["resid_out_image"], np.float32)
    missing = [f for f in features if f not in feats]
    if missing:
        raise KeyError(f"{ref.frame_id}: held-out dump has no {missing}")
    with np.load(p / "derived.npz") as z:
        a = z["serving_L8H2_last_d0"].astype(np.float64).reshape(-1)
        n = z["noun_target_L10H7"].astype(np.float64).reshape(-1)
    maps = {"A": a, "N": n, "P": R.normalize_max(a) * R.normalize_max(n)}
    return R.FrameInputs(ref, feats, maps, imgs, raw_state, prompt, qrows)


def install():
    R.load_frame = load_heldout_frame
    from AG3S.ag3s.experiments.studies import ifg_memory as M
    M.FEATS = ("prefix_L8",)


def heldout_refs(dump: pathlib.Path, prep: pathlib.Path):
    idx = {json.loads(l)["frame_id"]: json.loads(l) for l in open(prep / "index.jsonl")}
    bad = {json.loads(l)["frame_id"] for l in open(prep / "s0.jsonl") if not json.loads(l)["valid"]}
    refs, skipped = [], {"no_prep": 0, "s0_invalid": 0}
    for stratum, sub in SUB.items():
        for rd in sorted((dump / sub).glob("C1a_ep*")):
            ep = int(rd.name.split("_ep")[1].split("_")[0])
            for sd in sorted(rd.glob("step_*")):
                dj = sd / "done.json"
                if not dj.exists():
                    continue
                done = json.loads(dj.read_text())
                k = int(done["frame"])
                fid = f"HO_{stratum}_ep{ep}_f{k:03d}"
                if fid not in idx:
                    skipped["no_prep"] += 1; continue
                if fid in bad:
                    skipped["s0_invalid"] += 1; continue
                it = idx[fid]
                refs.append(R.FrameRef("HO", f"{stratum}_ep{ep}", k, it["grasp_state"], "success" if it["success"] else "failure", sd,
                                       {**done, "grasp_state": it["grasp_state"], "stratum": stratum}))
    return refs, skipped


def pseudo_run(prep: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    run = out / "run"
    for link, target in (("s0/replay", prep / "replay"), ("s0/labels", prep / "labels"), ("rounds/R6/robot_mask", prep / "robot_mask_render")):
        p = run / link
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            os.symlink(os.path.relpath(target, p.parent), p)
    return run


def _ns(args):
    install()
    from AG3S.ag3s.experiments.studies import ifg_round_ns as NS
    ref, run = args
    NS.frame((ref, run, "R5", None))
    NS.frame((ref, run, "R6", run / "rounds/R6/robot_mask"))


def _mem(args):
    install()
    from AG3S.ag3s.experiments.studies import ifg_memory as M
    return M.episode(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=pathlib.Path, required=True); ap.add_argument("--prep", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--stage", choices=["ns", "memory"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    install()
    run = pseudo_run(a.prep, a.out)
    refs, skipped = heldout_refs(a.dump, a.prep)
    (a.out / "frames.json").write_text(json.dumps(dict(n=len(refs), skipped=skipped,
                                                       frames=[dict(frame_id=r.frame_id, episode_id=r.episode_id, kind=r.kind,
                                                                    grasp_state=r.moment, stratum=r.done["stratum"]) for r in refs]), indent=0))
    print(len(refs), "frames", skipped, flush=True)
    if a.stage == "ns":
        for rnd in ("R5", "R6"):
            (run / "rounds" / rnd / "rows").mkdir(parents=True, exist_ok=True)
        todo = [r for r in refs if not (run / "rounds/R6/rows" / f"{r.frame_id}.jsonl").exists()]
        with mp.get_context("spawn").Pool(min(a.workers, 4)) as p:
            for i, _ in enumerate(p.imap_unordered(_ns, [(r, run) for r in todo], chunksize=4)):
                if i % 200 == 0:
                    print("ns", i, "/", len(todo), flush=True)
        return
    (run / "rounds/R10b/rows").mkdir(parents=True, exist_ok=True); (run / "rounds/R11b").mkdir(parents=True, exist_ok=True)
    eps = {}
    for r in refs:
        eps.setdefault(r.episode_id, []).append(r)
    allconf = []
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as p:
        for rows, conf in p.imap_unordered(_mem, [(v, run, True) for v in eps.values()]):
            for fid, rs in rows.items():
                (run / "rounds/R10b/rows" / f"{fid}.jsonl").write_text("".join(json.dumps(x, default=float) + "\n" for x in rs))
            allconf += conf
            print("memory episode done", len(allconf), flush=True)
    with open(run / "rounds/R11b/conf.jsonl", "w") as f:
        for c in allconf:
            f.write(json.dumps(c, default=float) + "\n")
    print("done", len(allconf))


if __name__ == "__main__":
    main()
