"""T55 — kv_L4 and noun-token features of re-simulated frames, instruction-changed prompts and T48a val C1 (GPU).

`AG3S/ag3s/docs/handoff/T55.task.md` 구현 3. The producers are T53's, unchanged: `t48_hsr_features.
FeatureExtractor` (kv_L4, batch 1) and `t53_extract.NounTokenExtractor` through `t53_extract.feature_frame`
(noun-token V cache per layer + final hidden at the query rows; Q = target, basket, last_text, where
"target" is the **noun of the prompt actually fed** — for an instruction-changed prompt that is the new
fruit). One model load per process (`t53_extract._feat_session`).

Jobs
----
* ``resim`` — every ``<key>__<tag>.npz`` of ``$AG3S_RUN_DIR/verify/T55/resim/<set>/`` (`t55_resim`): the
  re-simulated images / state of that run with its own prompt (``orig``). For ``E0`` additionally one
  **instruction-changed** prompt per non-target fruit of the episode (`randomized_episodes.jsonl`
  ``non_target_fruits``): the prompt's fruit noun replaced (`t48_hsr_core.prompt_template` / ``fill`` — the
  T48a C1 method), same image and state (``instr_<fruit>``).
* ``val`` (dev only) — T48a val episodes (`t48_hsr_io.VAL_EPISODES`, SUBTASK-b frames: stride 6, phase
  class >= 0, frames inside the video): C0 (the dataset prompt, ``orig``) and C1 (``instr_<distractor>``).
  These frames have no physics; their answer is the **phase label** (`t48_hsr_core.phase_to_class`: place
  = holding the instructed target for ``orig``, never for ``instr_*``), kept apart as ``truth_kind =
  phase_label`` by `t55_study`. Video decoding needs ``av`` (`.venv-openpi-live`).

Outputs: ``$AG3S_RUN_DIR/verify/T55/feat/<set>/<key>__<tag>__<prompt_tag>.npz`` (``val``:
``feat/dev/val/ep_<N>__<prompt_tag>.npz``) — `t53_extract.assemble_feat` arrays + ``t55_stamp`` (source md5,
tag, prompt tag, noun). Finished files with the same stamp are skipped; a different stamp is refused
(``--force`` redoes).

GPU stage ``extract`` refuses unless the shared lock ``/tmp/ag3s-vla-gpu-validation.lock`` is held by the
caller's ``flock`` wrapper (non-blocking probe, `t53_extract.lock_is_held`) and nvidia-smi lists no compute
process (`t51b_noun_extract.gpu_guard`). Stage ``plan`` / ``check`` are CPU and import no jax::

    source scripts/setup/env.sh; export AG3S_RUN_DIR=<the T55 run>; M=AG3S.ag3s.experiments.studies.t55_features
    .venv-ag3s/bin/python -m $M --stage plan --set dev --val
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage extract --set dev --val
    .venv-ag3s/bin/python -m $M --stage check --set test --t53-run-dir <T53 run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np

STEP = "T55"
GPU_LOCK = "/tmp/ag3s-vla-gpu-validation.lock"
STAGES = ("plan", "extract", "check")
SETS = ("dev", "test")
T55_FEAT_VERSION = "t55-feat-v1"
N_LAYERS = 18
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


# ------------------------------------------------------------------------------ jobs (CPU)


def instruction_prompts(prompt: str, target: str, others: Sequence[str]) -> list[tuple[str, str, str]]:
    """``[(prompt_tag, prompt, noun)]``: the original, then one per other fruit (sorted) with the noun swapped."""
    from AG3S.ag3s.experiments.studies.t48_hsr_core import fill, prompt_template

    template, fruit = prompt_template(prompt)
    if fruit != target:
        raise ValueError(f"prompt fruit {fruit!r} != target {target!r}: {prompt!r}")
    out = [("orig", prompt, target)]
    for f in sorted(set(others)):
        if f == target:
            continue
        out.append((f"instr_{f}", fill(template, f), f))
    return out


def _npz_scalar(path: pathlib.Path, names: Sequence[str]) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {n: (z[n].item() if z[n].ndim == 0 else z[n]) for n in names if n in z.files}


def md5_of(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def resim_jobs(resim_dir: pathlib.Path, feat_dir: pathlib.Path, *, set_name: str,
               keys: Optional[Sequence[str]] = None, tags: Optional[Sequence[str]] = None,
               instruction: bool = True, meta_ds: Optional[dict] = None) -> list[dict]:
    """One job per (re-sim run, prompt). Instruction prompts only for ``E0`` (same image and state)."""
    resim_dir, feat_dir = pathlib.Path(resim_dir), pathlib.Path(feat_dir)
    jobs = []
    for p in sorted(resim_dir.glob("*__*.npz")):
        if p.name.endswith(".partial.npz"):
            continue
        key, tag = p.stem.split("__", 1)
        if keys and key not in keys:
            continue
        if tags and tag not in tags:
            continue
        info = _npz_scalar(p, ("prompt", "target", "episode"))
        prompts = [("orig", str(info["prompt"]), str(info["target"]))]
        if instruction and tag == "E0":
            if meta_ds is None:
                from AG3S.ag3s.experiments.studies import t53_records as rc

                meta_ds = rc.load_dataset_meta()
            others = meta_ds[int(info["episode"])]["non_target_fruits"]
            prompts = instruction_prompts(str(info["prompt"]), str(info["target"]), others)
        for ptag, prompt, noun in prompts:
            jobs.append({"kind": "resim", "set": set_name, "key": key, "tag": tag, "prompt_tag": ptag,
                         "prompt": prompt, "noun": noun, "src": str(p),
                         "out": str(feat_dir / f"{key}__{tag}__{ptag}.npz")})
    return jobs


def val_jobs(feat_dir: pathlib.Path, episodes: Sequence[int]) -> list[dict]:
    """One job per val episode (all its prompt variants are written by that job)."""
    return [{"kind": "val", "set": "dev", "key": f"ep_{int(ep):06d}", "episode": int(ep),
             "out_dir": str(pathlib.Path(feat_dir) / "val")} for ep in episodes]


def t55_stamp(job: dict, *, src_md5: Optional[str], layers: Sequence[int], checkpoint: str) -> dict:
    return {"version": T55_FEAT_VERSION, "kind": job["kind"], "key": job["key"], "tag": job.get("tag"),
            "prompt_tag": job.get("prompt_tag"), "prompt": job.get("prompt"), "noun": job.get("noun"),
            "source": job.get("src"), "source_md5": src_md5, "noun_layers": [int(x) for x in layers],
            "checkpoint": str(checkpoint)}


def load_t55_stamp(path: pathlib.Path) -> Optional[dict]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["t55_stamp"]))
    except Exception:  # noqa: BLE001
        return None


def job_status(job: dict, stamp: dict) -> str:
    """``todo`` / ``done`` (same stamp) / ``conflict`` (a file made for another stamp)."""
    path = pathlib.Path(job["out"])
    if not path.is_file():
        return "todo"
    got = load_t55_stamp(path)
    return "done" if got == json.loads(json.dumps(stamp, sort_keys=True)) else "conflict"


# ------------------------------------------------------------------------------ GPU


def require_gpu_lock(args) -> None:
    from AG3S.ag3s.experiments.studies.t53_extract import lock_is_held

    if getattr(args, "no_lock_check", False):
        return
    held = lock_is_held(args.gpu_lock)
    if not held:
        raise SystemExit(f"stage extract must run under the shared GPU lock: flock {args.gpu_lock} <this command> "
                         f"(lock held: {held}); --no-lock-check only when the lead says so")


def _frames_features(s: dict, layers: np.ndarray, states, images: dict, prompt: str, noun: str) -> list[dict]:
    from AG3S.ag3s.experiments.studies import t53_extract as ex

    kv_fn, obs_fn, noun_fn = ex._feat_fns(s, layers)  # noqa: SLF001 — T53's own closures
    fx = s["fx"]
    out = []
    for i in range(len(states)):
        inputs = fx.inputs(np.asarray(states[i], np.float64), {c: images[c][i] for c in CAMS}, prompt)
        out.append(ex.feature_frame(inputs, kv_fn=kv_fn, obs_fn=obs_fn, noun_fn=noun_fn, noun=noun,
                                    ids=s["ids"], bos=s["bos"], n_layers=len(layers), state=s["state"]))
    return out


def run_resim_job(s: dict, job: dict, layers: np.ndarray, ckpt: str, force: bool) -> dict:
    from AG3S.ag3s.experiments.studies import t53_extract as ex
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    src = pathlib.Path(job["src"])
    stamp = t55_stamp(job, src_md5=md5_of(src), layers=layers, checkpoint=ckpt)
    st = job_status(job, stamp)
    if st == "done" and not force:
        return {"out": job["out"], "cached": True}
    if st == "conflict" and not force:
        raise SystemExit(f"{job['out']} was made for another stamp — --force to redo")
    with np.load(src, allow_pickle=False) as z:
        images = {c: np.asarray(z[f"image_{c}"]) for c in CAMS}
        states = np.asarray(z["state"], np.float64)
        t_step = np.asarray(z["t_step"], np.int64)
    if images["cam_high"].ndim != 4:
        raise SystemExit(f"{src} has no images (re-simulated with --no-images)")
    t0 = time.time()
    frames = _frames_features(s, layers, states, images, job["prompt"], job["noun"])
    fstamp = ex.feat_stamp(key=f"{job['key']}__{job['tag']}__{job['prompt_tag']}", prompt=job["prompt"],
                           noun=job["noun"], checkpoint=ckpt, config=s["config"], fingerprint=stamp["source_md5"],
                           n_frames=len(frames), layers=[int(x) for x in layers])
    payload = ex.assemble_feat(frames, t_step, fstamp, extra={
        "t55_stamp": json.dumps(stamp, sort_keys=True), "key": job["key"], "tag": job["tag"],
        "prompt_tag": job["prompt_tag"]})
    rs.save_npz_atomic(pathlib.Path(job["out"]), **payload)
    return {"out": job["out"], "n_frames": len(frames), "noun_errors": int((~payload["noun_ok"]).sum()),
            "seconds": time.time() - t0}


def val_unit(dataset: pathlib.Path, ep: int, tasks: dict, meta: dict, *, max_frames: Optional[int] = None):
    """(unit, images per sampled row) — SUBTASK-b frames inside the decoded video (`t48_hsr_io.dataset_unit`)."""
    from AG3S.ag3s.experiments.studies import t48_hsr_io as io

    vids = {c: io.decode_video(io.video_path(dataset, ep, c)) for c in io.CAMS}
    nvid = min(len(v) for v in vids.values())
    unit = io.dataset_unit(dataset, ep, tasks, meta, n_video=nvid, max_frames=max_frames)
    images = {c: np.stack([vids[c][int(f)] for f in unit.clock]) for c in io.CAMS}
    return unit, images, nvid


def run_val_job(s: dict, job: dict, layers: np.ndarray, ckpt: str, force: bool, *, dataset: pathlib.Path,
                tasks: dict, meta: dict, max_frames: Optional[int]) -> dict:
    from AG3S.ag3s.experiments.studies import t53_extract as ex
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    ep = int(job["episode"])
    unit, images, nvid = val_unit(dataset, ep, tasks, meta, max_frames=max_frames)
    states = np.asarray([unit.state(i) for i in range(unit.n)], np.float64)
    results = []
    for ptag, prompt, noun in instruction_prompts(unit.prompt, unit.target, unit.distractors):
        sub = {**job, "tag": "val", "prompt_tag": ptag, "prompt": prompt, "noun": noun,
               "src": f"dataset:ep{ep}", "out": str(pathlib.Path(job["out_dir"]) / f"{job['key']}__{ptag}.npz")}
        stamp = t55_stamp(sub, src_md5=None, layers=layers, checkpoint=ckpt) | {"max_frames": max_frames}
        st = job_status(sub, stamp)
        if st == "done" and not force:
            results.append({"out": sub["out"], "cached": True})
            continue
        if st == "conflict" and not force:
            raise SystemExit(f"{sub['out']} was made for another stamp — --force to redo")
        t0 = time.time()
        frames = _frames_features(s, layers, states, images, prompt, noun)
        fstamp = ex.feat_stamp(key=f"{job['key']}__{ptag}", prompt=prompt, noun=noun, checkpoint=ckpt,
                               config=s["config"], fingerprint=f"dataset:ep{ep}:n_video={nvid}",
                               n_frames=len(frames), layers=[int(x) for x in layers])
        payload = ex.assemble_feat(frames, unit.clock, fstamp, extra={
            "t55_stamp": json.dumps(stamp, sort_keys=True), "key": job["key"], "tag": "val", "prompt_tag": ptag,
            "phase_class": unit.y, "phase_index": unit.extra["phase_index"], "rows": unit.rows,
            "target": unit.target, "episode": ep, "used_arm": str(meta[ep]["used_arm"]),
            "distractors": np.asarray(unit.distractors)})
        rs.save_npz_atomic(pathlib.Path(sub["out"]), **payload)
        results.append({"out": sub["out"], "n_frames": len(frames),
                        "noun_errors": int((~payload["noun_ok"]).sum()), "seconds": time.time() - t0})
    return {"episode": ep, "results": results}


def session(args) -> dict:
    from AG3S.ag3s.experiments.studies import t53_extract as ex

    s = ex._feat_session(args)  # noqa: SLF001 — gpu_guard + checkpoint + FeatureExtractor + NounTokenExtractor
    s["config"] = args.config
    return s


# ------------------------------------------------------------------------------ check (CPU)


def check_against_t53(feat_dir: pathlib.Path, t53_run_dir: pathlib.Path, keys: Sequence[str]) -> dict:
    """E0 / orig features (re-simulated images) vs T53's ``feat/<key>.npz`` (recorded images): max |Δ|."""
    out = {}
    for k in keys:
        a = feat_dir / f"{k}__E0__orig.npz"
        b = pathlib.Path(t53_run_dir) / "verify" / "T53" / "feat" / f"{k}.npz"
        if not (a.is_file() and b.is_file()):
            out[k] = {"missing": [str(p) for p in (a, b) if not p.is_file()]}
            continue
        with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
            r = {}
            for n in ("kv_l4", "noun_hidden_final"):
                if za[n].shape == zb[n].shape:
                    r[f"{n}_max_abs"] = float(np.abs(za[n].astype(np.float64) - zb[n]).max())
                else:
                    r[f"{n}_shape"] = [list(za[n].shape), list(zb[n].shape)]
            la, lb = list(za["noun_layers"]), list(zb["noun_layers"])
            common = [x for x in la if x in lb]
            if common:
                va = za["noun_v"][:, :, [la.index(x) for x in common]]
                vb = zb["noun_v"][:, :, [lb.index(x) for x in common]]
                if va.shape == vb.shape:
                    r["noun_v_max_abs"] = float(np.abs(va.astype(np.float64) - vb).max())
            r["noun_ok_equal"] = bool(np.array_equal(za["noun_ok"], zb["noun_ok"]))
            out[k] = r
    return out


# ------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--records", nargs="+", default=None, help="subset of re-sim keys")
    ap.add_argument("--tags", nargs="+", default=None, help="subset of re-sim tags (E0, empty_..., slip_...)")
    ap.add_argument("--no-instruction", action="store_true", help="skip the instruction-changed prompts")
    ap.add_argument("--val", action="store_true", help="dev: also the T48a val C0 / C1 frames")
    ap.add_argument("--val-episodes", type=int, nargs="+", default=None, help="subset of the 50 val episodes")
    ap.add_argument("--val-max-frames", type=int, default=None)
    ap.add_argument("--dataset", default=None, help="dataset root (default storage)")
    ap.add_argument("--t53-run-dir", default=None, help="the T53 run (read-only; check: feature parity)")
    ap.add_argument("--config", default=rc51.CONFIG)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--noun-layers", type=int, nargs="+", default=list(range(N_LAYERS)))
    ap.add_argument("--gpu-lock", default=GPU_LOCK)
    ap.add_argument("--no-lock-check", action="store_true")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap


def all_jobs(args) -> tuple[list[dict], list[dict], pathlib.Path]:
    from AG3S.ag3s.experiments.studies import t48_hsr_io as io
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    forbid = [pathlib.Path(args.t53_run_dir)] if args.t53_run_dir else []
    resim_dir = rs.out_root(f"resim/{args.set}", forbid=forbid)
    feat_dir = rs.out_root(f"feat/{args.set}", forbid=forbid)
    jobs = resim_jobs(resim_dir, feat_dir, set_name=args.set, keys=args.records, tags=args.tags,
                      instruction=not args.no_instruction)
    vjobs = []
    if args.val:
        if args.set != "dev":
            raise SystemExit("--val is dev data only (T53 is the test set)")
        vjobs = val_jobs(feat_dir, args.val_episodes or io.VAL_EPISODES)
    return jobs, vjobs, feat_dir


def main(argv: Optional[Sequence[str]] = None) -> dict:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    ap = build_parser()
    args = ap.parse_args(argv)
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    if args.config != rc51.CONFIG:
        ap.error(f"--config must be {rc51.CONFIG}")
    layers = sorted(set(args.noun_layers))
    if not layers or any(not 0 <= x < N_LAYERS for x in layers):
        ap.error(f"--noun-layers in 0..{N_LAYERS - 1}")
    jobs, vjobs, feat_dir = all_jobs(args)
    info = {"step": STEP, "stage": args.stage, "argv": sys.argv, "args": vars(args), "created": rs.now_kst(),
            "python": sys.executable, "n_resim_jobs": len(jobs), "n_val_jobs": len(vjobs)}
    if args.stage == "plan":
        counts: dict[str, int] = {}
        for j in jobs:
            counts[j["prompt_tag"].split("_")[0]] = counts.get(j["prompt_tag"].split("_")[0], 0) + 1
        info.update({"jobs": jobs, "val_jobs": vjobs, "prompt_kinds": counts,
                     "exists": sum(pathlib.Path(j["out"]).is_file() for j in jobs)})
        rs.write_json(feat_dir / "plan.json", info)
        print(f"[feat plan {args.set}] {len(jobs)} re-sim jobs {counts}, {len(vjobs)} val episodes -> {feat_dir}")
        return info
    if args.stage == "check":
        rows = []
        for j in jobs:
            p = pathlib.Path(j["out"])
            st = load_t55_stamp(p) if p.is_file() else None
            ok = None
            if st is not None:
                with np.load(p, allow_pickle=False) as z:
                    ok = int((~z["noun_ok"]).sum())
            rows.append({"out": j["out"], "exists": p.is_file(), "noun_errors": ok,
                         "stamp_prompt_ok": (st or {}).get("prompt") == j["prompt"] if st else None})
        info["files"] = rows
        info["missing"] = [r["out"] for r in rows if not r["exists"]]
        if args.t53_run_dir and args.set == "test":
            keys = sorted({j["key"] for j in jobs if j["tag"] == "E0"})
            info["vs_t53_feat"] = check_against_t53(feat_dir, pathlib.Path(args.t53_run_dir), keys)
        rs.write_json(feat_dir / "check.json", info)
        print(f"[feat check {args.set}] {len(rows)} files, missing {len(info['missing'])}")
        return info
    # extract (GPU)
    require_gpu_lock(args)
    s = session(args)
    ckpt = str(s["ckpt"])
    lay = np.asarray(layers, np.int32)
    res = []
    t0 = time.time()
    for j in jobs:
        r = run_resim_job(s, j, lay, ckpt, args.force)
        res.append(r)
        print(f"[feat {args.set}] {pathlib.Path(j['out']).name}: {r}", flush=True)
    if vjobs:
        from AG3S.ag3s.experiments.studies import t48_hsr_io as io
        from AG3S.ag3s.experiments.studies import t53_records as rc

        dataset = pathlib.Path(args.dataset) if args.dataset else rc.storage_root() / io.DATASET_REL
        tasks, meta = io.task_prompts(dataset), io.episode_meta(dataset)
        for j in vjobs:
            r = run_val_job(s, j, lay, ckpt, args.force, dataset=dataset, tasks=tasks, meta=meta,
                            max_frames=args.val_max_frames)
            res.append(r)
            print(f"[feat val] {j['key']}: {[x.get('n_frames', 'cached') for x in r['results']]}", flush=True)
    info.update({"results": res, "wall_s": time.time() - t0, "checkpoint": ckpt,
                 "gpu_compute_pids_before": s["pids"], "kv_timing": s["fx"].timing(), "memory": s["fx"].memory(),
                 "noun_compile_s": s["state"].get("compiled"),
                 "noun_ms_median": (float(np.median(s["nt"].t_call) * 1e3) if s["nt"].t_call else None)})
    rs.write_json(feat_dir / f"extract_meta_{time.strftime('%Y%m%dT%H%M%S')}.json", info)
    return info


__all__ = ["check_against_t53", "instruction_prompts", "job_status", "load_t55_stamp", "main", "require_gpu_lock",
           "resim_jobs", "t55_stamp", "val_jobs"]


if __name__ == "__main__":
    main()
