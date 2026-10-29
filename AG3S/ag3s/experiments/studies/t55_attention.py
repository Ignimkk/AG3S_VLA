"""T55 — the serving action attention (L8H2) and subtask label of re-simulated frames, per prompt (GPU).

`AG3S/ag3s/docs/handoff/T55.task.md` 추가 사전 고정 2: the ``reground`` perception (`t55_perception`) re-runs the
serving AG3S grounding on every re-simulated frame, so it needs the attention the serving policy would have
returned for that frame — and, for the instruction-change condition, for the **changed prompt**.

Producer = the serving code, unchanged: `AG3S.trajopt.attention_policy.AttentionPolicy` (the class
`serve_safe` wraps the policy with) over a policy loaded by `pi05_attention.load_policy` (= the server's
`load_attention_model`: same config ``pi05_rby1_randomized_pick_place_16d_lora``, same checkpoint step as every
T47–T53 attention asset, `t51_extract.checkpoint_path`), default ``num_denoise_steps = 10``, ``noise_seed = 0``,
the shipped ``kv_L4_v1`` subtask probe (the T53 servers ran without ``--subtask-probe``). Per frame the policy
observation is the client's (`build_obs`: CHW images, float64 state, the prompt) and
`AttentionPolicy._attention_and_feature` returns the serving cell — denoise step 0, aggregation ``last``,
layer 8, head 2 (`attention_policy._DENOISE_STEP / _AGG / _LAYER / _HEAD`) — sliced per camera into 16x16 raw
softmax mass (float32, as served), plus the ``kv_L4`` feature of the same prefix pass; the probe's
``predict`` gives the ``result["subtask"]`` dict the serving `SafePolicy` received.

Jobs = `t55_features.resim_jobs` (every re-sim run of ``verify/T55/resim/<set>/`` with its own prompt
``orig``; ``E0`` also one instruction-changed prompt per non-target fruit, ``instr_<fruit>``).

Outputs ``$AG3S_RUN_DIR/verify/T55/attention/<set>/<key>__<tag>__<prompt_tag>.npz``: ``attention`` (F, 3, 16, 16)
float32 (cameras cam_high, cam_left_wrist, cam_right_wrist), ``subtask_json`` (F,) str, ``kv_l4`` (F, 512),
``t_step``, ``stamp`` (version, re-sim md5, prompt, checkpoint, config, cell, probe md5). Same stamp -> skipped;
another stamp -> refused (``--force`` redoes).

Stages: ``plan`` (CPU), ``extract`` (GPU: refuses unless the shared lock is held by the caller's ``flock`` and
nvidia-smi lists no compute process), ``check`` (CPU: test ``E0`` / ``orig`` vs the attention the serving
AG3S received — server chunks ``attention_*`` float16 — and vs T53's producer npz cell; subtask argmax vs the
recorded one; numbers only)::

    source scripts/setup/env.sh; export AG3S_RUN_DIR=<the T55 run>; M=AG3S.ag3s.experiments.studies.t55_attention
    .venv-ag3s/bin/python -m $M --stage plan --set dev
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage extract --set dev
    .venv-ag3s/bin/python -m $M --stage check --set test --t53-run-dir $R --episodes-root ... --server-roots ...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

STEP = "T55"
STAGES = ("plan", "extract", "check")
SETS = ("dev", "test")
ATTN_VERSION = "t55-attn-v1"
GPU_LOCK = "/tmp/ag3s-vla-gpu-validation.lock"
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
NUM_DENOISE = 10      # AttentionPolicy default (serve_safe passes none)
NOISE_SEED = 0        # AttentionPolicy default


def serving_cell() -> dict:
    from AG3S.trajopt import attention_policy as apol

    return {"denoise_step": int(apol._DENOISE_STEP), "agg": str(apol._AGG), "layer": int(apol._LAYER),  # noqa: SLF001
            "head": int(apol._HEAD), "num_denoise": NUM_DENOISE, "noise_seed": NOISE_SEED}  # noqa: SLF001


def attn_stamp(job: dict, *, src_md5: str, checkpoint: str, config: str, probe: Optional[dict]) -> dict:
    return {"version": ATTN_VERSION, "key": job["key"], "tag": job["tag"], "prompt_tag": job["prompt_tag"],
            "prompt": job["prompt"], "noun": job["noun"], "source": job["src"], "source_md5": src_md5,
            "checkpoint": str(checkpoint), "config": str(config), "cell": serving_cell(), "probe": probe}


def load_stamp(path: pathlib.Path) -> Optional[dict]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["stamp"]))
    except Exception:  # noqa: BLE001
        return None


def job_status(job: dict, stamp: dict) -> str:
    path = pathlib.Path(job["out"])
    if not path.is_file():
        return "todo"
    return "done" if load_stamp(path) == json.loads(json.dumps(stamp, sort_keys=True)) else "conflict"


def write_attention(path: pathlib.Path, *, attention: np.ndarray, subtask: Sequence[Optional[dict]],
                    kv_l4: Optional[np.ndarray], t_step: np.ndarray, stamp: dict) -> None:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    att = np.asarray(attention, np.float32)
    if att.ndim != 4 or att.shape[1:] != (3, 16, 16) or len(subtask) != len(att) or len(t_step) != len(att):
        raise ValueError(f"attention {att.shape}, {len(subtask)} subtask, {len(t_step)} t_step")
    rs.save_npz_atomic(pathlib.Path(path), attention=att,
                       subtask_json=np.asarray([json.dumps(s, sort_keys=True) if s is not None else "" for s in subtask]),
                       kv_l4=(np.zeros((len(att), 0), np.float32) if kv_l4 is None else np.asarray(kv_l4, np.float32)),
                       t_step=np.asarray(t_step, np.int64), cameras=np.asarray(CAMS),
                       stamp=np.asarray(json.dumps(stamp, sort_keys=True)))


def all_jobs(args) -> tuple[list[dict], pathlib.Path]:
    from AG3S.ag3s.experiments.studies import t55_features as tf
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    forbid = [pathlib.Path(args.t53_run_dir)] if args.t53_run_dir else []
    resim_dir = rs.out_root(f"resim/{args.set}", forbid=forbid)
    adir = rs.out_root(f"attention/{args.set}", forbid=forbid)
    jobs = tf.resim_jobs(resim_dir, adir, set_name=args.set, keys=args.records, tags=args.tags,
                         instruction=not args.no_instruction)
    return jobs, adir


# ------------------------------------------------------------------------------ GPU


def require_gpu_lock(args) -> None:
    from AG3S.ag3s.experiments.studies.t53_extract import lock_is_held

    if getattr(args, "no_lock_check", False):
        return
    held = lock_is_held(args.gpu_lock)
    if not held:
        raise SystemExit(f"stage extract must run under the shared GPU lock: flock {args.gpu_lock} <this command> "
                         f"(lock held: {held}); --no-lock-check only when the lead says so")


def session(args) -> dict:
    """gpu_guard + checkpoint + the serving `AttentionPolicy` (one model load per process)."""
    from AG3S.ag3s.experiments.sources.pi05_attention import load_policy
    from AG3S.ag3s.experiments.studies import t51_extract as ex51
    from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne
    from AG3S.ag3s.stages.subtask_probe import default_probe_path
    from AG3S.ag3s.experiments.studies import t55_resim as rs
    from AG3S.trajopt.attention_policy import AttentionPolicy

    pids = ne.gpu_guard(args)
    ckpt = ex51.checkpoint_path(args)
    t0 = time.time()
    policy = load_policy(args.config, str(ckpt))
    ap = AttentionPolicy(policy, policy._model, num_denoise_steps=NUM_DENOISE, noise_seed=NOISE_SEED,  # noqa: SLF001
                         subtask_probe="default")
    probe = ap.subtask_probe
    pinfo = None if probe is None else {**probe.record(), "path": str(default_probe_path()),
                                         "md5": rs.md5_of(default_probe_path())}
    return {"pids": pids, "ckpt": ckpt, "ap": ap, "probe": pinfo, "load_s": time.time() - t0}


def frames_attention(ap, images: dict, states: np.ndarray, prompt: str) -> tuple[np.ndarray, list, np.ndarray]:
    """Per frame the serving `AttentionPolicy` call on the client's observation."""
    att, sub, kv = [], [], []
    for i in range(len(states)):
        obs = {"state": np.asarray(states[i], np.float64),
               "images": {c: np.ascontiguousarray(np.transpose(images[c][i], (2, 0, 1))) for c in CAMS},
               "prompt": prompt}
        a, feat = ap._attention_and_feature(obs)  # noqa: SLF001 — the serving extraction itself
        att.append(np.stack([np.asarray(a[c], np.float32) for c in CAMS]))
        if feat is not None and ap.subtask_probe is not None:
            sub.append(ap.subtask_probe.predict(feat))
            kv.append(np.asarray(feat, np.float32).reshape(-1))
        else:
            sub.append(None)
            kv.append(np.zeros(0, np.float32))
    kvs = np.stack(kv) if kv and all(len(x) == len(kv[0]) for x in kv) else np.zeros((len(att), 0), np.float32)
    return np.asarray(att, np.float32), sub, kvs


def run_job(s: dict, job: dict, args) -> dict:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    src = pathlib.Path(job["src"])
    stamp = attn_stamp(job, src_md5=rs.md5_of(src), checkpoint=str(s["ckpt"]), config=args.config, probe=s["probe"])
    st = job_status(job, stamp)
    if st == "done" and not args.force:
        return {"out": job["out"], "cached": True}
    if st == "conflict" and not args.force:
        raise SystemExit(f"{job['out']} was made for another stamp — --force to redo")
    with np.load(src, allow_pickle=False) as z:
        images = {c: np.asarray(z[f"image_{c}"]) for c in CAMS}
        states = np.asarray(z["state"], np.float64)
        t_step = np.asarray(z["t_step"], np.int64)
    if images["cam_high"].ndim != 4:
        raise SystemExit(f"{src} has no images (re-simulated with --no-images)")
    t0 = time.time()
    att, sub, kv = frames_attention(s["ap"], images, states, job["prompt"])
    write_attention(pathlib.Path(job["out"]), attention=att, subtask=sub, kv_l4=kv, t_step=t_step, stamp=stamp)
    return {"out": job["out"], "n_frames": int(len(att)), "seconds": time.time() - t0}


# ------------------------------------------------------------------------------ check (CPU)


def compare_cells(new: np.ndarray, ref: np.ndarray) -> dict:
    """Two ``(F, 3, 16, 16)`` stacks of the serving cell: argmax-patch share, max / mean |Δ|, Pearson r."""
    a, b = np.asarray(new, np.float64), np.asarray(ref, np.float64)
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    ok = np.isfinite(b).all(axis=(-1, -2)) & np.isfinite(a).all(axis=(-1, -2))
    fa, fb = a[ok].reshape(-1, 256), b[ok].reshape(-1, 256)
    if not len(fa):
        return {"n_maps": 0}
    d = np.abs(fa - fb)
    return {"n_maps": int(len(fa)), "argmax_patch_same_share": float((fa.argmax(1) == fb.argmax(1)).mean()),
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "mean_abs_rel_to_mass": float(d.mean() / max(float(np.abs(fb).sum(1).mean()) / 256.0, 1e-30)),
            "pearson_r": float(np.corrcoef(fa.reshape(-1), fb.reshape(-1))[0, 1])}


def check_test(adir: pathlib.Path, args) -> dict:
    from AG3S.ag3s.experiments.studies import t55_perception as tp
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    out = {}
    for p in sorted(adir.glob("*__E0__orig.npz")):
        key = p.name.split("__")[0]
        r: dict[str, Any] = {}
        with np.load(p, allow_pickle=False) as z:
            att = np.asarray(z["attention"], np.float32)
            sub = [json.loads(str(x)) if str(x) else None for x in z["subtask_json"]]
        try:
            src = rs.t53_source(key, t53_run_dir=pathlib.Path(args.t53_run_dir),
                                episodes_root=None if args.episodes_root is None else pathlib.Path(args.episodes_root),
                                server_roots=[pathlib.Path(x) for x in args.server_roots])
        except SystemExit as exc:
            out[key] = {"error": str(exc)}
            continue
        if src.ref_chunk_files:
            srv = tp.server_chunk_inputs(src.ref_chunk_files)
            r["vs_server_recorded"] = compare_cells(att, srv["attention"])
            rec_arg = [None if s is None else s["argmax"] for s in srv["subtask"]]
            got_arg = [None if s is None else s["argmax"] for s in sub]
            both = [(a, b) for a, b in zip(got_arg, rec_arg) if a is not None and b is not None]
            r["subtask_argmax_equal"] = f"{sum(a == b for a, b in both)}/{len(both)}"
            dp = [max(abs(s["p"][c] - t["p"][c]) for c in s["p"]) for s, t in zip(sub, srv["subtask"])
                  if s is not None and t is not None]
            r["subtask_p_max_abs"] = float(max(dp)) if dp else None
        try:
            r["vs_t53_producer"] = compare_cells(att, tp.t53_attention_inputs(pathlib.Path(args.t53_run_dir), key))
        except Exception as exc:  # noqa: BLE001
            r["vs_t53_producer"] = {"error": f"{type(exc).__name__}: {exc}"}
        out[key] = r
    return out


# ------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument("--records", nargs="+", default=None)
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--no-instruction", action="store_true", help="skip the instruction-changed prompts")
    ap.add_argument("--t53-run-dir", default=None, help="the T53 run (read-only; check)")
    ap.add_argument("--episodes-root", default=None)
    ap.add_argument("--server-roots", nargs="*", default=[])
    ap.add_argument("--config", default=rc51.CONFIG)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--gpu-lock", default=GPU_LOCK)
    ap.add_argument("--no-lock-check", action="store_true")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> dict:
    from AG3S.ag3s.experiments.studies import t51_records as rc51
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    ap = build_parser()
    args = ap.parse_args(argv)
    if args.config != rc51.CONFIG:
        ap.error(f"--config must be {rc51.CONFIG}")
    jobs, adir = all_jobs(args)
    info = {"step": STEP, "stage": args.stage, "argv": sys.argv, "args": vars(args), "created": rs.now_kst(),
            "python": sys.executable, "n_jobs": len(jobs), "cell": serving_cell()}
    if args.stage == "plan":
        n_frames = 0
        for j in jobs:
            with np.load(j["src"], allow_pickle=False) as z:
                n_frames += int(len(z["t_step"]))
        info.update({"jobs": jobs, "n_frames": n_frames, "exists": sum(pathlib.Path(j["out"]).is_file() for j in jobs)})
        adir.mkdir(parents=True, exist_ok=True)
        rs.write_json(adir / "plan.json", info)
        print(f"[attention plan {args.set}] {len(jobs)} jobs, {n_frames} frames -> {adir}")
        return info
    if args.stage == "check":
        rows = [{"out": j["out"], "exists": pathlib.Path(j["out"]).is_file(),
                 "stamp_prompt_ok": ((load_stamp(pathlib.Path(j["out"])) or {}).get("prompt") == j["prompt"])
                 if pathlib.Path(j["out"]).is_file() else None} for j in jobs]
        info["files"] = rows
        info["missing"] = [r["out"] for r in rows if not r["exists"]]
        if args.set == "test" and args.t53_run_dir:
            info["serving_parity"] = check_test(adir, args)
        rs.write_json(adir / "check.json", info)
        print(f"[attention check {args.set}] {len(rows)} files, missing {len(info['missing'])}")
        return info
    require_gpu_lock(args)
    s = session(args)
    res, t0 = [], time.time()
    for j in jobs:
        r = run_job(s, j, args)
        res.append(r)
        print(f"[attention {args.set}] {pathlib.Path(j['out']).name}: {r}", flush=True)
    info.update({"results": res, "wall_s": time.time() - t0, "load_s": s["load_s"], "checkpoint": str(s["ckpt"]),
                 "probe": s["probe"], "gpu_compute_pids_before": s["pids"]})
    rs.write_json(adir / f"extract_meta_{time.strftime('%Y%m%dT%H%M%S')}.json", info)
    return info


__all__ = ["ATTN_VERSION", "attn_stamp", "check_test", "compare_cells", "frames_attention", "job_status", "load_stamp",
           "main", "require_gpu_lock", "serving_cell", "write_attention"]


if __name__ == "__main__":
    main()
