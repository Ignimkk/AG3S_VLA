"""T48a — HSR counterfactual: does the phase probe follow the prompt target, the state or the image?

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"   # once
    # CPU: plan (variants, train stats, SUBTASK-b data parity); --check-tokens needs openpi (CPU only)
    JAX_PLATFORMS=cpu .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.t48_hsr_counterfactual \\
        --stage plan --check-tokens
    # GPU (verifier, after the lead's slot): prefix-only kv_L4 per variant, cached per unit, resumable
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m \\
        AG3S.ag3s.experiments.studies.t48_hsr_counterfactual --stage extract [--batch 8]
    # CPU: probe + baselines + metrics (+ C0 parity vs SUBTASK-b stored features)
    .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t48_hsr_counterfactual --stage score

All raw output goes to ``$AG3S_RUN_DIR/verify/T48/`` (the CLI refuses to run without
``AG3S_RUN_DIR``):

``plan/``      train_stats.json, summary.json, variants_<set>.csv, token_check.json (--check-tokens)
``feat/<tag>/<set>/<unit>.npz``  features (V, 512) + the plan rows they belong to + plan digest +
               the decoded prompt the model saw; existing files with the same digest are skipped
``extract/``   timing.json (per-input prefix ms, GPU memory, wall), extract_meta.json
``score/``     c0_check.json, parity.json, metrics.json, predictions_<set>.csv

Variant definitions and baselines: `t48_hsr_core`. Data split and sources: `t48_hsr_io`.
GPU path: `t48_hsr_features`. Task: `AG3S/ag3s/docs/handoff/T48.task.md`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t48_hsr_core as core
from AG3S.ag3s.experiments.studies import t48_hsr_io as io

SETS = ("val", "ep1807")
STAGES = ("plan", "extract", "score")
METHODS = ("probe", "rule", "proxy", "frame_index")
#: SUBTASK-b val accuracy of kv_L4 (`kv_L4_v1.json` parity.val_acc_numpy; verify.json 0.9808).
C0_REFERENCE_ACC = 0.9807766243752403
C0_TOLERANCE = 0.002  # `kv_L4_v1.json` parity.tolerance
#: Frame-index fallback per set (`t48_hsr_core.frame_index_predict`).
FI_FALLBACK = {"val": "zero", "ep1807": "nearest"}


# ----------------------------------------------------------------------------- plumbing


def out_root() -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T48`` — refuses when the run dir is unset (no default location)."""
    from common.run_output import resolve_output_path

    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit(
            "AG3S_RUN_DIR is not set. Allocate a run first:\n"
            '  export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"')
    root = resolve_output_path("verify/T48", run_dir=run_dir, default_name="verify/T48")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return str(o)
    raise TypeError(type(o))


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=_json_default))
    tmp.replace(path)


def git_state() -> dict:
    root = pathlib.Path(__file__).resolve().parents[4]

    def run(*a):
        try:
            return subprocess.check_output(["git", "-C", str(root), *a], text=True,
                                           stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"head": run("rev-parse", "HEAD").strip(), "branch": run("branch", "--show-current").strip(),
            "dirty_files": [ln for ln in run("status", "--porcelain").splitlines() if ln]}


def now_kst() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(timespec="seconds")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--seed", type=int, default=0, help="C2 pair sampling seed")
    ap.add_argument("--max-units", type=int, default=None, help="first N val episodes only (partial run)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="first N sampled frames per unit (smoke); cache goes to feat/maxN/, not feat/full/")
    ap.add_argument("--batch", type=int, default=1, help="extract: inputs per prefix call (1 = serving path)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="extract: run even if nvidia-smi lists other compute processes")
    ap.add_argument("--check-tokens", action="store_true",
                    help="plan: run the CPU input transform (openpi, no model) on sample variants")
    ap.add_argument("--force", action="store_true", help="score: continue when C0 is not reproduced")
    ap.add_argument("--storage", default=None, help="default $AG3S_STORAGE_ROOT")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=io.CONFIG_NAME)
    ap.add_argument("--ep1807", default=None, help="recorded run dir (default storage results/workspace/...)")
    ap.add_argument("--subtask-b-feat", default=None, help="SUBTASK-b feat/ dir for C0 parity")
    ap.add_argument("--probe", default=None, help="probe asset npz (default kv_L4_v1)")
    args = ap.parse_args(argv)
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    for k in ("max_units", "max_frames"):
        v = getattr(args, k)
        if v is not None and v < 1:
            ap.error(f"--{k.replace('_', '-')} must be >= 1")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.check_tokens and args.stage != "plan":
        ap.error("--check-tokens belongs to --stage plan")
    return args


def paths_from(args) -> io.Paths:
    return io.Paths.from_storage(args.storage, dataset=args.dataset, checkpoint=args.checkpoint,
                                 ep1807=args.ep1807, subtask_b_feat=args.subtask_b_feat)


def feat_tag(args) -> str:
    return "full" if args.max_frames is None else f"max{args.max_frames}"


# ----------------------------------------------------------------------------- plan


class Plan:
    """Units + variant tables for the requested sets (deterministic; rebuilt by every stage)."""

    def __init__(self, args, paths: io.Paths, root: pathlib.Path):
        self.args, self.paths, self.root = args, paths, root
        self.meta = io.episode_meta(paths.dataset)
        self.tasks = io.task_prompts(paths.dataset)
        self.templates = core.templates_from_tasks(self.tasks.values())
        self.stats = self._train_stats()
        self.reps = io.reps_from_stats(self.stats)
        self.majority = io.majority_from_stats(self.stats)

    def _train_stats(self) -> dict:
        p = self.root / "plan" / "train_stats.json"
        if p.exists():
            st = json.loads(p.read_text())
            if st.get("dataset") == str(self.paths.dataset):
                return st
        st = io.train_stats(self.paths.dataset, self.meta)
        st["dataset"] = str(self.paths.dataset)
        write_json(p, st)
        return st

    def units(self, set_name: str):
        if set_name == "val":
            eps = io.VAL_EPISODES[: self.args.max_units] if self.args.max_units else io.VAL_EPISODES
            for ep in eps:
                yield io.dataset_unit(self.paths.dataset, ep, self.tasks, self.meta,
                                      max_frames=self.args.max_frames)
        elif set_name == "ep1807":
            yield io.ep1807_unit(self.paths.ep1807, self.meta, max_frames=self.args.max_frames)
        else:
            raise ValueError(set_name)

    def table(self, unit: core.Unit) -> core.VariantTable:
        return core.build_variants(unit, self.templates, self.reps, seed=self.args.seed)


def stage_plan(args, paths: io.Paths, root: pathlib.Path) -> dict:
    plan = Plan(args, paths, root)
    ns = io.norm_stats_state(paths.checkpoint)
    summary: dict = {"created": now_kst(), "git": git_state(), "args": vars(args),
                     "paths": {k: str(v) for k, v in vars(paths).items()},
                     "templates": plan.templates,
                     "gripper_reps": {"open": plan.reps.open, "closed": plan.reps.closed,
                                      "threshold": plan.reps.threshold},
                     "gripper_bins": {
                         "note": "numpy replica of the serving state string (t48_hsr_core.discretize_state)",
                         "state7_open": int(core.discretize_state(np.r_[np.zeros(7), plan.reps.open, np.zeros(8)],
                                                                  ns["q01"], ns["q99"])[7]),
                         "state7_closed": int(core.discretize_state(np.r_[np.zeros(7), plan.reps.closed, np.zeros(8)],
                                                                    ns["q01"], ns["q99"])[7]),
                         "state15_open": int(core.discretize_state(np.r_[np.zeros(15), plan.reps.open],
                                                                   ns["q01"], ns["q99"])[15]),
                         "state15_closed": int(core.discretize_state(np.r_[np.zeros(15), plan.reps.closed],
                                                                     ns["q01"], ns["q99"])[15])},
                     "train_stats": {k: v for k, v in plan.stats.items() if k != "frame_index_majority"},
                     "sets": {}}
    for s in args.sets:
        rows_csv, per_set = [], {"units": 0, "frames": 0, "per_class": [0, 0, 0], "variants": {},
                                 "digests": {}, "distractor_disp_max_m": 0.0, "units_info": []}
        for unit in plan.units(s):
            t = plan.table(unit)
            per_set["units"] += 1
            per_set["frames"] += unit.n
            for c in range(3):
                per_set["per_class"][c] += int((unit.y == c).sum())
            for fam in core.VARIANTS:
                per_set["variants"][fam] = per_set["variants"].get(fam, 0) + int((t.variant == fam).sum())
            per_set["digests"][unit.unit_id] = t.digest()
            disp = unit.extra.get("distractor_displacement_m") or unit.extra.get("distractor_max_displacement_m")
            per_set["distractor_disp_max_m"] = max(per_set["distractor_disp_max_m"], max(disp.values()))
            per_set["units_info"].append({"unit": unit.unit_id, "prompt": unit.prompt, "target": unit.target,
                                          "distractors": list(unit.distractors),
                                          "gripper_index": unit.gripper_index, "n": unit.n,
                                          "distractor_displacement_m": disp})
            for v in range(len(t)):
                a = int(t.host[v])
                rows_csv.append([unit.unit_id, t.variant[v], t.sub[v], a, int(t.image_row[v]),
                                 int(t.state_row[v]), int(unit.clock[a]), int(unit.y[a]),
                                 int(t.y_image[v]), int(t.y_state[v]), int(t.gt[v]), t.prompt[v],
                                 " ".join(f"{x:.6g}" for x in t.state[v])])
        per_set["total_variants"] = int(sum(per_set["variants"].values()))
        summary["sets"][s] = per_set
        with open(root / "plan" / f"variants_{s}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["unit", "variant", "sub", "host_row", "image_row", "state_row", "clock_host",
                        "y_host", "y_image", "y_state", "gt", "prompt", "state16"])
            w.writerows(rows_csv)
    summary["subtask_b_data_parity"] = subtask_b_data_parity(plan, args.sets)
    if args.check_tokens:
        summary["token_check"] = token_check(plan, args, paths)
        write_json(root / "plan" / "token_check.json", summary["token_check"])
    write_json(root / "plan" / "summary.json", summary)
    for s, v in summary["sets"].items():
        print(f"[plan {s}] units={v['units']} frames={v['frames']} per_class={v['per_class']} "
              f"variants={v['variants']} total={v['total_variants']}")
    print(f"[plan] gripper reps open={plan.reps.open:.4f} closed={plan.reps.closed:.4f}; "
          f"templates={plan.templates}")
    print(f"[plan] SUBTASK-b data parity: {json.dumps(summary['subtask_b_data_parity'].get('summary'))}")
    return summary


def subtask_b_data_parity(plan: Plan, sets: Sequence[str]) -> dict:
    """Our frame set / state / phase / rule history vs SUBTASK-b's stored feature files (no features
    compared here — that is `score/parity.json`)."""
    root = plan.paths.subtask_b_feat
    out: dict = {"dir": str(root), "found": root.exists(), "units": {}}
    if not root.exists():
        return out
    bad = 0
    n_units = 0
    for s in sets:
        for unit in plan.units(s):
            ref = _subtask_b_file(root, unit)
            if ref is None:
                continue
            n_units += 1
            with np.load(ref, allow_pickle=False) as z:
                r = _match_rows(unit, z)
                if r is None:
                    out["units"][unit.unit_id] = {"matched": 0}
                    bad += 1
                    continue
                ours, theirs = r
                g = core.gripper_min(unit.seq_states)
                closed_upto = np.cumsum(g < core.RULE_CLOSED_BELOW) > 0  # any closed in [0, i]
                d = {"matched": int(len(ours)),
                     "state_max_abs": float(np.abs(unit.seq_states[unit.rows[ours]] - z["state"][theirs]).max()),
                     "ever_closed_equal": bool(np.array_equal(closed_upto[unit.rows[ours]],
                                                              z["ever_closed"][theirs].astype(bool)))}
                if "phase_index" in z.files:
                    d["class_equal"] = bool(np.array_equal(unit.y[ours], core.phase_to_class(z["phase_index"][theirs])))
                    d["n_ref"] = int((core.phase_to_class(z["phase_index"]) >= 0).sum())
                else:
                    d["class_equal"] = bool(np.array_equal(unit.y[ours], [core.GT_MAP[g_] for g_ in z["gt"][theirs]]))
                    d["n_ref"] = int(len(z["gt"]))
                d["prompt_equal"] = str(z["prompt"]) == unit.prompt
                ok = (d["matched"] == unit.n == d["n_ref"] and d["state_max_abs"] == 0.0
                      and d["ever_closed_equal"] and d["class_equal"] and d["prompt_equal"])
                d["ok"] = bool(ok)
                bad += 0 if ok else 1
                out["units"][unit.unit_id] = d
    out["summary"] = {"found": True, "units_compared": n_units, "units_not_ok": bad}
    return out


def _subtask_b_file(root: pathlib.Path, unit: core.Unit) -> Optional[pathlib.Path]:
    p = (root / "val" / f"{unit.unit_id}.npz") if unit.set_name == "val" else (root / "closed" / "ep1807_apple.npz")
    return p if p.exists() else None


def _match_rows(unit: core.Unit, z) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """(our rows, their rows) matched by frame_index (val) or record index (ep1807)."""
    if unit.set_name == "val":
        theirs_fi = np.asarray(z["frame_index"], np.int64)
        keep = core.phase_to_class(z["phase_index"]) >= 0
        pos = {int(f): j for j, f in enumerate(theirs_fi) if keep[j]}
        ours = [a for a in range(unit.n) if int(unit.clock[a]) in pos]
        theirs = [pos[int(unit.clock[a])] for a in ours]
    else:
        fidx = np.asarray(z["file_index"], np.int64)
        pos = {int(f): j for j, f in enumerate(fidx)}
        ours = [a for a in range(unit.n) if int(unit.rows[a]) in pos]
        theirs = [pos[int(unit.rows[a])] for a in ours]
        if not np.array_equal(np.asarray(z["frame_index"])[theirs], unit.clock[ours]):
            return None
    if not ours:
        return None
    return np.asarray(ours, np.int64), np.asarray(theirs, np.int64)


_STATE_RE = re.compile(r"State: ([-\d ]+);")


def parse_state_string(text: str) -> Optional[np.ndarray]:
    m = _STATE_RE.search(text)
    return None if m is None else np.asarray([int(x) for x in m.group(1).split()], np.int64)


def token_check(plan: Plan, args, paths: io.Paths) -> dict:
    """CPU-only: run the policy input transform (no model) on the first frames of the first unit of
    each set, for every family, and record the decoded prompt the model would see. Checks that C2g
    changes exactly one state number, C1 / C1p change no state number, and C0 matches the numpy
    replica of the state string."""
    from AG3S.ag3s.experiments.studies.t48_hsr_features import build_cpu_input_transform

    fn, decode, info = build_cpu_input_transform(args.config, str(paths.checkpoint))
    ns = io.norm_stats_state(paths.checkpoint)
    blank = {c: np.zeros((3, 224, 224), np.uint8) for c in io.CAMS}
    out: dict = {"transform": info, "sets": {}}
    for s in args.sets:
        unit = next(plan.units(s))
        t = plan.table(unit)
        hosts = sorted(set(int(h) for h in t.host))[:2] + [int(h) for h in t.host[t.y_image != t.y_state][:1]]
        rows, bins = [], {}
        for v in range(len(t)):
            if int(t.host[v]) not in hosts:
                continue
            x = fn({"state": np.asarray(t.state[v], np.float32), "images": dict(blank), "prompt": str(t.prompt[v])})
            text = decode(x)
            b = parse_state_string(text)
            rep = core.discretize_state(t.state[v], ns["q01"], ns["q99"])
            bins[v] = b
            rows.append({"row": v, "variant": str(t.variant[v]), "sub": str(t.sub[v]), "host": int(t.host[v]),
                         "decoded": text, "n_text_tokens": int(np.asarray(x["tokenized_prompt_mask"]).sum()),
                         "replica_equal": bool(b is not None and np.array_equal(b, rep))})
        checks = []
        for v, r in zip([r["row"] for r in rows], rows):
            c0 = next((u for u in bins if t.variant[u] == "C0" and t.host[u] == t.host[v]), None)
            if c0 is None or bins[v] is None:
                continue
            nd = int((bins[v] != bins[c0]).sum())
            checks.append({"row": v, "variant": r["variant"], "n_state_numbers_changed_vs_C0": nd,
                           "changed_dims": np.flatnonzero(bins[v] != bins[c0]).tolist()})
        out["sets"][s] = {"unit": unit.unit_id, "rows": rows, "vs_C0": checks,
                          "all_replica_equal": all(r["replica_equal"] for r in rows)}
        print(f"[token_check {s}] {len(rows)} rows, replica_equal={out['sets'][s]['all_replica_equal']}")
        for r in rows[:9]:
            print(f"   {r['variant']:4s} {r['sub']:7s} {r['decoded']!r}")
    return out


# ----------------------------------------------------------------------------- extract (GPU)


def gpu_compute_processes() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return [f"nvidia-smi failed: {e!r}"]
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def unit_images(unit: core.Unit, paths: io.Paths, needed_rows: np.ndarray) -> dict[int, dict]:
    """{sampled row: {cam: HWC uint8}} for the rows a unit's variants feed."""
    rows = sorted(set(int(r) for r in needed_rows))
    if unit.set_name == "val":
        ep = int(unit.extra["episode"])
        vids = {c: io.decode_video(io.video_path(paths.dataset, ep, c)) for c in io.CAMS}
        nvid = min(len(v) for v in vids.values())
        too_far = [int(unit.clock[r]) for r in rows if int(unit.clock[r]) >= nvid]
        if too_far:  # SUBTASK-b extract.py:200 would have skipped these -> the plan is not SUBTASK-b's
            raise RuntimeError(f"{unit.unit_id}: frames {too_far[:5]} beyond video length {nvid}")
        unit.extra["n_video"] = nvid
        return {r: {c: vids[c][int(unit.clock[r])] for c in io.CAMS} for r in rows}
    rec = io.ep1807_records(paths.ep1807, with_images=True)
    return {r: {c: rec["images"][c][int(unit.rows[r])] for c in io.CAMS} for r in rows}


def stage_extract(args, paths: io.Paths, root: pathlib.Path) -> dict:
    procs = gpu_compute_processes()
    if procs and not args.allow_busy_gpu:
        raise SystemExit("other GPU compute processes are running (nvidia-smi):\n  " + "\n  ".join(procs)
                         + "\nWait for them (do not stop them) or pass --allow-busy-gpu if the lead approved.")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    plan = Plan(args, paths, root)
    t_wall = time.time()
    from AG3S.ag3s.experiments.studies.t48_hsr_features import FeatureExtractor

    ex = FeatureExtractor(args.config, str(paths.checkpoint), batch=args.batch)
    meta: dict = {"created": now_kst(), "git": git_state(), "args": vars(args), "gpu_processes_before": procs,
                  "load_s": ex.load_s, "units": {}, "batch_parity": None}
    ns = io.norm_stats_state(paths.checkpoint)
    warmed = False
    for s in args.sets:
        fdir = root / "feat" / feat_tag(args) / s
        fdir.mkdir(parents=True, exist_ok=True)
        for unit in plan.units(s):
            t = plan.table(unit)
            dig = t.digest()
            path = fdir / f"{unit.unit_id}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if str(z["plan_digest"]) == dig:
                        print(f"[extract {s}] {unit.unit_id} cached", flush=True)
                        continue
                raise SystemExit(f"{path} exists with a different plan digest — the plan changed "
                                 f"(seed / data / code). Use a new AG3S_RUN_DIR; not overwriting.")
            t0 = time.time()
            imgs = unit_images(unit, paths, t.image_row)

            def make(v):
                return ex.inputs(t.state[v], imgs[int(t.image_row[v])], str(t.prompt[v]))

            if not warmed:
                first = [make(v) for v in range(min(args.batch, len(t)))]
                ex.warmup(first[0], args.warmup)
                meta["batch_parity"] = ex.batch_parity(first)
                warmed = True
            n_before = len(ex.t_prefix)
            feats, decoded, n_txt = [], [], []
            for i in range(0, len(t), args.batch):   # streamed: inputs of one batch at a time
                xs = [make(v) for v in range(i, min(i + args.batch, len(t)))]
                feats.append(ex.features(xs))
                decoded += [ex.decoded_prompt(x) for x in xs]
                n_txt += [int(np.asarray(x["tokenized_prompt_mask"]).sum()) for x in xs]
            feats = np.concatenate(feats)
            decoded = np.asarray(decoded)
            n_txt = np.asarray(n_txt, np.int64)
            bins = np.stack([parse_state_string(d) if parse_state_string(d) is not None
                             else np.full(16, -9, np.int64) for d in decoded])
            rep = core.discretize_state(t.state, ns["q01"], ns["q99"])
            arrs = t.as_arrays("plan__")
            arrs.update(feature=feats.astype(np.float32), plan_digest=np.asarray(dig),
                        decoded_prompt=decoded, n_text_tokens=n_txt, state_bins_serving=bins,
                        state_bins_replica_mismatch=np.asarray(int((bins != rep).any(1).sum())),
                        unit_id=np.asarray(unit.unit_id), set_name=np.asarray(s),
                        clock=unit.clock, y=unit.y, rows=unit.rows,
                        prefix_s=np.asarray(ex.t_prefix[n_before:], np.float64),
                        n_video=np.asarray(int(unit.extra.get("n_video", -1))))
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, **arrs)
            tmp.replace(path)
            per = np.asarray(ex.t_prefix[n_before:]) / np.asarray(ex.n_per_call[n_before:])
            meta["units"][unit.unit_id] = {"variants": len(t), "wall_s": time.time() - t0,
                                           "per_input_ms_median": float(np.median(per) * 1e3),
                                           "replica_mismatch": int(arrs["state_bins_replica_mismatch"])}
            print(f"[extract {s}] {unit.unit_id} V={len(t)} {time.time() - t0:.1f}s "
                  f"prefix/input={np.median(per) * 1e3:.2f}ms mem={ex.memory()}", flush=True)
            write_json(root / "extract" / "extract_meta.json", meta)
    timing = {"wall_s": time.time() - t_wall, **ex.timing(), "memory": ex.memory(),
              "gpu_processes_before": procs, "created": now_kst()}
    write_json(root / "extract" / "timing.json", timing)
    write_json(root / "extract" / "extract_meta.json", meta)
    print("[extract] timing", json.dumps(timing), flush=True)
    return timing


# ----------------------------------------------------------------------------- score (CPU)


def _flat(plan: Plan, set_name: str, fdir: pathlib.Path, probe) -> dict:
    """Flat per-variant arrays for one set: plan rows, baseline and probe predictions."""
    cols: dict[str, list] = {k: [] for k in ("unit", "variant", "sub", "host", "image_row", "state_row",
                                             "clock", "y_host", "y_image", "y_state", "gt", "prompt",
                                             "host_key", "probe", "rule", "proxy", "frame_index",
                                             "fi_fallback", "p")}
    missing, units_info, c0_feats = [], {}, {}
    for unit in plan.units(set_name):
        t = plan.table(unit)
        b = core.baseline_predictions(unit, t, plan.majority, fallback=FI_FALLBACK[set_name])
        path = fdir / f"{unit.unit_id}.npz"
        p = np.full((len(t), 3), np.nan)
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z["plan_digest"]) != t.digest():
                    raise SystemExit(f"{path}: plan digest differs from the current plan "
                                     f"(seed/data/code changed since extract)")
                feats = np.asarray(z["feature"], np.float64)
                units_info[unit.unit_id] = {"n_text_tokens_by_variant": {
                    fam: sorted(set(int(x) for x in z["n_text_tokens"][t.variant == fam])) for fam in core.VARIANTS},
                    "state_bins_replica_mismatch": int(z["state_bins_replica_mismatch"])}
            p = probe.proba(feats)
            m0 = t.variant == "C0"
            c0_feats[unit.unit_id] = (t.host[m0], feats[m0])
        else:
            missing.append(unit.unit_id)
        pred = np.where(np.isnan(p).any(1), -1, np.nan_to_num(p).argmax(1))
        for v in range(len(t)):
            a = int(t.host[v])
            cols["unit"].append(unit.unit_id)
            cols["variant"].append(str(t.variant[v]))
            cols["sub"].append(str(t.sub[v]))
            cols["host"].append(a)
            cols["image_row"].append(int(t.image_row[v]))
            cols["state_row"].append(int(t.state_row[v]))
            cols["clock"].append(int(unit.clock[a]))
            cols["y_host"].append(int(unit.y[a]))
            cols["y_image"].append(int(t.y_image[v]))
            cols["y_state"].append(int(t.y_state[v]))
            cols["gt"].append(int(t.gt[v]))
            cols["prompt"].append(str(t.prompt[v]))
            cols["host_key"].append(f"{unit.unit_id}:{a}")
            cols["probe"].append(int(pred[v]))
            for k in ("rule", "proxy", "frame_index"):
                cols[k].append(int(b[k][v]))
            cols["fi_fallback"].append(bool(b["_frame_index_fallback"][v]))
            cols["p"].append(p[v])
        units_info.setdefault(unit.unit_id, {})["unit"] = {"n": unit.n, "target": unit.target,
                                                          "distractors": list(unit.distractors)}
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["p"] = np.asarray(cols["p"], np.float64).reshape(-1, 3)
    out["missing_units"] = missing
    out["units_info"] = units_info
    out["c0_feats"] = c0_feats
    return out


def c0_parity(plan: Plan, set_name: str, c0_feats: dict, probe) -> dict:
    """C0 features vs SUBTASK-b's stored `normal__kv_L4` on the same frames."""
    root = plan.paths.subtask_b_feat
    res: dict = {"dir": str(root), "found": root.exists(), "units": {}}
    if not root.exists():
        return res
    rel_all, agree, n_all, acc_ref, acc_new, n_lab = [], 0, 0, 0, 0, 0
    for unit in plan.units(set_name):
        if unit.unit_id not in c0_feats:
            continue
        ref = _subtask_b_file(root, unit)
        if ref is None:
            continue
        hosts, feats = c0_feats[unit.unit_id]
        with np.load(ref, allow_pickle=False) as z:
            r = _match_rows(unit, z)
            if r is None:
                continue
            ours, theirs = r
            pos = {int(h): i for i, h in enumerate(hosts)}
            sel = [(pos[int(a)], j) for a, j in zip(ours, theirs) if int(a) in pos]
            if not sel:
                continue
            fi, fj = np.asarray([s[0] for s in sel]), np.asarray([s[1] for s in sel])
            got, want = feats[fi], np.asarray(z["normal__kv_L4"], np.float64)[fj]
            rel = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
            pg, pw = probe.proba(got).argmax(1), probe.proba(want).argmax(1)
            y = unit.y[np.asarray([int(hosts[i]) for i in fi])]
            rel_all.append(rel)
            agree += int((pg == pw).sum())
            n_all += len(rel)
            acc_new += int((pg == y).sum())
            acc_ref += int((pw == y).sum())
            n_lab += len(y)
            res["units"][unit.unit_id] = {"n": int(len(rel)), "rel_l2_median": float(np.median(rel)),
                                          "rel_l2_max": float(rel.max()),
                                          "max_abs": float(np.abs(got - want).max()),
                                          "argmax_agree": int((pg == pw).sum())}
    if n_all:
        r = np.concatenate(rel_all)
        res["summary"] = {"n_frames": n_all, "n_units": len(res["units"]),
                          "rel_l2_median": float(np.median(r)), "rel_l2_p99": float(np.percentile(r, 99)),
                          "rel_l2_max": float(r.max()), "argmax_agree": agree,
                          "argmax_agree_ratio": agree / n_all,
                          "acc_new_features": acc_new / n_lab, "acc_subtask_b_features_same_frames": acc_ref / n_lab}
    return res


def stage_score(args, paths: io.Paths, root: pathlib.Path) -> dict:
    from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

    probe = SubtaskProbe.load(args.probe)
    plan = Plan(args, paths, root)
    sdir = root / "score"
    sdir.mkdir(parents=True, exist_ok=True)
    flats = {s: _flat(plan, s, root / "feat" / feat_tag(args) / s, probe) for s in args.sets}

    # ---- C0 gate (stop here when not reproduced, unless --force)
    c0: dict = {"reference_acc": C0_REFERENCE_ACC, "tolerance": C0_TOLERANCE, "sets": {}}
    parity = {}
    for s, f in flats.items():
        m = (f["variant"] == "C0") & (f["probe"] >= 0)
        acc = float((f["probe"][m] == f["gt"][m]).mean()) if m.any() else None
        parity[s] = c0_parity(plan, s, f["c0_feats"], probe)
        c0["sets"][s] = {"n": int(m.sum()), "probe_acc": acc, "missing_units": f["missing_units"],
                         "confusion": core.confusion(f["gt"][m], f["probe"][m]),
                         "acc_subtask_b_features_same_frames":
                             parity[s].get("summary", {}).get("acc_subtask_b_features_same_frames")}
    v = c0["sets"].get("val")
    full_val = (v is not None and not v["missing_units"] and args.max_units is None and args.max_frames is None)
    if v is None or v["probe_acc"] is None:
        c0["reproduced"] = None
        c0["note"] = "no val C0 features"
    elif full_val:
        c0["reproduced"] = bool(abs(v["probe_acc"] - C0_REFERENCE_ACC) <= C0_TOLERANCE)
        c0["note"] = "full val (50 episodes, 2601 frames) vs SUBTASK-b 0.9808"
    else:
        ref = v["acc_subtask_b_features_same_frames"]
        c0["reproduced"] = None if ref is None else bool(abs(v["probe_acc"] - ref) <= C0_TOLERANCE)
        c0["note"] = "partial val: compared with the probe on SUBTASK-b stored features of the same frames"
    write_json(sdir / "c0_check.json", c0)
    write_json(sdir / "parity.json", parity)
    print(f"[score] C0: {json.dumps({s: (x['n'], x['probe_acc']) for s, x in c0['sets'].items()})} "
          f"reproduced={c0['reproduced']}")
    for s, p in parity.items():
        print(f"[score] parity {s}: {json.dumps(p.get('summary'))}")
    if c0["reproduced"] is False and not args.force:
        print("[score] C0 NOT reproduced — stopping before the counterfactual metrics "
              "(T48.task.md: report the difference first). --force to continue.", file=sys.stderr)
        return {"c0": c0, "stopped": True}

    metrics: dict = {"created": now_kst(), "git": git_state(), "args": vars(args), "c0": c0,
                     "definitions": DEFINITIONS, "sets": {}}
    for s, f in flats.items():
        pair_key = np.asarray([hk if fam in ("C2s", "C2i") else "" for hk, fam in zip(f["host_key"], f["variant"])])
        ms = {}
        for meth in METHODS:
            ms[meth] = core.score_method(f["unit"], f["variant"], f["sub"], f["host_key"], f["gt"], f["y_host"],
                                         f["y_image"], f["y_state"], f[meth], pair_key=pair_key)
        metrics["sets"][s] = {
            "methods": ms,
            "n_variants": {fam: int((f["variant"] == fam).sum()) for fam in core.VARIANTS},
            "n_units": int(len(set(f["unit"].tolist()))), "missing_units": f["missing_units"],
            "frame_index_fallback_rows": int(f["fi_fallback"].sum()),
            "units_info": f["units_info"],
        }
        with open(sdir / f"predictions_{s}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["unit", "variant", "sub", "host_row", "image_row", "state_row", "clock_host", "y_host",
                        "y_image", "y_state", "gt", "prompt", "probe", "p_pick", "p_place", "p_home", "rule",
                        "proxy", "frame_index", "frame_index_fallback"])
            for i in range(len(f["unit"])):
                w.writerow([f["unit"][i], f["variant"][i], f["sub"][i], f["host"][i], f["image_row"][i],
                            f["state_row"][i], f["clock"][i], f["y_host"][i], f["y_image"][i], f["y_state"][i],
                            f["gt"][i], f["prompt"][i], f["probe"][i], *[f"{x:.6g}" for x in f["p"][i]],
                            f["rule"][i], f["proxy"][i], f["frame_index"][i], int(f["fi_fallback"][i])])
    write_json(sdir / "metrics.json", metrics)
    for s, m in metrics["sets"].items():
        line = " ".join(f"{meth}:C1->pick={_fmt(m['methods'][meth].get('C1', {}).get('to_pick_all', {}).get('pooled'))}"
                        for meth in METHODS)
        print(f"[score {s}] {line}")
    return metrics


def _fmt(x) -> str:
    return "NA" if x is None else f"{x:.3f}"


DEFINITIONS = {
    "host": "every variant row is attached to host frame A (its sampled row); C2s = image A + state B, "
            "C2i = image B + state A from the same pair (A, B)",
    "gt": {"C0": "phase class of A (val) / closed_loop_gt of A (ep1807)", "C1": "pick",
           "C1p": "class of A", "C2g": "class of A", "C2s": "class of A (image side)",
           "C2i": "class of B (image side)"},
    "baseline_history": "counterfactual replaces the current observation only: rule/proxy see A's original "
                        "history before t and the variant state at t; frame-index uses A's clock",
    "rule": "SUBTASK-b probe.py:101-105: min(s7,s15)<0.9 -> place; else ever closed (frames <= t) -> home; else pick",
    "proxy": "offline strong-rule PROXY (not the strong rule): LatchConfig thresholds on min(s7,s15) per request "
             "(dataset: every 8th frame ending at t; ep1807: every record); PRE/CLOSING->pick, HELD->place, PLACED->home",
    "frame_index": "SUBTASK-b controls.py:35-38 majority class per train frame_index; fallback val 0 (as SUBTASK-b), "
                   "ep1807 nearest train frame_index (t_step on the 15 Hz clock; SUBTASK-b did not evaluate it there)",
    "C1p.retain": "pred == C0 pred of the same host and method",
    "C2g.changed": "pred != C0 pred of the same host and method",
    "C2.follow": "state = pred equals the class of the state source; image = class of the image source; other = neither",
    "aggregation": "pooled k/n over frames AND per-episode ratio -> median, q25, q75 (episodes without the subset dropped)",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = out_root()
    paths = paths_from(args)
    for k in ("dataset",) + (("checkpoint",) if args.stage != "score" else ()) + \
             (("ep1807",) if "ep1807" in args.sets else ()):
        if not getattr(paths, k).exists():
            raise SystemExit(f"{k} not found: {getattr(paths, k)}")
    (root / "plan").mkdir(parents=True, exist_ok=True)
    if args.stage == "plan":
        stage_plan(args, paths, root)
    elif args.stage == "extract":
        stage_extract(args, paths, root)
    else:
        res = stage_score(args, paths, root)
        if res.get("stopped"):
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
