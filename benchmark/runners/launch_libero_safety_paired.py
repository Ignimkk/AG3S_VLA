"""Paired LIBERO-Safety runs (VLA-only / VLA+V11AB) through ``launch_libero_paired`` (BM-SETUP-B, ``.venv-ag3s``).

Thin, temporary wrapper: the shared launcher (``launch_libero_paired.py``, owned by BM-SETUP-A) is **not edited**.
This module replaces, in its own process only, the launcher's module-level hooks that are LIBERO / SafeLIBERO
specific -- ``client_argv`` (client script ``libero_safety_client.py`` under ``.venv-libero-safety``),
``protocol_settings``, ``runtime_schedule`` and ``server_argv`` (openpi config of the chosen checkpoint) -- and then
calls the launcher's own ``run_mode`` / ``paired_summary`` (GPU lock, server lifetime, manifests unchanged).
When the shared launcher gains these hooks (BM-SETUP-B.impl.md "hook requests") this file goes away.

``--record-constraints {none,occupancy,full}`` (SL10g, default ``none``) is the shared launcher's option: ``lp.run_mode``
appends ``--record-constraints <run_dir>/server_constraints --record-constraints-esdf <value>`` to the vla_v11ab server
and writes ``settings.record_constraints`` / ``launcher_result.record_constraints``; ``none`` leaves argv unchanged.

Refusals (before any run directory is allocated):
  * ``--policy-checkpoint pi05_libero_safety`` while openpi has no ``pi05_libero_safety`` TrainConfig (its norm
    stats live under ``assets/lerobot``; ``pi05_libero`` reads ``assets/physical-intelligence/libero``) -- hook H1;
  * ``vla_v11ab`` with ``pi05_libero_safety``: the Panda conversion (``AG3S/adapters/panda/action.py``) models the
    robosuite 1.4.1 OSC (0.05 m / 0.5 rad per unit, kp 150) but this policy emits actions for the LIBERO-Safety
    fork (2.0 / 2.0, kp 750) -- hook H2;
  * dev split = init states 0-4 (user decision 2026-10-23); anything else needs ``--data-split test`` (5-49);
  * ``--record-constraints`` other than ``none`` together with the ``vla_only`` mode.

    source scripts/setup/env.sh
    .venv-ag3s/bin/python benchmark/runners/launch_libero_safety_paired.py --task-suite-name human_safety --level 0 \
        --task-ids 0 --episode-ids 0 --run-id bmls-smoke --policy-checkpoint pi05_libero --conversion B --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runners import launch_libero_paired as lp  # noqa: E402
from benchmark.runners import launch_libero_pi05_only as l0  # noqa: E402
from benchmark.runners import libero_safety_client as lsc  # noqa: E402

CLIENT_SCRIPT = PROJECT_ROOT / "benchmark/runners/libero_safety_client.py"
ENV_WRAPPER = PROJECT_ROOT / "benchmark/LIBERO-Safety/libero/libero/envs/env_wrapper.py"
OPENPI_CONFIG = PROJECT_ROOT / "third_party/openpi/src/openpi/training/config.py"
SPLITS = {"dev": range(0, 5), "test": range(5, lsc.INIT_STATES_PER_TASK)}
SPLIT_TEXT = {"dev": "dev (LIBERO-Safety init 0-4, user decision 2026-10-23)",
              "test": "test (LIBERO-Safety init 5-49, user decision 2026-10-23)"}


def libero_safety_python() -> Path:
    return PROJECT_ROOT / ".venv-libero-safety/bin/python"


def checkpoint_dir(name: str) -> Path:
    return l0.shared_storage_root() / lsc.POLICY_CHECKPOINTS[name]["storage"]


def checkpoint_identity(name: str, require: bool = True) -> dict:
    """``l0.checkpoint_identity`` with the checkpoint's own norm-stats path (module constant swapped for the call)."""
    saved = l0.NORM_STATS_RELATIVE
    l0.NORM_STATS_RELATIVE = lsc.POLICY_CHECKPOINTS[name]["norm_stats"]
    try:
        info = l0.checkpoint_identity(checkpoint_dir(name), require=require)
    finally:
        l0.NORM_STATS_RELATIVE = saved
    info["policy_checkpoint"] = name
    return info


def openpi_has_config(name: str) -> bool:
    return re.search(r'name\s*=\s*"{}"'.format(re.escape(name)), OPENPI_CONFIG.read_text()) is not None


def make_server_argv(policy_checkpoint: str):
    config = lsc.POLICY_CHECKPOINTS[policy_checkpoint]["openpi_config"]

    def server_argv(mode, checkpoint, port, version="V11AB"):
        argv = _ORIG_SERVER_ARGV(mode, checkpoint, port, version)
        argv[argv.index("--config") + 1] = config
        return argv
    return server_argv


def make_client_argv(args):
    def client_argv(run_dir, port, _args=None):
        a = args
        out = [str(libero_safety_python()), str(CLIENT_SCRIPT), "--run-dir", str(run_dir), "--host", "127.0.0.1",
               "--port", str(port), "--task-suite-name", a.task_suite_name, "--level", str(a.level),
               "--task-ids", ",".join(map(str, a.task_ids)), "--episode-ids", ",".join(map(str, a.episode_ids)),
               "--seed", str(a.seed), "--policy-checkpoint", a.policy_checkpoint,
               "--violation-termination", a.violation_termination]
        if a.capture_segmentation:
            out.append("--capture-segmentation")
        for flag, value in (("--image-transform", a.image_transform), ("--action-adapter", a.action_adapter),
                            ("--max-steps", a.max_steps)):
            if value is not None:
                out += [flag, str(value)]
        return out
    return client_argv


def protocol_settings(args) -> dict:
    out = l0.protocol_settings(argparse.Namespace(**{**vars(args), "task_suite_name": "libero_spatial"}))
    ck = lsc.POLICY_CHECKPOINTS[args.policy_checkpoint]
    out.update({"suite": args.task_suite_name, "level": args.level, "benchmark": "LIBERO-Safety",
                "max_steps": args.max_steps or lsc.MAX_STEPS[args.task_suite_name],
                "num_steps_wait": lsc.NUM_STEPS_WAIT,
                "image_preprocessing": "{} of the raw render, resize_with_pad 224, uint8".format(
                    args.image_transform or ck["image_transform"]),
                "action_adapter": args.action_adapter or ck["action_adapter"],
                "violation_termination": args.violation_termination,
                "policy_checkpoint": args.policy_checkpoint, "policy_checkpoint_spec": ck,
                "libero_safety": lsc.protocol(), "reference": "benchmark/runners/libero_safety_client.py"})
    return out


def runtime_schedule(args) -> dict:
    out = l0.runtime_schedule(args)
    hz = l0.libero_control_freq(ENV_WRAPPER)
    out.update({"control_freq_hz": hz, "control_period_s": 1.0 / hz,
                "control_freq_source": f"{ENV_WRAPPER.relative_to(PROJECT_ROOT)} ControlEnv.__init__ default",
                "policy_call_period_sim_s": out["replan_steps"] / hz, "num_steps_wait": lsc.NUM_STEPS_WAIT,
                "first_policy_call_env_step": lsc.NUM_STEPS_WAIT})
    return out


_ORIG_SERVER_ARGV = lp.server_argv


def install_hooks(args):
    """Swap the launcher's LIBERO-specific module hooks (this process only)."""
    lp.CLIENT_SCRIPT = CLIENT_SCRIPT
    lp.client_argv = make_client_argv(args)
    lp.protocol_settings = protocol_settings
    lp.runtime_schedule = runtime_schedule
    lp.server_argv = make_server_argv(args.policy_checkpoint)
    l0.libero_python = libero_safety_python  # versions(): interpreter record of the client side


NATIVE_RESPONSE = "libero_safety_fork_native"
NATIVE_ADAPTER = "fork_native_unit"


def native_fork_version(version: str) -> bool:
    """SL-LS1 -- does this AG3S version's server argv select the native fork response model?"""
    from AG3S.adapters.panda import profile

    flags = list(profile.AG3S_VERSIONS.get(version, ()))
    return any(f == "--response-model" and i + 1 < len(flags) and flags[i + 1] == NATIVE_RESPONSE for i, f in enumerate(flags))


def check(args, modes) -> None:
    ck = lsc.POLICY_CHECKPOINTS[args.policy_checkpoint]
    if not openpi_has_config(ck["openpi_config"]):
        raise ValueError(f"openpi has no TrainConfig {ck['openpi_config']!r} (hook H1, BM-SETUP-B.impl.md)")
    if args.policy_checkpoint == "pi05_libero_safety" and "vla_v11ab" in modes and not args.allow_unscaled_v11ab:
        # hook H2 -> SL-LS1: the version must carry the native fork response model (server works in a_fork / NATIVE_UNIT),
        # and the client must multiply the rows back; the VLA-alone arm uses adapter none, so the two modes cannot share a run.
        if not native_fork_version(args.ag3s_version):
            raise ValueError("vla_v11ab with pi05_libero_safety needs the Panda conversion for the LIBERO-Safety "
                             "controller (hook H2, BM-SETUP-B.impl.md): use an AG3S version with --response-model "
                             f"{NATIVE_RESPONSE} (suffix -lsnative, SL-LS1); refused")
        if modes != ["vla_v11ab"]:
            raise ValueError("pi05_libero_safety + vla_v11ab runs alone (--modes vla_v11ab): its client action adapter "
                             f"{NATIVE_ADAPTER} differs from the VLA-alone arm's 'none'")
        if args.action_adapter not in (None, NATIVE_ADAPTER):
            raise ValueError(f"--action-adapter must be {NATIVE_ADAPTER} for this version (got {args.action_adapter})")
        args.action_adapter = NATIVE_ADAPTER
    elif native_fork_version(args.ag3s_version) and "vla_v11ab" in modes:
        raise ValueError(f"--response-model {NATIVE_RESPONSE} versions are for --policy-checkpoint pi05_libero_safety only")
    allowed = SPLITS[args.data_split]
    if any(e not in allowed for e in args.episode_ids):
        raise ValueError(f"{SPLIT_TEXT[args.data_split]}: episode ids {args.episode_ids} outside")
    if not (0 <= min(args.task_ids) and max(args.task_ids) < lsc.TASKS_PER_LEVEL):
        raise ValueError("task ids index the 5 tasks of the level")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-suite-name", choices=lsc.SUITES, required=True)
    ap.add_argument("--level", type=int, choices=lsc.LEVELS, required=True)
    ap.add_argument("--task-ids", type=lambda s: [int(x) for x in s.split(",")], required=True)
    ap.add_argument("--episode-ids", type=lambda s: [int(x) for x in s.split(",")], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--modes", default=",".join(lp.MODES))
    ap.add_argument("--step", default="BM-LS")
    ap.add_argument("--data-split", choices=sorted(SPLITS), default="dev")
    ap.add_argument("--policy-checkpoint", choices=sorted(lsc.POLICY_CHECKPOINTS), required=True)
    ap.add_argument("--violation-termination", choices=lsc.VIOLATION_TERMINATION, default="official")
    ap.add_argument("--image-transform", choices=lsc.IMAGE_TRANSFORMS, default=None)
    ap.add_argument("--action-adapter", choices=lsc.ACTION_ADAPTERS, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--capture-segmentation", action="store_true")
    ap.add_argument("--server-timeout", type=float, default=1800.0)
    ap.add_argument("--busy-gpu-wait", type=float, default=1800.0)
    ap.add_argument("--ag3s-version", default="V11AB")
    ap.add_argument("--conversion", choices=("A", "B", "C", "D", "E"), required=True)
    ap.add_argument("--allow-unscaled-v11ab", action="store_true",
                    help="diagnostic only: run vla_v11ab with pi05_libero_safety although the Panda conversion "
                         "assumes the robosuite 1.4.1 OSC scale")
    ap.add_argument("--record-constraints", choices=lp.RECORD_CONSTRAINTS_CHOICES, default="none",
                    help="SL10g -- vla_v11ab server only: serve_safe --record-constraints <run_dir>/server_constraints "
                         "--record-constraints-esdf <value> (per-call constraint npz; same as launch_libero_paired.py). "
                         "none = argv and settings unchanged")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", lsc.NUM_STEPS_WAIT)):
        setattr(args, name, value)
    args.safety_level = None  # SafeLIBERO field of the shared launcher; LIBERO-Safety uses --level
    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    modes = [m for m in args.modes.split(",") if m]
    check(args, modes)
    if args.record_constraints != "none" and "vla_only" in modes:   # same refusal as launch_libero_paired.main
        raise ValueError(f"--record-constraints {args.record_constraints} cannot run with the vla_only mode "
                         "(--no-safe server, no constraints); pass --modes vla_v11ab")
    install_hooks(args)
    identity = checkpoint_identity(args.policy_checkpoint, require=True)
    args.data_split = SPLIT_TEXT[args.data_split]
    run_id = args.run_id
    args.run_id = f"{run_id}-{args.task_suite_name}-L{args.level}-{args.policy_checkpoint}"
    if args.dry_run:
        print(json.dumps({m: lp.server_argv(m, identity["path"], 0, args.ag3s_version)
                          + lp.record_constraints_argv(m, lp.RUN_DIR_TOKEN, args.record_constraints) for m in modes},
                         indent=1))
        print(json.dumps(lp.client_argv("<run>", 0, args) + lp.oracle_client_argv("vla_v11ab", args)))   # SL12 (else unchanged)
        if any(lp.oracle_segmentation_required(m, args) for m in modes):   # SL-GTTO (off: no line)
            print(json.dumps({"client_argv_by_mode": {m: lp.gt_oracle_client_argv(m, args, lp.client_argv("<run>", 0, args))
                                                      for m in modes}}))
        print(json.dumps({"run_id": args.run_id, "protocol": protocol_settings(args),
                          "runtime_schedule": runtime_schedule(args), "checkpoint": identity}, default=str))
        return 0
    if lp.gpu_lock_held_by_caller():
        try:
            lp.check_caller_holds_gpu_lock()
        except lp.GpuLockNotHeld as exc:
            print(f"refused: {exc}", file=sys.stderr, flush=True)
            return 2
    shared = {"versions": lp.versions(), "benchmark": "LIBERO-Safety", "launcher": str(Path(__file__).name)}
    runs = {m: lp.run_mode(m, args, identity, shared) for m in modes}
    if set(runs) == set(lp.MODES):
        summary = lp.paired_summary(runs)
        with open(runs["vla_v11ab"] / "paired_summary.json", "x") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps({k: summary[k] for k in ("n", "vla_only_success", "vla_v11ab_success")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
