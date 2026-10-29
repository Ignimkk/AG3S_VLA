"""SUBTASK-g — the subtask gate is on by default.

User decision 2026-10-04 (evidence `SUBTASK-f.verify.json`): `clustering.subtask_gate` defaults to
True and `serve_safe` serves with the gate on unless `--no-subtask-gate`.

1. CLI: no flag = on · `--no-subtask-gate` = off · `--subtask-gate` (kept for compatibility) = on ·
   both = refused by argparse.
2. Start-up rule: with `--no-safe` · `--no-attention` · `--no-perception` no label reaches AG3S.
   The default gate is then turned off with one start-log line (no error); an explicit
   `--subtask-gate` with them stays an error (SUBTASK-c).
3. `build_ag3s` gets a bool from `main` and always writes it into the config (on or off).
4. In-process / offline paths follow the config default (on) — and they carry no label.
5. **No label → the gate decides nothing**: with the label None the gate-on decisions are
   bit-identical to the gate-off ones (TargetConfirm, the pipeline's constraint set, and the
   certification predicate); the only difference is the recorded `metrics["subtask"]["gate"]`.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, ClusteringConfig
from AG3S.ag3s.stages.target_grounding import TargetConfirm
from AG3S.trajopt import serve_safe

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
NO_LABEL_FLAGS = ["--no-safe", "--no-attention", "--no-perception"]


def _args(*extra):
    ap = serve_safe.build_parser()
    return ap, ap.parse_args(BASE_ARGV + list(extra))


def _startup(*extra):
    """What `main` does before anything reads the gate: refuse, then resolve to a bool."""
    ap, args = _args(*extra)
    serve_safe.reject_bad_flag_combinations(ap, args)
    note = serve_safe.resolve_subtask_gate(args)
    return args, note


def _config(args):
    return AG3SConfig.from_dict({"collision_backend": "esdf",
                                 **serve_safe.subtask_config_section(args.subtask_gate,
                                                                     args.subtask_probe)})


# ----------------------------------------------------------------------------------- 1. CLI


def test_config_default_is_on():
    assert ClusteringConfig().subtask_gate is True
    assert AG3SConfig().clustering.subtask_gate is True
    assert AG3SConfig.from_dict({}).clustering.subtask_gate is True


def test_server_default_is_on_and_written_into_the_config():
    args, note = _startup()
    assert args.subtask_gate is True and note is None
    assert serve_safe.subtask_config_section(args.subtask_gate, None) == {
        "clustering": {"subtask_gate": True}}
    assert _config(args).clustering.subtask_gate is True


def test_no_subtask_gate_turns_it_off_and_writes_false():
    args, note = _startup("--no-subtask-gate")
    assert args.subtask_gate is False and note is None
    assert serve_safe.subtask_config_section(args.subtask_gate, None) == {
        "clustering": {"subtask_gate": False}}
    assert _config(args).clustering.subtask_gate is False


def test_subtask_gate_flag_is_kept_for_compatibility():
    args, note = _startup("--subtask-gate", "--subtask-probe", "p.npz")
    assert args.subtask_gate is True and note is None
    cfg = _config(args)
    assert cfg.clustering.subtask_gate is True and cfg.clustering.subtask_probe_path == "p.npz"


def test_both_gate_flags_are_refused():
    with pytest.raises(SystemExit):
        _args("--subtask-gate", "--no-subtask-gate")


def test_resolve_is_idempotent():
    args, _ = _startup()
    assert serve_safe.resolve_subtask_gate(args) is None and args.subtask_gate is True
    args, _ = _startup("--no-attention")
    assert serve_safe.resolve_subtask_gate(args) is None and args.subtask_gate is False


# -------------------------------------------------------------------- 2. start-up rule


@pytest.mark.parametrize("flag", NO_LABEL_FLAGS)
def test_default_gate_with_no_label_starts_with_the_gate_off_and_one_log_line(flag):
    args, note = _startup(flag)
    assert args.subtask_gate is False
    assert isinstance(note, str) and "\n" not in note
    assert note.startswith("subtask gate: off") and flag in note


@pytest.mark.parametrize("flag", NO_LABEL_FLAGS)
def test_explicit_subtask_gate_with_no_label_is_still_an_error(flag):
    ap, args = _args("--subtask-gate", flag)
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


@pytest.mark.parametrize("flag", NO_LABEL_FLAGS)
def test_no_subtask_gate_with_no_label_is_silent(flag):
    args, note = _startup("--no-subtask-gate", flag)
    assert args.subtask_gate is False and note is None


def test_main_resolves_before_anything_reads_the_gate_and_logs_the_line():
    src = inspect.getsource(serve_safe.main)
    i_reject = src.index("reject_bad_flag_combinations(ap, args)")
    i_resolve = src.index("gate_note = resolve_subtask_gate(args)")
    assert i_reject < i_resolve
    assert src.index("logging.basicConfig") < i_resolve  # the line reaches the start log
    assert "logging.info(gate_note)" in src
    # AG3S-DYNAMIC-FIELD: `main` builds AG3S through `ag3s_from_args`, which reads the gate — its call counts as a read
    assert "subtask_gate=args.subtask_gate" in inspect.getsource(serve_safe.ag3s_from_args)
    first_read = min(src.index(k) for k in ("if args.subtask_gate else",
                                             "bool(args.subtask_gate)",
                                             "ag3s_from_args("))
    assert i_resolve < first_read
    # the record manifest keeps `subtask.gate` (now the resolved bool)
    assert '"subtask": {"gate": bool(args.subtask_gate)' in src


def test_build_ag3s_signature_leaves_the_default_to_the_config():
    sig = inspect.signature(serve_safe.build_ag3s)
    assert sig.parameters["subtask_gate"].default is None
    assert serve_safe.subtask_config_section(None, None) == {}


# ------------------------------------------------------------ 4. in-process / offline paths


def test_in_process_paths_follow_the_config_default():
    """`bringup.build_live_pipeline` and the offline replays (`safe_replay`, `a7`) build their
    `AG3SConfig` without a `clustering.subtask_gate` key — so the gate is on there too."""
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt import bringup
    from tests.ag3s.fixtures import PlanarTwoLinkArm

    src = inspect.getsource(bringup.build_live_pipeline)
    assert "subtask_gate" not in src
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
                                "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                                         "exclude_support_surfaces": False}})
    ag = AG3S(cfg, robot_model=PlanarTwoLinkArm())
    assert ag._target_confirm.subtask_gate is True                            # noqa: SLF001
    assert TargetConfirm(ClusteringConfig()).subtask_gate is True


# -------------------------------------------------------------- 5. label None = bit-identical


def _gate_script_module():
    import tests.ag3s.test_subtask_gate as g
    return g


@pytest.mark.parametrize("feed", ["none", "absent"])
@pytest.mark.parametrize("freeze_from", [None, 1, 5])
@pytest.mark.parametrize("placed_from", [None, 0, 8])
def test_target_confirm_label_none_gate_on_equals_gate_off(feed, freeze_from, placed_from):
    """The SUBTASK-c script (first · keep · hold · switch · occluded · destination) with the label
    None (fed as None, or never fed), with and without a freeze, with and without PLACED."""
    g = _gate_script_module()

    def run(gate):
        c = g._confirm(gate=gate)                                              # noqa: SLF001
        out = []
        for k, items in enumerate(g.SCRIPT):
            if feed == "none":
                c.note_subtask(None)
            if freeze_from is not None:
                c.freeze(k >= freeze_from)
            if placed_from is not None:
                c.note_placed(k >= placed_from)
            d = g._select(c, *items)                                           # noqa: SLF001
            out.append((g._key(d), c.subtask_label, c.subtask_blocked,         # noqa: SLF001
                        c.subtask_would_block, c.subtask_released, c.placed_seen,
                        None if c.manipulated is None else c.manipulated.id))
        return out

    on, off = run(True), run(False)
    assert on == off
    modes = {k[0][0] for k in off}
    assert "first" in modes
    if freeze_from is None:
        assert "switch" in modes  # the script exercises a switch the gate could have blocked


def _snapshot(value, path=""):
    """Everything a constraint set holds, flattened to comparable leaves (arrays byte-exact)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {}
        for f in dataclasses.fields(value):
            out.update(_snapshot(getattr(value, f.name), f"{path}.{f.name}"))
        return out
    if isinstance(value, dict):
        out = {}
        for k in sorted(value, key=str):
            out.update(_snapshot(value[k], f"{path}[{k!r}]"))
        return out or {path: "{}"}
    if isinstance(value, (list, tuple)):
        out = {f"{path}#len": len(value)}
        for i, v in enumerate(value):
            out.update(_snapshot(v, f"{path}[{i}]"))
        return out
    if isinstance(value, np.ndarray):
        return {path: (value.dtype.str, value.shape, value.tobytes())}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return {path: value.item()}
    if type(value).__module__.split(".")[0] == "casadi":
        # a symbolic expression (`parameter_symbol`): its text, not its SWIG pointer
        return {path: (type(value).__name__, str(value))}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        # a non-dataclass object (field handles etc.): its public attributes
        out = {f"{path}#type": type(value).__name__}
        for k, v in sorted(vars(value).items()):
            if not k.startswith("_"):
                out.update(_snapshot(v, f"{path}.{k}"))
        return out
    return {path: value}


#: Wall-clock fields: differ between two identical runs, so they are not decisions.
_CLOCK = (".profile", ".timestamp")


def _decisions(cs):
    snap = _snapshot(cs)
    return {k: v for k, v in snap.items()
            if not any(k.startswith(c) for c in _CLOCK) and "latency" not in k
            and not k.endswith("_ms']") and "_sec']" not in k}


def _pipeline_run(gate, feed):
    from AG3S.ag3s.runtime.pipeline import AG3S
    from tests.ag3s.fixtures import (DEFAULT_ARM_Q, LABEL_BOX, LABEL_SPHERE, PlanarTwoLinkArm,
                                     make_scene)

    arm = PlanarTwoLinkArm()
    scenes = {lab: make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q, target_label=lab)
              for lab in (LABEL_SPHERE, LABEL_BOX)}

    def inputs(lab, t):
        s = scenes[lab]
        return dict(attention_map=s.attention_grid, depth=s.depth,
                    camera_intrinsics=s.camera_intrinsics, T_base_cam=s.T_base_cam,
                    robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=s.hw, timestamp=t)

    ag = AG3S(AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.05},
                                    "clustering": {"subtask_gate": gate,
                                                   "subtask_confirm_frames": 1}}),
              robot_model=arm)
    # attention on the sphere, then on the box (a switch), a grasp in progress, then PLACED
    plan = ([(LABEL_SPHERE, False, False)] * 3 + [(LABEL_BOX, False, False)] * 4
            + [(LABEL_BOX, True, False)] * 2 + [(LABEL_SPHERE, False, True)] * 4)
    out = []
    for k, (lab, grasp, placed) in enumerate(plan):
        if feed == "none":
            ag.set_subtask(None)
        elif feed == "home":  # control: a label the gate acts on
            ag.set_subtask({"pick": 0.01, "place": 0.09, "home": 0.9})
        ag.set_grasp_active(grasp)
        ag.set_placed(placed)
        out.append(ag.process(**inputs(lab, 1.0 + 0.1 * k)))
    return out


@pytest.mark.parametrize("feed", ["none", "absent"])
def test_pipeline_label_none_gate_on_equals_gate_off(feed):
    from AG3S.trajopt.safe_policy import geometry_certified

    on, off = _pipeline_run(True, feed), _pipeline_run(False, feed)
    assert len(on) == len(off)
    for a, b in zip(on, off):
        assert a.metrics["subtask"]["gate"] is True and b.metrics["subtask"]["gate"] is False
        da, db = _decisions(a), _decisions(b)
        gate_key = [k for k in da if k.endswith("['subtask']['gate']")]
        assert len(gate_key) == 1
        for k in gate_key:
            da.pop(k), db.pop(k)
        assert da.keys() == db.keys()
        diff = [k for k in da if da[k] != db[k]]
        assert diff == [], diff[:5]
        assert geometry_certified(a) is geometry_certified(b)
        assert a.metrics["subtask"]["label"] is None and a.metrics["subtask"]["blocked"] is None
        assert a.grounding_status.value != "subtask_gated"


def test_safe_policy_hands_none_when_the_policy_carries_no_label(monkeypatch):
    """The in-process / offline wrappers (`_RecordedPolicy`, `_Recorded`, the bare policy) return no
    `subtask` — `SafePolicy._scene_fn` then hands `None` to AG3S, so the label stays None."""
    from AG3S.trajopt import wire
    from tests.trajopt.test_subtask_wiring import _policy, _spy

    sp = _policy(None)  # default config → gate on
    assert sp.ag3s._target_confirm.subtask_gate is True                       # noqa: SLF001
    sp.infer({"state": np.zeros(wire.ACTION_WIDTH), wire.PREFIX + "cameras": []})
    calls, stop = _spy(sp, monkeypatch)
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert calls["subtask"] == [None] and sp.ag3s.subtask_label is None


def test_control_the_comparison_sees_a_label_the_gate_acts_on():
    """The same comparison with a confirmed `home` label must differ — otherwise the bit-identity
    above would say nothing."""
    on, off = _pipeline_run(True, "home"), _pipeline_run(False, "home")
    differing = 0
    for a, b in zip(on, off):
        da, db = _decisions(a), _decisions(b)
        if any(da.get(k) != db.get(k) for k in set(da) | set(db)
               if not k.endswith("['subtask']['gate']")):
            differing += 1
    assert differing > 0
    assert any(cs.grounding_status.value == "subtask_gated" for cs in on)
    assert not any(cs.grounding_status.value == "subtask_gated" for cs in off)
