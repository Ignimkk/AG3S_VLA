"""main 02fe59e6 + a0b16d0a merge into feature/panda-benchmark (2026-11-03, user rulings) — CPU only.

1. Every option the merge adds (but the nine SL7 ported for P0m) is refused by the Panda server until its port (`UNPORTED_FINAL_CONFIG_DESTS`), and the
   RB-Y1-trained hidden-probe attention source is refused.
2. `--hsr6-objective-window` (T67 ②) and `--hsr6-objective-scope nbc` (SL6-A A9) are refused together at start.
"""
from __future__ import annotations

import pytest

from tests.benchmark.test_sl6a_panda_final import _final_args, gpu_ok  # noqa: F401 (fixture)

#: SL7 (P0m, 2026-11-03) ported nine of the merge options (see `PORTED_MERGE_DESTS`): the attention-surface extent and its
#: checks, the GPU surface selection, hull fill, contact allowance, and the gate label from the grasp latch (T92).
PORTED_MERGE_DESTS = (
    'attention_surface_device', 'surface_contact_allowance_mm', 'surface_grasp_hold', 'surface_hull_fill',
    'surface_keep_last', 'surface_no_ball', 'surface_target_check', 'subtask_from_latch', 'target_extent',
    'passthrough_when_clear')   # SL7b
MERGE_DESTS = (
    'grasp_corridor_ahead_mm', 'grasp_corridor_allow', 'grasp_corridor_min_target_points',
    'grasp_set_budget_ms', 'grasp_set_field_only', 'grasp_set_full_horizon', 'grasp_set_gate_mm', 'grasp_set_to',
    'grasp_set_track_scale', 'hsr_gt_stage', 'hsr5', 'hsr6_objective_window',
    'support_surface_offset_mm', 'surface_crease_angle_deg', 'surface_crease_cut',
    'surface_crease_reattach_ratio', 'surface_crease_voxel_mm', 'surface_memory_target_sticky', 'surface_reattach_hull',
    'surface_split', 'target_vacancy_delta_mm', 'target_vacancy_min_points', 'target_vacancy_p_empty',
    'target_vacancy_p_present', 'target_vacancy_release', 'target_vacancy_samples',
    # main a0b16d0a (T91-T93)
    'subtask_from_phase_machine', 'fm_guide', 'fm_guide_scale', 'fm_guide_steps',
    'fm_guide_max_step_rad', 'fm_guide_sweep')


def test_merge_options_are_parser_dests():
    from AG3S.trajopt import serve_safe as ss

    dests = {a.dest for a in ss.build_parser()._actions}
    assert set(MERGE_DESTS) | set(PORTED_MERGE_DESTS) <= dests


def test_ported_merge_options_are_no_longer_refused_by_name():
    from AG3S.adapters.panda import server as ps

    for dest in PORTED_MERGE_DESTS:
        assert dest in ps.PORTED_FINAL_CONFIG_DESTS and dest not in ps.UNPORTED_FINAL_CONFIG_DESTS


@pytest.mark.parametrize("dest", MERGE_DESTS)
def test_panda_refuses_each_merge_option(dest, gpu_ok):
    from AG3S.adapters.panda import server as ps

    assert dest in ps.UNPORTED_FINAL_CONFIG_DESTS and dest not in ps.PORTED_FINAL_CONFIG_DESTS
    args = _final_args("AG3S-final-panda")
    ps.check_unported_flags(args, None)                                              # the final version as before
    setattr(args, dest, "set-by-test")
    with pytest.raises(ValueError, match="not ported"):
        ps.check_unported_flags(args, None)


@pytest.mark.parametrize("spec", ["hidden_probe", "hidden_probe:/x/probe.npz"])
def test_panda_refuses_the_hidden_probe_source(spec, gpu_ok):
    from AG3S.adapters.panda import server as ps

    args = _final_args("AG3S-final-panda")
    args.attention_source = spec
    with pytest.raises(ValueError, match="hidden_probe is an RB-Y1-trained probe"):
        ps.check_unported_flags(args, None)


def test_objective_window_and_scope_refused_together():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    base = ["--config", "c", "--checkpoint", "k", "--model-xml", "m", "--hsr6-objective", "--neighbour-contact-mm", "20"]
    ss.reject_hsr6_flags(ap, ap.parse_args(base + ["--hsr6-objective-window"]))      # each alone: accepted
    ss.reject_hsr6_flags(ap, ap.parse_args(base + ["--hsr6-objective-scope", "nbc"]))
    with pytest.raises(SystemExit):
        ss.reject_hsr6_flags(ap, ap.parse_args(base + ["--hsr6-objective-window", "--hsr6-objective-scope", "nbc"]))
