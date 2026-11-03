"""SL7 — development's final configuration "P0m" on the Panda server (task `AG3S/ag3s/docs/handoff/SL7.task.md`), CPU only.

1. Versions: `AG3S-final-panda-p0m` / `-p0m-nohull` / `-p0m-nbc` = the Panda final flags minus what P0m does not use
   (HSR6 rules, `--nontarget-evidence`, TrajOpt multi-seed, D1 neighbour allowances) plus the P0 surface-extent flags and
   `--subtask-from-latch`; none of the forbidden options (task 0.5); launcher argv; read-back; the older versions' flag
   sets and `version_flag_values` keys unchanged.
2. Panda head: `--target-extent attention-surface` takes `noun_target:L8H6` on the Panda profile / `--robot panda` only,
   the RB-Y1 rule (`noun_target:L10H7`) is untouched.
3. Server: refusals (what is still unported), `check_unported_flags` passes the P0m versions without an HSR6 probe,
   CPU build (config, SafePolicy, gate label), config dump / recorder meta keys only with the option on.
4. Cameras: the surface selection needs head / left_wrist / right_wrist; the Panda has no right wrist, `dummy_view`
   fills it (default off), the hull-fill palm point, the gate label path, and one stored SafeLIBERO frame through
   `AG3S.process_multi_debug` (surface region on CPU; the cuRobo field build is stubbed -- it needs the GPU).
5. `PANDA_HSR6_PROBE` follows `AG3S_STORAGE_ROOT` (task 0.7).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import _Stub, _final_args, gpu_ok  # noqa: F401 (fixture)

ROOT = pathlib.Path(__file__).resolve().parents[2]
P0M, NOHULL, NBC = "AG3S-final-panda-p0m", "AG3S-final-panda-p0m-nohull", "AG3S-final-panda-p0m-nbc"
TERMSOFF, PANDAFIX = "AG3S-final-panda-p0m-termsoff", "AG3S-final-panda-p0m-pandafix"   # SL7b
OLD_FINAL = ("AG3S-final-panda", "AG3S-final-panda-g1a4", "AG3S-final-panda-v2", "AG3S-final-panda-v3",
             "AG3S-final-panda-v3-e0to", "AG3S-final-panda-v3-e0to-pc")

#: stored SafeLIBERO VLA-only rollout (geometry / call frames) and the stored noun-attention rows of the same episode
STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", ROOT.parent / "AG3S_VLA-storage"))
RUN = STORAGE / "results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only"
NOUN = STORAGE / "results/benchmark/20261102T110500_96fda38abd1b_safelibero-seg-dev-noun-rows-20261102"
STEM = "safelibero_spatial_t0_e0"
NEEDS_FRAMES = pytest.mark.skipif(
    not ((RUN / "episodes" / f"{STEM}_geom.npz").exists() and (NOUN / "noun_rows" / f"{STEM}_noun.npz").exists()),
    reason="stored SafeLIBERO rollout / noun rows not in this storage root")


def _tokens(flags):
    return [t for t in flags if t.startswith("--")]


def _args(version, extra=()):
    """The Panda server's parsed args for a P0m version (no HSR6 probe: `version_server_flags` gives none)."""
    from AG3S.adapters.panda import profile as pf

    argv = pf.version_server_argv(version)
    argv[argv.index("--config") + 1] = "pi05_libero"
    argv += ["--robot", "panda"] + pf.version_server_flags(version) + list(extra)
    return pf.parse_v11ab_args(argv=argv)


# ------------------------------------------------------------------------------------------------ 1. versions
def test_p0m_flags_are_final_minus_removed_plus_p0(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    final = list(pf.FINAL_CONFIG_FLAGS)
    p0m = list(pf.AG3S_VERSIONS[P0M])
    removed = {"--hsr6-attach-assist", "--hsr6-held-gate", "--hsr6-target-exempt", "--hsr6-objective",
               "--hsr6-objective-scope", "--nontarget-evidence", "--to-multi-seed", "--to-gpu-seeds",
               "--to-gpu-seeds-check", "--neighbour-contact-mm", "--neighbour-palm-contact-mm"}
    added = ["--target-extent", "attention-surface", "--surface-target-check", "overlap", "--surface-keep-last",
             "--surface-grasp-hold", "--surface-no-ball", "--attention-surface-device", "cuda", "--surface-hull-fill",
             "--surface-contact-allowance-mm", "10", "--subtask-from-latch"]
    assert removed <= set(_tokens(final))                                            # each removed flag was in the final
    assert not removed & set(_tokens(p0m))
    assert p0m[-len(added):] == added                                                # exactly the P0 additions, last
    kept = p0m[:-len(added)]
    assert _strip(final, removed) == kept
    # P0 values that the Panda keeps (task 0.6): the finger cover / palm shell / ROI carve / wrist tracking / after-grasp terms
    for flag in ("--palm-shell", "--palm-shell-margin", "--wrist-track-scale", "--roi-carve", "--continuity-after-grasp",
                 "--deflection-rate-after-grasp", "--placed-no-seed-certify", "--kept-geometry-follow"):
        assert flag in p0m
    assert p0m[p0m.index("--attention-source") + 1] == "noun_target:L8H6"            # task 0.2: the Panda head


def _strip(flags, dropped):
    """`flags` without the `dropped` flags and their value tokens (a value token = a token not starting with `--`)."""
    out, i = [], 0
    while i < len(flags):
        if flags[i] in dropped:
            i += 1
            while i < len(flags) and not flags[i].startswith("--"):
                i += 1
        else:
            out.append(flags[i])
            i += 1
    return out


def test_nohull_and_nbc_differ_from_p0m_exactly(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    p0m, nohull, nbc = (list(pf.AG3S_VERSIONS[v]) for v in (P0M, NOHULL, NBC))
    hull = ["--surface-hull-fill", "--surface-contact-allowance-mm", "10"]
    i = p0m.index("--surface-hull-fill")
    assert p0m[i:i + 3] == hull and nohull == p0m[:i] + p0m[i + 3:]
    assert "--surface-hull-fill" not in nohull and "--surface-contact-allowance-mm" not in nohull
    assert nbc == p0m + ["--neighbour-contact-mm", "20", "--neighbour-palm-contact-mm", "10"]
    # parsed: the neighbour allowance is on in nbc only (D1), the hull pair in p0m / nbc only
    a0, a1, a2 = _args(P0M), _args(NOHULL), _args(NBC)
    assert (a0.neighbour_contact_mm, a0.neighbour_palm_contact_mm) == (None, None) == (a1.neighbour_contact_mm,
                                                                                       a1.neighbour_palm_contact_mm)
    assert (a2.neighbour_contact_mm, a2.neighbour_palm_contact_mm) == (20.0, 10.0)
    assert a0.surface_hull_fill and a0.surface_contact_allowance_mm == 10.0 and a2.surface_hull_fill
    assert not a1.surface_hull_fill and a1.surface_contact_allowance_mm is None


@pytest.mark.parametrize("version", [P0M, NOHULL, NBC, TERMSOFF, PANDAFIX])
def test_p0m_versions_parse_and_have_no_forbidden_option(version, gpu_ok):
    args = _args(version)
    # 0.1 / 0.5: HSR6, hsr5, G1, A4, multi-seed, nontarget evidence, split / crease / sticky / corridor / pass-through off
    assert not any(getattr(args, d) for d in ("hsr6_attach_assist", "hsr6_held_gate", "hsr6_target_exempt",
                                              "hsr6_objective", "hsr6_placement_contact", "geo_grasp", "surface_split",
                                              "surface_crease_cut", "surface_memory_target_sticky",
                                              "passthrough_when_clear", "grasp_corridor_allow"))
    assert args.hsr5 is None and args.hsr6_probe is None and args.nontarget_evidence is None
    assert args.grasp_anchor is None and args.to_multi_seed in (None, 0, 1) and args.to_gpu_seeds in (None, "off")
    assert args.target_extent == "attention-surface" and args.surface_target_check == "overlap"
    assert args.surface_keep_last and args.surface_grasp_hold and args.surface_no_ball
    assert args.attention_surface_device == "cuda" and args.subtask_from_latch and args.subtask_gate is True
    assert args.attention_source == "noun_target:L8H6" and args.robot == "panda"
    assert args.robot_mask is None                                                    # the FK-sphere mask (ARM_SPHERES)


def test_p0m_versions_take_no_hsr6_probe(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    for v in (P0M, NOHULL, NBC):
        flags = pf.version_server_flags(v)
        assert "--hsr6-probe" not in flags and flags == list(pf.AG3S_VERSIONS[v])
        with pytest.raises(ValueError, match="no HSR6 rule"):
            pf.version_server_flags(v, hsr6_probe="/p.npz")
        argv = pf.version_server_argv(v)
        assert "--no-roi-carve" not in argv                                            # final base argv (ROI carve on)
        assert pf.is_final_version(v) and v in pf.P0M_VERSIONS and v not in pf.FINAL_CONFIG_VERSIONS
    for v in OLD_FINAL:                                                                 # the older final versions as before
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]
        assert v not in pf.P0M_VERSIONS


def test_older_versions_are_unchanged(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert pf.AG3S_VERSIONS["AG3S-final-panda"] == pf.FINAL_CONFIG_FLAGS
    assert pf.AG3S_VERSIONS["AG3S-final-panda-g1a4"] == pf.FINAL_CONFIG_FLAGS + pf.G1A4_FLAGS
    assert pf.FINAL_CONFIG_VERSIONS == OLD_FINAL
    assert list(pf.AG3S_VERSIONS)[:len(OLD_FINAL) + 8][:8] == ["V11AB"] + [f"V11AB-pb{i}" for i in range(1, 8)]
    # no P0m-only key in an older version's recorded flag values (the record keys are unchanged)
    p0m_only = {"target_extent", "surface_target_check", "surface_keep_last", "surface_grasp_hold", "surface_no_ball",
                "attention_surface_device", "surface_hull_fill", "surface_contact_allowance_mm", "subtask_from_latch"}
    for v in ("V11AB", "V11AB-pb7", *OLD_FINAL):
        assert not p0m_only & set(pf.version_flag_values(_final_args(v) if v in OLD_FINAL else pf.parse_v11ab_args()))
    assert p0m_only <= set(pf.version_flag_values(_args(P0M)))


def test_launcher_argv_and_readback(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import launch_libero_paired as lp

    for v in (P0M, NOHULL, NBC):
        argv = lp.server_argv("vla_v11ab", "/ck", 9, v)
        assert argv[argv.index("--config") + 1] == "pi05_libero" and argv[argv.index("--port") + 1] == "9"
        assert "--no-roi-carve" not in argv and "--hsr6-probe" not in argv and "--roi-carve" in argv
        assert argv[argv.index("--robot") + 1] == "panda" and argv[-len(pf.AG3S_VERSIONS[v]):] == list(pf.AG3S_VERSIONS[v])
        assert pf.match_ag3s_version(_args(v)) == v
    parser = lp.build_parser()
    a = parser.parse_args(["--task-ids", "1", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                           "--ag3s-version", P0M])
    assert lp.version_options(a) == {}
    # older versions read back as before (and are not mistaken for a P0m version)
    for v in OLD_FINAL:
        assert pf.match_ag3s_version(_final_args(v)) == v
    # the per-run attention cell does not rename the version; the Panda head only is accepted
    args = _args(P0M)
    assert pf.match_ag3s_version(args) == P0M
    with pytest.raises(SystemExit):
        _args(P0M, extra=["--attention-source", "noun_target:L8H2"])                  # not the Panda seed-map head


# ------------------------------------------------------------------------------------------------ 2. the Panda head
def test_seed_map_label_is_panda_only():
    from AG3S.adapters.panda.attention import PANDA_SEED_MAP_LABELS
    from AG3S.trajopt import attention_source as asrc
    from AG3S.trajopt import serve_safe as ss

    assert PANDA_SEED_MAP_LABELS == ("noun_target:L8H6", "action:last:L8H2")   # SL7d adds the action cell
    L10, L8 = asrc.parse_attention_source("noun_target:L10H7"), asrc.parse_attention_source("noun_target:L8H6")
    assert asrc.seed_map_source_ok(L10) and not asrc.seed_map_source_ok(L8) and not asrc.seed_map_source_ok(None)
    assert asrc.seed_map_source_ok(L8, PANDA_SEED_MAP_LABELS) and not asrc.seed_map_source_ok(
        asrc.parse_attention_source("noun_target:L8H2"), PANDA_SEED_MAP_LABELS)
    base = ["--config", "c", "--checkpoint", "k", "--model-xml", "m", "--esdf-backend", "curobo", "--fine-voxel", "0.005",
            "--target-field-policy", "exclude-authorized", "--target-extent", "attention-surface"]
    ap = ss.build_parser()

    def reject(extra):
        args = ap.parse_args(base + extra)
        ss.reject_bad_flag_combinations(ap, args)

    reject(["--attention-source", "noun_target:L10H7"])                                # RB-Y1: as before
    with pytest.raises(SystemExit):
        reject(["--attention-source", "noun_target:L8H6"])                             # RB-Y1 server: L8H6 still refused
    with pytest.raises(SystemExit):
        reject(["--robot", "panda", "--attention-source", "noun_target:L8H2"])
    reject(["--robot", "panda", "--attention-source", "noun_target:L8H6"])             # --robot panda: the Panda head
    reject(["--robot", "panda", "--attention-source", "noun_target:L10H7"])
    args = ap.parse_args(base + ["--attention-source", "noun_target:L8H6"])
    ss.reject_bad_flag_combinations(ap, args, seed_map_labels=PANDA_SEED_MAP_LABELS)   # the Panda profile's explicit form


# ------------------------------------------------------------------------------------------------ 3. server
def test_ported_and_still_refused_dests():
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    ported = ("target_extent", "surface_target_check", "surface_keep_last", "surface_grasp_hold", "surface_no_ball",
              "attention_surface_device", "surface_hull_fill", "surface_contact_allowance_mm", "subtask_from_latch",
              "passthrough_when_clear")   # SL7b
    refused = ("surface_split", "surface_crease_cut", "surface_crease_voxel_mm", "surface_crease_angle_deg",
               "surface_crease_reattach_ratio", "surface_memory_target_sticky", "surface_reattach_hull",
               "target_vacancy_release", "grasp_corridor_allow", "grasp_set_to", "hsr_gt_stage", "hsr5",
               "hsr6_objective_window", "support_surface_offset_mm",
               "subtask_from_phase_machine", "fm_guide", "robot_mask", "oracle_extent_identity",
               "oracle_max_dist_voxels", "hybrid_identity", "hybrid_other_exclude_voxels", "target_free_coarse")
    dests = {a.dest for a in ss.build_parser()._actions}
    for d in ported:
        assert d in dests and d in ps.PORTED_FINAL_CONFIG_DESTS and d not in ps.UNPORTED_FINAL_CONFIG_DESTS, d
    for d in refused:
        assert d in ps.UNPORTED_FINAL_CONFIG_DESTS and d not in ps.PORTED_FINAL_CONFIG_DESTS, d
    assert not set(ps.UNPORTED_FINAL_CONFIG_DESTS) & set(ps.PORTED_FINAL_CONFIG_DESTS)


@pytest.mark.parametrize("version", [P0M, NOHULL, NBC, TERMSOFF, PANDAFIX])
def test_p0m_passes_check_unported_flags_without_a_probe(version, gpu_ok):
    from AG3S.adapters.panda import server as ps

    ps.check_unported_flags(_args(version), None)


@pytest.mark.parametrize("dest, value", [
    ("surface_split", True), ("surface_crease_cut", True), ("surface_memory_target_sticky", True),
    ("hsr_gt_stage", True), ("subtask_from_phase_machine", True),
    ("robot_mask", "render"), ("target_free_coarse", True), ("grasp_corridor_allow", True),
    ("target_extent", "oracle-visible"), ("target_extent", "attention-surface-hybrid")])
def test_p0m_still_refuses_what_is_not_ported(dest, value, gpu_ok):
    from AG3S.adapters.panda import server as ps

    args = _args(P0M)
    setattr(args, dest, value)
    with pytest.raises(ValueError, match="not ported"):
        ps.check_unported_flags(args, None)


def test_hsr6_still_needs_a_probe_and_hidden_probe_is_refused(gpu_ok):
    from AG3S.adapters.panda import server as ps

    args = _args(P0M)
    args.hsr6_attach_assist = True
    with pytest.raises(ValueError, match="needs --hsr6-probe"):
        ps.check_unported_flags(args, None)
    args = _args(P0M)
    args.attention_source = "hidden_probe"
    with pytest.raises(ValueError, match="hidden_probe is an RB-Y1-trained probe"):
        ps.check_unported_flags(args, None)


@pytest.mark.parametrize("version", [P0M, NOHULL, NBC, TERMSOFF, PANDAFIX])
def test_p0m_builds_on_cpu(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt.grasp_latch import GraspPhase
    from AG3S.trajopt.safe_policy import LATCH_SUBTASK

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    args = _args(version)
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    safe = served._policy
    e = safe.ag3s.config.esdf
    # surface extent in the AG3S config (task: target_extent == attention_surface and the surface keys)
    assert e.target_extent == "attention_surface" and e.surface_target_check == "overlap"
    assert e.surface_keep_last and e.surface_grasp_hold and e.surface_no_ball and not e.surface_split
    assert e.target_exclusion == "ball_owned" and e.robot_mask == "spheres"
    hull = version != NOHULL
    assert bool(e.surface_hull_fill) is hull and e.surface_hull_fill_palm_mm == 80.0
    assert e.surface_contact_allowance_mm == (10.0 if hull else None)                  # nohull: both off
    assert not e.surface_crease_cut and not e.surface_memory_target_sticky
    # no HSR6, no multi-seed (E0), no G1 / A4, neighbour allowances only in nbc (D1)
    assert safe.hsr6 is None and safe._geo is None
    cost, col = safe.to_config.cost, safe.to_config.collision
    assert int(getattr(safe.to_config.sqp, "multi_seed", 0) or 0) <= 1 and getattr(safe.to_config.sqp, "gpu_seeds", "off") == "off"
    nbc = version == NBC
    assert (float(col.neighbour_contact_depth), float(col.neighbour_palm_contact_depth)) == ((0.02, 0.01) if nbc else (0.0, 0.0))
    # the gate label comes from the latch; the gate is on (clustering.subtask_gate)
    assert safe.subtask_from_latch is True and safe.ag3s.config.clustering.subtask_gate is True
    assert safe.hold_site_offset_m == (0.0, 0.0, 0.097)
    for phase in GraspPhase:
        safe._latch = types.SimpleNamespace(phase=phase)
        p = safe._subtask_gate_input()
        want = LATCH_SUBTASK.get(phase)
        assert (p is None) is (want is None)
        if p is not None:
            assert max(p, key=p.get) == want and sorted(p.values()) == [0.0, 0.0, 1.0]
    assert {k.value: v for k, v in LATCH_SUBTASK.items()} == {"searching": "pick", "latched": "pick", "closing": "pick",
                                                              "held": "place", "placed": "home"}
    # the A12 terms stay (the Panda value, task 0.6), after-grasp only
    if version == TERMSOFF:                       # SL7b (c): both terms off in every phase
        assert cost.w_continuity == 0.0 and not cost.continuity_after_grasp and not cost.deflection_rate_after_grasp
    else:
        assert cost.continuity_after_grasp and cost.deflection_rate_after_grasp
    # Panda cameras: the surface selection pads the missing right wrist
    assert safe.ag3s.surface_pad_missing_cameras is True and info["ag3s"]["surface_pad_missing_cameras"] is True
    # config dump
    fc = info["final_config"]
    assert fc["version"] == version and fc["hsr6"] is None and fc["multi_seed"] is None and fc["geo_grasp"] is None
    sr = fc["attention_surface"]
    assert sr["seed_map"] == "noun_target:L8H6" and sr["dummy_cameras"] == ["right_wrist"] and sr["arm"] == "top1-only-ns:N"
    assert sr["device_requested"] == "cuda" and sr["surface_hull_fill"] is hull
    assert fc["subtask_from_latch"]["source"] == "latch" and fc["subtask_from_latch"]["kv_L4_v1_probe"] is None
    assert fc["inert_on_panda"] == []                      # placed-no-seed-certify gets a `home` label from the latch
    assert (fc["neighbour_contact"] is not None) is nbc and (fc["neighbour_palm_contact"] is not None) is nbc
    if nbc:
        assert fc["neighbour_palm_contact"]["site"] == "panda_grip_site" and fc["neighbour_palm_contact"]["depth_m"] == 0.01
        assert fc["neighbour_palm_contact"]["site_offset_m"] == [0.0, 0.0, 0.097]
        assert fc["neighbour_contact"]["finger_links"] == {"left": ("panda_leftfinger", "panda_rightfinger")} or \
            fc["neighbour_contact"]["finger_links"] == {"left": ["panda_leftfinger", "panda_rightfinger"]}
    json.dumps(info, default=str)


def test_older_versions_keep_their_dump_keys(gpu_ok, monkeypatch):
    """No SL7 key appears in an older version's build `info` / config dump (SafePolicy arguments: the sl6a tests)."""
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _final_args("AG3S-final-panda"), gpu_warm_up=False)
    assert "surface_pad_missing_cameras" not in info["ag3s"] and "attention_surface" not in info["final_config"]
    assert "subtask_from_latch" not in info["final_config"]
    assert [r["flag"] for r in info["final_config"]["inert_on_panda"]] == ["--placed-no-seed-certify"]
    assert served._policy.ag3s.surface_pad_missing_cameras is False and served._policy.subtask_from_latch is False
    assert "subtask_source" not in json.dumps(served._policy.__dict__.get("_last_subtask_latch", {}))


def test_recorder_meta_p0m_keys(gpu_ok, tmp_path, monkeypatch):
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    args = _args(P0M, extra=["--record-constraints", str(tmp_path / "rc"), "--record-constraints-esdf", "occupancy"])
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    meta = served.recorder.meta
    assert meta["ag3s_version"] == P0M
    assert meta["subtask"]["source"] == "latch" and meta["subtask"]["placed_no_seed_certify"] is True
    assert "placed_no_seed_certify_inert_on_panda" not in meta["subtask"]
    assert meta["target_extent"] == "attention_surface" and meta["attention_surface_device"] == "cuda"
    assert meta["surface_keep_last"] and meta["surface_hull_fill"] and meta["surface_contact_allowance_mm"] == 10.0
    assert meta["surface_target_check"] == "overlap" and "hsr6" not in meta
    assert meta["final_config"]["attention_surface"]["seed_map"] == "noun_target:L8H6"
    ps.close_recorder(served)


# ------------------------------------------------------------------------------------------------ 4. cameras, gate, frame
def test_dummy_view_and_padding_are_off_by_default():
    from AG3S.ag3s.runtime import attention_surface as asurf
    from AG3S.ag3s.runtime.pipeline import AG3S

    depth = np.full((8, 8), 0.5)
    K = np.array([[10.0, 0, 4], [0, 10.0, 4], [0, 0, 1]])
    T = np.eye(4)
    real = [asurf.SurfaceView(n, depth, K, T, np.zeros((8, 8), bool), np.ones((16, 16), np.float32))
            for n in ("head", "left_wrist")]
    padded, names = asurf.pad_missing_views(real)
    assert names == ["right_wrist"] and [v.name for v in padded] == list(asurf.CAMERA_ORDER)
    d = padded[2]
    assert d.depth_m.shape == (8, 8) and int((d.depth_m > 0).sum()) == 1 and d.robot_mask.all()
    assert d.attention.shape == (16, 16) and not d.attention.any() and np.array_equal(d.K, K)
    assert padded[0] is real[0] and padded[1] is real[1]
    full, none = asurf.pad_missing_views(padded)                      # nothing missing: the same views, no dummy
    assert none == [] and [v.name for v in full] == list(asurf.CAMERA_ORDER) and full[2] is padded[2]
    assert asurf.pad_missing_views([]) == ([], [])
    # unpadded, two cameras abstain (what the RB-Y1 AG3S does without the Panda switch)
    assert asurf.select_target_surface(real).record["why"] == "missing_camera:right_wrist"
    assert AG3S().surface_pad_missing_cameras is False


def test_hull_fill_palm_point_is_the_finger_sphere_mean_near_the_hold_site():
    """Task 4: the hull-fill window (80 mm) measures from the mean of the finger-link constraint spheres, not from the A6
    hold site; on the Panda the two are within ~16 mm."""
    from AG3S.adapters.panda import hand_geometry as hg
    from AG3S.adapters.panda import model as pm

    m = pm.build_panda_models("gripper").constraint
    pair = m.gripper_joint_pairs()[0]
    assert pair["parent_link"] == "panda_hand" and tuple(pair["children"]) == tuple(pm.PANDA_FINGER_LINKS)
    q = np.zeros(len(m.joint_names))
    for name, val in zip(pm.PANDA_ARM_JOINTS, [0, -0.4, 0, -2.2, 0, 1.9, 0.8]):
        q[m.joint_names.index(name)] = val
    names = list(m.sphere_link_names)
    centres = np.asarray(m.sphere_centers_numeric(q)[0])
    idx = [i for i, n in enumerate(names) if n in pm.PANDA_FINGER_LINKS]
    assert idx
    T = np.asarray(m.link_pose(q, "panda_hand"))
    site = T[:3, :3] @ np.asarray(hg.PANDA_HOLD_SITE_OFFSET_M) + T[:3, 3]
    assert np.linalg.norm(centres[idx].mean(axis=0) - site) < 0.030


def _replay_ag3s(version, monkeypatch, pad=True):
    """The P0m (or other) Panda AG3S fed one stored frame at a time; the cuRobo field build (GPU) is stubbed out."""
    sys.path.insert(0, str(ROOT / "benchmark/runners"))
    import perception_replay as PR
    from AG3S.adapters.panda import profile as pf
    from AG3S.ag3s.runtime import pipeline

    class _Stop(Exception):
        pass

    def stop(self, *a, **k):
        raise _Stop

    monkeypatch.setattr(pipeline.AG3S, "_build_esdf", stop)
    args = _args(version)
    ag3s, _ = pf.build_panda_ag3s(kw=pf.v11ab_build_kwargs(args))
    ag3s.surface_pad_missing_cameras = pad
    with np.load(RUN / "episodes" / f"{STEM}_geom.npz", allow_pickle=False) as z:
        geom = {k: z[k] for k in z.files}
    with np.load(RUN / "episodes" / f"{STEM}_calls.npz", allow_pickle=False) as z:
        call_obs = z["call_obs_index"]
    with np.load(NOUN / "noun_rows" / f"{STEM}_noun.npz", allow_pickle=False) as z:
        noun = {k: z[k] for k in ("noun_image", "layers", "noun_ok")}
    rows_of = {int(o): i for i, o in enumerate(geom["obs_index"])}
    layer = int(np.flatnonzero(noun["layers"] == 8)[0])
    phase = pf.manifest_entry()["client_phase"]

    def step(k, subtask=None):
        v = noun["noun_image"][k, layer, 6]                                       # the Panda head L8H6
        grids = np.stack([v[j * 256:(j + 1) * 256].reshape(16, 16) for j in range(2)])
        obs, fingers, rs9, _ = PR.build_observations(geom, rows_of[int(call_obs[k])], grids, "unmirror", ag3s.robot_model)
        PR.apply_frame_state(ag3s, fingers, rs9)
        if subtask is not None:
            ag3s.set_subtask(subtask)
        try:
            ag3s.process_multi_debug(obs, phase=phase, active_manipulators=["left"], static_geometry=None)
        except _Stop:
            pass                                                                    # everything before the field build ran
        return obs

    ag3s.reset()
    return ag3s, step


@NEEDS_FRAMES
def test_one_stored_frame_makes_a_surface_region_on_cpu(gpu_ok, monkeypatch):
    ag3s, step = _replay_ag3s(P0M, monkeypatch)
    obs = step(0)
    assert [o.camera_id.value for o in obs] == ["head", "left_wrist"]               # the Panda has two cameras
    sf = ag3s._surface_frame
    assert sf["applied"] is True and sf["cameras"] == ["head", "left_wrist", "right_wrist"]
    assert sf["robot_mask_source"] == {"head": "spheres", "left_wrist": "spheres", "right_wrist": "dummy"}
    assert sf["arm"] == "top1-only-ns:N" and sf["n_selected_nodes"] > 0 and sf["n_robot_px"] > 0
    post = sf["t78"]
    assert post["target_check"]["passed"] is True and post["target_check"]["rule"] == "region_on_ats"
    assert post["target_check"]["n_ats_points"] > 0
    # the region reached the oracle memory (the samples the hand layer removes): code-1 pixels exist, only on real cameras
    codes = ag3s._surface_codes
    assert set(codes) == {"head", "left_wrist", "right_wrist"} and not (codes["right_wrist"] == 1).any()
    assert sum(int((c == 1).sum()) for c in codes.values()) > 0
    assert ag3s._oracle_memory is not None and len(ag3s._oracle_memory) > 0
    assert ag3s._manipulated is not None


@NEEDS_FRAMES
def test_without_the_dummy_the_two_camera_frame_abstains(gpu_ok, monkeypatch):
    ag3s, step = _replay_ag3s(P0M, monkeypatch, pad=False)
    step(0)
    sf = ag3s._surface_frame
    assert sf["applied"] is False and sf["why"] == "missing_camera:right_wrist"


@NEEDS_FRAMES
def test_the_latch_label_drives_the_gate_on_a_panda_frame(gpu_ok, monkeypatch):
    """Task 4: the gate works on the Panda path -- a `place` label (what the latch gives while HELD) confirmed over three
    requests withholds the first manipulated object; with the `pick` label the same frame adopts it."""
    place = {"pick": 0.0, "place": 1.0, "home": 0.0}
    pick = {"pick": 1.0, "place": 0.0, "home": 0.0}
    for label, adopted in ((pick, True), (place, False)):
        ag3s, step = _replay_ag3s(P0M, monkeypatch)
        for _ in range(3):
            ag3s._target_confirm.note_subtask(label)                                # three earlier requests
        step(0, subtask=label)
        tc = ag3s._target_confirm
        assert tc.subtask_gate is True and tc.subtask.label == max(label, key=label.get)
        assert (ag3s._manipulated is not None) is adopted
        assert tc.subtask_blocked == (None if adopted else "first")


# ------------------------------------------------------------------------------------------------ 5. probe path (0.7)
def test_hsr6_probe_is_found_under_the_storage_root():
    from AG3S.adapters.panda import profile as pf
    from common.workspace import shared_storage_root

    assert pf.PANDA_HSR6_PROBE == str(shared_storage_root() / pf.PANDA_HSR6_PROBE_REL)
    assert pf.PANDA_HSR6_PROBE_REL.endswith("probe/hsr6_panda_kv_L4.npz")
    if shared_storage_root() == pathlib.Path("/mnt/dev/AG3S_VLA-storage"):
        # the earlier fixed path: with the default root the recorded string is the same
        assert pf.PANDA_HSR6_PROBE == ("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261029T192223_5ea56af44722_"
                                       "sl6b-hsr6-panda-labels/probe/hsr6_panda_kv_L4.npz")
    code = ("from AG3S.adapters.panda import profile as pf; print(pf.PANDA_HSR6_PROBE)")
    env = dict(os.environ, AG3S_STORAGE_ROOT="/tmp/sl7-other-storage", PYTHONPATH=str(ROOT))
    out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=str(ROOT), text=True,
                                  stderr=subprocess.DEVNULL).strip().splitlines()[-1]
    assert out == "/tmp/sl7-other-storage/" + pf.PANDA_HSR6_PROBE_REL
    # the SHA-256 check stays where it was (the sl6a tests read the file when present)
    assert len(pf.PANDA_HSR6_PROBE_SHA256) == 64

def test_sl7b_termsoff_and_pandafix_differ_from_p0m_exactly(gpu_ok):
    """SL7b — (c) termsoff = P0m with the continuity / deflection-rate terms off in every phase; (b) pandafix = P0m + the
    Panda corrections of pc판 beyond P0m (V11AB-pb7 minus the pb2 gate), HSR6 still off."""
    from AG3S.adapters.panda import profile as pf

    p0m, off, fix = (list(pf.AG3S_VERSIONS[v]) for v in (P0M, TERMSOFF, PANDAFIX))
    after = ("--continuity-after-grasp", "--deflection-rate-after-grasp")
    assert off == [t for t in p0m if t not in after] + ["--w-continuity", "0", "--w-deflection-rate", "0"]
    pb7 = list(pf.AG3S_VERSIONS["V11AB-pb7"])
    i = pb7.index("--minimal-intervention-mm")
    assert fix == p0m + pb7[:i] + pb7[i + 2:]
    a_off, a_fix = _args(TERMSOFF), _args(PANDAFIX)
    assert a_off.w_continuity == 0.0 and a_off.w_deflection_rate == 0.0 and not a_off.continuity_after_grasp
    assert a_fix.reach_at_capture and a_fix.links == "arms" and a_fix.minimal_intervention_mm is None
    assert not a_fix.hsr6_placement_contact and a_fix.hsr6_probe is None
    for a in (a_off, a_fix):                                       # target extent and phase labels stay P0m's
        assert a.target_extent == "attention-surface" and a.subtask_from_latch and a.surface_hull_fill


@pytest.mark.parametrize("version,base", [("AG3S-final-panda-p0m-termsoff-pass", TERMSOFF),
                                          ("AG3S-final-panda-p0m-pandafix-pass", PANDAFIX)])
def test_sl7b_pass_versions_apply_passthrough_on_the_panda(version, base, gpu_ok, monkeypatch):
    """SL7b (d) — base + `--passthrough-when-clear`; the Panda server accepts it and its TrajOpt config carries
    `sqp.passthrough_when_clear` (serve_safe.trajopt_config_from_args, the call the Panda server makes)."""
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    assert list(pf.AG3S_VERSIONS[version]) == list(pf.AG3S_VERSIONS[base]) + ["--passthrough-when-clear"]
    args = _args(version)
    ps.check_unported_flags(args, None)
    assert ss.trajopt_config_from_args(args).sqp.passthrough_when_clear is True
    assert ss.trajopt_config_from_args(_args(base)).sqp.passthrough_when_clear is False


BALL = "AG3S-final-panda-p0m-ball"   # SL7c ①


def test_sl7c_ball_version_is_p0m_with_the_ball_extent(gpu_ok, monkeypatch):
    """SL7c ① — diagnostic version: P0m minus the surface-extraction / hull-fill / contact-allowance flags plus 1판's
    `--nontarget-evidence surface`; the AG3S config keeps the ball extent, the rest (latch gate, no HSR6, E0, D1) is P0m's."""
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    ball, p0m = list(pf.AG3S_VERSIONS[BALL]), list(pf.AG3S_VERSIONS[P0M])
    final = list(pf.FINAL_CONFIG_FLAGS)
    i = final.index("--nontarget-evidence")
    assert final[i:i + 2] == ["--nontarget-evidence", "surface"] and "--roi-carve-device" in ball
    assert ball[ball.index("--roi-carve-device") + 1] == "cuda"
    assert not ({"--target-extent", "--surface-hull-fill", "--surface-contact-allowance-mm", "--surface-no-ball"} & set(ball))
    # same as P0m apart from the extent: remove both surface blocks from P0m and add back the nontarget evidence
    j = p0m.index("--target-extent")
    p0m_wo = p0m[:j] + p0m[j + len(pf.P0M_SURFACE_FLAGS) + len(pf.P0M_HULL_FLAGS):]
    assert sorted(ball) == sorted(p0m_wo + ["--nontarget-evidence", "surface"])
    assert BALL in pf.P0M_VERSIONS and "--hsr6-probe" not in pf.version_server_flags(BALL)
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    args = _args(BALL)
    ps.check_unported_flags(args, None)
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    safe = served._policy
    e = safe.ag3s.config.esdf
    assert e.target_extent != "attention_surface" and e.target_exclusion == "ball_owned"
    assert not e.surface_hull_fill and e.surface_contact_allowance_mm is None
    assert safe.subtask_from_latch is True and safe.hsr6 is None and safe._geo is None
    col = safe.to_config.collision
    assert (float(col.neighbour_contact_depth), float(col.neighbour_palm_contact_depth)) == (0.0, 0.0)
    assert int(getattr(safe.to_config.sqp, "multi_seed", 0) or 0) <= 1
    assert safe.to_config.cost.continuity_after_grasp and safe.to_config.cost.deflection_rate_after_grasp
    # the build of the 1판 nontarget evidence is the same option as 1판's
    assert args.nontarget_evidence == pf.parse_v11ab_args(argv=_args_argv_final()).nontarget_evidence
    json.dumps(info, default=str)


def _args_argv_final():
    from AG3S.adapters.panda import profile as pf

    argv = pf.version_server_argv("AG3S-final-panda")
    argv[argv.index("--config") + 1] = "pi05_libero"
    return argv + ["--robot", "panda"] + pf.version_server_flags("AG3S-final-panda")


L8H2, PCLATCH = "AG3S-final-panda-p0m-l8h2", "AG3S-final-panda-v3-e0to-pc-latch"   # SL7d


def test_sl7d_versions_differ_by_exactly_one_change(gpu_ok):
    """SL7d — (가) = P0m with `--attention-source action:last:L8H2`; (나) = pc판 + `--subtask-from-latch`."""
    from AG3S.adapters.panda import profile as pf

    p0m, l8 = list(pf.AG3S_VERSIONS[P0M]), list(pf.AG3S_VERSIONS[L8H2])
    i = p0m.index("--attention-source")
    assert p0m[i + 1] == "noun_target:L8H6" and l8 == p0m[:i + 1] + ["action:last:L8H2"] + p0m[i + 2:]
    pc = list(pf.AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"])
    assert list(pf.AG3S_VERSIONS[PCLATCH]) == pc + ["--subtask-from-latch"] and "--attention-source" not in pc
    assert L8H2 in pf.P0M_VERSIONS and "--hsr6-probe" not in pf.version_server_flags(L8H2)
    fl = pf.version_server_flags(PCLATCH)
    assert pf.is_final_version(PCLATCH) and fl[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]
    assert PCLATCH not in pf.FINAL_CONFIG_VERSIONS and PCLATCH not in pf.P0M_VERSIONS


def test_sl7d_l8h2_builds_with_the_action_cell_as_seed_map(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    args = _args(L8H2)
    ps.check_unported_flags(args, None)
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    safe = served._policy
    e = safe.ag3s.config.esdf
    assert e.target_extent == "attention_surface" and e.surface_hull_fill and e.surface_contact_allowance_mm == 10.0
    assert safe.subtask_from_latch is True and safe.hsr6 is None
    fc = info["final_config"]
    assert fc["attention_surface"]["seed_map"] == "action:last:L8H2"
    kw, rec = ps.attention_policy_kwargs(args, tokenizer=False)
    assert rec["attention_source"] == "action:last:L8H2" and "noun_rule" not in rec


def test_sl7d_pc_latch_builds_with_hsr6_and_the_latch_gate(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    if not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(PCLATCH)
    argv[argv.index("--config") + 1] = "pi05_libero"
    args = pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(PCLATCH))
    assert args.subtask_from_latch and args.hsr6_placement_contact and args.attention_source is None
    ps.check_unported_flags(args, None)
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    safe = served._policy
    assert safe.subtask_from_latch is True and safe.ag3s.config.clustering.subtask_gate is True
    assert safe.ag3s.config.esdf.target_extent != "attention_surface"
    assert info["final_config"]["subtask_from_latch"]["source"] == "latch"


PCSURF, PCNOHSR6 = "AG3S-final-panda-v3-e0to-pc-surface", "AG3S-final-panda-v3-e0to-pc-nohsr6"   # SL7e


def _pc_args(version):
    from AG3S.adapters.panda import profile as pf

    argv = pf.version_server_argv(version)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(version))


def test_sl7e_pc_variants_flags(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    pc = list(pf.AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"])
    surf, nh = list(pf.AG3S_VERSIONS[PCSURF]), list(pf.AG3S_VERSIONS[PCNOHSR6])
    j = pc.index("--nontarget-evidence")
    assert surf == pc[:j] + pc[j + 2:] + ["--attention-source", "action:last:L8H2"] + list(pf.P0M_SURFACE_FLAGS) + list(pf.P0M_HULL_FLAGS)
    assert not any(t.startswith("--hsr6") for t in nh)
    assert [t for t in pc if not t.startswith("--hsr6") and t != "nbc"] == [t for t in nh if t != "nbc"]
    assert pf.version_server_flags(PCSURF)[-2] == "--hsr6-probe" and "--hsr6-probe" not in pf.version_server_flags(PCNOHSR6)


def test_sl7e_pc_variants_build(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    if not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    a = _pc_args(PCSURF)
    ps.check_unported_flags(a, None)
    served, info = ps.build_safe_server(_Stub(), a, gpu_warm_up=False)
    safe = served._policy
    e = safe.ag3s.config.esdf
    assert e.target_extent == "attention_surface" and e.surface_hull_fill and e.nontarget_evidence == "proximity"
    assert safe.hsr6 is not None and info["final_config"]["attention_surface"]["seed_map"] == "action:last:L8H2"
    b = _pc_args(PCNOHSR6)
    ps.check_unported_flags(b, None)
    served, info = ps.build_safe_server(_Stub(), b, gpu_warm_up=False)
    assert served._policy.hsr6 is None and served._policy.ag3s.config.esdf.target_extent != "attention_surface"
