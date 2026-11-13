"""SOTA clearance candidates (dev-57, 2026-11-11): `-to5` / `-to0` (+ `-tro`) are the base
version's argv plus the named flags only; the evaluated bases are unchanged; the server parser takes the later value."""
from AG3S.adapters.panda import profile as pf

BASE = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-"


def test_names_and_flags():
    assert len(pf.SOTA_VERSIONS) == (8 + len(pf.SOTA_OM_VERSIONS) + len(pf.SOTA_NTC_VERSIONS) + len(pf.SOTA_STALL_VERSIONS)
                                     + len(pf.SOTA_GATE_VERSIONS) + len(pf.SOTA_HELD_VERSIONS)
                                     + len(pf.SOTA_SCALED_VERSIONS) + len(pf.SOTA_HANDB_VERSIONS)
                                     + len(pf.SOTA_PEAK_VERSIONS) + len(pf.SOTA_ROTREL_VERSIONS)
                                     + len(pf.SOTA_REACHSTICKY_VERSIONS) + len(pf.SOTA_ZONE_VERSIONS)
                                     + len(pf.SOTA_ZONEPALM_VERSIONS) + len(pf.SOTA_ZONEPALM5_VERSIONS)
                                     + len(pf.SOTA_DEADCAM_VERSIONS) + len(pf.SOTA_HELDBAND_VERSIONS) + len(pf.SOTA_LSNATIVE_VERSIONS) + len(pf.SOTA_HANDSIZED_VERSIONS) + len(pf.SOTA_HANDSIZED_ATT_VERSIONS) + len(pf.SOTA_GTO_VERSIONS) + len(pf.SOTA_SPLITWIDE_VERSIONS))
    assert set(pf.SOTA_VERSIONS) <= set(pf.GT15_VERSIONS)
    for b in (BASE + "pe123458", BASE + "pe12345811"):
        base = tuple(pf.AG3S_VERSIONS[b])
        assert "--ee-action-qp-translation-only" not in base and base.count("--passthrough-obstacle-tolerance-mm") == 1
        for suffix, flags in pf._SOTA_SUFFIXES.items():
            assert tuple(pf.AG3S_VERSIONS[b + suffix]) == base + tuple(flags)
            assert tuple(pf.AG3S_VERSIONS[b + suffix + "-tro"]) == base + tuple(flags) + ("--ee-action-qp-translation-only",)
            assert pf.version_server_flags(b + suffix)   # accepted by the Panda profile (final-base rules)


def test_later_value_wins_in_the_server_parser():
    from AG3S.trajopt import serve_safe as ss
    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV
    a = ss.build_parser().parse_args(ARGV + ["--passthrough-obstacle-tolerance-mm", "10",
                                              "--passthrough-obstacle-tolerance-mm", "0"])
    assert float(a.passthrough_obstacle_tolerance_mm) == 0.0


def test_v11ab_record_keys_unchanged():
    fam = pf.version_flag_values(pf.parse_v11ab_args())
    assert "palm_shell" not in fam and "passthrough_obstacle_tolerance_mm" not in fam


def test_no_target_certify_versions():
    assert len(pf.SOTA_NTC_VERSIONS) == 24 and len(set(pf.SOTA_VERSIONS)) == len(pf.SOTA_VERSIONS)
    for name in pf.SOTA_NTC_VERSIONS:
        parent = name[:-len("-tro")] if name.endswith("-tro") else name[:-len("-ntc")]
        extra = "--ee-action-qp-translation-only" if name.endswith("-tro") else "--no-target-certify"
        assert tuple(pf.AG3S_VERSIONS[name]) == tuple(pf.AG3S_VERSIONS[parent]) + (extra,)
        assert tuple(pf.AG3S_VERSIONS[name]).count("--no-target-certify") == 1
        assert pf.version_server_flags(name)


def test_stall_versions():
    assert len(pf.SOTA_STALL_VERSIONS) == 6 * 2 * 16 and len(set(pf.SOTA_VERSIONS)) == len(pf.SOTA_VERSIONS)
    name = BASE + "pe12345811-to0-om15p10s40-ntc-pt6-pg4-tro"
    assert name in pf.SOTA_STALL_VERSIONS
    v = tuple(pf.AG3S_VERSIONS[name])
    assert v[-5:] == ("--obstacle-row-patience", "6", "--ee-action-qp-progress", "0.4", "--ee-action-qp-translation-only")
    for n in pf.SOTA_STALL_VERSIONS:
        assert pf.version_server_flags(n)
    fam = pf.version_flag_values(pf.parse_v11ab_args())
    assert "ee_action_qp_progress" not in fam and "obstacle_row_patience" not in fam


def test_scaled_qp_versions():
    """SL-SOTA6: ``-qs`` = the scaled action QP (SL-GTTO11 F9a) on the SOTA versions; flag only, default off."""
    from AG3S.adapters.panda import server as srv
    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-"
    name = base + "pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-qs-tro"
    assert name in pf.SOTA_SCALED_VERSIONS and set(pf.SOTA_SCALED_VERSIONS) <= set(pf.GT15_VERSIONS)
    assert tuple(pf.AG3S_VERSIONS[name])[-2:] == ("--ee-action-qp-scaled", "--ee-action-qp-translation-only")
    assert tuple(pf.AG3S_VERSIONS[name])[:-2] == tuple(pf.AG3S_VERSIONS[name.replace("-qs-tro", "")])
    for n in (base + "pe123458-qs", base + "pe12345811-to0-om15p10s40-ntc-qs-tro", base + "pe12345811-to0-om15p10s40-ntc-pt6-pg8-hr150-qs-tro"):
        assert n in pf.SOTA_SCALED_VERSIONS
    for n in pf.SOTA_SCALED_VERSIONS:
        assert pf.version_server_flags(n)
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(pf.AG3S_VERSIONS[name]))
    assert args.ee_action_qp_scaled is True and "ee_action_qp_scaled" in srv.PORTED_FINAL_CONFIG_DESTS
    assert "--ee-action-qp-scaled" not in pf.AG3S_VERSIONS[name.replace("-qs-tro", "-tro")]


def test_attention_peak_versions():
    """SL-SOTA9: ``-ap`` = SL-PE9 (--attention-peak-seed 5) on the PE11-based ...-hr150[-hb]-qs versions."""
    from AG3S.trajopt import serve_safe as ss

    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150"
    for mid in ("-qs", "-hb-qs"):
        name = base + mid + "-ap-tro"
        assert name in pf.SOTA_PEAK_VERSIONS and name in pf.GT15_VERSIONS
        v = tuple(pf.AG3S_VERSIONS[name])
        assert v[-3:] == ("--attention-peak-seed", "5", "--ee-action-qp-translation-only")
        assert v[:-3] == tuple(pf.AG3S_VERSIONS[base + mid])
        assert "--scene-obstacle-target-cut" in v                      # PE9 only with PE11
        p = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(v))
        assert ss.t43t_identity_section(p)["attention_peak_seed"] == 5.0
    assert all("pe12345811-" in n and "-hr150" in n for n in pf.SOTA_PEAK_VERSIONS)
    for n in pf.SOTA_PEAK_VERSIONS:
        assert pf.version_server_flags(n)


def test_dead_camera_seen_ok_versions():
    """SL-SOTA16 `-dc`: the frozen version's argv + SL-QE1 `--esdf-dead-camera-seen-ok`, nothing else; off everywhere else."""
    from AG3S.adapters.panda import server as srv
    frozen = BASE + "pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs-tro"
    name = frozen[:-len("-tro")] + "-dc-tro"
    assert name in pf.SOTA_DEADCAM_VERSIONS and name in pf.SOTA_VERSIONS and name in pf.GT15_VERSIONS
    assert len(pf.SOTA_DEADCAM_VERSIONS) == len(pf.SOTA_REACHSTICKY_VERSIONS)
    v = tuple(pf.AG3S_VERSIONS[name])
    assert v[-1] == "--esdf-dead-camera-seen-ok" and v[:-1] == tuple(pf.AG3S_VERSIONS[frozen])
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(v))
    assert args.esdf_dead_camera_seen_ok is True
    assert pf.version_server_flags(name)
    assert "esdf_dead_camera_seen_ok" in srv.PORTED_FINAL_CONFIG_DESTS
    for other, flags in pf.AG3S_VERSIONS.items():
        if other not in pf.SOTA_DEADCAM_VERSIONS and not (other in pf.SOTA_HELDBAND_VERSIONS and "-dc-" in other):
            assert "--esdf-dead-camera-seen-ok" not in flags, other

