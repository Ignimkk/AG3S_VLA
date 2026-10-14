from AG3S.ag3s.experiments.studies import t60b_return as B


def _m(**kw):
    m = {"hold_after_P": 0, "exec_dev_max": 0.0, "to_dev_max": 0.0, "pred8_ref_min": 1.0,
         "tcp_move_m": 0.02, "tcp_target_final_m": 0.2}
    m.update(kw)
    return m


def test_no_return_proposed_is_default():
    assert B.classify(_m(), 0.35) == "a_no_return_proposed"


def test_hold_after_placement_blocks():
    assert B.classify(_m(hold_after_P=1), 0.35) == "b_blocked"


def test_to_change_blocks_only_when_policy_proposed_return():
    assert B.classify(_m(to_dev_max=0.07, pred8_ref_min=0.2), 0.35) == "b_blocked"
    assert B.classify(_m(to_dev_max=0.07, pred8_ref_min=0.9), 0.35) == "a_no_return_proposed"


def test_retreat_elsewhere_needs_both_distances():
    assert B.classify(_m(tcp_move_m=0.12, tcp_target_final_m=0.15), 0.35) == "c_retreat_elsewhere"
    assert B.classify(_m(tcp_move_m=0.12, tcp_target_final_m=0.05), 0.35) == "a_no_return_proposed"
