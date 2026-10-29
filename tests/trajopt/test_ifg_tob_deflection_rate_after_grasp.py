"""IFG-TOB — the T43 TA (B) deflection-rate term only from the grasp on.

`cost.deflection_rate_after_grasp` / `serve_safe --deflection-rate-after-grasp` (needs `--w-deflection-rate > 0`), default off.
IFG ep954: a neighbour-fruit avoidance deflection taken while approaching a static apple was carried over five chunks by
the term (row-0 deflection 10 -> 7 -> 3 mm) and the policy's intended grasp point followed the pushed hand.

What this file pins:
1. Config: default False, bool check, `to_dict` / `from_dict` round trip.
2. `SafePolicy._deflection_rate_gate`: a reason on searching / latched / closing, None on held / placed; None when the flag
   is off or the weight is 0.
3. Refiner: with `context["deflection_rate_off"]` the optimizer (and its window optimizer) solves with the weight at 0, the
   config is restored afterwards and the reason is recorded in the metrics; without the key the call is as before.
4. CLI: flag -> cost key; refused without a positive `--w-deflection-rate` and on a `--no-safe` server.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_latch import GraspPhase

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_config_default_check_and_round_trip():
    assert TrajOptConfig().cost.deflection_rate_after_grasp is False
    cfg = TrajOptConfig.from_dict({"cost": {"w_deflection_rate": 20.0, "deflection_rate_after_grasp": True}})
    assert cfg.cost.deflection_rate_after_grasp is True
    assert TrajOptConfig.from_dict(cfg.to_dict()).cost.deflection_rate_after_grasp is True
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"cost": {"deflection_rate_after_grasp": "yes"}})


def _policy(w=20.0, flag=True, phase=GraspPhase.LATCHED):
    from AG3S.trajopt.safe_policy import SafePolicy

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"cost": {"w_deflection_rate": w, "deflection_rate_after_grasp": flag}})
    p._latch = types.SimpleNamespace(phase=phase)
    return p


def test_gate_drops_the_term_only_before_the_grasp():
    for ph in (GraspPhase.SEARCHING, GraspPhase.LATCHED, GraspPhase.CLOSING):
        why = _policy(phase=ph)._deflection_rate_gate()
        assert why is not None and ph.value in why
    for ph in (GraspPhase.HELD, GraspPhase.PLACED):
        assert _policy(phase=ph)._deflection_rate_gate() is None
    assert _policy(flag=False)._deflection_rate_gate() is None
    assert _policy(w=0.0)._deflection_rate_gate() is None


def _refiner(weights_seen):
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from AG3S.trajopt.types import TrajOptResult, TrajOptStatus

    class _Opt:
        def __init__(self, cfg):
            self.config = cfg
            self.window_optimizer = types.SimpleNamespace(config=cfg)

        def solve(self, reference, q_now, scene, **kw):
            weights_seen.append((self.config.cost.w_deflection_rate,
                                 self.window_optimizer.config.cost.w_deflection_rate, sorted(kw)))
            return TrajOptResult(chunk=np.zeros((2, 2)), trajectory=reference, status=TrajOptStatus.FEASIBLE,
                                 iterations=0, solve_time_ms=0.0, max_violation=0.0, reference_violation=0.0,
                                 slack_norm=0.0, cost=0.0, reference_deviation=0.0, metrics={})

    cfg = TrajOptConfig.from_dict({"cost": {"w_deflection_rate": 20.0}})
    r = object.__new__(TrajOptChunkRefiner)
    r.config = cfg
    r.layout = types.SimpleNamespace(chunk_to_trajectory=lambda c: np.zeros((2, c.shape[0])), n_valid=2)
    r.scene_fn = lambda ctx: (object(), np.zeros(2), True)
    r.on_result = None
    r.optimizer = _Opt(cfg)
    r._failure_count = 0
    r._previous = None
    return r


def test_refiner_solves_with_the_weight_at_zero_and_restores_it():
    seen = []
    r = _refiner(seen)
    r.refine(np.zeros((8, 2)), {"t_step": 1, "deflection_history": np.zeros(2)})
    assert seen[-1][:2] == (20.0, 20.0) and "deflection_history" in seen[-1][2]
    assert "deflection_rate_off_by" not in r.last_result.metrics
    r.refine(np.zeros((8, 2)), {"t_step": 2, "deflection_rate_off": "grasp latch latched (test)"})
    assert seen[-1][:2] == (0.0, 0.0) and "deflection_history" not in seen[-1][2]
    assert r.last_result.metrics["deflection_rate_off_by"] == "grasp latch latched (test)"
    assert r.optimizer.config.cost.w_deflection_rate == 20.0
    assert r.optimizer.window_optimizer.config.cost.w_deflection_rate == 20.0


def test_cli_flag_maps_and_is_refused_without_the_weight(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args([*BASE_ARGV, "--w-deflection-rate", "20", "--deflection-rate-after-grasp"])
    over = serve_safe.cost_overrides(args)
    assert over == {"w_deflection_rate": 20.0, "deflection_rate_after_grasp": True}
    assert "deflection_rate_after_grasp" not in serve_safe.cost_overrides(
        serve_safe.build_parser().parse_args([*BASE_ARGV, "--w-deflection-rate", "20"]))
    for extra, needle in ((["--deflection-rate-after-grasp"], "--deflection-rate-after-grasp"),
                          (["--no-safe", "--w-deflection-rate", "20", "--deflection-rate-after-grasp"], "--w-*")):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert needle in capsys.readouterr().err
