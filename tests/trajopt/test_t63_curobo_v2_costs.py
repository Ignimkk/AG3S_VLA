"""T63 — the extra-cost hook (`curobo_v2/costs.py`). Needs `curobo` importable (`.venv-curobo`);
skipped elsewhere. No CUDA needed: only class composition and the switch are tested."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("curobo")

from AG3S.trajopt.curobo_v2.costs import (  # noqa: E402
    activate_extra_terms, extensible_cost_manager_cfg)


class _Term:
    def __init__(self, name):
        self.name = name
        self.enabled = True
        self.calls = 0

    def setup_batch_tensors(self, batch_size, horizon):
        pass

    def forward(self, state, goal):
        self.calls += 1
        return 0.0


def test_factory_builds_a_cfg_whose_managers_carry_the_terms():
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg

    term = _Term("vla_track")
    cfg_type = extensible_cost_manager_cfg([term])
    assert issubclass(cfg_type, RobotCostManagerCfg)
    cfg = cfg_type.create({})
    assert type(cfg) is cfg_type  # the rollout checks `type(cfg) == instance type`
    assert issubclass(cfg.class_type, RobotCostManager)
    assert cfg.class_type.extra_terms == (term,)
    assert cfg.class_type.extra_active is False


def test_names_must_be_unique_and_not_curobo_own():
    with pytest.raises(ValueError):
        extensible_cost_manager_cfg([_Term("a"), _Term("a")])
    with pytest.raises(ValueError):
        extensible_cost_manager_cfg([_Term("scene_collision")])


def test_activate_switches_only_cost_managers():
    cfg_type = extensible_cost_manager_cfg([_Term("x")])
    manager = cfg_type.manager_type

    class _M:  # stands in for an instance; the switch is an attribute on the instance
        extra_terms = manager.extra_terms
        extra_active = False

    class _Rollout:
        def __init__(self):
            self.cost_manager, self.metrics_cost_manager = _M(), _M()
            self.constraint_manager = _M()

    r = _Rollout()
    assert activate_extra_terms([r]) == 2
    assert r.cost_manager.extra_active and r.metrics_cost_manager.extra_active
    assert not r.constraint_manager.extra_active
