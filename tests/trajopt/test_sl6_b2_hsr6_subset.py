"""SL6 B2 — `hsr6.load_hsr6_probe` accepts an ordered subset of the 6 phases (Panda: no retreat).

Pinned: the shipped RB-Y1 6-class probe loads as the plain `SubtaskProbe` with the same predictions as before
(bit-identical); a 5-class (no retreat) probe loads as `Hsr6SubsetProbe`, its `predict` reports all 6 phases with
retreat = 0.0, the 5 present probabilities = `SubtaskProbe.predict`'s, argmax over the present ones, `phase_of` never
returns retreat (③ never fires); wrong order / unknown class / duplicates / one class are refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.stages.subtask_probe import SubtaskProbe
from AG3S.trajopt import hsr6
from AG3S.trajopt.hsr6 import PHASES6, Hsr6SubsetProbe, is_ordered_subset, load_hsr6_probe, phase_of

FIVE = ("approach", "grasp", "lift", "transport", "placement")


def _write(path, classes, *, seed=0, dim=512):
    rng = np.random.default_rng(seed)
    K = len(classes)
    np.savez(path, classes=np.array(classes), mean=rng.normal(size=dim), scale=rng.uniform(0.5, 2, dim),
             coef=rng.normal(size=(K, dim)) * 0.1, intercept=rng.normal(size=K), C=0.03, feature="kv_L4")
    return path


def test_rby1_probe_unchanged():
    p = load_hsr6_probe()
    assert type(p) is SubtaskProbe and tuple(p.classes) == PHASES6
    ref = SubtaskProbe.load(hsr6.default_probe_path())
    X = np.random.default_rng(1).normal(size=(20, 512)).astype(np.float32)
    assert np.array_equal(p.proba(X), ref.proba(X))
    for x in X:
        assert p.predict(x) == ref.predict(x)
    assert p.record() == ref.record()


def test_five_class_subset_pads_retreat_with_zero(tmp_path):
    path = _write(tmp_path / "p5.npz", FIVE)
    p = load_hsr6_probe(path)
    assert isinstance(p, Hsr6SubsetProbe) and isinstance(p, SubtaskProbe)
    assert p.classes == FIVE and p.absent == ("retreat",) and p.dim == 512
    plain = SubtaskProbe.load(path)
    X = np.random.default_rng(2).normal(size=(50, 512)) * 3
    seen = set()
    for x in X:
        out, ref = p.predict(x), plain.predict(x)
        assert list(out["p"]) == list(PHASES6) and out["p"]["retreat"] == 0.0
        assert {c: out["p"][c] for c in FIVE} == ref["p"] and out["argmax"] == ref["argmax"]
        assert phase_of(out) == ref["argmax"] != "retreat"
        seen.add(out["argmax"])
    assert len(seen) >= 2
    assert np.array_equal(p.proba(X), plain.proba(X))
    assert p.record()["phases_absent"] == ["retreat"] and p.record()["classes"] == list(FIVE)
    # ③ (retreat) never fires on a subset probe without retreat
    assert not any(hsr6.withdraw_target_exemption(phase_of(p.predict(x))) for x in X)


@pytest.mark.parametrize("classes,ok", [
    (FIVE, True), (("approach", "grasp", "transport", "placement"), True), (("grasp", "lift"), True),
    (("grasp", "approach", "lift", "transport", "placement"), False),     # order
    (("approach", "grasp", "lift", "transport", "home"), False),          # unknown
    (("approach", "approach", "lift"), False), (("grasp",), False)])
def test_ordered_subset_rule(tmp_path, classes, ok):
    assert is_ordered_subset(classes) is ok
    path = _write(tmp_path / "p.npz", classes)
    if ok:
        assert load_hsr6_probe(path).classes == tuple(classes)
    else:
        with pytest.raises(ValueError, match="ordered subset"):
            load_hsr6_probe(path)


def test_dim_still_checked(tmp_path):
    with pytest.raises(ValueError, match="dim"):
        load_hsr6_probe(_write(tmp_path / "p.npz", FIVE, dim=256))
