"""T43 T — recorded replays (fixed input, CPU). Skipped when the records are not on this machine.

Harness: `outputs/impl/T43T/p/` (`qnow.py` scene from the record — reproduces the recorded verdict
classification on 31/31 held chunks; `recov_replay.py` TO; `ground_replay.py` grounding; `body_emul.py`
the field with the pre-grasp observation freed, **emulated** on the recorded ESDF).

1. Off is bit-identical: the same harness on the pre-fix commit (`682976e`, archived) wrote
   `outputs/impl/T43T/identity_head.npz`; the working tree must reproduce it exactly (TO refined chunks
   of four held deadlock chunks; grounding records of T41 c C/D ep1982 seq 1-47).
2. Fix 3(a): T41 c C/D ep1982 — off, the recorded switch to the hurdle bar at seq 39; on, no switch
   in seq 37-47 (the apple keeps id 0).
3. Step 2 (i) on the T42-1 held deadlocks (ep1908 right, ep1992 left): the attach snapshot from the
   pre-closing geometry executes and lifts (off HOLD); fix 2 (pre-grasp observation freed, 15 mm) also
   executes. Step 2 on closing chunks of T41 a: the TO's push away from the apple is gone and no row
   meets anything outside the carve ball below 0.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np
import pytest

P = pathlib.Path(_ag3s_workspace_path('scripts/testing/t43t'))
HEAD_NPZ = pathlib.Path(_ag3s_workspace_path('legacy/outputs/impl/T43T/identity_head.npz'))
C1982 = _ag3s_workspace_path('legacy/outputs/verify/T41/c/runs/E3b/C/E3b_ep1982_s19822')
D1982 = _ag3s_workspace_path('legacy/outputs/verify/T41/c/runs/E3b/D/E3b_ep1982_s19822')
R1908 = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1908_s19082'))

pytestmark = pytest.mark.skipif(
    not (P.exists() and HEAD_NPZ.exists() and pathlib.Path(C1982).exists() and R1908.exists()),
    reason="T43 T records / harness not on this machine")


@pytest.fixture(scope="module")
def harness():
    sys.path.insert(0, str(P))
    import ground_replay as G   # noqa: E402
    import identity as I        # noqa: E402
    import recov_replay as RR   # noqa: E402
    return G, RR, I


# ------------------------------------------------------------------------------- 1. off identity

def test_off_is_bit_identical_to_the_pre_fix_commit(harness, tmp_path):
    _, _, I = harness
    out = tmp_path / "work.npz"
    I.main(str(out))
    a, b = np.load(HEAD_NPZ), np.load(out)
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        if k == "meta":
            continue
        assert np.array_equal(a[k], b[k]), k
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    for k in ma:
        if k != "benchmark":
            assert ma[k] == mb[k], k


# ------------------------------------------------------------------------------- 2. fix 3(a)

def _ids(G, rd, clustering):
    return {seq: (None if m is None else m.id, None if last is None else last.mode)
            for seq, _, m, last, _ in G.replay(rd, overrides={"clustering": clustering},
                                              seqs=range(1, 48))}


@pytest.mark.parametrize("rd", [C1982, D1982])
def test_3a_keeps_the_apple_where_the_bar_took_over(harness, rd):
    G = harness[0]
    off = _ids(G, rd, {"subtask_gate": False})
    assert off[39] == (1, "switch")                              # the recorded switch to the bar
    on = _ids(G, rd, {"subtask_gate": False, "latched_identity_hold": True})
    assert all(on[s][0] == 0 for s in range(37, 48)), {s: on[s] for s in range(37, 48)}
    assert not any(on[s][1] == "switch" for s in range(37, 48))
    assert all(on[s] == off[s] for s in range(1, 35))            # identical before the latch mattered


# ------------------------------------------------------------------------------- 3. step 2 (i) · fix 2

@pytest.mark.parametrize("key,seq", [("T42_R1908", 16), ("T42_L1992", 16)])
def test_step2_attach_snapshot_from_the_pre_closing_geometry_leaves_the_held_deadlock(harness, key, seq):
    """`closing_keep_geometry`: AG3S.attach rebuilt on the recorded inputs with the last latched chunk's
    points (114 / 112) instead of the closing cap (33 / 34); its held volume freed (emulated held_free).
    The rebuild on the recorded inputs reproduces the recorded spheres (fidelity)."""
    sys.path.insert(0, str(P))
    import familyA_step2 as F  # noqa: E402
    RR = harness[1]
    run = RR.Run(key)
    assert run.solve(seq)["hold"]                                 # off, as recorded
    rd = RR.Q.RUNS[key]
    r2 = F.R2.Run(rd, RR.Q.IX[rd])
    a = r2.attaches()[0]
    held = r2.summ[a]["grasp"]["attached"]["held"]
    import re
    f = int(re.match(r"frame (\d+)", held["snapshot_state"]).group(1))
    pre = max(k for k in sorted(r2.summ) if k < a and r2.summ[k]["grasp"]["state"] == "latched")
    with np.load(r2.npz[a]) as z:
        pts_a = np.asarray(z["target_points"], float)
    with np.load(r2.npz[pre]) as z:
        pts_p = np.asarray(z["target_points"], float)
    _, loc0, _, _ = F.rebuild(r2, a, pts_a, f + 1)
    rec = np.array([c["centre_m"] for c in held["spheres_parent_frame"]])
    assert np.abs(loc0 - rec).max() < 1e-5
    att1, loc1, rad1, _ = F.rebuild(r2, a, pts_p, pre)
    assert len(pts_p) > 3 * len(pts_a)
    G = run.G
    z_min = float(held["support"]["plane"]["offset_m"]) + 0.005

    def scene_fn(s_, s, qn):
        with np.load(run.ix["chunks"][str(s_)]["npz"], allow_pickle=False) as zz:
            z = F.BE.Z({k: zz[k] for k in zz.files})
        T = np.asarray(G["robot"].link_pose(qn, att1.parent_link), float)
        fz, _ = F.BE.emulated_npz(z, list(zip(loc1 @ T[:3, :3].T + T[:3, 3], rad1)), z_min)
        sc, _ = F.BE.scene_with(G, z, s, qn, loc1, rad1, att1.parent_link, fz)
        return sc

    on = run.solve(seq, scene_fn=scene_fn)
    assert not on["hold"], on["reasons"]
    assert on["n_collision"] == 0 and on["n_unresolved"] == 0
    assert RR.lift_mm(run, on)[0] > 5.0                           # the plan lifts the held fruit


@pytest.mark.parametrize("key,seq", [("T42_R1908", 16), ("T42_L1992", 16)])
def test_fix2_frees_the_pre_grasp_observation_and_leaves_the_deadlock(harness, key, seq):
    RR = harness[1]
    run = RR.Run(key)
    r = run.solve(seq, free_obs_pad=0.015)
    assert not r["hold"], r["reasons"]
    assert r["n_collision"] == 0 and r["n_unresolved"] == 0


@pytest.mark.parametrize("key,seq", [("T41/a/D/E3b_ep1807_s18071", 16),
                                     ("T41/a/B/E3b_ep1995_s19952", 19)])
def test_step2_closing_chunk_no_deflection_away_from_the_apple(key, seq):
    """Closing chunks where the recorded TO pushed the fingertips away from the apple (U): with the
    pre-closing geometry carved (emulated target-free layer) the push is gone, and no robot row on
    the recorded main field goes below 0 against anything outside the carve ball."""
    sys.path.insert(0, str(P))
    import closing_replay as C  # noqa: E402
    r = C.one((key, seq, "apple", "left"))
    assert r["apple_in_ball"]["pre"]
    assert r["off"]["radial_mm"] > 10.0                           # away from the apple, as recorded
    assert r["i"]["radial_mm"] < 2.0 and r["i"]["defl_mm"] < 2.0
    assert r["i"]["nonfruit_n_below"] == 0 and r["i"]["nonfruit_min_mm"] > 0.0
