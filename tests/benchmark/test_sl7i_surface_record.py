"""SL7i — the per-call attention-surface record of the Panda server (`server.surface_call_record` -> response key
`panda/surface` -> client row `surface`), logging only; absent for versions without the attention-surface extent."""
from __future__ import annotations

import types

import numpy as np


def _safe(extent, frame):
    esdf = types.SimpleNamespace(target_extent=extent)
    ag3s = types.SimpleNamespace(config=types.SimpleNamespace(esdf=esdf), _surface_frame=frame)
    return types.SimpleNamespace(ag3s=ag3s)


def test_no_key_without_the_surface_extent():
    from AG3S.adapters.panda.server import surface_call_record

    assert surface_call_record(_safe("estimated", {"applied": True})) is None
    assert surface_call_record(types.SimpleNamespace()) is None


def test_states_and_scalars_only():
    from AG3S.adapters.panda.server import surface_call_record

    assert surface_call_record(_safe("attention_surface", None)) == {"state": "none"}
    frame = {"applied": False, "why": "no_overlap", "n_nodes": 12, "codes": np.zeros((4, 4)),
             "t78": {"target_check": {"ok": False, "why": "no_overlap", "overlap": 0.0, "pts": [1, 2]},
                     "region": {"n_nodes": 12}, "other": {"x": 1}}}
    rec = surface_call_record(_safe("attention_surface", frame))
    assert rec["state"] == "fallback" and rec["why"] == "no_overlap" and rec["n_nodes"] == 12 and "codes" not in rec
    assert rec["target_check"] == {"ok": False, "why": "no_overlap", "overlap": 0.0} and rec["region"] == {"n_nodes": 12}
    assert "other" not in rec and "t78" not in rec
    rec = surface_call_record(_safe("attention_surface", {"applied": True, "t78": {}}))
    assert rec == {"applied": True, "state": "surface"}


def test_client_writes_the_row_only_when_present():
    src = open("benchmark/runners/libero_safe_client.py").read()
    assert 'if "panda/surface" in resp:' in src and 'row["surface"] = base._jsonable(resp.get("panda/surface"))' in src
