"""``libero_safe_client.py --filter-every-step`` (F2, 2026-11-10): off = byte-identical client; on = the official loop
asks every env step, the policy requests are the same requests at the same env steps.

The client runs in ``.venv-libero`` (Python 3.8) through ``client_harness.py`` (fake env / geometry / websocket); this
test runs from ``.venv-openpi-live`` like ``test_client_privileged.py``::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD:$PWD/third_party/openpi/src \
        .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_filter_rate_client.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from Baseline.paper_baselines.closed_loop.tests.test_client_privileged import HARNESS, PY38, ROOT, harness

#: the client before F2 (HEAD e52f3fea: KNOWS F1 + AEGIS naming, 2026-11-09)
PRE_F2 = "e52f3fea"


@pytest.mark.parametrize("mode,seg,emit", [("vla_only", False, False), ("vla_v11ab", False, False),
                                           ("vla_v11ab", True, True)])
def test_flag_off_is_byte_identical_to_the_pre_f2_client(tmp_path, mode, seg, emit):
    pre = harness(tmp_path / "pre", PRE_F2, mode, emit=emit, seg=seg)
    new = harness(tmp_path / "new", "worktree", mode, emit=emit, seg=seg)
    assert new["requests"] == pre["requests"]  # keys, order, msgpack bytes per key and per request
    assert new["save"] == pre["save"]
    assert new["calls_jsonl_sha"] == pre["calls_jsonl_sha"]
    assert new["calls_npz"] == pre["calls_npz"]
    a, b = pre["parsed_sl5c_argv"], dict(new["parsed_sl5c_argv"])
    assert b.pop("filter_every_step") is False and b == a  # the only new argument, off by default


def loop(tmp, kind) -> dict:
    if not PY38.exists():
        pytest.skip(".venv-libero missing")
    sys.path.insert(0, str(ROOT))
    from benchmark.adapters.libero_panda.client_env import client_environment

    env = client_environment(dict(os.environ), ROOT)
    env["OMP_NUM_THREADS"] = "1"
    out = subprocess.run([str(PY38), str(HARNESS), "--source", "worktree", "--loop", kind, "--out", str(tmp)],
                         capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_step_mode_keeps_the_policy_requests_and_filters_every_step(tmp_path):
    c, s = loop(tmp_path / "c", "chunk"), loop(tmp_path / "s", "step")
    pc = [r for r in c["requests"] if r["kind"] == "policy"]
    ps = [r for r in s["requests"] if r["kind"] == "policy"]
    assert len(c["requests"]) == 4 and len(s["requests"]) == 17 == s["env_steps"] - 3  # 17 policy steps, 3 wait
    # the policy requests: same count, seq, reset, seed and the same bytes of every key but ag3s/filter_step
    assert [(r["seq"], r["reset"], r["policy_seed"], r["sha_policy_keys"]) for r in pc] == \
           [(r["seq"], r["reset"], r["policy_seed"], r["sha_policy_keys"]) for r in ps]
    assert all(r["keys"][-1] == "ag3s/filter_step" and r["k"] == 0 for r in ps)
    assert all("ag3s/filter_step" not in r["keys"] for r in c["requests"])
    # every other step: a filter-step request of the same chunk (seq of the last policy call, k = 1..4, no seed/reset)
    fs = [r for r in s["requests"] if r["kind"] == "filter"]
    assert [r["k"] for r in s["requests"]] == ([0, 1, 2, 3, 4] * 4)[:17]
    assert all(r["seq"] == i // 4 and not r["reset"] and r["policy_seed"] is None for i, r in enumerate(fs))
    # executed: chunk mode rows 0-4 of each chunk; step mode the server's row of each step (fake filter: +1e-3)
    ex_c, ex_s = c["executed"][3:], s["executed"][3:]
    assert len(ex_c) == len(ex_s) == 17
    assert all(abs(a - b - 1e-3) < 1e-6 for rc, rs in zip(ex_c, ex_s) for a, b in zip(rs, rc))
    assert s["save"]["filter_every_step"] == {"env_steps": 17, "policy_calls": 4, "replan_steps": 5}
    assert "filter_every_step" not in c["save"]
    assert s["client_arrays"]["actions_executed"] == c["client_arrays"]["actions_executed"] == [4, 5, 7]
    assert s["client_arrays"]["step_actions"] == [17, 7]
