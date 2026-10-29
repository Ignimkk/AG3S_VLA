"""Recorded index JSONs name inputs under the retired /mnt/dev/work checkout; `rebase_legacy_paths`
maps those keys and nested values onto the shared legacy input root without touching anything else."""
from common.workspace import (OLD_CHECKOUT_PREFIX, bind_historical_client_inputs, legacy_input_path,
                              rebase_legacy_path, rebase_legacy_paths)


def test_rebase_maps_keys_and_nested_values_onto_legacy_root():
    run = "outputs/verify/T40/G/runs/E3b/C/E3b_ep1968_s19681"
    npz = "outputs/verify/T40/G/runs/E3b/server_1/server_constraints/run_0000/chunk_00000.npz"
    idx = {OLD_CHECKOUT_PREFIX + run: {"server": OLD_CHECKOUT_PREFIX + "outputs/verify/T40/G/runs/E3b/server_1",
                                       "n_plan": 75, "complete": True,
                                       "chunks": {"1": {"npz": OLD_CHECKOUT_PREFIX + npz, "plan_t_step": 0}},
                                       "list": [OLD_CHECKOUT_PREFIX + npz, 3]}}
    out = rebase_legacy_paths(idx)
    key = str(legacy_input_path(run))
    assert list(out) == [key]
    assert out[key]["server"] == str(legacy_input_path("outputs/verify/T40/G/runs/E3b/server_1"))
    assert out[key]["chunks"]["1"]["npz"] == str(legacy_input_path(npz))
    assert out[key]["list"] == [str(legacy_input_path(npz)), 3]
    assert out[key]["n_plan"] == 75 and out[key]["complete"] is True and out[key]["chunks"]["1"]["plan_t_step"] == 0
    assert OLD_CHECKOUT_PREFIX not in repr(out)
    assert OLD_CHECKOUT_PREFIX in repr(idx)                       # input is not mutated


def test_rebase_leaves_other_strings_alone():
    for s in ("/mnt/dev/workspace/x", "relative/path", "/mnt/dev/AG3S_VLA-storage/a", "T41/a/runs"):
        assert rebase_legacy_path(s) == s
    assert rebase_legacy_paths({"a": "/tmp/x", 1: None}) == {"a": "/tmp/x", 1: None}


def test_client_checkpoint_binding_still_uses_the_same_mapping():
    class C:
        MODELS = {"m": {"checkpoint": OLD_CHECKOUT_PREFIX + "checkpoints/x/30000"},
                  "n": {"checkpoint": "/elsewhere/ckpt"}}

    bind_historical_client_inputs(C)
    assert C.MODELS["m"]["checkpoint"] == str(legacy_input_path("checkpoints/x/30000"))
    assert C.MODELS["n"]["checkpoint"] == "/elsewhere/ckpt"
