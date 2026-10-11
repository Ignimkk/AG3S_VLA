import numpy as np

from openpi import transforms
from openpi.policies import aloha_policy
from openpi.training import config as training_config


def test_16d_aloha_passthrough_and_output_slice():
    state = np.arange(16, dtype=np.float32)
    inputs = aloha_policy.AlohaInputs(adapt_to_pi=False)({
        "state": state.copy(),
        "images": {"cam_high": np.zeros((3, 4, 4), dtype=np.uint8)},
    })
    assert np.array_equal(inputs["state"], state)
    outputs = aloha_policy.AlohaOutputs(adapt_to_pi=False, action_dim=16)({
        "actions": np.arange(64, dtype=np.float32).reshape(2, 32),
    })
    assert outputs["actions"].shape == (2, 16)


def test_16d_delta_mask_keeps_grippers_absolute():
    state = np.arange(16, dtype=np.float32)
    actions = np.stack([state + 1.0, state + 2.0])
    delta = transforms.DeltaActions(transforms.make_bool_mask(7, -1, 7, -1))({
        "state": state,
        "actions": actions,
    })["actions"]
    assert np.allclose(delta[:, :7], [[1.0] * 7, [2.0] * 7])
    assert np.allclose(delta[:, 8:15], [[1.0] * 7, [2.0] * 7])
    assert np.array_equal(delta[:, [7, 15]], actions[:, [7, 15]])


def test_16d_is_padded_to_pi05_32d_then_restored():
    state = np.arange(16, dtype=np.float32)
    actions = np.arange(32, dtype=np.float32).reshape(2, 16)
    padded = transforms.PadStatesAndActions(32)({
        "state": state.copy(),
        "actions": actions.copy(),
    })
    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (2, 32)
    assert np.array_equal(padded["state"][:16], state)
    assert np.array_equal(padded["actions"][:, :16], actions)
    restored = aloha_policy.AlohaOutputs(adapt_to_pi=False, action_dim=16)({
        "actions": padded["actions"],
    })["actions"]
    assert np.array_equal(restored, actions)


def test_rby1_16d_training_config_and_legacy_default():
    assert training_config.LeRobotAlohaDataConfig().arm_joint_dim == 6
    config = training_config.get_config("pi05_rby1_randomized_pick_place_16d_lora")
    assert config.data.arm_joint_dim == 7
    assert config.data.adapt_to_pi is False
    assert tuple(config.data.base_config.episodes) == tuple(range(1600))
