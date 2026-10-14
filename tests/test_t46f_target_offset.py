"""T46f pilot — `pi05_infer.apply_target_offset_xy` moves only the target's initial xy and leaves the record intact."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("_t46f_pi05_infer", ROOT / "integration/rby1/bringup/pi05_infer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_apply_target_offset_moves_only_target_xy():
    mod = _load()
    rec = {"target_fruit": "pear",
           "initial_fruit_poses": {"pear": [0.5, 0.3, 0.85, 1, 0, 0, 0], "apple": [0.45, 0.3, 0.85, 1, 0, 0, 0]},
           "basket_pose": [0.5, 0.0, 0.88, 1, 0, 0, 0]}
    out = mod.apply_target_offset_xy(rec, (0.01, -0.06))
    assert out["initial_fruit_poses"]["pear"] == [0.51, 0.24, 0.85, 1.0, 0.0, 0.0, 0.0]
    assert out["initial_fruit_poses"]["apple"] == rec["initial_fruit_poses"]["apple"]
    assert out["basket_pose"] == rec["basket_pose"]
    assert rec["initial_fruit_poses"]["pear"] == [0.5, 0.3, 0.85, 1, 0, 0, 0]   # original untouched
