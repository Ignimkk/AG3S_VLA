"""BM-SETUP-A: ``benchmark/configs/suites.json`` agrees with the code that runs and with the official sources (CPU)."""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def reg():
    from benchmark.runners import suite_registry as reg

    return reg


def test_libero_steps_match_client_and_openpi_example(reg):
    from benchmark.runners import libero_pi05_client as base

    src = (ROOT / "third_party/openpi/examples/libero/main.py").read_text()
    official = {m.group(1): int(m.group(2)) for m in
                re.finditer(r'task_suite_name == "(\w+)":\s*\n\s*max_steps = (\d+)', src)}
    wait = int(re.search(r"num_steps_wait: int = (\d+)", src).group(1))
    for name in reg.suite_names(in_scope_only=False, family="libero"):
        s = reg.suite(name)
        assert s["max_steps"] == base.MAX_STEPS[name] == official[name]
        assert s["num_steps_wait"] == base.DEFAULT_NUM_STEPS_WAIT == wait
        assert reg.policy_steps(name) == s["max_steps"]
    assert sorted(official) == sorted(base.MAX_STEPS)


def test_safelibero_steps_match_client_and_main_aegis(reg):
    from benchmark.runners import libero_safe_client as sc

    src = (ROOT / "benchmark/SafeLIBERO/main/main_aegis.py").read_text()
    official = {m.group(1): int(m.group(2)) for m in
                re.finditer(r'task_suite_name == "(\w+)":\s*\n\s*max_steps = (\d+)', src)}
    wait = int(re.search(r"num_steps_wait: int = (\d+)", src).group(1))
    assert sorted(official) == sorted(sc.SAFELIBERO_TOTAL_STEPS) == sorted(reg.suite_names(family="safelibero"))
    for name, total in official.items():
        s = reg.suite(name)
        assert s["total_steps"] == sc.SAFELIBERO_TOTAL_STEPS[name] == total
        assert s["num_steps_wait"] == sc.SAFELIBERO_NUM_STEPS_WAIT == wait == 20
        assert reg.policy_steps(name) == sc.safelibero_policy_steps(name) == total - wait  # SL5c default
        assert reg.policy_steps(name, "main_aegis") == sc.safelibero_policy_steps(name, "main_aegis") == total
    from benchmark.runners import launch_libero_paired as lp

    assert lp.SAFELIBERO_STEP_BUDGETS == sc.SAFELIBERO_STEP_BUDGETS
    assert lp.SAFELIBERO_DEFAULT_STEP_BUDGET == sc.SAFELIBERO_DEFAULT_STEP_BUDGET == "total_incl_wait"


def test_main_aegis_resets_t_after_the_wait():
    """The finding behind ``--safelibero-step-budget main_aegis``: t restarts at 0 before ``while t < max_steps``."""
    lines = (ROOT / "benchmark/SafeLIBERO/main/main_aegis.py").read_text().splitlines()
    wait = next(i for i, l in enumerate(lines) if l.strip() == "while t < args.num_steps_wait:")
    loop = next(i for i, l in enumerate(lines) if l.strip() == "while t < max_steps:")
    resets = [i for i in range(wait + 1, loop) if lines[i].strip() == "t = 0"]
    assert resets, "no t = 0 between the wait loop and the policy loop"
    indent = len(lines[loop]) - len(lines[loop].lstrip())
    assert len(lines[resets[-1]]) - len(lines[resets[-1]].lstrip()) == indent  # same block: unconditional
    assert (wait + 1, resets[-1] + 1, loop + 1) == (166, 234, 254)  # line numbers cited in suites.json / client


def test_task_names_match_the_benchmark_sources(reg):
    from benchmark.adapters.libero_panda import task_target as tt

    for name in reg.suite_names(family="libero"):
        assert reg.suite(name)["tasks"]["-"] == tt.suite_task_names(name)
    sl = ROOT / "benchmark/SafeLIBERO/safelibero/libero/libero"
    for name in reg.suite_names(family="safelibero"):
        for level in ("I", "II"):
            names = reg.suite(name)["tasks"][level]
            assert len(names) == reg.suite(name)["n_tasks"] == 4
            for task in names:
                assert (sl / "bddl_files" / name / (task + ".bddl")).is_file()
                assert (sl / "init_files" / name / f"{task}_level_{level}.pruned_init").is_file()
    goal = reg.suite("safelibero_goal")["tasks"]
    assert goal["I"][:3] == goal["II"][:3] and goal["I"][3] != goal["II"][3]


def test_split_labels_and_selection_checks(reg):
    from benchmark.runners import launch_libero_paired as lp

    assert reg.split_episodes("dev") == [0, 1, 2, 3, 4] and reg.split_episodes("test") == list(range(5, 50))
    assert reg.split_label("libero_spatial", None, "dev") == lp.DEFAULT_SPLIT
    # the SL5 SafeLIBERO Spatial runs carry this label (both levels)
    assert reg.split_label("safelibero_spatial", "I", "dev") == "dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14"
    assert reg.split_label("safelibero_spatial", "II", "dev") == reg.split_label("safelibero_spatial", "I", "dev")
    for name in reg.suite_names():
        for level in reg.levels(name):
            for split in reg.SPLITS:
                label = reg.split_label(name, level, split)
                assert re.fullmatch(r"[^\n]+", label)
                reg.check_selection(name, level, [0], reg.split_episodes(split), label)
    with pytest.raises(ValueError, match="not in the dev split"):
        reg.check_selection("libero_goal", None, [0], [5], reg.split_label("libero_goal", None, "dev"))
    with pytest.raises(ValueError, match="belongs to"):
        reg.check_selection("libero_goal", None, [0], [0], reg.split_label("libero_object", None, "dev"))
    with pytest.raises(ValueError, match="task ids"):
        reg.check_selection("safelibero_long", "I", [4], [0])
    with pytest.raises(ValueError, match="episode ids"):
        reg.check_selection("libero_10", None, [0], [50])
    with pytest.raises(ValueError, match="safety-level"):
        reg.check_selection("safelibero_goal", None, [0], [0])
    reg.check_selection("libero_goal", None, [0], [0, 4], lp.DEFAULT_SPLIT)  # historic default label: launcher's check
    assert not reg.suite("libero_90")["in_scope"] and "libero_90" not in reg.suite_names()


def test_probe_esdf_box_is_the_panda_profile_box():
    from AG3S.adapters.panda import profile

    tree = ast.parse((ROOT / "benchmark/evaluators/suite_probe.py").read_text())
    consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and t.id.startswith("ESDF_BOX")}
    lo, hi = profile.panda_esdf_bounds()
    assert consts["ESDF_BOX_LOWER"] == pytest.approx(lo) and consts["ESDF_BOX_UPPER"] == pytest.approx(hi)


def test_fixture_aliases_resolve_the_support_bodies():
    from benchmark.adapters.libero_panda import task_target as tt

    bodies = ["world", "table", "floor", "study_table", "living_room_table", "living_room_table_col",
              "wooden_cabinet_1_main", "wooden_cabinet_1_base"]
    assert tt.instance_body_ids("main_table", bodies) == [1]          # LIBERO Spatial: unchanged
    assert tt.instance_body_ids("kitchen_table", bodies) == [1]
    assert tt.instance_body_ids("floor", bodies) == [2]
    assert tt.instance_body_ids("study_table", bodies) == [3]
    assert tt.instance_body_ids("living_room_table", bodies) == [5]    # prefix rule (the _col body)
    assert tt.instance_body_ids("wooden_cabinet_1", bodies) == [6, 7]
