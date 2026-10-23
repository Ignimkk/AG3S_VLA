"""Suite protocol registry (BM-SETUP-A): one table for every LIBERO / SafeLIBERO suite the runner can evaluate.

The data is ``benchmark/configs/suites.json`` (steps, wait, split, success / safety metrics, task names, published
reference numbers, each with its source line). The runners keep their own constants (``libero_pi05_client.MAX_STEPS``,
``libero_safe_client.SAFELIBERO_TOTAL_STEPS`` ...); ``tests/benchmark/test_suite_registry.py`` checks that the two
agree, so the table cannot drift from what actually runs. Python 3.8 compatible, stdlib only.

Split labels: the ``--data-split`` strings of the launcher. LIBERO Spatial dev keeps the launcher's historic default
(``launch_libero_paired.DEFAULT_SPLIT``) and SafeLIBERO Spatial dev the label of the SL5 runs, so new runs carry the
same label as the runs they are paired with.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "benchmark/configs/suites.json"
SPLITS = ("dev", "test")
#: label of the SL5 LIBERO Spatial dev runs (= launch_libero_paired.DEFAULT_SPLIT)
LIBERO_SPATIAL_DEV_LABEL = "dev (LIBERO init 0-4, plan §6.0)"
SAFELIBERO_LEVEL_LABEL_DATE = {"safelibero_spatial": "2026-10-14"}
SPLIT_DATE = "2026-10-23"

_CACHE = {}


def load(path=None):
    path = pathlib.Path(path) if path else REGISTRY_PATH
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = json.loads(path.read_text())
    return _CACHE[key]


def suite(name):
    suites = load()["suites"]
    if name not in suites:
        raise ValueError("unknown suite {!r}; known: {}".format(name, sorted(suites)))
    return suites[name]


def suite_names(in_scope_only=True, family=None):
    return [n for n, s in load()["suites"].items()
            if (s["in_scope"] or not in_scope_only) and (family is None or s["family"] == family)]


def is_safelibero(name):
    return suite(name)["family"] == "safelibero"


def levels(name):
    """``[None]`` for LIBERO suites, ``["I", "II"]`` for SafeLIBERO."""
    return list(suite(name)["levels"])


def policy_steps(name, budget="total_incl_wait"):
    """Policy-step budget the client loops for (``max_steps`` of ``run_episode``)."""
    s = suite(name)
    if s["family"] == "libero":
        return int(s["max_steps"])
    if budget not in s["policy_steps"]:
        raise ValueError("SafeLIBERO step budget must be one of {}, got {!r}".format(sorted(s["policy_steps"]), budget))
    return int(s["policy_steps"][budget])


def total_env_steps(name, budget="total_incl_wait"):
    """Upper bound of env steps per episode, wait steps included."""
    return policy_steps(name, budget) + int(suite(name)["num_steps_wait"])


def split_episodes(split):
    if split not in SPLITS:
        raise ValueError("split must be one of {}, got {!r}".format(SPLITS, split))
    return list(load()["splits"][split])


def _title(name):
    return name.split("_", 1)[1].capitalize()


def split_label(name, level, split):
    """The ``--data-split`` label of ``split`` for ``name`` (and SafeLIBERO ``level``)."""
    eps = split_episodes(split)
    span = "{}-{}".format(eps[0], eps[-1])
    if is_safelibero(name):
        if level not in ("I", "II"):
            raise ValueError("SafeLIBERO suites need level I or II")
        date = SAFELIBERO_LEVEL_LABEL_DATE.get(name, SPLIT_DATE)
        return "{}_SafeLIBERO_{}_ep{}_user_{}".format(split, _title(name), span, date)
    if level is not None:
        raise ValueError("LIBERO suites have no safety level")
    if name == "libero_spatial":
        return LIBERO_SPATIAL_DEV_LABEL if split == "dev" else "test (LIBERO init 5-49, plan §6.0)"
    return "{} (LIBERO {} init {}, user {})".format(split, _title(name) if name != "libero_10" else "10", span,
                                                     SPLIT_DATE)


def parse_split_label(label):
    """``(suite, split)`` of a registry label, else ``None`` (free-form labels are not checked). SafeLIBERO labels
    carry no level (the SL5 label is the same for Level I and II)."""
    for name in suite_names(in_scope_only=True):
        for level in levels(name):
            for split in SPLITS:
                if split_label(name, level, split) == label:
                    return name, split
    return None


def check_selection(name, level, task_ids, episode_ids, data_split=None):
    """Raise ``ValueError`` unless the selection exists in the suite (and fits a registry split label)."""
    s = suite(name)
    if (s["family"] == "safelibero") != (level is not None):
        raise ValueError("--safety-level is required for SafeLIBERO suites and only for them")
    bad_t = [t for t in task_ids if not 0 <= int(t) < int(s["n_tasks"])]
    if bad_t:
        raise ValueError("{}: task ids {} outside 0..{}".format(name, bad_t, int(s["n_tasks"]) - 1))
    bad_e = [e for e in episode_ids if not 0 <= int(e) < int(s["n_init_states"])]
    if bad_e:
        raise ValueError("{}: episode ids {} outside 0..{}".format(name, bad_e, int(s["n_init_states"]) - 1))
    if data_split is not None and data_split != LIBERO_SPATIAL_DEV_LABEL:
        # the launcher's historic default label is left to the launcher's own (unchanged) episode check
        parsed = parse_split_label(data_split)
        if parsed is not None:
            l_name, l_split = parsed
            if l_name != name:
                raise ValueError("--data-split {!r} belongs to {}, not {}".format(data_split, l_name, name))
            outside = sorted(set(int(e) for e in episode_ids) - set(split_episodes(l_split)))
            if outside:
                raise ValueError("--data-split {!r}: episodes {} are not in the {} split".format(data_split, outside, l_split))


def task_name(name, level, task_id):
    return suite(name)["tasks"]["-" if level is None else level][int(task_id)]
