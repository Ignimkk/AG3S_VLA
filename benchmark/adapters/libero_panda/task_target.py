"""LIBERO task target (BDDL) -> segmentation instance (SL3). Python 3.8 compatible, numpy only.

ATS needs to know, per task, which scene object is the **target** (the object the policy must grasp)
so an attention map can be scored against that object's pixels. LIBERO states it in the task's BDDL
problem file:

* ``(:obj_of_interest A B)`` -- the objects the task is about (LIBERO Spatial: the bowl to move and
  the plate it goes on);
* ``(:goal (And (On A B)))`` -- which of them moves: the first argument of the goal predicate.

:func:`task_target` reads both and names the target (``A``) and the destination (``B``). The parser is
the tokenizer of LIBERO's own ``bddl.parsing.scan_tokens`` (comments dropped, **lower-cased**,
parenthesis tree) and the section rules of ``libero.libero.envs.bddl_utils.robosuite_parse_problem``,
restated so it runs in ``.venv-ag3s`` / ``.venv-openpi-live`` without LIBERO / torch;
``tests/benchmark/libero_py38_checks.py`` (``bddl_target_matches_libero_parser``) compares it with
LIBERO's parser and task order for every LIBERO Spatial task.

Task order: ``benchmark.get_benchmark_dict()[suite]()`` uses ``task_orders[0]`` = identity over
``libero_suite_task_map[suite]`` (both read here from the LIBERO source by AST, not copied).

Segmentation link: the adapter's ``segmentation_geom/<cam>`` holds a MuJoCo **geom id** per pixel
(SL1); ``model_geom_bodyid`` / ``model_body_names`` in the ``_geom.npz`` give each geom's body. A
LIBERO object ``X`` is the bodies named ``X_main`` and ``X_<part>`` (``wooden_cabinet_1_base``,
``..._cabinet_top``, ...); the fixture ``main_table`` is the arena body ``table``
(:data:`FIXTURE_BODY_ALIASES`). :func:`instance_geom_ids` / :func:`instance_mask` turn an object name
into geom ids / a pixel mask.
"""
from __future__ import annotations

import ast
import hashlib
import pathlib
import re

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
LIBERO_ROOT = PROJECT_ROOT / "benchmark/LIBERO/libero/libero"
#: = ``get_libero_path("bddl_files")`` with this checkout's ``.runtime/libero/config.yaml`` (py3.8 check).
BDDL_ROOT = LIBERO_ROOT / "bddl_files"
TASK_MAP_FILE = LIBERO_ROOT / "benchmark/libero_suite_task_map.py"
BENCHMARK_INIT_FILE = LIBERO_ROOT / "benchmark/__init__.py"
#: BDDL fixtures whose MuJoCo body does not carry the fixture name (robosuite arena table).
FIXTURE_BODY_ALIASES = {"main_table": ("table",)}
#: Goal predicates whose first argument is the object that is moved / acted on.
OBJECT_FIRST_PREDICATES = ("on", "in")


# --- BDDL -------------------------------------------------------------------------------------------
def scan_tokens(text):
    """LIBERO ``bddl.parsing.scan_tokens``: drop ``;`` comments, lower-case, parenthesis tree."""
    text = re.sub(r";.*$", "", text, flags=re.MULTILINE).lower()
    stack, tokens = [], []
    for t in re.findall(r"[()]|[^\s()]+", text):
        if t == "(":
            stack.append(tokens)
            tokens = []
        elif t == ")":
            if not stack:
                raise ValueError("BDDL: missing open parenthesis")
            done, tokens = tokens, stack.pop()
            tokens.append(done)
        else:
            tokens.append(t)
    if stack:
        raise ValueError("BDDL: missing close parenthesis")
    if len(tokens) != 1:
        raise ValueError("BDDL: malformed expression")
    return tokens[0]


def _typed_list(group):
    """``a b - cat c - cat2`` -> ``{cat: [a, b], cat2: [c]}`` (untyped tail under the default key)."""
    out, pending = {}, []
    group = list(group)
    while group:
        if group[0] == "-":
            group.pop(0)
            out[group.pop(0)] = pending
            pending = []
        else:
            pending.append(group.pop(0))
    return out, pending


def parse_bddl(source):
    """A LIBERO BDDL problem (path or text) -> dict, the fields ``robosuite_parse_problem`` returns that
    SL3 uses: ``language`` (str), ``objects`` / ``fixtures`` ({category: [names]}), ``obj_of_interest``
    ([names]), ``goal`` ([[predicate, args...]]), plus ``object_names`` / ``fixture_names`` (flat)."""
    text = pathlib.Path(source).read_text() if isinstance(source, pathlib.Path) or (
        isinstance(source, str) and "(" not in source) else str(source)
    tokens = scan_tokens(text)
    if not (isinstance(tokens, list) and tokens and tokens[0] == "define"):
        raise ValueError("BDDL: not a (define ...) problem")
    out = {"language": "", "objects": {}, "fixtures": {}, "obj_of_interest": [], "goal": []}
    for group in tokens[1:]:
        if not isinstance(group, list) or not group:
            continue
        head = group[0]
        if head == ":objects":
            typed, rest = _typed_list(group[1:])
            if rest:
                typed.setdefault("object", []).extend(rest)
            out["objects"] = typed
        elif head == ":fixtures":
            typed, rest = _typed_list(group[1:])
            if rest:
                typed.setdefault("fixture", []).extend(rest)
            out["fixtures"] = typed
        elif head == ":obj_of_interest":
            out["obj_of_interest"] = list(group[1:])
        elif head == ":language":
            out["language"] = " ".join(group[1:])
        elif head == ":goal":
            body = group[1]
            out["goal"] = [list(p) for p in (body[1:] if body and body[0] == "and" else [body])]
    out["object_names"] = [n for names in out["objects"].values() for n in names]
    out["fixture_names"] = [n for names in out["fixtures"].values() for n in names]
    return out


def goal_roles(problem):
    """``(target, destination)`` from the goal: the first goal predicate whose first argument is an object
    of interest. ``On/In`` -> (arg1, arg2); a unary predicate (``Open``, ``TurnOn``) -> (arg1, None)."""
    interest = list(problem["obj_of_interest"])
    for predicate in problem["goal"]:
        if len(predicate) >= 2 and predicate[1] in interest:
            if predicate[0] in OBJECT_FIRST_PREDICATES and len(predicate) == 3:
                return predicate[1], predicate[2]
            if len(predicate) == 2:
                return predicate[1], None
    raise ValueError("BDDL: no goal predicate acts on an object of interest: goal {} interest {}".format(
        problem["goal"], interest))


# --- LIBERO task order --------------------------------------------------------------------------------
def _assigned_literal(path, name):
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("{} not assigned in {}".format(name, path))


def suite_task_names(suite, task_order_index=0):
    """Task names in ``benchmark.get_benchmark_dict()[suite](task_order_index)`` order."""
    names = list(_assigned_literal(TASK_MAP_FILE, "libero_task_map")[suite])
    if suite == "libero_90":
        return names
    order = _assigned_literal(BENCHMARK_INIT_FILE, "task_orders")[task_order_index]
    return [names[i] for i in order]


def bddl_path(suite, task_id, task_order_index=0):
    return BDDL_ROOT / suite / (suite_task_names(suite, task_order_index)[int(task_id)] + ".bddl")


def task_target(suite, task_id, task_order_index=0):
    """Target of LIBERO task ``(suite, task_id)``: JSON-able dict (names lower-case, as LIBERO parses)."""
    path = bddl_path(suite, task_id, task_order_index)
    problem = parse_bddl(path)
    target, destination = goal_roles(problem)
    return {"suite": suite, "task_id": int(task_id), "task_name": path.stem,
            "bddl_file": str(path.relative_to(PROJECT_ROOT)),
            "bddl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "language": problem["language"], "obj_of_interest": list(problem["obj_of_interest"]),
            "goal": problem["goal"], "target": target, "destination": destination,
            "objects": problem["object_names"], "fixtures": problem["fixture_names"],
            "rule": "target = first argument of the first goal predicate acting on an obj_of_interest "
                    "(On/In: destination = second argument)"}


# --- segmentation instances -----------------------------------------------------------------------------
def instance_body_ids(name, body_names):
    """Body ids of LIBERO object / fixture ``name``: ``<name>_main`` and ``<name>_<part>`` (or the alias)."""
    names = [str(b) for b in body_names]
    if name in FIXTURE_BODY_ALIASES:
        ids = [i for i, b in enumerate(names) if b in FIXTURE_BODY_ALIASES[name]]
    else:
        ids = [i for i, b in enumerate(names) if b.startswith(name + "_")]
    if not ids:
        raise ValueError("no MuJoCo body for LIBERO object {!r}".format(name))
    return ids


def instance_geom_ids(name, geom_bodyid, body_names):
    """Geom ids (segmentation values) of LIBERO object ``name``."""
    bodies = set(instance_body_ids(name, body_names))
    return [g for g, b in enumerate(np.asarray(geom_bodyid).tolist()) if int(b) in bodies]


def instance_mask(segmentation_geom, geom_ids):
    """Boolean pixel mask of a geom-id segmentation image (OpenCV rows) for ``geom_ids``."""
    return np.isin(np.asarray(segmentation_geom), np.asarray(list(geom_ids), dtype=np.int64))


def scene_instances(geom_tables, problem_or_target):
    """``{object / fixture name: [geom ids]}`` for every BDDL object and fixture of the task.

    ``geom_tables``: mapping with ``model_geom_bodyid`` and ``model_body_names`` (the ``_geom.npz``).
    ``problem_or_target``: :func:`parse_bddl` or :func:`task_target` output.
    """
    objects = problem_or_target.get("object_names", problem_or_target.get("objects"))
    fixtures = problem_or_target.get("fixture_names", problem_or_target.get("fixtures"))
    gb, bn = geom_tables["model_geom_bodyid"], geom_tables["model_body_names"]
    out = {}
    for name in list(objects) + list(fixtures):
        out[name] = instance_geom_ids(name, gb, bn)
    return out
