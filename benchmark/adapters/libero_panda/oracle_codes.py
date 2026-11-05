"""SL12 -- the simulator ground-truth code image of the Panda LIBERO clients (visible-target oracle, diagnostic).
Python 3.8 compatible, numpy only (the clients run in ``.venv-libero`` / ``.venv-libero-safety``, which cannot import AG3S).

The server side is the RB-Y1 T46e A13 oracle (``AG3S/ag3s/runtime/oracle_segmentation.py``; ``--target-extent
oracle-visible``): it reads, per camera of the request, an ``(H, W)`` uint8 **code image** next to the depth and builds the
episode's labelled surface samples from them. This module makes that image on the client from the MuJoCo geom-id
segmentation the adapter already renders (``camera.capture(..., segmentation=True)``, ``--capture-segmentation``: OpenCV row
order, aligned with ``depth_m`` -- both are ``raw[::-1]``).

======  ============================================================================================================
code    meaning (the table of ``oracle_segmentation``; ``tests/benchmark/test_sl12_oracle_panda.py`` pins it)
======  ============================================================================================================
0       nothing hit, or the **robot** (``robot0_*`` / ``gripper0_*`` / ``mount0_*`` bodies): never evidence
1       the task's **target object** (:func:`choose_target`)
2       static: table, fixtures (stove, cabinet ...), floor, wall -- everything that is neither robot nor an object
3, 4..  every **other movable object** of the task (``:objects`` of the BDDL), one code per object, in ``:objects`` order
======  ============================================================================================================

Which object is the target (:func:`choose_target`, printed in the run settings and recorded per episode):

1. the first predicate of the BDDL ``:goal`` whose first argument is a movable object of the task -- the object the goal acts
   on (LIBERO ``On/In A B``: A; LIBERO-Safety ``Checkgrippercontactpart A (..)``: A);
2. else the first movable ``:obj_of_interest``;
3. else the task has no pick target and the client refuses to start (the oracle would label nothing as target).

For the LIBERO Spatial / Object suites this is the same object ``benchmark.adapters.libero_panda.task_target.task_target``
(SL3) names. Limits: a task that moves two objects in turn (``put both ...``) keeps its first object as the target for the
whole episode; a task whose goal acts on a fixture only (``Open wooden_cabinet``, ``Turnon flat_stove``) is refused.

An object ``X`` is the bodies ``X_main`` and ``X_<part>`` (``task_target.instance_body_ids``); a geom takes the code of its body.
"""
from __future__ import annotations

import collections

import numpy as np

CODE_NONE = 0
CODE_TARGET = 1
CODE_STATIC = 2
CODE_OBJECT_BASE = 3
#: the memory's truncation band keeps one bit per code in a uint32 (``OracleMemory._update_band``)
MAX_CODE = 31
#: wire format version of the code image (``oracle_segmentation.CODE_VERSION``)
CODE_VERSION = 1
#: request keys (``AG3S.trajopt.wire.PREFIX + ORACLE_SEG`` / ``ORACLE_SEG_META``) and the server metadata key
#: (``wire.ORACLE_SEG_REQUIRED``); ``tests/benchmark/test_sl12_oracle_panda.py`` compares them with ``wire``
REQUEST_PREFIX = "ag3s/oracle_seg/"
REQUEST_META_KEY = "ag3s/oracle_seg_meta"
SERVER_REQUIRES_KEY = "oracle_segmentation_required"
#: bodies of the robot itself (robosuite Panda: arm, hand, fingers, mount)
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mount0_")
#: client flag (``libero_safe_client`` / ``libero_safety_client``) the launchers pass for the ``-oraclegt`` versions
CLIENT_FLAG = "--oracle-segmentation"


def is_robot_body(name):
    return str(name).startswith(ROBOT_BODY_PREFIXES)


# --- the task's target object ---------------------------------------------------------------------------------
def problem_from_parsed(parsed):
    """Normalise a LIBERO ``robosuite_parse_problem`` dict (``env.env.parsed_problem``) or a
    ``task_target.parse_bddl`` dict to ``{"goal", "obj_of_interest", "objects"}`` (lower-case names, ``:objects`` order)."""
    objects = parsed.get("object_names")
    if objects is None:
        objects = [n for names in dict(parsed.get("objects") or {}).values() for n in names]
    goal = parsed.get("goal")
    if goal is None:
        goal = parsed.get("goal_state") or []
    return {"goal": [list(p) for p in goal], "obj_of_interest": list(parsed.get("obj_of_interest") or []),
            "objects": list(objects)}


def choose_target(problem):
    """``(target object name, rule)`` of a normalised problem (module docstring); ``ValueError`` if there is none."""
    objects = list(problem["objects"])
    for predicate in problem["goal"]:
        if len(predicate) >= 2 and isinstance(predicate[1], str) and predicate[1] in objects:
            return predicate[1], "first_goal_predicate_first_argument"
    for name in problem["obj_of_interest"]:
        if name in objects:
            return name, "first_movable_obj_of_interest"
    raise ValueError("SL12 oracle: the task has no movable pick target (goal {}, obj_of_interest {}, objects {}); the "
                     "visible-target oracle needs a target object".format(problem["goal"], problem["obj_of_interest"],
                                                                         objects))


def instance_body_ids(name, body_names):
    """Body ids of LIBERO object ``name``: ``<name>_main`` and ``<name>_<part>`` (``task_target.instance_body_ids``)."""
    return [i for i, b in enumerate(str(x) for x in body_names) if b.startswith(name + "_")]


# --- the code table ---------------------------------------------------------------------------------------------
def geom_code_table(body_names, geom_bodyid, target, objects):
    """``(lut, codes)``: ``lut[geom_id + 1]`` = code of that geom (``lut[0]`` = no geom hit = 0), ``codes`` = ``{object: code}``
    of the target (1) and the other movable objects (3, 4, ...). Raises if ``target`` is not one of ``objects``, an object has
    no MuJoCo body, or there are too many codes for the memory's band mask."""
    objects = [str(o) for o in objects]
    target = str(target)
    if target not in objects:
        raise ValueError("oracle target {!r} is not a movable object {}".format(target, objects))
    codes = collections.OrderedDict([(target, CODE_TARGET)])
    for name in (o for o in objects if o != target):
        codes[name] = CODE_OBJECT_BASE + len(codes) - 1
    if max(codes.values()) > MAX_CODE:
        raise ValueError("{} movable objects: codes would exceed {} (the memory's band mask)".format(len(objects), MAX_CODE))
    body_names = [str(b) for b in body_names]
    body_code = np.full(len(body_names), CODE_STATIC, np.uint8)
    for i, name in enumerate(body_names):
        if is_robot_body(name):
            body_code[i] = CODE_NONE
    for name, code in codes.items():
        ids = instance_body_ids(name, body_names)
        if not ids:
            raise ValueError("no MuJoCo body for LIBERO object {!r}".format(name))
        body_code[ids] = code
    gb = np.asarray(geom_bodyid, np.int64)
    lut = np.full(len(gb) + 1, CODE_NONE, np.uint8)
    lut[1:] = body_code[gb]
    return lut, dict(codes)


def label_geom_image(geom_ids, lut):
    """(H, W) uint8 code image from a geom-id image (``segmentation_geom``, ``-1`` = no geom hit -> 0)
    (``oracle_segmentation.label_image`` for geom ids)."""
    b = np.asarray(geom_ids, np.int64)
    out = lut[np.clip(b + 1, 0, len(lut) - 1)]
    out[(b < -1) | (b + 1 >= len(lut))] = CODE_NONE
    return out.astype(np.uint8, copy=False)


class OracleLabeler(object):
    """One episode's labeller: the task's target, the geom -> code table and the request keys of a frame set."""

    def __init__(self, problem, body_names, geom_bodyid, *, suite="", task_id=None):
        self.target, self.rule = choose_target(problem)
        self.objects = list(problem["objects"])
        self.lut, self.codes = geom_code_table(body_names, geom_bodyid, self.target, self.objects)
        self.suite, self.task_id = suite, task_id
        self.n_frames = 0
        self.last_pixels = {}

    @classmethod
    def from_env(cls, env, tables, **kwargs):
        """``env``: the LIBERO ``ControlEnv`` (``env.env.parsed_problem``); ``tables``: ``camera.model_tables(env)``."""
        inner = getattr(env, "env", env)
        return cls(problem_from_parsed(inner.parsed_problem), tables["model_body_names"], tables["model_geom_bodyid"],
                   **kwargs)

    def labels(self, frames):
        """``{camera: (H, W) uint8}`` from frames carrying ``segmentation_geom`` (``camera.capture(segmentation=True)``)."""
        out = collections.OrderedDict()
        for cam, frame in frames.items():
            if "segmentation_geom" not in frame:
                raise KeyError("frame {!r} has no segmentation_geom: start the client with --capture-segmentation "
                               "(the oracle flag implies it)".format(cam))
            out[cam] = label_geom_image(frame["segmentation_geom"], self.lut)
        return out

    def request_keys(self, frames):
        """The request keys of one observation: ``ag3s/oracle_seg/<cam>`` per camera and ``ag3s/oracle_seg_meta``."""
        lab = self.labels(frames)
        self.n_frames += 1
        self.last_pixels = {cam: int((img == CODE_TARGET).sum()) for cam, img in lab.items()}
        out = collections.OrderedDict()
        for cam, img in lab.items():
            out[REQUEST_PREFIX + cam] = np.ascontiguousarray(img)
        out[REQUEST_META_KEY] = {"version": CODE_VERSION, "target": self.target, "codes": dict(self.codes)}
        return out

    def record(self):
        """Per-episode record (``episodes.jsonl`` ``safe_client.oracle``)."""
        return {"target": self.target, "rule": self.rule, "codes": dict(self.codes), "suite": self.suite,
                "task_id": self.task_id, "frames_labelled": int(self.n_frames),
                "robot_body_prefixes": list(ROBOT_BODY_PREFIXES), "code_version": CODE_VERSION}


def settings_record(suite_rule=None):
    """The run-settings entry (launcher manifest) that says what the oracle is and how its target is chosen."""
    return {"oracle_segmentation": {
        "what": "SL12 diagnostic: the client sends the MuJoCo GT code image per camera (0 robot/none, 1 target, 2 static, "
                ">=3 other movable objects); the server (--target-extent oracle-visible --oracle-extent-identity gt) "
                "removes the target's visible surface + truncation band from the target-free layer",
        "target_rule": ["first BDDL :goal predicate whose first argument is a movable object (LIBERO On/In A B: A; "
                        "LIBERO-Safety Checkgrippercontactpart A: A)", "else first movable :obj_of_interest",
                        "else the client refuses the task"],
        "limits": "put-both tasks keep their first object as the target; fixture-only goals are refused",
        "segmentation_instant": "rendered at env.step return (camera.capture), depth at the image substep",
        **({} if suite_rule is None else {"suite": suite_rule})}}
