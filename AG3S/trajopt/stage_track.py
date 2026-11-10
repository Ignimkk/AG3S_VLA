"""제약 단계 추종 (stage track, P5b 2026-11-10) — Panda 서버의 단계 추정기: π0.5 prefix hidden h_10 + gripper feedback.

Why: P2 (`outputs/2026-11-10_제약단계_추종/README.md` §7) showed the grasp latch of pe123458 misses 14.5 % of truth carry
requests (S3) and leaves 44 / 98 empty closes undetected (latch stuck in CLOSING). P4 / P5a (§8, §10) picked, by a rule fixed
before the results, a 4-stage classifier on the layer-10 residual stream of the policy's own prefix pass plus the gripper
feedback: S3 miss 197 -> 69, undetected E-fail 44 -> 6, false HELD 67 -> 125 (86 of them in two episodes where the robot held a
non-target object).

Stages (user decision 2026-11-10): S1 approach · S2 grasp (closing until both fingers touch the target) · S3 carry+place
(both fingers on the target) · S4 released (after a carry that ended inside the destination box).

Server use (`serve_safe --stage-track ASSET`, Panda only, default off):
  * `AttentionPolicy(stage_hidden=True)` runs one extra jitted prefix pass with the gemma hidden stack and puts
    :data:`STAGE_HIDDEN_KEY` = :func:`stage_hidden_feature` (6144 floats) into the policy result (`SafePolicy` pops it: never on
    the wire).
  * `SafePolicy` keeps a **shadow latch** (the evidence latch as configured, fed the same inputs, never forced) whose state is a
    feature, builds :func:`grip_features`, and runs :class:`StageTrackProbe`. The prediction drives the real latch through the
    `--oracle-latch` path (S3 -> held, else not held) plus the closing release (S1 while CLOSING -> SEARCHING).

Feature definitions are the offline ones verbatim (`experiments/segmentation/stage_track/p3_extract_hidden.py` pooling,
`p4_feature_compare.load` GRIP), so the asset trained there applies unchanged.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

STAGES: tuple[str, ...] = ("approach", "grasp", "carry_place", "released")   # S1 .. S4
HELD_STAGE = 2                                                                  # S3
APPROACH_STAGE = 0                                                              # S1
HIDDEN_LAYER = 10
#: P3 rows used by P4 (agentview mean, wrist mean, valid-text mean) — the third camera block (512:768) is the padded right wrist.
CAMERA_RANGES: tuple[tuple[int, int], ...] = ((0, 256), (256, 512))
N_IMAGE_TOKENS = 768
HIDDEN_DIM = 2048
FEATURE_DIM = 3 * HIDDEN_DIM
LATCH_STATES: tuple[str, ...] = ("searching", "latched", "closing", "held", "placed")
GRIP_DIM = 5 + len(LATCH_STATES)
#: `AttentionPolicy` result key of the pooled hidden feature (popped by `SafePolicy`, never sent to the client).
STAGE_HIDDEN_KEY = "_stage_track_hidden"


def stage_hidden_feature(xp, hidden_layer, mask):
    """``hidden_layer`` = h_10 ``[T, D]`` of one sample, ``mask`` = prefix mask ``[T]`` -> ``[3 * D]``: mean over the agentview
    tokens, mean over the wrist tokens (unmasked, as P3), mean over the valid text tokens; rounded through float16 as P3 stored
    it. `xp` = numpy or jax.numpy."""
    h = hidden_layer.astype(xp.float32)
    rows = [h[a:b].mean(0) for a, b in CAMERA_RANGES]
    tm = xp.asarray(mask)[N_IMAGE_TOKENS:].astype(xp.float32)
    rows.append((h[N_IMAGE_TOKENS:] * tm[:, None]).sum(0) / tm.sum())
    return xp.concatenate(rows).astype(xp.float16).astype(xp.float32)


def grip_features(*, gap_mm: float, prev_gap_mm: Optional[float], measured_opening: Optional[float],
                  latch_state: Optional[str], last_command: Optional[float]) -> np.ndarray:
    """P4 GRIP, in its column order: gap (mm, finger qpos difference), previous request's gap (first request: the same gap),
    their difference, measured opening (`latch_signal.value`; None -> 1.0), latch state one-hot (searching, latched, closing,
    held, placed; unknown -> zeros), last executed gripper action (first request: -1.0)."""
    prev = float(gap_mm) if prev_gap_mm is None else float(prev_gap_mm)
    onehot = np.zeros(len(LATCH_STATES))
    if latch_state in LATCH_STATES:
        onehot[LATCH_STATES.index(latch_state)] = 1.0
    return np.r_[float(gap_mm), prev, float(gap_mm) - prev, 1.0 if measured_opening is None else float(measured_opening),
                 onehot, -1.0 if last_command is None else float(last_command)]


@dataclasses.dataclass
class StageTrackProbe:
    """PCA (train mean / components) on the hidden feature, then StandardScaler + multinomial logistic on [PCA | GRIP]."""

    pca_mean: np.ndarray        # (FEATURE_DIM,)
    pca_components: np.ndarray  # (k, FEATURE_DIM)
    mean: np.ndarray            # (k + GRIP_DIM,)
    scale: np.ndarray
    coef: np.ndarray            # (4, k + GRIP_DIM)
    intercept: np.ndarray       # (4,)
    classes: tuple[str, ...]
    name: str
    md5: str
    meta: dict

    @classmethod
    def load(cls, path) -> "StageTrackProbe":
        path = pathlib.Path(path)
        raw = path.read_bytes()
        with np.load(path, allow_pickle=False) as z:
            classes = tuple(str(c) for c in z["classes"])
            meta = {k[len("meta_"):]: z[k].item() for k in z.files if k.startswith("meta_")}
            probe = cls(pca_mean=z["pca_mean"].astype(np.float64), pca_components=z["pca_components"].astype(np.float64),
                        mean=z["mean"].astype(np.float64), scale=z["scale"].astype(np.float64),
                        coef=z["coef"].astype(np.float64), intercept=z["intercept"].astype(np.float64), classes=classes,
                        name=path.stem, md5=hashlib.md5(raw).hexdigest(), meta=meta)
        probe.check()
        return probe

    def check(self) -> None:
        k = self.pca_components.shape[0]
        if self.classes != STAGES:
            raise ValueError(f"stage-track probe {self.name}: classes {self.classes} != {STAGES}")
        if self.pca_mean.shape != (FEATURE_DIM,) or self.pca_components.shape != (k, FEATURE_DIM):
            raise ValueError(f"stage-track probe {self.name}: PCA shapes {self.pca_mean.shape} {self.pca_components.shape}")
        d = k + GRIP_DIM
        if (self.mean.shape != (d,) or self.scale.shape != (d,) or self.coef.shape != (len(STAGES), d)
                or self.intercept.shape != (len(STAGES),)):
            raise ValueError(f"stage-track probe {self.name}: classifier shapes do not match PCA {k} + GRIP {GRIP_DIM}")

    def design(self, hidden: np.ndarray, grip: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, np.float64).reshape(-1, FEATURE_DIM)
        grip = np.asarray(grip, np.float64).reshape(-1, GRIP_DIM)
        return np.concatenate([(hidden - self.pca_mean) @ self.pca_components.T, grip], 1)

    def proba(self, hidden: np.ndarray, grip: np.ndarray) -> np.ndarray:
        Z = ((self.design(hidden, grip) - self.mean) / self.scale) @ self.coef.T + self.intercept
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predict(self, hidden, grip) -> dict[str, Any]:
        """One request -> ``{"stage", "index", "p": {stage: p}}`` (argmax; ties -> the earlier stage)."""
        p = self.proba(hidden, grip)[0]
        i = int(np.argmax(p))
        return {"stage": STAGES[i], "index": i, "p": {s: round(float(v), 5) for s, v in zip(STAGES, p)}}

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "md5": self.md5, "hidden_layer": HIDDEN_LAYER, "pca_dim": int(self.pca_components.shape[0]),
                "grip_dim": GRIP_DIM, "classes": list(self.classes), **{k: v for k, v in self.meta.items()}}


def default_asset_path() -> pathlib.Path:
    """The shipped asset (`experiments/segmentation/stage_track/p5b_build_asset.py`)."""
    return pathlib.Path(__file__).resolve().parents[1] / "ag3s" / "asset" / "stage_track" / "h10_grip_v1.npz"


def last_executed_command(prev_actions: Optional[np.ndarray], n_exec: Optional[int], n_steps: Optional[int] = None,
                          column: int = 6) -> Optional[float]:
    """The gripper value of the last row the client executed from the previous response (feedback ``n_exec`` / ``n_steps``):
    row ``n_exec - 1``; a HOLD chunk reports ``n_exec == 0`` and the client runs its first ``n_steps`` rows -> row
    ``n_steps - 1`` (without ``n_steps``: the last row). None before the first response. Checked against the client's
    ``actions_executed`` on the 104 pe123458 dev episodes (`p5b_build_asset.py` check 2)."""
    if prev_actions is None:
        return None
    a = np.asarray(prev_actions, np.float64)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] <= column:
        return None
    row = int(n_exec) - 1 if n_exec else (int(n_steps) - 1 if n_steps else a.shape[0] - 1)
    row = min(max(row, 0), a.shape[0] - 1)
    return float(a[row, column])


def gap_mm_of_state(state: Optional[Sequence[float]], columns: tuple[int, int] = (6, 7)) -> Optional[float]:
    """LIBERO policy state = [eef pos 3, axis-angle 3, gripper qpos 2] -> finger gap in mm ((q0 - q1) * 1000)."""
    if state is None:
        return None
    s = np.asarray(state, np.float64).reshape(-1)
    if s.shape[0] <= max(columns):
        return None
    return float((s[columns[0]] - s[columns[1]]) * 1000.0)


def build_stage_hidden_pass(sampler):
    """A jitted ``inputs (no batch axis) -> stage_hidden_feature`` on the attention sampler's model state (the policy's own
    weights): `prefix_hidden_impl` (one prefix pass, gemma `return_hidden` on for it only), h_10 of sample 0. Only the 6144
    floats leave the device."""
    import jax
    import jax.numpy as jnp

    from AG3S.trajopt.attention_source import prefix_hidden_impl
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    def _pass(state, obs):
        m = sampler._nnx.merge(sampler._graphdef, state)  # noqa: SLF001
        obs = _model.preprocess_observation(None, obs, train=False)
        _, mask, hidden = prefix_hidden_impl(m, obs, make_attn_mask=make_attn_mask, xp=jnp)
        return stage_hidden_feature(jnp, hidden[HIDDEN_LAYER][0], mask[0])

    jitted = jax.jit(_pass)

    def run(inputs) -> np.ndarray:
        from openpi.shared import array_typing as at

        batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], dict(inputs))
        obs = _model.Observation.from_dict(batched)
        with at.disable_typechecking():
            return np.asarray(jitted(sampler._state, obs), np.float32)  # noqa: SLF001

    return run


__all__ = ["STAGES", "HELD_STAGE", "APPROACH_STAGE", "HIDDEN_LAYER", "FEATURE_DIM", "GRIP_DIM", "LATCH_STATES",
           "STAGE_HIDDEN_KEY", "stage_hidden_feature", "grip_features", "StageTrackProbe", "last_executed_command",
           "gap_mm_of_state", "build_stage_hidden_pass", "default_asset_path"]
