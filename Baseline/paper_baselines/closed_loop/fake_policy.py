"""CPU stand-ins for the openpi ``Policy`` (CPU smoke only; no VLA, no GPU).

:class:`FakePolicy` has what the server path needs from an openpi policy: ``_rng`` (``SeededPolicy`` re-keys it),
``metadata``, ``infer(obs) -> {"actions" (H, 7), "state", "policy_timing"}`` with the same ``_rng`` split per call,
and the hooks of :class:`guided.SurrogateBackend` (target chunk in the normalized model space, output mapping).

Chunk sources:

* ``constant`` -- every call the same slow descent with the gripper open (= ``BM-SETUP-B`` fake server);
* ``recorded`` -- the chunks a recorded run's server returned (``episodes/<stem>_calls.npz`` ``actions_response``),
  looked up by the episode's ``policy_seed`` (decoded task / episode) and the request's ``ag3s/seq``. With the
  recorded VLA-only run of the same task / init / seed, ``--method none`` replays that run's actions exactly.
  ``source_key="nominal"`` reads a **method** run's unguided / unfiltered chunks instead
  (``method_calls/ep<k>_seed<s>.npz`` ``nominal``, by ``seq``): replaying a filter / guidance run's own observations
  with the VLA chunks that run's method saw (CLOSED-LOOP-FIX CPU replay).

``infer`` returns the source chunk itself (``exact``); the guided path samples D's surrogate flow towards it
(``tracking``: unguided = target up to ~1e-7).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np


def decode_policy_seed(policy_seed: int | None) -> dict:
    """``libero_safe_client.episode_seed`` = seed*100000 + task_id*1000 + episode_idx."""
    if policy_seed is None:
        return {"policy_seed": None, "seed": None, "task_id": None, "episode_idx": None}
    ps = int(policy_seed)
    return {"policy_seed": ps, "seed": ps // 100000, "task_id": (ps % 100000) // 1000, "episode_idx": ps % 1000}


SOURCE_KEYS = ("actions_response", "nominal")


class RecordedChunks:
    def __init__(self, run_dir: str | pathlib.Path, source_key: str = "actions_response"):
        if source_key not in SOURCE_KEYS:
            raise ValueError(f"source_key must be one of {SOURCE_KEYS}")
        self.run_dir = pathlib.Path(run_dir)
        self.source_key = source_key
        self.by_seed: dict[int, str] = {}
        for line in (self.run_dir / "episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ps = (r.get("safe_client") or {}).get("policy_seed")
            if ps is not None:
                self.by_seed[int(ps)] = r["npz"].rsplit("/", 1)[-1][: -len(".npz")]
        self._cache: dict[str, np.ndarray] = {}

    def chunks(self, policy_seed: int) -> np.ndarray:
        stem = self.by_seed[int(policy_seed)]
        if stem not in self._cache:
            if self.source_key == "nominal":
                path = sorted((self.run_dir / "method_calls").glob(f"ep*_seed{int(policy_seed)}.npz"))
                if len(path) != 1:
                    raise FileNotFoundError(f"method_calls/ep*_seed{policy_seed}.npz: {len(path)} files in {self.run_dir}")
                with np.load(path[0]) as z:
                    seq, nom = np.asarray(z["seq"], np.int64), np.asarray(z["nominal"], np.float64)
                out = np.full((int(seq.max()) + 1,) + nom.shape[1:], np.nan)
                out[seq] = nom
                self._cache[stem] = out
            else:
                with np.load(self.run_dir / "episodes" / f"{stem}_calls.npz") as z:
                    self._cache[stem] = np.asarray(z["actions_response"], np.float64)
        return self._cache[stem]

    def chunk(self, policy_seed: int | None, seq: int) -> np.ndarray | None:
        if policy_seed is None or int(policy_seed) not in self.by_seed:
            return None
        c = self.chunks(policy_seed)
        if int(seq) >= len(c) or not np.all(np.isfinite(c[int(seq)])):
            return None
        return c[int(seq)].copy()


class FakePolicy:
    def __init__(self, norm: dict, *, source: str = "constant", run_dir: str | None = None, H: int = 10, D: int = 32,
                 kind: str = "tracking", num_steps: int = 10, exact: bool = True, source_key: str = "actions_response"):
        import jax

        if source not in ("constant", "recorded"):
            raise ValueError("source must be constant | recorded")
        self.norm, self.source, self.H, self.D = norm, source, int(H), int(D)
        self.kind, self.num_steps, self.exact = kind, int(num_steps), bool(exact)
        self.recorded = RecordedChunks(run_dir, source_key) if source == "recorded" else None
        self._rng = jax.random.key(0)
        self.context = {"policy_seed": None, "seq": 0}
        self.n_missing = 0
        self._sample = None

    @property
    def metadata(self) -> dict:
        return {"fake": True, "source": self.source, "kind": self.kind,
                "run": None if self.recorded is None else str(self.recorded.run_dir),
                "source_key": None if self.recorded is None else self.recorded.source_key}

    def reset(self) -> None:
        pass

    def set_context(self, *, policy_seed=None, seq=None) -> None:
        self.context = {"policy_seed": policy_seed, "seq": int(seq or 0)}

    def target_raw(self, obs: dict) -> np.ndarray:
        if self.recorded is not None:
            c = self.recorded.chunk(self.context["policy_seed"], self.context["seq"])
            if c is not None:
                return c
            self.n_missing += 1
        a = np.zeros((self.H, 7))
        a[:, 2], a[:, 6] = -0.3, -1.0  # slow descent, gripper open
        return a

    def target_model_space(self, obs: dict) -> np.ndarray:
        from Baseline.paper_baselines.methods.guidance_common.core import chunk_to_model_space

        return chunk_to_model_space(self.target_raw(obs), self.norm, self.D)

    def finish(self, obs: dict, x: np.ndarray) -> dict:
        from Baseline.paper_baselines.methods.guidance_common.core import model_space_to_chunk

        return {"actions": model_space_to_chunk(x, self.norm), "state": np.asarray(obs.get("observation/state", []))}

    def infer(self, obs: dict, **kwargs) -> dict:
        import jax

        self._rng, rng = jax.random.split(self._rng)  # Policy.infer's split (keeps the RNG stream in step)
        if self.exact:
            out = {"actions": self.target_raw(obs), "state": np.asarray(obs.get("observation/state", []))}
        else:
            from Baseline.paper_baselines.methods.guidance_common.core import surrogate_sample

            noise = jax.random.normal(rng, (1, self.H, self.D))
            x = surrogate_sample(self.target_model_space(obs), noise, num_steps=self.num_steps, kind=self.kind)
            out = self.finish(obs, np.asarray(x))
        out["policy_timing"] = {"infer_ms": 0.0}
        return out


__all__ = ["FakePolicy", "RecordedChunks", "SOURCE_KEYS", "decode_policy_seed"]
