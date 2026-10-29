"""Feasibility probe: can pi0.5 (openpi) emit a high-level subtask in text?

openpi ships no text-decoding path for pi0.5. This probe adds one *outside* the repo:
prefix = 3 SigLIP grids + text tokens (prefix-LM: images+prompt bidirectional, generated tokens causal),
logits = final-normed hidden @ embedder.input_embedding.T (Gemma ties the LM head to the embedding).

Two measurements per recorded frame, for two checkpoints (pi05_base, our 16D LoRA fine-tune):
  1. free greedy generation under several prompt templates
  2. constrained scoring: length-normalised log p(candidate subtask | frame, template)
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import glob
import json
import pathlib
import sys
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

sys.path.insert(0, _ag3s_workspace_path(''))

from openpi.models import model as _model  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.models.pi0 import make_attn_mask  # noqa: E402
from openpi.policies import policy_config as _pc  # noqa: E402
from openpi.training import config as _config  # noqa: E402

CKPT_16D = (_ag3s_workspace_path('legacy/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999'))
BASE_PARAMS = _ag3s_workspace_path('legacy/pi05_TO_hybrid/cache/openpi/openpi-assets/checkpoints/pi05_base/params')
TOKENIZER = _ag3s_workspace_path('legacy/pi05_TO_hybrid/cache/openpi/big_vision/paligemma_tokenizer.model')
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
T_BUF = 128  # text buffer (prompt + generated)
EOS = 1

# --- templates: text that precedes the generated span ------------------------------------------------
# {p}: task prompt, {s}: discretised state string exactly as openpi builds it
TEMPLATES = {
    "openpi_action": "Task: {p}, State: {s};\nAction: ",      # the format both models were trained on
    "state_subtask": "Task: {p}, State: {s};\nSubtask: ",
    "task_subtask": "Task: {p}. Subtask: ",
    "issue701": "{p}. Subtask: ",
    "vqa_doing": "answer en What is the robot doing?\n",       # PaliGemma VQA format
    "vqa_holding": "answer en What is the robot holding?\n",
    "caption": "caption en\n",
}
SCORE_TEMPLATES = ("state_subtask", "task_subtask", "issue701")
CANDIDATES = {
    "pick": ["pick up the apple", "grasp the apple", "reach for the apple"],
    "place": ["place the apple in the basket", "put the apple in the basket", "move the apple to the basket"],
}


class TextHead:
    """Prefix-LM text decoding on a pi0.5 `Pi0` model, one jitted forward per call."""

    def __init__(self, model):
        self.graphdef, self.state = nnx.split(model)
        pure = self.state.to_pure_dict()
        self.emb = jnp.asarray(pure["PaliGemma"]["llm"]["embedder"]["input_embedding"])  # (V, D)
        self._img = jax.jit(self._img_tokens)
        self._fwd = jax.jit(self._forward)

    def _img_tokens(self, state, obs):
        model = nnx.merge(self.graphdef, state)
        toks, masks = [], []
        for name in obs.images:
            t, _ = model.PaliGemma.img(obs.images[name], train=False)
            toks.append(t)
            masks.append(jnp.repeat(obs.image_masks[name][:, None], t.shape[1], axis=1))
        return jnp.concatenate(toks, 1), jnp.concatenate(masks, 1)

    def _forward(self, state, emb, img_tok, img_mask, ids, valid, ar):
        """Per text position i: log p(ids[i+1] | ...) and top-5 of the next-token distribution."""
        model = nnx.merge(self.graphdef, state)
        txt = model.PaliGemma.llm(ids, method="embed")
        tokens = jnp.concatenate([img_tok, txt], 1)
        in_mask = jnp.concatenate([img_mask, valid], 1)
        ar_mask = jnp.concatenate([jnp.zeros(img_mask.shape, bool), ar], 1)
        attn = make_attn_mask(in_mask, ar_mask)
        pos = jnp.cumsum(in_mask, axis=1) - 1
        (out, _), _ = model.PaliGemma.llm([tokens, None], mask=attn, positions=pos)
        h = out[:, img_tok.shape[1]:].astype(jnp.float32)  # (B, T, D)
        logp = jax.nn.log_softmax(h @ emb.astype(jnp.float32).T, axis=-1)  # (B, T, V)
        nxt = jnp.concatenate([ids[:, 1:], jnp.zeros_like(ids[:, :1])], 1)
        lp_next = jnp.take_along_axis(logp, nxt[..., None], -1)[..., 0]
        top_lp, top_id = jax.lax.top_k(logp, 5)
        return lp_next, top_lp, top_id

    def image_tokens(self, obs):
        return self._img(self.state, obs)

    def run(self, img, ids_list, n_prefix):
        ids = np.zeros((1, T_BUF), np.int32)
        ids[0, : len(ids_list)] = ids_list
        valid = np.zeros((1, T_BUF), bool)
        valid[0, : len(ids_list)] = True
        ar = np.zeros((1, T_BUF), bool)
        ar[0, n_prefix:] = True  # generated / candidate span is causal
        out = self._fwd(self.state, self.emb, img[0], img[1], jnp.asarray(ids), jnp.asarray(valid), jnp.asarray(ar))
        return [np.asarray(o[0]) for o in out]

    def greedy(self, img, prefix_ids, max_new=24):
        ids = list(prefix_ids)
        steps = []
        for _ in range(max_new):
            _, top_lp, top_id = self.run(img, ids, len(prefix_ids))
            i = len(ids) - 1
            tok = int(top_id[i, 0])
            steps.append((tok, float(top_lp[i, 0]), top_id[i].tolist(), top_lp[i].tolist()))
            if tok == EOS or len(ids) >= T_BUF - 1:
                break
            ids.append(tok)
        return steps

    def score(self, img, prefix_ids, cand_ids):
        ids = list(prefix_ids) + list(cand_ids)
        lp_next, _, _ = self.run(img, ids, len(prefix_ids))
        n = len(prefix_ids)
        lps = lp_next[n - 1 : n - 1 + len(cand_ids)]
        return float(lps.sum()), float(lps.mean())


def load_models(which):
    cfg = _config.get_config("pi05_rby1_randomized_pick_place_16d_lora")
    policy = _pc.create_trained_policy(cfg, CKPT_16D)
    models = {}
    if "ft16d" in which:
        models["ft16d"] = policy._model  # noqa: SLF001
    if "base" in which:
        models["base"] = pi0_config.Pi0Config(pi05=True).load(
            _model.restore_params(BASE_PARAMS, dtype=jnp.bfloat16))
    return policy, models


def frame_obs(policy, d, prompt):
    images = {k: np.transpose(d[f"image_{k}"], (2, 0, 1)) for k in CAMS}
    inputs = policy._input_transform({"state": d["state"], "images": images, "prompt": prompt})  # noqa: SLF001
    tok_text = inputs["tokenized_prompt"][inputs["tokenized_prompt_mask"]]
    batched = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    obs = _model.preprocess_observation(None, _model.Observation.from_dict(batched), train=False)
    return obs, tok_text


def ground_truth(d):
    q, s = d["qpos"], d["state"]
    apple, crate = q[38:41], q[31:34]
    closed = bool(s[7] < 0.9)  # left gripper (the grasping hand in ep1807)
    lifted = bool(apple[2] > 0.87)
    in_crate = bool(np.linalg.norm(apple[:2] - crate[:2]) < 0.12)
    if in_crate and not closed:
        phase = "released"
    elif closed:
        phase = "held"
    else:
        phase = "reach"
    return dict(phase=phase, closed=closed, lifted=lifted, in_crate=in_crate,
                apple=apple.tolist(), left_grip=float(s[7]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=_ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000'))
    ap.add_argument("--models", nargs="+", default=["base", "ft16d"])
    ap.add_argument("--gen-every", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER)
    meta = json.load(open(pathlib.Path(args.run) / "meta.json"))
    prompt = args.prompt or meta["prompt"]
    files = sorted(glob.glob(str(pathlib.Path(args.run) / "step_*.npz")))[: args.limit]
    policy, models = load_models(args.models)
    heads = {k: TextHead(m) for k, m in models.items()}
    cand_ids = {c: sp.encode(c) for cs in CANDIDATES.values() for c in cs}

    records = []
    t0 = time.time()
    for fi, f in enumerate(files):
        d = np.load(f)
        obs, tok_text = frame_obs(policy, d, prompt)
        openpi_prompt = sp.decode(tok_text.tolist())
        state_str = openpi_prompt.split("State: ")[1].split(";")[0]
        clean = prompt.strip().replace("_", " ").replace("\n", " ")
        rec = dict(t_step=int(d["t_step"]), gt=ground_truth(d), gen={}, score={})
        for mk, head in heads.items():
            img = head.image_tokens(obs)
            rec["gen"][mk], rec["score"][mk] = {}, {}
            for tk, tpl in TEMPLATES.items():
                text = tpl.format(p=clean, s=state_str)
                prefix_ids = sp.encode(text, add_bos=True)
                if tk == "openpi_action":
                    assert prefix_ids == tok_text.tolist(), "template must reproduce openpi's own prompt"
                if fi % args.gen_every == 0:
                    steps = head.greedy(img, prefix_ids)
                    toks = [s[0] for s in steps if s[0] != EOS]
                    rec["gen"][mk][tk] = dict(
                        text=sp.decode(toks), ids=toks, lp=[s[1] for s in steps],
                        first_top5=[sp.id_to_piece(i) for i in steps[0][2]], first_top5_lp=steps[0][3])
                if tk in SCORE_TEMPLATES:
                    rec["score"][mk][tk] = {c: head.score(img, prefix_ids, ids) for c, ids in cand_ids.items()}
        records.append(rec)
        if fi % 5 == 0:
            g = rec["gen"]
            line = " | ".join(f"{mk}:{g[mk].get('state_subtask', {}).get('text', '')[:40]!r}" for mk in g)
            print(f"[{fi + 1}/{len(files)} t={rec['t_step']} {rec['gt']['phase']}] {line} ({time.time() - t0:.0f}s)",
                  flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(run=args.run, prompt=prompt, templates=TEMPLATES, candidates=CANDIDATES,
                   models=list(heads), records=records), open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
