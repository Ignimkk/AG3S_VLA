"""Template discovery: which text prefix makes pi0.5 speak its subtask, per checkpoint variant.

Variants
  base      pi05_base (SigLIP + LLM as released)
  ft_nolora 16D checkpoint with every lora_b zeroed -> released LLM weights (frozen by LoRA) + fine-tuned SigLIP
  ft16d     16D checkpoint as served (LoRA on)
"""
from common.workspace import workspace_path as _ag3s_workspace_path

import glob
import json
import pathlib
import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from subtask_probe import EOS, TOKENIZER, TextHead, frame_obs, ground_truth, load_models  # noqa: E402

RUN = _ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000')
FRAMES = (0, 25, 40)  # t=0 reach, t=200 held/carry, t=320 released
PREFIXES = {
    "T:p": "Task: {p}",
    "T:p,": "Task: {p}, ",
    "T:p;nl": "Task: {p};\n",
    "T:p.nl": "Task: {p}\n",
    "p_nl": "{p}\n",
    "caption": "caption en\n",
    "T:p.Sub": "Task: {p}. Subtask: ",
    "T:p,Sub": "Task: {p}, Subtask: ",
    "T:p;nlSub": "Task: {p};\nSubtask: ",
    "T:p_nlSub": "Task: {p}\nSubtask: ",
    "T:p,S;nlSub": "Task: {p}, State: {s};\nSubtask: ",
}


def zero_lora(model):
    graphdef, state = nnx.split(model)
    pure = state.to_pure_dict()

    def z(path, x):
        return jnp.zeros_like(x) if any(getattr(p, "key", None) == "lora_b" for p in path) else x

    state.replace_by_pure_dict(jax.tree_util.tree_map_with_path(z, pure))
    return nnx.merge(graphdef, state)


def main(out):
    sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER)
    meta = json.load(open(f"{RUN}/meta.json"))
    prompt = meta["prompt"]
    clean = prompt.strip()
    policy, models = load_models(["base", "ft16d"])
    heads = {"base": TextHead(models["base"]), "ft16d": TextHead(models["ft16d"])}
    heads["ft_nolora"] = TextHead(zero_lora(models["ft16d"]))
    files = sorted(glob.glob(f"{RUN}/step_*.npz"))
    res = []
    for fi in FRAMES:
        d = np.load(files[fi])
        obs, tok_text = frame_obs(policy, d, prompt)
        state_str = sp.decode(tok_text.tolist()).split("State: ")[1].split(";")[0]
        gt = ground_truth(d)
        for mk, head in heads.items():
            img = head.image_tokens(obs)
            for pk, tpl in PREFIXES.items():
                for causal in (False, True):
                    if causal and pk not in ("T:p", "caption", "T:p.Sub"):
                        continue
                    ids = sp.encode(tpl.format(p=clean, s=state_str), add_bos=True)
                    n_pre = 1 if causal else len(ids)  # causal: only BOS is "prefix"
                    steps = head.greedy(img, ids, max_new=40) if not causal else _greedy_ar(head, img, ids, n_pre)
                    toks = [s[0] for s in steps if s[0] != EOS]
                    text = sp.decode(toks)
                    res.append(dict(frame=fi, t_step=int(d["t_step"]), phase=gt["phase"], model=mk, prefix=pk,
                                    causal=causal, text=text, first_lp=steps[0][1],
                                    first_top5=[sp.id_to_piece(i) for i in steps[0][2]],
                                    n_fast=sum(254976 <= t <= 257023 for t in toks), n_tok=len(toks)))
                    print(f"t={int(d['t_step']):3d} {gt['phase']:8s} {mk:9s} {pk:12s}{' C' if causal else '  '} -> {text[:70]!r}",
                          flush=True)
    json.dump(res, open(out, "w"), indent=1)


def _greedy_ar(head, img, ids, n_pre, max_new=40):
    ids = list(ids)
    steps = []
    for _ in range(max_new):
        _, top_lp, top_id = head.run(img, ids, n_pre)
        i = len(ids) - 1
        tok = int(top_id[i, 0])
        steps.append((tok, float(top_lp[i, 0]), top_id[i].tolist(), top_lp[i].tolist()))
        if tok == EOS:
            break
        ids.append(tok)
    return steps


if __name__ == "__main__":
    main(sys.argv[1])
