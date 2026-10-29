"""Full-episode measurement: does the subtask pi0.5 speaks track the true task phase?

Per frame, per variant (base / ft16d / ft_nolora):
  gen    greedy text under 3 templates -> first verb class {pick, place, home, move, other, action_tokens}
  score  log p("Subtask: <candidate>" | frame, "Task: {p};\\n") grouped into pick / place / home
Ground truth phase from MuJoCo qpos: reach (gripper open, fruit not in crate) / held (gripper closed)
/ released (gripper open, fruit inside crate footprint).
"""
from common.workspace import workspace_path as _ag3s_workspace_path

import glob
import json
import pathlib
import re
import sys
import time

import numpy as np
import sentencepiece

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from subtask_discover import zero_lora  # noqa: E402
from subtask_probe import EOS, TOKENIZER, TextHead, frame_obs, load_models  # noqa: E402

RUNS = {
    "ep1807_apple": _ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000'),
    "ep0_orange": _ag3s_workspace_path('outputs/live_test/20260925_train/ep0/run_0000'),
    "ep500_banana": _ag3s_workspace_path('outputs/live_test/20260925_train/ep500/run_0000'),
    "ep1200_pear": _ag3s_workspace_path('outputs/live_test/20260925_train/ep1200/run_0000'),
}
FRUIT_QPOS = {"apple": 38, "banana": 45, "orange": 52, "pear": 59}  # free-joint qpos address (after crate at 31)
GEN_TEMPLATES = {
    "T;nl": ("Task: {p};\n", False),
    "p_nl": ("{p}\n", False),
    "T.Sub_causal": ("Task: {p}. Subtask: ", True),
}
VERBS = [("place", r"\b(place|put|drop|release|insert)\b"), ("pick", r"\b(pick|grasp|grab|reach)\b"),
         ("home", r"\b(return|home|retract|reset)\b"), ("move", r"\b(go|move|navigate|approach|lift)\b")]


def classify(text):
    t = text.lower().replace("subtask:", " ")
    if not t.strip():
        return "empty"
    non_ascii = sum(ord(c) > 127 for c in t) / max(1, len(t))
    first = None
    for cls, rx in VERBS:
        m = re.search(rx, t)
        if m and (first is None or m.start() < first[1]):
            first = (cls, m.start())
    if first and first[1] < 12:
        return first[0]
    if non_ascii > 0.3:
        return "action_tokens"
    return first[0] if first else "other"


def ground_truth(d, fruit):
    q, s = d["qpos"], d["state"]
    a = FRUIT_QPOS[fruit]
    obj, crate = q[a:a + 3], q[31:34]
    closed = bool(min(s[7], s[15]) < 0.9)
    in_crate = bool(np.linalg.norm(obj[:2] - crate[:2]) < 0.15 and obj[2] < 1.0)
    phase = "held" if closed else ("released" if in_crate else "reach")
    return dict(phase=phase, closed=closed, in_crate=in_crate, obj=obj.tolist(),
                grip_l=float(s[7]), grip_r=float(s[15]))


def logsumexp(x):
    m = max(x)
    return m + float(np.log(np.sum(np.exp(np.asarray(x) - m))))


def main(out, stride_long=2):
    sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER)
    policy, models = load_models(["base", "ft16d"])
    heads = {"base": TextHead(models["base"]), "ft16d": TextHead(models["ft16d"]),
             "ft_nolora": TextHead(zero_lora(models["ft16d"]))}
    result = dict(runs={}, gen_templates={k: v[0] for k, v in GEN_TEMPLATES.items()}, timing={})
    fwd_times = []
    for rk, run in RUNS.items():
        meta = json.load(open(f"{run}/meta.json"))
        prompt = meta["prompt"]
        fruit = next(f for f in FRUIT_QPOS if f in prompt.lower())
        cands = {"pick": [f"pick up the {fruit}", f"grasp the {fruit}"],
                 "place": [f"place the {fruit} in the basket", f"put the {fruit} in the basket"],
                 "home": ["return to home position"]}
        files = sorted(glob.glob(f"{run}/step_*.npz"))
        if len(files) > 80:
            files = files[::stride_long]
        frames = []
        t0 = time.time()
        for f in files:
            d = np.load(f)
            obs, _ = frame_obs(policy, d, prompt)
            rec = dict(t_step=int(d["t_step"]), gt=ground_truth(d, fruit), gen={}, score={})
            for mk, head in heads.items():
                img = head.image_tokens(obs)
                rec["gen"][mk] = {}
                for tk, (tpl, causal) in GEN_TEMPLATES.items():
                    ids = sp.encode(tpl.format(p=prompt.strip()), add_bos=True)
                    n_pre = 1 if causal else len(ids)
                    ids_run = list(ids)
                    toks = []
                    for _ in range(16):
                        ts = time.perf_counter()
                        _, _, top_id = head.run(img, ids_run, n_pre)
                        fwd_times.append(time.perf_counter() - ts)
                        tok = int(top_id[len(ids_run) - 1, 0])
                        if tok == EOS:
                            break
                        toks.append(tok)
                        ids_run.append(tok)
                    text = sp.decode(toks)
                    rec["gen"][mk][tk] = dict(text=text, cls=classify(text),
                                              n_fast=sum(254976 <= t <= 257023 for t in toks))
                pre = sp.encode(f"Task: {prompt.strip()};\n", add_bos=True)
                grp = {}
                for g, cs in cands.items():
                    grp[g] = logsumexp([head.score(img, pre, sp.encode("Subtask: " + c))[0] for c in cs])
                rec["score"][mk] = grp
            frames.append(rec)
            if len(frames) % 10 == 1:
                g = rec["gen"]
                print(f"[{rk} {len(frames)}/{len(files)} t={rec['t_step']} {rec['gt']['phase']}] "
                      + " | ".join(f"{mk}:{g[mk]['T;nl']['text'][:32]!r}" for mk in g)
                      + f" ({time.time() - t0:.0f}s)", flush=True)
        result["runs"][rk] = dict(run=run, prompt=prompt, fruit=fruit, candidates=cands, frames=frames)
        json.dump(result, open(out, "w"), indent=1)
    ft = np.asarray(fwd_times[50:])
    result["timing"] = dict(n=len(ft), median_ms=float(np.median(ft) * 1e3), p90_ms=float(np.percentile(ft, 90) * 1e3),
                            note="naive decoding: one full prefix+text forward (768 image + 128 text tokens) per token, no KV cache, H200, bf16")
    json.dump(result, open(out, "w"), indent=1)
    print("wrote", out, result["timing"])


if __name__ == "__main__":
    main(sys.argv[1])
