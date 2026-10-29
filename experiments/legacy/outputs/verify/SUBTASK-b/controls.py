"""SUBTASK-b supplementary controls (not in task.md; added by verifier, .venv-curobo).

C1 frame_index-only: per frame_index majority 3-class label over train episodes -> val.
   (closed-loop t_step is not on the dataset frame clock -> not evaluated there)
C2 arm-joint probe: StandardScaler + LogisticRegression on raw state[0:7, 8:15] (14 arm joints,
   gripper removed), C by GroupKFold(5) -> val, closed246, closed414.
C3 phase-sequence identity: how many sampled episodes share the exact same phase_index sequence.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from probe import CLASSES, choose_C, load_set, make, metrics, subset  # noqa: E402

OUT = pathlib.Path(__file__).parent
ARM = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]


def main():
    tr, va, cl = load_set("train"), load_set("val"), load_set("closed")
    seqs = {}
    for d in (tr, va):
        for ep in np.unique(d["episode"]):
            seqs.setdefault(tuple(d["phase"][d["episode"] == ep].tolist()), []).append(str(ep))
    sizes = sorted((len(v) for v in seqs.values()), reverse=True)
    tr, va = subset(tr, tr["y"] >= 0), subset(va, va["y"] >= 0)
    res = dict(C3_phase_sequence=dict(n_episodes=int(sum(sizes)), n_distinct_sequences=len(sizes),
                                      group_sizes=sizes,
                                      singletons=[v[0] for v in seqs.values() if len(v) == 1]))
    # C1
    maj = {}
    for t in np.unique(tr["t"]):
        maj[int(t)] = int(np.bincount(tr["y"][tr["t"] == t], minlength=3).argmax())
    pred = np.asarray([maj.get(int(t), 0) for t in va["t"]])
    res["C1_frame_index_val"], _ = metrics(va["y"], np.eye(3)[pred], [0, 1, 2])
    res["C1_frame_index_val"]["auroc_note"] = "hard label; AUROC on one-hot"
    # C2
    X = tr["state"][:, ARM]
    C, cv = choose_C(X, tr["y"], tr["episode"])
    m = make(C).fit(X, tr["y"])
    res["C2_arm_joint_probe"] = dict(C=C, cv_acc=cv)
    for e, d in {"val": va, "closed246": subset(cl, cl["stride2"]), "closed414": cl}.items():
        res["C2_arm_joint_probe"][e], _ = metrics(d["y"], m.predict_proba(d["state"][:, ARM]), [0, 1, 2])
    json.dump(res, open(OUT / "controls_results.json", "w"), indent=1)
    print("C3", len(sizes), sizes[:5])
    print("C1 val acc", res["C1_frame_index_val"]["accuracy"])
    print("C2", {e: res["C2_arm_joint_probe"][e]["accuracy"] for e in ("val", "closed246", "closed414")}, CLASSES)


if __name__ == "__main__":
    main()
