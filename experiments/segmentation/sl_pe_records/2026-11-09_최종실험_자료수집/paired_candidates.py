"""(2026-11-09 12:4x KST dev-4a) Fig. 1 v3 (upper SafeLIBERO row) / supplementary video candidates from E1a Spatial LI
(test init 5-14, seed 7): per (task, init) the outcome of every arm (SS = safe success, S+C, F+C, F) and the video paths of
pi0.5 and the AG3S arm. Candidate = AG3S safe success where pi0.5 collided and the detector baselines did not all succeed
safely. Outcome classes from episodes.jsonl (official_success, safelibero.collided). usage: python paired_candidates.py [ag3s tag]"""
import glob
import json
import sys

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
OUT = "/mnt/dev/AG3S_VLA/outputs/2026-11-09_최종실험_자료수집/"
ARMS = [("VLA", "2*qe-vla-e1a-spatial-LI-t*"), ("AEGIS", "*_e1a-aegis-spatial-LI-*"), ("CBF", "*_e1a-multilink_cbf-spatial-LI-*"),
        ("KNOWS", "*_e1a-knows-spatial-LI-*"), ("CBF_GT", "*_e1a-multilink_cbf-gt-spatial-LI-*"),
        ("KNOWS_GT", "*_e1a-knows-gt-spatial-LI-*"), ("B5_GT", "2*qe-gt23t100-e1a-spatial-LI-t*")]


def load(pat):
    out = {}
    for f in sorted(glob.glob(B + pat + "/episodes.jsonl")):
        if "INVALID" in f:
            continue
        for l in open(f):
            e = json.loads(l)
            s, c = bool(e["official_success"]), bool(e["safelibero"]["collided"])
            out[(int(e["task_id"]), int(e["episode_idx"]))] = ({(True, False): "SS", (True, True): "S+C", (False, True): "F+C", (False, False): "F"}[(s, c)],
                                                                e.get("video"), e["safelibero"]["obstacle"].replace("_obstacle_1", ""))
    return out


def main(tag):
    arms = ARMS + [("AG3S", f"2*qe-{tag}-e1a-spatial-LI-t*")]
    data = {k: load(p) for k, p in arms}
    keys = sorted(data["AG3S"])
    lines = ["| task · init | obstacle | " + " | ".join(k for k, _ in arms) + " | 후보 |", "|---|---|" + "---|" * (len(arms) + 1)]
    cands = []
    for k in keys:
        row = [data[a].get(k, ("-", None, None))[0] for a, _ in arms]
        ag, vla = data["AG3S"][k][0], data["VLA"].get(k, ("-",))[0]
        det = [data[a].get(k, ("-",))[0] for a in ("AEGIS", "CBF", "KNOWS")]
        cand = ag == "SS" and vla in ("S+C", "F+C") and not all(x == "SS" for x in det)
        if cand:
            cands.append(dict(task=k[0], init=k[1], obstacle=data["AG3S"][k][2], arms=dict(zip([a for a, _ in arms], row)),
                              video_ag3s=data["AG3S"][k][1], video_vla=data["VLA"].get(k, (None, None))[1]))
        lines.append(f"| t{k[0]} · {k[1]} | {data['AG3S'][k][2]} | " + " | ".join(row) + f" | {'**후보**' if cand else ''} |")
    open(OUT + f"paired_outcomes_{tag}.md", "w").write("\n".join(lines) + "\n")
    json.dump(cands, open(OUT + f"fig1_video_candidates_{tag}.json", "w"), indent=1)
    print("\n".join(lines))
    print(f"\ncandidates: {len(cands)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pe123458")
