"""SUBTASK-d V1 run lists: the T39 S2-2 E3b set (24 episodes x seed 10*ep+rep, rep 1, 2) split into 4 chains G1..G4
(G1/G2 = T39 chain A episodes at even/odd positions, G3/G4 = chain B likewise); inside a chain rep 1 of every episode first.
usage: plan_runs.py -> p/lists/E3bG_<1..4>.txt, p/lists/V0.txt"""
from common.workspace import workspace_path as _ag3s_workspace_path
import pathlib
EPS = [1800, 1807, 1808, 1818, 1819, 1824, 1828, 1836, 1863, 1878, 1887, 1888, 1896, 1914, 1925, 1927,
       1939, 1965, 1967, 1968, 1976, 1982, 1983, 1995]
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-d/p/lists')); D.mkdir(exist_ok=True)
A, B = EPS[0::2], EPS[1::2]
allitems = []
for i, eps in enumerate((A[0::2], A[1::2], B[0::2], B[1::2]), 1):
    items = [f"{e}:{10 * e + rep}" for rep in (1, 2) for e in eps]
    allitems += items
    (D / f"E3bG_{i}.txt").write_text(" ".join(items) + "\n")
    print(i, len(items), " ".join(items))
assert sorted(allitems) == sorted(f"{e}:{10 * e + r}" for e in EPS for r in (1, 2)) and len(allitems) == 48
(D / "V0.txt").write_text("1807:18071 1800:18001\n")
