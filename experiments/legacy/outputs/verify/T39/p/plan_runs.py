"""T39 S2-2 run lists. 24 episodes x seed 10*ep+rep (rep 1, 2) x {E0, E3b}; per condition 2 chains (A = even list positions,
B = odd), each chain does both seeds of its episodes, rep 1 of every episode first then rep 2 (an early stop still covers
whole pairs across conditions because E0 and E3b chains use the same order). usage: plan_runs.py -> p/lists/<COND>_<A|B>.txt"""
from common.workspace import workspace_path as _ag3s_workspace_path
import pathlib
EPS = [1800, 1807, 1808, 1818, 1819, 1824, 1828, 1836, 1863, 1878, 1887, 1888, 1896, 1914, 1925, 1927,
       1939, 1965, 1967, 1968, 1976, 1982, 1983, 1995]
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39/p/lists')); D.mkdir(exist_ok=True)
for chain, eps in (("A", EPS[0::2]), ("B", EPS[1::2])):
    items = [f"{e}:{10 * e + rep}" for rep in (1, 2) for e in eps]
    for cond in ("E0", "E3b"):
        (D / f"{cond}_{chain}.txt").write_text(" ".join(items) + "\n")
    print(chain, len(items), " ".join(items))
