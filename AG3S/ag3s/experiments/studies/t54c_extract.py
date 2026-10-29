"""T54c — `t53_extract` run on the T54c registry (thin wrapper; `t53_extract` itself is unchanged).

`T54c.task.md` "추출": the T53 order attention · noun · feat · gt · grasp · check, on the 16 new episodes of
`t54c_records`, with every output under ``<run>/verify/T54c/{attention,noun,feat,gt}/`` (+ ``layout.json``,
``check.json``). Same stages, flags and producers as `t53_extract` — this wrapper only swaps the registry
object `t53_extract` reads (`t53_extract.rc`, `t53_extract.STEP`) for the duration of one call::

    source scripts/setup/env.sh; R=<the T54c run>; M=AG3S.ag3s.experiments.studies.t54c_extract
    .venv-ag3s/bin/python -m $M --run-dir $R --stage layout --server-roots <server_8292_*> <server_8293_*>
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --run-dir $R --stage attention
    ...                                               (noun · feat on GPU, gt · grasp · check on CPU)

`t53_extract --help` documents every stage and flag.
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional, Sequence


@contextlib.contextmanager
def t54c_registry() -> Iterator:
    """`t53_extract` reading `t54c_records.records()` (restored afterwards)."""
    from AG3S.ag3s.experiments.studies import t53_extract as ex
    from AG3S.ag3s.experiments.studies import t54c_records as reg

    rc = reg.records()
    saved = (ex.rc, ex.STEP)
    ex.rc, ex.STEP = rc, reg.STEP
    try:
        yield ex
    finally:
        ex.rc, ex.STEP = saved


def main(argv: Optional[Sequence[str]] = None) -> None:
    with t54c_registry() as ex:
        ex.main(argv)


if __name__ == "__main__":
    main()
