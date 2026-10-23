"""Constrained flow matching [5] re-implemented as a ``SamplingGuidance`` (see README.md; upstream has no license)."""
from Baseline.paper_baselines.methods.constrained_flow_matching.guidance import ConstrainedFlowMatchingGuidance


def make_method(**kwargs) -> ConstrainedFlowMatchingGuidance:
    return ConstrainedFlowMatchingGuidance(**kwargs)


__all__ = ["ConstrainedFlowMatchingGuidance", "make_method"]
