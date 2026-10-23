"""NUDGE [9] implemented from the paper as a ``SamplingGuidance`` (see README.md). ``get_method("nudge")``."""
from Baseline.paper_baselines.methods.nudge.guidance import NudgeGuidance, calibrate_kappa


def make_method(**kwargs) -> NudgeGuidance:
    return NudgeGuidance(**kwargs)


__all__ = ["NudgeGuidance", "calibrate_kappa", "make_method"]
