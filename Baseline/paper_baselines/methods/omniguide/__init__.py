"""OmniGuide [19] as a ``SamplingGuidance`` (see README.md). ``get_method("omniguide", **params)``."""
from Baseline.paper_baselines.methods.omniguide.guidance import OmniGuideGuidance


def make_method(**kwargs) -> OmniGuideGuidance:
    return OmniGuideGuidance(**kwargs)


__all__ = ["OmniGuideGuidance", "make_method"]
