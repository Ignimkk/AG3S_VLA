"""Lan-o3dp [15] obstacle guidance ADAPTED to pi0.5 (see README.md). ``get_method("lan_o3dp_guidance_adapted")``."""
from Baseline.paper_baselines.methods.lan_o3dp_guidance_adapted.guidance import LanO3dpGuidanceAdapted


def make_method(**kwargs) -> LanO3dpGuidanceAdapted:
    return LanO3dpGuidanceAdapted(**kwargs)


__all__ = ["LanO3dpGuidanceAdapted", "make_method"]
