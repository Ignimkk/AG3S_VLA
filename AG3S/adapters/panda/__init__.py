"""Panda (LIBERO) robot adapter for AG3S (SL2): URDF sphere chain, self-filter cover, link sets.

`model` builds the V11AB-structured models; `cover` fits the self-filter covering capsules from the
robosuite meshes. Nothing here modifies AG3S core -- it composes `UrdfSphereChain`,
`self_filter_covering_capsules` and `depth_robot_mask` as they are.
"""
