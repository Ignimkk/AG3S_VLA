# Figure 2 caption (draft)

**Finding the target is not enough; its boundary and the querying link decide contact.**
(a) If every observed surface is an obstacle, the target repels the fingers and the grasp is blocked.
(b) Removing a ball around the target for every link permits the grasp, but the ball also swallows the
touching neighbour, so its surface is no longer protected. (c) AG3S keeps the ball only as an outer
bound and removes only the cells owned by the target's observed points. It then gives this target-free
field to the contact links of the active hand alone (d); every other queried link reads the full scene field.
All panels show one recorded frame just before the grasp, with the observed surface drawn as 1 cm voxels.

<!-- Scope note (do not print): panel (c)'s ownership rule is the serving function `_ball_owned_split`
(T44 I1, `--target-exclusion ball-owned`), which is on in the evaluated arms but off by code default.
The rendered panel (c) also applies the adopted T46 F1 surface evidence offline to the recorded chunk;
the drawn run (T53 C1a) used proximity evidence.
In the evaluated configuration only gripper links are queried (`--links gripper`), so "other links" are
the other hand's gripper. Extending the queries to wrist links is planned (T46f) and has no result. -->
