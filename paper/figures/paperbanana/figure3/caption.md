# Figure 3 caption (draft)

**One frozen VLA supplies both the action and the geometry it may touch.** For every action chunk,
the frozen policy produces a nominal chunk. A second pass of the same checkpoint returns
action-to-image attention and prefix features. Attention seeds multi-view 3D grounding: back-projection,
voxel fusion, region growing, candidate filtering and confirmation over frames. A sub-task probe on the
frozen features stops new targets from being adopted during place and home. The confirmed target bounds
a removal ball, and an ownership split keeps every cell nearer a non-target surface. Contact links of the
active hand query the resulting target-free layer; all other links query the cuRobo ESDF of the scene. Once
a grasp state machine confirms the grasp from executed gripper feedback, the object becomes an attached
sphere that queries the scene field. An executed-horizon SQP refines the chunk under
$d(x) - r \ge m$, and a post-check either executes the refined chunk or holds the arm.

<!-- Scope note (do not print): the manipulation phase is supplied by the client, not inferred
(constant "approach" in the evaluated runs). Attention uses layer 8, head 2 of the action expert
(serving default, unchanged by T47-T51). The evaluated arm plans 16 rows (8 executed + 8 look-ahead). -->
