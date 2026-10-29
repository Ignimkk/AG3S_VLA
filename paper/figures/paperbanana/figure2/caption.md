# Figure 2 caption (draft)

**AG3S turns the frozen policy's own attention into contact geometry.** (a) An end-to-end VLA never
checks contact and drives the apple into the bar. A geometric safety filter treats every surface as an
obstacle, so the target blocks its own grasp; removing the target with an external segmenter would instead
remove it for every link. AG3S reads attention and sub-task state from the frozen VLA, grounds the target in
3D, and builds a task- and link-conditioned field that a constrained refinement (CCR) uses to correct each
action chunk. (b) One scene: ① attention peaks on the apple; ② depth turns it into a 3D target cluster
separate from the touching pear and the bar; ③ the target's voxels are removed only for the grasping links;
④ a sub-task probe and the grasp state keep the target locked while placing and attach the held object;
⑤ CCR lifts the policy's tool path over the bar.

<!-- Scope (do not print): scene T43 V11AB ep1982 s19822 (overbar), replay renders; ③ uses the recorded
target-free ESDF of the same chunk; card (a) middle reuses the recorded scene ESDF with the target kept;
card (a) left = the Figure 1 VLA-only place frame (E0 s19823). -->
