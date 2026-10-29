# Figure 2 caption (v2 draft)

**What may be touched, by which link, and when — all decided by one frozen VLA.** *Top:* AG3S reads
attention and hidden state from the same frozen policy that produces the action chunk and turns them into
a single link-specific contact representation, which a constrained refinement (CCR) enforces.
*Bottom:* each contribution against the closest prior practice. **C1 What:** instead of an external
segmenter, the policy's own attention is lifted into the target's 3D extent. **C2 Who:** instead of freeing
a region for every link, only target-owned surface is freed, and only for the grasping links; the touching
neighbour stays an obstacle. **C3 When:** instead of a fixed role, a sub-task probe (HSR) and the grasp state
move the object from target to attached object to released obstacle.

<!-- Scope (do not print): C1 prior = GT target label of T53 ep1842 standing in for an ideal segmenter;
C1/C2 AG3S panels = T53 C1a ep1842 renders; C3 = T43 V11AB ep1982 s19822 chunks 5 (target), 22 (attached,
carried by FK), 34 (placed, HSR=home, exclusion released; recorded scene ESDF). -->
