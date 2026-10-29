# Reference analysis R3 (draft 2): web search for contributions 1–3 and missing foundations

2026-10-06 KST. A literature sub-agent searched the web; the writer re-read the arXiv abstract page (or HTML) of every
paper that is cited in the manuscript. Items marked UNVERIFIED were not cited. No page numbers: the local PDF reader failed
for lack of poppler.

## Papers that touch the contributions

| Paper | Reads / does | Relation to us |
|---|---|---|
| KNOWS, arXiv 2606.09749 | pi0.5 action->vision attention (L12,H3) ranks objects segmented by a fine-tuned YOLOE-seg; the winner is exempted; ellipsoids + per-step CBF-QP on the end effector | Claims attention-based Target selection with exemption. We differ: no segmenter (depth clustering grows the object), dense ESDF, chunk SQP |
| WBAG, arXiv 2610.01083 (1 Oct 2026) | Frozen pi0.5; target/goal/support excluded with simulator-given identities; whole-body and grasp-conditioned attached geometry; CBF constraints; SafeLIBERO | Closest on exemption + payload. Differences: perceived Target, dense ESDF, chunk SQP, phase probe; WBAG protects the whole body, our evaluated configuration protects gripper links only |
| OmniGuide, arXiv 2603.10052 | Repeller SDF from the collision point cloud; sec. III-C: "We utilize CLIP features and mesh rendering to remove task-relevant regions like the object of interest and robot body from the collision point cloud." | Removes the object for every query; not attention based |
| Multi-Link Safety Filtering, arXiv 2609.40007 | Five ellipsoids (gripper, wrist, forearm); the grasp target is not exempted | Whole-arm, no exemption |
| Constrained flow matching, arXiv 2607.01378 | Minimum-norm correction of predicted trajectories during denoising | Nearest chunk-level correction (inside denoising) |
| NUDGE, arXiv 2609.35231 | SDF gradients injected into generative-policy sampling | Sampling-time guidance |
| Kang et al., arXiv 2503.06287 (CVPR 2025) | Few attention heads of a frozen LVLM localise a referred object | Method precedent for head-based attention grounding (VLM, text->image) |
| Jeong et al., arXiv 2603.13782 | Attention heads of a frozen navigation VLA detect path deviation | Same framing, detection only |
| Bhardwaj et al., arXiv 2608.13474 | Linear probe on pi0.5 residual stream decodes task progress (continuous) | Closest "linear probe on frozen pi0.5"; not a discrete label, not a gate |
| Zhang et al., arXiv 2605.28527 | Linear probes on frozen pi0.5/OpenVLA features predict success value | Probe on frozen VLA; success, not phase |
| ProbeAct, arXiv 2606.09740 | Probe on OpenVLA-OFT hidden states regresses object positions | Hidden-state probe to 3D position; the arXiv page prints "Shariar" |
| SAFE, arXiv 2506.09937 (NeurIPS 2025) | Failure score from VLA internal features | Hidden-state failure detector |

## Findings for the claims

1. Contribution 1: "first to use VLA attention to pick the Target" is not available (KNOWS). The claim that survives is
   segmenter-free, attention-seeded object extraction in 3D that feeds a dense distance field.
2. Contribution 2: no paper found that reads a discrete pick/place/home label from frozen pi0.5 hidden states and uses it
   as a gate. The search is not exhaustive; repeat before submission. Internal evidence (SUBTASK_FACTS.md): the probe is
   not shown to beat a rule baseline and gives the same accuracy from layer 0, so claim "reads a closed phase label", not
   "reasons about sub-tasks".
3. Contribution 3: part-wise overlap with WBAG, OmniGuide and the shared-autonomy paper arXiv 2607.17323 (not cited).
   "Exempt only for gripper links" was not stated by any paper opened.

## Unverified / not cited

SimpleVLA-Guard, GUARD (2608.04510), FailureSpot (2609.04277), SHIELD/FAPS, Affordance Field Intervention (2512.07472;
masking claim from a snippet), RB-Y1 platform paper (none found; only the manufacturer page), LIBERO (2306.03310; cite when
the benchmark is used), SafeLIBERO (introduced in VLSA).
