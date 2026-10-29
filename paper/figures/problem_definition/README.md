# Three gaps between object grounding and safe contact

`three_gaps_safe_contact.svg` is the editable source and
`three_gaps_safe_contact.png` is the 3200×1800 presentation export.
The raster panel assets are embedded in the SVG, so the figure can be moved or
opened without resolving the original relative image paths.

The figure is a **conceptual motivation / problem-definition figure**, not a
quantitative result figure.

## Panel sources

- Policy Grounding Gap
  - `../paperbanana/figure2/assets/v2_segmenter_mask.png`: GT target mask used as
    a conceptual stand-in for an ideal external segmenter.
  - `../paperbanana/figure3/assets/f3_att_right_wrist.png`
  - `../paperbanana/figure3/assets/f3_target_cluster.png`
- Safety Geometry Gap
  - `../paperbanana/figure2/assets/f2b_ball_render.png`
  - `../paperbanana/figure2/assets/f2c_ownership_render.png`: boundary treatment
    applied offline to the recorded scene; do not present as a paired rollout.
- Contact Permission Gap
  - `../paperbanana/figure2/assets/f2c_link_render.png`
  - `../paperbanana/figure2/assets/v2_role_attached.png`
  - `../paperbanana/figure2/assets/v2_role_released.png`

## Suggested caption

**Three gaps between object grounding and safe contact.** An external object
mask does not identify the object currently relevant to the VLA policy (left).
Correct target identity does not determine a safe 3D contact-exemption boundary;
coarse exclusion may remove neighbouring surfaces (middle). The reconstructed
geometry must therefore be exposed selectively according to robot link and
manipulation phase, while an attached object remains collision geometry during
transport and returns to the scene after release (right).
