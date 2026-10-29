Create Figure 3 (method pipeline) of a CVPR paper, full text width, about 2.3 : 1. Follow the attached mockup (`mockup_v2.png`) for layout. Readers should be able to follow the whole method from left to right. Each stage is explained mainly by its image, with only a few words of text.

Style: clean vector academic figure, white background, rounded stage boxes with a pale fill, thin orthogonal arrows, sans-serif text, math in italic serif. Each stage has a bold name and a short subtitle. Keep labels to five words or fewer. No icons, shadows or 3D effects. Use the attached panels exactly as given: crop or resize them, but never redraw, regenerate or retouch them.

Layout, left to right:

1. Inputs (grey). Three stacked camera images (01a, 01b, 01c) labelled "RGB, 3 views", the instruction box "Put the pear in the basket.", and a grey box "depth (robot removed) + robot state". A thin grey line runs from that box along the bottom to ATS and LTE.

2. Frozen VLA (pale blue). A box reading "π0.5, no update" with a row of tokens (grey image tokens, one gold noun token, blue action tokens). Two outputs leave it: "noun attention" (gold arrow to ATS) and "features" (teal arrow to the identity check). An orange arrow labelled "KV" goes down to HSR. A blue lane across the top carries "action chunk A_t (policy output, unchanged)" straight into CCR.

3. ATS: "which object, where it ends" (pale gold). Three images in a row with arrows between them:
   02 "seed patches" → 03 "(Λ+L)m = Λy" → 04 "surface T_t".
   Below the row is the identity memory strip, which should read as "the method remembers what the target looks like and finds it again". Start with the teal words "same object?". Then:
   - 11a, an earlier frame with the remembered target region in orange, with an orange frame and the small caption "memory". Directly under it goes the first row of 11c, the remembered feature code.
   - an arrow to 11b, the current frame. Each candidate region is tinted from grey to orange by how similar it is to the memory: the target is bright orange, the touching orange fruit pale orange, everything else grey. Caption it "now".
   - next to 11b, the three lower rows of 11c, the feature codes of the three most similar candidates. The top row nearly matches the memory code; mark it with a small orange tick and fade the other two.
   - end with the small red words "low → alert".
   Draw no bar chart and print no numbers.

4. HSR: "when its role changes" (pale orange, under ATS). One image only: 05, the episode timeline. It has a purple "held" band on top and pick / place / home probability curves. Caption it "probe + gripper + geometry". Remove the earlier text rows "phase pick→place→home" and "grasp free→held→free"; the timeline replaces them. An orange arrow leads from HSR to the attached-object image in LTE.

5. LTE: "who may touch it" (pale purple). At the top is a wide box "cuRobo TSDF → ESDF" with two small images on its right: 10a → 10b, captioned "visible free-space carving". In 10a, red voxels mark where the moved apple used to be, still stored in the map. In 10b, yellow camera rays pass through that place and most of the red is cleared. Keep 10a and 10b the same size and crop.
   Below this box are two field slices side by side, the same size and aligned:
   06 captioned "d_t · other links" and 07 captioned "d_t^X · contact links".
   Under them, the rule δ_i = d_t^X iff i ∈ L ∧ e_t. At the bottom is image 08 with the text "after grasp: object → attached spheres".
   06 and 07 are a vertical cut through the target apple (orange) and the banana touching it (blue), resting on the table (tan); the blue glow is distance to the nearest obstacle. In 06 both fruits are obstacles. In 07 the target is gone, leaving only a dashed orange outline: the contact links no longer treat it as an obstacle, while the touching banana stays solid blue. This contrast is the key message, so keep both slices large enough to read. Do not add sphere drawings to them.

6. CCR (light grey). Image 09 (gold refined path vs dashed policy path over the red bar), then "refine A_t s.t. h_ik ≥ 0, run first K", then a box "check prefix" that splits into a green "execute" box and a red "hold" box.

Colours: target orange #E8711A, neighbour/obstacle blue #2E75B6, contact links gold #C99A06, attached object purple #7030A0, hold red #C00000, frozen VLA pale blue #DEEBF7.
Do not add modules, objects, numbers or comparisons that are not listed here.
