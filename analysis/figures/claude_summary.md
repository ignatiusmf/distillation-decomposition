What the figures tell you
PCA Variance & Effective Dimensionality
The teacher concentrates variance more tightly — 3 components capture 90% at layer 1, vs 4 for both students. At layer 2 the gap widens (11 vs 14). By layer 3 they converge again (~42). The students spread variance across more dimensions in early/mid layers, suggesting the deeper teacher learns a more compressed representation that the shallower students can't fully replicate.

CKA (Representation Similarity)
The big finding: CKA drops monotonically from layer 1 → 3. Early representations are nearly identical (0.97-0.98) but layer 3 drops to ~0.71. This means early feature extraction converges regardless of depth/distillation method, but later abstract representations diverge. The cross-layer heatmaps confirm representations stay layer-aligned (diagonal dominance) — no evidence of "layer shifting" where student layer 2 matches teacher layer 3.

Logit KD has slightly higher CKA at layers 1-2 (0.977, 0.921) vs factor transfer (0.971, 0.889), but factor transfer edges ahead at layer 3 (0.721 vs 0.708). This is interesting — factor transfer explicitly supervises intermediate features but actually produces less similar mid-layer representations, while matching better at the final layer where it ultimately matters for accuracy.

Principal Angles
Both students show nearly identical subspace alignment with the teacher. The top few principal directions are well-aligned (small angles), then angles grow steadily. No meaningful difference between logit KD and factor transfer — the variance subspaces are equally aligned.

ICA (Independent Components)
This is where the methods diverge most. At layer 3, logit KD preserves 43/64 independent components strongly (|corr| > 0.5) vs factor transfer's 40/64. At layer 1, ICA matching is weak for both (0.24, 0.27 mean correlation) — early features don't preserve teacher-specific independent factors. The heatmaps show a clear diagonal at layers 2-3 (components match up) but diffuse structure at layer 1.

The takeaway: ICA reveals structure that PCA/CKA miss. CKA says factor transfer is slightly better at layer 3, but ICA says logit KD preserves more independent factors there. These are different kinds of alignment.

Class Separability (Fisher Criterion)
Both distilled students have higher class separability than the teacher at layer 3 (logit: 1.18, factor: 1.06, teacher: 0.92). The smaller student is forced to be more discriminative in its final representation to compensate for fewer parameters. This is a genuine insight — distillation doesn't just copy the teacher's representation, it creates a more class-separative version.

PCA Scatter
The 2D projections (in teacher's PC basis) show all three models produce similar class cluster geometry at layer 3, but the students' clusters are tighter and more separated — consistent with the Fisher criterion result.

