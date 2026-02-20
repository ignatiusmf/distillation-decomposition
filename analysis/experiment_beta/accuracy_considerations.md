# Accuracy Gap Considerations

## The Concern

In experiment_alpha, KD methods typically produced only ~0.5–1% accuracy improvement over a pure student (no distillation). On CIFAR-10 the gap is even smaller — the pure ResNet-56 student already reaches ~93.2%, and most KD methods land in the 93.5–94.1% range. On CIFAR-100 the teacher-student gap itself is only ~1.1%.

If all methods produce nearly identical top-line accuracy, it becomes harder to make a compelling argument that the *method of knowledge transfer* matters — which is the whole thesis premise.

## Experiment Alpha Numbers (for reference)

**CIFAR-10:** Pure student 93.2% → best KD (attention transfer) 94.1%. Gap: ~0.9%.
**CIFAR-100:** Pure student 71.2% → best KD (logit, seed 2) 73.2%. Gap: ~2.0%.

## Counter-argument: Latent Space Differences Are Still Valid

Even if top-line accuracy is similar, the *internal representations* can differ substantially. Two models can agree on class labels for 93% of test images while organising their latent spaces in fundamentally different geometric configurations. This is arguably the more interesting finding:

- **Same accuracy, different geometry** → the KD method shaped *how* the student represents knowledge, not just *whether* it gets the right answer.
- PCA variance profiles, effective dimensionality, CKA alignment, and principal angles can all reveal structural differences invisible to accuracy alone.
- This is exactly the "explanatory gap" the thesis addresses: KD works, accuracy improves (or stays the same), but *what changed inside the network?*

## Practical Reality

Larger accuracy gaps would make the analysis *easier to interpret* — clearer signal in the representation metrics, more dramatic before/after comparisons, stronger narratives in the results chapter.

## Implications for Experiment Beta

Options to consider for getting more differentiated results:
- More aggressive hyperparameters (higher alpha, different temperature)
- Larger capacity gap between teacher and student
- Harder datasets where the student genuinely struggles without KD
- Fix RKD (completely broken in alpha — 18% CIFAR-10, 1% CIFAR-100)
- Ensure all 6 methods × all seeds are complete for fair comparison

---

## Claude's Analysis

### Why the gaps are small — it's the architecture pair

The core issue is that ResNet-56 is already a strong model for CIFAR-10/100. With 93.2% on CIFAR-10, it's within 0.3% of the teacher (93.55%). There simply isn't much room for KD to help. This is a well-known phenomenon in the KD literature — distillation benefits are most pronounced when the student is significantly weaker than the teacher. The ResNet-112→ResNet-56 pair shares the same 3-group architecture with the same channel widths (16→32→64); the only difference is depth (18 vs 9 blocks per group). They have roughly similar representational capacity for tasks this "easy."

### The numbers actually tell an interesting story already

Looking more carefully at the alpha results:

1. **Logit KD has alarming seed variance on CIFAR-100** (69.39%, 72.96%, 73.16% — a 3.77% spread). Seed 0 is *worse than pure*, while seed 2 *exceeds the teacher*. This isn't just noise — `alpha=0.5` means 50% of the gradient signal comes from matching teacher logits, and if the teacher's soft label distribution for seed 0 is somehow adversarial to the student's initialisation, it can genuinely hurt. This is worth investigating: does the logit seed-0 student have a *different* representation geometry than seeds 1-2, or did it just converge to a worse minimum with similar structure?

2. **FitNets hurt on CIFAR-100** (70.23% vs 71.17% pure). This is consistent with the known FitNets problem: forcing intermediate layer alignment can over-constrain the student if the hint/guided layer pairing is suboptimal. The student's layer 2 (32ch) may not have enough capacity to simultaneously match the teacher's layer 2 representation *and* learn features useful for classification. This is a representational conflict that our decomposition tools should be able to detect (e.g., lower effective dimensionality in the guided layer, or CKA showing high teacher-student alignment at the guided layer but low alignment at the output layer).

3. **Attention Transfer is the best method on both datasets** (94.12% CIFAR-10, 71.56% CIFAR-100) and the only one that consistently exceeds the teacher. AT's loss operates on attention maps (spatial statistics) rather than full representations, which gives the student more freedom in *how* it represents features while still inheriting *where* the teacher looks. This is a lighter constraint than FitNets or factor transfer, and the representation analysis should show it: AT students should preserve their own representation geometry more than FitNets students while still aligning on spatial attention patterns.

4. **RKD is catastrophically broken.** 18% on CIFAR-10 (10 classes → ~10% is random) and 1% on CIFAR-100 (100 classes → 1% is literally random). The RKD loss (distance + angle between sample pairs) may be dominating the CE loss at `alpha=0.5`, producing gradients that push the student to match inter-sample relations at the expense of learning any discriminative features. This needs debugging — try `alpha=0.1` or check if the RKD loss scale is orders of magnitude larger than CE.

### What this means for the thesis

The small accuracy gaps are actually *not* a problem for the thesis argument — they're the *whole point*. The thesis is about using linear decomposition to understand what KD does to representations. If all methods produced wildly different accuracies, the story would be simple: "better method → better accuracy." But when methods produce *similar* accuracy with potentially *different* internal representations, that's where decomposition analysis becomes essential. The thesis can argue:

> "Accuracy alone cannot distinguish between distillation strategies that produce similar performance. We show that linear decomposition of latent representations reveals systematic structural differences between methods that are invisible to top-line metrics."

This framing turns the "problem" (small gaps) into the "motivation" (we need better tools than accuracy).

### Practical recommendations for experiment beta

1. **Don't change the architecture pair.** Keeping ResNet-112→56 with small accuracy gaps actually *strengthens* the thesis narrative. The analysis tools need to work hardest (and prove their value most) when accuracy doesn't discriminate.

2. **Do fix RKD.** A working RKD that achieves ~93% on CIFAR-10 alongside the other methods gives you 6 methods with similar accuracy but different distillation objectives — ideal for showing that representation structure varies even when output performance doesn't.

3. **Do run all 6 methods × 3 seeds on all datasets.** The seed variance in logit KD (especially the CIFAR-100 anomaly) is genuinely interesting. You need 3 seeds everywhere to separate method effects from initialisation effects.

4. **Do fix SVHN** (`pip install scipy`). SVHN is a useful middle ground — harder than CIFAR-10 but with a different data distribution (natural images of house numbers vs object categories). It might show different KD dynamics.

5. **Consider TinyImageNet carefully.** The student (55.65%) *outperforming* the teacher (53.31% at epoch 127) is a red flag. Either the teacher needs more epochs, or the 64×64→reduced resolution pipeline isn't suited to deep ResNets. If the teacher can't beat the student, KD on TinyImageNet becomes meaningless.

6. **Alpha=0.5 is quite aggressive.** For methods like RKD where the distillation loss may have a very different scale than CE, consider method-specific alpha values or at minimum normalising the distillation loss. The literature often uses alpha=0.1–0.3 for feature-based methods.
