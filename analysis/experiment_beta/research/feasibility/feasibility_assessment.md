# Thesis Feasibility & Novelty Assessment

> **Question being answered:** Is this thesis recreating a wheel someone else has already made?
> **Short answer:** No. The specific combination does not exist. But the landscape has gotten active — details matter.

---

## 1. What the Thesis Claims to Do

Apply a suite of linear decomposition tools — **PCA, ICA, CKA, principal angles, effective dimensionality** — across intermediate layer activations of teacher/student model pairs trained with **multiple distillation methods** (logit, attention, feature-based, relational) to *understand structurally what each method transfers and why*.

The end product is **interpretation**, not a new KD method.

---

## 2. Threat Map — What Could Kill Novelty

| Threat | Verdict | Detail |
|--------|---------|--------|
| Someone already used PCA *analytically* on distilled representations | **SAFE** | TCS (AAAI 2025) uses PCA as a distillation mechanism, not as an analytical lens. Zero interpretive analysis of what components mean. |
| Someone already applied ICA to distilled model activations | **VERY SAFE** | No paper found in any search. ICA + KD appears only in medical imaging (ADHD classification, functional connectivity) — completely different context. |
| Someone already measured effective dimensionality of distilled vs. non-distilled models | **SAFE** | Effective rank paper (2512.00792) does this for ViTs but not in a KD context and not comparing distillation methods. |
| Someone already used principal angles to compare teacher/student subspaces | **SAFE** | VkD (CVPR 2024) uses orthogonal projections as a KD *loss*, not as an analytical measurement. CosPress (2411.15239) preserves cosine similarity as a method. No one measures these as interpretive quantities. |
| Someone already did a systematic multi-method comparison of what different KD methods transfer at the representation level | **SAFE** | Several benchmark repos compare *accuracy* across methods. No paper applies decomposition tools across multiple methods to compare *representation structure*. |
| CKA on KD is already saturated | **PARTIAL THREAT** | Saha et al. (BMVC 2022) use CKA as a distillation loss. Zhou et al. RCKA (IJCAI 2024) decompose CKA. But no one uses CKA systematically *across methods* as an analytical tool comparing logit vs. attention vs. feature distillation. |

**Bottom line:** The specific thesis angle — linear decomposition as interpretive framework across multiple distillation methods — does not exist in the literature. The closest work uses these tools to *build methods*, not to *explain phenomena*.

---

## 3. Closest Competing Papers (Read These Carefully)

### 3.1 TCS — "All You Need Is a Tailored Coordinate System" (Zhou, Zhu & Wu, AAAI 2025)
**arXiv:** 2412.09388

**What they do:** Extract a PCA coordinate system from teacher features (one forward pass). Student aligns to this coordinate system. Achieves SOTA with half training time.

**What they DON'T do (your space):**
- They never examine what eigenvectors encode semantically
- No comparison of spectral structure between teacher and student
- No layer-wise alignment analysis
- No comparison across different distillation objectives
- Confirmed by reading the full paper: "purely method paper, lacks analytical depth" — the only interpretive experiment confirms the method works, not why

**Differentiation:** "TCS uses PCA to engineer a better distillation objective. We use PCA (and ICA, and CKA, and principal angles) to *measure* what happens when you use any distillation objective. These are complementary, not competing."

---

### 3.2 VkD — "Improving KD using Orthogonal Projections" (Miles et al., CVPR 2024)
**arXiv:** 2403.06213

**What they do:** Constrain the student's projection layer to be orthogonal, preserving intra-batch feature similarity structure. SOTA on ImageNet.

**What they DON'T do (your space):**
- No measurement of how much subspace overlap exists between teacher and student
- Orthogonality is a constraint they impose, not a quantity they measure
- Single distillation objective, no cross-method comparison
- No spectral analysis of the representations

---

### 3.3 CosPress — "Preserving Angles Improves Feature Distillation" (Mannix et al., 2024)
**arXiv:** 2411.15239

**What they do:** Preserve cosine similarities between embeddings as a distillation objective. Better robustness and OOD detection.

**What they DON'T do (your space):**
- Cosine similarity preservation is the method, not a measurement tool
- No decomposition of which angular relationships matter
- No comparison to other distillation methods' angular properties

---

### 3.4 Flex-KD — "What Should Feature Distillation Transfer in LLMs?" (Saadi & Wang, 2025)
**arXiv:** 2507.10155

**What they do:** Gradient-based importance scoring identifies which teacher representation dimensions functionally matter per task. Selects top-d_S dimensions for distillation. Outperforms projection-based methods on LLMs.

**What they DON'T do (your space):**
- Gradient-based functional importance ≠ variance-based PCA importance
- No ICA analysis
- LLM-focused, no CIFAR CNN comparison
- No cross-method comparison (single distillation objective)
- No ICA to find statistically independent components

**Key relationship:** Flex-KD provides a theoretical framing ("functional geometry") that *supports* our approach. Their gradient-based "dominant directions" and our PCA "principal components" are measuring different but related things. If they align → interesting. If they diverge → even more interesting.

---

### 3.5 LELP — "Linear Projections of Teacher Embeddings for Few-Class Distillation" (2024)
**arXiv:** 2409.20449

**What they do:** Identify informative linear subspaces in teacher's embedding space, split into pseudo-subclasses for student training. NLP-focused, few-class problems.

**What they DON'T do (your space):**
- Analytical study of representations, not empirical comparison
- No ICA, no effective dimensionality, no layer-wise analysis
- NLP domain only

---

### 3.6 Geometry-Aware Representational Alignment (2024)
**arXiv:** 2509.25253

**What they do:** Argue that CKA and MSE cannot capture feature structure even at zero loss. Propose Procrustes distance and Feature Gram Matrix Frobenius norm as better distillation losses.

**What they DON'T do (your space):**
- Analysis reveals CKA's theoretical limits, but proposes a new method as solution
- No PCA/ICA component analysis
- No cross-method empirical comparison
- Language model domain

---

## 4. The Actual Novelty Space (What You Own)

```
Existing work:
  PCA / subspace / angles → used to BUILD better KD methods
  CKA → used to BUILD better KD methods (occasionally used analytically, but single-method)
  ICA → not used in KD context at all
  Effective dimensionality → not measured in KD context
  Multi-method systematic comparison → done for accuracy, never for representation structure

Your thesis:
  All of the above → used to EXPLAIN what different KD methods do to representations
  ICA → first application to KD intermediate layer activations
  Layer-wise spectral analysis → first depth-resolved characterization across methods
  Cross-method comparison → first systematic "what does each method transfer" via decomposition
```

**The core novelty claim:** *This is the first study to use linear decomposition methods as an interpretive framework — rather than as engineering components — to systematically characterize and compare what different knowledge distillation objectives actually transfer in intermediate representations.*

---

## 5. The Framing That Makes This Defensible

Your thesis needs a clean research question. Based on the gap analysis, the strongest framing is:

> **"What structural properties of intermediate representations enable, constrain, and characterize effective knowledge distillation?"**

This is answered through three sub-questions:
1. **PCA lens:** How does distillation reshape the variance structure of representations (eigenspectrum, effective dimensionality)?
2. **ICA lens:** Does distillation produce statistically independent components in student activations, and do these match the teacher's independent components?
3. **CKA + principal angles lens:** Which layers achieve strong teacher-student alignment, and does this vary by distillation method?

Each of these is answerable with your current experimental setup (6 methods × 4 datasets × 3 architectures).

---

## 6. What Could Make It Even Stronger

In order of implementation feasibility:

### 6.1 The "Stanton Follow-up" Angle (HIGH VALUE, LOW COST)
Stanton et al. (NeurIPS 2021) showed students don't match teachers in function space even with capacity to spare. They never looked inside at representations. **You can directly answer: what does align?** PCA/ICA shows which specific components transfer despite function-space failure. This positions your thesis as directly extending a widely-cited open question.

### 6.2 The "Regularization vs. Knowledge" Separation (HIGH VALUE, MEDIUM COST)
Yuan et al. (CVPR 2020) argue KD ≈ label smoothing. ICA should be able to separate a class-independent smoothing component (uniform across classes) from a structured class-relationship component (correlated with inter-class similarity). This is a concrete testable hypothesis ICA is uniquely suited for.

### 6.3 The Capacity Gap Lens (MEDIUM VALUE, LOW COST)
Mirzadeh et al. (AAAI 2020) show larger teacher-student gaps hurt distillation. Your ResNet112→ResNet56→ResNet20 pairs let you directly ask: as capacity gap grows, which PCA components are lost first? Are the top-k components always preserved, or does distillation fail selectively in the mid-rank spectrum?

### 6.4 Method Taxonomy From Decomposition (HIGH VALUE, MEDIUM COST)
After running all 6 distillation methods through your analysis, you may find natural clusters: "feature-based methods (FitNets, AT, NST) align mid-layer spectral structure while logit-based methods (Hinton) only align final-layer geometry." This taxonomy *from the data* rather than from method design is a genuinely publishable finding.

---

## 7. Risk Register

| Risk | Probability | Mitigation |
|------|------------|------------|
| All methods produce nearly identical spectral alignment (null result) | Low-medium | Null result is still publishable: "distillation methods are representationally equivalent despite different objectives" — that itself is interesting |
| TCS (AAAI 2025) is too close and reviewers confuse it with your work | Low | Clear differentiation: TCS engineers a method, you measure representations. Cite it prominently and distinguish in intro. |
| ICA on CNN activations doesn't converge or gives uninterpretable components | Medium | Well-known issue with noisy activations. Mitigation: apply FastICA on GAP-pooled layer outputs, condition on class, use multiple random restarts. Hyvarinen's identifiability results explicitly cover this case. |
| Very recent paper (2025) does exactly this while you're finishing | Low | The combination (PCA + ICA + CKA + principal angles) as a unified interpretive framework across multiple methods is hard to replicate accidentally. Even if a paper uses 1-2 of these analytically, the systematic multi-tool comparison remains novel. |
| CIFAR-scale models too small to draw general conclusions | Medium | Frame as "controlled study on small-scale models where ground truth is known" — standard in representation analysis literature. SVCCA, CKA papers all use CIFAR/ImageNet. |

---

## 8. Verdict on Your Current Plan

```
1. Train a bunch of things ✓ (6 methods × 4 datasets × 3 architectures = good factorial design)
2. Apply analysis tools ✓ (PCA, ICA, CKA, principal angles, effective dimensionality)
3. Look at results ✓
4. Refine analysis tools iteratively ✓ (scientifically sound, especially for ICA hyperparams)
5. Find interesting results ✓ (almost certain given nobody has looked)
```

**The plan is sound.** The risk of wheel-reinvention is low. The risk of null results is non-zero but manageable (even a null result is interpretable and novel at this scope).

The one thing that WOULD kill novelty: a paper appearing that *uses PCA + ICA together analytically across multiple distillation methods on CNN intermediate layers*. That specific combination does not exist as of February 2026.

---

## 9. Potential Pivots (If Needed)

If something too close surfaces during writing, here are defensible pivots that don't require rerunning experiments:

1. **Focus entirely on ICA** — no one has done ICA on distilled CNN representations. Narrow scope but completely clear blue ocean.
2. **Focus on the Stanton puzzle** — "what does align in representation space when function-space matching fails?" — explicitly framed as follow-up to NeurIPS 2021.
3. **Method taxonomy pivot** — let the empirical results define a taxonomy of distillation methods by their representational effects, rather than claiming comprehensive analysis of all tools.
4. **Layer-wise dynamics** — focus exclusively on how spectral alignment evolves across depth, which no existing work covers for any distillation method.

---

## 10. Papers to Track Going Forward (Pre-Publication Check)

Run these search queries monthly as you write:
- `"ICA" OR "independent component" "knowledge distillation" representations`
- `"PCA" "knowledge distillation" analysis interpret`
- `"effective dimensionality" OR "participation ratio" "distillation"`
- `"principal angles" "teacher" "student" representations`

Set a Google Scholar alert for: `knowledge distillation representation analysis spectral`

---

*Assessment compiled: February 2026. Based on searches across arXiv, Google Scholar, CVPR/NeurIPS/ICML/ICLR/AAAI/IJCAI proceedings 2019–2025.*
