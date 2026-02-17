# KD Interpretation Landscape — Research Exploration Map

> **Purpose:** scaffold for deeper literature dives. Each section = a research angle explaining *what KD transfers and why it works*, not methods for doing KD better. Sections marked with `[DEEP DIVE]` need full reads; `[SKIM]` means awareness-level; `[CORE]` means directly relevant to our thesis.

---

## 1. Dark Knowledge & Soft-Label Analysis

What do soft targets actually encode beyond the hard label?

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Hinton, Vinyals & Dean 2015 | NeurIPS-W | Soft targets encode inter-class similarity ("dark knowledge"); temperature controls how much relational structure is exposed |
| Tang et al. 2020 | arXiv | Decomposes teacher knowledge into 3 levels: universe (label smoothing), domain (class-relationship geometry), instance-specific |
| Cheng et al. 2020 | CVPR | Quantifies visual concepts — KD makes DNNs learn *more* concepts *simultaneously*, with more stable optimization |
| "Exploring Dark Knowledge under Various Teacher Capacities and Addressing Capacity Mismatch" 2026 | Frontiers CS | Distinctness among incorrect classes is the essence of dark knowledge; larger teachers *lack* this → capacity mismatch |

**What to dig into:**
- `[DEEP DIVE]` Tang et al.'s 3-level decomposition — maps onto our PCA/ICA decomposition nicely
- `[DEEP DIVE]` Cheng et al.'s concept quantification — could our eigenvectors correspond to "visual concepts"?
- `[SKIM]` Capacity mismatch literature (helps frame *when* our lens fails)

**Connection to us:** Our PCA/ICA decomposition of teacher activations could directly quantify *which components* of dark knowledge carry inter-class vs. instance-specific information. Eigenvector analysis of soft-label distributions reveals the rank and spectral structure of relational information.

---

## 2. Representation Similarity Metrics (CKA, CCA, SVCCA) `[CORE]`

The toolbox people use to compare teacher-student representations.

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Raghu et al. 2017 (SVCCA) | NeurIPS | SVD + CCA finds important directions; subspace directions disproportionately important vs neuron-aligned |
| Morcos et al. 2018 (PWCCA) | NeurIPS | Networks that *generalize* converge to more similar representations than ones that *memorize* |
| Kornblith et al. 2019 (CKA) | ICML | CCA fails when dim > n_samples; CKA reliably identifies correspondences across initializations |
| Saha, Bialkowski & Khalifa 2022 — "Distilling Representational Similarity using CKA" | BMVC | CKA *as* a distillation loss — student learns features at different scale to teacher |
| Zhou et al. 2024 (RCKA) | IJCAI | Decouples CKA into upper bound of MMD + constant; existing CKA-based KD "fails to uncover the essence of CKA" |

**What to dig into:**
- `[DEEP DIVE]` Zhou et al. 2024 — CKA-MMD decomposition could inform our theoretical framing
- `[DEEP DIVE]` Morcos et al. — generalization ↔ representational similarity link is central to our argument
- `[SKIM]` Saha et al. — CKA-as-loss is method-focused but useful context

**Key gap we fill:** Most work uses CKA/CCA as a *metric* or *loss*. Nobody uses PCA/ICA *on top of* alignment measures to decompose *which specific representational components* align. That's us.

---

## 3. Information-Theoretic Explanations

Mutual information, information bottleneck, compression perspectives.

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Tishby & Zaslavsky (IB principle) | — | Networks trade compression of input against preservation of task-relevant information |
| "Text Repr. Distillation via IB" 2023 | arXiv | Maximize MI(teacher, student) while minimizing MI(student, input) |
| "IT Criteria for KD in Multimodal" 2025 | arXiv | KD effective when MI(teacher, student) > MI(student, labels) — formal condition |
| SGD-Based KD with Bayesian Teachers 2026 | arXiv | Bayesian posteriors → variance reduction; excess risk bounds via L2 teacher-posterior distance |

**What to dig into:**
- `[SKIM]` IB-based KD — useful framing but orthogonal to our spectral approach
- `[SKIM]` Bayesian teacher paper — variance reduction is interesting but not our angle

**Connection to us:** PCA eigenvalues = variance explained ≈ information content. Spectral structure of teacher-student covariance matrices relates to the information "channel" between them. ICA independence criterion → which information channels are preserved vs. lost.

---

## 4. Geometric / Manifold Perspectives `[CORE]`

Understanding distillation through the geometry of learned representations.

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Park et al. 2019 (RKD) | CVPR | Transfer *mutual relations* (pairwise distances, angles) not individual outputs |
| GeomKD (Zhang et al.) 2026 | ICONIP 2025 / Springer | Riemannian curvature quantifies transfer limits; Fisher information matching |
| Bhattarai, Amjad, Zhylko & Alhanai 2025 | arXiv | Spherical geometry + Procrustes distance "more faithfully captures geometric alignment than CKA" |
| Saadi & Wang 2025 (Flex-KD) | arXiv | Distillation should retain *dominant directions of functional contribution*, not full features |

**What to dig into:**
- `[DEEP DIVE]` Saadi & Wang — "dominant directions of functional contribution" is essentially what PCA extracts; their framing could strengthen ours
- `[DEEP DIVE]` Bhattarai et al. — if Procrustes > CKA, what does that mean for our analysis?
- `[SKIM]` GeomKD — curvature-based view is interesting but probably too far from our scope

**Connection to us:** Our eigenvectors define the principal directions of the representation manifold. Comparing teacher-student eigenspaces *is* comparing manifold geometry. Flex-KD's theoretical framing provides natural language for our decomposition work.

---

## 5. Linear Decomposition / Spectral Approaches `[CORE — MOST DIRECT]`

The closest existing work to ours.

| Paper | Venue | Key claim |
|-------|-------|-----------|
| **"All You Need Is a Tailored Coordinate System" 2024** | arXiv (2412.09388) | **Dark knowledge is encoded in the PCA coordinate system of teacher features.** Single forward pass captures this. Teacher-free after extraction. SOTA with half training time. |
| Chiu et al. 2022 | CVPR | PCA as dimension reduction in KD for style transfer |
| Neural Spectral Decomp. 2024 | ECCV | SVD-based decomposition for *dataset* distillation; information per dimension is low-rank |
| Meller et al. 2023 (SVR) | PMLR | SVD of successive linear maps → graph representation of networks via meaningful input/output directions |
| Guillaume Alain & Yoshua Bengio 2016 | ICLR-W 2017 | Linear classifier probes on frozen intermediate representations; foundational probing work |
| C2G-KD 2025 | arXiv | PCA constraints on generator for data-free KD |

**What to dig into:**
- `[DEEP DIVE — CRITICAL]` TCS paper (2412.09388) — closest existing work. They show PCA subspaces encode dark knowledge but focus on *method design*, not *interpretation*. We need to clearly differentiate.
- `[DEEP DIVE]` Meller et al. SVR — SVD of linear maps across layers is very close to our layer-wise analysis
- `[SKIM]` Others — useful context but method-oriented

**Key gap we fill:**
1. TCS uses PCA for a method; we use PCA/ICA/CKA for *systematic interpretation*
2. Nobody compares PCA (variance-dominant) vs. ICA (statistically independent) vs. eigenvector analysis head-to-head as explanatory lenses
3. Nobody characterizes *which* eigenvectors carry which types of information (inter-class similarity, instance-specific, regularization)

---

## 6. Feature Attribution / Interpretability

Gradient and attention-based explanations of what transfers.

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Adhane et al. 2025 (UniCAM) | WACV | Distinguishes *distilled features* (teacher-guided, focused on textures/parts) from *residual features* (diffused, irrelevant areas). Proposes FSS and RS metrics. |
| Integrated Gradients for KD 2025 | arXiv | Feature-level guidance via integrated gradients, complementary to output-level KD |

**What to dig into:**
- `[DEEP DIVE]` Adhane et al. — distilled vs. residual feature distinction maps onto our PCA/ICA components (aligned vs. unaligned)
- `[SKIM]` IG paper

**Connection to us:** Their distilled/residual distinction is qualitative. Our decomposition makes it quantitative: components aligned between teacher-student = distilled; components present in one but not other = residual.

---

## 7. Theoretical / Generalization Bounds

Formal analyses of *why* distillation improves generalization.

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Phuong & Lampert 2019 | ICML | First theoretical analysis for linear/deep-linear classifiers. Three factors: *data geometry*, optimization bias, strong monotonicity |
| Liu 2025 | WIREs | Comprehensive review of theoretical perspectives |
| High-dim analysis of KD 2024 | — | Sharp risk characterization for high-dim regression; non-asymptotic bounds |

**What to dig into:**
- `[DEEP DIVE]` Phuong & Lampert — "data geometry" factor directly connects to our eigenvalue analysis; spectral structure of covariance determines when KD succeeds
- `[SKIM]` Liu review — useful for related work section

**Connection to us:** Their data geometry factor *is* the spectral structure we analyze. Linear-network setting = where our decomposition is most theoretically grounded. Potential formal backbone for our empirical findings.

---

## 8. Label Smoothing / Regularization Interpretation

Is KD just fancy regularization?

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Yuan, Tay, Li, Wang & Feng 2020 | CVPR | KD ≈ *learned* label smoothing. Even poorly-trained teachers improve students. |
| Zhou, Song, Chen, Zhou, Wang, Yuan & Zhang 2021 | ICLR | Soft labels = supervision + regularization. Bias-variance tradeoff varies sample-wise. |
| Zhang & Sabuncu 2020 | NeurIPS | Self-distillation ≈ instance-specific label smoothing |
| "KD ≈ Label Smoothing: Fact or Fallacy?" | OpenReview | Challenges the equivalence |

**What to dig into:**
- `[DEEP DIVE]` Yuan et al. — if poorly-trained teachers work, what does that mean for representation alignment? Does our PCA analysis show alignment even with bad teachers?
- `[SKIM]` bias-variance papers

**Connection to us:** The regularization perspective means some of what our decomposition detects might be regularization, not genuine knowledge transfer. ICA could potentially *separate* the uniform smoothing signal from structured, class-dependent patterns.

---

## 9. Loss Landscape / Optimization

Does KD work because of *what* is transferred or *how* it changes optimization?

| Paper | Venue | Key claim |
|-------|-------|-----------|
| **Stanton et al. 2021** | NeurIPS | **Students don't match teachers in function space** even with capacity. "Surprisingly large discrepancy." Optimization problem, not capacity. |
| Mirzadeh et al. 2020 (Teacher Assistant) | AAAI | Capacity gap affects loss landscape; teacher assistants bridge it |

**What to dig into:**
- `[DEEP DIVE — CRITICAL]` Stanton et al. — if function-space matching fails, then our representation-space decomposition may capture the *actual mechanism* by which KD helps (partial alignment, not full imitation)
- `[SKIM]` Teacher assistant — useful context for capacity mismatch

**Connection to us:** If students don't match teachers in function space, what *does* align in representation space? Our decomposition can precisely characterize the partial alignment. This could be a key framing for our thesis.

---

## 10. Emerging: Topology

| Paper | Venue | Key claim |
|-------|-------|-----------|
| "Topological Persistence Guided KD for Wearable Sensor Data" (Jeon et al.) 2024 | Eng. Appl. of AI (ScienceDirect) | Persistent homology captures topological structure (connected components, loops, voids) |

**What to dig into:**
- `[SKIM]` — complementary to our approach (they capture qualitative structure, we capture metric structure). Worth mentioning in future work.

---

## 11. Nonlinear ICA Theory (Foundation)

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Hyvarinen et al. 2023 | Patterns | Linear ICA identifiable; nonlinear ICA needs auxiliary info. With conditioning (e.g., class labels), nonlinear ICA becomes identifiable. |

**What to dig into:**
- `[DEEP DIVE]` Identifiability results — our ICA on neural activations is linear ICA on nonlinear features. Hyvarinen's work could strengthen theoretical foundation.

---

## Priority Queue for Deep Dives

**Tier 1 — Must read before writing Chapter 4/5:**
1. TCS paper (2412.09388) — closest work, need clear differentiation
2. Stanton et al. 2021 — key puzzle our work can address
3. Phuong & Lampert 2019 — theoretical grounding for spectral analysis
4. Saadi & Wang 2025 (Flex-KD) — "dominant directions" framing

**Tier 2 — Should read for depth:**
5. Zhou et al. 2024 (RCKA) — CKA-MMD decomposition
6. Tang et al. 2020 — 3-level knowledge decomposition
7. Adhane et al. 2025 (UniCAM) — distilled vs. residual features
8. Morcos et al. 2018 — generalization ↔ representational similarity
9. Yuan et al. 2020 — regularization interpretation

**Tier 3 — Useful context:**
10. Meller et al. 2023 (SVR) — SVD across layers
11. Cheng et al. 2020 — concept quantification
12. Hyvarinen et al. 2023 — ICA identifiability

---

## Gaps Our Thesis Fills (Working Thesis Positioning)

1. **Interpretation, not method.** Everyone uses PCA/CKA to build better KD. We use it to *explain* KD.
2. **Head-to-head decomposition comparison.** No one compares PCA vs. ICA vs. spectral analysis as explanatory lenses for distillation.
3. **Spectral characterization of dark knowledge.** Which eigenvectors carry inter-class similarity? Which carry instance-specific info? Which are regularization noise?
4. **Reconciling the Stanton puzzle.** Function-space matching fails, so what *does* align? Our representation decomposition can answer this.
5. **Separating regularization from knowledge.** ICA can potentially tease apart uniform smoothing from structured knowledge.
6. **Layer-wise decomposition dynamics.** How does spectral teacher-student alignment change across depth?
