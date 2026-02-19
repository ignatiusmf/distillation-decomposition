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

### Hinton, Vinyals & Dean 2015 — "Distilling the Knowledge in a Neural Network"
The paper introduces knowledge distillation via "soft targets": instead of training a student on hard labels, it trains on the full softmax output of a teacher (or ensemble), with a temperature parameter T that smooths the distribution to expose inter-class similarity structure. At higher temperatures, the softmax reveals how the teacher distributes probability mass across incorrect classes — this is the "dark knowledge" that hard labels discard. Experiments on MNIST and a large-scale speech recognition system (Google's acoustic model) show that a single compressed student recovers most of the ensemble's performance. The core claim is that the teacher's probability distribution over non-target classes carries rich similarity information that hard labels do not.

**Thesis connection:** This is the foundational paper defining the signal our thesis aims to decompose. PCA/ICA on student intermediate representations can reveal whether the inter-class similarity structure encoded in soft targets actually manifests as identifiable geometric structure in learned feature spaces, rather than only at the logit layer.

### Tang et al. 2020 — "Understanding and Improving Knowledge Distillation"
The authors decompose KD's effect into three hierarchical levels: (1) universe-level, where soft targets act as adaptive label smoothing providing regularization; (2) domain-level, where the teacher's probability distribution over incorrect classes encodes inter-class semantic relationships that shape the student's logit-layer geometry (shown via heatmap correlation analysis revealing block-diagonal class structure); and (3) instance-level, where the teacher's confidence on each example rescales per-instance gradients, effectively performing curriculum-like weighting. Critically, they diagnose two failure modes: label-smoothing applied to the teacher destroys the class relationship information in soft targets, and over-capacity teachers predict uniformly high confidence, eliminating the gradient rescaling signal.

**Thesis connection:** This decomposition directly motivates using PCA/ICA to separate these three effects in the student's intermediate representations. If domain-level knowledge produces specific geometric structure (e.g., cluster separability along principal components) while instance-level knowledge affects gradient variance, dimensionality-reduction tools should be able to disentangle them empirically in CIFAR teacher-student pairs.

### Cheng et al. 2020 — "Explaining Knowledge Distillation by Quantifying the Knowledge"
The authors define "visual concepts" as image regions where the network discards significantly less information, measured via pixel-wise conditional entropy. They introduce metrics for concept counts, discriminative ratio, learning speed/simultaneity, and optimization stability. Testing on AlexNet, VGG, and ResNet fine-tuned on ILSVRC-DET, CUB-200, and Pascal VOC, they find that distilled networks learn more foreground concepts, fewer background concepts, learn them more simultaneously rather than sequentially, and follow more stable optimization trajectories. The paper provides information-theoretic evidence that KD changes *what* representations encode, not just how well they perform.

**Thesis connection:** Their finding that distillation changes the structure and simultaneity of concept acquisition is exactly the kind of phenomenon PCA/ICA should capture. If distilled students learn concepts more concurrently, their principal component spectra should differ systematically from normally-trained networks, and CKA between teacher-student pairs should reveal whether this concurrent learning produces representational convergence.

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

### Raghu et al. 2017 — SVCCA
SVCCA combines SVD (to reduce each layer's activation matrix to its significant directions, filtering noise) with CCA (to find maximum-correlation alignments between two representations). This makes the comparison invariant to affine transformations and computationally tractable. Applied to CNNs and RNNs, they discover that networks converge "bottom up" during training — lower layers stabilize first — and that many layers are over-parameterized, with intrinsic dimensionality far below the number of neurons. They also propose "freeze training" (freezing converged lower layers early) as a practical speedup.

**Thesis connection:** SVCCA's SVD step is effectively PCA on layer activations, making it a direct methodological ancestor of our approach. The bottom-up convergence finding provides a testable hypothesis: in KD, does the student's bottom-up convergence pattern differ from a normally-trained network, and does SVCCA reveal that distilled students converge to teacher-like representations layer-by-layer?

### Morcos et al. 2018 — Projection-Weighted CCA
Introduces Projection-Weighted CCA (PWCCA), which weights CCA directions by how much variance they explain in the original representation, fixing SVCCA's equal-weighting of all canonical correlations regardless of importance. Key finding: networks which *generalize* converge to more similar representations than networks which *memorize*, measured across independently trained instances. They also find that wider networks converge to more similar solutions than narrow ones, and that different learning rates produce distinct representational clusters.

**Thesis connection:** The generalization-similarity link is directly relevant: if distilled students generalize better than normally-trained students of the same architecture (a common KD finding), PWCCA/CKA should show they converge closer to the teacher's representation. This provides a representational explanation for KD's generalization benefit that the thesis can test on CIFAR CNN pairs.

### Kornblith et al. 2019 — CKA
The paper proves a fundamental limitation: no similarity metric invariant to invertible linear transformations (including CCA, SVCCA, PWCCA) can produce meaningful results when representation dimensionality exceeds the number of data points, because any two random representations become perfectly correlated under CCA in that regime. They propose Centered Kernel Alignment (CKA), which operates on representational similarity matrices (Gram matrices) rather than raw activations, equivalent to HSIC normalized by representation norms. CKA reliably identifies correspondences between layers of networks trained from different initializations, where CCA-family methods fail. With linear kernels, CKA reduces to comparing correlation structure; with RBF kernels, it captures nonlinear relationships.

**Thesis connection:** CKA is our primary similarity metric. Its ability to compare representations across architectures of different widths (teacher vs. student) without requiring dimension-matching projections makes it the natural tool for measuring how much of the teacher's representational structure the student actually acquires during distillation.

### Zhou et al. 2024 — RCKA
The authors prove that maximizing CKA similarity is equivalent to minimizing an upper bound on MMD (Maximum Mean Discrepancy), plus a constant term that acts as a weight regularizer. They operationalize this as RCKA, applying CKA-based losses to both feature maps and logits. Tested on CIFAR-100 with pairs like ResNet-110/20, WRN-40-2/16-2, and cross-architecture pairs (ResNet-32x4 to ShuffleNet-V1), achieving state-of-the-art on most configurations.

**Thesis connection:** They use CKA as a training loss; we use CKA as an analytical lens. The MMD-CKA equivalence provides theoretical grounding: when we measure CKA between teacher and student, we are implicitly measuring a distributional distance, giving the similarity scores a concrete statistical interpretation beyond mere correlation.

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
| "Text Repr. Distillation via IB" 2023 | EMNLP | Maximize MI(teacher, student) while minimizing MI(student, input) |
| "IT Criteria for KD in Multimodal" 2025 | arXiv | KD effective when MI(teacher, student) > MI(student, labels) — formal condition |
| SGD-Based KD with Bayesian Teachers 2026 | arXiv | Bayesian posteriors → variance reduction; excess risk bounds via L2 teacher-posterior distance |

### Zhang et al. 2023 — IBKD (Text Representation Distillation via IB)
Applies the Information Bottleneck framework to representation distillation for pre-trained language models: maximize mutual information I(Z_s; Z_t) between student and teacher representations while minimizing I(Z_s; X) between student representation and raw input, forcing the student to retain only task-relevant information. Evaluated on Semantic Textual Similarity and Dense Retrieval benchmarks, IBKD outperforms standard MSE-based distillation. NLP-focused (distilling from large PLMs), not vision.

**Thesis connection:** The IB framework provides a theoretical vocabulary for what PCA/ICA decomposition reveals: principal components capturing high I(Z_s; Z_t) are the "distilled knowledge" dimensions, while components with high I(Z_s; X) but low I(Z_s; Z_t) represent noise the student failed to filter. This framing could give information-theoretic meaning to the spectral structure observed in CIFAR experiments.

### Xie et al. 2025 — IT Criteria for KD in Multimodal Learning
Proposes the Cross-modal Complementarity Hypothesis: cross-modal KD succeeds when I(Z_t; Z_s) > I(Z_s; Y), i.e., the teacher representation provides more information about the student's representation than the student's representation provides about the labels alone. Validated theoretically in a joint Gaussian model and empirically across image, text, video, audio, and cancer omics datasets. When the condition holds, distillation helps; when it doesn't, distillation can hurt.

**Thesis connection:** Though focused on cross-modal settings, the condition generalizes: KD should help when the teacher carries information the student cannot extract from labels alone. PCA/ICA on teacher-student pairs can test this by measuring whether the variance explained by shared principal components exceeds what the student learns independently — a geometric proxy for the mutual information condition.

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

### Park et al. 2019 — Relational Knowledge Distillation
Instead of matching individual output activations (point-wise KD), RKD transfers the relational structure between data examples: pairwise Euclidean distances (distance-wise loss) and triplet angles (angle-wise loss) computed over mini-batch representations. The intuition is that what matters is not the absolute position of each example in representation space but the geometric configuration of examples relative to each other. Tested on metric learning (Cars-196, CUB-200), classification, and few-shot learning, RKD students significantly outperform point-wise KD and sometimes surpass their own teachers — particularly in metric learning where relational structure is the entire objective.

**Thesis connection:** RKD's distance-wise and angle-wise losses are directly related to what CKA measures (kernel alignment over Gram matrices captures exactly these pairwise relational structures). We can use CKA/PCA to ask whether standard logit-based KD implicitly transfers relational structure (as RKD explicitly does), or whether the two mechanisms produce different representational geometries in CIFAR student networks.

### Saadi & Wang 2025 — Flex-KD
Takes a "functional" perspective: rather than matching raw hidden representations, it identifies which teacher neurons functionally matter by computing gradient-based importance scores (dF(x)/dh_T averaged over training examples). Only the top-d_S most important teacher dimensions (matching the student's hidden size) are selected for distillation, discarding weakly-contributing neurons. Architecture-agnostic and parameter-free. Evaluated on LLM distillation across GLUE, instruction-following, and summarization, Flex-KD outperforms projection-based methods by up to 3.75% ROUGE, with particular gains under severe teacher-student dimension mismatch.

**Thesis connection:** Flex-KD's gradient-based neuron selection is functionally a task-conditioned PCA: it identifies the dominant directions that most affect output. Our PCA/ICA performs a similar dimensionality reduction but in a post-hoc interpretive mode. Comparing Flex-KD's "functional dimensions" with PCA's principal components could reveal whether the directions that matter for distillation align with the directions that capture the most variance.

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
| **"All You Need Is a Tailored Coordinate System" (Zhou, Zhu & Wu) 2024** | AAAI 2025 | **Dark knowledge is encoded in the PCA coordinate system of teacher features.** Single forward pass captures this. Teacher-free after extraction. SOTA with half training time. |
| Chiu et al. 2022 | CVPR | PCA as dimension reduction in KD for style transfer |
| Neural Spectral Decomp. (Yang et al.) 2024 | ECCV | SVD-based decomposition for *dataset* distillation; information per dimension is low-rank |
| Meller & Berkouk 2023 (SVR) | AISTATS | SVD of successive linear maps → graph representation of networks via meaningful input/output directions |
| Guillaume Alain & Yoshua Bengio 2016 | ICLR-W 2017 | Linear classifier probes on frozen intermediate representations; foundational probing work |
| C2G-KD 2025 | arXiv | PCA constraints on generator for data-free KD |

### Zhou, Zhu & Wu 2024 — "All You Need Is a Tailored Coordinate System" (TCS)
Introduces TCS, which extracts dark knowledge from SSL-pretrained teachers by capturing the linear subspace in which teacher features reside, rather than matching feature values directly. The key insight is that the coordinate system (the basis of the feature subspace) itself encodes transferable structure, and the student can be aligned to this subspace with only a single forward pass through the teacher — no task-specific teacher fine-tuning needed. Achieves higher accuracy than prior KD methods at roughly half the training time and GPU memory, supports cross-architecture distillation, and handles large teacher-student capacity gaps. Accepted at AAAI 2025.

**Thesis connection:** This paper directly validates the idea that linear subspace structure (the kind PCA recovers) is a meaningful lens for understanding what transfers. But TCS uses this for a *method* — we use it for *systematic interpretation*. Applying PCA/ICA to teacher and student feature spaces on CIFAR CNN pairs could reveal whether the student converges toward the teacher's principal subspace, and CKA could quantify that alignment layer by layer.

### Meller & Berkouk 2023 — Singular Value Representation (SVR)
Applies SVD factorization to each layer's weight matrix to construct a weighted graph of "spectral neurons" — nodes corresponding to singular vectors that capture distinct activation patterns. They develop a statistical framework for discriminating meaningful connections between spectral neurons in both fully connected and convolutional layers. Two empirical findings: VGG networks exhibit a dominant spectral connection spanning multiple deep layers, and batch normalization induces significant connections between near-kernel components, producing spontaneous sparsification that emerges without any input data. Published at AISTATS 2023.

**Thesis connection:** SVR provides a concrete precedent for using SVD/spectral decomposition to characterize internal network structure, which is exactly the analytical frame we apply via PCA singular spectra. Comparing teacher and student SVR graphs (or simply their singular value distributions) on CIFAR CNN pairs could reveal whether distillation compresses the spectral structure or preserves the dominant singular directions.

### Yang et al. 2024 — Neural Spectral Decomposition for Dataset Distillation
Treats an entire dataset as a high-dimensional, low-rank observation and learns spectrum tensors plus transformation matrices that reconstruct the data distribution through matrix operations. Achieves state-of-the-art dataset distillation on CIFAR-10, CIFAR-100, Tiny ImageNet, and ImageNet subsets. Though this is *dataset* distillation (compressing datasets, not models), its core assumption — that the data distribution is low-rank and decomposable via spectral methods — directly parallels our premise.

**Thesis connection:** Provides evidence that spectral decomposition is a natural analytical vocabulary for distillation phenomena more broadly. The low-rank assumption underlying their method is the same assumption underlying our use of PCA to analyze feature spaces.

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

### Adhane et al. 2025 — UniCAM
Introduces a gradient-based visual explanation method that separates "distilled features" (task-relevant attributes like textures and object parts the student learns from the teacher) from "residual features" (irrelevant information like backgrounds). Proposes Feature Similarity Score (FSS) and Relevance Score (RS) as quantitative metrics. Experiments on CIFAR-10, ASIRRA, and Plant Disease datasets demonstrate that UniCAM can visually and quantitatively characterize what transfers during KD.

**Thesis connection:** UniCAM localizes transferred knowledge *spatially* (pixel regions); our PCA/ICA/CKA localizes it *spectrally* (which principal/independent components carry the distilled information). The two views are complementary — combining both on CIFAR CNN pairs would give a richer picture of what the student actually acquires.

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

### Phuong & Lampert 2019 — "Towards Understanding Knowledge Distillation"
The first rigorous generalization bound for knowledge distillation, analyzing linear and deep linear classifiers trained on soft teacher outputs. They prove that the expected risk of a distillation-trained linear classifier converges rapidly, and identify three factors governing convergence: the geometric properties of the data distribution (specifically class separation), an optimization bias whereby gradient descent finds favorable minima of the distillation objective, and a strong monotonicity property ensuring expected risk decreases as training data grows. The analysis explains why training on soft labels can outperform training on hard ground-truth labels.

**Thesis connection:** Their emphasis on data geometry — class separation in particular — directly motivates using PCA to measure how distillation reshapes class separability in the student's feature space. Their linear-classifier theory also justifies analyzing the linear subspace structure (via PCA/ICA) as a first-order approximation of what drives distillation success on CIFAR CNN pairs.

**What to dig into:**
- `[DEEP DIVE]` Phuong & Lampert — "data geometry" factor directly connects to our eigenvalue analysis; spectral structure of covariance determines when KD succeeds
- `[SKIM]` Liu review — useful for related work section

**Connection to us:** Their data geometry factor *is* the spectral structure we analyze. Linear-network setting = where our decomposition is most theoretically grounded. Potential formal backbone for our empirical findings.

---

## 8. Label Smoothing / Regularization Interpretation

Is KD just fancy regularization?

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Yuan, Tay, Li, Wang & Feng 2020 | CVPR (oral) | KD ≈ *learned* label smoothing. Even poorly-trained teachers improve students. |
| Zhou, Song, Chen, Zhou, Wang, Yuan & Zhang 2021 | ICLR | Soft labels = supervision + regularization. Bias-variance tradeoff varies sample-wise. |
| Zhang & Sabuncu 2020 | NeurIPS | Self-distillation ≈ instance-specific label smoothing |
| "KD ≈ Label Smoothing: Fact or Fallacy?" | OpenReview | Challenges the equivalence |

### Yuan, Tay, Li, Wang & Feng 2020 — "Revisiting KD via Label Smoothing Regularization"
Argues that KD succeeds primarily as a form of learned label smoothing regularization, not because the teacher transmits category-similarity information. Supported by two striking experiments: reversed KD (a weaker student improving a stronger teacher) and poorly-trained teachers still benefiting students. Based on this, they propose Teacher-free KD (Tf-KD), where the student distills from its own predictions or a fixed uniform-smoothing distribution, achieving performance comparable to conventional KD — including 0.65% improvement on ImageNet. CVPR 2020 oral.

**Thesis connection:** If KD's benefit is substantially regularization rather than structural knowledge transfer, then PCA/ICA analysis should show the student does *not* closely replicate the teacher's spectral structure but instead develops a smoother, lower-variance representation. This paper sets up a testable null hypothesis: spectral alignment (measured by CKA) between teacher and student may be weaker than expected if regularization dominates.

### Zhou, Song et al. 2021 — "Rethinking Soft Labels: Bias-Variance Tradeoff"
Decomposes soft labels' effect through a bias-variance lens and discovers the tradeoff operates at the *individual sample level*, not uniformly across the dataset. They identify "regularization samples" — instances where soft labels increase bias but reduce variance — and show that removing these entirely degrades performance, meaning the variance reduction they provide is essential. Proposes weighted soft labels that adaptively balance the per-sample tradeoff, outperforming fixed-temperature distillation.

**Thesis connection:** The sample-wise bias-variance decomposition suggests that PCA/ICA analysis should be conditioned on sample difficulty or class membership, not just computed globally. For CIFAR CNN pairs, one could stratify the spectral analysis by these "regularization samples" to see whether distillation reshapes the feature space differently for high-variance vs. low-variance instances.

### Zhang & Sabuncu 2020 — "Self-Distillation as Instance-Specific Label Smoothing"
Shows that multi-generational self-distillation (a model distilling into itself repeatedly) improves generalization, explained by framing teacher-student training as amortized MAP estimation where the teacher's predictions provide instance-specific regularization. They formally connect self-distillation to label smoothing but emphasize that self-distillation goes further by producing instance-specific smoothing driven by predictive diversity. Their proposed method often outperforms classical uniform label smoothing across multiple architectures.

**Thesis connection:** The instance-specific regularization view predicts that self-distillation should progressively diversify the feature representation across generations, which PCA analysis could detect as increasing effective dimensionality or spreading of the singular spectrum. Testing whether CKA between successive self-distillation generations on CIFAR CNNs decreases (diversification) or plateaus would directly probe this mechanism.

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

### Stanton et al. 2021 — "Does Knowledge Distillation Really Work?"
Investigates whether students actually learn to match teacher predictive distributions. They find a "surprisingly large discrepancy" that persists even when the student has sufficient capacity to represent the teacher function exactly. Students improve in generalization accuracy but fail to faithfully approximate the teacher's full output distribution. They identify optimization difficulty — not model capacity — as the primary bottleneck. Counterintuitively, closer function-space matching does not necessarily yield better student generalization, suggesting KD works through a mechanism other than simple imitation.

**Thesis connection:** If students provably fail to match teachers in function space, then asking *what does align* in representation space becomes the central question. PCA/ICA decomposition of teacher-student activations on CIFAR CNNs can precisely characterize the partial, selective alignment that actually occurs — and identify which representational components transfer versus which are lost.

### Mirzadeh et al. 2020 — "Improved KD via Teacher Assistant"
Demonstrates that student performance degrades when the capacity gap between teacher and student becomes too large — a teacher can only effectively transfer knowledge to students above a certain size threshold. They propose inserting an intermediate-sized "teacher assistant" network that bridges the gap via multi-step distillation. Experiments on CIFAR-10, CIFAR-100, and ImageNet using CNN and ResNet architectures show consistent improvement. The work establishes that the teacher-student capacity ratio is a first-order variable governing distillation success.

**Thesis connection:** The capacity gap implies that what transfers depends on the spectral complexity the student can absorb. PCA/ICA decomposition on CIFAR CNN pairs of varying capacity ratios could reveal exactly which principal components survive the gap — whether students capture only the top-k eigendirections, and whether the teacher assistant restores access to mid-ranked components.

**What to dig into:**
- `[DEEP DIVE — CRITICAL]` Stanton et al. — if function-space matching fails, then our representation-space decomposition may capture the *actual mechanism* by which KD helps (partial alignment, not full imitation)
- `[SKIM]` Teacher assistant — useful context for capacity mismatch

**Connection to us:** If students don't match teachers in function space, what *does* align in representation space? Our decomposition can precisely characterize the partial alignment. This could be a key framing for our thesis.

---

## 10. Emerging: Topology

| Paper | Venue | Key claim |
|-------|-------|-----------|
| "Topological Persistence Guided KD for Wearable Sensor Data" (Jeon et al.) 2024 | Eng. Appl. of AI (ScienceDirect) | Persistent homology captures topological structure (connected components, loops, voids) |

### Jeon et al. 2024 — Topological Persistence Guided KD
A dual-teacher KD framework for human activity recognition from wearable sensor data. One teacher is trained on raw time-series signals while the second is trained on persistence images derived from topological data analysis (TDA), capturing topological features like connected components and loops. The student distills from both but requires only raw data at inference. They introduce orthogonality constraints on feature correlation maps and an annealing schedule to absorb complementary information from both modalities.

**Thesis connection:** Demonstrates that structurally distinct representations (metric vs. topological) carry complementary knowledge for distillation. PCA/ICA captures metric/variance structure while topology captures qualitative shape — acknowledging this boundary clarifies what our decomposition can and cannot explain, and positions topological analysis as a natural complement in future work.

**What to dig into:**
- `[SKIM]` — complementary to our approach (they capture qualitative structure, we capture metric structure). Worth mentioning in future work.

---

## 11. Nonlinear ICA Theory (Foundation)

| Paper | Venue | Key claim |
|-------|-------|-----------|
| Hyvarinen et al. 2023 | Patterns | Linear ICA identifiable; nonlinear ICA needs auxiliary info. With conditioning (e.g., class labels), nonlinear ICA becomes identifiable. |

### Hyvarinen, Khemakhem & Morioka 2023 — "Nonlinear ICA for Principled Disentanglement"
A comprehensive review of nonlinear ICA theory, addressing the fundamental problem that nonlinear ICA is not identifiable in general — unlike linear ICA, where the mixing can be uniquely recovered. The key theoretical advance: nonlinear ICA *becomes* identifiable when auxiliary information (temporal structure, class labels, domain indices) is available. They show that several self-supervised learning algorithms (e.g., contrastive learning) can be formally reinterpreted as estimating nonlinear ICA models. Covers both identifiability proofs under various conditioning assumptions and practical algorithms.

**Thesis connection:** Our thesis applies linear ICA to intermediate CNN activations, which are themselves nonlinear functions of the input. Hyvarinen's identifiability results justify this: linear ICA on features from a fixed nonlinear encoder is well-posed, and conditioning on class labels (available in CIFAR) strengthens identifiability. This provides theoretical backing for interpreting ICA components of teacher-student activations as genuinely distinct knowledge factors rather than arbitrary rotations.

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
10. Meller & Berkouk 2023 (SVR) — SVD across layers
11. Cheng et al. 2020 — concept quantification
12. Hyvarinen et al. 2023 — ICA identifiability

---

## Gaps Our Thesis Fills (Expanded)

**1. Interpretation, not method.** The overwhelming majority of work using PCA, ICA, or CKA in the knowledge distillation literature deploys these tools to *build better distillation objectives* — TCS (2412.09388) uses PCA coordinate systems for a more efficient KD method, Saha et al. (2022) use CKA directly as a loss, Zhou et al.'s RCKA (2024) decomposes CKA primarily to argue for a better loss function. No existing work uses spectral and independence-based tools systematically as an *explanatory* framework to understand what knowledge transfers and why. Our thesis would be the first dedicated study where the decomposition is the end product — generating hypotheses about distillation mechanisms rather than engineering better performance numbers.

**2. Head-to-head decomposition comparison.** PCA maximizes variance explained, ICA maximizes statistical independence, and CKA measures representational similarity — these are fundamentally different lenses on the same activation data, yet no one has compared what each reveals about the same teacher-student pair. The closest work is SVCCA (Raghu et al., 2017), which chains SVD into CCA, but this is a single pipeline rather than a comparative study. Morcos et al. (2018) use PWCCA to study generalization but never contrast it against ICA or raw PCA. Our thesis systematically applies all three to the same CIFAR CNN teacher-student activations, revealing whether variance-dominant components (PCA), statistically independent components (ICA), or kernel-alignment structure (CKA) best explain which knowledge transfers.

**3. Spectral characterization of dark knowledge.** Tang et al. (2020) decompose teacher knowledge into universe-level, domain-level, and instance-specific categories, but this is a conceptual taxonomy, not a data-driven spectral one. TCS shows PCA subspaces encode dark knowledge but does not characterize *which* eigenvectors carry inter-class similarity versus instance-specific information versus regularization noise. Cheng et al. (2020) quantify visual concepts but don't connect these to spectral rank or eigenvalue structure. Our study would assign semantic roles to specific eigenvectors and ICA components — mapping top-k principal components to inter-class relational structure, mid-range to fine-grained instance features, and tail to noise, grounded in empirical measurement.

**4. Reconciling the Stanton puzzle.** Stanton et al. (2021) establish that students fail to match teachers in function space even with sufficient capacity, but do not investigate *what does* align at the representation level — their analysis remains at the output distribution level (agreement rates, calibration). Mirzadeh et al. (2020) show that capacity gaps modulate this failure but likewise don't decompose internal representations. PCA/ICA analysis of teacher-student activation pairs on CIFAR CNNs would directly address this: measuring which principal and independent components align versus diverge, characterizing the "partial alignment" that actually drives student improvement even when full function-space matching fails.

**5. Separating regularization from knowledge.** Yuan et al. (2020) show KD approximates learned label smoothing, and Zhou et al. (2021) decompose soft labels into supervision and regularization terms, but both analyses operate at the output level and cannot distinguish these effects within internal representations. The regularization interpretation implies that some of what a student learns from a teacher is a uniform smoothing signal rather than structured class-relationship knowledge. ICA, which finds statistically independent sources, is uniquely suited to separate a uniform/diffuse regularization component from structured, class-dependent components in teacher activations. Our study would apply ICA to teacher and student activations and test whether one or more independent components correspond to a class-independent smoothing effect — the first internal-representation evidence for or against the "KD as regularization" hypothesis.

**6. Layer-wise decomposition dynamics.** Existing representation similarity work (Kornblith et al., 2019; Morcos et al., 2018) typically reports CKA or CCA as a single similarity score per layer pair, producing block-diagonal heatmaps that show *where* alignment occurs but not *what* aligns. Meller et al.'s SVR (2023) decomposes individual layers via SVD but focuses on characterizing single networks rather than teacher-student correspondence. No existing work tracks how the *spectral composition* of teacher-student alignment evolves across depth — whether early layers share low-frequency/high-variance components while deeper layers align on class-discriminative independent components. Our thesis computes layer-wise PCA eigenspectra and ICA component overlap for CIFAR CNN pairs, producing a depth-resolved map of which types of representational structure transfer at each stage of the hierarchy.
