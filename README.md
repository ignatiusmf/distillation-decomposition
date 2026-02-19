================================================================================
DISTILLATION-DECOMPOSITION REPOSITORY EXPLORATION
================================================================================

Repository Location: /home/ignatius/Lab/studies/repos/distillation-decomposition/
Git Status: main branch (clean, no uncommitted changes)

================================================================================
1. PROJECT OVERVIEW
================================================================================

This is a thesis/research project investigating the structural properties of
neural network representations that enable knowledge distillation (KD). The work
decomposes KD mechanisms using linear latent analysis (PCA, ICA, CKA, principal
angles, Fisher criterion) to understand what is actually transferred between
teacher and student models.

Key Research Question:
"What structural properties of neural network representations enable effective
knowledge distillation?"

Focus: Representation-level lens and linear projection-based explanatory probes
to understand multiple KD strategies (logit distillation vs. factor transfer).

================================================================================
2. TOP-LEVEL PROJECT STRUCTURE
================================================================================

analysis/
  ├── extract.py              - Extracts GAP-pooled layer representations from trained models
  ├── analyze.py              - 7-part analysis: PCA, effective dim, CKA, principal angles, ICA, class separability, PCA scatter
  ├── representations/        - Contains .npz files with saved layer representations
  │   ├── teacher_ResNet112_seed0.npz
  │   ├── student_ResNet56_logit_seed0.npz
  │   └── student_ResNet56_factor_seed0.npz
  └── figures/                - Generated analysis plots (PNG + markdown summary)

experiments/
  ├── Cifar10/
  │   ├── pure/               - Baseline models (no distillation)
  │   ├── logit/              - Logit KD experiments
  │   └── factor_transfer/    - Factor Transfer experiments
  ├── Cifar100/
  │   ├── pure/
  │   ├── logit/
  │   └── factor_transfer/
  └── [Each experiment has seed variations 0, 1, 2]

data/
  ├── cifar-10-batches-py/    - CIFAR-10 dataset
  ├── cifar-10-python.tar.gz  - Compressed CIFAR-10 (163M)
  ├── cifar-100-python/       - CIFAR-100 dataset
  └── cifar-100-python.tar.gz - Compressed CIFAR-100 (162M)

toolbox/
  ├── models.py               - ResNet architectures (ResNet112, ResNet56, ResNet20, ResNetBaby)
  ├── data_loader.py          - Data loading for Cifar10, Cifar100, TinyImageNet
  ├── distillation.py         - 3 distillation methods: NoDistillation, LogitDistillation, FactorTransfer
  └── utils.py                - Training utilities and evaluation

train.py                       - Main training script (150 epochs, SGD, CosineAnnealing)
runner.py                      - Experiment launcher for PBS job queue
pyproject.toml                 - Python project metadata
nuwe_structure.md              - Thesis structure with word/reference/figure budgets (30k words target)
notes                          - Development notes and rsync commands

================================================================================
3. ANALYSIS/ - COMPLETE CONTENT OVERVIEW
================================================================================

3.1 extract.py
--------------
Purpose: Extract GAP-pooled representations from trained models on test set

Key Functions:
  extract(model, dataloader, device) -> dict
    - Runs model on dataset
    - Extracts 3 intermediate layers via Global Average Pooling: layer1, layer2, layer3
    - Returns logits and labels for each test sample
    - Output shape: [num_test_samples, num_channels] per layer

Main Flow:
  1. Load trained model weights from experiments/[dataset]/[method]/[seed]/best.pth
  2. For each of 3 models: teacher_ResNet112, student_ResNet56_logit, student_ResNet56_factor
  3. Process entire test set (no gradients)
  4. Save to .npz files: analysis/representations/[model_name]_seed[seed].npz
  5. Each .npz contains: layer1, layer2, layer3, logits, labels

Currently configured for: Cifar100 (100 classes), seed=0
Can be run with different seeds/datasets via --seed, --device flags


3.2 analyze.py - 7-PART ANALYSIS PIPELINE
-------------------------------------------
Purpose: Comprehensive representation analysis producing thesis-quality figures

Configuration Constants:
  - MODEL_KEYS: [teacher_ResNet112, student_ResNet56_logit, student_ResNet56_factor]
  - LAYERS: [layer1, layer2, layer3] with channel counts [16, 32, 64]
  - DISPLAY COLORS: teacher=blue, logit_kd=orange, factor_transfer=green
  - Figure DPI: 150 (display), 300 (save)

ANALYSIS 1: PCA Cumulative Variance (pca_variance.png)
  What: Plots cumulative variance explained across principal components
  For: Each of 3 layers, comparing 3 models
  Output: 1x3 subplot showing variance curves up to 95%+
  Calculation: Eigendecomposition of covariance matrix, sorted eigenvalues
  Key Metric: effective_dim(cum_var, threshold=0.90/0.95) = min components to reach threshold

ANALYSIS 2: Effective Dimensionality (effective_dim.png)
  What: Bar chart showing # PCA components needed for 90% and 95% variance
  For: Each layer, all 3 models side-by-side
  Pattern: Lower eff_dim = more compressed representation
  Teaching Insight: Teacher concentrates variance more tightly than students

ANALYSIS 3: CKA - Representation Similarity (cka_cross_layer.png & cka_same_layer.png)
  What: Linear CKA (Kornblith et al., 2019) measures subspace alignment
  Formula: CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
  
  Part A - Cross-layer heatmap:
    - Teacher self-similarity (diagonal strong, off-diagonal weak)
    - Teacher vs Logit KD student
    - Teacher vs Factor Transfer student
    - Shows if student layer 2 matches teacher layer 3 (layer shifting) - NONE OBSERVED
  
  Part B - Same-layer bar chart:
    - Direct comparison of corresponding layers
    - Pattern: CKA drops layer1(~0.97) → layer2(~0.92) → layer3(~0.71)
    - Insight: Early representations converge regardless of method/depth; late diverge

ANALYSIS 4: Principal Angles (principal_angles.png)
  What: Subspace alignment via principal angles between PCA bases
  For: Each layer, comparing student vs teacher subspaces
  Method:
    1. Get PCA eigenvectors covering 95% variance for each model
    2. Take orthonormal basis (QR decomposition)
    3. Compute SVD of U^T V (cross-product of bases)
    4. arccos(singular_values) = principal angles in degrees
  Output: Line plot per layer showing angle spectrum (0-90 degrees)
  Insight: Top few directions align well (small angles), then diverge

ANALYSIS 5: ICA Component Matching (ica_correlation.png & ica_summary.png)
  What: Independent Component Analysis reveals non-Gaussian structure
  For: Teacher and each student, extract n_components = n_channels per layer
  Method:
    1. FastICA with 2000 iterations on normalized data
    2. Cross-correlate extracted components (optimal matching via Hungarian algorithm)
    3. Reorder to maximize diagonal correlation
    4. Visualize |correlation| heatmap (darker = more matched)
  
  Summary metrics:
    - Mean matched |correlation|: avg strength of best-matched components
    - Strongly matched (|corr| > 0.5): count of components with strong correspondence
  
  Pattern: Layer 1 weak matching (0.24-0.27 mean) → Layer 2 moderate → Layer 3 best
  Divergence: Logit KD preserves more ICA structure at layer 3 than factor transfer
  Key Insight: ICA reveals patterns CKA/PCA miss; shows structure independence

ANALYSIS 6: Class Separability - Fisher Criterion (class_separability.png)
  What: Between-class vs within-class variance ratio (discriminative power)
  Formula: Fisher(X,labels) = Tr(S_between) / Tr(S_within)
  Calculation:
    1. Mean per class, overall mean
    2. S_b = sum over classes: n_c * ||m_c - m||^2
    3. S_w = sum over classes: sum samples: ||x - m_c||^2
    4. Return S_b / S_w
  
  Pattern: Students higher than teacher at layer 3
    - Logit KD: 1.18, Factor Transfer: 1.06, Teacher: 0.92
  Insight: Smaller student forced to be MORE discriminative to compensate for parameters
  Implication: Distillation doesn't copy; creates more separable representation

ANALYSIS 7: PCA Scatter - 2D Projection (pca_scatter.png)
  What: Visualize 10 evenly-spaced classes (0, 10, 20, ..., 90) in 2D PC space
  For: Layer 3 only, using teacher's PCA basis as common coordinate system
  Method:
    1. Compute teacher PCA: get eigenvectors and mean
    2. Project all 3 models' layer3 features: (X - teacher_mean) @ PC[:, :2]
    3. Plot with class color coding
  
  Pattern: All 3 models show similar class cluster geometry
  But: Student clusters tighter/more separated (confirms Fisher criterion finding)
  Labels: Legend showing 10 classes with scaled markers


3.3 Summary Table Output
------------------------
For each layer, prints:
  - Effective dimensionality (90% and 95% variance)
  - CKA (Teacher vs each student)
  - Fisher criterion (class separability)

This is printed to stdout during analysis execution.


3.4 Generated Figures
---------------------
All saved to analysis/figures/ as high-res PNGs (300 DPI):
  1. pca_variance.png         - 3 variance curves (1 per layer)
  2. effective_dim.png        - 2 bar charts (90%, 95% thresholds)
  3. cka_cross_layer.png      - 3 heatmaps (self, vs logit, vs factor)
  4. cka_same_layer.png       - Bar chart of same-layer CKA
  5. principal_angles.png     - 3 line plots (1 per layer)
  6. ica_correlation.png      - 2x3 heatmaps (2 students × 3 layers)
  7. ica_summary.png          - 2 bar charts (mean corr, strong count)
  8. class_separability.png   - Single bar chart
  9. pca_scatter.png          - 3 scatter plots (1 per model)


3.5 claude_summary.md
---------------------
Interpretive summary of findings:

PCA Variance & Effective Dimensionality:
  - Teacher concentrates variance tighter (3 components @ 90% layer1 vs 4 for students)
  - Gap widens at layer2 (11 vs 14), converges at layer3 (~42)
  - Shallower students can't fully replicate deep teacher's compression

CKA Drops Monotonically:
  - Layer1: 0.97-0.98 (near identical early features)
  - Layer2: 0.89-0.92 (diverging)
  - Layer3: 0.71 (significantly different abstract reps)
  - No layer-shifting evidence
  - Logit KD slightly higher at 1-2; Factor Transfer slightly better at 3

Principal Angles:
  - Both students show identical subspace alignment
  - Top PCs well-aligned (small angles), then diverge steadily
  - No meaningful difference between methods

ICA Independent Components:
  - Layer1: Weak matching (0.24-0.27 mean correlation)
  - Layer3: Logit KD preserves 43/64 strong components, Factor Transfer 40/64
  - ICA reveals structure that PCA/CKA miss
  - Different kinds of alignment

Class Separability:
  - Students MORE separable than teacher at layer3
  - Smaller model forced to be more discriminative
  - Distillation creates compressed, class-optimized representation

PCA Scatter:
  - All models produce similar cluster geometry
  - Students' clusters tighter and more separated


================================================================================
4. EXPERIMENTS/ - COMPLETE DIRECTORY STRUCTURE & RESULTS
================================================================================

4.1 Experiment Directory Structure
-----------------------------------

experiments/
├── Cifar10/
│   ├── pure/                          [BASELINE - NO DISTILLATION]
│   │   ├── ResNet112/
│   │   │   ├── 0/                    [seed=0]
│   │   │   ├── 1/                    [seed=1]
│   │   │   └── 2/                    [seed=2]
│   │   └── ResNet56/
│   │       ├── 0/
│   │       ├── 1/
│   │       └── 2/
│   ├── logit/                         [LOGIT DISTILLATION]
│   │   └── ResNet112_to_ResNet56/
│   │       ├── 0/
│   │       ├── 1/
│   │       └── 2/
│   └── factor_transfer/               [FACTOR TRANSFER]
│       └── ResNet112_to_ResNet56/
│           ├── 0/
│           ├── 1/
│           └── 2/
│
├── Cifar100/                          [SAME STRUCTURE AS ABOVE]
│   ├── pure/
│   │   ├── ResNet112/ {0,1,2}/
│   │   └── ResNet56/ {0,1,2}/
│   ├── logit/
│   │   └── ResNet112_to_ResNet56/ {0,1,2}/
│   └── factor_transfer/
│       └── ResNet112_to_ResNet56/ {0,1,2}/


4.2 Per-Experiment Files
------------------------

Each seed experiment directory contains:

  best.pth              - Model weights (best test accuracy)
                          Format: {'weights': state_dict}
                          Used for: inference and as teacher weights for distillation

  checkpoint.pth        - Full training checkpoint (can resume from)
                          Contains: model_state, optimizer_state, scheduler_state,
                                   distillation_state_dict (if applicable),
                                   all loss/acc curves, max_acc, config

  metrics.json          - Training curves (150 epochs)
                          Keys: train_loss, train_acc, test_loss, test_acc (lists of 150 floats)
                          Plus: config (all hyperparameters used)

  status.json           - Quick status check
                          Keys: status ('completed'|'in_progress'), epoch (int), 
                                max_acc (float), config (dict)

  Accuracy.png          - Plot of train/test accuracy curves (matplotlib)
  Loss.png              - Plot of train/test loss curves (matplotlib)

  logs                  - Directory (if created during training)
  errors                - Directory (if created during training)


4.3 CIFAR-10 RESULTS SUMMARY
----------------------------

Dataset: 50,000 training, 10,000 test images (32×32 RGB, 10 classes)

PURE BASELINE (no distillation):
  Model         Seed  Max Test Acc  Final Test Acc  Status
  ResNet112      0     92.54%        92.54%         COMPLETED
  ResNet112      1     94.21%        94.21%         COMPLETED
  ResNet112      2     93.89%        93.89%         COMPLETED
                 AVG   93.55% ± 0.78%
  
  ResNet56       0     93.11%        93.11%         COMPLETED
  ResNet56       1     93.29%        93.29%         COMPLETED
  ResNet56       2     93.19%        93.19%         COMPLETED
                 AVG   93.20% ± 0.09%

Observation: ResNet56 nearly matches ResNet112 on CIFAR-10 despite fewer parameters!
This is the student-baseline comparison.

LOGIT DISTILLATION (Hinton et al., 2015):
  ResNet112 → ResNet56, T=4.0, α=0.5
  
  Seed  Max Acc   Final Acc   vs Pure ResNet56   Status
   0     92.35%    92.35%     -0.76%            COMPLETED
   1     93.97%    93.97%     +0.68%            COMPLETED
   2     93.80%    93.80%     +0.61%            COMPLETED
               AVG 93.37% ± 0.82%

Interpretation: Logit KD provides modest benefit (avg +0.18% on seed average)
with high variance. Not always helpful on CIFAR-10.

FACTOR TRANSFER (Kim et al., 2018):
  ResNet112 → ResNet56, α=0.5, factor_dim=64
  Paraphraser + Translator on all 3 layers [0,1,2]
  
  Seed  Max Acc   Final Acc   vs Pure ResNet56   Status
   0     93.74%    93.74%     +0.63%            COMPLETED
   1     93.57%    93.57%     +0.28%            COMPLETED
   2     93.89%    93.89%     +0.70%            COMPLETED
               AVG 93.73% ± 0.16%

Interpretation: Factor Transfer slightly better than Logit KD on CIFAR-10
(+0.53% avg vs +0.18%), with lower variance.


4.4 CIFAR-100 RESULTS SUMMARY
-----------------------------

Dataset: 50,000 training, 10,000 test images (32×32 RGB, 100 classes)
Much harder task than CIFAR-10 (10× classes).

PURE BASELINE:
  Model         Seed  Max Test Acc  Status
  ResNet112      0     71.59%        COMPLETED
  ResNet112      1     72.88%        COMPLETED
  ResNet112      2     72.43%        COMPLETED
                 AVG   72.30% ± 0.63%
  
  ResNet56       0     70.43%        COMPLETED
  ResNet56       1     71.54%        COMPLETED
  ResNet56       2     71.54%        COMPLETED
                 AVG   71.17% ± 0.61%

Observation: On CIFAR-100, gap widens (~1.1%) vs CIFAR-10 (0.35%)
Teacher advantage more pronounced with 100 classes.

LOGIT DISTILLATION:
  ResNet112 → ResNet56, T=4.0, α=0.5
  
  Seed  Max Acc   vs Pure ResNet56   Status
   0     69.39%    -1.04%            COMPLETED
   1     72.96%    +1.42%            COMPLETED
   2     73.16%    +1.62%            COMPLETED
               AVG 71.84% ± 2.20%

Interpretation: HIGHLY VARIABLE. Logit KD can help (seed 1,2) or hurt (seed 0)
Large variance suggests instability or seed-dependent effects.

FACTOR TRANSFER:
  ResNet112 → ResNet56, α=0.5, factor_dim=64
  
  Seed  Max Acc   vs Pure ResNet56   Status
   0     71.20%    +0.77%            COMPLETED
   1     70.64%    -0.90%            COMPLETED
   2     72.13%    +0.59%            COMPLETED
               AVG 71.32% ± 0.75%

Interpretation: Factor Transfer more stable than Logit KD (std ±0.75 vs ±2.20)
But modest improvements overall (~+0.15% on average vs Pure).


4.5 Key Findings from Experiments
----------------------------------

1. CIFAR-10 vs CIFAR-100:
   - KD methods more effective on harder datasets (CIFAR-100)
   - On CIFAR-10, student nearly matches teacher without distillation
   - Variance increases with task difficulty

2. Logit KD vs Factor Transfer:
   - Factor Transfer more stable (lower seed variance)
   - Factor Transfer slightly better on CIFAR-10
   - Logit KD more variable on CIFAR-100 (can help or hurt)

3. Practical Impact:
   - ResNet56 without distillation: 93.20% (CIFAR-10), 71.17% (CIFAR-100)
   - With best distillation: 93.73% (CIFAR-10, Factor), 71.84% (CIFAR-100, Logit)
   - Gains: +0.53% (CIFAR-10), +0.67% (CIFAR-100)

4. Reproducibility:
   - 3 seeds per configuration (0, 1, 2)
   - All 24 experiments completed
   - Seed variance present but generally consistent


4.6 Architecture Details
------------------------

ResNet112:  [18, 18, 18] basic blocks per layer
ResNet56:   [9, 9, 9] basic blocks per layer (2× fewer)
ResNet20:   [3, 3, 3] basic blocks per layer (not used in these experiments)
ResNetBaby: [1, 1, 1] basic blocks per layer (not used)

All variants:
  - Input: 32×32 RGB images
  - Layer1: 16 channels
  - Layer2: 32 channels (stride=2)
  - Layer3: 64 channels (stride=2)
  - GAP: Global average pooling to (1,1,1)
  - Output: num_classes (10 or 100) logits
  - Returns: [layer1_feat, layer2_feat, layer3_feat, logits] during forward pass

This output format enables both distillation and representation extraction.


4.7 Training Configuration
--------------------------

All experiments (train.py):
  - Epochs: 150
  - Batch size: 128
  - Optimizer: SGD with momentum=0.9, weight_decay=5e-4
  - Scheduler: Cosine Annealing (T_max=150)
  - Learning rate: starts at 0.1, annealed via cosine schedule
  - Loss: Cross-entropy with label_smoothing=0.1 (baseline)
  - Distillation loss: Weighted combination (1-α)*CE + α*KD_loss

Distillation hyperparameters:
  - α (weight): 0.5 (equal balance between CE and KD)
  - Temperature: 4.0 (for logit and factor transfer)
  - Factor dimension: 64


================================================================================
5. TOOLBOX/ - IMPLEMENTATION DETAILS
================================================================================

5.1 distillation.py - 3 Methods
------------------------------

Base Class: DistillationMethod
  - Abstract methods: extra_loss(), get_trainable_modules(), to()
  - Registry: DISTILLATION_METHODS = {'none', 'logit', 'factor_transfer'}

Method 1: NoDistillation (pure training)
  - Returns loss = 0
  - For baseline experiments

Method 2: LogitDistillation (Hinton et al., 2015)
  - Softens teacher logits with temperature: softmax(logits/T)
  - KL divergence loss between soft teacher and soft student
  - Scaled by T² as per original paper
  - No extra trainable modules

Method 3: FactorTransfer (Kim et al., 2018)
  - Paraphraser (1×1 Conv path): teacher features → factor space
  - Translator (1×1 Conv path): student features → factor space
  - Architecture per module:
    Conv2d(in_ch, 64, k=1) → BatchNorm2d(64) → ReLU → Conv2d(64, 64, k=1)
  - L2 normalize factors
  - MSE loss between normalized factors
  - Applied to all 3 layers [0, 1, 2]
  - Handles spatial mismatch via adaptive_avg_pool2d
  - Trainable modules: paraphrasers (nn.ModuleList), translators (nn.ModuleList)


5.2 models.py - ResNet Architectures
------------------------------------

BasicBlock: Simple residual block
  - Conv3×3 (stride-dependent)
  - BatchNorm2d
  - Conv3×3 (stride=1)
  - BatchNorm2d
  - Shortcut path (1×1 conv if stride≠1 or channel change)
  - ReLU after addition

ResNet_simple: Base class
  - Input conv: 3ch → 16ch (k=3, p=1, no stride)
  - Layer1: 16ch, num_blocks[0] blocks
  - Layer2: 32ch, num_blocks[1] blocks, stride=2 (spatial reduction)
  - Layer3: 64ch, num_blocks[2] blocks, stride=2
  - Global Average Pool
  - Linear: 64*expansion → num_classes
  - Forward returns: [layer1_feat, layer2_feat, layer3_feat, logits]

Variants (controlled by num_blocks parameter):
  - ResNet112: [18, 18, 18] = 54 blocks total
  - ResNet56:  [9, 9, 9] = 27 blocks total
  - ResNet20:  [3, 3, 3] = 9 blocks total
  - ResNetBaby: [1, 1, 1] = 3 blocks total


================================================================================
6. TRAINING PIPELINE (train.py)
================================================================================

Execution Flow:
  1. Parse arguments: model, dataset, distillation method, seed
  2. Set up directory: experiments/[dataset]/[method]/[model_to_model]/[seed]/
  3. Load or initialize status.json
  4. If training already completed → skip (unless --force_restart)
  5. Set seeds (torch, cuda, numpy, random) for reproducibility
  6. Load data (trainloader, testloader with batch_size=128)
  7. Create student model, load teacher if distillation
  8. Create distillation method instance
  9. Setup optimizer (SGD) and scheduler (CosineAnnealing)
  10. Attempt to resume from checkpoint.pth if exists
  11. Main loop (150 epochs):
      - Forward pass through student
      - Compute CE loss
      - Compute distillation loss (if applicable)
      - Backward, step, scheduler update
      - Save metrics (loss/acc for train and test)
      - Save checkpoint and best.pth
      - Plot curves
  12. Save final metrics.json
  13. Mark status as 'completed'

Key Design:
  - Checkpoint contains full training state (optimizer, scheduler, distillation modules)
  - Best model has weights only (efficient for distillation)
  - Status.json allows quick skip of completed runs
  - Resume mechanism prevents re-training


================================================================================
7. EXPERIMENTAL LAUNCHER (runner.py)
================================================================================

Purpose: Queue all experiments via PBS (Portable Batch System)

Experiment Matrix:
  - Datasets: [Cifar10, Cifar100]
  - Models: Teacher=ResNet112, Student=ResNet56
  - Methods: [pure, logit, factor_transfer]
  - Seeds: [0, 1, 2]
  - Total runs: 2 datasets × (2 pure + 2 distillation methods) × 3 seeds
             = 24 total experiments

Order of Execution (in runner.py):
  1. All pure training first (ensures teacher weights exist)
  2. Then distillation experiments (use trained teachers)
  3. Skip mechanism: if status.json shows 'completed', skip submission

Job Generation:
  - Uses run.job template (PBS header)
  - Substitutes experiment_name and python_cmd
  - Submits via qsub command
  - Tracks submission count against queue limit


================================================================================
8. THESIS STRUCTURE (from nuwe_structure.md)
================================================================================

Target: ~30,000 words, ~80 references, 18-25 figures, 12-14 equations

Chapter Breakdown:
  Ch1: Introduction (1.6-2.0k words, 5-8 refs, 0-1 fig)
       - Motivation, research question, objectives, contributions, outline
       - Epistemic role: pose explanatory problem, not solve it

  Ch2: Neural Networks as Representation-Learning Systems (3.2-3.6k)
       - Beyond function approximation, distributed representations, training

  Ch3: CNNs and Hierarchical Feature Spaces (3.6-4.2k)
       - Inductive bias, feature maps, hierarchies, residual connections

  Ch4: Knowledge Distillation: Success and Fragmentation (6.0-6.5k)
       - Empirical effectiveness (citation-heavy)
       - What's claimed to be transferred
       - The explanatory gap

  Ch5: Latent Spaces and Linear Projections as Lens (4.8-5.2k)
       - Latent spaces, linear probes, eigenvectors, alignment
       - Why linear structure is plausible substrate

  Ch6: Methodology: Probing via Latent Alignment (3.0-3.3k)
       - Experimental design, extraction, projection, evaluation

  Ch7: Results and Analysis (3.8-4.5k, 6-10+ figures)
       - Alignment patterns, when alignment explains performance, failures

  Ch8: Conclusions and Implications (1.4-1.7k)
       - Summary, limits, future work


================================================================================
9. KEY INSIGHTS FROM THE REPOSITORY
================================================================================

1. RESEARCH HYPOTHESIS:
   Linear representation geometry explains what KD transfers, not just how
   or "dark knowledge" metaphors.

2. ANALYSIS TOOLSET:
   - PCA: variance structure and dimensionality
   - ICA: independent factors (reveals non-Gaussian structure)
   - CKA: subspace similarity
   - Principal angles: directional alignment
   - Fisher criterion: class separability
   - Together: multi-faceted lens on representation alignment

3. EARLY FINDINGS (from claude_summary.md):
   - Early features (layer1) converge; late features (layer3) diverge
   - Factor transfer and logit KD use different alignment strategies
   - ICA reveals patterns PCA/CKA miss
   - Student representations MORE class-separable than teacher
   - This is genuine distillation effect, not just copying

4. METHODOLOGICAL INSIGHT:
   "Linear structure is a plausible explanatory substrate"
   - Not claiming linearity IS the mechanism
   - But showing alignment in linear projections captures meaningful signal
   - Enables probe-based analysis without full model understanding

5. PRACTICAL OBSERVATIONS:
   - Distillation more effective on harder tasks (CIFAR-100 vs CIFAR-10)
   - Factor Transfer more stable than Logit KD across seeds
   - Individual seeds show high variance; proper statistical averaging needed
   - All 24 experiments completed successfully for reproducibility


================================================================================
10. FILE STATISTICS
================================================================================

Total Files in Repository:
  - Python code: 9 files (train.py, runner.py, analyze.py, extract.py, 4 toolbox files, setup.sh, tools.py)
  - Experiments: 24 directories × 8-10 files each = ~240 result files
  - Data: 2 datasets (CIFAR-10, CIFAR-100)
  - Analysis: 3 .npz representation files + 9 PNG figures + 1 markdown summary
  - Configuration: 2 markdown/text files (nuwe_structure.md, notes), 1 TOML, 1 gitignore

Repository Size: ~500MB (mostly datasets)
Git History: 5 commits (Experimients, Initial experiments, Training finished, training)


================================================================================
