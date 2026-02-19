You’re right — I dropped those in the last recapitulation. Here’s the **full revised structure again**, this time with **word + reference + figure + equation targets** at every chapter, plus brief per-section notes where it matters.

I’ll keep it operational: numbers you can actually aim at.

---

# REVISED STRUCTURE WITH ALL TARGETS

**Total words:** ~28 500–31 500 (nominal ~30 000)
**Total references:** ~75–95 (target ~80)
**Total figures:** ~18–25
**Total equations:** ~12–14 (mostly Chapter 5)


================================================================================================================================================
## Chapter 1 — Introduction

**Epistemic role:** pose the *explanatory problem*, not solve it.

**Words:** 1 600–2 000
**Refs:** 5–8
**Figs:** 0–1
**Eqs:** 0

### 1.1 Motivation — 600–750

Refs: 3–5 (mostly KD + “why explanation matters”)
Figs/Eqs: none

#### 1
* Model compression is mature
* Knowledge distillation works empirically
* **But** its explanations are fragmented, informal, and method-specific
* What lives here:
  * cost of large models
  * success of KD
  * lack of explanatory clarity
* Do **not** explain mechanisms.
* If this goes over 1 200 words, you’re editorializing.

**Purpose:** Define the *explanatory problem*, not the field.
#### 2
Epistemic role: define the problem, not the solution
This chapter answers why the thesis exists at all.
##### Motifivaiton
It must do three things, in order:
You start from a fact the community already agrees on:
- Deep models are powerful but expensive
- Knowledge distillation works and is widely used
Then you pivot sharply:
- Despite empirical success, KD lacks a clear, unifying explanation
- Existing accounts are fragmented, metaphorical (“dark knowledge”), or method-bound
Crucially, you are not saying:
- KD is mysterious
- Prior work is wrong
You are saying:
- Prior work explains how to apply KD, not what is structurally transferred
This frames your thesis as a clarification effort, not a challenge.

### 1.2 Research Question — 150–250

Refs: 0–1 (optional)
Figs/Eqs: none

* *What structural properties of neural network representations enable effective knowledge distillation?*
  (Not “how to distill better”, but “what is being transferred?”)
* Single, focused section.
  * state the central question
  * clarify scope (CNNs, classification, feature-based KD)
  * explicitly exclude what you’re *not* doing
* This should feel sharp and slightly austere.

### 1.3 Objectives — 250–350

Refs: 0
Figs/Eqs: none

* Establish a representation-level lens
* Use linear latent analysis as an explanatory probe
* Test whether this lens coherently explains multiple KD strategies
* Translate the question into *operational aims*.
* Each objective should:
  * begin with a verb
  * map cleanly to later chapters
* Avoid implementation detail.

### 1.4 Contributions — 250–350

Refs: 0
Figs/Eqs: none

* Reframing KD as representation alignment under projection
* Empirical evidence that linear structure captures explanatory signal
* Clarification of when this framing breaks down
* This is not a list of “things I did”.
* Each contribution should be:
* framed as intellectual movement
* phrased cautiously (reframing, probing, clarifying)

### 1.5 Thesis Outline — 200–300

Refs: 0
Figs/Eqs: none

* (Explicitly mention the *lens* appears before method evaluation.)
* Walk the reader through the **logic**, not the table of contents.
* If this feels boring, that’s good — it means it’s clear.

**Guardrail:** If Chapter 1 needs equations or more than 1 figure, it’s bleeding content from later chapters.
### References
**Low density. 5–8 total.**
* 1–2 general deep learning textbooks/surveys
* 2–3 KD foundational papers
* 1–2 papers explicitly noting KD’s empirical success
**Rule:**
Every reference here should justify *why the problem matters*, not *how it works*.
No citation clusters. No inline citation storms.
### Figures
**0–1 figure (optional).**
If included, it should be **conceptual**, not technical:
* Teacher → student schematic
* “Empirical success vs explanatory clarity” diagram
No plots. No architectures.
### Equations
**Zero.**
Any equation here is a mistake.
### Failure modes
* Explaining KD mechanisms (too early)
* Claiming novelty
* Defining latent spaces



================================================================================================================================================






## Chapter 2 — Neural Networks as Representation-Learning Systems

**Words:** 3 200–3 600
**Refs:** 3–5
**Figs:** 1–2
**Eqs:** 1–3

### 2.1 Beyond Function Approximation — 600–700

Refs: 1–2 (universal approximation + 1 textbook/survey)
Figs: 0
Eqs: 0

### 2.2 Neurons, Layers, Distributed Representations — 900–1 100

Refs: 1 (textbook anchor)
Figs: 1 (simple layer/representation diagram)
Eqs: 1 (neuron: (y=\sigma(w^\top x+b)))

### 2.3 Activation Functions and Nonlinearity — 600–700

Refs: 0–1 (optional; usually same textbook)
Figs: 0
Eqs: 0–1 (only if you define a generic nonlinearity)

### 2.4 Training and Backpropagation (Minimal) — 700–900

Refs: 1 (textbook or classic backprop reference)
Figs: 0–1 (optional computational graph sketch)
Eqs: 0–1 (loss notation at most)

**Guardrail:** No gradient derivations. Equations here are definitional, not analytical.


================================================================================================================================================
## Chapter 3 — CNNs and Hierarchical Feature Spaces

**Words:** 3 600–4 200
**Refs:** 6–10
**Figs:** 2–4
**Eqs:** 0–2

### 3.1 Inductive Bias and Spatial Structure — 700–900

Refs: 1–2 (CNN origin + inductive bias source)
Figs: 0–1 (optional: locality/weight sharing cartoon)
Eqs: 0

### 3.2 Feature Maps as Intermediate Representations — 1 200–1 400

Refs: 2–3
Figs: 1–2 (feature map illustration is ideal)
Eqs: 0–1 (optional convolution definition, symbolic only)

### 3.3 Hierarchies of Abstraction — 900–1 100

Refs: 1–2
Figs: 1 (early vs late features schematic)
Eqs: 0

### 3.4 Residual Connections and Representation Stability — 600–700

Refs: 1 (ResNet)
Figs: 1 (residual block schematic)
Eqs: 0–1 (optional: (y=x+F(x)))

**Guardrail:** Don’t do a CNN taxonomy. The visuals should explain “representations evolve”.

================================================================================================================================================

## Chapter 4 — Knowledge Distillation: Success and Explanatory Fragmentation

**Words:** 6 000–6 500
**Refs:** 30–40
**Figs:** 1–2
**Eqs:** 0–1

### 4.1 Empirical Effectiveness of KD — 1 200–1 400

Refs: 8–12 (foundational KD + representative results papers)
Figs: 0–1 (optional KD pipeline figure if not earlier)
Eqs: 0–1 (KD loss with temperature, no derivation)

### 4.2 What Is Claimed to Be Transferred — 2 000–2 400

Refs: 15–20 (this is the citation-heavy core)
Figs: 0–1 (optional taxonomy graphic; can also live in 4.3)
Eqs: 0

### 4.3 The Explanatory Gap — 1 800–2 200

Refs: 6–10 (papers that *claim* explanations, surveys, critiques)
Figs: 1 (taxonomy or “claims map” is strongly recommended)
Eqs: 0

**Guardrail:** This chapter must *end* without your solution. It should feel like “we need a lens”.

================================================================================================================================================
# NOTE: As Anna mentioned, maybe chapter 5 should be split into two chapters? One for purely PCA and ICA. One for how it applies to neural networks representations

## Chapter 5 — Latent Spaces and Linear Projections as a Lens

**Words:** 4 800–5 200
**Refs:** 15–20
**Figs:** 3–5
**Eqs:** 4–6

### 5.1 Latent Spaces in Neural Networks — 1 200–1 400

Refs: 3–5 (representation/geometry/manifold basics)
Figs: 1 (high-D data on low-D manifold cartoon)
Eqs: 0–1 (notation only)

### 5.2 Linear Projections as Probes, Not Models — 1 200–1 400

Refs: 4–6 (PCA/SVD + probing/interpretability framing)
Figs: 1 (projection as probe schematic)
Eqs: 1 (projection notation)

### 5.3 Eigenvectors, Variance, and Alignment — 1 000–1 200

Refs: 3–5
Figs: 1–2 (PCA axes / variance directions / alignment vs misalignment)
Eqs: 2–3 (SVD, PCA projection, variance explained)

### 5.4 Why Linear Structure Is a Plausible Explanatory Substrate — 900–1 100

Refs: 2–4 (limits + why linear probes are defensible)
Figs: 0–1 (optional assumptions/limits diagram)
Eqs: 0–1 (if you formalize an “alignment score” conceptually)

**Guardrail:** Equations here should clarify the lens, not prove theorems.

---

## Chapter 6 — Methodology: Probing Distillation Through Latent Alignment

**Words:** 3 000–3 300
**Refs:** 5–8
**Figs:** 2–3
**Eqs:** 2–3

### 6.1 Experimental Design — 700–900

Refs: 2–3 (datasets, base architectures)
Figs: 0–1 (overall experiment schematic)
Eqs: 0

### 6.2 Representation Extraction — 700–800

Refs: 0–1 (if following a standard practice)
Figs: 1 (layer tap points diagram)
Eqs: 0

### 6.3 Projection and Comparison Procedure — 800–1 000

Refs: 1–2
Figs: 1 (workflow: extract → project → compare)
Eqs: 1–2 (metric definitions; reuse Chapter 5 notation)

### 6.4 Evaluation Criteria — 600–700

Refs: 1–2 (if criteria inspired by prior probing work)
Figs: 0
Eqs: 0–1 (final metric summary)

**Guardrail:** No new conceptual machinery here — only operationalization.

---

## Chapter 7 — Results and Analysis

**Words:** 3 800–4 500
**Refs:** 0–3
**Figs:** 6–10+
**Eqs:** 0

### 7.1 Alignment Patterns Across KD Methods — 1 400–1 700

Refs: 0–1
Figs: 2–4 (core plots)

### 7.2 When Alignment Explains Performance — 1 200–1 400

Refs: 0–1
Figs: 2–3

### 7.3 Failure Cases and Limitations — 1 200–1 400

Refs: 0–1
Figs: 2–3 (include at least one “failure” figure)

**Guardrail:** Results should be figure-driven. If you’re writing walls of text, you’re narrating instead of analyzing.

---

## Chapter 8 — Conclusions and Implications

**Words:** 1 400–1 700
**Refs:** 0–3
**Figs:** 0
**Eqs:** 0

### 8.1 Summary of Findings — 600–700

Refs: 0

### 8.2 Limits of Linear Explanations — 400–600

Refs: 0–1 (optional)

### 8.3 Implications and Future Work — 400–500

Refs: 0–2 (optional positioning)

**Guardrail:** No new claims, no new concepts, no new figures.

---

# CONSOLIDATED TOTALS (TARGETED)

* **Words:** ~30k
* **References:** ~80 (range ~75–95)
* **Figures:** ~18–25
* **Equations:** ~12–14

  * Ch2: 1–3
  * Ch3: 0–2
  * Ch4: 0–1
  * Ch5: 4–6
  * Ch6: 2–3
  * Ch7–8: 0

If you want, I can turn this into a **one-page checklist** you keep beside you while writing (chapter budgets + “allowed content” per chapter). That alone prevents 80% of thesis drift.
