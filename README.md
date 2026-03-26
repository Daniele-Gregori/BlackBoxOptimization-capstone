# Black-Box Optimization Capstone Project

## Table of Contents

- [Project Overview](#project-overview)
- [Inputs and Outputs](#inputs-and-outputs)
- [Challenge Objectives](#challenge-objectives)
- [Technical Approach](#technical-approach)
- [Results Summary](#results-summary)
- [Documentation](#documentation)
- [Setup and Reproduction](#setup-and-reproduction)

---

## Project Overview

### What is this project?

This repository contains my complete solution to a Black-Box Optimization (BBO) capstone project, developed over ten iterative rounds as part of a machine learning and artificial intelligence module. The challenge involves finding the inputs that maximize eight unknown functions of varying dimensionality, ranging from 2 to 8 dimensions, with no access to the functions' analytical forms, gradients, or structural properties. The only interaction with the functions is through an oracle that returns exact evaluations at queried points.

### Why is this relevant?

Black-box optimization is fundamental to many real-world machine learning applications where the objective function is expensive, opaque, or both. Common examples include:

- **Hyperparameter tuning** — finding optimal learning rates, regularization strengths, and architecture choices for neural networks without a closed-form relationship between hyperparameters and model performance
- **Drug discovery** — optimizing molecular properties where each evaluation requires costly laboratory experiments or simulations
- **Materials science** — identifying compositions or process parameters that yield desired material properties
- **Engineering design** — optimizing aerodynamic shapes, circuit layouts, or structural configurations where each evaluation requires expensive simulation

In all these settings, the core challenge is the same: how to make intelligent decisions about where to query next when evaluations are limited and the landscape is unknown. This project provides a controlled environment to develop and test strategies for exactly this problem.

### Career relevance

This project develops skills that are directly transferable to data science and ML engineering roles, including surrogate modeling, uncertainty quantification, sequential decision-making under uncertainty, and the practical engineering of computational pipelines. The iterative nature of the challenge, where each round's strategy must be informed by previous results, mirrors the reality of applied ML projects where models are continuously refined based on new data and feedback. The experience of working with extremely sparse data in high-dimensional spaces is particularly relevant, as many real-world optimization problems share this fundamental constraint.

---

## Inputs and Outputs

### Input Format

Each query consists of a continuous-valued vector whose dimensionality depends on the function being optimized:

```
Function 1: x = [x1, x2]                                    # 2 dimensions
Function 2: x = [x1, x2]                                    # 2 dimensions
Function 3: x = [x1, x2, x3]                                # 3 dimensions
Function 4: x = [x1, x2, x3, x4]                            # 4 dimensions
Function 5: x = [x1, x2, x3, x4]                            # 4 dimensions
Function 6: x = [x1, x2, x3, x4, x5]                        # 5 dimensions
Function 7: x = [x1, x2, x3, x4, x5, x6]                   # 6 dimensions
Function 8: x = [x1, x2, x3, x4, x5, x6, x7, x8]          # 8 dimensions
```

All input dimensions are bounded within known continuous ranges. One query per function is permitted per round, for a total of ten rounds.

### Output Format

The oracle returns a single scalar value for each query:

```
y = f(x)    # continuous real-valued output
```

### Example

```python
# Query for Function 3 (3 dimensions)
query = [0.45, 0.72, 0.31]

# Oracle response
response = 2.847
```

### Data Storage

Query-response pairs are stored as Python dictionaries mapping input tuples to output values and exported as CSV files for persistence. Intermediate model evaluations, including GP predictions across grids, are stored as separate CSV files.

---

## Challenge Objectives

### Primary Goal

**Maximize** each of the eight unknown functions by selecting optimal query points across ten rounds of submission.

### Constraints and Limitations

| Constraint | Detail |
|-----------|--------|
| **Query budget** | One query per function per round, ten rounds total |
| **Function access** | Black-box only — no analytical form, derivatives, or structural information |
| **Initial data** | Approximately 10 provided points per function |
| **Total data** | ~20 points per function by final round |
| **Dimensionality** | Ranges from 2D to 8D across the eight functions |
| **Domain bounds** | Known continuous ranges for each input dimension |
| **Evaluation type** | Exact (no observation noise from the oracle, though model uncertainty remains) |

### Core Challenge

The fundamental difficulty is the extreme sparsity of data relative to the dimensionality of the search space. With approximately 20 points in 8 dimensions, the coverage of the input space is negligibly small. Every query must therefore be chosen to maximize the information gained while also pursuing the optimization objective — the classic exploration versus exploitation tradeoff.

---

## Technical Approach

This section documents the evolution of my optimization strategy across ten rounds. It serves as a living record of methodological decisions, their outcomes, and the reasoning that guided each transition.

### Phase 1: Baseline Establishment (Rounds 1–2)

**Core method:** Gaussian Process regression with UCB acquisition function

The initial approach used a Gaussian Process as the surrogate model, chosen for its ability to provide both predicted values and uncertainty estimates at unsampled points. The GP was evaluated over uniform grids spanning the input domain, with subdivisions ranging from 5 to 15 per dimension depending on the function's dimensionality.

```
Grid sizes ranged from:
  - 16^2  =        256 points (2D functions)
  - 16^5  =  1,048,576 points (5D function)
  - 11^6  =  1,771,561 points (6D function)
  - 8^8   = 16,777,216 points (8D function)
```

The Upper Confidence Bound acquisition function was used to select query points, with beta initially set as the ratio of mean predicted values to standard deviations. In round 2, beta was increased to approximately 4 after observing that no new maxima were being discovered, shifting the balance toward exploration.

**Computation:** Parallel evaluation across 12–24 CPU cores, producing approximately 6.5 GB of intermediate CSV files across 70 files.

**Key insight:** Function 1 was identified as anomalous, having essentially one non-negligible output value. It was handled manually rather than through the standard GP pipeline.

> **Notebook sections:** 1–4 (Data Wrangling, Progress Analysis, Grid Search, Gaussian Processes)

### Phase 2: Model Diversification (Rounds 3–6)

**Core methods:** GP + classification (Logistic Regression, SVM), Bayesian Neural Network, gradient analysis

**Round 3 — Classification for region identification:**
The optimization problem was recast as a classification task by labeling outputs above the upper quartile as positive. Both a Logistic Regression classifier (providing probability estimates) and an SVM with RBF kernel were trained to identify promising regions. The classification boundaries informed where to push UCB exploration, creating a hybrid strategy where classifiers provided strategic direction and the GP provided precise predictions.

**Rounds 4–6 — Bayesian Neural Network:**
A BNN was implemented from scratch in PyTorch with three Bayesian linear layers (32→16→8 units). Instead of fixed weights, distributions were learned, with the loss function combining negative log-likelihood and KL divergence regularization. Multiple forward passes with different weight samples provided mean predictions and uncertainty estimates.

Critically, gradient computation was implemented to assess the sensitivity of outputs to each input dimension. This enabled both feature importance ranking through gradient magnitude and gradient ascent from best known points as a directed search strategy.

**Visualization:** 3D plots of pairwise dimension slices revealed four distinct surface types across the eight functions:

| Type | Description | Functions |
|------|-------------|-----------|
| Near-linear | Output resembles a plane | Some low-dimensional |
| Single peak | Clear unimodal maximum | Several functions |
| Mixed extrema | Multiple peaks and valleys | Mid-dimensional |
| Oscillatory | Periodic-like variations | Most challenging |

> **Notebook sections:** 5–7 (Logistic Regression, SVM, Neural Networks)

### Phase 3: Systematic Refinement (Rounds 7–8)

**Core method:** Hyperparameter tuning with cross-validation

Round 7 represented the most impactful methodological advance. A systematic grid search tested 48 configurations:

- **8 kernels:** RBF, Matérn (ν=1.5, ν=2.5), Rational Quadratic, composite kernels, and ARD variants
- **6 alpha values:** [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
- **Evaluation:** 3-fold cross-validation

**Key findings:**

| Finding | Impact |
|---------|--------|
| RQ and Matérn 2.5 each best on 3/8 functions | No single kernel universally optimal — function-specific selection essential |
| Matérn 2.5 ARD best for 8D function | ARD critical for high-dimensional problems |
| Optimal alpha bimodal: 1e-8 or 1e-3 | No simple dimensionality–alpha relationship |
| Worst/best RMSE ratio up to 5.7× | Default hyperparameters substantially suboptimal |

**Round 8 — LLM-assisted code generation:**
The Anthropic API was used with Claude Opus to generate complete Bayesian optimization implementations. Experiments with temperature (default vs 1.0) and top-p (default vs 0.9) sampling revealed that T=1.0 produced the most complete and reliable code, while higher top-p introduced creative but structurally problematic choices.

> **Notebook section:** 8 (GP Hyperparameter Tuning)

### Phase 4: Synthesis (Rounds 9–10)

**Core method:** Unified pipeline combining best elements from all phases

The final strategy integrated the strongest findings into a single pipeline:

1. **Fit tuned GP** using Matérn 2.5 + ARD kernel with alpha=1e-4
2. **Extract ARD length scales** to identify irrelevant dimensions — length scales ranged from 0.006 to 1e5 (sklearn bound), with clear relevance separation across all functions
3. **Fix irrelevant dimensions** at current best values, reducing effective search space (8D→4D, 6D→3D, even 2D→1D)
4. **Expected Improvement** maximization via multi-start L-BFGS-B (200 restarts) in the reduced space
5. **Gradient ascent** on the GP mean surface from the best known point
6. **UCB cross-check** (beta≈3) to maintain exploration pressure
7. **Select best candidate** across all three methods

```python
# Simplified final pipeline pseudocode
for each function:
    gp = fit_tuned_gp(data, best_kernel, best_alpha, ARD=True)
    important_dims = identify_relevant_dimensions(gp.length_scales)
    x_ei = multistart_lbfgsb(expected_improvement, gp, dims=important_dims)
    x_ga = gradient_ascent(gp, x_best, dims=important_dims)
    x_ucb = multistart_lbfgsb(ucb, gp, beta=3, dims=important_dims)
    x_final = best_candidate(x_ei, x_ga, x_ucb)
    submit(x_final)
```

> **Notebook sections:** 9–10 (ARD Analysis & Dimension Reduction, Unified Optimization Pipeline)

### Exploration vs. Exploitation Balance

The balance between exploration and exploitation evolved deliberately across the project:

- **Rounds 1–6:** Heavily exploration-focused, driven by the observation that promising regions had not yet been identified
- **Round 7:** Balanced — systematic tuning improved both the model's predictive accuracy and its ability to guide exploitation
- **Rounds 8–10:** Shifted toward exploitation for well-characterized functions while maintaining exploration for challenging high-dimensional ones

The transition was informed by function-specific landscape understanding: near-linear surfaces warranted aggressive exploitation, while oscillatory or complex surfaces required continued exploration.

### What Makes This Approach Unique

The approach is distinctive in three ways. First, the deliberate progression from simple to sophisticated methods, with honest evaluation of which innovations actually improved performance. Second, the creative combination of models not typically used together in BBO, particularly the SVM-GP hybrid for region identification. Third, the use of ARD length scales not just for model improvement but as a strategic tool for dimensionality reduction, which transformed the tractability of higher-dimensional functions.

---

## Results Summary

| Function | Dims | Eff. Dims | Winner | Improved? | Key Observation |
|----------|------|-----------|--------|-----------|-----------------|
| 1 | 2 | 1 | GA | No | Already well-exploited; dim 0 irrelevant (ls=705) |
| 2 | 2 | 1 | GA | No | Already well-exploited; dominated by dim 0 |
| 3 | 3 | 2 | EI | Yes | EI found candidate below best observed (min) |
| 4 | 4 | 2 | EI | Yes | Uniform length scales (2.6–3.0), ambiguous reduction |
| 5 | 4 | 2 | GA | No | Dim 0 highly irrelevant (ls=19.3) |
| 6 | 5 | 3 | GA | Yes | Dim 0 irrelevant (ls=19,444); GA improved over best |
| 7 | 6 | 3 | GA | Yes | Dim 2 irrelevant (ls=1e5); GA found better candidate |
| 8 | 8 | 4 | EI | Yes | Dim 7 irrelevant (ls=1e5); EI best on highest-dim |

Detailed performance metrics are available in the [Model Card](docs/model_card.md).

---

## Documentation

- **[Datasheet](docs/datasheet.md)** — Complete documentation of the query dataset including motivation, composition, collection process, preprocessing, and distribution details
- **[Model Card](docs/model_card.md)** — Comprehensive description of the optimization approach including intended use, technical details, performance summary, assumptions, limitations, and ethical considerations

---

## Setup and Reproduction

### Requirements

```
python >= 3.9
scikit-learn >= 1.0
pytorch >= 1.12
numpy >= 1.21
scipy >= 1.7
pandas >= 1.4
matplotlib >= 3.5
```

---

*This project is being carried out as part of an EdTech ML/AI module. The iterative development process across ten rounds is documented in full to support transparency, reproducibility, and learning.*