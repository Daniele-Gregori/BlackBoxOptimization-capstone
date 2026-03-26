# Datasheet for BBO Capstone Query Dataset

## Motivation

This dataset was created to support a black-box optimization (BBO) capstone project in which the objective is to identify input values that maximize eight unknown functions of varying dimensionality. The dataset contains the complete history of query-response interactions with the oracle system across ten rounds of optimization.

The dataset serves two purposes. First, it supports the iterative optimization task itself, as each new round of queries is informed by all previously accumulated data. Second, it enables retrospective analysis of how different optimization strategies performed under severe data constraints.

The dataset was created by a single learner as part of a structured academic module on machine learning and artificial intelligence. No external funding or commercial interest motivated its creation. The intended beneficiaries are the learner for iterative model improvement, academic assessors evaluating the project, and any future students or researchers who wish to study how different optimization strategies perform with extremely limited function evaluations.

## Composition

The dataset consists of query-response pairs for **eight functions** with the following structure:

| Function | Input Dimensions | Approximate Data Points |
|----------|-----------------|------------------------|
| 1 | 2 | ~19 |
| 2 | 2 | ~19 |
| 3 | 3 | ~19 |
| 4 | 4 | ~19 |
| 5 | 4 | ~19 |
| 6 | 5 | ~19 |
| 7 | 6 | ~19 |
| 8 | 8 | ~19 |

Each instance comprises an input vector of continuous values bounded within known domain ranges and a scalar continuous output representing the function evaluation at that point. The total dataset contains approximately **160 instances** across all functions, starting from roughly 10 provided initial points per function and growing by one point per round over ten rounds.

Data is stored in dictionary format mapping input arrays to output values and exported as CSV files. All values are numerical. There are no missing values, as every submitted query received a valid response. There is no confidential, sensitive, or personally identifiable information.

**Key gaps:**
- The dataset is extremely sparse relative to the input dimensionality, particularly for functions 7 and 8 where 19 points in 6 or 8 dimensions provides negligible coverage of the input space
- Queries are clustered around model-identified promising regions, leaving vast areas of the domain entirely unsampled
- Function 1 is anomalous, having essentially only one non-negligible output value among the initial data points

## Collection Process

Data was collected over **approximately ten weeks**, with one new query per function submitted each round. The initial data points were provided as part of the project setup. Subsequent queries were generated through an evolving set of strategies:

- **Rounds 1–2:** Gaussian Process surrogate models with Upper Confidence Bound (UCB) acquisition function, evaluated over brute-force uniform grids with 5–16 subdivisions per dimension
- **Round 3:** Hybrid GP-SVM approach, using Support Vector Machine classification of the upper quartile to identify promising regions and inform UCB exploration
- **Rounds 4–6:** Bayesian Neural Network trained from scratch with gradient-based sensitivity analysis and gradient ascent from best known points
- **Round 7:** Systematic hyperparameter tuning of GP kernel type, ARD configuration, and noise parameter alpha across 48 configurations with 3-fold cross-validation
- **Round 8:** LLM-generated optimization code using the Anthropic API with Claude Opus, tested across temperature and sampling parameter variations
- **Rounds 9–10:** Synthesis strategy combining tuned GP with ARD-based dimension reduction, gradient ascent, and UCB validation

All queries after the initial provided points were model-driven rather than randomly sampled. Computation was performed on two personal laptops with 8/24 cores and intermittently on a remote batch computation provider with 192 cores. Intermediate grid evaluations produced approximately **6.5 GB** of CSV files across 70 files.

## Preprocessing and Uses

**Transformations applied:**
- Standard scaling of inputs for neural network training
- Binary classification encoding of outputs for SVM analysis, with values above the upper quartile labeled as 1 and all others as 0
- No transformations applied to outputs for GP fitting beyond regularization through the alpha parameter

**Intended uses:**
- Iterative black-box optimization through surrogate modeling and acquisition functions
- Retrospective analysis of optimization strategy effectiveness
- Educational demonstration of Bayesian optimization under data constraints

**Inappropriate uses:**
- Treating this dataset as representative of the true function landscapes given its extreme sparsity
- Training general-purpose regression models expected to generalize across the full input domain
- Drawing conclusions about function behavior in unsampled regions without acknowledging the profound uncertainty involved

## Distribution and Maintenance

The dataset is stored locally on the learners's machines and within Jupyter notebooks submitted as part of the capstone project. It is **not publicly distributed**. There are no licensing restrictions beyond standard academic use policies.

No formal versioning system is in place, though each round's submissions are tracked chronologically and intermediate model outputs are preserved as dated CSV files. The learner is the sole maintainer and no updates are planned beyond the capstone submission period.

No ethical review was required as the data involves only mathematical function evaluations with no human subjects, sensitive information, or potential for societal harm.