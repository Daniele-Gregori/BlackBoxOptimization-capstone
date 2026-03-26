# BBO Capstone Project - Final Reflection

## Initial Codebase

My codebase was built from scratch in Python within Jupyter notebooks. I chose this approach rather than using an existing Bayesian optimization library like BoTorch or GPyOpt because I wanted to understand each component from the ground up. The core dependencies were scikit-learn for Gaussian Process regression, NumPy and SciPy for numerical computation, and matplotlib for visualization.

In retrospect, this was both a strength and a limitation. Building from scratch gave me deep understanding of how GP-based optimization works — from kernel selection to acquisition function evaluation to gradient computation. However, it also meant I spent significant time implementing functionality that existing libraries provide out of the box, time that could have been spent on higher-level strategy refinement.

For data handling, I imported the provided data as dictionaries mapping inputs to outputs and built a small visualization app within the notebook that displayed histograms of output distributions, quartile statistics, and multi-dimensional data descriptions for each function simultaneously. This early investment in data exploration tools paid dividends throughout the project by giving me immediate intuition about each function's characteristics. For example, I identified in round one that function 1 was exceptional, having essentially only one non-vanishing output value, which led me to handle it separately through direct inspection rather than surrogate modeling.


## Code Modifications Week by Week

### Rounds 1-3: Foundation and Early Exploration

**Round 1** established the baseline framework. I trained Gaussian Process regressors for each function (except function 1) and evaluated them on uniform grids over the input domain. Grid subdivisions ranged from 5 to 15 per dimension, yielding up to 16,777,216 grid points for the 8-dimensional function. I computed the UCB acquisition function with beta set as the ratio of average predicted means to average standard deviations, and selected the grid point with maximum acquisition value. This required parallel computation across 12 CPU cores and produced approximately 6.5 GB of stored results across 70 CSV files.

**Round 2** adjusted the UCB beta parameter to emphasize exploration after observing that no new maxima were being identified. I heuristically set beta to 4 after experimenting with values in the 3-4 range. The impact was minimal — exploration increased but without a well-calibrated surrogate, this amounted to near-random search in hindsight.

**Round 3** introduced two changes. First, I incorporated classification models by labeling outputs in the upper quartile as positive and training both a Logistic Regression classifier and a Support Vector Machine classifier alongside the GP. The Logistic Regression provided probability estimates and served as a simpler baseline, while the SVM with RBF kernel identified promising regions. I adjusted beta until UCB-selected points fell within those classified regions. Second, I migrated the grid computation to a remote batch computing provider with 192 cores, reducing computation time for the 6-dimensional function from 2 hours to 20 minutes.

**Impact:** These early rounds produced baseline results but no significant optimization progress. The grid-based approach was fundamentally limited by the curse of dimensionality.

### Rounds 4-6: Model Experimentation

**Round 4** was architecturally ambitious. I built a Bayesian Neural Network from scratch with weight distributions instead of fixed weights, using a loss function combining negative log-likelihood with KL divergence regularization. I implemented gradient computation through backpropagation, obtaining partial derivatives with respect to each input dimension. This gave me both sensitivity analysis and a direction for gradient ascent.

**Round 5** refined the BNN architecture to a hierarchical structure of 32→16→8 units, inspired by progressive feature extraction principles. However, with only approximately 16-17 data points per function, this network had far more parameters than data points, leading to overfitting.

**Round 6** was my weakest round, focused primarily on conceptual reflection rather than algorithmic advancement. No significant code changes were made.

**Impact:** The BNN was ultimately abandoned in favor of the GP, but the gradient computation technique from round 4 became a permanent and valuable addition to my toolkit. The key lesson was that model complexity must match data availability — a BNN with hundreds of parameters cannot be reliably trained on fewer than 20 data points.

### Round 7: The Turning Point

This was the single most impactful round. I implemented systematic hyperparameter tuning for the GP, testing 48 configurations comprising 8 kernel types (RBF, Matérn 1.5, Matérn 2.5, Rational Quadratic, and ARD variants) crossed with 6 alpha values (1e-8 through 1e-3), evaluated using 3-fold cross-validation.

Key findings:
- Rational Quadratic and Matérn 2.5 (including ARD variants) each won on 3 of 8 functions, emerging as the two strongest kernel families
- ARD variants were critical for the highest-dimensional function (function 8, 8D), where Matérn 2.5 ARD achieved the best RMSE
- Optimal alpha was bimodal: very small values (1e-8) worked best for some functions while large values (1e-3) worked best for others, with no simple relationship to dimensionality
- The worst/best RMSE ratio reached 5.7× (function 5, 4D), confirming that default hyperparameters can be substantially suboptimal
- Function-specific tuning proved essential — no single kernel-alpha pair was universally optimal

**Impact:** This round transformed my approach from ad hoc to evidence-based. Every subsequent decision was grounded in these cross-validated results.

### Round 8: LLM-Assisted Development

I used the Anthropic API (accessed via Wolfram Language) to generate a complete sequential Bayesian optimization implementation. I tested different temperature and top-p settings, finding that temperature=1.0 produced the most complete solution with sklearn-based GP, multiple acquisition functions, and proper data scaling. However, the LLM-generated code used default alpha=1e-6, which my own tuning had shown was suboptimal. This highlighted that domain-specific empirical knowledge cannot be replaced by generic code generation.

**Impact:** Provided a cleaner implementation framework, but the real value came from combining it with my empirically-derived hyperparameters from round 7.

### Rounds 9-10: Synthesis

**Round 10** was my strongest technical submission. I unified the best elements from previous rounds into a coherent pipeline:

1. Fit GP with Matérn 2.5 kernel and ARD, alpha=1e-4, on all accumulated observations
2. Extract ARD length scales to identify relevant dimensions — length scales ranged from 0.006 (highly relevant) to 1e5 (hitting sklearn's upper bound, confirming true irrelevance)
3. Fix irrelevant dimensions at best known values across all functions, reducing effective dimensionality (e.g. 8D→4D for function 8, 6D→3D for function 7, and even 2D→1D for functions 1 and 2)
4. Run gradient ascent on the GP surrogate from the best known point in the reduced space
5. Cross-check candidate against UCB acquisition function (beta≈3) to maintain exploration pressure

This pipeline replaced the grid search entirely, running in seconds rather than hours.

**Impact:** First submission where computation, model quality, and search strategy were all working together coherently.

### Rounds 11-13: Refinement and Reflection

Round 11 considered clustering perspectives for identifying promising regions but did not introduce major algorithmic changes. Round 12 provided a comprehensive justification for all model selection decisions with quantitative evidence. Round 13 analyzed the project through the lens of reinforcement learning, recognizing that Bayesian optimization is structurally equivalent to model-based RL with the GP serving as the learned environment model.

## Final Results

My scores improved most significantly after round 7's hyperparameter tuning and round 10's unified pipeline. The trajectory was roughly:

- **Rounds 1-6:** Incremental and inconsistent progress, with different approaches producing variable results
- **Rounds 7-10:** Measurable improvement as the surrogate model became well-calibrated and the search strategy became principled
- **Rounds 11-13:** Modest additional gains as the strategy stabilized and remaining improvements came from accumulated data rather than methodological changes

The pipeline predicted improvement on 5 of 8 functions. Contrary to my expectations, the low-dimensional functions (1 and 2, both 2D) showed no predicted improvement — gradient ascent simply converged back to the best known point, indicating those functions were already well-exploited. The predicted gains came from mid- and high-dimensional functions: function 7 (6D→3D effective) and function 8 (8D→4D effective) both showed predicted improvement, as did the minimization functions 3 and 4. Gradient ascent was the winning method on 5 functions, while Expected Improvement won on the remaining 3 — notably on both minimization functions and the highest-dimensional function (8D). UCB never produced the best candidate, suggesting its role is better suited as a cross-check than as a primary acquisition strategy.

### If I Had More Time or a Fresh Start

I would make three fundamental changes:

**First, implement sequential Bayesian optimization with gradient-based acquisition function optimization from round 1.** My grid-based approach consumed enormous computational resources for marginal benefit. Using L-BFGS-B with multi-start optimization to maximize Expected Improvement would have been faster by orders of magnitude and more effective. I eventually implemented this in the unified pipeline (round 10), where the EI acquisition function with 200 multi-start L-BFGS-B runs replaced the grid search entirely, running in seconds rather than hours.

**Second, conduct systematic hyperparameter tuning in round 1, not round 7.** The six rounds between baseline and proper tuning were largely wasted in terms of optimization performance.

**Third, use an established BO library like BoTorch as a benchmark** alongside my from-scratch implementation, to quickly identify whether my results were competitive or whether implementation issues were holding me back.

## Trade-offs and Decisions

### Exploration vs. Exploitation

This was the central tension throughout the project. My approach evolved through three distinct phases:

- **Phase 1 (Rounds 1-3):** Aggressive exploration with high beta values, driven by failure to find new maxima. In retrospect, the problem was surrogate quality, not insufficient exploration.
- **Phase 2 (Rounds 4-9):** Experimentation with multiple models and strategies. Exploration was happening at the meta-level — I was exploring the space of methods rather than the input space.
- **Phase 3 (Rounds 10-13):** Dimension-adaptive strategy combining gradient ascent (exploitation) with Expected Improvement (balanced exploration-exploitation). In practice, the low-dimensional functions were already well-exploited and showed no further gains, while the higher-dimensional functions benefited most from the unified pipeline after ARD-based dimension reduction.

The most important realization was that **surrogate model quality matters more than exploration strategy**. A well-calibrated GP with moderate exploration dramatically outperforms a poorly calibrated model with any exploration policy.

### Breadth vs. Depth of Methods

I explored GP, Logistic Regression, BNN, SVM classification, LLM code generation, and gradient-based optimization. This breadth provided valuable learning but came at the cost of optimization performance. The BNN in particular consumed two rounds of effort (4-5) without improving results. A more disciplined approach would have been to exhaust the potential of the GP before exploring alternatives.

### Computational Investment vs. Algorithmic Efficiency

My most consequential early mistake was investing in hardware solutions — a second laptop, remote cloud computing with 192 cores — to scale a fundamentally inefficient algorithm (grid search). The eventual switch to gradient-based acquisition optimization made all that infrastructure unnecessary. **Algorithmic efficiency almost always trumps computational power.**

## Learning and Application

### Most Important Lesson

The single most important lesson is deceptively simple:

> **Match your method complexity to your data availability, and optimize your best method thoroughly before trying alternatives.**

With 15-21 data points, the GP was always going to be the right surrogate model. The BNN was too parameter-rich. The SVM classification was an interesting idea but addressed a problem I didn't actually have. The grid search was solving the wrong computational problem. If I had spent rounds 1-3 properly tuning the GP and implementing gradient-based acquisition optimization, I would likely have achieved better final results with far less total effort.

This lesson applies directly to real-world ML projects where data is often limited, computational budgets are constrained, and the temptation to try the latest complex method is strong. A well-tuned logistic regression frequently outperforms a poorly tuned neural network. A carefully implemented GP with ARD frequently outperforms a hastily built BNN. **Thoroughness beats novelty.**

### Application to Future Work

For future competitions or real-world optimization problems, I would:

1. **Start with established libraries** (BoTorch, Ax) as a strong baseline before building custom solutions
2. **Invest in surrogate model validation early** through cross-validation, not after multiple rounds of poor results
3. **Use ARD or sensitivity analysis immediately** to identify relevant dimensions and reduce the search space
4. **Track quantitative results rigorously** from round 1 with clear tables showing improvement over time
5. **Treat acquisition function optimization as seriously as surrogate modeling** — a perfect surrogate is useless if you can't efficiently find the maximum of the acquisition function

### What Surprised Me

What surprised me most about my own process was how long I persisted with approaches that weren't working. The grid search survived nine rounds despite clear evidence of diminishing returns. The BNN consumed two rounds despite immediate signs of overfitting. I believe this persistence was partly driven by sunk cost reasoning — having invested heavily in grid computation infrastructure, I was reluctant to abandon it — and partly by the natural human tendency to add complexity rather than refine simplicity.

Looking at the project holistically, the pattern is clear: my three strongest submissions (rounds 1, 7, and 10) were the ones where I focused on doing one thing well rather than trying to incorporate multiple new ideas. This is a lesson I expect to carry forward into every future ML project.
