# Verification primitives: 20 foundational methodologies across domains

Verification is fundamentally easier than generation because the **rule space is vastly smaller than the action space**. This framework systematically organizes 20 core evaluation methodologies—each with rigorous mathematical foundations—that transfer across clinical trials, software engineering, forensics, operations research, and AI evaluation. These methodologies share a common structure: they reduce complex verification problems to computable quantities with known statistical properties.

The table below presents each methodology across three layers: foundational mathematics, standard implementation, and advanced extensions. Together, they form a **universal verification toolkit** applicable to AI agent systems.

---

## Summary table: 20 core verification methodologies

| # | Methodology | Domain Origin | Core Math | Cross-Domain Transfer |
|---|-------------|---------------|-----------|----------------------|
| 1 | Hypothesis Testing Frameworks | Clinical trials | Z/t-tests, CI bounds, TOST | A/B testing, model comparison |
| 2 | Power Analysis | Clinical trials | Effect size, Type I/II errors | Experiment design universally |
| 3 | Sequential Testing | Clinical trials | Alpha-spending functions | Online A/B testing, continuous monitoring |
| 4 | Multiplicity Corrections | Clinical trials | FWER, FDR control | Multi-metric evaluation |
| 5 | Survival Analysis | Clinical trials | Kaplan-Meier, Cox hazards | Churn analysis, time-to-failure |
| 6 | Diagnostic Accuracy | Medicine | ROC/AUC, LR+/LR- | Classification evaluation |
| 7 | Agreement Metrics | Medicine/NLP | Cohen's κ, ICC, Krippendorff α | Annotation quality |
| 8 | Bayesian Evidence Reasoning | Forensics | Likelihood ratios, posterior odds | Hypothesis evaluation |
| 9 | Provenance Verification | Forensics | Hash functions, Merkle trees | Data integrity, audit trails |
| 10 | Decision Theory | OR | Expected utility, minimax | Resource allocation |
| 11 | Multi-Criteria Decision Making | OR | AHP, TOPSIS | Complex rankings |
| 12 | Queuing Theory | OR | Little's Law, M/M/c | System performance |
| 13 | Monte Carlo Simulation | OR | LLN, variance reduction | Uncertainty propagation |
| 14 | Reliability Theory | Engineering | Weibull, hazard functions | Component failure, degradation |
| 15 | Fault Tree Analysis | Engineering | Boolean algebra, minimal cut sets | Risk decomposition |
| 16 | Metamorphic Testing | Software QA | Metamorphic relations | Oracle-free verification |
| 17 | Coverage Metrics | Software QA | MC/DC, mutation score | Test completeness |
| 18 | Calibration Metrics | ML | ECE, Brier decomposition | Confidence trustworthiness |
| 19 | Ranking & Preference Learning | ML/Gaming | Bradley-Terry, Elo | Model comparison |
| 20 | Conformal Prediction | ML | Coverage guarantee, quantiles | Distribution-free UQ |

---

## Detailed three-layer breakdown

### 1. Hypothesis testing frameworks (superiority, non-inferiority, equivalence)

**Layer 1 — Core Math**

Three fundamental paradigms structure statistical comparison:

- **Superiority**: H₀: μ_test − μ_control ≤ 0 | H₁: μ_test − μ_control > 0. Reject if 95% CI excludes zero.
- **Non-inferiority (NI)**: H₀: μ_test − μ_control ≤ −Δ | H₁: μ_test − μ_control > −Δ. Declare NI if lower CI bound > −Δ.
- **Equivalence (TOST)**: H₀: |μ_test − μ_control| ≥ Δ | H₁: −Δ < difference < Δ. Requires entire CI within (−Δ, +Δ).

The test statistic for two-sample comparison: **Z = (X̄₁ − X̄₂) / SE(difference)**, where SE = √(σ₁²/n₁ + σ₂²/n₂).

**Layer 2 — Basic Application**

| Trial Type | α Convention | Population | Key Requirement |
|------------|--------------|------------|-----------------|
| Superiority | Two-sided 0.05 | ITT preferred | p < 0.05 |
| Non-inferiority | One-sided 0.025 | Both ITT and PP | Pre-specified margin Δ |
| Equivalence | Two-sided 0.05 | Both ITT and PP | CI entirely within bounds |

FDA guidance requires the non-inferiority margin M₂ ≤ 50% of the historical effect size M₁, with assay sensitivity and constancy assumptions validated.

**Layer 3 — Variations & Advanced**

The **synthesis method** tests preserved fraction: θ_test > f × θ_control (typically f = 0.5 or 0.67). **Three-arm trials** (test/active control/placebo) provide gold-standard evidence when ethical. In tech, NI frameworks enable "no worse than baseline by X%" evaluations for feature launches—directly transferable from FDA methodology to production ML.

---

### 2. Power analysis and sample size determination

**Layer 1 — Core Math**

The foundational sample size formula for comparing two means:

**n = 2[(Z_{α/2} + Z_β)² × σ²] / δ²**

Where Z_{α/2} = 1.96 (α = 0.05), Z_β = 0.84 (80% power), σ = pooled standard deviation, and δ = minimum detectable effect. **Cohen's d** standardizes effect sizes: d = 0.2 (small), 0.5 (medium), 0.8 (large).

For proportions: **n = [Z_{α/2}√(2p̄q̄) + Z_β√(p₁q₁ + p₂q₂)]² / (p₁ − p₂)²**

The relationship between power, sample size, and effect size is governed by: **Power = 1 − β = P(reject H₀ | H₁ true)**.

**Layer 2 — Basic Application**

Standard parameters: α = 0.05, power ≥ 80%, clinically meaningful effect size, dropout adjustment N_adj = n/(1 − dropout_rate). For unequal allocation (1:k ratio): n₁ = n×(1+k)/k. Event-driven trials require: **E = 4(Z_{α/2} + Z_β)² / (log HR)²** events.

**Layer 3 — Variations & Advanced**

**Adaptive sample size re-estimation** permits mid-trial variance recalculation—blinded (without unblinding) or unblinded (with alpha-spending adjustment). In tech experiments, Minimum Detectable Effect (MDE) replaces clinical difference, with sample size calculators universally applicable: n ∝ σ²/(MDE)².

---

### 3. Sequential testing and alpha-spending functions

**Layer 1 — Core Math**

Sequential methods control Type I error under repeated testing:

- **O'Brien-Fleming**: Z_c(k) = Z_OBF × √(K/k). Very conservative early (p < 0.001), preserves ~0.05 at final look.
- **Pocock**: Z_c(k) = constant. Equal boundaries (~0.016 for 5 looks).
- **Lan-DeMets α-spending** (O'Brien-Fleming type): **α(t) = 2[1 − Φ(Z_{α/2}/√t)]** where t = information fraction.

The **information fraction** t = n_current/N_total (sample-based) or events_current/Events_total (survival trials).

**Layer 2 — Basic Application**

| Method | Interim α | Final α | Sample Inflation |
|--------|-----------|---------|------------------|
| O'Brien-Fleming | Very small (p<0.001) | ~0.04 | 1-2% |
| Pocock | Equal (~0.016) | ~0.016 | 15-20% |
| Haybittle-Peto | p<0.001 | p<0.05 | Minimal |

Stopping boundaries: efficacy (exceeds upper), futility (below lower), continue (between).

**Layer 3 — Variations & Advanced**

**Always-valid inference** maintains coverage at all stopping times: P(τ ∈ CI_n for all n ≥ 1) ≥ 1−α. This enables continuous monitoring in production ML systems. **Pampallona-Tsiatis designs** handle simultaneous efficacy/futility testing. Tech implementation: "peeking" at experiments requires alpha-spending to maintain validity.

---

### 4. Multiplicity corrections (FWER and FDR)

**Layer 1 — Core Math**

Two error paradigms govern multiple testing:

- **FWER** (Family-Wise Error Rate): P(at least one false positive). For m independent tests: FWER = 1 − (1−α)^m ≈ mα.
- **FDR** (False Discovery Rate): E(false positives / total rejections | R > 0).

Key corrections:
- **Bonferroni**: α′_i = α/m (most conservative)
- **Holm step-down**: α′_(i) = α/(m−i+1), test from smallest p-value
- **Benjamini-Hochberg FDR**: Reject H_(1)...H_(k) where k = max{i : p_(i) ≤ (i/m)×q}

**Layer 2 — Basic Application**

| Method | Error Control | Use Case |
|--------|---------------|----------|
| Bonferroni | FWER (strict) | Few comparisons, confirmatory |
| Holm | FWER (less strict) | Few comparisons, more power |
| BH | FDR | Many comparisons, exploratory |
| BY | FDR (conservative) | Dependent tests |

Decision rule: FWER when any false positive is unacceptable (regulatory); FDR when proportion of false leads is acceptable (discovery).

**Layer 3 — Variations & Advanced**

**Gatekeeping procedures** enforce fixed sequence testing (primary → secondary → exploratory). **Graphical approaches** pass α between hypotheses upon rejection. **Storey's q-value** adaptively estimates π₀ for improved FDR power. In ML, these apply to feature selection, multi-metric A/B testing, and hyperparameter search with controlled false discovery.

---

### 5. Survival analysis (time-to-event methods)

**Layer 1 — Core Math**

The **survival function** S(t) = P(T > t) and **hazard function** h(t) = f(t)/S(t) are related by cumulative hazard H(t) = −ln(S(t)).

**Kaplan-Meier estimator** (product-limit):
**Ŝ(t) = ∏_{t_i ≤ t} (1 − d_i/n_i)**
where n_i = number at risk, d_i = events at time t_i.

**Cox proportional hazards**:
**h(t|X) = h₀(t) × exp(β₁X₁ + ... + β_pX_p)**
with hazard ratio HR = exp(β). The **log-rank test** compares survival curves: χ² = Σ[(O_j − E_j)²/E_j].

**Layer 2 — Basic Application**

HR interpretation: HR = 1 (no difference), HR = 0.5 (50% hazard reduction), HR = 2.0 (doubled hazard). Median survival occurs where S(t) = 0.5. The proportional hazards assumption (constant HR over time) must be verified via parallel log(−log(S(t))) curves.

**Layer 3 — Variations & Advanced**

**Parametric models** (Weibull, exponential, log-logistic) provide distribution-specific inference. **Competing risks** require cumulative incidence functions and Fine-Gray models. **Frailty models** add random effects for unobserved heterogeneity. Cross-domain: customer churn, time-to-conversion, subscription retention, component failure modeling.

---

### 6. Diagnostic accuracy metrics (ROC/AUC, likelihood ratios)

**Layer 1 — Core Math**

From the 2×2 confusion matrix:
- **Sensitivity** = TP/(TP+FN) = P(Test+ | Disease+)
- **Specificity** = TN/(TN+FP) = P(Test− | Disease−)
- **PPV** = TP/(TP+FP); **NPV** = TN/(TN+FN)

**Likelihood ratios** connect to Bayesian updating:
- **LR+** = Sensitivity/(1−Specificity)
- **LR−** = (1−Sensitivity)/Specificity
- **Post-test odds** = Pre-test odds × LR

**ROC curve** plots TPR vs FPR; **AUC** = P(X_positive > X_negative) for random samples. **Youden's J** = Sensitivity + Specificity − 1 identifies optimal cutoffs.

**Layer 2 — Basic Application**

| AUC | Interpretation |
|-----|----------------|
| 0.5 | Random (no discrimination) |
| 0.7-0.8 | Acceptable |
| 0.8-0.9 | Excellent |
| >0.9 | Outstanding |

Clinical heuristics: **SnNOut** (sensitive test rules OUT) and **SpPIn** (specific test rules IN). Sensitivity/specificity are prevalence-independent; PPV/NPV are prevalence-dependent.

**Layer 3 — Variations & Advanced**

**Partial AUC** focuses on clinically relevant ranges. **DeLong's method** provides confidence intervals for AUC comparison. Multi-class extensions use one-vs-rest ROC. Direct transfer to ML: classification evaluation, fraud detection (precision-recall for imbalanced data), safety-critical threshold selection.

---

### 7. Agreement and reliability metrics (Kappa, ICC, Krippendorff)

**Layer 1 — Core Math**

**Cohen's Kappa** (two raters, categorical):
**κ = (P_o − P_e)/(1 − P_e)**
where P_o = observed agreement (diagonal sum), P_e = expected agreement by chance.

**Intraclass Correlation Coefficient** (continuous, multiple raters):
**ICC(2,1) = (MSR − MSE)/[MSR + (k−1)MSE + (k/n)(MSC − MSE)]**
using ANOVA components (MSR = subjects, MSE = error, MSC = raters).

**Krippendorff's Alpha**:
**α = 1 − (D_o/D_e)**
handling missing data, any number of raters, and multiple data types (nominal, ordinal, interval, ratio).

**Bland-Altman limits of agreement**: LOA = d̄ ± 1.96 × SD(differences).

**Layer 2 — Basic Application**

| κ/ICC Range | Interpretation |
|-------------|----------------|
| < 0.50 | Poor |
| 0.50-0.75 | Moderate |
| 0.75-0.90 | Good |
| > 0.90 | Excellent |

Krippendorff thresholds: α ≥ 0.8 (reliable), 0.667 ≤ α < 0.8 (tentative), α < 0.667 (discard/re-annotate).

**Layer 3 — Variations & Advanced**

**Gwet's AC1/AC2** resist prevalence effects better than kappa. **Lin's concordance correlation** ρ_c combines precision and accuracy. **Dawid-Skene models** jointly estimate true labels and annotator quality. Critical for ML annotation pipelines, LLM-as-judge consistency measurement, and reproducibility assessment.

---

### 8. Bayesian evidence reasoning (likelihood ratios)

**Layer 1 — Core Math**

**Bayes' theorem** (odds form):
**Posterior Odds = Likelihood Ratio × Prior Odds**
**P(H|E)/P(¬H|E) = [P(E|H)/P(E|¬H)] × [P(H)/P(¬H)]**

The **likelihood ratio** LR = P(Evidence | H_prosecution)/P(Evidence | H_defense) quantifies evidential weight without requiring prior probability assessment.

Key principle: Always assess P(Evidence|Hypothesis), never P(Hypothesis|Evidence) directly.

**Layer 2 — Basic Application**

The **prosecutor's fallacy** confuses P(E|Innocence) with P(Innocence|E). Example: "1 in 400 match probability" ≠ "1 in 400 chance of innocence." The **defense attorney's fallacy** dismisses evidence by noting many could match, ignoring how rare characteristics update probabilities.

Verbal LR scales: LR = 1-10 (limited support), 10-100 (moderate), 100-1000 (moderately strong), >1000 (very strong).

**Layer 3 — Variations & Advanced**

**Bayesian networks** (DAGs) handle complex interdependent evidence. Hierarchical evidence levels (sub-source, source, activity propositions) require separate LR calculations. NIST and Royal Statistical Society guidance formalize forensic LR reporting. Transfers directly to: hypothesis evaluation in science, model comparison via Bayes factors, and AI claim verification.

---

### 9. Provenance verification (hash functions, Merkle trees)

**Layer 1 — Core Math**

**Cryptographic hash functions** H(m) → fixed-length digest with properties:
- **Collision resistance**: Computationally infeasible to find H(x) = H(y) where x ≠ y
- **Pre-image resistance**: Given H(x), infeasible to find x
- **Avalanche effect**: Small input change → drastically different output

**Merkle trees** enable O(log n) verification:
**Root = H(H(H(L₁)||H(L₂)) || H(H(L₃)||H(L₄)))**
where || denotes concatenation. Verification requires only the path from leaf to root (log n hashes).

**Layer 2 — Basic Application**

Standard algorithms: SHA-256 (256-bit, current standard), SHA-3 (latest). MD5/SHA-1 are deprecated for security applications. Process: create forensic image → generate hash at acquisition → verify hash at each transfer point → document chain of custody.

Legal precedent: FRE 902(13) and 902(14) recognize hash-authenticated evidence as self-authenticating.

**Layer 3 — Variations & Advanced**

**Blockchain-based audit trails** provide immutable ledgers with smart contracts for access control. **Timestamped verification** detects procrastinating auditors. Applications: Git version control, IPFS content addressing, supply chain provenance, AI training data lineage, model version control.

---

### 10. Decision theory under uncertainty

**Layer 1 — Core Math**

**Expected utility**:
**EU(a) = Σᵢ p(sᵢ) × U(a, sᵢ)**
maximizes expected payoff when probabilities are known.

Under uncertainty (unknown probabilities):
- **Maximin (Wald)**: Choose a* = argmax_a[min_s Payoff(a,s)] — maximize worst case
- **Minimax regret (Savage)**: Regret(a,s) = max_{a′} Payoff(a′,s) − Payoff(a,s); minimize maximum regret
- **Hurwicz**: H(a) = α×max_s + (1−α)×min_s, interpolating optimism-pessimism

**Layer 2 — Basic Application**

| Situation | Decision Rule |
|-----------|---------------|
| Known probabilities (risk) | Expected value/utility |
| Unknown probabilities (uncertainty) | Maximin, maximax, or minimax regret |
| Partial information | Bayesian updating |

**Dominance analysis** eliminates actions inferior under all criteria.

**Layer 3 — Variations & Advanced**

**Value of Perfect Information**: EVPI = E[max_a U(a,θ)] − max_a E[U(a,θ)] quantifies the worth of resolving uncertainty. **EVSI** (Expected Value of Sample Information) guides research prioritization. Regret minimization increasingly used for robust planning under deep uncertainty—requires only scenario analysis, not probability estimates. Applicable to AI deployment decisions, resource allocation, and risk management.

---

### 11. Multi-criteria decision making (AHP, TOPSIS)

**Layer 1 — Core Math**

**AHP (Analytic Hierarchy Process)** — Saaty (1980):

Pairwise comparison matrix A = [a_ij] where a_ij = importance of i vs j (1-9 scale), a_ji = 1/a_ij.

**Priority vector** w from eigenvector: A×w = λ_max×w

**Consistency Ratio**: CR = CI/RI where CI = (λ_max − n)/(n−1). Acceptable if **CR ≤ 0.10**.

**TOPSIS** ranks by closeness to ideal solution:
1. Normalize: r_ij = x_ij/√(Σ x_kj²)
2. Weight: v_ij = w_j × r_ij
3. Ideal solutions: A⁺ = {max for benefits, min for costs}; A⁻ = opposite
4. Distances: D_i⁺ = √[Σ(v_ij − v_j⁺)²]; D_i⁻ = √[Σ(v_ij − v_j⁻)²]
5. **Relative closeness**: C_i = D_i⁻/(D_i⁺ + D_i⁻) — rank by descending C_i

**Layer 2 — Basic Application**

AHP process: define hierarchy (goal → criteria → alternatives) → pairwise compare → calculate weights → check consistency → synthesize. TOPSIS requires no pairwise comparison, directly using normalized weighted values.

**Layer 3 — Variations & Advanced**

**Fuzzy TOPSIS** handles linguistic variables via triangular fuzzy numbers. **Sensitivity analysis** tests ranking stability under weight variations. **AHP-TOPSIS hybrid** uses AHP for weights, TOPSIS for ranking. Applications: vendor selection, technology evaluation, model selection with multiple metrics.

---

### 12. Queuing theory

**Layer 1 — Core Math**

**Little's Law** (universally applicable):
**L = λW**
where L = average number in system, λ = arrival rate, W = average time in system.

**M/M/1 queue** (Poisson arrivals, exponential service, 1 server):
- Utilization: ρ = λ/μ (must be < 1 for stability)
- Mean in system: E[L] = ρ/(1−ρ)
- Mean queue length: E[L_q] = ρ²/(1−ρ)
- Mean system time: E[W] = 1/(μ−λ)
- Mean wait time: E[W_q] = ρ/(μ−λ)

**Layer 2 — Basic Application**

**M/M/c** (c servers): ρ = λ/(cμ). **Erlang C** gives probability of waiting. Example: λ = 40/hr, W = 6 min → L = 40 × 0.1 = 4 customers average.

**PASTA property**: Poisson Arrivals See Time Averages—arriving customers observe equilibrium.

**Layer 3 — Variations & Advanced**

**Pollaczek-Khinchine formula** (M/G/1, general service): E[L_q] = (λ²σ² + ρ²)/[2(1−ρ)] where σ² = service variance. **Queuing networks** model computer systems, manufacturing, hospitals. Directly applicable to: API rate limiting, load balancing, inference server capacity planning.

---

### 13. Monte Carlo simulation

**Layer 1 — Core Math**

**Basic estimator**:
**μ̂_n = (1/N) Σᵢ f(Xᵢ)**

**Convergence** (Strong Law of Large Numbers): μ̂_n → μ as N → ∞.

**Standard error** = σ/√N, giving **convergence rate O(N^{−1/2})** independent of dimension.

**Central Limit Theorem**: √N(μ̂_n − μ) →_d N(0, σ²)

**Confidence interval**: CI = μ̂_n ± z_{α/2} × (s/√N)

**Layer 2 — Basic Application**

Probability estimation: P̂(E) = (1/N)Σ 1_E(Xᵢ). Integration: ∫f(x)dx ≈ (1/N)Σf(Xᵢ). Applications: financial derivatives (Black-Scholes), VaR estimation, engineering reliability, clinical outcome simulation.

**Layer 3 — Variations & Advanced**

**Variance reduction techniques**:
- **Antithetic variates**: Use [f(U) + f(1−U)]/2 with negatively correlated samples
- **Control variates**: μ̂_cv = μ̂ − β(Ȳ − E[Y]) using known-mean controls
- **Importance sampling**: E_p[f(X)] = E_q[f(X)×p(X)/q(X)] to emphasize important regions
- **Latin Hypercube**: Space-filling design for multi-dimensional inputs

**Quasi-Monte Carlo** (Halton, Sobol sequences) achieves O(N^{−1}) for smooth integrands. Universal tool for uncertainty propagation through complex systems.

---

### 14. Reliability theory (Weibull, hazard functions)

**Layer 1 — Core Math**

**Weibull distribution** (foundational reliability model):
- PDF: f(t) = (β/η)(t/η)^{β−1} exp(−(t/η)^β)
- Reliability: R(t) = exp(−(t/η)^β)
- Hazard: h(t) = (β/η)(t/η)^{β−1}

**Shape parameter β** determines failure behavior (**bathtub curve**):
- β < 1: Decreasing hazard (infant mortality)
- β = 1: Constant hazard (exponential, random failures)
- β > 1: Increasing hazard (wear-out)

**System reliability**:
- Series: R_sys = ∏R_i (all must work)
- Parallel: R_sys = 1 − ∏(1−R_i) (any must work)

**Layer 2 — Basic Application**

**MTTF** = ηΓ(1 + 1/β). Parameter estimation via MLE (preferred for censored data) or least squares on Weibull probability plots. Tools: Minitab, JMP, ReliaSoft Weibull++.

**Layer 3 — Variations & Advanced**

**Mixed Weibull** for competing failure modes. **Accelerated Life Testing** (Arrhenius, Eyring models) extrapolates from stress conditions. **Physics-of-failure models** combine with degradation analysis. Applicable to: ML model performance degradation, component reliability, hardware failure prediction.

---

### 15. Fault tree analysis and FMEA

**Layer 1 — Core Math**

**Boolean algebra** models failure propagation:
- OR gate: T = A ∪ B (either causes failure)
- AND gate: T = A ∩ B (both required for failure)

**Minimal cut sets** (MCS): smallest combinations causing top event.
**Top event probability**: P(TOP) ≈ Σ P(MCS_i) for rare events, where P(MCS_i) = ∏_j P(E_j).

**Importance measures**:
- **Birnbaum**: ∂P(TOP)/∂P(E_i) — sensitivity
- **Fussell-Vesely**: Contribution of event to top probability

**FMEA Risk Priority Number**:
**RPN = Severity × Occurrence × Detection** (each 1-10)

**Layer 2 — Basic Application**

FTA process: define top event → identify causes → decompose with AND/OR gates → compute minimal cut sets (MOCUS algorithm) → calculate probabilities. FMEA: list failure modes → assess S/O/D → calculate RPN → prioritize corrective actions.

**Layer 3 — Variations & Advanced**

**Dynamic fault trees** include sequence dependencies. **Common cause failure** modeling via beta-factor method. **Fuzzy FMEA** addresses RPN limitations. AIAG/VDA FMEA (2019) replaces RPN with **Action Priority (AP)**. Applications: AI agent failure mode analysis, safety case construction, risk decomposition.

---

### 16. Metamorphic testing

**Layer 1 — Core Math**

**Metamorphic relations (MR)** describe necessary relationships between inputs and outputs:
For function f: X → Y, relation R describes: R((x₁, f(x₁)), (x₂, f(x₂)), ..., (x_n, f(x_n)))

**Common mathematical MRs**:
- Commutativity: f(a,b) = f(b,a)
- Symmetry: |path(a,b)| = |path(b,a)|
- Scaling: f(kx) = k×f(x) for linear functions
- Subset: filter(search(q)) ⊆ search(q)

The **oracle problem**—not knowing expected output—is bypassed by checking relation consistency rather than absolute correctness.

**Layer 2 — Basic Application**

Process: identify MRs from domain knowledge → generate source test cases → apply transformations for follow-up cases → execute both → verify MR satisfaction → violation indicates fault. Highly effective for search engines, compilers, ML systems, numerical software.

**Layer 3 — Variations & Advanced**

**Automated MR identification** via machine learning. **Metamorphic fault tolerance** applies MRs at runtime. **Semi-proving** integrates with formal verification. Google acquired GraphicsFuzz (2018) for metamorphic graphics testing. Critical for AI verification where ground truth is unavailable.

---

### 17. Coverage metrics (MC/DC, mutation testing)

**Layer 1 — Core Math**

**Modified Condition/Decision Coverage (MC/DC)** per DO-178C:
Each condition must independently affect the decision outcome. For condition C in decision D: test cases T₁, T₂ exist where C differs, all other conditions match, and decision outcome differs.

**Minimum tests**: n+1 for n conditions (vs 2^n for full combinatorial).

**Mutation testing score**:
**Score = Killed Mutants / (Total − Equivalent Mutants) × 100**

**Coupling hypothesis**: Simple mutations couple to reveal complex faults.

**Layer 2 — Basic Application**

| Standard | Required Coverage |
|----------|-------------------|
| DO-178C Level A | MC/DC (100%) |
| ISO 26262 ASIL D | MC/DC |
| IEC 61508 SIL 4 | MC/DC recommended |

Tools: LDRA, VectorCAST (MC/DC); Stryker, PIT (mutation testing).

**Layer 3 — Variations & Advanced**

**Object code verification** at machine level. **Higher-order mutation** applies multiple simultaneous changes. **Mutant sampling** enables statistical approaches for large codebases. Applicable to: neural network decision coverage, policy path coverage in RL agents, state space exploration metrics.

---

### 18. Calibration metrics (ECE, Brier score)

**Layer 1 — Core Math**

**Expected Calibration Error**:
**ECE = Σ_{m=1}^M (|B_m|/n) × |acc(B_m) − conf(B_m)|**
where B_m = samples in bin m, acc = accuracy, conf = mean confidence.

**Brier score decomposition** (Murphy, 1973):
**BS = Reliability − Resolution + Uncertainty**
- Reliability: mean squared deviation of predictions from observed frequencies
- Resolution: variance of conditional outcomes (higher is better)
- Uncertainty: inherent data randomness

**Temperature scaling** (post-hoc calibration):
**q_i = exp(z_i/T) / Σ_j exp(z_j/T)**
where T > 1 softens probabilities, T < 1 sharpens them.

**Layer 2 — Basic Application**

**Reliability diagrams** plot observed accuracy vs predicted confidence; perfect calibration = diagonal. Tools: sklearn.calibration, torchmetrics.CalibrationError, MAPIE. Standard benchmarks: CIFAR, ImageNet with various architectures.

**Layer 3 — Variations & Advanced**

**SmoothECE** (Błasiok et al., 2023) replaces hard binning with kernel smoothing. **Class-wise ECE** assesses per-class calibration. **Dirichlet calibration** (Kull et al., NeurIPS 2019) extends to multiclass. **Full-ECE** for LLMs evaluates token-level calibration. Critical for trustworthy AI confidence estimates.

---

### 19. Ranking and preference learning (Bradley-Terry, Elo)

**Layer 1 — Core Math**

**Bradley-Terry model**:
**P(i ≻ j) = π_i/(π_i + π_j) = 1/(1 + exp(−(x_i − x_j)))**
where π_i = exp(x_i) represents strength. MLE estimation from pairwise comparisons.

**Elo rating system**:
- Expected score: E_A = 1/(1 + 10^{(R_B−R_A)/400})
- Update: R′_A = R_A + K(S_A − E_A)
where K = learning rate (16-32), 400 = scale factor (400-point difference ≈ 10:1 expected ratio).

**Plackett-Luce** (full rankings):
**P(y₁ ≻ ... ≻ y_N) = ∏_i [p_{y_i} / Σ_{k≥i} p_{y_k}]**

**TrueSkill** models skill as Gaussian s_i ~ N(μ_i, σ_i²) with Bayesian updating.

**Layer 2 — Basic Application**

Chatbot Arena uses Bradley-Terry with MLE, ties as 0.5-0.5, bootstrap CIs (100+ iterations). Libraries: BradleyTerryScalable (R), choix (Python), leaderbot (ICLR 2025).

**Layer 3 — Variations & Advanced**

**Factored Tie Model** (Ameli et al., ICLR 2025) adds pair-specific tie parameters. **Crowd-BT** accounts for annotator reliability. **Extended Bradley-Terry** jointly ranks models, tools, and frameworks. Foundation of modern LLM evaluation (LMSYS Chatbot Arena, MT-Bench).

---

### 20. Conformal prediction (distribution-free uncertainty)

**Layer 1 — Core Math**

**Coverage guarantee** (finite-sample, distribution-free):
**P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 − α**

**Split conformal algorithm**:
1. Compute nonconformity scores on calibration set: s_i = s(X_i, Y_i)
2. Set threshold: q̂ = Quantile_{(1−α)(1+1/n)}({s₁, ..., s_n})
3. Prediction set: Ĉ(X_new) = {y : s(X_new, y) ≤ q̂}

**Common scores**:
- Regression: s(x,y) = |y − f̂(x)|
- Classification: s(x,y) = 1 − p̂_y(x)

**Layer 2 — Basic Application**

Metrics: empirical coverage, average set size (efficiency). Libraries: MAPIE (Python), conformal (R). Exchangeability assumption required (IID or exchangeable data).

**Layer 3 — Variations & Advanced**

**Conformalized Bayesian inference** combines Bayesian posteriors with conformal guarantees. **Conformal risk control** generalizes beyond coverage to arbitrary loss functions. **Adaptive conformal prediction** handles distribution shift with time-varying coverage. **Conditional conformal** provides group-specific guarantees. **VRCP** offers verifiable robustness against adversarial attacks. The only uncertainty quantification method with **provable finite-sample guarantees** regardless of model or distribution.

---

## Framework synthesis: the verification advantage

These 20 methodologies share structural properties that make verification fundamentally tractable:

**Dimension reduction**: Each methodology reduces high-dimensional outputs to scalar metrics (AUC, κ, ECE, coverage rate) or binary decisions (reject/fail-to-reject, pass/fail).

**Compositionality**: Methods combine hierarchically—FTA decomposes system failure into component failures; sequential testing builds on single-test hypothesis frameworks; Merkle trees reduce O(n) verification to O(log n).

**Exchangeability of verification targets**: The same mathematical framework (e.g., likelihood ratios, coverage metrics, agreement coefficients) applies across domains—forensic evidence evaluation uses the same Bayesian structure as model comparison.

**Known error rates**: Unlike generation (where quality is often subjective), verification methods have quantifiable error rates: Type I/II errors, false positive/negative rates, calibration error bounds.

This supports the core thesis: **rule space ≪ action space**. An AI agent may take arbitrary actions, but verifying those actions against specifications uses a compact set of mathematical primitives. The 20 methodologies above constitute a near-complete toolkit for that verification task.

---

## Key references by domain

**Clinical/Statistical**: FDA Non-Inferiority Guidance (2016); ICH E9/E10; Cohen (1988) Power Analysis; Lan & DeMets (1983); Benjamini & Hochberg (1995); Kaplan & Meier (1958); Cox (1972)

**Engineering**: MIL-HDBK-338; NUREG-0492 Fault Tree Handbook; NASA/TM-2001-210876 MC/DC Tutorial; DO-178C; Chen et al. (2018) Metamorphic Testing Survey

**Forensics/OR**: NIST/Lund & Iyer (2017) Likelihood Ratios; Daubert v. Merrell Dow (1993); Saaty (1980) AHP; Little (1961)

**ML Evaluation**: Guo et al. (ICML 2017) Calibration; Bradley & Terry (1952); Angelopoulos & Bates (2022) Conformal Prediction; Shi et al. (2024) Contamination Detection; Chiang et al. (2024) Chatbot Arena