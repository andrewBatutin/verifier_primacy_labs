# Detecting Hidden LLM Model Drift with MMD²

*A statistical approach to catching silent API changes before they break your production systems*

---

**TL;DR:** LLM providers silently update models behind stable API endpoints. This can break your production systems without warning. Maximum Mean Discrepancy (MMD²) provides a mathematically rigorous way to detect these hidden changes using only model outputs—no internal access required.

---

## The Problem: Silent Model Drift

You've built a production system on top of `gemini-2.5-pro` or `gemini-3-flash-preview`. Your evaluations pass. Your prompts are tuned. Everything works.

Then one day, your system starts behaving differently. Outputs are longer. The tone has shifted. Edge cases that worked before now fail. But you didn't change anything.

**What happened?** The model behind the API endpoint was silently updated.

### Why This Matters

- **Production breakage**: Carefully tuned prompts may no longer work as expected
- **Invalid evaluations**: Benchmark results from last month may not reflect current behavior
- **Cost surprises**: Verbose outputs increase token costs
- **Compliance risks**: Behavioral changes may violate regulatory requirements

### The Challenge

You can't see inside the model. You only have access to its outputs. How do you detect that something has changed?

The answer: **compare output distributions statistically**.

---

## The Solution: MMD² (Maximum Mean Discrepancy)

MMD² is a statistical test that answers the question:

> *"Do these two sets of samples come from the same distribution?"*

For LLM drift detection, this becomes:

> *"Are today's model outputs statistically distinguishable from last week's outputs?"*

If yes → the model has drifted. If no → the model appears stable.

---

## The Math Behind MMD²

### Core Idea

Compare two distributions P and Q by mapping their samples to a high-dimensional feature space and measuring the distance between their means.

### The Formula

$$\text{MMD}^2(P, Q) = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]$$

Where:
- x, x' ~ P (samples from distribution P)
- y, y' ~ Q (samples from distribution Q)
- k(·, ·) is a kernel function

### Intuition

Think of it as measuring three things:
1. **How similar are samples within P to each other?** → E[k(x, x')]
2. **How similar are samples within Q to each other?** → E[k(y, y')]
3. **How similar are samples across P and Q?** → E[k(x, y)]

If P = Q, all three terms should be equal, making MMD² = 0.

If P ≠ Q, the cross-term will be smaller (less similar), making MMD² > 0.

---

## Why Kernels? The High-Dimensional Challenge

LLM outputs, when embedded, live in **384-1024 dimensional space**. Direct comparison in such high dimensions fails due to the curse of dimensionality—distances become meaningless as everything appears equally far apart.

### The Kernel Trick

Kernels implicitly map data to an even higher (potentially infinite) dimensional space called a **Reproducing Kernel Hilbert Space (RKHS)** where comparison becomes tractable.

### RBF Kernel (Radial Basis Function)

$$k(x, y) = \exp\left(-\gamma \|x - y\|^2\right)$$

- Returns 1 when x = y (identical)
- Approaches 0 as x and y diverge
- γ controls sensitivity to distance

The RBF kernel is a **universal approximator**—it can detect any difference between distributions given enough samples.

---

## Beyond Text: Multimodal Drift Detection

MMD² works with **any embedding space**. The same math applies to:

| Modality | Embedding Model | Use Case |
|----------|-----------------|----------|
| Text | sentence-transformers, E5 | LLM output drift |
| Images | CLIP, DINOv2 | Image generation drift |
| Audio | wav2vec, Whisper | Speech synthesis drift |
| Code | CodeBERT, StarCoder | Code generation drift |

If you can embed it, you can detect drift in it.

---

## Why MMD² Beats the Alternatives

| Method | Weakness | MMD² Advantage |
|--------|----------|----------------|
| **KL Divergence** | Requires density estimation; fails catastrophically in high dimensions | Works directly on samples |
| **Rule-based checks** | Brittle; misses subtle drift; requires constant maintenance | Captures any distribution-wide shift |
| **LLM-as-judge** | Expensive; adds noise; subjective; itself subject to drift | Deterministic, cheap, objective |
| **Cosine similarity** | Single-point comparison; misses distributional changes | Captures full distribution shape |
| **Output length checks** | Only catches one symptom | Catches style, tone, verbosity, accuracy shifts |

### Key Properties of MMD²

- **Noise-stable**: Permutation testing provides statistical rigor
- **High-dimensional native**: Designed for exactly this setting
- **Captures variety**: Style drift, verbosity changes, tone shifts
- **Low false alarms**: Null hypothesis testing prevents over-detection
- **Computationally efficient**: O(n²) for n samples—fast enough for real-time monitoring

---

## Experiment Setup

I implemented MMD² drift detection and tested it on real LLM outputs from Gemini models.

Responses were collected from three different Gemini models:
- **gemini-3-flash-preview**: Latest preview model (fast, capable)
- **gemini-2.5-pro**: Current flagship model (slower, more capable)
- **gemini-2.0-flash-lite**: Lightweight model (fastest, most concise)

### Two Prompt Types

To demonstrate MMD² works across different application architectures, I tested with **two prompt styles**:

#### 1. INSURANCE_CHAT (Conversational)
Direct chat-based conversation without structured context—simulating a user chatting with an AI assistant.

**Example prompt:**
```
I have a home insurance policy with a $1,000 deductible. My roof was damaged
in a storm and repairs cost $3,500. How much will I receive from the claim?
```

#### 2. RAG_CHAT (RAG Application)
Structured Query + Context format—simulating a RAG (Retrieval-Augmented Generation) application.

**Example prompt:**
```
Query: How much will I pay out of pocket if my roof damage costs $8,500?
Context: Policy Section 4.2 - Deductibles
Your dwelling coverage deductible is $1,500 per occurrence. For wind/hail damage,
a separate deductible of 2% of Coverage A limit applies. Your Coverage A limit
is $250,000. The higher of the two deductibles applies to wind/hail claims.
```

### Dataset Summary

Each model was queried with 25 prompts per type, with 10 samples per prompt:
- **250 outputs per model per prompt type**
- **750 outputs per prompt type** (3 models × 250)
- **1,500 total outputs** (750 × 2 prompt types)

### Embedding Model

I used `intfloat/e5-large-v2`, a high-quality 1024-dimensional text embedding model. The higher dimensionality captures more nuanced semantic differences than smaller models.

---

## Why We Need Statistical Testing

### The Problem with Raw MMD²

Even if two samples come from the **exact same distribution**, MMD² will almost never be exactly zero due to sampling variance. This creates a critical question:

> *"Is this MMD² value meaningful, or just random noise?"*

### Null Hypothesis Testing

We frame this as a hypothesis test:

- **H₀ (Null)**: Both sample sets come from the same distribution
- **H₁ (Alternative)**: The distributions are different (model has drifted)

We need to compute: *"How likely is this MMD² value if H₀ is true?"*

### Permutation Test (Non-parametric)

The permutation test is ideal here because:
- No assumptions about distribution shape
- Works in high dimensions
- Provides exact p-values

**The procedure:**
1. Pool all samples together
2. Randomly shuffle (permute) the labels
3. Recompute MMD² with shuffled labels
4. Repeat 1000 times → builds null distribution
5. P-value = fraction of null MMD² values ≥ observed MMD²

### Interpreting Results

| P-value | Interpretation |
|---------|-----------------|
| p < 0.05 | Reject H₀ → Drift detected with 95% confidence |
| p < 0.01 | Strong evidence of drift |
| p ≥ 0.05 | Cannot reject H₀ → No evidence of drift |

**Low p-value + high MMD² = strong signal of model drift**

---

## Results: Model Comparisons

### INSURANCE_CHAT (Conversational Prompts)

| Comparison | MMD² | P-value | Result |
|------------|------|---------|--------|
| G3 Flash vs G2.5 Pro | 0.006609 | p < 0.001 | DRIFT |
| G3 Flash vs G2.0 Lite | 0.012641 | p < 0.001 | DRIFT |
| G2.5 Pro vs G2.0 Lite | 0.010224 | p < 0.001 | DRIFT |

### RAG_CHAT (Query + Context Prompts)

| Comparison | MMD² | P-value | Result |
|------------|------|---------|--------|
| G3 Flash vs G2.5 Pro | 0.005577 | p < 0.001 | DRIFT |
| G3 Flash vs G2.0 Lite | 0.007281 | p < 0.001 | DRIFT |
| G2.5 Pro vs G2.0 Lite | 0.006808 | p < 0.001 | DRIFT |

All model pairs are statistically distinguishable. MMD² successfully identifies that different models produce different output distributions.

---

## Validation: Does MMD² Actually Work?

### Test 1: Truncation (Should Detect Drift)

I took outputs from one model and artificially truncated each response to 200 characters. If MMD² works, it should detect this obvious change.

**Results:**
- Original avg length: 1,230 chars
- Truncated avg length: 200 chars
- **Observed MMD²: 0.041088**
- **P-value: p < 0.001**
- **Result: ✓ SUCCESS** - Truncation drift detected!

### Test 2: Self-Split (Should NOT Detect Drift)

I took samples from one model, shuffled them, and split 50/50 into two groups. Since both halves come from the same distribution, MMD² should find no significant difference.

**Results:**
- Half A: 125 samples
- Half B: 125 samples
- **Observed MMD²: -0.000123** (essentially zero)
- **P-value: p = 0.488**
- **Result: ✓ SUCCESS** - No false positive!

These validation tests confirm MMD² works correctly:
- Detects real drift (truncation)
- Doesn't flag same-distribution samples (self-split)

---

## Effect Sizes: Beyond P-Values

P-values alone can be misleading. **Effect size** tells us how far the observed MMD² is from the null distribution in terms of standard deviations (σ):

$$\text{Effect Size} = \frac{\text{Observed MMD}^2 - \mu_{\text{null}}}{\sigma_{\text{null}}}$$

This gives us intuitive interpretation:
- **~0σ**: No difference (within noise)
- **2-3σ**: Moderate difference
- **5-10σ**: Large difference
- **>10σ**: Extreme difference (very confident detection)

### Effect Size Summary (All Experiments)

| Category | Comparison | MMD² | P-value | Effect (σ) | Result |
|----------|------------|------|---------|------------|--------|
| CHAT Models | G3 Flash vs G2.5 Pro | 0.006609 | p < 0.001 | +12.2σ | DRIFT |
| CHAT Models | G3 Flash vs G2.0 Lite | 0.012641 | p < 0.001 | +23.6σ | DRIFT |
| CHAT Models | G2.5 Pro vs G2.0 Lite | 0.010224 | p < 0.001 | +19.2σ | DRIFT |
| RAG Models | G3 Flash vs G2.5 Pro | 0.005577 | p < 0.001 | +7.3σ | DRIFT |
| RAG Models | G3 Flash vs G2.0 Lite | 0.007281 | p < 0.001 | +10.4σ | DRIFT |
| RAG Models | G2.5 Pro vs G2.0 Lite | 0.006808 | p < 0.001 | +9.1σ | DRIFT |
| Validation | Truncation (should detect) | 0.041088 | p < 0.001 | +59.7σ | ✓ Correct |
| Validation | Self-split (should NOT detect) | -0.000123 | p = 0.488 | -0.1σ | ✓ Correct |

**Interpretation:**
- Model comparisons show 7-24σ effect sizes — these are huge, unambiguous differences
- Self-split shows ~0σ — correctly identifies same distribution
- Truncation shows 60σ — extreme signal for obvious artificial change

---

## Embedding Model Robustness Check

Do the results depend on embedding model choice? I re-ran the RAG comparisons with a different embedding model: `all-MiniLM-L6-v2` (384 dimensions) instead of `e5-large-v2` (1024 dimensions).

| Pair | E5-large (1024d) | MiniLM (384d) | Match? |
|------|------------------|---------------|--------|
| G3 Flash vs G2.5 Pro | DRIFT p < 0.001 | No drift p = 0.070 | ✗ |
| G3 Flash vs G2.0 Lite | DRIFT p < 0.001 | DRIFT p = 0.018 | ✓ |
| G2.5 Pro vs G2.0 Lite | DRIFT p < 0.001 | DRIFT p = 0.021 | ✓ |

**Finding:** Higher-dimensional embeddings (E5-large) are more sensitive to distributional differences. MiniLM missed one comparison that E5-large caught. Use the best embedding model you can afford.

---

## Canary Deployment: Using This in Production

Here's how to implement MMD² drift detection in a real production system.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Canary Prompts │ ──> │   LLM API       │ ──> │  Embed + Store  │
│  (scheduled)    │     │   (Gemini)      │     │  Responses      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         v
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Alert if       │ <── │  MMD² Test      │ <── │  Compare to     │
│  p < threshold  │     │  (permutation)  │     │  Baseline       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Implementation Steps

1. **Collect Baseline**
   - Run canary prompts on your model (e.g., `gemini-2.5-pro`)
   - Generate 50-100+ responses per prompt
   - Embed and store as your baseline distribution

2. **Schedule Canary Runs**
   - Run canary prompts daily/weekly
   - Collect fresh response samples
   - Embed with the same model (critical: keep embedder consistent)

3. **Run MMD² Test**
   - Compare new batch to baseline
   - Use permutation test for p-value
   - Log results for trending

4. **Alert on Drift**
   - If p < 0.05: potential drift, investigate
   - If p < 0.01: likely drift, high priority
   - Update baseline if change is intentional (new model version)

### Practical Recommendations

| Parameter | Recommendation | Rationale |
|-----------|----------------|-----------|
| Sample size | 100-250 per batch | Strong statistical power |
| Prompts | 10-25 diverse prompts | Cover different response types |
| Samples per prompt | 5-10 | Capture response variance |
| Frequency | Daily to weekly | Depends on criticality |
| Threshold | p < 0.05 (standard) | Adjust based on tolerance |
| Embedding model | Keep consistent | Changing embedder invalidates baseline |

### Cost Optimization

- **Reuse embeddings**: Store embeddings, not just responses
- **Batch API calls**: Reduce latency with batched requests
- **Use `gemini-2.0-flash-lite`** for canary prompts: Fastest & cheapest
- **Cache baselines**: Don't recompute baseline embeddings
- **Subsample for gamma**: Median heuristic doesn't need all pairs

---

## Conclusion

MMD² provides a **mathematically rigorous**, **computationally efficient**, and **practically useful** method for detecting hidden LLM model drift.

### Key Takeaways

1. **Silent model updates are real** and can break production systems
2. **MMD² works in high dimensions** where other methods fail
3. **Permutation testing** provides statistical rigor without assumptions
4. **Effect sizes matter** — our experiments showed 7-24σ for model differences vs ~0σ for same-model splits
5. **The method is validated**: Detected artificial truncation drift, no false positives on self-split
6. **Implementation is straightforward** with standard libraries

### When to Use MMD²

- Monitoring production LLM APIs (Gemini, GPT, Claude)
- Validating model updates before deployment
- Comparing different model versions or providers
- Detecting training data drift in fine-tuned models

### Limitations

- Requires sufficient samples (n > 50) for reliable results
- Detects *that* drift occurred, not *what* changed
- Embedding model choice affects sensitivity
- Computational cost grows quadratically with sample size

---

*The complete code for this article is available at [github.com/abatutin/verifier_primacy](https://github.com/abatutin/verifier_primacy). For more on verification methodologies, see the Verifier Primacy project.*
