# Verifier Primacy

Verification methodologies from clinical trials, forensics, QA, and operations research applied to AI agent evaluation.

**Core thesis**: Verification is fundamentally easier than generation because the rule space is vastly smaller than the action space.

## MMD² Drift Detection

Detect hidden LLM model drift using Maximum Mean Discrepancy (MMD²). See [notebooks/mmd_article.ipynb](notebooks/mmd_article.ipynb).

### Key Results

| Experiment | Effect Size | Result |
|------------|-------------|--------|
| Self-split (same model) | ~0σ | No drift (correct) |
| Different models | 7-24σ | Drift detected |
| Truncated outputs | 60σ | Drift detected |

## Quick Start

```bash
# Install dependencies
uv sync

# Run MMD notebook
uv run jupyter notebook notebooks/mmd_article.ipynb

# Run verification primitives examples
uv run python src/verification_primitives_examples.py
```

## Project Structure

```
├── notebooks/
│   ├── mmd_article.ipynb      # MMD² drift detection (main)
│   └── kl_divergence_playground.ipynb
├── src/
│   └── verification_primitives_examples.py  # 16 verification methodologies
├── data/
│   └── mmd_test/              # Experiment outputs (JSON, PNG)
└── docs/                      # Research documentation
```

## Verification Methodologies

Drawn from 5 domains: Clinical Trials, Forensics, Operations Research, Engineering, and ML/Software QA.

- Diagnostic accuracy (ROC/AUC, sensitivity/specificity)
- Agreement metrics (Cohen's κ, ICC, Krippendorff α)
- Hypothesis testing (Z/t-tests, TOST)
- Calibration metrics (ECE, Brier decomposition)
- Conformal prediction (distribution-free uncertainty)
- Sequential testing (alpha-spending, O'Brien-Fleming)
- Metamorphic testing (oracle-free verification)
- Bayesian evidence reasoning (likelihood ratios)

## License

MIT License - see [LICENSE](LICENSE)

## Author

Andriy Batutin
