# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Verifier Primacy is a research project exploring verification methodologies from clinical trials, forensics, QA, and operations research applied to AI agent evaluation. The core thesis: verification is fundamentally easier than generation because the rule space is vastly smaller than the action space.

## Commands

```bash
# Install dependencies (uses UV package manager)
uv sync

# Run verification primitives examples
python src/verification_primitives_examples.py

# Run notebooks
jupyter notebook notebooks/mmd_test.ipynb
jupyter notebook notebooks/kl_divergence_playground.ipynb

# Linting and formatting
ruff check .
ruff format .

# Testing
pytest
```

## Architecture

### Core Components

- **`src/verification_primitives_examples.py`** - Main implementation with 16 verification methodologies, each containing three layers: Core Mathematics, Basic Application with NumPy examples, and Advanced Extensions

- **`notebooks/`** - Active experiments
  - `mmd_test.ipynb` - Maximum Mean Discrepancy for LLM model drift detection (queries Gemini models)
  - `kl_divergence_playground.ipynb` - KL Divergence exploration for distribution shift measurement

- **`docs/`** - Research documentation including the 20-methodology framework and AI evals ecosystem references

- **`data/mmd_test/`** - Experiment outputs (JSON model responses, PNG visualizations)

### The 20 Verification Methodologies

Drawn from 5 domains: Clinical Trials, Forensics, Operations Research, Engineering, and ML/Software QA. Key implementations include:
- Diagnostic accuracy (ROC/AUC, sensitivity/specificity)
- Agreement metrics (Cohen's κ, ICC, Krippendorff α)
- Hypothesis testing (Z/t-tests, TOST)
- Calibration metrics (ECE, Brier decomposition)
- Conformal prediction (distribution-free uncertainty)
- Sequential testing (alpha-spending, O'Brien-Fleming)
- Metamorphic testing (oracle-free verification)
- Bayesian evidence reasoning (likelihood ratios)

## Environment

Requires Python 3.12+. Uses `.env` file for `GEMINI_API_KEY` when running notebook experiments.
