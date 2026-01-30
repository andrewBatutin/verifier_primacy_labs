# MMD² Drift Detection Notebook

Detecting hidden LLM model drift using Maximum Mean Discrepancy (MMD²).

## Quick Start

```bash
uv sync
uv run jupyter notebook notebooks/mmd_article.ipynb
```

## What It Does

Compares output distributions from different Gemini models to detect if they're statistically distinguishable. Uses semantic embeddings + permutation testing.

## Key Results

| Experiment | Effect Size | Result |
|------------|-------------|--------|
| Self-split (same model) | ~0σ | No drift (correct) |
| Different models | 7-24σ | Drift detected |
| Truncated outputs | 60σ | Drift detected |

## Dataset

- **1,500 total outputs** from 3 Gemini models
- **Two prompt types**: Conversational (CHAT) and RAG-style (Query+Context)
- **250 samples per model per prompt type**

## Embedding Models Tested

- `intfloat/e5-large-v2` (1024-dim) - primary
- `all-MiniLM-L6-v2` (384-dim) - robustness check

## Files

- `mmd_article.ipynb` - Main notebook with full analysis
- `../data/mmd_test/experiment_outputs/` - Raw JSON data from Gemini API
- `../data/mmd_test/*.png` - Generated visualizations
