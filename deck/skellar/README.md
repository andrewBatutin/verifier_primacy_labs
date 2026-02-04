# Verifier Primacy Presentation

## Files

- `verifier_primacy.qmd` — Quarto markdown source
- `custom.scss` — Custom theme styling (purple/teal branding)

## Rendering

### Option 1: Quarto CLI (recommended)

```bash
# Install Quarto from https://quarto.org/docs/get-started/

# Render to HTML (reveal.js)
quarto render verifier_primacy.qmd

# Preview with live reload
quarto preview verifier_primacy.qmd
```

### Option 2: RStudio

1. Open `verifier_primacy.qmd` in RStudio
2. Click "Render" button

### Option 3: VS Code

1. Install Quarto extension
2. Open `verifier_primacy.qmd`
3. Cmd/Ctrl + Shift + K to render

## Output

The rendered presentation will be `verifier_primacy.html` — a fully interactive reveal.js presentation.

### Features
- Press `S` for speaker notes
- Press `O` for overview mode
- Press `F` for fullscreen
- Arrow keys or swipe to navigate
- Math equations rendered with MathJax

## Customization

Edit `custom.scss` to adjust:
- Colors (purple/teal theme)
- Fonts
- Callout box styles
- Stack diagram layers

## Exporting

```bash
# Export to PDF (requires Chrome)
quarto render verifier_primacy.qmd --to pdf

# Export to PowerPoint
quarto render verifier_primacy.qmd --to pptx
```

## Content Structure

1. **Hook** — The guaranteed executive failure
2. **Stochasticity Paradox** — Why AI can't be reliable (math)
3. **RAG Case Study** — Insurance failure with $120k consequence
4. **RAG Methodology** — Hamel Husain framework
5. **Agentic Case Study** — Travel agent with $20.5k exposure
6. **Agentic Methodology** — Hamel Husain framework
7. **Verifier Primacy** — Core insight (rule space << action space)
8. **Verification Stack** — 6 layers with math foundations
9. **Tale of Two Executives** — The choice
10. **Close** — Stop praying, start verifying

## Math Equations Included

- Entropy/generalization tradeoff
- Information asymmetry (H(Query) << H(Document))
- Embedding geometry curse
- Combinatorial explosion
- Constraint satisfaction
- NLI classification
- Deontic logic for domain rules
- KL divergence for judge alignment
- Cohen's kappa for calibration
- Cost-sensitive escalation threshold
