# Darkness Visible: Reading the Exception Handler of a Language Model

Code, data, and paper for "Darkness Visible: Reading the Exception Handler of a Language Model" by Peter Balogh.

## Overview

The final MLP of GPT-2 Small exhibits a fully legible routing program — 27 named neurons organized into a three-tier exception handler — while the knowledge it routes remains entangled across ~3,040 residual neurons.

## Structure

- `src/` — Core library: `transparent_model.py` (TransparentGPT2 wrapper), `architecture.py`, `evaluate.py`
- `code/` — Experiment scripts: bootstrap crossover, knowledge extraction, garden-path surprisal, cross-layer analysis, tuned lens
- `data/` — Result JSON files for all experiments
- `paper/` — LaTeX source, figures, references

## Key Results

- **Consensus-exception crossover** at 4-5/7 consensus neurons (Table 4, bootstrap 95% CIs all exclude zero)
- **Knowledge neurons are routing infrastructure**, not fact storage (36.5× enrichment overlap with routing circuit)
- **Reversed garden-path effect** — GPT-2 uses verb subcategorization immediately (Wilcoxon W=12, p=0.018)
- **Terminal crystallization** — routing structure appears only at L11 (max Jaccard 0.384 at any other layer vs. 0.998 at L11)

## Related Papers

- Balogh (2026), "The Discrete Charm of the MLP" — arXiv:2603.10985 (predecessor: identifies the consensus/exception architecture)

## Requirements

```
torch
transformers
scipy
numpy
```

## Citation

```bibtex
@article{balogh2026darkness,
  title={Darkness Visible: Reading the Exception Handler of a Language Model},
  author={Balogh, Peter},
  year={2026}
}
```
