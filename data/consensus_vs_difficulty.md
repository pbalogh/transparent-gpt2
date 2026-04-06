# Consensus Level vs L11 MLP Helpfulness
**Date: 2026-03-11, 200×1024 tokens, probability-based**

## Key Finding: Crossover at 3-4/7 consensus

| Level | Tokens | Avg P(L10) | Avg P(L11) | ΔP | Effect |
|-------|--------|-----------|-----------|-----|--------|
| 0/7 | 210 | 0.629 | 0.774 | +0.145 | **L11 helps** |
| 1/7 | 950 | 0.715 | 0.781 | +0.066 | L11 helps |
| 2/7 | 2,584 | 0.700 | 0.741 | +0.041 | L11 helps |
| 3/7 | 5,559 | 0.611 | 0.624 | +0.013 | ~neutral |
| 4/7 | 11,932 | 0.427 | 0.405 | -0.022 | **L11 hurts** |
| 5/7 | 32,448 | 0.336 | 0.290 | -0.046 | L11 hurts |
| 6/7 | 67,972 | 0.276 | 0.218 | -0.058 | L11 hurts |
| 7/7 | 82,945 | 0.249 | 0.178 | -0.071 | L11 hurts |

Crossover at ~3-4/7 — exactly matching "Discrete Charm" causal analysis.

## Note on inverted base rates
Low consensus tokens (0-2/7) have HIGHER avg P because they're rare tokens 
(paragraph breaks, etc.) followed by predictable continuations. High consensus 
tokens are common function words followed by harder-to-predict content.
