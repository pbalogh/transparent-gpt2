# Logit Lens: Layer-by-Layer Prediction Emergence
**Date: 2026-03-11, GPT-2 Small, 12 factual prompts**

## Key Finding: Facts emerge at different layers through different mechanisms

| Fact | First top-1 | Mechanism | L11 MLP effect |
|------|-------------|-----------|----------------|
| Lincoln | L4 mlp | L0 MLP: rank 1 immediately | 99.8% → 99.3% (slight degradation) |
| Shakespeare | L4 attn | L2 attn: rank 2 from "William" | 98.2% → 54.3% (heavy redistribution) |
| Fahrenheit | L7 mlp | Celsius leads L3-L6, flip at L7 | 78.2% → 29.8% (heavy redistribution) |
| da Vinci (Vin) | L8 mlp | Gradual attention buildup | 95.6% → 99.9% (L11 HELPS here) |
| relativity | L8 attn | L6-L7 attention narrows to top-5 | 92.3% → 52.8% (heavy redistribution) |
| moon | L10 mlp | "planet" dominates L6-L9, L10 MLP flips | 75.4% → 61.9% (moderate redistribution) |
| French Rev (17) | L11 attn | Late: rank 30+ until L10 MLP | L11 attn: 22.4%, L11 MLP: 46.7% (HELPS) |
| center (gravity) | L7 mlp | L5-L6: "end" leads, L7 MLP flips | Complex: bounces between center/earth/Earth |
| second (speed) | L11 mlp | "hour" leads until L11 | L11 MLP is decisive (3.5% → 49.9%) |
| ic (DNA) | L11 mlp | "otide" leads L5-L10 | L11 MLP is decisive (4.9% → 74.1%) |
| C (degrees) | L11 mlp | Celsius leads most of the way | L11 MLP is decisive (1.8% → 16.3%) |
| Paris | never | Never reaches top-1! Model predicts "the" | Top-4 at L11, prob 3.2% |

## Three modes of L11 MLP behavior:
1. **Redistributive** (Shakespeare, relativity, Fahrenheit): Answer already correct, L11 MLP spreads probability to alternatives. This is the consensus-path behavior — the MLP is counterproductive.
2. **Decisive** (second, ic, C, 17): Answer not yet correct, L11 MLP makes the final call. This is the exception-path behavior — the MLP is essential.
3. **Neutral** (Lincoln, moon): Answer already correct, L11 MLP makes minor adjustments.

## Implications for "Darkness Visible"
- Facts are built progressively by attention (copying context) and earlier MLPs (reshaping)
- L11 MLP is NOT a knowledge retrieval layer — it's a decision/adjustment layer
- For "easy" facts (high confidence by L10), L11 MLP actively degrades performance
- For "hard" facts (low confidence by L10), L11 MLP is the tiebreaker
- This perfectly matches the consensus/exception architecture: consensus tokens get degraded, exception tokens get helped
