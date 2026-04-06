# Transparent GPT-2: First Results

**Date:** March 11, 2026
**Data:** 200 × 1024-token sequences from WikiText-103

## Three-Column Comparison

| Mode | PPL | Δ | Note |
|---|---|---|---|
| Standard GPT-2 | 31.79 | — | Black box baseline |
| Transparent GPT-2 | 31.79 | +0.00% | Same math, routing visible |
| Bypass (all, v1) | 5870 | 💀 | Cascade failure from L0 |

**Key validation:** Transparent mode produces IDENTICAL perplexity.
The routing detection adds zero divergence.

## Single-Layer Zero-Bypass (MLP zeroed at consensus)

| Layer | PPL | Δ | Tokens bypassed | Phase |
|---|---|---|---|---|
| L10 | 32.38 | +1.9% | 42.2% | Decision |
| L8  | 32.40 | +1.9% | 42.7% | Decision |
| L9  | 32.77 | +3.1% | 70.8% | Decision |
| L7  | 32.77 | +3.1% | 79.6% | Decision |
| L11 | 33.99 | +6.9% | 55.5% | Decision |
| L0  | 858.52 | +2601% | 82.6% | Scaffold ⚠️ |

**Findings:**
1. Decision layers (L7-11) tolerate zero-bypass at 2-7% PPL cost
2. L0 is SACRED — its "consensus" structure is load-bearing for all downstream layers
3. L11 costs the most to bypass despite being best-characterized — it's doing real work even at consensus

## Cumulative Zero-Bypass

| Layers bypassed | PPL | Δ |
|---|---|---|
| L11 | 33.99 | +6.9% |
| L7+L11 | 35.55 | +11.8% |
| L0+L7+L11 | 928.86 | +2822% ⚠️ |

**Finding:** Degradation is roughly additive for L7-11.
Adding L0 is catastrophic. L0 must be excluded from bypass.

## Linear Bypass (W_out @ W_in, skip GELU): CATASTROPHIC

| Layer | PPL | Δ |
|---|---|---|
| L11 | 19,615 | +61,606% |
| L10 | 113 | +256% |

**Finding:** The naive linear approximation (skip GELU) is useless.
This is actually the paper's point made visceral: you can't smooth-approximate a valve.
The GELU isn't doing "a little nonlinearity" — it's implementing a binary switch,
and removing it destroys the switch.

Zero-bypass (just skip the MLP entirely) works VASTLY better than
linear bypass (approximate the MLP without GELU). Because the MLP's
contribution at consensus is counterproductive — the best approximation
of "slightly harmful" is "nothing," not "linear version of harmful."

## Routing Report (Transparent Mode)

```
L 0 🔧 scaffold  | consensus=82.6%, exception=9.3%
L 1 🔧 scaffold  | gateway=35.5%
L 2 🔧 scaffold  | gateway=17.5%
L 3 🔧 scaffold  | gateway=38.8%
L 4 🌊 diffuse   | (no routing structure)
L 5 🌊 diffuse   | (no routing structure)
L 6 🌊 diffuse   | (no routing structure)
L 7 ⚡ decision  | consensus=79.6%, exception=6.4%
L 8 ⚡ decision  | consensus=42.7%, exception=28.8%
L 9 ⚡ decision  | consensus=70.8%, exception=17.2%
L10 ⚡ decision  | consensus=42.2%, exception=21.7%
L11 ⚡ decision  | consensus=55.5%, exception=11.5%
```

## Next Steps

1. Decompose `full_mlp()` — which neurons actually contribute at exception?
2. Decompose `attend()` — head-level routing (skip BOS sinks?)
3. Test bypass on LAMBADA (harder task, more sensitive to errors)
4. Per-token routing visualization: color-code text by routing path
5. Investigate L0's special role — what makes its consensus sacred?
