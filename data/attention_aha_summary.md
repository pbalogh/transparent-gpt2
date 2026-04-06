# Attention at "Aha" Layers
**Date: 2026-03-11, 10 factual prompts**

## Key Patterns

### H8 at L11: The Content Head
At every L11 lock-in, H8 has near-zero BOS attention and attends to
semantically relevant content tokens:
- "C" (degrees): freezes(0.33), zero(0.31), at(0.23)
- "second" (speed): per(0.25), 300(0.24), 000(0.19)
- "ic" (DNA): on(0.35), oxy(0.24), rib(0.18)
- "17" (French Rev): began(0.39), Revolution(0.25), in(0.22)

All other L11 heads are BOS sinks (0.80-0.99 to BOS).
H8 is the sole content-processing head at L11.

### H1 at L0: Copy Head
Copies immediately preceding token with ~0.98-1.00 attention weight.
Simple induction pattern that bootstraps the prediction.

### Lock-in Mechanisms by Difficulty
- **Trivial (Lincoln)**: L0 MLP, "Abraham"→"Lincoln" memorized
- **Easy (Shakespeare)**: L2 attention starts, H10 attends to content words
- **Medium (relativity)**: L8 H8 attends to "Einstein"(0.70) — entity lookup
- **Medium (moon)**: L10 H11 attends to "In"(0.77) — temporal context signal
- **Hard (second, ic, C, 17)**: L11 MLP decisive, H8 provides content attention

### H11 at L4-L8: Focus Head
At mid-layers, H11 achieves ~1.0 attention to a SINGLE content word:
- L4: 'was'(1.00) for Lincoln, 'theory'(1.00) for relativity
- Maximally focused — single-token extraction

## Implication
Facts arrive through attention copying semantic content, then MLPs at the
corresponding layer reshape the distribution. The "aha moment" is when the
right attention head locks onto the right source token and the MLP amplifies
it past competitors.
