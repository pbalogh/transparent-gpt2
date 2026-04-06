# Darkness Visible: Reading the Exception Handler of a Language Model

**Peter Balogh**

## Abstract

We decompose the final MLP layer of GPT-2 Small into a legible routing program. The layer's 3,072 neurons organize into a three-tier exception handler — 5 fused Core neurons that reset vocabulary toward function words, 10 Differentiators that suppress wrong candidates, and 5 Specialists that detect structural boundaries — all gated by 7 Consensus neurons that collectively detect "normal language." This architecture is unique to the terminal layer; earlier layers (L7, L10) have proto-routing but no structured specialization. We show that factual knowledge emerges not from individual neurons but from the intersection of ~100 simultaneously active "conceptual lenses" — a mechanism analogous to a game of 20 Questions, where each binary neuron narrows the possibility space until only the correct completion survives. The routing logic is fully legible (27 named neurons with documented behavior); the knowledge it routes is distributed across the remaining ~3,040 neurons in a combinatorial code.

## 1. Introduction

The MLP layers of transformer language models are typically treated as opaque nonlinear transformations. Recent work has shown that individual neurons can be interpreted [citations], and that MLP layers implement key-value memories [Geva et al., 2021], but a complete, legible account of what an MLP layer *does* — readable as pseudocode — has remained elusive.

We provide such an account for Layer 11 of GPT-2 Small, the model's final MLP. Our decomposition preserves the original weights exactly (cosine similarity 0.99999994 with the unmodified layer) while revealing:

1. **A binary routing switch**: A single exception neuron (N2123, 11.3% fire rate) partitions all tokens into "routine" and "exceptional" processing paths.

2. **A consensus detector**: Seven neurons that each monitor a different linguistic dimension — syntax (N2361), temporality (N2460), entities (N1831), semantics (N1245), sentence structure (N2600) — and collectively signal "this token is predictable."

3. **A three-tier exception handler**: When the exception neuron fires, processing routes through Core (vocabulary reset), Differentiators (candidate suppression), and Specialists (boundary detection).

4. **Combinatorial knowledge retrieval**: The ~3,040 residual neurons function as "conceptual lenses" whose intersection produces specific factual completions — a mechanism we call the **20 Questions principle**.

5. **Terminal crystallization**: This architecture is unique to L11. Earlier layers with nominal consensus structure (L7, L10) show no enrichment, no specialization, and no tight co-firing clusters. Legible routing crystallizes only at the network's final decision point.

## 2. Background and Related Work

### 2.1 MLP Interpretability
- Geva et al. (2021): MLPs as key-value memories
- Meng et al. (2022): Locating and editing factual associations (ROME)
- Gurnee et al. (2023): Finding neurons in a haystack
- Bricken et al. (2023): SAE decomposition of MLP neurons

### 2.2 Binary Routing in Neural Networks
- Dettmers et al. (2022): Outlier features and quantization
- Balogh (2026): "The Discrete Charm of the MLP" — consensus/exception architecture at L11
- This paper extends Balogh (2026) from *detecting* the routing structure to *reading* it

### 2.3 Attention Sinks and Structural Heads
- Xiao et al. (2023): Attention sinks — BOS tokens as no-op destinations
- Our finding: L11H7 (6× most important attention head) sends 45% of attention to BOS, more for exception tokens (47% vs 37%)

## 3. Methods

### 3.1 Transparent Forward Pass
We construct a `TransparentGPT2` that wraps the original model with no weight changes. At Layer 11, the MLP's output is decomposed into tier contributions computed from the same GELU activations:

```
output = Core(x) + Differentiators(x) + Specialists(x) + Residual(x)
```

Each tier is defined by a fixed neuron mask. The sum equals the original MLP output exactly.

### 3.2 Neuron Characterization
For 512,000 tokens (500 × 1024 sequences from WikiText-103):
- Forward hooks capture pre-GELU activations at L11
- Binary firing determined by |post-GELU| > 0.1
- Exception/consensus status computed per token
- Co-firing measured by Jaccard similarity
- Tier assignment by exception-conditional fire rate

### 3.3 Knowledge Extraction
For 20 factual prompts, we decompose the MLP's contribution to the correct prediction by tier, identifying which residual neurons carry the factual signal.

## 4. The Exception Handler

### 4.1 Three-Tier Architecture

**Core (5 neurons, fire rate 90-100% among exceptions):**
N2123, N2910, N740, N1611, N2044. These form a single fused mega-neuron (pairwise Jaccard ≥ 0.91, peak 0.998). They push predictions toward generic function words: `the`, `in`, `and`, `a`. This is a **vocabulary reset** — clearing the prediction slate before the residual contributes specific knowledge.

**Differentiators (10 neurons, 35-88%):**
Including suppression pair N584 + N2378 (Jaccard 0.889) and subword repair neurons N1602, N1715. These neurons don't promote correct answers — they suppress wrong ones, narrowing the candidate set.

**Specialists (5 neurons, 14-37%):**
N737 is particularly notable: a solo paragraph boundary detector (Jaccard < 0.15 with all other neurons). It fires independently of the Core, activating only at structural transitions.

### 4.2 Norm Contributions
Among exception tokens (sampled):
- Core: 54% of output norm (alignment +0.37 with full output)
- Differentiators: 23% (+0.34)
- Specialists: 4% (+0.06)
- Residual: 94% (+0.76)

The residual dominates in magnitude, but its direction is shaped by the tiers' vocabulary reset and suppression.

### 4.3 The Exception Neuron N2123
Output direction: pushes toward `,`, `the`, `and`, `.` (positive logits ~2-3)
Suppresses: garbage tokens, byte-pair artifacts (logits ~ -6)
Input sensitivity: activated by subword fragments (`anny`, `arter`), inhibited by complete content words (`destroy`, `challenge`, `conclude`)

**Interpretation:** N2123 detects tokens where the model "doesn't yet know what's happening" — incomplete words, structural breaks, rare patterns. Its vocabulary reset buys time for the residual to compute a contextually appropriate completion.

## 5. The Consensus Detector

### 5.1 Seven Dimensions of "Normal"

Each consensus neuron monitors a different linguistic property:

| Neuron | Fire Rate | Monitors | Enriched tokens |
|--------|-----------|----------|-----------------|
| N2 | 88.1% | Sentence boundaries | `.`, `In`, `\n` |
| N2361 | 85.6% | Syntactic relations | `were`, `which`, `be`, `who` |
| N2460 | 87.6% | Temporal/aspectual | `been`, `during`, `when`, `where` |
| N2928 | 92.0% | Structure/formatting | `@`, `\n`, `;`, `In` |
| N1831 | 81.1% | Entity/discourse | `Fey`, `Shiva`, `This`, `which` |
| N1245 | 86.0% | Semantic content | `gods`, `deities`, `are`, `is` |
| N2600 | 77.7% | Sentence structure | `.`, `\n`, `I`, `He`, `(` |

### 5.2 Output Direction Alignment

All 7 consensus neurons push in the same direction: toward high-frequency function words (`,`, `the`, `and`). Their pairwise cosine similarities range from 0.52 to 0.73, except N2600 which is orthogonal to the rest (cos ≈ 0.0).

The exception neuron N2123 is **aligned with the consensus mean** (cos = 0.838). Both routing paths push toward the same safe vocabulary — the difference is activation context, not direction.

### 5.3 Consensus as Linguistic Predictability

| Level | Tokens | % | Character |
|-------|--------|---|-----------|
| 0/7 | 486 | 0.1% | Paragraph breaks (27%), rare subwords |
| 1-2/7 | 8,908 | 1.7% | `\n\n`, subword fragments, markup |
| 3-4/7 | 45,481 | 8.9% | Transition: `=` headers, emerging function words |
| 5-6/7 | 250,928 | 49.0% | Function words, punctuation dominant |
| 7/7 | 206,197 | 40.3% | Pure routine: `,`, `.`, `the`, `of`, `"` |

**Universal consensus breakers** (depleted by nearly all 7 neurons): `\n\n`, `j` (subword fragment), `Tru` (truncated proper noun), `-`, `@`.

## 6. Where Knowledge Lives (And Where It Doesn't)

### 6.1 The MLP Doesn't Retrieve Facts

A natural hypothesis is that the ~3,040 residual neurons function as a combinatorial knowledge store — each neuron encoding a "conceptual lens" whose intersection narrows to specific facts, like a game of 20 Questions. We tested this directly.

For 12 factual prompts ("The French Revolution began in ___", "In 1969, astronauts landed on the ___", etc.), we accumulated residual neuron output directions one at a time, sorted by contribution magnitude to the correct answer. If residual neurons retrieve facts combinatorially, the target's rank should drop rapidly as neurons accumulate.

**It doesn't.** Across all 12 prompts, the correct token never reaches top-10 from residual neurons alone. Individual neuron output directions are dominated by high-frequency tokens (`,`, `the`, `.`) regardless of context. The residual neurons, taken in isolation, cannot produce factual completions.

### 6.2 Knowledge Arrives Via the Residual Stream

The model predicts "moon" with 61.9% probability for "In 1969, astronauts landed on the" — but this signal doesn't come from the L11 MLP. It's already in the residual stream when it arrives, built up by 11 layers of attention (which copy context-relevant information) and earlier MLPs (which reshape distributions).

The L11 MLP's role is not retrieval but **adjustment**:
- For exception tokens: reshape the distribution (vocabulary reset + correction)
- For consensus tokens: leave the distribution alone (or actively degrade it — boost drops below 1.0× from 4/7 consensus onward)

### 6.3 Three Roles, Not One

The progressive prediction analysis (§4) reveals the MLP's actual contribution:

1. **Core neurons** provide a vocabulary reset — pushing predictions toward generic function words (`the`, `in`, `and`), clearing the slate regardless of what the residual stream carried in.

2. **Differentiator neurons** suppress wrong candidates — not by promoting the right answer but by demoting competitors.

3. **Residual neurons** provide a contextual correction signal that, when added to the already-informed residual stream, nudges the distribution. For "1969 landed on the ___", the residual stream already contains "moon" information from attention; the residual MLP neurons provide a small push that, combined with the pre-existing signal, lands on the right answer.

The key insight: **the MLP's residual neurons don't contain knowledge — they contain adjustments that interact with knowledge already present.** No single component retrieves "the French Revolution began in 1789." The fact is an emergent property of the full forward pass.

### 6.4 Implications for Knowledge Editing

This reframes the ROME paradigm. Meng et al. (2022) locate factual associations in specific MLP layers and edit individual neurons. Our analysis suggests a more nuanced picture: the "fact" is distributed across the entire forward pass, and the MLP layer provides a *correction* to an answer largely determined by attention.

Our finding that Core neurons are 120× more efficient injection points than residual neurons for fact insertion is consistent: the Core provides a clean vocabulary reset that downstream processing can build on, whereas injecting into the distributed residual creates interference with the ~3,040 other contextual adjustment signals.

### 6.5 The Routing IS the Contribution

If the MLP doesn't retrieve knowledge, what does it do? It **routes**. The consensus/exception architecture determines whether the MLP should reshape the distribution (exception path) or get out of the way (consensus path). This binary decision — intervene or abstain — is the MLP's primary contribution to the forward pass at L11. The routing is legible; the knowledge it routes around is not.

## 7. Terminal Crystallization

### 7.1 Earlier Layers Lack Structure

We replicated the characterization analysis at L7 and L10, layers with nominal consensus architectures (identified in Balogh, 2026). Neither shows the structured exception handler found at L11:

| Property | L7 | L10 | L11 |
|----------|----|----|-----|
| Exception fire rate | 44.5% | 50.3% | 11.3% |
| Enrichment ratio | ~1.0× | ~1.0× | Up to 2.0× |
| Specialist neurons | 0 | 0 | 5 |
| Max Jaccard (co-firing) | ~0.7 (uniform) | ~0.7 (uniform) | 0.998 (fused core) |
| Consensus neurons | 1 | 3 | 7 |

L7 and L10's "exception" neurons fire on ~half of all tokens — they're broad binary splits, not selective exception detectors. The progression from diffuse splitting (L7) through weak structure (L10) to crystallized routing (L11) reflects the model's developmental arc: early layers establish broad categories, later layers refine them, and only the terminal layer crystallizes a legible decision architecture.

### 7.2 L11H7: The Attention Sink

The most important attention head at L11 (6× importance of any other head) sends 45% of its weight to the BOS token. Exception tokens attend to BOS even more (47% vs 37% for consensus) and with lower entropy (3.01 vs 3.51 bits).

This is consistent with the attention sink phenomenon: when the MLP is doing the real work (exception path), the attention head has less to contribute and dumps weight on the no-op BOS position. L11H7's importance may derive from its role as a residual stream normalizer rather than an information mover.

## 8. Discussion

### 8.1 The Program and the Database

Our decomposition separates GPT-2's L11 MLP into two functionally distinct components:

1. **The Program** (27 named neurons): Binary routing logic that determines processing path, resets vocabulary, suppresses candidates, and detects boundaries. This is fully legible — we can write it as pseudocode with named variables.

2. **The Database** (~3,040 residual neurons): Combinatorial knowledge retrieval via the 20 Questions principle. This is distributed and not directly legible, but its *mechanism* is legible — we know how facts emerge even if we can't enumerate them.

This is analogous to reading a software system's control flow (if/else branches, routing tables) without access to its database. You can understand the *logic* of decision-making without knowing every stored fact.

### 8.2 Why the Terminal Layer?

We hypothesize that legible routing crystallizes at L11 because it is the last opportunity for the model to adjust its prediction. Earlier layers can afford diffuse, distributed processing because subsequent layers can correct errors. L11 has no safety net — its decisions are final. This creates evolutionary pressure (during training) for clear, reliable routing at the terminal layer.

### 8.3 Limitations

- **Single model**: All analysis is on GPT-2 Small (124M parameters). Larger models may have different or more complex routing.
- **Single layer**: We characterize L11 in depth but only sketch L7 and L10. A complete "pseudocode" account would need all 12 layers.
- **Static analysis**: Our weight-based output directions are approximate — actual behavior depends on the full residual stream context at each position.
- **20 Questions is a metaphor**: We demonstrate the principle on 20 prompts. A rigorous information-theoretic analysis of how many neurons are needed for unique identification remains future work.

## 9. Conclusion

We have shown that the final MLP of GPT-2 Small implements a legible routing program: 7 consensus neurons detect "normal language" along distinct linguistic dimensions, a single exception neuron gates entry to a three-tier processing hierarchy, and ~3,040 residual neurons retrieve factual knowledge through combinatorial intersection. This architecture crystallizes only at the terminal layer — a phenomenon we call terminal crystallization — suggesting that legible decision structures emerge under the pressure of finality.

The title alludes to Milton's "darkness visible" — the paradoxical light that illuminates Hell in *Paradise Lost*. What we find inside the MLP is similarly paradoxical: the routing program is transparent and readable, but the knowledge it routes is irreducibly distributed. We can see the darkness — we can read the program that navigates it — but the knowledge itself remains a combinatorial shadow, retrievable only through the intersection of many dim lights.

## Reproducibility

All code and data: https://github.com/pbalogh/transparent-gpt2
Companion paper (consensus architecture): https://github.com/pbalogh/discrete-charm-mlp
