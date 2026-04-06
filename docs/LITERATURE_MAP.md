# Literature Map — Darkness Visible

How each reference supports the paper's claims about GPT-2 Small's L11 exception handler.

---

## Core Claims & Supporting Literature

### Claim 1: MLP layers are key-value memories with readable structure
- **Geva et al 2021** — MLP layers as key-value memories. We extend this: L11's keys aren't just associative — they implement a binary routing program with named neurons.
- **Geva et al 2023** — Dissecting factual recall. Their subject→last-token→MLP pipeline is what our exception handler *routes*. The 20 Questions experiment (§6) tests whether individual neurons encode facts (they don't — knowledge arrives via attention).
- **Dai et al 2022** — Knowledge neurons. They found individual neurons storing facts; our finding is more nuanced — routing neurons (Core 5) are 120× more efficient for injection than residual neurons, but the knowledge itself is distributed.

### Claim 2: Three-tier exception handler with consensus quorum
- **Balogh 2026 (Discrete Charm)** — Our own prior work. Established the 7-neuron consensus quorum and binary routing. Darkness Visible extends this to the full 27-neuron program with tier structure.
- **Gurnee et al 2023** — Sparse probing finds individual neurons with interpretable roles. We go further: not just finding neurons, but showing they form a *program* with compositional structure (Core fuses, Differentiators specialize, Specialists detect).

### Claim 3: Progressive prediction through depth (logit lens / tuned lens)
- **nostalgebraist 2020** — The original logit lens. Our developmental arc (scaffold → diffuse → decision) uses logit lens to track prediction quality per layer.
- **Belrose et al 2023** — Tuned lens fixes logit lens bias in early layers. We validate: tuned lens adds +6.3pp at L0-L3, converges at L11. Both show the same three-phase arc — it's real, not an artifact.
- **Tenney et al 2019** — BERT rediscovers the NLP pipeline. Our layer-by-layer developmental arc is the GPT-2 version of this finding.

### Claim 4: Consensus predicts MLP intervention helpfulness
- **Conmy et al 2023** — Automated circuit discovery (ACDC). We don't use ACDC but our causal interventions (bypass MLP at varying consensus) are the manual version of circuit ablation.
- **Wang et al 2023** — IOI circuit. The gold standard for circuit-level mechanistic claims. Our 27-neuron program is a comparable level of mechanistic detail for the MLP.
- **Hanna et al 2024** — Greater-than circuits in GPT-2. Another example of readable computation in GPT-2 Small, validating that this model is interpretable enough for such claims.

### Claim 5: Terminal crystallization (L11-specific, not present at L7/L10)
- **Elhage et al 2021** — Mathematical framework for transformer circuits. The residual stream + attention + MLP decomposition is the language we use to describe how consensus emerges only at L11.
- **Olsson et al 2022** — Induction heads. Attention mechanism reference point. Our finding that routing is in MLP (not attention) for L11 contrasts with their attention-centric view.

### Garden-Path Negative Result
- **Van Gompel & Pickering 2001** — Lexical guidance in garden-path resolution. Source of our 15 minimal pair stimuli.
- **Frazier & Rayner 1982** — Classic garden-path theory. We test whether N2123 detects structural reanalysis — it doesn't. GPT-2 is a one-stage parser.

### Methodology
- **Clark et al 2019** — BERT attention analysis. Attention visualization methods we adapt.
- **Voita et al 2019** — Head pruning (specialized heads do heavy lifting). Our finding that 7 consensus neurons do the routing while 3,040 residual neurons carry knowledge is the MLP version.
- **Bills et al 2023** — LLMs explain neurons. We take the opposite approach: instead of asking an LLM to label neurons, we read them directly through activation patterns.
- **Merullo et al 2024** — Circuit reuse across tasks. Relevant to whether our L11 program generalizes.
- **Nanda et al 2023** — Progress measures for grokking. Methodological reference for tracking learned structure.

### Technical
- **Dettmers et al 2022** — LLM.int8() quantization. Referenced for efficient inference context.
- **Xiao et al 2023** — Attention sinks. The "streaming" interpretation of early-position tokens connects to how consensus neurons monitor global state.
- **Bricken et al 2023** — SAE/dictionary learning. Alternative approach to neuron interpretation; we don't use SAEs here but acknowledge them as complementary.

---

## What's NOT in the bib but should be
- Anthropic's scaling monosemanticity (May 2024) — SAE features at scale, contrasts with our individual-neuron approach
- McDougall et al 2024 — Copy suppression heads, relevant to our "function word promotion" finding

---

## Cognitive Foundations

### Feldman 2006 — *From Molecule to Metaphor*
**What it gives us:** Recruitment learning: uncommitted neurons get "recruited" to represent new facts when they happen to be well-connected to all relevant inputs. Only a few neurons (out of thousands) satisfy the connectivity requirement — they become the dedicated circuit for that knowledge via rapid weight change. Also: the 2/3 activation rule in triangle nodes — activating any two of three connected concepts activates the third.
**Our connection:** Recruitment learning is how consensus neurons EMERGE. N2, N2361, N2460, N2928, N1831, N1245, N2600 weren't designed as consensus monitors — they were recruited through training because they happened to be well-connected to the features distinguishing high-consensus from low-consensus contexts. The 2/3 rule maps to our finding that consensus at 3-4/7 is the crossover point where MLP intervention becomes helpful — it's the minimum "quorum" needed for reliable activation.

---

*24 references + 1 cognitive foundation. Paper: `paper/main.tex`. arXiv: pending.*
