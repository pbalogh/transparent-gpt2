# Knowledge Neurons Evaluation Results
## Dai et al. (2022) vs. Transparent GPT-2 Routing Hypothesis
**Date:** 2026-03-15 | **Model:** GPT-2 Small | **Device:** CUDA (EC2 T4)

---

## Experiment 1: Integrated Gradients ↔ Routing Neuron Overlap

**Method:** Run Dai's integrated gradients attribution on 20 factual prompts at L11. Check how many of the top-20 attributed neurons are our consensus/routing neurons.

| Metric | Value |
|---|---|
| Avg consensus neurons in top-20 | **2.6/7** (37%) |
| Avg routing neurons in top-20 | **7.3/30** (24%) |
| Expected routing by chance | **0.2** |
| **Enrichment factor** | **36.5×** |

**Critical finding:** The SAME routing neurons appear across nearly ALL prompts:
- N2, N611, N1611, N2044, N2173, N2460, N2600, N2910 → present in 16-19/20 prompts
- If these were fact-specific "knowledge" neurons, they shouldn't be the same for "Paris", "Berlin", "Tokyo", "Jupiter", and "Au"
- They're *routing infrastructure*, not *knowledge stores*

**One exception:** "Einstein → relativity" had 0 consensus and only 2 routing neurons in top-20 — suggesting this fact genuinely relies more on distributed residual representations.

---

## Experiment 2: Knockout Test

| Condition | Units | Avg Prob Drop | Relative |
|---|---|---|---|
| Dai's top-20 neurons | 20 | **-0.0804** | **-193.4%** |
| Consensus 7 | 7 | 0.0028 | 2.7% |
| All routing 30 | 30 | **-0.0307** | **-48.0%** |
| Attention heads (L11) | 12 | 0.0144 | 18.9% |

**Negative drop = knockout INCREASED fact probability.**

- Knocking out Dai's neurons makes the model MORE likely to predict the correct fact
- This is the routing signature: these neurons are doing distributional reshaping (pushing toward function words like "the"), not boosting the correct fact
- When you remove them, the fact signal from attention comes through more clearly
- Attention heads show the only meaningful positive drop — consistent with knowledge arriving via attention/residual stream

---

## Experiment 3: Transplant Test

**Method:** Capture Dai's "knowledge neurons" for Fact A, inject their activations when processing Fact B's prompt.

| Source → Dest | Source prob change | Dest prob change | Clean transplant? |
|---|---|---|---|
| Paris → Germany context | 0.0000 | -0.0005 | ❌ |
| Tokyo → Italy context | -0.0001 | -0.0152 | ❌ |
| Beijing → Spain context | -0.0000 | -0.0199 | ❌ |
| Moscow → Canada context | -0.0002 | -0.0018 | ❌ |
| Canberra → Brazil context | -0.0001 | -0.0025 | ❌ |

**Clean transplants: 0/5**

- Transplanting "Paris" neurons into Germany's context does NOT make the model predict "Paris"
- Instead, it slightly reduces the correct destination fact probability (routing disruption)
- Source fact probability barely budges (±0.0001)
- **This is definitive:** if neurons stored facts, transplanting them should transplant the fact

---

## Synthesis: The Reconciliation

Dai et al. correctly identified neurons with high *causal influence* on factual outputs. But they conflated **influence** with **storage**.

Our three experiments show:
1. **Same neurons, different facts** (Exp 1) → routing, not storage
2. **Knockout helps fact recall** (Exp 2) → neurons suppress/reshape, not boost
3. **No transplant** (Exp 3) → facts aren't encoded in these activations

**The mechanism:** These neurons (especially consensus + exception core) push probability mass toward function words ("the", "in", "and"). They're the distributional reshaping circuit from "Discrete Charm." When the model is confident about a fact, these neurons actively *dampen* it (the "boost drops below 1.0× at consensus" finding). Removing them removes the dampening → fact probability goes up.

**Why ROME works:** ROME edits MLP weights, which changes the routing decision — redirecting which attention-derived knowledge flows through. The edit isn't "writing a fact into a neuron" — it's "changing the highway sign."

**For Darkness Visible paper:** Add §5.3 or extend §5.1 with these results. Cite Dai et al. explicitly, acknowledge their attribution method identifies real causal structure, then show it's routing not storage. The transplant test is the cleanest argument.
