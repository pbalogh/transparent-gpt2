# Darkness Visible — Review Notes

Rotating critical review passes. Each dated entry focuses on one aspect.

---

## 2026-03-13 — Pass 1: Statistical Claims

Focus: Are all numbers, p-values, effect sizes, and sample sizes defensible?

### 🔴 CRITICAL

**1. Token count inconsistency: 204,800 vs 204,600**
§5.3 text says "for 204,800 tokens, we measure whether L11's MLP improves…" but Table 4 sums to exactly 204,600 (210+950+2584+5559+11932+32448+67972+82945). Table 5 caption also says "204,600 tokens." Table 6 (logit lens) uses "204,600 tokens." The 204,800 in the prose is wrong — fix to 204,600 or reconcile.

### 🟡 MODERATE

**2. Phantom "1 out of 142" figure (§5.3)**
Text claims "almost no correct predictions are lost (1 out of 142 at 0/7)." But Table 3 shows 486 tokens at 0/7 (512K dataset) and Table 4 shows 210 tokens at 0/7 (204.6K dataset). Where does 142 come from? Likely the number of *correct* top-1 predictions at 0/7 consensus, but this isn't stated. Needs clarification — a reviewer will flag this.

**3. No significance tests on the consensus-helpfulness crossover**
The central claim — that L11 MLP crosses from helpful to harmful at 3–4/7 consensus — rests entirely on Table 4's ΔP values. No confidence intervals, no bootstrap, no permutation test. The crossover is the paper's key quantitative result. At minimum, add standard errors on ΔP. Given the sample sizes (210 tokens at 0/7, 82,945 at 7/7), the low-consensus estimates are noisy.

**4. "Importance score 91.5" for L11H7 — undefined metric**
§6.3 introduces an "importance score" for attention heads without defining it. Is this mean absolute attention weight? Gradient-based? Some norm of the OV matrix? A reviewer cannot assess the "6× the next head" claim without knowing the metric. Define it in §3 (Methods) or at point of use.

**5. Only 6 of 12 prompts shown in Table 2**
The negative knowledge-retrieval result ("the MLP does not retrieve facts") is based on 12 prompts, but Table 2 shows only 6. This invites cherry-picking suspicion. Either show all 12 (appendix is fine) or report aggregate statistics (median rank, IQR) over all 12. N=12 is already small for a strong universal negative claim — don't also hide half the data.

**6. "Exactly" vs floating-point reality**
Abstract and §3.1 both say the decomposition preserves the original "exactly" / "with no weight changes," then report cosine sim 0.99999994 and max error < 6e-5. These are floating-point arithmetic artifacts, but "exactly" is technically false and a pedantic reviewer will ding it. Suggest: "to floating-point precision" or "up to numerical precision."

### 🟢 MINOR

**7. Table 3 arithmetic checks out.** 486+8908+45481+250928+206197 = 512,000. Percentages verified within rounding.

**8. Table 4 net calculations are correct.** 17,949 − 7,795 = 10,154 ✓. 7,674 − 3,530 = 4,144 ✓. ΔP signs and magnitudes consistent.

**9. 500 × 1,024 = 512,000 ✓.** Dataset size internally consistent for the characterization experiments.

**10. Threshold robustness acknowledged.** The |GELU| > 0.1 threshold is a modeling choice, but the paper correctly notes prior robustness analysis across 0.01–1.0. Sufficient.

### Summary

The biggest issue is the missing statistical backing for the consensus-helpfulness crossover (item 3) — this is the paper's central quantitative finding and it has zero inferential statistics. The 204,800/204,600 discrepancy (item 1) is an easy fix but embarrassing if caught by a reviewer. The undefined importance metric (item 4) and incomplete prompt table (item 5) are both things a methods-oriented reviewer will flag.

**Priority fixes:** Items 1, 3, 5, then 2 and 4.

---

## 2026-03-14 — Pass 2: Logical Flow

Focus: Does the argument build coherently? Any logical leaps, unsupported inferences, or structural issues?

### 🔴 CRITICAL

**1. The "vocabulary reset" narrative contradicts the ablation results**
§4.1 frames Core neurons as architecturally central — "a fused mega-neuron" performing "vocabulary reset," accounting for "54% of the exception-path output norm." The language implies this is *the* key mechanism. But §5.5 (Table 5) shows Core ablation costs only +0.2% PPL while Differentiators cost +1.3% — Core is 6× *less* important. The paper never reconciles this contradiction. It reads like the Core was the original hypothesis, the ablation disproved it, and the narrative wasn't updated. This is the biggest logical gap in the paper. Suggestion: either reframe the Core as scaffolding (big in magnitude, small in impact — like a DC offset that gets subtracted) or explain *why* 54% of output norm translates to 0.2% PPL impact (e.g., because the residual stream already contains the function-word baseline).

**2. "Booster rocket" analogy obscures rather than clarifies**
§4.3 says the Core acts as a "booster rocket — providing an initial push (vocabulary reset) that is discarded once the residual contributes the actual content." But the Core isn't discarded — it's part of the final output. What the analogy seems to mean is that Core shifts the distribution toward a generic prior and the residual overrides it. But that's not discarding; that's *overwriting*. And if the Core is routinely overwritten by the residual (which is what +0.2% PPL suggests), then the Core is nearly a no-op in practice. The paper needs to decide: is the Core functionally important (as §4 claims) or nearly redundant (as §5.5 shows)?

### 🟡 MODERATE

**3. Circular definition risk in "consensus detects predictability"**
§5.2 claims consensus neurons detect "linguistic predictability," but the evidence is that high-consensus tokens are function words and punctuation (Table 3). Function words and punctuation *are* high-frequency, high-predictability tokens by definition. So the argument is: consensus fires on predictable tokens → consensus detects predictability. That's nearly tautological. A stronger version would show that consensus level correlates with model confidence (pre-MLP top-1 probability) *after controlling for* token frequency. The data for this probably exists in Table 4 (Avg P(L10) does decrease with consensus level) but the paper doesn't frame it as evidence against the circularity concern.

**4. Negative result ("MLP does not retrieve facts") is too strong for the evidence**
§5.1 tests 12 hand-picked prompts and concludes "the MLP does not retrieve facts." But N=12 doesn't support a universal negative. The prompts are all simple entity completions (moon, Shakespeare, relativity). What about: multi-hop facts? Numeric facts? Facts requiring composition? The paper should scope the claim: "We find no evidence that L11's MLP retrieves simple factual associations" — not that it never retrieves any facts.

**5. Terminal crystallization hypothesis is post-hoc and unfalsifiable**
§6.2 hypothesizes that routing crystallizes at L11 because it's "the last opportunity" and there's "gradient pressure for clear, reliable routing." This is plausible but unfalsifiable as stated. You could equally argue that the *first* layer should crystallize (to set the processing agenda) or that *middle* layers should (where most knowledge emerges, per Table 6). Without a testable prediction — e.g., "in a 24-layer model, we'd expect crystallization at L23" — this is just a story. Label it as speculation or derive a testable prediction.

**6. Section ordering front-loads the mechanism, back-loads the justification**
The paper presents the exception handler architecture (§4) before showing it matters (§5). A reader who reaches §4.1's detailed neuron-by-neuron characterization will reasonably ask "so what?" for 3 pages before getting to the ablation and knowledge results. Consider: either add a forward reference at the start of §4 ("We will show in §5 that this architecture correctly predicts when MLP intervention helps vs. harms") or restructure so the functional results (§5.3–5.5) precede the mechanistic details.

### 🟢 MINOR

**7. Specialists "slightly improve PPL when ablated" is handwaved**
Table 5 shows Specialist ablation gives −0.3% PPL (i.e., *better* without them). The paper calls this "marginally counterproductive on average" but doesn't investigate why. If 5 named neurons are actively harmful, that's interesting and worth a sentence. Are they overcorrecting at boundaries? Fighting with attention heads?

**8. The "two orthogonal axes" framing (§5.2) is clean but underspecified**
The paper says six content neurons have pairwise cosine similarities of 0.52–0.73 and N2600 has cos ≈ 0.0 with all six. But 0.52–0.73 is a wide range — are these six neurons really "an axis" or are they several sub-clusters? A PCA or factor analysis on the 7 output directions would strengthen (or complicate) the claim.

**9. Transition from §5 to §6 is abrupt**
§5 ends with "the routing is legible… the knowledge it routes around is not" — a nice closing line. Then §6 (Terminal Crystallization) starts with cross-layer analysis that feels like a different paper. A bridging sentence would help: "Having characterized L11's architecture, we now ask: is this structure unique to the terminal layer?"

### Summary

The paper's biggest logical vulnerability is the Core narrative (items 1–2). A reviewer will notice that the mechanism presented as architecturally central in §4 is shown to be nearly irrelevant in §5.5, and the paper doesn't address this tension. The negative knowledge-retrieval claim (item 4) is overclaimed for N=12 prompts. The terminal crystallization hypothesis (item 5) needs to be flagged as speculative.

**Priority fixes:** Items 1 and 2 (reconcile Core importance), then 4 (scope the negative claim), then 3 (address circularity).

---

## 2026-03-15 — Pass 3: Related Work

Focus: Missing citations, under-engaged prior work, claims that contradict published results.

### 🔴 CRITICAL

**1. Dai et al. 2022 "Knowledge Neurons in Pretrained Transformers" — not cited, directly contradicted**
This paper explicitly identifies "knowledge neurons" in MLP layers that store factual associations, using attribution and suppression to show individual MLP neurons are necessary for specific facts. The present paper's central negative claim ("the MLP does not retrieve facts," §5.1) directly contradicts Dai et al.'s findings. Not citing this is a glaring omission — any reviewer in mech interp will notice immediately. The paper *must* engage with this work. Possible reconciliation: Dai et al. work on BERT (bidirectional), the present paper is on GPT-2 (autoregressive); or Dai et al. use attribution which may confuse routing with retrieval; or Dai et al.'s "knowledge neurons" could be analogous to the present paper's consensus/exception neurons (routing, not retrieval). But this argument needs to be *made*, not ignored.

**2. Meng et al. 2022 (ROME) is cited but the tension is never addressed**
The paper cites Meng 2022 as background, but ROME's entire premise is that factual knowledge *can be edited by modifying MLP weights* — implying the MLP *stores* facts. If the MLP doesn't retrieve facts (as §5.1 claims), why does ROME work? The paper needs to address this head-on. Possible answer: ROME edits change the *routing* behavior rather than modifying stored facts, or ROME works at mid-layers (L17 in GPT-J) where the architecture differs from L11 of GPT-2 Small. But the paper doesn't say any of this.

**3. Only 8 references for a NeurIPS submission**
This is remarkably thin. Most papers in this area have 30–50+ citations. A reviewer will infer either (a) insufficient literature review or (b) deliberate avoidance of work that complicates the narrative. Neither is good. The related work section (§2) is only ~15 lines. This needs significant expansion.

### 🟡 MODERATE

**4. Belrose et al. 2023 "Eliciting Latent Predictions with the Tuned Lens" — not cited**
§5.2 uses the logit lens extensively (Table 6, case studies), but the tuned lens paper documents known biases in the logit lens: it becomes unreliable at early layers because the residual stream hasn't yet rotated into the unembedding basis. This directly affects the "three-phase developmental arc" interpretation. The early-layer accuracy numbers (0.8% at embedding, 4.8% at L0–L3) may underestimate actual information because the logit lens is miscalibrated there. At minimum, acknowledge this limitation; ideally, replicate with tuned lens to confirm the arc holds.

**5. Geva et al. 2023 "Dissecting Recall of Factual Associations in LMs" — not cited**
This is a direct follow-up to Geva 2021 (which *is* cited). It traces factual recall step by step through attention and MLP layers, finding that MLPs at subject-last positions *do* contribute to promoting correct attributes. This is more nuanced than the present paper's binary "attention brings knowledge, MLP just routes" story. The present paper's §5.6 ("The Routing Is the Contribution") should engage with this.

**6. Elhage et al. 2021/2022 — "A Mathematical Framework for Transformer Circuits" and "Toy Models of Superposition" — not cited**
These are foundational Anthropic/Transformer Circuits papers. The mathematical framework paper introduces the key-value/output-value decomposition of attention heads and the concept of residual stream communication that the present paper relies on implicitly. "Toy Models" introduces superposition, which is directly relevant to why the ~3,040 residual neurons are "irreducibly distributed." Not citing these in a mech interp paper is like not citing Vaswani in a transformer paper.

**7. Wang et al. 2023 "Interpretability in the Wild" (IOI paper) — not cited**
This paper provides a complete circuit-level account of a behavior in GPT-2 Small — the same model studied here. It's the closest precedent for what this paper attempts (full legible decomposition) and should be discussed, either as a point of comparison (they traced a circuit; we traced an entire layer) or contrast (they focused on attention; we focus on MLP).

**8. Conmy et al. 2023 "Towards Automated Circuit Discovery" (ACDC) — not cited**
Relevant to methodology. The paper manually identifies 27 named neurons; ACDC provides automated methods for the same task. A reviewer may ask why automated discovery wasn't used. At minimum, cite and briefly note the methodological choice.

**9. Bills et al. 2023 (OpenAI) "Language Models Can Explain Neurons" — not cited**
Directly relevant to neuron characterization methodology. The present paper characterizes neurons via enrichment/depletion analysis; Bills et al. uses GPT-4 to generate neuron descriptions. Methodological comparison would strengthen §3.

### 🟢 MINOR

**10. Tenney et al. 2019 "BERT Rediscovers the Classical NLP Pipeline"**
Relevant to the "terminal crystallization" claim (§6). Tenney showed layer-wise specialization in BERT (POS → syntax → semantics). The present paper's developmental arc (scaffold → decision → terminal) could be compared to this, even acknowledging the BERT/GPT difference.

**11. Clark et al. 2019 "What Does BERT Look At?"**
Early attention head analysis work, relevant to the L11H7 characterization in §6.3. Minor but would round out the attention-related citations.

**12. Self-citation balance**
The paper cites Balogh 2026 (the companion paper) appropriately and builds on it clearly. No concern here — but the *ratio* of self-citation to external citation (1:7) is skewed only because the total citation count is so low. Adding 15–20 more external citations would make the self-citation ratio unremarkable.

### Summary

The most damaging omission is Dai et al. 2022 (item 1) — the paper makes a strong negative claim ("MLP does not retrieve facts") while ignoring a published paper claiming the opposite. Combined with the unaddressed tension with ROME (item 2), this creates a vulnerability where a reviewer could reasonably reject on grounds of insufficient engagement with prior work. The thin reference list (item 3) compounds this: 8 citations for a NeurIPS paper signals incomplete scholarship.

Beyond the critical gaps, the logit lens bias issue (item 4) could undermine Table 6's developmental arc, and the missing Transformer Circuits foundations (item 6) are conspicuous for a mechanistic interpretability paper.

**Priority fixes:** Item 1 (cite and engage with Dai et al.), item 2 (address ROME tension), item 3 (expand references to 25+ minimum), item 6 (add Elhage et al.), item 4 (acknowledge or mitigate logit lens bias), item 5 (engage with Geva 2023).

---

## 2026-03-16 — Pass 4: Writing Quality

Focus: Awkward phrasing, unclear sentences, AI-detectable patterns, register consistency, readability.

### 🔴 CRITICAL

**1. The abstract buries the lede**
The abstract opens with method ("We decompose the final MLP layer…") rather than the finding. The actual headline — that an MLP layer implements a legible routing program readable as pseudocode — doesn't land until the reader parses a dense 6-line sentence. Suggested rewrite pattern: lead with what you found ("The final MLP of GPT-2 Small implements a fully legible routing program…"), then how you found it, then what it means. Compare: the current abstract's first sentence has 23 words before the reader encounters a concrete claim ("a legible routing program"). That's too long for the hook.

**2. Overuse of em-dashes and parenthetical interruptions**
The paper uses em-dashes (—) 0 times but uses long parenthetical asides constantly. Examples:
- "the MLP's primary contribution is a binary routing decision: intervene (exception path) or abstain (consensus path)" — the parentheticals break the rhythm of what should be a clean, punchy statement.
- "27 named neurons with documented behavior readable as pseudocode" — "readable as pseudocode" dangles. Should be "…documented behavior, readable as pseudocode" or restructured.
- §4.2: "Its input sensitivity (from W_fc) responds to…" — "(from W_fc)" could be a footnote or moved to Methods; it interrupts a characterization paragraph.

This pattern is consistent throughout. A pass to remove or restructure 50%+ of parentheticals would tighten the prose significantly.

**3. AI-detectable phrasing patterns**
Several constructions read as LLM-generated prose:
- "This is analogous to reading a software system's control flow without access to its database" (§7.1) — the analogy is fine but the construction "This is analogous to X" is a GPT tell. Try: "Think of it as reading a program's control flow without its database."
- "yielding the structured architecture we observe" (§6.2) — "yielding" + "we observe" is a classic hedging pattern. Try: "which produces the structured architecture."
- "consistent with the attention sink phenomenon" (§6.3) — the word "consistent" appears 5 times in the paper, always in "consistent with X." This is a citation-hedging reflex. Vary: "matches," "aligns with," "as predicted by," or simply "this is the attention sink phenomenon."
- "Darkness visible: we can see the structure that navigates the dark, even when the dark itself cannot be illuminated" (final sentence) — this tries too hard. The Milton reference in the title and intro is effective; repeating it as the closing line with an extended metaphor feels forced. Consider ending on the concrete: "We can read 27 neurons as a routing program. The 3,040 neurons they route remain opaque."

### 🟡 MODERATE

**4. Inconsistent register: oscillates between vivid and flat**
The paper has some genuinely good writing ("buying time," "booster rocket," "the dark itself cannot be illuminated") mixed with passages of dry, mechanical prose ("We compute the following metrics for each neuron…"). This inconsistency is jarring. The vivid passages work well — lean into them more consistently, or cut the metaphors and go fully technical. The current mix reads like two authors (or an author and an LLM editor).

**5. "We show / we find / we demonstrate" — excessive hedging**
Counted 14 instances of "we show," "we find," "we demonstrate," or "we observe" across the paper. Some are fine (topic sentences in results sections). But many could be cut or replaced:
- "We show that the MLP does not retrieve factual knowledge" → "The MLP does not retrieve factual knowledge"
- "We find that L11's most important attention head…" → "L11's most important attention head…"
The paper would read more confidently with ~50% of these removed. The findings should feel like facts, not observations.

**6. Table/figure reference density is uneven**
§4 has 3 table references in 2 pages. §5 has 5 table references in 3 pages. But §6 has only 1 table reference and relies heavily on inline numbers. The inline numbers in §6.3 ("45.4%," "47.0% vs 37.3%," "3.01 vs 3.51 bits") should probably be a small table — they're hard to parse in running text and easy to misread.

**7. The phrase "irreducibly distributed" is used 3 times**
Abstract, §4.3, and implicitly in §7.1. It's a good phrase the first time. By the third, it feels like a catchphrase being hammered. Use it once (abstract or conclusion) and find alternatives elsewhere: "entangled across thousands of neurons," "not decomposable into named units," etc.

**8. §3 (Methods) is too terse for reproducibility**
§3.1 describes the transparent forward pass in 6 lines. §3.2 describes the dataset and neuron characterization in 5 lines. §3.4 describes consensus profiling in 4 lines. These are barely method sketches. A reviewer wanting to reproduce the enrichment analysis, the co-firing measurement, or the progressive prediction experiments would need to reverse-engineer from results. Either expand §3 or add an appendix with full methodological details.

### 🟢 MINOR

**9. "Percentage-point" vs "%" inconsistency**
§2 uses "94.3 percentage-point drop" (correct for a difference). §5.3 uses "+14.5pp" and "−7.1pp" (abbreviation, less formal but clear). §5.4 uses "+8.0 percentage points." Pick one convention and stick with it. Suggest: spell out "percentage points" in prose, use "pp" only in tables.

**10. The Milton epigraph works well**
The title ("Darkness Visible") and the brief explication in §1 are effective and non-gimmicky. The allusion gives the paper a memorable identity. Keep as-is (but see item 3 above re: the closing line).

**11. Missing article before "Layer 11" in some places**
"At Layer 11" (correct) vs "L11 has no safety net" (reads as abbreviation, fine). But "Layer 11 of GPT-2 Small, the model's final MLP" — "Layer 11" here refers to the layer, but the appositive says it's "the model's final MLP." Layer 11 *contains* the final MLP; it also contains attention heads. Minor imprecision but a careful reader will notice.

**12. "~3,040 residual neurons" — explain the approximation**
3,072 total − 27 named − 5 overlap (Core counted once) = 3,040? The arithmetic isn't shown. A parenthetical "(3,072 − 27 named = 3,045 remaining…)" or similar would remove the ambiguity. The "~" suggests uncertainty about the count of your own categorization.

### Summary

The writing is generally strong — better than most mech interp papers — but has two main vulnerabilities: (1) AI-detectable phrasing patterns that a savvy reviewer will notice, and (2) a register inconsistency that undermines the paper's voice. The abstract needs restructuring to lead with the finding. The "consistent with" reflex and excessive "we show/find" hedging are the most pervasive stylistic issues. The closing line tries too hard; end on concrete results, not extended metaphor.

**Priority fixes:** Item 1 (restructure abstract), item 3 (scrub AI-tells), item 5 (reduce hedging), item 2 (thin parentheticals), item 8 (expand Methods for reproducibility).

---

## 2026-03-17 — Pass 5: Methodology

Focus: Experimental design issues, confounds, missing controls, reproducibility gaps.

### 🔴 CRITICAL

**1. Binary firing threshold (|GELU| > 0.1) is load-bearing but unjustified**
The *entire* architecture — tiers, consensus levels, Jaccard similarities, fire rates — depends on binarizing GELU activations at 0.1. §7 acknowledges this is "a modeling choice" and points to robustness analysis in Balogh 2026, but that's a different paper. *This* paper should demonstrate threshold robustness for its own claims. Specifically: do the three tiers (Core/Differentiator/Specialist) survive at thresholds 0.01 and 1.0? Do the Jaccard similarities in Table 1 hold? A reviewer can reasonably ask: "What if the tier structure is an artifact of the threshold?" Without a sensitivity analysis table (even in appendix), this is a methodological hole.

**2. Neuron characterization method is underspecified**
§3.2 says "binary firing is determined by |GELU(x_n)| > 0.1" and "co-firing structure is measured by Jaccard similarity." But the *enrichment analysis* that drives §4–5 (which tokens are overrepresented when a neuron fires) has no methodological description at all. What's the statistical test? Is it a simple ratio (fire-rate-for-token / base-fire-rate)? Is there a significance threshold? How are ties broken? The consensus neuron profiles in Table 2 list specific tokens ("fires on `and`, `but`, `also`") — how were these selected from what must be thousands of enriched tokens? Without this, the neuron characterizations look like cherry-picked anecdotes rather than systematic analysis.

**3. No control model / baseline for the architecture claims**
The paper claims L11's exception handler is a *discovered* architecture, not a statistical artifact. But there's no null model. What does the same analysis produce on: (a) a randomly initialized GPT-2 with the same architecture? (b) a permutation baseline where neuron activations are shuffled across tokens? (c) a different model (GPT-2 Medium, GPT-Neo) at the final layer? Without any control, a reviewer can argue that *any* MLP layer analyzed this way would produce apparent tier structure — neurons naturally vary in fire rate, and binning by fire rate will always produce "tiers." The L7/L10 replication (§6.1) partially addresses this (those layers lack structure), but that's the same model, and the paper admits they were chosen because they have "consensus structure" from the prior paper. A random-init control would be much stronger.

### 🟡 MODERATE

**4. WikiText-103 as sole dataset introduces domain confound**
All experiments use WikiText-103 (Wikipedia text). The consensus neuron characterizations ("fires on clausal continuation," "fires on concrete reference") are derived entirely from encyclopedic prose. Would the same characterizations hold on: dialogue? Code? Poetry? Legal text? The paper doesn't claim domain generality, but the framing ("seven dimensions of normal," "linguistic predictability") implies it. A brief sanity check on even one alternate domain (e.g., 10K tokens of OpenWebText or BookCorpus) would significantly strengthen or appropriately scope the claims. Without it, "normal language" really means "normal Wikipedia."

**5. Progressive prediction experiment (§4.3) lacks systematic evaluation**
The "In 1969, astronauts landed on the ___" walkthrough is a single illustrative example. The paper presents it as demonstration, not evidence, but readers will treat it as evidence for the Core-as-vocabulary-reset claim. How many of the 12 factual prompts show this tier-by-tier pattern? Is it always Core→reset→Diff→suppress→Residual→restore? If even 2 of 12 prompts deviate, the narrative weakens. This should be systematized: for all 12 prompts, report the model's top-1 prediction after each tier is added.

**6. Knowledge extraction test (§5.1) has a flawed premise**
The test accumulates residual neuron *output directions* (columns of W_proj projected through unembedding) one at a time and checks if the target rank improves. But this ignores context: a neuron's output direction is its *static* vocabulary contribution, while its *actual* contribution depends on activation magnitude, which depends on the input. Sorting by "contribution magnitude" (§3.5) partially addresses this, but the paper doesn't specify whether magnitudes are computed for the specific prompt or averaged. If averaged, the test is checking whether generic neuron directions reconstruct facts — which of course they don't. If prompt-specific, the test is more meaningful but still ignores interaction effects (the output directions aren't independent; their sum produces the actual MLP output, which we already know is correct). The "20 Questions" framing is vivid but the experiment may be testing the wrong thing.

**7. Activation transplant experiment (§5.4, Exp. 3) has tiny N and no quantitative threshold**
5 country-capital pairs is an extremely small sample. "Source-fact probability changed by <0.02pp" — is 0.02pp the right threshold for "no effect"? What's the variance? With N=5, even a suggestive trend would be invisible. This experiment is interesting but statistically underpowered. Either expand to 50+ pairs or present it as "illustrative" rather than "evidence."

**8. Attention head importance metric (§6.3) — still undefined despite Pass 1 flagging**
Pass 1 (item 4) flagged that the "importance score" for L11H7 is never defined. This is a *methodology* issue, not just a reporting issue: without knowing how importance is computed, the claim that L11H7 is "6× the importance of any other head" is unverifiable. Is this norm of the OV matrix? Mean absolute attention weight? Gradient-based attribution? Different metrics could give different rankings. This needs to be in §3.

### 🟢 MINOR

**9. 500 sequences × 1024 tokens: no description of how sequences are selected**
§3.2 says "500 sequences of 1,024 tokens each" from WikiText-103. Are these the first 500? Random? Stratified by article length? WikiText-103 has articles of varying quality and length; selection method matters for representativeness.

**10. No held-out validation set**
All analysis and all evaluation use the same 512K tokens. The consensus-helpfulness crossover (Table 4), the ablation results (Table 5), and the neuron characterizations are all computed on the same data. There's no train/test or even train/validation split. For descriptive/exploratory analysis this is acceptable, but the paper makes quantitative claims (crossover at 3–4/7, PPL impacts) that could be inflated by overfitting the analysis to the dataset. A split — characterize on 256K, validate on 256K — would cost nothing and add credibility.

**11. Jaccard similarity as co-firing metric**
Jaccard is appropriate for binary vectors but sensitive to base rate: two neurons that both fire on 90% of tokens will have high Jaccard regardless of actual co-occurrence patterns. The Core neurons (90–100% conditional fire rate) are exactly in this regime. The 0.998 Jaccard between N2123 and N2910 is impressive but should be contextualized: what Jaccard would two independent neurons with 90% fire rates produce? (Answer: ~0.81.) The 0.998 is still well above this, but the paper should note the base-rate issue to preempt the criticism.

**12. Cross-layer replication only at 2 other layers**
§6.1 tests L7 and L10 — chosen because the prior paper identified them as having consensus structure. But GPT-2 Small has 12 layers. Testing only layers pre-selected for having *some* structure biases toward finding "L11 is special." What about L0–L6, L8–L9? Even a quick fire-rate and max-Jaccard survey across all 12 layers would show whether L11 is truly unique or just the strongest of a gradient.

### Summary

The biggest methodological vulnerability is the absence of a null model (item 3). The entire paper's architecture story could be an artifact of the analysis method applied to any MLP layer — and without a random-init or permutation control, there's no way to rule this out. The L7/L10 comparison helps but doesn't fully address this because those layers were selected, not random.

The threshold sensitivity (item 1) and underspecified enrichment analysis (item 2) are compounding issues: the neuron characterizations are the paper's core contribution, and both the binarization method and the characterization method lack sufficient detail for a reviewer to assess or reproduce.

The knowledge extraction test's flawed premise (item 6) is subtle but important — if the experiment is testing the wrong thing, the strong negative conclusion ("MLP does not retrieve facts") rests on weaker ground than presented.

**Priority fixes:** Item 3 (add null model/control), item 1 (threshold sensitivity analysis), item 2 (specify enrichment methodology), item 6 (reconsider or reframe knowledge extraction test), item 4 (test on non-Wikipedia data).

---

## 2026-03-18 — Pass 6: Figures/Tables

Focus: Are tables clear, necessary, correctly referenced, and do they support the claims being made?

### 🔴 CRITICAL

**1. Table 2 (tab:twenty_q) is the paper's Achilles heel**
This table is meant to be the key evidence for "the MLP does not retrieve facts" (§5.1), but it's a disaster:
- **Only 6 of 12 prompts shown** — already flagged in Pass 1 (item 5), but examining the table now: the missing 6 prompts could completely change the interpretation. If 6 more prompts showed successful retrieval, the "negative result" narrative collapses.
- **No clear split between "high confidence" and "genuinely difficult"** — the text groups prompts into these categories (lines after the table), but the table has no visual separation, no bolding, nothing. A reader scanning the table cannot tell which 6 succeeded and which 6 failed without re-reading the prose.
- **Column headers are confusing** — "Static (all)" vs "Context (all)" — what does "all" mean? All 3,040 residual neurons? The distinction between "50" and "All" is never explained in the caption or table notes.
- **Target ranks in the tens of thousands are unreadable** — "15,666", "45,230", "13,339" — these numbers mean "buried so deep it's hopeless," but a reader has to mentally convert rank to percentile. Add a column: "Percentile" or "Top-X%" to make the failure intuitive.
- **The table contradicts the text's split** — Text says context-dependent retrieval works for high-confidence facts (rank 1) but fails for difficult ones (rank 5,515+). But looking at the table: "da → Vinci" gets rank 9 (borderline), "Wall of → China" gets rank 332 (moderate failure). These don't fit the binary "works vs. fails" narrative. The table shows a *gradient*, not a split.

**RECOMMENDATION:** Either show all 12 prompts (move to full-page table or appendix), add visual grouping (horizontal rule between categories), and add a percentile column — OR replace the table with aggregate statistics (median rank, IQR, success rate at top-10) and move the full table to appendix.

**2. Table 9 (tab:cross_layer) is too sparse to support "terminal crystallization"**
The table claims L7 and L10 lack structure while L11 has it, but the evidence is thin:
- Only 5 metrics shown (exception fire rate, enrichment, specialists, max Jaccard, consensus neurons)
- "~1.0×" and "~0.7" are imprecise approximations — actual numbers needed
- No comparison of *within-layer variance* — L7 might have 0 specialists but high variance in some other structural metric not shown
- Missing key metrics that would strengthen the claim: e.g., tier count, output norm concentration, consensus-exception anticorrelation strength

The narrative claims L11 undergoes "terminal crystallization" — a developmental phase shift — but the table shows scalar differences, not qualitative ones. A 44.5% exception rate (L7) vs 11.3% (L11) is a 4× difference, but is that crystallization or just refinement?

**RECOMMENDATION:** Expand to 8+ metrics, add L0, L3, L6, L8, L9 as spot checks (currently only 3 of 12 layers shown), compute actual numbers not "~", and add a column showing whether each metric is *significantly different* from L11 (statistical test).

**3. Table 1 (tab:tiers) lists "N2462, N2173, N1602, N1800, N2379, …" with ellipsis**
The Differentiator tier has "10" neurons but only 5 are named, with "…" standing in for the other 5. This is unacceptable for a table that's supposed to document the architecture. Either list all 10 (it's just neuron IDs, not that much space) or say "N2462, N2173, +8 others" with a footnote. The ellipsis makes the table look incomplete, like the authors didn't bother to finish it.

**4. No visual representations (figures) at all**
The paper describes:
- A three-tier architecture (Fig: stacked bars showing Core/Diff/Spec output norm contribution)
- Consensus-exception anticorrelation (Fig: line plot of exception rate vs. consensus level 0-7)
- Progressive prediction tier-by-tier (Fig: waterfall chart showing prediction shift after each tier)
- Logit lens developmental arc (Fig: line plot of top-1 accuracy across layers, with phases annotated)
- Consensus neuron output directions in embedding space (Fig: 2D PCA showing N2600 orthogonal to the other 6)

All of these are described *in words* when they should be *shown visually*. A NeurIPS paper with 0 figures will stand out negatively. The "three-phase developmental arc" in Table 6 is a perfect candidate: the phases (scaffold, decision, terminal) are invisible in a table of numbers but would be obvious in a line plot.

**RECOMMENDATION:** Add at minimum 3 figures: (1) exception rate vs consensus level, (2) logit lens accuracy curve with phases, (3) tier contribution breakdown (stacked bar or pie).

### 🟡 MODERATE

**5. Table 7 (tab:enrichment_stats) is in §7 (Controls) but never forward-referenced**
The enrichment statistics are core methodology — they determine which neurons are "significantly enriched" under exception firing. But the table appears in the *appendix-like Controls section*, not in §4 where the tiers are introduced. A reader encountering "significantly enriched" in §4 has no idea what test, what threshold, what p-value. The table should be in §3 (Methods) or §4 (Results), with a brief version in main text and full version in appendix if space-constrained.

**6. Table 3 (tab:consensus_levels) rounds percentages inconsistently**
0.1%, 1.7%, 8.9%, 49.0%, 40.3% — the first three have 1 decimal place, the last two have 1 decimal place. But looking at the "Tokens" column: 486/512000 = 0.0949% ≈ 0.09% (not 0.1%). The rounding is inconsistent between columns. Either round all to 1 decimal or explain the rounding rule in a note.

**7. Table 4 (tab:consensus_help) buries the crossover in the middle**
The most important row — where ΔP crosses from positive to negative (the 3-4/7 crossover) — is visually unmarked. Add a horizontal rule or bold the 3/7 and 4/7 rows to highlight the transition. Currently the reader has to scan the ΔP column to find the crossover; it should jump off the page.

**8. Table 5 (tab:ablation) shows "−0.3%" for Specialists but doesn't flag the sign**
Negative PPL delta means *improvement* when the tier is removed — the Specialists are *counterproductive*. This is buried in a cell. Either bold it, add a footnote marker (†), or add a "Direction" column (Harmful/Beneficial). The paper mentions this in passing ("slightly improve PPL when ablated") but it deserves more prominence — 5 named neurons are making the model worse!

**9. Table 6 (tab:logit_lens) has a "Never" row at 54.1%**
This means 54.1% of tokens *never* reach top-1 across any layer. That's a striking finding (majority of predictions are always wrong!), but it's presented as a table footnote. This deserves a sentence in the text: "Even at the final layer, only 45.9% of tokens are correctly predicted as top-1 — the remaining 54.1% never reach the correct answer."

**10. Table 2 (tab:consensus) has a 4.5cm paragraph column**
The "Key Evidence" column has `p{4.5cm}` width and contains dense prose ("Fires on ordinals, lists, parallel structure; highest mean activation (0.79)"). This is hard to read in a table — consider reformatting as bullet points, shortening to keywords only ("ordinals, lists, parallel"), or moving the detailed evidence to a supplementary table and keeping only dimension labels here.

### 🟢 MINOR

**11. Table 8 (tab:knowledge_neurons) has only 2 data rows**
"Consensus neurons (7)" and "All routing neurons (27)" — this could be a sentence ("Consensus neurons appear in 2.6 of 20 top-attributed neurons on average (52× over chance); all routing neurons appear in 7.3 (36.5×)"). It doesn't need to be a table. Consider in-text presentation.

**12. Table captions are inconsistent in detail level**
Table 1: terse ("Fire rates are conditional on N2123 firing")
Table 2: verbose (full sentences explaining dimension derivation)
Table 6: medium (explains what "First lock-in" means)
Pick a style and stick with it. NeurIPS style tends toward concise captions with details in text.

**13. Missing table: "Universal consensus breakers"**
§5.2 mentions "universal consensus breakers — tokens depleted by nearly all 7 neurons" and lists 5 examples (\n\n, j, Tru, -, @) in prose. This should be a small table showing: Token | Avg consensus fire rate | N depleted by. Would take 4 lines and make the claim concrete.

**14. Table 10 (tab:knowledge_neurons) title doesn't match content**
Caption says "Integrated gradients attribution overlap with routing neurons" but the table shows "In top-20" vs "Expected by chance" — the word "overlap" suggests intersection counts, but it's actually enrichment ratios. Retitle: "Routing neuron enrichment in top-20 attributed neurons (integrated gradients, 20 prompts)."

**15. No table for the "illustrative case studies" (§5.2)**
Text describes three modes of factual emergence with specific examples ("Abraham → Lincoln locks in at L0", "moon overtakes planet at L10"). These are mini case studies that should be a table: Prompt | Target | Lock-in layer | Mode | Probability at lock-in. Would support the qualitative categorization.

### Summary

The paper has **no figures** — a major weakness for a NeurIPS submission claiming to reveal legible architecture (which is inherently visual). Table 2 (the knowledge retrieval test) is the most problematic: incomplete data (6/12 prompts), confusing headers, and a claimed binary split (success/failure) that the actual numbers don't support. Table 9 (cross-layer) is too sparse to substantiate "terminal crystallization." Table 1 uses ellipsis for named neurons, which looks sloppy.

Three high-impact additions: (1) a figure showing exception rate vs consensus level (the paper's core finding visualized), (2) a figure showing the logit lens developmental arc, (3) expanding or replacing Table 2 to show all 12 prompts with clearer categorization.

Several tables could be demoted to in-text prose (Table 8, Table 10) or reorganized (Table 7 to Methods section).

**Priority fixes:** Item 1 (fix/expand Table 2), item 4 (add 2-3 figures), item 2 (expand Table 9 cross-layer comparison), item 3 (complete Table 1 neuron list), item 5 (move enrichment stats to Methods).

---

## 2026-03-19 — Pass 7: Abstract/Intro/Conclusion Alignment

Focus: Do the abstract, introduction, and conclusion accurately reflect what the paper actually demonstrates? Are contributions overclaimed or underclaimed? Is there drift between the promise and the delivery?

**Context note:** The paper has been substantially revised since Passes 1–6. Many previously flagged issues are now addressed: abstract leads with finding, Dai et al. engaged, ROME tension addressed, tuned lens validation added, figures included, Table 2 expanded to 160 prompts with category breakdown, bootstrap CIs on crossover, null model control, threshold robustness, cross-layer survey expanded to all 12 layers, enrichment stats specified with Fisher's exact test + Bonferroni. This pass reviews the *current* state.

### 🔴 CRITICAL

**1. Abstract claims "to floating-point precision; cosine similarity 0.99999994" — but this is the *decomposition*, not a *finding***
The abstract's second sentence mixes method (the decomposition preserves weights) with results (what the decomposition reveals). The cosine similarity number is about numerical fidelity of the implementation, not about the scientific claim. It reads like padding — as if the precision of floating-point arithmetic is a contribution. A reviewer will notice this is a trivially achievable property of any additive decomposition (you're just partitioning a sum). Move this to Methods or demote it to a parenthetical. The abstract's prime real estate should be reserved for non-trivial findings.

**2. Abstract lists 6 contributions but §1 (Intro) lists the same 6 — exact duplication**
The abstract's enumerated structure (routing indicator, consensus detector, three-tier exception handler, negative knowledge result, negative garden-path result, terminal crystallization) is repeated almost verbatim in the introduction's numbered list. This is wasted space: a reader going abstract → intro encounters the same claims twice before seeing any evidence. The intro should *expand and contextualize* the abstract's claims, not restate them. For instance, the intro could briefly say *why* each finding matters — what prior assumption it overturns, what it enables — rather than re-listing the same descriptors.

**3. Conclusion undersells the garden-path finding**
The conclusion compresses the garden-path result into a single sentence: "The exception handler detects token-level predictability… but not syntactic reparse." But this is arguably the paper's most surprising and novel finding — GPT-2 has a *reversed* garden-path effect, parsing intransitive verbs correctly from the start. This contradicts a large body of psycholinguistic work on incremental parsing and suggests a fundamental difference between transformer and human sentence processing. The conclusion should give this at least a full paragraph. It's the finding most likely to generate discussion and citations.

### 🟡 MODERATE

**4. Abstract's "27 named neurons whose behavior can be read as pseudocode" — the pseudocode is never shown**
The abstract and intro both promise neurons "readable as pseudocode." This is a vivid, compelling claim — and the paper never delivers actual pseudocode. The neuron characterizations are in natural language (Table 2: "Fires on `and`, `but`, `also` mid-clause; silent at clause boundaries"). That's a description, not pseudocode. Either (a) add a figure or listing showing the actual pseudocode interpretation (e.g., `if consensus < 5/7: activate_core(); suppress_candidates(); ...`), or (b) soften the claim to "readable as a routing program" or "characterizable in functional terms." "Pseudocode" sets an expectation of formal, algorithmic description that the paper doesn't meet.

**5. Introduction's "No light, but rather darkness visible" epigraph is effective but the Milton thread frays**
The title, epigraph, and intro's "structured darkness" passage work beautifully. But the paper then uses Milton inconsistently: §5.3 introduces "Belial's Counsel" (a section title with its own Milton explication), and §5.3 introduces "Moloch" in passing. These are different characters with different meanings in Paradise Lost, and the paper doesn't sustain the conceit. Belial = persuasive but misleading (knowledge neurons); Moloch = brute force even when counterproductive (MLP at consensus). The mapping is clever but not explained enough for readers unfamiliar with Milton. Suggestion: either commit to the extended metaphor (add a sentence explaining each character's relevance at point of use) or drop Moloch/Belial and keep only the "darkness visible" thread. Currently it reads like the author is showing off literary knowledge without integrating it into the argument.

**6. Conclusion doesn't address the "so what" for practitioners**
The conclusion restates findings but doesn't discuss implications. Three practical implications are buried in §7 (Discussion) but missing from the conclusion:
- *Efficiency*: bypassing MLP for high-consensus tokens could save compute — this is quantified (6.9% PPL cost for 40% token savings) but not in the conclusion
- *Interpretability methodology*: the finding that knowledge neurons are routing infrastructure has implications for ROME, model editing, and attribution methods — worth a concluding sentence
- *Scaling prediction*: terminal crystallization makes a testable prediction about deeper models — the conclusion should end on this forward-looking note

**7. Abstract's "all eight levels exclude zero" — readers won't understand without context**
The abstract says "bootstrap 95% CIs, all eight levels exclude zero" for the consensus-exception crossover. This is technically correct but a reader encountering this in the abstract has no framework for "eight levels" (consensus 0/7 through 7/7). It reads as a statistical incantation rather than a meaningful summary. Suggestion: "the crossover is statistically sharp (all bootstrap 95% CIs exclude zero)" — simpler, and the "eight levels" detail can stay in the results.

### 🟢 MINOR

**8. Intro's "A single exception neuron (N2123, 11.3% fire rate)" — too much implementation detail for the intro**
Neuron IDs and fire rates belong in Results, not the Introduction. The intro should frame the *conceptual* finding (a binary routing indicator) and leave the specifics for later. "A single neuron reliably indicates which processing path is active" is sufficient for the intro.

**9. Abstract's sentence structure: the first sentence is now 26 words and lands well**
"The final MLP of GPT-2 Small implements a fully legible routing program — 27 named neurons whose behavior can be read as pseudocode" — strong opening. This addresses Pass 4's item 1 (abstract buried the lede). Good fix.

**10. Conclusion's final line is now concrete and effective**
"We can read 27 neurons as a routing program. The 3,040 neurons they route remain opaque." This addresses Pass 4's item 3 (closing line tried too hard with extended metaphor). Much better than the previous version. Keep.

**11. "Terminal crystallization" is introduced in the abstract without definition**
The term first appears in the abstract's final contribution bullet: "terminal crystallization, with a testable prediction." But what terminal crystallization *means* isn't clear until §6. The abstract should include a gloss: "terminal crystallization — the concentration of legible routing at the model's final layer."

**12. Intro's enumerated list uses inconsistent formatting**
Items 1–3 are bold-titled ("Binary routing indicator:", "Consensus detector:", etc.). Items 4–6 use different bolding patterns ("Negative result on MLP knowledge retrieval" vs "Terminal crystallization"). Minor but a reviewer scanning the list will notice the inconsistency.

### Summary

The abstract and intro are much stronger than pre-revision — the lede is no longer buried, the closing line works, and the contributions are clearly enumerated. The main remaining issues: (1) the "pseudocode" promise is never fulfilled — either deliver it or soften the claim; (2) the abstract and intro duplicate each other rather than building on each other; (3) the conclusion undersells the garden-path finding and lacks practical implications; (4) the Milton conceit (Belial, Moloch) is evocative but not fully integrated.

The garden-path result deserves more concluding prominence — it's the finding most likely to attract attention from the psycholinguistics and cognitive science communities, and burying it in one sentence of the conclusion is a missed opportunity.

**Priority fixes:** Item 4 (pseudocode promise), item 3 (expand garden-path in conclusion), item 2 (differentiate intro from abstract), item 6 (add implications to conclusion).

---

## 2026-03-20 — Pass 8: Statistical Claims (Second Round, Post-Revision)

Focus: Re-examine statistical claims in the revised paper. The paper now has bootstrap CIs, 160-prompt knowledge test, Fisher's exact + Bonferroni enrichment, null model, threshold robustness, tuned lens, and cross-layer survey of all 12 layers. Are the NEW statistical claims defensible?

### 🔴 CRITICAL

**1. Table 7 (enrichment stats): N2123 base rate 70.9% but earlier text says 11.3% fire rate**
§4.2 says "N2123 fires on 11.3% of tokens." Table 7 (§7.1) says N2123's base rate is 70.9% and exception rate is 100.0%. These cannot both be right. One is likely the *conditional* fire rate under exception (which is definitionally 100%), and the other is the unconditional rate — but 70.9% and 11.3% are drastically different numbers. The 11.3% figure is consistent with the rest of the paper (Table 3 shows low-consensus tokens are ~10% of data). The 70.9% in Table 7 appears to be using a different definition of "fires" — possibly |GELU| > 0.1 (which gives a higher base rate than the exception-path definition). If so, the paper is conflating two different notions of "firing" without flagging the distinction. This is a **serious inconsistency** that undermines the enrichment analysis. A reviewer will notice that N2123's enrichment ratio of 1.41× (from 70.9% to 100%) is modest, while the rest of the paper frames exception firing as a rare event (11.3%). These are apparently different things, and the paper never says so.

**Likely explanation:** The 11.3% is the rate at which N2123 fires in its role as *exception indicator* (i.e., high activation, perhaps above a higher threshold or defined by the consensus-exception partition). The 70.9% is the rate at which |GELU(N2123)| > 0.1, the generic "any activation" threshold. If this is correct, the paper has two overlapping definitions of "N2123 fires" and needs to distinguish them explicitly — perhaps "N2123 activates" (70.9%, generic) vs. "exception path active" (11.3%, routing-relevant). Without this clarification, the enrichment analysis in Table 7 is testing the wrong condition.

**2. Garden-path statistics are underpowered for the claim being made**
§6.6 reports $t(14) = -0.45, p = 0.66$ for the N2123 activation difference between intransitive and transitive conditions. With 15 pairs and a non-significant result, the paper concludes the exception handler "does not detect syntactic reparse." But a non-significant result is not evidence of absence — especially at $N = 15$. The observed effect size ($d = 0.102 - 0.105 = -0.003$) is tiny, but what power does this test have to detect a small-to-medium effect? At $N = 15$, power to detect $d = 0.5$ is approximately 0.34 — the test would miss a moderate effect 66% of the time. The paper should either: (a) report a power analysis showing the test could detect a meaningful effect, (b) use equivalence testing (TOST) to positively establish that the effect is negligibly small, or (c) soften the claim to "we found no evidence of" rather than "the exception handler does not detect." Currently the paper over-interprets a null result.

**3. "36.5× enrichment over chance" (§5.4, Table 8) — the chance baseline is wrong**
The paper says expected overlap is $27/3072 × 20 = 0.18$ routing neurons in any top-20 set. But the top-20 integrated-gradient neurons are selected *because they have high causal influence on the output*. Routing neurons — by definition — have high causal influence on the output. So the "chance" baseline (uniform random from 3,072) is not the right null. The proper comparison would be: among the top-20 neurons by *any* high-activation metric (e.g., highest absolute GELU activation), how many are routing neurons? If high-activation neurons generally overlap with routing neurons, the 36.5× enrichment might shrink dramatically. This doesn't invalidate the finding but the comparison to uniform chance inflates the reported enrichment.

### 🟡 MODERATE

**4. Bootstrap CIs on crossover (Table 4) — unclear what's resampled**
Table 4 reports "10,000 bootstrap resamples" with CIs that all exclude zero. But bootstrap of *what*? Are you resampling tokens within each consensus level? Sequences? The bootstrap unit matters: if individual tokens are resampled but tokens within a sequence are autocorrelated (which they are — consecutive tokens share context), the CIs will be too narrow. Sequence-level or block bootstrap would be more conservative and appropriate. The paper should state the resampling unit.

**5. Knowledge test: "median rank 1,905 across all 160 prompts" — misleading summary**
Table 2's bottom panel shows median ranks ranging from 19 (historical events) to 27,124 (languages). The overall median of 1,905 averages over wildly heterogeneous categories. This single number obscures the bimodal distribution: some categories succeed (historical events: 4/10 top-10), others fail completely (capitals: 0/15 top-10). The overall median is dominated by the large number of failing categories and gives a misleading sense of uniform failure. Better summary: "Retrieval succeeds for highly constrained completions (4/10 historical events reach top-10) but fails for arbitrary associations (0/15 capitals). The overall top-10 rate is 11%."

**6. Ablation PPL changes (Table 5) — no confidence intervals or significance tests**
Core ablation: +0.2% PPL. Specialists: −0.3% PPL. Are these significant? PPL is computed over 204,800 tokens, so the SEs should be estimable. A +0.2% change in PPL (31.79 → 31.86) could easily be within noise, especially given that PPL is sensitive to a small number of high-loss tokens. The Specialist result (−0.3%, i.e., *improvement*) is particularly important to validate — if it's within noise, the "Specialists are counterproductive" claim falls apart. Add bootstrap CIs or paired-sample tests (PPL with vs. without ablation, resampled at sequence level).

**7. Enrichment p-values "< 10^{-300}" (Table 7) — all six significant neurons**
Reporting $p < 10^{-300}$ for Fisher's exact test is technically correct but raises questions. With 512,000 tokens, Fisher's exact test on a 2×2 table will produce astronomically small p-values for even tiny enrichment ratios (e.g., N611: 67.2% → 71.8%, enrichment 1.07×). A 4.6 percentage-point difference with $N = 512K$ is "significant" by any test but practically negligible. The paper should report effect sizes (enrichment ratios, which it does) alongside p-values, and note that the p-values reflect sample size more than effect magnitude. Otherwise a reviewer might worry about p-hacking via massive sample size.

### 🟢 MINOR

**8. "Maximum elementwise difference < 6 × 10^{-5}" (§3.1) — needs units**
Is this in logit space? Pre-softmax? Post-softmax probability? The magnitude interpretation depends on the space. For logit-space differences, 6e-5 is negligible. For probability differences on rare tokens, it could matter.

**9. Table 4 token counts still sum to 204,800 ✓**
206+913+3,051+6,283+12,617+34,155+69,625+77,950 = 204,800. This fixes the 204,800/204,600 discrepancy flagged in Pass 1. 

**10. Cross-layer Table 9 expansion is strong**
All 12 layers surveyed, with L11 (known) showing 0.998 Jaccard vs. max 0.384 at any other layer. The qualitative gap is now convincingly documented. This addressed Pass 5's item 12 and Pass 6's item 2 well.

**11. 160 prompts across 15 categories is a major improvement over the original 12**
The knowledge retrieval test is now much more convincing as a negative result. The category breakdown reveals the mechanism (constrained vs. arbitrary completions) rather than just showing failure. Good revision.

### Summary

The revised paper is statistically much stronger than the original — bootstrap CIs, Bonferroni correction, null model, expanded datasets all help. The remaining issues: (1) the N2123 fire rate inconsistency (70.9% vs. 11.3%) is confusing and suggests two conflated definitions; (2) the garden-path null result is over-interpreted without a power analysis; (3) the 36.5× enrichment uses a naive chance baseline. The ablation results (item 6) still lack CIs, which is surprising given that the crossover analysis now has them — apply the same rigor everywhere.

**Priority fixes:** Item 1 (clarify N2123 dual fire rates), item 2 (power analysis or soften garden-path claim), item 3 (improve enrichment baseline), item 4 (specify bootstrap resampling unit), item 6 (add CIs to ablation table).

---

## 2026-03-21 — Pass 9: Framing & Novelty Claims

Focus: Is the paper correctly positioned relative to the field? Are novelty claims accurate and defensible? Will the framing survive contact with skeptical reviewers?

### 🔴 CRITICAL

**1. The central novelty claim — "a complete, legible account of what an MLP layer does" — is overclaimed**
The intro says: "a complete, legible account of what an MLP layer *does* — readable as pseudocode with named variables — has remained elusive. We provide such an account." But the paper characterizes 27 of 3,072 neurons (0.9%) and explicitly states the remaining ~3,040 are "distributed and not directly legible." That's not a "complete account" — it's a complete account of the *routing logic* and an admission of opacity for everything else. The claim should be: "We provide a complete account of the routing logic within L11's MLP — 27 neurons that form a legible program — while showing that the remaining ~3,040 neurons operate as a distributed, non-decomposable knowledge substrate." The current framing invites the obvious rebuttal: "You characterized 0.9% of neurons and called it complete."

**2. "The first fully legible MLP layer" framing risks the Geva 2021/2023 objection**
Geva et al. (2021) already described MLP layers as key-value memories with interpretable keys and values. Geva et al. (2023) traced factual recall step-by-step through MLP layers. Bricken et al. (2023) decomposed MLP activations into monosemantic features. A reviewer steeped in this work will ask: "How is your 'legible routing program' different from what Geva et al. already showed — that MLPs implement key-value memories with interpretable outputs?" The paper needs a sharper differentiation. Suggestion: emphasize that prior work characterized *individual neurons or features* (what a neuron responds to), while this paper characterizes *the functional organization* (how neurons coordinate into a routing program). The difference is between a parts list and a circuit diagram. This distinction is implicit but should be made explicit in the intro.

**3. "Knowledge neurons are routing infrastructure" — the claim is scoped to L11 but framed as universal**
§5.4's title ("Belial's Counsel: Knowledge Neurons Are Routing Neurons") and the abstract's claim ("'knowledge neurons' are routing infrastructure, not fact storage") are stated as general truths. But the evidence is specific to L11 of GPT-2 Small. Dai et al. (2022) worked on BERT; Meng et al. (2022) worked on GPT-J at different layers. The paper's own terminal crystallization hypothesis implies that the routing architecture is unique to the final layer — so L11's routing neurons may be a special case, not a refutation of knowledge neurons at other layers. The paper acknowledges this in Limitations ("Single model") but the framing in the abstract and §5.4 doesn't carry that qualification. A reviewer from the knowledge-editing community will push back hard on a universal claim backed by single-model, single-layer evidence.

### 🟡 MODERATE

**4. The "exception handler" metaphor may confuse software engineering readers**
In software, an exception handler catches *unexpected errors* — rare, abnormal conditions. The paper's "exception handler" fires on 11.3% of tokens and handles subword fragments, structural breaks, and rare patterns. These aren't "exceptions" in the software sense — they're minority-class tokens that need special routing. The metaphor works poetically (connecting to "darkness visible" — exceptions are the dark territory the model must navigate) but technically it's misleading. A reader might expect the exception handler to catch *errors* (hallucinations, contradictions, out-of-distribution inputs) rather than structurally predictable minority cases. Consider: "routing switch," "minority-path handler," or keep "exception handler" but add a clarifying sentence: "We use 'exception' in the statistical sense (tokens that deviate from the majority pattern) rather than the error-handling sense."

**5. The "one-stage parser" claim (§6.6) overreaches from 15 sentence pairs**
The garden-path experiment produces a genuinely interesting finding (reversed garden-path effect), but the conclusion — "transformer language models are fundamentally one-stage parsers" — generalizes from GPT-2 Small to all transformers, from 15 sentence pairs to all syntactic constructions, and from one type of garden path (intransitive/transitive verb ambiguity) to all parsing phenomena. This is a five-paragraph finding extrapolated into a universal claim. The paper should say: "GPT-2 Small behaves as a one-stage parser for verb subcategorization ambiguities" and flag the broader claim as a hypothesis for future work. As stated, it invites immediate counterexamples from anyone who's studied transformer parsing behavior.

**6. "Terminal crystallization" as a named phenomenon may be premature**
Coining a term ("terminal crystallization") for a finding observed in one model at one layer is risky. If GPT-2 Medium doesn't show the predicted structure at L23, the term becomes an embarrassment. The paper is careful to frame it as a "testable prediction," which helps, but the act of naming it implies it's a robust, general phenomenon rather than a single observation with a plausible explanation. Consider: present the finding descriptively ("L11's routing structure is unique among all 12 layers, suggesting that legible routing may be concentrated at the terminal layer") and introduce the term tentatively ("we call this *terminal crystallization*, pending validation in deeper models").

**7. The paper positions itself as mechanistic interpretability but doesn't use the field's standard tools**
No sparse autoencoders, no ACDC/automated circuit discovery, no causal tracing (in the Meng et al. sense), no activation patching. The paper uses enrichment analysis, Jaccard similarity, logit lens, and ablation — all valid but older methods. A mech interp reviewer might view this as methodologically dated. The paper addresses this obliquely (§2: "we deliberately use manual characterization to demonstrate full legibility, though automated methods could scale the approach") but could be more direct: "Our goal is interpretability, not scalability — we show what a manual, complete decomposition reveals that automated methods miss."

### 🟢 MINOR

**8. The NeurIPS 2026 preprint header may be premature**
The paper uses `\usepackage[preprint]{neurips_2026}`. If this is going to arXiv first, the venue-specific formatting signals that the authors are targeting NeurIPS, which can bias reviewers (if it's later submitted there, reviewers may have seen the preprint and formed opinions). Consider using a generic preprint format for the arXiv version.

**9. "27 named neurons" — but not all are individually characterized in the paper**
The abstract says "27 named neurons." The paper names 20 exception-handler neurons (5 Core + 10 Differentiator + 5 Specialist) and 7 consensus neurons. But in Table 1, the Differentiator tier lists all 10 neurons explicitly now. Good — this was a previous issue (Pass 6 item 3). Verify the actual paper text matches.

**10. The "program and database" analogy (§7.1) is clean and effective**
This framing — open-source control flow, encrypted database — is one of the paper's best contributions to how people think about MLP layers. It's memorable, accurate, and useful. Keep and possibly promote to the abstract or conclusion.

**11. Contribution 4 ("terminal crystallization") overlaps with the prior paper**
Balogh 2026 already identified the consensus/exception structure at L11. The present paper's L11-specificity finding partially overlaps. The paper handles this reasonably ("extends from *detecting* to *reading*") but a reviewer might still feel that the cross-layer uniqueness claim is shared intellectual property with the prior paper rather than new to this one.

### Summary

The paper's biggest framing risk is the "complete account" claim (item 1) — characterizing 0.9% of neurons and calling it complete will draw immediate fire. The "knowledge neurons are routing" claim (item 3) is scoped to one layer of one model but framed as a universal correction of Dai et al. — this asymmetry between evidence and claim is the most likely rejection trigger from a knowledge-editing reviewer. The "one-stage parser" extrapolation (item 5) similarly overreaches from a small experiment.

The framing *strengths* are considerable: the "program and database" analogy is excellent, the Milton thread gives the paper a distinctive identity, and the pseudocode representation (Figure 1) is a genuinely novel way to present MLP function. The paper just needs to match claim scope to evidence scope more carefully.

**Priority fixes:** Item 1 (scope "complete account" to routing logic), item 3 (qualify "knowledge neurons" claim to L11/GPT-2), item 5 (scope "one-stage parser" to tested constructions), item 2 (differentiate from Geva et al. explicitly), item 6 (present "terminal crystallization" more tentatively).

---

## 2026-03-22 — Pass 10: Revision Verification & Remaining Gaps

Focus: Systematic check of what's been fixed from Passes 1–9 and what remains unresolved. Also catching new issues introduced by revisions.

### 🔴 CRITICAL

**1. Author block violates USER.md instructions**
The paper still has:
```latex
\author{
  Peter Balogh \\
  Independent Researcher \\
  \texttt{peter@example.com}
}
```
USER.md explicitly says: "Affiliation: None — do NOT put 'Independent Researcher' or placeholder emails on papers. Just name + real email." The correct block is:
```latex
\author{
  Peter Balogh \\
  \texttt{palexanderbalogh@gmail.com}
}
```
This is not a style nitpick — `peter@example.com` is a *placeholder* that will be on the arXiv preprint. Fix before any submission.

**2. N2123 fire rate inconsistency STILL UNRESOLVED (Pass 8, item 1)**
§4.2 says N2123 fires on "11.3% of tokens." Table 7 (§7.1, enrichment stats) says N2123's base rate is 70.9%. The paper never explains this discrepancy. These appear to be two different definitions:
- 11.3% = exception *path* activation (the routing-relevant definition used throughout §4–6)
- 70.9% = |GELU(N2123)| > 0.1 (the generic binary firing threshold used for enrichment analysis)

If that's right, Table 7's enrichment analysis is testing "does N2123 activate slightly more when the exception path is active?" (70.9% → 100%, enrichment 1.41×) rather than "does N2123 fire as exception indicator?" These are very different questions. The enrichment ratio of 1.41× looks *unimpressive* for the paper's most important neuron — a reviewer will ask why the "exception indicator" is only 1.41× enriched. Add a sentence to §7.1 explaining the dual definition, or use the 11.3% definition consistently.

**3. Garden-path null result still over-interpreted (Pass 8, item 2)**
The paper says "N2123 shows no differential response" based on $t(14) = -0.45, p = 0.66$. With $N = 15$ and no power analysis, this remains an absence-of-evidence claim presented as evidence-of-absence. The paper should either:
- Add a power analysis (even one sentence: "This test has 80% power to detect $d \geq X$")
- Use equivalence testing (TOST procedure)
- Soften to "we found no evidence that" rather than "the exception handler does not detect"
The conclusion still says "confirming the exception handler operates at token-level predictability, not syntactic structure" — "confirming" is too strong for a null result at $N = 15$.

### 🟡 MODERATE — Remaining from earlier passes

**4. "Complete, legible account" overclaim partially fixed (Pass 9, item 1)**
The abstract now says "fully legible routing program" which is accurate. But the intro still says: "a complete, legible account of what an MLP layer *does* — readable as pseudocode with named variables — has remained elusive. We provide such an account." This claims a complete account of what the MLP *does*, not just of its routing logic. The intro text needs to match the abstract's more careful framing. Suggested fix: "a complete, legible account of an MLP layer's *routing logic*…"

**5. "36.5× enrichment over chance" baseline still naive (Pass 8, item 3)**
The chance baseline ($27/3072 \times 20 = 0.18$) assumes uniform random selection. But integrated-gradient top-20 neurons are selected for high causal influence, and routing neurons have high causal influence by definition. A better baseline: top-20 by absolute activation magnitude. Without this, the 36.5× figure is inflated. At minimum, add a sentence acknowledging the baseline limitation.

**6. Bootstrap resampling unit unspecified (Pass 8, item 4)**
Table 4's caption says "10,000 bootstrap resamples" but doesn't say what's resampled. If individual tokens (which are autocorrelated within sequences), CIs are anti-conservative. Should specify: "sequence-level bootstrap" or "token-level bootstrap (treating tokens as independent)" with a note on the autocorrelation caveat.

**7. Ablation results (Table 5) still lack CIs (Pass 8, item 6)**
Core: +0.2% PPL, Specialists: −0.3% PPL. No confidence intervals. The Specialist result is particularly load-bearing — "Specialists are counterproductive" is claimed based on a −0.3% change that could be noise. The crossover analysis has bootstrap CIs; the ablation table should have them too.

**8. "One-stage parser" claim still overgeneralized (Pass 9, item 5)**
The conclusion says "transformer language models are fundamentally one-stage parsers" — generalizing from 15 sentence pairs on GPT-2 Small to all transformers and all syntactic constructions. Should be scoped: "GPT-2 Small behaves as a one-stage parser for verb subcategorization ambiguities."

### 🟢 CONFIRMED FIXED (tracking)

The following issues from Passes 1–9 are now addressed in the paper text:
- ✅ Token count inconsistency (204,800 now consistent, Pass 1 item 1)
- ✅ Bootstrap CIs on crossover (Table 4, Pass 1 item 3)
- ✅ "Exactly" → "to numerical precision" (Pass 1 item 6)
- ✅ Core narrative reconciled with "DC offset" framing (Pass 2 items 1–2)
- ✅ Circularity check added (§5.2, Pass 2 item 3)
- ✅ Terminal crystallization has testable prediction (Pass 2 item 5)
- ✅ Dai et al. cited and engaged (§2, §5.4, Pass 3 item 1)
- ✅ ROME tension addressed (§5.4, Pass 3 item 2)
- ✅ References expanded from 8 to ~24 (Pass 3 item 3)
- ✅ Tuned lens validation added (Table 6, Fig 3, Pass 3 item 4)
- ✅ Geva 2023 cited (Pass 3 item 5)
- ✅ Elhage 2021, Olsson 2022 cited (Pass 3 item 6)
- ✅ Wang 2023 (IOI) cited (Pass 3 item 7)
- ✅ Conmy 2023 (ACDC), Bills 2023 cited (Pass 3 items 8–9)
- ✅ Abstract leads with finding (Pass 4 item 1)
- ✅ Closing line concrete (Pass 4 item 3)
- ✅ Null model control added (§7.1, Pass 5 item 3)
- ✅ Threshold robustness documented (§7.1, Pass 5 item 1)
- ✅ Enrichment method specified: Fisher's exact + Bonferroni (Pass 5 item 2)
- ✅ Cross-layer survey covers all 12 layers (Table 9, Pass 5 item 12)
- ✅ Figures added: logit lens curve, crossover plot, tier contribution (Pass 6 item 4)
- ✅ Table 2 expanded to 160 prompts with category breakdown (Pass 6 item 1)
- ✅ Pseudocode figure added (Figure 1, Pass 7 item 4)
- ✅ Table 1 Differentiator neurons fully listed (Pass 6 item 3)
- ✅ Knowledge test scoped with "amplification, not retrieval" framing (Pass 2 item 4)
- ✅ Single-domain limitation acknowledged (Pass 5 item 4)
- ✅ Progressive prediction systematized: "10/12 show pattern" (Pass 5 item 5)
- ✅ Activation transplant labeled as "illustrative" (Pass 5 item 7)

### 🟡 NEW ISSUES FROM REVISIONS

**9. Figure 1 pseudocode uses `exception = N2123.fires` but the paper says N2123 is "diagnostic, not causal"**
The pseudocode (Figure 1) presents N2123 firing as a conditional branch: `if exception: ...`. But §4.2 and §2 (citing Balogh 2026) emphasize that N2123 is "diagnostic rather than causal" — "a readable summary of the distributed routing decision." The pseudocode implies N2123 *controls* the routing path, which contradicts the diagnostic framing. The pseudocode should have a comment: `// diagnostic readout, not causal gate` or restructure to show the routing decision emerging from collective activity with N2123 as an observable indicator.

**10. §5.4 Experiment 3 (activation transplant) says N=5 is "too small for strong inference" — then why include it?**
The paper now correctly hedges ("illustrative rather than definitive"), which is good. But including an admittedly underpowered experiment alongside two stronger experiments creates a pattern where a reviewer might wonder about the standards for inclusion. Consider either: (a) expanding to N=20+ to make it informative, or (b) moving to a footnote/appendix as supplementary. Currently it occupies a full paragraph for something labeled "not standalone evidence."

**11. The Milton characters (Belial, Moloch) are still under-integrated (Pass 7, item 5)**
§5.4 has a full paragraph explaining Belial ("In Paradise Lost, the demon Belial…"), which works. But "Moloch" in §5.3 appears with only a brief phrase ("the fallen angel Moloch — 'the strongest and the fiercest Spirit'") before the analogy. Readers unfamiliar with Paradise Lost won't know why Moloch is relevant (brute force aggression even when counterproductive). One additional sentence would help: "Moloch advocates war against Heaven even when the cause is hopeless — choosing action over abstention not from strategy but from incapacity for restraint."

### Summary

**Revision quality is strong.** Of ~45 issues flagged across Passes 1–9, approximately 25 are confirmed fixed in the current text. The paper is substantially better than the original — the knowledge test expansion (12→160 prompts), bootstrap CIs, null model, tuned lens validation, and figure additions all significantly strengthen the submission.

**Top remaining vulnerabilities (in priority order):**
1. Author block: placeholder email + forbidden "Independent Researcher" (trivial fix, high embarrassment risk)
2. N2123 dual fire rate (70.9% vs 11.3%): confusing, undermines enrichment table credibility
3. Garden-path null result over-interpreted without power analysis
4. Intro still claims "complete account of what an MLP does" (should be "routing logic")
5. Ablation PPL changes lack CIs despite crossover having them (inconsistent rigor)
6. "One-stage parser" overgeneralized from N=15 on one model

**Assessment:** The paper is close to arXiv-ready. Items 1–2 are straightforward fixes. Items 3–6 are framing/scoping adjustments. No remaining structural or methodological issues that would warrant major revision — the experimental foundation is now solid.

---

## 2026-03-23 — Pass 11: Logical Flow (Second Round, Post-Revision)

Focus: Does the revised paper's argument cohere end-to-end? Have the revisions introduced new logical gaps or tensions? Does the evidence actually support the narrative arc?

### 🔴 CRITICAL

**1. The "diagnostic, not causal" framing creates a central incoherence**
The paper simultaneously presents two framings of N2123 that pull in opposite directions:
- **Framing A (§4.2, §2):** "N2123 is diagnostic rather than causal… a readable summary of the distributed routing decision… Removing the counter does not change the election."
- **Framing B (Figure 1, §4, title, abstract):** The entire paper is organized around the "exception handler" — a program with `if exception: [Core, Diff, Spec]` logic, where N2123's firing *defines* the routing path.

These are contradictory. If N2123 is purely diagnostic (Framing A), then the pseudocode in Figure 1 is misleading — there is no `if` branch controlled by N2123; there's a distributed computation that N2123 happens to correlate with. The "exception handler" is then a *description* of observed correlations, not a *mechanism*. But if the architecture genuinely routes through an exception path (Framing B), then N2123 is part of the mechanism, not just a readout.

The paper tries to have it both ways: using the causal language of programs ("exception handler," "routing," "fires," `if/else` pseudocode) while disclaiming causality. This is the paper's deepest logical tension. A careful reviewer will ask: "Is this a program or a statistical pattern? You can't call it pseudocode and then say the key variable isn't causal."

**Resolution needed:** Either (a) reframe the pseudocode as a *descriptive model* — "the following pseudocode summarizes the observed activation patterns, though the routing decision is distributed, not controlled by any single neuron" — or (b) argue that the distributed routing *implements* the program (N2123 is a readout of a real computation, not an epiphenomenon) and that the pseudocode captures the functional organization even if no single neuron is a bottleneck. Option (b) is more defensible and what the paper seems to believe, but it needs to be stated explicitly.

**2. The knowledge retrieval argument has an internal contradiction between §5.1 and §5.4**
- §5.1 (Knowledge Extraction Test): Tests whether MLP *residual neurons* can reconstruct facts. Conclusion: "This is amplification, not retrieval."
- §5.4 (Experiment 2, Knockout): "Zeroing Dai's neurons *increases* target probability by 8.0 percentage points… removing them lets the factual signal from attention pass through unimpeded."

But wait — if removing routing neurons *improves* factual recall, that means the routing neurons are *actively suppressing* the correct answer in many cases. This is a stronger claim than "the MLP doesn't retrieve facts" — it means the MLP's routing infrastructure is *fighting against* factual accuracy. The paper doesn't connect these dots. The narrative goes: "MLP doesn't retrieve → knowledge neurons are routing → removing them helps." But the logical implication — that the exception handler actively harms factual recall — contradicts the ablation results in Table 5, where removing all 20 exception neurons costs +2.1% PPL (makes the model *worse* overall).

**Resolution:** These are reconcilable — the exception handler helps on average (+2.1% PPL when removed) but hurts on *factual* tokens specifically (where it suppresses the attention-derived signal). But this reconciliation is never stated. The paper should explicitly note: "The exception handler improves general prediction (Table 5) while degrading factual recall on specific prompts — it is optimized for the common case (function word prediction) at the cost of rare factual completions."

**3. The "DC offset" explanation for Core is logically incomplete**
The revised paper explains the Core's low PPL impact (+0.2%) despite high output norm (54%) by calling it a "DC offset toward function words that the residual stream already provides." This is better than the original, but it raises an unanswered question: *why does the architecture waste 54% of its output norm on a redundant signal?*

If the residual stream already contains the function-word baseline, the Core is doing unnecessary work. In an optimized system (which GPT-2 approximately is, via gradient descent), redundant computation should be eliminated. The paper offers no explanation for why 5 neurons generating 54% of output norm survive training if their contribution is nearly zero. Possible explanations:
- The Core *was* important during training (providing a stable prior early in optimization) and became redundant as other mechanisms matured, but gradient signal wasn't strong enough to zero it out.
- The Core serves a function not captured by PPL (e.g., calibration, interaction effects with other layers).
- The +0.2% PPL is misleading because it averages over contexts where the Core matters greatly (low-consensus, where it's +2.4%) and contexts where it's counterproductive (high-consensus, where it's −0.4%).

The third explanation is supported by the paper's own data (§5.5: "Core ablation disproportionately affects low-consensus tokens") but isn't surfaced as the resolution. Instead, the paper leaves the paradox hanging: big magnitude, small impact, no explanation.

### 🟡 MODERATE

**4. The "amplification, not retrieval" framing obscures a more interesting finding**
§5.1's conclusion — "This is amplification, not retrieval" — frames the knowledge test as a negative result. But the category breakdown (Table 2, bottom panel) reveals something positive and interesting: the MLP *can* reconstruct highly constrained completions (4/10 historical events reach top-10) but *cannot* reconstruct arbitrary associations (0/15 capitals). This is evidence for a *gradient of MLP capability* indexed by how many plausible completions exist in context. The paper mentions this ("Highly constrained completions succeed") but subordinates it to the negative framing.

The positive finding — that MLP residual neurons can reconstruct facts *when there's only one plausible completion* — is actually more informative for the field than "the MLP doesn't retrieve facts." It suggests the MLP provides a kind of *contextual constraint satisfaction*, narrowing from many candidates to few, which works when the constraint space is small but fails when it's large. This is different from both "retrieval" and "pure routing" — it's something in between.

**5. The garden-path result is logically compelling but statistically unsupported**
The *logic* of the reversed garden-path effect is elegant: GPT-2 uses verb subcategorization immediately, so intransitive verbs produce *lower* surprisal at disambiguation (the model was already right) while transitive verbs produce *higher* surprisal (the model must revise its object-reading commitment). This makes clear predictions and is testable.

But the paper presents this as a *confirmed* finding based on 15 pairs with no significance test on the reversed effect itself. The $t$-test is on N2123 activation (non-significant), not on the surprisal reversal. The surprisal examples are presented as illustrative ("struggled" 5.4 vs "scratched" 11.6; "sneezed" 6.8 vs "visited" 16.2) but no paired $t$-test or Wilcoxon test on the surprisal difference across all 15 pairs is reported. If the reversed effect is real, it should be significant. If it's not tested, the paper can't claim it as a finding.

**6. Terminal crystallization argument has a survivorship bias**
The paper surveys all 12 layers, finds structure only at L11, and concludes this is "terminal crystallization." But L11 is also the layer the authors *chose to study first* (in Balogh 2026) and *built their analysis tools around*. The characterization methodology — binary firing at 0.1, Jaccard similarity, enrichment analysis — was developed *for* L11's architecture. It's possible that other layers have equally interesting structure that this methodology can't detect because it was designed for L11's specific patterns.

The null model control (§7.1) addresses one concern (random init doesn't show structure) but not this one. The question isn't "does any MLP show apparent structure?" but "does L11's specific type of structure exist elsewhere?" The answer is "no," but the methodology was optimized to find L11's type of structure.

This doesn't invalidate the finding — L11 genuinely has tighter co-firing, higher Jaccard, and clearer tier separation than other layers. But the "terminal crystallization" conclusion should acknowledge that alternative forms of structure at other layers may exist but be invisible to this analysis.

**7. The Belial/Moloch Milton thread now works better but still has a logical asymmetry**
Belial (§5.4): Persuasive voice that makes "the worse appear the better reason" → knowledge neurons appear to store knowledge but actually route. The mapping is precise — integrated gradients (eloquent attribution) misleads about the neurons' function.

Moloch (§5.3): "The strongest and the fiercest Spirit" who "argues for open war even when war is unwinnable" → MLP intervenes on high-consensus tokens even when intervention is counterproductive. The mapping works but is less precise — Moloch *chooses* war; the MLP has no choice (it's architecturally incapable of abstention). The paper notes this ("Moloch's compulsion is structural, not a bug") but the analogy breaks because Moloch's defining trait is *agency in choosing violence*, whereas the MLP's defining trait is *lack of agency*. They're almost opposites.

Minor point, but a literary-minded reviewer (or a Milton scholar on the program committee) will notice.

### 🟢 MINOR

**8. The "two orthogonal axes" claim (§5.2) doesn't follow from the evidence**
Six neurons have pairwise cosine similarities of 0.52–0.73. N2600 has cos ≈ 0.0 with all six. The paper calls these "two orthogonal axes" — but the six content neurons with 0.52–0.73 pairwise similarity are not "an axis." An axis implies a single direction; 0.52–0.73 cosine similarity means these six vectors span a subspace, not a line. "Two groups" or "a content cluster and an orthogonal outlier" would be more accurate. "Two axes" implies a 2D factorization that the data doesn't support.

**9. "The routing is the contribution" (§5.6) is a strong rhetorical closer for §5 but logically follows from §5.1, not §5.4–5.5**
§5.6 says "The MLP does not retrieve knowledge — facts arrive through attention" and points to L11 H8 as the factual signal provider. But the evidence for this is in §5.1 (knowledge extraction failure) and the logit lens (§5.2, facts emerge before L11). §5.4 (knowledge neurons) and §5.5 (ablation) provide supporting evidence but don't independently establish the claim. The section ordering suggests §5.6 follows from §5.4–5.5, when it actually follows from §5.1–5.2. Consider a forward reference: "As established in §5.1–5.2, factual knowledge arrives via attention; the following experiments (§5.4–5.5) explain why attribution methods erroneously assign knowledge to MLP neurons."

**10. The paper never explains why there are exactly 7 consensus neurons and 20 exception neurons**
The numbers 7 and 20 (totaling 27) appear to be empirically discovered but the paper doesn't say how. Were these identified by thresholding some metric? By manual inspection? By the prior paper's methodology? A reader could wonder: are there really exactly 27, or could there be 30 or 25 with slightly different criteria? The Methods section (§3.2) describes how neurons are characterized but not how the 27 were *selected* from 3,072. This is a gap — the entire architecture story depends on which neurons are "in the club."

### Summary

The revised paper's argument is substantially more coherent than the original, but three logical tensions remain:

1. **Diagnostic vs. causal** (item 1): The paper uses causal language (pseudocode, routing, exception handler) for a structure it explicitly says isn't causal. This is the paper's deepest conceptual issue and the most likely source of reviewer confusion. It needs a clear statement reconciling the two framings.

2. **Knowledge retrieval contradiction** (item 2): Removing routing neurons *helps* factual recall (Exp. 2) while removing exception neurons *hurts* general PPL (Table 5). These are reconcilable but the reconciliation isn't stated.

3. **DC offset paradox** (item 3): 54% of output norm, +0.2% PPL impact, no explanation for why training preserves this. The data for an explanation exists (consensus-conditional Core impact) but isn't surfaced.

The garden-path finding (item 5) is the paper's most citable result but lacks the statistical test that would make it a *confirmed* rather than *observed* pattern. Adding a paired test on surprisal reversal would cost one sentence and significantly strengthen the claim.

**Priority fixes:** Item 1 (reconcile diagnostic/causal framing), item 2 (explain routing-vs-factual tension), item 5 (add surprisal reversal test), item 3 (explain Core persistence), item 10 (explain neuron selection criteria).

---

## 2026-03-25 — Pass 12: Writing Quality (Second Round, Post-Revision)

Focus: Prose quality, readability, and register consistency in the revised paper. Checking whether revisions introduced new issues and whether earlier writing concerns (Pass 4) were addressed.

### Status of Pass 4 Issues

- ✅ Abstract leads with finding (item 1) — fixed, strong opening sentence
- ✅ Closing line concrete (item 3) — "We can read 27 neurons… The 3,040 neurons they route remain opaque." Excellent.
- ✅ "Consistent with" overuse (item 3) — reduced but not eliminated (see below)
- ⚠️ Parenthetical overuse (item 2) — still present, addressed below
- ⚠️ "We show/find" hedging (item 5) — partially reduced
- ✅ "Irreducibly distributed" repetition (item 7) — replaced with "entangled across" in abstract; term doesn't recur excessively

### 🔴 CRITICAL

**1. The Belial paragraph is too long for a results section**
§5.4 opens with a 4-line literary explication ("In Paradise Lost, the demon Belial is the most persuasive voice in the infernal council — 'his tongue / Dropt manna, and could make the worse appear / The better reason.'"). This is a *results* section. The literary context is charming but takes 50+ words before the reader encounters any science. For a NeurIPS audience scanning for results, this is friction. Suggestion: compress to one sentence. "The neurons Dai et al. identified as 'knowledge neurons' are Belial — eloquent but misleading." Move the full Milton quote to a footnote if it matters. The section title "Belial's Counsel" already signals the allusion; the paragraph doesn't need to re-explain it.

The *Reconciliation* paragraph also circles back: "Belial's counsel is eloquent but misleading; attribution scores are high but point to routing infrastructure, not warehouses." This is the third statement of the same point (title, opening paragraph, reconciliation). Once is evocative; three times is a tic.

**2. The abstract is now 15 lines — too dense for quick comprehension**
The abstract tries to convey: (1) the routing program, (2) the decomposition breakdown (5/10/5/7), (3) the crossover with bootstrap CIs, (4) three knowledge-neuron experiments, (5) the reversed garden-path effect, (6) the tuned lens validation, and (7) terminal crystallization with a prediction. That's 7 distinct claims in 15 lines. A NeurIPS reviewer reading 20 abstracts in a row will lose the thread by claim 4.

Suggestion: cut the neuron counts from the abstract ("5 fused Core neurons that reset vocabulary toward function words, 10 Differentiators that suppress wrong candidates, 5 Specialists that detect structural boundaries, and 7 Consensus neurons") — this is architectural detail that belongs in the body. Replace with: "organized into consensus detectors, a three-tier exception handler, and ~3,040 distributed residual neurons." Saves 2 lines and preserves the structure without forcing readers to track four number-name pairs.

### 🟡 MODERATE

**3. "consistent with" still appears 3 times**
- "consistent with the exception handler operating at token-level predictability" (abstract)
- "consistent with the exception handler operating at the level of token-level predictability" (§7.2 interpretation — near-duplicate of abstract)
- "consistent with the attention sink phenomenon" (§6.3)

The abstract and §7.2 instances are virtually identical phrases. This reads as copy-paste. Vary one: the abstract version could be "revealing that the exception handler operates at token-level predictability" (more assertive); the §7.2 version could be "confirming that N2123 tracks vocabulary uncertainty, not syntactic structure."

**4. The Moloch paragraph works but has a tonal mismatch with its surroundings**
The surrounding prose is quantitative ("promotes 17,949 tokens to top-1… 7,674 tokens while losing 3,530"). Then suddenly: "In Milton's Paradise Lost, the fallen angel Moloch — 'the strongest and the fiercest Spirit' — argues for open war against Heaven even when war is unwinnable." This jarring register shift — from counts and percentages to 17th-century epic poetry — will delight some readers and annoy others. The Belial section works better because it opens a subsection (the reader expects a fresh register). The Moloch analogy is mid-paragraph.

Suggestion: set Moloch off with a paragraph break or a transitional sentence: "This has a structural consequence best captured by analogy." The current transition from "At 7/7: gains 7,674 but loses 3,530" → "In Milton's Paradise Lost" is whiplash.

**5. Several sentences are too long for their information density**
Examples:
- "The consensus-exception crossover — where MLP intervention shifts from helpful to harmful — is statistically sharp (bootstrap 95% CIs exclude zero at all consensus levels; crossover between 4/7 and 5/7)." — 32 words, 3 parenthetical interruptions (em-dashes + parentheses + semicolon). Split: "The crossover from helpful to harmful is statistically sharp: all bootstrap 95% CIs exclude zero, with the transition at 4–5/7 consensus."
- "Critically, \citet{balogh2026discrete} showed that N2123 is diagnostic rather than causal: zeroing it alone has no measurable effect on perplexity (<0.1%)." — Fine on its own, but this is the 3rd restatement of the diagnostic-not-causal point (also in intro item 1, Figure 1 caption, and §4.2). At some point repetition becomes anxiety rather than emphasis.

**6. §5.6 "The Routing Is the Contribution" is only 7 lines**
This subsection title promises a major synthesis, but the section is remarkably brief — 3 sentences about attention head H8, then the closing "The routing is legible… the knowledge it routes around is not." For a section that's meant to be the §5 capstone, it's underdeveloped. Either expand (one more paragraph connecting H8 to the broader attention-derives-knowledge theme) or merge into the preceding subsection's closing paragraph.

**7. The "DC offset" metaphor appears 3 times without formal introduction**
- Figure 3 caption: "The DC offset paradox"
- §5.5 prose: "a DC offset toward function words that the residual stream already provides"
- §4.1: "The Core establishes the baseline vocabulary distribution"

"DC offset" is an electrical engineering term that some NeurIPS readers (especially from NLP/linguistics backgrounds) won't know. The first use should include a brief gloss: "a DC offset — a constant baseline signal — toward function words." After that, the term can stand alone.

**8. Parenthetical asides still interrupt flow in several key passages**
- §4.2: "Its input sensitivity (from W_fc) responds to subword fragments" — the W_fc attribution is methodology, not characterization. Move to a footnote or earlier in Methods.
- §5.1: "(Table~\ref{tab:twenty_q})" appears 0 times in the knowledge extraction section prose — the table is introduced only via the subsection but never explicitly referenced in the flowing text. Add "Table X shows…" at first use.
- Table 4 caption: "(204,800 tokens, 10,000 sequence-level bootstrap resamples, resampling 500 sequences with replacement to account for within-sequence autocorrelation)" — this is a 22-word parenthetical in a caption. Move the methodological detail to §3 or a footnote; keep the caption to "204,800 tokens, sequence-level bootstrap."

### 🟢 MINOR

**9. The introduction's "darkness visible" passage is well-written and distinctive**
"Language models are routinely called 'black boxes' — as if the darkness inside were uniform. It is not." This is genuinely good prose. It sets the paper's identity without being overwrought. Keep as-is.

**10. The garden-path section (§7.2) reads cleanly**
The experiment-results-interpretation structure is crisp. The minimal pair example is well-chosen. The Wilcoxon test adds statistical backing that was missing in earlier drafts. The scoped conclusion ("For verb subcategorization ambiguities, GPT-2 Small behaves as a one-stage parser") is appropriately hedged.

**11. Verb tense consistency: mostly present tense (good)**
The paper uses present tense for findings ("The Core accounts for 54%…") and past tense for experiments ("We replicated the full characterization…"). This is standard and consistent.

**12. One remaining AI-detectable pattern: enumerated contribution lists**
The introduction's 4-item enumerated list of contributions, each with bold heading + explanation, is the standard GPT-4 paper-writing template. It's also a genuinely useful format. The risk is low — many human-written NeurIPS papers use this structure. But if the paper is run through a detector, this is the passage most likely to flag.

**13. "The darkness is visible precisely because it has structure" — slight logic issue**
Darkness is visible in Milton because it's illuminated by hellfire, not because it has structure. The paper's interpretation ("visible because structured") is a *reinterpretation* of Milton, not the original meaning. This is fine for a scientific paper (you're adapting the metaphor), but a Milton scholar would note the deviation. Minor — the sentence works on its own terms.

### Summary

The prose is in strong shape for a NeurIPS submission. The main remaining writing issues: (1) the Belial paragraph is too literary for a results section — compress; (2) the abstract tries to do too much — cut architectural neuron counts; (3) the "consistent with" near-duplication between abstract and §7.2 needs varying; (4) the Moloch paragraph needs a register transition. These are all fixable in 30 minutes. The paper's distinctive voice (Milton thread, "DC offset paradox," "program and database") is an asset — most mech interp papers are monotonously technical. The risk is occasional tonal whiplash, not bland writing.

**Priority fixes:** Item 2 (tighten abstract), item 1 (compress Belial), item 4 (smooth Moloch transition), item 7 (gloss "DC offset"), item 5 (split overlong sentences).

---

## 2026-03-26 — Pass 13: Methodology (Second Round, Post-Revision)

Focus: Re-examine experimental design and methods in the revised paper. Pass 5 (3/17) flagged threshold robustness, missing null model, underspecified enrichment, single-domain confound, knowledge-test premise, underpowered transplant, and undefined head importance. Which are fixed? What new methodological issues emerged from revisions?

### Status of Pass 5 Issues

- ✅ Threshold robustness (item 1): §7.1 now documents sweep across θ ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 1.0}. Core co-firing and consensus-exception anticorrelation persist. Addressed.
- ✅ Enrichment method specified (item 2): §3.2 now specifies one-sided Fisher's exact test on 2×2 contingency tables, Bonferroni correction for 3,072 neurons (α = 1.6×10⁻⁵). Table 7 shows results. Addressed.
- ✅ Null model added (item 3): §7.1 reports random-init GPT-2 control — no consensus-exception structure, flat exception rate, low Jaccard. Addressed.
- ⚠️ Single-domain confound (item 4): Limitations section now acknowledges "Single domain: All experiments use WikiText-103." Acknowledged but not tested. Still a gap — see below.
- ✅ Progressive prediction systematized (item 5): §4.3 now reports "10/12 show Core→reset→Diff→suppress pattern; 2 exceptions involve subword-initial predictions." Addressed.
- ⚠️ Knowledge extraction premise (item 6): §5.1 now has both static and context-dependent versions. The concern about testing the wrong thing (static directions vs. actual contributions) is partially addressed by the context-dependent version. See new assessment below.
- ✅ Transplant experiment labeled illustrative (item 7): "illustrative rather than definitive… N=5 is too small for strong inference." Addressed.
- ⚠️ Head importance metric undefined (item 8): §6.3 says "measured by mean absolute attention-weighted value norm across 204,800 tokens." This is now defined but still only at point of use, not in §3 (Methods). Partially addressed.
- ✅ Sequence selection described (item 9): §3.2 says "500 sequences of 1,024 tokens each" — still no description of how sequences are selected from WikiText-103, but this is standard practice for the dataset and unlikely to draw reviewer fire.
- ⚠️ No held-out validation (item 10): Still all analysis on same 512K/204.8K tokens. Unaddressed but common in mech interp papers.
- ✅ Jaccard base-rate issue (item 11): Not explicitly acknowledged but mitigated by the random-init null model, which shows untrained neurons at 90%+ fire rates produce Jaccard of only 0.53 vs. trained Core's 0.86+.
- ✅ Cross-layer survey expanded (item 12): Table 9 now covers all 12 layers. Addressed.

### 🔴 CRITICAL

**1. The enrichment analysis and the exception-path definition use incompatible "firing" criteria**
This was flagged in Pass 8 (item 1) and Pass 10 (item 2) but remains unresolved and is fundamentally a *methodology* problem, not just a reporting one.

The paper uses two definitions of N2123 "firing":
- **Routing definition** (§4, §5, throughout): N2123 fires on 11.3% of tokens, indicating the exception path. This is the definition that drives the entire architectural story.
- **Enrichment definition** (§3.2, Table 7): |GELU(N2123)| > 0.1, yielding a 70.9% base rate.

Table 7's enrichment analysis tests whether neurons are enriched *under the generic firing definition* conditioned on the exception path being active. This means the enrichment analysis is asking: "When the exception path is active (11.3% of tokens), does |GELU(N2123)| > 0.1 more often?" Answer: yes, 100% vs 70.9%, enrichment 1.41×.

But this is the *wrong question*. The interesting question is: "When the exception path is active, which other neurons fire more often?" — using the *same* definition of "exception path active" as the rest of the paper. Table 7 appears to use N2123's generic activation as the conditioning variable, but the paper never defines what "exception rate" means in the enrichment context. Is the condition "N2123 fires at the routing threshold (whatever defines the 11.3%)" or "N2123's |GELU| > 0.1"? If the former, the base rates in Table 7 should change (conditioned on 11.3% of tokens, not 70.9%). If the latter, the enrichment analysis is testing correlation with a threshold artifact, not with the routing decision.

**Fix needed:** Either (a) define the exception-path condition explicitly in §3 and use it consistently in Table 7, or (b) add a note to Table 7 explaining the dual thresholds and why the enrichment is still meaningful.

**2. The garden-path experiment lacks a manipulation check**
The 15 sentence pairs contrast intransitive vs. transitive verbs and measure surprisal at the disambiguation point. But the experiment assumes GPT-2 *knows* verb subcategorization — that it treats "struggled" as intransitive and "scratched" as transitive. This is never verified.

If GPT-2 assigns non-negligible probability to "struggled" taking a direct object (which it might — "struggled" can appear in constructions like "struggled against," "struggled with," where a following NP is expected), then the intransitive/transitive contrast is weaker than assumed, and the "reversed" garden-path effect might reflect something else entirely (e.g., frequency differences between the verbs).

A manipulation check is straightforward: for each verb, measure P(NP follows | "After the dog [verb]") and confirm that intransitive verbs produce lower NP-continuation probability than transitive verbs. Without this, the experiment's internal validity is uncertain.

### 🟡 MODERATE

**3. The context-dependent knowledge extraction test (§5.1) still has a methodological gap**
The context-dependent version scales each neuron's output direction by its actual post-GELU activation. This is better than the static version, but it still accumulates neurons *one at a time* and checks when the target reaches top-10. The order of accumulation matters — sorted by "contribution magnitude" — but magnitude to what? The paper says "contribution magnitude" without specifying whether this is |activation × output_direction_norm|, or |activation × projection_onto_target_direction|, or something else. The choice of sorting criterion determines the accumulation curve and could make the test trivially easy (sort by projection onto target = guaranteed early convergence) or misleadingly hard (sort by generic magnitude = facts get buried under function-word pushes).

Also: accumulating individual neuron contributions independently ignores interaction effects. The actual MLP output is the *sum* of all neuron contributions; individual accumulation doesn't account for cancellation between neurons. Two neurons with opposing effects might each rank low individually but jointly produce the correct prediction. The test can show that no *single* neuron retrieves facts, but not that no *combination* does (which is a different claim from "the MLP does not retrieve facts").

**4. Bootstrap resampling unit now specified — but the method description is buried**
Table 4's caption now says "10,000 sequence-level bootstrap resamples, resampling 500 sequences with replacement." This is the right approach (sequence-level to account for within-sequence autocorrelation) — good fix. But it's only in the Table 4 caption, not in §3 (Methods). A reviewer looking at Methods won't find the bootstrap procedure. Should be added to §3 as the general statistical framework.

**5. Ablation methodology (§5.5, Table 5) still has no CIs**
Pass 8 (item 6) flagged this. The crossover analysis (Table 4) has bootstrap CIs. The ablation analysis (Table 5) does not. The Specialist result (−0.3% PPL, i.e., model *improves* when Specialists are removed) is load-bearing for the claim that "Specialists are marginally counterproductive." Without a CI, we don't know if this is signal or noise. Given that you already have the bootstrap machinery, adding CIs to Table 5 should be trivial — just recompute PPL for each bootstrap resample with and without each tier ablated.

**Update:** Checking Table 5 more carefully — the caption now says "sequence-level bootstrap 95% CIs from 500 sequences" and the table includes a CI column. **This is fixed.** The CIs are: Core [+0.1%, +0.4%], Differentiators [+0.9%, +1.7%], Specialists [−0.6%, −0.1%], All exception [+1.6%, +2.5%]. The Specialist CI excludes zero on the negative side — the counterproductive effect is statistically significant. Good.

**6. The "attention head importance" metric is domain-specific and might not generalize**
§6.3 defines importance as "mean absolute attention-weighted value norm." This is a reasonable metric for measuring how much signal a head contributes to the residual stream, but it doesn't account for *useful* vs. *harmful* contributions. A head that adds large-magnitude noise would score as "important." The paper's interpretation — that H7's high importance reflects its dominance in L11's processing — assumes magnitude ≈ utility, which the paper itself shows is false for the Core neurons (54% magnitude, 0.2% utility). The same caveat should apply to attention heads.

**7. The Wilcoxon test on garden-path surprisal (§7.2) is good but the stimuli design has a BPE confound acknowledged but not controlled**
The Limitations section says: "In some garden-path pairs, the disambiguation word falls at different absolute positions due to BPE tokenization, potentially confounding position-sensitive effects." This is honest, but BPE confounds could be controlled by selecting pairs where the disambiguation word falls at the same token position, or by including position as a covariate. With only 15 pairs, losing some to position-matching might be too costly, but the paper should at least report how many pairs are affected and whether the effect holds in the subset with matched positions.

### 🟢 MINOR

**8. The 160-prompt knowledge test is well-designed**
15 categories, 10+ prompts each, both static and context-dependent versions, category-level breakdown. This is a major improvement over the original 12 prompts. The category structure reveals the mechanism (constrained vs. arbitrary) rather than just showing uniform failure. Methodologically sound.

**9. The random-init null model is the right control**
Same architecture, untrained weights, identical analysis pipeline. No consensus-exception structure, flat exception rate, low Jaccard. This rules out the concern that any MLP analyzed this way would show apparent tier structure. Well done.

**10. Tuned lens validation is methodologically appropriate**
Using Belrose et al.'s tuned lens alongside the logit lens addresses the known bias of logit lens at early layers. The developmental arc holds under both methods. Table 6 shows both side-by-side, which is the right presentation.

### Summary

The methodology is substantially stronger than the original version. The main remaining issues:

1. **Dual firing-threshold inconsistency** (item 1) — this is the same issue flagged in Passes 8 and 10, now examined as a methodology problem rather than just a statistical reporting one. It undermines Table 7's enrichment analysis because the conditioning variable is ambiguous.
2. **Garden-path manipulation check** (item 2) — the experiment assumes GPT-2 correctly categorizes verbs as transitive/intransitive, but never verifies this. A 2-line computation (P(NP | verb)) would establish internal validity.
3. **Knowledge extraction accumulation order** (item 3) — "contribution magnitude" is underspecified and the independent-accumulation design can't test combinatorial retrieval (which is what "20 Questions" implies).
4. **Bootstrap method in caption not Methods** (item 4) — trivial relocation.

Of the original Pass 5 critical items, all three (threshold, enrichment, null model) are now addressed. The remaining methodology issues are moderate — they won't sink the paper but a thorough reviewer will flag items 1 and 2.

**Priority fixes:** Item 1 (clarify enrichment conditioning), item 2 (add manipulation check or acknowledge), item 3 (specify accumulation sort criterion), item 4 (move bootstrap to §3).
## 2026-03-27 — Pass 14: Figures/Tables (Second Round, Post-Revision)

Focus: Re-examine all figures and tables in the revised paper. Pass 6 (3/18) identified zero figures and problematic tables. The paper now has 4 figures and 10 tables. Are they well-executed? Do they support the claims effectively?

### Status: Major Improvement

The paper now has **4 figures** and **10 tables** (vs. 0 figures and problematic tables in the original). This addresses Pass 6's critical item 4. The figure additions are well-chosen and the table expansions are substantial. However, several issues remain.

### 🔴 CRITICAL

**1. Table 4 (tab:twenty_q) has confusing structure and unclear "Base" column**
The table caption says "Base" but doesn't define it. Context suggests this is the rank before L11 MLP (i.e., from the residual stream at L10), but readers will have to infer this. The caption should explicitly state: "Base: rank before L11 MLP; Static: accumulated raw W_proj columns; Context: scaled by post-GELU activation."

Also, the two-panel structure (top: examples, bottom: category summary) is clever but hard to parse on first read. The transition from the examples to the summary table happens with just a `\vspace{4pt}` — readers scanning quickly might think these are two separate tables. Consider adding a horizontal rule or a subheading in the caption: "**Top panel:** representative prompts. **Bottom panel:** median ranks by category."

**2. Figure 1 (pseudocode) has an ellipsis in the DIFF tier: "N2462,N2173,N1602,...,N2378"**
This is the paper's headline figure — the "routing program readable as pseudocode" — and it uses "..." for 6 of the 10 Differentiator neurons. This looks unfinished. If space is truly an issue, use "N2462, N2173, +8 others" or list all 10 on multiple lines:
```
DIFF(N2462, N2173, N1602, N1800,
     N2379, N1715, N611, N3066,
     N584, N2378):
```
The ellipsis in the paper's most important figure undermines the "complete account" framing.

**3. Table 7 (tab:consensus_help) caption is 3 lines long and hard to parse**
Caption: "L11 MLP effect on next-token probability, by consensus level (204,800 tokens, 10,000 sequence-level bootstrap resamples, resampling 500 sequences with replacement to account for within-sequence autocorrelation). All CIs exclude zero. The crossover from helpful to harmful occurs at 4–5/7."

This is 53 words. The methodological detail about resampling belongs in §3 (Methods), not in every table caption. Readers encountering this table will spend cognitive load parsing the bootstrap procedure rather than understanding the result. Suggest: "L11 MLP effect on next-token probability by consensus level (204,800 tokens, sequence-level bootstrap). All CIs exclude zero; crossover at 4–5/7." Move the detailed bootstrap description to Methods.

### 🟡 MODERATE

**4. Figure 3 (crossover) and Table 7 are redundant**
Figure 3 shows the consensus-crossover effect visually (bars with error shading), and Table 7 shows the exact same data in tabular form with CIs. This is good practice for a statistics-heavy finding, but the figure doesn't add much beyond the table — the crossover is equally obvious from the ΔP column flipping sign. Consider: could Figure 3 show something *additional*, like how the crossover varies across different types of tokens (content words vs. function words vs. subwords)? As-is, it's a well-executed but somewhat redundant figure.

**5. Table 2 (consensus neuron specs) has a 4.5cm paragraph column that's hard to read**
The "Key Evidence" column uses `p{4.5cm}` and contains dense prose: "Fires on `and`, `but`, `also` mid-clause; silent at clause boundaries." This is harder to scan than bulleted keywords or a structured format. Consider reformatting:
- Current: prose paragraph
- Better: "Fires: `and`, `but`, `also` (mid-clause). Silent: clause boundaries."
Or move detailed evidence to supplementary material and keep only dimension labels + fire rate in the main table.

**6. Figure 4 (tier contribution) caption explains the "DC offset paradox" but the figure doesn't visually show the paradox**
Caption: "The DC offset paradox. **Left:** Exception-path output norm by tier. The Core dominates at 54%. **Right:** PPL impact when each tier is ablated. Despite its large magnitude, the Core contributes only +0.2% PPL—a DC offset toward function words that the residual stream already provides."

The figure shows two bar charts: output norm (left) and PPL impact (right). The "paradox" is that Core is largest in the left panel but smallest in the right panel. This is visible but not obvious to a casual reader — the two charts don't share the same y-axis scale, so the visual mismatch isn't striking. Consider: overlay the two metrics on the same chart (dual y-axis: output norm in blue, PPL impact in red) so the paradox is immediately visible, or annotate the figure with a caption-style comment: "Core: 54% norm → 0.2% impact."

**7. Table 10 (enrichment stats) has an unexplained note in the caption about N2123's dual threshold**
The caption includes a long explanation: "N2123 has a generic activation rate of 70.9% (|GELU| > 0.1), but only 11.3% of tokens produce the high-magnitude activations that indicate exception-path routing (§4.2). The enrichment analysis here uses the generic threshold; the 1.41× enrichment reflects the concentration of high-activation events within exception tokens."

This is good — it addresses Pass 8/10/13's concern about the dual firing thresholds. But it's 46 words in a caption, and the explanation is conceptually important enough to be in the main text (§3 or §7.1) rather than hidden in Table 10's caption. Most readers won't read table captions this carefully. Move this to the text and reference it from the caption: "See §7.1 for explanation of the dual thresholds."

**8. Table 9 (cross_layer) uses ranges for L0–L3, L4–L6, L7–L9 but only reports max/min, not all layers**
The table groups early layers (L0–L3) and reports their range (e.g., "6–15%" exception fire rate, "0.06–0.16" Jaccard). This is space-efficient but loses granularity. A reviewer might wonder: is there a smooth gradient from L0 to L3, or is L3 an outlier? The claim that "L11's structure is unique" would be stronger if the table showed all 12 layers individually (even if in a supplementary table), confirming that *every* earlier layer lacks structure, not just that the range across layers is lower than L11's.

**9. Figure 2 (logit lens) has solid and dashed lines that are hard to distinguish in grayscale**
The caption says "logit lens (dashed) and tuned lens (solid)," but if the figure is printed in grayscale or viewed by colorblind readers, the distinction might be subtle. Consider: (a) use different line styles (dashed vs. dotted vs. dash-dot), or (b) add labels directly on the curves ("Logit Lens", "Tuned Lens") rather than relying solely on the legend, or (c) note in the caption that color is needed: "solid blue / dashed orange lines."

### 🟢 MINOR — Confirmed Good Practices

**10. All tables use booktabs style (toprule/midrule/bottomrule) ✓**
Clean, professional typesetting throughout. No vertical rules, good spacing.

**11. Table 4's two-panel structure (examples + category summary) effectively shows both specifics and aggregate ✓**
The top panel gives intuition (specific prompts that succeed/fail); the bottom panel gives the statistical summary (median rank by category). This is the right design for showing heterogeneity.

**12. Figure 1 (pseudocode) is the paper's best visual contribution ✓**
The pseudocode format makes the routing program immediately graspable. The comments (PPL impacts, "diagnostic readout not causal gate") are well-placed. This is genuinely novel visual communication for mech interp. The ellipsis issue (item 2) is fixable; the concept is strong.

**13. Figure 3 uses color meaningfully: blue (helps) vs. red (hurts) ✓**
The visual encoding (upward blue triangles = MLP helps, downward red triangles = MLP hurts) is intuitive. The gray line for token count (right y-axis) provides context without cluttering.

**14. Table 6 (logit_lens) includes both logit and tuned lens side-by-side ✓**
This directly addresses Pass 3's item 4 (tuned lens validation). Showing both lenses in the same table makes the robustness of the developmental arc obvious. The "First lock-in (Tuned)" column adds useful detail (what % of tokens first reach top-1 at each layer).

**15. All figures have .pdf extensions, suggesting vector graphics ✓**
Good practice for LaTeX. The figures will scale cleanly and look professional.

### 🟢 MINOR — Issues

**16. Table 1 (tiers) still has multi-line neuron lists that could be formatted more compactly**
The Differentiator row spans 2 lines for neuron IDs and another line for the function description. Consider:
```
Differentiator & N2462, N2173, N1602, N1800, N2379, & 10 & 35–88\% & 0.15–0.89 & Candidate \\
               & N1715, N611, N3066, N584, N2378   &    &          &            & suppression \\
```
This is acceptable but makes the table harder to scan. An alternative: list neurons in smaller font or use a footnote.

**17. Table 3 (consensus_levels) "Dominated by" column has inconsistent specificity**
- 0/7: specific token + percentage (\n\n at 27%)
- 1–2/7: token categories (paragraph breaks, fragments)
- 3–4/7: conceptual labels (headers, emerging function words)
- 5–6/7: categories (function words, punctuation)
- 7/7: specific tokens + percentages (, at 7.3%)

The 0/7 and 7/7 rows have quantitative breakdowns; the middle rows don't. Either add percentages for all rows (e.g., "the" at 5.1%, "a" at 4.2% for 5–6/7) or remove percentages from 0/7 and 7/7. The current mix is inconsistent.

**18. No table shows the Wilcoxon test result for garden-path surprisal reversal**
§7.2 (garden-path discussion) reports: "A paired Wilcoxon signed-rank test across all 15 pairs shows the transitive condition produces significantly higher surprisal at disambiguation (W = 12, p = 0.018, median difference +3.1 bits)." This is the paper's most interesting psycholinguistic finding but appears only in prose. A small table (even 3 rows: Intransitive median, Transitive median, Test statistic) would make the result more prominent and easier to cite.

**19. Figure/table numbering is correct but some forward references are distant**
Figure 1 (pseudocode) is referenced in the intro before it appears (fine — standard LaTeX float behavior). But Table 8 (ablation) is first mentioned in §4.1 ("PPL impact from Table~\ref{tab:ablation}") and doesn't appear until §5.5, 15 pages later in the compiled PDF. LaTeX floats will place it near §5.5, but readers at §4.1 will have to hunt. Consider: mention the forward reference more explicitly: "PPL impact from the tier ablation study (Table~\ref{tab:ablation}, §5.5)."

**20. Table 5 (knowledge_neurons) could be a single sentence instead of a table**
The table shows:
- Consensus neurons: 2.6 avg in top-20, 0.05 expected
- All routing neurons: 7.3 avg in top-20, 0.18 expected
- Enrichment: 36.5×

This is 3 data points. The table takes more space than prose: "Routing neurons appear in an average of 7.3 of the top-20 attributed neurons (36.5× over the chance expectation of 0.18), with consensus neurons alone averaging 2.6 (vs. 0.05 expected)." Consider demoting to in-text statistics and reserving table space for data that benefits from tabular structure.

### Summary

**Major improvements over Pass 6's assessment:** The paper now has figures (4, up from 0) and substantially expanded tables (160 prompts in Table 4, up from 12). Figure 1 (pseudocode) is an excellent contribution and the figures are well-chosen for the claims. Table expansions (knowledge test, logit lens with tuned lens, cross-layer survey, bootstrap CIs) address many earlier concerns.

**Remaining critical issues:**
1. Figure 1's ellipsis (item 2) — trivial fix, high visual impact
2. Table 4's undefined "Base" column (item 1) — caption clarity
3. Table 7's overcrowded caption (item 3) — move methods detail to §3

**Moderate improvements:**
- Table 2's paragraph column (item 5) could be reformatted
- Figure 4 (item 6) could make the "DC offset paradox" more visually obvious
- Figure 3 and Table 7 (item 4) are somewhat redundant but both are defensible
- Table 10's dual-threshold explanation (item 7) should be in main text

**Bottom line:** The figures/tables are now publication-ready with minor fixes. The pseudocode figure is a genuine contribution to how mech interp results can be communicated. The table expansions (especially 160-prompt knowledge test) significantly strengthen the empirical grounding. Items 1–3 are the only must-fix issues; the rest are quality-of-life improvements.

**Priority fixes:**
1. Figure 1: replace ellipsis with full neuron list or "+8 others" notation
2. Table 4: define "Base" in caption
3. Table 7: move bootstrap detail to Methods, shorten caption
4. Table 10: move dual-threshold explanation to §7.1 prose
5. Add supplementary table with all 12 layers' metrics (supplement Table 9)

---

## 2026-04-05 — Pass 21: Appendix Quality & Supporting Evidence

Focus: Do the appendices adequately support the main text claims? Are there internal inconsistencies between appendix data and main text assertions? Would a skeptical reviewer checking supplementary material find gaps? This is the first pass dedicated to appendix-level scrutiny.

### Overall Assessment

The paper has 6 appendix sections (A–F): Consensus Characterization, Knowledge Extraction Full Results, Knowledge Neurons Full Details, Logit Lens & Developmental Arc, Garden-Path Experiment, and Controls + Limitations. The appendices are functional but uneven — some are data-rich (B, D), others are thin sketches (C, E). Several carry load-bearing evidence that the main text relies on without adequate cross-referencing.

### 🔴 CRITICAL

**1. Appendix B (Knowledge Extraction): Category table only shows 7 of 15 claimed categories**
§5.1 says "160 factual cloze prompts spanning 15 categories." Appendix B's category summary table shows only 7 categories (Historical events, Physics, Mathematics, Technology, Biology, Capitals, Languages) summing to 75 prompts — not 160. The "All (160)" row at the bottom claims 160, but the listed categories add to 75. Where are the other 85 prompts across 8 unlisted categories? Either the table is incomplete (missing 8 categories, a serious omission), or the "15 categories" claim in the main text is wrong. A reviewer checking the math will immediately flag this: 7 × 10 + 1 × 15 = 85 ≠ 160.

**Fix:** Either list all 15 categories in the table (the most natural solution — it adds 8 rows, minimal space), or explain the discrepancy. If some categories were merged or the count is wrong, correct the main text. This is a data integrity issue that undermines the knowledge-retrieval experiment's credibility.

**2. Appendix C (Knowledge Neurons): The knockout effect (+8.0pp) has no per-prompt breakdown**
The main text's strongest claim in §5.4 — that zeroing knowledge neurons *increases* target probability — is supported only by the aggregate "+8.0pp on average." But with 20 prompts, the variance could be enormous. How many of the 20 show improvement? How many show degradation? If 15 improve by +12pp and 5 degrade by −4pp, that's very different from 20 prompts uniformly improving by +8pp. The appendix says "on average" and gives no distribution information. A reviewer who finds this suspicious (the claim directly contradicts Dai et al.) will want per-prompt data.

**Fix:** Add a small table or histogram showing the distribution of knockout effects across 20 prompts: how many positive, how many negative, range, IQR. Even a one-line summary ("18/20 prompts showed increased target probability; range: −2.1pp to +24.3pp") would address this.

**3. Appendix E (Garden-Path): Only one example pair shown**
The appendix shows a single minimal pair ("struggled"/"scratched") and reports aggregate statistics. But with only 15 pairs and a p=0.018 result, the stimulus set IS the experiment. A reviewer will want to see all 15 pairs to assess quality: Are the verbs matched for frequency? Are the sentence frames natural? Are there any pairs where the intransitive/transitive classification is debatable? Showing all 15 pairs (even in a compact table: Pair # | Intransitive verb | Transitive verb | Surprisal_int | Surprisal_trans) would cost ~15 lines and dramatically increase transparency.

**Fix:** Add a table with all 15 pairs and their individual surprisal values at disambiguation. This also lets readers verify the Wilcoxon test themselves and check for outlier-driven effects.

### 🟡 MODERATE

**4. Appendix A (Consensus Neurons): "Key Evidence" column mixes statistical and anecdotal evidence**
Table entries like "Fires on `and`, `but`, `also` mid-clause; silent at clause boundaries" are characterizations, not evidence. What's the enrichment ratio? What's the p-value? The enrichment analysis methodology is described in §3 (Fisher's exact, Bonferroni), but Appendix A doesn't report enrichment statistics for the specific tokens listed. Contrast with Table 7 (enrichment stats in Controls appendix), which does report p-values but for a different set of neurons. The consensus neuron characterizations in Appendix A appear to be qualitative descriptions rather than systematic statistical results.

**Suggestion:** Add enrichment ratios to the "Key Evidence" column: "Fires on `and` (enrichment 2.3×, p < 10⁻⁵⁰), `but` (1.8×), `also` (2.1×)". This grounds the qualitative characterizations in the same statistical framework used elsewhere.

**5. Appendix D (Logit Lens): "First lock-in (Tuned)" column is poorly defined**
The column header says "First lock-in (Tuned)" with values like 3.5%, 6.2%, 1.2%, etc. The main text never defines what "first lock-in" means. Is this the percentage of tokens that *first* reach top-1 accuracy at this layer (i.e., never correct before, correct here for the first time)? If so, the values should sum to 100% minus "Never" — and indeed: 3.5+6.2+1.2+1.4+1.4+1.3+4.0+3.4+5.9+6.0+5.9+3.7+3.0+53.0 = 99.9% ≈ 100% ✓. Good internal consistency. But the definition needs to be in the table caption or a table note, not left implicit.

**6. Appendix D: L11H7 attention head analysis feels orphaned**
The "L11H7: the dominant attention head" paragraph appears at the end of the Logit Lens appendix but is about attention, not the logit lens. It discusses BOS attention weights (45.4%), exception-token BOS attendance (47.0% vs 37.3%), and attention sink behavior. This material belongs either in a separate appendix ("Attention Head Analysis") or in the main text's Terminal Crystallization section (§6). Its current placement — tacked onto the logit lens appendix — suggests it was added late and no better home was found.

**7. Appendix F (Controls): Null model numbers are presented as assertions, not data**
The null model paragraph says: "exception fires on ~64% of tokens (vs. 11.3% trained), Core Jaccard is 0.53 (vs. 0.86 trained), and exception rate is flat across consensus levels (63–65%, no anticorrelation)." These are three numbers with no table. For the paper's most important control experiment — ruling out that the architecture is an analysis artifact — the evidence should be presented as systematically as the trained model's results. At minimum, a small table comparing trained vs. random-init on the key metrics (exception rate, max Jaccard, enrichment, consensus-exception correlation) would make the control convincing rather than perfunctory.

**8. Appendix F (Limitations): Missing the "SAE alternative" limitation**
The limitations list covers: single model, single domain, single layer depth, garden-path stimuli, BPE confounds, transplant N=5. Missing: "We characterize neurons individually rather than via dictionary learning (sparse autoencoders), which might reveal additional structure in the ~3,040 residual neurons." This is especially important given that Templeton et al. 2024 (still uncited — see Passes 15/17/20) demonstrated SAE decomposition at scale. The limitations section should acknowledge this methodological alternative.

Also missing: "The enrichment analysis conditions on the generic threshold (|GELU| > 0.1) rather than the exception-path threshold (GELU > 1.0), creating a dual-threshold issue documented in §3." Pass 8/10/13/19 flagged this repeatedly. Even though it's now explained in the methods, acknowledging it as a limitation would show self-awareness.

### 🟢 MINOR

**9. Appendix A: N2123's alignment with consensus mean (cos 0.838) is a strong finding buried in an appendix**
The fact that both routing paths push toward the same vocabulary (safe/function words) — and differ only in activation context, not direction — is conceptually important. It means the exception handler isn't *redirecting* the distribution to a different vocabulary; it's *adjusting the strength* of the same push. This supports the "DC offset" framing. Consider promoting this to the main text (§4 or §5.2).

**10. Appendix B: The progressive prediction walkthrough (moon example) is well-executed**
The step-by-step ("Before MLP → After Core → After Diff → After Residual") with the 10/12 success rate is convincing evidence for the tier-by-tier processing story. The 2 exceptions (subword-initial predictions) are honestly reported. This is good appendix content.

**11. Appendix C: The enrichment baseline caveat is well-stated**
"The 36.5× enrichment uses a uniform baseline... the true enrichment over a causally-matched baseline would be lower. The qualitative finding — that the same routing neurons recur across diverse facts — is the more robust evidence." This directly addresses Pass 8's concern. Good.

**12. Appendix C: Knockout-vs-ablation reconciliation is valuable**
"The handler is optimized for the common case (function-word prediction at low consensus) at the cost of rare factual completions." This resolves the Pass 11 tension between knockout improving factual recall (+8pp) while full ablation worsens PPL (+2.1%). This reconciliation is important enough that it should also appear in the main text (§5.4 or §7).

**13. Appendix E: The Britt 1992 connection is a nice scholarly touch**
Connecting the garden-path results to Britt et al.'s selective modularity work adds depth and shows engagement with the psycholinguistics literature beyond just the garden-path paradigm. Minor but strengthens the paper's interdisciplinary credibility.

**14. Appendix F: Threshold robustness sweep is well-reported**
The range (θ = 0.01 to 1.0) with specific metrics (Jaccard ≥ 0.45, spread ≥ 0.11) demonstrates that the architecture isn't a threshold artifact. The sweep covers two orders of magnitude. Solid.

### Summary

The appendices are functional but have three notable gaps:

1. **Category table incomplete** (item 1) — 7 of 15 categories shown for the 160-prompt knowledge test. This is a data integrity issue. Must be fixed: either list all 15 categories or correct the "15 categories" claim.

2. **Knockout distribution missing** (item 2) — the +8.0pp knockout result is reported as an average with no variance information. For a claim that directly contradicts Dai et al., per-prompt data is essential.

3. **Garden-path stimuli hidden** (item 3) — only 1 of 15 pairs shown. For a psycholinguistic experiment with N=15 and borderline significance, full transparency about stimuli is standard practice and expected.

The moderate issues (consensus enrichment statistics, null model presentation, missing limitations) are quality improvements that would strengthen the appendices' persuasive power.

**Priority fixes:**
1. Complete the category table in Appendix B (add 8 missing categories)
2. Add knockout distribution data to Appendix C
3. Add full stimulus table to Appendix E
4. Add SAE-alternative to limitations list in Appendix F
5. Present null model results as a comparison table, not just prose

**Overall observation:** The appendices currently serve as data dumps rather than persuasive arguments. The best NeurIPS appendices anticipate reviewer skepticism and proactively address it with systematic data presentation. Items 1–3 are cases where a reviewer *will* go looking for supporting detail and find it missing. Items 4–5 are cases where better presentation would convert a skeptical reader into a convinced one.

---

## 2026-03-28 — Pass 15: Related Work (Second Round, Post-Revision)

Focus: The paper went from 8 to ~24 references after Pass 3 flagged critical gaps. Are the new citations well-integrated? Are there remaining gaps given the current claims? Does the paper fairly engage with work it contradicts?

### Status of Pass 3 Issues

- ✅ Dai et al. 2022 cited and engaged (§2, §5.4) — the paper now devotes a full subsection to rebutting their "knowledge neurons" claim. Well done.
- ✅ ROME tension addressed (§5.4 Reconciliation paragraph) — "editing MLP weights changes routing decisions, not stored facts." Clear.
- ✅ References expanded from 8 to ~24. Major improvement.
- ✅ Belrose et al. 2023 (tuned lens) cited and used (§5.2, Table 6). Integrated, not just cited.
- ✅ Geva et al. 2023 cited (§2). Acknowledged but engagement is thin — see below.
- ✅ Elhage et al. 2021 cited (§2). Foundational reference now present.
- ✅ Wang et al. 2023 (IOI) cited (§2). Good differentiation: "they traced a circuit; we trace an entire layer."
- ✅ Conmy et al. 2023 (ACDC) cited (§2). Methodological choice acknowledged.
- ✅ Bills et al. 2023 cited (§2). Brief but sufficient.
- ✅ Tenney et al. 2019 cited (§2). Connected to developmental arc.
- ✅ Clark et al. 2019 cited (§2). Attention head context.
- ✅ Self-citation ratio now reasonable (~1:23).

### 🔴 CRITICAL

**1. Geva et al. 2023 engagement is still too thin — this is the paper's most dangerous under-citation**
§2 says: "Geva et al. (2023) refined this picture, tracing factual recall step by step through attention and MLP layers and finding that MLPs at subject-last positions promote correct attributes — a more nuanced view than our binary 'routing vs. retrieval' framing."

This is one sentence, and it *concedes the tension without resolving it*. Geva 2023 provides evidence that MLPs *do* promote factual attributes — not just route. The present paper claims MLPs route but don't retrieve. These are directly contradictory findings. The one-sentence acknowledgment makes it look like the paper is hoping reviewers won't notice the contradiction.

The paper needs at least a paragraph (ideally in §5) explaining *why* its findings differ from Geva 2023. Possible explanations:
- Geva 2023 studies mid-layers (where MLPs may have a different role); this paper studies L11 (the terminal layer, where routing dominates)
- Geva 2023's "promotion" could be reinterpreted as routing (amplifying attention-derived signals, consistent with this paper's framing)
- Geva 2023 uses a different methodology (causal tracing vs. enrichment analysis) that could explain different conclusions

Without engaging with *why* the results differ, a Geva-lab reviewer will reject on grounds of insufficient engagement with contradicting evidence.

**2. Missing: Templeton et al. 2024 "Scaling Monosemanticity" (Anthropic)**
This is the large-scale follow-up to Bricken et al. 2023 (which is cited). Templeton et al. applied sparse autoencoders to Claude 3 Sonnet and found interpretable features at scale, including features that map to specific concepts, facts, and behaviors. This is directly relevant because:
- It provides evidence that MLP neurons *can* be individually meaningful (via SAE decomposition), which this paper claims is true for 27 but not 3,040 neurons
- It raises the question: would SAE decomposition of L11's ~3,040 "opaque" residual neurons reveal additional interpretable structure?
- The paper claims the residual neurons are "irreducibly distributed" — Templeton et al.'s success in finding interpretable features from distributed representations challenges this claim

Not citing Scaling Monosemanticity in a 2026 mech interp paper is a significant gap. It was the most discussed interpretability result of 2024.

**3. Missing: Marks et al. 2024 "Sparse Feature Circuits" / Anthropic's feature-circuit work**
This line of work (building on Bricken 2023 and Templeton 2024) connects individual features to circuits, bridging the gap between feature-level and circuit-level interpretability. The present paper occupies exactly this bridge — it identifies features (neurons) and connects them to a functional program (circuit). Not citing the SAE-based circuit work misses an opportunity to position the paper's manual approach against the automated/SAE approach.

### 🟡 MODERATE

**4. The §2 Background section is well-organized but reads as a literature dump**
The section has 6 paragraphs, each covering a different area (MLP interpretability, Transformer circuits, Binary routing, Layer-wise development, Attention sinks). Each paragraph is 4-6 sentences of "X showed Y. Z extended this. W applied it to…" This is functional but reads like a checklist rather than a narrative. Compare the best NeurIPS related work sections, which tell a *story* — "The question of how MLPs process information has evolved through three phases: the key-value memory view (Geva 2021), the factual editing view (Meng 2022, Dai 2022), and the circuits view (Elhage 2021, Wang 2023). Our work suggests a fourth phase: MLPs as routing programs."

The current structure doesn't build toward the paper's contribution — it just lists what others did. A narrative framing would make the contribution's novelty clearer and preempt the "how is this different from Geva?" question.

**5. Missing: Neel Nanda's work on grokking and mechanistic interpretability of learned algorithms**
Nanda et al. 2023 "Progress measures for grokking via mechanistic interpretability" showed that interpretability can reveal learned algorithms in toy models. The present paper claims to have found a "learned algorithm" (the routing program) in a real model. Citing Nanda's work would strengthen the claim that models can implement legible algorithms, and differentiate by scale (toy model → real model at GPT-2 scale).

**6. Missing: Hanna et al. 2024 "How does GPT-2 compute greater-than over the list?" and related circuit analyses**
This paper and Merullo et al. 2024 provide circuit-level analyses of specific behaviors in GPT-2 — the same model. They're natural comparisons for the "what does GPT-2 do internally?" question and would round out the Transformer circuits paragraph.

**7. Dettmers et al. 2022 (LLM.int8()) is missing but relevant**
The paper discusses outlier features and the finding that a small number of features account for most activation magnitude. This is relevant to the Core neurons (5 neurons = 54% of output norm), which are exactly the kind of outlier features Dettmers describes. Citing this would ground the Core's large magnitude in a known phenomenon.

**8. The Dai et al. engagement in §5.4 is strong but slightly unfair**
The "Belial's Counsel" framing positions Dai et al. as *deceived* — they were misled by integrated gradients into thinking knowledge neurons store knowledge. But Dai et al. worked on BERT, not GPT-2, and used suppression experiments (not just attribution) showing that specific neurons were necessary for specific facts. The present paper's replication is on a different model (GPT-2) at a different layer (L11, the terminal layer). It's possible that:
- Knowledge neurons genuinely store facts in BERT's bidirectional architecture but not in GPT-2's autoregressive one
- Knowledge neurons store facts at mid-layers but not at the terminal layer

The paper acknowledges the model difference ("Our §5.4 tests this on GPT-2") but doesn't acknowledge that the results might not transfer back to BERT. The Belial framing implies Dai et al. were *wrong*; a fairer framing is that the phenomenon may be architecture-dependent. Add one sentence: "Whether Dai et al.'s findings in BERT — a bidirectional model with different information flow — represent genuine knowledge storage or also reduce to routing remains an open question."

### 🟢 MINOR

**9. Voita et al. 2019 "Analyzing Multi-Head Self-Attention" — cited but not discussed**
Listed in §2 alongside Clark 2019 in the attention paragraph but gets no specific engagement. Either remove (Clark 2019 makes the same point) or add a sentence about Voita's contribution (showing most attention heads are prunable, relevant to H7's dominance).

**10. Xiao et al. 2023 (attention sinks) is well-cited and well-used ✓**
The connection between H7's BOS attention (45.4%) and the attention sink phenomenon is clear and well-supported. Good citation practice.

**11. Olsson et al. 2022 (induction heads) is cited but the connection is loose**
§2 says Olsson et al. "identified induction heads as a fundamental circuit motif." This is true but the connection to the present paper is unclear — induction heads are an attention mechanism, and this paper is about MLP routing. Either make the connection explicit (e.g., "induction heads in attention provide the signal that MLP routing amplifies or suppresses") or drop the citation if it's only included for completeness.

**12. The paper correctly doesn't cite Anthropic's Constitutional AI, RLHF, or alignment work**
These are often shoehorned into mech interp papers for Anthropic-reviewer goodwill. This paper correctly limits citations to technically relevant work. Good editorial discipline.

**13. Missing but minor: Li et al. 2023 "Inference-Time Intervention"**
ITI provides an alternative to ROME for steering model behavior via activation modification. Relevant to the efficiency discussion (§7.3) — if the consensus detector can identify when MLP is counterproductive, ITI-style intervention could bypass or redirect the MLP at those positions. Minor citation but would strengthen the practical implications.

### Summary

The reference expansion from 8 to ~24 was a major improvement, and the critical gaps from Pass 3 (Dai et al., ROME, Elhage, Wang, Belrose) are now addressed. However, three gaps remain:

1. **Geva 2023 engagement** (item 1) — the paper concedes the tension in one sentence without resolving it. This is the most likely rejection trigger from a knowledge-editing reviewer, because Geva 2023 directly contradicts the "MLP doesn't retrieve" claim with step-by-step evidence.

2. **Scaling Monosemanticity** (item 2) — not citing Templeton et al. 2024 in a 2026 mech interp paper is conspicuous. The SAE line of work directly challenges the claim that ~3,040 residual neurons are "irreducibly distributed" — SAE methods might decompose them further.

3. **Dai et al. fairness** (item 8) — the Belial framing is rhetorically effective but slightly unfair. The BERT/GPT-2 difference should be acknowledged more prominently.

The §2 Background is functional but could be more narrative (item 4). The remaining missing citations (items 5-7, 13) are moderate gaps, not critical.

**Priority fixes:**
1. Expand Geva 2023 engagement to a full paragraph in §5 (explain why findings differ)
2. Cite Templeton et al. 2024 and engage with SAE challenge to "irreducibly distributed" claim
3. Acknowledge BERT/GPT-2 architecture difference more prominently in §5.4
4. Restructure §2 from literature dump to narrative arc
5. Add Dettmers 2022, Nanda 2023, Hanna 2024 to round out citations to ~28-30

---

## 2026-03-29 — Pass 16: Pre-Submission Readiness Audit

Focus: Cross-cutting synthesis of unresolved issues from all 15 prior passes. What MUST be fixed before arXiv? What can wait? Overall assessment of submission readiness.

### Outstanding Critical Issues (Must-Fix Before arXiv)

**1. N2123 dual fire-rate inconsistency (Passes 8/10/13, still open)**
The paper uses 11.3% (routing definition) and 70.9% (|GELU| > 0.1 enrichment definition) for the same neuron without reconciling them. Table 7's caption now has a 46-word explanation, but it's buried in a caption rather than addressed in Methods or main text. A reviewer will see "fires 11.3%" in §4 and "base rate 70.9%" in Table 7 and assume an error. **Fix: Add 2–3 sentences to §3.2 defining both thresholds explicitly and explaining their relationship. Reference this from Table 7's caption.**

**2. Garden-path claim still over-interpreted (Passes 8/10/11)**
Three related issues remain:
- No power analysis for the null result on N2123 ($t(14) = -0.45, p = 0.66$)
- No manipulation check that GPT-2 treats the verbs as intransitive/transitive
- Conclusion still says "confirming the exception handler operates at token-level predictability, not syntactic structure" — "confirming" is too strong for a null at N=15

The Wilcoxon test on surprisal reversal ($W = 12, p = 0.018$) is good and properly reported. The issue is the framing around the *null* N2123 result.

**Fix: (a) Change "confirming" to "consistent with" in the conclusion. (b) Add one sentence in §6.6: "This test has limited power (N=15); we report it as consistent with but not conclusive for the token-level predictability interpretation." (c) Optionally add the manipulation check (P(NP | verb) for each verb) — this would take ~5 lines of code and strengthen internal validity significantly.**

**3. Geva 2023 engagement still too thin (Passes 3/15)**
One sentence in §2 acknowledges the tension; no resolution offered. Geva 2023 found MLPs *promote* correct factual attributes at subject-last positions. This paper says L11's MLP *doesn't retrieve* facts. These are in tension.

**Fix: Add a paragraph in §5.1 or §5.4 addressing why: (a) Geva 2023 studies mid-layer MLPs, not the terminal layer — different functional role; (b) "promotion" in mid-layers is compatible with "routing" — the MLP amplifies attention-derived signals (which is what this paper also finds for constrained completions); (c) the distinction sharpens at L11, where routing dominates because it's the last intervention point. This reconciliation actually *strengthens* the terminal crystallization argument.**

**4. Scaling Monosemanticity not cited (Pass 15)**
Templeton et al. 2024 is the most prominent mech interp result of 2024. Its relevance: SAE decomposition of the ~3,040 "irreducibly distributed" residual neurons might reveal additional structure the paper's method can't detect. Not citing it is conspicuous.

**Fix: Add to §2 (MLP interpretability paragraph): "Templeton et al. (2024) extended sparse autoencoder methods to production-scale models, finding interpretable features in superposition — an approach that could potentially decompose the ~3,040 residual neurons our manual method cannot resolve." This also preempts the reviewer question "have you tried SAEs?"**

### Outstanding Moderate Issues (Should Fix, Lower Priority)

**5. "Complete account" language in intro (Passes 9/10)**
Intro still says "a complete, legible account of what an MLP layer *does*." Should be "of an MLP layer's *routing logic*." The abstract got this right; the intro didn't follow.

**6. "One-stage parser" overgeneralized in conclusion (Passes 9/10)**
Conclusion should scope to "GPT-2 Small behaves as a one-stage parser for verb subcategorization ambiguities" rather than implying all transformers and all constructions.

**7. Figure 1 ellipsis in Differentiator list (Pass 14)**
The pseudocode figure — the paper's headline visual — uses "..." for 6/10 Differentiator neurons. List all 10 or say "+8 others." Trivial fix, high visual impact.

**8. 36.5× enrichment baseline is naive (Passes 8/10)**
Uniform chance ($27/3072 \times 20$) is the wrong null for top-20 integrated-gradient neurons, which are selected for high causal influence. The paper now has a parenthetical acknowledging this ("the true enrichment over a causally-matched baseline would be lower"), which is adequate for arXiv but should be expanded for venue submission.

**9. Bootstrap description in Table 4 caption, not in Methods (Pass 13)**
The sequence-level bootstrap procedure is described only in the Table 4 caption. Should be in §3 as the general statistical framework.

**10. Belial paragraph too literary for a results section (Pass 12)**
The 4-line Paradise Lost explication in §5.4 slows down a results section. Compress to one sentence; move the Milton quote to a footnote.

### Confirmed Fixed (For Tracking)

All items from Passes 1–15 marked ✅ in previous passes remain fixed in the current text. Key fixes verified:
- ✅ Author block correct (name + real email, no "Independent Researcher")
- ✅ Token count consistent at 204,800
- ✅ Bootstrap CIs on crossover (Table 4)
- ✅ Ablation CIs added (Table 5)
- ✅ Null model (random init) in §7.1
- ✅ Threshold robustness sweep in §7.1
- ✅ Tuned lens validation (Table 6, Figure 2)
- ✅ Cross-layer survey all 12 layers (Table 9)
- ✅ 160-prompt knowledge test with categories (Table 2)
- ✅ Enrichment method: Fisher's exact + Bonferroni (Table 7)
- ✅ Figures added (4 total: pseudocode, logit lens, crossover, tier contribution)
- ✅ Pseudocode figure with "diagnostic, not causal" comment (Pass 11 item 1 partially addressed)
- ✅ References expanded to ~24
- ✅ Dai et al., ROME, Elhage, Wang, Belrose all cited and engaged
- ✅ Wilcoxon test on garden-path surprisal reversal ($W = 12, p = 0.018$)
- ✅ DC offset framing for Core narrative
- ✅ "To numerical precision" (not "exactly")
- ✅ Closing line concrete ("We can read 27 neurons… The 3,040 neurons they route remain opaque.")

### Assessment: ArXiv Readiness

**Status: Near-ready. 4 must-fix items, ~6 should-fix items.**

The paper is substantially stronger than the original version reviewed in Pass 1. The experimental foundation is solid: 160-prompt knowledge test, bootstrap CIs, null model, tuned lens, all-layer survey, enrichment statistics with proper corrections. The writing is distinctive (Milton thread gives the paper a memorable identity) and mostly well-executed.

The 4 must-fix items (N2123 dual threshold, garden-path over-interpretation, Geva 2023 engagement, Scaling Monosemanticity citation) are all addressable in a single focused editing session — estimated 2–3 hours of work. None require new experiments.

The should-fix items are framing and polish: scoping claims, improving figure details, relocating methods descriptions. These can be done in parallel or deferred to the camera-ready version if submitting to a venue.

**Recommendation: Fix items 1–4 before arXiv posting. Items 5–10 can be addressed in parallel or in the next revision cycle. The paper is ready for preprint with those four fixes.**

### What Worked Well in This Review Process

Looking back over 16 passes:
- The rotating focus ensured no category was neglected
- Second-round passes on the same topic caught issues the first round missed (e.g., Pass 8 found the dual-threshold issue that Pass 1's statistics pass didn't)
- The paper improved dramatically: 0→4 figures, 8→24 references, 12→160 prompts, no CIs → bootstrap CIs everywhere, no controls → null model + threshold sweep
- The revision verification pass (Pass 10) was essential for tracking what was actually fixed vs. flagged-but-unfixed
- The most impactful single pass was Pass 3 (Related Work) — going from 8 to 24 references addressed the paper's biggest vulnerability

**Next pass suggestion:** If reviews continue, Pass 17 should be a "Devil's Advocate" pass — read the paper as a hostile reviewer looking for the single strongest rejection argument, then draft the author response.

---

## 2026-04-01 — Pass 17: Devil's Advocate

Focus: Read the paper as a hostile but competent reviewer (R2, the skeptic). Identify the single strongest rejection argument, secondary rejection arguments, and draft author rebuttals. Goal: stress-test the paper before it encounters real reviewers.

### The Strongest Rejection Argument

**"This paper characterizes 0.9% of neurons in one layer of one small, obsolete model and calls it a 'complete, legible routing program.' The claims far outstrip the evidence."**

A hostile reviewer would frame it like this:

> GPT-2 Small (124M parameters, 2019) is 5+ generations behind current frontier models. The paper studies exactly one MLP layer (L11) of this one model. It characterizes 27 of 3,072 neurons (0.9%) and explicitly acknowledges the remaining 99.1% are "distributed and not directly legible." Yet the abstract promises "a fully legible routing program," the intro claims to provide "a complete, legible account," and the framing implies general truths about how MLPs work ("knowledge neurons are routing infrastructure," "terminal crystallization").
>
> The terminal crystallization prediction (structure at the final layer of deeper models) is untested. The knowledge-neuron rebuttal is tested only at L11 of GPT-2 Small but framed as overturning Dai et al.'s findings on BERT. The garden-path experiment uses 15 sentence pairs on one model and concludes GPT-2 is "a one-stage parser."
>
> Every interesting claim in this paper is either (a) scoped to one layer of one model with no generalization evidence, or (b) stated as a general truth without adequate generalization evidence. The gap between evidence scope and claim scope is the paper's fatal flaw.

**Why this is dangerous:** It's largely correct. The paper *is* a single-model, single-layer study making broad claims. The defense must be: (a) single-model depth studies have high value in mech interp (cf. Wang et al. 2023 IOI, also one model), (b) the terminal crystallization prediction is explicitly testable and we flag it as such, (c) the knowledge-neuron finding is scoped in Limitations, and (d) "0.9% of neurons" misframes the contribution — we characterize 100% of the *routing logic* while showing the remaining neurons serve a different function (distributed knowledge). But this defense needs to be *in the paper*, not just in the rebuttal.

### Secondary Rejection Arguments

**2. "The 'routing program' is a post-hoc narrative imposed on activation statistics, not a discovered algorithm."**

> The paper takes neurons with different fire rates, groups them into tiers by fire rate, and calls this a "three-tier exception handler." But any set of neurons binned by activation frequency will produce apparent tiers. The null model control (random init) addresses one version of this concern, but the deeper issue remains: is this a *program* or a *statistical description*?
>
> The pseudocode (Figure 1) implies algorithmic structure — `if exception: Core → Diff → Spec`. But the paper simultaneously says N2123 is "diagnostic, not causal" and that "the routing decision is distributed, not controlled by any single neuron." If no neuron controls the routing, there is no `if` statement. The pseudocode is metaphor dressed as mechanism.

**Rebuttal:** The pseudocode captures *functional organization* — how neurons coordinate — not literal control flow. The diagnostic/causal distinction is explicitly stated. The three tiers are not arbitrary fire-rate bins: they have distinct functional roles (vocabulary reset, candidate suppression, boundary detection) confirmed by output direction analysis and ablation. The null model shows untrained networks don't produce this structure. The threshold robustness analysis shows the tiers survive across 2 orders of magnitude. But the paper should more explicitly frame the pseudocode as a *functional description* rather than a *causal mechanism*.

**3. "The knowledge retrieval negative result is undermined by the paper's own data."**

> Table 2 shows that 18/160 prompts (11%) *do* reach top-10 via context-dependent accumulation. Historical events achieve 4/10 top-10. This isn't "the MLP doesn't retrieve facts" — it's "the MLP retrieves some facts but not others." The paper frames this as "amplification, not retrieval," but retrieval of highly constrained completions *is* a form of retrieval. The distinction between "amplification" and "retrieval" is semantic, not empirical.
>
> Furthermore, Geva et al. (2023) showed that MLPs at subject-last positions promote correct factual attributes — a finding the paper acknowledges in one sentence (§2) without substantive engagement. If mid-layer MLPs promote facts and L11 sometimes retrieves constrained facts (11% of the time), the "MLP doesn't retrieve" framing is misleading.

**Rebuttal:** The 11% success rate is precisely the evidence for "amplification" — it succeeds only when there's essentially one plausible completion (speed of light → "second", 100 degrees → "C"), meaning the MLP is narrowing an already-tiny candidate set, not selecting from many options. Arbitrary associations (capitals, languages) uniformly fail. The Geva 2023 reconciliation belongs in §5 and should explain the mid-layer vs. terminal-layer distinction explicitly. *This is the paper's most under-defended claim and the most likely source of reviewer pushback.*

**4. "The garden-path finding is interesting but statistically inadequate."**

> N=15 pairs, no power analysis, no manipulation check, null result on the primary measure (N2123 activation), significant result only on a secondary measure (surprisal reversal) tested post-hoc. The reversed garden-path effect is presented as the paper's most novel psycholinguistic finding but rests on a Wilcoxon test with $p = 0.018$ — not corrected for the 4 measures tested (N2123, consensus, surprisal, MLP delta). After Bonferroni correction across 4 tests, $p = 0.072$ — no longer significant.

**Rebuttal:** The surprisal reversal was the predicted direction based on the one-stage parser hypothesis, not a post-hoc finding — but the paper doesn't state this clearly enough. The 4 measures are not independent (they all probe the same phenomenon from different angles), so Bonferroni is conservative. Nevertheless, $p = 0.018$ uncorrected on N=15 is thin for a headline finding. **The paper should either: (a) expand to 30+ pairs to get convincing power, (b) present the garden-path result as "suggestive/exploratory" rather than "confirmed," or (c) pre-register the prediction and run a stronger test.** Currently the claim-to-evidence ratio is unfavorable.

**5. "The paper ignores SAE-based interpretability, which could resolve the 'opaque' residual neurons."**

> Bricken et al. (2023) and Templeton et al. (2024) showed that sparse autoencoders can decompose MLP activations into interpretable monosemantic features. The paper claims ~3,040 residual neurons are "distributed and not directly legible" without attempting SAE decomposition. This is like claiming a text is unreadable without trying to decode it. If SAE methods could decompose the residual into interpretable features, the paper's central dichotomy (27 legible neurons vs. 3,040 opaque ones) collapses.

**Rebuttal:** SAE decomposition is complementary, not contradictory — SAE features and individual neurons operate at different levels of description. The paper *should* cite Templeton 2024 and acknowledge this possibility: "Sparse autoencoder methods (Bricken et al. 2023, Templeton et al. 2024) could potentially resolve additional structure within the residual neurons; our manual decomposition establishes the routing/knowledge boundary that SAE methods could then refine." This turns a weakness into a future-work opportunity. **The missing Templeton citation is the single easiest improvement the paper can make to preempt this objection.**

### Mock Review Score

If I were assigning scores as a NeurIPS reviewer right now:

- **Soundness:** 3/4 — Experimental methodology is solid post-revision (null model, bootstrap, threshold robustness), but the knowledge-retrieval claim and garden-path claim are both somewhat overclaimed for their evidence.
- **Significance:** 3/4 — The routing program decomposition is genuinely novel and the consensus-crossover finding is clean. But single-model scope limits impact.
- **Novelty:** 3/4 — The pseudocode representation of MLP function is new. The "knowledge neurons are routing" finding adds to an ongoing debate. The garden-path reversal is surprising if it holds up.
- **Clarity:** 3.5/4 — Distinctive voice, good figures post-revision, clear structure. The Milton thread is memorable. The dual-threshold issue and diagnostic/causal tension hurt clarity.
- **Overall:** 6/10 — Borderline accept. Strong execution on a limited-scope study. The gap between evidence scope (one layer, one model) and claim scope (general truths about MLPs, knowledge neurons, parsing) is the main weakness. Tightening claims to match evidence would push this to 7/10.

### Top 5 Actions to Improve Acceptance Probability

1. **Scope the claims to match the evidence.** Intro: "routing logic" not "what an MLP does." Knowledge neurons: "at L11 of GPT-2 Small" not universal. Garden-path: "suggestive" not "confirmed." This is 30 minutes of editing and removes the strongest rejection argument.

2. **Cite Templeton 2024 and engage with the SAE challenge.** One paragraph in §2 or §7. Preempts objection 5 and shows awareness of the field's trajectory. 15 minutes.

3. **Expand Geva 2023 engagement.** A paragraph in §5.1 explaining mid-layer promotion vs. terminal-layer routing. Preempts objection 3 and strengthens the terminal crystallization argument. 20 minutes.

4. **Clarify the N2123 dual threshold in §3.2.** Three sentences defining "activates" (70.9%) vs. "exception path active" (11.3%) and their relationship. Removes the most confusing inconsistency. 10 minutes.

5. **Soften garden-path to "exploratory."** Change "confirming" to "consistent with." Add power caveat. Note Bonferroni concern. Reframe as hypothesis-generating rather than hypothesis-confirming. 15 minutes.

**Total estimated time for all 5: ~90 minutes. These would collectively shift the paper from borderline-accept to solid accept territory.**

### What the Paper Does Right (Positive Reviewer Perspective)

A fair review should also note what works:
- The pseudocode representation (Figure 1) is genuinely novel — no prior mech interp paper has presented MLP function this way
- The consensus-crossover finding (Table 4) is the paper's strongest quantitative result: clean, well-powered, properly bootstrapped
- The 160-prompt knowledge test is thorough and the category breakdown reveals mechanism, not just failure
- The null model, threshold robustness, and tuned lens validation show methodological care
- The writing is distinctive and memorable — the Milton thread gives the paper an identity most mech interp papers lack
- The "program and database" analogy is a useful conceptual contribution that could influence how people think about MLP layers

**Priority fixes for arXiv:** Items 1–5 above, in order. The paper's experimental foundation is solid; the remaining work is claim-scoping and citation completeness.

---

## 2026-04-02 — Pass 18: Scope Alignment & Claim-Evidence Audit

Focus: For each major claim in the abstract/intro/conclusion, verify that the evidence in the paper actually supports the claim *as stated*. The Devil's Advocate pass (Pass 17) identified "evidence scope vs. claim scope" as the strongest rejection argument. This pass checks every claim systematically.

### Claim-by-Claim Audit

**Claim 1 (Abstract): "The final MLP of GPT-2 Small implements a fully legible routing program — 27 named neurons organized into a three-tier exception handler"**

- *Evidence:* Table 1 (tiers with fire rates, Jaccard), Table 2 (consensus neuron characterizations), Figure 1 (pseudocode), Table 7 (enrichment stats with Fisher's exact + Bonferroni), null model control, threshold robustness sweep.
- *Scope match:* ✅ Well-scoped. Says "GPT-2 Small," says "routing program," doesn't overclaim beyond L11.
- *Issue:* The word "implements" implies mechanism; the paper says the program is "diagnostic, not causal." A routing program that nothing "implements" is a description, not a program. See item below.

**Claim 2 (Abstract): "the knowledge it routes remains entangled across ~3,040 residual neurons"**

- *Evidence:* Table 2's knowledge extraction test (160 prompts, 11% top-10 rate), progressive prediction (10/12 prompts show the pattern).
- *Scope match:* ✅ Appropriate. "Entangled" is honest — doesn't say "irreducibly distributed" (which would be stronger and less defensible given SAE methods).
- *Minor:* The abstract doesn't note that 11% of prompts *do* succeed, which complicates "entangled." But "entangled" doesn't mean "impossible to extract," just "not individually legible," which is accurate.

**Claim 3 (Abstract): "The consensus-exception crossover — where MLP intervention shifts from helpful to harmful — is statistically sharp (bootstrap 95% CIs exclude zero at all consensus levels; crossover between 4/7 and 5/7)"**

- *Evidence:* Table 4 (204,800 tokens, sequence-level bootstrap, all 8 CIs exclude zero).
- *Scope match:* ✅ This is the paper's best-supported claim. Clean data, proper statistics, large sample. No overclaim.

**Claim 4 (Abstract): "Three experiments show that 'knowledge neurons' are routing infrastructure, not fact storage"**

- *Evidence:* Exp 1: attribution overlap (36.5× enrichment, 20 prompts). Exp 2: knockout (+8.0pp, opposite of storage prediction). Exp 3: activation transplant (0/5, labeled illustrative).
- *Scope match:* 🟡 Partially overclaimed. The claim is stated as a general truth ("knowledge neurons are routing infrastructure") but the evidence is from L11 of GPT-2 Small only. Dai et al. (2022) worked on BERT. The paper now has a sentence in §5.4 acknowledging the BERT/GPT-2 difference ("Whether Dai et al.'s findings in BERT... represent genuine storage or also reduce to routing remains an open question") — but the abstract doesn't carry this qualifier.
- **Fix needed:** Abstract should say "at L11" or "in GPT-2 Small" — adding 3 words scopes the claim properly without weakening it.

**Claim 5 (Abstract): "A garden-path experiment reveals a reversed garden-path effect — GPT-2 uses verb subcategorization immediately, consistent with the exception handler operating at token-level predictability rather than syntactic structure"**

- *Evidence:* 15 minimal pairs, Wilcoxon W=12, p=0.018 on surprisal reversal. N2123 null result t(14)=-0.45, p=0.66, powered for d≥0.78.
- *Scope match:* 🟡 The surprisal reversal is supported (p=0.018). The "consistent with" framing for the exception handler claim is appropriately hedged. But "powered for d≥0.78" means only large effects are detectable — a medium effect (d=0.5) on N2123 would be missed 66% of the time. The abstract doesn't flag this.
- *Status:* Acceptable for arXiv. The "consistent with" hedging does the work. The power limitation is in the appendix.

**Claim 6 (Abstract): "This architecture crystallizes only at the terminal layer — in deeper models, we predict equivalent structure at the final layer, not at layer 11"**

- *Evidence:* Table 9 (all 12 layers surveyed, max Jaccard 0.384 at any non-L11 layer vs. 0.998 at L11).
- *Scope match:* ✅ Well-phrased. "We predict" explicitly flags it as untested. The cross-layer survey is thorough (all 12 layers, not a subset).

**Claim 7 (Intro §1, item 2): "'Knowledge neurons' are routing infrastructure"**

- Repeats Claim 4. Same scope concern. The intro version is slightly better: "Neurons identified by Dai et al. as storing factual knowledge appear across nearly all facts tested — they are highway signs, not warehouses."
- *Issue:* "Highway signs, not warehouses" is vivid but scoped to the intro's context of L11. The abstract's version lacks this scoping.

**Claim 8 (§5.1): "This is amplification, not retrieval"**

- *Evidence:* 160 prompts, 18/160 (11%) reach top-10 via context-dependent accumulation. Category breakdown shows success correlates with constraint (historical events 4/10 vs. capitals 0/15).
- *Scope match:* 🟡 The claim is stated as a binary ("not retrieval") but the evidence shows a gradient. 11% success rate for highly constrained completions *is* a form of retrieval. The paper reframes this as "amplification" — boosting already-strong signals — which is a reasonable interpretation but presented as dichotomous when the data is continuous.
- **Suggestion:** "Primarily amplification rather than independent retrieval" would be more accurate than the current binary.

**Claim 9 (§4.1): "a fused mega-neuron… vocabulary reset"**

- *Evidence:* Jaccard ≥0.91 (peak 0.998), output directions toward function words, 54% output norm but +0.2% PPL (CI [+0.1%, +0.4%]).
- *Scope match:* ✅ The DC offset framing resolves the magnitude-vs-impact paradox. The Simpson's paradox note (+2.4% at low consensus, −0.4% at high) explains the apparent contradiction.

**Claim 10 (§7, Discussion): "Bypassing the MLP for 7/7 consensus tokens (40% of traffic) costs only 6.9% PPL"**

- *Evidence:* Not shown in any table. This appears to be stated without supporting data in the main text.
- *Scope match:* 🔴 **Where does 6.9% come from?** The ablation table (Table 5) shows PPL impacts of *tier-level* ablation, not *consensus-conditional* bypass. 7/7 tokens are 77,950/204,800 = 38% (close to 40%). But the PPL impact of bypassing MLP *only* for 7/7 tokens isn't in any table. This is either (a) computed but not shown, (b) estimated from Table 4's ΔP values, or (c) unreferenced. **This needs a citation to its own table or a computation shown in the appendix.** A reviewer will ask "where does 6.9% come from?"

### 🔴 CRITICAL FINDINGS

**1. The 6.9% PPL bypass claim (Claim 10) has no supporting data in the paper**
This is a new issue not caught in any prior pass. The Discussion section claims a specific efficiency gain (6.9% PPL for 40% compute savings) that doesn't correspond to any table or experiment. Either add a supplementary table computing PPL when L11 MLP is bypassed for 7/7 consensus tokens only, or remove/soften the claim. As stated, it's an unsubstantiated number.

**2. Abstract's knowledge-neuron claim (Claim 4) lacks model-scoping qualifier**
The abstract says "knowledge neurons are routing infrastructure, not fact storage" without scoping to GPT-2 Small or L11. The body (§5.4) has the appropriate scoping. Add "at L11" or "in this model" to the abstract — 3 words, major defensive improvement.

### 🟡 MODERATE FINDINGS

**3. "Implements" vs. "diagnostic, not causal" tension persists**
The abstract says "implements a fully legible routing program." The intro (item 1) says "diagnostic, not causal." These coexist in the paper but create cognitive dissonance. The intro's framing is the right one — add "captures" or "exhibits" instead of "implements" to the abstract. "Implements" implies the neurons *execute* the program; "exhibits" or "is described by" implies the program is a model of observed behavior. Subtle but important for the diagnostic-not-causal framing.

Alternatively: keep "implements" and lean into the argument that a diagnostic readout of a distributed computation *is* an implementation — just not a single-bottleneck one. The pseudocode captures real functional structure even though ablating any single neuron doesn't break the program. This is analogous to reading assembly code that describes a SIMD operation — no single lane is the "gate," but the program is real. If this is the intended meaning, one sentence in the intro making this argument would resolve the tension.

**4. "Amplification, not retrieval" is presented as binary; data shows a gradient**
The knowledge test reveals a continuous relationship between contextual constraint and MLP retrieval success. The binary framing ("not retrieval") is rhetorically clean but empirically inaccurate. "Primarily amplification, with limited retrieval capacity for highly constrained completions" would be more precise. This matters because the Geva 2023 tension (mid-layer MLPs promote attributes) is better resolved by a gradient model than a binary one.

**5. The diagnostic/causal distinction now appears in 5 places — approaching over-repetition**
- Abstract: not explicitly, but pseudocode caption addresses it
- Intro item 1: "diagnostic, not causal"
- Figure 1 caption: "diagnostic readout, not causal gate"
- §4 (Exception indicator paragraph): "It is not causal: zeroing it changes PPL by <0.1%"
- §2: "N2123 is diagnostic rather than causal"

Five statements of the same point. The first two are necessary (abstract and intro establish the framing). Figure 1 caption is good (in-context where readers might misread). §4 and §2 are both also good positions. But the cumulative effect is protesting too much — as if the paper doesn't fully believe its own framing. Consider dropping the §2 mention (which is just citing Balogh 2026) and letting the other four occurrences stand.

### 🟢 WELL-SCOPED CLAIMS (No Action Needed)

- Consensus-crossover: perfectly scoped, well-evidenced
- Terminal crystallization: explicitly flagged as a prediction
- Core as DC offset: well-explained Simpson's paradox
- Specialists as counterproductive: CI [-0.6%, -0.1%] excludes zero, properly hedged
- Garden-path "consistent with": appropriate hedging for N=15
- "~3,040 residual neurons" as entangled: honest about what's not decomposed

### Summary

The paper's claim-to-evidence alignment is generally good post-revision. The three remaining scope mismatches:

1. **6.9% PPL bypass** — unsupported number in Discussion (new finding, not caught before)
2. **Abstract knowledge-neuron scope** — needs "at L11" or model qualifier (3-word fix)
3. **"Implements" vs. diagnostic framing** — either change verb or add one sentence of argument

Of these, #1 is the most concerning because it's a specific quantitative claim with no backing data. A reviewer checking Table 5 for "6.9%" won't find it. Either show the computation or remove the number.

**Priority fixes:**
1. Support or remove the 6.9% bypass claim (add table/computation, or soften to qualitative)
2. Add model scope to abstract's knowledge-neuron claim
3. Resolve "implements" vs. diagnostic tension with one sentence of argument
4. Consider softening "amplification, not retrieval" to acknowledge the gradient

---

## 2026-04-03 — Pass 19: Reproducibility & Technical Precision

Focus: Can an independent researcher reproduce the key results from what the paper describes? Are equations, definitions, and numerical claims internally consistent and sufficiently specified? This is distinct from the methodology passes (design-level) — this is implementation-level verification.

### 🔴 CRITICAL

**1. The "high-magnitude" threshold for N2123's exception-path regime is never numerically defined**
§3 says: "At the generic threshold, N2123 activates on 70.9% of tokens. However, only 11.3% produce the *high-magnitude* activations that co-occur with exception-path behavior." But what defines "high-magnitude"? The generic threshold is |GELU(x)| > 0.1. Is the high-magnitude threshold |GELU(x)| > 1.0? > 2.0? The 95th percentile of N2123's activation distribution? A bimodal separation point?

This is the single most important reproducibility gap in the paper. The entire architecture — tiers, consensus crossover, pseudocode — depends on which 11.3% of tokens are labeled "exception path." Without a numerical definition, no one can replicate the core finding. The threshold robustness analysis (§7.1) sweeps the *generic* threshold across 0.01–1.0, but never defines or sweeps the *exception-path* threshold.

**Fix:** Add to §3: "We define exception-path activation as N2123 post-GELU activation exceeding [X], corresponding to the [bimodal separation / top 11.3% / specific criterion]. This threshold was selected because [distribution shows clear bimodal separation at X / it maximizes the Jaccard gap between Core and non-Core neurons / etc.]." Without this, the paper's central claim is unreproducible.

**2. The MLP decomposition equation lacks the GELU activation step**
§3 gives: $\text{MLP}(x) = \text{Core}(x) + \text{Diff}(x) + \text{Spec}(x) + \text{Residual}(x)$, and says "each tier applies a binary mask to post-GELU activations before projecting through the output matrix." But the full computation is:
$$\text{MLP}(x) = W_\text{proj} \cdot \text{GELU}(W_\text{fc} \cdot x + b_\text{fc}) + b_\text{proj}$$
The decomposition masks the intermediate activation vector $h = \text{GELU}(W_\text{fc} \cdot x + b_\text{fc})$. So:
$$\text{Core}(x) = W_\text{proj}[:, \mathcal{C}] \cdot (h \odot m_\mathcal{C})$$
where $\mathcal{C}$ is the Core neuron index set and $m_\mathcal{C}$ is a binary mask. But where does the bias $b_\text{proj}$ go? Is it assigned to one tier? Split proportionally? Absorbed into the residual?

This matters for the "preserves to numerical precision" claim (cosine sim 0.99999994). If the bias isn't handled, the decomposition is approximate. If it's handled but not described, the paper is under-specified.

**Fix:** Either show the full decomposition equation including bias handling, or add: "The output bias $b_\text{proj}$ is [assigned to the residual tier / split proportionally / absorbed into Core because...]." One sentence removes the ambiguity.

**3. "Contribution magnitude" for knowledge extraction accumulation is still undefined (flagged Pass 13, item 3)**
§5.1 says neurons are accumulated in order of "contribution magnitude" but doesn't define this. Possible meanings:
- $|h_n| \cdot \|W_\text{proj}[:, n]\|$ (activation × output direction norm)
- $|h_n \cdot W_\text{proj}[:, n]^\top e_\text{target}|$ (projection onto target token direction)
- $\|h_n \cdot W_\text{proj}[:, n]\|$ (output vector norm)

The first and third are target-agnostic; the second is target-specific. If target-specific, the test is biased toward finding retrieval (sorting by relevance to the answer). If target-agnostic, it's a fair test of whether generic neuron importance order recovers facts. This distinction matters for interpreting the 11% success rate.

**Fix:** Define "contribution magnitude" in §3 or §5.1. One sentence: "Neurons are accumulated by decreasing $|h_n| \cdot \|W_\text{proj}[:, n]\|_2$ (activation-weighted output norm, target-agnostic)." Or whatever the actual definition is.

### 🟡 MODERATE

**4. The Jaccard similarity computation assumes binary vectors, but the binarization is asymmetric**
Jaccard is computed on binary firing patterns: neuron fires (1) or doesn't (0) based on |GELU(x)| > 0.1. But GELU is asymmetric around zero — negative pre-activations are suppressed to near-zero, positive ones pass through. For neurons with negative mean pre-activation, |GELU(x)| > 0.1 will rarely trigger (low fire rate). For neurons with positive mean pre-activation, it'll trigger frequently (high fire rate).

This means Jaccard similarities between two high-firing neurons will be inflated by the shared high base rate (as noted in Pass 5, item 11). The paper has the random-init null model which mitigates this concern (untrained neurons have Jaccard ~0.53 vs. trained Core's 0.998). But the paper never states what Jaccard two *independent* neurons with the *same* marginal fire rates as the Core neurons would produce. With 90-100% fire rates, even independent neurons would have Jaccard ~0.81-1.0.

**Fix:** Add to §7.1 or a footnote: "For comparison, two independent neurons each firing at 95% would produce Jaccard = 0.90; the observed Core Jaccard of 0.998 exceeds this independence baseline by [X standard deviations under a permutation null]." This controls for the base-rate inflation.

**5. "max elementwise difference < 6 × 10⁻⁵" — in what space?**
Flagged in Pass 8 (item 8) but still unanswered. This is the decomposition fidelity metric. Is this measured on:
- The 3,072-dimensional intermediate activation vector?
- The 768-dimensional MLP output (post-projection)?
- The 50,257-dimensional logit vector?

The magnitude's significance depends on the space. In logit space, 6e-5 is negligible (typical logit values are ~±10). In probability space after softmax, it could matter for rare tokens. Specifying the space takes 3 words: "max elementwise difference in MLP output space < 6 × 10⁻⁵."

**6. Statistical methods section doesn't specify software or random seeds**
§3 says "bootstrap (10,000 resamples)" and "Fisher's exact test with Bonferroni correction" but doesn't mention: Python version, numpy/scipy version, random seed for bootstrap, or whether the Fisher's test uses an exact computation or an approximation (at N=512K, many implementations use chi-squared approximation). For reproducibility, at minimum state the software: "All statistical tests use scipy.stats (version X.Y) with random seed 42 for bootstrap resamples."

**7. Token count for garden-path experiment is ambiguous**
§6.6 says "15 minimal pairs" but doesn't specify: are these 15 pairs × 2 conditions = 30 sentences, each with multiple tokens measured? Which token position is "disambiguation"? The examples suggest it's the word after the ambiguity region ("took" in "After the dog struggled the vet took off the muzzle"), but is it always exactly one token? What about multi-token disambiguation words? The Wilcoxon test has $N = 15$ (one measurement per pair), but what exactly is measured — surprisal at a single token, or average surprisal over the disambiguation region?

**8. The "7 consensus neurons" selection criterion is underspecified (flagged Pass 11, item 10)**
How were these 7 identified from 3,072? The paper says they "monitor distinct linguistic properties" and have fire rates of 79-91% (Table 2). But many neurons probably fire at 79-91%. What separates these 7 from, say, 50 other neurons with similar fire rates? Is the criterion:
- Correlation with exception-path suppression?
- Manual inspection of enriched/depleted tokens?
- Output direction similarity to a "consensus axis"?
- Imported from Balogh 2026?

If imported from the prior paper, say so. If independently rediscovered, describe the selection process. This matters because the consensus-crossover finding (the paper's strongest result) depends on *which* 7 neurons define the consensus count.

### 🟢 MINOR

**9. "512,000 tokens from WikiText-103, processed in 500 sequences of 1,024 tokens each" — verified ✓**
500 × 1,024 = 512,000. The split into characterization (512K) and evaluation (204,800) datasets is stated. The 204,800 = 200 sequences × 1,024. Arithmetic is consistent.

**10. Table 4 token counts sum correctly ✓**
206 + 913 + 3,051 + 6,283 + 12,617 + 34,155 + 69,625 + 77,950 = 204,800. Verified.

**11. The TransparentGPT2 wrapper approach is sound in principle**
Binary masks on post-GELU activations before projection is a clean, mathematically exact decomposition (given proper bias handling). The code is available at the anonymous repo. The approach doesn't modify weights, only observes and partitions the intermediate computation.

**12. Bonferroni correction factor is correctly computed**
α = 0.05 / 3,072 = 1.63 × 10⁻⁵. With p-values < 10⁻³⁰⁰, all enrichment results survive any reasonable correction.

### Summary

The paper's main reproducibility gap is the **undefined exception-path threshold for N2123** (item 1). This is the paper's most critical unreproducible element — the entire architecture story depends on which tokens are "exception-path," and the definition of that set isn't numerically specified. The MLP decomposition's bias handling (item 2) and the knowledge-test accumulation criterion (item 3) are secondary but both affect reproducibility.

A researcher reading this paper could reproduce the generic analysis pipeline (Fisher's exact on binary firing, Jaccard similarity, consensus counting, logit lens) but could *not* reproduce the specific 27-neuron architecture without knowing: (a) the exception-path activation threshold, (b) the neuron selection criteria for the 7 consensus neurons, and (c) the accumulation order for the knowledge test.

**Priority fixes:**
1. Define the exception-path threshold numerically (must-fix before arXiv — this is the paper's central construct)
2. Specify bias handling in the decomposition equation
3. Define "contribution magnitude" for knowledge accumulation
4. Add Jaccard independence baseline for high-fire-rate neurons
5. Specify consensus neuron selection criterion (or cite Balogh 2026 and state which are inherited)

---

## 2026-04-04 — Pass 20: Fresh Eyes on Current Text

Focus: Read the *current* paper as a first-time NeurIPS reviewer. Many issues from Passes 1–19 are now resolved in the text. This pass assesses the paper *as it stands today*, noting what a reviewer would flag, what's impressive, and what's the most likely path to rejection or acceptance.

**Key observation:** Comparing the current paper text against prior review notes, the following are NOW RESOLVED in the manuscript:
- ✅ N2123 bimodal threshold explicitly defined (GELU > 1.0, captures 11.3%, bimodal minimum)
- ✅ Threshold robustness range stated (0.7–1.5)
- ✅ Decomposition bias handling specified ($b_\text{proj}$ assigned to Residual tier)
- ✅ Full decomposition equation with masking notation
- ✅ Bootstrap: sequence-level, 10K resamples of 500 sequences, seed 42, within-sequence autocorrelation
- ✅ Software: scipy.stats v1.12
- ✅ Consensus neuron selection: >75% fire rate, >10× enrichment, cos sim >0.4 to mean MLP output
- ✅ Knowledge accumulation: $|h_n| \cdot \|W_\text{proj}[:, n]\|_2$, explicitly target-agnostic
- ✅ Jaccard base-rate contextualized (independent 95% neurons → ~0.90, observed 0.998)
- ✅ Random-init null model control (Jaccard 0.53 only)
- ✅ Intro scoped to "routing logic" not "what an MLP does"
- ✅ Abstract scopes knowledge neurons "at L11 of this model"
- ✅ Garden-path: power noted ("powered for d ≥ 0.78"), "consistent with" framing
- ✅ Wilcoxon test on surprisal reversal (W=12, p=0.018)
- ✅ 6.9% PPL claim REMOVED — Discussion now says "an empirical question we leave to future work"
- ✅ Figure 1 pseudocode lists all 10 Differentiator neurons (no ellipsis)
- ✅ Figure 1 has "diagnostic readout, not causal gate" comment
- ✅ Dettmers 2022 cited
- ✅ Author block correctly formatted (commented out for anonymous submission, de-anonymized version uses correct email, no "Independent Researcher")
- ✅ "To numerical precision" (not "exactly")
- ✅ Max elementwise difference specified as MLP output space (768-dim)
- ✅ Geva 2023 engaged in §2 with "more nuanced picture" acknowledgment
- ✅ Simpson's paradox for Core (+2.4% at low consensus, −0.4% at high)

### 🔴 CRITICAL — Still Outstanding

**1. Templeton et al. 2024 "Scaling Monosemanticity" — still not cited**
This remains the most conspicuous omission. In a 2026 mech interp paper, not citing Anthropic's highest-profile interpretability result is like a 2018 NLP paper not citing ELMo. The paper claims ~3,040 residual neurons are "entangled" — SAE methods from Templeton et al. specifically challenge this by extracting interpretable features from distributed representations. One paragraph in §2 engaging with this would preempt the obvious reviewer question ("have you tried SAEs on the residual neurons?") and would naturally cite both Bricken 2023 (already cited) and Templeton 2024. Suggested addition to §2 MLP interpretability paragraph:

> "\citet{templeton2024scaling} extended sparse autoencoder methods to production-scale models, suggesting that the $\sim$3{,}040 residual neurons our manual analysis cannot resolve may yield to SAE decomposition — a complementary approach that could refine the routing/knowledge boundary."

**2. Geva 2023 reconciliation is still only in §2, not in §5 where it matters**
The §2 engagement is better ("a more nuanced picture than our binary routing/retrieval framing"), but the actual reconciliation — *why* Geva 2023's mid-layer fact promotion doesn't contradict L11's routing role — is never made explicit. The strongest version: "Geva et al.'s finding that mid-layer MLPs promote correct attributes is compatible with our terminal-layer routing picture: earlier layers contribute to building the prediction, while the final layer routes rather than retrieves, consistent with terminal crystallization." This argument strengthens the paper's own thesis but is currently absent. A knowledge-editing reviewer who knows Geva 2023 well will notice the gap.

### 🟡 MODERATE — Remaining Concerns

**3. The "knowledge neurons are routing infrastructure" claim in Contribution 2 is still broad**
The intro says: "This reframes ROME: editing MLP weights changes routing decisions, not stored facts." This is stated as a general truth about ROME, not scoped to L11/GPT-2. ROME operates on GPT-J/GPT-2-XL at different layers. The reframing prediction ("edits should generalize across phrasings but fail across domains where routing structure differs") is interesting but untested. A sentence scoping this would help: "At L11 of GPT-2 Small, this reframes ROME..."

**4. The Belial footnote is delightful but the Moloch paragraph (§5.3) is underdeveloped**
The Belial footnote is perfectly placed and exactly the right length. But "Moloch's compulsion" in §5.3 appears as a paragraph heading with just two sentences: the MLP promotes tokens to top-1 while losing net probability, and the architecture has no bypass. The *why this matters* — that the transformer architecture forces a computational cost on high-consensus tokens with no benefit — deserves one more sentence connecting to the efficiency discussion in §7. Currently the Moloch metaphor is introduced and immediately dropped.

**5. The abstract is dense but now well-scoped**
At ~200 words, it's within NeurIPS limits. The neuron-count detail (5/10/5/7) is still heavy for an abstract but serves the "fully legible" claim — you need to show the decomposition is complete. Borderline. Could save ~15 words by condensing "5 fused Core neurons that reset vocabulary toward function words, 10 Differentiators that suppress wrong candidates, 5 Specialists that detect structural boundaries, and 7 Consensus neurons that each monitor a distinct linguistic dimension" to "5 Core (vocabulary reset), 10 Differentiators (candidate suppression), 5 Specialists (boundary detection), and 7 Consensus neurons (linguistic dimensions)" — but this is a style preference, not an error.

**6. No explicit statement of the paper's theoretical contribution vs. empirical contribution**
The paper blends theoretical claims (the MLP is a routing program, knowledge neurons are routing infrastructure, terminal crystallization) with empirical results (consensus-crossover statistics, ablation PPL, knowledge test ranks). A reviewer asking "what's the contribution?" might struggle to separate the (strong) empirical findings from the (more speculative) theoretical framing. Consider adding one sentence to the intro: "Our primary contribution is empirical: a complete neuron-by-neuron decomposition with statistical validation. The routing-program interpretation is a descriptive framework for organizing these findings."

**7. The cross-layer table (Table 9) groups early layers into ranges**
L0–L3, L4–L6, L7–L9 are shown as ranges. L10 and L11 get individual rows. The grouping is space-efficient but prevents readers from seeing whether there's a gradient (L9 closer to L11 than L0?) or a sharp break. Given the "terminal crystallization" claim — that structure appears *only* at L11 — showing each layer individually would be stronger evidence. An appendix table with all 12 individual rows would cost little and add precision.

### 🟢 POSITIVE — What Works Well in Current Text

**8. The pseudocode figure (Figure 1) is excellent**
The comments ("54% of output norm, +0.2% PPL" etc.) give the reader both structure and impact at a glance. The "diagnostic readout, not causal gate" comment addresses the diagnostic/causal tension *in situ* where readers will have the question. This figure is the paper's most distinctive contribution to mech interp communication.

**9. The consensus-crossover analysis (Table 4, Figure 2) is the paper's strongest result**
204,800 tokens, sequence-level bootstrap, all CIs exclude zero, clean crossover at 4–5/7. This is publication-quality statistical analysis. The blue/red visual encoding in Figure 2 makes the result immediately graspable. No reviewer can reasonably object to this finding's statistical support.

**10. The 160-prompt knowledge test with category breakdown is compelling**
The category structure (Table 2 bottom panel) reveals mechanism rather than just showing failure. The gradient from historical events (4/10 top-10) to capitals (0/15 top-10) supports "amplification, not retrieval" better than any aggregate statistic could. The "amplifies or suppresses signals already present in the residual stream, with capability indexed by contextual constraint" framing in the abstract is well-supported.

**11. The diagnostic/causal framing is now clean and consistent**
Intro item 1 establishes it ("diagnostic, not causal"), Figure 1 reinforces it, §4 (exception indicator paragraph) provides the evidence (zeroing changes PPL by <0.1%). The paper no longer protests too much — the point is made firmly in three well-chosen locations without excessive repetition.

**12. The garden-path finding is well-hedged**
"consistent with" in the abstract, "powered for d ≥ 0.78" in the appendix, "reversed effect" rather than "proves one-stage parsing." The Wilcoxon test (W=12, p=0.018) on surprisal reversal provides statistical support for the interesting positive finding while the N2123 null is properly characterized as limited-power.

**13. The Milton thread is distinctive without being overbearing**
Title → epigraph → "structured darkness" → Belial footnote → Moloch heading → closing line ("The 3,040 neurons they route remain opaque"). The thread is present throughout without demanding attention. The Belial footnote is the highlight — perfectly placed, perfectly toned.

### Assessment: Current Submission Readiness

**Status: ArXiv-ready with 2 recommended fixes. NeurIPS-competitive.**

The paper has addressed the vast majority of issues from 19 prior review passes. The experimental foundation is solid, the writing is distinctive, and the claims are now well-scoped.

**Must-fix before submission:**
1. Cite Templeton et al. 2024 (Scaling Monosemanticity) — one paragraph in §2
2. Expand Geva 2023 reconciliation — one paragraph in §5.1 or §7

**Should-fix (quality improvements):**
3. Scope ROME reframing to L11/GPT-2 in intro
4. Add individual-layer appendix table to supplement Table 9
5. One sentence connecting Moloch's compulsion to §7 efficiency discussion

**Mock reviewer scores for current text:**
- Soundness: 3.5/4
- Significance: 3/4 (single-model scope; strong within that scope)
- Novelty: 3.5/4 (pseudocode representation is genuinely new; routing program concept adds to the field)
- Clarity: 3.5/4 (distinctive voice, good figures, clean structure)
- Overall: 7/10 — Accept (weak). Would be 7.5–8 with the two must-fix items addressed.

**The paper's greatest strength:** The consensus-crossover result (Table 4) combined with the pseudocode representation (Figure 1) — these make a convincing case that MLP routing is legible and functionally meaningful, even if the scope is limited to one model.

**The paper's greatest vulnerability:** A reviewer who prioritizes breadth/generalization over depth/completeness will argue single-model findings on GPT-2 Small (2019, 124M params) have limited relevance to 2026 frontier models. The terminal crystallization prediction partially addresses this, but without validation on deeper models, it's a promissory note.

**Comparison to Pass 1 assessment:** The paper has improved dramatically — from ~15 critical issues across statistics, methodology, related work, and figures to 2 remaining citation/engagement gaps. The revision trajectory suggests a careful, responsive author who takes feedback seriously. This is itself a positive signal for the review process.

---

## 2026-04-06 — Pass 22: Writing Quality (Third Round)

Focus: Sentence-level prose quality, readability, register consistency, and remaining AI-detectable patterns in the current revised text. The paper has been substantially revised since Pass 12 (3/25). This pass reads the current text as a NeurIPS reviewer scanning for polish issues that signal either carelessness or machine-generated prose.

### Status of Pass 12 Issues

- ✅ Belial paragraph compressed (now a footnote, not a paragraph — elegant fix)
- ✅ Abstract neuron-count detail retained but well-structured with colons and commas
- ✅ "Consistent with" reduced from 5 to 3 uses — still present but no longer a tic
- ✅ Closing line concrete and effective (unchanged, as recommended)
- ⚠️ Abstract still dense at ~200 words — see below
- ⚠️ Moloch paragraph still brief — see below

### 🔴 CRITICAL

**1. Appendix B category table shows 7 of 15 claimed categories (data/writing mismatch)**
This was flagged in Pass 21 (item 1) as a data integrity issue but it's also a *writing* problem: the table's "All (160)" summary row contradicts the visible rows that sum to 75. A reader encountering this table will either (a) assume 8 categories are missing and wonder what they show, or (b) conclude the authors miscounted. Neither is good. The text in §5.1 says "160 factual cloze prompts spanning 15 categories" — so the table must show all 15 categories or the discrepancy must be explained. This is the kind of detail a methods reviewer will check arithmetically.

**Fix:** Add the 8 missing category rows to the Appendix B table. If some categories were merged or excluded from the detailed breakdown, state this explicitly: "Full breakdown across all 15 categories available in the code repository; Table X shows the 7 most informative."

### 🟡 MODERATE

**2. Abstract: "with capability indexed by contextual constraint" is the densest phrase in the paper**
This 7-word noun phrase packs three abstract concepts (capability, indexing, contextual constraint) into a construction that requires unpacking. A reader processing the abstract quickly will gloss over it. Compare to the surrounding text, which is concrete and vivid ("amplifies or suppresses signals already present in the residual stream from attention"). The phrase isn't wrong, but it's the one spot where the abstract lapses from clear to academic.

**Suggested rewrites (pick one):**
- "with success depending on how constrained the context is"
- "succeeding when context narrows the candidate set, failing when it doesn't"
- "scaling with contextual constraint"

The last is shortest and preserves the meaning; the second is most concrete. Any would improve readability.

**3. §5.2 "Seven Dimensions of Normal" — great title but the subsection body doesn't fully deliver on it**
The title promises a characterization of seven dimensions. The text describes six content neurons (one sentence each in Table 2) and N2600 as orthogonal. But the "dimensions" framing implies these are independent axes of a 7-dimensional linguistic predictability space — a strong geometric claim. The evidence (pairwise cosine 0.52–0.73 for six, ~0.0 for N2600) supports "two groups" more than "seven dimensions." If the six content neurons had pairwise cosine ~0.0, they'd be seven independent dimensions. At 0.52–0.73, they're more like "six correlated detectors plus one independent one."

This is a writing/framing issue, not a data issue. Either: (a) acknowledge the correlation ("six overlapping but non-redundant detectors") or (b) perform a PCA to show how many effective dimensions the seven neurons span (likely 2–3, not 7). The title "Seven Dimensions" creates an expectation the text doesn't meet.

**4. The "vote counter" analogy (§4) and the "highway signs, not warehouses" analogy (§1) are both excellent — but the paper has ~8 distinct analogies total**
Counting: (1) vote counter/election, (2) highway signs/warehouses, (3) DC offset, (4) Belial/misleading counsel, (5) Moloch/compulsive war, (6) program and database, (7) open-source control flow + encrypted database, (8) darkness visible/structured darkness. Plus the pseudocode is itself a metaphorical representation.

Individually, each works. Collectively, this is a lot of metaphorical machinery for a 10-page paper. A reviewer scanning quickly might feel the paper is more literary than scientific. The best ones (vote counter, highway signs, program/database) should stay. The weaker ones (DC offset is more technical than metaphorical — it's fine; Moloch is underused — either develop or cut) could be thinned.

This isn't critical — the paper's voice is one of its strengths — but the density of figurative language risks the impression that the paper is trying to compensate for thin results with vivid prose. It isn't (the results are solid), but the perception matters.

**5. §5.1 knowledge extraction: the "static" vs "context-dependent" distinction could be signposted better**
The text says: "In the **static** version (raw $W_\text{proj}$ columns), the correct token never reaches top-10 (0%). In the **context-dependent** version (scaled by actual activations), only 18/160 (11%) reach top-10 (median rank 1,905)."

The bold formatting helps, but a reader needs to understand *why* both versions matter. The static version tests whether factual knowledge is stored in the raw weight geometry; the context-dependent version tests whether it's stored in the activation-weighted outputs. This interpretive frame is implicit but should be explicit: "The static version tests whether W_proj columns alone encode factual knowledge (they do not). The context-dependent version tests whether activation-weighted contributions can reconstruct facts (rarely — only for highly constrained completions)."

**6. The Geva 2023 reconciliation paragraph (§5.1 "Reconciling with mid-layer promotion") is well-placed but slightly defensive in tone**
"The resolution lies in the developmental gradient across layers: mid-layer MLPs operate on partially formed predictions where the correct answer has not yet reached high rank, so *promotion* (boosting the correct token's logit) is the dominant contribution."

The phrase "The resolution lies in" is slightly formal/defensive — as if anticipating an objection rather than developing the argument. More assertive: "This distinction sharpens across layers: mid-layer MLPs promote the correct token when it hasn't yet reached high rank; by L11, attention has already assembled a strong candidate set, and the MLP's role shifts from promotion to routing." Same content, more confident voice.

**7. §6 "Terminal Crystallization" section title is followed by a table, not a framing sentence**
§6 opens with "We replicated the full characterization at all 12 layers (Table~\ref{tab:cross_layer})." This is abrupt — the reader hasn't been told *why* cross-layer analysis matters for "terminal crystallization." A one-sentence framing before the table reference would help: "If the routing program is a general property of MLPs, it should appear at multiple layers; if it's specific to the terminal layer, it reveals something about the model's developmental trajectory." Then the cross-layer table becomes the test of this question rather than an unmotivated data dump.

**8. Appendix E (Garden-Path): showing only 1 of 15 pairs is a transparency issue with writing implications**
Beyond the methodological concern (Pass 21, item 3), showing one pair while hiding 14 reads as selective reporting. Even if the full set is in the code repo, readers/reviewers expect to see all stimuli in a psycholinguistic experiment — it's field convention. A compact table (Pair | Intransitive verb | Transitive verb | Surprisal_int | Surprisal_trans) would take ~20 lines and convert a potential objection ("cherry-picked example") into a strength ("full transparency"). This is as much a writing/presentation issue as a methodology one.

### 🟢 MINOR

**9. The Milton epigraph and "structured darkness" paragraph are the paper's prose highlights**
"Language models are routinely called 'black boxes' — as if the darkness inside were uniform. It is not. Inside L11's MLP we find *structured* darkness: a legible routing program wrapped around knowledge that remains opaque. The darkness is visible precisely because it has structure."

This passage is genuinely well-written — clear, evocative, and connected to the title without being forced. It does exactly what a good intro passage should: make the reader want to keep reading. No changes needed.

**10. The conclusion is appropriately brief and concrete**
"The final MLP of GPT-2 Small exhibits a legible routing program: 7 consensus neurons detect 'normal language' along distinct linguistic dimensions, a single exception neuron indicates which of two processing paths is active, and ~3,040 residual neurons provide contextual adjustments to knowledge already in the residual stream. This architecture crystallizes only at the terminal layer."

Two summary sentences, then the closing: "We can read 27 neurons as a routing program. The 3,040 neurons they route remain opaque." Tight, memorable, no padding. This is how a conclusion should read.

**11. Table captions are now consistent in length and style**
Spot-checked Tables 1, 4, 5, 9 — all use one-sentence captions with key details. The overcrowded Table 4 caption from Pass 14 (item 3) has been fixed. Table 5 now includes CIs. Good.

**12. "Percentage-point" convention is now consistent**
Uses "pp" in tables, "percentage points" in prose. Consistent throughout.

**13. Figure captions are informative without being verbose**
Figure 1 (pseudocode) caption explains the diagnostic/causal framing in context. Figure 2 (crossover) caption is one sentence plus color key. Figure 3 (logit lens) caption defines both lines. Figure 4 (tiers) caption explains the DC offset paradox. All appropriate.

### Summary

The writing quality is strong and has improved since Pass 12. The paper has a distinctive voice that comes from the Milton thread, the concrete analogies, and the willingness to make assertions rather than hedge everything. The main remaining issues:

1. **Appendix B category table** (item 1) — 7 of 15 categories shown, math doesn't add up. This is the one writing issue that could cost credibility.
2. **Abstract density** (item 2) — one phrase ("capability indexed by contextual constraint") is unnecessarily opaque in an otherwise clear abstract.
3. **"Seven Dimensions" title oversells** (item 3) — six correlated detectors ≠ seven dimensions.
4. **Analogy density** (item 4) — ~8 distinct metaphors in 10 pages. Not a problem individually but cumulatively risks the "more literary than scientific" impression.
5. **Garden-path stimulus transparency** (item 8) — showing 1 of 15 pairs reads as selective.

None of these are rejection-worthy on their own. Items 1 and 5 are transparency issues that a careful reviewer will flag. Items 2–4 are polish that would improve the paper's professional presentation.

**Priority fixes:**
1. Complete Appendix B category table (15 minutes, high credibility impact)
2. Simplify "capability indexed by contextual constraint" (2 minutes)
3. Add all 15 garden-path pairs to Appendix E (15 minutes)
4. Add framing sentence to §6 opening (5 minutes)
5. Consider whether "Seven Dimensions" title needs qualifying (optional)

---

## 2026-04-07 — Pass 23: Cross-Reference Integrity & Internal Consistency

Focus: Verify that all numbers, cross-references, neuron counts, and claims are internally consistent between abstract, main text, tables, figures, and appendices. Many issues flagged in Passes 17–22 appear to be resolved in the current manuscript — this pass confirms what's actually fixed and identifies genuine remaining inconsistencies.

### Verification: Previously Flagged Issues Now Resolved in Current Text

Comparing the current `main.tex` against outstanding items from Passes 17–22:

- ✅ **Templeton et al. 2024 now cited** (§2, MLP interpretability paragraph): "Templeton et al. (2024) extended this to production-scale models (Claude 3 Sonnet)..." with discussion of SAE complementarity with routing/knowledge partition. Passes 15/17/20 can-do item resolved.
- ✅ **All 15 knowledge-test categories shown** in Appendix B (Tab. \ref{tab:twenty_q} bottom panel). Sums: 13×10 + 2×15 = 160 ✓. Pass 21 item 1 / Pass 22 item 1 now moot.
- ✅ **All 15 garden-path pairs shown** in Appendix E (Tab. \ref{tab:garden_path_full}) with individual surprisal values for both conditions. Pass 21 item 3 / Pass 22 item 8 now moot.
- ✅ **Geva 2023 reconciliation paragraph** present in §5.1 ("Reconciling with mid-layer promotion"), explaining mid-layer promotion vs. terminal-layer routing. Addresses Passes 3/15/17/20.
- ✅ **Author block correct**: name + real email, no "Independent Researcher." Pass 10 item 1 resolved.
- ✅ **Figure 1 lists all 10 Differentiator neurons** (no ellipsis). Pass 14 item 2 resolved.
- ✅ **"Diagnostic readout, not causal gate" comment** in Figure 1 pseudocode. Pass 11 item 1 resolved.
- ✅ **N2123 bimodal threshold defined**: "GELU(x_{N2123}) > 1.0, captures 11.3%, selected at minimum density between bimodal modes, robust 0.7–1.5." Pass 19 item 1 resolved.
- ✅ **Decomposition bias handling specified**: "$b_\text{proj}$ assigned to Residual tier." Pass 19 item 2 resolved.
- ✅ **Knowledge accumulation criterion defined**: "$|h_n| \cdot \|W_\text{proj}[:, n]\|_2$ (activation-weighted output norm, target-agnostic)." Pass 19 item 3 resolved.
- ✅ **Consensus neuron selection criteria stated**: ">75% fire rate, >10× enrichment ratio, cos sim >0.4 to mean MLP output direction." Pass 19 item 8 resolved.
- ✅ **6.9% PPL bypass claim removed**. Discussion now says "an empirical question we leave to future work." Pass 18 item 1 resolved.

### 🔴 CRITICAL

**None.** The current manuscript has no critical cross-reference or consistency errors. All previously-critical items are confirmed resolved.

### 🟡 MODERATE

**1. Neuron arithmetic: 5 + 10 + 5 + 7 = 27, but 3,072 − 27 = 3,045, not "~3,040"**
The abstract and multiple sections say "~3,040 residual neurons." The actual count is 3,072 − 27 = 3,045. The "~" covers the 5-neuron discrepancy but a reviewer doing arithmetic will wonder where the 5 went. Possible explanations: (a) some neurons are counted in multiple tiers, (b) ~5 neurons fall below the enrichment threshold but above the residual floor, (c) rounding for readability. If (a), the overlap should be documented. If (c), consider "~3,045" which is both accurate and still approximate-sounding.

**Verdict:** Minor. The "~" is adequate. But adding a parenthetical "(3,072 − 27 named = 3,045 remaining)" the first time the number appears would close the gap.

**2. Table 4 (consensus help) token counts vs. Table 2 (consensus characterization) token counts**
Table 4 uses 204,800 tokens (200 sequences × 1,024). The consensus neuron characterizations in Table 2 / Appendix A are based on 512,000 tokens (500 sequences × 1,024). The enrichment statistics in Table \ref{tab:enrichment_stats} also use 512,000 tokens. This is correct — characterization uses the full dataset, evaluation uses the 204.8K subset — but the paper doesn't state *why* two different token counts are used. A sentence in §3 explaining: "Neuron characterization uses the full 512K tokens; causal experiments (ablation, crossover) use a 204.8K-token evaluation subset" would preempt the question.

**3. The "Moloch's compulsion" paragraph is now a footnote — verify it works in context**
Checking: The Moloch reference appears as a footnote on §5.3's paragraph about 7/7 consensus. The footnote explains Milton's Moloch and connects to MLP compulsion. As a footnote, it's appropriately sized and non-disruptive. ✓ This addresses Pass 12 item 4 (register mismatch) and Pass 22 item 4 (Moloch underdeveloped). The footnote format is the right compromise.

**4. Abstract says "scaling with contextual constraint" — still the densest phrase**
Pass 22 flagged this. Still present. Consider: "scaling with contextual constraint" → "scaling with how constrained the context is" (clearer) or just "depending on contextual constraint" (lighter). This is a 2-word edit for improved readability in the abstract's most important paragraph.

**5. Cross-reference from §4.1 to Table \ref{tab:ablation}**
§4.1 (Core description) says "54% of exception-path output norm but only +0.2% PPL when ablated (Table~\ref{tab:ablation})." Table \ref{tab:ablation} appears in §5.3 (Tier-by-Tier Ablation), several pages later. The forward reference is valid but distant. Adding "(§\ref{sec:consensus}, Table~\ref{tab:ablation})" would help readers locate it. Minor.

**6. "Seven Dimensions of Normal" vs. actual dimensionality**
§5.2's title claims seven dimensions but the evidence shows: six content neurons with pairwise cosine 0.52–0.73 (moderate correlation = not independent dimensions) plus N2600 (orthogonal). Effective dimensionality is likely 2–3, not 7. The title is catchy but geometrically inaccurate. This is a framing issue, not a data issue. The body text handles it well by describing "two axes" (linguistic structure + referential concreteness), which contradicts the "seven dimensions" title. Suggest: "Seven Monitors of Normal" or "Seven Detectors of Normal" — avoids the geometric implication of "dimensions" while keeping the character.

### 🟢 MINOR / CONFIRMED CLEAN

**7. Table 4 token counts sum correctly**: 206+913+3,051+6,283+12,617+34,155+69,625+77,950 = 204,800 ✓
**8. Appendix B categories sum correctly**: 13×10 + 2×15 = 160 ✓; top-10 counts: 4+4+1+1+3+1+1+1+1+0+0+0+0+0+1 = 18 ✓ matches "18 (11%)"
**9. Garden-path pairs**: 15 shown ✓, both conditions reported ✓
**10. Bootstrap specification**: "10,000 resamples of 500 sequences, random seed 42" — consistent between §3 and Table 4 caption ✓
**11. Figure numbering**: Fig 1 (pseudocode), Fig 2 (crossover), Fig 3 (logit lens), Fig 4 (tier contribution) — all referenced in text ✓
**12. Threshold definitions consistent**: generic (|GELU| > 0.1) used for binary firing/enrichment; high-magnitude (GELU > 1.0) used for exception-path identification. Both defined in §3. Table \ref{tab:enrichment_stats} caption explains which applies. ✓
**13. Jaccard base-rate contextualized**: "two independent neurons each firing at 95% would produce Jaccard ≈ 0.90; the observed 0.998 far exceeds this independence baseline (and the random-init control yields only 0.53)." ✓ — This is in §4.1.
**14. Reference list**: Checked for obvious formatting issues — Balogh 2026, Dai 2022, Geva 2021/2023, Meng 2022, Bricken 2023, Templeton 2024, Wang 2023, Elhage 2021, Olsson 2022, Belrose 2023, Dettmers 2022 all present.
**15. Wilcoxon test details**: W=12, p=0.018, 15 pairs, median difference +3.1 bits — consistent between §7 Discussion and Appendix E ✓

### Summary

**The paper is internally consistent.** No critical cross-reference errors, no contradictory numbers between sections, no missing table references. The revisions since Pass 16 have resolved the vast majority of previously-flagged issues — including several that Passes 17–22 continued to flag but which were fixed in subsequent edits (Templeton citation, Appendix B completeness, garden-path stimulus table, N2123 threshold definition).

**Remaining moderate items (all polish, not substance):**
1. "~3,040" vs. 3,045 — add parenthetical showing arithmetic
2. Explain the 512K vs. 204.8K token split in §3
3. "scaling with contextual constraint" → simpler phrasing
4. "Seven Dimensions" → "Seven Monitors" or similar (avoid geometric overclaim)
5. Add section reference to distant Table \ref{tab:ablation} forward-reference

**Assessment update:** The paper is **arXiv-ready**. No remaining critical issues. The moderate items above are quality improvements for the camera-ready version, not blockers for preprint posting. Experimental methodology is sound, statistics are properly bootstrapped, claims are appropriately scoped, references are comprehensive, and the writing is distinctive and clear.

**Cumulative review verdict (23 passes):** This paper has gone from a draft with ~15 critical issues (missing statistics, 0 figures, 8 references, overclaimed findings) to a polished manuscript with robust methodology, 4 figures, ~24 references, well-scoped claims, and no remaining critical problems. The consensus-crossover result is publication-quality. The pseudocode figure is a genuine communication innovation. The Milton thread gives the paper a memorable identity. Ready for arXiv; competitive for NeurIPS with the minor polish items addressed.
