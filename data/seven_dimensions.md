# The Seven Consensus Dimensions: Deep Characterization
**Date: 2026-03-11, 512,000 tokens (500×1024)**

## Summary

The 7 consensus neurons form two orthogonal groups:
- **Content group (6 neurons, cos 0.52–0.73)**: Monitor linguistic properties
- **Structure group (1 neuron, cos ≈ 0)**: Monitors document/discourse structure

## Pairwise Cosine Similarity (Output Directions)
```
         N2    N2361  N2460  N2928  N1831  N1245  N2600
N2      1.00   0.52   0.61   0.62   0.53   0.64   0.13
N2361          1.00   0.65   0.64   0.61   0.64   0.01
N2460                 1.00   0.63   0.65   0.73   0.03
N2928                        1.00   0.52   0.61   0.10
N1831                               1.00   0.64  -0.09
N1245                                      1.00   0.00
N2600                                             1.00
```

## Individual Neuron Profiles

### N2 — Clausal Continuation
**Fire rate:** 88.4% (highest) | **Mean |act|:** 0.64

**What activates it most:** Conjunctions and clause connectors within complex sentences. Top activations: `@` (number formatting), `and`, `but`, `lost`, `created`, `opened`, `also`, `as`. All occur mid-sentence continuing an ongoing clause.

**What it rejects:** Subword fragments (`urt`, `kel`, `arn`), sentence-initial capitalized words (`According`, `Best`, `Cre`). Fires least on tokens that START new ideas.

**Disagrees-silent (47,989 tokens):** When N2 stays silent but others fire, top tokens are `the`, `of`, `@`, `and` — function words in sentence-INITIAL position or after structural breaks.

**Dimension:** Monitors whether we're INSIDE a continuing clause vs. at a structural boundary.

---

### N2361 — Syntactic Elaboration
**Fire rate:** 84.1% | **Mean |act|:** 0.56

**What activates it most:** Subordinate/relative clause markers and narrative elaboration. Top activations: `Villaret` (named entity in complex sentences), `perform`, `that`, `partly`, `fully`, `fifth`, `entire`, `neither`, `while`, `but`. Context shows elaborate syntactic constructions.

**What it rejects:** Named entities and nouns at new-topic positions: `Cardinal`, `Christian`, `United`, `There`. Also subword fragments.

**Most depleted:** `gh` (30%), `od` (34.5%), `Ant` (35.4%), `O` (37.3%) — morphological fragments that interrupt syntactic flow.

**Dimension:** Monitors syntactic complexity. Fires on tokens embedded in elaborate syntactic structures (relative clauses, subordination, multi-clause sentences). Silent on simple declarative starts.

---

### N2460 — Prepositional/Relational Context  
**Fire rate:** 86.0% | **Mean |act|:** 0.57

**What activates it most:** Prepositions, articles in prepositional phrases, and relational words. Top activations: `an`, `the`, `into`, `other`, `per` — all within prepositional or relational phrases. Also abstract nouns: `anger`, `consent`, `inability`, `deities`.

**What it rejects:** Strongly depletes `According` (3.3%!), `Fin` (7.7%), `Tru` (13.1%), `Sap` (17.6%) — sentence-initial discourse markers and specific entity fragments. Also `\n\n` (21.5%) — paragraph breaks.

**Dimension:** Detects relational/prepositional structure. Fires when the current position is inside a grammatical relationship (X of Y, X in Y, X with Y). Silent at discourse boundaries and topic-introducing positions.

---

### N2928 — Sequential/List Structure
**Fire rate:** 91.4% (second-highest) | **Mean |act|:** 0.79 (highest)

**What activates it most:** Ordinals, list elements, sequences, and parallel structures. Top activations: `too` (in parallel), `Best` (in award lists), `th` (ordinals), `%` (statistics), `,` (in lists), `each`, `heavily`, `that`, `another`, `Film`, `Form`. All appear in enumerated or structured sequences.

**What it rejects:** `According` (1.6%!!), `vest` (5.2%), `Fin` (9.0%), `Tru` (10.7%) — same discourse markers/fragments as N2460 but even more extreme.

**Disagrees-fire (6,957 tokens):** `\n\n` is 24% — fires on paragraph breaks when others don't, detecting structural formatting that involves sequence.

**Dimension:** Monitors parallel structure, enumeration, and sequential patterning. Highest fire rate and strongest activation — this is the "backbone" consensus neuron that fires whenever text has regular, predictable structure.

---

### N1831 — Discourse Coherence / Topic Continuity
**Fire rate:** 81.0% | **Mean |act|:** 0.54

**What activates it most:** Topic-continuing phrases and discourse-level references. Top activations: `All` (in "All Star"), `all` (in "all-time"), `South` (in album name), `synthesis`, `public`, `Republic`, `scope`, `play`, `co` — all continue or elaborate on an established topic.

**What it rejects:** `According` (2.4%), `how` (4.0%), `vest` (5.8%), `Tru` (14.3%) — topic-shifting and fragmentary tokens. Also `\n\n` (17.4%).

**Disagrees-silent (77,823 tokens — largest!):** `the` (7.5%!), `.` (4.5%), `a` (2.3%), `was` (1.5%). N1831 has by far the most disagreement with the group. It stays silent on generic function words that could appear in ANY topic — it only fires when there's clear topic continuity.

**Dimension:** Monitors discourse coherence. Fires when the current token continues or elaborates an established topic. Silent on generic function words that don't carry topic information. The most "discourse-level" of the content neurons.

---

### N1245 — Semantic Role / Argument Structure
**Fire rate:** 85.5% | **Mean |act|:** 0.56

**What activates it most:** Semantic role markers — tokens that establish relationships between entities. Top activations: `leadership`, `Player`, `patronage`, `validity`, `one` (in "one of the"), `invitation`, `Chairman`, `nature`, `character`, `Deacon`, `supervision`, `responsibility`. All mark semantic roles: X is the Y of Z.

**What it rejects:** `According` (2.4%), `Tru` (6.0%), `vest` (6.3%), `Fin` (17.9%) — same discourse markers. Also entity-initial fragments: `Gal`, `Lock`, `Che`.

**Disagrees-fire (4,720 tokens):** `the` (1.9%), `of` (1.7%), `a` (1.2%), `part` (0.9%), `number` (0.5%) — specifically fires on function words that establish semantic relationships (part of, number of).

**Dimension:** Monitors semantic argument structure. Fires when the current position establishes or continues a semantic role relationship (X of Y, under the Z of W). The most "semantic" neuron.

---

### N2600 — Quantitative/Referential Anchoring ★ORTHOGONAL★
**Fire rate:** 79.2% (lowest) | **Mean |act|:** 0.86 (second-highest! very strong when active)

**What activates it most:** Dollar signs, currency amounts, timestamps, dates. Top 20 activations are ALL `$` or `:` in monetary/temporal contexts. This neuron fires MOST STRONGLY on quantitative reference points.

**Most enriched tokens:** `$` (100%), months (`March`, `July`, `August`, `April`, `October`, `November`), proper nouns (`America`, `David`, `Paul`), derived forms (`ations`, `bs`, `son`).

**What it rejects:** ABSTRACT ADJECTIVES. Most depleted: `natural` (1.8%!), `social` (3.1%), `religious` (4.3%), `main` (4.8%), `human` (6.8%), `various` (10.1%), `whose` (7.7%), `different` (15.7%), `my` (15.6%), `their` (17.9%), `small` (18.3%), `major` (18.6%).

**Disagrees-silent (95,739 tokens — MASSIVE):** `the` (9.0%!), `a` (5.1%), `and` (2.3%), `s` (1.7%), `his` (1.2%), `was` (1.3%). Silent on generic determiners and possessives.

**Disagrees-fire (12,782 tokens):** `\n\n` (16.9%), `=` (5.7%), `According` (0.8%).

**Dimension:** Monitors CONCRETE REFERENCE — specific dates, amounts, names, places — versus ABSTRACT DESCRIPTION. This is why it's orthogonal: the content neurons monitor linguistic structure (syntax, semantics, discourse), while N2600 monitors whether the content is concrete/referential vs. abstract/descriptive. This is a fundamentally different axis.

---

## Universal Consensus Breakers

Tokens that ALL 7 neurons tend to reject (low fire rate across all):
- `According` — discourse attribution marker (new info source)
- `Tru`, `Fin`, `Sap`, `kel`, `vest` — specific named entity fragments (proper noun subwords)
- `urt`, `arn` — subword fragments mid-word
- `how` — interrogative/manner word
- `\n\n` — paragraph breaks (though N2600 and N2928 still fire on these)

These are tokens at MAXIMUM UNPREDICTABILITY: discourse shifts, rare entities, structural breaks.

## Revised Labels

| Neuron | Old Label | New Label | Fire Rate |
|--------|-----------|-----------|-----------|
| N2 | Sentence boundaries | **Clausal continuation** | 88.4% |
| N2361 | Syntactic relations | **Syntactic elaboration** | 84.1% |
| N2460 | Temporal/aspectual | **Prepositional/relational context** | 86.0% |
| N2928 | Structure/formatting | **Sequential/list structure** | 91.4% |
| N1831 | Entity/discourse | **Discourse coherence** | 81.0% |
| N1245 | Semantic content | **Semantic role/argument structure** | 85.5% |
| N2600 | Sentence structure | **Concrete reference vs. abstract description** | 79.2% |

## The Two Axes

**Content axis (6 neurons):** Is the language structurally predictable?
- Clause continuation (N2) + syntactic complexity (N2361) + relational embedding (N2460) = syntactic normalcy
- Sequential pattern (N2928) + topic continuity (N1831) + argument structure (N1245) = discourse normalcy

**Referential axis (N2600):** Is the content concrete and referential?
- Concrete (dates, amounts, names) → fires → contributes to consensus
- Abstract (qualities, descriptions) → silent → withholds consensus

Full consensus (7/7) requires BOTH structural predictability AND concrete referentiality.
