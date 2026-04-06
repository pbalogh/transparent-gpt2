# Cross-Layer Exception Handler Characterization
**Date: 2026-03-11, 500K tokens, 1024-seq**

## Summary

Only L11 has a true three-tier exception handler. L7 and L10 have binary splits but no enrichment, no specialization, no tight co-firing clusters.

## L7 (Exception N1990, Consensus [N2489])
- **Exception fire rate: 44.5%** — nearly half of all tokens
- **Enrichment: ~1.0× for all top neurons** — no selectivity
- **Tiers: 5 "core" (>85%), 15 "diff" (78-84%), 0 specialists**
- **Token overlap**: "the", "and", "of" appear in both exception and consensus top tokens
- **Interpretation**: Broad binary split, not true exception handling

## L10 (Exception N1858, Consensus [N1486, N1109, N928])
- **Exception fire rate: 50.3%** — exactly half
- **Enrichment: ~1.0× for all top neurons** — zero selectivity
- **Tiers: 7 "core" (>85%), 13 "diff" (81-85%), 0 specialists**
- **Jaccard: ~0.7 everywhere** — uniform, not clustered (vs L11's 0.998 fused core)
- **Token overlap**: Same tokens in both categories
- **Interpretation**: Near-random binary split

## L11 (Exception N2123, Consensus [N2, N2361, N2460, N2928, N1831, N1245, N2600])
- **Exception fire rate: 11.3%** — highly selective minority
- **Enrichment: up to 2.0× for core neurons**
- **Tiers: 5 core (90-100%, fused at Jaccard 0.998), 10 diff (35-88%, includes suppression pair), 5 specialists (14-37%, N737 solo at <0.15)**
- **Token analysis**: Exception tokens are syntactically distinct (rare words, subwords, boundaries)
- **Interpretation**: True structured exception handler with legible routing

## Key Finding

Routing **crystallizes through depth** but only the terminal layer (L11) achieves the tight, legible, tiered architecture readable as pseudocode. Earlier layers have proto-routing (binary splits) but not structured exception handling.

This supports the "progressive crystallization" narrative: the three-phase arc (scaffold → diffuse → decision) is not just about consensus neuron count — it's about the QUALITY of the routing structure. L11 is qualitatively different, not just quantitatively more.
