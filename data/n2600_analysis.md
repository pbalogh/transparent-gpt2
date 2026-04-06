# N2600: The Document Structure Detector
**Date: 2026-03-11**

## Key Finding
N2600 is orthogonal to the other 6 consensus neurons (cos ≈ 0.0) because it monitors a fundamentally different dimension: **document structure** vs. **linguistic content**.

## Evidence

### When N2600 disagrees with the other 6:
- **Case A (others fire, N2600 silent, 32,943 tokens)**: `the` (10.1%), `a` (5.4%), determiners, possessives — content words that ARE linguistically normal but don't carry structural information
- **Case B (N2600 fires, others silent, 5,060 tokens)**: `\n\n` (16.7%), `=` headers (5.5%), formatting markers — structural tokens the other 6 can't classify linguistically

### Interpretation
- Other 6 neurons: "What kind of language is this?" (syntax, tense, entities, semantics)
- N2600: "Where are we in the document?" (paragraph breaks, headers, sections)

The 7-neuron consensus detector has TWO orthogonal components:
1. **Content consensus** (6 neurons, correlated at cos 0.52-0.73): linguistic normalcy
2. **Structure consensus** (N2600, orthogonal): document-level positioning

Full consensus (7/7) requires BOTH content normalcy AND structural predictability.
