# L11 Consensus Neuron Characterization
**Date: 2026-03-11, 512K tokens (500×1024)**

## The 7 Consensus Neurons: A Distributed "Normal Language" Detector

Each consensus neuron checks a different linguistic dimension. When all 7 agree → predictable context → skip nonlinear processing.

### Individual Neuron Specializations

| Neuron | Fire Rate | Detects (enriched) | Misses (depleted) |
|--------|-----------|--------------------|--------------------|
| N2 (88.1%) | Sentence boundaries | `.`, `In`, `\n`, `She` | `\n\n`, `-`, `@` |
| N2361 (85.6%) | Syntactic relations | `were`, `which`, `be`, `has`, `who`, `but` | `-`, `\n\n`, `@` |
| N2460 (87.6%) | Temporal/aspectual | `been`, `during`, `when`, `where`, `ed` | `\n\n`, `j`, `Tru` |
| N2928 (92.0%) | Structural/formatting | `@`, `\n`, `were`, `;`, `In` | `j`, `Tru`, `\n\n` |
| N1831 (81.1%) | Entity/discourse | `Fey`, `Shiva`, `This`, `which`, `gods` | `j`, `Tru`, `\n\n` |
| N1245 (86.0%) | Semantic content | `gods`, `deities`, `divine`, `are`, `is` | `j`, `Tru`, `\n\n` |
| N2600 (77.7%) | Sentence structure | `.`, `\n`, `I`, `He`, `(` | `their`, `its`, `a` |

### Consensus Level = Linguistic Predictability

| Level | Tokens | % | Dominated by |
|-------|--------|---|-------------|
| 0/7 | 486 | 0.1% | `\n\n` (27%), rare subwords |
| 1/7 | 2,284 | 0.4% | `\n\n` (22%), rare tokens |
| 2/7 | 6,624 | 1.3% | `\n\n` (17%), subwords, markup |
| 3/7 | 14,357 | 2.8% | `=` headers, `\n\n`, rare |
| 4/7 | 31,124 | 6.1% | Transition: common words appearing |
| 5/7 | 82,465 | 16.1% | `the`, `of`, `a`, `and` |
| 6/7 | 168,463 | 32.9% | Punctuation + function words |
| 7/7 | 206,197 | 40.3% | `,`, `.`, `the`, `of`, `"` |

### Key Insight

0/7 consensus = "this token is alien" (paragraph breaks, rare subwords, non-linguistic tokens)
7/7 consensus = "this token is bread-and-butter English" (punctuation, function words, common structure)

The gradient is continuous: as more neurons "recognize" the token, the linguistic context becomes more predictable, and the MLP's nonlinear processing becomes less necessary (and eventually counterproductive).

### Universal Depletions (what breaks consensus)
- `\n\n` (paragraph breaks) — depleted by ALL 7 neurons
- `j` (subword fragment) — depleted by N2460, N2928, N1831, N1245
- `Tru` (truncated proper noun) — depleted by N2460, N2928, N1831, N1245
- `-` (hyphen) — depleted by N2, N2361, N2928
- `@` — depleted by N2, N2361

These are exactly the tokens where the model needs exception handling: boundaries, fragments, and non-linguistic markup.
