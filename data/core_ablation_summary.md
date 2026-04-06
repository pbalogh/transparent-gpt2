# Core Ablation Results
**Date: 2026-03-11, 204,600 tokens (200×1024)**

## Tier-by-Tier PPL Impact
| Tier | Neurons | PPL | Change |
|------|---------|-----|--------|
| Baseline | — | 31.79 | — |
| Core (5) | N2123/N2910/N740/N1611/N2044 | 31.86 | +0.2% |
| Differentiators (10) | N2462/N2173/N1602/etc. | 32.20 | +1.3% |
| Specialists (5) | N2921/N2709/N971/N2679/N737 | 31.69 | -0.3% |
| All exception (20) | All above | 32.44 | +2.1% |

## Key Finding: The Core's "vocabulary reset" is almost free
Zeroing the Core costs only +0.2% PPL. The Differentiators (suppression +
subword repair) are 6× more important. Specialists actually *improve* PPL
when removed — they may be slightly counterproductive.

## By Consensus Level (Core ablation)
| Level | Base PPL | Ablated PPL | Change |
|-------|----------|-------------|--------|
| 0/7 | 11.30 | 11.34 | +0.3% |
| 1/7 | 7.50 | 7.65 | +1.9% |
| 2/7 | 8.95 | 9.10 | +1.6% |
| 3/7 | 10.24 | 10.49 | +2.4% |
| 4/7 | 15.90 | 16.16 | +1.6% |
| 5/7 | 25.07 | 25.34 | +1.1% |
| 6/7 | 35.28 | 35.30 | +0.0% |
| 7/7 | 40.48 | 40.32 | -0.4% |

Core ablation hurts most at 1-3/7 consensus (exception territory) and
slightly HELPS at 7/7 (the vocabulary reset was unnecessary there).
