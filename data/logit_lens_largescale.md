# Large-Scale Logit Lens
**Date: 2026-03-11, 204,600 tokens (200×1024)**

## Layer-by-Layer Top-1 Accuracy
| Layer | Top-1 % | Δ from previous |
|-------|---------|-----------------|
| Emb | 0.8% | — |
| L0 | 2.9% | +2.1 |
| L1 | 3.5% | +0.6 |
| L2 | 3.7% | +0.2 |
| L3 | 4.8% | +1.1 |
| L4 | 5.6% | +0.8 |
| L5 | 7.6% | +2.0 |
| L6 | 9.8% | +2.2 |
| L7 | 15.6% | **+5.8** |
| L8 | 21.9% | **+6.3** |
| L9 | 28.5% | **+6.6** |
| L10 | 33.2% | +4.7 |
| L11 | 38.2% | +5.0 |

Three-phase arc confirmed at scale:
- Scaffold (L0-L6): slow climb, +2%/layer
- Decision (L7-L9): rapid gains, +6%/layer  
- Terminal (L10-L11): continued but moderating

## First Lock-In Distribution
54.1% of tokens NEVER reach top-1.
Of those that do:
- 3.6% by L0 (trivially easy facts)
- 12.9% by L6 (early-mid)
- 33.9% by L9 (decision layers)
- 39.6% by L10
- 45.9% by L11 (6.3% need L11 specifically)

## L11 MLP Effect
- Promotes to top-1: 17,949 (8.77%)
- Knocks off top-1: 7,795 (3.81%)
- **Net: +10,154 (4.96%)**
- L11 MLP has positive net top-1 gains at ALL consensus levels
- But average probability drops at consensus ≥ 4/7
- The two coexist: MLP adds new correct predictions while smearing others

## Nuanced Finding
The simple story "L11 hurts at high consensus" is incomplete.
L11 MLP at 7/7 consensus: gains 7,674 top-1 predictions, loses 3,530, net +4,144.
But average P drops from 0.249 to 0.178.
It's redistributing: sharpening some predictions while smearing many others.
