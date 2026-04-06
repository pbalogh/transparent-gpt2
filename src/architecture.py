"""
Architecture map for GPT-2 Small, derived from 500K-token analysis.

This is the "discovered blueprint" — the routing structure that gradient
descent built but never documented.
"""

# ============================================================
# LAYER PHASES
# ============================================================
# Three developmental phases, like embryonic differentiation:
#   Scaffold (L0-3): Simple formatting, gateway neurons
#   Diffuse  (L4-6): Distributed processing, no routing structure
#   Decision (L7-11): Binary consensus/exception architecture

PHASES = {
    0: 'scaffold', 1: 'scaffold', 2: 'scaffold', 3: 'scaffold',
    4: 'diffuse',  5: 'diffuse',  6: 'diffuse',
    7: 'decision', 8: 'decision', 9: 'decision', 10: 'decision', 11: 'decision',
}

# ============================================================
# MLP ROUTING MAP
# ============================================================

# Exception handler neurons: fire at consensus breakdown,
# high-intensity output, trigger full nonlinear processing.
# The "pressure relief valve."
EXCEPTION_NEURONS = {
    0: 2053,   # 9% fire rate, primitive
    7: 1990,   # 8% fire rate, first decision layer
    8: 589,    # 27% fire rate (highest — L8 is messy)
    9: 1999,   # 15% fire rate
    10: 1858,  # 20% fire rate
    11: 2123,  # 10% fire rate, best characterized
}

# Consensus neurons: "default-ON" valves. When the quorum holds,
# the token takes the linear path. When it breaks, exception fires.
# Quorum grows with depth: 1 → 3 → 7.
CONSENSUS_NEURONS = {
    0:  [800],                                          # 1 neuron  (primitive)
    7:  [2489],                                         # 1 neuron  (decision begins)
    8:  [1640, 2579, 1374],                             # 3 neurons
    9:  [1305, 1889],                                   # 2 neurons
    10: [1486, 1109, 928],                              # 3 neurons
    11: [2, 2361, 2460, 2928, 1831, 1245, 2600],        # 7 neurons (full quorum)
}

# Gateway neurons: simpler routing primitive in scaffold layers.
# Single neuron that fires MORE for barely-nonlinear tokens.
# "One manual shutoff valve" vs the consensus parliament.
GATEWAY_NEURONS = {
    1: 2882,   # 19/20 top enriched patterns contain this neuron
    2: 2380,   # 20/20
    3: 746,    # 17/20
}

# Layers where MLP can be safely zeroed at consensus (< 7% PPL cost)
BYPASSABLE_LAYERS = {7, 8, 9, 10, 11}

# ============================================================
# L11 EXCEPTION HANDLER: TIERED NEURON ARCHITECTURE
# From 200K-token characterization of exception-path activations.
# ============================================================

# TIER 1 — "The Core" (fire 90-100% of exception tokens)
# These five neurons are effectively one fused unit (Jaccard 0.91-0.998).
# They all push toward common function words: the, in, and, a, of.
# Function: "vocabulary reset" — when consensus breaks, push the
# residual stream back toward the most probable tokens.
EXCEPTION_CORE = {
    2123: 'exception handler (the switch itself)',
    2910: 'vocabulary reset: the, in, and, a (Jaccard 0.998 with N2123)',
    740:  'punctuation/novelty signal (Jaccard 0.985 with N2123)',
    1611: 'vocabulary redistribution: the, and, a, in',
    2044: 'vocabulary redistribution: the, a, and, in, of',
}

# TIER 2 — "The Differentiators" (fire 35-88%)
# Fire selectively based on token type. Two sub-circuits:
#   A) Suppression pair: N584+N2378 (Jaccard 0.889) push AWAY from
#      common words — they anti-boost, preventing wrong predictions
#   B) Subword handlers: N1602 (word fragments), N1715 (multi-token
#      word continuation like "acebook", "archment", "incinn")
EXCEPTION_DIFFERENTIATORS = {
    2462: 'lower-threshold vocab reset (88%)',
    2173: 'vocab reset (75%)',
    1602: 'subword/rare token handler: j, g, z, Sap (54%)',
    1800: 'vocab reset (46%)',
    2379: 'morphological processing (44%)',
    1715: 'multi-token word continuation (42%)',
    611:  'suppression: anti-boost common words (40%)',
    3066: 'punctuation correction (38%)',
    584:  'suppression pair with N2378 (37%)',
    2378: 'suppression pair with N584 (36%)',
}

# TIER 3 — "The Specialists" (fire 14-37%)
# Structurally independent from Tiers 1-2 (low Jaccard).
# N737 is nearly solo: Jaccard < 0.15 with everything.
EXCEPTION_SPECIALISTS = {
    2921: 'section boundary handler (37%)',
    2709: 'section boundary + vocab (36%)',
    971:  'section boundary (34%)',
    2679: 'formatting/special characters: @, –, . (21%)',
    737:  'paragraph-only specialist — fires almost exclusively on \\n\\n (14%)',
}

# L0 is NOT bypassable despite having consensus structure.
# Its "consensus" serves a different function (positional formatting)
# that all downstream layers depend on. Zeroing it = catastrophe.
SACRED_LAYERS = {0}

# ============================================================
# ATTENTION MAP (from 144-head catalog)
# ============================================================

# Heads that dump attention onto BOS token (position 0).
# 93 of 144 heads. Effectively no-ops for most tokens.
# (Full list TBD — need per-head BOS-sink classification)
N_BOS_SINK_HEADS = 93

# The one head that matters most: L11H7
# 6x more important than any other head (importance score 91.5)
DOMINANT_HEAD = (11, 7)

# ============================================================
# ROUTING PARAMETERS
# ============================================================

# Fraction of consensus neurons that must fire for "consensus holds"
CONSENSUS_THRESHOLD = 0.85

# GELU activation above this = neuron "fires"
FIRE_THRESHOLD = 0.1
