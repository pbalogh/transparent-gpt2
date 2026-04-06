"""
Transparent GPT-2 Small: Same weights, visible routing, readable control flow.

The forward pass is semantically identical to standard GPT-2 but written
as human-readable pseudocode with real matrix operations underneath.

Three modes:
  'standard'    — vanilla forward pass (baseline verification)
  'transparent' — identical math, routing decisions logged
  'bypass'      — exploit routing: zero MLP at consensus for decision layers
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from .architecture import (
    PHASES, EXCEPTION_NEURONS, CONSENSUS_NEURONS, GATEWAY_NEURONS,
    BYPASSABLE_LAYERS, CONSENSUS_THRESHOLD, FIRE_THRESHOLD,
    EXCEPTION_CORE, EXCEPTION_DIFFERENTIATORS, EXCEPTION_SPECIALISTS,
)


# ============================================================
# ROUTING PRIMITIVES
# These are the "valves" — binary decisions about token routing.
# ============================================================

def neuron_fires(activations, neuron_idx):
    """Does this neuron's GELU activation exceed the firing threshold?
    
    This is the fundamental binary decision. Despite living in a
    continuous network, the routing choice is yes/no.
    Binary vs continuous classification accuracy: 79.2% vs 78.8%.
    Binarization loses essentially nothing.
    """
    return activations[:, :, neuron_idx] > FIRE_THRESHOLD


def consensus_holds(activations, layer):
    """Do enough "default-ON" neurons agree that this token is normal?
    
    When the quorum holds: token takes the cheap path (or no path).
    When the quorum breaks: exception handler fires, full nonlinear circuit.
    
    The quorum grows with depth:
      L0: 1 neuron must agree   (primitive)
      L7: 1 neuron              (decision begins)
      L8: 3 neurons             (committee)
      L11: 7 neurons            (full parliament)
    """
    if layer not in CONSENSUS_NEURONS:
        return torch.zeros(
            activations.shape[0], activations.shape[1],
            dtype=torch.bool, device=activations.device
        )
    neurons = CONSENSUS_NEURONS[layer]
    votes = torch.stack([neuron_fires(activations, n) for n in neurons], dim=-1)
    fraction_agreeing = votes.float().mean(dim=-1)
    return fraction_agreeing >= CONSENSUS_THRESHOLD


def gateway_fires(activations, layer):
    """Does the single gateway neuron say "this token needs work"?
    
    Scaffold layers (L1-3) use a simpler routing primitive than consensus:
    one neuron acts as a manual shutoff valve. If it fires, the token
    gets routed through full nonlinear processing. If not, linear path.
    """
    if layer not in GATEWAY_NEURONS:
        return torch.zeros(
            activations.shape[0], activations.shape[1],
            dtype=torch.bool, device=activations.device
        )
    return neuron_fires(activations, GATEWAY_NEURONS[layer])


def exception_fires(activations, layer):
    """Is the exception handler active? (For logging/transparency.)
    
    The exception handler fires at 3x normal intensity when the
    consensus breaks down. It's try/catch in floating point.
    """
    if layer not in EXCEPTION_NEURONS:
        return torch.zeros(
            activations.shape[0], activations.shape[1],
            dtype=torch.bool, device=activations.device
        )
    return neuron_fires(activations, EXCEPTION_NEURONS[layer])


# ============================================================
# MLP IMPLEMENTATIONS
# These are the "pipes" — what happens to the continuous signal
# after the binary routing decision is made.
# ============================================================

def full_mlp(block, x_normed):
    """Full nonlinear MLP path: project up → GELU → project down.
    
    This is the standard transformer MLP. 768 → 3072 → 768.
    The GELU activation is where the nonlinear magic happens —
    it's NOT a smooth function along the data manifold, it's a switch.
    
    Cost: ~4.7M multiply-adds per token.
    When it fires: exception handling, disambiguation, rare patterns.
    """
    # Passthrough to original implementation.
    # For the decomposed view, see decomposed_mlp_L11().
    return block.mlp(x_normed)


def full_mlp_with_activations(block, x_normed):
    """Same as full_mlp but also returns post-GELU activations for routing.
    
    We need the activations to check consensus/gateway/exception,
    but we don't want to compute them twice. So this returns both.
    """
    h = block.mlp.c_fc(x_normed)       # Project up: 768 → 3072
    a = block.mlp.act(h)               # GELU: the nonlinear switch
    out = block.mlp.c_proj(a)          # Project down: 3072 → 768
    out = block.mlp.dropout(out)
    return out, a


# ============================================================
# DECOMPOSED MLP: The tiered view
# ============================================================
# 
# Instead of treating the MLP as 3072 inscrutable neurons,
# we decompose it into functional circuits discovered via
# co-firing analysis and output-direction characterization.
#
# The math is identical: output = sum(a_i * W_out[i]) + bias.
# But we group the terms by function.

_CORE_NEURONS = list(EXCEPTION_CORE.keys())
_DIFF_NEURONS = list(EXCEPTION_DIFFERENTIATORS.keys())
_SPEC_NEURONS = list(EXCEPTION_SPECIALISTS.keys())
_NAMED_NEURONS = set(_CORE_NEURONS + _DIFF_NEURONS + _SPEC_NEURONS)


def _neuron_group_contribution(activations, W_out, neuron_indices):
    """Compute the output contribution of a specific group of neurons.
    
    Each neuron i contributes: activation[i] * W_out[i, :]
    This returns the sum of contributions for the given group.
    """
    # activations: [B, T, 3072], W_out: [3072, 768]
    # Select just these neurons and project
    a_subset = activations[:, :, neuron_indices]     # [B, T, K]
    W_subset = W_out[neuron_indices, :]              # [K, 768]
    return a_subset @ W_subset                       # [B, T, 768]


def decomposed_mlp_L11(block, x_normed, activations=None):
    """Layer 11 MLP decomposed into interpretable circuits.
    
    Produces IDENTICAL output to full_mlp — same math, different grouping.
    
    The exception handler is not 3072 inscrutable neurons.
    It's three circuits:
    
    TIER 1 — "Vocabulary Reset" (5 neurons, fire 90-100%)
        N2123, N2910, N740, N1611, N2044
        When consensus breaks, push residual stream toward common
        function words (the, in, and, a). "I don't know what this
        token should predict, so default to the most likely words."
        These five are effectively one fused unit (Jaccard ≥ 0.91).
    
    TIER 2 — "Selective Correction" (10 neurons, fire 35-88%)
        Two sub-circuits:
        a) Suppression pair (N584 + N2378, Jaccard 0.889):
           Push AWAY from common words. Anti-boost wrong candidates.
        b) Subword repair (N1602, N1715):
           Handle word fragments and multi-token continuations
           ("acebook" → Facebook, "archment" → parchment).
    
    TIER 3 — "Structural Specialists" (5 neurons, fire 14-37%)
        N737: paragraph boundary detector (fires almost solo).
        N2921/N2709/N971: section boundary handlers.
        N2679: formatting and special characters.
        Structurally independent from Tiers 1-2 (Jaccard < 0.15).
    
    RESIDUAL — The remaining ~3040 neurons.
        Collectively contribute ~25% of the output signal.
        Distributed fine-tuning corrections with no clear individual roles.
    """
    W_out = block.mlp.c_proj.weight.data  # [3072, 768]
    b_out = block.mlp.c_proj.bias.data    # [768]
    
    if activations is None:
        h = block.mlp.c_fc(x_normed)
        activations = block.mlp.act(h)
    
    # === TIER 1: Vocabulary Reset ===
    # "Consensus broke. Default to common words."
    # 5 neurons, fused unit, fires on ~every exception token.
    core_output = _neuron_group_contribution(activations, W_out, _CORE_NEURONS)
    
    # === TIER 2: Selective Correction ===
    # Suppression (N584+N2378): "Not THOSE common words."
    # Subword repair (N1602, N1715): "This is a word fragment."
    diff_output = _neuron_group_contribution(activations, W_out, _DIFF_NEURONS)
    
    # === TIER 3: Structural Specialists ===
    # N737: "This is a paragraph boundary."
    # N2921/N2709/N971: "This is a section boundary."
    # N2679: "This is special formatting."
    spec_output = _neuron_group_contribution(activations, W_out, _SPEC_NEURONS)
    
    # === RESIDUAL: Everything else ===
    # ~3040 neurons contributing distributed fine-tuning.
    # Collectively ~25% of signal. No clear individual roles.
    all_indices = list(range(3072))
    residual_indices = [i for i in all_indices if i not in _NAMED_NEURONS]
    residual_output = _neuron_group_contribution(activations, W_out, residual_indices)
    
    # Sum = identical to standard MLP output
    total = core_output + diff_output + spec_output + residual_output + b_out
    
    return total, {
        'core': core_output,
        'differentiators': diff_output,
        'specialists': spec_output,
        'residual': residual_output,
    }


def linear_bypass(x_normed):
    """The "consensus path": token is normal, MLP adds nothing.
    
    When consensus holds, the MLP's contribution is not just small —
    it's actively counterproductive (boost < 1.0x at full consensus).
    The cheapest and BEST thing to do is: nothing.
    
    Cost: 0 multiply-adds.
    This is the whole point: 40-80% of tokens can skip the MLP entirely.
    
    TODO: Investigate whether a tiny learned correction (rank-1? rank-4?)
    could capture whatever small residual the MLP contributes at consensus.
    """
    # Return zeros. The residual connection will pass x through unchanged.
    return torch.zeros_like(x_normed)


def attend(block, x_normed, **kwargs):
    """Attention: who does this token look at?
    
    93 of 144 heads are BOS sinks (dumping attention to position 0).
    L11H7 alone has 6x the importance of any other head.
    
    TODO: Decompose into functional head groups:
      - BOS sinks (93 heads): safe to skip?
      - Induction heads: pattern matching / in-context learning
      - Previous-token heads: local context
      - L11H7: the dominant head — what does it actually do?
    """
    # Currently a passthrough. Future: per-head routing.
    return block.attn(x_normed, **kwargs)[0]


# ============================================================
# THE TRANSPARENT FORWARD PASS
# ============================================================

class TransparentGPT2:
    """GPT-2 Small with its routing logic made visible.
    
    Standard GPT-2:
        for layer in layers:
            x = x + attention(x)
            x = x + mlp(x)
    
    Transparent GPT-2 (same math, visible routing):
        for layer in layers:
            x = x + attend(x)
            if phase == 'scaffold':
                if gateway_fires(x): x = x + full_mlp(x)
                else:                x = x + linear_bypass(x)
            elif phase == 'diffuse':
                x = x + full_mlp(x)  # no shortcuts possible
            elif phase == 'decision':
                if consensus_holds(x): x = x + linear_bypass(x)
                else:                  x = x + full_mlp(x)
    """
    
    def __init__(self, device='cuda'):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.model.eval()
        self.device = device
    
    def forward(self, input_ids, mode='transparent', bypass_layers=None):
        """
        The readable forward pass.
        
        Args:
            input_ids: [B, T] token ids
            mode: 'standard' | 'transparent' | 'bypass'
            bypass_layers: set of layers to actually bypass (bypass mode only).
                           Default: BYPASSABLE_LAYERS (L7-11)
        
        Returns:
            logits: [B, T, vocab_size]
            routing_log: per-layer routing statistics
        """
        if bypass_layers is None:
            bypass_layers = BYPASSABLE_LAYERS
        
        B, T = input_ids.shape
        routing_log = {}
        
        # === EMBEDDING ===
        # Token identity + position → initial representation
        token_embeddings = self.model.transformer.wte(input_ids)
        position_embeddings = self.model.transformer.wpe(
            torch.arange(T, device=self.device)
        )
        x = self.model.transformer.drop(token_embeddings + position_embeddings)
        
        # === THE 12 LAYERS ===
        for i, block in enumerate(self.model.transformer.h):
            phase = PHASES[i]
            
            # ---- Step 1: Attention ----
            # "Who should I look at?"
            x_norm_attn = block.ln_1(x)
            x = x + attend(block, x_norm_attn)
            
            # ---- Step 2: MLP ----
            # "What should I do with what I saw?"
            x_normed = block.ln_2(x)
            
            if mode == 'standard':
                # Black box: just run it
                x = x + full_mlp(block, x_normed)
                
            elif mode == 'transparent':
                # Same math, but we observe the routing
                mlp_out, activations = full_mlp_with_activations(block, x_normed)
                x = x + mlp_out  # identical to standard
                
                # Log what happened (doesn't change computation)
                routing_log[i] = self._log_routing(activations, i, B * T)
                
            elif mode == 'bypass':
                if i not in bypass_layers:
                    # This layer isn't being bypassed — run normally
                    mlp_out, activations = full_mlp_with_activations(block, x_normed)
                    x = x + mlp_out
                    routing_log[i] = self._log_routing(activations, i, B * T)
                else:
                    # This layer IS being bypassed — route based on consensus
                    mlp_out, activations = full_mlp_with_activations(block, x_normed)
                    
                    if phase == 'scaffold' and i in GATEWAY_NEURONS:
                        # Gateway routing: full MLP only if gateway fires
                        gw = gateway_fires(activations, i).unsqueeze(-1).float()
                        x = x + mlp_out * gw  # zero for non-gateway tokens
                        
                    elif i in CONSENSUS_NEURONS:
                        # Consensus routing: zero MLP when consensus holds
                        cons = consensus_holds(activations, i).unsqueeze(-1).float()
                        x = x + mlp_out * (1 - cons)  # zero for consensus tokens
                        
                    else:
                        # No routing structure — must use full MLP
                        x = x + mlp_out
                    
                    routing_log[i] = self._log_routing(activations, i, B * T)
        
        # === OUTPUT ===
        # Final layer norm → project to vocabulary
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x)
        
        return logits, routing_log
    
    def _log_routing(self, activations, layer, total_tokens):
        """Record routing decisions for this layer."""
        phase = PHASES[layer]
        log = {
            'phase': phase,
            'total': total_tokens,
            'consensus': 0,
            'exception': 0,
            'gateway': 0,
        }
        
        if layer in CONSENSUS_NEURONS:
            log['consensus'] = consensus_holds(activations, layer).sum().item()
        if layer in EXCEPTION_NEURONS:
            log['exception'] = exception_fires(activations, layer).sum().item()
        if layer in GATEWAY_NEURONS:
            log['gateway'] = gateway_fires(activations, layer).sum().item()
        
        return log
    
    def print_routing_report(self, routing_log):
        """Human-readable routing report."""
        markers = {'scaffold': '🔧', 'diffuse': '🌊', 'decision': '⚡'}
        
        print(f"\n{'='*65}")
        print(f"  ROUTING REPORT")
        print(f"{'='*65}")
        
        for i in range(12):
            if i not in routing_log:
                phase = PHASES[i]
                print(f"  L{i:2d} {markers[phase]} {phase:9s} | (not instrumented)")
                continue
            
            log = routing_log[i]
            phase = log['phase']
            total = log['total']
            
            parts = []
            if log['consensus']:
                parts.append(f"consensus={100*log['consensus']/total:.1f}%")
            if log['exception']:
                parts.append(f"exception={100*log['exception']/total:.1f}%")
            if log['gateway']:
                parts.append(f"gateway={100*log['gateway']/total:.1f}%")
            
            detail = ', '.join(parts) if parts else '(no routing structure)'
            print(f"  L{i:2d} {markers[phase]} {phase:9s} | {detail}")
        
        print(f"{'='*65}\n")
