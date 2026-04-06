"""
Progressive Prediction: Watch the MLP build its answer tier by tier.

For each token, show what the model would predict if you stopped
at each stage of the exception handler:

  Stage 0: No MLP at all (just attention + residual stream)
  Stage 1: + Core circuit (vocabulary reset)
  Stage 2: + Differentiators (suppression + subword repair)
  Stage 3: + Specialists (paragraph/section boundaries)
  Stage 4: + Residual (full MLP, all 3072 neurons)

This reveals the coarse-to-fine computation:
  Core provides energy (big push, approximate direction)
  Each tier adds precision (steering toward the right token)
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '/Users/peter/clawd/projects/transparent-gpt2')

from src.architecture import (
    CONSENSUS_NEURONS, FIRE_THRESHOLD, CONSENSUS_THRESHOLD,
    EXCEPTION_CORE, EXCEPTION_DIFFERENTIATORS, EXCEPTION_SPECIALISTS,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer


_CORE = list(EXCEPTION_CORE.keys())
_DIFF = list(EXCEPTION_DIFFERENTIATORS.keys())
_SPEC = list(EXCEPTION_SPECIALISTS.keys())
_NAMED = set(_CORE + _DIFF + _SPEC)
_RESIDUAL = [i for i in range(3072) if i not in _NAMED]


def neuron_group_output(activations, W_out, indices):
    """Output contribution from a group of neurons."""
    a_sub = activations[:, indices]   # [T, K]
    W_sub = W_out[indices, :]         # [K, 768]
    return a_sub @ W_sub              # [T, 768]


def get_top_predictions(logits, tokenizer, k=5):
    """Top-k predicted tokens from logits."""
    probs = F.softmax(logits, dim=-1)
    topk = probs.topk(k)
    tokens = [tokenizer.decode([i]) for i in topk.indices.tolist()]
    scores = [round(p, 4) for p in topk.values.tolist()]
    return list(zip(tokens, scores))


def main():
    device = 'cpu'
    print("Loading GPT-2 Small...")
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Test sequences — mix of easy and hard tokens
    test_texts = [
        "The capital of France is",
        "She walked into the room and",
        "The quick brown fox jumps over the",
        "In 1969, humans first landed on the",
        "\n\nThe following year, the team",
    ]
    
    block = model.transformer.h[11]
    W_out = block.mlp.c_proj.weight.data   # [3072, 768]
    b_out = block.mlp.c_proj.bias.data     # [768]
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    
    for text in test_texts:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        T = input_ids.shape[1]
        
        print(f"\n{'='*80}")
        print(f"  INPUT: \"{text}\"")
        print(f"{'='*80}")
        
        with torch.no_grad():
            # Forward pass to L11 attention output
            wte = model.transformer.wte(input_ids)
            wpe = model.transformer.wpe(torch.arange(T, device=device))
            x = model.transformer.drop(wte + wpe)
            
            for i in range(12):
                b = model.transformer.h[i]
                attn_out = b.attn(b.ln_1(x))[0]
                x = x + attn_out
                if i < 11:
                    x = x + b.mlp(b.ln_2(x))
                else:
                    # L11: decompose
                    x_normed = b.ln_2(x)
                    h = b.mlp.c_fc(x_normed)
                    activations = b.mlp.act(h)[0]  # [T, 3072]
                    
                    # Check consensus for each token
                    cons_neurons = CONSENSUS_NEURONS[11]
                    votes = torch.stack([activations[:, n] > FIRE_THRESHOLD for n in cons_neurons], dim=-1)
                    frac = votes.float().mean(dim=-1)
                    is_consensus = frac >= CONSENSUS_THRESHOLD
                    
                    # Compute tier outputs
                    core_out = neuron_group_output(activations, W_out, _CORE)
                    diff_out = neuron_group_output(activations, W_out, _DIFF)
                    spec_out = neuron_group_output(activations, W_out, _SPEC)
                    resid_out = neuron_group_output(activations, W_out, _RESIDUAL)
                    
                    # Base = residual stream after L11 attention, before MLP
                    x_base = x[0]  # [T, 768]
            
            # For each position, show progressive predictions
            # We care about the LAST token (what comes next?) but also
            # show a few interior tokens for illustration
            
            positions_to_show = list(range(max(0, T-5), T))
            
            for pos in positions_to_show:
                tok = tokenizer.decode([input_ids[0, pos].item()])
                route = "CONSENSUS" if is_consensus[pos].item() else "EXCEPTION"
                
                # Build progressive MLP outputs
                stages = {
                    'No MLP':       torch.zeros(768),
                    '+ Core':       core_out[pos] + b_out,
                    '+ Diff':       core_out[pos] + diff_out[pos] + b_out,
                    '+ Spec':       core_out[pos] + diff_out[pos] + spec_out[pos] + b_out,
                    '+ Residual':   core_out[pos] + diff_out[pos] + spec_out[pos] + resid_out[pos] + b_out,
                }
                
                print(f"\n  Token: \"{tok}\" [{route}]")
                print(f"  {'Stage':<14s} {'Top-1':>12s} {'p':>6s} | {'Top-2':>12s} {'p':>6s} | {'Top-3':>12s} {'p':>6s} | {'Top-4':>12s} {'p':>6s} | {'Top-5':>12s} {'p':>6s}")
                print(f"  {'-'*100}")
                
                for stage_name, mlp_contribution in stages.items():
                    # Add MLP contribution to residual stream
                    x_with_mlp = x_base[pos] + mlp_contribution
                    
                    # Final layer norm + unembedding
                    x_final = ln_f(x_with_mlp.unsqueeze(0))
                    logits = lm_head(x_final)[0]  # [vocab]
                    
                    preds = get_top_predictions(logits, tokenizer, k=5)
                    
                    parts = []
                    for t, p in preds:
                        parts.append(f"{t:>12s} {p:>6.3f}")
                    
                    print(f"  {stage_name:<14s} {' | '.join(parts)}")
    
    # Also show a concrete narrative example
    print(f"\n\n{'='*80}")
    print("  NARRATIVE: Watch 'Paris' emerge from noise")
    print(f"{'='*80}")
    
    text = "The capital of France is"
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    T = input_ids.shape[1]
    
    with torch.no_grad():
        wte = model.transformer.wte(input_ids)
        wpe = model.transformer.wpe(torch.arange(T, device=device))
        x = model.transformer.drop(wte + wpe)
        
        for i in range(12):
            b = model.transformer.h[i]
            attn_out = b.attn(b.ln_1(x))[0]
            x = x + attn_out
            if i < 11:
                x = x + b.mlp(b.ln_2(x))
            else:
                x_normed = b.ln_2(x)
                h = b.mlp.c_fc(x_normed)
                activations = b.mlp.act(h)[0]
                
                core_out = neuron_group_output(activations, W_out, _CORE)
                diff_out = neuron_group_output(activations, W_out, _DIFF)
                spec_out = neuron_group_output(activations, W_out, _SPEC)
                resid_out = neuron_group_output(activations, W_out, _RESIDUAL)
                
                x_base = x[0]
    
    # Last position: predicting what comes after "is"
    pos = T - 1
    print(f"\n  Predicting the token after \"...France is\":\n")
    
    stages = [
        ('No MLP (attention only)', torch.zeros(768)),
        ('+ Core (5 neurons)',      core_out[pos] + b_out),
        ('+ Diff (+10 neurons)',    core_out[pos] + diff_out[pos] + b_out),
        ('+ Spec (+5 neurons)',     core_out[pos] + diff_out[pos] + spec_out[pos] + b_out),
        ('+ Residual (all 3072)',   core_out[pos] + diff_out[pos] + spec_out[pos] + resid_out[pos] + b_out),
    ]
    
    for stage_name, mlp_contribution in stages:
        x_with_mlp = x_base[pos] + mlp_contribution
        x_final = ln_f(x_with_mlp.unsqueeze(0))
        logits = lm_head(x_final)[0]
        
        preds = get_top_predictions(logits, tokenizer, k=8)
        top1 = preds[0]
        rest = preds[1:6]
        
        # Check rank of " Paris"
        paris_id = tokenizer.encode(" Paris")[0]
        probs = F.softmax(logits, dim=-1)
        paris_prob = probs[paris_id].item()
        paris_rank = (probs > paris_prob).sum().item() + 1
        
        print(f"  {stage_name:<30s}")
        print(f"    Top-1: {top1[0]} (p={top1[1]:.4f})")
        print(f"    Also:  {', '.join(f'{t}({p:.3f})' for t,p in rest)}")
        print(f"    \" Paris\": rank={paris_rank}, p={paris_prob:.4f}")
        print()
    
    print("Done!")


if __name__ == '__main__':
    main()
