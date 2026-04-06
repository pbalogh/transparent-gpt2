#!/usr/bin/env python3
"""
Bootstrap confidence intervals for L11 MLP consensus-helpfulness crossover.

For each consensus level (0-7), compute ΔP (P_L11 - P_L10) for the correct
next token, then bootstrap to get 95% CIs.

This is the paper's central quantitative claim: the crossover from helpful
to harmful occurs at 3-4/7 consensus.
"""
import torch
import numpy as np
import json
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import functional as F

device = 'cpu'  # GPT-2 small is fine on CPU
print(f"Using device: {device}")

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Get consensus neuron indices (from Discrete Charm paper)
CONSENSUS_NEURONS = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION_NEURON = 2123
FIRE_THRESHOLD = 0.1


def compute_delta_p_by_consensus(n_sequences=200, seq_len=1024):
    """
    For each token, compute:
    1. Consensus level (how many of 7 consensus neurons fire)
    2. P(correct) before L11 MLP (P_L10)  
    3. P(correct) after L11 MLP (P_L11)
    4. ΔP = P_L11 - P_L10
    
    Returns dict: consensus_level -> list of ΔP values
    """
    from datasets import load_dataset
    
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    text = '\n\n'.join([t for t in dataset['text'] if t.strip()])
    tokens = tokenizer.encode(text)
    
    # Collect per-token data
    delta_p_by_level = {i: [] for i in range(8)}
    total_tokens = 0
    
    for seq_idx in range(n_sequences):
        start = seq_idx * seq_len
        if start + seq_len + 1 > len(tokens):
            break
            
        input_ids = torch.tensor([tokens[start:start + seq_len]], device=device)
        target_ids = tokens[start + 1:start + seq_len + 1]
        
        # Hook to capture L11 MLP pre and post
        pre_mlp = {}
        post_gelu = {}
        
        def hook_pre_mlp(module, input, output):
            pre_mlp['val'] = input[0].detach()  # residual stream before MLP
            
        def hook_post_gelu(module, input, output):
            post_gelu['val'] = output.detach()
        
        h_pre = model.transformer.h[11].register_forward_pre_hook(
            lambda m, i: None)  # dummy, we need a different approach
        
        # Actually, let's just do two forward passes:
        # 1. Normal forward pass to get P_L11
        # 2. Forward pass with L11 MLP zeroed to get P_L10
        
        # Remove dummy hook
        h_pre.remove()
        
        # Normal forward pass + capture post-GELU activations
        def capture_gelu(module, input, output):
            post_gelu['val'] = output.detach()
            
        handle_gelu = model.transformer.h[11].mlp.act.register_forward_hook(capture_gelu)
        
        with torch.no_grad():
            outputs_normal = model(input_ids)
            logits_l11 = outputs_normal.logits[0]  # (seq_len, vocab)
        
        handle_gelu.remove()
        gelu_acts = post_gelu['val'][0]  # (seq_len, 3072)
        
        # Zero L11 MLP forward pass
        def zero_mlp(module, input, output):
            return torch.zeros_like(output)
        
        handle_zero = model.transformer.h[11].mlp.register_forward_hook(zero_mlp)
        
        with torch.no_grad():
            outputs_no_mlp = model(input_ids)
            logits_l10 = outputs_no_mlp.logits[0]  # (seq_len, vocab)
        
        handle_zero.remove()
        
        # Compute per-token metrics
        probs_l11 = F.softmax(logits_l11, dim=-1)
        probs_l10 = F.softmax(logits_l10, dim=-1)
        
        for pos in range(seq_len):
            target = target_ids[pos]
            
            p_l11 = probs_l11[pos, target].item()
            p_l10 = probs_l10[pos, target].item()
            delta_p = p_l11 - p_l10
            
            # Compute consensus level
            acts = gelu_acts[pos]
            consensus = sum(1 for n in CONSENSUS_NEURONS if abs(acts[n].item()) > FIRE_THRESHOLD)
            
            delta_p_by_level[consensus].append(delta_p)
        
        total_tokens += seq_len
        if (seq_idx + 1) % 20 == 0:
            print(f"  Processed {seq_idx + 1}/{n_sequences} sequences ({total_tokens} tokens)")
    
    return delta_p_by_level, total_tokens


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    data = np.array(data)
    n = len(data)
    if n == 0:
        return 0, 0, 0
    
    means = np.array([np.mean(np.random.choice(data, size=n, replace=True)) 
                      for _ in range(n_bootstrap)])
    
    alpha = (1 - ci) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return np.mean(data), lo, hi


print("Computing ΔP by consensus level...")
t0 = time.time()
delta_p_by_level, total_tokens = compute_delta_p_by_consensus(n_sequences=200, seq_len=1024)
elapsed = time.time() - t0
print(f"\nDone: {total_tokens} tokens in {elapsed:.0f}s")

print("\nBootstrapping 95% CIs (10,000 resamples)...")
print(f"\n{'Level':>6s} | {'N':>7s} | {'Mean ΔP':>10s} | {'95% CI':>22s} | {'Sig?':>5s}")
print("-" * 65)

results = {}
for level in range(8):
    data = delta_p_by_level[level]
    mean, lo, hi = bootstrap_ci(data)
    sig = "YES" if (lo > 0 or hi < 0) else "no"
    print(f"  {level}/7  | {len(data):>7d} | {mean:>+10.4f} | [{lo:>+.4f}, {hi:>+.4f}] | {sig:>5s}")
    results[f"{level}/7"] = {
        "n": len(data),
        "mean": round(mean, 5),
        "ci_lo": round(lo, 5),
        "ci_hi": round(hi, 5),
        "significant": sig == "YES"
    }

# Check crossover
print(f"\nCrossover analysis:")
for level in range(8):
    data = delta_p_by_level[level]
    mean, lo, hi = bootstrap_ci(data)
    if lo <= 0 <= hi:
        print(f"  {level}/7: CI spans zero [{lo:+.4f}, {hi:+.4f}] — crossover region")

# Save results
import os
outpath = os.path.expanduser('~/clawd/projects/transparent-gpt2/data/bootstrap_crossover.json')
with open(outpath, 'w') as f:
    json.dump({
        'total_tokens': total_tokens,
        'n_bootstrap': 10000,
        'ci_level': 0.95,
        'results': results,
    }, f, indent=2)
print(f"\nSaved to {outpath}")
