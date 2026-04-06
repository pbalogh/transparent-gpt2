#!/usr/bin/env python3
"""
Survey exception handler structure across ALL 12 GPT-2 Small MLP layers.
Lightweight version — no scipy, pure numpy.
"""
import torch
import numpy as np
import json
import time
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

device = 'cpu'
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

FIRE_THRESHOLD = 0.1
N_SEQUENCES = 100  # 100K tokens is enough for structure detection
SEQ_LEN = 1024

dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
text = '\n\n'.join([t for t in dataset['text'] if t.strip()])
tokens = tokenizer.encode(text)

print(f"Analyzing {N_SEQUENCES} × {SEQ_LEN} = {N_SEQUENCES * SEQ_LEN} tokens")


def analyze_layer(layer_idx):
    """Analyze MLP structure at a given layer. Fast version."""
    
    # Collect binary activations
    all_binary = []
    for seq_idx in range(N_SEQUENCES):
        start = seq_idx * SEQ_LEN
        if start + SEQ_LEN > len(tokens):
            break
        
        input_ids = torch.tensor([tokens[start:start + SEQ_LEN]], device=device)
        activations = {}
        
        def hook_fn(module, input, output):
            activations['val'] = output.detach()
        
        handle = model.transformer.h[layer_idx].mlp.act.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(input_ids)
        handle.remove()
        
        binary = (activations['val'][0].abs() > FIRE_THRESHOLD).cpu().numpy()
        all_binary.append(binary)
    
    all_binary = np.vstack(all_binary)  # (n_tokens, 3072)
    n_tokens = all_binary.shape[0]
    fire_rates = all_binary.mean(axis=0)
    
    # Find top-7 highest fire rate neurons (consensus candidates)
    top7_idx = np.argsort(fire_rates)[-7:]
    consensus = all_binary[:, top7_idx].sum(axis=1)  # 0-7 per token
    
    # Find most anticorrelated neuron with consensus
    best_anticorr = 0
    exception_idx = 0
    for n in range(3072):
        if fire_rates[n] < 0.02 or fire_rates[n] > 0.98:
            continue
        corr = np.corrcoef(all_binary[:, n].astype(float), 
                          consensus.astype(float))[0, 1]
        if corr < best_anticorr:
            best_anticorr = corr
            exception_idx = n
    
    exc_firing = all_binary[:, exception_idx].astype(bool)
    exc_count = exc_firing.sum()
    
    # Max Jaccard with exception neuron (sample 200 neurons for speed)
    candidate_neurons = np.argsort(fire_rates)[::-1][:200]  # top 200 by fire rate
    max_jaccard = 0
    high_jaccard_count = 0
    for n in candidate_neurons:
        if n == exception_idx:
            continue
        intersection = (exc_firing & all_binary[:, n].astype(bool)).sum()
        union = (exc_firing | all_binary[:, n].astype(bool)).sum()
        if union > 0:
            j = intersection / union
            if j > max_jaccard:
                max_jaccard = j
            if j > 0.5:
                high_jaccard_count += 1
    
    # Enrichment: max(exc_rate / base_rate) across neurons
    max_enrichment = 1.0
    if exc_count > 10:
        for n in candidate_neurons:
            if n == exception_idx:
                continue
            base = fire_rates[n]
            if base > 0.01 and base < 0.99:
                exc_rate = all_binary[exc_firing, n].mean()
                enrich = exc_rate / base
                if enrich > max_enrichment:
                    max_enrichment = enrich
    
    # Exception rate by consensus level
    exc_by_consensus = {}
    for level in range(8):
        mask = consensus == level
        if mask.sum() > 0:
            exc_by_consensus[level] = float(exc_firing[mask].mean())
    
    # Anticorrelation spread
    spread = exc_by_consensus.get(0, 0) - exc_by_consensus.get(7, 0)
    
    return {
        'layer': layer_idx,
        'exception_neuron': int(exception_idx),
        'exception_fire_rate': round(float(fire_rates[exception_idx]), 4),
        'exception_corr': round(float(best_anticorr), 4),
        'max_jaccard': round(float(max_jaccard), 4),
        'high_jaccard_count': int(high_jaccard_count),
        'max_enrichment': round(float(max_enrichment), 3),
        'consensus_mean_rate': round(float(fire_rates[top7_idx].mean()), 4),
        'anticorr_spread': round(float(spread), 4),
        'consensus_neurons': [int(x) for x in top7_idx],
    }


results = []
t_start = time.time()

for layer in range(12):
    t0 = time.time()
    r = analyze_layer(layer)
    elapsed = time.time() - t0
    
    print(f"L{layer:2d}: exc={r['exception_fire_rate']:.3f} "
          f"J={r['max_jaccard']:.3f} "
          f"hiJ={r['high_jaccard_count']} "
          f"enr={r['max_enrichment']:.2f} "
          f"spr={r['anticorr_spread']:.3f} "
          f"corr={r['exception_corr']:.3f} "
          f"({elapsed:.0f}s)")
    results.append(r)

print(f"\nTotal: {time.time() - t_start:.0f}s")

# Summary
print(f"\n{'Layer':>5s} | {'Exc%':>6s} | {'MaxJ':>6s} | {'HiJ':>4s} | "
      f"{'Enrich':>7s} | {'Spread':>7s} | {'Corr':>6s}")
print("-" * 55)
for r in results:
    print(f"  L{r['layer']:<3d} | {r['exception_fire_rate']*100:>5.1f}% | "
          f"{r['max_jaccard']:>6.3f} | {r['high_jaccard_count']:>4d} | "
          f"{r['max_enrichment']:>7.2f} | {r['anticorr_spread']:>7.3f} | "
          f"{r['exception_corr']:>6.3f}")

outpath = os.path.expanduser('~/clawd/projects/transparent-gpt2/data/all_layers_structure.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outpath}")
