#!/usr/bin/env python3
"""
Darkness Visible — Reviewer Controls
Addresses four methodological concerns:
1. Null model: random-init GPT-2 → does tier structure appear?
2. Threshold robustness: sweep 0.01-1.0, measure tier stability
3. Enrichment specification: document the statistical test
4. Context-dependent knowledge test: use actual activations, not static W_proj

Run on Mac mini (CPU, GPT-2 Small). ~15 min total.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from collections import defaultdict
import json
import time
import os

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "projects/transparent-gpt2/data/controls"
os.makedirs(OUT_DIR, exist_ok=True)

# L11 neuron IDs from architecture.py
CONSENSUS = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION = 2123
CORE = [2123, 2910, 740, 1611, 2044]
DIFFERENTIATORS = [584, 2378, 611, 2173, 1602, 1715, 935, 2856, 1427, 377]
SPECIALISTS = [737, 1375, 1037, 1994, 1224]
ALL_ROUTING = CORE + DIFFERENTIATORS + SPECIALISTS + [n for n in CONSENSUS if n not in CORE]

def load_tokens(n=100_000):
    """Load WikiText tokens. Use fewer for speed on CPU."""
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    # Use wikitext from HF datasets or a local cache
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text = " ".join([t for t in ds["text"][:5000] if t.strip()])
        tokens = tok.encode(text)[:n]
    except:
        # Fallback: generate synthetic text
        print("WARNING: Using random token IDs as fallback")
        tokens = list(range(1000, 1000 + n))
    return tokens

def get_l11_activations(model, tokens, seq_len=256, max_tokens=None):
    """Run forward pass, capture L11 MLP pre-GELU activations."""
    if max_tokens:
        tokens = tokens[:max_tokens]
    n_seqs = len(tokens) // seq_len
    tokens_t = torch.tensor(tokens[:n_seqs * seq_len], dtype=torch.long).reshape(n_seqs, seq_len)
    
    all_pre_gelu = []
    captured = {}
    
    def hook(module, input, output):
        captured['pre_gelu'] = output.detach().cpu()
    
    handle = model.transformer.h[11].mlp.c_fc.register_forward_hook(hook)
    
    with torch.no_grad():
        for i in range(n_seqs):
            batch = tokens_t[i:i+1].to(DEVICE)
            model(batch)
            all_pre_gelu.append(captured['pre_gelu'].squeeze(0))  # (seq_len, 3072)
    
    handle.remove()
    return torch.cat(all_pre_gelu, dim=0).numpy()  # (total_tokens, 3072)


def gelu(x):
    """Numpy GELU approximation."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ================================================================
# CONTROL 1: Null model (random-init GPT-2)
# ================================================================
def run_null_model(tokens):
    print("\n" + "="*60)
    print("CONTROL 1: Random-init GPT-2 (null model)")
    print("="*60)
    
    # Random-init model (same architecture, random weights)
    config = GPT2Config()
    random_model = GPT2LMHeadModel(config).to(DEVICE).eval()
    
    # Trained model
    trained_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    
    results = {}
    for name, model in [("random", random_model), ("trained", trained_model)]:
        print(f"\n--- {name} model ---")
        pre_gelu = get_l11_activations(model, tokens, max_tokens=50_000)
        post_gelu = gelu(pre_gelu)
        firing = np.abs(post_gelu) > 0.1
        
        fire_rates = firing.mean(axis=0)  # (3072,)
        
        # Measure tier structure: do CORE neurons co-fire?
        core_firing = firing[:, CORE]  # (tokens, 5)
        core_jaccard = []
        for i in range(len(CORE)):
            for j in range(i+1, len(CORE)):
                intersection = (core_firing[:, i] & core_firing[:, j]).sum()
                union = (core_firing[:, i] | core_firing[:, j]).sum()
                if union > 0:
                    core_jaccard.append(float(intersection / union))
        
        # Measure consensus-exception anticorrelation
        cons_firing = firing[:, CONSENSUS]
        cons_count = cons_firing.sum(axis=1)  # how many consensus neurons fire per token
        exc_firing = firing[:, EXCEPTION]
        
        # Exception rate at each consensus level
        exc_by_consensus = {}
        for level in range(8):
            mask = cons_count == level
            if mask.sum() > 0:
                exc_by_consensus[level] = float(exc_firing[mask].mean())
        
        # Enrichment: do any neurons show >1.5x enrichment when exception fires?
        exc_mask = exc_firing.astype(bool)
        n_exc = exc_mask.sum()
        enrichments = []
        if n_exc > 100:
            for n_id in range(3072):
                base_rate = fire_rates[n_id]
                if base_rate > 0.01:
                    exc_rate = firing[exc_mask, n_id].mean()
                    enrichments.append(float(exc_rate / base_rate))
                else:
                    enrichments.append(1.0)
        
        n_enriched_15x = sum(1 for e in enrichments if e > 1.5) if enrichments else 0
        n_enriched_2x = sum(1 for e in enrichments if e > 2.0) if enrichments else 0
        max_enrichment = max(enrichments) if enrichments else 0
        
        res = {
            "fire_rates": {
                "exception_N2123": float(fire_rates[EXCEPTION]),
                "consensus_mean": float(fire_rates[CONSENSUS].mean()),
                "core_mean": float(fire_rates[CORE].mean()),
            },
            "core_jaccard_mean": float(np.mean(core_jaccard)) if core_jaccard else 0,
            "core_jaccard_min": float(np.min(core_jaccard)) if core_jaccard else 0,
            "exc_by_consensus": exc_by_consensus,
            "enrichment": {
                "n_enriched_1.5x": n_enriched_15x,
                "n_enriched_2x": n_enriched_2x,
                "max_enrichment": max_enrichment,
            },
        }
        results[name] = res
        
        print(f"  Exception fire rate: {res['fire_rates']['exception_N2123']:.3f}")
        print(f"  Core Jaccard mean: {res['core_jaccard_mean']:.3f} (min: {res['core_jaccard_min']:.3f})")
        print(f"  Exc by consensus: {exc_by_consensus}")
        print(f"  Neurons >1.5x enriched: {n_enriched_15x}, >2x: {n_enriched_2x}")
    
    # Clean up
    del random_model, trained_model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    
    with open(f"{OUT_DIR}/null_model.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/null_model.json")
    return results


# ================================================================
# CONTROL 2: Threshold robustness sweep
# ================================================================
def run_threshold_sweep(tokens):
    print("\n" + "="*60)
    print("CONTROL 2: Threshold robustness sweep")
    print("="*60)
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    pre_gelu = get_l11_activations(model, tokens, max_tokens=50_000)
    post_gelu = gelu(pre_gelu)
    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    
    thresholds = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    results = {}
    
    for thresh in thresholds:
        firing = np.abs(post_gelu) > thresh
        fire_rates = firing.mean(axis=0)
        
        # Core co-firing
        core_firing = firing[:, CORE]
        core_jaccard = []
        for i in range(len(CORE)):
            for j in range(i+1, len(CORE)):
                inter = (core_firing[:, i] & core_firing[:, j]).sum()
                union = (core_firing[:, i] | core_firing[:, j]).sum()
                if union > 0:
                    core_jaccard.append(float(inter / union))
        
        # Consensus-exception anticorrelation
        cons_count = firing[:, CONSENSUS].sum(axis=1)
        exc_firing = firing[:, EXCEPTION]
        exc_at_0 = float(exc_firing[cons_count == 0].mean()) if (cons_count == 0).sum() > 0 else None
        exc_at_7 = float(exc_firing[cons_count == 7].mean()) if (cons_count == 7).sum() > 0 else None
        
        res = {
            "threshold": thresh,
            "exception_fire_rate": float(fire_rates[EXCEPTION]),
            "consensus_mean_fire_rate": float(fire_rates[CONSENSUS].mean()),
            "core_jaccard_mean": float(np.mean(core_jaccard)) if core_jaccard else 0,
            "core_jaccard_min": float(np.min(core_jaccard)) if core_jaccard else 0,
            "exc_at_0_consensus": exc_at_0,
            "exc_at_7_consensus": exc_at_7,
            "anticorrelation_spread": (exc_at_0 - exc_at_7) if (exc_at_0 is not None and exc_at_7 is not None) else None,
        }
        results[str(thresh)] = res
        print(f"  θ={thresh:.2f}: exc_rate={res['exception_fire_rate']:.3f}, "
              f"jaccard={res['core_jaccard_mean']:.3f}, "
              f"spread={res['anticorrelation_spread']}")
    
    with open(f"{OUT_DIR}/threshold_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/threshold_sweep.json")
    return results


# ================================================================
# CONTROL 3: Enrichment specification (document the method)
# ================================================================
def run_enrichment_specification(tokens):
    """
    The enrichment method used in the paper:
    
    For each neuron N and condition C (e.g., "exception fires"):
      enrichment(N, C) = P(N fires | C) / P(N fires)
    
    This is a simple frequency ratio. Significance is assessed by:
    - Fisher's exact test on the 2x2 contingency table
    - Bonferroni correction for 3072 neurons
    
    This function runs the full enrichment with proper stats.
    """
    from scipy import stats
    
    print("\n" + "="*60)
    print("CONTROL 3: Enrichment analysis with statistical tests")
    print("="*60)
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    pre_gelu = get_l11_activations(model, tokens, max_tokens=50_000)
    post_gelu = gelu(pre_gelu)
    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    
    firing = np.abs(post_gelu) > 0.1
    exc_mask = firing[:, EXCEPTION].astype(bool)
    n_total = len(exc_mask)
    n_exc = exc_mask.sum()
    n_non_exc = n_total - n_exc
    
    print(f"  Total tokens: {n_total}, Exception: {n_exc} ({n_exc/n_total:.1%})")
    
    # For each routing neuron, compute enrichment + Fisher's exact test
    bonferroni = 3072
    results = {}
    
    for n_id in ALL_ROUTING:
        n_fires = firing[:, n_id].astype(bool)
        
        # 2x2 table: [exc & fires, exc & !fires], [!exc & fires, !exc & !fires]
        a = (exc_mask & n_fires).sum()
        b = (exc_mask & ~n_fires).sum()
        c = (~exc_mask & n_fires).sum()
        d = (~exc_mask & ~n_fires).sum()
        
        base_rate = n_fires.mean()
        exc_rate = n_fires[exc_mask].mean() if n_exc > 0 else 0
        enrichment = exc_rate / base_rate if base_rate > 0 else float('inf')
        
        # Fisher's exact test (one-sided: enrichment)
        _, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        
        tier = ("core" if n_id in CORE else 
                "differentiator" if n_id in DIFFERENTIATORS else
                "specialist" if n_id in SPECIALISTS else
                "consensus")
        
        results[str(n_id)] = {
            "tier": tier,
            "base_rate": float(base_rate),
            "exc_rate": float(exc_rate),
            "enrichment": float(enrichment),
            "p_value": float(p_value),
            "significant_bonferroni": bool(p_value < 0.05 / bonferroni),
            "contingency": [int(a), int(b), int(c), int(d)],
        }
    
    # Summary
    sig_count = sum(1 for r in results.values() if r["significant_bonferroni"])
    print(f"  {sig_count}/{len(results)} routing neurons significantly enriched (Bonferroni p<{0.05/bonferroni:.1e})")
    for n_id, r in sorted(results.items(), key=lambda x: -x[1]["enrichment"]):
        print(f"  N{n_id} ({r['tier']}): {r['enrichment']:.2f}x enrichment, p={r['p_value']:.2e}")
    
    with open(f"{OUT_DIR}/enrichment_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/enrichment_stats.json")
    return results


# ================================================================
# CONTROL 4: Context-dependent knowledge test
# ================================================================
def run_context_knowledge_test():
    """
    The paper's 20 Questions test used static W_proj columns.
    This version uses actual context-dependent activations:
    post_gelu(x) * W_proj column = the actual contribution of each neuron.
    """
    print("\n" + "="*60)
    print("CONTROL 4: Context-dependent knowledge extraction")
    print("="*60)
    
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Factual prompts from the paper
    prompts = [
        ("In 1969, Neil Armstrong landed on the", "moon"),
        ("Water freezes at zero degrees", "C"),
        ("The playwright was William", "Shakespeare"),
        ("Einstein's theory of", "relativity"),
        ("The speed of light is 300,000 km per", "second"),
        ("The French Revolution began in", "17"),
        ("The chemical symbol for gold is", "Au"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("The capital of France is", "Paris"),
        ("DNA stands for deoxyribonucleic", "acid"),
        ("The Mona Lisa was painted by Leonardo da", "Vinci"),
        ("The Great Wall of", "China"),
    ]
    
    # Get W_proj for L11
    W_proj = model.transformer.h[11].mlp.c_proj.weight.detach().cpu().numpy()  # (768, 3072)
    ln_weight = model.transformer.ln_f.weight.detach().cpu().numpy()
    ln_bias = model.transformer.ln_f.bias.detach().cpu().numpy()
    unembed = model.lm_head.weight.detach().cpu().numpy()  # (50257, 768)
    
    results = []
    
    for prompt_text, target_word in prompts:
        input_ids = tokenizer.encode(prompt_text)
        target_id = tokenizer.encode(" " + target_word)[0]
        
        # Get L11 pre-GELU activations for last token
        captured = {}
        def hook(module, inp, out):
            captured['pre_gelu'] = out.detach().cpu()
        handle = model.transformer.h[11].mlp.c_fc.register_forward_hook(hook)
        
        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(DEVICE))
            model_pred = out.logits[0, -1].softmax(-1)
            model_pred_id = model_pred.argmax().item()
            model_pred_prob = model_pred[target_id].item()
        
        handle.remove()
        
        pre_gelu_last = captured['pre_gelu'][0, -1].numpy()  # (3072,)
        post_gelu_last = gelu(pre_gelu_last)
        
        # --- STATIC method (paper's approach): use W_proj columns directly ---
        # Each neuron's static output direction
        static_contributions = W_proj.T  # (3072, 768) — each row = one neuron's output direction
        
        # --- CONTEXT method: scale by actual activation ---
        context_contributions = post_gelu_last[:, None] * W_proj.T  # (3072, 768)
        
        # For both methods: accumulate residual neurons by magnitude, track target rank
        residual_mask = np.ones(3072, dtype=bool)
        for n_id in ALL_ROUTING:
            if n_id < 3072:
                residual_mask[n_id] = False
        residual_ids = np.where(residual_mask)[0]
        
        def rank_target(contributions, neuron_ids, target_token_id):
            """Accumulate neuron contributions and get target rank in vocab space."""
            # Sort by contribution magnitude
            magnitudes = np.linalg.norm(contributions[neuron_ids], axis=1)
            sorted_idx = neuron_ids[np.argsort(-magnitudes)]
            
            ranks_at_n = {}
            accumulated = np.zeros(768)
            for i, n_id in enumerate(sorted_idx):
                accumulated += contributions[n_id]
                if (i+1) in [5, 50, 200, 500, len(sorted_idx)]:
                    # Project through LN + unembed (approximate)
                    normed = (accumulated - accumulated.mean()) / (accumulated.std() + 1e-5)
                    normed = normed * ln_weight + ln_bias
                    logits = unembed @ normed
                    rank = (logits > logits[target_token_id]).sum() + 1
                    ranks_at_n[i+1] = int(rank)
            return ranks_at_n
        
        static_ranks = rank_target(static_contributions, residual_ids, target_id)
        context_ranks = rank_target(context_contributions, residual_ids, target_id)
        
        res = {
            "prompt": prompt_text,
            "target": target_word,
            "target_id": target_id,
            "model_pred": tokenizer.decode([model_pred_id]),
            "model_target_prob": float(model_pred_prob),
            "static_ranks": static_ranks,
            "context_ranks": context_ranks,
        }
        results.append(res)
        print(f"  '{prompt_text}' → '{target_word}'")
        print(f"    Static:  {static_ranks}")
        print(f"    Context: {context_ranks}")
    
    del model
    torch.mps.empty_cache() if DEVICE == "mps" else None
    
    with open(f"{OUT_DIR}/knowledge_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/knowledge_test.json")
    return results


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    t0 = time.time()
    
    print("Loading tokens...")
    tokens = load_tokens(n=100_000)
    print(f"  Loaded {len(tokens)} tokens")
    
    # Run all four controls
    null_results = run_null_model(tokens)
    threshold_results = run_threshold_sweep(tokens)
    enrichment_results = run_enrichment_specification(tokens)
    knowledge_results = run_context_knowledge_test()
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All controls complete in {elapsed:.0f}s")
    print(f"Results saved to {OUT_DIR}/")
    print(f"{'='*60}")
