"""
Evaluation harness: compare standard vs transparent vs bypass modes.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from .transparent_model import TransparentGPT2
from .architecture import BYPASSABLE_LAYERS


def eval_perplexity(model, tokens, mode='transparent', bypass_layers=None,
                    seq_len=1024, max_seqs=200, verbose=True):
    """Evaluate perplexity on a token array."""
    n_seqs = min(max_seqs, len(tokens) // seq_len)
    total_loss = 0.0
    total_count = 0
    all_logs = {}
    
    if verbose:
        print(f"  Evaluating '{mode}' mode ({n_seqs} × {seq_len} tokens)...")
    
    t0 = time.time()
    with torch.no_grad():
        for s in range(n_seqs):
            chunk = tokens[s * seq_len : (s + 1) * seq_len]
            input_ids = torch.tensor(chunk, dtype=torch.long, device=model.device).unsqueeze(0)
            
            logits, routing_log = model.forward(input_ids, mode=mode, bypass_layers=bypass_layers)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), reduction='sum'
            )
            total_loss += loss.item()
            total_count += shift_labels.numel()
            
            # Accumulate routing logs
            for layer, log in routing_log.items():
                if layer not in all_logs:
                    all_logs[layer] = {k: 0 for k in log}
                    all_logs[layer]['phase'] = log['phase']
                for k in ['total', 'consensus', 'exception', 'gateway']:
                    all_logs[layer][k] += log[k]
            
            if verbose and (s + 1) % 50 == 0:
                ppl = np.exp(total_loss / total_count)
                print(f"    {s+1}/{n_seqs}, PPL so far: {ppl:.2f}")
    
    elapsed = time.time() - t0
    ppl = np.exp(total_loss / total_count)
    
    if verbose:
        print(f"  → PPL = {ppl:.2f} ({elapsed:.1f}s)\n")
    
    return ppl, all_logs


def run_comparison(token_path, device='cuda', max_seqs=200):
    """Run the three-column comparison: standard vs transparent vs bypass."""
    
    print("Loading tokens...")
    tokens = np.load(token_path)[:max_seqs * 1024]
    print(f"  {len(tokens)} tokens\n")
    
    model = TransparentGPT2(device=device)
    
    print("=" * 65)
    print("  THREE-COLUMN COMPARISON")
    print("=" * 65)
    
    # Column 1: Standard (black box)
    print("\n[1/3] STANDARD GPT-2 (black box)")
    ppl_std, _ = eval_perplexity(model, tokens, mode='standard')
    
    # Column 2: Transparent (same math, visible routing)
    print("[2/3] TRANSPARENT GPT-2 (same math, routing visible)")
    ppl_trans, trans_logs = eval_perplexity(model, tokens, mode='transparent')
    model.print_routing_report(trans_logs)
    
    # Column 3: Bypass (exploit routing)
    print("[3/3] BYPASS GPT-2 (zero MLP at consensus, L7-11)")
    ppl_bypass, bypass_logs = eval_perplexity(
        model, tokens, mode='bypass', bypass_layers=BYPASSABLE_LAYERS
    )
    model.print_routing_report(bypass_logs)
    
    # Summary table
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  {'Mode':<20s} {'PPL':>8s} {'Δ':>10s} {'Note'}")
    print(f"  {'-'*58}")
    print(f"  {'Standard':<20s} {ppl_std:>8.2f} {'—':>10s} Black box baseline")
    
    delta_trans = 100 * (ppl_trans - ppl_std) / ppl_std
    print(f"  {'Transparent':<20s} {ppl_trans:>8.2f} {delta_trans:>+9.2f}% Same math, visible routing")
    
    delta_bypass = 100 * (ppl_bypass - ppl_std) / ppl_std
    print(f"  {'Bypass (L7-11)':<20s} {ppl_bypass:>8.2f} {delta_bypass:>+9.2f}% Zero MLP at consensus")
    
    # Compute bypass savings
    total_tokens = sum(l['total'] for l in bypass_logs.values())
    bypassed = sum(l['consensus'] for l in bypass_logs.values())
    print(f"\n  MLP computations skipped: {bypassed}/{total_tokens} ({100*bypassed/total_tokens:.1f}%)")
    print("=" * 65)


if __name__ == '__main__':
    import sys
    token_path = sys.argv[1] if len(sys.argv) > 1 else '/data/tokens/wikitext103_train_tokens.npy'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_comparison(token_path, device=device)
