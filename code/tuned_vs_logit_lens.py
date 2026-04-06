#!/usr/bin/env python3
"""
Compare logit lens vs tuned lens for Table 6 of Darkness Visible.

Belrose et al. (2023) objection: logit lens is unreliable at early layers.
The tuned lens learns an affine probe per layer that corrects for this.

If the developmental arc (scaffold → decision → terminal) holds under tuned
lens, the finding is robust.
"""

import torch
import numpy as np
import json
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tuned_lens import TunedLens

device = 'cpu'
print("Loading model + tuned lens...")
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tuned_lens = TunedLens.from_model_and_pretrained(model).to(device).eval()

ln_f = model.transformer.ln_f
lm_head = model.lm_head

N_SEQUENCES = 200
SEQ_LEN = 1024


def run_comparison():
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    text = '\n\n'.join([t for t in dataset['text'] if t.strip()])
    tokens = tokenizer.encode(text)

    # Per-layer accumulators
    logit_correct = np.zeros(13)  # emb + 12 layers
    tuned_correct = np.zeros(13)
    logit_first_lockin = np.zeros(13)
    tuned_first_lockin = np.zeros(13)
    logit_never = 0
    tuned_never = 0
    total_tokens = 0
    t0 = time.time()

    for seq_idx in range(N_SEQUENCES):
        start = seq_idx * SEQ_LEN
        if start + SEQ_LEN + 1 > len(tokens):
            break

        input_ids = torch.tensor([tokens[start:start + SEQ_LEN]], device=device)
        target_ids = torch.tensor(tokens[start + 1:start + SEQ_LEN + 1], device=device)

        # Collect all hidden states in one pass
        hidden_states = []

        hooks = []
        def make_hook(storage):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    storage.append(output[0].detach())
                else:
                    storage.append(output.detach())
            return hook

        hooks.append(model.transformer.drop.register_forward_hook(make_hook(hidden_states)))
        for i in range(12):
            hooks.append(model.transformer.h[i].register_forward_hook(make_hook(hidden_states)))

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        # Now batch-compute predictions for all layers at once
        # Each hidden_states[i] is (1, seq_len, 768)
        n_tokens = SEQ_LEN - 1  # skip last (no target)

        with torch.no_grad():
            for layer_idx in range(13):
                h = hidden_states[layer_idx][0, :n_tokens]  # (n_tokens, 768)

                # Logit lens: final LN + unembed
                logit_logits = lm_head(ln_f(h))  # (n_tokens, vocab)
                logit_preds = logit_logits.argmax(dim=-1)  # (n_tokens,)
                logit_hits = (logit_preds == target_ids[:n_tokens])

                # Tuned lens
                if layer_idx < 12:
                    tuned_logits = tuned_lens(h, layer_idx)  # (n_tokens, vocab)
                    tuned_preds = tuned_logits.argmax(dim=-1)
                else:
                    tuned_preds = logit_preds  # identity at final layer

                tuned_hits = (tuned_preds == target_ids[:n_tokens])

                logit_correct[layer_idx] += logit_hits.sum().item()
                tuned_correct[layer_idx] += tuned_hits.sum().item()

                # Track first lock-in (need per-token tracking)
                if layer_idx == 0:
                    # Initialize per-token trackers for this sequence
                    logit_first = np.full(n_tokens, -1, dtype=int)
                    tuned_first = np.full(n_tokens, -1, dtype=int)

                logit_mask = logit_hits.cpu().numpy()
                tuned_mask = tuned_hits.cpu().numpy()

                for j in range(n_tokens):
                    if logit_mask[j] and logit_first[j] == -1:
                        logit_first[j] = layer_idx
                    if tuned_mask[j] and tuned_first[j] == -1:
                        tuned_first[j] = layer_idx

            # Accumulate first lock-in stats
            for j in range(n_tokens):
                if logit_first[j] >= 0:
                    logit_first_lockin[logit_first[j]] += 1
                else:
                    logit_never += 1
                if tuned_first[j] >= 0:
                    tuned_first_lockin[tuned_first[j]] += 1
                else:
                    tuned_never += 1

            total_tokens += n_tokens

        if (seq_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {seq_idx+1}/{N_SEQUENCES} sequences ({total_tokens} tokens, {elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {total_tokens} tokens in {elapsed:.0f}s")

    # Compute percentages
    logit_pct = logit_correct / total_tokens * 100
    tuned_pct = tuned_correct / total_tokens * 100
    logit_lk_pct = logit_first_lockin / total_tokens * 100
    tuned_lk_pct = tuned_first_lockin / total_tokens * 100
    logit_never_pct = logit_never / total_tokens * 100
    tuned_never_pct = tuned_never / total_tokens * 100

    # Print table
    layer_names = ['Emb'] + [f'L{i}' for i in range(12)]
    print(f"\n{'Layer':<6} {'Logit%':>8} {'Tuned%':>8} {'Δ':>8} {'Logit Lock%':>12} {'Tuned Lock%':>12}")
    print('-' * 56)
    for i, name in enumerate(layer_names):
        d = tuned_pct[i] - logit_pct[i]
        print(f"{name:<6} {logit_pct[i]:>8.1f} {tuned_pct[i]:>8.1f} {d:>+8.1f} {logit_lk_pct[i]:>12.1f} {tuned_lk_pct[i]:>12.1f}")
    print(f"{'Never':<6} {'':>8} {'':>8} {'':>8} {logit_never_pct:>12.1f} {tuned_never_pct:>12.1f}")

    # Developmental arc check
    scaffold_gain_logit = (logit_pct[6] - logit_pct[1]) / 5  # L0→L5
    scaffold_gain_tuned = (tuned_pct[6] - tuned_pct[1]) / 5
    decision_gain_logit = (logit_pct[9] - logit_pct[7]) / 2  # L7→L9
    decision_gain_tuned = (tuned_pct[9] - tuned_pct[7]) / 2

    print(f"\nDevelopmental arc:")
    print(f"  Scaffold gain/layer:  logit {scaffold_gain_logit:+.2f}pp  tuned {scaffold_gain_tuned:+.2f}pp")
    print(f"  Decision gain/layer:  logit {decision_gain_logit:+.2f}pp  tuned {decision_gain_tuned:+.2f}pp")

    arc_holds = decision_gain_tuned > scaffold_gain_tuned
    print(f"  Arc holds under tuned lens: {'YES ✓' if arc_holds else 'NO ✗'}")

    early_boost = np.mean(tuned_pct[1:5]) - np.mean(logit_pct[1:5])
    late_boost = np.mean(tuned_pct[8:13]) - np.mean(logit_pct[8:13])
    print(f"\n  Early (L0-L3) tuned boost: {early_boost:+.1f}pp")
    print(f"  Late (L7-L11) tuned boost: {late_boost:+.1f}pp")

    # Save
    results = {
        'total_tokens': total_tokens,
        'logit_top1_pct': logit_pct.tolist(),
        'tuned_top1_pct': tuned_pct.tolist(),
        'logit_lockin_pct': logit_lk_pct.tolist(),
        'tuned_lockin_pct': tuned_lk_pct.tolist(),
        'logit_never_pct': float(logit_never_pct),
        'tuned_never_pct': float(tuned_never_pct),
        'scaffold_gain_logit': float(scaffold_gain_logit),
        'scaffold_gain_tuned': float(scaffold_gain_tuned),
        'decision_gain_logit': float(decision_gain_logit),
        'decision_gain_tuned': float(decision_gain_tuned),
        'arc_holds': bool(arc_holds),
        'early_boost_pp': float(early_boost),
        'late_boost_pp': float(late_boost),
    }
    outpath = '../data/tuned_vs_logit_lens.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    run_comparison()
