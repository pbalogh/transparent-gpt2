#!/usr/bin/env python3
"""
Garden-Path Surprisal Test: Does N2123 fire at disambiguation points?

Hypothesis (from MEMORY.md "Garden-Path Recovery" paper idea):
When GPT-2 encounters a garden-path sentence like:
  "After the dog struggled the vet took off the muzzle"
the model initially treats "the vet" as the object of "struggled" (late closure),
then must reparse at "took" (the disambiguation point).

At that disambiguation point, we should see:
  - Higher surprisal (lower P(correct next token))
  - Lower consensus level (fewer of 7 consensus neurons firing)
  - Higher N2123 (exception indicator) activation
  - Larger MLP delta norm (more nonlinear intervention needed)

Control: transitive-verb versions where "the vet" IS a valid object:
  "After the dog scratched the vet took off the muzzle"

Materials based on Van Gompel & Pickering (2001), "Lexical guidance in
sentence processing," which replicated Mitchell (1987) garden-path effects
using eye-tracking with intransitive vs transitive verbs.

Output: per-token trace of surprisal, consensus, N2123 activation, MLP delta
for each sentence pair, with visualization.
"""

import torch
import numpy as np
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import functional as F

device = 'cpu'
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

CONSENSUS_NEURONS = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION_NEURON = 2123
FIRE_THRESHOLD = 0.1

# ── Stimulus materials ────────────────────────────────────────────────────
# Format: (intransitive_sentence, transitive_sentence, disambig_region)
# disambig_region: the word(s) where reparse should happen
# Based on Van Gompel & Pickering (2001) / Mitchell (1987) style materials

STIMULI = [
    # Original Van Gompel & Pickering materials (adapted)
    {
        'id': 'vgp1',
        'intransitive': "After the dog struggled the vet took off the muzzle.",
        'transitive': "After the dog scratched the vet took off the muzzle.",
        'disambig_word': 'took',
        'np_region': 'the vet',
        'label': 'struggled/scratched'
    },
    {
        'id': 'vgp2',
        'intransitive': "After the child sneezed the doctor prescribed a course of medicine.",
        'transitive': "After the child visited the doctor prescribed a course of medicine.",
        'disambig_word': 'prescribed',
        'np_region': 'the doctor',
        'label': 'sneezed/visited'
    },
    {
        'id': 'vgp3',
        'intransitive': "While the man dozed the woman next to him read her book quietly.",
        'transitive': "While the man watched the woman next to him read her book quietly.",
        'disambig_word': 'next',
        'np_region': 'the woman',
        'label': 'dozed/watched'
    },
    {
        'id': 'vgp4',
        'intransitive': "After the prisoner escaped the guard searched the entire building.",
        'transitive': "After the prisoner attacked the guard searched the entire building.",
        'disambig_word': 'searched',
        'np_region': 'the guard',
        'label': 'escaped/attacked'
    },
    {
        'id': 'vgp5',
        'intransitive': "While the student slept the professor standing nearby noticed the textbook.",
        'transitive': "While the student ignored the professor standing nearby noticed the textbook.",
        'disambig_word': 'standing',
        'np_region': 'the professor',
        'label': 'slept/ignored'
    },
    # Additional items to increase power
    {
        'id': 'add1',
        'intransitive': "After the baby cried the mother in the next room came running.",
        'transitive': "After the baby woke the mother in the next room came running.",
        'disambig_word': 'in',
        'np_region': 'the mother',
        'label': 'cried/woke'
    },
    {
        'id': 'add2',
        'intransitive': "While the cat purred the owner sitting nearby stroked its fur.",
        'transitive': "While the cat bit the owner sitting nearby stroked its fur.",
        'disambig_word': 'sitting',
        'np_region': 'the owner',
        'label': 'purred/bit'
    },
    {
        'id': 'add3',
        'intransitive': "After the patient fainted the nurse on duty called for assistance.",
        'transitive': "After the patient alarmed the nurse on duty called for assistance.",
        'disambig_word': 'on',
        'np_region': 'the nurse',
        'label': 'fainted/alarmed'
    },
    {
        'id': 'add4',
        'intransitive': "While the horse galloped the rider on its back held the reins tightly.",
        'transitive': "While the horse threw the rider on its back held the reins tightly.",
        'disambig_word': 'on',
        'np_region': 'the rider',
        'label': 'galloped/threw'
    },
    {
        'id': 'add5',
        'intransitive': "After the volcano erupted the residents of the town evacuated immediately.",
        'transitive': "After the volcano destroyed the residents of the town evacuated immediately.",
        'disambig_word': 'of',
        'np_region': 'the residents',
        'label': 'erupted/destroyed'
    },
    {
        'id': 'add6',
        'intransitive': "While the engine sputtered the mechanic under the hood checked the spark plugs.",
        'transitive': "While the engine startled the mechanic under the hood checked the spark plugs.",
        'disambig_word': 'under',
        'np_region': 'the mechanic',
        'label': 'sputtered/startled'
    },
    {
        'id': 'add7',
        'intransitive': "After the witness lied the lawyer at the front of the courtroom objected.",
        'transitive': "After the witness accused the lawyer at the front of the courtroom objected.",
        'disambig_word': 'at',
        'np_region': 'the lawyer',
        'label': 'lied/accused'
    },
    {
        'id': 'add8',
        'intransitive': "While the singer performed the audience in the theater applauded loudly.",
        'transitive': "While the singer impressed the audience in the theater applauded loudly.",
        'disambig_word': 'in',
        'np_region': 'the audience',
        'label': 'performed/impressed'
    },
    {
        'id': 'add9',
        'intransitive': "After the soldier marched the commander at the base issued new orders.",
        'transitive': "After the soldier disobeyed the commander at the base issued new orders.",
        'disambig_word': 'at',
        'np_region': 'the commander',
        'label': 'marched/disobeyed'
    },
    {
        'id': 'add10',
        'intransitive': "While the fire blazed the firefighter near the entrance rescued the child.",
        'transitive': "While the fire trapped the firefighter near the entrance rescued the child.",
        'disambig_word': 'near',
        'np_region': 'the firefighter',
        'label': 'blazed/trapped'
    },
]


def get_token_trace(text):
    """
    Run text through GPT-2 and return per-token metrics:
    - token: the actual token string
    - surprisal: -log2(P(token | context))
    - consensus: how many of 7 consensus neurons fire (0-7)
    - n2123_act: absolute GELU activation of N2123
    - n2123_fires: whether |GELU(N2123)| > threshold
    - mlp_delta_norm: L2 norm of L11 MLP output
    """
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], device=device)
    
    # Hooks to capture L11 MLP activations
    l11_post_gelu = {}
    l11_mlp_output = {}
    
    def hook_post_gelu(module, input, output):
        # GPT-2's MLP: c_fc → GELU → c_proj
        # After GELU, before c_proj
        l11_post_gelu['act'] = output.detach()
    
    def hook_mlp_output(module, input, output):
        l11_mlp_output['out'] = output.detach()
    
    # Register hooks on L11 MLP
    h1 = model.transformer.h[11].mlp.act.register_forward_hook(hook_post_gelu)
    h2 = model.transformer.h[11].mlp.register_forward_hook(hook_mlp_output)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
    
    h1.remove()
    h2.remove()
    
    post_gelu = l11_post_gelu['act'][0]  # (seq_len, 3072)
    mlp_out = l11_mlp_output['out'][0]   # (seq_len, 768)
    
    results = []
    for i in range(len(tokens)):
        token_str = tokenizer.decode([tokens[i]])
        
        # Surprisal: for token i, use logits at position i-1 (predicting token i)
        if i == 0:
            surprisal = 0.0  # no prediction for first token
        else:
            probs = F.softmax(logits[i-1], dim=-1)
            p = probs[tokens[i]].item()
            surprisal = -np.log2(max(p, 1e-10))
        
        # Consensus level at this position
        consensus = 0
        for n in CONSENSUS_NEURONS:
            if abs(post_gelu[i, n].item()) > FIRE_THRESHOLD:
                consensus += 1
        
        # N2123 activation
        n2123_act = abs(post_gelu[i, EXCEPTION_NEURON].item())
        n2123_fires = n2123_act > FIRE_THRESHOLD
        
        # MLP delta norm
        delta_norm = torch.norm(mlp_out[i]).item()
        
        results.append({
            'position': i,
            'token': token_str,
            'token_id': tokens[i],
            'surprisal': round(surprisal, 4),
            'consensus': consensus,
            'n2123_act': round(n2123_act, 4),
            'n2123_fires': n2123_fires,
            'mlp_delta_norm': round(delta_norm, 4),
        })
    
    return results


def find_token_position(trace, word):
    """Find the position of a word in the token trace. 
    Handles BPE tokenization — looks for the first token that starts the word."""
    # Build the running text to find where the word starts
    text_so_far = ""
    for i, t in enumerate(trace):
        text_so_far += t['token']
        # Check if the word just appeared
        if word in text_so_far and word not in text_so_far[:-len(t['token'])]:
            return i
    return None


def run_experiment():
    """Run all stimuli and compute garden-path effects."""
    print(f"Running garden-path surprisal test ({len(STIMULI)} sentence pairs)")
    print(f"Consensus neurons: {CONSENSUS_NEURONS}")
    print(f"Exception neuron: N{EXCEPTION_NEURON}")
    print(f"Fire threshold: {FIRE_THRESHOLD}")
    print()
    
    all_results = []
    
    # Aggregate metrics at disambiguation point
    intransitive_surprisals = []
    transitive_surprisals = []
    intransitive_consensus = []
    transitive_consensus = []
    intransitive_n2123 = []
    transitive_n2123 = []
    intransitive_delta = []
    transitive_delta = []
    
    for stim in STIMULI:
        print(f"── {stim['label']} ──")
        
        # Run both versions
        trace_intr = get_token_trace(stim['intransitive'])
        trace_tran = get_token_trace(stim['transitive'])
        
        # Find disambiguation point
        disambig = stim['disambig_word']
        pos_intr = find_token_position(trace_intr, disambig)
        pos_tran = find_token_position(trace_tran, disambig)
        
        if pos_intr is None or pos_tran is None:
            print(f"  ⚠ Could not find '{disambig}' in traces, skipping")
            continue
        
        # Get metrics at disambiguation point
        d_intr = trace_intr[pos_intr]
        d_tran = trace_tran[pos_tran]
        
        # Also get metrics at the NP region (the ambiguous NP itself)
        np_word = stim['np_region'].split()[-1]  # Last word of NP (e.g., "vet")
        np_pos_intr = find_token_position(trace_intr, np_word)
        np_pos_tran = find_token_position(trace_tran, np_word)
        
        print(f"  Intransitive: {stim['intransitive'][:60]}...")
        print(f"    Disambig '{disambig}' at pos {pos_intr}: "
              f"surprisal={d_intr['surprisal']:.2f} bits, "
              f"consensus={d_intr['consensus']}/7, "
              f"N2123={'FIRES' if d_intr['n2123_fires'] else 'silent'} ({d_intr['n2123_act']:.3f}), "
              f"Δnorm={d_intr['mlp_delta_norm']:.2f}")
        
        print(f"  Transitive:   {stim['transitive'][:60]}...")
        print(f"    Disambig '{disambig}' at pos {pos_tran}: "
              f"surprisal={d_tran['surprisal']:.2f} bits, "
              f"consensus={d_tran['consensus']}/7, "
              f"N2123={'FIRES' if d_tran['n2123_fires'] else 'silent'} ({d_tran['n2123_act']:.3f}), "
              f"Δnorm={d_tran['mlp_delta_norm']:.2f}")
        
        # Compute effects
        surprisal_effect = d_intr['surprisal'] - d_tran['surprisal']
        consensus_effect = d_intr['consensus'] - d_tran['consensus']
        n2123_effect = d_intr['n2123_act'] - d_tran['n2123_act']
        delta_effect = d_intr['mlp_delta_norm'] - d_tran['mlp_delta_norm']
        
        print(f"  EFFECT: Δsurprisal={surprisal_effect:+.2f} bits, "
              f"Δconsensus={consensus_effect:+d}, "
              f"ΔN2123={n2123_effect:+.3f}, "
              f"ΔΔnorm={delta_effect:+.2f}")
        
        # Direction check
        directions = []
        if surprisal_effect > 0: directions.append("✓ surprisal↑")
        else: directions.append("✗ surprisal↓")
        if consensus_effect < 0: directions.append("✓ consensus↓")
        else: directions.append("✗ consensus≥")
        if n2123_effect > 0: directions.append("✓ N2123↑")
        else: directions.append("✗ N2123↓")
        print(f"  Predictions: {', '.join(directions)}")
        print()
        
        # Collect for aggregation
        intransitive_surprisals.append(d_intr['surprisal'])
        transitive_surprisals.append(d_tran['surprisal'])
        intransitive_consensus.append(d_intr['consensus'])
        transitive_consensus.append(d_tran['consensus'])
        intransitive_n2123.append(d_intr['n2123_act'])
        transitive_n2123.append(d_tran['n2123_act'])
        intransitive_delta.append(d_intr['mlp_delta_norm'])
        transitive_delta.append(d_tran['mlp_delta_norm'])
        
        # Store full results
        all_results.append({
            'id': stim['id'],
            'label': stim['label'],
            'intransitive': stim['intransitive'],
            'transitive': stim['transitive'],
            'disambig_word': disambig,
            'trace_intransitive': trace_intr,
            'trace_transitive': trace_tran,
            'disambig_pos_intr': pos_intr,
            'disambig_pos_tran': pos_tran,
            'np_pos_intr': np_pos_intr,
            'np_pos_tran': np_pos_tran,
            'effect': {
                'surprisal': round(surprisal_effect, 4),
                'consensus': consensus_effect,
                'n2123': round(n2123_effect, 4),
                'delta_norm': round(delta_effect, 4),
            }
        })
    
    # ── Aggregate results ──
    n = len(intransitive_surprisals)
    print("=" * 70)
    print(f"AGGREGATE RESULTS ({n} sentence pairs)")
    print("=" * 70)
    
    mean_surp_intr = np.mean(intransitive_surprisals)
    mean_surp_tran = np.mean(transitive_surprisals)
    mean_cons_intr = np.mean(intransitive_consensus)
    mean_cons_tran = np.mean(transitive_consensus)
    mean_n2123_intr = np.mean(intransitive_n2123)
    mean_n2123_tran = np.mean(transitive_n2123)
    mean_delta_intr = np.mean(intransitive_delta)
    mean_delta_tran = np.mean(transitive_delta)
    
    print(f"\nAt disambiguation point:")
    print(f"  {'Metric':<20} {'Intransitive':>14} {'Transitive':>14} {'Effect':>14} {'Direction':>10}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
    print(f"  {'Surprisal (bits)':<20} {mean_surp_intr:>14.2f} {mean_surp_tran:>14.2f} {mean_surp_intr-mean_surp_tran:>+14.2f} {'✓' if mean_surp_intr > mean_surp_tran else '✗':>10}")
    print(f"  {'Consensus (0-7)':<20} {mean_cons_intr:>14.2f} {mean_cons_tran:>14.2f} {mean_cons_intr-mean_cons_tran:>+14.2f} {'✓' if mean_cons_intr < mean_cons_tran else '✗':>10}")
    print(f"  {'N2123 activation':<20} {mean_n2123_intr:>14.3f} {mean_n2123_tran:>14.3f} {mean_n2123_intr-mean_n2123_tran:>+14.3f} {'✓' if mean_n2123_intr > mean_n2123_tran else '✗':>10}")
    print(f"  {'MLP Δ norm':<20} {mean_delta_intr:>14.2f} {mean_delta_tran:>14.2f} {mean_delta_intr-mean_delta_tran:>+14.2f} {'✓' if mean_delta_intr > mean_delta_tran else '✗':>10}")
    
    # Effect sizes (Cohen's d)
    from scipy import stats
    
    surp_d = (mean_surp_intr - mean_surp_tran) / np.sqrt((np.var(intransitive_surprisals) + np.var(transitive_surprisals)) / 2)
    cons_d = (mean_cons_intr - mean_cons_tran) / np.sqrt((np.var(intransitive_consensus) + np.var(transitive_consensus)) / 2)
    n2123_d = (mean_n2123_intr - mean_n2123_tran) / np.sqrt((np.var(intransitive_n2123) + np.var(transitive_n2123)) / 2)
    delta_d = (mean_delta_intr - mean_delta_tran) / np.sqrt((np.var(intransitive_delta) + np.var(transitive_delta)) / 2)
    
    # Paired t-tests
    t_surp, p_surp = stats.ttest_rel(intransitive_surprisals, transitive_surprisals)
    t_cons, p_cons = stats.ttest_rel(intransitive_consensus, transitive_consensus)
    t_n2123, p_n2123 = stats.ttest_rel(intransitive_n2123, transitive_n2123)
    t_delta, p_delta = stats.ttest_rel(intransitive_delta, transitive_delta)
    
    print(f"\n  {'Effect sizes (Cohens d):'}")
    print(f"  Surprisal: d={surp_d:+.3f}, t({n-1})={t_surp:.3f}, p={p_surp:.4f}")
    print(f"  Consensus: d={cons_d:+.3f}, t({n-1})={t_cons:.3f}, p={p_cons:.4f}")
    print(f"  N2123:     d={n2123_d:+.3f}, t({n-1})={t_n2123:.3f}, p={p_n2123:.4f}")
    print(f"  MLP delta: d={delta_d:+.3f}, t({n-1})={t_delta:.3f}, p={p_delta:.4f}")
    
    # N2123 fire rate comparison
    intr_fires = sum(1 for x in intransitive_n2123 if x > FIRE_THRESHOLD)
    tran_fires = sum(1 for x in transitive_n2123 if x > FIRE_THRESHOLD)
    print(f"\n  N2123 fires at disambig: intransitive {intr_fires}/{n} ({100*intr_fires/n:.0f}%), "
          f"transitive {tran_fires}/{n} ({100*tran_fires/n:.0f}%)")
    
    # Summary verdict
    print(f"\n{'='*70}")
    correct_directions = sum([
        mean_surp_intr > mean_surp_tran,  # higher surprisal for garden path
        mean_cons_intr < mean_cons_tran,   # lower consensus for garden path
        mean_n2123_intr > mean_n2123_tran, # more exception firing for garden path
        mean_delta_intr > mean_delta_tran, # larger MLP intervention for garden path
    ])
    print(f"VERDICT: {correct_directions}/4 predictions in correct direction")
    if correct_directions == 4:
        print("🔥 All four metrics support garden-path effect at L11!")
    elif correct_directions >= 3:
        print("✓ Strong support for garden-path effect at L11")
    elif correct_directions >= 2:
        print("~ Mixed support — some metrics align, some don't")
    else:
        print("✗ Weak or no support for garden-path effect at L11")
    print(f"{'='*70}")
    
    # Save results
    output = {
        'n_pairs': n,
        'stimuli': all_results,
        'aggregate': {
            'surprisal': {
                'intransitive_mean': round(mean_surp_intr, 4),
                'transitive_mean': round(mean_surp_tran, 4),
                'effect': round(mean_surp_intr - mean_surp_tran, 4),
                'cohens_d': round(surp_d, 4),
                't': round(t_surp, 4),
                'p': round(p_surp, 6),
            },
            'consensus': {
                'intransitive_mean': round(mean_cons_intr, 4),
                'transitive_mean': round(mean_cons_tran, 4),
                'effect': round(mean_cons_intr - mean_cons_tran, 4),
                'cohens_d': round(cons_d, 4),
                't': round(t_cons, 4),
                'p': round(p_cons, 6),
            },
            'n2123': {
                'intransitive_mean': round(mean_n2123_intr, 4),
                'transitive_mean': round(mean_n2123_tran, 4),
                'effect': round(mean_n2123_intr - mean_n2123_tran, 4),
                'cohens_d': round(n2123_d, 4),
                't': round(t_n2123, 4),
                'p': round(p_n2123, 6),
                'fire_rate_intransitive': intr_fires / n,
                'fire_rate_transitive': tran_fires / n,
            },
            'mlp_delta': {
                'intransitive_mean': round(mean_delta_intr, 4),
                'transitive_mean': round(mean_delta_tran, 4),
                'effect': round(mean_delta_intr - mean_delta_tran, 4),
                'cohens_d': round(delta_d, 4),
                't': round(t_delta, 4),
                'p': round(p_delta, 6),
            },
            'correct_directions': correct_directions,
        }
    }
    
    outpath = os.path.join(os.path.dirname(__file__), '..', 'data', 'garden_path_surprisal.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, bool): return obj
        return obj
    
    import copy
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        else:
            return convert(obj)
    
    with open(outpath, 'w') as f:
        json.dump(deep_convert(output), f, indent=2)
    print(f"\nResults saved to {outpath}")
    
    return output


if __name__ == '__main__':
    run_experiment()
