"""
Knowledge Neurons Evaluation: Empirical test of Dai et al. (2022) claims
against our routing/consensus framework.

Three experiments:
1. INTEGRATED GRADIENTS OVERLAP — Run Dai's attribution method, check how
   many "knowledge neurons" are actually our consensus/routing neurons
2. KNOCKOUT TEST — Zero out (a) Dai's neurons, (b) our consensus neurons,
   (c) attention heads → compare fact recall damage
3. TRANSPLANT TEST — Activate Dai's neurons for Fact A in Fact B's context
   → does the fact transplant cleanly or just disrupt routing?

Uses GPT-2 Small. All experiments at L11 (our primary decision layer)
with optional extension to L10.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# === ARCHITECTURE MAP (from Transparent GPT-2) ===
CONSENSUS_L11 = [2, 2361, 2460, 2928, 1831, 1245, 2600]
EXCEPTION_CORE = [2123, 2910, 740, 1611, 2044]
EXCEPTION_DIFF = [2462, 2173, 1602, 1800, 2379, 1715, 611, 3066, 584, 2378]
EXCEPTION_SPEC = [2921, 2709, 971, 2679, 737]
ALL_ROUTING_L11 = set(CONSENSUS_L11 + EXCEPTION_CORE + EXCEPTION_DIFF + EXCEPTION_SPEC)
RESIDUAL_L11 = [i for i in range(3072) if i not in ALL_ROUTING_L11]

CONSENSUS_L10 = [1486, 1109, 928]

# === FACTUAL PROMPTS ===
FACTUAL_PROMPTS = [
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Spain is", " Madrid"),
    ("The capital of China is", " Beijing"),
    ("The capital of Russia is", " Moscow"),
    ("The capital of Brazil is", " Bras"),  # Brasília
    ("The capital of Australia is", " Canberra"),
    ("The capital of Canada is", " Ottawa"),
    ("The largest planet in the solar system is", " Jupiter"),
    ("Water freezes at", " 0"),
    ("The speed of light is approximately", " 300"),
    ("In 1969, humans first landed on the", " Moon"),
    ("The chemical symbol for gold is", " Au"),
    ("Albert Einstein developed the theory of", " relat"),
    ("The Great Wall is located in", " China"),
    ("The currency of Japan is the", " yen"),
    ("Shakespeare wrote", " Hamlet"),
    ("The Mona Lisa was painted by", " Leonardo"),
]


def setup_model(device=None):
    """Load GPT-2 Small."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    return model, tokenizer, device


def get_target_prob(model, tokenizer, prompt, target, device):
    """Get probability of target token given prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    target_id = tokenizer.encode(target)[0]
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
        probs = F.softmax(logits, dim=-1)
    return probs[target_id].item()


def get_top_prediction(model, tokenizer, prompt, device):
    """Get top-1 predicted token and its probability."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
        probs = F.softmax(logits, dim=-1)
    top_id = torch.argmax(probs).item()
    return tokenizer.decode([top_id]), probs[top_id].item()


# =====================================================
# EXPERIMENT 1: Integrated Gradients Attribution
# =====================================================

def integrated_gradients_mlp(model, tokenizer, prompt, target, device,
                              layer=11, steps=50):
    """
    Compute integrated gradients attribution for each MLP neuron at
    the specified layer, following Dai et al.'s approach.

    Returns: attribution scores for all 3072 neurons.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    target_id = tokenizer.encode(target)[0]
    seq_len = input_ids.shape[1]

    # We need the intermediate MLP activations (after GELU, before W_out)
    # Hook into the MLP's activation function output
    mlp = model.transformer.h[layer].mlp

    # Get baseline (zero) and actual activations
    activations = {}
    def hook_fn(module, input, output):
        activations['value'] = output

    handle = mlp.act.register_forward_hook(hook_fn)

    # Forward pass to get actual activations
    model.zero_grad()
    with torch.no_grad():
        _ = model(input_ids)
    actual_act = activations['value'].detach().clone()  # [1, seq_len, 3072]
    handle.remove()

    # Baseline: zero activations
    baseline_act = torch.zeros_like(actual_act)

    # Integrated gradients: interpolate between baseline and actual
    attributions = torch.zeros(3072, device=device)

    for step in range(steps):
        alpha = step / steps
        # Interpolated activation
        interp_act = baseline_act + alpha * (actual_act - baseline_act)
        interp_act.requires_grad_(True)

        # Hook to replace activations with interpolated values
        def replace_hook(module, input, output, interp=interp_act):
            return interp

        handle = mlp.act.register_forward_hook(replace_hook)

        logits = model(input_ids).logits[0, -1]
        target_logit = logits[target_id]
        target_logit.backward()

        if interp_act.grad is not None:
            # Attribution at the last token position
            attributions += interp_act.grad[0, -1].detach()

        model.zero_grad()
        handle.remove()

    # Scale by (actual - baseline) / steps
    attributions = attributions * (actual_act[0, -1] - baseline_act[0, -1]).detach() / steps

    return attributions.cpu().numpy()


def experiment1_overlap(model, tokenizer, device, top_k=20):
    """
    For each factual prompt, find top-K "knowledge neurons" via integrated
    gradients and check overlap with our routing/consensus neurons.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Integrated Gradients ↔ Routing Neuron Overlap")
    print("=" * 70)

    results = []

    for prompt, target in FACTUAL_PROMPTS:
        print(f"\n  {prompt} → {target}")

        attrs = integrated_gradients_mlp(model, tokenizer, prompt, target,
                                          device, layer=11)
        # Top-K by absolute attribution
        top_neurons = np.argsort(np.abs(attrs))[-top_k:][::-1]
        top_set = set(top_neurons.tolist())

        # Check overlaps
        consensus_overlap = top_set & set(CONSENSUS_L11)
        routing_overlap = top_set & ALL_ROUTING_L11
        residual_in_top = top_set - ALL_ROUTING_L11

        result = {
            'prompt': prompt,
            'target': target,
            'top_neurons': top_neurons.tolist(),
            'top_attrs': attrs[top_neurons].tolist(),
            'consensus_overlap': list(consensus_overlap),
            'routing_overlap': list(routing_overlap),
            'n_consensus_in_top': len(consensus_overlap),
            'n_routing_in_top': len(routing_overlap),
            'n_residual_in_top': len(residual_in_top),
        }
        results.append(result)

        # Expected by chance: routing = 30/3072 ≈ 1% of neurons
        # In top-20: expect ~0.2 by chance
        print(f"    Consensus in top-{top_k}: {len(consensus_overlap)}/7 "
              f"({sorted(consensus_overlap)})")
        print(f"    Routing in top-{top_k}: {len(routing_overlap)}/30 "
              f"({sorted(routing_overlap)})")
        print(f"    Residual in top-{top_k}: {len(residual_in_top)}")

    # Summary
    avg_consensus = np.mean([r['n_consensus_in_top'] for r in results])
    avg_routing = np.mean([r['n_routing_in_top'] for r in results])
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: Avg consensus in top-{top_k}: {avg_consensus:.1f}/7")
    print(f"         Avg routing in top-{top_k}: {avg_routing:.1f}/30")
    print(f"         Expected by chance (routing): "
          f"{top_k * len(ALL_ROUTING_L11) / 3072:.1f}")
    print(f"{'=' * 70}")

    return results


# =====================================================
# EXPERIMENT 2: Knockout Test
# =====================================================

def knockout_neurons(model, layer, neuron_ids):
    """Zero out specific neurons' output weights in MLP at given layer."""
    with torch.no_grad():
        for n_id in neuron_ids:
            model.transformer.h[layer].mlp.c_proj.weight[n_id, :] = 0.0


def knockout_attention_heads(model, layer, head_ids=None):
    """Zero out attention heads' output projection at given layer."""
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    if head_ids is None:
        head_ids = list(range(n_head))  # all heads
    with torch.no_grad():
        for h_id in head_ids:
            start = h_id * head_dim
            end = start + head_dim
            model.transformer.h[layer].attn.c_proj.weight[start:end, :] = 0.0


def experiment2_knockout(model, tokenizer, device):
    """
    Compare fact recall under four knockout conditions:
    (a) Dai's top knowledge neurons (from Exp 1)
    (b) Our consensus neurons
    (c) Our full routing circuit
    (d) Attention heads at L11
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Knockout Test — What kills fact recall?")
    print("=" * 70)

    import copy

    # First, get baseline fact probs and Dai's neurons
    baselines = {}
    dai_neurons_per_prompt = {}

    for prompt, target in FACTUAL_PROMPTS:
        prob = get_target_prob(model, tokenizer, prompt, target, device)
        baselines[(prompt, target)] = prob

        # Get Dai's top-20 knowledge neurons for this prompt
        attrs = integrated_gradients_mlp(model, tokenizer, prompt, target,
                                          device, layer=11)
        dai_neurons_per_prompt[(prompt, target)] = \
            np.argsort(np.abs(attrs))[-20:][::-1].tolist()

    conditions = {
        'dai_neurons': None,  # per-prompt, set below
        'consensus_7': CONSENSUS_L11,
        'all_routing_30': list(ALL_ROUTING_L11),
        'attention_heads': 'attention',  # special handling
    }

    results = {}

    for cond_name in conditions:
        print(f"\n  Condition: {cond_name}")
        cond_results = []

        for prompt, target in FACTUAL_PROMPTS:
            # Fresh model copy for each knockout
            ko_model = copy.deepcopy(model)

            if cond_name == 'dai_neurons':
                neurons = dai_neurons_per_prompt[(prompt, target)]
                knockout_neurons(ko_model, 11, neurons)
            elif cond_name == 'attention_heads':
                knockout_attention_heads(ko_model, 11)
            else:
                neurons = conditions[cond_name]
                knockout_neurons(ko_model, 11, neurons)

            prob = get_target_prob(ko_model, tokenizer, prompt, target, device)
            baseline = baselines[(prompt, target)]
            drop = baseline - prob

            cond_results.append({
                'prompt': prompt,
                'target': target,
                'baseline_prob': baseline,
                'knockout_prob': prob,
                'prob_drop': drop,
                'relative_drop': drop / max(baseline, 1e-8),
            })

            del ko_model

        results[cond_name] = cond_results

        avg_drop = np.mean([r['prob_drop'] for r in cond_results])
        avg_rel = np.mean([r['relative_drop'] for r in cond_results])
        print(f"    Avg prob drop: {avg_drop:.4f} "
              f"(relative: {avg_rel:.1%})")

    print(f"\n{'=' * 70}")
    print("SUMMARY:")
    for cond_name, cond_results in results.items():
        avg_drop = np.mean([r['prob_drop'] for r in cond_results])
        avg_rel = np.mean([r['relative_drop'] for r in cond_results])
        n_neurons = (20 if cond_name == 'dai_neurons'
                     else 7 if cond_name == 'consensus_7'
                     else 30 if cond_name == 'all_routing_30'
                     else 12)  # attention heads
        print(f"  {cond_name:20s}: drop={avg_drop:.4f} "
              f"({avg_rel:.1%}) [{n_neurons} units]")
    print(f"{'=' * 70}")

    return results


# =====================================================
# EXPERIMENT 3: Transplant Test
# =====================================================

def experiment3_transplant(model, tokenizer, device):
    """
    Activate Dai's "knowledge neurons" for Fact A in Fact B's context.
    If Dai is right: Fact A should appear in Fact B's context.
    If we're right: routing disruption, not clean transplant.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Transplant Test — Storage vs. Routing")
    print("=" * 70)

    # Fact pairs: transplant A's neurons into B's context
    transplant_pairs = [
        # (source_prompt, source_target, dest_prompt, dest_target)
        ("The capital of France is", " Paris",
         "The capital of Germany is", " Berlin"),
        ("The capital of Japan is", " Tokyo",
         "The capital of Italy is", " Rome"),
        ("The capital of China is", " Beijing",
         "The capital of Spain is", " Madrid"),
        ("The capital of Russia is", " Moscow",
         "The capital of Canada is", " Ottawa"),
        ("The capital of Australia is", " Canberra",
         "The capital of Brazil is", " Bras"),
    ]

    results = []

    for src_prompt, src_target, dst_prompt, dst_target in transplant_pairs:
        print(f"\n  Transplant: '{src_prompt}→{src_target}' neurons "
              f"into '{dst_prompt}'")

        # Get source fact's knowledge neurons + activations
        src_ids = tokenizer.encode(src_prompt, return_tensors='pt').to(device)
        src_target_id = tokenizer.encode(src_target)[0]
        dst_target_id = tokenizer.encode(dst_target)[0]

        attrs = integrated_gradients_mlp(model, tokenizer, src_prompt,
                                          src_target, device, layer=11)
        dai_neurons = np.argsort(np.abs(attrs))[-20:][::-1].tolist()

        # Capture source activations at those neurons
        mlp = model.transformer.h[11].mlp
        src_activations = {}

        def capture_hook(module, input, output):
            src_activations['act'] = output.detach().clone()

        handle = mlp.act.register_forward_hook(capture_hook)
        with torch.no_grad():
            model(src_ids)
        src_act = src_activations['act'][0, -1].clone()  # [3072]
        handle.remove()

        # Now run dest prompt, replacing those neurons' activations
        dst_ids = tokenizer.encode(dst_prompt, return_tensors='pt').to(device)

        # Baseline dest probs
        with torch.no_grad():
            dst_logits = model(dst_ids).logits[0, -1]
            dst_probs = F.softmax(dst_logits, dim=-1)
        baseline_src_prob = dst_probs[src_target_id].item()
        baseline_dst_prob = dst_probs[dst_target_id].item()
        baseline_top, baseline_top_prob = get_top_prediction(
            model, tokenizer, dst_prompt, device)

        # Transplant: replace activations at Dai's neurons
        def transplant_hook(module, input, output,
                           neurons=dai_neurons, source=src_act):
            modified = output.clone()
            for n_id in neurons:
                modified[0, -1, n_id] = source[n_id]
            return modified

        handle = mlp.act.register_forward_hook(transplant_hook)
        with torch.no_grad():
            mod_logits = model(dst_ids).logits[0, -1]
            mod_probs = F.softmax(mod_logits, dim=-1)
        transplant_src_prob = mod_probs[src_target_id].item()
        transplant_dst_prob = mod_probs[dst_target_id].item()
        transplant_top_id = torch.argmax(mod_probs).item()
        transplant_top = tokenizer.decode([transplant_top_id])
        transplant_top_prob = mod_probs[transplant_top_id].item()
        handle.remove()

        # How many of Dai's neurons are routing neurons?
        dai_routing = set(dai_neurons) & ALL_ROUTING_L11
        dai_residual = set(dai_neurons) - ALL_ROUTING_L11

        result = {
            'src_prompt': src_prompt, 'src_target': src_target,
            'dst_prompt': dst_prompt, 'dst_target': dst_target,
            'dai_neurons': dai_neurons,
            'dai_routing_count': len(dai_routing),
            'dai_residual_count': len(dai_residual),
            'baseline_src_prob': baseline_src_prob,
            'baseline_dst_prob': baseline_dst_prob,
            'baseline_top': baseline_top,
            'transplant_src_prob': transplant_src_prob,
            'transplant_dst_prob': transplant_dst_prob,
            'transplant_top': transplant_top,
            'src_prob_change': transplant_src_prob - baseline_src_prob,
            'dst_prob_change': transplant_dst_prob - baseline_dst_prob,
            'clean_transplant': (transplant_top.strip() ==
                                  src_target.strip()),
        }
        results.append(result)

        print(f"    Dai neurons: {len(dai_routing)} routing, "
              f"{len(dai_residual)} residual")
        print(f"    Before transplant: P({src_target.strip()})="
              f"{baseline_src_prob:.4f}, "
              f"P({dst_target.strip()})={baseline_dst_prob:.4f}, "
              f"top='{baseline_top.strip()}'")
        print(f"    After transplant:  P({src_target.strip()})="
              f"{transplant_src_prob:.4f}, "
              f"P({dst_target.strip()})={transplant_dst_prob:.4f}, "
              f"top='{transplant_top.strip()}'")
        if result['clean_transplant']:
            print(f"    ✅ CLEAN TRANSPLANT — supports Dai (storage)")
        else:
            print(f"    ❌ NO CLEAN TRANSPLANT — supports routing hypothesis")

    # Summary
    n_clean = sum(1 for r in results if r['clean_transplant'])
    avg_src_boost = np.mean([r['src_prob_change'] for r in results])
    avg_dst_drop = np.mean([r['dst_prob_change'] for r in results])
    print(f"\n{'=' * 70}")
    print(f"SUMMARY:")
    print(f"  Clean transplants: {n_clean}/{len(results)}")
    print(f"  Avg source fact prob change: {avg_src_boost:+.4f}")
    print(f"  Avg dest fact prob change: {avg_dst_drop:+.4f}")
    if n_clean <= 1:
        print(f"  → SUPPORTS routing hypothesis: neurons gate, "
              f"don't store facts")
    elif n_clean >= 4:
        print(f"  → SUPPORTS Dai: neurons do store factual knowledge")
    else:
        print(f"  → MIXED: partial storage + routing")
    print(f"{'=' * 70}")

    return results


# =====================================================
# MAIN
# =====================================================

def main():
    print("Knowledge Neurons Evaluation")
    print("Dai et al. (2022) vs. Transparent GPT-2 Routing Hypothesis")
    print("=" * 70)

    model, tokenizer, device = setup_model()
    print(f"Device: {device}")
    print(f"Routing neurons: {len(ALL_ROUTING_L11)} "
          f"(7 consensus + 5 core + 10 diff + 5 spec + 3 L10)")
    print(f"Residual neurons: {len(RESIDUAL_L11)}")

    # Run all three experiments
    exp1_results = experiment1_overlap(model, tokenizer, device)
    exp2_results = experiment2_knockout(model, tokenizer, device)
    exp3_results = experiment3_transplant(model, tokenizer, device)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'knowledge_neurons_eval.json')

    all_results = {
        'experiment1_overlap': exp1_results,
        'experiment2_knockout': exp2_results,
        'experiment3_transplant': exp3_results,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
