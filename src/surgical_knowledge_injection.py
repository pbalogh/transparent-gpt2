"""
Surgical Knowledge Injection: Can we inject a fact by modifying ONLY
residual neurons in decision layers, while preserving routing circuits?

Experiment: Make GPT-2 believe "The capital of France is Berlin"
(a wrong fact, easy to verify)

Three conditions:
1. FULL fine-tune: Update all MLP weights in L10-11 (standard approach)
2. SURGICAL: Update only residual neuron weights in L10-11
3. ROUTING-ONLY: Update only routing circuit weights (should NOT learn the fact)

Measure:
- Target fact accuracy: does "The capital of France is" → "Berlin"?
- Collateral damage: perplexity on unrelated text
- Routing preservation: do consensus/exception fire rates stay the same?
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Architecture map
EXCEPTION_CORE = [2123, 2910, 740, 1611, 2044]
EXCEPTION_DIFF = [2462, 2173, 1602, 1800, 2379, 1715, 611, 3066, 584, 2378]
EXCEPTION_SPEC = [2921, 2709, 971, 2679, 737]
ROUTING_NEURONS = set(EXCEPTION_CORE + EXCEPTION_DIFF + EXCEPTION_SPEC)
# Also include consensus neurons as routing
CONSENSUS_L11 = [2, 2361, 2460, 2928, 1831, 1245, 2600]
CONSENSUS_L10 = [1486, 1109, 928]
ALL_ROUTING = ROUTING_NEURONS | set(CONSENSUS_L11) | set(CONSENSUS_L10)
RESIDUAL_NEURONS = [i for i in range(3072) if i not in ALL_ROUTING]

# Fact to inject
PROMPT = "The capital of France is"
TARGET = " Berlin"  # wrong fact, easy to test
ORIGINAL = " Paris"

# Control prompts (should NOT be affected)
CONTROL_PROMPTS = [
    ("The capital of Germany is", " Berlin"),  # already true
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("Water boils at 100 degrees", " Fahrenheit"),
    ("She walked into the room and", " looked"),
    ("The cat sat on the", " floor"),
    ("Once upon a", " time"),
    ("1, 2, 3, 4,", " 5"),
]


def get_target_prob(model, tokenizer, prompt, target, device):
    """Get the probability of target token given prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    target_id = tokenizer.encode(target)[0]
    
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]  # last position
        probs = F.softmax(logits, dim=-1)
        target_prob = probs[target_id].item()
        
        # Also get rank
        rank = (probs > target_prob).sum().item() + 1
        
        # Top 5
        topk = probs.topk(5)
        top5 = [(tokenizer.decode([i]), round(p, 4)) for i, p in zip(topk.indices.tolist(), topk.values.tolist())]
    
    return target_prob, rank, top5


def eval_perplexity_quick(model, tokenizer, device, n_samples=20):
    """Quick perplexity on some standard text."""
    texts = [
        "The United States of America is a country in North America.",
        "Machine learning is a branch of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog near the river.",
        "In the year 2024, many advances were made in technology.",
        "She picked up the book and started reading the first chapter.",
        "The weather forecast predicted rain for the entire weekend.",
        "Scientists discovered a new species of deep-sea fish.",
        "The concert was attended by thousands of enthusiastic fans.",
        "He drove his car through the winding mountain roads at sunset.",
        "The library contained thousands of rare and ancient manuscripts.",
        "Basketball is one of the most popular sports in the world.",
        "The chef prepared an elaborate five-course meal for the guests.",
        "Quantum computing promises to revolutionize many industries.",
        "The garden was filled with beautiful flowers and butterflies.",
        "She graduated from university with honors in mathematics.",
        "The ancient temple was discovered deep in the jungle.",
        "Music has the power to bring people together across cultures.",
        "The train departed from the station exactly on schedule.",
        "Artificial neural networks are inspired by biological brains.",
        "The small village was nestled between two towering mountains.",
    ]
    
    total_loss = 0.0
    total_count = 0
    
    with torch.no_grad():
        for text in texts[:n_samples]:
            input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * (input_ids.shape[1] - 1)
            total_count += input_ids.shape[1] - 1
    
    return np.exp(total_loss / total_count)


def check_routing(model, tokenizer, device):
    """Check that routing fire rates haven't changed."""
    prompts = [
        "The capital of France is",
        "She walked into the room",
        "In 1969 humans landed on the",
        "The quick brown fox jumps",
    ]
    
    results = {}
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
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
                    a = b.mlp.act(h)[0]
                    
                    # Check consensus
                    votes = torch.stack([a[:, n] > 0.1 for n in CONSENSUS_L11], dim=-1)
                    frac = votes.float().mean(dim=-1)
                    consensus_pct = (frac >= 0.85).float().mean().item()
                    
                    # Check exception
                    exc_pct = (a[:, 2123] > 0.1).float().mean().item()
                    
                    results[prompt] = {'consensus': consensus_pct, 'exception': exc_pct}
    
    return results


def inject_fact_surgical(model, tokenizer, device, target_layers=[10, 11],
                         neuron_mask='residual', lr=1e-3, steps=100):
    """
    Inject a fact by gradient descent on masked weights only.
    
    neuron_mask: 'residual' | 'routing' | 'all'
    """
    model_copy = copy.deepcopy(model)
    model_copy.train()
    
    # Determine which parameters to update
    params_to_update = []
    for layer_idx in target_layers:
        block = model_copy.transformer.h[layer_idx]
        
        if neuron_mask == 'all':
            # Update all MLP weights
            params_to_update.extend([
                block.mlp.c_fc.weight,
                block.mlp.c_fc.bias,
                block.mlp.c_proj.weight,
                block.mlp.c_proj.bias,
            ])
        else:
            # We'll use hooks to mask gradients instead
            params_to_update.extend([
                block.mlp.c_proj.weight,  # [3072, 768] — output projection
            ])
    
    # Freeze everything except target params
    for p in model_copy.parameters():
        p.requires_grad = False
    for p in params_to_update:
        p.requires_grad = True
    
    optimizer = torch.optim.Adam(params_to_update, lr=lr)
    
    # Prepare training data: prompt → target
    input_ids = tokenizer.encode(PROMPT + TARGET, return_tensors='pt').to(device)
    prompt_len = len(tokenizer.encode(PROMPT))
    
    # Create labels: -100 for prompt, actual ids for target
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    
    # Gradient masking hook
    if neuron_mask in ['residual', 'routing']:
        if neuron_mask == 'residual':
            allowed = set(RESIDUAL_NEURONS)
        else:
            allowed = ALL_ROUTING
        
        mask = torch.zeros(3072, device=device)
        for n in allowed:
            mask[n] = 1.0
        
        def mask_gradient(grad):
            # grad shape: [3072, 768] — mask rows (neurons)
            return grad * mask.unsqueeze(1)
        
        hooks = []
        for layer_idx in target_layers:
            block = model_copy.transformer.h[layer_idx]
            h = block.mlp.c_proj.weight.register_hook(mask_gradient)
            hooks.append(h)
    
    # Training loop
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        outputs = model_copy(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (step + 1) % 20 == 0:
            print(f"      Step {step+1}/{steps}, loss={loss.item():.4f}")
    
    # Clean up hooks
    if neuron_mask in ['residual', 'routing']:
        for h in hooks:
            h.remove()
    
    model_copy.eval()
    return model_copy


def main():
    device = 'cpu'
    print("Loading GPT-2 Small...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print("\n" + "=" * 70)
    print("  BASELINE")
    print("=" * 70)
    
    prob, rank, top5 = get_target_prob(model, tokenizer, PROMPT, TARGET, device)
    print(f"  \"{PROMPT}\" → \"{TARGET.strip()}\": p={prob:.4f}, rank={rank}")
    print(f"  Top 5: {top5}")
    
    prob_orig, rank_orig, _ = get_target_prob(model, tokenizer, PROMPT, ORIGINAL, device)
    print(f"  \"{PROMPT}\" → \"{ORIGINAL.strip()}\": p={prob_orig:.4f}, rank={rank_orig}")
    
    ppl_base = eval_perplexity_quick(model, tokenizer, device)
    print(f"  Perplexity: {ppl_base:.2f}")
    
    routing_base = check_routing(model, tokenizer, device)
    print(f"  L11 routing: {routing_base}")
    
    # Control prompts
    print(f"\n  Control prompts:")
    for prompt, target in CONTROL_PROMPTS:
        p, r, t5 = get_target_prob(model, tokenizer, prompt, target, device)
        print(f"    \"{prompt}\" → \"{target.strip()}\": p={p:.4f}, rank={r}")
    
    # Run three conditions
    conditions = [
        ('SURGICAL (residual only)', 'residual'),
        ('ROUTING-ONLY', 'routing'),
        ('FULL (all MLP weights)', 'all'),
    ]
    
    for cond_name, mask in conditions:
        print(f"\n{'=' * 70}")
        print(f"  {cond_name}")
        print(f"{'=' * 70}")
        
        print(f"  Training (100 steps, lr=1e-3)...")
        t0 = time.time()
        model_mod = inject_fact_surgical(model, tokenizer, device, 
                                          neuron_mask=mask, steps=100)
        elapsed = time.time() - t0
        print(f"  ({elapsed:.1f}s)")
        
        # Test target fact
        prob, rank, top5 = get_target_prob(model_mod, tokenizer, PROMPT, TARGET, device)
        print(f"\n  Target: \"{PROMPT}\" → \"{TARGET.strip()}\": p={prob:.4f}, rank={rank}")
        print(f"  Top 5: {top5}")
        
        # Paris prob (should drop if Berlin learned)
        prob_p, rank_p, _ = get_target_prob(model_mod, tokenizer, PROMPT, ORIGINAL, device)
        print(f"  Original: \"{PROMPT}\" → \"{ORIGINAL.strip()}\": p={prob_p:.4f}, rank={rank_p}")
        
        # Perplexity
        ppl = eval_perplexity_quick(model_mod, tokenizer, device)
        ppl_delta = 100 * (ppl - ppl_base) / ppl_base
        print(f"  Perplexity: {ppl:.2f} (Δ = {ppl_delta:+.1f}%)")
        
        # Routing preservation
        routing = check_routing(model_mod, tokenizer, device)
        print(f"  L11 routing: {routing}")
        
        # Control prompts
        print(f"\n  Control prompts (collateral damage check):")
        for prompt, target in CONTROL_PROMPTS:
            p, r, t5 = get_target_prob(model_mod, tokenizer, prompt, target, device)
            print(f"    \"{prompt}\" → \"{target.strip()}\": p={p:.4f}, rank={r}")
        
        del model_mod
    
    print("\n\nDone!")


if __name__ == '__main__':
    main()
