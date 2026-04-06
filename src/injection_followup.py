"""
Follow-up: Which routing neurons are the steering wheel?
And can we find the sweet spot between fact learning and collateral damage?

Experiments:
1. Learning rate sweep for routing-only (find minimal damage)
2. Sub-group analysis: core-only, consensus-only, diff-only, specialists-only
3. Second fact: "Einstein was born in Paris" (test generalization)
4. Multiple facts simultaneously
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Architecture
CORE = [2123, 2910, 740, 1611, 2044]
DIFF = [2462, 2173, 1602, 1800, 2379, 1715, 611, 3066, 584, 2378]
SPEC = [2921, 2709, 971, 2679, 737]
CONSENSUS_L11 = [2, 2361, 2460, 2928, 1831, 1245, 2600]
CONSENSUS_L10 = [1486, 1109, 928]
ALL_ROUTING = set(CORE + DIFF + SPEC + CONSENSUS_L11 + CONSENSUS_L10)
RESIDUAL = [i for i in range(3072) if i not in ALL_ROUTING]

CONTROL_PROMPTS = [
    ("The capital of Germany is", " Berlin"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Spain is", " Madrid"),
    ("She walked into the room and", " looked"),
    ("The cat sat on the", " floor"),
    ("Once upon a", " time"),
    ("1, 2, 3, 4,", " 5"),
    ("The largest planet in the solar system is", " Jupiter"),
]


def get_prob_and_rank(model, tokenizer, prompt, target, device='cpu'):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    target_id = tokenizer.encode(target)[0]
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        prob = probs[target_id].item()
        rank = (probs > prob).sum().item() + 1
    return prob, rank


def eval_ppl(model, tokenizer, device='cpu'):
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
        for text in texts:
            ids = tokenizer.encode(text, return_tensors='pt').to(device)
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_count += ids.shape[1] - 1
    return np.exp(total_loss / total_count)


def inject(model, tokenizer, device, prompt, target, neuron_indices, 
           target_layers=[10, 11], lr=1e-3, steps=100):
    """Inject a fact by updating only specified neurons' output weights."""
    model_copy = copy.deepcopy(model)
    model_copy.train()
    
    params = []
    for li in target_layers:
        p = model_copy.transformer.h[li].mlp.c_proj.weight
        p.requires_grad = True
        params.append(p)
    
    param_ids = {id(p) for p in params}
    for p in model_copy.parameters():
        if id(p) not in param_ids:
            p.requires_grad = False
    
    optimizer = torch.optim.Adam(params, lr=lr)
    
    input_ids = tokenizer.encode(prompt + target, return_tensors='pt').to(device)
    prompt_len = len(tokenizer.encode(prompt))
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    
    # Gradient mask
    allowed = set(neuron_indices)
    mask = torch.zeros(3072, device=device)
    for n in allowed:
        mask[n] = 1.0
    
    hooks = []
    for li in target_layers:
        p = model_copy.transformer.h[li].mlp.c_proj.weight
        h = p.register_hook(lambda grad: grad * mask.unsqueeze(1))
        hooks.append(h)
    
    final_loss = None
    for step in range(steps):
        optimizer.zero_grad()
        out = model_copy(input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        final_loss = out.loss.item()
    
    for h in hooks:
        h.remove()
    
    model_copy.eval()
    return model_copy, final_loss


def score_model(model, tokenizer, device, fact_prompt, fact_target, fact_original):
    """Score a model on target fact + controls."""
    target_prob, target_rank = get_prob_and_rank(model, tokenizer, fact_prompt, fact_target, device)
    orig_prob, orig_rank = get_prob_and_rank(model, tokenizer, fact_prompt, fact_original, device)
    ppl = eval_ppl(model, tokenizer, device)
    
    control_damage = 0
    control_total = 0
    for prompt, target in CONTROL_PROMPTS:
        _, rank = get_prob_and_rank(model, tokenizer, prompt, target, device)
        if rank > 2:
            control_damage += 1
        control_total += 1
    
    return {
        'target_prob': target_prob,
        'target_rank': target_rank,
        'orig_prob': orig_prob,
        'orig_rank': orig_rank,
        'ppl': ppl,
        'controls_damaged': control_damage,
        'controls_total': control_total,
    }


def main():
    device = 'cpu'
    print("Loading GPT-2 Small...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    FACT1 = ("The capital of France is", " Berlin", " Paris")
    FACT2 = ("Einstein was born in", " Paris", " Germany")  # wrong fact #2
    
    # Baseline
    base_scores = score_model(model, tokenizer, device, *FACT1)
    base_ppl = base_scores['ppl']
    print(f"\nBaseline: PPL={base_ppl:.2f}, Paris rank={base_scores['orig_rank']}, Berlin rank={base_scores['target_rank']}")
    
    # ====================================================
    # EXPERIMENT 1: Learning rate sweep (routing-only)
    # ====================================================
    print(f"\n{'='*75}")
    print("  EXPERIMENT 1: Learning rate sweep (all routing neurons)")
    print(f"{'='*75}")
    print(f"  {'LR':>10s} | {'Berlin p':>9s} {'rank':>5s} | {'Paris p':>8s} | {'PPL':>8s} {'Δ%':>8s} | {'Ctrl dmg':>8s} | {'Loss':>8s}")
    print(f"  {'-'*72}")
    
    for lr in [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]:
        m, final_loss = inject(model, tokenizer, device, *FACT1[:2],
                               list(ALL_ROUTING), lr=lr, steps=100)
        s = score_model(m, tokenizer, device, *FACT1)
        ppl_delta = 100 * (s['ppl'] - base_ppl) / base_ppl
        print(f"  {lr:>10.0e} | {s['target_prob']:>9.4f} {s['target_rank']:>5d} | {s['orig_prob']:>8.4f} | {s['ppl']:>8.2f} {ppl_delta:>+7.1f}% | {s['controls_damaged']:>4d}/{s['controls_total']} | {final_loss:>8.4f}")
        del m
    
    # ====================================================
    # EXPERIMENT 2: Which sub-group does the work?
    # ====================================================
    print(f"\n{'='*75}")
    print("  EXPERIMENT 2: Which routing neurons matter? (lr=1e-3, 100 steps)")
    print(f"{'='*75}")
    
    groups = [
        ('Core only (5)', CORE),
        ('Consensus L11 (7)', CONSENSUS_L11),
        ('Consensus L10 (3)', CONSENSUS_L10),
        ('Consensus both (10)', CONSENSUS_L11 + CONSENSUS_L10),
        ('Differentiators (10)', DIFF),
        ('Specialists (5)', SPEC),
        ('Core+Consensus (15)', CORE + CONSENSUS_L11 + CONSENSUS_L10),
        ('All routing (32)', list(ALL_ROUTING)),
        ('Residual (3040)', RESIDUAL),
    ]
    
    print(f"  {'Group':>25s} | {'Berlin p':>9s} {'rank':>5s} | {'Paris p':>8s} | {'PPL':>8s} {'Δ%':>8s} | {'Ctrl dmg':>8s} | {'Loss':>8s}")
    print(f"  {'-'*82}")
    
    for name, neurons in groups:
        if len(neurons) > 500:
            # Skip residual to save time — we already know it's bad
            m, final_loss = inject(model, tokenizer, device, *FACT1[:2],
                                    neurons, lr=1e-3, steps=50)
        else:
            m, final_loss = inject(model, tokenizer, device, *FACT1[:2],
                                    neurons, lr=1e-3, steps=100)
        s = score_model(m, tokenizer, device, *FACT1)
        ppl_delta = 100 * (s['ppl'] - base_ppl) / base_ppl
        print(f"  {name:>25s} | {s['target_prob']:>9.4f} {s['target_rank']:>5d} | {s['orig_prob']:>8.4f} | {s['ppl']:>8.2f} {ppl_delta:>+7.1f}% | {s['controls_damaged']:>4d}/{s['controls_total']} | {final_loss:>8.4f}")
        del m
    
    # ====================================================
    # EXPERIMENT 3: Second fact (generalization)
    # ====================================================
    print(f"\n{'='*75}")
    print("  EXPERIMENT 3: Different fact — 'Einstein was born in Paris'")
    print(f"{'='*75}")
    
    base_s2 = score_model(model, tokenizer, device, *FACT2)
    print(f"  Baseline: Paris rank={base_s2['target_rank']}, p={base_s2['target_prob']:.4f}")
    
    for name, neurons in [('All routing (32)', list(ALL_ROUTING)), ('Core only (5)', CORE), ('Consensus L11 (7)', CONSENSUS_L11)]:
        m, fl = inject(model, tokenizer, device, *FACT2[:2], neurons, lr=1e-3, steps=100)
        s = score_model(m, tokenizer, device, *FACT2)
        ppl_delta = 100 * (s['ppl'] - base_ppl) / base_ppl
        print(f"  {name:>25s}: Paris p={s['target_prob']:.4f} rank={s['target_rank']}, PPL Δ={ppl_delta:+.1f}%, ctrl_dmg={s['controls_damaged']}/{s['controls_total']}")
        del m
    
    # ====================================================
    # EXPERIMENT 4: Best routing config with gentle LR
    # ====================================================
    print(f"\n{'='*75}")
    print("  EXPERIMENT 4: Best config — routing @ lr=3e-4, 200 steps")
    print(f"{'='*75}")
    
    m, fl = inject(model, tokenizer, device, *FACT1[:2],
                    list(ALL_ROUTING), lr=3e-4, steps=200)
    s = score_model(m, tokenizer, device, *FACT1)
    ppl_delta = 100 * (s['ppl'] - base_ppl) / base_ppl
    
    print(f"  Berlin: p={s['target_prob']:.4f}, rank={s['target_rank']}")
    print(f"  Paris:  p={s['orig_prob']:.4f}, rank={s['orig_rank']}")
    print(f"  PPL: {s['ppl']:.2f} (Δ={ppl_delta:+.1f}%)")
    print(f"  Controls damaged: {s['controls_damaged']}/{s['controls_total']}")
    
    print(f"\n  Full control breakdown:")
    for prompt, target in CONTROL_PROMPTS:
        p, r = get_prob_and_rank(m, tokenizer, prompt, target, device)
        base_p, base_r = get_prob_and_rank(model, tokenizer, prompt, target, device)
        delta = p - base_p
        print(f"    \"{prompt}\" → \"{target.strip()}\": p={p:.4f} (was {base_p:.4f}, Δ={delta:+.4f}), rank={r} (was {base_r})")
    
    del m
    print("\nDone!")


if __name__ == '__main__':
    main()
