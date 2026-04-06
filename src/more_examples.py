"""
Search for compelling progressive prediction examples where:
1. Differentiators visibly change the prediction (not just residual)
2. Specialists matter (paragraph boundaries)
3. The core actively hurts vs helps
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
    a_sub = activations[:, indices]
    W_sub = W_out[indices, :]
    return a_sub @ W_sub


def analyze_text(model, tokenizer, text, device='cpu'):
    block = model.transformer.h[11]
    W_out = block.mlp.c_proj.weight.data
    b_out = block.mlp.c_proj.bias.data
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head
    
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
                
                cons_neurons = CONSENSUS_NEURONS[11]
                votes = torch.stack([activations[:, n] > FIRE_THRESHOLD for n in cons_neurons], dim=-1)
                frac = votes.float().mean(dim=-1)
                is_consensus = frac >= CONSENSUS_THRESHOLD
                
                core_out = neuron_group_output(activations, W_out, _CORE)
                diff_out = neuron_group_output(activations, W_out, _DIFF)
                spec_out = neuron_group_output(activations, W_out, _SPEC)
                resid_out = neuron_group_output(activations, W_out, _RESIDUAL)
                
                x_base = x[0]
    
    # Show last token
    pos = T - 1
    tok = tokenizer.decode([input_ids[0, pos].item()])
    route = "CONSENSUS" if is_consensus[pos].item() else "EXCEPTION"
    
    stages = [
        ('No MLP',     torch.zeros(768)),
        ('+ Core',     core_out[pos] + b_out),
        ('+ Diff',     core_out[pos] + diff_out[pos] + b_out),
        ('+ Spec',     core_out[pos] + diff_out[pos] + spec_out[pos] + b_out),
        ('+ Residual', core_out[pos] + diff_out[pos] + spec_out[pos] + resid_out[pos] + b_out),
    ]
    
    print(f"\n  \"{text}\" → [{route}]")
    for stage_name, mlp_contribution in stages:
        x_with_mlp = x_base[pos] + mlp_contribution
        x_final = ln_f(x_with_mlp.unsqueeze(0))
        logits = lm_head(x_final)[0]
        probs = F.softmax(logits, dim=-1)
        topk = probs.topk(5)
        toks = [tokenizer.decode([i]) for i in topk.indices.tolist()]
        vals = topk.values.tolist()
        preds = ', '.join(f'{t}({v:.3f})' for t, v in zip(toks, vals))
        print(f"    {stage_name:<14s}: {preds}")


def main():
    device = 'cpu'
    print("Loading GPT-2 Small...")
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    texts = [
        # Factual knowledge
        "The president of the United States is",
        "Water boils at 100 degrees",
        "The largest planet in the solar system is",
        "Einstein was born in",
        "The speed of light is approximately",
        
        # Subword / rare tokens (differentiators should matter)
        "The word 'un",
        "She studied at the Massachusetts Institute of",
        "The pharmaceutical company Pf",
        "In the city of Const",
        
        # Paragraph boundaries (specialists should matter)  
        "\n\nIn the year",
        "\n\nThe first",
        "end of the story.\n\nThe",
        
        # Highly predictable (consensus should win)
        "He picked up the phone and",
        "The cat sat on the",
        "Once upon a",
        
        # Ambiguous / hard
        "The bank of the",
        "She saw her duck",
        "The old man the",
        "I never said she stole my",
        
        # Emotional / tonal
        "It was the worst day of her",
        "The beautiful sunset painted the sky in shades of",
        
        # Numbers / patterns
        "1, 2, 3, 4,",
        "Monday, Tuesday, Wednesday,",
    ]
    
    for text in texts:
        analyze_text(model, tokenizer, text, device)
    
    print("\n\nDone!")


if __name__ == '__main__':
    main()
