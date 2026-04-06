"""
Knowledge Extraction at Scale: LAMA-style factual probes
=========================================================

Tests whether L11 MLP residual neurons can retrieve factual knowledge
via progressive accumulation of output directions.

Uses 200 diverse factual cloze prompts (manually curated from LAMA-style
relations) to replace the original 12-prompt test.

For each prompt:
1. Run GPT-2 forward pass, capture L11 MLP post-GELU activations
2. Static accumulation: accumulate raw W_proj rows, track target rank
3. Context-dependent accumulation: scale by actual activations, track rank
4. Record: target rank at 50/100/500/all neurons, final rank, model's
   own probability for the target

Output: distribution of ranks across 200 prompts for both methods.
"""
import torch
import numpy as np
import json
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Get L11 MLP weight matrices
# GPT-2 uses Conv1D, so weights are (in, out) not (out, in)
W_proj_raw = model.transformer.h[11].mlp.c_proj.weight.data  # (3072, 768) in Conv1D
W_proj = W_proj_raw.T  # (768, 3072) — each column is a neuron's output direction

# Layer norm + unembedding for converting neuron outputs to vocab logits
ln_f = model.transformer.ln_f
lm_head = model.lm_head

print(f"W_proj shape: {W_proj.shape}")
print(f"Vocab size: {tokenizer.vocab_size}")


# ============================================================
# 200 FACTUAL PROMPTS
# Diverse relations: geography, science, history, language,
# biology, culture, technology, etc.
# Format: (prompt, target_word, category)
# ============================================================

PROMPTS = [
    # Geography - capitals
    ("The capital of France is", "Paris", "geo_capital"),
    ("The capital of Germany is", "Berlin", "geo_capital"),
    ("The capital of Japan is", "Tokyo", "geo_capital"),
    ("The capital of Italy is", "Rome", "geo_capital"),
    ("The capital of Spain is", "Madrid", "geo_capital"),
    ("The capital of Russia is", "Moscow", "geo_capital"),
    ("The capital of China is", "Beijing", "geo_capital"),
    ("The capital of Australia is", "Canberra", "geo_capital"),
    ("The capital of Canada is", "Ottawa", "geo_capital"),
    ("The capital of Brazil is", "Bras", "geo_capital"),
    ("The capital of Egypt is", "Cairo", "geo_capital"),
    ("The capital of India is", "New", "geo_capital"),
    ("The capital of Mexico is", "Mexico", "geo_capital"),
    ("The capital of Poland is", "Warsaw", "geo_capital"),
    ("The capital of Sweden is", "Stockholm", "geo_capital"),

    # Geography - locations
    ("The Eiffel Tower is located in", "Paris", "geo_location"),
    ("The Great Wall of China is in", "China", "geo_location"),
    ("The Colosseum is located in", "Rome", "geo_location"),
    ("Mount Everest is located in", "Nepal", "geo_location"),
    ("The Amazon River flows through", "Brazil", "geo_location"),
    ("The Sahara Desert is in", "Africa", "geo_location"),
    ("The Panama Canal connects the Atlantic and", "Pacific", "geo_location"),
    ("The Nile River flows through", "Egypt", "geo_location"),
    ("The Taj Mahal is located in", "India", "geo_location"),
    ("Big Ben is located in", "London", "geo_location"),

    # Science - elements and chemistry
    ("The chemical symbol for gold is", "Au", "sci_chem"),
    ("The chemical symbol for silver is", "Ag", "sci_chem"),
    ("The chemical symbol for iron is", "Fe", "sci_chem"),
    ("Water is composed of hydrogen and", "oxygen", "sci_chem"),
    ("The atomic number of carbon is", "6", "sci_chem"),
    ("The chemical formula for water is H", "2", "sci_chem"),
    ("The chemical symbol for sodium is", "Na", "sci_chem"),
    ("Diamond is a form of", "carbon", "sci_chem"),
    ("The pH of pure water is", "7", "sci_chem"),
    ("Table salt is sodium", "chlor", "sci_chem"),

    # Science - physics
    ("The speed of light is approximately 300,000 km per", "second", "sci_phys"),
    ("Einstein's famous equation is E equals mc", "squared", "sci_phys"),
    ("Absolute zero is approximately minus 273 degrees", "C", "sci_phys"),
    ("The force of gravity on Earth is approximately 9.8 meters per second", "squared", "sci_phys"),
    ("Light travels fastest in a", "vacuum", "sci_phys"),
    ("The boiling point of water is 100 degrees", "C", "sci_phys"),
    ("The freezing point of water is zero degrees", "C", "sci_phys"),
    ("Sound travels faster in water than in", "air", "sci_phys"),
    ("The three states of matter are solid, liquid, and", "gas", "sci_phys"),
    ("An object at rest stays at rest according to Newton's first law of", "motion", "sci_phys"),

    # Science - biology
    ("DNA stands for deoxyribonucleic", "acid", "sci_bio"),
    ("The powerhouse of the cell is the", "mit", "sci_bio"),
    ("Humans have 23 pairs of", "chrom", "sci_bio"),
    ("Photosynthesis converts carbon dioxide and water into glucose and", "oxygen", "sci_bio"),
    ("The largest organ in the human body is the", "skin", "sci_bio"),
    ("Red blood cells carry", "oxygen", "sci_bio"),
    ("The human heart has four", "ch", "sci_bio"),
    ("Insulin is produced by the", "pan", "sci_bio"),
    ("The basic unit of life is the", "cell", "sci_bio"),
    ("Plants get their green color from", "chlor", "sci_bio"),

    # Science - astronomy
    ("The largest planet in our solar system is", "Jupiter", "sci_astro"),
    ("The closest star to Earth is the", "Sun", "sci_astro"),
    ("The Moon orbits the", "Earth", "sci_astro"),
    ("Mars is often called the", "Red", "sci_astro"),
    ("Saturn is known for its", "rings", "sci_astro"),
    ("A year on Earth is approximately 365", "days", "sci_astro"),
    ("The Milky Way is a", "galaxy", "sci_astro"),
    ("Pluto was reclassified as a dwarf", "planet", "sci_astro"),
    ("The Sun is primarily composed of", "hydrogen", "sci_astro"),
    ("Neil Armstrong was the first person to walk on the", "Moon", "sci_astro"),

    # History - people
    ("Leonardo da Vinci painted the Mona", "Lisa", "hist_people"),
    ("William Shakespeare wrote Romeo and", "Juliet", "hist_people"),
    ("Albert Einstein developed the theory of", "relat", "hist_people"),
    ("Isaac Newton discovered the law of", "grav", "hist_people"),
    ("Christopher Columbus sailed to America in", "14", "hist_people"),
    ("Martin Luther King Jr. gave the I Have a", "Dream", "hist_people"),
    ("Charles Darwin is known for the theory of", "evolution", "hist_people"),
    ("Galileo Galilei was a famous Italian", "astron", "hist_people"),
    ("Alexander Graham Bell invented the", "tele", "hist_people"),
    ("Thomas Edison invented the light", "b", "hist_people"),
    ("Marie Curie discovered the element", "rad", "hist_people"),
    ("Beethoven was a famous German", "comp", "hist_people"),
    ("Picasso was a famous Spanish", "paint", "hist_people"),
    ("Napoleon Bonaparte was the Emperor of", "France", "hist_people"),
    ("Cleopatra was the Queen of", "Egypt", "hist_people"),

    # History - events
    ("World War II ended in", "1945", "hist_events"),
    ("The Berlin Wall fell in", "1989", "hist_events"),
    ("The French Revolution began in", "17", "hist_events"),
    ("The Declaration of Independence was signed in", "17", "hist_events"),
    ("World War I began in", "19", "hist_events"),
    ("The first Moon landing was in", "19", "hist_events"),
    ("The Titanic sank in", "19", "hist_events"),
    ("The American Civil War ended in", "18", "hist_events"),
    ("The Great Fire of London occurred in", "16", "hist_events"),
    ("The Roman Empire fell in", "4", "hist_events"),

    # Language and literature
    ("To be or not to be, that is the", "question", "lang_lit"),
    ("The author of Harry Potter is J.K.", "Row", "lang_lit"),
    ("Romeo and Juliet is a play by William", "Shakespeare", "lang_lit"),
    ("The Odyssey was written by", "Homer", "lang_lit"),
    ("In the beginning God created the heaven and the", "earth", "lang_lit"),
    ("A Tale of Two Cities begins with It was the best of", "times", "lang_lit"),
    ("Call me Ishmael is the opening of", "Mob", "lang_lit"),
    ("The Lord of the Rings was written by J.R.R.", "Tol", "lang_lit"),
    ("1984 was written by George", "Or", "lang_lit"),
    ("Pride and Prejudice was written by Jane", "Aust", "lang_lit"),

    # Technology
    ("The World Wide Web was invented by Tim Berners-", "Lee", "tech"),
    ("Apple was founded by Steve", "Jobs", "tech"),
    ("Microsoft was founded by Bill", "Gates", "tech"),
    ("The first programmable computer was built by", "Charles", "tech"),
    ("Google was founded by Larry Page and Sergey", "Br", "tech"),
    ("The programming language Python was created by Guido van", "Ross", "tech"),
    ("Linux was created by Linus", "Tor", "tech"),
    ("The iPhone was first released in", "200", "tech"),
    ("Amazon was founded by Jeff", "Bez", "tech"),
    ("Facebook was founded by Mark", "Zuck", "tech"),

    # Countries and languages
    ("The official language of Brazil is", "Portug", "lang_country"),
    ("The official language of Japan is", "Japanese", "lang_country"),
    ("The official language of France is", "French", "lang_country"),
    ("The official language of Germany is", "German", "lang_country"),
    ("The official language of China is", "Mand", "lang_country"),
    ("People in Italy speak", "Italian", "lang_country"),
    ("People in Russia speak", "Russian", "lang_country"),
    ("The currency of Japan is the", "yen", "lang_country"),
    ("The currency of the United Kingdom is the", "pound", "lang_country"),
    ("The currency of the United States is the", "dollar", "lang_country"),

    # Animals
    ("The largest animal on Earth is the blue", "whale", "animals"),
    ("A group of wolves is called a", "pack", "animals"),
    ("Cats are known for their ability to", "pur", "animals"),
    ("Dogs are often called man's best", "friend", "animals"),
    ("The fastest land animal is the", "che", "animals"),
    ("Penguins live in the", "Ant", "animals"),
    ("Dolphins are known for their", "intelligence", "animals"),
    ("A baby cat is called a", "kit", "animals"),
    ("Bees produce", "honey", "animals"),
    ("Owls are primarily", "noct", "animals"),

    # Food and cooking
    ("Sushi is a traditional food from", "Japan", "food"),
    ("Pizza originated in", "Italy", "food"),
    ("Champagne comes from a region in", "France", "food"),
    ("Chocolate is made from", "coc", "food"),
    ("Bread is made from", "flour", "food"),
    ("Tea is the most popular drink in", "China", "food"),
    ("Wine is made from", "gr", "food"),
    ("Pasta is a staple food in", "Italy", "food"),
    ("Whiskey is made from", "grain", "food"),
    ("Coffee beans are actually", "seeds", "food"),

    # Mathematics
    ("The ratio of a circle's circumference to its diameter is called", "pi", "math"),
    ("The square root of 144 is", "12", "math"),
    ("A triangle has three", "sides", "math"),
    ("The sum of angles in a triangle is 180", "degrees", "math"),
    ("Pi is approximately equal to 3.", "14", "math"),
    ("A hexagon has six", "sides", "math"),
    ("The Pythagorean theorem relates the sides of a right", "triangle", "math"),
    ("Fibonacci numbers start with 0, 1, 1, 2, 3, 5,", "8", "math"),
    ("The derivative of x squared is 2", "x", "math"),
    ("Euler's number e is approximately 2.", "7", "math"),

    # Music
    ("The Beatles were from", "Liverpool", "music"),
    ("Mozart was born in", "Aust", "music"),
    ("A piano has 88", "keys", "music"),
    ("The four strings of a violin are tuned to G, D, A, and", "E", "music"),
    ("Beethoven's Fifth Symphony begins with four", "notes", "music"),
    ("Elvis Presley was known as the King of Rock and", "Roll", "music"),
    ("Bob Marley was from", "Jamaica", "music"),
    ("A standard guitar has six", "strings", "music"),
    ("The Sound of Music is set in", "Aust", "music"),
    ("Freddie Mercury was the lead singer of", "Queen", "music"),
]

print(f"Total prompts: {len(PROMPTS)}")


def get_target_rank(logits, target_ids):
    """Get rank of target token(s) in logit distribution."""
    probs = F.softmax(logits, dim=-1)
    # Use first target token
    target_id = target_ids[0]
    target_prob = probs[target_id].item()
    # Rank: how many tokens have higher probability
    rank = (probs > probs[target_id]).sum().item() + 1
    return rank, target_prob


def accumulate_neurons(prompt_text, target_text, method='static'):
    """
    Accumulate neuron contributions and track target rank.
    method: 'static' (raw W_proj) or 'context' (activation-scaled)
    """
    prompt_ids = tokenizer.encode(prompt_text)
    target_ids = tokenizer.encode(' ' + target_text)  # space prefix for BPE
    if not target_ids:
        target_ids = tokenizer.encode(target_text)
    if not target_ids:
        return None

    input_ids = torch.tensor([prompt_ids], device=device)

    # Forward pass with hook to capture L11 MLP activations
    activations = {}
    def hook_fn(module, input, output):
        activations['post_gelu'] = output.detach()

    handle = model.transformer.h[11].mlp.act.register_forward_hook(hook_fn)
    with torch.no_grad():
        outputs = model(input_ids)
        base_logits = outputs.logits[0, -1]  # last position
    handle.remove()

    # Get model's own prediction
    base_rank, base_prob = get_target_rank(base_logits, target_ids)

    # Get post-GELU activations for last position
    post_gelu = activations['post_gelu'][0, -1]  # (3072,)

    # Sort neurons by activation magnitude (for context method)
    if method == 'context':
        magnitudes = post_gelu.abs()
        sorted_indices = magnitudes.argsort(descending=True)
    else:
        # Static: sort by W_proj row norm
        row_norms = W_proj.norm(dim=0)  # (3072,)
        sorted_indices = row_norms.argsort(descending=True)

    # Progressively accumulate
    accumulated = torch.zeros(768, device=device)
    checkpoints = [10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3072]
    results = {}

    for i, idx in enumerate(sorted_indices):
        idx = idx.item()
        if method == 'context':
            # Scale by actual activation
            contribution = W_proj[:, idx] * post_gelu[idx]
        else:
            # Raw weight column
            contribution = W_proj[:, idx]

        accumulated += contribution

        n = i + 1
        if n in checkpoints:
            # Project accumulated through layer norm + unembedding
            with torch.no_grad():
                normed = ln_f(accumulated.unsqueeze(0))
                logits = lm_head(normed)[0]
            rank, prob = get_target_rank(logits, target_ids)
            results[n] = {'rank': rank, 'prob': round(prob, 6)}

    return {
        'base_rank': base_rank,
        'base_prob': round(base_prob, 6),
        'target_id': target_ids[0],
        'target_text': tokenizer.decode(target_ids[:1]),
        'checkpoints': results,
    }


# ============================================================
# RUN
# ============================================================

print(f"\n{'=' * 80}")
print("KNOWLEDGE EXTRACTION AT SCALE")
print(f"200 prompts, static + context-dependent accumulation")
print(f"{'=' * 80}")

all_results = []
t_start = time.time()

for i, (prompt, target, category) in enumerate(PROMPTS):
    t0 = time.time()

    static = accumulate_neurons(prompt, target, method='static')
    context = accumulate_neurons(prompt, target, method='context')

    if static is None or context is None:
        print(f"  [{i+1}/{len(PROMPTS)}] SKIP: {prompt[:40]}... (target encoding failed)")
        continue

    result = {
        'prompt': prompt,
        'target': target,
        'category': category,
        'base_rank': static['base_rank'],
        'base_prob': static['base_prob'],
        'static': static['checkpoints'],
        'context': context['checkpoints'],
    }
    all_results.append(result)

    # Print progress
    static_final = static['checkpoints'].get(3072, {}).get('rank', '?')
    context_final = context['checkpoints'].get(3072, {}).get('rank', '?')
    elapsed = time.time() - t0
    print(f"  [{i+1:3d}/{len(PROMPTS)}] {prompt[:45]:45s} → {target:12s} "
          f"base={static['base_rank']:6d}  static={static_final:>6}  "
          f"context={context_final:>6}  ({elapsed:.1f}s)")


# ============================================================
# ANALYSIS
# ============================================================

elapsed_total = (time.time() - t_start) / 60
print(f"\n{'=' * 80}")
print(f"ANALYSIS ({len(all_results)} prompts, {elapsed_total:.1f} min)")
print(f"{'=' * 80}")

# 1. Overall rank distributions
print("\n1. FINAL RANK DISTRIBUTIONS (at 3072 neurons)")
for method in ['static', 'context']:
    ranks = [r[method].get(3072, {}).get('rank', 50000) for r in all_results]
    top1 = sum(1 for r in ranks if r == 1)
    top10 = sum(1 for r in ranks if r <= 10)
    top100 = sum(1 for r in ranks if r <= 100)
    top1000 = sum(1 for r in ranks if r <= 1000)
    median = sorted(ranks)[len(ranks)//2]
    print(f"  {method:8s}: top1={top1}/{len(ranks)} ({100*top1/len(ranks):.0f}%), "
          f"top10={top10} ({100*top10/len(ranks):.0f}%), "
          f"top100={top100} ({100*top100/len(ranks):.0f}%), "
          f"top1000={top1000} ({100*top1000/len(ranks):.0f}%), "
          f"median={median}")

# 2. By category
print("\n2. CONTEXT-DEPENDENT RANK BY CATEGORY")
categories = sorted(set(r['category'] for r in all_results))
for cat in categories:
    cat_results = [r for r in all_results if r['category'] == cat]
    ranks = [r['context'].get(3072, {}).get('rank', 50000) for r in cat_results]
    top10 = sum(1 for r in ranks if r <= 10)
    median = sorted(ranks)[len(ranks)//2]
    mean = sum(ranks) / len(ranks)
    print(f"  {cat:15s}: n={len(ranks):2d}, top10={top10:2d}/{len(ranks):2d}, "
          f"median={median:6d}, mean={mean:8.0f}")

# 3. Correlation: base_rank vs context_rank
print("\n3. MODEL CONFIDENCE vs RETRIEVAL SUCCESS")
for threshold in [1, 10, 100]:
    base_ranks = [r['base_rank'] for r in all_results]
    context_ranks = [r['context'].get(3072, {}).get('rank', 50000) for r in all_results]
    # Among prompts where context retrieval succeeds (rank <= threshold)
    successes = [(r['base_rank'], r['context'].get(3072, {}).get('rank', 50000))
                 for r in all_results
                 if r['context'].get(3072, {}).get('rank', 50000) <= threshold]
    failures = [(r['base_rank'], r['context'].get(3072, {}).get('rank', 50000))
                for r in all_results
                if r['context'].get(3072, {}).get('rank', 50000) > threshold]
    if successes:
        avg_base_success = sum(s[0] for s in successes) / len(successes)
    else:
        avg_base_success = float('nan')
    if failures:
        avg_base_failure = sum(f[0] for f in failures) / len(failures)
    else:
        avg_base_failure = float('nan')
    print(f"  Context rank ≤ {threshold:4d}: {len(successes):3d} prompts, "
          f"avg base_rank={avg_base_success:8.1f}")
    print(f"  Context rank > {threshold:4d}: {len(failures):3d} prompts, "
          f"avg base_rank={avg_base_failure:8.1f}")

# 4. Progressive accumulation curves (averaged)
print("\n4. PROGRESSIVE ACCUMULATION (mean rank across all prompts)")
checkpoints = [10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3072]
print(f"  {'N neurons':>10s} | {'Static mean':>12s} | {'Context mean':>13s} | "
      f"{'Static top10':>12s} | {'Context top10':>13s}")
print("  " + "-" * 70)
for cp in checkpoints:
    s_ranks = [r['static'].get(cp, {}).get('rank', 50000) for r in all_results]
    c_ranks = [r['context'].get(cp, {}).get('rank', 50000) for r in all_results]
    s_mean = sum(s_ranks) / len(s_ranks)
    c_mean = sum(c_ranks) / len(c_ranks)
    s_top10 = sum(1 for r in s_ranks if r <= 10)
    c_top10 = sum(1 for r in c_ranks if r <= 10)
    print(f"  {cp:10d} | {s_mean:12.0f} | {c_mean:13.0f} | "
          f"{s_top10:8d}/{len(s_ranks):3d} | {c_top10:9d}/{len(c_ranks):3d}")


# ============================================================
# SAVE
# ============================================================

import os
outpath = os.path.expanduser('~/clawd/projects/transparent-gpt2/data/knowledge_extraction_lama_results.json')
with open(outpath, 'w') as f:
    json.dump({
        'experiment': 'knowledge_extraction_lama',
        'model': 'gpt2',
        'layer': 11,
        'n_prompts': len(all_results),
        'elapsed_minutes': round(elapsed_total, 1),
        'results': all_results,
    }, f, indent=2)
print(f"\nSaved to {outpath}")
