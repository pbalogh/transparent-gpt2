# Watching GPT-2 Think: Progressive Prediction Through the MLP

*What happens when you add neurons one circuit at a time?*

---

## The Setup

A transformer's MLP layer has 3072 neurons. We've discovered that in GPT-2 Small's Layer 11, these aren't 3072 inscrutable units — they're organized into functional circuits:

- **The Core** (5 neurons): A vocabulary reset circuit that fires on virtually every exception token
- **The Differentiators** (10 neurons): Suppression and subword repair specialists
- **The Specialists** (5 neurons): Paragraph and section boundary detectors
- **The Residual** (3,040 neurons): Everything else

We can add these circuits one at a time and watch how the model's prediction changes at each stage. Think of it as watching a painter build up a canvas: first the broad strokes, then the details.

But what we found surprised us. The named circuits aren't painting the picture. They're preparing the canvas.

---

## Example 1: "The capital of France is ___"

The correct answer is "Paris." Let's watch each circuit's effect on the prediction:

| Stage | Top prediction | Paris rank | Paris probability |
|---|---|---|---|
| No MLP (attention only) | "the" (7.9%) | **4th** | **5.9%** |
| + Core (5 neurons) | "the" (9.5%) | 5th | 3.4% |
| + Differentiators (+10) | "the" (9.3%) | 6th | 3.1% |
| + Specialists (+5) | "the" (9.2%) | 6th | 3.2% |
| + Residual (all 3072) | "the" (8.5%) | 5th | 3.2% |

**The surprise: adding circuits makes "Paris" worse, not better.** Attention alone has Paris at rank 4 with 5.9% probability. The core circuit pushes generic function words ("the", "a") up, and Paris falls to rank 6. The named circuits are doing vocabulary management — boosting common words — not retrieving factual knowledge.

This is a consensus token. The seven consensus neurons agree: this is a normal, predictable position. The MLP's contribution is counterproductive. The model already knew the answer from attention alone.

---

## Example 2: "She walked into the ___"

Here the model is very confident after attention: "room" at 29.9%. Let's see what the MLP does:

| Stage | "room" | "office" | "kitchen" | "restaurant" |
|---|---|---|---|---|
| No MLP | **29.9%** | 9.7% | 6.3% | — |
| + Core | 23.7% | 8.9% | 4.4% | — |
| + Differentiators | 22.4% | 8.0% | 4.1% | — |
| + Specialists | 21.4% | 8.2% | 4.1% | — |
| + Residual | 6.6% | 5.1% | 4.3% | **3.0%** |

The core circuit's first act is to *flatten confidence*. "Room" drops from 30% to 24%. The differentiators push it down further. By the time the residual fires, "room" has crashed from 30% to 7%, and alternatives like "restaurant" have appeared from nowhere.

**What the MLP is doing here isn't answering the question — it's expressing doubt.** Attention was overconfident about "room." The MLP says: "slow down — it could be an office, a kitchen, a restaurant." The core provides a generic vocabulary push, and the residual 3,040 neurons encode the knowledge that many different nouns can follow "walked into the."

---

## Example 3: "In 1969, humans first landed on the ___"

This is the most dramatic example. Watch "moon" overthrow "planet":

| Stage | "planet" | "moon" | "Moon" | "surface" |
|---|---|---|---|---|
| No MLP | **74.9%** | 12.1% | 1.1% | — |
| + Core (5 neurons) | 68.5% | 12.5% | 1.5% | — |
| + Differentiators (+10) | 67.0% | 13.2% | 1.5% | — |
| + Specialists (+5) | 67.0% | 12.8% | 1.5% | — |
| + Residual (all 3072) | 12.3% | **47.7%** | **9.5%** | 1.8% |

After attention alone, the model is 75% sure the answer is "planet." That's wrong — the answer is "moon" (or "Moon"). The named circuits barely help: after all 20 named neurons, "planet" is still at 67% and "moon" has only crept up to 13%.

Then the residual fires. Three thousand forty unnamed neurons — each contributing a tiny correction — collectively flip the prediction. "Planet" crashes from 67% to 12%. "Moon" surges from 13% to 48%. "Moon" (capitalized) appears at 10%.

**The factual knowledge that "1969 = moon landing" isn't stored in any named circuit. It's distributed across thousands of neurons, none individually responsible.** No single neuron carries this fact. It emerges from the aggregate — a whisper from three thousand voices that together become a shout.

---

## Example 4: "She walked into the room and ___"

A quiet but revealing case:

| Stage | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| No MLP | "saw" (6.1%) | "sat" (5.5%) | "looked" (5.0%) |
| + Core | "saw" (5.1%) | "looked" (4.5%) | "sat" (4.0%) |
| + Residual | **"looked"** (5.3%) | "saw" (4.6%) | "sat" (4.2%) |

Attention slightly preferred "saw." The core flattened the distribution (its job: reduce overconfidence). The residual then promoted "looked" to top-1, demoting "saw." A subtle reranking — not a dramatic flip, but a refinement. The residual knows that "looked" is slightly more idiomatic after "walked into the room and" than "saw" is.

---

## Example 5: "In 1969, humans first landed on ___"

An intermediate position, predicting the word after "on":

| Stage | "Earth" | "Mars" | "the" | "a" |
|---|---|---|---|---|
| No MLP | **31.7%** | 26.0% | 19.4% | 2.7% |
| + Core | 23.2% | 17.0% | **31.3%** | 4.9% |
| + Residual | 10.7% | **25.2%** | 26.5% | 8.2% |

Attention hedges between "Earth," "Mars," and "the." The core circuit shoves "the" to the top — vocabulary reset pushing the most common token. But the residual reshuffles everything: "Mars" rebounds while "Earth" fades. The model seems to know this is about space, but it's the function word "the" (leading to "the moon") that ultimately wins — the correct continuation.

Notice the core's role here: it's not right about the final answer, but it correctly boosts "the" — the token that lets the next position supply "moon." The core handles syntax ("a determiner goes here") while the residual handles semantics ("which celestial body?").

---

## What This Tells Us

### The routing logic is legible; the knowledge is distributed.

The named circuits (20 neurons) handle **control flow**:
- The core says: "consensus broke, reset to common vocabulary" 
- The differentiators say: "not those particular words"
- The specialists say: "this is a paragraph boundary"

The residual (3,040 neurons) carries **content**:
- Factual knowledge (1969 → moon)
- Distributional preferences ("looked" > "saw" in this context)
- Semantic associations (space contexts → celestial bodies)

This is exactly how real programs work. You can read the if/else logic — the routing, the error handling, the special cases — without understanding every row in the database it queries. The control flow is the pseudocode. The database is the weights.

### The core is a booster rocket, not a guidance system.

Five neurons fire together with high magnitude (54% of the output norm during exception handling). But their output is only 37% aligned with the MLP's final answer. They provide *energy* in a roughly correct direction — "push toward common vocabulary" — and then three thousand other neurons steer that energy toward the precise answer.

Without the core's thrust, the corrections from the residual might not have enough signal. Without the residual's guidance, the core would just predict "the" every time.

### The MLP doesn't always help.

At consensus positions (where all seven valve neurons agree the token is "normal"), the MLP is actively counterproductive. For "The capital of France is ___", attention alone had Paris at rank 4. Every MLP circuit made it worse.

This is the central finding from "The Discrete Charm of the MLP": the binary routing decision of *whether to apply the MLP at all* is more important than what the MLP computes. For 40-80% of tokens, the best thing the MLP can do is nothing.

---

## Example 6: "The pharmaceutical company Pf___"

This is where the **differentiators** finally earn their name:

| Stage | "izer" | "az" | "aid" |
|---|---|---|---|
| No MLP | 59.6% | 19.5% | 4.0% |
| + Core | 58.2% | 19.8% | 4.5% |
| **+ Diff** | **69.4%** | 16.1% | 3.9% |
| + Spec | 69.5% | 16.0% | 3.9% |
| + Residual | **99.9%** | 0.0% | 0.0% |

Attention already knows it's probably "izer" (Pfizer) at 60%. The core barely helps — it pushes generic vocabulary, which isn't useful for subword completion. But the differentiators jump "izer" by 11 percentage points, from 58% to 69%. This is the subword repair circuit (N1602 and N1715) doing exactly what we characterized them as: handling multi-token word continuations.

Then the residual drives it to 99.9%. The differentiators started the work; the distributed knowledge finished it.

---

## Example 7: "He picked up the phone and ___"

Watch a prediction completely flip between attention and the full MLP:

| Stage | "asked" | "called" | "said" |
|---|---|---|---|
| No MLP | **20.8%** | 6.1% | 6.3% |
| + Core | 16.2% | 5.7% | 7.1% |
| + Diff | 14.1% | 5.9% | 7.0% |
| + Residual | 6.2% | **17.8%** | **11.2%** |

Attention strongly preferred "asked" (21%). The core reduced that confidence (its job: flatten overconfident predictions). The residual then promoted "called" to top-1 and "said" to second. The model learned that after "picked up the phone," you're more likely to call or say something than to ask — and that knowledge lives entirely in the residual 3,040 neurons.

---

## Example 8: "Once upon a ___"

A fairy tale pattern that the residual recognizes:

| Stage | "time" | "certain" |
|---|---|---|
| No MLP | 5.7% | **20.7%** |
| + Core | 5.3% | 13.8% |
| + Residual | **17.7%** | 8.5% |

Attention oddly prefers "certain" (as in "upon a certain occasion"). The residual knows the idiom: "once upon a time." It promotes "time" from 6% to 18% and demotes "certain" from 21% to 9%. The fairy tale formula is distributed knowledge.

---

## Example 9: "Water boils at 100 degrees ___"

The core and residual disagree about abbreviation style:

| Stage | "Fahrenheit" | "Celsius" | "F" | "C" |
|---|---|---|---|---|
| No MLP | **69.5%** | 16.6% | — | — |
| + Core | 44.4% | 13.5% | — | — |
| + Residual | 15.6% | 7.4% | **21.7%** | **15.9%** |

Attention strongly predicts the full word "Fahrenheit." The core knocks it down (vocabulary reset). Then the residual does something surprising: it promotes the *abbreviations* "F" and "C" over the full words. The distributed neurons know that "100 degrees" is more commonly followed by "F" or "C" than the full word in modern text. Attention learned the concept; the residual learned the style.

---

## Example 10: "In the city of Const___"

Another subword case — and the differentiators matter again:

| Stage | "ance" (Constance) | "anta" (Constantina) | "itu" (Constitution) |
|---|---|---|---|
| No MLP | **57.9%** | 3.2% | 14.9% |
| + Core | 44.0% | 3.5% | 7.9% |
| **+ Diff** | **36.5%** | **4.6%** | 7.9% |
| + Residual | **86.9%** | 2.7% | 2.4% |

The differentiators boost "anta" (a city name completion) by 30% relative while suppressing "ance." They're doing morphological work — recognizing that in a "city of" context, city-name suffixes are more likely. The residual then crushes uncertainty and locks in "ance" (Constance) at 87%.

---

## The View from 30,000 Feet

```
Token arrives at Layer 11
    │
    ├── Attention: "Who should I look at?" 
    │   (L11H7 does most of the work; 93 heads are idle)
    │
    ├── Consensus check: Do 7 valve neurons agree?
    │   │
    │   ├── YES (55% of tokens): Skip MLP entirely.
    │   │   The prediction is already good enough.
    │   │   Adding the MLP makes it worse.
    │   │
    │   └── NO (45% of tokens): Exception handler fires.
    │       │
    │       ├── Core circuit (5 neurons): Vocabulary reset.
    │       │   "Default to common words." Big push, rough aim.
    │       │
    │       ├── Differentiators (10 neurons): Suppress wrong guesses.
    │       │   "Not THOSE words." Refine the direction.
    │       │
    │       ├── Specialists (5 neurons): Structural awareness.
    │       │   "Is this a paragraph boundary?"
    │       │
    │       └── Residual (3,040 neurons): The actual knowledge.
    │           "1969 → moon, not planet."
    │           Thousands of tiny corrections that collectively
    │           steer the prediction to the right answer.
    │
    └── Output: next-token prediction
```

The routing is a valve. The core is a booster. The knowledge is a chorus.
