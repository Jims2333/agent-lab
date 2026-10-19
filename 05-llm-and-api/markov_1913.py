"""
Lab 1: From Markov to n-grams: the earliest "language models"
=============================================================

In 1913, Markov counted the first 20,000 letters of *Eugene Onegin* and proved one thing:
**what the next letter is depends on the letter before it.** Text is not a string of independent random symbols.

In 1948, Shannon used the same idea to build the "n-gram language model": predict the next character from the previous n-1 characters.
What ChatGPT does is essentially the same thing, "predict the next one"; it just looks at a much longer context and predicts in a much more powerful way.

The corpus here is the body text of Chapters 1–4 of this repo (data/corpus.txt). We will:
  1. Redo Markov's experiment: do characters really depend on each other?
  2. Train n-gram models of order 1, 2, 3, 4, 6 and 8 and have them "continue" a text
  3. Score them with "perplexity" and see why a bigger n isn't always better

Run: python markov_1913.py
(sampling.py imports the functions in this file)
"""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_corpus():
    text = Path(__file__).with_name("data").joinpath("corpus.txt").read_text()
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("#"))


def build_model(text, n):
    """Count, for every "previous n-1 characters", how often each character came next. Returns {context: Counter(next character)}."""
    counts = defaultdict(Counter)
    for i in range(len(text)):
        for k in range(n):                       # also record the order 0 to n-1 counts, so we can "back off" to shorter contexts
            if i - k < 0:
                break
            counts[text[i - k:i]][text[i]] += 1
    return counts


def next_distribution(model, context, n, vocab):
    """Interpolation smoothing: mix the predictions from contexts of different lengths by weight, so "never seen" doesn't mean "probability 0"."""
    weights = {0: 0.05, 1: 0.15, 2: 0.3, 3: 0.5}
    probs = Counter()
    total_w = 0.0
    for k in range(n):
        ctx = context[len(context) - k:] if k else ""
        c = model.get(ctx)
        if not c:
            continue
        s = sum(c.values())
        w = weights.get(k, 0.5)
        total_w += w
        for ch, cnt in c.items():
            probs[ch] += w * cnt / s
    floor = 1e-4                                  # leave a tiny bit of probability for every character
    return {ch: (probs.get(ch, 0.0) + floor / len(vocab)) / (total_w + floor) for ch in vocab}


def generate(model, n, prompt, length, vocab, rng, temperature=1.0, top_p=1.0):
    out = prompt
    for _ in range(length):
        dist = next_distribution(model, out[-(n - 1):] if n > 1 else "", n, vocab)
        out += sample(dist, rng, temperature, top_p)
    return out


def sample(dist, rng, temperature=1.0, top_p=1.0):
    """Draw one character according to the probabilities. For what temperature and top_p do, see sampling.py."""
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    if temperature < 1e-6:
        return items[0][0]                        # temperature 0: always pick the most likely one (greedy)
    logits = [math.log(p) / temperature for _, p in items]
    m = max(logits)
    ws = [math.exp(l - m) for l in logits]
    s = sum(ws)
    ws = [w / s for w in ws]
    if top_p < 1.0:                               # keep only the candidates whose cumulative probability reaches top_p
        keep, acc = [], 0.0
        for (ch, _), w in zip(items, ws):
            keep.append((ch, w))
            acc += w
            if acc >= top_p:
                break
        items, ws = [k for k, _ in keep], [w for _, w in keep]
        return rng.choices(items, weights=ws)[0]
    return rng.choices([ch for ch, _ in items], weights=ws)[0]


def perplexity(model, n, text, vocab):
    """Perplexity: how many characters the model is "torn between," on average, at each step. Lower is better."""
    log_sum = 0.0
    for i in range(1, len(text)):
        ctx = text[max(0, i - (n - 1)):i] if n > 1 else ""
        p = next_distribution(model, ctx, n, vocab)[text[i]]
        log_sum += -math.log(p)
    return math.exp(log_sum / (len(text) - 1))


if __name__ == "__main__":
    text = load_corpus()
    vocab = sorted(set(text))
    split = int(len(text) * 0.9)
    train, test = text[:split], text[split:]
    print(f"Corpus: {len(text):,} characters, {len(vocab)} distinct characters. The first 90% is for training, the last 10% for testing.\n")

    # ------------------------------------------------------------ 1. Markov's question
    print("[1] Redoing Markov's experiment: does the next character depend on the one before it?")
    uni = Counter(train)
    bi = build_model(train, 2)
    total = sum(uni.values())
    for prev, nxt in [("t", "h"), ("c", "k"), ("x", "p"), ("q", "u")]:
        p_alone = uni[nxt] / total
        p_after = bi[prev][nxt] / max(sum(bi[prev].values()), 1)
        print(f"   Chance of '{nxt}': at a random spot {p_alone:6.2%}  →  right after '{prev}' {p_after:6.1%}"
              f"  ({p_after / p_alone:,.0f}x)")
    print("   If characters were independent of each other, the two probabilities would be about the same. A gap of 7x to 48x shows that text is a 'chain' with memory.")
    print("   When Markov counted the vowels and consonants of Onegin in 1913, he came to the same conclusion.\n")

    # ------------------------------------------------------------ 2. Continuing a text
    print("[2] Have n-gram models of different orders continue 'neural network' (every model uses the same random seed)")
    models = {n: build_model(train, n) for n in (1, 2, 3, 4, 6, 8)}
    for n, model in models.items():
        rng = random.Random(1913)
        out = generate(model, n, "neural network", 50, vocab, rng)
        hint = {1: "character frequencies only, no context", 2: "looks at the previous 1 character",
                3: "looks at the previous 2 characters", 4: "looks at the previous 3 characters",
                6: "looks at the previous 5 characters", 8: "looks at the previous 7 characters"}[n]
        print(f"   {n}-gram ({hint}): {out.replace(chr(10), ' ')}")
    print("   The bigger n is, the more fluent the text is locally; but the model is only splicing together pieces it has seen. It doesn't 'understand' anything.\n")

    # ------------------------------------------------------------ 3. Perplexity
    print("[3] Scoring on test text the models have never seen (perplexity, lower is better)")
    for n, model in models.items():
        ppl = perplexity(model, n, test[:3000], vocab)
        print(f"   {n}-gram: perplexity {ppl:7.1f}  {'█' * int(ppl)}")
    print(f"""   A perplexity of 10 means: each time the model predicts a character, it is, on average, like guessing blindly among 10 characters.
   From the 1-gram to the 4-gram the gains are big; after that they shrink fast, and from the 6-gram to the 8-gram the score barely moves:
   the longer the context, the smaller the chance that exactly the same context ever appeared in the training corpus (data sparsity).
   That is the ceiling of n-grams. Bengio's neural language model in 2003, and later the Transformer,
   replaced "rote memorization" with "vectors," so that a model can generalize from similar contexts.

Think about it:
  1. Swap the corpus for your own writing (say, chat logs or a diary). What style do the continuations take on?
  2. What would the 8-gram's perplexity be on the training set? Why is it so different from the test set? (Hint: overfitting)
  3. For an n-gram model that looks at the previous 1,000 characters, how much text would it need to have "seen" enough contexts?""")
