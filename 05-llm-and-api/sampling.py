"""
Lab 3: Temperature, top-k, top-p: how a model "picks" the next character
=========================================================================

At every step, a language model outputs a **probability distribution**: the next character is "e" with probability 12%, "a" with probability 8%, ...
Which character actually gets written is decided by the "sampling strategy." That is what the temperature and top_p parameters in an API adjust.

We demo it here with the 3-gram model trained in Lab 1:
  1. What the same distribution looks like at different temperatures
  2. Why greedy decoding tends to get "stuck" and repeat itself over and over
  3. How temperature and top-p change the style of a continuation

Run: python sampling.py
"""

import math
import random

from markov_1913 import build_model, generate, load_corpus, next_distribution


def pad(text, width):
    """Pad with spaces to a display width (a CJK character takes up two columns)."""
    return text + " " * max(width - sum(2 if ord(c) > 127 else 1 for c in text), 0)


def reshape(dist, temperature):
    logits = {ch: math.log(p) / temperature for ch, p in dist.items()}
    m = max(logits.values())
    ws = {ch: math.exp(v - m) for ch, v in logits.items()}
    s = sum(ws.values())
    return {ch: w / s for ch, w in ws.items()}


text = load_corpus()
vocab = sorted(set(text))
model = build_model(text, 3)

# ---------------------------------------------------------------- 1. Temperature changes the shape of the distribution
context = "th"
dist = next_distribution(model, context, 3, vocab)
print(f"[1] The probability distribution of the next character after '{context}' (top 6 only)")
for t in (0.5, 1.0, 2.0):
    d = reshape(dist, t)
    top = sorted(d.items(), key=lambda kv: -kv[1])[:6]
    cells = "  ".join(f"{ch!r}:{p:5.1%}" for ch, p in top)
    print(f"   temperature {t}: {cells}")
print("   Temperature < 1: the strong get stronger, the distribution gets 'sharper' and the model more cautious; temperature > 1: the distribution is 'flattened,' and rare characters get a chance too.\n")

# ---------------------------------------------------------------- 2. top-k and top-p
dist = next_distribution(model, "ne", 3, vocab)
ranked = sorted(dist.items(), key=lambda kv: -kv[1])
print("[2] top-k and top-p: first prune the obviously bad candidates, then draw")
print(f"   The most likely characters after 'ne': " + "  ".join(f"{ch!r}:{p:.1%}" for ch, p in ranked[:5]))
acc, k_needed = 0.0, 0
for ch, p in ranked:
    acc += p
    k_needed += 1
    if acc >= 0.9:
        break
print(f"   top-k = 5: draw only from the top 5")
print(f"   top-p = 0.9: add up the probabilities from the top down until they reach 90%. Here that takes the top {k_needed} characters.")
print("   The nice thing about top-p is that it 'scales itself': when the model is sure, there are few candidates; when it isn't, there are more.\n")

# ---------------------------------------------------------------- 3. Comparing continuations
print("[3] Same model, different sampling strategies, continuing 'attention':")
settings = [("greedy (temp 0)", 0.0, 1.0), ("temp 0.5", 0.5, 1.0), ("temp 1.0", 1.0, 1.0),
            ("temp 1.0 + top-p 0.9", 1.0, 0.9), ("temp 1.8", 1.8, 1.0)]
for name, temp, top_p in settings:
    rng = random.Random(1951)
    out = generate(model, 3, "attention", 45, vocab, rng, temperature=temp, top_p=top_p)
    print(f"   {pad(name, 22)}{out.replace(chr(10), ' ')}")

print("\n   A single run involves some luck. Run each setting 20 times and measure two things:")
print("   · distinct: what share of the continuation's 3-character pieces are unique (lower = more repetitive)")
print("   · novel: what share of the continuation's 3-character pieces never appear in the corpus (higher = more made up)")
print(f"   {pad('setting', 22)}| distinct | novel")
for name, temp, top_p in settings:
    distinct, novel = 0.0, 0.0
    for seed in range(20):
        out = generate(model, 3, "attention", 45, vocab, random.Random(seed), temperature=temp, top_p=top_p)
        grams = [out[i:i + 3] for i in range(len(out) - 2)]
        distinct += len(set(grams)) / len(grams) / 20
        novel += sum(g not in text for g in grams) / len(grams) / 20
    print(f"   {pad(name, 22)}|   {distinct:4.0%}   |  {novel:4.0%}")

print("""
   * Greedy decoding always picks the most likely character, so it easily walks into a loop it can't get out of and repeats the same words again and again;
   * Low temperature: it almost only copies pieces it has seen, safe but dull; high temperature: far more of the pieces are newly spliced together, colorful but gibberish;
   * A temperature around 1 plus top-p is usually the compromise between "some variety" and "not too wild," and many APIs' default settings are around there.

   When you use an API, you can choose like this: for code, math and information extraction, where you want stable answers, turn the temperature down; for stories and brainstorming, turn it up a little.
   (Some newer models don't let you change these parameters when thinking mode is on; check each provider's documentation.)

Think about it:
  1. Why does greedy decoding repeat itself? Can you find the reason in the distribution from part 1?
  2. Is temperature approaching 0 the same thing as top-k = 1?
  3. If you want the model to "give exactly the same answer every time" (say, for testing), how should you set things up?""")
