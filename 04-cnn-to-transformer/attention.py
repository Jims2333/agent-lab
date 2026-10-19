"""
Lab 2: Attention by hand
========================

In 2014, Bahdanau let a translation model "look back" at the most relevant words of the source
sentence every time it produced a word. In 2017, the Transformer took the idea all the way: the whole
network runs on attention alone, with no recurrence at all.

The attention formula is a single line:  output = softmax(Q·Kᵀ / √d) · V
Here we take it apart in four steps:
  1. Think of attention as a "fuzzy dictionary lookup"
  2. Self-attention on a sentence: let "it" find "kitten"
  3. Multi-head attention + a causal mask: one head handles "what does this refer to," another "the previous word"
  4. Why divide by √d

Run: python attention.py
"""

import numpy as np

np.random.seed(2017)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # subtract the max so exp doesn't overflow
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention. Q: (n, d), K: (m, d), V: (m, dv). Returns the output and the attention weights."""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])         # how well each query "matches" each key
    if mask is not None:
        scores = np.where(mask, scores, -1e9)        # masked positions get a hugely negative score
    weights = softmax(scores, axis=-1)               # turn them into weights that add up to 1
    return weights @ V, weights                      # weighted average of the "values"


def pad(text, width):
    """Pad with spaces by display width (a CJK character takes two columns), so mixed text still lines up."""
    shown = sum(2 if ord(ch) > 127 else 1 for ch in text)
    return text + " " * max(width - shown, 0)


def shade(v):
    for limit, ch in ((0.05, " "), (0.25, "░"), (0.5, "▒"), (0.75, "▓")):
        if v < limit:
            return ch
    return "█"


def heatmap(w, labels, title):
    print(title)
    print("   " + pad("", 5) + "looks at →" + " ".join(f"{i:>2}" for i in range(len(labels))))
    for i, row in enumerate(w):
        print(f"   {i:>2} {pad(labels[i], 7)}     " + " ".join(shade(v) * 2 for v in row))
    print("   (" + "  ".join(f"{i}:{t}" for i, t in enumerate(labels)) + "; ░ a little ▒ more ▓ a lot █ almost all)\n")


# ---------------------------------------------------------------- 1. fuzzy dictionary lookup
print("[1] Attention = a fuzzy dictionary lookup")
features = ["sweet", "sour", "red", "yellow"]
keys = np.array([[0.8, 0.2, 1, 0],      # apple
                 [0.0, 1.0, 0, 1],      # lemon
                 [0.9, 0.0, 0, 1],      # banana
                 [0.9, 0.3, 1, 0]])     # strawberry
names = ["apple", "lemon", "banana", "strawberry"]
values = np.array([[5.0], [3.0], [4.0], [12.0]])     # price per pound of each fruit
query = np.array([[1.0, 0.0, 1.0, 0.0]])             # what I want: "something sweet and red"
out, w = attention(query * 3, keys * 3, values)      # ×3 makes the differences in match stronger
print("   The dictionary's 'keys' are the fruits' features (sweet, sour, red, yellow); the 'values' are prices.")
print("   Query: 'How much is a sweet, red fruit?'")
for n, wi in zip(names, w[0]):
    print(f"     {pad(n + ':', 12)}weight {wi:.2f} {'█' * int(wi * 40)}")
print(f"   A normal dictionary can only look up one exact key; attention gives every key a weight by similarity, and the answer is the weighted average: ${out[0, 0]:.2f}")
print("   Apples and strawberries are both 'sweet and red', so they get most of the weight.\n")

# ---------------------------------------------------------------- 2. self-attention: let "it" find "kitten"
tokens = ["The", "kitten", "chases", "the", "yarn", ";", "it", "runs", "fast"]
n = len(tokens)
IT = tokens.index("it")                  # position of the pronoun "it"
#                  animal object action pronoun function-word
semantic = np.array([[0, 0, 0, 0, 1],   # The
                     [1, 0, 0, 0, 0],   # kitten
                     [0, 0, 1, 0, 0],   # chases
                     [0, 0, 0, 0, 1],   # the
                     [0, 1, 0, 0, 0],   # yarn
                     [0, 0, 0, 0, 1],   # ;
                     [0, 0, 0, 1, 0],   # it
                     [0, 0, 1, 0, 0],   # runs
                     [0, 0, 1, 0, 0]],  # fast
                    dtype=float)
position = np.eye(n)                     # position information: word i has a 1 in dimension i
X = np.hstack([semantic, position])      # each word's vector = semantic features + position features
d = X.shape[1]

# Head A (coreference head): a pronoun's "query" looks for an animal's "key"
Wq_a = np.zeros((d, 4)); Wq_a[3, 0] = 4.0      # pronoun → sends "I'm looking for an animal" in dimension 0
Wk_a = np.zeros((d, 4)); Wk_a[0, 0] = 4.0      # animal → answers "I am an animal" in dimension 0
Wv_a = np.eye(d)[:, :5]                          # value: pass the semantic features through unchanged
out_a, w_a = attention(X @ Wq_a, X @ Wk_a, X @ Wv_a)

print("[2] Self-attention: every word in the sentence 'looks at' all the other words")
print("   Sentence: The kitten chases the yarn; it runs fast")
print(f"   How 'it' spreads its attention over the words:")
for t, wi in zip(tokens, w_a[IT]):
    print(f"     {pad(t, 7)}{wi:.2f} {'█' * int(wi * 30)}")
print(f"   In the output vector for 'it', the 'animal' dimension = {out_a[IT, 0]:.2f}: it picked up the message 'I refer to an animal' from 'kitten'.")
print("   (Wq and Wk here are designed by hand; in a real model, they are learned.)\n")

# ---------------------------------------------------------------- 3. multi-head + causal mask
# Head B (previous-word head): the query at position i looks for the key at position i-1
shift = np.zeros((d, n))
for i in range(1, n):
    shift[5 + i, i - 1] = 1.0                    # query at position i → dimension i-1
Wq_b = shift * 6
Wk_b = np.zeros((d, n)); Wk_b[5:, :] = np.eye(n) * 6   # key at position j → dimension j
causal = np.tril(np.ones((n, n), dtype=bool))  # causal mask: each word may only see itself and the words before it
_, w_b = attention(X @ Wq_b, X @ Wk_b, X @ Wv_a, mask=causal)

print("[3] Multi-head attention: several heads work at once, each watching a different relationship")
heatmap(w_a, tokens, "   Head A (coreference head): the 'it' row focuses on 'kitten'; the other words have nothing to look for, so they look at all words evenly")
heatmap(w_b, tokens, "   Head B (previous-word head, with a causal mask): every word looks at the word just before it; the upper-right corner is empty")
print("   The causal mask lets each word see only the words before it, so when the model generates text it can't 'peek at the answer'.")
print("   Concatenate the two heads' outputs, and every word knows both 'who I refer to' and 'who comes before me'.")
print("   Without the position features, head B couldn't work at all: attention itself doesn't know word order, so a Transformer has to add positional encoding.\n")

# ---------------------------------------------------------------- 4. why divide by √d
print("[4] Why divide by √d?")
print("    dim d | avg max weight, unscaled | divided by √d")
for dim in (4, 64, 512):
    raw, scaled = [], []
    for _ in range(200):
        q = np.random.randn(1, dim)
        k = np.random.randn(16, dim)
        s = q @ k.T
        raw.append(softmax(s).max())
        scaled.append(softmax(s / np.sqrt(dim)).max())
    print(f"   {dim:6d} |           {np.mean(raw):.2f}           |   {np.mean(scaled):.2f}")
print("""   The higher the dimension, the bigger the dot products, and the closer softmax gets to "pick just one." The gradient is then almost 0, and the model can't learn.
   Dividing by √d pulls the scores back into a normal range, so attention stays "selective, but not extreme."

Think about it:
  1. Change the query in part [1] to "sour and yellow." What price do you get?
  2. Remove the causal mask from head B. Where does word 0 (The) look? Why?
  3. For a sentence of n words, how many scores does self-attention compute? If the sentence doubles in length, how many times more computation is that?""")
