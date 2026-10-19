"""
Lab 4: Putting together a complete Transformer
==============================================

The first three labs were the parts. Here we assemble them into the architecture from the 2017 paper:
  word embeddings + positional encoding → [multi-head self-attention → feed-forward network]×N
  (every block has residual connections and layer normalization) → predict the next word

The weights are random and untrained, so the predictions mean nothing. The point of this lab is to see clearly:
  1. The shape of the data at every layer
  2. Where all the parameters go
  3. Why residual connections let a network be stacked very deep (echoing Chapter 3's vanishing gradients)

Run: python transformer_block.py
"""

import numpy as np

rng = np.random.default_rng(2017)

VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, SEQ = 1000, 64, 4, 256, 2, 10
D_HEAD = D_MODEL // N_HEADS


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x, eps=1e-5):
    """Layer normalization: adjust each word's vector to mean 0 and variance 1, so the numbers stay in a stable range."""
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


def positional_encoding(n, d):
    """The sinusoidal positional encoding from the 2017 paper: each dimension uses a wave of a different frequency, and together they mark every position uniquely."""
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / (10000 ** (2 * i / d))
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe


def init(*shape):
    return rng.normal(0, 1 / np.sqrt(shape[0]), shape)


def make_block():
    return {"Wq": init(D_MODEL, D_MODEL), "Wk": init(D_MODEL, D_MODEL),
            "Wv": init(D_MODEL, D_MODEL), "Wo": init(D_MODEL, D_MODEL),
            "W1": init(D_MODEL, D_FF), "W2": init(D_FF, D_MODEL)}


def multi_head_attention(x, p, verbose=False):
    n = x.shape[0]
    q = (x @ p["Wq"]).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)   # (head, word, dim per head)
    k = (x @ p["Wk"]).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
    v = (x @ p["Wv"]).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(D_HEAD)                 # (head, word, word)
    scores = np.where(np.tril(np.ones((n, n), bool)), scores, -1e9)      # causal mask
    out = softmax(scores) @ v                                            # (head, word, dim per head)
    out = out.transpose(1, 0, 2).reshape(n, D_MODEL)                     # concatenate the heads back together
    if verbose:
        print(f"      split into {N_HEADS} heads: Q/K/V shape {q.shape}, attention scores {scores.shape}")
    return out @ p["Wo"]


def block(x, p, residual=True, verbose=False):
    a = multi_head_attention(layer_norm(x), p, verbose)                  # 1. self-attention: words exchange information
    x = x + a if residual else a
    f = np.maximum(layer_norm(x) @ p["W1"], 0) @ p["W2"]                 # 2. feed-forward network: each word "thinks it over" on its own
    x = x + f if residual else f
    return x


# ---------------------------------------------------------------- 1. one forward pass
embed = init(VOCAB, D_MODEL) * np.sqrt(VOCAB) / np.sqrt(D_MODEL)
unembed = init(D_MODEL, VOCAB)
blocks = [make_block() for _ in range(N_LAYERS)]
tokens = rng.integers(0, VOCAB, SEQ)

print("[1] A sentence (10 words) flows through the Transformer; the shape at every step:")
print(f"   Input: the IDs of 10 words {tokens.tolist()}")
x = embed[tokens]
print(f"   Word embedding: look up each word's vector               → {x.shape} (words × model dimension)")
x = x + positional_encoding(SEQ, D_MODEL)
print(f"   Add positional encoding: tells the model the word order  → {x.shape}")
for i, p in enumerate(blocks):
    print(f"   Transformer block {i + 1}:")
    x = block(x, p, verbose=True)
    print(f"      after attention + feed-forward + residual connections → {x.shape} (same shape, updated content)")
logits = layer_norm(x) @ unembed
probs = softmax(logits[-1])
print(f"   Output layer: scores {VOCAB} words at each position         → {logits.shape}")
print(f"   The last position predicts the 'next word': most likely word #{probs.argmax()}, probability {probs.max():.4f}")
print(f"   (It is untrained, so the probability is close to the uniform 1/{VOCAB} = {1 / VOCAB:.4f}; training is about making the correct next word more likely.)\n")

# ---------------------------------------------------------------- 2. where the parameters are
print("[2] Where do all the parameters go?")
n_embed = VOCAB * D_MODEL * 2
n_attn = 4 * D_MODEL * D_MODEL
n_ffn = 2 * D_MODEL * D_FF
print(f"   Embeddings + output layer:         {n_embed:>9,}")
print(f"   Attention per block (Q, K, V, O):  {n_attn:>9,}  × {N_LAYERS} blocks")
print(f"   Feed-forward network per block:    {n_ffn:>9,}  × {N_LAYERS} blocks")
print(f"   Total:                             {n_embed + N_LAYERS * (n_attn + n_ffn):>9,}")
print("   The feed-forward networks have 2 times the parameters of attention: in large models, most of the 'knowledge' lives in the feed-forward networks.")
print("   Grow the dimension from 64 to over 10,000 and stack the layers from 2 to around 100, and the parameter count reaches the hundreds of billions.\n")

# ---------------------------------------------------------------- 3. residual connections
def similarity(x):
    """The average cosine similarity between word vectors at different positions. 1.0 means every word has become identical."""
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    S = xn @ xn.T
    return (S.sum() - len(x)) / (len(x) * (len(x) - 1))


print("[3] Stacking 48 layers: with residual connections vs. without")
print("   How similar the different words' vectors are in each layer's output (1.00 = every word has become the same vector and all information is lost):")
deep = [make_block() for _ in range(48)]
for p in deep:                     # GPT-2's trick: scale down the residual branches' output weights by 1/√(2×number of layers)
    p["Wo"] *= 1 / np.sqrt(2 * 48)
    p["W2"] *= 1 / np.sqrt(2 * 48)
x0 = rng.normal(0, 1, (VOCAB, D_MODEL))[tokens] + positional_encoding(SEQ, D_MODEL)
for residual in (True, False):
    x = x0.copy()
    marks = [f"input: {similarity(x):.2f}"]
    for i, p in enumerate(deep, 1):
        x = block(x, p, residual)
        if i in (1, 6, 24, 48):
            marks.append(f"layer {i:2d}: {similarity(x):.2f}")
    print(f"   {'with residuals   ' if residual else 'without residuals'} | " + "  ".join(marks))
print("""   Without residual connections, after a few layers every word's vector becomes identical (researchers call this "rank collapse"),
   so the model can no longer tell which word is which, and the gradient can't flow back either.
   With residual connections (x = x + f(x)), each layer only "adds a small edit" on top of what was there, and the input's information has a direct highway through.
   This is exactly the secret that let Kaiming He and colleagues train a 152-layer residual network (ResNet) in 2015, and the Transformer inherited it.

Think about it:
  1. Change N_HEADS from 4 to 8. Does the parameter count change? Why?
  2. Without positional encoding, can the model still tell "the cat chases the dog" from "the dog chases the cat"?
  3. Why does the output layer only need the result at the last position to predict the next word?""")
