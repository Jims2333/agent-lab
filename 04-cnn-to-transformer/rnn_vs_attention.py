"""
Lab 3: Why attention beat recurrent networks
============================================

A recurrent neural network (RNN) reads a sentence the way a person reads a book: one word after
another, keeping its memory in a "hidden state." The problem: for information from word 1 to reach
word 100, it has to be passed along 99 times, and every pass discounts it a little (the vanishing
gradients of Chapter 3).

The 1997 LSTM added a "conveyor belt" that lets information travel forward almost without loss;
the 2017 Transformer simply lets every word see all the other words **directly**, so no matter how
far apart they are, they are only one step away.

Here we compare the three:
  1. How word 1's influence on word t changes with distance
  2. How many steps a sentence takes when computed in order (can it run in parallel?)

Run: python rnn_vs_attention.py
"""

import numpy as np

rng = np.random.default_rng(1997)
H = 32                                   # dimension of the hidden state

# ---------------------------------------------------------------- 1. influence decays with distance
# plain RNN: h_t = tanh(W·h_{t-1} + U·x_t)
W = rng.normal(0, 1, (H, H))
W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))   # set W's "amplification factor" to 0.9 so it doesn't explode right away
U = rng.normal(0, 0.5, (H, H))


def rnn_influence(T, trials=20):
    """How sensitive the hidden state at step T is to the hidden state at step 1 (the norm of the Jacobian)."""
    total = 0.0
    for _ in range(trials):
        h = np.zeros(H)
        J = np.eye(H)                               # dh_t / dh_1, accumulated starting from the identity matrix
        for t in range(T):
            h = np.tanh(W @ h + U @ rng.normal(size=H))
            if t > 0:
                J = (np.diag(1 - h ** 2) @ W) @ J   # chain rule: multiply by "this step's derivative" at every step
        total += np.linalg.norm(J, 2)
    return total / trials


def lstm_influence(T, forget=0.99):
    """The LSTM memory cell is c_t = f·c_{t-1} + …; along this conveyor belt, the influence is only multiplied by the forget gate f."""
    return forget ** (T - 1)


print("[1] How much influence does word 1 still have on word t? (1.0 = fully preserved)")
print("   dist t |    plain RNN   | LSTM (forget≈0.99) | attention (path length)")
for T in (2, 5, 10, 20, 50, 100):
    r = rnn_influence(T)
    l = lstm_influence(T)
    print(f"   {T:5d}  | {r:12.2e}   |    {l:10.3f}      |  always just 1 step")
print("""
   A plain RNN multiplies by a matrix "smaller than 1" at every step, so after a few dozen steps the influence is negligible: it "forgets" the start of the sentence.
   On the LSTM's memory conveyor belt, there is only the forget gate to multiply by: once the forget gate learns to stay close to 1, information can travel hundreds of steps (Hochreiter's "constant error carousel").
   In attention, every pair of words has a direct connection. How much information gets through depends on the attention weights, not on distance.
""")

# ---------------------------------------------------------------- 2. sequential vs. parallel computation
print("[2] Processing a sentence of n words: how many steps must run one after another? How many multiplications in total? (64-dimensional word vectors)")
print("      n    | RNN steps | attention steps | RNN multiplications | attention multiplications")
D = 64
for n in (128, 512, 2048):
    rnn_mults = n * 2 * D * D                 # each step: two matrix-vector products, h·Wh and x·Wx
    attn_mults = 3 * n * D * D + 2 * n * n * D   # computing Q/K/V, plus the n×n scores and the weighted sums
    print(f"   {n:6d}  |  {n:6d}   |        1        | {rnn_mults:19,} | {attn_mults:25,}")
print("""
   An RNN's step t has to wait for step t-1 to finish, so the steps can only queue up and go one at a time;
   attention actually does more computation in total (it grows with n²), but all positions can be computed at once, in a few big matrix multiplications.
   What GPUs do best is exactly this: huge amounts of parallel computation at once. So on a GPU, attention ends up much faster.
   This is the key reason the Transformer could scale models and data up by several orders of magnitude.

   The price is n²: double the sentence length, and attention's computation becomes nearly 4 times as large.
   That is also why the "context window" has always been an important limit of large models (more in Chapter 5).

Think about it:
  1. Change W's amplification factor from 0.9 to 1.1. What happens to the plain RNN column?
  2. If the forget gate were 0.9 instead of 0.99, how much would the LSTM have left after 100 steps?
  3. If attention is so good, why was Bahdanau's 2014 attention still built on top of an RNN?""")
