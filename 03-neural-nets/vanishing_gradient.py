"""
Lab 3: Vanishing and exploding gradients
========================================

In 1991, Hochreiter analyzed a troublesome phenomenon in his diploma thesis:
once a network gets deep, the gradients in the layers near the input become tiny (vanish) or huge
(explode), and the network simply can't be trained.

Here we build a 30-layer network, skip training, and do just one backward pass to see how big the
error signal still is when it gets back to each layer. We compare different activation functions
and initializations:
  * sigmoid: its derivative is at most 0.25, so the signal gets discounted at every layer on the way back
  * ReLU + too small an initialization: the signal gets weaker layer by layer
  * ReLU + He initialization (2015): the gradient stays about the same size at every layer
  * ReLU + too large an initialization: the gradient grows layer by layer and explodes

Run: python vanishing_gradient.py
"""

import numpy as np

DEPTH, WIDTH, BATCH = 30, 64, 128

ACTS = {
    # name: (activation function, its derivative)
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z, a: a * (1 - a)),
    "tanh": (np.tanh, lambda z, a: 1 - a ** 2),
    "relu": (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(float)),
}

CONFIGS = [
    ("sigmoid, init std 1/√n", "sigmoid", 1.0),
    ("tanh,    init std 1/√n", "tanh", 1.0),
    ("ReLU,    init std 1/√n (too small)", "relu", 1.0),
    ("ReLU,    He init √(2/n)", "relu", np.sqrt(2)),
    ("ReLU,    init std 1.5×√(2/n) (too large)", "relu", 1.5 * np.sqrt(2)),
]


def layer_grad_norms(act_name, gain, seed=1991):
    rng = np.random.default_rng(seed)
    f, df = ACTS[act_name]
    Ws = [rng.normal(0, gain / np.sqrt(WIDTH), (WIDTH, WIDTH)) for _ in range(DEPTH)]

    # forward: record each layer's input, weighted sum and output
    h = rng.normal(size=(BATCH, WIDTH))
    cache = []
    for W in Ws:
        z = h @ W
        a = f(z)
        cache.append((h, z, a))
        h = a

    # backward: assume the gradient of the loss w.r.t. the last layer's output is random noise of "standard size", and see how much is left as it travels back layer by layer
    g = rng.normal(size=(BATCH, WIDTH))
    top = np.linalg.norm(g)
    norms = [0.0] * DEPTH
    for i in reversed(range(DEPTH)):
        _, z, a = cache[i]
        gz = g * df(z, a)                       # through the activation function: multiply by its derivative
        g = gz @ Ws[i].T                        # through the weights: multiply by the weight matrix, pass to the previous layer
        norms[i] = np.linalg.norm(g) / top      # at layer i's input, how many times its original size the signal is
    return norms


def bar(value, ref):
    """Draw a bar on a log scale: every two █ stand for 10×."""
    if value == 0:
        return "(exactly 0)"
    k = np.log10(value / ref)
    if k >= 0:
        return "│" + "█" * min(int(round(k * 2)), 30) + f"  ×{value / ref:.1e}"
    return "█" * min(int(round(-k * 2)), 30) + f"│  ×{value / ref:.1e}"


print(f"A {DEPTH}-layer network with {WIDTH} neurons per layer: as the error signal travels back from the output, how big is it when it reaches each layer?")
print("(The signal at the output is ×1. A bar to the left means it shrank, a bar to the right means it grew; every two blocks = 10×.)\n")

for title, act, gain in CONFIGS:
    norms = layer_grad_norms(act, gain)
    print(f"■ {title}")
    for i in (DEPTH - 1, 24, 19, 14, 9, 4, 0):
        b = bar(norms[i], 1.0)
        pad = 30 - (len(b.split("│")[0]) if "│" in b else 0)
        print(f"   layer {i + 1:2d} {' ' * pad}{b}")
    ratio = norms[0]
    verdict = "vanishing gradients" if ratio < 1e-2 else ("exploding gradients" if ratio > 1e2 else "roughly stable ✓")
    print(f"   By layer 1 it is {ratio:.1e} times its original size  → {verdict}\n")

print("""Why does this happen?
  During backpropagation, every time the gradient passes through a layer, it gets multiplied by "this layer's weights" and "the derivative of the activation function."
  * The sigmoid's derivative is at most 0.25, so over 30 layers you multiply lots of numbers smaller than 1 together, and the product keeps shrinking;
  * Too small an initialization, and each layer's weights shrink the signal a little; too large, and each layer grows it a little. Raised to the 30th power, both go to extremes;
  * ReLU's derivative is exactly 1 on the positive side, so with the right initialization (He initialization), the signal keeps just the same size at every layer.

Think about it:
  1. Change DEPTH to 5. Is the sigmoid problem still serious? Why were early networks all shallow?
  2. If every layer had a "shortcut" that adds its input straight to its output (residual connections, coming in Chapter 4), what would happen to the gradient?
  3. Chapter 1's Lab 3 used sigmoid. Why didn't it run into this problem?""")
