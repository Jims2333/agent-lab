"""
Lab 3: 1986, add a layer + backpropagation, and XOR is learned
==============================================================

In their 1986 *Nature* paper, Rumelhart, Hinton and Williams showed that
if you just add a "hidden layer" between the input and the output, then use backpropagation
to pass the error back layer by layer and work out how each weight should change,
the network can learn XOR on its own.

Here we don't use any deep learning framework, just numpy, written by hand:
  forward pass → compute the error → backpropagate to get the gradients → update the weights with gradient descent

Run: python backprop_1986.py
"""

import numpy as np

rng = np.random.default_rng(1986)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)          # XOR

HIDDEN = 3        # number of neurons in the hidden layer
LR = 1.0          # learning rate: how far each step goes along the gradient
EPOCHS = 10000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Random initialization: the weights of both layers
W1 = rng.normal(0, 1, (2, HIDDEN))
b1 = np.zeros(HIDDEN)
W2 = rng.normal(0, 1, (HIDDEN, 1))
b2 = np.zeros(1)

print("Training a tiny 2 → 3 → 1 network to learn XOR")
print("-" * 52)
for epoch in range(EPOCHS + 1):
    # ---------- 1. Forward pass: compute from the input, layer by layer, to the output ----------
    h = sigmoid(X @ W1 + b1)          # hidden layer output, shape (4, HIDDEN)
    out = sigmoid(h @ W2 + b2)        # network output, shape (4, 1)

    # ---------- 2. Error: mean squared error (the 1986 paper also used squared error) ----------
    loss = np.mean((out - Y) ** 2)

    if epoch % 1000 == 0:
        bar = "█" * int(loss * 160)
        print(f"Epoch {epoch:5d}  error {loss:.4f}  {bar}")

    # ---------- 3. Backpropagation: use the chain rule to compute gradients from the output backward ----------
    d_out = 2 * (out - Y) / len(X)            # ∂loss/∂out
    d_z2 = d_out * out * (1 - out)            # back through the output layer's sigmoid
    d_W2 = h.T @ d_z2
    d_b2 = d_z2.sum(axis=0)

    d_h = d_z2 @ W2.T                         # the error is "passed back" to the hidden layer
    d_z1 = d_h * h * (1 - h)                  # back through the hidden layer's sigmoid
    d_W1 = X.T @ d_z1
    d_b1 = d_z1.sum(axis=0)

    # ---------- 4. Gradient descent: nudge each weight a little in the direction that makes the error smaller ----------
    W2 -= LR * d_W2
    b2 -= LR * d_b2
    W1 -= LR * d_W1
    b1 -= LR * d_b1

print("-" * 52)
h = sigmoid(X @ W1 + b1)
out = sigmoid(h @ W2 + b2)
print("\nTraining done. Let's look at the results:")
print("   x1  x2 | answer   | output  | ok?")
for x, y, o in zip(X, Y, out):
    ok = "✓" if round(o[0]) == y[0] else "✗"
    print(f"    {x[0]:.0f}   {x[1]:.0f} |    {y[0]:.0f}     |  {o[0]:.3f}  |  {ok}")

# ---------- Opening the black box: what did the hidden layer learn? ----------
NAMES = {
    (0, 0, 0, 1): "x1 AND x2", (0, 1, 1, 1): "x1 OR x2",
    (1, 1, 1, 0): "NAND",      (1, 0, 0, 0): "NOR",
    (0, 1, 1, 0): "XOR",       (1, 0, 0, 1): "XNOR",
    (0, 0, 1, 1): "x1",        (0, 1, 0, 1): "x2",
    (1, 1, 0, 0): "NOT x1",    (1, 0, 1, 0): "NOT x2",
    (0, 0, 1, 0): "x1 AND NOT x2", (0, 1, 0, 0): "x2 AND NOT x1",
    (1, 1, 0, 1): "NOT x1 OR x2",  (1, 0, 1, 1): "x1 OR NOT x2",
    (0, 0, 0, 0): "always 0",  (1, 1, 1, 1): "always 1",
}
print("\nOpening the black box: how each hidden neuron responds to the 4 possible inputs")
print("   input:    (0,0)  (0,1)  (1,0)  (1,1)  → roughly computes")
for j in range(HIDDEN):
    acts = h[:, j]
    pattern = tuple(int(a > 0.5) for a in acts)
    vals = "  ".join(f"{a:.2f} " for a in acts)
    print(f"   neuron {j + 1}: {vals}  → {NAMES[pattern]}")

print("""
Nobody told the network which intermediate results to compute. It invented the intermediate concepts above
in its hidden layer on its own, then combined them into XOR in the output layer. For example, XOR = OR and NAND,
that is, "at least one of them is 1, but not both".
Learning useful intermediate representations on its own is what "learning representations" means,
and that is exactly the title of that 1986 paper.

Think about it:
  1. Change HIDDEN to 2 and try a few different random seeds. Does it sometimes fail to learn? (Hint: local minima)
  2. Change LR to 0.1 or 10. How does the error curve change?
  3. Remove the hidden layer's sigmoid (make it h = X @ W1 + b1). Can it still learn? Why?""")
