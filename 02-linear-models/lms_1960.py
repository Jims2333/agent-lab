"""
Lab 4: 1951–1960, one example at a time: stochastic gradient descent
====================================================================

The gradient descent so far goes through **all the data** for every single step.
If there are millions of data points, or the data keeps "streaming" in (like signals on a telephone line), that's far too slow.

In 1951, Robbins and Monro proved that if you take each step using a noisy "estimate", as long as the step size
slowly shrinks, you still get to the answer in the end.
In 1960, at Stanford, Widrow and Hoff proposed the LMS rule: **every time an example comes in, adjust the weights once
based on its error**. This is the forerunner of the "stochastic gradient descent" (SGD) used to train every neural network today.

The task: learn y = 3·x1 - 2·x2 + 1 from a stream of data (the data is noisy)

Run: python lms_1960.py
"""

import numpy as np

rng = np.random.default_rng(1960)
TRUE_W, TRUE_B = np.array([3.0, -2.0]), 1.0
N = 5000


def stream(n):
    X = rng.normal(size=(n, 2))
    y = X @ TRUE_W + TRUE_B + rng.normal(0, 0.5, n)
    return X, y


X, y = stream(N)


def error(w, b):
    """How far the parameters are from the true answer."""
    return np.sqrt(np.sum((w - TRUE_W) ** 2) + (b - TRUE_B) ** 2)


def batch_gd(seen_budget, lr=0.1):
    """Batch gradient descent: every step looks at all N examples."""
    w, b, seen = np.zeros(2), 0.0, 0
    while seen + N <= seen_budget:
        r = y - (X @ w + b)
        w += lr * 2 * (r @ X) / N
        b += lr * 2 * r.mean()
        seen += N
    return w, b


def lms(seen_budget, lr=0.02, decay=False, batch=1):
    """LMS / stochastic gradient descent: update once after looking at just `batch` examples."""
    w, b = np.zeros(2), 0.0
    for step, start in enumerate(range(0, seen_budget, batch), 1):
        xb, yb = X[start:start + batch], y[start:start + batch]
        r = yb - (xb @ w + b)                     # the error on these few examples
        eta = lr / (1 + step / 200) if decay else lr
        w += eta * (r @ xb) / len(r)              # the Widrow–Hoff rule: error × input
        b += eta * r.mean()
    return w, b


budgets = [100, 500, 1000, 5000]
methods = [  # (name, training function, how many examples one step needs at minimum)
    ("Batch GD (all 5,000 per step)", lambda s: batch_gd(s), N),
    ("LMS / SGD (1 at a time)", lambda s: lms(s), 1),
    ("SGD + shrinking learning rate", lambda s: lms(s, lr=0.05, decay=True), 1),
    ("Mini-batch SGD (32 at a time)", lambda s: lms(s, lr=0.2, batch=32), 32),
]

print("After 'seeing' the same number of examples, how far are the parameters from the true answer? (smaller is better)\n")
print(f"   {'Method':<34} |" + "".join(f"   seen {b:>5}" for b in budgets))
for name, fn, per_step in methods:
    cells = []
    for s in budgets:
        w, b = fn(s)
        cells.append("  no step yet" if s < per_step else f"     {error(w, b):8.3f}")
    print(f"   {name:<{34 - sum(1 for ch in name if ord(ch) > 127)}} |" + "".join(cells))

w, b = lms(N, lr=0.05, decay=True)
print(f"\n   What SGD learned: y = {w[0]:.2f}·x1 {w[1]:+.2f}·x2 {b:+.2f} (true: y = 3·x1 - 2·x2 + 1)")

# ---------------------------------------------------------------- Look at how the path jitters
print("\nLooking at 1 example at a time, the path 'jitters'. Here's how w1 changes over the last 10 steps:")
w_hist = []
w_, b_ = np.zeros(2), 0.0
for i in range(N):
    r = y[i] - (X[i] @ w_ + b_)
    w_ += 0.02 * r * X[i]
    b_ += 0.02 * r
    w_hist.append(w_[0])
print("   fixed learning rate:    ", " ".join(f"{v:.3f}" for v in w_hist[-10:]))
w_hist = []
w_, b_ = np.zeros(2), 0.0
for i in range(N):
    r = y[i] - (X[i] @ w_ + b_)
    eta = 0.05 / (1 + (i + 1) / 200)
    w_ += eta * r * X[i]
    b_ += eta * r
    w_hist.append(w_[0])
print("   shrinking learning rate:", " ".join(f"{v:.3f}" for v in w_hist[-10:]))

print("""
Takeaways:
  * Every step of batch gradient descent is "accurate", but every step is also "expensive": the more data, the slower each step.
  * Every SGD step is "cheap". Its direction is a bit off and it jitters, but it takes many steps, fast,
    and overall it gets close to the finish line early on.
  * When the learning rate gradually shrinks, the jitter shrinks with it (this is exactly the condition Robbins and Monro proved).
  * Mini-batches are a compromise between the two, and the standard way to train large models today:
    look at tens to thousands of examples at a time.

Compare it with the perceptron from Chapter 1:
  Perceptron rule: change only after a wrong guess, by a fixed amount (w += y·x)
  LMS rule:        change every time, in proportion to "how wrong" it was (w += η·error·x)

Think about it:
  1. Change the fixed learning rate from 0.02 to 0.5. What happens?
  2. Why is SGD the only option when the data "streams" in?""")
