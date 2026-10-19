"""
Lab 2: 1969, the perceptron hits the XOR wall
=============================================

In their book *Perceptrons*, Minsky and Papert proved that a single-layer perceptron can only draw
"one straight line" to separate two kinds of things. AND and OR can both be separated by one straight line;
XOR (exclusive or) cannot.

This script uses exactly the same perceptron as Lab 1 to learn AND, OR and XOR in turn.
You'll see it learn the first two quickly, while XOR never gets learned, no matter how many epochs it trains.

Run: python xor_1969.py
"""

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
TASKS = {
    "AND": np.array([-1, -1, -1, +1]),   # output 1 only when both are 1
    "OR":  np.array([-1, +1, +1, +1]),   # output 1 if either one is 1
    "XOR": np.array([-1, +1, +1, -1]),   # output 1 only when the two are "different"
}
MAX_EPOCHS = 1000


def accuracy(w, b, y):
    out = np.where(X @ w + b > 0, 1, -1)
    return np.mean(out == y)


def train_perceptron(y):
    """Return (best w, best b, the epoch it learned in or None, best accuracy, whether it went in a loop)."""
    w, b = np.zeros(2), 0.0
    best = (w.copy(), b, accuracy(w, b, y))
    seen = set()
    looped = False
    for epoch in range(1, MAX_EPOCHS + 1):
        mistakes = 0
        for x, t in zip(X, y):
            if t * (x @ w + b) <= 0:          # a wrong guess (or exactly on the line)
                w += t * x
                b += t
                mistakes += 1
                acc = accuracy(w, b, y)
                if acc > best[2]:
                    best = (w.copy(), b, acc)
        if mistakes == 0:
            return w, b, epoch, 1.0, False
        state = (*w, b)
        if state in seen:
            looped = True
        seen.add(state)
    return best[0], best[1], None, best[2], looped


def draw(w, b, y):
    """Draw the perceptron's decision on a grid over [-0.5, 1.5]: '+' regions output 1, '.' regions output 0."""
    pts = {(0, 0): y[0], (0, 1): y[1], (1, 0): y[2], (1, 1): y[3]}
    for x2 in np.linspace(1.5, -0.5, 9):            # 0.25 per row, so it passes exactly through 0 and 1
        row = ""
        for x1 in np.linspace(-0.5, 1.5, 17):       # 0.125 per column, so it passes exactly through 0 and 1
            key = (round(float(x1), 3), round(float(x2), 3))
            if key in pts:
                row += "●" if pts[key] > 0 else "○"
            else:
                row += "+" if x1 * w[0] + x2 * w[1] + b > 0 else "."
        label = f"x2={x2:.0f} " if x2 in (0.0, 1.0) else "     "
        print("   " + label + row)
    print("        x1 runs from -0.5 to 1.5 (● should output 1, ○ should output 0; + marks where the perceptron says 1)\n")


for name, y in TASKS.items():
    print("=" * 64)
    w, b, epoch, best_acc, looped = train_perceptron(y)
    if epoch:
        print(f"{name}: learned it by epoch {epoch}!")
        print(f"   The line it found: {w[0]:+.0f}·x1 {w[1]:+.0f}·x2 {b:+.0f} = 0")
    else:
        print(f"{name}: still not learned after {MAX_EPOCHS} epochs.")
        if looped:
            print("   And the weights are going around in circles: after each loop they return to an earlier state, so it never stops.")
        print(f"   Even the best line it ever found gets only {best_acc:.0%} of the points right:")
    draw(w, b, y)

print("=" * 64)
# Forget the learning rule: brute-force 100,000 random lines and see how many points the best one gets right
rng = np.random.default_rng(1969)
lines = rng.normal(size=(100_000, 3))                     # each row is (w1, w2, b)
outs = np.where(X @ lines[:, :2].T + lines[:, 2] > 0, 1, -1)   # (4, 100000)
best_by_brute = (outs == TASKS["XOR"][:, None]).mean(axis=0).max()
print(f"Now try another approach: draw 100,000 random lines to split XOR. Even the best one gets only {best_by_brute:.0%} right.\n")

print("""Why can't it learn XOR?

    x2
    1 |  ●(0,1)        ○(1,1)
      |
    0 |  ○(0,0)        ●(1,0)
      +-----------------------  x1
         0              1

The two ● sit on one diagonal, and the two ○ sit on the other.
Try it yourself: on paper, draw "one straight line" with both ● on one side and both ○ on the other. You can't.
At heart, a perceptron just draws a straight line (a plane, in higher dimensions), so no matter how you train it,
it can get at most 3 of the 4 points right.

After this book came out in 1969, funding and enthusiasm for neural network research drained away.
The way out didn't catch on until 1986: add a layer, then train with backpropagation → see Lab 3.""")
