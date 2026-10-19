"""
Lab 2: 1847, Cauchy's "downhill method": gradient descent
=========================================================

Least squares has a ready-made formula, but when there are thousands of unknowns and the equations are nonlinear,
there's no formula to use. In 1847, Cauchy proposed a general method:
**wherever you're standing, take a small step in the steepest downhill direction, and keep doing it.**

Here we use Galton's real 1886 data (parent height → child height) to fit a straight line,
and watch gradient descent work its way, step by step, to the least squares answer. We'll also see:
  * what happens when the learning rate is too small, just right, or too big
  * why "centering" the data first makes gradient descent hundreds of times faster

Run: python gradient_descent_1847.py
"""

from pathlib import Path

import numpy as np

data = np.genfromtxt(Path(__file__).with_name("data") / "galton_1886.csv",
                     delimiter=",", skip_header=4)
x, y = data[:, 0], data[:, 1]          # x: average parent height, y: child height (inches)


def loss(a, b, x):
    return np.mean((y - (a + b * x)) ** 2)


def gradient(a, b, x):
    r = y - (a + b * x)                 # residual: actual value - predicted value
    return -2 * r.mean(), -2 * (r * x).mean()


def descend(x, lr, steps, a=0.0, b=0.0):
    path = [(a, b)]
    for _ in range(steps):
        ga, gb = gradient(a, b, x)
        a, b = a - lr * ga, b - lr * gb       # take a small step downhill
        path.append((a, b))
        if not np.isfinite(a) or abs(a) > 1e6:
            break
    return a, b, path


# The reference answer: the least squares closed-form solution
X = np.column_stack([np.ones_like(x), x])
a_best, b_best = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"Least squares reference answer: child height = {a_best:.2f} + {b_best:.3f} × average parent height\n")

# ---------------------------------------------------------------- Act 1: straight downhill
print("[Act 1] Walk downhill on the raw data (parent heights are between 64 and 73 inches)")
lam = np.linalg.eigvalsh(2 * X.T @ X / len(x))          # how steep the loss "bowl" is in its two directions
lr = 1.9 / lam.max()                                    # as large a learning rate as possible without diverging
a, b, _ = descend(x, lr, 5000)
print(f"   Learning rate {lr:.6f} (any bigger and it diverges), after 5,000 steps:")
print(f"   slope = {b:.3f} (reference answer {b_best:.3f}), loss {loss(a, b, x):.2f} (minimum {loss(a_best, b_best, x):.2f})")
print(f"   Why: this 'bowl' is {lam.max() / lam.min():,.0f} times steeper in one direction than in the other. It's a long, narrow canyon.")
print("   Big steps bang back and forth between the canyon walls; small steps barely move along the canyon floor.\n")

# ---------------------------------------------------------------- Act 2: centering
xc = x - x.mean()                                       # centering: make the average height the origin
lam_c = np.linalg.eigvalsh(2 * np.column_stack([np.ones_like(xc), xc]).T
                           @ np.column_stack([np.ones_like(xc), xc]) / len(xc))
print("[Act 2] First subtract the mean from the parent heights (centering), then walk downhill")
print(f"   Now one direction is only {lam_c.max() / lam_c.min():.1f} times steeper than the other: the 'canyon' has become a rounder 'bowl'.\n")

settings = [("too small", 0.02 / lam_c.max()), ("just right", 1.0 / lam_c.max()), ("too big", 2.05 / lam_c.max())]
print("   learning rate    |    step 1     step 5    step 20   step 100  (loss)")
for name, lr in settings:
    _, _, path = descend(xc, lr, 100)
    ls = [loss(*path[min(i, len(path) - 1)], xc) for i in (1, 5, 20, 100)]
    cells = "  ".join(f"{v:9.2f}" if v < 1e6 else " blown up" for v in ls)
    print(f"   {name:<10} {lr:.3f} | {cells}")

lr = 1.0 / lam_c.max()
a, b, path = descend(xc, lr, 100)
print(f"\n   With the learning rate just right, after 100 steps: slope = {b:.3f} (reference answer {b_best:.3f}) ✓")

# ---------------------------------------------------------------- Show where each step lands
print("\nWhere the slope is after each step (│ is the reference answer, ● is the current position):")
lo, hi, width = -1.2, 2.5, 46


def marker_line(value):
    cells = [" "] * width
    star = int((b_best - lo) / (hi - lo) * (width - 1))
    cells[star] = "│"
    if value < lo:
        cells[0] = "◀"
    elif value > hi:
        cells[-1] = "▶"
    else:
        cells[int((value - lo) / (hi - lo) * (width - 1))] = "●"
    return "".join(cells)


for name, factor in [("just right", 1.0), ("a bit big, but still converges", 1.8), ("too big, jumps farther and farther", 2.05)]:
    _, _, path = descend(xc, factor / lam_c.max(), 8)
    print(f"\n   Learning rate {name} ({factor / lam_c.max():.3f})")
    for step in (0, 1, 2, 3, 4, 8):
        print(f"   step {step} [{marker_line(path[step][1])}] slope {path[step][1]:+.3f}")

print("""
Think about it:
  1. The "too big" learning rate is only a tiny bit bigger than the "a bit big" one. Why does one converge and the other diverge?
  2. In Act 1, if you patiently took 1,000,000 steps, would you reach the reference answer?
  3. If you also divide by the standard deviation (standardize), how many times steeper is one direction than the other? How big can the learning rate be?""")
