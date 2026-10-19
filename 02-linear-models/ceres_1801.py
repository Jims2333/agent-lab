"""
Lab 1: 1801, using least squares to find the "lost planet" again
================================================================

On New Year's Day 1801, Piazzi discovered Ceres. After 41 days of tracking and about two dozen observations,
it vanished into the Sun's glare.
To find it again a year later, you have to take those few dozen **observations, each with some error**,
and work out the most believable orbit.

Here is a greatly simplified version:
  * assume the "planet's" angle in the sky changes at a constant rate: angle = a + b × day
  * observe on 24 randomly chosen nights within 41 days; every observation has a little error
  * the task: predict where it will be on day 364 (December 31)

Real orbit calculations are much more complicated (an elliptical orbit has 6 unknowns, and the Earth is moving too).
Here we keep only the core idea: **use many observations with errors to find the most believable parameters.**

Run: python ceres_1801.py
"""

import numpy as np

TRUE_A, TRUE_B = 51.5, 0.214     # the true starting angle (degrees) and how far it moves each day (degrees/day)
NOISE = 0.05                     # error of each observation (degrees)
N_OBS, LAST_DAY, TARGET_DAY = 24, 40, 364

rng = np.random.default_rng(1801)


def observe():
    days = np.sort(rng.choice(np.arange(LAST_DAY + 1), size=N_OBS, replace=False)).astype(float)
    angles = TRUE_A + TRUE_B * days + rng.normal(0, NOISE, N_OBS)
    return days, angles


def two_point_fit(t, y):
    """The most naive approach: use only the first and the last observation, and draw a straight line through them."""
    b = (y[-1] - y[0]) / (t[-1] - t[0])
    return y[0] - b * t[0], b


def least_squares_fit(t, y):
    """Least squares: find the straight line that minimizes the 'sum of squared errors' between all the observations and the line.
    Take the derivatives and set them to 0 to get two equations (the normal equations): a 2×2 system you can solve by hand."""
    n, st, sy = len(t), t.sum(), y.sum()
    stt, sty = (t * t).sum(), (t * y).sum()
    b = (n * sty - st * sy) / (n * stt - st * st)
    a = (sy - b * st) / n
    return a, b


# ---------------------------------------------------------------- One specific "history"
t, y = observe()
a2, b2 = two_point_fit(t, y)
a, b = least_squares_fit(t, y)
truth = TRUE_A + TRUE_B * TARGET_DAY

print(f"Piazzi's {N_OBS} observations (days 0–{LAST_DAY}), each with an error of about ±{NOISE}°")
print("   days:", " ".join(f"{d:.0f}" for d in t[:12]), "…")
print()

# Draw the observations and the least squares line in ASCII (so you can see them, the vertical axis shows the "deviation from the true line")
print("Deviation of the observations (●) and of the least squares line (─) from the true position:")
rows = np.linspace(0.12, -0.12, 9)
for r in rows:
    line = ""
    for day in range(LAST_DAY + 1):
        fit_dev = (a + b * day) - (TRUE_A + TRUE_B * day)
        obs = [yy - (TRUE_A + TRUE_B * tt) for tt, yy in zip(t, y) if tt == day]
        if obs and abs(obs[0] - r) < 0.0151:
            line += "●"
        elif abs(fit_dev - r) < 0.0151:
            line += "─"
        else:
            line += " "
    print(f"   {r:+.2f}° |{line}")
print("           " + "day 0" + " " * 30 + f"day {LAST_DAY}\n")

print("The least squares 'normal equations' (you can solve them by hand):")
print(f"   n = {len(t)},  Σt = {t.sum():.0f},  Σt² = {(t * t).sum():.0f},  Σy = {y.sum():.2f},  Σty = {(t * y).sum():.1f}")
print(f"   Solution: a = {a:.4f}°, b = {b:.5f}°/day")
lstsq = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), y, rcond=None)[0]
print(f"   Checked with numpy: a = {lstsq[0]:.4f}°, b = {lstsq[1]:.5f}°/day  ✓\n")

print(f"Predicted position on day {TARGET_DAY} (true position {truth:.2f}°):")
print(f"   Two-point method: {a2 + b2 * TARGET_DAY:.2f}°, error {abs(a2 + b2 * TARGET_DAY - truth):.2f}°")
print(f"   Least squares:    {a + b * TARGET_DAY:.2f}°, error {abs(a + b * TARGET_DAY - truth):.2f}°\n")

# ---------------------------------------------------------------- Replay history 2000 times
errs2, errs = [], []
for _ in range(2000):
    t, y = observe()
    a2, b2 = two_point_fit(t, y)
    a, b = least_squares_fit(t, y)
    errs2.append(abs(a2 + b2 * TARGET_DAY - truth))
    errs.append(abs(a + b * TARGET_DAY - truth))
errs2, errs = np.array(errs2), np.array(errs)

print("Replaying this history 2,000 times and tallying the prediction errors:")
print(f"   Two-point method: mean error {errs2.mean():.2f}°, within half a degree {np.mean(errs2 < 0.5):.0%} of the time")
print(f"   Least squares:    mean error {errs.mean():.2f}°, within half a degree {np.mean(errs < 0.5):.0%} of the time")
print("""
Using only two observations throws the other 22 away, and the errors get magnified by a "lever" stretched out to 364 days.
Least squares puts every observation to work, the errors cancel each other out, and the prediction is much steadier.

Think about it:
  1. Raise NOISE to 0.2. What does the share within half a degree become?
  2. Shorten the observation window from 41 days to 10 (LAST_DAY = 10). Why does the prediction get much worse?
  3. Why minimize the "sum of squared errors" and not the "sum of absolute errors"? (Hint: see the Concepts in the README)""")
