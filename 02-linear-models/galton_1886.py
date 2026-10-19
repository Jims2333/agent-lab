"""
Lab 3: 1886, Galton discovers "regression"
==========================================

This is Galton's real data from back then: the heights (in inches) of 928 adult children from 205 families.
"Parent height" is the average of the father and the mother (the mother's height is first multiplied by 1.08,
and so are the daughters' heights; this is how Galton did it).

Let's recreate his discovery:
  * children of very tall parents are tall on average too, but not as tall as their parents
  * children of very short parents are short on average too, but not as short as their parents
  * all of them "go back" some of the way toward the average, which is where the word "regression" comes from

Run: python galton_1886.py
"""

from pathlib import Path

import numpy as np

data = np.genfromtxt(Path(__file__).with_name("data") / "galton_1886.csv",
                     delimiter=",", skip_header=4)
parent, child = data[:, 0], data[:, 1]
print(f"{len(child)} children in all. Mean parent height (the father–mother average): {parent.mean():.1f}; mean child height: {child.mean():.1f}\n")

# ---------------------------------------------------------------- Group by parent height
print("The children's average height, grouped by parent height:")
print("   parents  | kids | kids avg | vs. parents   | pulled back toward the mean")
mean = parent.mean()
for h in sorted(set(parent)):
    kids = child[parent == h]
    if len(kids) < 10:
        continue
    diff = kids.mean() - h
    # Too close to the mean, this ratio gets blown up by noise, so we skip it
    pulled = 1 - (kids.mean() - mean) / (h - mean) if abs(h - mean) >= 1.5 else None
    pulled_txt = f"{pulled:5.0%}" if pulled is not None else "   —"
    arrow = "↓ shorter" if diff < 0 else "↑ taller "
    print(f"   {h:7.1f}  | {len(kids):4d} |  {kids.mean():6.1f}  | {arrow} {abs(diff):.1f} | {pulled_txt}")

# ---------------------------------------------------------------- Least squares fit
b, a = np.polyfit(parent, child, 1)
print(f"\nLeast squares fit: child height ≈ {a:.1f} + {b:.2f} × parent height")
print(f"Slope {b:.2f}: when the parents are 3 inches above average, their children are, on average, only {3 * b:.1f} inches above average. "
      "Galton's own estimate back then was about 2/3.\n")

# ---------------------------------------------------------------- Scatter plot
print("Scatter plot (the denser the character, the more people at that spot):")
px = np.arange(64, 74)
cy = np.arange(74, 61, -1)
for row in cy:
    line = ""
    for col in px:
        n = np.sum((np.abs(parent - col - 0.25) <= 0.5) & (np.abs(child - row) <= 0.5))
        line += f" {' .:-=+*#%@'[min(n // 3, 9)] if n else ' '} "
    print(f"   child {row} |{line}")
print("      parent " + "".join(f"{c:^3d}" for c in px))
print("   The cloud is a tilted ellipse, but flatter than the 45° line: for each extra inch of parent height,"
      " children are on average only about 0.65 inches taller.")

# ---------------------------------------------------------------- The other way around
zc = (child - child.mean()) / child.std()
zp = (parent - parent.mean()) / parent.std()
print(f"""
Now flip it around: if you use "the child's height" to predict "the parents' height", do you get "tall children have even taller parents"?
Switch to standard scores (how many standard deviations from the mean) and look:
   Predicting children from parents: slope {np.polyfit(zp, zc, 1)[0]:.2f}
   Predicting parents from children: slope {np.polyfit(zc, zp, 1)[0]:.2f}
Both directions are less than 1, and exactly the same (both equal the correlation coefficient between the two).
So "regression to the mean" isn't some mysterious hereditary force. It's a statistical phenomenon:
as long as two quantities aren't perfectly correlated, when you use an extreme value of one to predict the other,
the prediction always lands closer to the mean.

Think about it:
  1. A basketball player has an amazing year and makes a magazine cover; the next year, his performance drops. Does this prove the "cover jinx"?
  2. Students who do especially badly on one exam often improve on the next one. Is that necessarily thanks to extra tutoring?""")
