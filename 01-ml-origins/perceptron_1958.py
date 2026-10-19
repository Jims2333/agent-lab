"""
Lab 1: Recreating Rosenblatt's 1958 perceptron demo
===================================================

In July 1958, Rosenblatt gave a demo in Washington: after seeing 50 punched cards, an IBM 704
learned to tell cards "marked on the left" from cards "marked on the right".

Here we simulate the same thing with numpy:
  * a "card" = a 20×20 grid of pixels (the Mark I Perceptron had exactly 20×20 = 400 photocells)
  * each card has a small square, either in the left half or in the right half, plus a sprinkle of noise
  * the perceptron has just 400 weights + 1 bias, and its rule fits in one sentence: **change only when wrong**

Run: python perceptron_1958.py
"""

import numpy as np

SIZE = 20          # a 20×20 "retina"
BLOCK = 4          # side length of the marker square
NOISE = 0.03       # each pixel has a 3% chance of lighting up at random (a dirty card)
LEFT, RIGHT = -1, +1

rng = np.random.default_rng(1958)


def make_card(side):
    """Make one card: the square is in the left half when side=LEFT, in the right half when side=RIGHT."""
    card = (rng.random((SIZE, SIZE)) < NOISE).astype(float)
    row = rng.integers(0, SIZE - BLOCK + 1)
    if side == LEFT:
        col = rng.integers(0, SIZE // 2 - BLOCK + 1)
    else:
        col = rng.integers(SIZE // 2, SIZE - BLOCK + 1)
    card[row:row + BLOCK, col:col + BLOCK] = 1.0
    return card


def make_deck(n):
    sides = rng.choice([LEFT, RIGHT], size=n)
    cards = np.stack([make_card(s) for s in sides])
    return cards.reshape(n, -1), sides          # flatten each card into a 400-dimensional vector


def show_card(flat, title):
    print(title)
    grid = flat.reshape(SIZE, SIZE)
    for r in range(SIZE):
        print("   " + "".join("█" if v else "·" for v in grid[r]))
    print()


def show_weights(w):
    """Draw the 400 weights back as a 20×20 grid: '+' means "if this spot is lit, vote right", '-' means vote left."""
    grid = w.reshape(SIZE, SIZE)
    t = np.abs(w).max() * 0.15
    print("Learned weight map (+ votes 'right', - votes 'left', blank ≈ 0):")
    for r in range(SIZE):
        line = "".join("+" if v > t else "-" if v < -t else " " for v in grid[r])
        print("   |" + line + "|")
    print()


def predict(w, b, x):
    return RIGHT if x @ w + b > 0 else LEFT


def train(w, b, xs, ys, max_epochs=20):
    """Go through the same deck again and again until a whole epoch has no mistakes."""
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for x, y in zip(xs, ys):
            if predict(w, b, x) != y:
                # Rosenblatt's learning rule: on a wrong guess, "add" this card into the weights, or "subtract" it out
                w += y * x
                b += y
                mistakes += 1
        print(f"   Epoch {epoch} ({len(xs):,} cards seen): mistakes {mistakes:3d}")
        if mistakes == 0:
            print("   A whole epoch without a single mistake: it has learned this deck.")
            break
    return w, b


def exam(w, b):
    acc = np.mean([predict(w, b, x) == y for x, y in zip(test_x, test_y)])
    print(f"   Exam on 1,000 new cards it has never seen: accuracy {acc:.1%}\n")


# 400 weights, matching the 400 motor-driven potentiometers on the Mark I
w = np.zeros(SIZE * SIZE)
b = 0.0
test_x, test_y = make_deck(1000)

# ---------------------------------------------------------------- Act 1: 50 cards
train_x, train_y = make_deck(50)  # just like the 1958 demo: 50 trials
show_card(train_x[0], f"The first training card (answer: {'right' if train_y[0] == RIGHT else 'left'}):")

print("[Act 1] Just like in 1958: show it only 50 cards")
w, b = train(w, b, train_x, train_y)
exam(w, b)
show_weights(w)

# ---------------------------------------------------------------- Act 2: more data
more_x, more_y = make_deck(1000)
print("[Act 2] Now show it 1,000 more cards")
w, b = train(w, b, more_x, more_y)
exam(w, b)
show_weights(w)

print("""Getting every training card right is not the same as getting every new card right.
With only 50 cards, any spot where the square never happened to appear still has a weight of 0,
so a little noise easily makes it guess wrong. The more cards it sees, the cleaner the weight map
and the better the exam score. That's "generalization", and it's also what the ImageNet story
later on is about: data matters.

Think about it:
  1. Why is the weight map '-' on the left and '+' on the right? Nobody told it the concept of "left" and "right".
  2. What happens if you set NOISE to 0.3?
  3. If the task became "is the square in the top half or the bottom half?", would the learning rule need to change?""")
