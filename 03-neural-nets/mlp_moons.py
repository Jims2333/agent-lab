"""
Lab 2: Train a neural network with the engine you wrote
=======================================================

Chapter 1's perceptron could only draw a straight line. The data here is two interlocking "moons,"
which no straight line can separate. The 1989 "universal approximation theorem" says that with enough
neurons in the hidden layer, a network can in theory approximate a boundary of any shape.
So we take the Value engine from Lab 1, train a small network from scratch, and see whether it can
draw a curved boundary.

Everything is pure-Python scalar arithmetic, so it's a bit slow (about half a minute). That is exactly
what lets you feel why GPUs and vectorization were needed later.

Run: python mlp_moons.py
"""

import math
import random

from autograd import Value

random.seed(1989)


# ---------------------------------------------------------------- data: two moons
def make_moons(n, noise=0.12):
    pts = []
    for i in range(n):
        t = math.pi * random.random()
        if i % 2 == 0:                                  # upper moon, label +1
            x, y, label = math.cos(t), math.sin(t), 1
        else:                                           # lower moon, label -1
            x, y, label = 1 - math.cos(t), 0.5 - math.sin(t), -1
        pts.append((x + random.gauss(0, noise), y + random.gauss(0, noise), label))
    return pts


# ---------------------------------------------------------------- network: neuron → layer → multi-layer perceptron
class Neuron:
    def __init__(self, n_in, act="relu"):
        # Initialization: scale by the number of inputs. ReLU "switches off" half the signal, so use the slightly larger scale Kaiming He proposed in 2015
        scale = math.sqrt(3 / n_in) if act == "tanh" else math.sqrt(6 / n_in)
        self.w = [Value(random.uniform(-1, 1) * scale) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act                                  # "relu", "tanh", or None (no activation on the output layer)

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.act == "relu":
            return z.relu()
        if self.act == "tanh":
            return z.tanh()
        return z

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, act):
        self.neurons = [Neuron(n_in, act) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, sizes, act="relu"):
        last = len(sizes) - 2                           # the last layer outputs a score, with no activation
        self.layers = [Layer(a, b, act=None if i == last else act)
                       for i, (a, b) in enumerate(zip(sizes, sizes[1:]))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def train(model, data, steps, lr, quiet=False):
    params = model.parameters()
    for step in range(steps + 1):
        scores = [model([x, y]) for x, y, _ in data]
        # hinge loss: if the point is classified correctly with enough margin, the loss is 0
        losses = [(1 - label * s).relu() for (_, _, label), s in zip(data, scores)]
        loss = sum(losses, Value(0.0)) * (1.0 / len(data))
        acc = sum((s.data > 0) == (label > 0) for (_, _, label), s in zip(data, scores)) / len(data)
        if not quiet and step % 10 == 0:
            print(f"   step {step:3d}  loss {loss.data:.4f}  accuracy {acc:.0%}")
        if step == steps:
            break

        for p in params:                       # 1. zero the gradients
            p.grad = 0.0
        loss.backward()                        # 2. backpropagation
        step_lr = lr * (1 - 0.9 * step / steps)   # the learning rate slowly shrinks (see Chapter 2)
        for p in params:                       # 3. gradient descent
            p.data -= step_lr * p.grad
    return acc


def draw(model, data, title):
    """ASCII decision boundary: the '+' region is classified as the upper moon, the '.' region as the lower moon."""
    print(title)
    xs = [-1.4 + i * 0.1 for i in range(36)]
    ys = [1.4 - j * 0.15 for j in range(16)]
    for y0 in ys:
        row = ""
        for x0 in xs:
            near = [lab for x, y, lab in data if abs(x - x0) < 0.05 and abs(y - y0) < 0.075]
            if near:
                row += "●" if near[0] > 0 else "○"
            else:
                row += "+" if model([x0, y0]).data > 0 else "."
        print("   " + row)
    print("   (● upper moon, ○ lower moon)\n")


data = make_moons(80)

print("[Control group] A network with no hidden layer (just a linear classifier, like Chapter 1's perceptron)")
linear = MLP([2, 1])
acc = train(linear, data, 80, lr=1.0, quiet=True)
print(f"   Accuracy after 80 training steps: {acc:.0%}")
draw(linear, data, "   All it can draw is a straight line:")

print("[Experimental group] Two hidden layers with 10 ReLU neurons each: 2 → 10 → 10 → 1")
random.seed(2015)
model = MLP([2, 10, 10, 1])
print(f"   {len(model.parameters())} parameters in total, and every one of them is a Value from Lab 1.")
acc = train(model, data, 80, lr=1.0)
draw(model, data, f"\n   The decision boundary after training (accuracy {acc:.0%}):")

print("""Think about it:
  1. Change the network to [2, 2, 1] (only 2 hidden neurons). Can it still separate the two moons?
  2. Switch the activation to tanh (MLP([2, 10, 10, 1], act="tanh")). How do the training speed and the boundary shape change?
  3. Change scale in Neuron to 1 / math.sqrt(n_in) (a smaller initialization). How far does accuracy drop?
  4. How many Value objects does this network create at every step? Why do real frameworks use matrix operations and GPUs?""")
