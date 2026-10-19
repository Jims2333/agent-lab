"""
Lab 1: Build an autodiff engine by hand
=======================================

In 1970, the Finnish student Linnainmaa came up with "reverse-mode automatic differentiation":
break a complicated computation into small steps, record how they connect (the computational graph),
then start from the result and **walk backward once**, using the chain rule to get the derivative of
the result with respect to every input in a single pass.

Today's PyTorch and TensorFlow are both built on this idea. Here we implement it ourselves in under
100 lines of code, following the same approach as micrograd, which Karpathy wrote in 2020.

Run: python autograd.py
(Other labs import the Value class from here, so the demo code lives in __main__ at the end of the file.)
"""

import math


class Value:
    """A scalar, plus: how it was computed (_prev, _op), and the derivative of the result with respect to it (grad)."""

    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0                 # d(final result)/d(me); only filled in after backprop
        self._prev = _prev              # which Values I was computed from
        self._op = _op                  # which operation was used
        self._backward = lambda: None   # how to pass my grad on to _prev
        self.label = label

    # ------------------------------------------------------------ basic operations
    # Every operation does two things: compute the result (forward), and define "how the derivative flows back" (backward)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                # addition: pass the derivative unchanged to both sides
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                # multiplication: each side gets multiplied by the other side's value
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant powers are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    # every other operation can be built from the ones above
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other ** -1
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return (-self) + other
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): return Value(other) * self ** -1

    # ------------------------------------------------------------ backpropagation
    def backward(self):
        """Starting from myself, call every node's _backward in reverse "topological order"."""
        # Depth-first search for the topological order: every node comes after its inputs.
        # An explicit stack instead of recursion, so even a very deep graph (say, a sum of a thousand terms)
        # won't hit Python's recursion limit.
        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            v, inputs_done = stack.pop()
            if inputs_done:
                order.append(v)
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            stack.extend((child, False) for child in v._prev if id(child) not in seen)

        self.grad = 1.0                 # d(myself)/d(myself) = 1
        for v in reversed(order):
            v._backward()

    def __repr__(self):
        return f"Value({self.label or ''}={self.data:.4f}, grad={self.grad:.4f})"


def draw_graph(root):
    """Draw the computational graph as an ASCII tree (looking from the result toward the inputs)."""
    lines = []

    def walk(v, prefix, is_last):
        name = v.label or v._op or "const"
        lines.append(f"{prefix}{'└─ ' if is_last else '├─ '}{name:<6} value={v.data:8.4f}   grad={v.grad:8.4f}")
        kids = list(v._prev)
        for i, k in enumerate(kids):
            walk(k, prefix + ("   " if is_last else "│  "), i == len(kids) - 1)
    walk(root, "", True)
    return "\n".join(lines)


def numerical_grad(f, xs, i, h=1e-6):
    """Numerical gradient: nudge input i a tiny bit and see how much the result changes. Slow, but never wrong, so we use it for checking."""
    up = list(xs); up[i] += h
    down = list(xs); down[i] -= h
    return (f(up) - f(down)) / (2 * h)


if __name__ == "__main__":
    # ---------------------------------------------------------------- 1. a small expression
    print("[1] A small expression: L = (a·b + c) · f")
    a, b, c, f = Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c"), Value(-2.0, label="f")
    e = a * b; e.label = "e=a·b"
    d = e + c; d.label = "d=e+c"
    L = d * f; L.label = "L=d·f"
    L.backward()
    print(draw_graph(L))
    print("""
   Checking by hand (chain rule):
     dL/dd = f = -2; dL/de = dL/dd × dd/de = -2 × 1 = -2
     dL/da = dL/de × de/da = -2 × b = -2 × (-3) = 6  ✓
     dL/db = dL/de × de/db = -2 × a = -2 × 2 = -4    ✓
""")

    # ---------------------------------------------------------------- 2. one neuron
    print("[2] One neuron: out = tanh(w1·x1 + w2·x2 + b), with a gradient check")
    vals = [2.0, 0.0, -3.0, 1.0, 6.8813735870195432]    # x1, x2, w1, w2, b

    def neuron_plain(v):                                # the same neuron, computed with plain floats
        x1, x2, w1, w2, bias = v
        return math.tanh(x1 * w1 + x2 * w2 + bias)

    inputs = [Value(t, label=n) for t, n in zip(vals, ["x1", "x2", "w1", "w2", "b"])]
    x1, x2, w1, w2, bias = inputs
    out = (x1 * w1 + x2 * w2 + bias).tanh()
    out.backward()
    print("   param  autodiff    numerical   difference")
    for i, v in enumerate(inputs):
        num = numerical_grad(neuron_plain, vals, i)
        print(f"   {v.label:<4} {v.grad:10.6f}   {num:10.6f}   {abs(v.grad - num):.1e}")
    print("   The two methods agree almost perfectly: our backpropagation is correct.\n")

    # ---------------------------------------------------------------- 3. why "reverse"?
    print("[3] Why work backward from the result?")
    n = 1000
    xs = [Value(1.0 + i / n) for i in range(n)]
    total = sum((x * x for x in xs), Value(0.0))
    total.backward()
    print(f"   A function with {n} inputs: f = x1² + x2² + … + x{n}²")
    print(f"   Reverse mode: 1 forward pass + 1 backward pass gives all {n} derivatives (for example, df/dx1 = {xs[0].grad:.3f})")
    print(f"   With numerical gradients or forward mode, every input needs its own pass: {n} passes or more.")
    print("   A neural network has millions to hundreds of billions of parameters but only 1 loss value, so reverse mode is the only practical choice.")

    print("""
Think about it:
  1. Why does _backward use += instead of =? (Hint: try b = a + a.)
  2. Add a sin operation to Value yourself, and use numerical_grad to check that it is correct.
  3. What goes wrong if backward() doesn't call _backward in topological order?""")
