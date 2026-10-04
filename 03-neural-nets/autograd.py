"""
实验 1：手写一个自动求导引擎
==============================

1970 年，芬兰学生林纳因马（Linnainmaa）想出了"反向模式自动微分"：
把一个复杂的计算拆成一步步的小运算，记下它们之间的连接（计算图），
然后从结果出发**倒着走一遍**，用链式法则一次算出结果对所有输入的导数。

今天的 PyTorch、TensorFlow 都建立在这个想法上。这里用不到 100 行代码亲手实现它，
和 Karpathy 2020 年写的 micrograd 是同一个思路。

运行：python autograd.py
（其他实验会 import 这里的 Value 类，所以演示代码放在文件末尾的 __main__ 里）
"""

import math


class Value:
    """一个标量，外加：它是怎么算出来的（_prev, _op），以及结果对它的导数（grad）。"""

    def __init__(self, data, _prev=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0                 # d(最终结果)/d(我)，反向传播之后才有值
        self._prev = _prev              # 我是由哪些 Value 算出来的
        self._op = _op                  # 用的什么运算
        self._backward = lambda: None   # 怎么把我的 grad 传给 _prev
        self.label = label

    # ------------------------------------------------------------ 基本运算
    # 每个运算都做两件事：算出结果（前向），并定义"导数怎么往回传"（反向）
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                # 加法：导数原样分给两边
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                # 乘法：交换着乘对方的值
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "只支持常数次幂"
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

    # 其余运算都可以用上面几个拼出来
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other ** -1
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return (-self) + other
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): return Value(other) * self ** -1

    # ------------------------------------------------------------ 反向传播
    def backward(self):
        """从自己出发，按"拓扑顺序"的反方向，把每个节点的 _backward 调一遍。"""
        # 深度优先排出拓扑顺序：保证每个节点都排在它的输入之后。
        # 用显式的栈而不是递归，计算图再深（比如一千项相加）也不会超出 Python 的递归上限。
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

        self.grad = 1.0                 # d(自己)/d(自己) = 1
        for v in reversed(order):
            v._backward()

    def __repr__(self):
        return f"Value({self.label or ''}={self.data:.4f}, grad={self.grad:.4f})"


def draw_graph(root):
    """把计算图画成一棵 ASCII 树（从结果往输入看）。"""
    lines = []

    def walk(v, prefix, is_last):
        name = v.label or v._op or "常数"
        lines.append(f"{prefix}{'└─ ' if is_last else '├─ '}{name:<6} 值={v.data:8.4f}   梯度={v.grad:8.4f}")
        kids = list(v._prev)
        for i, k in enumerate(kids):
            walk(k, prefix + ("   " if is_last else "│  "), i == len(kids) - 1)
    walk(root, "", True)
    return "\n".join(lines)


def numerical_grad(f, xs, i, h=1e-6):
    """数值梯度：把第 i 个输入挪一点点，看结果变多少。慢，但不会错，用来检查。"""
    up = list(xs); up[i] += h
    down = list(xs); down[i] -= h
    return (f(up) - f(down)) / (2 * h)


if __name__ == "__main__":
    # ---------------------------------------------------------------- 1. 一个小算式
    print("【1】一个小算式：L = (a·b + c) · f")
    a, b, c, f = Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c"), Value(-2.0, label="f")
    e = a * b; e.label = "e=a·b"
    d = e + c; d.label = "d=e+c"
    L = d * f; L.label = "L=d·f"
    L.backward()
    print(draw_graph(L))
    print("""
   手算验证（链式法则）：
     dL/dd = f = -2；dL/de = dL/dd × dd/de = -2 × 1 = -2
     dL/da = dL/de × de/da = -2 × b = -2 × (-3) = 6  ✓
     dL/db = dL/de × de/db = -2 × a = -2 × 2 = -4    ✓
""")

    # ---------------------------------------------------------------- 2. 一个神经元
    print("【2】一个神经元：out = tanh(w1·x1 + w2·x2 + b)，并做梯度检查")
    vals = [2.0, 0.0, -3.0, 1.0, 6.8813735870195432]    # x1, x2, w1, w2, b

    def neuron_plain(v):                                # 同一个神经元，用普通浮点数算
        x1, x2, w1, w2, bias = v
        return math.tanh(x1 * w1 + x2 * w2 + bias)

    inputs = [Value(t, label=n) for t, n in zip(vals, ["x1", "x2", "w1", "w2", "b"])]
    x1, x2, w1, w2, bias = inputs
    out = (x1 * w1 + x2 * w2 + bias).tanh()
    out.backward()
    print("   参数   自动求导     数值梯度     差别")
    for i, v in enumerate(inputs):
        num = numerical_grad(neuron_plain, vals, i)
        print(f"   {v.label:<4} {v.grad:10.6f}   {num:10.6f}   {abs(v.grad - num):.1e}")
    print("   两种方法几乎完全一致：说明我们的反向传播写对了。\n")

    # ---------------------------------------------------------------- 3. 为什么要"反向"
    print("【3】为什么要从结果往回算？")
    n = 1000
    xs = [Value(1.0 + i / n) for i in range(n)]
    total = sum((x * x for x in xs), Value(0.0))
    total.backward()
    print(f"   一个有 {n} 个输入的函数：f = x1² + x2² + … + x{n}²")
    print(f"   反向模式：前向算 1 遍 + 反向算 1 遍，就得到全部 {n} 个导数（比如 df/dx1 = {xs[0].grad:.3f}）")
    print(f"   如果用数值方法或前向模式：每个输入都要单独算一遍，共 {n} 遍以上。")
    print("   神经网络有几百万到几千亿个参数，但只有 1 个损失值，所以反向模式是唯一可行的选择。")

    print("""
想一想：
  1. 为什么 _backward 里用的是 +=，而不是 =？（提示：试试 b = a + a）
  2. 自己给 Value 加一个 sin 运算，并用 numerical_grad 检查它对不对。
  3. backward() 里如果不按拓扑顺序调用 _backward，会出什么错？""")
