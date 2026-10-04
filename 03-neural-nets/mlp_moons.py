"""
实验 2：用自己写的引擎训练一个神经网络
========================================

第一章的感知机只能画一条直线。这里的数据是两个交错的"月亮"，任何一条直线都分不开。
1989 年的"万能逼近定理"说：只要隐藏层的神经元够多，网络在理论上能逼近任何形状的边界。
我们就用实验 1 写的 Value 引擎，从零训练一个小网络，看它能不能画出弯曲的边界。

全部是纯 Python 标量运算，所以会慢一点（大约半分钟），这正好让你感受到：为什么后来需要 GPU 和向量化。

运行：python mlp_moons.py
"""

import math
import random

from autograd import Value

random.seed(1989)


# ---------------------------------------------------------------- 数据：两个月亮
def make_moons(n, noise=0.12):
    pts = []
    for i in range(n):
        t = math.pi * random.random()
        if i % 2 == 0:                                  # 上面的月亮，标签 +1
            x, y, label = math.cos(t), math.sin(t), 1
        else:                                           # 下面的月亮，标签 -1
            x, y, label = 1 - math.cos(t), 0.5 - math.sin(t), -1
        pts.append((x + random.gauss(0, noise), y + random.gauss(0, noise), label))
    return pts


# ---------------------------------------------------------------- 网络：神经元 → 层 → 多层感知机
class Neuron:
    def __init__(self, n_in, act="relu"):
        # 初始化：按输入个数缩放。ReLU 会"关掉"一半信号，所以用何恺明 2015 年提出的更大一点的尺度
        scale = math.sqrt(3 / n_in) if act == "tanh" else math.sqrt(6 / n_in)
        self.w = [Value(random.uniform(-1, 1) * scale) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act                                  # "relu"、"tanh"，或 None（输出层不加激活）

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
        last = len(sizes) - 2                           # 最后一层输出分数，不加激活
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
        # 合页损失（hinge loss）：分对了并且有足够把握，损失就是 0
        losses = [(1 - label * s).relu() for (_, _, label), s in zip(data, scores)]
        loss = sum(losses, Value(0.0)) * (1.0 / len(data))
        acc = sum((s.data > 0) == (label > 0) for (_, _, label), s in zip(data, scores)) / len(data)
        if not quiet and step % 10 == 0:
            print(f"   第 {step:3d} 步  损失 {loss.data:.4f}  准确率 {acc:.0%}")
        if step == steps:
            break

        for p in params:                       # 1. 梯度清零
            p.grad = 0.0
        loss.backward()                        # 2. 反向传播
        step_lr = lr * (1 - 0.9 * step / steps)   # 学习率慢慢变小（第二章讲过）
        for p in params:                       # 3. 梯度下降
            p.data -= step_lr * p.grad
    return acc


def draw(model, data, title):
    """ASCII 决策边界：'+' 区域判为上面的月亮，'.' 区域判为下面的月亮。"""
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
    print("   （● 上面的月亮，○ 下面的月亮）\n")


data = make_moons(80)

print("【对照组】没有隐藏层的网络（就是一个线性分类器，相当于第一章的感知机）")
linear = MLP([2, 1])
acc = train(linear, data, 80, lr=1.0, quiet=True)
print(f"   训练 80 步后准确率：{acc:.0%}")
draw(linear, data, "   它只能画一条直线：")

print("【实验组】两个隐藏层、每层 10 个 ReLU 神经元：2 → 10 → 10 → 1")
random.seed(2015)
model = MLP([2, 10, 10, 1])
print(f"   一共 {len(model.parameters())} 个参数，每一个都是实验 1 里的 Value。")
acc = train(model, data, 80, lr=1.0)
draw(model, data, f"\n   训练后的决策边界（准确率 {acc:.0%}）：")

print("""想一想：
  1. 把隐藏层改成 [2, 2, 1]（只有 2 个神经元），还能分开两个月亮吗？
  2. 把激活函数换成 tanh（MLP([2, 10, 10, 1], act="tanh")），训练速度和边界形状有什么不同？
  3. 把 Neuron 里的 scale 改成 1 / math.sqrt(n_in)（更小的初始化），准确率会掉到多少？
  4. 这个网络每走一步，要创建多少个 Value 对象？为什么真正的框架要用矩阵运算和 GPU？""")
