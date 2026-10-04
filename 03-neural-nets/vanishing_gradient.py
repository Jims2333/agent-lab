"""
实验 3：梯度消失与梯度爆炸
============================

1991 年，霍赫赖特（Hochreiter）在毕业论文里分析了一个让人头疼的现象：
网络一深，靠近输入那几层的梯度会变得极小（消失）或极大（爆炸），根本训练不动。

这里搭一个 30 层的网络，不训练，只做一次反向传播，看看误差信号传回到每一层时还剩多大。
比较不同的激活函数和初始化：
  * sigmoid：导数最大只有 0.25，每往回传一层就打一次折扣
  * ReLU + 太小的初始化：信号一层层变弱
  * ReLU + 何恺明初始化（2015）：每层的梯度大小基本不变
  * ReLU + 太大的初始化：梯度一层层放大，爆炸

运行：python vanishing_gradient.py
"""

import numpy as np

DEPTH, WIDTH, BATCH = 30, 64, 128

ACTS = {
    # 名字: (激活函数, 它的导数)
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z, a: a * (1 - a)),
    "tanh": (np.tanh, lambda z, a: 1 - a ** 2),
    "relu": (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(float)),
}

CONFIGS = [
    ("sigmoid，初始化标准差 1/√n", "sigmoid", 1.0),
    ("tanh，   初始化标准差 1/√n", "tanh", 1.0),
    ("ReLU，   初始化标准差 1/√n（偏小）", "relu", 1.0),
    ("ReLU，   何恺明初始化 √(2/n)", "relu", np.sqrt(2)),
    ("ReLU，   初始化标准差 1.5×√(2/n)（偏大）", "relu", 1.5 * np.sqrt(2)),
]


def layer_grad_norms(act_name, gain, seed=1991):
    rng = np.random.default_rng(seed)
    f, df = ACTS[act_name]
    Ws = [rng.normal(0, gain / np.sqrt(WIDTH), (WIDTH, WIDTH)) for _ in range(DEPTH)]

    # 前向：记下每一层的输入、加权和、输出
    h = rng.normal(size=(BATCH, WIDTH))
    cache = []
    for W in Ws:
        z = h @ W
        a = f(z)
        cache.append((h, z, a))
        h = a

    # 反向：假设损失对最后一层输出的梯度是"标准大小"的随机数，看它一层层传回去还剩多少
    g = rng.normal(size=(BATCH, WIDTH))
    top = np.linalg.norm(g)
    norms = [0.0] * DEPTH
    for i in reversed(range(DEPTH)):
        _, z, a = cache[i]
        gz = g * df(z, a)                       # 穿过激活函数：乘上它的导数
        g = gz @ Ws[i].T                        # 穿过权重：乘上权重矩阵，传给前一层
        norms[i] = np.linalg.norm(g) / top      # 传到第 i 层输入时，信号是原来的多少倍
    return norms


def bar(value, ref):
    """用对数刻度画条：每个 █ 代表 10 倍。"""
    if value == 0:
        return "（完全为 0）"
    k = np.log10(value / ref)
    if k >= 0:
        return "│" + "█" * min(int(round(k * 2)), 30) + f"  ×{value / ref:.1e}"
    return "█" * min(int(round(-k * 2)), 30) + f"│  ×{value / ref:.1e}"


print(f"一个 {DEPTH} 层、每层 {WIDTH} 个神经元的网络：误差信号从输出端传回来，到达各层时还剩多大")
print("（以输出端的信号为 ×1。往左的条表示变小了，往右表示变大了；每两格 = 10 倍）\n")

for title, act, gain in CONFIGS:
    norms = layer_grad_norms(act, gain)
    print(f"■ {title}")
    for i in (DEPTH - 1, 24, 19, 14, 9, 4, 0):
        b = bar(norms[i], 1.0)
        pad = 30 - (len(b.split("│")[0]) if "│" in b else 0)
        print(f"   第 {i + 1:2d} 层 {' ' * pad}{b}")
    ratio = norms[0]
    verdict = "梯度消失" if ratio < 1e-2 else ("梯度爆炸" if ratio > 1e2 else "基本稳定 ✓")
    print(f"   传到第 1 层时是原来的 {ratio:.1e} 倍  → {verdict}\n")

print("""为什么会这样？
  反向传播时，梯度每经过一层，都要乘上"这一层的权重"和"激活函数的导数"。
  * sigmoid 的导数最大是 0.25，30 层乘下来，就是好多个小于 1 的数连乘，越乘越小；
  * 初始化太小，每层的权重把信号缩小一点；太大，每层放大一点。30 次方之后都会走向极端；
  * ReLU 在正半轴导数恰好是 1，配合合适的初始化（何恺明初始化），每一层的信号大小正好保持不变。

想一想：
  1. 把 DEPTH 改成 5，sigmoid 的问题还严重吗？为什么早期的网络都很浅？
  2. 如果每一层都有一条"捷径"直接把输入加到输出上（残差连接，第四章会讲），梯度会怎样？
  3. 第一章实验 3 里用的是 sigmoid，为什么它没遇到这个问题？""")
