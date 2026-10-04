"""
实验 2：1969 年，感知机撞上了 XOR 这堵墙
==========================================

Minsky 和 Papert 在《Perceptrons》一书中证明：单层感知机只能画"一条直线"
来分开两类东西。AND、OR 都能用一条直线分开，XOR（异或）却不行。

这个脚本用和实验 1 一模一样的感知机，分别去学 AND、OR、XOR，
你会看到前两个很快学会，XOR 无论训练多少轮都学不会。

运行：python xor_1969.py
"""

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
TASKS = {
    "AND": np.array([-1, -1, -1, +1]),   # 两个都是 1 才输出 1
    "OR":  np.array([-1, +1, +1, +1]),   # 有一个是 1 就输出 1
    "XOR": np.array([-1, +1, +1, -1]),   # 两个"不一样"才输出 1
}
MAX_EPOCHS = 1000


def accuracy(w, b, y):
    out = np.where(X @ w + b > 0, 1, -1)
    return np.mean(out == y)


def train_perceptron(y):
    """返回 (最好的 w, 最好的 b, 学会时的轮数或 None, 最高准确率, 是否出现循环)。"""
    w, b = np.zeros(2), 0.0
    best = (w.copy(), b, accuracy(w, b, y))
    seen = set()
    looped = False
    for epoch in range(1, MAX_EPOCHS + 1):
        mistakes = 0
        for x, t in zip(X, y):
            if t * (x @ w + b) <= 0:          # 猜错了（或者正好在线上）
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
    """在 [-0.5, 1.5] 的方格里画出感知机的判断：'+' 区域输出 1，'.' 区域输出 0。"""
    pts = {(0, 0): y[0], (0, 1): y[1], (1, 0): y[2], (1, 1): y[3]}
    for x2 in np.linspace(1.5, -0.5, 9):            # 每行 0.25，正好经过 0 和 1
        row = ""
        for x1 in np.linspace(-0.5, 1.5, 17):       # 每列 0.125，正好经过 0 和 1
            key = (round(float(x1), 3), round(float(x2), 3))
            if key in pts:
                row += "●" if pts[key] > 0 else "○"
            else:
                row += "+" if x1 * w[0] + x2 * w[1] + b > 0 else "."
        label = f"x2={x2:.0f} " if x2 in (0.0, 1.0) else "     "
        print("   " + label + row)
    print("        x1 从 -0.5 到 1.5（● 应该输出 1，○ 应该输出 0；+ 区域是感知机判为 1 的地方）\n")


for name, y in TASKS.items():
    print("=" * 64)
    w, b, epoch, best_acc, looped = train_perceptron(y)
    if epoch:
        print(f"{name}：第 {epoch} 轮就学会了！")
        print(f"   找到的直线：{w[0]:+.0f}·x1 {w[1]:+.0f}·x2 {b:+.0f} = 0")
    else:
        print(f"{name}：训练了 {MAX_EPOCHS} 轮还是没学会。")
        if looped:
            print("   而且权重在原地打转：转一圈又回到以前的状态，永远停不下来。")
        print(f"   它找到过的最好的直线，也只能分对 {best_acc:.0%} 的点：")
    draw(w, b, y)

print("=" * 64)
# 不靠学习规则，直接暴力乱试 10 万条随机直线，看看最好能分对几个点
rng = np.random.default_rng(1969)
lines = rng.normal(size=(100_000, 3))                     # 每行是 (w1, w2, b)
outs = np.where(X @ lines[:, :2].T + lines[:, 2] > 0, 1, -1)   # (4, 100000)
best_by_brute = (outs == TASKS["XOR"][:, None]).mean(axis=0).max()
print(f"再换个思路：随机乱画 10 万条直线去分 XOR，最好的一条也只能分对 {best_by_brute:.0%}。\n")

print("""为什么 XOR 学不会？

    x2
    1 |  ●(0,1)        ○(1,1)
      |
    0 |  ○(0,0)        ●(1,0)
      +-----------------------  x1
         0              1

两个 ● 在一条对角线上，两个 ○ 在另一条对角线上。
你试试看：在纸上画"一条直线"，让两个 ● 在一边、两个 ○ 在另一边，画不出来。
感知机的本质就是画一条直线（高维时是一个平面），所以无论怎么训练，
4 个点里最多只能分对 3 个。

1969 年这本书出版后，神经网络研究的经费和热情大幅退潮。
破局的办法要等到 1986 年才流行起来：多加一层，然后用反向传播训练 → 见实验 3。""")
