"""
实验 3：1986 年，多加一层 + 反向传播，学会 XOR
================================================

Rumelhart、Hinton、Williams 在 1986 年的《Nature》论文里展示：
只要在输入和输出之间加一层"隐藏层"，再用反向传播（backpropagation）
把误差一层层往回传、算出每个权重该怎么调，网络就能自己学会 XOR。

这里我们不用任何深度学习框架，只用 numpy 手写：
  前向传播 → 算误差 → 反向传播求梯度 → 梯度下降更新权重

运行：python backprop_1986.py
"""

import numpy as np

rng = np.random.default_rng(1986)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)          # XOR

HIDDEN = 3        # 隐藏层的神经元个数
LR = 1.0          # 学习率：每一步沿梯度走多远
EPOCHS = 10000


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 随机初始化：两层权重
W1 = rng.normal(0, 1, (2, HIDDEN))
b1 = np.zeros(HIDDEN)
W2 = rng.normal(0, 1, (HIDDEN, 1))
b2 = np.zeros(1)

print("训练一个 2 → 3 → 1 的小网络去学 XOR")
print("-" * 52)
for epoch in range(EPOCHS + 1):
    # ---------- 1. 前向传播：输入一层层算到输出 ----------
    h = sigmoid(X @ W1 + b1)          # 隐藏层输出，形状 (4, HIDDEN)
    out = sigmoid(h @ W2 + b2)        # 网络输出，形状 (4, 1)

    # ---------- 2. 误差：均方误差（1986 年论文用的也是平方误差）----------
    loss = np.mean((out - Y) ** 2)

    if epoch % 1000 == 0:
        bar = "█" * int(loss * 160)
        print(f"第 {epoch:5d} 轮  误差 {loss:.4f}  {bar}")

    # ---------- 3. 反向传播：链式法则，从输出往回算梯度 ----------
    d_out = 2 * (out - Y) / len(X)            # ∂loss/∂out
    d_z2 = d_out * out * (1 - out)            # 穿过输出层的 sigmoid
    d_W2 = h.T @ d_z2
    d_b2 = d_z2.sum(axis=0)

    d_h = d_z2 @ W2.T                         # 误差"传回"隐藏层
    d_z1 = d_h * h * (1 - h)                  # 穿过隐藏层的 sigmoid
    d_W1 = X.T @ d_z1
    d_b1 = d_z1.sum(axis=0)

    # ---------- 4. 梯度下降：每个权重往"让误差变小"的方向挪一小步 ----------
    W2 -= LR * d_W2
    b2 -= LR * d_b2
    W1 -= LR * d_W1
    b1 -= LR * d_b1

print("-" * 52)
h = sigmoid(X @ W1 + b1)
out = sigmoid(h @ W2 + b2)
print("\n训练完成，看看结果：")
print("   x1  x2 | 正确答案 | 网络输出 | 判断")
for x, y, o in zip(X, Y, out):
    ok = "✓" if round(o[0]) == y[0] else "✗"
    print(f"    {x[0]:.0f}   {x[1]:.0f} |    {y[0]:.0f}     |  {o[0]:.3f}  |  {ok}")

# ---------- 打开黑箱：隐藏层学到了什么？----------
NAMES = {
    (0, 0, 0, 1): "x1 AND x2", (0, 1, 1, 1): "x1 OR x2",
    (1, 1, 1, 0): "NAND",      (1, 0, 0, 0): "NOR",
    (0, 1, 1, 0): "XOR",       (1, 0, 0, 1): "XNOR",
    (0, 0, 1, 1): "x1",        (0, 1, 0, 1): "x2",
    (1, 1, 0, 0): "NOT x1",    (1, 0, 1, 0): "NOT x2",
    (0, 0, 1, 0): "x1 AND NOT x2", (0, 1, 0, 0): "x2 AND NOT x1",
    (1, 1, 0, 1): "NOT x1 OR x2",  (1, 0, 1, 1): "x1 OR NOT x2",
    (0, 0, 0, 0): "恒为 0",     (1, 1, 1, 1): "恒为 1",
}
print("\n打开黑箱：每个隐藏神经元对 4 种输入的反应")
print("   输入:    (0,0)  (0,1)  (1,0)  (1,1)   → 它大致在算")
for j in range(HIDDEN):
    acts = h[:, j]
    pattern = tuple(int(a > 0.5) for a in acts)
    vals = "  ".join(f"{a:.2f} " for a in acts)
    print(f"   神经元{j + 1}: {vals}  → {NAMES[pattern]}")

print("""
没有人告诉网络该算哪些中间结果。它自己在隐藏层里发明了上面这些中间概念，
再在输出层把它们组合成 XOR。比如 XOR = OR 且 NAND，也就是"至少有一个是 1，但不是两个都是 1"。
自己学出有用的中间表示，就是"学习表示"（learning representations），
这也正是 1986 年那篇论文的标题。

想一想：
  1. 把 HIDDEN 改成 2，换几个随机种子，会不会偶尔学不会？（提示：局部最小值）
  2. 把 LR 改成 0.1 或 10，误差曲线有什么变化？
  3. 去掉隐藏层的 sigmoid（改成 h = X @ W1 + b1），还能学会吗？为什么？""")
