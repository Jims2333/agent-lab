"""
实验 4：1951–1960 年，一次只看一个样本：随机梯度下降
======================================================

前面的梯度下降，每走一步都要把**全部数据**算一遍。
如果数据有几百万条，或者数据是源源不断"流"进来的（比如电话线上的信号），这样就太慢了。

1951 年，罗宾斯和门罗证明：用带噪声的"估计"来走每一步，只要步长慢慢变小，最后也能走到答案。
1960 年，维德罗和霍夫在斯坦福提出了 LMS 规则：**每来一个样本，就根据它的误差调整一次权重**。
这就是今天训练所有神经网络都在用的"随机梯度下降"（SGD）的前身。

任务：从一串数据里学出 y = 3·x1 - 2·x2 + 1（数据带噪声）

运行：python lms_1960.py
"""

import numpy as np

rng = np.random.default_rng(1960)
TRUE_W, TRUE_B = np.array([3.0, -2.0]), 1.0
N = 5000


def stream(n):
    X = rng.normal(size=(n, 2))
    y = X @ TRUE_W + TRUE_B + rng.normal(0, 0.5, n)
    return X, y


X, y = stream(N)


def error(w, b):
    """参数离真实答案有多远。"""
    return np.sqrt(np.sum((w - TRUE_W) ** 2) + (b - TRUE_B) ** 2)


def batch_gd(seen_budget, lr=0.1):
    """批量梯度下降：每一步都看全部 N 个样本。"""
    w, b, seen = np.zeros(2), 0.0, 0
    while seen + N <= seen_budget:
        r = y - (X @ w + b)
        w += lr * 2 * (r @ X) / N
        b += lr * 2 * r.mean()
        seen += N
    return w, b


def lms(seen_budget, lr=0.02, decay=False, batch=1):
    """LMS / 随机梯度下降：每次只看 batch 个样本，就更新一次。"""
    w, b = np.zeros(2), 0.0
    for step, start in enumerate(range(0, seen_budget, batch), 1):
        xb, yb = X[start:start + batch], y[start:start + batch]
        r = yb - (xb @ w + b)                     # 这几个样本的误差
        eta = lr / (1 + step / 200) if decay else lr
        w += eta * (r @ xb) / len(r)              # 维德罗-霍夫规则：误差 × 输入
        b += eta * r.mean()
    return w, b


budgets = [100, 500, 1000, 5000]
methods = [  # (名字, 训练函数, 每走一步至少要看多少个样本)
    ("批量梯度下降（每步看全部 5000 个）", lambda s: batch_gd(s), N),
    ("LMS / SGD（每次 1 个）", lambda s: lms(s), 1),
    ("SGD + 学习率逐渐变小", lambda s: lms(s, lr=0.05, decay=True), 1),
    ("小批量 SGD（每次 32 个）", lambda s: lms(s, lr=0.2, batch=32), 32),
]

print("同样只'看'了这么多个样本之后，参数离真实答案还有多远（越小越好）：\n")
print("   方法                               |" + "".join(f" 看了{b:>5}个" for b in budgets))
for name, fn, per_step in methods:
    cells = []
    for s in budgets:
        w, b = fn(s)
        cells.append("  还没走一步" if s < per_step else f"   {error(w, b):8.3f}")
    print(f"   {name:<{34 - sum(1 for ch in name if ord(ch) > 127)}} |" + "".join(cells))

w, b = lms(N, lr=0.05, decay=True)
print(f"\n   SGD 学到的：y = {w[0]:.2f}·x1 {w[1]:+.2f}·x2 {b:+.2f}（真实：y = 3·x1 - 2·x2 + 1）")

# ---------------------------------------------------------------- 看看路径的抖动
print("\n每次只看 1 个样本，走的路是'抖'的。看 w1 在最后 10 步里的变化：")
w_hist = []
w_, b_ = np.zeros(2), 0.0
for i in range(N):
    r = y[i] - (X[i] @ w_ + b_)
    w_ += 0.02 * r * X[i]
    b_ += 0.02 * r
    w_hist.append(w_[0])
print("   固定学习率：", " ".join(f"{v:.3f}" for v in w_hist[-10:]))
w_hist = []
w_, b_ = np.zeros(2), 0.0
for i in range(N):
    r = y[i] - (X[i] @ w_ + b_)
    eta = 0.05 / (1 + (i + 1) / 200)
    w_ += eta * r * X[i]
    b_ += eta * r
    w_hist.append(w_[0])
print("   逐渐变小：  ", " ".join(f"{v:.3f}" for v in w_hist[-10:]))

print("""
结论：
  * 批量梯度下降每一步都很"准"，但每一步都很"贵"：数据越多，每一步越慢。
  * SGD 每一步都很"便宜"，方向有点不准、会抖，但走得多、走得快，总体上早早就到了终点附近。
  * 学习率逐渐变小，抖动也跟着变小（这正是罗宾斯和门罗证明的条件）。
  * 小批量是两者的折中，也是今天训练大模型的标准做法：一次看几十到几千个样本。

和第一章的感知机比一比：
  感知机规则：只有猜错了才改，改的幅度固定（w += y·x）
  LMS 规则：  每次都改，改的幅度和"错了多少"成正比（w += η·误差·x）

想一想：
  1. 把固定学习率从 0.02 调到 0.5，会发生什么？
  2. 为什么数据是"流"进来的时候，只能用 SGD？""")
