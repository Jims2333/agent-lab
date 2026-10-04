"""
实验 2：1847 年，柯西的"下山法"：梯度下降
==========================================

最小二乘有现成的公式，可当未知数成千上万、方程是非线性的时候，就没有公式可用了。
1847 年，柯西提出了一个通用的办法：**站在哪里，就朝最陡的下坡方向走一小步，反复走。**

这里用高尔顿 1886 年的真实数据（父母身高 → 子女身高）来拟合一条直线，
看看梯度下降怎么一步步走到最小二乘的答案，以及：
  * 学习率太小、刚好、太大，分别会怎样
  * 为什么先把数据"中心化"，梯度下降会快几百倍

运行：python gradient_descent_1847.py
"""

from pathlib import Path

import numpy as np

data = np.genfromtxt(Path(__file__).with_name("data") / "galton_1886.csv",
                     delimiter=",", skip_header=4)
x, y = data[:, 0], data[:, 1]          # x：父母平均身高，y：子女身高（英寸）


def loss(a, b, x):
    return np.mean((y - (a + b * x)) ** 2)


def gradient(a, b, x):
    r = y - (a + b * x)                 # 残差：真实值 - 预测值
    return -2 * r.mean(), -2 * (r * x).mean()


def descend(x, lr, steps, a=0.0, b=0.0):
    path = [(a, b)]
    for _ in range(steps):
        ga, gb = gradient(a, b, x)
        a, b = a - lr * ga, b - lr * gb       # 朝下坡方向走一小步
        path.append((a, b))
        if not np.isfinite(a) or abs(a) > 1e6:
            break
    return a, b, path


# 标准答案：最小二乘的公式解
X = np.column_stack([np.ones_like(x), x])
a_best, b_best = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"最小二乘的标准答案：子女身高 = {a_best:.2f} + {b_best:.3f} × 父母平均身高\n")

# ---------------------------------------------------------------- 第一幕：直接下山
print("【第一幕】直接用原始数据下山（父母身高在 64～73 英寸之间）")
lam = np.linalg.eigvalsh(2 * X.T @ X / len(x))          # 损失"碗"在两个方向上的陡峭程度
lr = 1.9 / lam.max()                                    # 不发散的前提下，尽量大的学习率
a, b, _ = descend(x, lr, 5000)
print(f"   学习率 {lr:.6f}（再大一点就会发散），走了 5000 步：")
print(f"   斜率 = {b:.3f}（标准答案 {b_best:.3f}），损失 {loss(a, b, x):.2f}（最小 {loss(a_best, b_best, x):.2f}）")
print(f"   原因：这个'碗'一个方向比另一个方向陡 {lam.max() / lam.min():,.0f} 倍，是一条又窄又长的峡谷。")
print("   步子大了会在峡谷两壁之间来回撞，步子小了沿着谷底几乎走不动。\n")

# ---------------------------------------------------------------- 第二幕：中心化
xc = x - x.mean()                                       # 中心化：以平均身高为原点
lam_c = np.linalg.eigvalsh(2 * np.column_stack([np.ones_like(xc), xc]).T
                           @ np.column_stack([np.ones_like(xc), xc]) / len(xc))
print("【第二幕】先把父母身高减去平均值（中心化），再下山")
print(f"   现在两个方向的陡峭程度只差 {lam_c.max() / lam_c.min():.1f} 倍，'峡谷'变成了一个圆一点的'碗'。\n")

settings = [("太小", 0.02 / lam_c.max()), ("刚好", 1.0 / lam_c.max()), ("太大", 2.05 / lam_c.max())]
print("   学习率  |  第 1 步    第 5 步    第 20 步   第 100 步  的损失")
for name, lr in settings:
    _, _, path = descend(xc, lr, 100)
    ls = [loss(*path[min(i, len(path) - 1)], xc) for i in (1, 5, 20, 100)]
    cells = "  ".join(f"{v:9.2f}" if v < 1e6 else "   爆炸了" for v in ls)
    print(f"   {name} {lr:.3f} | {cells}")

lr = 1.0 / lam_c.max()
a, b, path = descend(xc, lr, 100)
print(f"\n   学习率刚好时，100 步后：斜率 = {b:.3f}（标准答案 {b_best:.3f}）✓")

# ---------------------------------------------------------------- 画出每一步的位置
print("\n每一步之后，斜率走到了哪里（│ 是标准答案，● 是当前位置）：")
lo, hi, width = -1.2, 2.5, 46


def marker_line(value):
    cells = [" "] * width
    star = int((b_best - lo) / (hi - lo) * (width - 1))
    cells[star] = "│"
    if value < lo:
        cells[0] = "◀"
    elif value > hi:
        cells[-1] = "▶"
    else:
        cells[int((value - lo) / (hi - lo) * (width - 1))] = "●"
    return "".join(cells)


for name, factor in [("刚好", 1.0), ("偏大，但还能收敛", 1.8), ("太大，越跳越远", 2.05)]:
    _, _, path = descend(xc, factor / lam_c.max(), 8)
    print(f"\n   学习率{name}（{factor / lam_c.max():.3f}）")
    for step in (0, 1, 2, 3, 4, 8):
        print(f"   第 {step} 步 [{marker_line(path[step][1])}] 斜率 {path[step][1]:+.3f}")

print("""
想一想：
  1. "太大"的学习率只比"偏大"大了一点点，为什么一个收敛、一个发散？
  2. 第一幕里，如果耐心走 100 万步，能走到标准答案吗？
  3. 如果再除以标准差（标准化），陡峭程度会差几倍？学习率可以取多大？""")
