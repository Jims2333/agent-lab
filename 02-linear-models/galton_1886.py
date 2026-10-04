"""
实验 3：1886 年，高尔顿发现的"回归"
====================================

这是高尔顿当年真实的数据：205 个家庭、928 个成年子女的身高（单位：英寸）。
"父母身高"是父亲和母亲的平均（母亲的身高先乘以 1.08，女儿的也一样，这是高尔顿的做法）。

我们来重现他的发现：
  * 父母很高的，孩子平均也高，但没有父母那么高
  * 父母很矮的，孩子平均也矮，但没有父母那么矮
  * 都往平均值"退回"了一些，这就是"回归"（regression）这个词的来历

运行：python galton_1886.py
"""

from pathlib import Path

import numpy as np

data = np.genfromtxt(Path(__file__).with_name("data") / "galton_1886.csv",
                     delimiter=",", skip_header=4)
parent, child = data[:, 0], data[:, 1]
print(f"共 {len(child)} 个孩子。父母平均身高的均值 {parent.mean():.1f}，孩子身高的均值 {child.mean():.1f}\n")

# ---------------------------------------------------------------- 按父母身高分组看
print("按父母身高分组，看孩子的平均身高：")
print("   父母身高 | 人数 | 孩子平均 | 比父母   | 往平均值退回了")
mean = parent.mean()
for h in sorted(set(parent)):
    kids = child[parent == h]
    if len(kids) < 10:
        continue
    diff = kids.mean() - h
    # 离平均值太近时，这个比例会被噪声放大，就不算了
    pulled = 1 - (kids.mean() - mean) / (h - mean) if abs(h - mean) >= 1.5 else None
    pulled_txt = f"{pulled:5.0%}" if pulled is not None else "   —"
    arrow = "↓ 矮了" if diff < 0 else "↑ 高了"
    print(f"   {h:7.1f}  | {len(kids):4d} |  {kids.mean():6.1f}  | {arrow} {abs(diff):.1f} | {pulled_txt}")

# ---------------------------------------------------------------- 最小二乘拟合
b, a = np.polyfit(parent, child, 1)
print(f"\n最小二乘拟合：孩子身高 ≈ {a:.1f} + {b:.2f} × 父母身高")
print(f"斜率 {b:.2f}：父母比平均高 3 英寸，孩子平均只比平均高 {3 * b:.1f} 英寸。高尔顿当年估计约为 2/3。\n")

# ---------------------------------------------------------------- 散点图
print("散点图（字符越密，那个位置的人越多）：")
px = np.arange(64, 74)
cy = np.arange(74, 61, -1)
for row in cy:
    line = ""
    for col in px:
        n = np.sum((np.abs(parent - col - 0.25) <= 0.5) & (np.abs(child - row) <= 0.5))
        line += f" {' .:-=+*#%@'[min(n // 3, 9)] if n else ' '} "
    print(f"   孩子 {row} |{line}")
print("           父母 " + "".join(f"{c:^3d}" for c in px))
print("   点云是一个斜着的椭圆，但比 45° 线平：父母每高 1 英寸，孩子平均只高约 0.65 英寸。")

# ---------------------------------------------------------------- 反过来看
zc = (child - child.mean()) / child.std()
zp = (parent - parent.mean()) / parent.std()
print(f"""
反过来想：如果用"孩子的身高"去预测"父母的身高"，会不会出现"孩子高的，父母更高"？
换成标准分（离平均值有几个标准差）来看：
   用父母预测孩子：斜率 {np.polyfit(zp, zc, 1)[0]:.2f}
   用孩子预测父母：斜率 {np.polyfit(zc, zp, 1)[0]:.2f}
两个方向都小于 1，而且一模一样（都等于两者的相关系数）。
所以"回归平均值"不是什么神秘的遗传力量，而是一个统计现象：
只要两个量不是完全相关，极端的一方去预测另一方，结果总会更靠近平均值。

想一想：
  1. 一个篮球队员今年表现特别好，上了杂志封面，第二年表现变差了。这能说明"封面诅咒"吗？
  2. 一次考试考得特别差的学生，下次考试往往会进步。这一定是补课的功劳吗？""")
