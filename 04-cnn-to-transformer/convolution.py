"""
实验 1：卷积：教机器像猫的视觉皮层一样"看边缘"
================================================

1959 年，休伯尔和维泽尔发现：猫的视觉皮层里，有的神经元只对**特定方向的边缘**有反应。
1980 年的 Neocognitron 和 1989 年 LeCun 的卷积网络，都是在模仿这种"局部的、找边缘的小探测器"。

这里用 numpy 手写卷积：
  1. 用 3×3 的"边缘探测器"扫过一张小图，看它在哪里有反应
  2. 复现休伯尔和维泽尔的"方向调谐曲线"：同一个探测器，对不同角度的线条反应多强
  3. 看池化怎么让识别"不怕平移"
  4. 数一数：卷积比全连接省了多少参数

运行：python convolution.py
"""

import numpy as np


def conv2d(image, kernel):
    """最朴素的二维卷积（严格说是互相关，深度学习里都这么叫）：小窗口滑过整张图，逐个位置做加权求和。"""
    kh, kw = kernel.shape
    h, w = image.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


def max_pool(x, size=2):
    h, w = x.shape[0] // size, x.shape[1] // size
    return x[:h * size, :w * size].reshape(h, size, w, size).max(axis=(1, 3))


def show(x, title, chars=" .:-=+*#%@"):
    print(title)
    m = np.abs(x).max() or 1.0
    for row in x:
        print("   " + "".join(chars[int(abs(v) / m * (len(chars) - 1))] * 2 for v in row))
    print()


# ---------------------------------------------------------------- 1. 一张小图
img = np.zeros((14, 14))
img[3:11, 3:11] = 1.0              # 一个方块
img[5:9, 5:9] = 0.0                # 中间挖空，变成一个"口"字

KERNELS = {
    "竖直边缘": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),   # Sobel 算子
    "水平边缘": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
}

show(img, "【1】原图：一个'口'字（14×14 像素）")
for name, k in KERNELS.items():
    out = np.maximum(conv2d(img, k), 0) + np.maximum(-conv2d(img, k), 0)   # 两个方向的边缘都算
    show(out, f"   用 3×3 的'{name}'探测器扫一遍之后（越密 = 反应越强）：")
print("   同一个 3×3 的小探测器（只有 9 个数），在整张图的每个位置重复使用。")
print("   竖直探测器只在左右两条边上亮，水平探测器只在上下两条边上亮。\n")


# ---------------------------------------------------------------- 2. 方向调谐曲线
def bar_image(angle_deg, size=21, width=0.8):
    """画一根穿过中心、指定角度的亮线（边缘柔和，像投影仪打出来的光条）。"""
    c = size // 2
    t = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.abs((xx - c) * np.sin(t) - (c - yy) * np.cos(t))   # 每个像素到线条的距离
    return np.exp(-dist ** 2 / (2 * width ** 2))


print("【2】复现休伯尔和维泽尔的实验：给'竖直边缘'探测器看不同角度的线条，它的最强反应是多少？")
responses = []
for angle in range(0, 181, 15):
    r = conv2d(bar_image(angle), KERNELS["竖直边缘"])
    responses.append((angle, np.abs(r).max()))
top = max(v for _, v in responses)
for angle, v in responses:
    print(f"   线条角度 {angle:3d}° {'█' * int(round(v / top * 30)):<30} {v / top:4.0%}")
print("   线条越接近竖直（90°），反应越强；变成水平（0° 和 180°）时最弱。")
print("   这就是一个'方向选择性'的神经元，和猫的视觉皮层里发现的一样。\n")

# ---------------------------------------------------------------- 3. 池化与平移不变性
print("【3】池化：图像挪动一点点，识别结果还稳不稳？")
k = KERNELS["竖直边缘"]
a = np.abs(conv2d(img, k))
shifted = np.roll(img, 1, axis=1)          # 整张图向右挪 1 个像素
b = np.abs(conv2d(shifted, k))
pa, pb = max_pool(a, 4), max_pool(b, 4)
print(f"   卷积之后：两张特征图有 {np.mean(np.abs(a - b) > 1e-9):.0%} 的位置数值不一样")
print(f"   再做 4×4 最大池化：只有 {np.mean(np.abs(pa - pb) > 1e-9):.0%} 的位置不一样")
print("   池化只关心'这一片区域里有没有出现边缘'，不关心精确位置，所以对小的平移不敏感。")
print("   福岛邦彦 1980 年的论文标题里就写着这个目标：'不受位置移动影响的模式识别'。\n")

# ---------------------------------------------------------------- 4. 参数量
H = W = 224
full = (H * W) * (H * W)
conv = 3 * 3
print("【4】为什么要卷积：参数量对比（以 224×224 的灰度图为例）")
print(f"   全连接：每个输出像素都连到全部输入像素，需要 {full:,} 个权重")
print(f"   卷积：  一个 3×3 探测器在所有位置共享，只要 {conv} 个权重")
print(f"   相差 {full // conv:,} 倍。这就是'局部连接 + 权重共享'的威力。")

print("""
想一想：
  1. 自己设计一个"对角线边缘"探测器（3×3），它的调谐曲线峰值会在哪个角度？
  2. 如果把两层卷积叠起来，第二层的一个神经元能"看到"原图多大的区域？
  3. CNN 里的探测器是训练出来的，不是手工设计的。你觉得训练好的第一层会长什么样？""")
