"""
实验 1：复现 1958 年 Rosenblatt 的感知机演示
================================================

1958 年 7 月，Rosenblatt 在华盛顿演示：一台 IBM 704 看了 50 张打孔卡片之后，
学会了分辨"标记在左边"和"标记在右边"的卡片。

这里我们用 numpy 模拟同样的事情：
  * 一张"卡片" = 20×20 的像素网格（Mark I 感知机正好有 20×20 = 400 个光电管）
  * 卡片上有一个小方块，要么在左半边，要么在右半边，再撒一点噪点
  * 感知机只有 400 个权重 + 1 个偏置，规则只有一句话：**猜错了才改**

运行：python perceptron_1958.py
"""

import numpy as np

SIZE = 20          # 20×20 的"视网膜"
BLOCK = 4          # 标记方块的边长
NOISE = 0.03       # 每个像素有 3% 的概率被随机点亮（模拟脏卡片）
LEFT, RIGHT = -1, +1

rng = np.random.default_rng(1958)


def make_card(side):
    """生成一张卡片：side=LEFT 时方块在左半边，side=RIGHT 时在右半边。"""
    card = (rng.random((SIZE, SIZE)) < NOISE).astype(float)
    row = rng.integers(0, SIZE - BLOCK + 1)
    if side == LEFT:
        col = rng.integers(0, SIZE // 2 - BLOCK + 1)
    else:
        col = rng.integers(SIZE // 2, SIZE - BLOCK + 1)
    card[row:row + BLOCK, col:col + BLOCK] = 1.0
    return card


def make_deck(n):
    sides = rng.choice([LEFT, RIGHT], size=n)
    cards = np.stack([make_card(s) for s in sides])
    return cards.reshape(n, -1), sides          # 每张卡片拉平成 400 维向量


def show_card(flat, title):
    print(title)
    grid = flat.reshape(SIZE, SIZE)
    for r in range(SIZE):
        print("   " + "".join("█" if v else "·" for v in grid[r]))
    print()


def show_weights(w):
    """把 400 个权重画回 20×20：'+' 表示"看到这里亮就投右边"，'-' 表示投左边。"""
    grid = w.reshape(SIZE, SIZE)
    t = np.abs(w).max() * 0.15
    print("学到的权重图（+ 投票给'右'，- 投票给'左'，空格≈0）：")
    for r in range(SIZE):
        line = "".join("+" if v > t else "-" if v < -t else " " for v in grid[r])
        print("   |" + line + "|")
    print()


def predict(w, b, x):
    return RIGHT if x @ w + b > 0 else LEFT


def train(w, b, xs, ys, max_epochs=20):
    """反复看同一叠卡片，直到一整轮都不出错。"""
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for x, y in zip(xs, ys):
            if predict(w, b, x) != y:
                # Rosenblatt 的学习规则：猜错了，就把这张卡片"加进"或"减出"权重
                w += y * x
                b += y
                mistakes += 1
        print(f"   第 {epoch} 轮（看了 {len(xs)} 张卡片）：猜错 {mistakes:3d} 次")
        if mistakes == 0:
            print("   一整轮都没猜错，这叠卡片学会了。")
            break
    return w, b


def exam(w, b):
    acc = np.mean([predict(w, b, x) == y for x, y in zip(test_x, test_y)])
    print(f"   拿 1000 张从没见过的新卡片考试：准确率 {acc:.1%}\n")


# 400 个权重，对应 Mark I 上 400 个由电机拧动的电位器
w = np.zeros(SIZE * SIZE)
b = 0.0
test_x, test_y = make_deck(1000)

# ---------------------------------------------------------------- 第一幕：50 张卡片
train_x, train_y = make_deck(50)  # 和 1958 年的演示一样：50 次试验
show_card(train_x[0], f"第一张训练卡片（答案：{'右' if train_y[0] == RIGHT else '左'}）：")

print("【第一幕】像 1958 年那样，只给它看 50 张卡片")
w, b = train(w, b, train_x, train_y)
exam(w, b)
show_weights(w)

# ---------------------------------------------------------------- 第二幕：更多数据
more_x, more_y = make_deck(1000)
print("【第二幕】再多给它看 1000 张卡片")
w, b = train(w, b, more_x, more_y)
exam(w, b)
show_weights(w)

print("""在训练卡片上全对，不等于在新卡片上全对。
50 张卡片里，方块恰好没出现过的位置，权重就还是 0，噪点一干扰就容易猜错；
看得越多，权重图越干净，考试成绩越好。这就是"泛化"，也是后来 ImageNet 的故事要讲的：数据很重要。

想一想：
  1. 权重图为什么左边是 '-'、右边是 '+'？没有人告诉它"左右"这个概念。
  2. 把 NOISE 调到 0.3，会发生什么？
  3. 如果任务换成"方块在上半边还是下半边"，需要改学习规则吗？""")
