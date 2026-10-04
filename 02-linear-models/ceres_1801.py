"""
实验 1：1801 年，用最小二乘法找回"丢失的行星"
================================================

1801 年元旦，皮亚齐发现了谷神星，追踪了 41 天、约两打观测之后，它消失在太阳的光芒里。
要在一年后重新找到它，就得用这几十个**带误差的观测**，算出最可信的轨道。

这里做一个大大简化的版本：
  * 假设"行星"在天上的角度随时间匀速变化：角度 = a + b × 天数
  * 41 天里随机挑 24 个晚上观测，每次观测都有一点误差
  * 任务：预测第 364 天（12 月 31 日）它在哪里

真实的轨道计算复杂得多（椭圆轨道有 6 个未知数，地球自己也在动）。
这里只保留最核心的想法：**用很多个有误差的观测，求出最可信的参数。**

运行：python ceres_1801.py
"""

import numpy as np

TRUE_A, TRUE_B = 51.5, 0.214     # 真实的起始角度（度）和每天移动的角度（度/天）
NOISE = 0.05                     # 每次观测的误差（度）
N_OBS, LAST_DAY, TARGET_DAY = 24, 40, 364

rng = np.random.default_rng(1801)


def observe():
    days = np.sort(rng.choice(np.arange(LAST_DAY + 1), size=N_OBS, replace=False)).astype(float)
    angles = TRUE_A + TRUE_B * days + rng.normal(0, NOISE, N_OBS)
    return days, angles


def two_point_fit(t, y):
    """最朴素的办法：只用第一次和最后一次观测，连一条直线。"""
    b = (y[-1] - y[0]) / (t[-1] - t[0])
    return y[0] - b * t[0], b


def least_squares_fit(t, y):
    """最小二乘：找一条直线，让所有观测点到直线的'误差平方和'最小。
    求导令其为 0，得到两个方程（正规方程），手算 2×2 就能解出来。"""
    n, st, sy = len(t), t.sum(), y.sum()
    stt, sty = (t * t).sum(), (t * y).sum()
    b = (n * sty - st * sy) / (n * stt - st * st)
    a = (sy - b * st) / n
    return a, b


# ---------------------------------------------------------------- 一次具体的"历史"
t, y = observe()
a2, b2 = two_point_fit(t, y)
a, b = least_squares_fit(t, y)
truth = TRUE_A + TRUE_B * TARGET_DAY

print(f"皮亚齐的 {N_OBS} 次观测（第 0～{LAST_DAY} 天），每次误差约 ±{NOISE}°")
print("   天数:", " ".join(f"{d:.0f}" for d in t[:12]), "…")
print()

# 用 ASCII 画出观测点和最小二乘直线（为了看得清，纵轴画的是"减去真实直线后的偏差"）
print("观测点（●）和最小二乘直线（─）相对真实位置的偏差：")
rows = np.linspace(0.12, -0.12, 9)
for r in rows:
    line = ""
    for day in range(LAST_DAY + 1):
        fit_dev = (a + b * day) - (TRUE_A + TRUE_B * day)
        obs = [yy - (TRUE_A + TRUE_B * tt) for tt, yy in zip(t, y) if tt == day]
        if obs and abs(obs[0] - r) < 0.0151:
            line += "●"
        elif abs(fit_dev - r) < 0.0151:
            line += "─"
        else:
            line += " "
    print(f"   {r:+.2f}° |{line}")
print("          " + "第 0 天" + " " * 28 + f"第 {LAST_DAY} 天\n")

print("最小二乘的'正规方程'（手算即可）：")
print(f"   n = {len(t)},  Σt = {t.sum():.0f},  Σt² = {(t * t).sum():.0f},  Σy = {y.sum():.2f},  Σty = {(t * y).sum():.1f}")
print(f"   解出：a = {a:.4f}°，b = {b:.5f}°/天")
lstsq = np.linalg.lstsq(np.column_stack([np.ones_like(t), t]), y, rcond=None)[0]
print(f"   用 numpy 验算：a = {lstsq[0]:.4f}°，b = {lstsq[1]:.5f}°/天  ✓\n")

print(f"预测第 {TARGET_DAY} 天的位置（真实位置 {truth:.2f}°）：")
print(f"   两点法：  {a2 + b2 * TARGET_DAY:.2f}°，误差 {abs(a2 + b2 * TARGET_DAY - truth):.2f}°")
print(f"   最小二乘：{a + b * TARGET_DAY:.2f}°，误差 {abs(a + b * TARGET_DAY - truth):.2f}°\n")

# ---------------------------------------------------------------- 重演 2000 次历史
errs2, errs = [], []
for _ in range(2000):
    t, y = observe()
    a2, b2 = two_point_fit(t, y)
    a, b = least_squares_fit(t, y)
    errs2.append(abs(a2 + b2 * TARGET_DAY - truth))
    errs.append(abs(a + b * TARGET_DAY - truth))
errs2, errs = np.array(errs2), np.array(errs)

print("把这段历史重演 2000 次，统计预测误差：")
print(f"   两点法：  平均误差 {errs2.mean():.2f}°，误差在半度以内的比例 {np.mean(errs2 < 0.5):.0%}")
print(f"   最小二乘：平均误差 {errs.mean():.2f}°，误差在半度以内的比例 {np.mean(errs < 0.5):.0%}")
print("""
只用两个观测，另外 22 个就白白浪费了，误差还会被拉长 364 天的"杠杆"放大。
最小二乘让每个观测都出一份力，误差互相抵消，预测稳得多。

想一想：
  1. 把 NOISE 调大到 0.2，半度以内的比例会变成多少？
  2. 观测天数从 41 天缩短到 10 天（LAST_DAY = 10），为什么预测会差很多？
  3. 为什么是"误差的平方和"最小，而不是"误差的绝对值之和"最小？（提示：见 README 的概念解释）""")
