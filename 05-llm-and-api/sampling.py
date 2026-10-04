"""
实验 3：温度、top-k、top-p：模型怎么"挑"下一个字
===================================================

语言模型每一步输出的是一个**概率分布**：下一个字是"的"的概率 12%，是"是"的概率 8%……
最后到底写哪个字，由"采样策略"决定。API 里的 temperature、top_p 参数，调的就是它。

这里用实验 1 训练的 3-gram 模型来演示：
  1. 同一个分布，在不同温度下长什么样
  2. 贪心解码为什么容易"卡住"、不停重复
  3. 温度和 top-p 怎么影响续写的风格

运行：python sampling.py
"""

import math
import random

from markov_1913 import build_model, generate, load_corpus, next_distribution


def pad(text, width):
    """按显示宽度补空格（一个汉字占两格）。"""
    return text + " " * max(width - sum(2 if ord(c) > 127 else 1 for c in text), 0)


def reshape(dist, temperature):
    logits = {ch: math.log(p) / temperature for ch, p in dist.items()}
    m = max(logits.values())
    ws = {ch: math.exp(v - m) for ch, v in logits.items()}
    s = sum(ws.values())
    return {ch: w / s for ch, w in ws.items()}


text = load_corpus()
vocab = sorted(set(text))
model = build_model(text, 3)

# ---------------------------------------------------------------- 1. 温度改变分布的形状
context = "反向"
dist = next_distribution(model, context, 3, vocab)
print(f"【1】在'{context}'后面，下一个字的概率分布（只显示前 6 个）")
for t in (0.5, 1.0, 2.0):
    d = reshape(dist, t)
    top = sorted(d.items(), key=lambda kv: -kv[1])[:6]
    cells = "  ".join(f"{ch!r}:{p:5.1%}" for ch, p in top)
    print(f"   温度 {t}：{cells}")
print("   温度 < 1：强者更强，分布更'尖'，模型更保守；温度 > 1：分布被'压平'，冷门的字也有机会。\n")

# ---------------------------------------------------------------- 2. top-k 和 top-p
dist = next_distribution(model, "神经", 3, vocab)
ranked = sorted(dist.items(), key=lambda kv: -kv[1])
print("【2】top-k 和 top-p：先把明显不靠谱的候选剪掉，再抽签")
print(f"   在'神经'后面，概率最高的几个字：" + "  ".join(f"{ch!r}:{p:.1%}" for ch, p in ranked[:5]))
acc, k_needed = 0.0, 0
for ch, p in ranked:
    acc += p
    k_needed += 1
    if acc >= 0.9:
        break
print(f"   top-k = 5：只在前 5 个里抽")
print(f"   top-p = 0.9：从高到低累加概率，凑够 90% 为止。这里只需要前 {k_needed} 个字。")
print("   top-p 的好处是会'自动伸缩'：模型很确定时候选很少，模型拿不准时候选会多一些。\n")

# ---------------------------------------------------------------- 3. 续写对比
print("【3】同一个模型，不同的采样策略，续写'注意力'：")
settings = [("贪心（温度 0）", 0.0, 1.0), ("温度 0.5", 0.5, 1.0), ("温度 1.0", 1.0, 1.0),
            ("温度 1.0 + top-p 0.9", 1.0, 0.9), ("温度 1.8", 1.8, 1.0)]
for name, temp, top_p in settings:
    rng = random.Random(1951)
    out = generate(model, 3, "注意力", 45, vocab, rng, temperature=temp, top_p=top_p)
    print(f"   {pad(name, 22)}{out.replace(chr(10), ' ')}")

print("\n   单看一次的结果有运气成分。把每种设置都跑 20 次，统计两个指标：")
print("   · 不重复度：续写里不重复的三字片段占多少（越低越啰嗦）")
print("   · 新造比例：续写里有多少三字片段在语料中从没出现过（越高越像胡编）")
print(f"   {pad('设置', 22)}| 不重复度 | 新造比例")
for name, temp, top_p in settings:
    distinct, novel = 0.0, 0.0
    for seed in range(20):
        out = generate(model, 3, "注意力", 45, vocab, random.Random(seed), temperature=temp, top_p=top_p)
        grams = [out[i:i + 3] for i in range(len(out) - 2)]
        distinct += len(set(grams)) / len(grams) / 20
        novel += sum(g not in text for g in grams) / len(grams) / 20
    print(f"   {pad(name, 22)}|   {distinct:4.0%}   |  {novel:4.0%}")

print("""
   * 贪心解码每次都选最可能的字，很容易绕进一个圈子里出不来，不停地重复同一段话；
   * 温度低：几乎只会照搬见过的片段，稳妥但呆板；温度高：大部分片段都是新拼出来的，五花八门但胡言乱语；
   * 温度 1 左右再加上 top-p，通常是"有变化、又不离谱"的折中，很多 API 的默认设置就在这附近。

   用 API 时可以这样选：写代码、做数学、抽取信息，想要稳定的答案，就把温度调低；写故事、头脑风暴，就调高一些。
   （有些新模型在开启深度思考时不允许修改这些参数，以各家文档为准。）

想一想：
  1. 为什么贪心解码会重复？在第 1 部分的分布里能找到原因吗？
  2. 温度趋近 0 和 top-k = 1 是不是一回事？
  3. 如果要让模型"每次回答都一模一样"（比如做测试），应该怎么设置？""")
