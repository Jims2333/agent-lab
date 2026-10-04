"""
实验 1：从马尔可夫到 n-gram：最早的"语言模型"
===============================================

1913 年，马尔可夫数了《叶甫盖尼·奥涅金》的前 2 万个字母，证明了一件事：
**下一个字母是什么，和前一个字母有关。** 文字不是一串互相独立的随机符号。

1948 年，香农用同样的思路造出了"n-gram 语言模型"：根据前 n-1 个字符，预测下一个字符。
ChatGPT 做的事情本质上一样，都是"预测下一个"，只是看的上下文长得多、预测的方法强得多。

这里的语料是本仓库第 1～4 章的正文（data/corpus.txt），我们：
  1. 重做马尔可夫的实验：字和字之间真的有依赖吗？
  2. 训练 1～4 阶的 n-gram 模型，让它们"续写"
  3. 用"困惑度"给它们打分，看看为什么 n 越大不一定越好

运行：python markov_1913.py
（sampling.py 会 import 这里的函数）
"""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_corpus():
    text = Path(__file__).with_name("data").joinpath("corpus.txt").read_text()
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("#"))


def build_model(text, n):
    """统计每个"前 n-1 个字"后面，各个字出现了多少次。返回 {上下文: Counter(下一个字)}。"""
    counts = defaultdict(Counter)
    for i in range(len(text)):
        for k in range(n):                       # 同时记下 0～n-1 阶的统计，方便"退回"到短上下文
            if i - k < 0:
                break
            counts[text[i - k:i]][text[i]] += 1
    return counts


def next_distribution(model, context, n, vocab):
    """插值平滑：把不同长度上下文的预测按权重混在一起，避免"没见过就是 0 概率"。"""
    weights = {0: 0.05, 1: 0.15, 2: 0.3, 3: 0.5}
    probs = Counter()
    total_w = 0.0
    for k in range(n):
        ctx = context[len(context) - k:] if k else ""
        c = model.get(ctx)
        if not c:
            continue
        s = sum(c.values())
        w = weights.get(k, 0.5)
        total_w += w
        for ch, cnt in c.items():
            probs[ch] += w * cnt / s
    floor = 1e-4                                  # 给所有字留一点点概率
    return {ch: (probs.get(ch, 0.0) + floor / len(vocab)) / (total_w + floor) for ch in vocab}


def generate(model, n, prompt, length, vocab, rng, temperature=1.0, top_p=1.0):
    out = prompt
    for _ in range(length):
        dist = next_distribution(model, out[-(n - 1):] if n > 1 else "", n, vocab)
        out += sample(dist, rng, temperature, top_p)
    return out


def sample(dist, rng, temperature=1.0, top_p=1.0):
    """按概率抽一个字。temperature 和 top_p 的作用见 sampling.py。"""
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    if temperature < 1e-6:
        return items[0][0]                        # 温度为 0：总是选最可能的那个（贪心）
    logits = [math.log(p) / temperature for _, p in items]
    m = max(logits)
    ws = [math.exp(l - m) for l in logits]
    s = sum(ws)
    ws = [w / s for w in ws]
    if top_p < 1.0:                               # 只保留累计概率达到 top_p 的那些候选
        keep, acc = [], 0.0
        for (ch, _), w in zip(items, ws):
            keep.append((ch, w))
            acc += w
            if acc >= top_p:
                break
        items, ws = [k for k, _ in keep], [w for _, w in keep]
        return rng.choices(items, weights=ws)[0]
    return rng.choices([ch for ch, _ in items], weights=ws)[0]


def perplexity(model, n, text, vocab):
    """困惑度：模型平均每一步"在几个字之间犹豫"。越低越好。"""
    log_sum = 0.0
    for i in range(1, len(text)):
        ctx = text[max(0, i - (n - 1)):i] if n > 1 else ""
        p = next_distribution(model, ctx, n, vocab)[text[i]]
        log_sum += -math.log(p)
    return math.exp(log_sum / (len(text) - 1))


if __name__ == "__main__":
    text = load_corpus()
    vocab = sorted(set(text))
    split = int(len(text) * 0.9)
    train, test = text[:split], text[split:]
    print(f"语料：{len(text):,} 个字符，{len(vocab)} 种不同的字符。前 90% 用来训练，后 10% 用来测试。\n")

    # ------------------------------------------------------------ 1. 马尔可夫的问题
    print("【1】重做马尔可夫的实验：下一个字和前一个字有关吗？")
    uni = Counter(train)
    bi = build_model(train, 2)
    total = sum(uni.values())
    for prev, nxt in [("学", "习"), ("神", "经"), ("梯", "度"), ("模", "型")]:
        p_alone = uni[nxt] / total
        p_after = bi[prev][nxt] / max(sum(bi[prev].values()), 1)
        print(f"   '{nxt}' 出现的概率：随便一个位置 {p_alone:6.2%}  →  紧跟在'{prev}'后面 {p_after:6.1%}"
              f"（{p_after / p_alone:,.0f} 倍）")
    print("   如果字和字互相独立，两个概率应该差不多。差了几百倍，说明文字是一条有记忆的'链'。")
    print("   马尔可夫 1913 年数《奥涅金》的元音和辅音，得出的是同一个结论。\n")

    # ------------------------------------------------------------ 2. 续写
    print("【2】让不同阶的 n-gram 模型续写'神经网络'（每个模型都用同一个随机种子）")
    models = {n: build_model(train, n) for n in (1, 2, 3, 4)}
    for n, model in models.items():
        rng = random.Random(1913)
        out = generate(model, n, "神经网络", 50, vocab, rng)
        hint = {1: "只看字频，不看上下文", 2: "看前 1 个字", 3: "看前 2 个字", 4: "看前 3 个字"}[n]
        print(f"   {n}-gram（{hint}）：{out.replace(chr(10), ' ')}")
    print("   n 越大，局部越通顺；但它只是在拼接见过的片段，并不'理解'内容。\n")

    # ------------------------------------------------------------ 3. 困惑度
    print("【3】在没见过的测试文本上打分（困惑度，越低越好）")
    for n, model in models.items():
        ppl = perplexity(model, n, test[:3000], vocab)
        print(f"   {n}-gram：困惑度 {ppl:7.1f}  {'█' * int(ppl / 15)}")
    print(f"""   困惑度 100 的意思是：模型每预测一个字，平均就像在 100 个字里瞎猜。
   从 1-gram 到 3-gram 进步很大；再往上，进步变小甚至变差：上下文越长，在训练语料里"一模一样出现过"的机会越少（数据稀疏）。
   这就是 n-gram 的天花板。2003 年本吉奥的神经语言模型、后来的 Transformer，
   正是用"向量"代替"死记硬背"，让模型能从相似的上下文里举一反三。

想一想：
  1. 把语料换成你自己的文字（比如聊天记录、日记），续写会变成什么风格？
  2. 4-gram 在训练集上的困惑度会是多少？为什么和测试集差这么多？（提示：过拟合）
  3. 一个看前 1000 个字的 n-gram 模型，需要多少语料才能"见过"足够多的上下文？""")
