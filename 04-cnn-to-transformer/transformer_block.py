"""
实验 4：拼出一个完整的 Transformer
====================================

前面三个实验是零件，这里把它们组装成 2017 年论文里的结构：
  词嵌入 + 位置编码 → [多头自注意力 → 前馈网络]×N（每一块都有残差连接和层归一化）→ 预测下一个词

权重是随机的、没有训练，所以预测毫无意义。这个实验的目的是看清：
  1. 数据在每一层的形状
  2. 参数都花在了哪里
  3. 为什么残差连接能让网络堆得很深（呼应第三章的梯度消失）

运行：python transformer_block.py
"""

import numpy as np

rng = np.random.default_rng(2017)

VOCAB, D_MODEL, N_HEADS, D_FF, N_LAYERS, SEQ = 1000, 64, 4, 256, 2, 10
D_HEAD = D_MODEL // N_HEADS


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x, eps=1e-5):
    """层归一化：把每个词的向量调整成平均 0、方差 1，让数值保持在稳定的范围里。"""
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


def positional_encoding(n, d):
    """2017 年论文里的正弦位置编码：不同维度用不同频率的波，组合起来唯一地标记每个位置。"""
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / (10000 ** (2 * i / d))
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe


def init(*shape):
    return rng.normal(0, 1 / np.sqrt(shape[0]), shape)


def make_block():
    return {"Wq": init(D_MODEL, D_MODEL), "Wk": init(D_MODEL, D_MODEL),
            "Wv": init(D_MODEL, D_MODEL), "Wo": init(D_MODEL, D_MODEL),
            "W1": init(D_MODEL, D_FF), "W2": init(D_FF, D_MODEL)}


def multi_head_attention(x, p, verbose=False):
    n = x.shape[0]
    q = (x @ p["Wq"]).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)   # (头, 词, 每头维度)
    k = (x @ p["Wk"]).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
    v = (x @ p["Wv"]).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(D_HEAD)                 # (头, 词, 词)
    scores = np.where(np.tril(np.ones((n, n), bool)), scores, -1e9)      # 因果掩码
    out = softmax(scores) @ v                                            # (头, 词, 每头维度)
    out = out.transpose(1, 0, 2).reshape(n, D_MODEL)                     # 把各个头拼回去
    if verbose:
        print(f"      拆成 {N_HEADS} 个头：Q/K/V 形状 {q.shape}，注意力分数 {scores.shape}")
    return out @ p["Wo"]


def block(x, p, residual=True, verbose=False):
    a = multi_head_attention(layer_norm(x), p, verbose)                  # 1. 自注意力：词与词交流
    x = x + a if residual else a
    f = np.maximum(layer_norm(x) @ p["W1"], 0) @ p["W2"]                 # 2. 前馈网络：每个词自己"想一想"
    x = x + f if residual else f
    return x


# ---------------------------------------------------------------- 1. 走一遍前向计算
embed = init(VOCAB, D_MODEL) * np.sqrt(VOCAB) / np.sqrt(D_MODEL)
unembed = init(D_MODEL, VOCAB)
blocks = [make_block() for _ in range(N_LAYERS)]
tokens = rng.integers(0, VOCAB, SEQ)

print("【1】一句话（10 个词）流过 Transformer，每一步的形状：")
print(f"   输入：10 个词的编号 {tokens.tolist()}")
x = embed[tokens]
print(f"   词嵌入：查表得到每个词的向量         → {x.shape}（词数 × 模型维度）")
x = x + positional_encoding(SEQ, D_MODEL)
print(f"   加上位置编码：告诉模型词的顺序       → {x.shape}")
for i, p in enumerate(blocks):
    print(f"   第 {i + 1} 个 Transformer 块：")
    x = block(x, p, verbose=True)
    print(f"      注意力 + 前馈网络 + 残差连接之后  → {x.shape}（形状不变，内容被更新）")
logits = layer_norm(x) @ unembed
probs = softmax(logits[-1])
print(f"   输出层：每个位置对 {VOCAB} 个词打分     → {logits.shape}")
print(f"   最后一个位置预测'下一个词'：最可能是第 {probs.argmax()} 号词，概率 {probs.max():.4f}")
print(f"   （没训练过，所以概率接近平均的 1/{VOCAB} = {1 / VOCAB:.4f}；训练就是要让正确的下一个词概率变大。）\n")

# ---------------------------------------------------------------- 2. 参数都在哪里
print("【2】参数都花在哪里？")
n_embed = VOCAB * D_MODEL * 2
n_attn = 4 * D_MODEL * D_MODEL
n_ffn = 2 * D_MODEL * D_FF
print(f"   词嵌入 + 输出层：           {n_embed:>9,}")
print(f"   每块的注意力（Q、K、V、O）：  {n_attn:>9,}  × {N_LAYERS} 块")
print(f"   每块的前馈网络：             {n_ffn:>9,}  × {N_LAYERS} 块")
print(f"   合计：                      {n_embed + N_LAYERS * (n_attn + n_ffn):>9,}")
print("   前馈网络的参数是注意力的 2 倍：大模型里，大部分'知识'存在前馈网络里。")
print("   把维度从 64 放大到 1 万多、层数从 2 堆到 100 层左右，参数量就到了千亿级别。\n")

# ---------------------------------------------------------------- 3. 残差连接
def similarity(x):
    """不同位置的词向量之间，平均的余弦相似度。1.0 表示所有词变得一模一样。"""
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    S = xn @ xn.T
    return (S.sum() - len(x)) / (len(x) * (len(x) - 1))


print("【3】堆 48 层：有残差连接 vs 没有残差连接")
print("   看各层输出里，不同词的向量有多像（1.00 = 所有词都变成了同一个向量，信息全丢了）：")
deep = [make_block() for _ in range(48)]
for p in deep:                     # GPT-2 的做法：残差分支的输出权重按 1/√(2×层数) 缩小
    p["Wo"] *= 1 / np.sqrt(2 * 48)
    p["W2"] *= 1 / np.sqrt(2 * 48)
x0 = rng.normal(0, 1, (VOCAB, D_MODEL))[tokens] + positional_encoding(SEQ, D_MODEL)
for residual in (True, False):
    x = x0.copy()
    marks = [f"输入 {similarity(x):.2f}"]
    for i, p in enumerate(deep, 1):
        x = block(x, p, residual)
        if i in (1, 6, 24, 48):
            marks.append(f"第{i:2d}层 {similarity(x):.2f}")
    print(f"   {'有残差连接' if residual else '没有残差  '}：" + "  ".join(marks))
print("""   没有残差连接时，几层之后所有词的向量就变得一模一样（研究者称之为"秩崩塌"），
   模型再也分不清哪个词是哪个词，梯度也传不回去。
   有了残差连接（x = x + f(x)），每一层只是在原来的基础上"加一点修改"，输入的信息有一条直通的高速公路。
   这正是 2015 年何恺明等人用残差网络（ResNet）训练出 152 层网络的秘诀，Transformer 也继承了它。

想一想：
  1. 把 N_HEADS 从 4 改成 8，参数量会变吗？为什么？
  2. 去掉位置编码，模型还能区分"猫追狗"和"狗追猫"吗？
  3. 为什么输出层只需要用最后一个位置的结果来预测下一个词？""")
