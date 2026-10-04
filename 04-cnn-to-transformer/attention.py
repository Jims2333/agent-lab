"""
实验 2：手写注意力机制
========================

2014 年，巴达瑙（Bahdanau）让翻译模型在生成每个词时，"回头看"原句里最相关的那几个词。
2017 年，Transformer 把这个想法发挥到极致：整个网络只靠注意力，不再需要循环。

注意力的公式只有一行：  输出 = softmax(Q·Kᵀ / √d) · V
这里分四步把它拆开：
  1. 把注意力看成"模糊查字典"
  2. 在一句话上做自注意力：让"它"找到"小猫"
  3. 多头注意力 + 因果掩码：一个头管"指代"，一个头管"前一个词"
  4. 为什么要除以 √d

运行：python attention.py
"""

import numpy as np

np.random.seed(2017)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # 减去最大值，防止 exp 溢出
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """缩放点积注意力。Q: (n, d)，K: (m, d)，V: (m, dv)。返回输出和注意力权重。"""
    scores = Q @ K.T / np.sqrt(Q.shape[-1])         # 每个查询和每个键有多"匹配"
    if mask is not None:
        scores = np.where(mask, scores, -1e9)        # 被遮住的位置，分数设成极小
    weights = softmax(scores, axis=-1)               # 变成加起来等于 1 的权重
    return weights @ V, weights                      # 按权重把"值"加权平均


def pad(text, width):
    """按显示宽度补空格（一个汉字占两格），让中英文混排也能对齐。"""
    shown = sum(2 if ord(ch) > 127 else 1 for ch in text)
    return text + " " * max(width - shown, 0)


def shade(v):
    for limit, ch in ((0.05, " "), (0.25, "░"), (0.5, "▒"), (0.75, "▓")):
        if v < limit:
            return ch
    return "█"


def heatmap(w, labels, title):
    print(title)
    print("   " + pad("", 10) + "看向→ " + " ".join(f"{i:>2}" for i in range(len(labels))))
    for i, row in enumerate(w):
        print(f"   {i:>2} {pad(labels[i], 7)}     " + " ".join(shade(v) * 2 for v in row))
    print("   （" + "  ".join(f"{i}:{t}" for i, t in enumerate(labels)) + "；░ 少量 ▒ 较多 ▓ 很多 █ 几乎全部）\n")


# ---------------------------------------------------------------- 1. 模糊查字典
print("【1】注意力 = 模糊查字典")
features = ["甜", "酸", "红", "黄"]
keys = np.array([[0.8, 0.2, 1, 0],      # 苹果
                 [0.0, 1.0, 0, 1],      # 柠檬
                 [0.9, 0.0, 0, 1],      # 香蕉
                 [0.9, 0.3, 1, 0]])     # 草莓
names = ["苹果", "柠檬", "香蕉", "草莓"]
values = np.array([[5.0], [3.0], [4.0], [12.0]])     # 每种水果每斤的价格
query = np.array([[1.0, 0.0, 1.0, 0.0]])             # 我想要："又甜又红的"
out, w = attention(query * 3, keys * 3, values)      # ×3 让匹配度的差别更明显
print("   字典里的'键'是水果的特征（甜、酸、红、黄），'值'是价格。")
print("   查询：'又甜又红的水果多少钱？'")
for n, wi in zip(names, w[0]):
    print(f"     {n}：权重 {wi:.2f} {'█' * int(wi * 40)}")
print(f"   普通字典只能查到一个精确的键；注意力按相似度给每个键分配权重，答案是加权平均：{out[0, 0]:.2f} 元")
print("   苹果和草莓都'又甜又红'，所以它们分到了大部分权重。\n")

# ---------------------------------------------------------------- 2. 自注意力：让"它"找到"小猫"
tokens = ["小猫", "追", "毛线球", "，", "它", "跑", "得", "很", "快"]
n = len(tokens)
#               动物 物体 动作 代词 虚词
semantic = np.array([[1, 0, 0, 0, 0],   # 小猫
                     [0, 0, 1, 0, 0],   # 追
                     [0, 1, 0, 0, 0],   # 毛线球
                     [0, 0, 0, 0, 1],   # ，
                     [0, 0, 0, 1, 0],   # 它
                     [0, 0, 1, 0, 0],   # 跑
                     [0, 0, 0, 0, 1],   # 得
                     [0, 0, 0, 0, 1],   # 很
                     [0, 0, 1, 0, 0]],  # 快
                    dtype=float)
position = np.eye(n)                     # 位置信息：第 i 个词在第 i 维上是 1
X = np.hstack([semantic, position])      # 每个词的向量 = 语义特征 + 位置特征
d = X.shape[1]

# 头 A（指代头）：代词的"查询"去找动物的"键"
Wq_a = np.zeros((d, 4)); Wq_a[3, 0] = 4.0      # 代词 → 在第 0 维发出"我在找动物"
Wk_a = np.zeros((d, 4)); Wk_a[0, 0] = 4.0      # 动物 → 在第 0 维回应"我是动物"
Wv_a = np.eye(d)[:, :5]                          # 值：把语义特征原样传过去
out_a, w_a = attention(X @ Wq_a, X @ Wk_a, X @ Wv_a)

print("【2】自注意力：句子里的每个词都去'看'其他所有词")
print("   句子：小猫追毛线球，它跑得很快")
print(f"   '它'分给每个词的注意力：")
for t, wi in zip(tokens, w_a[4]):
    print(f"     {pad(t, 7)}{wi:.2f} {'█' * int(wi * 30)}")
print(f"   '它'的输出向量里，'动物'这一维 = {out_a[4, 0]:.2f}：它从'小猫'那里拿到了'我指的是一只动物'这条信息。")
print("   （这里的 Wq、Wk 是手工设计的；真实模型里，它们是训练出来的。）\n")

# ---------------------------------------------------------------- 3. 多头 + 因果掩码
# 头 B（前一个词头）：位置 i 的查询去找位置 i-1 的键
shift = np.zeros((d, n))
for i in range(1, n):
    shift[5 + i, i - 1] = 1.0                    # 位置 i 的查询 → 第 i-1 维
Wq_b = shift * 6
Wk_b = np.zeros((d, n)); Wk_b[5:, :] = np.eye(n) * 6   # 位置 j 的键 → 第 j 维
causal = np.tril(np.ones((n, n), dtype=bool))  # 因果掩码：只能看自己和前面的词
_, w_b = attention(X @ Wq_b, X @ Wk_b, X @ Wv_a, mask=causal)

print("【3】多头注意力：几个头同时工作，各自关注不同的关系")
heatmap(w_a, tokens, "   头 A（指代头）：'它'那一行的焦点落在'小猫'上；其他词没有要找的东西，就平均地看所有词")
heatmap(w_b, tokens, "   头 B（前一个词头，加了因果掩码）：每个词都看向它前面那个词，右上角全空")
print("   因果掩码让每个词只能看到它前面的词，这样模型在生成文字时就没法'偷看答案'。")
print("   把两个头的输出拼在一起，每个词就同时知道了'我指代谁'和'我前面是谁'。")
print("   没有位置特征的话，头 B 根本无法工作：注意力本身不知道词的顺序，所以 Transformer 必须加位置编码。\n")

# ---------------------------------------------------------------- 4. 为什么除以 √d
print("【4】为什么要除以 √d？")
print("   维度 d | 不缩放时最大权重的平均值 | 除以 √d 后")
for dim in (4, 64, 512):
    raw, scaled = [], []
    for _ in range(200):
        q = np.random.randn(1, dim)
        k = np.random.randn(16, dim)
        s = q @ k.T
        raw.append(softmax(s).max())
        scaled.append(softmax(s / np.sqrt(dim)).max())
    print(f"   {dim:6d} |          {np.mean(raw):.2f}          |   {np.mean(scaled):.2f}")
print("""   维度越高，点积的数值越大，softmax 就越接近"只选一个"，梯度几乎为 0，模型学不动。
   除以 √d 能把分数拉回正常范围，让注意力保持"有选择、但不极端"。

想一想：
  1. 把实验 1 里的 query 改成"又酸又黄"，结果会是多少钱？
  2. 头 B 去掉因果掩码后，第 0 个词（小猫）会看向哪里？为什么？
  3. 一句话有 n 个词，自注意力要算多少个分数？句子长度翻倍，计算量变成几倍？""")
