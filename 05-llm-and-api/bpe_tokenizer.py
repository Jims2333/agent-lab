"""
实验 2：从零训练一个 BPE 分词器
=================================

大模型读的不是"字"，也不是"词"，而是 **token**。token 是怎么切出来的？
今天 GPT 系列用的方法，源头是 1994 年菲利普·盖奇在《C 用户杂志》上发表的一个**压缩算法**：
  反复找出数据里出现次数最多的"相邻两个单位"，把它们合并成一个新单位。
2016 年，森里奇（Sennrich）等人把它用到了机器翻译上；GPT-2 又把它改成在**字节**上运行。

这里从零实现字节级 BPE：
  1. 在本仓库的中文语料上训练，看它先学会了哪些"词"
  2. 用它给中文、英文、代码分词，数一数 token
  3. 理解为什么"按 token 计费"时，不同语言的价格会不一样

运行：python bpe_tokenizer.py
"""

import re
from collections import Counter
from pathlib import Path

N_MERGES = 400
# 先粗切成小块，合并只在块内部进行（GPT-2 也是这么做的）：连续的汉字 / 英文单词 / 数字 / 其他符号
CHUNK = re.compile(r"[一-鿿]+| ?[A-Za-z]+| ?\d+|\s+|[^\sA-Za-z\d一-鿿]+")


def load_corpus():
    text = Path(__file__).with_name("data").joinpath("corpus.txt").read_text()
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("#"))


def train_bpe(text, n_merges):
    # 每个小块先表示成一串字节（一个汉字是 3 个字节）
    chunks = Counter(tuple(c.encode("utf-8")) for c in CHUNK.findall(text))
    vocab = {i: bytes([i]) for i in range(256)}           # 初始词表：256 个字节
    merges = []
    for step in range(n_merges):
        pairs = Counter()
        for seq, freq in chunks.items():
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        (a, b), freq = pairs.most_common(1)[0]           # 出现最多的相邻一对
        new_id = 256 + step
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append(((a, b), new_id, freq))
        new_chunks = Counter()
        for seq, f in chunks.items():                    # 把所有 (a, b) 替换成新编号
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            new_chunks[tuple(out)] += f
        chunks = new_chunks
    return vocab, merges


def encode(text, merges):
    ids = []
    for chunk in CHUNK.findall(text):
        seq = list(chunk.encode("utf-8"))
        for (a, b), new_id, _ in merges:                 # 按训练时的顺序，依次应用每一条合并规则
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        ids.extend(seq)
    return ids


def show_token(vocab, tid):
    raw = vocab[tid]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + raw.hex(" ") + ">"                   # 半个汉字（不完整的字节）


if __name__ == "__main__":
    text = load_corpus()
    print(f"语料：{len(text):,} 个字符 = {len(text.encode('utf-8')):,} 个字节\n")
    vocab, merges = train_bpe(text, N_MERGES)

    print(f"【1】训练：做了 {len(merges)} 次合并，词表从 256 个字节长到 {len(vocab)} 个 token")
    print("   最先学会的 12 个合并（出现最多的相邻一对）：")
    for (a, b), new_id, freq in merges[:12]:
        print(f"     {show_token(vocab, a)!s:>10} + {show_token(vocab, b)!s:<10} → {show_token(vocab, new_id)!r:<12} 出现 {freq} 次")
    long_tokens = sorted((v for v in vocab.values() if len(v) >= 6), key=len, reverse=True)
    words = []
    for v in long_tokens:
        try:
            words.append(v.decode("utf-8"))
        except UnicodeDecodeError:
            pass
    print("   学到的一些多字 token：" + "、".join(words[:15]))
    print("   一个汉字在 UTF-8 里是 3 个字节，所以最早的合并往往是在'拼出常见汉字'；之后才把常见的字拼成词。\n")

    print("【2】分词：同一个意思，不同的写法，各要多少个 token？")
    samples = [
        ("中文", "神经网络通过反向传播来学习。"),
        ("英文", "Neural networks learn through backpropagation."),
        ("中文（生僻）", "饕餮之徒在鳜鱼面前踟蹰。"),
        ("代码", "loss.backward()  # compute gradients"),
    ]
    for name, s in samples:
        ids = encode(s, merges)
        pieces = " | ".join(show_token(vocab, i) for i in ids)
        print(f"   {name}：{len(s)} 个字符 → {len(ids)} 个 token")
        print(f"     {pieces}")
    print("""
   * 训练语料里常见的词（比如'神经网络'、'学习'）会被合并成一个 token；没那么常见的（比如'传播'）还是一个字一个字地切；
   * 语料里没见过的生僻字，只能拆回字节，一个字要用 2～3 个 token；
   * 这个分词器是在中文语料上训练的，所以英文反而被切得很碎。真实模型的分词器在海量多语言数据上训练，
     但同样的规律依然存在：在训练数据里占比越高的语言，每个 token 能装的信息越多。

   这件事和钱有关：大模型 API 按 token 计费，上下文窗口也按 token 计算。
   同样的内容，换一种语言或者换一家模型（分词器不同），token 数可能差好几成。

想一想：
  1. 把 N_MERGES 改成 50 和 2000，同样的句子 token 数会怎么变？词表变大有什么代价？
  2. 为什么不直接把每个汉字当成一个 token？（提示：想想那些生僻字和其他语言）
  3. 'loss.backward()' 被切成了哪几块？大模型写代码时，为什么"空格"和"缩进"也是 token？""")
