"""
Lab 2: Training a BPE tokenizer from scratch
============================================

Large language models don't read "characters" or "words"; they read **tokens**. How are tokens cut out of text?
The method the GPT series uses today goes back to a **compression algorithm** Philip Gage published in *The C Users Journal* in 1994:
  over and over, find the "pair of adjacent units" that occurs most often in the data, and merge it into one new unit.
In 2016, Sennrich and colleagues applied it to machine translation; GPT-2 then changed it to run on **bytes**.

Here we implement byte-level BPE from scratch:
  1. Train it on this repo's English corpus and see which "words" it learns first
  2. Use it to tokenize English, Chinese and code, and count the tokens
  3. Understand why, when you're "billed by the token," different languages cost different amounts

Run: python bpe_tokenizer.py
"""

import re
from collections import Counter
from pathlib import Path

N_MERGES = 400
# First cut the text into small chunks; merges only happen inside a chunk (GPT-2 does this too): runs of CJK characters / English words / numbers / other symbols
CHUNK = re.compile(r"[一-鿿]+| ?[A-Za-z]+| ?\d+|\s+|[^\sA-Za-z\d一-鿿]+")


def load_corpus():
    text = Path(__file__).with_name("data").joinpath("corpus.txt").read_text()
    return "".join(line for line in text.splitlines(keepends=True) if not line.startswith("#"))


def train_bpe(text, n_merges):
    # Each chunk starts out as a sequence of bytes (an English letter is 1 byte, a Chinese character is 3)
    chunks = Counter(tuple(c.encode("utf-8")) for c in CHUNK.findall(text))
    vocab = {i: bytes([i]) for i in range(256)}           # initial vocabulary: the 256 bytes
    merges = []
    for step in range(n_merges):
        pairs = Counter()
        for seq, freq in chunks.items():
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        (a, b), freq = pairs.most_common(1)[0]           # the adjacent pair that occurs most often
        new_id = 256 + step
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append(((a, b), new_id, freq))
        new_chunks = Counter()
        for seq, f in chunks.items():                    # replace every (a, b) with the new id
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
        for (a, b), new_id, _ in merges:                 # apply every merge rule, in the order it was learned
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
        return "<" + raw.hex(" ") + ">"                   # part of a character (incomplete bytes)


if __name__ == "__main__":
    text = load_corpus()
    print(f"Corpus: {len(text):,} characters = {len(text.encode('utf-8')):,} bytes\n")
    vocab, merges = train_bpe(text, N_MERGES)

    print(f"[1] Training: {len(merges)} merges, and the vocabulary grows from 256 bytes to {len(vocab)} tokens")
    print("   The first 12 merges it learned (the most frequent adjacent pairs):")
    for (a, b), new_id, freq in merges[:12]:
        print(f"     {show_token(vocab, a)!r:>10} + {show_token(vocab, b)!r:<10} → {show_token(vocab, new_id)!r:<12} seen {freq} times")
    long_tokens = sorted((v for v in vocab.values() if len(v) >= 6), key=len, reverse=True)
    words = []
    for v in long_tokens:
        try:
            words.append(v.decode("utf-8"))
        except UnicodeDecodeError:
            pass
    print("   Some of the long tokens it learned: " + ", ".join(repr(w) for w in words[:15]))
    print("   In UTF-8 an English letter is a single byte, so the earliest merges glue common letter pairs together (' ' + 't', 'h' + 'e');")
    print("   only after that are those pieces put together into words (' t' + 'he' → ' the'). A leading space is part of the token.\n")

    print("[2] Tokenizing: the same meaning, written different ways. How many tokens does each take?")
    samples = [
        ("English", "Neural networks learn through backpropagation."),
        ("Chinese", "神经网络通过反向传播来学习。"),
        ("English (rare words)", "The gourmand dithered before the mandarin fish."),
        ("Code", "loss.backward()  # compute gradients"),
    ]
    for name, s in samples:
        ids = encode(s, merges)
        pieces = " | ".join(show_token(vocab, i) for i in ids)
        print(f"   {name}: {len(s)} characters → {len(ids)} tokens")
        print(f"     {pieces}")
    print("""
   * Words that are common in the training corpus (like ' networks' and ' learn') get merged into a single token; less common ones (like 'backpropagation') are still cut into pieces;
   * Rare words the corpus has never seen fall apart into small pieces, often just a letter or two per token (' gourmand' → ' g | ou | r | m | and');
   * This tokenizer was trained on an English corpus with no Chinese in it, so the Chinese sentence is cut up very finely: every character falls back to its 3 UTF-8 bytes, 3 tokens per character.
     Real models' tokenizers are trained on huge multilingual datasets, but the same pattern still holds: the bigger a language's share of the training data, the more information each token can carry.

   This is about money: LLM APIs charge by the token, and context windows are measured in tokens too.
   The same content, in another language or on another provider's model (with a different tokenizer), can take tens of percent more or fewer tokens.

Think about it:
  1. Change N_MERGES to 50 and to 2000. How does the token count for the same sentences change? What does a bigger vocabulary cost?
  2. Why not just make every word a token? (Hint: think about rare words and other languages)
  3. What pieces is 'loss.backward()' cut into? When a large model writes code, why are "spaces" and "indentation" tokens too?""")
