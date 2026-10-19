# Chapter 5 — Large Language Models and APIs: A Century-Long Game of "Guess the Next Word"

> When you chat with ChatGPT, Claude or DeepSeek, what they are doing, once you strip it down, is just one thing: **predicting the next token**.
> The idea goes back to 1913, when a Russian mathematician took a long poem by Pushkin and counted it, letter by letter.
> This chapter is about how this game of "guess the next word" has been played all the way to today, and how to use these models through an API.

**How to read this chapter:** Read the stories first, then run the labs, and save the [Concepts](#concepts) for last. The labs use a mock model by default, so no API key is needed.
Starting with this chapter, the labs use the shared client in the repo root, [`common/llm.py`](../common/llm.py), which puts Claude, GPT and DeepSeek behind one common interface.

| When | What happened | Lab / concept |
|---|---|---|
| 1913 | Markov counts the letters of *Eugene Onegin* | 🧪 `markov_1913.py` |
| 1948–1951 | Shannon's n-grams and "guess the letter" experiment | Language models, perplexity |
| 2003–2013 | Neural language models and word vectors | Word embeddings |
| 1994–2016 | BPE: from compression algorithm to tokenizer | 🧪 `bpe_tokenizer.py` |
| 2019–2022 | GPT-2, scaling laws, GPT-3, Chinchilla | Scaling laws |
| 2022 | InstructGPT and RLHF | Pretraining, fine-tuning, RLHF |
| 2025 | DeepSeek-R1 | Reasoning models, open weights |
| — | Temperature and sampling | 🧪 `sampling.py` |
| — | Using large models through an API | 🧪 `chat_api.py` |

---

## Prologue — 1913, St. Petersburg: The Mathematician Who Counted Letters

**Andrey Markov** was a mathematician in St. Petersburg, famous for his stubborn temper.
At the time, another mathematician, **Pavel Nekrasov**, had put forward an argument: the "law of large numbers" in probability theory holds only for **independent** events;
since social statistics (crime rates, for example) are very stable, people's choices must be independent of each other, and from that you can conclude that "free will" exists.
Markov disliked this way of dragging probability theory in to vouch for theology. More importantly, he believed the mathematical premise of the argument was simply wrong:
**events that depend on each other can have stable averages too.**

To prove it, he did something that looked rather dumb: he took Pushkin's novel in verse *Eugene Onegin*
and sorted its first 20,000 letters, one by one, into vowels and consonants. The count came out at about 43% vowels and about 57% consonants.
Then he counted again: when a letter is a vowel, what is the probability that the next one is a vowel? And when it is a consonant?
The two were very different: **the next letter depends on the letter before it**. And yet the statistics of the text as a whole stayed stable.

In 1913, he published this study. This kind of chain, where "the next state depends only on the current state," later came to be called a **Markov chain**.
For the first time, text was treated as a sequence that could be described with probabilities.

> 🧪 **Try it**: run [`markov_1913.py`](./markov_1913.py). Redo Markov's experiment on the text of this repo's first four chapters,
> then train n-gram models of order 1, 2, 3, 4, 6 and 8 and have them continue "neural network".

> 💡 Look ahead: → [Language models](#1-language-models)

---

## Act 1 — 1948–1951: Shannon's "Guess the Letter" Game

In 1948, **Claude Shannon** of Bell Labs published the founding work of information theory, *A Mathematical Theory of Communication*.
It contains a delightful demo: he "built" English with more and more elaborate statistical rules.
- If every letter appears at random with equal probability, what comes out is "XFOML RXKHRJFFJUJ ZLPWCFWKCYJ…", pure gibberish;
- If you draw the letters according to how often they appear in English, it starts to look a bit more like the real thing;
- If you also take into account "what the previous one or two letters were," the output looks more and more like English;
- And if you switch to whole words as the unit, it can even string together a few phrases that read smoothly.

This is the **n-gram language model**: predict the next symbol from the n-1 symbols before it.

In 1951, Shannon published *Prediction and Entropy of Printed English*. He had people look at the beginning of a passage, guess the next letter, and recorded how many guesses it took to get it right.
From the results, he estimated that each letter of English carries only about 0.6 to 1.3 bits of information, far below the 4.7 bits of a random letter.
**Human language is highly redundant, and what comes next can often be guessed.** Seventy years later, the training objective of large models is exactly this game: guess the next token.

> 💡 Look ahead: → [n-grams and data sparsity](#2-n-grams-and-data-sparsity), [Perplexity](#3-perplexity)

---

## Act 2 — 2003–2013: Turning Words into Vectors

n-grams have a fatal weakness: all they can do is "memorize by rote." If the training corpus contains "the cat is sleeping on the sofa,"
the model still knows nothing about "the dog is sleeping on the sofa," because it doesn't know that "cat" and "dog" are alike. The longer the context, the more combinations it has never seen (you can see this in Lab 1).

In 2003, **Bengio** and colleagues published *A Neural Probabilistic Language Model* in the *Journal of Machine Learning Research*.
Their core idea: give every word a **vector** (a few dozen to a few hundred numbers), so that words with similar meanings get similar vectors.
That way, what the model learns from "cat" carries over naturally to "dog." This is the **word embedding**.

In 2013, **Tomas Mikolov** and colleagues at Google released **word2vec**, which made training word vectors fast and cheap.
People were amazed to find semantic relationships hidden in these vectors: compute `king - man + woman`, and the word closest to the resulting vector is `queen`.

Today's large models still start by turning each token into a vector (the "word embeddings" in Lab 4 of Chapter 4).

> 💡 Look ahead: → [Word embeddings](#5-word-embeddings)

---

## Act 3 — 1994–2016: A Compression Algorithm Becomes a Tokenizer

What unit should a model read text in? With "letters" as the unit, the sequences get too long; with "words," the vocabulary gets too big, and new words, typos and names are all unknown to it.

The answer came from an unexpected place. In February 1994, the programmer **Philip Gage** published a short article, *A New Algorithm for Data Compression*, in *The C Users Journal*.
The method was simple: find the **pair of adjacent bytes that occurs most often** in the data, replace it with a new byte that hasn't been used yet, and repeat.
It is called **BPE (byte pair encoding)**.

More than twenty years later, **Rico Sennrich**, Barry Haddow and Alexandra Birch at the University of Edinburgh
applied BPE to machine translation: common words are kept whole, and rare words are split into smaller "subwords."
That keeps the vocabulary size under control without ever running into an "unknown word." Later, GPT-2 changed BPE to run directly on bytes, so it can handle any language and any symbol.
Almost every large model's tokenizer today is a variation on this idea.

> 🧪 **Try it**: run [`bpe_tokenizer.py`](./bpe_tokenizer.py). Train a BPE tokenizer from scratch and see which "words" it learns first,
> then compare how many tokens English, Chinese, rare words and code each take.

> 💡 Look ahead: → [Tokens and BPE](#4-tokens-and-bpe)

---

## Act 4 — 2019–2022: Is Bigger Better?

In 2018, OpenAI released the first GPT (covered in Chapter 1). In February 2019, when they released **GPT-2**, they made an unusual decision:
out of concern about misuse, they **held back** the largest version, with 1.5 billion parameters, for the time being. They released only a small model at first, then released bigger versions in stages over several months.
The largest model wasn't fully released until November 2019. This "staged release" set off a long-running debate in the AI world about openness and safety.

In January 2020, **Jared Kaplan** of OpenAI and colleagues published *Scaling Laws for Neural Language Models*.
They found that a model's performance (its test loss) follows a very smooth, predictable power-law relationship with **the number of parameters, the amount of data and the amount of compute**.
In other words, as long as you put in more resources in proportion, the model gets steadily better. A few months later, **GPT-3** (175 billion parameters) pushed this bet to the extreme,
and also showed a new ability: give it just a few examples in the prompt, with no retraining, and the model can pick up a new task ("in-context learning").

In 2022, DeepMind's **Chinchilla** paper revised this story: they trained a 70-billion-parameter model on 1.4 trillion tokens,
and it substantially outperformed Gopher (280 billion parameters), which had 4 times as many parameters but was trained on only 300 billion tokens.
The conclusion: earlier large models had generally been **"underfed,"** and data and parameters should grow together.

> 💡 Look ahead: → [Scaling laws](#9-scaling-laws)

---

## Act 5 — 2022: From "Good at Continuing" to "Good at Listening"

A pretrained model is great at **continuing** text, but not necessarily at **answering**. Ask it a question, and it may just go on to make up more questions.

In 2022, Long Ouyang and colleagues at OpenAI published the **InstructGPT** paper:
first, have people write high-quality "question + answer" demonstrations and fine-tune the model on them; next, have people rank several of the model's answers and train a "scoring model";
finally, use reinforcement learning to get the model to go after high scores. This method is called **RLHF** (reinforcement learning from human feedback).

The result was surprising: labelers preferred the answers of InstructGPT, with only **1.3 billion parameters**, over those of the original GPT-3, with **175 billion parameters**.
A model's "ability" and "whether it does what people intend" are two different things. In November of the same year, ChatGPT, trained with a similar method, went live.

> 💡 Look ahead: → [Pretraining, fine-tuning and RLHF](#8-pretraining-fine-tuning-and-rlhf)

---

## Act 6 — Jan 2025: The DeepSeek Shock

On January 20, 2025, the Chinese company **DeepSeek** released the reasoning model **DeepSeek-R1** and made its model weights openly available.
On tasks that need reasoning, such as math and programming, its performance came close to the strongest closed models of the time.

What also caused a stir was the cost. In its earlier V3 technical report, DeepSeek wrote that the final official training run of V3 took about 2.788 million H800 GPU hours,
or about $5.6 million at $2 per hour. That figure doesn't include the earlier research and experiments, but it was still far below what people imagined it cost to train a top model.

On January 27, DeepSeek's app shot into the top ranks of the US App Store's free chart, and Nvidia's stock fell about 17% that day, wiping out about $590 billion in market value,
a record single-day loss of market value in the US stock market.

For this repo, DeepSeek matters in one more way: its API is compatible with OpenAI's interface, and that's how Side Quest 1 plugs it in.

> 💡 Look ahead: → [Reasoning models](#10-reasoning-models), [Open weights and APIs](#12-open-weights-and-closed-apis)

---

## Act 7 — Today: Models Become a Service

The biggest models have hundreds of billions of parameters; an ordinary computer simply can't hold them. So most people use large models by calling an **API** that a provider offers over the network:
send a conversation over, get the model's reply back, and pay by the number of tokens used.

APIs have a few pitfalls that beginners are bound to fall into, and Lab 4 demos them one by one:
1. It **doesn't remember you**: every request starts from scratch, and any "memory" depends on you sending the conversation history along with it;
2. So the longer the conversation, the more **expensive** and the **slower** each request gets;
3. Generating takes time, so you use **streaming** to show the output while it is being generated.

> 🧪 **Try it**: run [`sampling.py`](./sampling.py) to see how the sampling parameters change the output,
> then run [`chat_api.py`](./chat_api.py) to experience these three properties of APIs with the mock model or a real model.

> 💡 Look ahead: → [Sampling](#7-sampling-temperature-top-k-top-p), [The basic rules of APIs](#11-the-basic-rules-of-apis)

---

## Epilogue — The Task That Never Changed

From Markov counting letters, to Shannon guessing letters, to today's large models, the task has never changed: **given what came before, predict what comes next**.
Three things did change: the context a model can see has grown longer and longer (from 1 letter to millions of tokens); the way of predicting has gone from counting to neural networks;
and the training data has gone from one long poem to the better part of the internet.

But a model that can only "talk" can't yet "do things" for you. In the next chapter, we connect it to tools and turn it into an **agent**.

---

## Labs

```bash
pip install numpy
python markov_1913.py     # 1913/1948: Markov's experiment and n-gram language models (about 2 seconds)
python bpe_tokenizer.py   # 1994/2016: training a BPE tokenizer from scratch (about 2 seconds)
python sampling.py        # temperature, top-k, top-p
python chat_api.py        # APIs: statelessness, tokens and cost, streaming (uses the mock model by default)

# Switch to a real model (needs the matching API key and SDK: pip install anthropic openai)
python chat_api.py --provider claude      # needs ANTHROPIC_API_KEY
python chat_api.py --provider deepseek    # needs DEEPSEEK_API_KEY
python chat_api.py --provider gpt --interactive
```

| Lab | What you should see | Think about it (also at the end of each script) |
|---|---|---|
| `markov_1913.py` | A 'u' right after a 'q' is 48 times as likely as usual; perplexity falls from 24.1 for the 1-gram to 5.9 for the 4-gram, then levels off (4.8 for the 6-gram, 4.6 for the 8-gram) | What is the 8-gram's perplexity on the training set? |
| `bpe_tokenizer.py` | The first merges are common letter pairs like `' '` + `'t'` and `'h'` + `'e'`, and the 7th is already `' the'`; `' networks'` and `' learn'` are single tokens; the Chinese sentence costs 3 tokens per character | Why not just make every word a token? |
| `sampling.py` | Greedy decoding gets stuck in a "the the the" loop; at temperature 1.8, 44% of the pieces are newly spliced together | How do you make a model give the same answer every time? |
| `chat_api.py` | Without the history, the model doesn't know your name; over 8 turns, every turn sends more tokens than the one before | How does the total cost of a 100-turn conversation relate to the number of turns? |

About real-model mode: I tested the real-API path in `common/llm.py` against a local server that mimics the Anthropic and OpenAI API formats (including tool calls and streaming),
but I have not run it with a real API key. If you run into problems, please open an issue.

---

## Concepts

### 1. Language Models

**In one sentence**: given some preceding text, compute the probability of each candidate for the next token.

With that ability, generating text is easy: compute the probabilities, pick one, append it, compute the next one... and keep repeating.
Markov's chains, Shannon's n-grams and today's GPT are all language models; the difference is "how that probability is computed."

📖 First appears in: [Prologue](#prologue--1913-st-petersburg-the-mathematician-who-counted-letters), [Act 1](#act-1--19481951-shannons-guess-the-letter-game)

### 2. n-grams and Data Sparsity

An n-gram model looks only at the previous n-1 tokens and uses "the frequency counted in the corpus" as the probability.
The bigger n is, the longer the context it sees and the more accurate its predictions, but the data it needs grows exponentially: the chance that a long context appeared in the corpus "exactly as is" keeps shrinking.
This is called **data sparsity**. In Lab 1, beyond the 6-gram the perplexity hardly falls any further.

📖 First appears in: [Act 1](#act-1--19481951-shannons-guess-the-letter-game)　🧪 `markov_1913.py`

### 3. Perplexity

Perplexity measures how "surprised" a model is by a piece of text it **has never seen**.
Intuitively: a perplexity of 100 means that each time the model predicts a token, it is, on average, like guessing blindly among 100 candidates. Lower is better.
It is the same thing as the cross-entropy loss used in training: perplexity = e^(average cross-entropy).

🧪 `markov_1913.py`

### 4. Tokens and BPE

A **token** is the smallest unit a model reads and writes: it may be a character, a word, half a word, or even a single byte.
**BPE** starts from single bytes and repeatedly merges the most common adjacent pair into a new token, until the vocabulary reaches a preset size (real models usually have tens of thousands to hundreds of thousands of tokens).

A few practical takeaways:
- Prices, speed and context windows are all counted in **tokens**, not in characters or words;
- The same content can have very different token counts in different languages and on different models;
- A model "sees" text token by token, which is why it sometimes can't count how many letters are in a word.

📖 First appears in: [Act 3](#act-3--19942016-a-compression-algorithm-becomes-a-tokenizer)　🧪 `bpe_tokenizer.py`

### 5. Word Embeddings

Every token corresponds to a vector (in large models, usually thousands of dimensions). Training makes tokens with similar meanings and usage end up with similar vectors.
This solves the n-gram's "rote memorization" problem: the model can generalize from similar contexts.
The first layer of a large model is a huge "embedding table": look up a token's ID in the table to get its vector.

📖 First appears in: [Act 2](#act-2--20032013-turning-words-into-vectors)

### 6. The Context Window

The maximum number of tokens a model can handle at once, including everything you send it and the answer it generates.
Early GPT models had only a few thousand tokens; some models today have reached a million or more (Claude Opus 5.5, for example, has 1 million).
Content beyond the window is invisible to the model. And even within the window, the longer the content, the more expensive and slower the request. Chapter 7 covers how to manage it.

### 7. Sampling: Temperature, top-k, top-p

The model outputs a probability distribution; "which one to pick" is decided by the sampling strategy:
- **Greedy**: always pick the most likely one. Stable, but prone to repeating itself;
- **Temperature**: first "sharpen" the distribution (temperature < 1) or "flatten" it (temperature > 1), then draw;
- **top-k**: draw only from the k most likely candidates;
- **top-p**: add up the probabilities from the top down, and draw only from the candidates it takes to reach probability p; the number of candidates adjusts itself.

Use a low temperature for tasks that need accuracy and stability (code, math, information extraction), and a somewhat higher one for creative tasks.
Note: some newer models don't let you change these parameters when "thinking" mode is on; check each provider's documentation.

🧪 `sampling.py`

### 8. Pretraining, Fine-Tuning and RLHF

Training a chat model usually takes three steps:
1. **Pretraining**: "guess the next token" on a huge amount of text, to learn language and knowledge about the world. This is the most expensive step;
2. **Fine-tuning** (instruction tuning): keep training on human-written "question + good answer" pairs, so the model learns to answer instructions instead of blindly continuing the text;
3. **RLHF**: have people rank several answers, train a scoring model, then use reinforcement learning to make the model's answers better match people's preferences.
   Later, methods also appeared that replace part of the human work with feedback from the AI itself.

📖 First appears in: [Act 5](#act-5--2022-from-good-at-continuing-to-good-at-listening)

### 9. Scaling Laws

A model's performance improves in a predictable, power-law way as the **number of parameters, the amount of data and the amount of compute** grow.
They turned training large models from "trying your luck" into "engineering you can plan": experiment with small models first, then extrapolate how many resources a large model needs and what results it can reach.
Chinchilla added one more point: parameters and data should grow together in proportion; piling on parameters alone is a waste.

📖 First appears in: [Act 4](#act-4--20192022-is-bigger-better)

### 10. Reasoning Models

An ordinary chat model starts writing the answer as soon as it gets the question. A **reasoning model** first generates a long internal "thinking process":
it breaks the problem down, tries things, checks and corrects itself, and only then gives the answer. Reasoning models are usually trained with large-scale reinforcement learning: on tasks like math and programming, where right and wrong can be checked automatically,
the ways of thinking that end in a correct answer get rewarded. DeepSeek-R1, Claude's "extended thinking" and OpenAI's o series all belong to this category.
The price is that they are slower and use more tokens.

📖 First appears in: [Act 6](#act-6--jan-2025-the-deepseek-shock)

### 11. The Basic Rules of APIs

- **Stateless**: every request is independent. For a multi-turn conversation, you have to keep the history yourself and send the whole thing every time;
- **Message roles**: `system` (the overall setup and rules for the model), `user` (what the user says), `assistant` (the model's earlier answers);
- **Billed by the token**: input and output are priced separately, and output is usually more expensive. Because a multi-turn conversation resends the history every time, the total cost grows roughly with the square of the number of turns;
- **Streaming**: the server sends the output while it is generating it, so the user sees the beginning sooner;
- **API key**: this is like your account password. Keep it only in an environment variable; never write it into your code or commit it to GitHub.

📖 First appears in: [Act 7](#act-7--today-models-become-a-service)　🧪 `chat_api.py`

### 12. Open Weights and Closed APIs

- **Closed APIs** (Claude, GPT and others): the model runs only on the provider's servers, and you use it through an API;
- **Open weights** (DeepSeek-R1, Llama and others): the model's parameters can be downloaded, and if you have the hardware you can run, modify and fine-tune it on your own machine.

Each has its trade-offs: closed APIs are the least hassle to use, and usually the strongest; open weights give you more control and privacy, but you need your own compute.
Many open-weight models are also offered through an API, DeepSeek for example.

📖 First appears in: [Act 6](#act-6--jan-2025-the-deepseek-shock)

---

## References

- Wikipedia, [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov); B. Hayes, *First Links in the Markov Chain*, *American Scientist*, 2013
- C. E. Shannon, *A Mathematical Theory of Communication*, *Bell System Technical Journal*, 1948
- C. E. Shannon, [Prediction and Entropy of Printed English](https://www.nokia.com/bell-labs/publications-and-media/publications/prediction-and-entropy-of-printed-english/), 1951
- Y. Bengio et al., [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html), *JMLR*, 2003
- T. Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013 (word2vec)
- T. Mikolov et al., [Linguistic Regularities in Continuous Space Word Representations](https://aclanthology.org/N13-1090.pdf), 2013
- P. Gage, [A New Algorithm for Data Compression](https://jacobfilipp.com/DrDobbs/articles/CUJ/1994/9402/gage/gage.htm), *C Users Journal*, 1994
- R. Sennrich, B. Haddow, A. Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016; Wikipedia, [Byte-pair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding)
- I. Solaiman et al., [Release Strategies and the Social Impacts of Language Models](https://arxiv.org/abs/1908.09203), 2019 (GPT-2's staged release)
- J. Kaplan et al., *Scaling Laws for Neural Language Models*, 2020
- J. Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), 2022 (Chinchilla)
- L. Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, 2022; OpenAI, [Aligning language models to follow instructions](https://openai.com/index/instruction-following/)
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024; *DeepSeek-R1*, 2025
- Business Standard, [DeepSeek sparks AI stock selloff; Nvidia loses record $593 bn in mcap](https://www.business-standard.com/amp/markets/news/deepseek-sparks-ai-stock-selloff-nvidia-loses-record-593-bn-in-mcap-125012800095_1.html), 2025

---

⬅️ [Chapter 4 · From CNNs to Transformers](../04-cnn-to-transformer/) | [Contents](../README.md) | ➡️ [Chapter 6 · The Agent Loop](../06-agent-loop/)
