# Chapter 4 — From CNNs to Transformers: Giving Networks Eyes and Attention

> The networks of Chapter 3 are "universal," but faced with pictures and sentences, they don't know how to look.
> This chapter is about how researchers took inspiration from **a cat's brain** to design convolutional networks made for seeing images;
> and how, step by step, they solved the problem of "forgetting things while reading a long sentence," arriving at the architecture every large model shares today: **the Transformer**.

**How to read this chapter:** Read the stories first, then run the labs, and save the [Concepts](#concepts) for last. These labs need only Python and NumPy, so no API key is needed.

| When | What happened | Lab / concept |
|---|---|---|
| 1959 | Hubel and Wiesel discover "edge detectors" in the cat's visual cortex | 🧪 `convolution.py` |
| 1980 | Kunihiko Fukushima's Neocognitron | Convolution, pooling |
| 1989–2015 | Convolutional networks: from LeNet to ResNet | 🧪 `transformer_block.py` (residual connections) |
| 1997 | LSTM | 🧪 `rnn_vs_attention.py` |
| 2014 | seq2seq and Bahdanau attention | 🧪 `attention.py` |
| 2016 | Google Translate switches to neural networks | — |
| 2017 | Transformer | 🧪 `attention.py`, `transformer_block.py` |
| 2020 | ViT: an image is worth 16×16 words | — |

---

## Prologue — 1959: The Edge of a Slide

In the late 1950s, in a lab at Johns Hopkins University, two young neurophysiologists,
**David Hubel** and **Torsten Wiesel**, wanted to find out what the individual neurons in a cat's visual cortex were actually "looking at."

They put electrodes into the visual cortex of an anesthetized cat and used a projector to throw all kinds of patterns onto a screen: bright spots, dark spots... The neurons didn't respond at all.
As Hubel recalled in his Nobel lecture, they tried for hours with no result. Then, just as they were sliding a glass slide into the projector,
the loudspeaker suddenly burst into the dense crackle of a neuron firing like a machine gun.
What excited it wasn't the pattern on the slide but the line cast on the screen by **the edge of the slide**.

From this they discovered that some neurons in the visual cortex ("simple cells") respond only to **edges at a particular orientation**.
Turn the line to a different angle, and the response gets weaker. Other neurons ("complex cells") respond to edges at the same orientation but don't care exactly where they appear.
Vision is processed **in layers**: first find the edges, then combine edges into more complex shapes.

In 1981, the two of them won the Nobel Prize in Physiology or Medicine for these discoveries.

> 🧪 **Try it**: run [`convolution.py`](./convolution.py). Write a 3×3 "edge detector" by hand,
> then recreate Hubel and Wiesel's experiment: show it lines at different angles and plot its "orientation tuning curve."

> 💡 Look ahead: → [Convolution](#1-convolution)

---

## Act 1 — 1980, Tokyo: The Neocognitron

**Kunihiko Fukushima**, a researcher at NHK's broadcasting technology research laboratories in Japan, carried Hubel and Wiesel's findings straight into neural networks.
In 1980 he published the **Neocognitron** in the journal *Biological Cybernetics*, and its goal was written right into the paper's title:
"a Mechanism of Pattern Recognition **Unaffected by Shift in Position**."

His network alternated between two kinds of layers:
- **S layers** imitate simple cells, using small local detectors to look for particular patterns;
- **C layers** imitate complex cells, merging nearby responses so that recognition doesn't care about a pattern's exact position.

This is the early form of the "convolution layer + pooling layer" in today's convolutional networks. But the Neocognitron wasn't trained with backpropagation.
As Chapter 1 told it, it was in 1989 that Yann LeCun at Bell Labs combined a similar structure with backpropagation, and that gave us the **convolutional neural network (CNN)** in the full sense.

> 💡 Look ahead: → [Pooling and translation invariance](#3-pooling-and-translation-invariance)

---

## Act 2 — 1989–2015: Deeper and Deeper Convolutional Networks

Earlier chapters have already told part of what happened to convolutional networks next: LeCun's networks reading checks, and AlexNet stunning everyone on ImageNet in 2012.
After that, networks kept getting deeper, and also harder and harder to train (the vanishing gradients of Chapter 3).

In 2015, **Kaiming He**, Xiangyu Zhang, Shaoqing Ren and Jian Sun at Microsoft Research Asia introduced the **residual network (ResNet)**.
The idea is surprisingly simple: instead of outputting a new result directly, each layer outputs "a change to its input," and then the input is added back unchanged:
`output = input + f(input)`. This "shortcut" lets signals and gradients pass through many layers without obstruction.

They trained a network with **152 layers** and won the classification task of the 2015 ImageNet challenge (ILSVRC) with a 3.57% error rate.
Residual connections later became standard equipment in almost every deep network, the Transformer included.

> 🧪 **Try it**: run [`transformer_block.py`](./transformer_block.py) and look at part 3. Stack 48 layers and compare having residual connections with not having them.

> 💡 Look ahead: → [Residual connections](#9-residual-connections-and-layer-normalization)

---

## Act 3 — 1997: The Forgetful Recurrent Network, and a Conveyor Belt

Images are two-dimensional, but language is a sequence, one thing after another. The most natural way to handle a sequence is a **recurrent neural network (RNN)**:
like a person reading a book, it reads one word after another, keeps "what it has understood so far" in a hidden state, and updates it with every new word.

The problem is exactly what that 1991 diploma thesis from Chapter 3 found: for information from word 1 to reach word 100, it has to go through 99 updates,
each one discounting it a little, until almost nothing is left. RNNs have a hard time remembering long-range relationships.

In 1997, **Hochreiter** and **Schmidhuber** published **long short-term memory (LSTM)** in *Neural Computation*.
It added a "conveyor belt" to the RNN (the paper calls it the "constant error carousel"), and used a few "gates" to control what gets written onto the belt, what gets erased, and what gets read.
As long as the "forget gate" stays open, information can travel a long way along the belt almost without loss. The paper showed that an LSTM can learn dependencies spanning more than 1,000 time steps.

For nearly twenty years after that, the LSTM was the workhorse for sequence data such as speech and text.

> 🧪 **Try it**: run [`rnn_vs_attention.py`](./rnn_vs_attention.py). Compare how much word 1 influences word 100:
> a plain RNN keeps only 10⁻⁴⁸, an LSTM still keeps 0.37, and with attention it is always just one step away.

> 💡 Look ahead: → [RNNs and LSTMs](#4-rnns-and-lstms)

---

## Act 4 — 2014: The Bottleneck, and "Looking Back"

In 2014, **Sutskever**, **Oriol Vinyals** and **Quoc V. Le** at Google published the **seq2seq** (sequence-to-sequence) model:
one LSTM "reads" a whole English sentence into a fixed-length vector, and another LSTM "writes" that vector out as French.
It reached a BLEU score of 34.8 on English-to-French translation, showing the promise of neural networks for translation.

But the design has an obvious **bottleneck**: however long the sentence, it has to be squeezed into a vector of the same length. Once sentences get long, translation quality drops noticeably.

The same year, **Dzmitry Bahdanau**, a Belarusian student interning in **Bengio**'s lab at the University of Montreal,
published *Neural Machine Translation by Jointly Learning to Align and Translate* together with Kyunghyun Cho and Bengio.
Their approach: instead of squeezing the whole sentence into one vector, keep a representation of every word in the source sentence; **every time the model produces a word of the translation, it "looks back" over the source sentence**,
gives each source word a weight according to how relevant it is, and takes the weighted average.

This is the **attention mechanism**. According to Bahdanau's own recollection in a 2024 email to Karpathy, the name "attention" came from Bengio.

Two years later, these techniques entered the lives of hundreds of millions of people. In September 2016, Google released **GNMT**, a neural machine translation system built on LSTMs and attention.
The first language pair to go live was the notoriously hard **Chinese to English**: all Chinese-to-English translation on the Google Translate website and mobile app switched to GNMT, about 18 million translations a day.

> 🧪 **Try it**: run [`attention.py`](./attention.py). Think of attention as a "fuzzy dictionary lookup," then let the word "it" in a sentence find "kitten" on its own.

> 💡 Look ahead: → [seq2seq and the bottleneck](#5-seq2seq-and-the-bottleneck), [Attention](#6-attention-q-k-v)

---

## Act 5 — 2017: Attention Is All You Need

Chapter 1 told the story of the Transformer paper's eight authors and where its name came from. Here we look at what it did technically.

Bahdanau's attention was still built on top of an RNN: the RNN read one word at a time, and attention was only a helper.
The 2017 Transformer took the most radical step: **throw out recurrence completely**, and build the whole network from attention alone.
- **Self-attention**: every word in the sentence looks at all the other words at the same time, so a connection between any two words takes a single step;
- **Multiple heads**: several sets of attention run at once, each watching a different relationship;
- **Positional encoding**: attention itself doesn't know the order of the words, so each position gets its own unique "number tag" added on;
- **Residual connections + layer normalization**: these let the network be stacked very deep (from earlier work such as ResNet).

Since nothing has to be computed step by step in order anymore, a whole sentence can be handed to the GPU at once and processed in parallel. The paper's base model trained for only 12 hours on 8 P100 GPUs,
and the big model trained for 3.5 days, reaching a BLEU score of 28.4 on English-to-German translation, better than every model at the time, including ensembles of several models.

> 🧪 **Try it**: run [`transformer_block.py`](./transformer_block.py). Assemble the parts into a complete Transformer,
> and see the shape of a sentence as it flows through each layer, and where all the parameters go.

> 💡 Look ahead: → [Self-attention and multi-head attention](#7-self-attention-and-multi-head-attention), [Positional encoding](#8-positional-encoding), [The Transformer block](#10-the-transformer-block)

---

## Act 6 — 2020: An Image Is Worth 16×16 Words

The Transformer was originally designed for language. In October 2020, **Alexey Dosovitskiy** and colleagues at Google
published a paper with a fun title: *An Image is Worth 16x16 Words*.

They cut an image into small 16×16-pixel squares, treated each square as a "word," and handed them straight to a Transformer.
No convolutions, no structure designed especially for images. The result: given enough pre-training data, this **Vision Transformer (ViT)**
matched or even beat the best convolutional networks, while needing less compute to train.

From then on, the Transformer became a "general-purpose architecture": text, images, speech, proteins can all be cut into a string of "words" and handed to it.
Today's GPT, Claude and DeepSeek are all built on it.

---

## Epilogue — Two Ways of "Seeing"

| | CNNs | Transformers |
|---|---|---|
| Inspiration | The cat's visual cortex | "Looking back at the source" while translating |
| What it looks at | Small local regions, widening layer by layer | Everything, in one step |
| Prior knowledge | "Locality" and "translation invariance" built in | Almost none; everything is learned from data |
| With little data | Has the edge | Struggles to learn well |
| With lots of data | Gains slow down | The more data, the stronger |

Behind this lies a lesson that runs through the whole history of AI: **built-in human knowledge is very useful when data is scarce, but in the face of massive data and compute,
more general methods usually have the last laugh.** In 2019, the reinforcement-learning pioneer Richard Sutton wrote this up in a short essay titled *The Bitter Lesson*. In the next chapter, we'll see what happened once Transformers were scaled up enormously: large language models.

---

## Labs

```bash
pip install numpy
python convolution.py        # 1959/1980: edge detectors, orientation tuning curves, pooling
python attention.py          # 2014/2017: attention by hand, multiple heads, causal mask
python rnn_vs_attention.py   # 1997 vs 2017: long-range dependencies and parallel computation
python transformer_block.py  # 2017: assemble a complete Transformer and see what residual connections do
```

| Lab | What you should see | Think about it (also at the end of each script) |
|---|---|---|
| `convolution.py` | The vertical detector lights up only on vertical edges; the tuning curve peaks at 90°; after pooling, a shift barely affects the result | Stack two convolution layers: how big a region can they see? |
| `attention.py` | "it" gives all its attention to "kitten"; the "previous-word head" draws a diagonal line; without dividing by √d, the weights become extreme | If the sentence doubles in length, how many times more computation is that? |
| `rnn_vs_attention.py` | At distance 100, the RNN keeps 10⁻⁴⁸ and the LSTM keeps 0.37; the RNN needs n sequential steps, attention just 1 | Why was Bahdanau attention still built on top of an RNN? |
| `transformer_block.py` | The shape stays (10, 64) at every layer; the feed-forward network has 2 times the parameters of attention; without residual connections, after 6 layers every word's vector becomes identical | Without positional encoding, can the model still tell "the cat chases the dog" from "the dog chases the cat"? |

---

## Concepts

### 1. Convolution

Slide a small window (the kernel, say 3×3) across the whole image. At each position, multiply the pixels in the window by the matching kernel entries and add them up.
The result is a new "feature map" that says "how strongly this pattern shows up at each position."

Convolution has two key properties:
- **Local connections**: each output looks at only a small patch of the input;
- **Weight sharing**: the same kernel is reused at every position.

Together, these cut the number of parameters by several orders of magnitude, and they also give the network a built-in sense that "the same pattern in a different place is the same thing."
In a CNN the kernels are learned. After training, the first layer's kernels often really do look like edge detectors, just like the cat's simple cells.

📖 First appears in: [Prologue](#prologue--1959-the-edge-of-a-slide)　🧪 `convolution.py`

### 2. Receptive fields

The region of the input a neuron can "see" is called its receptive field. A 3×3 convolution sees a 3×3 region; stack two and it's 5×5; the more layers, the bigger the view.
That is why a CNN builds its understanding **from local to global**, layer by layer: shallow layers see edges, middle layers see textures and parts, deep layers see whole objects.

### 3. Pooling and translation invariance

**Max pooling** keeps only the largest value in a small region (say, 2×2).
It only cares whether "this feature shows up somewhere in this patch," not exactly which pixel it's on. Shift the image a little, and the pooled result barely changes. This is called **translation invariance**.
It is exactly what Fukushima's "C layers" do, corresponding to Hubel and Wiesel's "complex cells."

📖 First appears in: [Act 1](#act-1--1980-tokyo-the-neocognitron)　🧪 `convolution.py`

### 4. RNNs and LSTMs

**RNN**: `h_t = tanh(W·h_{t-1} + U·x_t)`. For each word it reads, it computes a new state from "the old state + the new word."
It is a natural fit for sequences, but at every step the information gets multiplied by a matrix once more, so over long distances it vanishes (or explodes).

An **LSTM** adds a separate **memory cell** c, plus three gates:
- **Forget gate**: how much of the old memory to keep;
- **Input gate**: how much new information to write in;
- **Output gate**: how much to output.

Updating the memory cell is mainly "multiply by the forget gate + add new content." When the forget gate is close to 1, information and gradients can both travel a long way almost without loss.

📖 First appears in: [Act 3](#act-3--1997-the-forgetful-recurrent-network-and-a-conveyor-belt)　🧪 `rnn_vs_attention.py`

### 5. seq2seq and the bottleneck

**seq2seq** = encoder + decoder: the encoder reads the input sequence into a vector, and the decoder generates the output sequence from that vector.
Translation, summarization and dialogue can all be seen as seq2seq problems.
Its **bottleneck**: the information of the whole sentence is squeezed into one fixed-length vector, and the longer the sentence, the more gets lost. Attention was born precisely to solve this problem.

📖 First appears in: [Act 4](#act-4--2014-the-bottleneck-and-looking-back)

### 6. Attention (Q, K, V)

You can think of attention as a "fuzzy dictionary lookup":
- **Query**: what I'm looking for;
- **Key**: each candidate's "label";
- **Value**: each candidate's actual content.

How it's computed:
1. Take the dot product of the query with each key to get a "match score";
2. Divide by √d (d is the vector dimension) to keep the scores from getting too big;
3. Use softmax to turn the scores into weights that add up to 1;
4. Use the weights to take a weighted average of all the values.

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
```

Unlike a normal dictionary, attention doesn't "take just one"; it "takes a little of each, according to relevance." This makes it differentiable everywhere, so it can be trained with backpropagation.

📖 First appears in: [Act 4](#act-4--2014-the-bottleneck-and-looking-back)　🧪 `attention.py`

### 7. Self-attention and multi-head attention

**Self-attention**: Q, K and V all come from the same sentence. Every word uses its own query to match against the keys of all the words in the sentence.
That way, "it" can find "kitten" directly, no matter how many words lie in between.

**Multi-head attention**: use several sets of Q/K/V projections at once (several "heads"). Each head can learn to watch a different relationship:
what a word refers to, grammar, which words sit next to each other... Finally, the heads' results are concatenated.

**Causal mask**: when generating text, each word may only look at the words before it (the upper-right corner of the attention matrix is masked out);
otherwise the model could "peek at the answer" during training. Models like GPT all use a causal mask.

📖 First appears in: [Act 5](#act-5--2017-attention-is-all-you-need)　🧪 `attention.py`

### 8. Positional encoding

Attention by itself is "orderless": shuffle the words in a sentence, and the attention result each word gets doesn't change. But "the cat chases the dog" and "the dog chases the cat" mean completely different things.
So each position gets a unique vector added to it (the **positional encoding**). The 2017 paper combined sine and cosine waves of different frequencies;
today's large models mostly use improved methods such as rotary positional embeddings (RoPE).

📖 First appears in: [Act 5](#act-5--2017-attention-is-all-you-need)　🧪 `attention.py`, `transformer_block.py`

### 9. Residual connections and layer normalization

- **Residual connections**: `output = input + f(input)`. Each layer only learns "a change," and the original information has a direct shortcut through.
  This solved the vanishing-gradient problem of deep networks, and it also prevents what you saw in Lab 4, where "every word's vector becomes identical" (researchers call this rank collapse).
- **Layer normalization**: adjust each word's vector to mean 0 and variance 1, so the numbers at every layer stay in a stable range and training goes more smoothly.

📖 First appears in: [Act 2](#act-2--19892015-deeper-and-deeper-convolutional-networks)　🧪 `transformer_block.py`

### 10. The Transformer block

One Transformer block = **multi-head self-attention** (words exchange information with each other) + a **feed-forward network** (each word goes through a two-layer transformation on its own),
with each part wrapped in a residual connection and layer normalization. Stack dozens to a hundred or more of these blocks, and you have the body of a large model.

The parameters live mainly in three places: the word embeddings and output layer, the Q/K/V/O matrices of attention, and the feed-forward networks. The feed-forward networks usually take the largest share.

📖 First appears in: [Act 5](#act-5--2017-attention-is-all-you-need)　🧪 `transformer_block.py`

### 11. ViT

Cut an image into small patches of a fixed size (say, 16×16 pixels), flatten each patch and treat it as a "word," add positional encoding, and hand it to a standard Transformer.
It shows that the Transformer is a general-purpose sequence processor: anything that can be cut into "a string of words" can be learned with it.

📖 First appears in: [Act 6](#act-6--2020-an-image-is-worth-1616-words)

---

## References

- D. Hubel, [Evolution of Ideas on the Primary Visual Cortex, 1955–1978](https://www.nobelprize.org/prizes/medicine/1981/hubel/lecture/) (Nobel lecture, 1981); [An accidental experiment discovered new cells in cat brains](https://next.massivesci.com/notes/simple-complex-cells-neurons-cats-eyes/)
- K. Fukushima, [Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position](https://www.cs.princeton.edu/courses/archive/spring08/cos598B/Readings/Fukushima1980.pdf), *Biological Cybernetics*, 1980
- K. He, X. Zhang, S. Ren, J. Sun, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015
- S. Hochreiter & J. Schmidhuber, *Long Short-Term Memory*, *Neural Computation* 9(8), 1997
- I. Sutskever, O. Vinyals, Q. V. Le, *Sequence to Sequence Learning with Neural Networks*, NIPS 2014
- D. Bahdanau, K. Cho, Y. Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2014
- Turing Post, [Who came up with the term "attention"?](https://www.turingpost.com/p/attention), 2024 (Bahdanau's email to Karpathy)
- Google, [A Neural Network for Machine Translation, at Production Scale](https://research.google/blog/a-neural-network-for-machine-translation-at-production-scale/), 2016
- A. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017
- A. Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), 2020
- R. Sutton, [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), 2019
- Y. Dong, J.-B. Cordonnier, A. Loukas, [Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth](https://arxiv.org/abs/2103.03404), 2021 (rank collapse)

---

⬅️ [Chapter 3 · Neural Networks and Autodiff](../03-neural-nets/) | [Contents](../README.md) | ➡️ [Chapter 5 · Large Language Models and APIs](../05-llm-and-api/)
