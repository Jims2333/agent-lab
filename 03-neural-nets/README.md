# Chapter 3 — Neural Networks and Autodiff: Teaching Machines to Take Their Own Derivatives

> In Chapter 1 we wrote backpropagation for the XOR network by hand, and we had to work out every derivative ourselves.
> That's fine for a three-layer network. But what about a hundred layers and hundreds of billions of parameters?
> This chapter is about how people handed the job of "taking derivatives" over to the machine completely, and why deep networks are so hard to train.

**How to read this chapter:** Read the stories first, then run the labs, and save the [Concepts](#concepts) for last. These labs need only Python and NumPy, so no API key is needed.

| When | What happened | Lab / concept |
|---|---|---|
| 1970 | Linnainmaa proposes reverse-mode automatic differentiation in his master's thesis | 🧪 `autograd.py` |
| 1989 | The universal approximation theorem | 🧪 `mlp_moons.py` |
| 1991 | Hochreiter analyzes vanishing gradients in his diploma thesis | 🧪 `vanishing_gradient.py` |
| 2010–2011 | Research on initialization, and the rise of ReLU | Activation functions, initialization |
| 2012 | Dropout | Regularization |
| 2015 | He initialization | 🧪 `mlp_moons.py`, `vanishing_gradient.py` |
| 2007–2017 | Theano, TensorFlow, PyTorch | Deep learning frameworks |
| 2020–2022 | Karpathy's micrograd | 🧪 `autograd.py` |

---

## Prologue — 1970, Copenhagen: An Afternoon in the Park

In 1970, **Seppo Linnainmaa**, a student at the University of Helsinki in Finland, was writing his master's thesis.
His problem had nothing to do with neural networks. Every time a computer does a floating-point operation, it makes a tiny rounding error.
As those little errors pile up step after step, how far off does the final result end up?

To answer that, you need to know **how sensitive the final result is to every intermediate step**. In other words, a whole pile of derivatives.
According to a later account by the mathematician Andreas Griewank, Linnainmaa said
the solution came to him on a sunny afternoon in 1970, in a park in Copenhagen:
**record the computation, then start from the final result and walk through it backward**, using the chain rule to pass the sensitivities back one step at a time.
A single backward pass gives you the derivative of the result with respect to every intermediate quantity. He even implemented it in FORTRAN.

This is **reverse-mode automatic differentiation**. More than a decade later, people used it to train neural networks and gave it another name: **backpropagation**.

> 🧪 **Try it**: run [`autograd.py`](./autograd.py). It implements an autodiff engine in under 100 lines of code,
> draws the computational graph, and uses "numerical gradients" to check that it computes correctly.

> 💡 Look ahead: → [Computational graphs](#1-computational-graphs), [Forward mode and reverse mode](#3-forward-mode-and-reverse-mode)

---

## Act 1 — 1989: Is One Layer Enough?

As Chapter 1 told it, in 1986 backpropagation made multi-layer networks hot again. But one basic question was still unanswered:
**how complicated a thing can a multi-layer network actually represent?**

In 1989, two papers gave the answer almost at the same time. The mathematician **George Cybenko**
proved in *Mathematics of Control, Signals and Systems* that with just **one** hidden layer and an S-shaped activation function like the sigmoid,
a network can approximate any continuous function to any accuracy you like. The same year, **Kurt Hornik**, Stinchcombe and White
published a paper with a resounding title in *Neural Networks*: *Multilayer Feedforward Networks are Universal Approximators*.

This is the **universal approximation theorem**. It is reassuring: in principle, there is no shape a neural network "can't draw."
But two things it doesn't say went on to become problems for decades: **how many neurons do you need**? And **how do you actually train the weights**?

> 🧪 **Try it**: run [`mlp_moons.py`](./mlp_moons.py). Use the engine you wrote yourself to train a network
> that separates two interlocking "moons." A network with no hidden layer can only draw a straight line; add hidden layers, and the boundary starts to bend.

> 💡 Look ahead: → [The universal approximation theorem](#5-the-universal-approximation-theorem)

---

## Act 2 — 1991, Munich: The Deeper, the Harder to Learn

On June 15, 1991, **Sepp Hochreiter**, a student at the Technical University of Munich, turned in his diploma thesis (Diplom).
It was titled *Untersuchungen zu dynamischen neuronalen Netzen* ("Studies of Dynamic Neural Networks"), it was written in German, and his advisor was **Jürgen Schmidhuber**.

The thesis analyzed a phenomenon that gave everyone headaches. As the error signal travels backward through the network, it gets multiplied by some factors at every layer.
If those factors are mostly smaller than 1, after a few dozen layers the signal is almost zero, and the layers near the input learn nothing at all. This is called **vanishing gradients**.
If they are mostly larger than 1, the signal grows bigger and bigger as it travels. This is called **exploding gradients**.

Schmidhuber later called this thesis one of the most important documents in the history of machine learning.
Six years later, Hochreiter and Schmidhuber invented the LSTM precisely to get around this problem (more on that in Chapter 4).

> 🧪 **Try it**: run [`vanishing_gradient.py`](./vanishing_gradient.py). Build a 30-layer network
> and see how much of the error signal is left when it gets back to layer 1: the sigmoid network keeps only 10⁻²⁰ of it, while a network with too large an initialization amplifies it more than ten thousand times.

> 💡 Look ahead: → [Vanishing and exploding gradients](#7-vanishing-and-exploding-gradients)

---

## Act 3 — 2010–2015: A New Activation Function and the Right Starting Weights

All through the 2000s, deep networks were known for being "hard to train." Around 2010, a few improvements that looked very plain turned the situation around completely:

- **In 2010**, **Xavier Glorot** and **Yoshua Bengio** of the University of Montreal
  published *Understanding the Difficulty of Training Deep Feedforward Neural Networks*. They carefully analyzed how the signal grows and shrinks from layer to layer,
  and proposed scaling the initial weights by "the number of inputs and outputs." This later became known as **Xavier initialization**.
- **In 2010**, **Vinod Nair** and **Hinton** at the University of Toronto proposed at ICML improving models with **rectified linear units (ReLU)**.
  The rule could hardly be simpler: if the input is above 0, pass it through unchanged; if it is below 0, output 0.
- **In 2011**, *Deep Sparse Rectifier Neural Networks* by Glorot, Bordes and Bengio showed that deep networks using ReLU
  can train very well even without complicated pre-training tricks.
- **In 2012**, the AlexNet paper reported that on the CIFAR-10 dataset, a network with ReLU reached a 25% training error rate
  **six times** faster than one with tanh. From then on, ReLU became the default choice.
- **In 2015**, **Kaiming He** and colleagues at Microsoft Research Asia published *Delving Deep into Rectifiers*. It proposed an initialization method designed specifically for ReLU
  (**He initialization**), which made it possible to train very deep networks from scratch. The paper's model reached a 4.94% top-5 error rate on ImageNet,
  the first result to beat human-level performance (5.1%) on this challenge.

In Lab 2 you'll see for yourself how much initialization matters: same network, same data, but if you pick too small a scale for the initial weights,
accuracy gets stuck at around 90%; switch to He initialization, and it reaches 100% in 80 steps.

> 💡 Look ahead: → [Activation functions](#6-activation-functions), [Initialization](#8-initialization)

---

## Act 4 — 2012: A Lesson from the Bank Tellers

The bigger the network, the easier it is to **overfit**: it memorizes the training data and then falls apart on new data.

As Hinton himself recalled in an online Q&A (on Reddit), he once went to the bank and noticed that the tellers kept changing. He asked one of them why.
The teller didn't know, only that they got moved around a lot. Hinton figured it was probably because defrauding the bank would take several employees working together;
if they were rotated often, they couldn't conspire. That gave him an idea: **at every training step, randomly "switch off" some of the neurons**.
Then they can't form fragile little "cliques," and the network is less likely to overfit.

In July 2012, Hinton and his students posted *Improving Neural Networks by Preventing Co-adaptation of Feature Detectors* on arXiv.
The method is called **Dropout**. AlexNet used it the same year.

> 💡 Look ahead: → [Overfitting, regularization and Dropout](#9-overfitting-regularization-and-dropout)

---

## Act 5 — 2007–2017: No More Derivatives by Hand

In Chapter 1's Lab 3, we had to work out every backpropagation formula ourselves. Change the network structure even a little, and you have to derive it all over again. It is very easy to make mistakes.

**Deep learning frameworks** changed all that. At their core is exactly Linnainmaa's reverse-mode automatic differentiation:
you just write the forward computation, and the framework records the computational graph automatically and then computes all the gradients for you.

- **In 2007**, a research team at the University of Montreal (later MILA) released **Theano**, one of the first Python frameworks to support automatic differentiation and GPUs;
- **On November 9, 2015**, Google open-sourced **TensorFlow**;
- **In January 2017**, Facebook's research team publicly released **PyTorch**. Its "build the graph as the code runs" approach is just as intuitive as our Value class,
  and it quickly became researchers' favorite tool.

From then on, researchers could spend their energy designing new network architectures instead of deriving formulas.

> 💡 Look ahead: → [Deep learning frameworks](#10-deep-learning-frameworks)

---

## Act 6 — 2020–2022: The Whole Idea in a Hundred Lines of Code

As frameworks grew more powerful, their insides grew more complicated too, and it became hard for beginners to see what they were actually doing.

In 2020, **Andrej Karpathy**, former head of AI at Tesla and one of the founding members of OpenAI,
released **micrograd** on GitHub: an autodiff engine of about 100 lines, plus a neural network library of about 50 lines,
with an API a lot like PyTorch's. In August 2022 he also recorded a video, more than two hours long, writing it line by line from scratch.

Labs 1 and 2 in this chapter follow the same path: write the engine yourself, then use it to train a network.
The next time you call `loss.backward()` in PyTorch, you'll know what is happening behind it.

---

## Epilogue — Three Questions

The three stories in this chapter match three key questions in training neural networks:
1. **How do we take derivatives?** Reverse-mode automatic differentiation: one backward pass gives you all the gradients;
2. **Can a network represent the answer?** The universal approximation theorem says yes, but it takes enough neurons and the right training;
3. **Can it be trained?** Vanishing and exploding gradients once stopped deep networks in their tracks; ReLU and good initialization (and later residual connections) cleared the way.

In the next chapter, we'll see how researchers designed network architectures specifically for images and language: from convolutional networks all the way to the Transformer.

---

## Labs

```bash
pip install numpy
python autograd.py            # a hand-written autodiff engine + gradient check
python mlp_moons.py           # train a network with your own engine (pure Python, about half a minute)
python vanishing_gradient.py  # vanishing and exploding gradients in a 30-layer network
```

| Lab | What you should see | Think about it (also at the end of each script) |
|---|---|---|
| `autograd.py` | The gradient at every node of the computational graph; differences from the numerical gradient on the order of 10⁻¹⁰ | Why does `_backward` use `+=`? |
| `mlp_moons.py` | With no hidden layer, only about 90%; the 2→10→10→1 ReLU network reaches 100% in 80 steps, and the boundary bends into an S shape | Make the initialization smaller: how far does accuracy drop? |
| `vanishing_gradient.py` | With sigmoid, only 10⁻²⁰ of the signal reaches layer 1; with ReLU + He initialization it barely decays; with too large an initialization it is amplified more than ten thousand times | Why were early networks all shallow? |

---

## Concepts

### 1. Computational graphs

Any computation can be broken down into small steps: add, multiply, tanh... Draw them as a graph, with each node a value
and each edge saying "who was computed from whom," and you have a **computational graph**. In Lab 1, a `Value` object's `_prev` holds its incoming edges.
The first thing a framework does is quietly record this graph while you write the forward computation.

📖 First appears in: [Prologue](#prologue--1970-copenhagen-an-afternoon-in-the-park)　🧪 `autograd.py`

### 2. The chain rule

If `L` depends on `d`, and `d` depends on `e`, then `dL/de = dL/dd × dd/de`.
Backpropagation is just the chain rule applied over and over on the computational graph: each node only needs to know "its own local derivative,"
multiply that by the gradient arriving from the nodes after it, and pass the result on to the nodes before it. The local derivative of each operation is simple:
- Addition: pass it unchanged to both sides;
- Multiplication: multiply by the other side's value;
- tanh: multiply by `1 - tanh²`.

### 3. Forward mode and reverse mode

Automatic differentiation can run in two directions:
- **Forward mode**: start from one input and compute the derivatives of all the results with respect to **that one input**. With n inputs, you need n passes.
- **Reverse mode**: start from one result, walk backward once, and get the derivatives of **that one result** with respect to all the inputs.

Here is the situation in a neural network: millions to hundreds of billions of inputs (parameters), and only 1 result (the loss). So reverse mode wins hands down. That is backpropagation.

📖 First appears in: [Prologue](#prologue--1970-copenhagen-an-afternoon-in-the-park)

### 4. Gradient checking

A numerical gradient takes the derivative the dumbest way possible: nudge one parameter a tiny bit (say, 10⁻⁶) and see how much the loss changes.
It is slow (every parameter needs two separate evaluations), but it almost never goes wrong, so it is used specifically to **check whether your autodiff is correct**.
In Lab 1, the two methods differ on the order of 10⁻¹⁰, which shows the engine is right.

### 5. The universal approximation theorem

With enough neurons in the hidden layer, a network with one hidden layer can approximate any continuous function to any accuracy.
It describes a neural network's **expressive power**, but it doesn't say how many neurons you need, or whether a training algorithm can find those weights.
In practice, "deeper" is often far more efficient than "wider": a multi-layer network can combine features layer by layer into more and more complex ones.

📖 First appears in: [Act 1](#act-1--1989-is-one-layer-enough)　🧪 `mlp_moons.py`

### 6. Activation functions

| Function | Shape | Strengths | Weaknesses |
|---|---|---|---|
| sigmoid | S-shape from 0 to 1 | Can be read as a probability | Its derivative is at most 0.25; it saturates at both ends, where the gradient is almost 0 |
| tanh | S-shape from -1 to 1 | Centered on 0 | Also saturates at both ends |
| ReLU | `max(0, x)` | Derivative is always 1 on the positive side; extremely fast to compute | Derivative is 0 on the negative side, so neurons can "die" |

Without an activation function (or with only linear ones), a multi-layer network is no better than a single layer. The activation function supplies the **nonlinearity** that lets the network draw curved boundaries.
Today's large models mostly use smooth versions of ReLU, such as GELU and SwiGLU.

📖 First appears in: [Act 3](#act-3--20102015-a-new-activation-function-and-the-right-starting-weights)

### 7. Vanishing and exploding gradients

During backpropagation, every time the gradient passes through a layer, it is multiplied by "this layer's weights" and "the derivative of the activation function."
- If these factors are mostly **smaller than 1** → multiplied dozens of times, they head toward 0 → **vanishing gradients**, and the early layers can't learn;
- If they are mostly **larger than 1** → multiplied dozens of times, they head toward infinity → **exploding gradients**, and training simply diverges.

The fixes include ReLU-style activation functions, good initialization, normalization layers, residual connections (Chapter 4), and the LSTM, designed specifically for sequences (Chapter 4).
For exploding gradients there is also a quick and blunt fix: **gradient clipping**, which just shrinks the gradient back under a cap when it gets too big.

📖 First appears in: [Act 2](#act-2--1991-munich-the-deeper-the-harder-to-learn)　🧪 `vanishing_gradient.py`

### 8. Initialization

Before training starts, the weights can't all be 0: then every neuron would be identical and could never learn to be different. So we initialize them randomly.
But the **scale** of the random numbers is crucial:
- Too small → the signal gets weaker layer by layer → vanishing gradients;
- Too large → the signal gets stronger layer by layer → exploding gradients.

The usual rule is to scale by "the number of inputs, n":
- **Xavier initialization** (standard deviation about √(1/n)): suits tanh and sigmoid;
- **He initialization** (standard deviation √(2/n)): suits ReLU. ReLU zeroes out half of the signal, so the scale is a bit bigger to make up for it.

📖 First appears in: [Act 3](#act-3--20102015-a-new-activation-function-and-the-right-starting-weights)　🧪 `mlp_moons.py`, `vanishing_gradient.py`

### 9. Overfitting, regularization and Dropout

**Overfitting** means learning the training data too well and then doing worse on new data (Chapter 1, concept 6). The methods for fighting it are collectively called **regularization**:
- **L2 regularization (weight decay)**: add "the sum of the squared weights" to the loss, which nudges the weights to stay small and the model to stay simple;
- **Dropout**: at every training step, randomly switch off some of the neurons (say, half), forcing the network not to rely on any particular "clique";
  at test time, switch them all back on. You can think of it as training many sub-networks at once and then averaging them;
- **Early stopping**: stop training once performance on the validation set starts getting worse;
- **More data**: the most fundamental fix of all.

📖 First appears in: [Act 4](#act-4--2012-a-lesson-from-the-bank-tellers)

### 10. Deep learning frameworks

Frameworks like PyTorch, TensorFlow and JAX do the same core job as Lab 1's Value class: record the computational graph and backpropagate automatically.
The differences:
- They work on **tensors** (multi-dimensional arrays), not individual scalars. One matrix multiplication can replace thousands upon thousands of scalar operations;
- The computation can run in parallel on a **GPU**;
- They come with ready-made layers, optimizers (like Adam from Chapter 2) and data-loading tools.

In Lab 2, training 151 parameters with pure-Python scalars takes half a minute. PyTorch on a GPU does the same thing in the blink of an eye. That is the power of vectorization and hardware acceleration.

📖 First appears in: [Act 5](#act-5--20072017-no-more-derivatives-by-hand), [Act 6](#act-6--20202022-the-whole-idea-in-a-hundred-lines-of-code)

---

## References

- Wikipedia, [Seppo Linnainmaa](https://en.wikipedia.org/wiki/Seppo_Linnainmaa); A. Griewank, [Who Invented the Reverse Mode of Differentiation?](https://papers.baulab.info/papers/also/Griewank-2012.pdf), *Documenta Mathematica*, 2012
- G. Cybenko, [Approximation by Superpositions of a Sigmoidal Function](https://papers.baulab.info/papers/Cybenko-1989.pdf), *Mathematics of Control, Signals and Systems*, 1989
- K. Hornik, M. Stinchcombe, H. White, *Multilayer Feedforward Networks are Universal Approximators*, *Neural Networks* 2(5), 1989
- S. Hochreiter, [Untersuchungen zu dynamischen neuronalen Netzen](https://www.bioinf.jku.at/publications/older/3804.pdf), Diplomarbeit, TU München, 1991; Wikipedia, [Sepp Hochreiter](https://en.wikipedia.org/wiki/Sepp_Hochreiter)
- X. Glorot & Y. Bengio, [Understanding the Difficulty of Training Deep Feedforward Neural Networks](https://proceedings.mlr.press/v9/glorot10a), AISTATS 2010
- V. Nair & G. Hinton, *Rectified Linear Units Improve Restricted Boltzmann Machines*, ICML 2010
- X. Glorot, A. Bordes, Y. Bengio, [Deep Sparse Rectifier Neural Networks](https://proceedings.mlr.press/v15/glorot11a.html), AISTATS 2011
- A. Krizhevsky, I. Sutskever, G. Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, NIPS 2012
- K. He, X. Zhang, S. Ren, J. Sun, [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852), ICCV 2015
- G. Hinton et al., [Improving Neural Networks by Preventing Co-adaptation of Feature Detectors](https://arxiv.org/abs/1207.0580), 2012
- The Import AI newsletter's retelling of Hinton's bank-teller story: [jack-clark.net, 2016-08-15](https://jack-clark.net/2016/08/15/)
- Wikipedia, [Theano](https://en.wikipedia.org/wiki/Theano_(software)); Google, [TensorFlow open-source announcement](https://research.google/blog/tensorflow-googles-latest-machine-learning-system-open-sourced-for-everyone/), 2015
- A. Karpathy, [micrograd](https://github.com/karpathy/micrograd); video: *The spelled-out intro to neural networks and backpropagation: building micrograd*, 2022

---

⬅️ [Chapter 2 · Linear Models and Gradient Descent](../02-linear-models/) | [Contents](../README.md) | ➡️ [Chapter 4 · From CNNs to Transformers](../04-cnn-to-transformer/)
