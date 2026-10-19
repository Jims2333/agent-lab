# Chapter 2 — Linear Models and Gradient Descent: Starting with a Lost Planet

> Chapter 1 was about where neural networks came from. But the "engine" neural networks use to learn,
> namely **define an error, then drive that error as low as it will go**, is far older than neural networks.
> Its story begins on New Year's Day 1801, with a lost planet.

**How to read this chapter:** As in Chapter 1, read the stories first, then run the labs, and save the [Concepts](#concepts) for last.

| When | What happened | Lab / concept |
|---|---|---|
| 1801 | Ceres is discovered and then lost; 24-year-old Gauss calculates where it is | 🧪 `ceres_1801.py` |
| 1805 | Legendre publishes the method of least squares | Least squares |
| 1809 | Gauss says he has been using it since 1795, and the priority dispute begins | — |
| 1847 | Cauchy proposes gradient descent | 🧪 `gradient_descent_1847.py` |
| 1875–1886 | Galton's sweet peas and height data; "regression" gets its name | 🧪 `galton_1886.py` |
| 1951 | Robbins and Monro propose stochastic approximation | 🧪 `lms_1960.py` |
| 1958 | Cox proposes logistic regression | From regression to classification |
| 1959–1960 | Widrow and Hoff's ADALINE and the LMS rule | 🧪 `lms_1960.py` |
| 2014 | The Adam optimizer | Optimizers |

---

## Prologue — New Year's Day 1801, Palermo: A "Star" That Moved

On January 1, 1801, the first night of the 19th century, the Italian astronomer **Giuseppe Piazzi**,
at the observatory in Palermo, Sicily, noticed a faint point of light. When he looked again the next night, it had moved a little.
It wasn't a star. It was a body orbiting the Sun between Mars and Jupiter, later named **Ceres**.

Piazzi tracked it for 41 days and recorded about two dozen observations. Then it drew close to the Sun and vanished into the Sun's glare.
When it came back out from behind the Sun, where in that vast sky should anyone look? A few dozen observations covered only a short arc of its orbit,
and at the time there was no reliable way to compute a full orbit from so little data. Europe's astronomers searched for months and found nothing.

That's when a 24-year-old German, **Carl Friedrich Gauss**, took on the problem.
He spent about three months on the calculations and worked out where Ceres should appear in December.

On December 31, 1801, the astronomer **Franz Xaver von Zach**, searching from Gotha where Gauss had predicted,
found Ceres again less than half a degree from the predicted position. The next day, Heinrich Olbers saw it too, in Bremen.
Discovered on New Year's Day, found again on New Year's Eve. Overnight, Gauss was famous all over Europe.

One of his secret weapons was a method for handling "observations with errors": what came to be called **the method of least squares**.

> 🧪 **Try it**: run [`ceres_1801.py`](./ceres_1801.py). Use 24 noisy observations to predict the position a year later,
> and compare which is more accurate: "just two points" or "least squares."

---

## Act 1 — 1805–1820: Who Really Invented Least Squares?

In 1805, the French mathematician **Adrien-Marie Legendre** published a book on computing the orbits of comets.
Its appendix (dated March 6, 1805) clearly laid out a method:
when there are more equations than unknowns, and every observation has some error, choose the solution that makes **the sum of the squares of all the errors as small as possible**.
He gave the method a name: **the method of least squares** (*méthode des moindres quarrés*).

Four years later, in 1809, Gauss published his great work on celestial mechanics, *Theoria Motus*. It also covered least squares
and mentioned Legendre, but it also said that **he had been using this method since 1795**.

Legendre was furious. On May 31, 1809, he wrote to Gauss, saying that priority should be settled by **publication**.
In 1820, he publicly rebutted Gauss again in a supplement to one of his works. Gauss later tried to prove that he really had been first,
but by his own account, his old calculation notes had been lost, and his colleagues either couldn't remember or didn't want to be drawn into the dispute.

Today, textbooks usually credit both: Legendre was the first to publish and name it, while Gauss connected it to a probability theory of errors (what became the "normal distribution")
and explained **why** you should use squares.

> 💡 Look ahead: Why "squares," of all things? → [Residuals, least squares and mean squared error](#3-residuals-least-squares-and-mean-squared-error)

---

## Act 2 — 1847: Cauchy's "Downhill Method"

For simple models like a straight line, least squares has a ready-made formula. But real orbit problems are **nonlinear**:
a celestial body's orbit has 6 unknowns, the equations are full of trigonometric functions, and there's simply no formula that solves them.

In 1847, the French mathematician **Augustin-Louis Cauchy** published a short note, just three pages long, in the *Comptes Rendus* of the French Academy of Sciences,
titled *Méthode générale pour la résolution des systèmes d'équations simultanées* ("General method for solving systems of simultaneous equations"). His starting point was exactly the orbit calculations of astronomy.
His idea was astonishingly simple:
1. Combine "the errors of all the equations" into one total error, which is a function of the unknowns;
2. From where you are now, see which direction makes the total error drop fastest;
3. Take a small step in that direction, then repeat.

This is **gradient descent**. More than 170 years later, models like ChatGPT and Claude are still trained with its descendants.

> 🧪 **Try it**: run [`gradient_descent_1847.py`](./gradient_descent_1847.py). Walk downhill on Galton's real data,
> and see what happens when the learning rate is too small, just right, or too big, and why "centering the data first" can make it hundreds of times faster.

> 💡 Look ahead: → [The gradient](#5-the-gradient), [Gradient descent and the learning rate](#6-gradient-descent-and-the-learning-rate), [Feature scaling](#7-feature-scaling)

---

## Act 3 — 1875–1886: Sweet Peas, Heights, and the Odd Name "Regression"

**Francis Galton** was Darwin's cousin, a Victorian scholar who wanted to measure just about everything.
He wanted to know: how are parents' traits passed on to their children?

In 1875, he ran a very creative experiment. He sorted sweet pea seeds by weight into 7 grades, from very large to very small,
put 10 seeds of each grade into packets, and mailed them to 7 friends, asking them to plant the seeds and send back the next generation of seeds they harvested.
What he found: the offspring of large seeds were also on the large side, but **not as large as their parents**; the offspring of small seeds were also on the small side, but **not as small as their parents**.
The offspring had all "gone back" some of the way toward the average.

In 1886, he confirmed the same phenomenon with human height. He collected the heights of 928 adult children from 205 families
and published the paper *Regression towards Mediocrity in Hereditary Stature*.
When parents were somewhat taller than average, their children were, on average, taller by only about two-thirds of that gap.

He called this phenomenon **regression** (literally, "going back"). Later, the whole set of methods for "describing the relationship between two quantities with a straight line"
took over the name, and we still use it today, even though most problems have nothing to do with "going back."

(Galton's history also has a dark side: he coined the word "eugenics," an idea that later caused serious harm.)

> 🧪 **Try it**: run [`galton_1886.py`](./galton_1886.py). It uses **Galton's real data from back then**.
> You'll compute that "about two-thirds" yourself, and you'll discover that regression to the mean is a purely statistical phenomenon.

> 💡 Look ahead: → [Regression to the mean](#10-regression-to-the-mean)

---

## Act 4 — 1951: Reaching the Finish Line, Noise and All

Every step of gradient descent requires computing the "exact" downhill direction. But what if every measurement is noisy, and all you can get is a **rough direction**?

In 1951, the statisticians **Herbert Robbins** and **Sutton Monro**
published *A Stochastic Approximation Method*. They proved that even if the direction of every step is noisy, as long as the step size **shrinks gradually** in the right way,
you'll still reach the right answer in the end.

This very "theoretical"-looking paper later became the theoretical foundation of **stochastic gradient descent (SGD)**.

> 💡 Look ahead: → [Batch, stochastic and mini-batch](#8-batch-stochastic-and-mini-batch-gradient-descent), [Learning rate decay](#9-learning-rate-decay)

---

## Act 5 — 1959–1960: Stanford's ADALINE

At the same time Rosenblatt was building the perceptron at Cornell (see Chapter 1), a young Stanford professor, **Bernard Widrow**,
and his first PhD student, **Marcian "Ted" Hoff**, came up with a different learning rule in the fall of 1959.

The perceptron adjusts only when it guesses wrong, and each adjustment is a fixed size. Widrow and Hoff's rule:
**for every example that comes in, adjust the weights in proportion to "how wrong" it was**. Very wrong, big adjustment; slightly wrong, small adjustment.
This rule is called **LMS (least mean squares)**, also known as the Widrow–Hoff rule or the delta rule. At heart, it's just **gradient descent that looks at one example at a time**.
They used it to build a device called **ADALINE** (adaptive linear neuron).

LMS later became a huge success in signal processing, widely used in adaptive filtering such as echo cancellation on telephone lines.

And that PhD student, Ted Hoff, later became Intel's employee number 12
and led the design of the **Intel 4004**, released in November 1971: **the world's first commercial microprocessor**.

> 🧪 **Try it**: run [`lms_1960.py`](./lms_1960.py). Compare "looking at all the data every step" with "looking at one example at a time":
> after seeing the same amount of data, which one is closer to the answer?

> 💡 Look ahead: → [LMS and the perceptron](#14-lms-and-the-perceptron)

---

## Act 6 — 1958–2014: From Regression to Classification, from SGD to Adam

Linear regression predicts a **number** (a height, a position). But many problems need a **yes or no**: Is this email spam? Is the mark on this card on the left or the right?

In 1958, the British statistician **David Cox** systematically laid out **logistic regression**:
first compute the linear model's output, then use an S-shaped curve (the logistic function, which is the sigmoid from Chapter 1's Lab 3) to squash it into the range 0 to 1, and treat that as the probability of "yes."
It is still one of the most widely used classification methods, and it's also the standard form of a single neuron in a neural network.

Gradient descent itself kept evolving too. In December 2014, **Diederik Kingma** and **Jimmy Ba** published the **Adam** optimizer:
it automatically adjusts the step size for each parameter and also takes the directions of the previous few steps into account, so noise is less likely to throw it off course.
Adam quickly became one of the most widely used optimizers in deep learning, and today most large language models are trained with variants from the Adam family.

From Cauchy's three-page note in 1847 to training hundreds of billions of parameters today, the core step has never changed: **compute the gradient, take a small step downhill**.

> 💡 Look ahead: → [From regression to classification](#12-from-regression-to-classification-logistic-regression), [Optimizers](#13-optimizers-momentum-and-adam)

---

## Epilogue — Two Ways to Find the Answer

This chapter showed two ways of "finding the best parameters":

1. **Solve the formula directly** (Legendre, Gauss): for linear models, least squares has an exact closed-form solution that gets there in one step.
2. **Walk there step by step** (Cauchy, Robbins–Monro, Widrow–Hoff): no formula needed, just the ability to compute gradients.

Why does deep learning today use only the second? Because neural networks are nonlinear and simply have no closed-form solution;
and when there are hundreds of billions of parameters and trillions of tokens of data, you couldn't compute it even if there were a formula.
Only gradient descent, especially stochastic gradient descent, can handle that kind of scale.

---

## Labs

```bash
pip install numpy
python ceres_1801.py             # 1801: finding Ceres again with least squares
python gradient_descent_1847.py  # 1847: gradient descent, learning rate, feature scaling
python galton_1886.py            # 1886: Galton's real data and "regression to the mean"
python lms_1960.py               # 1951–1960: stochastic gradient descent and LMS
```

| Lab | What you should see | Think about it (also at the end of each script) |
|---|---|---|
| `ceres_1801.py` | Replayed 2,000 times: least squares is within half a degree 90% of the time, the two-point method only about 58% | Why the sum of squares, rather than the sum of absolute values? |
| `gradient_descent_1847.py` | On the raw data, 5,000 steps still don't get there; after centering, 20 steps do; make the learning rate just a little bigger and it jumps farther and farther away | What sets the upper limit on the learning rate? |
| `galton_1886.py` | Slope 0.65, close to Galton's 2/3; in standard scores, the slope is 0.46 in both directions | Is the "cover jinx" real? |
| `lms_1960.py` | After only 1,000 examples, SGD is already very close to the answer, while batch gradient descent hasn't taken a single step | Why is SGD the only option when the data "streams" in? |

About the data: [`data/galton_1886.csv`](./data/galton_1886.csv) is the original data from Galton's 1886 paper (928 rows),
transcribed from the R package HistData, maintained by the statistician Michael Friendly.

---

## Concepts

### 1. Regression and Linear Models

**In one sentence**: regression means predicting a number from inputs; a linear model means "multiply each input by a weight, then add them up."

```
prediction = w1·x1 + w2·x2 + … + b
```
- **x** (features): the input information, like the parents' heights;
- **w** (weights): how much each feature matters, like 0.65;
- **b** (bias/intercept): the baseline value when all the features are 0.

Chapter 1's perceptron is a linear model too; it just adds one final step: "greater than 0, output 1; otherwise output 0."

📖 First appears in: [Act 3](#act-3--18751886-sweet-peas-heights-and-the-odd-name-regression)

### 2. What Exactly Is "Linear" About a Linear Model?

The "linear" in linear model means **linear in the parameters (w, b)**, not in the inputs.
So `y = w1·x + w2·x² + b` is a linear model too: just treat x² as a new feature.
This lets linear models fit curves while keeping least squares' closed-form solution.
The power of neural networks is that they can **learn on their own** which features to use (the hidden layer in Chapter 1's Lab 3), so no one has to design x² by hand.

### 3. Residuals, Least Squares and Mean Squared Error

- **Residual**: actual value − predicted value, that is, how far off each point is.
- **Least squares**: choose the parameters that make the **sum of squared residuals** as small as possible.
- **Mean squared error (MSE)**: the average of the squared residuals, the most common loss function for regression.

Why squares?
1. Positive and negative errors don't cancel each other out;
2. Big errors get an extra-heavy penalty, so the model works hard to avoid wild mistakes;
3. The square function is differentiable everywhere, so you can take derivatives and solve for a formula directly (the normal equations), and it also makes gradient descent easy;
4. Gauss proved that if the observation errors follow a normal distribution, least squares gives the "most likely" answer.

The downside is that it's very sensitive to **outliers**: one wildly off data point can drag the whole line toward it. In that case you can switch to other loss functions, such as absolute error.

📖 First appears in: [Prologue](#prologue--new-years-day-1801-palermo-a-star-that-moved), [Act 1](#act-1--18051820-who-really-invented-least-squares)　🧪 `ceres_1801.py`

### 4. The Normal Equations (Closed-Form Solution)

For a linear model, take the derivative of the mean squared error with respect to each parameter and set it to 0, and you get a set of equations called the **normal equations**:
```
(XᵀX) w = Xᵀy
```
Solving them gives the least squares answer. The 2×2 system solved by hand in Lab 1 is their simplest form.
The downsides: with d features, the work needed to solve them grows roughly in proportion to d³, so with many features it becomes too much to compute; and nonlinear models have no such formula at all.

### 5. The Gradient

The **gradient** is a vector that points in "the direction in which the function rises fastest," and its length tells you how steep it is.
Each of its components is the partial derivative of the loss with respect to one parameter: how much the loss changes if that parameter goes up a tiny bit.
Walk in **the opposite direction of the gradient**, and the loss drops fastest.

### 6. Gradient Descent and the Learning Rate

```
parameters ← parameters − learning rate × gradient
```
- **Learning rate too small**: every step is steady, but it takes forever;
- **Learning rate just right**: it reaches the bottom of the valley quickly;
- **Learning rate too big**: one step overshoots the bottom of the valley and lands higher up on the other side, it jumps farther and farther, and finally it "blows up."

The upper limit on the learning rate is set by how steep the loss function is in its steepest direction: the steeper it is, the smaller the steps have to be.
In Lab 2, going from "a bit big" to "too big" changes the learning rate only a tiny bit, yet one converges and the other diverges. That's because it crosses this limit.

📖 First appears in: [Act 2](#act-2--1847-cauchys-downhill-method)　🧪 `gradient_descent_1847.py`

### 7. Feature Scaling

If one direction is very steep and another is very flat, the loss function looks like a **long, narrow canyon**.
The learning rate has to be set for the steep direction (otherwise it diverges), and as a result it can barely move along the flat direction.

**Centering** (subtracting the mean) and **standardizing** (then also dividing by the standard deviation) turn the canyon into a nice round bowl, and gradient descent gets much faster.
In Lab 2, the raw data's canyon is millions of times steeper in one direction than in the other; after centering, the difference is only 3 times.
That's also why the first step of almost every machine learning pipeline is to scale the data to similar ranges.

### 8. Batch, Stochastic and Mini-Batch Gradient Descent

| | How much data per step | What it's like |
|---|---|---|
| **Batch gradient descent** | All of it | Accurate direction, but every step is expensive |
| **Stochastic gradient descent (SGD)** | 1 example | Every step is cheap; the direction is noisy and it jitters |
| **Mini-batch gradient descent** | Tens to thousands | A compromise between the two; today's standard practice, and the best fit for GPU parallelism |

**Epoch**: one complete pass through all the training data is called an epoch.

📖 First appears in: [Act 4](#act-4--1951-reaching-the-finish-line-noise-and-all), [Act 5](#act-5--19591960-stanfords-adaline)　🧪 `lms_1960.py`

### 9. Learning Rate Decay

SGD's direction is noisy, so near the finish line it keeps jittering around.
If you make the learning rate **shrink gradually**, the jitter shrinks too, and in the end it settles firmly on the answer.
What Robbins and Monro proved in 1951 was exactly how the step size should shrink to guarantee convergence.
When large models are trained today, the learning rate is usually "warmed up" first (slowly raised from a small value), then gradually decayed.

### 10. Regression to the Mean

When two quantities aren't perfectly correlated and one of them is extreme, the other tends to be **less extreme**, closer to the average.
This isn't some mysterious force; it's a statistical regularity: an extreme value usually includes a share of "luck," and luck isn't inherited and doesn't last.

Many "effects" in everyday life are really illusions caused by regression to the mean: the improvement after an especially bad exam, the slump after an especially good performance (the "cover jinx").
To judge whether a method really works, you need a control group; a before-and-after comparison isn't enough.

📖 First appears in: [Act 3](#act-3--18751886-sweet-peas-heights-and-the-odd-name-regression)　🧪 `galton_1886.py`

### 11. The Danger of Extrapolation

Predicting **inside** the range your data covers is called **interpolation**, and it's fairly reliable; predicting **outside** that range is called **extrapolation**, and it's dangerous.
Lab 1 uses 41 days of data to predict day 364. It works because the form of model we used (motion at a constant speed) happens to be right.
Gauss could find Ceres again because he, too, used the right physical model (Kepler's elliptical orbits).
If the form of the model itself is wrong, then no matter how well it fits the data, extrapolation can go wildly wrong.

### 12. From Regression to Classification: Logistic Regression

Logistic regression = linear model + sigmoid function:
```
probability = sigmoid(w·x + b) = 1 / (1 + e^-(w·x + b))
```
The sigmoid squashes any real number into the range 0–1, which is treated as the probability of "yes." Training usually doesn't use mean squared error,
but **cross-entropy** loss instead: if the true answer is "yes," the lower the probability the model gives to "yes," the heavier the penalty.

A logistic regression is really just "a neuron that outputs a probability." Connect lots of these neurons in layers, and you get the neural networks of Chapter 1.

📖 First appears in: [Act 6](#act-6--19582014-from-regression-to-classification-from-sgd-to-adam)

### 13. Optimizers: Momentum and Adam

Plain gradient descent looks only at the gradient of "the current step." There are two main kinds of improvements:
- **Momentum**: like a ball rolling downhill, it keeps some of its earlier speed. In a canyon it's less likely to bang back and forth, and on flat ground it can keep rolling forward.
- **Adaptive learning rates**: give each parameter its own step size. Parameters that often swing wildly take smaller steps; parameters that rarely get updated take bigger ones.

**Adam** combines the two and works with hardly any tuning, so it became the default choice in deep learning.

📖 First appears in: [Act 6](#act-6--19582014-from-regression-to-classification-from-sgd-to-adam)

### 14. LMS and the Perceptron

| | Perceptron rule (1958) | LMS rule (1960) |
|---|---|---|
| When it updates | Only on a wrong guess | On every example |
| How much it updates | A fixed amount: `w += y·x` | In proportion to the error: `w += η·(y − ŷ)·x` |
| What it really is | A heuristic rule | Stochastic gradient descent on the mean squared error |
| When no straight line can split the data perfectly | It never stops (Chapter 1's XOR) | It settles where the error is smallest |

LMS shows us that **a learning rule can be derived from "minimizing a loss function,"** instead of being designed by intuition.
Backpropagation in 1986 extended this idea to multilayer networks.

📖 First appears in: [Act 5](#act-5--19591960-stanfords-adaline)

---

## References

- [Gauss Predicts the Orbit of Ceres](https://thatsmaths.com/2021/06/24/gauss-predicts-the-orbit-of-ceres/) (ThatsMaths, 2021)
- Wikipedia, [Least squares](https://en.wikipedia.org/wiki/Least_squares) (the priority dispute between Legendre and Gauss)
- A.-L. Cauchy, [Méthode générale pour la résolution des systèmes d'équations simultanées](https://gtfp.cs.rhul.ac.uk/pulskamp/Cauchy/Orbits/1847%20CR%20536%28383%29.pdf), *Comptes Rendus* 25, 1847
- F. Galton, *Regression towards Mediocrity in Hereditary Stature*, Journal of the Anthropological Institute 15, 1886
- J. M. Stanton, [Galton, Pearson, and the Peas: A Brief History of Linear Regression for Statistics Instructors](https://jse.amstat.org/v9n3/stanton.html), *Journal of Statistics Education*, 2001 (the sweet pea experiment)
- M. Friendly, *HistData: Data Sets from the History of Statistics and Data Visualization* (R package; source of the Galton data)
- H. Robbins & S. Monro, [A Stochastic Approximation Method](https://projecteuclid.org/euclid.aoms/1177729391), *Annals of Mathematical Statistics*, 1951
- B. Widrow & M. E. Hoff, *Adaptive Switching Circuits*, IRE WESCON Convention Record, 1960
- Wikipedia, [ADALINE](https://en.wikipedia.org/wiki/ADALINE), [Least mean squares filter](https://en.wikipedia.org/wiki/Least_mean_squares_filter)
- Computer History Museum, [Ted Hoff](https://computerhistory.org/profile/ted-hoff/) (Intel 4004)
- D. R. Cox, *The Regression Analysis of Binary Sequences*, Journal of the Royal Statistical Society B, 1958
- D. P. Kingma & J. Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014

---

⬅️ [Chapter 1 · Where Machine Learning Came From](../01-ml-origins/) | [Contents](../README.md) | ➡️ [Chapter 3 · Neural Networks and Autodiff](../03-neural-nets/)
