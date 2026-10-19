# Chapter 1 — Where Machine Learning Came From

> On July 8, 1958, *The New York Times* ran a story under the headline
> **"New Navy Device Learns By Doing"**.
>
> More than sixty years later, we start this repo the same way.

This chapter is about how machines learned, step by step, to "learn." It runs from the neuron equations a runaway teenager wrote down in 1943
to a team of AI agents splitting up the work of writing code in 2026. Along the way there are geniuses, gambles and winters, and one person who changed history with two gaming graphics cards in his bedroom.

**How to read this chapter:**

1. Read the stories first (Acts 1–12). Every name, date, place and quote has a source, listed in the References at the end.
   For anecdotes that rest on a single source, I say so: "according to X's recollection" or "as X reported."
2. Then run the three small labs and recreate three key moments, from 1958, 1969 and 1986, yourself.
3. Save the [Concepts](#concepts) for last. Every concept planted in the stories is explained in full there.

| When | What happened | Lab / concept |
|---|---|---|
| 1943 | McCulloch and Pitts describe the neuron with logic | Neuron |
| 1949 | Hebb proposes that "neurons that fire together grow stronger connections" | Learning rule |
| 1950 | Turing publishes *Computing Machinery and Intelligence* | — |
| 1956 | The Dartmouth conference gives "artificial intelligence" its name | — |
| 1958 | Rosenblatt's perceptron makes *The New York Times* | 🧪 `perceptron_1958.py` |
| 1959 | Samuel's checkers paper makes the term "machine learning" catch on | Defining machine learning |
| 1969 | Minsky and Papert publish *Perceptrons* | 🧪 `xor_1969.py` |
| 1973 | The Lighthill Report, the first AI winter | — |
| 1986 | Backpropagation appears in *Nature* | 🧪 `backprop_1986.py` |
| 1989 | LeCun uses a convolutional network to read handwritten ZIP codes | CNN |
| 1997 | Deep Blue defeats Kasparov | Search vs. learning |
| 2009 | ImageNet makes its debut | Data |
| 2012 | AlexNet wins, and DNNresearch is auctioned off | GPUs, deep learning |
| 2014 | The GAN is born in a bar | Generative models |
| 2016 | AlphaGo vs. Lee Sedol, Move 37 | Reinforcement learning |
| 2017 | *Attention Is All You Need* | Transformer |
| 2022 | ChatGPT launches | Large language models |
| 2023– | ReAct, AutoGPT, Stanford's Smallville, multi-agent systems, harnesses | Agents, harness, multi-agent |

---

## Prologue — Jul 1958, Washington: A Machine That "Learns By Doing"

On July 7, 1958, the US Office of Naval Research held a press conference at the US Weather Bureau in Washington.
The star of the show was a 29-year-old psychologist from the Cornell Aeronautical Laboratory, **Frank Rosenblatt**.
His prop was an IBM 704 computer that weighed 5 tons and filled a whole room.

The demo sounded simple: show the machine one punched card after another, some marked on the left, some on the right.
Nobody wrote rules telling it what "left" and "right" meant. After 50 cards, **it had learned on its own to tell left from right**.

The next day, the headline in *The New York Times* read "New Navy Device Learns By Doing; Psychologist Shows Embryo of Computer
Designed to Read and Grow Wiser."
The article also said the Navy expected it would one day "be able to walk, talk, see, write, reproduce itself and be conscious of its existence."
Rosenblatt himself called it the first machine capable of having an original idea. *The New Yorker* went even further:
in its view, this was the first serious rival to the human brain ever devised.

The machine was called the **perceptron**. To understand why it got people so excited, and why it later fell so hard,
we first have to go back 15 years, to Chicago.

> 💡 Look ahead: The machine "learned," but what exactly did it learn? → [Models and parameters](#2-models-parameters-training-inference)

---

## Act 1 — 1943, Chicago: The Runaway Boy and the Logic of Neurons

**Walter Pitts** was born in 1923 into a poor family in Detroit.
According to his friend Jerome Lettvin's recollections (retold in a long 2015 article in *Nautilus*),
at age 12 Pitts ducked into the public library to escape some street bullies, and there he found Russell and Whitehead's *Principia Mathematica*.
He stayed in the library for three days, read the nearly 2,000-page work from cover to cover, found several mistakes in it, and wrote to Russell about them.
Russell wrote back inviting him to come to Cambridge as a graduate student. But he was only 12.

At 15, Pitts heard that Russell was coming to lecture at the University of Chicago, so he ran away from home to Chicago.
He wasn't enrolled. He got by on odd jobs and snuck into classes, and that's where he met a medical student, Lettvin.
Lettvin introduced him to the neurophysiologist **Warren McCulloch**.
McCulloch had long wanted to explain the brain with logic. He was captivated by the boy's ideas, and simply invited him to live in his house in the suburbs.

In 1943, the two published a paper, *A Logical Calculus of the Ideas Immanent in Nervous Activity*.
Its core idea: **a neuron can be thought of as a switch**. It receives signals from other neurons.
If the signals add up to more than some threshold, it "fires"; otherwise it stays silent. Connect such switches into a network, and you can compute any logical proposition.

It was the first time anyone had written down an "artificial neuron" in mathematics. But it had one flaw: the connections in the network were designed by people. **It couldn't learn.**

The rest of Pitts's life was a tragedy. He went to MIT, worked with Norbert Wiener, the father of cybernetics, and was seen as a genius.
In 1952, Wiener abruptly cut off McCulloch and everyone connected with him, Pitts included. Pitts was devastated. He started drinking heavily and never got his PhD. According to *Nautilus*, he burned his doctoral dissertation and years of notes.
He died in May 1969, at 46. McCulloch died that September.

> 💡 Look ahead: Neuron = weighted sum + threshold. → [Neurons and the perceptron](#7-neurons-and-the-perceptron)

---

## Act 2 — 1949–1950: A Rule for Learning, a Machine That Thinks

So now there was a neuron. But how does the brain *learn* things?

In 1949, the Canadian psychologist **Donald Hebb** proposed in his book *The Organization of Behavior*
that if neuron A repeatedly takes part in making neuron B fire, the connection from A to B grows stronger.
Later generations boiled it down to a catchy rhyme: "**cells that fire together, wire together**."
It was the first time anyone had clearly proposed that learning means **changing the strength of connections**.

A year later, in 1950, **Alan Turing** published *Computing Machinery and Intelligence* in the philosophy journal *Mind*.
Its very first sentence: "I propose to consider the question, 'Can machines think?'"
He introduced the famous "imitation game," later known as the Turing test.
The last section of the paper is titled "Learning Machines," and in it Turing wrote that
instead of writing a program to simulate an adult's mind, **why not write one that simulates a child's, and then educate it?**

A psychologist said learning is adjusting connections; a mathematician said we might as well build a machine that learns.
It would take Rosenblatt to truly bring these two ideas together.

> 💡 Look ahead: Adjusting connection strengths = adjusting parameters. → [Models and parameters](#2-models-parameters-training-inference)

---

## Act 3 — 1956–1959: One Summer, and a Game of Checkers

In 1955, four young scholars co-signed a funding proposal:
**John McCarthy**, **Marvin Minsky**,
**Nathaniel Rochester**, and **Claude Shannon**, the father of information theory.
They proposed a two-month, ten-person workshop at Dartmouth College in the summer of 1956.
**The term "artificial intelligence" first appeared in this very proposal.**

The proposal was wildly optimistic: "We think that a significant advance can be made in one or more of these problems
if a carefully selected group of scientists work on it together for a summer." The problems included getting machines to use language, form abstract concepts, and improve themselves.
(Spoiler: one summer was nowhere near enough. Seventy years later, some of these problems are still unsolved.)

Around the same time, IBM engineer **Arthur Samuel** was working on something more down-to-earth: teaching a computer to play checkers.
His program remembered the positions it had played and whether they eventually led to a loss or a win. The more it played, the better it got.
On February 24, 1956, the program made its public debut on television. According to historical material compiled by the University of Alberta's checkers AI team,
IBM president Thomas Watson arranged for the program to be shown to shareholders and predicted it would push IBM's stock up 15 points. It did.

In 1959, Samuel published his paper *Some Studies in Machine Learning Using the Game of Checkers*.
**That's how the term "machine learning" caught on.**
In 1962, his program beat Robert Nealey, a self-described checkers master.

One sentence is often cited as Samuel's definition of machine learning: "Field of study that gives computers the ability to learn without being explicitly programmed."
But **you won't find that sentence** in his 1959 paper. It's a later paraphrase. What the paper actually says is:
"Programming computers to learn from experience should eventually eliminate the need for much of this detailed programming effort."

> 💡 Look ahead: Learning from experience, getting better the more it plays. → [What is machine learning](#1-what-is-machine-learning), [Three ways to learn](#3-three-ways-to-learn)

---

## Act 4 — 1958–1960: Rosenblatt and His "Electronic Brain"

Now back to the press conference from the Prologue.

Rosenblatt pieced the earlier ideas together: take the McCulloch–Pitts neuron (weighted sum + threshold),
add a Hebb-like rule (adjust connection strengths based on experience), and you get **a machine that learns**.
His learning rule fits in one sentence: **if it guesses right, do nothing; if it guesses wrong, adjust the weights**.

The 1958 demo actually *simulated* the perceptron on the IBM 704. Two years later, on June 23, 1960,
purpose-built hardware, the **Mark I Perceptron**, was unveiled to the public:
- Its "eye" was an array of 20×20 = 400 photocells, in effect a 400-pixel retina;
- Each connection's weight was stored in a potentiometer;
- During learning, **electric motors turned those potentiometers**. Every wrong guess set a motor turning.

Today the Mark I Perceptron is on display at the National Museum of American History (part of the Smithsonian Institution).

Rosenblatt was a man of many talents: he played the piano well and even helped build an observatory.
On July 11, 1971, he died in a boating accident on Chesapeake Bay. It was his 43rd birthday.

> 🧪 **Try it**: run [`perceptron_1958.py`](./perceptron_1958.py) and recreate the demo yourself:
> 20×20 cards, 50 trials, learning to tell left from right. Then show it 1,000 more cards and see what happens.

> 💡 Look ahead: Getting every training card right doesn't mean getting every new card right. → [Generalization and overfitting](#6-training-set-test-set-generalization-and-overfitting)

---

## Act 5 — 1969–1973: A Book, a Wall, a Winter

Marvin Minsky, one of the organizers of the Dartmouth conference, had known Rosenblatt for a long time:
both went to the Bronx High School of Science in New York, with Minsky one year behind Rosenblatt.
The two held opposite views on neural networks and debated them publicly at academic conferences, but in private they reportedly stayed on friendly terms.

In 1969, Minsky and the MIT mathematician **Seymour Papert** published *Perceptrons*.
The book proved with rigorous math that **a single-layer perceptron can do very little**.
The most famous example is "exclusive or" (XOR): output 1 when the two inputs are "different," 0 when they're "the same."
A rule this simple, and the perceptron can never learn it.

The book's impact went far beyond the math itself. Funding and enthusiasm for neural network research drained away, and many researchers moved on to other things.
Two years later, Rosenblatt died in an accident, and the field lost its most powerful champion.

Winter didn't fall on neural networks alone. In 1973, the British mathematician **James Lighthill**,
commissioned by the UK's Science Research Council, wrote a report assessing artificial intelligence research. His verdict was harsh:
"In no part of the field have the discoveries made so far produced the major impact that was then promised."
The British government went on to cut AI research funding at most of its universities. That June, the BBC also recorded a televised debate at the Royal Institution:
Lighthill on one side, and three AI researchers, including John McCarthy, on the other.

This period came to be known as **the first AI winter**.

> 🧪 **Try it**: run [`xor_1969.py`](./xor_1969.py). The same perceptron learns AND and OR quickly,
> but on XOR it never stops. You'll also see a little chart showing why no straight line can separate XOR.

> 💡 Look ahead: Can one straight line separate them? → [Linear separability and XOR](#8-linear-separability-and-xor)

---

## Act 6 — 1986: Sending the Error Backward

People had actually long known the answer to the problem: **add another layer**. Put a "hidden layer" between the input and the output, and XOR can be solved.
The real puzzle was: how do you adjust the hidden layer's weights? When the output is wrong, which hidden neuron is to blame?

In 1986, **David Rumelhart**, **Geoffrey Hinton**, and
**Ronald Williams** published a paper just 4 pages long in *Nature*:
*Learning representations by back-propagating errors*. The method is called **backpropagation**:
first compute the error at the output, then use the chain rule from calculus to pass the error **backward**, layer by layer,
working out how much "blame" each weight should take, and then nudge each one a little.

To be fair, they weren't the first to invent the math.
The Finnish student Seppo Linnainmaa described a related automatic differentiation method in his 1970 master's thesis,
and the American researcher Paul Werbos proposed using it to train neural networks in his 1974 PhD thesis.
But the 1986 paper showed clearly that **the hidden layer learns useful intermediate features on its own**. With that, neural networks came back to life.

Hinton's own story is a fascinating one. He is the **great-great-grandson of the logician George Boole**:
the "Boolean logic" every computer uses today is named after his great-great-grandfather.
In 1987, he left the US for the University of Toronto in Canada. One reason: most American AI research at the time was funded by the Department of Defense, and he didn't want his research used for war.
He and his students then spent decades in Canada sticking with unfashionable neural networks, until they changed the world.
In 2018, he shared the Turing Award with Yoshua Bengio and Yann LeCun; in 2024, he shared the Nobel Prize in Physics with John Hopfield.

> 🧪 **Try it**: run [`backprop_1986.py`](./backprop_1986.py). It writes backpropagation by hand with nothing but numpy.
> Watch the error come down bit by bit, then open the black box and see which intermediate concepts the hidden layer invented on its own.

> 💡 Look ahead: The error flows backward, and each weight takes a small step downhill. → [Loss functions and gradient descent](#4-loss-functions), [Backpropagation](#10-backpropagation)

---

## Act 7 — 1989–1997: The Network That Read Checks, and the Champion That Couldn't Learn

In 1988, a young Frenchman, **Yann LeCun**, joined Bell Labs in the US.
In 1989, he used backpropagation to train a special kind of neural network to recognize handwritten ZIP codes from the US Postal Service.
Its design was inspired by the visual cortex: instead of looking at the whole image at once, it slides small "windows" across the image,
looking for local patterns like edges and corners. This is the **convolutional neural network (CNN)**.

The technology was later used to read bank checks. According to the ACM's Turing Award announcement, a check-reading system LeCun helped develop was widely deployed by NCR and other companies,
and in the late 1990s and early 2000s, **it read more than 10% of all checks in the US**.

But in academia, neural networks soon fell out of favor again. In 1995, Corinna Cortes and Vladimir Vapnik published the **support vector machine (SVM)**.
It had elegant mathematical theory behind it, and at the data sizes of the time, its results were also more reliable. Throughout the 2000s, "statistical learning" was the mainstream.
As LeCun, Hinton and others later recalled, neural network papers in those years were often rejected by conferences. This is sometimes called neural networks' "second cold spell."

Around the same time, another "AI beats human" story shook the world: in May 1997, IBM's **Deep Blue** defeated world chess champion Garry Kasparov in New York.
But Deep Blue could hardly **learn** at all. It relied on custom chips that searched more than a hundred million positions per second, plus evaluation rules hand-written by chess experts.
It was strong, but its strength had been "written in" by people, not "learned" on its own.

> 💡 Look ahead: → [Convolutional neural networks](#11-convolutional-neural-networks-cnns)

---

## Act 8 — 2006–2009: A Dam Made of Data

In 2006, a young assistant professor, **Fei-Fei Li**, began turning over a question:
everyone was competing on algorithms, but what if **data** was the real bottleneck? Children come to understand the world because they've seen an enormous number of things.

She decided to build an image database unlike anything before it: following the structure of WordNet, a database of English words, collect upwards of a thousand photos for each of thousands of kinds of objects.
That was far too much work for students to do. In 2007, the team turned to Amazon's crowdsourcing platform, Mechanical Turk:
about 49,000 workers from 167 countries sifted through and labeled more than 160 million candidate images.

When **ImageNet** made its debut in 2009 at CVPR, the top computer vision conference, it had 3.2 million images,
but it was just a poster in a corner of the venue. Starting in 2010, the team ran the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) every year:
1,000 categories and about 1.2 million training images. For the first two years, the winners all used traditional methods, and progress was slow.

> 💡 Look ahead: More data, better generalization. Remember Act 2 of Lab 1? → [Generalization and overfitting](#6-training-set-test-set-generalization-and-overfitting)

---

## Act 9 — 2012: Two Graphics Cards in a Bedroom, an Auction in a Casino

In 2012, two of Hinton's students in Toronto, **Alex Krizhevsky** and
**Ilya Sutskever**, decided to enter the ImageNet challenge.
It was Sutskever who talked Krizhevsky into it, and Krizhevsky happened to be exceptionally good at squeezing every last drop of performance out of GPUs.

Their network came to be known as **AlexNet**. It was too big to fit in one graphics card's memory, so they split it across **two NVIDIA GTX 580 gaming cards**
and trained it for five or six days. According to the Computer History Museum, the machine sat **in Krizhevsky's bedroom at his parents' house**.

The results stunned the field: AlexNet's top-5 error rate was **15.3%**, while the runner-up's was **26.2%**.
In a field used to improving by a fraction of a percentage point a year, it had jumped ahead by almost 11 points in one go. The era of deep learning had begun.

A few months later came an even more dramatic scene. Hinton and the two students had set up a three-person company, DNNresearch.
In December 2012, while the NIPS conference was being held at Lake Tahoe, he ran an auction from his room at Harrah's, the casino hotel where the conference took place.
According to Cade Metz's *Genius Makers*, the bidders were Google, Microsoft, Baidu and DeepMind, then still a small company,
all bidding by email. The price climbed all the way to **$44 million**. At that point Hinton stopped the auction and sold the company to Google.
In his view, finding the right home for the research mattered more than getting the highest price.

> 💡 Look ahead: → [Deep learning and GPUs](#12-deep-learning-and-gpus)

---

## Act 10 — 2014–2016: An Idea in a Bar, Move 37 on the Board

**2014, Montreal.** PhD student **Ian Goodfellow** went to a bar called Les 3 Brasseurs
to celebrate a fellow student who had just finished a PhD. His friends were discussing a hard problem: how do you get a computer to "draw" realistic photos on its own?
Goodfellow had an idea: **pit two neural networks against each other**. One makes fakes, the other tells real from fake.
The forger keeps getting better at forging, and the detector keeps getting better at detecting.
According to *MIT Technology Review*, when he got home his girlfriend was already asleep. He coded into the early hours of the morning, then gave it a test, and **it worked on the very first run**.
This was the **generative adversarial network (GAN)**.

**March 2016, Seoul.** DeepMind's **AlphaGo** faced one of the world's top players, **Lee Sedol**, in a five-game match.
On move 37 of Game 2, AlphaGo played a "shoulder hit" that none of the professional players could make sense of.
Lee Sedol got up and left the playing room. When he came back to the board, he stared at the move for nearly 15 minutes before he played.
European Go champion Fan Hui, who was there, said: "It's not a human move. I've never seen a human play this move." Then he added: "So beautiful."
AlphaGo itself estimated that the odds of a human player making this move were just 1 in 10,000.

In Game 4, it was Lee Sedol's turn to play a "divine move": move 78. AlphaGo likewise estimated the odds of a human playing it at just 1 in 10,000,
and its judgment then fell into confusion, and it lost the game. It was humanity's only win of the match; the final score was 4–1.
In November 2019, Lee Sedol announced his retirement. AI, he said, is "an entity that cannot be defeated."

AlphaGo was different from Deep Blue: it first learned from records of human games, then kept getting stronger by **playing against itself**. Nobody could have "written" move 37.

> 💡 Look ahead: → [Generative models and GANs](#13-generative-models-and-gans), [Three ways to learn](#3-three-ways-to-learn) (reinforcement learning)

---

## Act 11 — 2017–2022: Eight People and One Paper

In spring 2017, eight Google researchers wrote a paper titled ***Attention Is All You Need***.
According to a 2024 story in *Wired*:
- The core idea, "self-attention," came first from **Jakob Uszkoreit**. **He also came up with the name "Transformer,"** because he liked the sound of the word;
  an early design document was even illustrated with pictures of characters from *Transformers*;
- The paper's title is a nod to the Beatles song "All You Need Is Love";
- The eight authors decided to break with the usual convention for ordering names. They put an asterisk after every name, with a footnote reading "Equal contribution. Listing order is random.";
- Later, **all eight left Google**, to start their own companies or join other AI companies.

The Transformer threw out the old approach of reading a sentence in order, one word at a time. Instead, every word in the sentence "attends" to every other word **at the same time**.
This not only works better, it's also a great fit for parallel computing on GPUs, so models could finally keep getting bigger.

You may have heard the rest: in 2018, OpenAI released GPT ("Generative Pre-trained Transformer") and Google released BERT;
in 2020, GPT-3 had 175 billion parameters.

On **November 30, 2022**, OpenAI released ChatGPT. According to UBS, citing data from Similarweb,
it reached about 100 million monthly active users two months later, in January 2023, and was called the fastest-growing consumer app in history.
The analysts wrote: "In 20 years following the internet space, we cannot recall a faster ramp in a consumer internet app."

> 💡 Look ahead: → [Attention and the Transformer](#14-attention-and-the-transformer), [Large language models](#15-large-language-models-llms)

---

## Act 12 — 2022–2026: From Talking to Doing

Now there were models that could chat. The next question: **could they go and do things on their own?**

- **October 2022**: Shunyu Yao and colleagues at Princeton and Google published **ReAct**: have the model alternate between "thinking" and "acting."
  Think one step, call a tool (say, look something up on Wikipedia), see the result, then think about the next step. Almost every agent today runs this loop.
- **March 30, 2023**: Toran Bruce Richards released **AutoGPT** on GitHub: give GPT-4 a goal and let it break the goal into tasks and carry them out itself.
  It collected 30,000 stars in 13 days and passed 100,000 a few weeks later, becoming one of the fastest-growing open-source projects on GitHub at the time.
  It often went in circles and failed to get things done, but it showed everyone what agents could be.
- **April 2023**: Joon Sung Park and colleagues at Stanford and Google published "Generative Agents."
  They placed 25 residents, each driven by a large language model, in a pixel-art town called **Smallville**.
  The researchers told the café owner, Isabella, just one thing: she wanted to throw a Valentine's Day party.
  After that, she sent out the invitations herself, word spread through the town, someone asked their crush to go with them, and that evening several residents actually showed up at the party.
  **Nobody wrote a "throw a party" script.** It "emerged" from the interactions of multiple agents.
- **November 2024**: Anthropic released **MCP (Model Context Protocol)**, a single standard for connecting models to all kinds of tools and data sources.
  **April 2025**: Google released the **A2A (Agent2Agent) protocol**, which lets agents from different vendors discover each other, talk and work together;
  that June, it was donated to the Linux Foundation.
- **June 2025**: Anthropic described how its own multi-agent research system works: a "lead agent" does the planning and sends out several "subagents" to search in parallel,
  then pulls the results together. In their internal evaluation, a system with Claude Opus 4 as the lead agent and Claude Sonnet 4 as subagents outperformed Claude Opus 4 working alone by 90.2%; the price was that the multi-agent system used about 15 times as many tokens as an ordinary chat.
- **February 2026**: OpenAI published "Harness engineering": over about five months, a small team
  built a product of about 1 million lines of code, **without a single line written by a human**. AI agents wrote all of it.
  The engineers' job became designing the agents' working environment: documentation, rules, tests, checking tools, and loops in which agents review each other's work.
  This surrounding system is called the **harness**.

From the motors turning potentiometers in 1958 to the agents reviewing each other's code in 2026, "learning by doing" has never stopped.
The rest of the chapters in this repo are about building these things yourself, by hand.

> 💡 Look ahead: → [Agents](#16-agents), [Harness](#17-harness), [Multi-agent collaboration](#18-multi-agent-collaboration)

---

## Epilogue — Three Threads That Run Through It All

Looking back over these seventy years, three threads keep coming up:

1. **A battle between two approaches**: one camp believed intelligence comes from **rules and logic** (symbolism: Deep Blue, early AI);
   the other believed it comes from **connections learned from data** (connectionism: the perceptron, backpropagation, deep learning).
   In 1969 the first camp had the upper hand; after 2012 the second won across the board. Today's agents combine the two: a model that was learned, calling tools that were written by hand.
2. **Data**: from 50 cards, to ImageNet's more than a million images, to the better part of the internet that large language models have read.
3. **Compute**: from the 5-ton IBM 704, to two gaming graphics cards in a bedroom, to today's GPU clusters.

Many of the ideas, like multilayer networks and backpropagation, had been around for a long time; they were just waiting for data and compute to catch up.

---

## Labs

```bash
pip install numpy
python perceptron_1958.py   # 1958: the perceptron learns to tell left from right
python xor_1969.py          # 1969: the perceptron can't learn XOR
python backprop_1986.py     # 1986: add a layer + backpropagation, and XOR is learned
```

| Lab | What you should see | Think about it (also at the end of each script) |
|---|---|---|
| `perceptron_1958.py` | After 50 cards it gets the whole training set right and about 94% of new cards; after 1,000 more cards, about 99% | Why are the weights negative on the left and positive on the right? |
| `xor_1969.py` | AND / OR are learned in a few epochs; XOR goes around in circles forever, and even the best of 100,000 random lines gets only 75% right | If the task were "exactly one of three inputs is 1," could a perceptron learn it? |
| `backprop_1986.py` | The error falls from 0.34 to 0.0002 and all four examples come out right; the hidden layer learns intermediate concepts like NOR and NAND on its own | If you remove the hidden layer's sigmoid, can it still learn? |

---

## Concepts

> The concepts below are listed in the order they appear in the stories, and each one notes the act where it first appears.

### 1. What Is Machine Learning

**In one sentence**: instead of writing the rules directly, let the program work out the rules from data (experience) on its own.

Traditional programming: rules written by people + data → answers.
Machine learning: data + answers → the program finds the rules itself.

In 1997, Tom Mitchell gave a more precise definition in his textbook:
> A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**,
> if its performance at tasks in T, as measured by P, improves with experience E.

Applied to Lab 1: T = telling whether a card is marked on the left or the right, P = accuracy, E = the cards it has seen. The more it sees, the higher its accuracy, so it is "learning."

📖 First appears in: [Act 3](#act-3--19561959-one-summer-and-a-game-of-checkers) (Samuel's checkers)

### 2. Models, Parameters, Training, Inference

- **Model**: a function with lots of "knobs." Data goes in, predictions come out.
- **Parameters** (weights): those knobs. On the Mark I Perceptron, the parameters **really were knobs**: 400 potentiometers turned by electric motors.
  Today's large models have hundreds of billions of parameters. The principle is the same; there are just far more knobs.
- **Training**: adjusting the knobs bit by bit, based on data, so the predictions get more and more accurate.
- **Inference**: once training is done, the knobs are fixed and the model is used on new data. When you chat with a chatbot, it is doing "inference."

📖 First appears in: [Prologue](#prologue--jul-1958-washington-a-machine-that-learns-by-doing), [Act 4](#act-4--19581960-rosenblatt-and-his-electronic-brain)

### 3. Three Ways to Learn

| Type | What the data looks like | Examples from the stories |
|---|---|---|
| **Supervised learning** | Every example comes with the "right answer" | The perceptron's cards (labeled left/right), ImageNet's images (labeled with categories) |
| **Unsupervised learning** | Just data, no answers; the model finds the structure itself | Automatically grouping similar news stories together |
| **Reinforcement learning** | No right answers, only a reward signal for "how well it did" | Samuel's checkers (win or lose), AlphaGo playing against itself |

Pre-training large language models also uses a variant: **self-supervised learning**. The answers are hidden in the data itself:
for example, cover up the next word of a sentence, have the model guess it, then uncover it to check. No human labeling is needed, so you can train on huge amounts of text.

📖 First appears in: [Act 3](#act-3--19561959-one-summer-and-a-game-of-checkers), [Act 10](#act-10--20142016-an-idea-in-a-bar-move-37-on-the-board)

### 4. Loss Functions

**In one sentence**: a single number that measures how badly the model is wrong right now.

Lab 3 uses **mean squared error**: for each example, take "prediction minus correct answer," square it, add them all up, and take the average.
A perfect guess gives a loss of 0, and the more wrong it is, the bigger the loss. The goal of training is to make this number smaller and smaller.

### 5. Gradient Descent and the Learning Rate

Imagine you're on a mountain in thick fog. You can't see where the valley is; you can only feel which direction is steepest under your feet.
The simplest way down: **take a small step in the steepest downhill direction**, and repeat many times.

- That "steepest direction" is the **gradient** (the derivative of the loss with respect to each parameter);
- How big each step is, is the **learning rate**. Too small, and the descent is slow; too big, and you might step right over the valley and bounce back and forth between the two sides.

In Lab 3, `W -= LR * d_W` is that step.

📖 First appears in: [Act 6](#act-6--1986-sending-the-error-backward)

### 6. Training Set, Test Set, Generalization and Overfitting

- **Training set**: the data used to tune the parameters. **Test set**: data the model has never seen, kept just for the exam.
- **Generalization**: the ability to do well on data it hasn't seen. This is what we really want.
- **Overfitting**: doing well on the training set but poorly on new data, like memorizing the answers to the practice problems.

Lab 1 is a ready-made example: after seeing only 50 cards, it gets the whole training set right, but only about 94% of new cards;
after 1,000 more cards, its accuracy on new cards rises to about 99%. **More data, and more varied data, usually leads to better generalization.** That's what ImageNet was all about.

📖 First appears in: [Act 4](#act-4--19581960-rosenblatt-and-his-electronic-brain), [Act 8](#act-8--20062009-a-dam-made-of-data)

### 7. Neurons and the Perceptron

An artificial neuron does two things:
1. **Weighted sum**: multiply each input by a weight, add them up, then add a bias: `z = w1·x1 + w2·x2 + … + b`;
2. **Activation**: if z is greater than 0, output 1 (fire); otherwise output 0 (or -1).

- **McCulloch–Pitts neuron** (1943): the weights are set by people; **it doesn't learn**.
- **Perceptron** (1958): the weights start at 0 and are learned with one simple rule. When it guesses wrong, `w ← w + y·x` (y is the correct answer, ±1).

📖 First appears in: [Act 1](#act-1--1943-chicago-the-runaway-boy-and-the-logic-of-neurons), [Act 4](#act-4--19581960-rosenblatt-and-his-electronic-brain)

### 8. Linear Separability and XOR

The perceptron's decision rule, `w1·x1 + w2·x2 + b > 0`, draws **a straight line** on the plane: one side says 1, the other side says 0.
So a perceptron can only solve problems that one straight line can separate. This is called being **linearly separable**.

AND and OR are both linearly separable. XOR's two classes sit on the two diagonals of a square,
so **no straight line can separate them**, and a single-layer perceptron is doomed never to learn it. Minsky and Papert's book also proved limits more general than this.

📖 First appears in: [Act 5](#act-5--19691973-a-book-a-wall-a-winter)　🧪 `xor_1969.py`

### 9. Multilayer Networks and Activation Functions

The fix is to add one or more **hidden layers** between the input and the output.
Each layer can re-represent the data "from a new angle," and stacking several layers lets you draw curved, complicated boundaries.

But here's the key: every layer must be followed by a **nonlinear activation function** (like the sigmoid in Lab 3, or the ReLU that's common today).
Without it, stacked linear computations are still linear, no different from a single layer, and XOR still can't be learned. (That's Think about it question 3 in Lab 3.)

### 10. Backpropagation

Multilayer networks bring a new problem: when the output is wrong, **which weight in which layer is to blame?** This is called the "credit assignment problem."

Backpropagation's answer:
1. **Forward**: compute from the input, layer by layer, to the output;
2. **Compute the error**: compare with the correct answer to get the loss;
3. **Backward**: use the chain rule to pass "the derivative of the loss with respect to the output" back, layer by layer, getting the derivative of the loss with respect to every weight (the gradient);
4. **Update**: each weight takes a small gradient descent step.

Training today's large models with hundreds of billions of parameters still uses these four steps. Frameworks like PyTorch just do step 3 for you automatically.

📖 First appears in: [Act 6](#act-6--1986-sending-the-error-backward)　🧪 `backprop_1986.py`

### 11. Convolutional Neural Networks (CNNs)

An ordinary network connects every pixel of an image to every neuron. That's too many parameters, and it knows nothing about the structure of an "image." CNNs have two clever design choices:
- **Local receptive fields**: each neuron looks at only a small patch of the image (say 5×5), hunting for local patterns like edges and corners;
- **Weight sharing**: the same "pattern detector" slides across the whole image. Whether a cat's ear shows up in the top-left or the bottom-right corner, the same detector looks for it.

Shallow layers find lines; deeper layers combine lines into eyes, wheels and faces.

📖 First appears in: [Act 7](#act-7--19891997-the-network-that-read-checks-and-the-champion-that-couldnt-learn), [Act 9](#act-9--2012-two-graphics-cards-in-a-bedroom-an-auction-in-a-casino)

### 12. Deep Learning and GPUs

**Deep learning** just means "neural networks with many layers." The "deep" refers to the number of layers.

Most of a neural network's computation is **matrix multiplication**: thousands upon thousands of independent multiplications and additions.
GPUs were originally designed for game graphics, processing millions of pixels at once, so they're naturally good at this kind of massively parallel computation.
AlexNet proved that when you bring together **a big network + big data + GPUs**, deep learning can leave traditional methods far behind.

📖 First appears in: [Act 9](#act-9--2012-two-graphics-cards-in-a-bedroom-an-auction-in-a-casino)

### 13. Generative Models and GANs

Most of the models so far are **discriminative**: given an image, decide whether it's a cat or a dog.
A **generative model** works the other way around: it learns what the data looks like, then creates new, realistic data.

A GAN pits two networks against each other: the **generator** makes fakes, and the **discriminator** tells real from fake. Each pushes the other to improve, like a counterfeiter and the person checking for fake bills.
Today's image generation mostly uses "diffusion models," and large language models are essentially generative models too: they generate the next word.

📖 First appears in: [Act 10](#act-10--20142016-an-idea-in-a-bar-move-37-on-the-board)

### 14. Attention and the Transformer

When you read the word "it," you automatically look back to find what "it" refers to. That's exactly what the **attention mechanism** does:
as it processes each word, the model gives every other word in the sentence a "relevance" score, then pulls in their information, weighted by those scores.

The **Transformer** is a network built by stacking attention. It has two advantages:
- any two words in a sentence can "talk" directly, no matter how far apart they are;
- all the words can be computed **at the same time**, which suits GPUs perfectly, so models can be made enormous.

Almost every large language model today (GPT, Claude, DeepSeek…) is a Transformer.

📖 First appears in: [Act 11](#act-11--20172022-eight-people-and-one-paper)

### 15. Large Language Models (LLMs)

- **Token**: the smallest unit a model reads and writes. It might be a single character, a word, or part of a word.
- **Pre-training**: reading huge amounts of text and practicing "guess the next token" over and over (self-supervised learning). This step is where the model picks up language, knowledge, and some ability to reason.
- **Fine-tuning**: continuing training on smaller, more targeted data, for example to teach it to answer in a "conversation" format.
- **RLHF** (reinforcement learning from human feedback): people rank several of the model's answers, and reinforcement learning then makes the model lean toward the answers people prefer.
- **Context window**: the maximum number of tokens the model can "see" at once. Anything beyond that, it can't see.

A common misunderstanding: a large model's API itself **has no memory**. On every call, you have to send it the full conversation history again.
This is the key to understanding agents and multi-agent systems, and the [side quest](../side-quests/ai-talks-to-ai/) covers it in detail.

📖 First appears in: [Act 11](#act-11--20172022-eight-people-and-one-paper)

### 16. Agents

**In one sentence**: Agent = large model + tools + a loop.

A chat model can only "talk"; an agent can "do." It works through the ReAct loop:
```
while the task isn't done:
    Think: what should I do next?
    Act: call a tool (search, read a file, run code…)
    Observe: see what the tool returned
```
The key: **the model decides the next step itself**, instead of people hard-coding the process in advance.

📖 First appears in: [Act 12](#act-12--20222026-from-talking-to-doing)

### 17. Harness

A "harness" originally means **the gear for a horse**: the reins, straps and all the rest, which let the horse's strength be guided in the right direction.
In AI, the model is the horse, and **the harness is everything wrapped around the model that makes it work reliably**:
- the `while` loop itself;
- which tools are available, and how tool results are fed back to the model;
- how the context is managed: what goes in, what gets deleted, how to compress it when it gets too long;
- permissions and safety boundaries: which actions need human approval, and what sandbox it runs in;
- feedback loops: tests, code checks, another agent reviewing the work, so mistakes get caught and fixed in time.

OpenAI's experience: with the same model, how good the harness is makes a world of difference.
Tools you may be using right now, like Claude Code and Codex, are essentially harnesses.

📖 First appears in: [Act 12](#act-12--20222026-from-talking-to-doing)

### 18. Multi-Agent Collaboration

When one agent isn't enough, have several agents split up the work:
- **Orchestrator–worker**: a lead agent breaks down the task, hands out the work and pulls the results together, while several subagents work in parallel;
- **Writer–reviewer**: one writes, the other finds fault;
- **Routing**: hand each type of task to the model that's best at it;
- **Debate**: several models each give their view and argue against each other, then reach a conclusion.

Why split up the work? Because each agent's context window is limited, so a task that's too big won't fit in one agent; and independent subtasks can be done in parallel.
The price is that token use multiplies, for example the roughly 15x that Anthropic reported. So it's only worth it when the task is valuable enough.

How exactly do two AIs "talk" to each other? See the side quest: [How Do Two AIs "Talk"?](../side-quests/ai-talks-to-ai/).

📖 First appears in: [Act 12](#act-12--20222026-from-talking-to-doing)

---

## References

**1943–1973**
- Amanda Gefter, [The Man Who Tried to Redeem the World with Logic](https://nautil.us/the-man-who-tried-to-redeem-the-world-with-logic-235253), *Nautilus*, 2015 (Pitts's life)
- McCulloch & Pitts, *A Logical Calculus of the Ideas Immanent in Nervous Activity*, Bulletin of Mathematical Biophysics, 1943
- D. O. Hebb, *The Organization of Behavior*, 1949
- A. M. Turing, [Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238), *Mind*, 1950
- McCarthy, Minsky, Rochester, Shannon, [A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence](http://jmc.stanford.edu/articles/dartmouth/dartmouth.pdf), 1955
- A. L. Samuel, *Some Studies in Machine Learning Using the Game of Checkers*, IBM Journal, 1959
- [Arthur Samuel's Legacy](https://webdocs.cs.ualberta.ca/~chinook/project/legacy.html) (University of Alberta's Chinook project; source for the TV demo and the IBM stock price)
- IBM, [The games that helped AI evolve](https://www.ibm.com/history/early-games) (the 1962 match against Nealey)
- Cornell Chronicle, [Professor's perceptron paved the way for AI – 60 years too soon](https://nbb.cornell.edu/news/professors-perceptron-paved-way-ai-60-years-too-soon), 2019 (the 1958 demo and press coverage)
- Wikipedia, [Perceptron](https://en.wikipedia.org/wiki/Perceptron), [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt), [Perceptrons (book)](https://en.wikipedia.org/wiki/Perceptrons_(book))
- [Lighthill Report](https://www.aiai.ed.ac.uk/events/lighthill1973/) (University of Edinburgh archive, including the BBC debate)

**1986–2016**
- Rumelhart, Hinton & Williams, [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0), *Nature* 323, 1986
- ACM, [Fathers of the Deep Learning Revolution Receive ACM A.M. Turing Award](https://www.acm.org/media-center/2019/march/turing-award-2018), 2019 (includes LeCun's check-reading system)
- [The Nobel Prize in Physics 2024](https://www.nobelprize.org/prizes/physics/2024/summary/)
- Wikipedia, [ImageNet](https://en.wikipedia.org/wiki/ImageNet), [AlexNet](https://en.wikipedia.org/wiki/AlexNet)
- Computer History Museum, [CHM Releases AlexNet Source Code](https://computerhistory.org/blog/chm-releases-alexnet-source-code/), 2025
- Cade Metz, *Genius Makers*, 2021 (the DNNresearch auction)
- MIT Technology Review, [The GANfather](https://www.technologyreview.com/2018/02/21/145289/the-ganfather-the-man-whos-given-machines-the-gift-of-imagination/), 2018
- Wired, *The Sadness and Beauty of Watching Google's AI Play Go*, 2016
- [Lee Sedol](https://en.wikipedia.org/wiki/Lee_Sedol) (his retirement and "an entity that cannot be defeated")

**2017–2026**
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017
- Steven Levy, *8 Google Employees Invented Modern AI. Here's the Inside Story*, Wired, 2024
- Reuters, *ChatGPT sets record for fastest-growing user base*, 2023-02
- Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022
- [AutoGPT](https://en.wikipedia.org/wiki/AutoGPT)
- Park et al., [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442), 2023
- [Model Context Protocol](https://en.wikipedia.org/wiki/Model_Context_Protocol), [Agent2Agent](https://en.wikipedia.org/wiki/Agent2Agent)
- Anthropic, [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system), 2025
- OpenAI, [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/), 2026

---

⬅️ [Contents](../README.md) | ➡️ [Chapter 2 · Linear Models and Gradient Descent](../02-linear-models/) | Side quest: [How Do Two AIs "Talk"?](../side-quests/ai-talks-to-ai/)
