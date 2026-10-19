# Chapter 8 — Multi-Agent Collaboration: When One Agent Isn't Enough

> Chapter 7 put a harness on one agent. But some jobs are too big for a single agent, however hard it works: there's too much material to fit in one context,
> or the job needs several completely different skills. So people came up with the idea of "forming a team."
> The idea isn't new at all: as early as the 1970s, researchers were already asking how many small programs could cooperate like a team.
> This chapter starts there and goes all the way to the public argument in 2025 over "whether to build multi-agent systems at all."

**How to read this chapter:** Read the stories first, then run the labs, and save the [Concepts](#concepts) for last. The labs use a mock model by default, so no API key is needed.

| When | What happened | Lab / concept |
|---|---|---|
| Apr 2023 | Stanford's virtual town: 25 agents throw a party on their own | Multi-agent systems |
| 1973 | Hewitt's Actor model: cooperating through messages alone | Message passing |
| 1971–1976 | Hearsay-II: a group of "experts" around a blackboard | Blackboard architecture |
| 1978–1980 | Reid Smith's Contract Net: announce, bid, award | 🧪 `contract_net.py` |
| 1986 | Minsky's *The Society of Mind*: intelligence comes from diversity | Debate |
| 2023 | CAMEL, multi-agent debate, ChatDev, AutoGen | 🧪 `debate.py` |
| Mar 2025 | MAST: 14 ways multi-agent systems fail | 🧪 `write_review.py` |
| Jun 2025 | Anthropic's multi-agent research system vs. Cognition's *Don't Build Multi-Agents* | 🧪 `orchestrator.py` |

---

## Prologue — Apr 2023: A Valentine's Day Party in a Small Town

In April 2023, Joon Sung Park of Stanford University and his collaborators at Stanford and Google published a paper
called *Generative Agents: Interactive Simulacra of Human Behavior*.
Using large language models, they built a pixel-art town called **Smallville**, home to **25 agents**.
Each agent had its own name, job, personality and relationships. They got up, made breakfast, went to work, chatted, and remembered what had happened.

The researchers did just one thing: they told one of the agents, the café owner **Isabella Rodriguez**,
that she wanted to throw a Valentine's Day party from 5 to 7 pm on February 14.

Nobody wrote a script for what happened next. Whenever Isabella ran into friends and regulars, in the café or out in town, she invited them.
On the afternoon of February 13, she started decorating the café. Her good friend **Maria**, a regular, came by and was roped into helping.
Maria's character description said she had a crush on **Klaus**. That evening she invited Klaus to go to the party with her, and he happily said yes.
At 5 pm on Valentine's Day, five agents, Maria and Klaus among them, showed up at the café.
Over the two days of the simulation, news of the party reached about half of the town's residents, all through conversations between agents.

A single agent only acts according to its own description. But put many agents together, passing messages and influencing each other, and things happen that nobody planned.
That is exactly what makes multi-agent systems both fascinating and a headache. Still, the idea of "many programs cooperating" is fifty years older than large language models.

> 💡 Look ahead: → [Multi-agent systems](#1-multi-agent-systems)

---

## Act 1 — 1973, MIT: Everything Is an Actor

In late 1972, **Carl Hewitt** at MIT put aside the programming language he was designing,
because he and a few graduate students had come up with a new model of computation. In August 1973, at the third International Joint Conference on Artificial Intelligence (IJCAI), held at Stanford University,
Hewitt, Peter Bishop and Richard Steiger presented the paper *A Universal Modular ACTOR Formalism for Artificial Intelligence*.

There is only one kind of thing in this model: the **actor**. When an actor receives a message, it can do only three things:
1. send a finite number of messages to other actors;
2. create a finite number of new actors;
3. decide how it will respond to the next message it receives.

There is no shared memory, and nobody reads anybody else's variables: **the only way to cooperate is by sending messages**.
According to Hewitt himself, the model was inspired by physics (general relativity and quantum mechanics), and it also drew on Lisp, Simula, early Smalltalk and packet-switched networks.

Look at today's multi-agent systems fifty years later: the orchestrator sends a subagent a task description, the subagent finishes and hands back a conclusion,
and neither can see the other's context. That is almost exactly the actor's rulebook, except that the "actors" are now large language models.

> 💡 Look ahead: → [Message passing and the Actor model](#2-message-passing-and-the-actor-model), [Context isolation](#6-context-isolation-vs-shared-context)

---

## Act 2 — 1971–1976, Carnegie Mellon: Experts Around a Blackboard

In 1971, the US Department of Defense's Advanced Research Projects Agency (ARPA, later renamed DARPA) launched a five-year research program on speech understanding.
The goal: by 1976, a system that could understand connected speech from several speakers, with a vocabulary of about 1,000 words.
Carnegie Mellon University, BBN, SRI and others took part.

Understanding a sentence is hard: the sound has to be cut into phonemes, phonemes put together into syllables and words, and words into phrases and sentences, and every level is full of uncertainty.
Carnegie Mellon's **Hearsay-II** team (Lee Erman, Frederick Hayes-Roth, Victor Lesser, Raj Reddy and others) came up with a solution:
turn the knowledge at each level into an independent "knowledge source," and have them all work around a shared **blackboard**.
The acoustic knowledge source writes on the blackboard "this might be such-and-such a phoneme"; the word-building one sees it and writes "these phonemes might form such-and-such a word";
the grammar one then builds on top of that. Nobody gives orders to anybody directly: **everyone only reads the blackboard and only writes on the blackboard**, and a scheduler decides who acts next.

When the program ended in 1976, the system that actually met the goal was Harpy, another system from Carnegie Mellon.
Hearsay-II didn't win this race, but its "blackboard architecture" lived on and became a classic way for many later AI systems to organize multiple modules.

> 💡 Look ahead: → [Blackboard architecture](#3-blackboard-architecture)

---

## Act 3 — 1978–1980, Stanford: Announce, Bid, Award

The blackboard solved "how everyone shares information," but there was still another question: **who should do the job?**

In 1978, **Reid G. Smith** finished his PhD thesis at Stanford University. He studied "distributed problem solving":
how a group of loosely connected computers can cooperate to solve a problem. His answer borrowed the most common way human society divides up work: **tendering**.

In December 1980, he formally published the **Contract Net Protocol** in *IEEE Transactions on Computers*:
- a node with a task to be done acts as the **manager** and broadcasts the task; this is the **announcement**;
- the other nodes assess whether they can do it and how well, and those that want it submit a **bid**;
- the manager compares the bids and awards the **contract** to the best-suited node;
- the winning node does the work and **reports** the result back. If the task is too big, it can act as a manager itself and subcontract part of it.

The demos in the paper included the N-queens problem and a "distributed sensing" system in which many sensor nodes cooperate to monitor an area.
The cleverest part: **the manager doesn't need to know what each node is good at or how busy it is**; the nodes write that in their bids themselves.
Later, the Foundation for Intelligent Physical Agents (FIPA) turned it into a standard, and it became the classic protocol for "task allocation" in the multi-agent field.

> 🧪 **Try it**: run [`contract_net.py`](./contract_net.py). Four contractors bid according to their own skills. Watch how load balancing emerges on its own,
> and what the manager does when nobody bids. Then compare it with "taking turns, ignoring skills."

> 💡 Look ahead: → [The Contract Net](#4-the-contract-net)

---

## Act 4 — 1986: The Society of Mind

In the early 1970s, **Marvin Minsky** and **Seymour Papert** at MIT began to mull over a bold idea:
could the human mind itself be a society made up of many simple little "agents"?
According to Minsky, the theory's biggest source of inspiration was their attempt to get a machine to build with children's blocks, using a robot arm, a camera and a computer:
something that looks as simple as can be turned out, once you took it apart, to need many abilities working together: seeing, grasping, planning, correcting mistakes.

In 1986, Minsky wrote the theory up as a book: *The Society of Mind*.
The book is made up of 270 self-contained short essays, arranged in 30 chapters. It contains a famous passage:

> "What magical trick makes us intelligent? The trick is that there is no trick. The power of intelligence stems from our vast diversity, not from any single, perfect principle."

Nearly forty years later, large-language-model researchers dug the book out again.

> 💡 Look ahead: → [Multi-agent debate](#8-multi-agent-debate-and-majority-voting)

---

## Act 5 — 2023: Large Language Models Start Forming Teams

Once ChatGPT took off, researchers quickly wondered: if one model can chat, what happens if two models chat with each other?

**March 2023, CAMEL.** Guohao Li and others at King Abdullah University of Science and Technology (KAUST) published CAMEL.
They had two models play the "user" and the "assistant" and keep talking about a task on their own; this is called **role-playing**.
They also honestly recorded all the ways it went wrong: two agents in the middle of a conversation **swapped roles**; the assistant just repeated the instructions back;
replies were half-hearted; and there were **infinite loops**: two agents talking on and on without end,
even after they had realized they were stuck in a loop, because they were still "in character" and couldn't end the conversation.

**May 2023, multi-agent debate.** Yilun Du, Shuang Li, Antonio Torralba,
Joshua Tenenbaum and Igor Mordatch of MIT and Google Brain published a paper in which several model instances each answer the same question,
then look at one another's answers and reasoning, and after a few rounds of debate reach a common conclusion. The paper says explicitly that the method was inspired by Minsky's *The Society of Mind*.
The experiments showed that debate improved the models on both factual accuracy and reasoning problems.
But the story has a sequel: in 2025, the paper *Debate or Vote* by Hyeong Kyu Choi and others found, on seven benchmarks,
that most of the gains from multi-agent debate actually come from "majority voting" itself.

**July 2023, ChatDev.** Chen Qian and others, mostly from Tsinghua University, opened a "virtual software company":
the CEO, the CPO, the CTO, a programmer, a reviewer, a tester and an art designer were all played by models, developing software in a waterfall process of "design, coding, testing, documentation."
According to the paper's figures, making a small piece of software took less than 7 minutes on average and cost less than $1.

**August 2023, AutoGen.** Qingyun Wu, Chi Wang and others from Microsoft Research and several universities released the AutoGen framework,
which treats all multi-agent collaboration as "conversation": for example, an assistant that writes code, paired with a proxy that stands in for the user and can actually run the code, going back and forth until the task is done.

Within a year, "multi-agent" had become one of the hottest buzzwords. But behind all the excitement, one question still hadn't been taken seriously: where exactly do these systems go wrong?

> 🧪 **Try it**: run [`debate.py`](./debate.py). Three agents debate the classic "bat and ball" trick question.
> See how differently the debate turns out for agents that "check their work" and agents that "follow the crowd."

> 💡 Look ahead: → [Role-playing](#7-role-playing), [Multi-agent debate](#8-multi-agent-debate-and-majority-voting), [Stopping conditions](#11-stopping-conditions-and-going-in-circles-detection)

---

## Act 6 — Mar 2025, Berkeley: Why Multi-Agent Systems Fail

In March 2025, Mert Cemri, Melissa Z. Pan, Shuyi Yang and others at the University of California, Berkeley published the paper
*Why Do Multi-Agent LLM Systems Fail?* They collected run logs from several popular multi-agent frameworks and had several experts annotate them one by one
(the agreement between annotators, measured by Cohen's κ, reached 0.88), producing the first empirically grounded taxonomy of failures: **MAST**.

MAST sorts failures into three categories, 14 kinds in all:
- **Specification and system design**: disobeying the task specification, disobeying the role specification, repeating steps, losing the conversation history, not knowing when to stop;
- **Misalignment between agents**: conversation resets, acting without asking for clarification, getting off track, withholding information that should be shared, ignoring other agents' input, a mismatch between reasoning and action;
- **Verification and termination**: ending too early, no verification or incomplete verification, incorrect verification.

The paper's most important conclusion: the failures of multi-agent systems **come mainly from system design and from problems in how agents work together, not just from models not being smart enough**.
In other words, many of the problems don't have to wait for stronger models; better "organizational design" can fix them. For example, a conscientious reviewer, plus a rule that calls a halt.

> 🧪 **Try it**: run [`write_review.py`](./write_review.py). One agent writes code; another runs the tests in a subprocess with a timeout and gives feedback.
> Watch how an infinite loop gets caught by the timeout, and how a writer that "goes in circles" gets stopped.

> 💡 Look ahead: → [The writer–reviewer loop](#9-the-writerreviewer-loop), [How multi-agent systems fail](#10-how-multi-agent-systems-fail)

---

## Act 7 — Jun 2025: Two Voices in the Same Week

In mid-June 2025, almost at the same time, two companies that build agents published two articles with opposite views.

**One came from Anthropic** (June 13): *How we built our multi-agent research system*.
In Claude's Research feature, a lead agent first analyzes the question and makes a plan, then sends out several subagents to search in parallel.
Each subagent has **its own separate context window**, and in the end hands only its most important findings back to the lead agent to pull together.
On their internal research evaluation, a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 as the subagents
outperformed a single Claude Opus 4 by **90.2%**.
They were also frank about the costs: agents use about 4 times as many tokens as an ordinary chat, and multi-agent systems about **15 times** as many;
early versions were also prone to "spawning 50 subagents for simple queries, scouring the web endlessly for nonexistent sources, and distracting each other with excessive updates."
The article also contains a very important passage:

> "Some domains that require all agents to share the same context or involve many dependencies between agents are not a good fit for multi-agent systems today.
> For instance, most coding tasks involve fewer truly parallelizable tasks than research..."

**The other came from Cognition** (the company behind the AI programmer Devin). Its author, Walden Yan, didn't mince words in the title: *Don't Build Multi-Agents*.
He put forward two principles:
1. **Share context, and share full agent traces, not just individual messages**;
2. **Actions carry implicit decisions, and conflicting decisions carry bad results.**

He gave an example: suppose the task is "build a little Flappy Bird game," split between two subagents, one making the background and one making the bird.
The background subagent misunderstood and made a background in the style of *Super Mario Bros.*; the bird subagent made a bird that didn't look like a game asset and didn't move anything like the one in Flappy Bird.
Each subagent "completed its task," but put together, the pieces add up to nothing.

The two articles look like they are at loggerheads, but they don't actually contradict each other. **Research** can be split into many independent, read-only parts, and each subagent's findings can be compressed into a few sentences;
**writing code** involves implicit decisions at every step, and the steps depend tightly on each other, so once you split it up, everyone easily goes their own way.
Whether to use multiple agents depends on whether the task itself can be split cleanly.

> 🧪 **Try it**: run [`orchestrator.py`](./orchestrator.py). An orchestrator splits "research Chapters 1–7 of this repo" among 7 workers running in parallel,
> then compares speed, context size and tokens with "one agent reads everything from start to finish."

> 💡 Look ahead: → [Orchestrator–worker](#5-orchestratorworker), [Context isolation vs. shared context](#6-context-isolation-vs-shared-context), [Weighing the cost](#12-when-to-use-multiple-agents)

---

## Epilogue — Common Ways to Form a Team

| Pattern | How the work is split | Good for | Watch out for | Example in this chapter |
|---|---|---|---|---|
| Orchestrator–worker | A lead splits the task, several workers do the parts in parallel, and the lead pulls it together | Big tasks that split into independent parts, such as research | Splitting it unclearly; workers handing back too little information | `orchestrator.py` |
| Contract Net | Announce the task, everyone bids, award it to the best | Members with different abilities and changing workloads | Dishonest bids; nobody bidding | `contract_net.py` |
| Debate | Each answers, they challenge each other, then vote or let a judge decide | Questions with a standard answer that can be checked | Conformity; everyone making the same mistake | `debate.py` |
| Writer–reviewer loop | One writes, one reviews, revise again and again | Tasks with clear acceptance criteria | Going in circles; the reviewer itself being wrong | `write_review.py` |
| Blackboard | Everyone reads and writes the same shared area | Several kinds of knowledge that have to be pieced together step by step | Scheduling who goes first | (concept only) |

More agents isn't always better. Before you split anything up, ask yourself three questions:
1. Can this task **be split into independent parts**?
2. Once split, is each part's **information too big** to fit in one context?
3. Is the speed and quality you gain **worth** spending several times as many tokens?

Only if all three answers are "yes" should you consider multiple agents. In the next chapter, we put models from **different vendors** on one team and see how protocols let them talk to each other.

---

## Labs

```bash
python orchestrator.py       # orchestrator–worker: 7 workers research in parallel (uses the mock model by default)
python contract_net.py       # Contract Net: bidding for tasks
python debate.py             # multi-agent debate + a thought experiment
python write_review.py       # writer–reviewer loop: running the tests in a subprocess

# Switch to a real model (needs the matching API key; see Chapter 5)
python orchestrator.py --provider claude
python debate.py --provider deepseek
```

| Lab | What you should see | Think about it (also at the end of each script) |
|---|---|---|
| `orchestrator.py` | 7 workers in parallel, and the total time is about the same as one worker's; each worker's context is about 1/6 of the single agent's; in this example the multi-agent setup actually uses fewer tokens in total | If the Chapter 5 worker needed the Chapter 4 conclusion, could they still run in parallel? |
| `contract_net.py` | After the Programmer has taken two jobs, the Mathematician wins the coding task; when nobody can draw, the manager rewrites the task as "write the diagram in Mermaid"; the Contract Net is done in 8.8 hours, while taking turns takes 44.4 hours | If a contractor quoted half the time it really needs, how could the manager find out? |
| `debate.py` | The same start, one right and two wrong: the verifiers all end up right, the conformists all end up wrong; in the thought experiment, a debate in which everyone conforms gives exactly the same result as a straight vote | With models from three different vendors, would "all falling into the same trap" still hold? |
| `write_review.py` | Version 1 fails 3 tests; version 2's infinite loop is caught by the 2-second timeout; version 3 passes everything; the stubborn writer is stopped in round 2 | When the writer can see all the tests, how do you keep it from "fooling" them? |

About real-model mode: I tested the real-API paths of all four scripts against a mock server (to confirm they don't crash and have a fallback when the format is wrong),
but have not run them with a real API key. A real model will behave differently from the mock model's script; for the debate question, for example, today's large models will very likely all get it right in round 1.
The real mode of `write_review.py` runs the code the model writes on your computer. There are a timeout and resource limits, but this is not a real sandbox.

---

## Concepts

### 1. Multi-Agent Systems

A system made of several agents that work together in some way to complete a task.
Each agent has its own instructions, context and tools. How they communicate, how they divide the work and who has the final say: that is the system's "organizational design."
A multi-agent system can show "emergent" behavior you never see in a single agent, both good and bad.

📖 First appears in: [Prologue](#prologue--apr-2023-a-valentines-day-party-in-a-small-town)

### 2. Message Passing and the Actor Model

In the Actor model, each actor can communicate with others only through messages. On receiving a message, it can send messages, create new actors, and change how it will respond next time.
There is no shared memory, so two parties can never change the same variable at the same time, which makes it a natural fit for parallel and distributed computing.
Most of today's multi-agent systems work the same way: what passes between agents is pieces of text (task descriptions, conclusions), not the whole shared context.

📖 First appears in: [Act 1](#act-1--1973-mit-everything-is-an-actor)

### 3. Blackboard Architecture

Several specialized modules (knowledge sources) share one "blackboard": each module watches the blackboard, processes any information it knows how to handle, and writes the results back to the blackboard;
a scheduler decides which one to activate next. The modules don't talk to each other directly; they cooperate only indirectly, through the blackboard.
In LLM systems, a shared file, a database or a task list can all play the role of a blackboard.

📖 First appears in: [Act 2](#act-2--19711976-carnegie-mellon-experts-around-a-blackboard)

### 4. The Contract Net

A protocol that allocates tasks by "tendering": the manager sends an **announcement**, contractors submit **bids** (or decline), the manager awards the **contract**, and the contractor **reports** the result when it's done.
In FIPA's standard version, these messages are called cfp (call for proposals), propose / refuse, accept-proposal / reject-proposal, and inform.
The advantage is that the manager doesn't need to know the inner workings of each contractor. The hard parts: whether bids can be trusted, what to do when nobody bids, and what to do when the same contractor wins several contracts at once.

📖 First appears in: [Act 3](#act-3--19781980-stanford-announce-bid-award)　🧪 `contract_net.py`

### 5. Orchestrator–Worker

An **orchestrator** (also called the lead agent) understands the task, splits it into subtasks, hands them to **workers** (also called subagents),
and finally pulls the results together. The difference from a fixed workflow: how many pieces to split it into, and how, is decided on the spot by the orchestrator for the task at hand.
In *Building effective agents* (December 2024), Anthropic lists it as one of the common workflow patterns.

📖 First appears in: [Act 7](#act-7--jun-2025-two-voices-in-the-same-week)　🧪 `orchestrator.py`

### 6. Context Isolation vs. Shared Context

**Isolation**: each subagent has its own clean context, holding only its own part of the task. The benefits: it isn't distracted by other information,
the context is small, the work can run in parallel, and a large amount of raw material can be "compressed" into a few sentences of conclusions to hand back.
**Sharing**: every participant sees the full history and one another's decisions. The benefit: nobody goes off in their own direction.
It's a trade-off: Anthropic's research system chose isolation, because the parts of a research task don't depend on each other;
Cognition argues for sharing, because every step of writing code depends on the decisions made before it.

📖 First appears in: [Act 1](#act-1--1973-mit-everything-is-an-actor), [Act 7](#act-7--jun-2025-two-voices-in-the-same-week)　🧪 `orchestrator.py`

### 7. Role-Playing

Give each agent a role (product manager, programmer, tester...) and a matching system prompt, and let them talk to each other in character.
CAMEL uses two roles, "user" and "assistant"; ChatDev simulated a whole software company.
Roles make the division of labor clear, but problems like **role swapping**, **just repeating the instructions** and **not being able to stop** also show up, and have to be constrained in the prompts and in the program.

📖 First appears in: [Act 5](#act-5--2023-large-language-models-start-forming-teams)

### 8. Multi-Agent Debate and Majority Voting

**Debate**: several agents first answer independently, then see one another's answers and reasoning, decide whether to stick with their answer or change it, and finally vote or let a judge decide.
**Majority voting**: several agents (or several samples from the same model) answer on their own, and you simply take the most common answer, with no communication.
What debate is supposed to achieve is "the minority's right answer convincing the majority," which requires the agents to be able to **check** the others' claims **independently**;
if everyone just follows the crowd, debate degenerates into majority voting, and can even talk someone who was right into being wrong.
Also, when several agents use the same model, they tend to make the same mistakes, and voting can't save them.

📖 First appears in: [Act 4](#act-4--1986-the-society-of-mind), [Act 5](#act-5--2023-large-language-models-start-forming-teams)　🧪 `debate.py`

### 9. The Writer–Reviewer Loop

Also called the **evaluator-optimizer**: one agent produces, another evaluates and gives specific feedback, and the loop continues until the work passes.
Criteria a program can judge (tests, format validation, compiling) should be judged by a program first; what can't be tested (readability, style) can then go to a model for comments.
When a reviewer runs code someone else wrote, it must have a timeout, resource limits and an isolated working directory.

📖 First appears in: [Act 6](#act-6--mar-2025-berkeley-why-multi-agent-systems-fail)　🧪 `write_review.py`

### 10. How Multi-Agent Systems Fail

MAST sorts the failures of multi-agent systems into three categories: **specification and system design** (for example, disobeying roles, repeating steps, not knowing when to stop),
**misalignment between agents** (for example, acting without asking for clarification, not sharing what should be shared, ignoring other agents' input),
and **verification and termination** (for example, ending too early, no verification, incorrect verification).
Most of these problems can be eased with better design: clear task descriptions, an explicit hand-off format, independent verification and reliable stopping conditions.

📖 First appears in: [Act 6](#act-6--mar-2025-berkeley-why-multi-agent-systems-fail)　🧪 `write_review.py`

### 11. Stopping Conditions and Going-in-Circles Detection

When several agents talk to each other, it's easy for nobody to call a halt: they thank each other, revise the same spot again and again, repeat the same search.
Common brakes: a maximum number of rounds; a clear sign of completion (for example, all tests passing); detecting "this round is exactly the same as the last one";
token and time budgets (Chapter 7). Once it stops, hand the situation to a human instead of quietly ending.

📖 First appears in: [Act 5](#act-5--2023-large-language-models-start-forming-teams)　🧪 `write_review.py`

### 12. When to Use Multiple Agents

The benefits of multiple agents: parallelism (faster), context isolation (each agent is more focused), and division of labor (different prompts, tools, even different models).
The costs: more tokens (Anthropic reports about 15 times as many as an ordinary chat), more coordination failures, harder debugging.
A good fit: tasks that split into independent parts, where each part holds a lot of information and the result is worth spending more on, such as broad research.
Not a good fit: tasks whose steps depend tightly on each other, or where everyone needs to share the full context, such as most programming tasks.

📖 First appears in: [Act 7](#act-7--jun-2025-two-voices-in-the-same-week), [Epilogue](#epilogue--common-ways-to-form-a-team)　🧪 `orchestrator.py`

---

## References

- J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang, M. S. Bernstein, *Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023; [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- C. Hewitt, P. Bishop, R. Steiger, *A Universal Modular ACTOR Formalism for Artificial Intelligence*, IJCAI 1973; [IJCAI 1973 proceedings](https://www.ijcai.org/Proceedings/1973/)
- L. D. Erman, F. Hayes-Roth, V. R. Lesser, D. R. Reddy, *The Hearsay-II Speech-Understanding System: Integrating Knowledge to Resolve Uncertainty*, ACM Computing Surveys 12(2), 1980
- R. G. Smith, *The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver*, IEEE Transactions on Computers C-29(12), 1980; FIPA, *Contract Net Interaction Protocol Specification* (document no. 00029)
- M. Minsky, *The Society of Mind*, Simon & Schuster, 1986
- G. Li, H. A. A. K. Hammoud, H. Itani, D. Khizbullin, B. Ghanem, *CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society*, NeurIPS 2023; [arXiv:2303.17760](https://arxiv.org/abs/2303.17760)
- Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, I. Mordatch, *Improving Factuality and Reasoning in Language Models through Multiagent Debate*, ICML 2024; [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)
- H. K. Choi, X. Zhu, Y. Li, *Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?*, NeurIPS 2025; [arXiv:2508.17536](https://arxiv.org/abs/2508.17536)
- C. Qian et al., *ChatDev: Communicative Agents for Software Development*, ACL 2024; [arXiv:2307.07924](https://arxiv.org/abs/2307.07924)
- Q. Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*; [arXiv:2308.08155](https://arxiv.org/abs/2308.08155)
- M. Cemri, M. Z. Pan, S. Yang et al., *Why Do Multi-Agent LLM Systems Fail?*; [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)
- Anthropic, [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system), 2025-06-13
- Cognition (Walden Yan), [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents), 2025-06
- Anthropic, [Building effective agents](https://www.anthropic.com/research/building-effective-agents), 2024-12 (orchestrator-workers, evaluator-optimizer)
- S. Frederick, *Cognitive Reflection and Decision Making*, Journal of Economic Perspectives 19(4), 2005 (the bat-and-ball question)

---

⬅️ [Chapter 7 · Harness Engineering](../07-harness/) | [Contents](../README.md) | ➡️ [Chapter 9 · Cross-Model Collaboration](../09-cross-model/)
