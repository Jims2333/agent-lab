# Chapter 6 — The Agent Loop: From Talking to Doing

> The large language models of Chapter 5 are great talkers, but once they finish talking, that's it: they can't look things up, can't do math, can't edit files.
> This chapter is about how people spent nearly sixty years getting machines from "chatting with you" to "doing things for you,"
> and about the core that almost every AI agent today shares: a simple loop.

**How to read this chapter:** Read the stories first, then run the labs, and save the [Concepts](#concepts) for last. The labs use a mock model by default, so no API key is needed.

| When | What happened | Lab / concept |
|---|---|---|
| 1966 | ELIZA: the first chat program that people got emotionally attached to | 🧪 `eliza_1966.py` |
| 1966–1972 | Shakey: the first robot that could plan its own actions | Planning |
| 1968–1970 | SHRDLU: a blocks world that understood commands | — |
| 2022 | ReAct: thinking while doing | 🧪 `react_agent.py` |
| 2023 | Toolformer and function calling | 🧪 `tools.py` |
| 2023 | SWE-bench: testing agents with real GitHub issues | — |
| 2024–2025 | Computer Use and Claude Code | Agents vs. workflows |

---

## Prologue — 1966, MIT: Please Leave the Room

Between 1964 and 1966, MIT computer scientist **Joseph Weizenbaum** wrote a program called **ELIZA**.
Its most famous script, DOCTOR, played a psychotherapist: you said "I feel tired," and it asked "Why do you feel tired?"
How it worked was extremely simple: find a keyword, cut out part of your own words, swap the pronouns, and drop them into a pre-written sentence pattern. It had **no idea** what you were talking about.

But people's reactions took Weizenbaum by surprise. As he recalled, even his own secretary, someone who had watched him write the program,
asked him to leave the room after exchanging just a few lines with ELIZA, so that she could talk to it alone.

This deeply disturbed Weizenbaum. He later wrote: "What I had not realized is that extremely short exposures to a relatively simple computer program
could induce powerful delusional thinking in quite normal people." In 1976 he published *Computer Power and Human Reason*, warning people not to project understanding and feelings onto machines.
The phenomenon later became known as the **ELIZA effect**.

> 🧪 **Try it**: run [`eliza_1966.py`](./eliza_1966.py). A few dozen lines of rules are enough to make a "therapist"; then see where it gives itself away.

> 💡 Look ahead: → [The ELIZA effect](#9-the-eliza-effect)

---

## Act 1 — 1966–1972, Stanford Research Institute: A Robot That Thinks

ELIZA could only talk. Around the same time, at the Stanford Research Institute (SRI) in California, a team wanted a machine that could **act in the real world**.

They built a robot: a TV camera, a range finder and an antenna mounted on a base. It wobbled as it moved, so everyone called it **Shakey**.
Shakey was the first mobile robot that could **reason about its own actions**. Other robots had to be told what to do step by step,
but you could tell Shakey something like "go to room D and push block 9 to door 4," and it would break the goal down into a series of basic actions by itself,
plan a route, and then carry it out.

For Shakey, the researchers invented two things with a far-reaching influence:
- **The A\* search algorithm**: much of today's navigation software and pathfinding in games still uses it;
- **The STRIPS planner**: it automatically works out the steps to take from "the current state, the goal state, and the available actions."

In 1970, *Life* magazine called Shakey the "first electronic person."

> 💡 Look ahead: → [Planning](#7-planning)

---

## Act 2 — 1968–1970, MIT: The Blocks World

At almost the same time, MIT graduate student **Terry Winograd** wrote **SHRDLU**.
It lived in a virtual "blocks world": a table with blocks and pyramids, big and small, red and green. You could talk to it in plain English,
with requests along the lines of *pick up the big red block*, *put it in the box*, or *why did you pick up the green one?* Not only would it do what you asked, it also understood what "it" referred to,
and could explain why it had done something.

At the time, this looked like real intelligence. But every piece of SHRDLU's knowledge and every grammar rule was written by hand.
Outside that tiny blocks world, it couldn't do anything at all. Systems built on "hand-written rules" like this were hard to scale up, which was one of the causes of the AI winter described in Chapter 1.

> 💡 Look ahead: Decades later, large language models replaced "hand-written rules" with "learning from data," but the goal of "understand the instruction, plan the steps, carry them out" stayed the same.

---

## Act 3 — 2022: Thinking While Doing

Once large language models could talk, a natural question was: could they also, like Shakey, decide the next step themselves and then carry it out?

In October 2022, **Shunyu Yao** of Princeton University and collaborators at Google published **ReAct** (Reasoning + Acting).
The method is simple: have the model alternate between writing out "thoughts" and "actions."
- Thought: I need to look up X first;
- Action: call the search tool to look up X;
- Observation: the tool returns a passage of text;
- Thought: based on this text, I also need to look up Y...

On question-answering tasks that require looking things up, letting the model "think while it searches" through a simple Wikipedia interface clearly cut down on how often it made things up.
The skeleton of almost every agent today is this **think → act → observe** loop.

> 🧪 **Try it**: run [`react_agent.py`](./react_agent.py). Write a ReAct agent from scratch and watch how it searches the repo, uses a calculator,
> corrects itself after guessing the wrong file name, and finally gives an answer with its source.

> 💡 Look ahead: → [The ReAct loop](#4-the-react-loop)

---

## Act 4 — 2023: Giving the Model a Tool Manual

In ReAct, an "action" was originally a command that the model wrote out in a specific format inside its text, which the program then parsed. That broke easily: if the model got a single bracket wrong, the program couldn't understand it.

In February 2023, researchers at Meta and Pompeu Fabra University published **Toolformer**:
the model teaches itself when to call tools such as a calculator, a search engine or a translator, and with what arguments.

On June 13 of the same year, OpenAI added **function calling** to its API:
developers describe in JSON Schema which functions exist and what each parameter is; when the model needs one, it outputs a piece of structured JSON saying "which function I want to call, and with what arguments."
**The model only proposes the call; the developer's program does the actual running.** Soon Anthropic, Google and other providers offered similar "tool calling" features,
and it became the standard interface for agents.

> 🧪 **Try it**: run [`tools.py`](./tools.py). See what the tool manual the model sees looks like, and how the program runs tool calls safely,
> including blocking malicious code hidden in an "arithmetic expression."

> 💡 Look ahead: → [Tool calling](#2-tool-calling), [JSON Schema](#3-json-schema)

---

## Act 5 — 2023: An Exam Made of Real GitHub Issues

Just how capable are agents? In October 2023, researchers at Princeton University released **SWE-bench**:
**2,294 real GitHub issues** collected from 12 popular open-source Python projects. The AI has to change the code to fix them,
and then the projects' own tests check whether each fix is right.

The results were sobering: the best model at the time, Claude 2, solved only **1.96%**.
A year later, in October 2024, the upgraded Claude 3.5 Sonnet could already solve 49% on a human-verified subset (SWE-bench Verified).
Half of the progress came from the models themselves; the other half came from everything around the model: how the tools are designed, how feedback is given, what happens when something goes wrong (the subject of Chapter 7).

---

## Act 6 — 2024–2025: Agents Move into Computers and Terminals

- **October 22, 2024**: Anthropic released **Computer Use** as a public beta: Claude can look at screenshots, move the mouse, click and type, operating a computer the way a person does;
- **December 2024**: Anthropic published *Building effective agents*. Summing up their experience working with dozens of teams, their advice was surprisingly plain:
  the most successful systems usually don't rely on complex frameworks, but are built from simple, composable patterns; **start with the simplest solution, and add complexity only when it is truly needed**.
  The article also distinguishes two kinds of systems: **workflows**, where code lays out the flow in advance, and **agents**, where the model decides the flow itself;
- **February 24, 2025**: Anthropic released a research preview of **Claude Code**: a coding agent that runs in the terminal and can read and write files, run commands,
  and debug on its own until the task is done.

From ELIZA in 1966, which could only "talk," to today's agents that can "do" things on your computer, nearly sixty years have passed. Yet the core loop is simple enough to write in a few dozen lines of code.

> 💡 Look ahead: → [Agents vs. workflows](#8-agents-vs-workflows)

---

## Epilogue — Model, Tools, Loop

The three labs in this chapter match the three parts of an agent:
1. **The model**: does the thinking and decides the next step (`common/llm.py`);
2. **The tools**: the model's "hands and eyes," actually run by the program (`tools.py`);
3. **The loop**: turns the model's decisions into actions and feeds the results back in, until the job is done (`react_agent.py`).

But once an agent can read and write files and run commands, new problems appear: will it delete something it shouldn't? What do you do as the context keeps getting longer?
When it says "fixed it," how do you make sure it really is fixed? The next chapter is about the whole layer wrapped around the agent: the **harness**.

---

## Labs

```bash
python eliza_1966.py                  # 1966: mini ELIZA (use --interactive to chat with it yourself)
python tools.py                       # the tool manual and safe execution
python react_agent.py                 # a ReAct agent from scratch (uses the mock model by default)

# Switch to a real model and ask anything about this repo (needs an API key)
python react_agent.py --provider claude --task "How many hidden layers does mlp_moons.py in Chapter 3 use? How many neurons does each layer have?"
python react_agent.py --provider deepseek --task "How many chapters does this repo have? What is each chapter's title?"
```

| Lab | What you should see | Think about it (also at the end of each script) |
|---|---|---|
| `eliza_1966.py` | Every reply can be traced back to a hand-written rule; "I am telling you, it was a movie that scared me." gets forced into a strange sentence | What is the most fundamental difference between ELIZA and an agent? |
| `tools.py` | The JSON Schema the model sees; an "arithmetic expression" hiding code, a `../` path and a misspelled argument are all blocked | Why should error messages be passed back to the model? |
| `react_agent.py` | The task is done in 4 steps: wrong file guess → switches to search → uses the calculator → answer with a source (10.9 percentage points) | What kind of task is worth using an agent for? |

About real-model mode: the agent loop calls real models through `common/llm.py`. I tested this path against a mock API server, but have not run it with a real API key.

---

## Concepts

### 1. Agent

The classic textbook definition: a system that can **perceive its environment** and **take actions** to achieve a goal.
In the era of large language models, "agent" usually means **an LLM + tools + a loop**, where **the model itself decides** what to do next and when to stop.

📖 First appears in: [Act 1](#act-1--19661972-stanford-research-institute-a-robot-that-thinks), [Act 3](#act-3--2022-thinking-while-doing)

### 2. Tool Calling

Also called function calling. Here's how it goes:
1. You attach a list of tools to the request (names, descriptions, parameter formats);
2. If the model thinks it needs one, its reply contains a structured call request: which tool, with what arguments;
3. **Your program** runs the tool and sends the result back to the model as a new message;
4. The model continues its answer based on the result, or calls more tools.

Remember: **the model never runs anything itself**. The power to execute always stays with your program, and that's where every safety measure is anchored.

📖 First appears in: [Act 4](#act-4--2023-giving-the-model-a-tool-manual)　🧪 `tools.py`

### 3. JSON Schema

A standard format for describing "what a piece of JSON data should look like": which fields it has, what type each field is, and which ones are required.
A tool manual uses it to tell the model the format of the arguments, and the program can use it to check whether the model's arguments are right (`check_args` in `tools.py`).

**The description text matters a lot**: the model relies entirely on it to decide when to use the tool and how to fill in the arguments. The more specific it is, the less likely the model is to misuse the tool.

🧪 `tools.py`

### 4. The ReAct Loop

```
while not done and under the step limit:
    reply = model(the full history)
    add reply to the history
    if reply has no tool calls:
        return reply (the final answer)
    for each tool call:
        result = run the tool
        add result to the history
```
Because the API is stateless (Chapter 5), every round has to send the full history (including every tool result) to the model again.

📖 First appears in: [Act 3](#act-3--2022-thinking-while-doing)　🧪 `react_agent.py`

### 5. Tool Results and Error Handling

When a tool fails, **don't let the program crash; return a clear error message to the model as the tool result** (and mark it as an error).
A good error message tells the model what went wrong and how it might fix it, for example "file not found; try list_dir to find it first."
The agent in Lab 3 switched from "guessing the file name" to "searching" precisely after reading a message like that.

🧪 `react_agent.py`

### 6. Stopping Conditions

An agent has to know when to stop:
- **Normal finish**: the model stops calling tools and gives its answer directly;
- **Step limit**: keeps it from getting stuck in an endless loop, or trying forever at a task it can't do;
- **Budget limit**: stop when the tokens or the money run out (Chapter 7);
- **Human interrupt**: the user can call a halt at any time.

🧪 `react_agent.py --max-steps 2`

### 7. Planning

Breaking a big goal into a series of small steps that can be carried out. Shakey used a symbolic planner like STRIPS to derive the steps from "state + goal + available actions";
today's LLM agents plan in language, usually in one of two styles:
- **Plan first, then execute**: write out the full plan, then work through it step by step;
- **Think while doing** (ReAct): decide each next step based on the latest observation.

In practice the two are often combined: start with a rough plan, then adjust it as you go.

📖 First appears in: [Act 1](#act-1--19661972-stanford-research-institute-a-robot-that-thinks)

### 8. Agents vs. Workflows

In *Building effective agents*, Anthropic draws a very useful distinction:
- **Workflow**: LLMs and tools run along **a path written into the code in advance**, for example "translate, then summarize, then check";
- **Agent**: **the model itself decides** which tools to call, how many steps to take, and when to stop.

Workflows are more controllable, cheaper and easier to debug; agents are more flexible and suit open-ended tasks whose steps can't be known in advance.
Rule of thumb: **if a workflow can solve it, don't use an agent**. Chapter 8 covers several common workflow patterns and multi-agent patterns.

📖 First appears in: [Act 6](#act-6--20242025-agents-move-into-computers-and-terminals)

### 9. The ELIZA Effect

People can't help believing that a computer program "understands" them, and may even develop feelings for it, however simple the program actually is.
Today's large language models talk far more naturally, so the effect is much stronger. When you use AI, remember to tell apart "it sounds like it understood" and "it actually got it right."
That is also why an agent's results need to be checked (the "feedback loop" in Chapter 7).

📖 First appears in: [Prologue](#prologue--1966-mit-please-leave-the-room)　🧪 `eliza_1966.py`

---

## References

- Wikipedia, [ELIZA](https://en.wikipedia.org/wiki/ELIZA); J. Weizenbaum, *ELIZA—A Computer Program for the Study of Natural Language Communication between Man and Machine*, *Communications of the ACM*, 1966; *Computer Power and Human Reason*, 1976
- Computer History Museum, [Shakey the robot](https://www.computerhistory.org/revolution/artificial-intelligence-robotics/13/289); IEEE, [Milestones: Shakey, the World's First Mobile, Intelligent Robot, 1972](https://ieeemilestones.ethw.org/Milestones:Shakey:_The_World%E2%80%99s_First_Mobile,_Intelligent_Robot,_1972)
- Wikipedia, [SHRDLU](https://en.wikipedia.org/wiki/SHRDLU)
- S. Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022
- T. Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, 2023
- OpenAI, [Function calling and other API updates](https://openai.com/index/function-calling-and-other-api-updates/), 2023-06-13
- C. E. Jimenez et al., [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770), 2023
- Anthropic, *Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku*, 2024-10-22
- Anthropic, [Building effective agents](https://www.anthropic.com/research/building-effective-agents), 2024-12
- Anthropic, [Claude Code release notes](https://docs.anthropic.com/en/release-notes/claude-code)

---

⬅️ [Chapter 5 · Large Language Models and APIs](../05-llm-and-api/) | [Contents](../README.md) | ➡️ [Chapter 7 · Harness Engineering](../07-harness/)
