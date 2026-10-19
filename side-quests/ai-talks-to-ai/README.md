# Side Quest 1 — How Do Two AIs "Talk"?

> How it started: there's a video on Bilibili (a Chinese video site) about a casual chat between Claude and DeepSeek. The two AIs take turns, back and forth, like two people chatting.
> Which raises a question: **how can one AI talk to another AI?** The technology behind it happens to be the perfect way into agents and multi-agent collaboration.

---

## The One-Sentence Answer

**Neither AI actually knows the other one exists.** In the middle sits a "relay" program:
it sends what A said to B as a "user message", then sends B's answer back to A as a "user message".
That relay program is the simplest possible **harness**.

```
        ┌────────────── relay program (harness) ───────────────┐
        │                                                      │
 Claude ◀── "user" says: <what DeepSeek just said>             │
        ──▶ reply ──────────────┐                              │
        │                       ▼                              │
        │          "user" says: <what Claude just said> ───────▶ DeepSeek
        │                       ┌─────────────────────── reply ◀──
        └───────────────────────┴──────────────────────────────┘
```

Below, we break it down into five key technical points.

---

## 1. The API Is "Stateless": You Resend the Whole Conversation Every Time

When you chat with an AI on a web page, it feels like it "remembers" what was said earlier. But at the API level, **the model itself has no memory**.
On every call, the program has to package up **the entire conversation history** and send it over. The model's only job is to generate the "next" reply:

```python
messages = [
    {"role": "user",      "content": "Hi, my name is Xiao Ming"},
    {"role": "assistant", "content": "Hi, Xiao Ming!"},
    {"role": "user",      "content": "What's my name?"},     # ← the model has to answer this one
]
```

There are three roles:
- **system**: the model's "persona" and rules (for example, "You are a debater. Keep each reply to 3 sentences or fewer");
- **user**: what the other side said;
- **assistant**: what the model itself said earlier.

So the "memory" is really a list that **the program** maintains. Once that clicks, everything else falls into place.

## 2. Role Flipping: The Same Line Plays a Different Role on Each Side

The relay program has to keep **a separate** conversation history for each AI. The key step is this:

> A line that A says is `assistant` ("something I said") in A's history, and `user` ("something the other side said") in B's history.

```
History as Claude sees it                History as DeepSeek sees it
─────────────────────────                ───────────────────────────
user:      Host: kick things off
assistant: Claude, line 1        ──▶     user:      Claude, line 1
user:      DeepSeek, line 1      ◀──     assistant: DeepSeek, line 1
assistant: Claude, line 2        ──▶     user:      Claude, line 2
...                                      ...
```

In [`relay.py`](./relay.py), this takes just two lines:

```python
me["history"].append({"role": "assistant", "content": text})    # to myself: something I said
other["history"].append({"role": "user", "content": text})      # to the other side: something the user said
```

Run `python relay.py --show-history`. At the end it prints both sides' histories together, so you can see the flip with your own eyes.

## 3. System Prompt + Stop Condition: Giving the Conversation Shape and an Ending

Relaying messages isn't enough on its own. There are at least three more problems to solve:

| Problem | Fix |
|---|---|
| Both AIs are very polite, so "thanks for sharing" / "thank you too" can loop forever | Agree in the system prompt: when you feel you're done, output `[END]`; the program stops when it sees it |
| The program might run forever, getting more and more expensive | Set a maximum number of turns (`--turns`) |
| The longer the conversation, the longer the history you resend each time, so cost and latency keep climbing | Limit the length of each reply; longer conversations need the history "compressed" (Chapter 7, on harnesses, covers this) |

One more safety note: **one AI's output becomes another AI's input directly**.
If A's reply slips in something like "Ignore all your previous instructions…", B might be led astray.
This is called **prompt injection**, and it's a problem every multi-agent system has to take seriously.

## 4. How to Plug In Models from Different Providers (Claude / DeepSeek / GPT)

The good news: the major providers' APIs all look pretty similar:

| Model | Which SDK | Key settings |
|---|---|---|
| Claude | `anthropic` | `ANTHROPIC_API_KEY`; system is a separate parameter |
| DeepSeek | `openai` (DeepSeek's API is OpenAI-compatible) | `base_url="https://api.deepseek.com"` + `DEEPSEEK_API_KEY` |
| GPT | `openai` | `OPENAI_API_KEY` |

DeepSeek and GPT use the same code; **only the `base_url` and the key change**.
`relay.py` wraps all of them so they look the same: `reply(system, messages) -> str`.
That way the relay loop doesn't care at all who's on the other end. To add a new model, you just write one more `make_xxx` function.

> ⚠️ Model names change: this was written in Oct 2026, and the defaults are `claude-opus-5-5`, `deepseek-v4-flash` and `gpt-5.5`.
> If you get a "model not found" error, first check each provider's official docs for the latest model names, then pass them with `--model-a` / `--model-b`.

## 5. From "Small Talk" to "Dividing Up the Work": Multi-Agent Collaboration

If two AIs can chat, they can work together. Small talk is just the simplest pattern of collaboration. Here are some other common ones:

### 5.1 Wrapping Another AI as a "Tool"

Large models all support **tool calls** (tool use / function calling): you tell the model "there's a tool called `ask_deepseek`, and its parameter is a question",
and when the model needs it, it says "I want to call `ask_deepseek` with the argument …". Your program then actually calls DeepSeek and hands the result back.

```
User ──▶ Claude (the lead)
            │  "Have DeepSeek work through this math problem first"
            ├──▶ tool ask_deepseek("…") ──▶ DeepSeek ──▶ result
            │  "Have GPT polish this English text"
            ├──▶ tool ask_gpt("…") ──▶ GPT ──▶ result
            ▼
        Puts it all together and answers the user
```

Now they're no longer chatting as equals: it's **one lead and a few specialists**.

### 5.2 Common Collaboration Patterns

| Pattern | How it works | Good for |
|---|---|---|
| **Orchestrator–worker** | The lead agent splits up the task, several subagents work on it in parallel, and the lead agent combines the results | Research, and tasks that need many directions explored at once |
| **Routing** | First figure out what kind of task it is, then hand it to the model that's best at it | Some models are great at code; others are cheap and fast |
| **Writer–reviewer** | One writes, the other picks it apart, and they revise until it passes | Writing code, writing articles |
| **Debate** | Several models each give a view and rebut one another, and a judge sums up at the end | Questions that need judging from several angles |

A real example: Anthropic's research feature uses the orchestrator–worker pattern. A lead agent makes the plan and sends out several subagents to search in parallel.
They reported much better results than a single agent, but it uses about 15 times as many tokens as a normal chat. **Dividing up the work pays off, but it also costs.**

### 5.3 Standard Protocols: MCP and A2A

As agents multiply, everyone needs a common "plug standard":
- **MCP** (Model Context Protocol; Anthropic, 2024): defines **how models connect to tools and data**. Write a tool as an MCP server, and any AI that supports MCP can use it.
- **A2A** (Agent2Agent Protocol; Google, 2025, now under the Linux Foundation): defines **how agents discover, talk to and work with each other**, even when they come from different providers.

`relay.py` does the relaying by hand; MCP and A2A standardize it.
To see MCP in action, read on to Side Quest 2: [getting the ChatGPT and Claude desktop apps to pass messages to each other](../app-bridge/).

---

## Try It

```bash
# 1. No keys needed: watch the relay at work with two mock AIs
python relay.py --show-history

# 2. Have two real AIs chat (needs the matching API keys; costs a little money)
pip install anthropic openai
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
python relay.py --a claude --b deepseek --turns 6 --topic "whether AI can dream"

# 3. Switch to GPT and DeepSeek, or let two Claudes chat with each other
python relay.py --a gpt --b deepseek
python relay.py --a claude --b claude
```

> Note: mock mode has been tested and runs fine. Real-model mode needs your own API keys, and the author has not run it with real keys.
> If you run into problems, please open an issue.

**Exercises:**
1. Change `system_prompt()` so that A argues "for" and B argues "against", turning it into a debate.
2. Add a third-party "judge": when the conversation ends, send the full transcript to a third model and have it decide who won.
3. Add some stats: in each turn, how many messages, and how many characters, are sent to the API? Get a feel for "the context keeps getting longer".
4. (Advanced) Wrap DeepSeek as one of Claude's tools (`ask_deepseek`) and build the diagram from 5.1. That's exactly what Chapters 8 and 9 do.

---

⬅️ [Contents](../../README.md) | 📖 Review: [Chapter 1 · Where Machine Learning Came From](../../01-ml-origins/) | ➡️ [Side Quest 2 · App Bridge](../app-bridge/)
