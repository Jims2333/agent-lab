# Chapter 9 — Cross-Model Collaboration: Getting Claude, GPT and DeepSeek to Work Together

> This is the last chapter. Remember how, in Side Quest 1, Claude and DeepSeek chatted through a Python program sitting between them?
> That program was just "passing notes by hand." The question this chapter answers is: when there are more and more participants, from different companies,
> who have never met, what do they rely on to work together? The answer is plain: **protocols**.
> People already found this answer once, more than fifty years ago, when they built the internet.

**How to read this chapter:** Read the stories first, then run the labs, and save the [Concepts](#concepts) for last. The labs use a mock model by default, so no API key is needed.

| When | What happened | Lab / concept |
|---|---|---|
| Apr 1969 | A graduate student writes RFC 1 in a bathroom | Protocols |
| May 1974 | Cerf and Kahn: a network of networks | Adapter layer |
| Mar 1989 | Berners-Lee's proposal: "Vague but exciting" | Addresses and discovery |
| Jun 2016 | The Language Server Protocol: turning M×N into M+N | 🧪 `router.py` |
| Nov 2024 | MCP: a "USB-C port" for AI applications | MCP |
| Apr 2025 | A2A: business cards and conversations between agents | 🧪 `a2a_mini.py` |
| Mar 2026 | A2A 1.0 is released | 🧪 `cross_check.py`, `capstone.py` |

---

## Prologue — Apr 7, 1969: A Request for Comments Written in a Bathroom

In the spring of 1969, the ARPANET, funded by the US Department of Defense, was getting ready to connect the computers of a few universities.
Someone was in charge of the lines and the switching equipment, but **how two computers should "talk" once they were connected** was something nobody had specified.
The job fell to a group of graduate students at several universities, who called themselves the "Network Working Group."

**Steve Crocker**, a graduate student at the University of California, Los Angeles, was the one who wrote down everyone's discussions.
As he recalled in *The New York Times* in 2009, he was sharing a house with friends at the time, and he wrote the first set of notes late at night in the bathroom, so as not to wake his sleeping housemates.
He was also worried: would something written by a bunch of graduate students seem too presumptuous, and offend the eminent professors?
So he gave the notes as modest a name as he could: **"Request for Comments" (RFC)**.

On April 7, 1969, RFC 1, *Host Software*, was sent out.
The modest name is still in use today: the internet's specifications are still published as RFCs, and the numbering passed nine thousand long ago.
Later you'll see that a detail in the A2A specification, such as "at which address the business card goes," also cites an RFC.

> 💡 Look ahead: → [Protocols](#1-protocols)

---

## Act 1 — May 1974: A Network of Networks

Once the ARPANET was up and running, a new problem arrived: the world had more than one network. Satellite networks, packet radio networks...
Each network had a different packet size, failed in different ways and ran at a different speed. How could **different networks** be connected to each other?

In May 1974, **Vinton Cerf** and **Robert Kahn** published
*A Protocol for Packet Network Intercommunication* in *IEEE Transactions on Communications*. Their idea: **don't require every network to be the same**;
just require everyone to follow the same set of rules at the boundary (the gateways), and let each network work however it likes inside.
That set of rules was TCP, which was later split into the two layers familiar today, TCP and IP.

January 1, 1983 was the ARPANET's "flag day": every host connected to the network had to switch from the old protocol to TCP/IP.
Many people see that day as the internet's real birthday.

This is exactly what `common/llm.py` from Chapter 5 does: Claude's, GPT's and DeepSeek's APIs all differ,
and instead of changing them, we add an **adapter layer** at the boundary, so the programs above it only deal with one unified format.

> 💡 Look ahead: → [The adapter layer and the M×N problem](#3-the-adapter-layer-and-the-mn-problem)

---

## Act 2 — Mar 1989, CERN: Vague but Exciting

In March 1989, **Tim Berners-Lee** at CERN, the European Organization for Nuclear Research, handed his boss **Mike Sendall**
a proposal titled *Information Management: A Proposal*, its cover filled with interconnected circles and boxes.
Sendall wrote a comment on it that later became famous: **"Vague but exciting"**.

Next, Berners-Lee wrote the first browser and server, and laid down the concepts of HTTP, HTML and the URL.
Of these, the URL is the plainest and the most crucial: **every resource has an address, and anyone with the address can find it**.
On April 30, 1993, CERN announced that it was giving up all intellectual property rights to this software: anyone could use, modify and distribute it.
Openness let the World Wide Web spread across the world quickly.

More than thirty years later, the A2A protocol gives every agent a fixed address for its "business card," using the same idea:
**people have to be able to find you before any collaboration can begin.**

> 💡 Look ahead: → [A2A and the Agent Card](#5-a2a-and-the-agent-card)

---

## Act 3 — Jun 2016: Turning M×N into M+N

There are many code editors (VS Code, Vim, Emacs...) and many programming languages (Python, Java, Go...).
To make every editor support "go to definition" and "autocomplete" for every language, you'd have to write M×N plugins.

On June 27, 2016, at the DevNation conference in San Francisco, Microsoft announced that, together with Red Hat and Codenvy, it would push forward the **Language Server Protocol (LSP)**:
each language only has to implement one "language server," each editor only has to implement one "client," and the two sides communicate over a JSON-based protocol.
**M×N plugins became M+N implementations.**

Eight years later, someone carried the same idea over to AI.

> 🧪 **Try it**: run [`router.py`](./router.py). A product is connected to three vendors' models at once; the router hands out requests by type,
> and when one vendor goes down, it automatically switches to the next. All of this relies on the same adapter layer.

> 💡 Look ahead: → [The adapter layer and the M×N problem](#3-the-adapter-layer-and-the-mn-problem), [Routing and failover](#7-routing-failover-and-cascades)

---

## Act 4 — Nov 2024: A USB-C Port for AI

Every AI application wants to connect to all kinds of tools and data: files, databases, calendars, code repositories...
Writing the integration code once for every application and every tool is, again, an M×N problem.

On November 25, 2024, Anthropic released the **Model Context Protocol (MCP)**,
designed by the engineers David Soria Parra and Justin Spahr-Summers. It explicitly borrows from LSP:
tools and data sources implement an "MCP server," AI applications implement an "MCP client," and the two sides communicate over JSON-RPC 2.0.
The official MCP documentation offers an analogy: **"Think of MCP like a USB-C port for AI applications."**

What really turned MCP into an industry standard was a competitor joining in. In March 2025, OpenAI's CEO, **Sam Altman**, said publicly:
"people love MCP and we are excited to add support across our products".
On December 9, 2025, Anthropic donated MCP to the newly founded **Agentic AI Foundation** under the Linux Foundation,
a foundation started jointly by Anthropic, Block and OpenAI.

In Side Quest 2, the "mailbox" that lets the ChatGPT and Claude desktop apps pass messages to each other is an MCP server.

> 💡 Look ahead: → [MCP](#4-mcp)

---

## Act 5 — Apr 2025: Business Cards for Agents

MCP solves "how a model uses tools." But what if the other side isn't a tool, but **another agent**?
It has its own judgment, it may take a long time, it may need more information from you partway through, and behind it there may be another company and another vendor's model.

On April 9, 2025, at the Google Cloud Next conference, Google released the **A2A (Agent2Agent) protocol**,
with support from more than fifty partners at launch. Google positioned it as a complement to MCP: MCP connects agents to tools, A2A connects agents to agents.
On June 23, 2025, the Linux Foundation announced the A2A project; Google handed over the protocol specification, the SDKs and the developer tools,
and Amazon Web Services, Cisco, Microsoft, Salesforce, SAP, ServiceNow and other companies joined too.

A2A's core design looks a lot like a business exchange:
- every agent puts an **Agent Card** (a business card) at the fixed address `/.well-known/agent-card.json`, saying who it is, what it can do and how to reach it;
- the side that wants to work together fetches the card first, then, at the address on the card, calls `SendMessage` over JSON-RPC to send the task over;
- the other side returns a **Task**, which holds the task's status and its outputs (artifacts).

On March 12, 2026, A2A released version 1.0: method names were all changed to the `SendMessage` style,
and clients must state the protocol version they use in every request. (Digitally signing Agent Cards to prevent forgery has been possible since version 0.3, in July 2025.)

> 🧪 **Try it**: run [`a2a_mini.py`](./a2a_mini.py). It implements a minimal subset of A2A 1.0 with the standard library
> and starts two agents on your machine: one backed by a large language model, the other by nothing but a piece of code. See how agents that "have never met" work together.

> 🧪 **Try it**: run [`cross_check.py`](./cross_check.py) to see when having models from different vendors check each other's answers helps, and when it doesn't.
> Finally, run the capstone [`capstone.py`](./capstone.py): three vendors' models share the work of making a study card for this repo.

> 💡 Look ahead: → [A2A and the Agent Card](#5-a2a-and-the-agent-card), [How MCP and A2A divide the work](#6-how-mcp-and-a2a-divide-the-work),
> [Model diversity](#8-model-diversity-and-error-correlation), [Security in cross-organization collaboration](#10-security-in-cross-organization-collaboration)

---

## Epilogue — Protocols Let Strangers Work Together

| Protocol | Era | Who it lets talk to whom | Key design |
|---|---|---|---|
| The RFC series | 1969– | The people designing networks | Open discussion; anyone can comment |
| TCP/IP | 1974– | Different networks | Agree only on the rules at the boundary, not on how each side works inside |
| HTTP / URL | 1989– | Browsers and websites | Everything has an address; open and free |
| LSP | 2016– | Editors and programming languages | M×N becomes M+N |
| MCP | 2024– | AI applications and tools, data | A "USB-C port" for models |
| A2A | 2025– | Agents and agents | Business card + task; it doesn't matter who is behind the other side |

Not one of these protocols succeeded by "making everyone the same." They only specify **how to talk at the boundary**, and leave each side free inside.
That is also the right way to approach working with several vendors: there's no need to bet on which model is best; instead, build a system that lets them each play to their strengths, back each other up and check each other.

And with that, you've come all the way from a single artificial neuron in 1943 to a cross-vendor team of agents. The end.

---

## Labs

```bash
python router.py          # routing and failover (uses the mock model by default)
python cross_check.py     # cross-vendor checking + a thought experiment on error correlation
python a2a_mini.py        # mini A2A: Agent Cards, SendMessage, tasks, error handling
python capstone.py        # capstone: three vendors share the work of making a study card

# Vendors whose keys you've set use real models; the rest keep using the mock model
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, DEEPSEEK_API_KEY; one or two are enough
python capstone.py --live
```

| Lab | What you should see | Think about it (also at the end of each script) |
|---|---|---|
| `router.py` | 8 requests sorted into 4 categories; when gpt is down, the reasoning questions automatically go to claude; following the routing table costs about a third of "send everything to claude" | How could a "cascade" replace classification: cheap first, then expensive? |
| `cross_check.py` | Asking one vendor gets 4/6, asking the same vendor three times still gets 4/6, a vote of three vendors gets 5/6; a two-vendor check flags the 2 questionable questions but misses 1 where "both are wrong"; in the thought experiment, the more correlated the errors, the less it helps to ask more vendors | Are the three models' errors highly or weakly correlated? How would you measure it? |
| `a2a_mini.py` | Fetch the Agent Cards → hand out work by skill → get a Task back; nobody can book plane tickets; a missing version header gives -32009, the old method name gives -32601, and the code injection is rejected | What if someone sends a fake card pretending to be the translator? |
| `capstone.py` | Nine researchers in parallel; the sloppy one is sent back by the reviewer to redo its work; the translation is done over A2A; at the end, the card and a bill broken down by vendor | Why make the cheap model the researcher and the expensive one the reviewer? |

About real-model mode: I tested the real-API paths of all four scripts against a mock server (to confirm they don't crash and have a fallback when the format is wrong),
but have not run them with a real API key. Each vendor's answers will differ from the mock model's script; for example, today's models may get every trick question in the cross-check right.
The prices of GPT-5.5 and DeepSeek V4 Flash in `router.py` come from third-party price comparison sites; the vendors' official websites have the final word.

---

## Concepts

### 1. Protocols

A set of rules agreed on in advance: what the messages look like, in what order they are sent, and how to say something went wrong.
With a protocol, neither side needs to know how the other works inside; as long as both follow the same rules, they can work together.
Good protocols usually specify only the "boundary," and are open and free, so anyone can implement them.

📖 First appears in: [Prologue](#prologue--apr-7-1969-a-request-for-comments-written-in-a-bathroom)

### 2. JSON-RPC

A very simple "remote call" format: a request is a JSON object with `jsonrpc: "2.0"`, an `id`, a `method` (the method name) and `params` (the parameters);
the reply contains either a `result` or an `error` (with an error code and a message).
LSP, MCP and A2A's JSON-RPC binding are all built on it, so once you've learned one, the others are easy to follow.

📖 First appears in: [Act 3](#act-3--jun-2016-turning-mn-into-mn), [Act 4](#act-4--nov-2024-a-usb-c-port-for-ai)　🧪 `a2a_mini.py`

### 3. The Adapter Layer and the M×N Problem

If M users have to connect to N providers and every pair is connected directly, you have to write M×N pieces of code.
Agree on one unified interface, have each user and each provider implement it once, and you only need M+N.
This repo's `common/llm.py` is a small adapter layer: the programs above it only know one "neutral format,"
and it takes care of translating that into the formats of Claude, GPT and DeepSeek (where the system prompt goes, how tool calls are represented, how tool results are filled back in).

📖 First appears in: [Act 1](#act-1--may-1974-a-network-of-networks), [Act 3](#act-3--jun-2016-turning-mn-into-mn)　🧪 `router.py`

### 4. MCP

The Model Context Protocol. There are three roles: the **host** (such as Claude Desktop, ChatGPT or an IDE),
the **client** inside the host (which keeps a one-to-one connection with one server), and the **server** that provides capabilities.
A server can provide tools, resources and prompts (prompt templates).
Write an MCP server once, and every AI application that supports MCP can use it. The mailbox in Side Quest 2 is an example.

📖 First appears in: [Act 4](#act-4--nov-2024-a-usb-c-port-for-ai)

### 5. A2A and the Agent Card

A2A specifies how agents discover each other and how they delegate tasks:
- **Agent Card**: the business card at `/.well-known/agent-card.json`, with the name, a description, skills, how to reach it (supportedInterfaces) and the capabilities it supports;
- **Message**: one message, made up of one or more **Parts** (text, files, structured data), with the role `ROLE_USER` or `ROLE_AGENT`;
- **Task**: one delegated job, with a state (`TASK_STATE_WORKING`, `TASK_STATE_COMPLETED`, `TASK_STATE_FAILED`, `TASK_STATE_INPUT_REQUIRED` and so on) and outputs (artifacts);
- **Bindings**: the same semantics can be carried over JSON-RPC, gRPC or HTTP+JSON.

📖 First appears in: [Act 2](#act-2--mar-1989-cern-vague-but-exciting), [Act 5](#act-5--apr-2025-business-cards-for-agents)　🧪 `a2a_mini.py`

### 6. How MCP and A2A Divide the Work

A rough but useful distinction: **MCP connects agents to tools**; on the other side is "a capability that gets called," and once the call is done, it's over.
**A2A connects agents to agents**; on the other side is "a colleague with a mind of its own": a task may take a long time, may need more information partway through, and you don't know, and don't need to know, who is behind it.
An agent can perfectly well use both protocols at once: MCP inside to call its tools, and A2A outside to work with other agents.

📖 First appears in: [Act 5](#act-5--apr-2025-business-cards-for-agents)

### 7. Routing, Failover and Cascades

**Routing**: first work out what type of request it is, then hand it to a suitable model. **Failover**: when the first choice fails (rate limits, an outage), switch to a backup automatically.
**Cascade**: let a cheap model answer first, and pass the request to a more expensive model if the answer fails a check.
Stanford's 2023 FrugalGPT paper reported that, in their experiments, combined strategies like these could match the best single model at a much lower cost.
Whichever you use, the precondition is that you have a set of evals and know "which kinds of tasks it's enough to give to whom."

📖 First appears in: [Act 3](#act-3--jun-2016-turning-mn-into-mn)　🧪 `router.py`

### 8. Model Diversity and Error Correlation

Whether asking several models, taking the majority or having them check each other reduces errors depends on whether their errors are "correlated":
ask the same model many times, and its errors are almost perfectly correlated, so asking more doesn't help; models from different vendors are trained differently, so their errors may be less correlated, and only then does asking more help.
"Accept it when two vendors agree, send it to a human when they don't" doesn't need the right answer, but the price is that cases where "both are wrong in the same way" slip through.

📖 First appears in: [Act 5](#act-5--apr-2025-business-cards-for-agents)　🧪 `cross_check.py`

### 9. Version Negotiation

Protocols get upgraded, and an old client and a new server must be able to discover "we're not speaking the same version."
A2A requires clients to send an `A2A-Version` header with every request; if the server doesn't support that version, it returns a clear error (`VersionNotSupportedError`, JSON-RPC error code -32009),
instead of muddling along and processing the request in the wrong format.

📖 First appears in: [Act 5](#act-5--apr-2025-business-cards-for-agents)　🧪 `a2a_mini.py`

### 10. Security in Cross-Organization Collaboration

Working with someone else's agent means the trust boundary gets wider:
- **The card may be fake**: A2A supports digitally signing Agent Cards, so you can verify where a card came from;
- **Everything the other side sends is data, not instructions**: never `eval()` what the other side sends, and never let its replies change your own behavior (prompt injection, Chapter 7);
- **Least privilege**: give the other side only the information needed to complete the task; don't send it your whole context, let alone your API keys;
- **Keep keys only in environment variables**: don't write them into the code, and don't commit them to the repo.

📖 First appears in: [Act 5](#act-5--apr-2025-business-cards-for-agents)　🧪 `a2a_mini.py`

### 11. The Big Picture

Tying the whole book together: the model itself is a "guess the next word" machine (Chapters 1–5);
give it tools and a loop, and it becomes an agent (Chapter 6);
put a harness on it, and it can work reliably (Chapter 7);
several agents dividing the work can do what one agent can't (Chapter 8);
with an adapter layer and protocols, agents from different vendors and different companies can work together too (Chapter 9).

📖 First appears in: [Epilogue](#epilogue--protocols-let-strangers-work-together)　🧪 `capstone.py`

---

## References

- S. Crocker, *RFC 1: Host Software*, 1969-04-07; [RFC Editor](https://www.rfc-editor.org/rfc/rfc1.html)
- S. D. Crocker, *How the Internet Got Its Rules*, The New York Times, 2009-04-06
- V. Cerf, R. Kahn, *A Protocol for Packet Network Intercommunication*, IEEE Transactions on Communications 22(5), 1974
- T. Berners-Lee, [Information Management: A Proposal](https://info.cern.ch/Proposal.html), CERN, 1989-03
- CERN, [CERN puts the World Wide Web in the public domain](https://timeline.web.cern.ch/node/794), 1993-04-30
- Microsoft / Red Hat / Codenvy, Language Server Protocol, 2016-06; [specification](https://microsoft.github.io/language-server-protocol/)
- Anthropic, [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), 2024-11-25; [MCP documentation](https://modelcontextprotocol.io)
- Anthropic, [Donating the Model Context Protocol and establishing the Agentic AI Foundation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation), 2025-12-09
- Google, [Announcing the Agent2Agent Protocol (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/), 2025-04-09; [Google Cloud donates A2A to Linux Foundation](https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/), 2025-06-23
- A2A Project, [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/) (this chapter implements version 1.0); [GitHub](https://github.com/a2aproject/A2A)
- L. Chen, M. Zaharia, J. Zou, *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, 2023; [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)

---

⬅️ [Chapter 8 · Multi-Agent Collaboration](../08-multi-agent/) | [Contents](../README.md)
