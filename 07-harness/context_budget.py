"""
Lab 2: Managing context: what to keep, what to drop, and how
============================================================

Every step an agent takes makes the history a bit longer (Chapter 5: every request resends the whole history).
After 40 steps, a single request can be tens of thousands of tokens: expensive, slow, and it may hit the limit
of the context window.

Here we simulate a 40-step agent task (reading files, searching, running tests, ...) and compare five strategies
for managing the context:
  A. Keep everything
  B. Clear old tool results at every step (keep only the latest 5)
  C. Clear in batches once enough has piled up (above 8000 tokens, clear most old results at once)
  D. Summarize (above 8000 tokens, compress the old history into a summary)
  E. Notes + batch clearing (the agent writes key facts into a small notes file, combined with batch clearing)

Four measures:
  · how big the context is            · how many tokens are sent in total
  · how much hits the "prompt cache"  · whether three key facts are still remembered at the end

Run: python context_budget.py
"""

import random

STEPS = 40
rng = random.Random(2025)

# ---------------------------------------------------------------- generate one simulated agent task
# each step: the model says something (about 80 tokens) + one tool result (size depends on the tool)
TOOL_SIZES = {"read_file": (800, 2500), "search": (200, 600), "run_tests": (300, 900), "list_dir": (50, 150)}
FACTS = {  # the step at which each key fact appears, and the kind of message it appears in
    3: ("User requirement: every function needs type annotations", "user"),
    11: ("config.yaml line 7: tests must use Python 3.11", "tool"),
    24: ("run_tests output: test_parse fails only on Windows", "tool"),
}
history = [{"id": "task", "kind": "user", "tokens": 120, "fact": None}]
for step in range(1, STEPS + 1):
    history.append({"id": f"a{step}", "kind": "assistant", "tokens": 80, "fact": None})
    tool = rng.choice(list(TOOL_SIZES))
    fact = FACTS.get(step)
    if fact and fact[1] == "user":
        history.append({"id": f"u{step}", "kind": "user", "tokens": 60, "fact": fact[0]})
    history.append({"id": f"t{step}", "kind": "tool", "tokens": rng.randint(*TOOL_SIZES[tool]),
                    "fact": fact[0] if fact and fact[1] == "tool" else None})

SYSTEM = [{"id": "system+tools", "kind": "system", "tokens": 1500, "fact": None}]
PLACEHOLDER = 15        # a cleared tool result is replaced by a one-line "cleared" note
SUMMARY = 600           # size of the summary
THRESHOLD = 8000        # the trigger for batch clearing / summarizing
CACHE_PRICE = 0.1       # cached tokens cost about 1/10 of the normal price (taking Claude as the example)


def upto(step):
    """All messages produced up to and including step `step`."""
    n = next(i for i, m in enumerate(history) if m["id"] == f"t{step}") + 1
    return history[:n]


def size(view):
    return sum(m["tokens"] for m in view)


# ---------------------------------------------------------------- five strategies: each returns what is sent at this step
def keep_all(step, state):
    return SYSTEM + upto(step)


def clear_every_step(step, state, keep=5):
    msgs = upto(step)
    tools = [m for m in msgs if m["kind"] == "tool"]
    old = {m["id"] for m in tools[:-keep]}
    return SYSTEM + [{**m, "id": m["id"] + "~", "tokens": PLACEHOLDER, "fact": None} if m["id"] in old else m
                     for m in msgs]


def clear_in_batches(step, state, keep=3):
    msgs = upto(step)
    cleared = state.setdefault("cleared", set())
    view = lambda: [{**m, "id": m["id"] + "~", "tokens": PLACEHOLDER, "fact": None} if m["id"] in cleared else m
                    for m in msgs]
    if size(SYSTEM + view()) > THRESHOLD:                  # clear only past the threshold, then a big batch at once
        tools = [m for m in msgs if m["kind"] == "tool"]
        cleared.update(m["id"] for m in tools[:-keep])
    return SYSTEM + view() + state.get("notes", [])


def summarize(step, state, keep_msgs=6):
    msgs = upto(step)
    start = state.get("summary_upto", 0)
    view = state.get("summary", []) + msgs[start:]
    if size(SYSTEM + view) > THRESHOLD:
        cut = len(msgs) - keep_msgs
        covered = msgs[:cut]
        # a summary usually keeps the user's requirements, but easily loses details buried in long tool results
        kept = [m["fact"] for m in covered if m["fact"] and m["kind"] == "user"]
        old_kept = [f for s in state.get("summary", []) for f in s.get("facts", [])]
        state["summary"] = [{"id": f"summary@{step}", "kind": "summary", "tokens": SUMMARY,
                             "fact": None, "facts": old_kept + kept}]
        state["summary_upto"] = cut
        view = state["summary"] + msgs[cut:]
    return SYSTEM + view


def notes_and_batches(step, state):
    # each time the agent sees a key fact, it appends a line to its notes
    # (the notes always go last and are never cleared)
    for m in upto(step):
        if m["fact"] and m["fact"] not in state.setdefault("noted", set()):
            state["noted"].add(m["fact"])
            state.setdefault("notes", []).append(
                {"id": f"note:{m['id']}", "kind": "note", "tokens": 30, "fact": m["fact"]})
    return clear_in_batches(step, state)


STRATEGIES = [("A Keep all", keep_all), ("B Clear every step", clear_every_step),
              ("C Clear in batches", clear_in_batches), ("D Summarize", summarize),
              ("E Notes + batches", notes_and_batches)]


def facts_in(view):
    found = set()
    for m in view:
        if m.get("fact"):
            found.add(m["fact"])
        found.update(m.get("facts", []))
    return found


def common_prefix_tokens(prev, cur):
    """The prompt cache only matches a "prefix": only the part that is identical from the very start can hit."""
    total = 0
    for a, b in zip(prev, cur):
        if a["id"] != b["id"]:
            break
        total += b["tokens"]
    return total


print(f"Simulating a {STEPS}-step agent task. Size of the context sent at each step (in thousands of tokens):\n")
print("   Strategy           " + "".join(f"{s:>6}" for s in range(5, STEPS + 1, 5)))
results = []
for name, fn in STRATEGIES:
    state, prev = {}, []
    sizes, sent, cached = [], 0, 0
    for step in range(1, STEPS + 1):
        view = fn(step, state)
        s = size(view)
        sizes.append(s)
        sent += s
        cached += common_prefix_tokens(prev, view)
        prev = view
    remembered = facts_in(view)
    results.append((name, sizes, sent, cached, remembered))
    pad = name + " " * (19 - sum(2 if ord(c) > 127 else 1 for c in name))
    print("   " + pad + "".join(f"{sizes[s - 1] / 1000:6.1f}" for s in range(5, STEPS + 1, 5)))

print("\n   Strategy           | Max context | Total sent | Cache hits | Est. cost | Key facts kept")
base_cost = None
for name, sizes, sent, cached, remembered in results:
    cost = (sent - cached) + cached * CACHE_PRICE
    base_cost = base_cost or cost
    pad = name + " " * (19 - sum(2 if ord(c) > 127 else 1 for c in name))
    print(f"   {pad}| {max(sizes) / 1000:10.1f}k | {sent / 1000:9.0f}k | {cached / sent:10.0%} | "
          f"{cost / base_cost:9.0%} |      {len(remembered)}/3")

print("""   (Estimated cost is relative to "Keep all" = 100%; cached tokens are billed at 1/10 of the normal price.)

Reading the results:
  * A Keep all: no information is lost. Because the history is only ever appended to, 95% of it hits the cache,
    so the cost is actually not as high as you might think.
    But the context keeps growing, sooner or later it hits the window limit, and every step gets slower.
  * B Clear every step: the context gets much smaller, but every clearing changes the middle of the history,
    and nothing after that point can hit the cache. The cost ends up even higher than "Keep all" (148%).
    Saving tokens is not the same as saving money.
  * C Clear in batches: wait until enough has piled up, then clear it all at once. Only a few steps break the
    cache, so you get both "smaller" and "cheaper."
  * D Summarize: the smallest context and the lowest cost, but a summary easily loses details buried in tool
    results (two facts were lost here).
  * E Notes + batches: key facts go into the notes, so they aren't lost when tool results are cleared.
    This is exactly the "structured note-taking" approach Anthropic recommends in
    "Effective context engineering for AI agents."

One more constraint that is easy to miss: some models (such as Claude Opus 5.5) require you to send their earlier
replies back exactly as they were. If you change the history, their earlier "thinking" becomes invalid and is
dropped, or the request may even fail with an error.
That's why real harnesses prefer to "only append," or to use the server-side context management that providers
offer, instead of freely editing the history on the client side.

Think about it:
  1. If you raise THRESHOLD to 20000, how do the costs of C and E change?
  2. If D's summary were written better (also recording the key facts from tool results), how would it still
     differ from E?
  3. When you use tools like Claude Code or Cursor, have you noticed when they "compact the context"?""")
