"""
Capstone: a small cross-vendor team
===================================

Put the parts from the earlier chapters together, and have the three vendors' models share a real job:
  make a "study card" for this repo: nine chapters, one row each, with the English title, the Chinese title
  and the time span of the stories.

Who does what:
  · Orchestrator (claude): splits the task into nine subtasks ......................... Chapter 8, orchestrator–worker
  · Nine researchers (deepseek), in parallel: read each chapter's README with the
    ReAct agent from Chapter 6 .......................................................... Chapter 6, agent loop + tools
      the tool can only read files inside the repo ...................................... Chapter 7, sandbox
  · Translator (gpt): translates the titles into Chinese over the A2A protocol ........ Chapter 9, protocols
  · Reviewer (claude): checks the format with a program first, sends failures back
    to be redone, then asks a model for comments ........................................ Chapter 8, writer–reviewer loop
  · A ledger the whole way: how many calls each vendor made, how many tokens,
    roughly how much money .............................................................. Chapter 7, budgets

The mock model's script deliberately includes one "sloppy" researcher, to see whether the reviewer catches it.

Run:
  python capstone.py            # mock models
  python capstone.py --live     # vendors whose keys are set use real models (calls the APIs dozens of times)
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from team import ROOT, add_live_arg, ask, fit, label, model_for, provider_for, roster
from a2a_mini import A2AServer, fetch_card, send_text, task_text
from router import PRICES
from common.llm import MockLLM

sys.path.insert(0, str(ROOT / "06-agent-loop"))
from react_agent import run_agent  # noqa: E402

CHAPTERS = sorted(p.name for p in ROOT.iterdir() if re.match(r"0\d-", p.name))
TASK = ("Make a study card for the agent-lab repo: one row per chapter, with the English title, "
        "the Chinese title and the time span of the stories.")
LINE = re.compile(r"^Chapter \d+ \| .+ \| \d{4}–\d{4}$")
SLOPPY = 4                                  # in the script, the 5th researcher is sloppy and reads only the first 5 lines
TITLE_ZH = {                                # the titles the mock translator knows
    "Where Machine Learning Came From": "机器学习从哪里来",
    "Linear Models and Gradient Descent": "线性模型与梯度下降",
    "Neural Networks and Autodiff": "神经网络与自动求导",
    "From CNNs to Transformers": "从 CNN 到 Transformer",
    "Large Language Models and APIs": "大语言模型与 API",
    "The Agent Loop": "Agent 循环",
    "Harness Engineering": "Harness 工程",
    "Multi-Agent Collaboration": "多 Agent 协同",
    "Cross-Model Collaboration": "跨模型协作",
}


# ---------------------------------------------------------------- the ledger (Chapter 7: budgets)
class Ledger:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = {}

    def add(self, vendor, usage, calls=1):
        with self.lock:
            row = self.rows.setdefault(vendor, {"calls": 0, "in": 0, "out": 0})
            row["calls"] += calls
            row["in"] += usage["input_tokens"]
            row["out"] += usage["output_tokens"]

    def report(self, live):
        total = 0.0
        print(f"   {fit('Vendor', 22)}| Calls | Input tokens | Output tokens | Estimated cost (USD)")
        for vendor, r in self.rows.items():
            p_in, p_out = PRICES[vendor]
            cost = r["in"] / 1e6 * p_in + r["out"] / 1e6 * p_out
            total += cost
            print(f"   {fit(label(vendor, live), 22)}| {r['calls']:>5} | {r['in']:>12,} | {r['out']:>13,} | {cost:.5f}")
        print(f"   {fit('Total', 22)}|       |              |               | {total:.5f}")


LEDGER = Ledger()


# ---------------------------------------------------------------- the orchestrator
def plan(live):
    prompt = (f"{TASK}\nThe chapter folders in the repo: {', '.join(CHAPTERS)}.\n"
              "List the chapter folders to research, one subtask per chapter. Output only a JSON array of strings.")
    try:
        reply = ask("claude", prompt, live, MockLLM([json.dumps(CHAPTERS)]))
        LEDGER.add("claude", reply.usage)
        dirs = [d for d in json.loads(re.search(r"\[.*\]", reply.text, re.S).group(0)) if d in CHAPTERS]
    except Exception:                       # the call failed, or the format was wrong
        dirs = []
    return dirs or CHAPTERS                 # fallback when the model ignores the format


# ---------------------------------------------------------------- researchers (the ReAct agent from Chapter 6)
def researcher_policy(chapter, lines):
    def read(messages, tools):
        return {"text": f"Read the beginning of {chapter}.",
                "tool_calls": [{"name": "read_file", "args": {"path": f"{chapter}/README.md", "start": 1, "end": lines}}]}

    def summarize(messages, tools):
        text = next(m for m in reversed(messages) if m["role"] == "tool")["content"]
        title = re.search(r"^\d+: # (Chapter \d+) — (.+)$", text, flags=re.M)
        cells = re.findall(r"^\d+: \| ([^|]*?\d{4}[^|]*?) \|", text, flags=re.M)
        years = [int(y) for c in cells for y in re.findall(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", c)]
        span = f"{min(years)}–{max(years)}" if years else "?"
        if not title:
            return f"{chapter} | (couldn't find the title) | {span}"
        return f"{title.group(1)} | {title.group(2).split(': ')[0]} | {span}"
    return [read, summarize]


def research(i, chapter, live, feedback=None):
    task = (f"Read the beginning of {chapter}/README.md (the title and the timeline table), and output only one line: "
            "Chapter N | main title of the chapter (the part before the colon) | earliest year–latest year")
    if feedback:
        task += f"\nThe reviewer sent back your last result: {feedback}"
    lines = 5 if i == SLOPPY and not feedback else 45
    provider = provider_for("deepseek", live)
    try:
        answer, messages, usage = run_agent(task, provider, model_for("deepseek") if provider != "mock" else None,
                                            mock=MockLLM(researcher_policy(chapter, lines)), max_steps=6,
                                            verbose=False)
    except Exception as e:                  # one researcher failing mustn't bring down the whole team
        return f"{chapter} | (call failed: {type(e).__name__}) | ?"
    LEDGER.add("deepseek", usage, calls=sum(m["role"] == "assistant" for m in messages))
    return answer.strip().splitlines()[-1] if answer.strip() else ""


# ---------------------------------------------------------------- the translator (an A2A server, backed by gpt)
def translator_handler(live):
    def handler(text):
        source = text.split(":", 1)[-1].strip()
        try:
            reply = ask("gpt", f"Translate this chapter title of a book into Chinese. Output only the translation:\n{source}",
                        live, MockLLM([TITLE_ZH.get(source, "(unknown title)")]))
        except Exception as e:
            return "TASK_STATE_FAILED", f"the call to the model behind the translator failed: {type(e).__name__}"
        LEDGER.add("gpt", reply.usage)
        return "TASK_STATE_COMPLETED", reply.text.strip()
    return handler


# ---------------------------------------------------------------- the reviewer
def review(lines):
    """Whatever a program can check, check with a program first: the format, and whether there's a time span.
    Returns the failures as {index: reason}."""
    problems = {}
    for i, line in enumerate(lines):
        if not LINE.match(line):
            problems[i] = f"\"{line}\" has the wrong format or no time span; please read the whole timeline table"
    return problems


def main():
    parser = add_live_arg(argparse.ArgumentParser(description="Capstone: a small cross-vendor team"))
    args = parser.parse_args()
    live = args.live
    print(f"Task: {TASK}\nTeam: {roster(live)}\n")

    dirs = plan(live)
    print(f"[1] The orchestrator, {label('claude', live)}, split the task into {len(dirs)} subtasks.")

    print(f"\n[2] {len(dirs)} researchers, {label('deepseek', live)}, start in parallel...")
    with ThreadPoolExecutor(max_workers=len(dirs)) as pool:
        lines = list(pool.map(lambda a: research(*a, live), enumerate(dirs)))
    for line in lines:
        print(f"   · {line}")

    print(f"\n[3] The reviewer, {label('claude', live)}, checks the format with a program first...")
    for round_ in range(1, 3):
        problems = review(lines)
        if not problems:
            print("   ✅ Everything passes.")
            break
        for i, why in problems.items():
            print(f"   ❌ Researcher {i + 1}: {why}. Sent back to be redone.")
            lines[i] = research(i, dirs[i], live, feedback=why)
            print(f"      After redoing it: {lines[i]}")
    else:
        print("   ⚠️  Some rows still fail after two rounds; handing them to a human.")

    print(f"\n[4] Handing the titles over A2A to the translator, {label('gpt', live)}...")
    server = A2AServer("Translator", "Translates English chapter titles into Chinese", [
        {"id": "translate-title", "name": "Title translation", "description": "English titles into Chinese",
         "tags": ["translation"]}],
        translator_handler(live)).start()
    try:
        card = fetch_card(server.base_url)
        print(f"   Got the Agent Card: {card['name']}, skill {card['skills'][0]['id']}")
        chinese = []
        for line in lines:
            parts = line.split(" | ")
            title = parts[1] if len(parts) == 3 else line
            state, text = task_text(send_text(card, f"Translate: {title}"))
            chinese.append(text if state == "TASK_STATE_COMPLETED" else "(translation failed)")
    finally:
        server.stop()
    print(f"   {len(chinese)} titles done.")

    bad = len(review(lines))
    canned = ("All the chapters are there, each with a time span. The first five chapters form one main line from "
              "statistical learning to large language models, and Chapter 6 turns to agents: a sensible order."
              if not bad else f"The information for {bad} chapter(s) is still incomplete; "
                              "a human should fill it in before this is published.")
    try:
        comment = ask("claude", "Here is a study card. In a sentence or two, say whether it is complete and "
                      "whether it has any obvious problems:\n" + "\n".join(lines), live, MockLLM([canned]))
        LEDGER.add("claude", comment.usage)
        comment = comment.text.strip()
    except Exception as e:
        comment = f"(the review failed: {type(e).__name__})"
    print(f"\n[5] The reviewer's comments: {comment}")

    print("\n🎓 Study card\n")
    print("| Chapter | Title | 中文 | Years |\n|---|---|---|---|")
    for line, zh in zip(lines, chinese):
        parts = line.split(" | ") + ["", "", ""]
        print(f"| {parts[0]} | {parts[1]} | {zh} | {parts[2]} |")

    print("\n💰 The bill (the mock models' token counts are estimates; prices are in router.py)")
    LEDGER.report(live)

    print("""
🗺️  What this little project uses from the whole repo:

   The user's task
    └─ Orchestrator (claude): splits the task ........................... Chapter 8, orchestrator–worker
        ├─ Nine researchers (deepseek), in parallel ..................... Chapter 6, the ReAct loop and tools
        │    └─ read_file can only read files inside the repo ........... Chapter 7, sandbox
        ├─ Reviewer (claude): program check → send back → model review .. Chapter 8, writer–reviewer loop
        └─ Translator (gpt): strangers working together through
           an Agent Card and SendMessage ................................ Chapter 9, A2A
   Running through everything: the shared chat() adapter layer (Chapters 5 and 9) and the ledger (Chapter 7, budgets);
   and the model itself is the "guess the next word" machine from Chapters 1 to 5.

Think about it:
  1. Why make deepseek the researchers and claude the reviewer, and not the other way around? What would happen
     if you swapped them? (Hint: prices and the number of calls.)
  2. If the reviewer were also deepseek, how would that differ from "reviewing your own work"?
     (Think back to the "error correlation" in the cross-checking lab.)
  3. If you replaced the translator with an A2A agent someone else runs on the internet, which lines of this program
     would you have to change? And what extra risks would you have to guard against?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
