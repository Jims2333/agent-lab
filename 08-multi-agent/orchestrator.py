"""
Lab 1: Orchestrator–worker: one lead, several helpers working in parallel
=========================================================================

The multi-agent research system Anthropic described in 2025 uses exactly this pattern:
  1. The lead agent (the orchestrator) splits the task into a few independent subtasks
  2. It starts a subagent (a worker) for each subtask; each works in parallel in its own **clean context**
  3. The subagents hand back only short conclusions, and the lead agent pulls them together

The task here: research Chapters 1–7 of this repo, and sum up in one sentence which period each chapter's stories cover.
The workers reuse the ReAct agent written in Chapter 6 (react_agent.run_agent).
At the end we compare with "one agent reads everything from start to finish": speed, the size of each agent's context, total tokens.

Run:
  python orchestrator.py                     # mock model (each call simulates 0.3 seconds of network latency)
  python orchestrator.py --provider claude   # real model
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "06-agent-loop"))

from common.llm import MockLLM, add_provider_args, chat, estimate_tokens, user_message  # noqa: E402
from react_agent import run_agent  # noqa: E402

CHAPTERS = sorted(p.name for p in ROOT.iterdir() if re.match(r"0[1-7]-", p.name))
TASK = ("Research Chapters 1–7 of the agent-lab repo: for each chapter, sum up in one sentence "
        "which period its stories cover and what its topic is.")
LATENCY = 0.3


class SlowMock(MockLLM):
    """A mock model plus simulated network latency, so the effect of running "in parallel" is visible."""

    def __call__(self, messages, tools):
        time.sleep(LATENCY)
        return super().__call__(messages, tools)


# ---------------------------------------------------------------- the mock model's script
def worker_policy(chapter):
    """A worker: read the beginning of this chapter's README and boil the title and timeline table down to one sentence."""
    def read(messages, tools):
        return {"text": f"First, I'll read the beginning of {chapter}.",
                "tool_calls": [{"name": "read_file", "args": {"path": f"{chapter}/README.md", "start": 1, "end": 45}}]}

    def summarize(messages, tools):
        text = next(m for m in reversed(messages) if m["role"] == "tool")["content"]
        title = re.search(r"^\d+: # (Chapter \d+) — (.+)$", text, flags=re.M)
        # timeline rows: the first cell holds a year, e.g. "1958", "Jul 1958", "1971–1976", "2023–"
        cells = re.findall(r"^\d+: \| ([^|]*?\d{4}[^|]*?) \|", text, flags=re.M)
        years = [int(y) for c in cells for y in re.findall(r"(?<!\d)\d{4}(?!\d)", c)]
        span = f"{min(years)}–{max(years)}" if years else "(no timeline found)"
        name = f"{title.group(1)} · {title.group(2).split(': ')[0]}" if title else chapter
        return f"{name}: {span}, {len(cells)} milestones."
    return [read, summarize]


def lead_plan_mock(messages, tools):
    return json.dumps([f"Read {c}/README.md and sum up the chapter's time span and topic in one sentence."
                       for c in CHAPTERS])


def lead_summary_mock(messages, tools):
    reports = messages[-1]["content"]
    lines = re.findall(r"^Subtask \d+ result: (.+)$", reports, flags=re.M)
    return "Summary:\n" + "\n".join(f"  · {l}" for l in lines)


# ---------------------------------------------------------------- the orchestrator
def plan(provider, model, mock):
    prompt = (f"{TASK}\nThe chapter folders in the repo are: {', '.join(CHAPTERS)}.\n"
              "Split the task into independent subtasks that can be done in parallel, one per chapter. "
              "Output only a JSON array of strings, and nothing else.")
    reply = chat([user_message(prompt)], provider=provider, model=model, mock=mock, effort="low")
    try:
        subtasks = json.loads(re.search(r"\[.*\]", reply.text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        subtasks = [f"Read {c}/README.md and sum up the chapter in one sentence." for c in CHAPTERS]   # fallback when the model ignores the format
    return subtasks, reply.usage


def context_tokens(messages):
    """Roughly how many tokens the agent's context holds by the end (same estimate as the mock model's)."""
    return estimate_tokens(json.dumps([m["content"] for m in messages], ensure_ascii=False))


def run_worker(i, subtask, provider, model):
    mock = SlowMock(worker_policy(CHAPTERS[i])) if provider == "mock" else None
    t0 = time.perf_counter()
    answer, messages, usage = run_agent(subtask, provider, model, mock=mock, max_steps=6, verbose=False)
    return {"id": i + 1, "answer": answer, "steps": sum(m["role"] == "assistant" for m in messages),
            "tokens": usage["input_tokens"] + usage["output_tokens"], "context": context_tokens(messages),
            "seconds": time.perf_counter() - t0}


def synthesize(reports, provider, model, mock):
    body = "\n".join(f"Subtask {r['id']} result: {r['answer']}" for r in reports)
    prompt = (f"Original task: {TASK}\nHere are the conclusions the subagents handed back:\n{body}\n\n"
              "Pull them together into a short list, ordered by chapter.")
    reply = chat([user_message(prompt)], provider=provider, model=model, mock=mock, effort="low")
    return reply.text, reply.usage


# ---------------------------------------------------------------- control group: one agent reads everything
def single_agent_policy():
    steps = [{"text": f"Read {c}.", "tool_calls": [{"name": "read_file", "args": {"path": f"{c}/README.md", "start": 1, "end": 45}}]}
             for c in CHAPTERS]
    steps.append("(One agent has read all seven chapters and writes the summary in a single context.)")
    return steps


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="Orchestrator–worker"))
    args = parser.parse_args()
    lead_mock = SlowMock([lead_plan_mock, lead_summary_mock])

    print(f"Task: {TASK}\n")
    t0 = time.perf_counter()
    subtasks, plan_usage = plan(args.provider, args.model, lead_mock)
    print(f"[1] The orchestrator split the task into {len(subtasks)} subtasks:")
    for i, s in enumerate(subtasks, 1):
        print(f"   {i}. {s}")

    print(f"\n[2] {len(subtasks)} workers start at the same time (each with its own context)...")
    with ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        reports = list(pool.map(lambda a: run_worker(*a, args.provider, args.model), enumerate(subtasks)))
    for r in reports:
        print(f"   Worker {r['id']}: {r['steps']} steps, {r['seconds']:.1f} s, about {r['tokens']} tokens → {r['answer']}")

    summary, sum_usage = synthesize(reports, args.provider, args.model, lead_mock)
    multi_time = time.perf_counter() - t0
    multi_tokens = sum(r["tokens"] for r in reports) + sum(plan_usage.values()) + sum(sum_usage.values())
    print(f"\n[3] The orchestrator's summary:\n{summary}")

    print("\n[4] Control group: the same task, with one agent reading everything from start to finish")
    t0 = time.perf_counter()
    mock = SlowMock(single_agent_policy()) if args.provider == "mock" else None
    _, messages, usage = run_agent(TASK, args.provider, args.model, mock=mock, max_steps=12, verbose=False)
    single_time = time.perf_counter() - t0
    single_tokens = usage["input_tokens"] + usage["output_tokens"]
    single_context = context_tokens(messages)
    biggest_worker = max(r["context"] for r in reports)

    print("                  | Time    | Total tokens | Largest context of one agent (about)")
    print(f"   Multi-agent    | {multi_time:5.1f} s | {multi_tokens:12,} | {biggest_worker:,} tokens")
    print(f"   Single agent   | {single_time:5.1f} s | {single_tokens:12,} | {single_context:,} tokens")
    print(f"""
   * Speed: the workers run in parallel, so the total time is close to "the slowest one," not everyone added up.
   * Context: each worker holds only its own chapter, while the single agent has to fit all seven chapters into one context.
   * Cost: in this example the multi-agent total is actually lower, because the single agent resends its ever-longer
     history at every step. But real workers often explore for many steps on their own and repeat each other's
     searches, and on top of that come the planning, the summary and a system prompt for each one, so the bill is
     usually higher: Anthropic reports that their multi-agent system uses about 15 times as many tokens as a chat.
   So multi-agent setups suit tasks that "can be split into independent parts, each with a lot of information,"
   such as broad research.

Think about it:
  1. If the Chapter 5 worker needed the Chapter 4 worker's conclusion before it could start, could they still run in parallel?
  2. If a worker hands back only one sentence, could important details get lost? How do you decide "how much to hand back"?
  3. If you went from 7 workers to 70 (say, one per section), what new problems would you run into?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
