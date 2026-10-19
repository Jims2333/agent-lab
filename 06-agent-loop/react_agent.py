"""
Lab 3: A ReAct agent from scratch
=================================

The 2022 ReAct paper proposed a simple loop, and almost every agent today is a variation of it:

    while not done:
        Reason:  what should I do next?
        Act:     call a tool
        Observe: look at what the tool returned, then keep reasoning

Here we implement that loop in under 100 lines, together with the four tools from tools.py.
By default a mock model demonstrates one complete task. Its script reads the observation at each step,
and it makes a mistake and then corrects it, the same way a real model behaves.
Switch to a real model and you can ask it any question about this repo.

Run:
  python react_agent.py                                # mock model, built-in task
  python react_agent.py --provider claude --task "Which activation function does Chapter 3 use?"
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm import (MockLLM, add_provider_args, assistant_message, chat,  # noqa: E402
                        tool_message, user_message)
from tools import TOOLS, run_tool  # noqa: E402

SYSTEM = """You are a research assistant. You can use tools to look things up in agent-lab, a learning repo.
How to work: at each step, first explain your thinking in a sentence or two, then call a tool. Once you have enough
information, give the final answer directly and cite your source (file and line number).
Do not make up anything that isn't in the repo."""

DEFAULT_TASK = ("According to Chapter 1, how many percentage points lower was AlexNet's top-5 error rate than "
                "the runner-up's? First find the original sentence in the repo, then work it out with the calculator.")


# ---------------------------------------------------------------- the mock model's script
def last_observation(messages):
    return next(m for m in reversed(messages) if m["role"] == "tool")["content"]


def result_line(text):
    """The first search-result line that mentions AlexNet and the runner-up and gives at least two percentages."""
    return next((l for l in text.splitlines() if "AlexNet" in l and "runner-up" in l
                 and len(re.findall(r"\d+(?:\.\d+)?%", l)) >= 2), None)


def step_guess_file(messages, tools):
    return {"text": "Let me guess first: Chapter 1 probably has a file just about AlexNet.",
            "tool_calls": [{"name": "read_file", "args": {"path": "01-ml-origins/alexnet.md"}}]}


def step_search(messages, tools):
    return {"text": "That file doesn't exist, so my guess was wrong. I'll search the whole repo for the phrase instead.",
            "tool_calls": [{"name": "search_repo", "args": {"query": "top-5 error rate"}}]}


def step_calculate(messages, tools):
    line = result_line(last_observation(messages))
    if line is None:
        return "I couldn't find a sentence with both numbers in the search results, so I can't answer yet."
    ours, second = re.findall(r"(\d+(?:\.\d+)?)%", line)[:2]
    return {"text": f"Found it: AlexNet was at {ours}% and the runner-up at {second}%. I'll use the calculator "
                    "for the difference.",
            "tool_calls": [{"name": "calculator", "args": {"expression": f"{second} - {ours}"}}]}


def step_answer(messages, tools):
    source = next(result_line(m["content"]) for m in messages
                  if m["role"] == "tool" and result_line(m["content"])).split(": ")[0]
    diff = last_observation(messages)
    return f"AlexNet's top-5 error rate was {diff} percentage points lower than the runner-up's (source: {source})."


DEMO_SCRIPT = [step_guess_file, step_search, step_calculate, step_answer]


# ---------------------------------------------------------------- the agent loop itself
def run_agent(task, provider="mock", model=None, mock=None, max_steps=8, verbose=True):
    messages = [user_message(task)]
    usage = {"input_tokens": 0, "output_tokens": 0}
    for step in range(1, max_steps + 1):
        reply = chat(messages, system=SYSTEM, tools=TOOLS, provider=provider, model=model,
                     mock=mock, effort="medium")
        messages.append(assistant_message(reply))
        usage = {k: usage[k] + reply.usage[k] for k in usage}
        if verbose and reply.text and reply.tool_calls:
            print(f"   🤔 Step {step} Thought: {reply.text}")

        if not reply.tool_calls:                    # no more tool calls = this is the final answer
            return reply.text, messages, usage

        for call in reply.tool_calls:               # run every tool call the model asked for
            result, is_error = run_tool(call["name"], call["args"])
            messages.append(tool_message(call, result, is_error))
            if verbose:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                print(f"   🔧        Action: {call['name']}({args})")
                lines = result.splitlines()
                more = f"  ... ({len(lines)} lines in total)" if len(lines) > 2 else ""
                print(f"   {'❌' if is_error else '👀'}        Observation: " + " / ".join(lines[:2])[:110] + more)
    return f"(Hit the {max_steps}-step limit; the task is not finished.)", messages, usage


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="A ReAct agent from scratch"))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    if args.provider == "mock" and args.task != DEFAULT_TASK:
        raise SystemExit("The mock model can only demo the built-in task. To ask something else, "
                         "add --provider claude / gpt / deepseek.")

    print(f"Task: {args.task}\n")
    answer, messages, usage = run_agent(args.task, args.provider, args.model,
                                        mock=MockLLM(DEMO_SCRIPT), max_steps=args.max_steps)
    print(f"\n   ✅ Final answer: {answer}\n")

    roles = " → ".join({"user": "user", "assistant": "model", "tool": "tool result"}[m["role"]] for m in messages)
    print(f"Here is how the conversation history grew, one message at a time:\n   {roles}")
    print(f"   {len(messages)} messages in total, {sum(m['role'] == 'assistant' for m in messages)} model calls, "
          f"{usage['input_tokens']} tokens sent in all.")
    print("""
   Every round, the program sends the whole history (including every tool result) to the model again
   (Chapter 5 explained that the API is stateless).
   The model decides what to do next; the program runs the tools, writes the results back into the history,
   and decides when to stop.
   The "wrong file name" moment at step 2 is typical: an agent's value isn't that every step is right,
   but that it can correct itself based on what it observes.

Think about it:
  1. Run it again with --max-steps 2. What happens? Why must there always be a step limit?
  2. If the error message the model reads at step 2 were just the word "Error", could it still figure out
     how to correct itself?
  3. If you wrote this task as a fixed flow (search, then calculate, then answer), would you still need an agent?
     What kind of task is worth using an agent for?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
