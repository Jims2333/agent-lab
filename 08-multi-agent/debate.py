"""
Lab 3: Multi-agent debate: when are three heads better than one?
================================================================

The 2023 paper from MIT and Google Brain (Du et al.) proposed "multi-agent debate":
  Round 1: several agents each answer on their own
  Round 2: each agent sees the others' answers and reasoning, and decides whether to stick with its answer or change it
  At the end: a judge (or a majority vote) gives the final answer

The question is the most famous one from psychologist Shane Frederick's 2005 "Cognitive Reflection Test":
  "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
Intuition blurts out $0.10; the right answer is $0.05.

We give the mock model two "personalities" and see when a debate helps:
  · Verifier: plugs every candidate answer back into the question first, and trusts whichever one works
  · Conformist: looks at what most people say, and goes along with it

Run:
  python debate.py                     # mock model
  python debate.py --provider claude   # real model (all three agents use the same model; they will probably all get it right in round 1)
"""

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import MockLLM, add_provider_args, chat, user_message  # noqa: E402

QUESTION = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost (in dollars)?")
ANSWER = 0.05
FORMAT = "Keep your reasoning short, and on the last line write only: Answer: X dollars"
ANSWER_RE = r"Answer:\s*\$?(\d+(?:\.\d+)?)"

FIRST_ROUND = {   # the mock model's round-1 answers: two go with their gut, one actually writes an equation
    "A": "1.10 minus 1.00 is 0.10, so the ball costs 0.10 dollars.\nAnswer: 0.10 dollars",
    "B": "Let the ball be x, so the bat is x + 1.00: 2x + 1.00 = 1.10, x = 0.05.\n"
         "Check: 0.05 + 1.05 = 1.10 ✓\nAnswer: 0.05 dollars",
    "C": "You can see it at a glance: the ball costs 0.10 dollars.\nAnswer: 0.10 dollars",
}


def extract(text):
    found = re.findall(ANSWER_RE, text)
    return float(found[-1]) if found else None


def show(x):
    return "(no answer given)" if x is None else f"{x:.2f}"


def satisfies(ball):
    """Plug a candidate answer back into the question: do the ball and the bat add up to 1.10?"""
    return abs(ball + (ball + 1.00) - 1.10) < 1e-9


# ---------------------------------------------------------------- mock models with two "personalities"
def mock_agent(name, style):
    def respond(messages, tools):
        prompt = messages[-1]["content"]
        if "Other answers" not in prompt:
            return FIRST_ROUND[name]
        mine = extract(prompt.split("Your previous answer")[1].split("Other answers")[0])
        others = [float(x) for x in re.findall(ANSWER_RE, prompt.split("Other answers")[1])]
        if style == "verifier":
            if satisfies(mine):
                return (f"I plug mine back in: {mine:.2f} + {mine + 1:.2f} = {2 * mine + 1:.2f} ✓. "
                        f"I'm sticking with it.\nAnswer: {mine:.2f} dollars")
            good = [x for x in others if satisfies(x)]
            if good:
                x = good[0]
                return (f"I plug mine back in: {mine:.2f} + {mine + 1:.2f} = {2 * mine + 1:.2f}, not 1.10 ✗. "
                        f"Theirs: {x:.2f} + {x + 1:.2f} = 1.10 ✓, so I'm switching.\nAnswer: {x:.2f} dollars")
            return f"My answer doesn't check out, and neither do the others, so I'll keep mine for now.\nAnswer: {mine:.2f} dollars"
        majority = Counter([mine] + others).most_common(1)[0][0]
        if majority != mine:
            return f"Most people say {majority:.2f}; maybe I got it wrong.\nAnswer: {majority:.2f} dollars"
        return f"Everyone thinks the same as me, so it must be fine.\nAnswer: {mine:.2f} dollars"
    return respond


def judge_mock(messages, tools):
    votes = Counter(float(x) for x in re.findall(ANSWER_RE, messages[-1]["content"]))
    best, n = votes.most_common(1)[0]
    return f"{n} votes to {sum(votes.values()) - n}; going with the majority.\nAnswer: {best:.2f} dollars"


# ---------------------------------------------------------------- the debate
def debate(style, provider, model, rounds=2):
    names = list(FIRST_ROUND)
    agents = {n: MockLLM(mock_agent(n, style)) for n in names}
    answers = {}
    for r in range(1, rounds + 1):
        print(f"   —— Round {r} ——")
        new = {}
        for n in names:
            if r == 1:
                prompt = f"{QUESTION}\n{FORMAT}"
            else:
                others = "\n".join(f"[{m}] {answers[m]}" for m in names if m != n)
                prompt = (f"{QUESTION}\n\nYour previous answer:\n{answers[n]}\n\nOther answers:\n{others}\n\n"
                          f"Check every answer carefully (including your own), then decide whether to stick with "
                          f"yours or change it. {FORMAT}")
            reply = chat([user_message(prompt)], provider=provider, model=model, mock=agents[n], effort="low")
            new[n] = reply.text.strip()
            first_line = new[n].splitlines()[0]
            print(f"   {n}: {first_line[:70]}{'...' if len(first_line) > 70 else ''}  → {show(extract(new[n]))}")
        answers = new
    finals = "\n".join(f"[{n}] {answers[n]}" for n in names)
    prompt = (f"Question: {QUESTION}\n\nEach side's final answer after the debate:\n{finals}\n\n"
              f"You are the judge. Give the final answer. {FORMAT}")
    verdict = chat([user_message(prompt)], provider=provider, model=model, mock=MockLLM([judge_mock]), effort="low")
    result = extract(verdict.text)
    ok = result is not None and abs(result - ANSWER) < 1e-9
    print(f"   Judge: {verdict.text.strip().splitlines()[0]} → {show(result)}  {'✅ right' if ok else '❌ wrong'}")


# ---------------------------------------------------------------- thought experiment: many repetitions
def simulate(p, reliability, trials=20000, n=3, seed=0):
    """In round 1, each agent independently gets it right with probability p; when wrong, they all fall into the same intuitive trap.
    Round 2: each agent checks carefully with probability reliability (and can then recognize the right answer); otherwise it follows the majority."""
    rng = random.Random(seed)
    single = vote = debated = 0
    for _ in range(trials):
        first = [rng.random() < p for _ in range(n)]
        single += first[0]
        majority = sum(first) * 2 > n
        vote += majority
        second = []
        for _ in range(n):
            if rng.random() < reliability:
                second.append(any(first))       # checks carefully: if anyone gave the right answer, it recognizes it
            else:
                second.append(majority)         # conforms: follows the round-1 majority
        debated += sum(second) * 2 > n
    return single / trials, vote / trials, debated / trials


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="Multi-agent debate"))
    args = parser.parse_args()
    print(f"Question: {QUESTION}\n")
    for style in ["verifier", "conformist"]:
        print(f"[Scenario] All three agents are {style}s"
              + ("" if args.provider == "mock" else " (with a real model, the model decides its own personality)"))
        debate(style, args.provider, args.model)
        print()

    print("""In the mock model's script, round 1 is the same (one right, two wrong); only the personalities differ,
and the endings are complete opposites:
  · Verifiers: A and C plug their 0.10 back in, find that it doesn't work, switch to 0.05, and all three votes are right;
  · Conformists: B was right all along, but was talked out of it by "most people say 0.10," and in the end
    they all agreed on the wrong answer.
Whether a debate helps doesn't depend on "having more people"; it depends on whether each one **can independently check**
what the others say.
""")

    print("[Thought experiment] Repeat the process above 20,000 times (three agents; when they're wrong, "
          "they all fall into the same trap):")
    heads = ["One agent's accuracy p", "One agent", "Vote of three", "Debate: all verify",
             "Debate: half verify", "Debate: all conform"]
    print("   " + " | ".join(heads))
    for p in [0.3, 0.5, 0.7]:
        single, vote, full = simulate(p, 1.0)
        _, _, half = simulate(p, 0.5)
        _, _, none = simulate(p, 0.0)
        cells = [f"{p:.1f}"] + [f"{x:5.1%}" for x in (single, vote, full, half, none)]
        print(("   " + " | ".join(c.center(len(h)) for c, h in zip(cells, heads))).rstrip())
    print("""
   * When everyone conforms, the debate gives exactly the same result as "just voting": talking brought no new information.
   * When a single agent is right less than half the time (p = 0.3), a vote of three is actually worse than one agent:
     this is the flip side of Condorcet's 1785 "jury theorem": a majority vote only amplifies each person's original leaning.
   * Only when the agents really check the answers can a debate turn "a minority's right answer" into the final answer.
   * This is a thought experiment computed from our assumptions, not a measurement of real models. A 2025 study
     (Choi et al., "Debate or Vote") found on seven benchmarks that most of the gains from multi-agent debate
     actually come from majority voting itself.

Think about it:
  1. If the three agents were models from three different vendors, would the assumption "when they're wrong,
     they all fall into the same trap" still hold? (Chapter 9 picks this up.)
  2. Give an agent a real calculator tool and it can "check its work." Which questions can't be checked this way?
  3. If all three get it wrong in round 1, can a few more rounds of debate make them right?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
