"""
Lab 2: Cross-checking: letting models from different vendors check each other
==============================================================================

The debate lab in Chapter 8 left a question open: when several agents use the same model, they tend to fall
into the same trap, and voting can't save them.
So what about models from different vendors? Their training data and training methods differ, so their "blind spots"
may not be in the same places.

Four approaches are compared here:
  A. Ask only one vendor
  B. Ask the same vendor three times and take the majority
  C. Have each of the three vendors answer once and take the majority
  D. Have two vendors answer once each: if they agree, accept it; if not, send it to a human for review

Then a thought experiment shows that the more "correlated" the vendors' errors are, the less you gain by asking more of them.

Run:
  python cross_check.py            # mock models (the three vendors' "error logs" are set by us and don't reflect real models)
  python cross_check.py --live     # vendors whose keys are set use real models
"""

import argparse
import random
import re
from collections import Counter
from statistics import NormalDist

from team import VENDORS, add_live_arg, ask, fit, provider_for, roster
from common.llm import MockLLM

QUESTIONS = [   # (question, right answer): all well-known "trick questions"
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
     "How much does the ball cost (in dollars)?", "0.05"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines "
     "to make 100 widgets?", "5"),
    ("In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days "
     "for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?", "47"),
    ("How many times does the letter r appear in the word strawberry?", "3"),
    ("Which number is bigger, 9.11 or 9.9?", "9.9"),
    ("How many months in a year have 28 days?", "12"),
]
# The mock models' "error logs": which questions each vendor gets wrong, and what it answers (set by us, just for the demo)
MOCK_MISTAKES = {
    "claude": {3: "2", 4: "9.11"},
    "gpt": {0: "0.10", 4: "9.11"},
    "deepseek": {2: "24"},
}
FORMAT = "Give only the final answer, and on the last line write: Answer: X"


def name(i, vendor, live):
    """Real models show the vendor's name (the models are listed in the team roster at the top); mock models are labeled
    A/B/C, so nobody mistakes them for how real vendors perform."""
    return vendor if provider_for(vendor, live) != "mock" else f"Vendor {'ABC'[i]} (mock)"


def extract(text):
    """Take what comes after the last "Answer:" in the reply; if it's a number, keep only the number."""
    found = re.findall(r"Answer:\s*(.+)", text)
    raw = (found[-1] if found else text).strip()
    num = re.search(r"\d+(?:\.\d+)?", raw)
    return num.group(0) if num else raw


def key(x):
    """For comparing: 0.10 and 0.1 count as the same answer."""
    try:
        return f"{float(x):g}"
    except ValueError:
        return x


def same(a, b):
    return key(a) == key(b)


def answer_all(vendor, live):
    out = []
    for q, (question, truth) in enumerate(QUESTIONS):
        wrong = MOCK_MISTAKES[vendor].get(q)
        mock = MockLLM([f"Answer: {wrong if wrong else truth}"])
        try:
            out.append(extract(ask(vendor, f"{question}\n{FORMAT}", live, mock).text))
        except Exception as e:                       # network errors, rate limits... note it down rather than crash the whole lab
            out.append(f"(call failed: {type(e).__name__})")
    return out


def score(answers):
    return sum(same(a, t) for a, (_, t) in zip(answers, QUESTIONS))


def majority(*answer_lists):
    result = []
    for votes in zip(*answer_lists):
        top, n = Counter(key(v) for v in votes).most_common(1)[0]
        result.append(top if n * 2 > len(votes) else "(no majority)")
    return result


# ---------------------------------------------------------------- thought experiment: how "correlated" the errors are
def simulate(rho, error_rate=0.2, trials=40000, seed=0):
    """Each question has a "difficulty" shared by all vendors, plus each vendor's own randomness. The larger rho, the more
    likely the vendors are to go wrong on the same question.
    Assumption: when they're wrong, they all fall into the same trap (give the same wrong answer), the worst case for voting."""
    rng = random.Random(seed)
    t = NormalDist().inv_cdf(1 - error_rate)
    single = maj = undetected = flagged = 0
    for _ in range(trials):
        shared = rng.gauss(0, 1)
        wrong = [rho ** 0.5 * shared + (1 - rho) ** 0.5 * rng.gauss(0, 1) > t for _ in range(3)]
        single += wrong[0]
        maj += sum(wrong) >= 2
        undetected += wrong[0] and wrong[1]
        flagged += wrong[0] != wrong[1]
    return single / trials, maj / trials, undetected / trials, flagged / trials


def main():
    parser = add_live_arg(argparse.ArgumentParser(description="Cross-checking across vendors"))
    args = parser.parse_args()
    live = args.live
    print(f"Team: {roster(live)}\n")

    answers = {v: answer_all(v, live) for v in VENDORS}
    names = {v: name(i, v, live) for i, v in enumerate(VENDORS)}
    print(f"   {fit('Question', 40)} | Correct | " + " | ".join(fit(names[v], 15) for v in VENDORS))
    for q, (question, truth) in enumerate(QUESTIONS):
        cells = [f"{answers[v][q]}{'' if same(answers[v][q], truth) else ' (wrong)'}" for v in VENDORS]
        print((f"   {fit(question, 40)} | {truth:<7} | " + " | ".join(fit(c, 15) for c in cells)).rstrip())

    a, b, c = VENDORS
    repeat = [answer_all(a, live) for _ in range(3)]
    n = len(QUESTIONS)
    print(f"\n   {fit(f'A. Ask only {names[a]}', 52)}right: {score(answers[a])}/{n}")
    print(f"   {fit(f'B. Ask {names[a]} three times, majority', 52)}right: {score(majority(*repeat))}/{n}")
    print(f"   {fit('C. Three vendors answer once each, majority', 52)}right: {score(majority(*answers.values()))}/{n}")
    agree = [same(x, y) for x, y in zip(answers[a], answers[b])]
    ok = sum(1 for q, g in enumerate(agree) if g and same(answers[a][q], QUESTIONS[q][1]))
    bad = sum(1 for q, g in enumerate(agree) if g and not same(answers[a][q], QUESTIONS[q][1]))
    print(f"   {fit(f'D. {names[a]} and {names[b]} cross-check', 52)}agree and right: {ok}, "
          f"agree but both wrong: {bad}, disagree and go to a human: {n - ok - bad}")

    if not live:
        print("""
   In the mock models' setup:
   * B is the same as A: ask the same model three times, and it gets the same questions wrong (its errors are perfectly correlated).
   * C fixes the questions where "only one vendor is wrong," but on question 5 Vendor A and Vendor B fell into
     the same trap, and the majority vote can't save it.
   * D doesn't need to know the right answer: just by checking "whether the two vendors say the same thing,"
     it picks out the questionable ones and hands them to a human. The price is that questions where
     "both agree but are both wrong" slip through.""")

    print("\n[Thought experiment] Each vendor alone is wrong 20% of the time; vary the correlation ρ between "
          "the vendors' errors (repeated over 40,000 questions):")
    heads = ["Correlation ρ", "One vendor wrong", "Majority of 3 wrong",
             "Two-vendor check: agree but both wrong (missed)", "Two-vendor check: needs a human"]
    print("   " + " | ".join(heads))
    for rho in [0.0, 0.3, 0.6, 0.9]:
        single, maj, undetected, flagged = simulate(rho)
        cells = [f"{rho:.1f}"] + [f"{x:5.1%}" for x in (single, maj, undetected, flagged)]
        print(("   " + " | ".join(c.center(len(h)) for c, h in zip(cells, heads))).rstrip())
    print("""
   * ρ = 0 (the vendors' errors are unrelated): a vote of three cuts the error rate from 20% to about 10%,
     and a two-vendor check misses only 4%.
   * The larger ρ, the less you gain from asking more vendors; as ρ approaches 1, it's about the same as
     "asking the same model three times."
   * So the value of cross-vendor collaboration depends on how different the vendors' "blind spots" are.
     Only your own test set can measure that; don't take it for granted. The numbers in the table are computed
     from assumptions, not measured on real models.

Think about it:
  1. How is having Vendor A "review" Vendor B's answer (instead of each answering on its own) different from D here?
     Could the reviewer be "led astray" by the answer it sees?
  2. If all three models were trained on similar web data, do you think their errors would be highly or weakly
     correlated? How would you measure it?
  3. How much manual work does "send it to a human when they disagree" create? In your situation, which costs more:
     the errors that slip through, or the human labor?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
