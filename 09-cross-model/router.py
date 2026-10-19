"""
Lab 1: Routing: which vendor should get this request?
=====================================================

A product is connected to the APIs of three vendors, Claude, GPT and DeepSeek, and gets all kinds of requests every day.
Send everything to the most expensive model and your wallet can't take it; send everything to the cheapest and some jobs won't be done well.
A router does two things:
  1. Classify: first decide which category the request belongs to
  2. Dispatch: hand it to the right vendor according to the routing table; if that vendor is down, move on to the next one in order (failover)

Note: the routing table below is only an **example configuration**; it says nothing about which vendor is better at what.
Each vendor's strengths change from version to version. A real routing table should be decided by your own tasks and evals
(the "acceptance check" idea from Chapter 7).

Run:
  python router.py            # mock model
  python router.py --live     # vendors whose keys are set use real models
"""

import argparse

from team import VENDORS, add_live_arg, ask, fit, label, provider_for, roster
from common.llm import MockLLM

CATEGORIES = {
    "chat": "everyday questions, greetings, simple common knowledge",
    "code": "writing programs, fixing bugs, explaining code",
    "long document": "summarizing and distilling very long material",
    "reasoning": "math proofs, logical analysis, judgments that take several steps of thinking",
}
ROUTES = {                              # example configuration: first choice and backups for each category, tried in order
    "chat": ["deepseek", "gpt", "claude"],
    "code": ["claude", "gpt", "deepseek"],
    "long document": ["deepseek", "claude", "gpt"],
    "reasoning": ["gpt", "claude", "deepseek"],
}
# Price per million tokens (USD): (input, output). Prices change often; the official websites have the final word.
PRICES = {
    "claude": (4.00, 20.00),            # claude-opus-5-5, from Anthropic's website
    "gpt": (5.00, 30.00),               # gpt-5.5, according to third-party price comparison sites (2026)
    "deepseek": (0.14, 0.28),           # deepseek-v4-flash, according to third-party price comparison sites (2026)
}
# The requests received in one day: (content, estimated input tokens, estimated output tokens)
REQUESTS = [
    ("Good morning! What should I wear today?", 30, 150),
    ("This Python code raises a KeyError, help me find out why: config['port']", 400, 600),
    ("Summarize these 30,000-word minutes as 5 conclusions", 30000, 800),
    ("Prove that the square root of 2 is irrational", 50, 900),
    ("Come up with 5 names for our new coffee shop", 40, 200),
    ("Write a quicksort function, with unit tests", 60, 1200),
    ("Distill this 8,000-word paper into a 1-page summary", 8000, 700),
    ("Someone says 'I am lying.' Is that true or false? Please analyze it", 40, 800),
]
KEYWORDS = {"code": ["Python", "function", "code", "bug", "error", "KeyError"],
            "long document": ["summarize", "distill", "summary", "minutes"],
            "reasoning": ["prove", "analyze", "true or false", "logic"]}


# ---------------------------------------------------------------- step 1: classify
def classify(text, live):
    """The mock model classifies by keywords; with real models, the cheapest vendor does it (classifying costs money too, so use a cheap one)."""
    def rule(messages, tools):
        for cat, words in KEYWORDS.items():
            if any(w.lower() in text.lower() for w in words):
                return cat
        return "chat"
    options = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    prompt = (f"Put the user's request into one of the categories below. Output only the category name, "
              f"nothing else:\n{options}\n\nRequest: {text}")
    try:
        cat = ask("deepseek", prompt, live, MockLLM([rule])).text.strip().strip(".").lower()
    except Exception:                                # if the classifier itself fails, fall back to the keyword rules
        cat = rule(None, None)
    return cat if cat in CATEGORIES else "chat"


# ---------------------------------------------------------------- step 2: dispatch + failover
def call(vendor, text, live, down):
    if vendor in down:
        raise ConnectionError("503 Service Unavailable (simulated)")
    return ask(vendor, text, live, MockLLM([f"({vendor}'s mock answer) Got it: {text[:12]}..."]), effort="medium")


def route(text, live, down):
    cat = classify(text, live)
    tried = []
    for vendor in ROUTES[cat]:
        try:
            reply = call(vendor, text, live, down)
            return cat, vendor, reply, tried
        except Exception as e:                       # network errors, rate limits, overloaded servers... all move on to the next vendor
            tried.append(f"{vendor} failed ({str(e)[:40]})")
    return cat, None, None, tried


def cost(vendor, tokens_in, tokens_out):
    p_in, p_out = PRICES[vendor]
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def run_day(live, down, title):
    print(f"\n{title}")
    print(f"   {fit('Request', 52)} | {fit('Category', 13)} | Answered by")
    total, answered = 0.0, []
    for text, t_in, t_out in REQUESTS:
        cat, vendor, reply, tried = route(text, live, down)
        if vendor and provider_for(vendor, live) != "mock":     # real models: use the real token counts
            t_in, t_out = reply.usage["input_tokens"], reply.usage["output_tokens"]
        note = f"  ← tried first: {'; '.join(tried)}" if tried else ""
        print(f"   {fit(text, 52)} | {fit(cat, 13)} | "
              f"{label(vendor, live) if vendor else '❌ all failed'}{note}")
        if vendor:
            total += cost(vendor, t_in, t_out)
            answered.append((t_in, t_out))
    return total, answered


def main():
    parser = add_live_arg(argparse.ArgumentParser(description="Routing across vendors"))
    args = parser.parse_args()
    print(f"Team: {roster(args.live)}")
    print("Routing table (example configuration): " + "; ".join(f"{k} → {' → '.join(v)}" for k, v in ROUTES.items()))

    total, answered = run_day(args.live, set(), "[1] A normal day:")
    total_down, _ = run_day(args.live, {"gpt"},
                            "[2] A day when gpt is down: the router automatically moves on to the next vendor, "
                            "and users don't notice")

    print("\n[3] The same 8 requests, comparing costs (from the estimated token counts and the price table above, in USD):")
    for v in VENDORS:
        all_v = sum(cost(v, a, b) for a, b in answered)
        print(f"   {fit('All to ' + v, 20)}{all_v:8.4f}")
    print(f"   {fit('By the routing table', 20)}{total:8.4f}")
    print("""
Reading the results:
  * Classifying also calls a model, so the cheapest vendor does it; in production, a small, specially trained
    classifier is common.
  * Long documents often have tens of thousands of input tokens, so sending them to a cheap model saves the most money.
    But cheap doesn't mean good enough: which kinds of tasks can go to a cheap model is something only your own
    evals can answer, and quality isn't measured here.
  * Failover makes the three vendors back each other up: if any one is rate-limited or down, the request still gets answered.
    This is one of the most practical benefits of "connecting to several vendors": not putting all your eggs in one basket.
  * All of this works thanks to the "adapter layer" in common/llm.py: the three vendors' APIs look different,
    but the router only has to call the same chat().

Think about it:
  1. What happens if the classification is wrong (a hard reasoning question is treated as chat)? How would you catch errors like that?
  2. Another approach is a "cascade": let a cheap model answer first, and pass it to an expensive one if the answer
     doesn't pass a check. What kinds of tasks can be "checked" automatically?
  3. After a failover to another vendor, the style and format of the answer may both be different.
     How should the downstream program cope?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
