"""
Side quest: the smallest "relay" program that lets two AIs chat (the simplest possible harness)
==============================================================================================

Neither AI actually knows the other one exists. Each of them is just answering messages from a "user",
and that "user" is really this program: it passes what A says on to B, and what B says back to A.

Usage:
  python relay.py                                    # default mock ↔ mock, no API key needed
  python relay.py --show-history                     # at the end, print each side's own messages list
  python relay.py --a claude --b deepseek --turns 6 --topic "whether AI can dream"
  python relay.py --a gpt --b deepseek

Environment variables you need (only for the models you actually use):
  ANTHROPIC_API_KEY   → claude   (pip install anthropic)
  DEEPSEEK_API_KEY    → deepseek (pip install openai; DeepSeek's API is OpenAI-compatible)
  OPENAI_API_KEY      → gpt      (pip install openai)
"""

import argparse
import os
import textwrap

# Default model for each provider; override with --model-a / --model-b.
# Providers rename their models from time to time. If a run fails, check the official docs for the latest names first.
DEFAULT_MODELS = {
    "claude": "claude-opus-5-5",
    "deepseek": "deepseek-v4-flash",
    "gpt": "gpt-5.5",
    "mock": "mock",
}
END_MARK = "[END]"


# ---------------------------------------------------------------------------
# Part 1: wrap the different providers' APIs so they all look the same
#   reply(system, messages) -> str
# system   : this AI's "persona" and rules
# messages : the conversation history from this AI's own point of view (user / assistant, alternating)
# ---------------------------------------------------------------------------

def make_claude(model):
    import anthropic

    client = anthropic.Anthropic()  # reads the key from ANTHROPIC_API_KEY

    def reply(system, messages):
        resp = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_config={"effort": "low"},       # small talk doesn't need deep thinking; cheaper and faster
            # if a line ever gets blocked by the safety classifier, automatically retry with another model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            return "(Claude declined to answer this one.)"
        return "".join(b.text for b in resp.content if b.type == "text")

    return reply


def make_openai_compatible(model, api_key_env, base_url=None):
    """DeepSeek and GPT both use OpenAI's API format; the only differences are base_url and the key."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    def reply(system, messages):
        resp = client.chat.completions.create(
            model=model,
            # in the OpenAI format, system is the first entry in messages
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content

    return reply


MOCK_LINES = {
    "first": [
        "Let me open with a claim: predicting the next word is not the same as understanding.",
        "But when people understand a sentence, there's 'prediction' going on in their heads too.",
        f"So our real disagreement is about how to define 'understanding'. Let's stop here for today {END_MARK}",
    ],
    "second": [
        "I don't quite agree. If it can use the sentence to do the right thing, doesn't that count as understanding?",
        "Right, so maybe the key isn't 'is there prediction' but 'what is being predicted'.",
        "Agreed. Next time we can start from the definitions.",
    ],
}


def make_mock(order):
    """Pretends to be an AI: no network, no cost, just a way to see the "relay" process clearly."""
    lines = MOCK_LINES[order]
    count = 0

    def reply(system, messages):
        nonlocal count
        heard = messages[-1]["content"].split('…"')[-1].lstrip(". ")   # drop the other side's quote of us
        text = f'Got "{heard[:24].rstrip()}…". {lines[min(count, len(lines) - 1)]}'
        count += 1
        return text

    return reply


def make_speaker(kind, model, order):
    if kind == "claude":
        return make_claude(model)
    if kind == "deepseek":
        return make_openai_compatible(model, "DEEPSEEK_API_KEY", "https://api.deepseek.com")
    if kind == "gpt":
        return make_openai_compatible(model, "OPENAI_API_KEY")
    return make_mock(order)


# ---------------------------------------------------------------------------
# Part 2: the relay loop (this is the heart of the harness)
# ---------------------------------------------------------------------------

def system_prompt(me, other, topic):
    return (
        f"You are {me}. You are chatting with another AI ({other}). A program in the middle relays the messages, "
        f"so every 'user' message you see is actually something {other} said. The topic is: {topic}. "
        "Keep each reply to 3 sentences or fewer, have opinions of your own, and feel free to ask the other side questions. "
        f"If you feel the topic has been talked through, add {END_MARK} at the end of your reply."
    )


def relay(a, b, turns, topic, show_history):
    speakers = [a, b]
    for me, other in ((a, b), (b, a)):
        me["system"] = system_prompt(me["name"], other["name"], topic)
        me["history"] = []

    # The host (that is, this program) says an opening line to A first; it is A's first user message
    a["history"].append({"role": "user", "content": f"Today's topic: {topic}. Why don't you kick things off?"})

    for turn in range(turns):
        me, other = speakers[turn % 2], speakers[(turn + 1) % 2]
        text = me["reply"](me["system"], me["history"])

        # The key step: the same line plays a different role in each side's history
        me["history"].append({"role": "assistant", "content": text})    # to myself: something I said
        other["history"].append({"role": "user", "content": text})      # to the other side: something the user said

        print(f"\n[{turn + 1}] {me['name']}:")
        print(textwrap.indent(textwrap.fill(text, 60), "    "))

        if END_MARK in text:
            print(f"\n({me['name']} said {END_MARK}, so the conversation is over)")
            break
    else:
        print(f"\n(Reached the {turns}-turn limit, so the conversation is over)")

    if show_history:
        for s in speakers:
            print(f"\n======== The conversation history as {s['name']} sees it (sent to the API in full on every call) ========")
            print(f"system: {s['system'][:90]}…")
            for m in s["history"]:
                print(f"  {m['role']:>9}: {m['content'][:80]}…")


def main():
    parser = argparse.ArgumentParser(description="Let two AIs chat with each other")
    choices = list(DEFAULT_MODELS)
    parser.add_argument("--a", choices=choices, default="mock", help="the AI that speaks first")
    parser.add_argument("--b", choices=choices, default="mock", help="the AI that speaks second")
    parser.add_argument("--model-a", help="override A's default model")
    parser.add_argument("--model-b", help="override B's default model")
    parser.add_argument("--turns", type=int, default=6, help="maximum number of lines (both sides combined)")
    parser.add_argument("--topic", default="whether a machine can truly 'understand' a sentence")
    parser.add_argument("--show-history", action="store_true", help="print both sides' messages at the end")
    args = parser.parse_args()

    # When both sides use the same kind of AI, add a label to tell them apart
    name_a = args.a.capitalize() + ("-A" if args.a == args.b else "")
    name_b = args.b.capitalize() + ("-B" if args.a == args.b else "")
    try:
        a = {"name": name_a,
             "reply": make_speaker(args.a, args.model_a or DEFAULT_MODELS[args.a], "first")}
        b = {"name": name_b,
             "reply": make_speaker(args.b, args.model_b or DEFAULT_MODELS[args.b], "second")}
    except ImportError as e:
        raise SystemExit(f"Missing SDK: {e.name}. Run pip install {e.name} first")
    except KeyError as e:
        raise SystemExit(f"Environment variable {e} not found. Run export {e.args[0]}=your-key first")

    print(f"Topic: {args.topic}")
    print(f"{a['name']} ⇄ relay program ⇄ {b['name']}")
    relay(a, b, args.turns, args.topic, args.show_history)


if __name__ == "__main__":
    main()
