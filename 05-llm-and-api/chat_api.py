"""
Lab 4: Calling an LLM API: statelessness, tokens and cost
=========================================================

The first three labs built "mini language models" on your own computer. Real large models are too big; they run on the provider's servers, and we call them through an API.
This lab uses the repo's shared client, common/llm.py, to demo three things you must know before using an API:
  1. The API is "stateless": it doesn't remember you; the "memory" is something you send over yourself every time
  2. How tokens and cost keep piling up in a multi-turn conversation
  3. Streaming: showing the output while it is being generated

Run:
  python chat_api.py                                 # uses the mock model by default, no API key needed
  python chat_api.py --provider claude               # use a real model (needs ANTHROPIC_API_KEY)
  python chat_api.py --provider deepseek --interactive   # chat freely with a real model
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so the script can find common/ in the repo root

from common.llm import (DEFAULT_MODELS, MockLLM, add_provider_args, assistant_message, chat,  # noqa: E402
                        cost_usd, user_message)


def memory_mock(messages, tools):
    """A mock model. Just like a real model, it can only see the messages sent in this one request, and remembers nothing else."""
    last = messages[-1]["content"]
    name = None
    for m in messages:
        found = re.search(r"my name is ([^.!?,]+)", m["content"], re.I)
        if m["role"] == "user" and found and "what" not in m["content"].lower():
            name = found.group(1).strip()
    if re.search(r"what('s| is) my name", last, re.I):
        return f"Your name is {name}." if name else "Sorry, I don't know your name. You haven't told me yet."
    if re.search(r"my name is", last, re.I):
        return f"Hi, {name}! Nice to meet you."
    count = f"{len(messages)} message" + ("s" if len(messages) != 1 else "")
    return f'(mock answer) About "{last[:40]}": this turn I received {count} in all.'


def show_request(messages):
    print("     messages sent: " + json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages], ensure_ascii=False))


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="Getting started with LLM APIs"))
    parser.add_argument("--interactive", action="store_true", help="enter free-chat mode")
    args = parser.parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    mock = MockLLM(memory_mock)
    system = "You are a concise, friendly assistant. Keep every answer to two sentences or fewer."

    def ask(messages, **kw):
        return chat(messages, system=system, provider=args.provider, model=model, mock=mock,
                    effort="low", **kw)

    print(f"Model: {args.provider} / {model}\n")

    if args.interactive:
        history = []
        print("Free-chat mode (enter an empty line or press Ctrl-D to quit)")
        while True:
            try:
                text = input("\nYou: ").strip()
            except EOFError:
                break
            if not text:
                break
            history.append(user_message(text))
            print("Model: ", end="", flush=True)
            reply = ask(history, on_text=lambda t: print(t, end="", flush=True))
            history.append(assistant_message(reply))
            print(f"\n     (this turn sent {reply.usage['input_tokens']} tokens and received {reply.usage['output_tokens']})")
        return

    # ------------------------------------------------------------ 1. Stateless
    print("[1] The API doesn't remember you: you have to bring the 'memory' yourself")
    first = [user_message("My name is Sam.")]
    r1 = ask(first)
    print(f"   First request:")
    show_request(first)
    print(f"     model: {r1.text}")

    fresh = [user_message("What's my name?")]
    r2 = ask(fresh)
    print(f"   Second request (a brand-new request that only asks the question):")
    show_request(fresh)
    print(f"     model: {r2.text}")

    full = first + [assistant_message(r1), user_message("What's my name?")]
    r3 = ask(full)
    print(f"   Third request (sending the earlier conversation along with it):")
    show_request(full)
    print(f"     model: {r3.text}")
    print("   The model itself keeps no conversation at all. A chatbot on the web 'remembers' you only because the program resends the whole history every time.\n")

    # ------------------------------------------------------------ 2. Tokens and cost
    print("[2] A multi-turn conversation: every turn resends the whole history")
    questions = ["What is gradient descent?", "What happens if the learning rate is too big?", "And if it's too small?",
                 "How do I choose a good learning rate?", "What is Adam?", "How is it different from SGD?",
                 "When should I use SGD?", "Sum up what we talked about today."]
    history, total_in, total_out = [], 0, 0
    price_model = model if cost_usd(model, {"input_tokens": 0, "output_tokens": 0}) is not None else "claude-opus-5-5"
    print("   turn | sent this turn | received this turn | total sent | total cost")
    for i, q in enumerate(questions, 1):
        history.append(user_message(q))
        reply = ask(history)
        history.append(assistant_message(reply))
        total_in += reply.usage["input_tokens"]
        total_out += reply.usage["output_tokens"]
        cost = cost_usd(price_model, {"input_tokens": total_in, "output_tokens": total_out})
        print(f"   {i:4d} | {reply.usage['input_tokens']:14,} | {reply.usage['output_tokens']:18,} | {total_in:10,} | ${cost:.5f}")
    note = " (the mock model's token counts are estimates)" if args.provider == "mock" else ""
    print(f"   Cost estimated at {price_model}'s prices{note}.")
    print("""   Every turn sends more tokens than the one before: all the earlier questions and answers have to be sent again.
   The longer the conversation, the more expensive and slower each turn gets, until it hits the limit of the "context window."
   That's why Chapter 7 covers "context management": what to keep, what to delete, and what to compress into a summary.
""")

    # ------------------------------------------------------------ 3. Streaming
    print("[3] Streaming: no need to wait for the whole thing; show it while it's being generated (| marks each small chunk)")
    print("   model: ", end="")
    ask([user_message("Explain what a token is in one sentence.")], on_text=lambda t: print(t + "|", end="", flush=True))
    print("\n   A large model generates text one token at a time. Streaming lets the user see the beginning sooner, which feels much faster.")

    print("""
Think about it:
  1. In a 100-turn conversation, roughly how many times as many tokens does turn 100 send as turn 1? How does the total cost relate to the number of turns?
  2. If you deleted part of the much earlier conversation before sending, what would be the upsides and downsides?
  3. Chat with a real model for a bit using --interactive, then ask it "What did we just talk about?" Start the program again, ask the same question, and compare.""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:          # missing SDK, no key set and the like: explain in plain words
        raise SystemExit(f"Something went wrong: {e}")
