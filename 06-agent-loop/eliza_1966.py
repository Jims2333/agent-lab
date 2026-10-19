"""
Lab 1: Mini ELIZA, the 1966 "therapist"
=======================================

In 1966, Joseph Weizenbaum at MIT wrote ELIZA. Its most famous script, DOCTOR, played a psychotherapist:
it found a keyword in what you said, dropped part of your words into a pre-written sentence pattern,
and tossed the question back to you. It understood **nothing** of what you said.
Yet, as Weizenbaum recalled, even his secretary asked him to leave the room so she could "chat" with ELIZA alone.

Here we build a small version in a few dozen lines. Once you see how it works, compare it with the agent in Lab 3:
  ELIZA: every rule is written by a person; it can only "talk," not "do"
  agent: the model decides the next step itself, and can call tools that change the outside world

Run:
  python eliza_1966.py                # watch a demo conversation
  python eliza_1966.py --interactive  # chat with it yourself
"""

import argparse
import random
import re

# Pronoun reflection: you say "my job", it answers "your job"
REFLECT = {"i": "you", "me": "you", "my": "your", "am": "are", "you": "me", "your": "my"}

# Rules: (keyword pattern, reply templates). {0} is replaced by the part cut out of what you said
RULES = [
    (r"I (?:feel|am feeling) (.+)", ["Why do you feel {0}?", "Do you often feel {0}?",
                                     "When you feel {0}, what do you do?"]),
    (r"I (?:want|would like) (.+)", ["What would it mean to you if you got {0}?", "Why do you want {0}?"]),
    (r"I am (?:so|very|really) (.+)", ["Why are you so {0}?", "What do you do when you are so {0}?"]),
    (r"I am (.+)", ["How long have you been {0}?", "Do you believe it is normal to be {0}?"]),
    (r"Because (.+)", ["Is that the real reason?", "Don't any other reasons come to mind?"]),
    (r".*\b(mother|father|family|parents)\b.*", ["Tell me more about your family.",
                                              "How do you get along with your {0}?"]),
    (r".*\b(computers?|machines?|programs?|AI)\b.*", ["Do computers worry you?",
                                                    "Do you think I can really understand you?"]),
    (r".*\b(sorry|apologize)\b.*", ["Please don't apologize."]),
    (r".+", ["Please go on.", "I see. What does that suggest to you?", "Hmm... can you tell me more?"]),
]


def reflect(text):
    # swap the pronouns one word at a time ("my" → "your", "me" → "you", ...)
    out = re.sub(r"[A-Za-z']+", lambda m: REFLECT.get(m.group().lower(), m.group()), text)
    return out.strip(".!?,;: ")


def pad(text, width):
    """Pad with spaces to a display width (a CJK character takes two columns)."""
    return text + " " * max(width - sum(2 if ord(c) > 127 else 1 for c in text), 0)


def respond(text, rng):
    for pattern, templates in RULES:
        m = re.match(pattern, text.strip(), re.IGNORECASE)
        if m:
            parts = [reflect(g) for g in m.groups()]
            return rng.choice(templates).format(*parts), pattern
    return "Please go on.", ".+"


def main():
    parser = argparse.ArgumentParser(description="Mini ELIZA")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    rng = random.Random(1966)

    if args.interactive:
        print("ELIZA: Hello, I am ELIZA. Tell me what has been on your mind lately. (Enter an empty line to quit)")
        while True:
            try:
                text = input("You:   ").strip()
            except EOFError:
                break
            if not text:
                break
            print("ELIZA: " + respond(text, rng)[0])
        return

    demo = ["I feel stressed about my studies lately.", "Because I want to learn how to build AI agents.",
            "I am a beginner.", "My mother thinks I should learn the basics first.",
            "Do you think AI will replace programmers?", "Sorry, I got off track."]
    print("A demo conversation (on the right: the rule ELIZA matched):\n")
    for line in demo:
        reply, rule = respond(line, rng)
        print(f"   You:   {line}")
        print(f"   ELIZA: {pad(reply, 51)} ← rule {rule}")
    print("""
It looks like it is listening carefully, but it only does three things:
  1. Find a keyword ("I feel", "because", "mother", ...)
  2. Cut out a piece of what you said and swap the pronouns (my → your)
  3. Drop it into a sentence pattern written in advance
It has no memory, doesn't understand what you mean, and can't do anything for you.

Weizenbaum later reflected on this in Computer Power and Human Reason (1976):
people can't help projecting understanding and feelings onto a simple program. This became known as the "ELIZA effect."
The warning is still useful today when you chat with a large language model.

Think about it:
  1. Type "I am telling you, it was a movie that scared me." What strange thing does ELIZA say? Why?
  2. Add a rule so that ELIZA can respond to "I hate ...".
  3. What is the most fundamental difference between ELIZA and the agent in Lab 3?
     (Hint: who decides the next step? Can it change the outside world?)""")


if __name__ == "__main__":
    main()
