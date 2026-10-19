"""
Lab 4: The writer–reviewer loop: one writes the code, one finds what's wrong with it
====================================================================================

In *Building effective agents* (December 2024), Anthropic calls this pattern the "evaluator-optimizer":
one agent writes, another evaluates and gives feedback, and the loop repeats until the work passes.

Who does what here:
  · The writer (a model): writes the function is_palindrome(s), based on the requirements and the last round's feedback
  · The reviewer's first check (a program): runs the tests in a **subprocess** with a timeout and resource limits,
    and feeds the failing cases back to the writer exactly as they are
  · The reviewer's second check (a model): once all the tests pass, takes a look from angles the tests can't measure, such as readability
  · The brakes: at most 5 rounds; if the writer hands in exactly the same code as last round, it's going in circles,
    so stop right away and hand it to a human

Run:
  python write_review.py                     # mock model: the writer revises three times, following the script
  python write_review.py --provider claude   # real model: a model plays the writer (note: this runs the model's code on your machine)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import MockLLM, add_provider_args, chat, user_message  # noqa: E402

SPEC = ("Write a Python function is_palindrome(s) that checks whether a string is a palindrome. "
        "It must ignore case, spaces and punctuation, work for Chinese too, and treat the empty string as a palindrome.")
CASES = [                                   # the reviewer's test cases; the writer can't see them
    ("上海自来水来自海上", True),
    ("A man, a plan, a canal: Panama", True),
    ("race a car", False),
    ("", True),
    ("Was it a car or a cat I saw?", True),
    ("ab", False),
    (".,!", True),
    ("1a2", False),
]
TIMEOUT = 2                                  # seconds
MAX_ROUNDS = 5

# ---------------------------------------------------------------- the writer's script (mock model)
V1 = '''def is_palindrome(s):
    return s == s[::-1]'''
V2 = '''def is_palindrome(s):
    i, j = 0, len(s) - 1
    while i < j:
        if not s[i].isalnum():
            continue
        if not s[j].isalnum():
            j -= 1
            continue
        if s[i].lower() != s[j].lower():
            return False
        i += 1
        j -= 1
    return True'''
V3 = '''def is_palindrome(s):
    chars = [ch.lower() for ch in s if ch.isalnum()]
    return chars == chars[::-1]'''

DILIGENT = [
    f"The simplest way: reverse the string and compare.\n```python\n{V1}\n```",
    f"The feedback says case and punctuation aren't handled. I'll switch to two pointers that walk in from both ends "
    f"and skip anything that isn't a letter or digit.\n```python\n{V2}\n```",
    f"The feedback says the 2nd test case timed out. When the left pointer hit punctuation, I only wrote continue "
    f"and forgot to move it forward, so it looped forever. "
    f"This time I'll use a way that's harder to get wrong: pick out the letters and digits, lowercase them, "
    f"and compare them with their reverse.\n```python\n{V3}\n```",
]
STUBBORN = [f"I think this is right.\n```python\n{V1}\n```"] * MAX_ROUNDS


# ---------------------------------------------------------------- the reviewer's first check: run the tests in a subprocess
RUNNER = '''
import json, sys
CASES = json.loads(sys.argv[1])
for s, want in CASES:
    try:
        got = is_palindrome(s)
        print(json.dumps([s, want, got, None], ensure_ascii=False), flush=True)
    except Exception as e:
        print(json.dumps([s, want, None, repr(e)], ensure_ascii=False), flush=True)
'''


def limit_resources():
    """(Only takes effect on Linux / macOS.) Limits the subprocess's CPU time and memory."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT + 1, TIMEOUT + 1))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 ** 2, 512 * 1024 ** 2))
    except (ImportError, ValueError, OSError):
        pass


def run_tests(code):
    """Returns (whether everything passed, a summary, the feedback for the writer)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(code + "\n" + RUNNER, encoding="utf-8")
        cmd = [sys.executable, "-I", str(path), json.dumps(CASES, ensure_ascii=False)]
        kwargs = {"preexec_fn": limit_resources} if os.name == "posix" else {}
        env = {"PYTHONIOENCODING": "utf-8"}                 # don't pass your environment variables (such as API keys) to the subprocess
        if "SYSTEMROOT" in os.environ:                      # on Windows, Python can't start without it
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                                  timeout=TIMEOUT, env=env, **kwargs)
            out, err, timed_out = proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err, timed_out = "", True
    results = [json.loads(line) for line in out.splitlines() if line.startswith("[")]
    problems = []
    for s, want, got, error in results:
        if error:
            problems.append(f"is_palindrome({s!r}) raised an exception: {error}")
        elif got != want:
            problems.append(f"is_palindrome({s!r}) should return {want}, but returned {got}")
    if timed_out:
        hung = CASES[len(results)][0]
        problems.append(f"is_palindrome({hung!r}) didn't return within {TIMEOUT} seconds, probably an infinite loop; "
                        "the remaining cases never got tested")
    elif len(results) < len(CASES):
        problems.append("The code didn't run to the end: " + (err.strip().splitlines() or ["(no error message)"])[-1])
    passed = len(results) - sum(1 for r in results if r[3] or r[2] != r[1])
    summary = f"{passed}/{len(CASES)} tests passed"
    return not problems, summary, problems


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------- the writer–reviewer loop
def write_review_loop(writer_mock, provider, model):
    feedback, last_code = None, None
    for rnd in range(1, MAX_ROUNDS + 1):
        prompt = (f"Requirements: {SPEC}\nOutput only one ```python code block; "
                  "you may explain your approach in a sentence or two before it.")
        if feedback:
            prompt += (f"\n\nYour previous version:\n```python\n{last_code}\n```\n"
                       f"The reviewer's feedback:\n{feedback}\nPlease revise it.")
        reply = chat([user_message(prompt)], provider=provider, model=model, mock=writer_mock)
        code = extract_code(reply.text)
        note = reply.text.split("```")[0].strip()
        print(f"\n✍️  Version {rnd}  Writer: {note}")
        print("    " + code.replace("\n", "\n    "))
        if code == last_code:
            print("🛑 This version is exactly the same as the last one: the writer is going in circles, and looping "
                  "any longer would only waste tokens. Stopping and handing it to a human.")
            return False
        ok, summary, problems = run_tests(code)
        print(f"🔍 Reviewer (running the tests): {summary}")
        for p in problems:
            print(f"     ✗ {p}")
        last_code = code
        if ok:
            review = chat([user_message(f"Requirements: {SPEC}\nThe code has passed all the tests:\n```python\n{code}\n```\n"
                                        "Give a short review (two or three sentences) on readability and edge cases.")],
                          provider=provider, model=model, effort="low",
                          mock=MockLLM(["Very direct: filter first, then compare, so you can tell at a glance whether it's right. "
                                        "It's O(n) time with O(n) extra memory; for very long strings, two pointers "
                                        "could save memory, but watch out for an infinite loop like the last version's."]))
            print(f"🧐 Reviewer (model's review): {review.text.strip()}")
            print(f"✅ Version {rnd} passes; the loop is done.")
            return True
        feedback = "\n".join(problems)
    print(f"🛑 Still not passing after {MAX_ROUNDS} rounds; stopping and handing it to a human.")
    return False


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="The writer–reviewer loop"))
    args = parser.parse_args()
    if args.provider != "mock":
        print("⚠️  Real mode runs the code the model writes in a subprocess on your machine. The subprocess has a timeout "
              "and resource limits, but it is not a real sandbox (see Chapter 7).\n")
    print(f"Requirements: {SPEC}")
    print("\n================ Scenario 1: a diligent writer ================")
    write_review_loop(MockLLM(DILIGENT), args.provider, args.model)
    if args.provider == "mock":
        print("\n================ Scenario 2: a stubborn writer (hands in the same code every time) ================")
        write_review_loop(MockLLM(STUBBORN), args.provider, args.model)

    print("""
Reading the results (based on the mock model's script):
  * Version 1: the feedback isn't just "there's a bug"; it says exactly which input, what was expected and what came back.
    The more specific the feedback, the faster the fix.
  * Version 2: the writer wrote an infinite loop. If the tests didn't run in a subprocess with a timeout,
    the whole review process would hang. When a reviewer runs code written by someone else (especially by a model),
    a timeout, resource limits and an isolated working directory are all a must.
  * Version 3: only after all the tests pass do we ask the model for a review. Whatever a program can judge
    right or wrong, don't leave to a model's "feel."
  * Scenario 2: without "going-in-circles detection," the loop would run all the way to the round limit and waste money.
    The 2025 MAST study summed up 14 ways multi-agent systems fail, and "step repetition" is one of them.

Think about it:
  1. If the writer could see all the test cases, it might write code like if s == "race a car": return False
     to "fool" the tests. How do you prevent that?
  2. What if the reviewer's tests themselves are wrong? Who reviews the reviewer?
  3. If the writer and the reviewer were models from two different vendors, would that be better than one model
     writing and reviewing its own work? Why?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
