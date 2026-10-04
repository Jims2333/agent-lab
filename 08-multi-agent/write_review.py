"""
实验 4：写-审循环：一个写代码，一个挑毛病
===========================================

Anthropic 在《Building effective agents》（2024 年 12 月）里把这种模式叫做"评估者-优化者"
（evaluator-optimizer）：一个 agent 负责写，另一个负责评估并给出反馈，循环到合格为止。

这里的分工：
  · 作者（模型）：根据需求和上一轮的反馈，写出 is_palindrome(s) 函数
  · 审稿人第一关（程序）：在**子进程**里跑测试，带超时和资源限制，把失败的用例原样反馈给作者
  · 审稿人第二关（模型）：测试全过以后，再从可读性等测试测不出来的角度看一眼
  · 刹车：最多 5 轮；如果作者交上来的代码和上一轮一模一样，说明在原地打转，立刻停下交给人类

运行：
  python write_review.py                     # 假模型：作者按剧本改三版
  python write_review.py --provider claude   # 真实模型：作者由模型扮演（注意：会在本机执行模型写的代码）
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

SPEC = ("写一个 Python 函数 is_palindrome(s)，判断字符串是不是回文。"
        "要求：忽略大小写、空格和标点符号；中文也要支持；空字符串算回文。")
CASES = [                                   # 审稿人手里的测试用例，作者看不到
    ("上海自来水来自海上", True),
    ("A man, a plan, a canal: Panama", True),
    ("race a car", False),
    ("", True),
    ("Was it a car or a cat I saw?", True),
    ("ab", False),
    ("。，！", True),
    ("1a2", False),
]
TIMEOUT = 2                                  # 秒
MAX_ROUNDS = 5

# ---------------------------------------------------------------- 作者的剧本（假模型）
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
    f"最简单的写法：把字符串反过来比较。\n```python\n{V1}\n```",
    f"反馈说大小写和标点没处理好。改成双指针，从两头往中间走，跳过不是字母数字的字符。\n```python\n{V2}\n```",
    f"反馈说第 2 个用例超时了。原来左指针遇到标点时只写了 continue，忘了往前挪，死循环了。"
    f"这次换个更不容易出错的写法：先把字母数字挑出来、统一小写，再和反转后的比较。\n```python\n{V3}\n```",
]
STUBBORN = [f"我觉得这样就对了。\n```python\n{V1}\n```"] * MAX_ROUNDS


# ---------------------------------------------------------------- 审稿人第一关：在子进程里跑测试
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
    """（只在 Linux / macOS 上生效）限制子进程的 CPU 时间和内存。"""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT + 1, TIMEOUT + 1))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 ** 2, 512 * 1024 ** 2))
    except (ImportError, ValueError, OSError):
        pass


def run_tests(code):
    """返回 (是否全部通过, 给作者看的反馈)。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(code + "\n" + RUNNER, encoding="utf-8")
        cmd = [sys.executable, "-I", str(path), json.dumps(CASES, ensure_ascii=False)]
        kwargs = {"preexec_fn": limit_resources} if os.name == "posix" else {}
        env = {"PYTHONIOENCODING": "utf-8"}                 # 不把你的环境变量（比如 API key）传给子进程
        if "SYSTEMROOT" in os.environ:                      # Windows 上没有它，Python 启动不了
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
            problems.append(f"is_palindrome({s!r}) 抛出了异常：{error}")
        elif got != want:
            problems.append(f"is_palindrome({s!r}) 应该返回 {want}，实际返回 {got}")
    if timed_out:
        hung = CASES[len(results)][0]
        problems.append(f"is_palindrome({hung!r}) 超过 {TIMEOUT} 秒没有返回，可能是死循环；后面的用例没来得及测")
    elif len(results) < len(CASES):
        problems.append("代码没能跑完：" + (err.strip().splitlines() or ["（没有错误信息）"])[-1])
    passed = len(results) - sum(1 for r in results if r[3] or r[2] != r[1])
    summary = f"通过 {passed}/{len(CASES)} 个测试"
    return not problems, summary, problems


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------- 写-审循环
def write_review_loop(writer_mock, provider, model):
    feedback, last_code = None, None
    for rnd in range(1, MAX_ROUNDS + 1):
        prompt = f"需求：{SPEC}\n只输出一段 ```python 代码块，前面可以用一两句话说明思路。"
        if feedback:
            prompt += f"\n\n你上一版的代码：\n```python\n{last_code}\n```\n审稿人的反馈：\n{feedback}\n请修改。"
        reply = chat([user_message(prompt)], provider=provider, model=model, mock=writer_mock)
        code = extract_code(reply.text)
        note = reply.text.split("```")[0].strip()
        print(f"\n✍️  第 {rnd} 版　作者：{note}")
        print("    " + code.replace("\n", "\n    "))
        if code == last_code:
            print("🛑 这一版和上一版一模一样：作者在原地打转，再循环下去只会浪费 token。停下来，交给人类。")
            return False
        ok, summary, problems = run_tests(code)
        print(f"🔍 审稿人（跑测试）：{summary}")
        for p in problems:
            print(f"     ✗ {p}")
        last_code = code
        if ok:
            review = chat([user_message(f"需求：{SPEC}\n代码已经通过了全部测试：\n```python\n{code}\n```\n"
                                        "请从可读性和边界情况的角度简短点评（两三句话）。")],
                          provider=provider, model=model, effort="low",
                          mock=MockLLM(["写法很直接：先过滤再比较，一眼就能看出对不对。"
                                        "复杂度 O(n)，额外占用 O(n) 内存；字符串很长时可以考虑双指针来省内存，"
                                        "但要小心上一版那样的死循环。"]))
            print(f"🧐 审稿人（模型点评）：{review.text.strip()}")
            print(f"✅ 第 {rnd} 版通过，循环结束。")
            return True
        feedback = "\n".join(problems)
    print(f"🛑 {MAX_ROUNDS} 轮还没通过，停下来交给人类。")
    return False


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="写-审循环"))
    args = parser.parse_args()
    if args.provider != "mock":
        print("⚠️  真实模式会在本机的子进程里执行模型写的代码。子进程有超时和资源限制，"
              "但这不是真正的沙箱（第七章讲过）。\n")
    print(f"需求：{SPEC}")
    print("\n================ 场景 1：认真的作者 ================")
    write_review_loop(MockLLM(DILIGENT), args.provider, args.model)
    if args.provider == "mock":
        print("\n================ 场景 2：固执的作者（每次都交同一份代码）================")
        write_review_loop(MockLLM(STUBBORN), args.provider, args.model)

    print("""
看懂这个结果（以假模型的剧本为例）：
  * 第 1 版：反馈不是一句"有 bug"，而是具体哪个输入、期望什么、实际得到什么。反馈越具体，改得越快。
  * 第 2 版：作者写出了死循环。如果测试不放在带超时的子进程里跑，整个审稿流程就卡死了。
    审稿人要执行别人（尤其是模型）写的代码时，超时、资源限制、隔离的工作目录都是必需的。
  * 第 3 版：测试全过以后才请模型点评。能用程序判断对错的，就不要交给模型去"感觉"。
  * 场景 2：没有"原地打转检测"的话，循环会一直跑到轮数上限，白白花钱。
    2025 年的 MAST 研究总结了多 agent 系统的 14 种失败方式，"重复步骤"（step repetition）就是其中之一。

想一想：
  1. 如果作者能看到全部测试用例，它可能会写出 if s == "race a car": return False 这样的代码来"骗过"测试。怎么防？
  2. 审稿人的测试本身写错了怎么办？谁来审审稿人？
  3. 把作者和审稿人换成两家不同厂商的模型，会比同一个模型自己写自己审更好吗？为什么？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
