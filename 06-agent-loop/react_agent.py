"""
实验 3：从零写一个 ReAct agent
================================

2022 年的 ReAct 论文提出了一个简单的循环，今天几乎所有 agent 都是它的变体：

    while 还没完成:
        思考（Reason）：下一步该做什么？
        行动（Act）：   调用一个工具
        观察（Observe）：看工具返回了什么，然后继续思考

这里用不到 100 行实现这个循环，配上 tools.py 里的四个工具。
默认用假模型演示一个完整的任务；它的剧本会读取每一步的观察结果，也会犯错再改正，和真模型的行为方式一样。
换成真实模型后，你可以给它任何关于本仓库的问题。

运行：
  python react_agent.py                                # 假模型，演示内置任务
  python react_agent.py --provider claude --task "第三章用的是哪种激活函数？"
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm import (MockLLM, add_provider_args, assistant_message, chat,  # noqa: E402
                        tool_message, user_message)
from tools import TOOLS, run_tool  # noqa: E402

SYSTEM = """你是一个研究助手，可以使用工具查阅 agent-lab 这个学习仓库。
工作方式：每一步先用一两句话说明你的想法，再调用工具；拿到足够的信息后，直接给出最终答案，并注明出处（文件和行号）。
不要编造仓库里没有的内容。"""

DEFAULT_TASK = "第一章里说，AlexNet 的 top-5 错误率比第二名低了多少个百分点？请先在仓库里查到原文，再用计算器算出来。"


# ---------------------------------------------------------------- 假模型的剧本
def last_observation(messages):
    return next(m for m in reversed(messages) if m["role"] == "tool")["content"]


def step_guess_file(messages, tools):
    return {"text": "我先猜一下：第一章可能有一个专门讲 AlexNet 的文件。",
            "tool_calls": [{"name": "read_file", "args": {"path": "01-ml-origins/alexnet.md"}}]}


def step_search(messages, tools):
    return {"text": "文件不存在，猜错了。那我直接在整个仓库里搜索关键词。",
            "tool_calls": [{"name": "search_repo", "args": {"query": "top-5 错误率"}}]}


def step_calculate(messages, tools):
    obs = last_observation(messages)
    line = next(l for l in obs.splitlines() if "AlexNet" in l and "第二名" in l)
    ours, second = re.findall(r"(\d+(?:\.\d+)?)%", line)[:2]
    return {"text": f"找到了：AlexNet 是 {ours}%，第二名是 {second}%。用计算器算差值。",
            "tool_calls": [{"name": "calculator", "args": {"expression": f"{second} - {ours}"}}]}


def step_answer(messages, tools):
    found = next(m["content"] for m in messages if m["role"] == "tool" and "第二名" in m["content"])
    source = next(l for l in found.splitlines() if "AlexNet" in l and "第二名" in l).split(": ")[0]
    diff = last_observation(messages)
    return f"AlexNet 的 top-5 错误率比第二名低了 {diff} 个百分点（出处：{source}）。"


DEMO_SCRIPT = [step_guess_file, step_search, step_calculate, step_answer]


# ---------------------------------------------------------------- agent 循环本身
def run_agent(task, provider="mock", model=None, mock=None, max_steps=8, verbose=True):
    messages = [user_message(task)]
    usage = {"input_tokens": 0, "output_tokens": 0}
    for step in range(1, max_steps + 1):
        reply = chat(messages, system=SYSTEM, tools=TOOLS, provider=provider, model=model,
                     mock=mock, effort="medium")
        messages.append(assistant_message(reply))
        usage = {k: usage[k] + reply.usage[k] for k in usage}
        if verbose and reply.text and reply.tool_calls:
            print(f"   🤔 第 {step} 步 思考：{reply.text}")

        if not reply.tool_calls:                    # 不再调用工具 = 给出了最终答案
            return reply.text, messages, usage

        for call in reply.tool_calls:               # 执行模型要求的每一个工具调用
            result, is_error = run_tool(call["name"], call["args"])
            messages.append(tool_message(call, result, is_error))
            if verbose:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                print(f"   🔧        行动：{call['name']}({args})")
                lines = result.splitlines()
                more = f"  ……（共 {len(lines)} 行）" if len(lines) > 2 else ""
                print(f"   {'❌' if is_error else '👀'}        观察：" + " / ".join(lines[:2])[:110] + more)
    return f"（达到了 {max_steps} 步的上限，任务没有完成。）", messages, usage


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="从零写一个 ReAct agent"))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    if args.provider == "mock" and args.task != DEFAULT_TASK:
        raise SystemExit("假模型只会演示内置任务。想问别的问题，请加上 --provider claude / gpt / deepseek。")

    print(f"任务：{args.task}\n")
    answer, messages, usage = run_agent(args.task, args.provider, args.model,
                                        mock=MockLLM(DEMO_SCRIPT), max_steps=args.max_steps)
    print(f"\n   ✅ 最终答案：{answer}\n")

    roles = " → ".join({"user": "用户", "assistant": "模型", "tool": "工具结果"}[m["role"]] for m in messages)
    print(f"整个过程中，对话历史是这样一条一条长出来的：\n   {roles}")
    print(f"   一共 {len(messages)} 条消息，调用模型 {sum(m['role'] == 'assistant' for m in messages)} 次，"
          f"累计发送 {usage['input_tokens']} 个 token。")
    print("""
   每一轮，程序都把完整的历史（包括所有工具结果）重新发给模型（第五章讲过 API 是无状态的）。
   模型决定下一步做什么；程序负责执行工具、把结果写回历史、决定什么时候停下来。
   第二步那个"猜错文件名"很有代表性：agent 的价值不在于每一步都对，而在于能根据观察结果改正。

想一想：
  1. 用 --max-steps 2 再跑一次，会发生什么？为什么一定要有步数上限？
  2. 如果把第二步的错误信息改成只返回"错误"两个字，模型还能知道该怎么改正吗？
  3. 这个任务如果写成固定流程（先搜索、再计算、再回答），还需要 agent 吗？什么样的任务才值得用 agent？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
