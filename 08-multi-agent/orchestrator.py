"""
实验 1：编排者-工作者：一个主管，多个并行的帮手
=================================================

2025 年 Anthropic 公开的多 agent 研究系统，用的就是这个模式：
  1. 主 agent（编排者）把任务拆成几个互不依赖的子任务
  2. 为每个子任务启动一个子 agent（工作者），各自用**干净的上下文**并行去做
  3. 子 agent 只把简短的结论交回来，由主 agent 汇总

这里的任务是：调研本仓库第 1～7 章，每章用一句话概括它讲了哪个时间段的故事。
工作者直接复用第六章写的 ReAct agent（react_agent.run_agent）。
最后和"一个 agent 从头读到尾"做对比：速度、每个 agent 的上下文大小、总 token。

运行：
  python orchestrator.py                     # 假模型（每次调用模拟 0.3 秒的网络延迟）
  python orchestrator.py --provider claude   # 真实模型
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "06-agent-loop"))

from common.llm import MockLLM, add_provider_args, chat, user_message  # noqa: E402
from react_agent import run_agent  # noqa: E402

CHAPTERS = sorted(p.name for p in ROOT.iterdir() if re.match(r"0[1-7]-", p.name))
TASK = "调研 agent-lab 仓库的第 1～7 章：每章用一句话概括它讲了哪个时间段的故事、主题是什么。"
LATENCY = 0.3


class SlowMock(MockLLM):
    """假模型 + 模拟网络延迟，好让"并行"的效果看得出来。"""

    def __call__(self, messages, tools):
        time.sleep(LATENCY)
        return super().__call__(messages, tools)


# ---------------------------------------------------------------- 假模型的剧本
def worker_policy(chapter):
    """工作者：读这一章 README 的开头，从标题和时间线表格里提炼一句话。"""
    def read(messages, tools):
        return {"text": f"先读 {chapter} 的开头部分。",
                "tool_calls": [{"name": "read_file", "args": {"path": f"{chapter}/README.md", "start": 1, "end": 45}}]}

    def summarize(messages, tools):
        text = next(m for m in reversed(messages) if m["role"] == "tool")["content"]
        title = re.search(r"# (第.+?章)　(.+)", text)
        years = re.findall(r"^\d+: \| (\d{4}(?:[–-]\d{4})?)", text, flags=re.M)
        if not years:                                   # 有的章节表头写的是"时间"
            years = re.findall(r"^\d+: \| (\d{4}) 年", text, flags=re.M)
        span = f"{years[0][:4]}–{years[-1][-4:]}" if years else "（没找到时间线）"
        return f"{title.group(1)}《{title.group(2)}》：时间跨度 {span}，共 {len(years)} 个里程碑。"
    return [read, summarize]


def lead_plan_mock(messages, tools):
    return json.dumps([f"读 {c}/README.md，用一句话概括这一章的时间跨度和主题。" for c in CHAPTERS],
                      ensure_ascii=False)


def lead_summary_mock(messages, tools):
    reports = messages[-1]["content"]
    lines = re.findall(r"^子任务 \d+ 的结论：(.+)$", reports, flags=re.M)
    return "汇总：\n" + "\n".join(f"  · {l}" for l in lines)


# ---------------------------------------------------------------- 编排者
def plan(provider, model, mock):
    prompt = (f"{TASK}\n仓库里的章节目录是：{', '.join(CHAPTERS)}。\n"
              "请把任务拆成互相独立、可以并行完成的子任务，每章一个。只输出一个 JSON 字符串数组，不要输出别的内容。")
    reply = chat([user_message(prompt)], provider=provider, model=model, mock=mock, effort="low")
    try:
        subtasks = json.loads(re.search(r"\[.*\]", reply.text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        subtasks = [f"读 {c}/README.md，用一句话概括这一章。" for c in CHAPTERS]   # 模型没按格式输出时的兜底
    return subtasks, reply.usage


def run_worker(i, subtask, provider, model):
    mock = SlowMock(worker_policy(CHAPTERS[i])) if provider == "mock" else None
    t0 = time.perf_counter()
    answer, messages, usage = run_agent(subtask, provider, model, mock=mock, max_steps=6, verbose=False)
    return {"id": i + 1, "answer": answer, "steps": sum(m["role"] == "assistant" for m in messages),
            "tokens": usage["input_tokens"] + usage["output_tokens"], "context": len(json.dumps(
                [m["content"] for m in messages], ensure_ascii=False)) // 2, "seconds": time.perf_counter() - t0}


def synthesize(reports, provider, model, mock):
    body = "\n".join(f"子任务 {r['id']} 的结论：{r['answer']}" for r in reports)
    prompt = f"原始任务：{TASK}\n下面是各个子 agent 交回来的结论：\n{body}\n\n请汇总成一份按章节排列的简短清单。"
    reply = chat([user_message(prompt)], provider=provider, model=model, mock=mock, effort="low")
    return reply.text, reply.usage


# ---------------------------------------------------------------- 对照组：一个 agent 从头读到尾
def single_agent_policy():
    steps = [{"text": f"读 {c}。", "tool_calls": [{"name": "read_file", "args": {"path": f"{c}/README.md", "start": 1, "end": 45}}]}
             for c in CHAPTERS]
    steps.append("（一个 agent 读完了全部七章，在同一个上下文里写出汇总。）")
    return steps


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="编排者-工作者"))
    args = parser.parse_args()
    lead_mock = SlowMock([lead_plan_mock, lead_summary_mock])

    print(f"任务：{TASK}\n")
    t0 = time.perf_counter()
    subtasks, plan_usage = plan(args.provider, args.model, lead_mock)
    print(f"【1】编排者拆出了 {len(subtasks)} 个子任务：")
    for i, s in enumerate(subtasks, 1):
        print(f"   {i}. {s}")

    print(f"\n【2】{len(subtasks)} 个工作者同时开工（各自独立的上下文）……")
    with ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
        reports = list(pool.map(lambda a: run_worker(*a, args.provider, args.model), enumerate(subtasks)))
    for r in reports:
        print(f"   工作者 {r['id']}：{r['steps']} 步，{r['seconds']:.1f} 秒，用了约 {r['tokens']} token → {r['answer']}")

    summary, sum_usage = synthesize(reports, args.provider, args.model, lead_mock)
    multi_time = time.perf_counter() - t0
    multi_tokens = sum(r["tokens"] for r in reports) + sum(plan_usage.values()) + sum(sum_usage.values())
    print(f"\n【3】编排者汇总：\n{summary}")

    print("\n【4】对照组：同样的任务，让一个 agent 自己从头读到尾")
    t0 = time.perf_counter()
    mock = SlowMock(single_agent_policy()) if args.provider == "mock" else None
    _, messages, usage = run_agent(TASK, args.provider, args.model, mock=mock, max_steps=12, verbose=False)
    single_time = time.perf_counter() - t0
    single_tokens = usage["input_tokens"] + usage["output_tokens"]
    single_context = len(json.dumps([m["content"] for m in messages], ensure_ascii=False)) // 2
    biggest_worker = max(r["context"] for r in reports)

    print("                  | 用时    | 总 token | 单个 agent 的上下文最大约")
    print(f"   多 agent       | {multi_time:5.1f} 秒 | {multi_tokens:7,} | {biggest_worker:,} token")
    print(f"   单个 agent     | {single_time:5.1f} 秒 | {single_tokens:7,} | {single_context:,} token")
    print(f"""
   * 速度：工作者是并行的，总用时接近"最慢的那一个"，而不是所有人加起来；
   * 上下文：每个工作者只装着自己那一章，单个 agent 却要把七章全装进同一个上下文；
   * 代价：这个例子里多 agent 的总 token 反而更少，因为单个 agent 每一步都要把越来越长的历史重发一遍。
     但真实的工作者往往要自己探索很多步、彼此还会做重复的搜索，再加上拆分、汇总和每份系统提示，
     总账通常更贵：Anthropic 报告他们的多 agent 系统用的 token 约为普通聊天的 15 倍。
   所以多 agent 适合"可以拆成独立部分、而且每部分信息量都很大"的任务，比如大范围的调研。

想一想：
  1. 如果第 5 章的工作者需要第 4 章工作者的结论才能开始，还能这样并行吗？
  2. 工作者只交回一句话，会不会丢掉重要的细节？怎么权衡"交回多少"？
  3. 把工作者数量从 7 个改成 70 个（比如每节一个），会遇到什么新问题？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
