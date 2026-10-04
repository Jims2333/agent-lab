"""
毕业项目：一支跨厂商的小团队
==============================

把前面几章的零件拼起来，让三家的模型分工完成一件真事：
  给本仓库做一张"学习路线卡片"：九章，每章一行，包括中文标题、英文标题、故事的时间跨度。

分工：
  · 编排者（claude）：把任务拆成九个子任务 ........................ 第八章 编排者-工作者
  · 九个调研员（deepseek），并行：用第六章的 ReAct agent 读每章的 README ... 第六章 agent 循环 + 工具
      工具只能读仓库里的文件 ...................................... 第七章 沙箱
  · 翻译员（gpt）：通过 A2A 协议把标题译成英文 ...................... 第九章 协议
  · 审稿员（claude）：先用程序检查格式，不合格的打回去重做，再请模型点评 ... 第八章 写-审循环
  · 全程记账：每家调用了几次、用了多少 token、大概花了多少钱 .......... 第七章 预算

假模型的剧本里，故意安排了一个"偷懒"的调研员，看看审稿员能不能抓住它。

运行：
  python capstone.py            # 假模型
  python capstone.py --live     # 配置了 key 的厂商用真实模型（会调用几十次 API）
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from team import ROOT, add_live_arg, ask, fit, label, model_for, provider_for, roster
from a2a_mini import A2AServer, fetch_card, send_text, task_text
from router import PRICES
from common.llm import MockLLM

sys.path.insert(0, str(ROOT / "06-agent-loop"))
from react_agent import run_agent  # noqa: E402

CHAPTERS = sorted(p.name for p in ROOT.iterdir() if re.match(r"0\d-", p.name))
TASK = "给 agent-lab 仓库做一张学习路线卡片：每章一行，包括中文标题、英文标题、故事的时间跨度。"
LINE = re.compile(r"^第.+章｜.+｜\d{4}–\d{4}$")
SLOPPY = 4                                  # 剧本里第 5 个调研员会偷懒，只读开头 5 行
TITLE_EN = {                                # 假翻译员认识的标题
    "机器学习从哪里来": "Where Machine Learning Came From",
    "线性模型与梯度下降": "Linear Models and Gradient Descent",
    "神经网络与自动求导": "Neural Networks and Automatic Differentiation",
    "从 CNN 到 Transformer": "From CNNs to Transformers",
    "大语言模型与 API": "Large Language Models and APIs",
    "Agent 循环": "The Agent Loop",
    "Harness 工程": "Harness Engineering",
    "多 Agent 协同": "Multi-Agent Collaboration",
    "跨模型协作": "Cross-Model Collaboration",
}


# ---------------------------------------------------------------- 账本（第七章：预算）
class Ledger:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = {}

    def add(self, vendor, usage, calls=1):
        with self.lock:
            row = self.rows.setdefault(vendor, {"calls": 0, "in": 0, "out": 0})
            row["calls"] += calls
            row["in"] += usage["input_tokens"]
            row["out"] += usage["output_tokens"]

    def report(self, live):
        total = 0.0
        print(f"   {fit('厂商', 22)}| 调用次数 | 输入 token | 输出 token | 估算费用（美元）")
        for vendor, r in self.rows.items():
            p_in, p_out = PRICES[vendor]
            cost = r["in"] / 1e6 * p_in + r["out"] / 1e6 * p_out
            total += cost
            print(f"   {fit(label(vendor, live), 22)}| {r['calls']:>6}   | {r['in']:>9,}  | {r['out']:>9,}  | {cost:.5f}")
        print(f"   {fit('合计', 22)}|          |            |            | {total:.5f}")


LEDGER = Ledger()


# ---------------------------------------------------------------- 编排者
def plan(live):
    prompt = (f"{TASK}\n仓库里的章节目录：{', '.join(CHAPTERS)}。\n"
              "请列出需要调研的章节目录，每章一个子任务。只输出 JSON 字符串数组。")
    try:
        reply = ask("claude", prompt, live, MockLLM([json.dumps(CHAPTERS)]))
        LEDGER.add("claude", reply.usage)
        dirs = [d for d in json.loads(re.search(r"\[.*\]", reply.text, re.S).group(0)) if d in CHAPTERS]
    except Exception:                       # 调用失败或格式不对
        dirs = []
    return dirs or CHAPTERS                 # 模型没按格式输出时的兜底


# ---------------------------------------------------------------- 调研员（第六章的 ReAct agent）
def researcher_policy(chapter, lines):
    def read(messages, tools):
        return {"text": f"读 {chapter} 的开头。",
                "tool_calls": [{"name": "read_file", "args": {"path": f"{chapter}/README.md", "start": 1, "end": lines}}]}

    def summarize(messages, tools):
        text = next(m for m in reversed(messages) if m["role"] == "tool")["content"]
        title = re.search(r"# (第.+?章)　(.+)", text)
        cells = re.findall(r"^\d+: \| ([^|]*?\d{4}[^|]*?) \|", text, flags=re.M)
        years = [int(y) for c in cells for y in re.findall(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", c)]
        span = f"{min(years)}–{max(years)}" if years else "？"
        if not title:
            return f"{chapter}｜（没读到标题）｜{span}"
        return f"{title.group(1)}｜{title.group(2).split('：')[0]}｜{span}"
    return [read, summarize]


def research(i, chapter, live, feedback=None):
    task = (f"读 {chapter}/README.md 的开头（标题和时间线表格），只输出一行："
            "第X章｜章节主标题（冒号前面的部分）｜最早年份–最晚年份")
    if feedback:
        task += f"\n上一次的结果被审稿员打回了：{feedback}"
    lines = 5 if i == SLOPPY and not feedback else 45
    provider = provider_for("deepseek", live)
    try:
        answer, messages, usage = run_agent(task, provider, model_for("deepseek") if provider != "mock" else None,
                                            mock=MockLLM(researcher_policy(chapter, lines)), max_steps=6,
                                            verbose=False)
    except Exception as e:                  # 一个调研员出故障，不能拖垮整个团队
        return f"{chapter}｜（调用失败：{type(e).__name__}）｜？"
    LEDGER.add("deepseek", usage, calls=sum(m["role"] == "assistant" for m in messages))
    return answer.strip().splitlines()[-1] if answer.strip() else ""


# ---------------------------------------------------------------- 翻译员（A2A 服务器，背后是 gpt）
def translator_handler(live):
    def handler(text):
        source = text.split("：", 1)[-1].strip()
        try:
            reply = ask("gpt", f"把这个书的章节标题翻译成英文，只输出译文：\n{source}", live,
                        MockLLM([TITLE_EN.get(source, "(unknown title)")]))
        except Exception as e:
            return "TASK_STATE_FAILED", f"翻译员背后的模型调用失败：{type(e).__name__}"
        LEDGER.add("gpt", reply.usage)
        return "TASK_STATE_COMPLETED", reply.text.strip()
    return handler


# ---------------------------------------------------------------- 审稿员
def review(lines):
    """程序能检查的先用程序检查：格式、有没有时间跨度。返回不合格的 {序号: 原因}。"""
    problems = {}
    for i, line in enumerate(lines):
        if not LINE.match(line):
            problems[i] = f"「{line}」格式不对或者没有找到时间跨度，请把时间线表格读完整"
    return problems


def main():
    parser = add_live_arg(argparse.ArgumentParser(description="毕业项目：跨厂商小团队"))
    args = parser.parse_args()
    live = args.live
    print(f"任务：{TASK}\n团队：{roster(live)}\n")

    dirs = plan(live)
    print(f"【1】编排者 {label('claude', live)} 拆出了 {len(dirs)} 个子任务。")

    print(f"\n【2】{len(dirs)} 个调研员 {label('deepseek', live)} 并行开工……")
    with ThreadPoolExecutor(max_workers=len(dirs)) as pool:
        lines = list(pool.map(lambda a: research(*a, live), enumerate(dirs)))
    for line in lines:
        print(f"   · {line}")

    print(f"\n【3】审稿员 {label('claude', live)} 先用程序检查格式……")
    for round_ in range(1, 3):
        problems = review(lines)
        if not problems:
            print("   ✅ 全部合格。")
            break
        for i, why in problems.items():
            print(f"   ❌ 第 {i + 1} 个调研员：{why}。打回重做。")
            lines[i] = research(i, dirs[i], live, feedback=why)
            print(f"      重做后：{lines[i]}")
    else:
        print("   ⚠️  两轮之后仍有不合格的，交给人类处理。")

    print(f"\n【4】通过 A2A 把标题交给翻译员 {label('gpt', live)}……")
    server = A2AServer("翻译员", "把中文章节标题译成英文", [
        {"id": "translate-title", "name": "标题翻译", "description": "中文标题译成英文", "tags": ["翻译"]}],
        translator_handler(live)).start()
    try:
        card = fetch_card(server.base_url)
        print(f"   取到名片：{card['name']}，技能 {card['skills'][0]['id']}")
        english = []
        for line in lines:
            parts = line.split("｜")
            title = parts[1] if len(parts) == 3 else line
            state, text = task_text(send_text(card, f"翻译：{title}"))
            english.append(text if state == "TASK_STATE_COMPLETED" else "（翻译失败）")
    finally:
        server.stop()
    print(f"   完成 {len(english)} 个标题。")

    bad = len(review(lines))
    canned = ("各章齐全，时间跨度都有；前五章是一条从统计学习到大模型的主线，第六章开始转向 agent，顺序合理。"
              if not bad else f"有 {bad} 章的信息还不完整，建议人工补上后再发布。")
    try:
        comment = ask("claude", "下面是一张学习路线卡片，请用一两句话点评它是否完整、有没有明显的问题：\n"
                      + "\n".join(lines), live, MockLLM([canned]))
        LEDGER.add("claude", comment.usage)
        comment = comment.text.strip()
    except Exception as e:
        comment = f"（点评失败：{type(e).__name__}）"
    print(f"\n【5】审稿员点评：{comment}")

    print("\n🎓 学习路线卡片\n")
    print("| 章 | 标题 | English | 故事的时间跨度 |\n|---|---|---|---|")
    for line, en in zip(lines, english):
        parts = line.split("｜") + ["", "", ""]
        print(f"| {parts[0]} | {parts[1]} | {en} | {parts[2]} |")

    print("\n💰 账单（假模型的 token 数是估算的；价格见 router.py）")
    LEDGER.report(live)

    print("""
🗺️  这个小项目用到了整个仓库的哪些东西：

   用户的任务
    └─ 编排者（claude）：拆任务 ................................ 第八章 编排者-工作者
        ├─ 九个调研员（deepseek），并行 ........................ 第六章 ReAct 循环与工具
        │    └─ read_file 只能读仓库里的文件 .................. 第七章 沙箱
        ├─ 审稿员（claude）：程序检查 → 打回重做 → 模型点评 ..... 第八章 写-审循环
        └─ 翻译员（gpt）：素不相识，靠名片和 SendMessage 合作 ... 第九章 A2A
   贯穿全程：统一的 chat() 适配层（第五、九章）；账本（第七章 预算）；
   而模型本身，是第一章到第五章讲的那台"猜下一个字"的机器。

想一想：
  1. 为什么让 deepseek 当调研员、claude 当审稿员，而不是反过来？换过来会怎样？（提示：价格和调用次数）
  2. 审稿员如果也是 deepseek，和"自己审自己"有什么区别？（回想交叉核对实验里的"错误相关性"）
  3. 把翻译员换成别人部署在网上的 A2A agent，这个程序要改哪几行？又要多防哪些风险？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
