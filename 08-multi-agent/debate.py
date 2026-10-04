"""
实验 3：多 agent 辩论：三个臭皮匠，什么时候能顶个诸葛亮？
==========================================================

2023 年 MIT 和 Google Brain 的论文（Du 等）提出"多 agent 辩论"：
  第 1 轮：几个 agent 各自独立回答
  第 2 轮：每个 agent 看到别人的答案和理由，决定坚持还是修改
  最后：由裁判（或多数票）给出最终答案

题目用的是心理学家 Shane Frederick 2005 年"认知反射测试"里最有名的一道：
  "一个球拍和一个球一共 1.10 美元，球拍比球贵 1.00 美元。球多少钱？"
直觉会脱口而出 0.10 美元，正确答案是 0.05 美元。

我们给假模型设定两种"性格"，看看辩论在什么情况下有用：
  · 验算型：先把每个候选答案代回题目算一遍，谁算得通就信谁
  · 从众型：看大多数人怎么说，就跟着怎么说

运行：
  python debate.py                     # 假模型
  python debate.py --provider claude   # 真实模型（三个 agent 都用同一个模型，很可能第一轮就全答对）
"""

import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import MockLLM, add_provider_args, chat, user_message  # noqa: E402

QUESTION = "一个球拍和一个球一共 1.10 美元，球拍比球贵 1.00 美元。球多少钱？"
ANSWER = 0.05
FORMAT = "请简短地写出理由，最后一行只写：答案：X 美元"

FIRST_ROUND = {   # 假模型第 1 轮的回答：两个跟着直觉走，一个认真列了方程
    "A": "1.10 减去 1.00 等于 0.10，所以球是 0.10 美元。\n答案：0.10 美元",
    "B": "设球是 x 美元，球拍就是 x + 1.00，两者相加 2x + 1.00 = 1.10，所以 x = 0.05。\n"
         "验算：0.05 + 1.05 = 1.10 ✓\n答案：0.05 美元",
    "C": "这题一眼就能看出来，球是 0.10 美元。\n答案：0.10 美元",
}


def extract(text):
    found = re.findall(r"答案[：:]\s*([\d.]+)", text)
    return float(found[-1]) if found else None


def show(x):
    return "（没给出答案）" if x is None else f"{x:.2f}"


def satisfies(ball):
    """把候选答案代回题目：球 + 球拍是不是 1.10？"""
    return abs(ball + (ball + 1.00) - 1.10) < 1e-9


# ---------------------------------------------------------------- 两种"性格"的假模型
def mock_agent(name, style):
    def respond(messages, tools):
        prompt = messages[-1]["content"]
        if "其他人的回答" not in prompt:
            return FIRST_ROUND[name]
        mine = extract(prompt.split("你上一轮的回答")[1].split("其他人的回答")[0])
        others = [float(x) for x in re.findall(r"答案：([\d.]+)", prompt.split("其他人的回答")[1])]
        if style == "验算型":
            if satisfies(mine):
                return (f"我把自己的答案代回去：球 {mine:.2f} + 球拍 {mine + 1:.2f} = {2 * mine + 1:.2f} ✓，"
                        f"坚持原答案。\n答案：{mine:.2f} 美元")
            good = [x for x in others if satisfies(x)]
            if good:
                x = good[0]
                return (f"我把自己的答案代回去：球 {mine:.2f} + 球拍 {mine + 1:.2f} = {2 * mine + 1:.2f}，"
                        f"不是 1.10 ✗。别人的 {x:.2f}：{x:.2f} + {x + 1:.2f} = 1.10 ✓，我改答案。\n答案：{x:.2f} 美元")
            return f"我的答案验算不通过，别人的也不通过，暂时保留。\n答案：{mine:.2f} 美元"
        majority = Counter([mine] + others).most_common(1)[0][0]
        if majority != mine:
            return f"大多数人都说 {majority:.2f}，可能是我想错了。\n答案：{majority:.2f} 美元"
        return f"大家和我想的一样，那就没问题了。\n答案：{mine:.2f} 美元"
    return respond


def judge_mock(messages, tools):
    votes = Counter(float(x) for x in re.findall(r"答案：([\d.]+)", messages[-1]["content"]))
    best, n = votes.most_common(1)[0]
    return f"{n} 票对 {sum(votes.values()) - n} 票，采纳多数意见。\n答案：{best:.2f} 美元"


# ---------------------------------------------------------------- 辩论流程
def debate(style, provider, model, rounds=2):
    names = list(FIRST_ROUND)
    agents = {n: MockLLM(mock_agent(n, style)) for n in names}
    answers = {}
    for r in range(1, rounds + 1):
        print(f"   —— 第 {r} 轮 ——")
        new = {}
        for n in names:
            if r == 1:
                prompt = f"{QUESTION}\n{FORMAT}"
            else:
                others = "\n".join(f"【{m}】{answers[m]}" for m in names if m != n)
                prompt = (f"{QUESTION}\n\n你上一轮的回答：\n{answers[n]}\n\n其他人的回答：\n{others}\n\n"
                          f"请认真检查所有答案（包括你自己的），决定坚持还是修改。{FORMAT}")
            reply = chat([user_message(prompt)], provider=provider, model=model, mock=agents[n], effort="low")
            new[n] = reply.text.strip()
            first_line = new[n].splitlines()[0]
            print(f"   {n}：{first_line[:70]}{'……' if len(first_line) > 70 else ''}  → {show(extract(new[n]))}")
        answers = new
    finals = "\n".join(f"【{n}】{answers[n]}" for n in names)
    prompt = f"题目：{QUESTION}\n\n辩论结束后各方的最终回答：\n{finals}\n\n你是裁判，请给出最终答案。{FORMAT}"
    verdict = chat([user_message(prompt)], provider=provider, model=model, mock=MockLLM([judge_mock]), effort="low")
    result = extract(verdict.text)
    ok = result is not None and abs(result - ANSWER) < 1e-9
    print(f"   裁判：{verdict.text.strip().splitlines()[0]} → {show(result)}  {'✅ 对了' if ok else '❌ 错了'}")


# ---------------------------------------------------------------- 思想实验：大量重复
def simulate(p, reliability, trials=20000, n=3, seed=0):
    """每个 agent 第 1 轮独立答对的概率是 p；答错时都掉进同一个直觉陷阱。
    第 2 轮：每个 agent 以 reliability 的概率认真验算（能认出正确答案），否则跟着多数走。"""
    rng = random.Random(seed)
    single = vote = debated = 0
    for _ in range(trials):
        first = [rng.random() < p for _ in range(n)]
        single += first[0]
        majority = sum(first) * 2 > n
        vote += majority
        second = []
        for _ in range(n):
            if rng.random() < reliability:
                second.append(any(first))       # 认真验算：只要有人给出了正确答案，就能认出来
            else:
                second.append(majority)         # 从众：跟着第 1 轮的多数走
        debated += sum(second) * 2 > n
    return single / trials, vote / trials, debated / trials


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="多 agent 辩论"))
    args = parser.parse_args()
    print(f"题目：{QUESTION}\n")
    for style in ["验算型", "从众型"]:
        print(f"【场景】三个 agent 都是「{style}」" + ("" if args.provider == "mock" else "（真实模型下，性格由模型自己决定）"))
        debate(style, args.provider, args.model)
        print()

    print("""在假模型的剧本里，同样的第 1 轮（一对两错），只是性格不同，结局完全相反：
  · 验算型：A 和 C 把自己的 0.10 代回去发现算不通，改成 0.05，三票全对；
  · 从众型：B 明明是对的，却被"大多数人都说 0.10"说服了，最后一致认定了错误答案。
辩论有没有用，关键不在"人多"，而在每个人**能不能独立检验**别人的说法。
""")

    print("【思想实验】把上面的过程重复 20000 次（三个 agent，答错时都掉进同一个陷阱）：")
    print("   单个 agent 答对率 p | 单个 agent | 三人投票 | 辩论：全员验算 | 辩论：一半验算 | 辩论：全员从众")
    for p in [0.3, 0.5, 0.7]:
        single, vote, full = simulate(p, 1.0)
        _, _, half = simulate(p, 0.5)
        _, _, none = simulate(p, 0.0)
        print(f"           {p:.1f}          |   {single:5.1%}   |  {vote:5.1%}  |     {full:5.1%}     |"
              f"     {half:5.1%}     |     {none:5.1%}")
    print("""
   * 全员从众时，辩论的结果和"直接投票"一模一样：交流没有带来新信息。
   * 单个 agent 答对率不到一半时（p = 0.3），三人投票反而比一个人更差：
     这是 1785 年孔多塞"陪审团定理"的另一面——多数票只会放大每个人原本的倾向。
   * 只有当 agent 真的会验算，辩论才能把"少数人的正确答案"变成最终答案。
   * 这是一个按我们的假设算出来的思想实验，不是真实模型的测量结果。2025 年有研究
     （Choi 等，《Debate or Vote》）在七个基准上发现：多 agent 辩论的提升大部分其实来自多数投票本身。

想一想：
  1. 把三个 agent 换成三家不同厂商的模型，"答错时都掉进同一个陷阱"这个假设还成立吗？（第九章会用到）
  2. 给 agent 一个真正的计算器工具，它就能"验算"了。哪些问题没法这样验算？
  3. 如果第 1 轮三个人都答错了，再辩论几轮能变对吗？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
