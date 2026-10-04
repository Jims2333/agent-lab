"""
实验 2：交叉核对：让不同厂商的模型互相检查
==========================================

第八章的辩论实验留了一个问题：几个 agent 用的是同一个模型时，它们往往会掉进同一个陷阱，投票也救不回来。
那么换成不同厂商的模型呢？它们的训练数据、训练方法都不一样，"盲区"也许不在同一个地方。

这里比较四种做法：
  A. 只问一家
  B. 同一家问三次，取多数
  C. 三家各答一次，取多数
  D. 两家各答一次：一致就放行，不一致就交给人复核

然后用一个思想实验说明：各家的错误越"相关"，多问几家的好处就越小。

运行：
  python cross_check.py            # 假模型（三家的"错题本"是我们设定的，不代表真实模型的表现）
  python cross_check.py --live     # 配置了 key 的厂商用真实模型
"""

import argparse
import random
import re
from collections import Counter
from statistics import NormalDist

from team import VENDORS, add_live_arg, ask, fit, provider_for, roster
from common.llm import MockLLM

QUESTIONS = [   # (题目, 正确答案)：都是流传很广的"陷阱题"
    ("一个球拍和一个球一共 1.10 美元，球拍比球贵 1.00 美元。球多少美元？", "0.05"),
    ("5 台机器 5 分钟能做 5 个零件，100 台机器做 100 个零件需要几分钟？", "5"),
    ("湖里有一片睡莲，面积每天翻一倍。48 天能盖满整个湖，盖满一半需要几天？", "47"),
    ("英文单词 strawberry 里有几个字母 r？", "3"),
    ("9.11 和 9.9 哪个数更大？", "9.9"),
    ("一年里有几个月有 28 天？", "12"),
]
# 假模型的"错题本"：每家在哪几题上犯错、错成什么（我们设定的，只为演示）
MOCK_MISTAKES = {
    "claude": {3: "2", 4: "9.11"},
    "gpt": {0: "0.10", 4: "9.11"},
    "deepseek": {2: "24"},
}
FORMAT = "只需要给出最终答案，最后一行写：答案：X"


def name(i, vendor, live):
    """真实模型显示厂商名（型号见开头的团队名单）；假模型用甲乙丙，免得让人以为这是真实厂商的表现。"""
    return vendor if provider_for(vendor, live) != "mock" else f"{'甲乙丙'[i]}（假）"


def extract(text):
    """从回复里取出最后一个"答案："后面的内容；是数字就只留数字。"""
    found = re.findall(r"答案[：:]\s*(.+)", text)
    raw = (found[-1] if found else text).strip()
    num = re.search(r"\d+(?:\.\d+)?", raw)
    return num.group(0) if num else raw


def key(x):
    """比较用：0.10 和 0.1 算同一个答案。"""
    try:
        return f"{float(x):g}"
    except ValueError:
        return x


def same(a, b):
    return key(a) == key(b)


def answer_all(vendor, live):
    out = []
    for q, (question, truth) in enumerate(QUESTIONS):
        wrong = MOCK_MISTAKES[vendor].get(q)
        mock = MockLLM([f"答案：{wrong if wrong else truth}"])
        try:
            out.append(extract(ask(vendor, f"{question}\n{FORMAT}", live, mock).text))
        except Exception as e:                       # 网络错误、限流……记下来，别让整个实验崩掉
            out.append(f"（调用失败：{type(e).__name__}）")
    return out


def score(answers):
    return sum(same(a, t) for a, (_, t) in zip(answers, QUESTIONS))


def majority(*answer_lists):
    result = []
    for votes in zip(*answer_lists):
        top, n = Counter(key(v) for v in votes).most_common(1)[0]
        result.append(top if n * 2 > len(votes) else "（没有多数）")
    return result


# ---------------------------------------------------------------- 思想实验：错误的"相关性"
def simulate(rho, error_rate=0.2, trials=40000, seed=0):
    """每道题有一个各家共享的"难度"，加上各家自己的随机性。rho 越大，各家越容易在同一题上犯错。
    假设：犯错时都掉进同一个陷阱（给出同一个错误答案），这是对投票最不利的情况。"""
    rng = random.Random(seed)
    t = NormalDist().inv_cdf(1 - error_rate)
    single = maj = undetected = flagged = 0
    for _ in range(trials):
        shared = rng.gauss(0, 1)
        wrong = [rho ** 0.5 * shared + (1 - rho) ** 0.5 * rng.gauss(0, 1) > t for _ in range(3)]
        single += wrong[0]
        maj += sum(wrong) >= 2
        undetected += wrong[0] and wrong[1]
        flagged += wrong[0] != wrong[1]
    return single / trials, maj / trials, undetected / trials, flagged / trials


def main():
    parser = add_live_arg(argparse.ArgumentParser(description="跨厂商交叉核对"))
    args = parser.parse_args()
    live = args.live
    print(f"团队：{roster(live)}\n")

    answers = {v: answer_all(v, live) for v in VENDORS}
    names = {v: name(i, v, live) for i, v in enumerate(VENDORS)}
    print(f"   {fit('题目', 34)} | 正确 | " + " | ".join(fit(names[v], 10) for v in VENDORS))
    for q, (question, truth) in enumerate(QUESTIONS):
        cells = [f"{answers[v][q]}{'' if same(answers[v][q], truth) else '（错）'}" for v in VENDORS]
        print(f"   {fit(question, 34)} | {truth:<4} | " + " | ".join(fit(c, 10) for c in cells))

    a, b, c = VENDORS
    repeat = [answer_all(a, live) for _ in range(3)]
    n = len(QUESTIONS)
    print(f"\n   {fit(f'A. 只问{names[a]}', 36)}答对 {score(answers[a])}/{n}")
    print(f"   {fit(f'B. {names[a]}问三次，取多数', 36)}答对 {score(majority(*repeat))}/{n}")
    print(f"   {fit('C. 三家各答一次，取多数', 36)}答对 {score(majority(*answers.values()))}/{n}")
    agree = [same(x, y) for x, y in zip(answers[a], answers[b])]
    ok = sum(1 for q, g in enumerate(agree) if g and same(answers[a][q], QUESTIONS[q][1]))
    bad = sum(1 for q, g in enumerate(agree) if g and not same(answers[a][q], QUESTIONS[q][1]))
    print(f"   {fit(f'D. {names[a]}和{names[b]}交叉核对', 36)}一致且正确 {ok} 题，一致但都错 {bad} 题，"
          f"不一致、交给人复核 {n - ok - bad} 题")

    if not live:
        print("""
   在假模型的设定里：
   * B 和 A 一样：同一个模型问三次，错的还是那几题（错误完全相关）。
   * C 纠正了"只有一家错"的题，但第 5 题甲和乙掉进了同一个陷阱，多数票也救不回来。
   * D 不需要知道正确答案，只看"两家说的是否一样"，就把有问题的题目挑出来交给人；
     代价是"两家一致但都错"的题会漏过去。""")

    print("\n【思想实验】每家单独答错的概率都是 20%，改变各家错误之间的相关程度 ρ（重复 4 万道题）：")
    print("   相关程度 ρ | 只问一家答错 | 三家多数票答错 | 两家核对：一致但都错（漏网） | 两家核对：需要人复核")
    for rho in [0.0, 0.3, 0.6, 0.9]:
        single, maj, undetected, flagged = simulate(rho)
        print(f"      {rho:.1f}     |    {single:5.1%}    |     {maj:5.1%}      |"
              f"           {undetected:5.1%}            |       {flagged:5.1%}")
    print("""
   * ρ = 0（各家的错误互不相关）：三家投票把错误率从 20% 压到 10% 左右，两家核对只漏掉 4%。
   * ρ 越大，多问几家的好处越小；ρ 接近 1 时，就和"同一个模型问三次"差不多了。
   * 所以跨厂商协作的价值，取决于各家的"盲区"有多不一样。这只能靠你自己的测试集去测，
     不能想当然。表里的数字是按假设算出来的，不是真实模型的测量结果。

想一想：
  1. 让甲来"审"乙的答案（而不是各答各的），和这里的 D 有什么区别？审稿人会不会被答案"带偏"？
  2. 如果三家模型都是用相似的网络数据训练的，你觉得它们的错误相关程度高还是低？怎么测？
  3. "不一致就交给人"会产生多少人工工作量？在你的场景里，漏网的错误和人工成本哪个更贵？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
