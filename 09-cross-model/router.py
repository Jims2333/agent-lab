"""
实验 1：路由：这个请求，该交给哪一家？
=======================================

一个产品同时接了 Claude、GPT、DeepSeek 三家的 API，每天收到各种各样的请求。
全部交给最贵的模型，钱包受不了；全部交给最便宜的，有些活又干不好。
路由器（router）做两件事：
  1. 分类：先判断请求属于哪一类
  2. 派单：按路由表把它交给合适的厂商；这一家出故障了，就按顺序换下一家（故障转移）

注意：下面的路由表只是**示例配置**，不代表哪家更擅长什么。
各家的强项会随着版本变化，真正的路由表应该用你自己的任务和评测来定（第七章的"验收"思想）。

运行：
  python router.py            # 假模型
  python router.py --live     # 配置了 key 的厂商用真实模型
"""

import argparse

from team import VENDORS, add_live_arg, ask, fit, label, provider_for, roster
from common.llm import MockLLM

CATEGORIES = {
    "闲聊": "日常问答、打招呼、简单常识",
    "写代码": "写程序、改 bug、解释代码",
    "长文档": "总结、提炼很长的材料",
    "推理": "数学证明、逻辑分析、需要多步思考的判断",
}
ROUTES = {                              # 示例配置：每一类的首选和备选，按顺序尝试
    "闲聊": ["deepseek", "gpt", "claude"],
    "写代码": ["claude", "gpt", "deepseek"],
    "长文档": ["deepseek", "claude", "gpt"],
    "推理": ["gpt", "claude", "deepseek"],
}
# 每百万 token 的价格（美元）：(输入, 输出)。价格经常变，以官网为准。
PRICES = {
    "claude": (4.00, 20.00),            # claude-opus-5-5，Anthropic 官网
    "gpt": (5.00, 30.00),               # gpt-5.5，据第三方价格汇总网站（2026 年）
    "deepseek": (0.14, 0.28),           # deepseek-v4-flash，据第三方价格汇总网站（2026 年）
}
# 一天里收到的请求：(内容, 估计的输入 token, 估计的输出 token)
REQUESTS = [
    ("早上好！今天适合穿什么？", 30, 150),
    ("这段 Python 报 KeyError，帮我找找原因：config['port']", 400, 600),
    ("总结这份三万字的会议纪要，列出五条结论", 30000, 800),
    ("证明根号 2 是无理数", 50, 900),
    ("给我们的新咖啡店起 5 个名字", 40, 200),
    ("写一个快速排序函数，并附上单元测试", 60, 1200),
    ("把这篇 8000 字的论文提炼成一页摘要", 8000, 700),
    ("一个人说'我正在说谎'，这句话是真是假？请分析", 40, 800),
]
KEYWORDS = {"写代码": ["Python", "函数", "代码", "bug", "报错", "KeyError"],
            "长文档": ["总结", "提炼", "摘要", "纪要"],
            "推理": ["证明", "分析", "真是假", "逻辑"]}


# ---------------------------------------------------------------- 第 1 步：分类
def classify(text, live):
    """假模型按关键词分类；真实模型让最便宜的那一家来分（分类本身也要花钱，所以用便宜的）。"""
    def rule(messages, tools):
        for cat, words in KEYWORDS.items():
            if any(w in text for w in words):
                return cat
        return "闲聊"
    options = "\n".join(f"- {k}：{v}" for k, v in CATEGORIES.items())
    prompt = f"把用户的请求归到下面某一类，只输出类别名，不要输出别的：\n{options}\n\n请求：{text}"
    try:
        cat = ask("deepseek", prompt, live, MockLLM([rule])).text.strip().strip("。")
    except Exception:                                # 分类器自己出故障时，退回关键词规则
        cat = rule(None, None)
    return cat if cat in CATEGORIES else "闲聊"


# ---------------------------------------------------------------- 第 2 步：派单 + 故障转移
def call(vendor, text, live, down):
    if vendor in down:
        raise ConnectionError("503 服务暂时不可用（模拟故障）")
    return ask(vendor, text, live, MockLLM([f"（{vendor} 的假回答）收到：{text[:12]}……"]), effort="medium")


def route(text, live, down):
    cat = classify(text, live)
    tried = []
    for vendor in ROUTES[cat]:
        try:
            reply = call(vendor, text, live, down)
            return cat, vendor, reply, tried
        except Exception as e:                       # 网络错误、限流、服务过载……都换下一家
            tried.append(f"{vendor} 失败（{str(e)[:40]}）")
    return cat, None, None, tried


def cost(vendor, tokens_in, tokens_out):
    p_in, p_out = PRICES[vendor]
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def run_day(live, down, title):
    print(f"\n{title}")
    print("   请求                                   | 类别   | 由谁回答")
    total, answered = 0.0, []
    for text, t_in, t_out in REQUESTS:
        cat, vendor, reply, tried = route(text, live, down)
        if vendor and provider_for(vendor, live) != "mock":     # 真实模型：用真实的 token 数
            t_in, t_out = reply.usage["input_tokens"], reply.usage["output_tokens"]
        note = f"  ← 先试了：{'；'.join(tried)}" if tried else ""
        print(f"   {fit(text, 38)} | {fit(cat, 6)} | "
              f"{label(vendor, live) if vendor else '❌ 全部失败'}{note}")
        if vendor:
            total += cost(vendor, t_in, t_out)
            answered.append((t_in, t_out))
    return total, answered


def main():
    parser = add_live_arg(argparse.ArgumentParser(description="跨厂商路由"))
    args = parser.parse_args()
    print(f"团队：{roster(args.live)}")
    print("路由表（示例配置）：" + "；".join(f"{k} → {' → '.join(v)}" for k, v in ROUTES.items()))

    total, answered = run_day(args.live, set(), "【1】正常的一天：")
    total_down, _ = run_day(args.live, {"gpt"}, "【2】gpt 出故障的一天：路由器自动换到下一家，用户感觉不到")

    print("\n【3】同样这 8 个请求，费用对比（按估计的 token 数和上面的价格表，单位：美元）：")
    for v in VENDORS:
        all_v = sum(cost(v, a, b) for a, b in answered)
        print(f"   {fit('全部交给 ' + v, 20)}{all_v:8.4f}")
    print(f"   {fit('按路由表分配', 20)}{total:8.4f}")
    print("""
看懂这个结果：
  * 分类本身也要调用模型，所以用最便宜的那一家来分类；生产环境里常用一个专门训练的小分类器。
  * 长文档的输入动辄几万 token，交给便宜的模型，省下的钱最多；
    但便宜不等于够用：哪类任务可以交给便宜的模型，要靠你自己的评测来回答，这里没有测质量。
  * 故障转移让三家互为备份：任何一家限流、宕机，请求都能被接住。
    这也是"接多家模型"最实在的好处之一：不把鸡蛋放在一个篮子里。
  * 这一切能成立，靠的是 common/llm.py 那层"适配层"：三家的接口长得不一样，
    但路由器只需要调用同一个 chat()。

想一想：
  1. 如果分类分错了（把一道难的推理题当成闲聊），会发生什么？怎样发现这类错误？
  2. 换一种思路"级联"：先让便宜的模型答，检查不过关再交给贵的。什么样的任务能自动"检查过不过关"？
  3. 故障转移时换了一家模型，回答的风格、格式可能都不一样。下游的程序要怎么应对？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
