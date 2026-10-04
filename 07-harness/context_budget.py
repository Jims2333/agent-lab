"""
实验 2：上下文管理：留什么、删什么、怎么删
============================================

agent 每走一步，历史就长一截（第五章：每次请求都要把整段历史重新发一遍）。
40 步之后，一次请求可能有几万个 token：又贵、又慢，还可能撞上上下文窗口的上限。

这里模拟一次 40 步的 agent 任务（读文件、搜索、跑测试……），比较五种管理上下文的策略：
  A. 全部保留
  B. 每一步都清理旧的工具结果（只留最近 5 条）
  C. 攒够了再批量清理（超过 8000 token 时，一次清掉大部分旧结果）
  D. 摘要压缩（超过 8000 token 时，把旧历史压缩成一段摘要）
  E. 笔记 + 批量清理（agent 把关键信息记进一个小小的笔记里，再配合批量清理）

评价四个指标：
  · 上下文有多大            · 累计一共发了多少 token
  · 有多少能命中"提示缓存"  · 三条关键信息到最后还记不记得

运行：python context_budget.py
"""

import random

STEPS = 40
rng = random.Random(2025)

# ---------------------------------------------------------------- 生成一次模拟的 agent 任务
# 每一步：模型说一段话（约 80 token）+ 一个工具结果（大小随工具不同）
TOOL_SIZES = {"read_file": (800, 2500), "search": (200, 600), "run_tests": (300, 900), "list_dir": (50, 150)}
FACTS = {  # 第几步出现的关键信息，出现在哪种消息里
    3: ("用户要求：所有函数都要加类型注解", "user"),
    11: ("config.yaml 第 7 行：测试必须用 Python 3.11", "tool"),
    24: ("run_tests 的输出：test_parse 只在 Windows 上失败", "tool"),
}
history = [{"id": "task", "kind": "user", "tokens": 120, "fact": None}]
for step in range(1, STEPS + 1):
    history.append({"id": f"a{step}", "kind": "assistant", "tokens": 80, "fact": None})
    tool = rng.choice(list(TOOL_SIZES))
    fact = FACTS.get(step)
    if fact and fact[1] == "user":
        history.append({"id": f"u{step}", "kind": "user", "tokens": 60, "fact": fact[0]})
    history.append({"id": f"t{step}", "kind": "tool", "tokens": rng.randint(*TOOL_SIZES[tool]),
                    "fact": fact[0] if fact and fact[1] == "tool" else None})

SYSTEM = [{"id": "system+tools", "kind": "system", "tokens": 1500, "fact": None}]
PLACEHOLDER = 15        # 被清理的工具结果，换成一行"已清理"的说明
SUMMARY = 600           # 摘要的大小
THRESHOLD = 8000        # 批量清理 / 摘要的触发线
CACHE_PRICE = 0.1       # 命中缓存的部分，价格大约是正常的 1/10（以 Claude 为例）


def upto(step):
    """到第 step 步为止产生的全部消息。"""
    n = next(i for i, m in enumerate(history) if m["id"] == f"t{step}") + 1
    return history[:n]


def size(view):
    return sum(m["tokens"] for m in view)


# ---------------------------------------------------------------- 五种策略：返回这一步真正发出去的内容
def keep_all(step, state):
    return SYSTEM + upto(step)


def clear_every_step(step, state, keep=5):
    msgs = upto(step)
    tools = [m for m in msgs if m["kind"] == "tool"]
    old = {m["id"] for m in tools[:-keep]}
    return SYSTEM + [{**m, "id": m["id"] + "~", "tokens": PLACEHOLDER, "fact": None} if m["id"] in old else m
                     for m in msgs]


def clear_in_batches(step, state, keep=3):
    msgs = upto(step)
    cleared = state.setdefault("cleared", set())
    view = lambda: [{**m, "id": m["id"] + "~", "tokens": PLACEHOLDER, "fact": None} if m["id"] in cleared else m
                    for m in msgs]
    if size(SYSTEM + view()) > THRESHOLD:                  # 超线了才清理，而且一次清掉一大批
        tools = [m for m in msgs if m["kind"] == "tool"]
        cleared.update(m["id"] for m in tools[:-keep])
    return SYSTEM + view() + state.get("notes", [])


def summarize(step, state, keep_msgs=6):
    msgs = upto(step)
    start = state.get("summary_upto", 0)
    view = state.get("summary", []) + msgs[start:]
    if size(SYSTEM + view) > THRESHOLD:
        cut = len(msgs) - keep_msgs
        covered = msgs[:cut]
        # 摘要通常会保留用户的要求，但容易丢掉埋在大段工具结果里的细节
        kept = [m["fact"] for m in covered if m["fact"] and m["kind"] == "user"]
        old_kept = [f for s in state.get("summary", []) for f in s.get("facts", [])]
        state["summary"] = [{"id": f"summary@{step}", "kind": "summary", "tokens": SUMMARY,
                             "fact": None, "facts": old_kept + kept}]
        state["summary_upto"] = cut
        view = state["summary"] + msgs[cut:]
    return SYSTEM + view


def notes_and_batches(step, state):
    # agent 每看到一条关键信息，就往笔记里追加一行（笔记永远放在最后，不会被清理）
    for m in upto(step):
        if m["fact"] and m["fact"] not in state.setdefault("noted", set()):
            state["noted"].add(m["fact"])
            state.setdefault("notes", []).append(
                {"id": f"note:{m['id']}", "kind": "note", "tokens": 30, "fact": m["fact"]})
    return clear_in_batches(step, state)


STRATEGIES = [("A 全部保留", keep_all), ("B 每步清理旧结果", clear_every_step),
              ("C 批量清理", clear_in_batches), ("D 摘要压缩", summarize),
              ("E 笔记 + 批量清理", notes_and_batches)]


def facts_in(view):
    found = set()
    for m in view:
        if m.get("fact"):
            found.add(m["fact"])
        found.update(m.get("facts", []))
    return found


def common_prefix_tokens(prev, cur):
    """提示缓存只认"前缀"：从开头起一模一样的部分才能命中。"""
    total = 0
    for a, b in zip(prev, cur):
        if a["id"] != b["id"]:
            break
        total += b["tokens"]
    return total


print(f"模拟一次 {STEPS} 步的 agent 任务。每一步发出去的上下文有多大（单位：千 token）：\n")
print("   策略               " + "".join(f"{s:>6}" for s in range(5, STEPS + 1, 5)))
results = []
for name, fn in STRATEGIES:
    state, prev = {}, []
    sizes, sent, cached = [], 0, 0
    for step in range(1, STEPS + 1):
        view = fn(step, state)
        s = size(view)
        sizes.append(s)
        sent += s
        cached += common_prefix_tokens(prev, view)
        prev = view
    remembered = facts_in(view)
    results.append((name, sizes, sent, cached, remembered))
    pad = name + " " * (19 - sum(2 if ord(c) > 127 else 1 for c in name))
    print("   " + pad + "".join(f"{sizes[s - 1] / 1000:6.1f}" for s in range(5, STEPS + 1, 5)))

print("\n   策略               | 最大上下文 | 累计发送 | 命中缓存 | 估算费用 | 记住的关键信息")
base_cost = None
for name, sizes, sent, cached, remembered in results:
    cost = (sent - cached) + cached * CACHE_PRICE
    base_cost = base_cost or cost
    pad = name + " " * (19 - sum(2 if ord(c) > 127 else 1 for c in name))
    print(f"   {pad}|  {max(sizes) / 1000:6.1f}k   | {sent / 1000:6.0f}k  |  {cached / sent:5.0%}   | "
          f"{cost / base_cost:6.0%}   |   {len(remembered)}/3")

print("""   （估算费用以"全部保留"为 100%；命中缓存的部分按正常价格的 1/10 计算。）

看懂这张表：
  * A 全部保留：信息一条不丢；因为只往后追加，95% 都能命中缓存，费用其实没有想象中那么高。
    但上下文越来越大，迟早撞上窗口上限，而且越到后面每一步越慢。
  * B 每步清理：上下文明显变小，可每清理一次，历史中间就变一次，之后的内容全部无法命中缓存，
    结果费用反而比"全部保留"还高（148%）。省 token 不等于省钱。
  * C 批量清理：攒够了再一次性清理，只有少数几步会破坏缓存，兼顾了"变小"和"便宜"。
  * D 摘要压缩：上下文最小、费用最低，但摘要容易丢掉埋在工具结果里的细节（这里丢了两条）。
  * E 笔记 + 批量清理：关键信息记进笔记，清理工具结果时不会跟着丢失。这正是 Anthropic 在
    《上下文工程》里推荐的"结构化笔记"做法。

还有一个容易被忽略的约束：有些模型（比如 Claude Opus 5.5）要求你把它之前的回复原样发回去，
如果改动了历史，它之前的"思考"内容就会失效，会被丢弃甚至直接报错。
所以真实的 harness 更愿意"只往后追加"，或者使用厂商提供的服务器端上下文管理功能，而不是在客户端随意修改历史。

想一想：
  1. 把 THRESHOLD 调大到 20000，C 和 E 的费用会怎么变？
  2. D 的摘要如果写得更好（把工具结果里的关键信息也记下来），和 E 还有什么区别？
  3. 你在用 Claude Code、Cursor 这类工具时，注意过它们什么时候"压缩上下文"吗？""")
