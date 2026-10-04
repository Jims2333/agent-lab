"""
实验 2：合同网：发公告、投标、中标
=====================================

1980 年 Reid G. Smith 发表的合同网协议（Contract Net Protocol），解决的是"活该交给谁干"：
  1. 公告（announce）：经理把任务广播出去，写明需要什么技能、资格门槛是多少
  2. 投标（bid）：每个承包者自己评估"我能不能干、多久能干完"，愿意就投标，不愿意就拒绝
  3. 中标（award）：经理比较所有标书，把合同交给最合适的那一个
  4. 汇报（report）：承包者干完后交回结果
后来 FIPA 把它写成了标准，消息名字叫 cfp / propose / refuse / accept-proposal / inform。

关键在于：经理**不需要知道**每个承包者忙不忙、擅长什么，这些由承包者自己在标书里说清楚。
这里还会演示：没人投标时怎么办，以及和"不看技能、轮流派活"相比效果差多少。

运行：
  python contract_net.py                     # 假模型：承包者按自己的技能表规则投标
  python contract_net.py --provider claude   # 真实模型：每份标书由模型写（会调用几十次 API）
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import MockLLM, add_provider_args, chat, user_message  # noqa: E402

# 承包者：技能熟练度（0～1）和每小时的价格。熟练度越高，干得越快、质量越好。
CONTRACTORS = [
    {"name": "研究员", "skills": {"调研": 0.9, "写作": 0.8, "摘要": 0.7}, "price": 3},
    {"name": "程序员", "skills": {"写代码": 0.9, "调试": 0.85, "写作": 0.4}, "price": 3},
    {"name": "数学家", "skills": {"数学": 0.95, "写代码": 0.6}, "price": 2},
    {"name": "速记员", "skills": {"摘要": 0.8, "翻译": 0.85, "写作": 0.6}, "price": 1},
]
# 任务：(编号, 描述, 需要的技能, 工作量——熟练度为 1 的人需要几小时)
TASKS = [
    ("T1", "调研 2023 年以来的多 agent 框架", "调研", 4),
    ("T2", "实现一个解析 JSON 日志的函数", "写代码", 3),
    ("T3", "推导注意力机制的计算复杂度", "数学", 2),
    ("T4", "把调研结果翻译成英文", "翻译", 2),
    ("T5", "修复一个失败的单元测试", "调试", 2),
    ("T6", "写一段 300 字的项目介绍", "写作", 2),
    ("T7", "实现一个 BPE 分词器", "写代码", 4),
    ("T8", "给 10 篇论文各写一段摘要", "摘要", 3),
    ("T9", "画一张系统架构图", "画图", 2),
]
THRESHOLD = 0.6            # 公告里的资格门槛：这项技能的熟练度至少要到多少
RELAXED = 0.3              # 没人投标时，第二次公告放宽到多少
SKILLS = sorted({s for c in CONTRACTORS for s in c["skills"]})


# ---------------------------------------------------------------- 承包者：自己评估、自己投标
def rule_bid(contractor, task, threshold, busy_until):
    """假模型的投标逻辑：有这项技能、而且达到门槛才投；工期 = 工作量 / 熟练度。"""
    _, _, skill, size = task
    level = contractor["skills"].get(skill, 0)
    if level < threshold:
        return {"bid": False, "reason": f"我的{skill}熟练度只有 {level}，达不到 {threshold} 的门槛"}
    hours = round(size / level, 1)
    return {"bid": True, "hours": hours, "reason": f"{skill}熟练度 {level}，手头的活到第 {busy_until:.1f} 小时做完"}


def ask_contractor(contractor, task, threshold, busy_until, provider, model):
    """把任务公告发给一个承包者，拿回它的标书。真实模式下，标书由模型来写。"""
    tid, desc, skill, size = task
    prompt = (f"你是团队里的「{contractor['name']}」，你的技能熟练度是 {json.dumps(contractor['skills'], ensure_ascii=False)}，"
              f"手头的工作要到第 {busy_until:.1f} 小时才做完。\n"
              f"任务公告 {tid}：{desc}。需要技能：{skill}；资格要求：该技能熟练度 ≥ {threshold}；"
              f"工作量：熟练度为 1 的人需要 {size} 小时。\n"
              '请诚实评估。只输出一个 JSON：{"bid": true 或 false, "hours": 你预计需要的小时数, "reason": "一句话理由"}')
    mock = MockLLM([lambda m, t: json.dumps(rule_bid(contractor, task, threshold, busy_until), ensure_ascii=False)])
    reply = chat([user_message(prompt)], provider=provider, model=model, mock=mock, effort="low")
    try:
        bid = json.loads(re.search(r"\{.*\}", reply.text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        return {"bid": False, "reason": "（标书格式不对，按拒绝处理）"}
    if bid.get("bid") and not isinstance(bid.get("hours"), (int, float)):
        return {"bid": False, "reason": "（标书没写清工期，按拒绝处理）"}
    return bid


# ---------------------------------------------------------------- 经理：公告、比较标书、授予合同
def reformulate(task, provider, model):
    """连放宽门槛都没人投标时，经理换个说法，把任务改写成团队里有人会做的形式。"""
    tid, desc, skill, size = task
    prompt = (f"任务 {tid}「{desc}」需要的技能是「{skill}」，但团队里没有人投标。"
              f"团队成员会的技能有：{'、'.join(SKILLS)}。\n"
              "请把这个任务改写成团队能完成的形式。只输出 JSON："
              '{"description": "改写后的任务", "skill": "上面列表里的一项技能"}')
    canned = {"description": "用 Mermaid 语法写出系统架构图的文本描述（可以自动渲染成图）", "skill": "写代码"}
    mock = MockLLM([json.dumps(canned, ensure_ascii=False)])
    reply = chat([user_message(prompt)], provider=provider, model=model, mock=mock, effort="low")
    try:
        new = json.loads(re.search(r"\{.*\}", reply.text, re.S).group(0))
        if new.get("skill") in SKILLS:
            return (tid + "'", new["description"], new["skill"], size)
    except (AttributeError, json.JSONDecodeError):
        pass
    return None


def contract_net(provider, model, verbose=True):
    busy = {c["name"]: 0.0 for c in CONTRACTORS}       # 每个承包者手头的活要到第几小时做完
    schedule = []                                       # (承包者, 任务编号, 开始, 结束, 熟练度)
    say = print if verbose else (lambda *a, **k: None)

    for task in TASKS:
        attempts = [(task, THRESHOLD), (task, RELAXED)]
        while attempts:
            current, threshold = attempts.pop(0)
            tid, desc, skill, size = current
            say(f"\n📢 公告 {tid}：{desc}（需要「{skill}」，门槛 {threshold}）")
            bids, refused = [], []
            for c in CONTRACTORS:
                b = ask_contractor(c, current, threshold, busy[c["name"]], provider, model)
                if b.get("bid"):
                    finish = busy[c["name"]] + b["hours"]
                    bids.append((finish, b["hours"] * c["price"], c, b))
                    say(f"   📝 {c['name']} 投标：{b['hours']} 小时，第 {finish:.1f} 小时交付，"
                        f"报价 {b['hours'] * c['price']:.1f}（{b['reason']}）")
                else:
                    refused.append(c["name"])
            if refused:
                say(f"   ✋ 拒绝：{'、'.join(refused)}")
            if bids:
                # 经理的评标规则：谁最早交付就给谁；同时交付时选便宜的
                finish, cost, winner, b = min(bids, key=lambda x: (x[0], x[1]))
                start = busy[winner["name"]]
                busy[winner["name"]] = finish
                schedule.append((winner["name"], tid, start, finish, winner["skills"].get(skill, 0), cost))
                say(f"   ✅ 中标：{winner['name']}（第 {start:.1f}～{finish:.1f} 小时）")
                break
            say("   ⚠️  没有人投标。")
            if not attempts and threshold == RELAXED:
                new = reformulate(current, provider, model)
                if new:
                    say(f"   🔁 经理把任务改写成：{new[1]}")
                    attempts.append((new, THRESHOLD))
                else:
                    say("   🙋 还是没人能做，上报给人类。")
            elif attempts:
                say(f"   🔁 放宽门槛，重新公告一次。")
    return schedule


# ---------------------------------------------------------------- 对照组：不看技能，轮流派活
def round_robin():
    busy = {c["name"]: 0.0 for c in CONTRACTORS}
    schedule = []
    for i, (tid, desc, skill, size) in enumerate(TASKS):
        c = CONTRACTORS[i % len(CONTRACTORS)]
        level = c["skills"].get(skill, 0.1)            # 完全不会的活，也只能硬着头皮慢慢做
        hours = round(size / level, 1)
        start = busy[c["name"]]
        busy[c["name"]] = start + hours
        schedule.append((c["name"], tid, start, start + hours, level, hours * c["price"]))
    return schedule


def gantt(schedule, scale):
    """用字符画甘特图：每个字符代表 1/scale 小时。"""
    end = max(s[3] for s in schedule)
    step = max(1, 4 // scale)                           # 刻度每隔几小时标一次，保证每个刻度占 4 个字符
    print("   " + " " * 8 + "".join(f"{h:<4}" for h in range(0, int(end) + step, step)))
    for c in CONTRACTORS:
        row = [" "] * (int(end * scale) + 2)
        for name, tid, start, finish, *_ in schedule:
            if name != c["name"]:
                continue
            a, b = int(round(start * scale)), max(int(round(finish * scale)), int(round(start * scale)) + 1)
            label = (tid + "·" * (b - a))[: b - a]
            row[a:b] = list(label)
            row[a] = "["
        pad = c["name"] + " " * (8 - 2 * len(c["name"]))
        print("   " + pad + "".join(row))


def report(name, schedule):
    makespan = max(s[3] for s in schedule)
    quality = sum(s[4] for s in schedule) / len(schedule)
    weak = sum(1 for s in schedule if s[4] < THRESHOLD)
    cost = sum(s[5] for s in schedule)
    print(f"   {name}| 全部完成要 {makespan:5.1f} 小时 | 平均熟练度 {quality:.2f} | "
          f"交给外行的任务 {weak} 个 | 总花费 {cost:5.1f}")


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="合同网协议"))
    args = parser.parse_args()
    print(f"团队：{'、'.join(c['name'] for c in CONTRACTORS)}；任务 {len(TASKS)} 个。经理按顺序公告每个任务。")
    cn = contract_net(args.provider, args.model)
    if not cn:
        print("\n一个任务都没有成交。用真实模型时，先检查它有没有按要求只输出 JSON 标书。")
        return

    print("\n合同网的排期（[1 表示 T1 开始，· 表示进行中，每格半小时，刻度单位是小时）：")
    gantt(cn, scale=2)
    rr = round_robin()
    print("\n对照组：不看技能、轮流派活的排期（注意：这张图每格一小时）：")
    gantt(rr, scale=1)
    print()
    report("合同网    ", cn)
    report("轮流派活  ", rr)
    print("""
看懂这个结果（以假模型的剧本为例）：
  * 经理从头到尾没有问过"谁会什么"，只看标书；承包者最了解自己，由它们来报工期。
  * 负载均衡是自动出现的：程序员接了两个活以后，T7 写代码的标书交付时间排到了后面，
    于是会写代码、手头又空的数学家中了标。
  * T9 没人会画图：放宽门槛还是没人投，经理就把任务改写成团队能做的形式；真找不到人就上报给人类。
  * 轮流派活看起来"公平"，却把调试（T5）和画图（T9）交给了研究员、把写作（T6）交给了程序员，
    研究员一个人要干 40 多个小时，其他人早早就闲着了，又慢又贵。

换成大模型以后要多想一步：模型写的标书**不一定诚实**，它可能高估自己的能力。
协议本身并不检查标书是否属实，所以真实系统里，经理还得参考每个承包者过去的成绩，或者在交付时验收。

想一想：
  1. 把评标规则改成"谁最便宜就给谁"，排期和总花费会怎么变？
  2. 如果某个承包者每次都把工期报少一半，会发生什么？经理怎样才能发现？
  3. 现在任务是一个一个公告的。如果 9 个任务同时公告，同一个承包者可能同时中好几个标，该怎么办？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
