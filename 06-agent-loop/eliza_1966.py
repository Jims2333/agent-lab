"""
实验 1：迷你 ELIZA：1966 年的"心理医生"
=========================================

1966 年，MIT 的魏岑鲍姆（Joseph Weizenbaum）写出了 ELIZA。它最有名的剧本 DOCTOR 扮演一位心理治疗师：
找到你话里的关键词，套进预先写好的句式，再把问题抛回给你。它对你说的内容**一无所知**。
可据魏岑鲍姆回忆，连他的秘书都请他离开房间，好和 ELIZA 单独"聊一会儿"。

这里用几十行代码做一个中文版。看完它是怎么工作的，再对比第 3 个实验里的 agent：
  ELIZA：规则全是人写的，只会"说"，不会"做"
  agent：由模型自己决定下一步，能调用工具改变外部世界

运行：
  python eliza_1966.py                # 看一段演示对话
  python eliza_1966.py --interactive  # 自己和它聊
"""

import argparse
import random
import re

# 人称转换：你说"我的工作"，它要回"你的工作"
REFLECT = [("我的", "你的"), ("我", "你"), ("你的", "我的"), ("你", "我")]

# 规则：（关键词模式, 回答模板）。{0} 会被替换成从你的话里截取的部分
RULES = [
    (r"我(?:觉得|感觉)(.+)", ["你为什么觉得{0}？", "你经常觉得{0}吗？", "觉得{0}的时候，你会做什么？"]),
    (r"我(?:想要|想)(.+)", ["如果真的{0}了，会怎么样？", "你为什么想{0}？"]),
    (r"我是(.+)", ["你是{0}多久了？", "你觉得自己是{0}，是因为什么？"]),
    (r"我很(.+)", ["你为什么很{0}？", "很{0}的时候，你会怎么办？"]),
    (r"因为(.+)", ["这是真正的原因吗？", "还有别的原因吗？"]),
    (r".*(妈妈|爸爸|家人|父母).*", ["和我说说你的家人吧。", "你和{0}的关系怎么样？"]),
    (r".*(电脑|机器|程序|人工智能|AI).*", ["你担心机器吗？", "你觉得我真的能理解你吗？"]),
    (r".*(对不起|抱歉).*", ["不用道歉。"]),
    (r".+", ["请继续说。", "我明白了。这对你意味着什么？", "嗯……能多说一点吗？"]),
]


def reflect(text):
    out, i = "", 0
    while i < len(text):
        for a, b in REFLECT:                         # 依次尝试替换（先长后短）
            if text.startswith(a, i):
                out += b
                i += len(a)
                break
        else:
            out += text[i]
            i += 1
    return out.strip("。！？!?，, ")


def pad(text, width):
    """按显示宽度补空格（一个汉字占两格）。"""
    return text + " " * max(width - sum(2 if ord(c) > 127 else 1 for c in text), 0)


def respond(text, rng):
    for pattern, templates in RULES:
        m = re.match(pattern, text.strip())
        if m:
            parts = [reflect(g) for g in m.groups()]
            return rng.choice(templates).format(*parts), pattern
    return "请继续说。", ".+"


def main():
    parser = argparse.ArgumentParser(description="迷你 ELIZA")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    rng = random.Random(1966)

    if args.interactive:
        print("ELIZA：你好，我是 ELIZA。说说你最近的心事吧。（输入空行退出）")
        while True:
            try:
                text = input("你：").strip()
            except EOFError:
                break
            if not text:
                break
            print("ELIZA：" + respond(text, rng)[0])
        return

    demo = ["我觉得最近学习压力很大。", "因为我想学会怎么做 AI agent。", "我是一个初学者。",
            "我妈妈觉得我应该先打好基础。", "你说人工智能会取代程序员吗？", "对不起，我扯远了。"]
    print("一段演示对话（右边是 ELIZA 匹配到的规则）：\n")
    for line in demo:
        reply, rule = respond(line, rng)
        print(f"   你：   {line}")
        print(f"   ELIZA：{pad(reply, 34)} ← 规则 {rule}")
    print("""
看起来像在认真倾听，其实它只做了三件事：
  1. 找关键词（"我觉得""因为""妈妈"……）
  2. 把你的话截一段下来，换一下人称（我 → 你）
  3. 套进事先写好的句式
它没有记忆，不懂你说的意思，也不能为你做任何事。

魏岑鲍姆后来在《计算机能力与人类理性》（1976）里反思了这件事：
人们会不由自主地把理解和情感投射到一台简单的程序上。这种现象后来被称为"ELIZA 效应"。
今天和大模型聊天时，这个提醒依然有用。

想一想：
  1. 输入"我是被一部电影吓到的"，ELIZA 会说出什么奇怪的话？为什么？
  2. 给 ELIZA 加一条规则，让它能回应"我讨厌……"。
  3. ELIZA 和第 3 个实验里的 agent，最本质的区别是什么？（提示：谁决定下一步？能不能改变外部世界？）""")


if __name__ == "__main__":
    main()
