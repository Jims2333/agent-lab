"""
番外：让两个 AI 聊天的最小"传话"程序（一个最简单的 harness）
==============================================================

两个 AI 其实都不知道对方存在。它们各自只是在回答"用户"的消息，
而这个"用户"其实是本程序：它把 A 说的话转给 B，再把 B 说的话转给 A。

用法：
  python relay.py                                    # 默认 mock ↔ mock，不需要任何 API key
  python relay.py --show-history                     # 结束后打印两边各自的 messages 列表
  python relay.py --a claude --b deepseek --turns 6 --topic "AI 会不会做梦"
  python relay.py --a gpt --b deepseek

需要的环境变量（只在用到对应模型时才需要）：
  ANTHROPIC_API_KEY   → claude   （pip install anthropic）
  DEEPSEEK_API_KEY    → deepseek （pip install openai，DeepSeek 兼容 OpenAI 的接口）
  OPENAI_API_KEY      → gpt      （pip install openai）
"""

import argparse
import os
import textwrap

# 每个模型的默认型号，可以用 --model-a / --model-b 覆盖。
# 各家会更新型号名，跑不通时先去官方文档确认最新的名字。
DEFAULT_MODELS = {
    "claude": "claude-opus-5-5",
    "deepseek": "deepseek-v4-flash",
    "gpt": "gpt-5.5",
    "mock": "mock",
}
END_MARK = "[END]"


# ---------------------------------------------------------------------------
# 第一部分：把不同厂商的 API 包装成同一个样子
#   reply(system, messages) -> str
# system   : 这个 AI 的"人设"和规则
# messages : 从这个 AI 自己的视角看到的对话历史（user / assistant 交替）
# ---------------------------------------------------------------------------

def make_claude(model):
    import anthropic

    client = anthropic.Anthropic()  # 从 ANTHROPIC_API_KEY 读取密钥

    def reply(system, messages):
        resp = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_config={"effort": "low"},       # 闲聊不需要深度思考，省钱也更快
            # 万一某一句被安全分类器拦下，自动换一个模型重试
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            return "（这一句 Claude 拒绝回答。）"
        return "".join(b.text for b in resp.content if b.type == "text")

    return reply


def make_openai_compatible(model, api_key_env, base_url=None):
    """DeepSeek 和 GPT 都用 OpenAI 的接口格式，区别只是 base_url 和密钥。"""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    def reply(system, messages):
        resp = client.chat.completions.create(
            model=model,
            # OpenAI 格式里，system 是 messages 里的第一条
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content

    return reply


MOCK_LINES = {
    "first": [
        "我先抛个观点：会预测下一个词，不等于理解。",
        "可人类理解一句话时，脑子里也有'预测'在发生啊。",
        f"那我们的分歧其实在'理解'的定义上。今天先聊到这里吧 {END_MARK}",
    ],
    "second": [
        "我不太同意。如果它能用这句话正确地做事，算不算理解？",
        "对，所以关键可能不是'有没有预测'，而是'预测的是什么'。",
        "同意，下次可以从定义开始聊。",
    ],
}


def make_mock(order):
    """假装是一个 AI：不联网、不花钱，只是为了看清"传话"的过程。"""
    lines = MOCK_LINES[order]
    count = 0

    def reply(system, messages):
        nonlocal count
        heard = messages[-1]["content"].split("」")[-1].lstrip("。")   # 去掉对方的引用部分
        text = f"收到「{heard[:10]}…」。{lines[min(count, len(lines) - 1)]}"
        count += 1
        return text

    return reply


def make_speaker(kind, model, order):
    if kind == "claude":
        return make_claude(model)
    if kind == "deepseek":
        return make_openai_compatible(model, "DEEPSEEK_API_KEY", "https://api.deepseek.com")
    if kind == "gpt":
        return make_openai_compatible(model, "OPENAI_API_KEY")
    return make_mock(order)


# ---------------------------------------------------------------------------
# 第二部分：传话循环（这就是 harness 的核心）
# ---------------------------------------------------------------------------

def system_prompt(me, other, topic):
    return (
        f"你是 {me}。你正在和另一个 AI（{other}）聊天，中间有一个程序负责传话，"
        f"你看到的'用户'消息其实都是 {other} 说的话。话题是：{topic}。"
        "每次回复不超过 3 句话，要有自己的观点，也可以反问对方。"
        f"如果你觉得话题已经聊完了，就在回复结尾加上 {END_MARK}。"
    )


def relay(a, b, turns, topic, show_history):
    speakers = [a, b]
    for me, other in ((a, b), (b, a)):
        me["system"] = system_prompt(me["name"], other["name"], topic)
        me["history"] = []

    # 主持人（也就是本程序）先对 A 说一句开场白，A 的第一条 user 消息就是它
    a["history"].append({"role": "user", "content": f"今天的话题：{topic}。你先开个头吧。"})

    for turn in range(turns):
        me, other = speakers[turn % 2], speakers[(turn + 1) % 2]
        text = me["reply"](me["system"], me["history"])

        # 关键的一步：同一句话，在两边的历史里角色不一样
        me["history"].append({"role": "assistant", "content": text})    # 对自己：我说的
        other["history"].append({"role": "user", "content": text})      # 对对方：用户说的

        print(f"\n[{turn + 1}] {me['name']}：")
        print(textwrap.indent(textwrap.fill(text, 60), "    "))

        if END_MARK in text:
            print(f"\n（{me['name']} 说了 {END_MARK}，对话结束）")
            break
    else:
        print(f"\n（达到 {turns} 轮上限，对话结束）")

    if show_history:
        for s in speakers:
            print(f"\n======== {s['name']} 眼里的对话历史（每次调用都要整份发给 API）========")
            print(f"system: {s['system'][:50]}…")
            for m in s["history"]:
                print(f"  {m['role']:>9}: {m['content'][:46]}…")


def main():
    parser = argparse.ArgumentParser(description="让两个 AI 互相聊天")
    choices = list(DEFAULT_MODELS)
    parser.add_argument("--a", choices=choices, default="mock", help="先发言的 AI")
    parser.add_argument("--b", choices=choices, default="mock", help="后发言的 AI")
    parser.add_argument("--model-a", help="覆盖 A 的默认型号")
    parser.add_argument("--model-b", help="覆盖 B 的默认型号")
    parser.add_argument("--turns", type=int, default=6, help="最多说几句（两边加起来）")
    parser.add_argument("--topic", default="机器能不能真正'理解'一句话")
    parser.add_argument("--show-history", action="store_true", help="结束后打印两边的 messages")
    args = parser.parse_args()

    # 两边用同一种 AI 时，加上编号区分
    name_a = args.a.capitalize() + ("-A" if args.a == args.b else "")
    name_b = args.b.capitalize() + ("-B" if args.a == args.b else "")
    try:
        a = {"name": name_a,
             "reply": make_speaker(args.a, args.model_a or DEFAULT_MODELS[args.a], "first")}
        b = {"name": name_b,
             "reply": make_speaker(args.b, args.model_b or DEFAULT_MODELS[args.b], "second")}
    except ImportError as e:
        raise SystemExit(f"缺少 SDK：{e.name}。先运行 pip install {e.name}")
    except KeyError as e:
        raise SystemExit(f"没有找到环境变量 {e}，先 export {e.args[0]}=你的密钥")

    print(f"话题：{args.topic}")
    print(f"{a['name']} ⇄ 传话程序 ⇄ {b['name']}")
    relay(a, b, args.turns, args.topic, args.show_history)


if __name__ == "__main__":
    main()
