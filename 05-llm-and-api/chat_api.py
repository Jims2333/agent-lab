"""
实验 4：调用大模型 API：无状态、token 和费用
=============================================

前三个实验在自己的电脑上造"迷你语言模型"。真正的大模型太大，跑在厂商的服务器上，我们通过 API 调用它。
这个实验用本仓库的公共客户端 common/llm.py，演示三件用 API 前必须知道的事：
  1. API 是"无状态"的：它不记得你，"记忆"是你每次自己发过去的
  2. 多轮对话里，token 和费用是怎么越滚越多的
  3. 流式输出：边生成边显示

运行：
  python chat_api.py                                 # 默认用假模型（mock），不需要 API key
  python chat_api.py --provider claude               # 用真实模型（需要 ANTHROPIC_API_KEY）
  python chat_api.py --provider deepseek --interactive   # 和真实模型自由聊天
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 让脚本能找到仓库根目录下的 common/

from common.llm import (DEFAULT_MODELS, MockLLM, add_provider_args, assistant_message, chat,  # noqa: E402
                        cost_usd, user_message)


def memory_mock(messages, tools):
    """一个假模型。和真模型一样，它只能看到这一次请求里发过来的 messages，除此之外什么都不记得。"""
    last = messages[-1]["content"]
    name = None
    for m in messages:
        if m["role"] == "user" and "我叫" in m["content"] and "什么" not in m["content"]:
            name = m["content"].split("我叫")[1].strip("。！!，, ")
    if "我叫什么" in last:
        return f"你叫{name}。" if name else "抱歉，我不知道你叫什么，你还没告诉过我。"
    if "我叫" in last:
        return f"你好，{name}！很高兴认识你。"
    return f"（假模型的回答）关于「{last[:14]}」，这一轮我一共收到了 {len(messages)} 条消息。"


def show_request(messages):
    print("     发出去的 messages：" + json.dumps(
        [{"role": m["role"], "content": m["content"]} for m in messages], ensure_ascii=False))


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="大模型 API 入门"))
    parser.add_argument("--interactive", action="store_true", help="进入自由聊天模式")
    args = parser.parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    mock = MockLLM(memory_mock)
    system = "你是一个简洁友好的助手，每次回答不超过两句话。"

    def ask(messages, **kw):
        return chat(messages, system=system, provider=args.provider, model=model, mock=mock,
                    effort="low", **kw)

    print(f"使用的模型：{args.provider} / {model}\n")

    if args.interactive:
        history = []
        print("自由聊天模式（输入空行或按 Ctrl-D 退出）")
        while True:
            try:
                text = input("\n你：").strip()
            except EOFError:
                break
            if not text:
                break
            history.append(user_message(text))
            print("模型：", end="", flush=True)
            reply = ask(history, on_text=lambda t: print(t, end="", flush=True))
            history.append(assistant_message(reply))
            print(f"\n     （这一轮发送 {reply.usage['input_tokens']} 个 token，收到 {reply.usage['output_tokens']} 个）")
        return

    # ------------------------------------------------------------ 1. 无状态
    print("【1】API 不记得你：'记忆'要自己带上")
    first = [user_message("我叫小明。")]
    r1 = ask(first)
    print(f"   第一次请求：")
    show_request(first)
    print(f"     模型：{r1.text}")

    fresh = [user_message("我叫什么？")]
    r2 = ask(fresh)
    print(f"   第二次请求（新开一个请求，只问问题）：")
    show_request(fresh)
    print(f"     模型：{r2.text}")

    full = first + [assistant_message(r1), user_message("我叫什么？")]
    r3 = ask(full)
    print(f"   第三次请求（把之前的对话一起发过去）：")
    show_request(full)
    print(f"     模型：{r3.text}")
    print("   模型本身不保存任何对话。网页版的聊天机器人之所以'记得'你，是因为程序每次都把整段历史重新发了一遍。\n")

    # ------------------------------------------------------------ 2. token 和费用
    print("【2】多轮对话：每一轮都要把整段历史重新发一遍")
    questions = ["什么是梯度下降？", "学习率太大会怎样？", "那太小呢？", "怎么选合适的学习率？",
                 "Adam 是什么？", "它和 SGD 有什么区别？", "什么时候该用 SGD？", "总结一下今天聊的。"]
    history, total_in, total_out = [], 0, 0
    price_model = model if cost_usd(model, {"input_tokens": 0, "output_tokens": 0}) is not None else "claude-opus-5-5"
    print("   轮次 | 这一轮发送 | 这一轮收到 | 累计发送 | 累计费用")
    for i, q in enumerate(questions, 1):
        history.append(user_message(q))
        reply = ask(history)
        history.append(assistant_message(reply))
        total_in += reply.usage["input_tokens"]
        total_out += reply.usage["output_tokens"]
        cost = cost_usd(price_model, {"input_tokens": total_in, "output_tokens": total_out})
        print(f"   {i:4d} | {reply.usage['input_tokens']:10,} | {reply.usage['output_tokens']:10,} | {total_in:8,} | ${cost:.5f}")
    note = "（假模型的 token 数是估算的）" if args.provider == "mock" else ""
    print(f"   费用按 {price_model} 的价格估算{note}。")
    print("""   每一轮发送的 token 都比上一轮多：之前所有的问题和回答都要重新发一遍。
   对话越长，每一轮越贵、越慢，直到撞上"上下文窗口"的上限。
   这就是为什么第七章要讲"上下文管理"：什么该留，什么该删，什么该压缩成摘要。
""")

    # ------------------------------------------------------------ 3. 流式输出
    print("【3】流式输出：不用等全部生成完，边生成边显示（| 是每一小段的分界）")
    print("   模型：", end="")
    ask([user_message("用一句话解释什么是 token。")], on_text=lambda t: print(t + "|", end="", flush=True))
    print("\n   大模型是一个 token 一个 token 地生成的。流式输出让用户更早看到开头，体验上快得多。")

    print("""
想一想：
  1. 一段 100 轮的对话，第 100 轮发送的 token 大约是第 1 轮的多少倍？总费用和轮数是什么关系？
  2. 如果把很早之前的对话删掉一部分再发送，会有什么好处和坏处？
  3. 用 --interactive 和真实模型聊几句，然后问它"我们刚才聊了什么"，再新开一次程序问同样的问题，对比一下。""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:          # 没装 SDK、没设置 key 之类的问题，给一句人话
        raise SystemExit(f"出错了：{e}")
