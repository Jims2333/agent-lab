"""
第 5～9 章共用的大模型客户端：一个接口，接 Claude / GPT / DeepSeek / 假模型
==========================================================================

为什么要有它？三家的 API 大同小异，但细节各不相同：
  * Claude（Anthropic SDK）：system 是单独的参数；工具调用是 tool_use 内容块；工具结果放在 user 消息里
  * GPT / DeepSeek（OpenAI SDK）：system 是第一条消息；工具调用在 tool_calls 字段里；工具结果是 role="tool" 的消息
这个文件用一种"中立格式"保存对话，在调用时再翻译成各家的格式。第 9 章会专门讲这个"适配层"。

中立格式的消息：
  {"role": "user", "content": "文字"}
  {"role": "assistant", "content": "文字", "tool_calls": [{"id", "name", "args"}], "raw": {...}}
  {"role": "tool", "tool_call_id": "...", "name": "工具名", "content": "结果文字", "is_error": False}
中立格式的工具：
  {"name": "...", "description": "...", "parameters": {JSON Schema}}

用法：
  from common.llm import chat, assistant_message, tool_message
  reply = chat(messages, system="你是……", tools=TOOLS, provider="mock", mock=MockLLM([...]))
  messages.append(assistant_message(reply))

环境变量：ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY
注意：真实模型模式需要你自己的 key，作者没有用真实 key 测试过；mock 模式不需要任何依赖。
"""

import json
import os
from dataclasses import dataclass, field

DEFAULT_MODELS = {
    "claude": "claude-opus-5-5",
    "gpt": "gpt-5.5",
    "deepseek": "deepseek-v4-flash",
    "mock": "mock-1",
}

# 每百万 token 的价格（美元），只列出核实过的。价格会变，以官网为准。
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

PROVIDERS = tuple(DEFAULT_MODELS)


@dataclass
class Reply:
    text: str = ""
    tool_calls: list = field(default_factory=list)   # [{"id", "name", "args"}]
    stop: str = "end"                                 # "end" | "tool_use" | "max_tokens" | "refusal"
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    provider: str = "mock"
    model: str = "mock-1"
    raw: object = None                                # 厂商原始的返回内容（原样回放时要用）


def estimate_tokens(text):
    """粗略估算 token 数：汉字约 1 个字 1 个 token，英文约 4 个字母 1 个 token。只用于假模型和演示。"""
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    return cjk + (len(text) - cjk) // 4 + 1


def cost_usd(model, usage):
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return usage["input_tokens"] / 1e6 * p_in + usage["output_tokens"] / 1e6 * p_out


# ---------------------------------------------------------------------------
# 构造中立格式的消息
# ---------------------------------------------------------------------------

def user_message(text):
    return {"role": "user", "content": text}


def assistant_message(reply):
    """把模型的回复变成一条可以存进历史的消息。raw 里保留厂商的原始内容，下次原样发回去。"""
    return {"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls,
            "raw": {reply.provider: reply.raw} if reply.raw is not None else {}}


def tool_message(call, result, is_error=False):
    return {"role": "tool", "tool_call_id": call["id"], "name": call["name"],
            "content": str(result), "is_error": is_error}


# ---------------------------------------------------------------------------
# 假模型：按剧本回答，用来在没有 API key 时演示 harness 的完整流程
# ---------------------------------------------------------------------------

class MockLLM:
    """剧本里的每一项是一轮回复，可以是：
         "一段文字"                                         → 直接回答
         {"text": "...", "tool_calls": [{"name", "args"}]}   → 调用工具
         一个函数 f(messages, tools) -> 上面两种之一         → 根据对话内容决定（比如读取工具结果）
       剧本也可以直接是一个函数：每一轮都用它来回答。
       剧本用完之后，统一回答 fallback。"""

    def __init__(self, script, fallback="（假模型的剧本已经演完了。）"):
        self.respond = script if callable(script) else None
        self.script = [] if callable(script) else list(script)
        self.fallback = fallback
        self.calls = 0

    def __call__(self, messages, tools):
        if self.respond:
            step = self.respond
        else:
            step = self.script.pop(0) if self.script else self.fallback
        if callable(step):
            step = step(messages, tools)
        if isinstance(step, str):
            step = {"text": step}
        self.calls += 1
        calls = [{"id": f"mock_{self.calls}_{i}", "name": c["name"], "args": c.get("args", {})}
                 for i, c in enumerate(step.get("tool_calls", []))]
        text = step.get("text", "")
        prompt_tokens = sum(estimate_tokens(json.dumps(m, ensure_ascii=False, default=str)) for m in messages)
        return Reply(text=text, tool_calls=calls, stop="tool_use" if calls else "end",
                     usage={"input_tokens": prompt_tokens,
                            "output_tokens": estimate_tokens(text) + 10 * len(calls)},
                     provider="mock", model="mock-1")


# ---------------------------------------------------------------------------
# Claude（Anthropic SDK）
# ---------------------------------------------------------------------------

def _to_anthropic(messages):
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            raw = m.get("raw", {}).get("claude")
            if raw is not None:
                # 同一个模型产生的回复（包括思考块）必须原样发回去，改动会让 API 拒绝或丢弃思考内容
                out.append({"role": "assistant", "content": raw})
            else:
                blocks = [{"type": "text", "text": m["content"]}] if m["content"] else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                           for c in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if m.get("is_error"):
                block["is_error"] = True
            # 连续的多个工具结果，要合并进同一条 user 消息
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


def _call_claude(messages, system, tools, model, effort, on_text):
    import anthropic

    client = anthropic.Anthropic()
    kwargs = dict(
        model=model,
        max_tokens=16000,
        messages=_to_anthropic(messages),
        output_config={"effort": effort or "medium"},
        # 1) 被安全分类器拒绝时，由服务器自动换模型重试
        # 2) 如果 harness 改动了历史（比如第 7 章的上下文压缩），丢弃失效的思考块而不是直接报错
        thinking={"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
        betas=["server-side-fallback-2026-07-01", "thinking-binding-controls-2026-08-01"],
        fallbacks="default",
    )
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                            "input_schema": t["parameters"]} for t in tools]
    if on_text:
        with client.beta.messages.stream(**kwargs) as stream:
            for chunk in stream.text_stream:
                on_text(chunk)
            resp = stream.get_final_message()
    else:
        resp = client.beta.messages.create(**kwargs)

    if resp.stop_reason == "refusal":
        return Reply(text="（模型拒绝了这个请求。）", stop="refusal", provider="claude", model=model,
                     raw=resp.content, usage=_claude_usage(resp))
    text = "".join(b.text for b in resp.content if b.type == "text")
    calls = [{"id": b.id, "name": b.name, "args": dict(b.input)}
             for b in resp.content if b.type == "tool_use"]
    stop = {"tool_use": "tool_use", "max_tokens": "max_tokens"}.get(resp.stop_reason, "end")
    return Reply(text=text, tool_calls=calls, stop=stop, usage=_claude_usage(resp),
                 provider="claude", model=resp.model or model, raw=resp.content)


def _claude_usage(resp):
    return {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


# ---------------------------------------------------------------------------
# GPT 和 DeepSeek（OpenAI 兼容接口）
# ---------------------------------------------------------------------------

def _to_openai(messages, system):
    out = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m["content"] or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"],
                                                   "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                                     for c in m["tool_calls"]]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out


def _call_openai_compatible(provider, messages, system, tools, model, on_text):
    from openai import OpenAI

    key_env, base_url = {"gpt": ("OPENAI_API_KEY", None),
                         "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com")}[provider]
    if not os.environ.get(key_env):
        raise RuntimeError(f"没有设置环境变量 {key_env}")
    client = OpenAI(api_key=os.environ[key_env], base_url=base_url)
    kwargs = dict(model=model, messages=_to_openai(messages, system))
    if tools:
        kwargs["tools"] = [{"type": "function",
                            "function": {"name": t["name"], "description": t["description"],
                                         "parameters": t["parameters"]}} for t in tools]

    if on_text and not tools:                      # 流式输出（这里只在不用工具时支持，保持简单）
        text, usage = "", {"input_tokens": 0, "output_tokens": 0}
        for chunk in client.chat.completions.create(**kwargs, stream=True,
                                                    stream_options={"include_usage": True}):
            if chunk.choices and chunk.choices[0].delta.content:
                on_text(chunk.choices[0].delta.content)
                text += chunk.choices[0].delta.content
            if chunk.usage:
                usage = {"input_tokens": chunk.usage.prompt_tokens,
                         "output_tokens": chunk.usage.completion_tokens}
        return Reply(text=text, usage=usage, provider=provider, model=model)

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message
    calls = []
    for c in msg.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw_arguments": c.function.arguments}
        calls.append({"id": c.id, "name": c.function.name, "args": args})
    stop = "tool_use" if calls else ("max_tokens" if choice.finish_reason == "length" else "end")
    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens} \
        if resp.usage else {"input_tokens": 0, "output_tokens": 0}
    return Reply(text=msg.content or "", tool_calls=calls, stop=stop, usage=usage,
                 provider=provider, model=resp.model or model)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def chat(messages, system=None, tools=None, provider="mock", model=None, effort=None,
         mock=None, on_text=None):
    """发一次请求，返回 Reply。messages 用中立格式；provider 是 claude / gpt / deepseek / mock。"""
    if provider not in PROVIDERS:
        raise ValueError(f"不认识的 provider：{provider}，可选：{', '.join(PROVIDERS)}")
    model = model or DEFAULT_MODELS[provider]
    if provider == "mock":
        reply = (mock or MockLLM([]))(messages, tools or [])
        if on_text:                                  # 假装在流式输出：一小段一小段地吐字
            for i in range(0, len(reply.text), 6):
                on_text(reply.text[i:i + 6])
        return reply
    try:
        if provider == "claude":
            return _call_claude(messages, system, tools, model, effort, on_text)
        return _call_openai_compatible(provider, messages, system, tools, model, on_text)
    except ImportError as e:
        raise RuntimeError(f"缺少 SDK：{e.name}。先运行 pip install {e.name}") from e
    except TypeError as e:
        if "authentication" in str(e):
            raise RuntimeError("没有找到 Claude 的凭据：请设置环境变量 ANTHROPIC_API_KEY") from e
        raise


def add_provider_args(parser, default="mock"):
    """给命令行脚本加上 --provider 和 --model 两个参数。"""
    parser.add_argument("--provider", choices=PROVIDERS,
                        default=os.environ.get("AGENT_LAB_PROVIDER", default),
                        help="用哪家的模型（默认 mock：不需要 API key 的假模型）")
    parser.add_argument("--model", help="覆盖默认型号")
    return parser
