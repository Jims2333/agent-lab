"""
The LLM client shared by Chapters 5–9: one interface for Claude / GPT / DeepSeek / the mock model
================================================================================================

Why does this exist? The three providers' APIs are broadly similar, but the details all differ:
  * Claude (Anthropic SDK): system is a separate parameter; tool calls are tool_use content blocks; tool results go in a user message
  * GPT / DeepSeek (OpenAI SDK): system is the first message; tool calls are in the tool_calls field; tool results are messages with role="tool"
This file stores the conversation in a "neutral format" and translates it into each provider's format at call time.
Chapter 9 covers this "adapter layer" in detail.

Messages in the neutral format:
  {"role": "user", "content": "text"}
  {"role": "assistant", "content": "text", "tool_calls": [{"id", "name", "args"}], "raw": {...}}
  {"role": "tool", "tool_call_id": "...", "name": "tool name", "content": "result text", "is_error": False}
Tools in the neutral format:
  {"name": "...", "description": "...", "parameters": {JSON Schema}}

Usage:
  from common.llm import chat, assistant_message, tool_message
  reply = chat(messages, system="You are ...", tools=TOOLS, provider="mock", mock=MockLLM([...]))
  messages.append(assistant_message(reply))

Environment variables: ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY
Note: real-model mode needs your own key, and the author has not tested it with real keys; mock mode needs no dependencies at all.
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

# Price per million tokens (USD); only verified prices are listed. Prices change, so the official website has the final word.
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
    raw: object = None                                # the provider's raw response content (needed to send it back verbatim)


def estimate_tokens(text):
    """Rough token estimate: about 1 token per Chinese character, about 1 token per 4 English letters. Only used for the mock model and demos."""
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    return cjk + (len(text) - cjk) // 4 + 1


def cost_usd(model, usage):
    if model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return usage["input_tokens"] / 1e6 * p_in + usage["output_tokens"] / 1e6 * p_out


# ---------------------------------------------------------------------------
# Building messages in the neutral format
# ---------------------------------------------------------------------------

def user_message(text):
    return {"role": "user", "content": text}


def assistant_message(reply):
    """Turn the model's reply into a message that can be stored in the history. raw keeps the provider's original content so it can be sent back verbatim next time."""
    return {"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls,
            "raw": {reply.provider: reply.raw} if reply.raw is not None else {}}


def tool_message(call, result, is_error=False):
    return {"role": "tool", "tool_call_id": call["id"], "name": call["name"],
            "content": str(result), "is_error": is_error}


# ---------------------------------------------------------------------------
# Mock model: answers from a script, to demo the harness's full flow when there's no API key
# ---------------------------------------------------------------------------

class MockLLM:
    """Each item in the script is one turn's reply, and can be:
         "some text"                                         → answer directly
         {"text": "...", "tool_calls": [{"name", "args"}]}   → call tools
         a function f(messages, tools) -> one of the above   → decide based on the conversation (e.g. read a tool result)
       The script can also be a single function: it is used to answer every turn.
       Once the script runs out, every answer is the fallback."""

    def __init__(self, script, fallback="(The mock model's script has run out.)"):
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
# Claude (Anthropic SDK)
# ---------------------------------------------------------------------------

def _to_anthropic(messages):
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            raw = m.get("raw", {}).get("claude")
            if raw is not None:
                # A reply produced by the same model (thinking blocks included) must be sent back exactly as it was;
                # any change makes the API reject it or drop the thinking content
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
            # several tool results in a row must be merged into one user message
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
        # 1) if the safety classifier refuses, the server automatically retries with another model
        # 2) if the harness has changed the history (e.g. Chapter 7's context compression),
        #    drop the invalidated thinking blocks instead of failing with an error
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
        return Reply(text="(The model refused this request.)", stop="refusal", provider="claude", model=model,
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
# GPT and DeepSeek (OpenAI-compatible API)
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
        raise RuntimeError(f"Environment variable {key_env} is not set")
    client = OpenAI(api_key=os.environ[key_env], base_url=base_url)
    kwargs = dict(model=model, messages=_to_openai(messages, system))
    if tools:
        kwargs["tools"] = [{"type": "function",
                            "function": {"name": t["name"], "description": t["description"],
                                         "parameters": t["parameters"]}} for t in tools]

    if on_text and not tools:                      # streaming output (only supported here when no tools are used, to keep it simple)
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
# Single entry point
# ---------------------------------------------------------------------------

def chat(messages, system=None, tools=None, provider="mock", model=None, effort=None,
         mock=None, on_text=None):
    """Send one request and return a Reply. messages use the neutral format; provider is claude / gpt / deepseek / mock."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Options: {', '.join(PROVIDERS)}")
    model = model or DEFAULT_MODELS[provider]
    if provider == "mock":
        reply = (mock or MockLLM([]))(messages, tools or [])
        if on_text:                                  # pretend to stream: emit the text a small chunk at a time
            for i in range(0, len(reply.text), 6):
                on_text(reply.text[i:i + 6])
        return reply
    try:
        if provider == "claude":
            return _call_claude(messages, system, tools, model, effort, on_text)
        return _call_openai_compatible(provider, messages, system, tools, model, on_text)
    except ImportError as e:
        raise RuntimeError(f"Missing SDK: {e.name}. Run pip install {e.name} first") from e
    except TypeError as e:
        if "authentication" in str(e):
            raise RuntimeError("No Claude credentials found: please set the environment variable ANTHROPIC_API_KEY") from e
        raise


def add_provider_args(parser, default="mock"):
    """Add the two arguments --provider and --model to a command-line script."""
    parser.add_argument("--provider", choices=PROVIDERS,
                        default=os.environ.get("AGENT_LAB_PROVIDER", default),
                        help="which provider's model to use (default mock: a mock model that needs no API key)")
    parser.add_argument("--model", help="override the default model")
    return parser
