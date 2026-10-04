"""
第九章共用的"团队名单"：Claude、GPT、DeepSeek 三家
====================================================

默认全部用假模型，不需要任何 API key。
加上 --live 以后：配置了 key 的厂商改用真实模型，没配置的继续用假模型，所以只有一两家的 key 也能跑。

  厂商        环境变量              默认型号（可以用环境变量覆盖）
  claude      ANTHROPIC_API_KEY     AGENT_LAB_CLAUDE_MODEL
  gpt         OPENAI_API_KEY        AGENT_LAB_GPT_MODEL
  deepseek    DEEPSEEK_API_KEY      AGENT_LAB_DEEPSEEK_MODEL
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.llm import DEFAULT_MODELS, chat, user_message  # noqa: E402

VENDORS = ["claude", "gpt", "deepseek"]
KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "gpt": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


def add_live_arg(parser):
    parser.add_argument("--live", action="store_true",
                        help="配置了 API key 的厂商改用真实模型（没配置的继续用假模型）")
    return parser


def provider_for(vendor, live):
    return vendor if live and os.environ.get(KEY_ENV[vendor]) else "mock"


def model_for(vendor):
    return os.environ.get(f"AGENT_LAB_{vendor.upper()}_MODEL") or DEFAULT_MODELS[vendor]


def label(vendor, live):
    return f"{vendor}（{model_for(vendor)}）" if provider_for(vendor, live) != "mock" else f"{vendor}（假）"


def ask(vendor, prompt, live, mock, system=None, effort="low"):
    """向某一家提一个问题。mock 是这家在假模型模式下的剧本。"""
    provider = provider_for(vendor, live)
    model = model_for(vendor) if provider != "mock" else None
    return chat([user_message(prompt)], system=system, provider=provider, model=model, mock=mock, effort=effort)


def fit(text, width):
    """按显示宽度（汉字算 2）截断并补齐，让表格对齐。"""
    full = sum(2 if ord(ch) > 127 else 1 for ch in text)
    if full <= width:
        return text + " " * (width - full)
    out, w = "", 0
    for ch in text:
        cw = 2 if ord(ch) > 127 else 1
        if w + cw > width - 2:
            return out + "…" + " " * (width - w - 1)
        out, w = out + ch, w + cw
    return out + " " * (width - w)


def roster(live):
    real = [v for v in VENDORS if provider_for(v, live) != "mock"]
    if not live:
        return "全部是假模型（加 --live 可以让配置了 key 的厂商用真实模型）"
    if not real:
        return "加了 --live，但没有找到任何 API key，所以全部还是假模型"
    return "真实模型：" + "、".join(label(v, live) for v in real) + \
        ("；其余用假模型" if len(real) < len(VENDORS) else "")
