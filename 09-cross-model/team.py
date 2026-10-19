"""
The "team roster" shared by Chapter 9: three vendors, Claude, GPT and DeepSeek
==============================================================================

By default everything uses the mock model, so no API key is needed.
With --live: vendors whose keys are set switch to real models, and the rest keep using the mock model,
so it still runs if you only have keys for one or two of them.

  Vendor      API key variable      Overrides the default model
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
                        help="use real models for the vendors whose API keys are set (the rest keep using the mock model)")
    return parser


def provider_for(vendor, live):
    return vendor if live and os.environ.get(KEY_ENV[vendor]) else "mock"


def model_for(vendor):
    return os.environ.get(f"AGENT_LAB_{vendor.upper()}_MODEL") or DEFAULT_MODELS[vendor]


def label(vendor, live):
    return f"{vendor} ({model_for(vendor)})" if provider_for(vendor, live) != "mock" else f"{vendor} (mock)"


def ask(vendor, prompt, live, mock, system=None, effort="low"):
    """Ask one vendor a question. mock is this vendor's script in mock mode."""
    provider = provider_for(vendor, live)
    model = model_for(vendor) if provider != "mock" else None
    return chat([user_message(prompt)], system=system, provider=provider, model=model, mock=mock, effort=effort)


def fit(text, width):
    """Truncate and pad to a display width (a Chinese character counts as 2), so that tables line up."""
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
        return "all mock models (add --live to use real models for the vendors whose keys are set)"
    if not real:
        return "--live was given, but no API key was found, so everything is still a mock model"
    return "real models: " + ", ".join(label(v, live) for v in real) + \
        ("; the rest use the mock model" if len(real) < len(VENDORS) else "")
