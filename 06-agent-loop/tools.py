"""
Lab 2: Getting "tools" ready for the model
==========================================

A model can only output text. To let it "do" things, we have to provide tools, plus a **manual** that tells it
what each tool can do and what arguments to pass.
The manual is written in JSON Schema, a format all three APIs (Claude, GPT, DeepSeek) accept.

The model never runs a tool itself. It only says "I want to call calculator with arguments {...}";
**our program does the actual running**, and then tells the model the result.

Four tools are defined here:
  calculator   safely evaluates an arithmetic expression
  search_repo  searches this repo's docs and code for a keyword
  read_file    reads a few lines of a file in the repo
  list_dir     lists the files in a directory of the repo

Run: python tools.py  (prints the tool manual the model sees, and demos a few calls, including ones that fail)
Other labs import TOOLS and run_tool from here.
"""

import ast
import json
import operator
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # the tools can only work inside this repo
TEXT_SUFFIXES = {".md", ".py", ".txt", ".csv"}

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluates an arithmetic expression. Supports + - * / ** and parentheses, e.g. '26.2 - 15.3'.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "The arithmetic expression to evaluate"}},
            "required": ["expression"],
        },
    },
    {
        "name": "search_repo",
        "description": ("Searches all docs and code in the agent-lab repo for a keyword. "
                        "Returns matching lines (file path:line number: content), at most 8."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "The keyword to search for; the more specific, the better"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": ("Reads some lines of a file in the repo. The path is relative to the repo root, "
                        "e.g. '02-linear-models/README.md'."),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the repo root"},
                "start": {"type": "integer", "description": "Line to start from (counting from 1); default 1"},
                "end": {"type": "integer", "description": "Last line to read (inclusive); default start + 39"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": ("Lists the files and subdirectories of a directory in the repo. "
                        "The path is relative to the repo root; '.' means the root."),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path relative to the repo root"}},
            "required": ["path"],
        },
    },
]


class ToolError(Exception):
    """A tool failed. The error message is passed to the model as is, so it gets a chance to correct itself."""


# ---------------------------------------------------------------- tool implementations
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression):
    """Only numbers and arithmetic are allowed. No eval: the string from the model could contain arbitrary code."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ToolError(f"unsupported part of the expression: {ast.dump(node)[:40]}")
    try:
        value = ev(ast.parse(expression, mode="eval").body)
    except SyntaxError:
        raise ToolError(f"invalid expression: {expression!r}")
    except ZeroDivisionError:
        raise ToolError("division by zero")
    return str(round(value, 10))


def _resolve(path):
    """Turn the model's relative path into a real path, and make sure it is still inside the repo
    (no escaping with ../../)."""
    target = (REPO / path).resolve()
    if target != REPO and REPO not in target.parents:
        raise ToolError(f"access outside the repo is not allowed: {path}")
    return target


def search_repo(query):
    if not query.strip():
        raise ToolError("the keyword must not be empty")
    hits = []
    for f in sorted(REPO.rglob("*")):
        if f.suffix not in TEXT_SUFFIXES or ".git" in f.parts or "data" in f.parts:
            continue
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if query in line:
                hits.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:120]}")
                if len(hits) >= 8:
                    return "\n".join(hits)
    return "\n".join(hits) if hits else f"no matches for {query!r}"


def read_file(path, start=1, end=None):
    f = _resolve(path)
    if not f.is_file():
        raise ToolError(f"file not found: {path} (try list_dir or search_repo to find it first)")
    lines = f.read_text(errors="ignore").splitlines()
    start = max(1, int(start))
    end = min(len(lines), int(end) if end else start + 39)
    body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return f"{path} ({len(lines)} lines in total, showing lines {start}–{end})\n{body}"


def list_dir(path):
    d = _resolve(path)
    if not d.is_dir():
        raise ToolError(f"directory not found: {path}")
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir()
                   if not p.name.startswith(".") and p.name != "__pycache__")
    return "\n".join(names)


IMPLEMENTATIONS = {"calculator": calculator, "search_repo": search_repo,
                   "read_file": read_file, "list_dir": list_dir}


def check_args(tool, args):
    """The most basic check against the JSON Schema: are the required arguments there, and are the types right?"""
    schema = tool["parameters"]
    for name in schema.get("required", []):
        if name not in args:
            raise ToolError(f"missing required argument: {name}")
    types = {"string": str, "integer": int, "number": (int, float)}
    for name, value in args.items():
        spec = schema["properties"].get(name)
        if spec is None:
            raise ToolError(f"unknown argument: {name}")
        if not isinstance(value, types[spec["type"]]):
            raise ToolError(f"argument {name} should be {spec['type']}, got {type(value).__name__}")


def run_tool(name, args, tools=TOOLS, implementations=IMPLEMENTATIONS):
    """Run one tool call. Returns (result text, is_error). Errors don't raise; they are reported to the model."""
    tool = next((t for t in tools if t["name"] == name), None)
    if tool is None:
        return f"No tool called {name}. Available tools: {', '.join(t['name'] for t in tools)}", True
    try:
        check_args(tool, args)
        return implementations[name](**args), False
    except ToolError as e:
        return f"Error: {e}", True


if __name__ == "__main__":
    print("[1] The tool manual the model sees (JSON Schema):")
    print(json.dumps(TOOLS[0], ensure_ascii=False, indent=2))
    print(f"   ...{len(TOOLS)} tools in total: {', '.join(t['name'] for t in TOOLS)}\n")

    print("[2] Our program runs the tools on the model's behalf:")
    demos = [
        ("calculator", {"expression": "26.2 - 15.3"}),
        ("calculator", {"expression": "__import__('os').system('rm -rf /')"}),
        ("search_repo", {"query": "Les 3 Brasseurs"}),
        ("read_file", {"path": "../../etc/passwd"}),
        ("read_file", {"path": "02-linear-models/README.md", "start": 1, "end": 3}),
        ("list_dir", {"path": "06-agent-loop"}),
        ("calculator", {"expr": "1+1"}),
        ("send_email", {"to": "boss"}),
    ]
    for name, args in demos:
        result, is_error = run_tool(name, args)
        flag = "❌" if is_error else "✅"
        first = result.splitlines()[0] if result else ""
        more = f"  (result: {len(result.splitlines())} lines)" if len(result.splitlines()) > 1 else ""
        print(f"   {flag} {name}({json.dumps(args, ensure_ascii=False)})")
        print(f"      → {first[:90]}{more}")

    print("""
   Look at the ❌ lines:
   * The calculator doesn't use eval, so code hidden in an "expression" is never run.
   * read_file won't let you escape the repo with ../
   * A misspelled argument name, or a call to a tool that doesn't exist, returns a clear error message
     instead of crashing the program.
   After seeing the error message, the model can usually correct itself. These "guards" are an early form
   of the harness in Chapter 7.

Think about it:
  1. If a tool's description is vague (say, just "search"), what mistakes will the model make?
  2. Why should a tool error be returned to the model as a message, instead of raising and ending the program?
  3. If you gave the model a tool that can run any shell command, what limits would you need to add?""")
